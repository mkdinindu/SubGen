import json
import os
import unittest
from unittest import mock

import ollama_client as client


class NormalizeBaseUrlTests(unittest.TestCase):
    def test_default_url(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(client.normalize_base_url(), "http://127.0.0.1:11434")

    def test_environment_host(self):
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": " http://192.168.1.10:11434/ "}):
            self.assertEqual(client.normalize_base_url(), "http://192.168.1.10:11434")

    def test_adds_missing_scheme(self):
        self.assertEqual(client.normalize_base_url("localhost:11434"), "http://localhost:11434")

    def test_rejects_unsupported_scheme(self):
        with self.assertRaises(client.OllamaError):
            client.normalize_base_url("ftp://example.com")


class FakeResponse:
    def __init__(self, body):
        self.body = body.encode("utf-8") if isinstance(body, str) else body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ListModelsTests(unittest.TestCase):
    def test_returns_sorted_unique_names_from_tags_endpoint(self):
        body = json.dumps({"models": [
            {"name": "gemma:2b"}, {"name": "llama3.2"}, {"name": "gemma:2b"},
        ]})
        with mock.patch.object(client.urllib.request, "urlopen",
                               return_value=FakeResponse(body)) as urlopen:
            result = client.list_models("http://host:1/")
        self.assertEqual(result, ["gemma:2b", "llama3.2"])
        self.assertEqual(urlopen.call_args.args[0].full_url,
                         "http://host:1/api/tags")

    def test_invalid_json_raises_ollama_error(self):
        with mock.patch.object(client.urllib.request, "urlopen",
                               return_value=FakeResponse("not json")):
            with self.assertRaises(client.OllamaError):
                client.list_models()

    def test_connection_error_is_retryable(self):
        error = client.urllib.error.URLError("refused")
        with mock.patch.object(client.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(client.OllamaError) as caught:
                client.list_models()
        self.assertTrue(caught.exception.retryable)


class PromptTests(unittest.TestCase):
    def test_user_prompt_contains_count_and_ids(self):
        payload = json.loads(client.build_user_prompt(["one", "two"]))
        self.assertEqual(payload["count"], 2)
        self.assertEqual([cue["id"] for cue in payload["cues"]], [1, 2])
        self.assertEqual([cue["text"] for cue in payload["cues"]], ["one", "two"])


class ParseTranslationsTests(unittest.TestCase):
    def test_parses_code_fenced_json(self):
        content = '```json\n{"translations": ["uno", "dos"]}\n```'
        self.assertEqual(client.parse_translations(content, 2), ["uno", "dos"])

    def test_orders_id_objects(self):
        content = '[{"id": 2, "text": "dos"}, {"id": 1, "text": "uno"}]'
        self.assertEqual(client.parse_translations(content, 2), ["uno", "dos"])

    def test_orders_dict_keys(self):
        self.assertEqual(client.parse_translations('{"2": "dos", "1": "uno"}', 2), ["uno", "dos"])

    def test_wrong_count_is_retryable(self):
        with self.assertRaises(client.OllamaError) as caught:
            client.parse_translations('{"translations": ["solo"]}', 2)
        self.assertTrue(caught.exception.retryable)


class TranslateBatchTests(unittest.TestCase):
    def test_empty_batch(self):
        self.assertEqual(
            client.translate_batch([], "English", "en", "Spanish", "es"),
            [],
        )

    def test_retries_then_succeeds(self):
        effects = [client.OllamaError("temporary", True), '{"translations": ["hola"]}']
        with mock.patch.object(client, "_post_chat", side_effect=effects) as post, \
                mock.patch.object(client.time, "sleep") as sleep:
            result = client.translate_batch(
                ["hello"], "English", "en", "Spanish", "es", retries=2
            )
        self.assertEqual(result, ["hola"])
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(client.DEFAULT_RETRY_DELAY)

    def test_stops_on_non_retryable_error(self):
        error = client.OllamaError("invalid request", False)
        with mock.patch.object(client, "_post_chat", side_effect=error) as post:
            with self.assertRaises(client.OllamaError):
                client.translate_batch(
                    ["hello"], "English", "en", "Spanish", "es", retries=3
                )
        self.assertEqual(post.call_count, 1)

    def test_request_uses_chat_endpoint_with_json_format(self):
        with mock.patch.object(
            client, "_post_chat", return_value='{"translations": ["hola"]}'
        ) as post:
            client.translate_batch(
                ["hello"], "English", "en", "Spanish", "es",
                model="test-model", base_url="http://host:1/", timeout=7,
            )
        url, payload, timeout = post.call_args.args
        self.assertEqual(url, "http://host:1/api/chat")
        self.assertEqual(timeout, 7)
        self.assertEqual(payload["model"], "test-model")
        self.assertTrue(payload["format"] == "json")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
