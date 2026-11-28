import os
import tempfile
import unittest
from unittest import mock

import subgen


def _word(text, start, end):
    return {"word": text, "start": start, "end": end}


def _segments(words):
    return [{"words": words}]


class CollectWordsTests(unittest.TestCase):
    def test_merges_standalone_punctuation_into_previous_word(self):
        words = [
            _word("Hello", 0.0, 0.4),
            _word(",", 0.4, 0.5),
            _word("world", 0.6, 1.0),
        ]
        merged = subgen._collect_words(_segments(words))
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["text"], "Hello,")
        self.assertEqual(merged[0]["end"], 0.5)
        self.assertEqual(merged[1]["text"], "world")

    def test_ignores_blank_words(self):
        merged = subgen._collect_words(_segments([_word("  ", 0, 1), _word("Hi", 1, 2)]))
        self.assertEqual([word["text"] for word in merged], ["Hi"])


class WrapTests(unittest.TestCase):
    def test_wrap_lines_respects_max_chars(self):
        lines = subgen._wrap_lines("one two three four", 7)
        self.assertTrue(all(len(line) <= 7 for line in lines))
        self.assertEqual(" ".join(lines), "one two three four")

    def test_wrap_text_uses_newlines(self):
        self.assertEqual(subgen._wrap_text("a b c", 3), "a b\nc")

    def test_count_lines_is_at_least_one(self):
        self.assertEqual(subgen._count_lines("", 42), 1)


class ChunkTests(unittest.TestCase):
    def test_chunks_respect_max_duration(self):
        words = [
            _word("Short", 0.0, 0.4),
            _word("chunk", 0.4, 0.8),
            _word("second", 4.0, 4.4),
            _word("chunk", 4.4, 4.8),
        ]
        chunks = subgen.build_subtitle_chunks(
            _segments(words),
            chars_per_line=20,
            max_lines=1,
            min_duration=0.0,
            max_duration=1.0,
        )
        self.assertEqual([chunk["text"] for chunk in chunks], ["Short chunk", "second chunk"])
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 1.0)

    def test_min_gap_clips_previous_cue(self):
        words = [
            _word("A", 0.0, 0.2),
            _word("B", 0.5, 0.7),
            _word("C", 2.0, 2.2),
        ]
        chunks = subgen.build_subtitle_chunks(
            _segments(words),
            chars_per_line=42,
            max_lines=2,
            min_duration=1.0,
            max_duration=1.0,
            min_gap=1.5,
        )
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0]["end"], 0.5)
        self.assertAlmostEqual(chunks[1]["end"], 3.0)

    def test_no_words_returns_no_chunks(self):
        self.assertEqual(subgen.build_subtitle_chunks([]), [])


class FormatTimeTests(unittest.TestCase):
    def test_rounds_milliseconds(self):
        self.assertEqual(subgen._fmt_time(0.0004), "00:00:00,000")
        self.assertEqual(subgen._fmt_time(1.2346), "00:00:01,235")
        self.assertEqual(subgen._fmt_time(1.9996), "00:00:02,000")
        self.assertEqual(subgen._fmt_time(3661.5), "01:01:01,500")


class SrtTests(unittest.TestCase):
    def setUp(self):
        self._tqdm_patcher = mock.patch.object(subgen, "tqdm", None)
        self._tqdm_patcher.start()

    def tearDown(self):
        self._tqdm_patcher.stop()

    def test_write_srt_without_translation(self):
        chunks = [
            {"start": 0.0, "end": 1.2346, "text": "Hello"},
            {"start": 2.0, "end": 3.0, "text": "World"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.srt")
            subgen.write_srt(chunks, path)
            with open(path, encoding="utf-8") as handle:
                content = handle.read()
        self.assertEqual(
            content,
            "1\n00:00:00,000 --> 00:00:01,235\nHello\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nWorld\n\n",
        )

    def test_write_srt_translates_in_batches(self):
        chunks = [
            {"start": index, "end": index + 1, "text": f"Cue {index}"}
            for index in range(5)
        ]
        seen = []

        def translator(texts):
            seen.append(list(texts))
            return [f"T:{text}" for text in texts]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.srt")
            subgen.write_srt(chunks, path, translator, batch_size=2)
            with open(path, encoding="utf-8") as handle:
                content = handle.read()

        self.assertEqual([len(batch) for batch in seen], [2, 2, 1])
        self.assertIn("T:Cue 4", content)

    def test_write_srt_rejects_wrong_translation_count(self):
        chunks = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.srt")
            with self.assertRaises(RuntimeError):
                subgen.write_srt(chunks, path, lambda texts: ["one", "too many"])


class TranslateCuesTests(unittest.TestCase):
    def test_delegates_to_shared_client(self):
        with mock.patch.object(subgen, "translate_batch", return_value=["hola"]) as patched:
            result = subgen.translate_cues(
                ["hello"], "English", "en", "Spanish", "es",
                model="gemma", base_url="http://host:1", timeout=7, retries=0,
            )
        self.assertEqual(result, ["hola"])
        patched.assert_called_once()
        self.assertEqual(patched.call_args.kwargs["base_url"], "http://host:1")
        self.assertEqual(patched.call_args.kwargs["timeout"], 7)


if __name__ == "__main__":
    unittest.main()
