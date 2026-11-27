#!/usr/bin/env python3
"""Translate a single string with Ollama through its HTTP API."""

import argparse
import sys

from ollama_client import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    OllamaError,
    translate_batch,
)


def translate(text, source_lang, source_code, target_lang, target_code,
              model="gemma", base_url=None, timeout=DEFAULT_TIMEOUT,
              retries=DEFAULT_RETRIES):
    return translate_batch(
        [text],
        source_lang=source_lang,
        source_code=source_code,
        target_lang=target_lang,
        target_code=target_code,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
    )[0]


def main():
    parser = argparse.ArgumentParser(description="Translate text using Ollama")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--source-lang", default="English", help="Source language name")
    parser.add_argument("--source-code", default="en", help="Source language code")
    parser.add_argument("--target-lang", default="Spanish", help="Target language name")
    parser.add_argument("--target-code", default="es", help="Target language code")
    parser.add_argument("--model", default="gemma", help="Ollama model name")
    parser.add_argument("--url", default=None,
                        help="Ollama server URL (default: OLLAMA_HOST or http://127.0.0.1:11434)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="Request timeout in seconds (default: 300)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="Retries for failed requests (default: 3)")

    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")

    try:
        result = translate(
            args.text, args.source_lang, args.source_code,
            args.target_lang, args.target_code, args.model,
            base_url=args.url, timeout=args.timeout, retries=args.retries,
        )
        print(result)
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
