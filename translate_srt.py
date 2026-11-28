#!/usr/bin/env python3
"""Translate an existing ``.srt`` file with Ollama.

This mode never runs ffmpeg or Whisper: cue numbers and timestamps are read
from the input file and reused unchanged in the translated file.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from ollama_client import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    OllamaError,
    normalize_base_url,
    translate_batch,
)
from srt_utils import SRTError, read_srt, write_srt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def default_output_path(input_path, target_code):
    """Return ``<name>.<target-code>.srt`` next to the input file."""
    safe_code = re.sub(r"[^A-Za-z0-9_-]+", "", str(target_code)) or "translated"
    return str(Path(input_path).with_suffix(f".{safe_code}.srt"))


def translate_texts(texts, source_lang, source_code, target_lang, target_code,
                    model="gemma", base_url=None, timeout=DEFAULT_TIMEOUT,
                    retries=DEFAULT_RETRIES, batch_size=DEFAULT_BATCH_SIZE):
    """Translate cue texts in batches, preserving order and blank cues."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    translated = [""] * len(texts)
    pending = [i for i, text in enumerate(texts) if text.strip()]
    pbar = tqdm(total=len(pending), desc="Translating", unit="seg",
                file=sys.stderr) if tqdm else None

    try:
        for start in range(0, len(pending), batch_size):
            indices = pending[start:start + batch_size]
            results = translate_batch(
                [texts[i] for i in indices],
                source_lang=source_lang,
                source_code=source_code,
                target_lang=target_lang,
                target_code=target_code,
                model=model,
                base_url=base_url,
                timeout=timeout,
                retries=retries,
            )
            if len(results) != len(indices):
                raise OllamaError(
                    f"Ollama returned {len(results)} translation(s) "
                    f"for {len(indices)} cue(s)"
                )
            for index, result in zip(indices, results):
                translated[index] = result
            if pbar:
                pbar.update(len(indices))
    finally:
        if pbar:
            pbar.close()

    return translated


def translate_srt(input_path, output_path, source_lang, source_code,
                  target_lang, target_code, model="gemma", base_url=None,
                  timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES,
                  batch_size=DEFAULT_BATCH_SIZE):
    """Translate every cue in ``input_path`` and write ``output_path``."""
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Output path must differ from the input SRT path")

    cues = read_srt(input_path)
    translated = translate_texts(
        [cue.text for cue in cues],
        source_lang=source_lang,
        source_code=source_code,
        target_lang=target_lang,
        target_code=target_code,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        batch_size=batch_size,
    )
    for cue, text in zip(cues, translated):
        cue.text = text

    write_srt(cues, output_path)
    return len(cues)


def main():
    parser = argparse.ArgumentParser(
        description="Translate an existing SRT file with Ollama (no ffmpeg/Whisper)"
    )
    parser.add_argument("srt", help="Path to the existing .srt file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output SRT path (default: <name>.<target-code>.srt)")
    parser.add_argument("--source-lang", default="English", help="Source language name")
    parser.add_argument("--source-code", default="en", help="Source language code")
    parser.add_argument("--target-lang", default="Spanish", help="Target language name")
    parser.add_argument("--target-code", default="es", help="Target language code")
    parser.add_argument("--ollama-model", default="gemma", help="Ollama model name")
    parser.add_argument("--ollama-url", default=None,
                        help="Ollama server URL (default: OLLAMA_HOST or http://127.0.0.1:11434)")
    parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="Seconds to wait for each Ollama request (default: 300)")
    parser.add_argument("--ollama-retries", type=int, default=DEFAULT_RETRIES,
                        help="Retries for failed Ollama requests (default: 3)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Subtitle cues per Ollama request (default: 20)")

    args = parser.parse_args()

    if args.ollama_timeout <= 0:
        parser.error("--ollama-timeout must be positive")
    if args.ollama_retries < 0:
        parser.error("--ollama-retries cannot be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not os.path.isfile(args.srt):
        print(f"Error: file '{args.srt}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        server_url = normalize_base_url(args.ollama_url)
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output = args.output or default_output_path(args.srt, args.target_code)
    print(
        f"Translating SRT cues via Ollama at {server_url} "
        f"(batch size {args.batch_size}) ...",
        file=sys.stderr,
    )

    try:
        count = translate_srt(
            args.srt, output,
            args.source_lang, args.source_code,
            args.target_lang, args.target_code,
            model=args.ollama_model,
            base_url=server_url,
            timeout=args.ollama_timeout,
            retries=args.ollama_retries,
            batch_size=args.batch_size,
        )
    except (OllamaError, SRTError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Done → {output} ({count} cue(s))", file=sys.stderr)


if __name__ == "__main__":
    main()
