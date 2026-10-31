#!/usr/bin/env python3
"""Translate text using Ollama. Spins up the model if needed."""

import argparse
import shutil
import subprocess
import sys
import time


def translate(text, source_lang, source_code, target_lang, target_code,
              model="gemma"):
    prompt = (
        f"You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text "
        f"while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_lang} translation, without any additional explanations or commentary. "
        f"Please translate the following {source_lang} text into {target_lang}:\n\n"
        f"{text}"
    )

    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=300
    )

    if proc.returncode != 0:
        raise RuntimeError(f"ollama failed: {proc.stderr.strip()}")

    output = proc.stdout.strip()
    # Remove the prompt echo if ollama repeats it
    if prompt[:80] in output:
        output = output.replace(prompt, "", 1).strip()
    return output


def main():
    parser = argparse.ArgumentParser(description="Translate text using Ollama")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--source-lang", default="English", help="Source language name")
    parser.add_argument("--source-code", default="en", help="Source language code")
    parser.add_argument("--target-lang", default="Spanish", help="Target language name")
    parser.add_argument("--target-code", default="es", help="Target language code")
    parser.add_argument("--model", default="gemma", help="Ollama model name")

    args = parser.parse_args()

    if not shutil.which("ollama"):
        print("Error: ollama not found. Install from https://ollama.com", file=sys.stderr)
        sys.exit(1)

    try:
        result = translate(
            args.text, args.source_lang, args.source_code,
            args.target_lang, args.target_code, args.model
        )
        print(result)
    except subprocess.TimeoutExpired:
        print("Error: Translation timed out (5 min).", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
