#!/usr/bin/env python3
"""Generate subtitles from a video file using Whisper (speech-to-text)
and optionally translate with Ollama. Progress bars for all stages."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def check_deps():
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install it with: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper not installed.", file=sys.stderr)
        print("Install it: pip install openai-whisper", file=sys.stderr)
        sys.exit(1)
    if not tqdm:
        print("Info: install tqdm for progress bars: pip install tqdm", file=sys.stderr)


def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def extract_audio(video_path, audio_path, total_duration):
    pbar = tqdm(total=100, desc="Extracting audio", unit="%", file=sys.stderr) if tqdm else None
    proc = subprocess.Popen(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path, "-y", "-progress", "pipe:1"],
        stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True
    )
    # ffmpeg -progress outputs to stdout
    for line in proc.stdout:
        m = re.match(r"out_time_us=(\d+)", line)
        if m and pbar:
            elapsed = int(m.group(1)) / 1_000_000
            pct = min(int(elapsed / total_duration * 100), 100)
            pbar.n = pct
            pbar.refresh()
    proc.wait()
    if pbar:
        pbar.n = 100
        pbar.refresh()
        pbar.close()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")


def transcribe(audio_path, model_name="base", language=None):
    import whisper
    # Whisper's transcribe already shows a tqdm bar when verbose=None (default)
    if tqdm:
        print(f"Loading Whisper model ({model_name}) ...", file=sys.stderr)
    else:
        print(f"Loading Whisper model ({model_name}) ...", file=sys.stderr)
    model = whisper.load_model(model_name)

    print(f"Transcribing ...", file=sys.stderr)
    opts = {"task": "transcribe", "word_timestamps": True, "verbose": None}
    if language:
        opts["language"] = language
    result = model.transcribe(audio_path, **opts)
    return result


def build_subtitle_chunks(segments, chars_per_line=42, max_lines=2,
                          min_duration=1.0, max_duration=6.0):
    words = []
    for seg in segments:
        for word in seg.get("words", []):
            words.append({
                "text": word["word"].strip(),
                "start": word["start"],
                "end": word["end"]
            })

    if not words:
        return []

    chunks = []
    current_chunk = []
    chunk_start = words[0]["start"]

    for w in words:
        candidate = current_chunk + [w]
        candidate_text = " ".join(wd["text"] for wd in candidate)
        candidate_duration = w["end"] - chunk_start
        lines = _count_lines(candidate_text, chars_per_line)

        if (lines > max_lines or candidate_duration > max_duration) and current_chunk:
            chunk_end = current_chunk[-1]["end"]
            duration = chunk_end - chunk_start
            if duration < min_duration:
                chunk_end = chunk_start + min_duration
            text = " ".join(wd["text"] for wd in current_chunk)
            chunks.append({"start": chunk_start, "end": chunk_end, "text": text})
            current_chunk = [w]
            chunk_start = w["start"]
        else:
            current_chunk.append(w)

    if current_chunk:
        chunk_end = current_chunk[-1]["end"]
        text = " ".join(wd["text"] for wd in current_chunk)
        chunks.append({"start": chunk_start, "end": chunk_end, "text": text})

    return chunks


def _count_lines(text, max_chars):
    if len(text) <= max_chars:
        return 1
    words = text.split()
    lines = 1
    current = 0
    for w in words:
        if current + len(w) + (1 if current > 0 else 0) > max_chars:
            lines += 1
            current = len(w)
        else:
            current += len(w) + (1 if current > 0 else 0)
    return lines


def translate_text(text, source_lang, source_code, target_lang, target_code,
                   model="gemma"):
    prompt = (
        f"You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text "
        f"while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_lang} translation, without any additional explanations or commentary. "
        f"There can be some unrelated, letters, artifact in input. avoid translating them or including them in output "
        f"Please translate the following {source_lang} text into {target_lang}:\n\n"
        f"{text}"
    )
    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True, text=True, timeout=300
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ollama failed: {proc.stderr.strip()}")
    output = proc.stdout.strip()
    if prompt[:80] in output:
        output = output.replace(prompt, "", 1).strip()
    return output


def write_srt(chunks, output_path, translate_fn=None):
    iterator = tqdm(chunks, desc="Translating" if translate_fn else "Writing subtitles",
                    unit="seg", file=sys.stderr) if tqdm else chunks
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(iterator, 1):
            text = chunk["text"]
            if translate_fn:
                text = translate_fn(text)
            start = _fmt_time(chunk["start"])
            end = _fmt_time(chunk["end"])
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def _fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from video")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("-o", "--output", default=None, help="Output SRT path (default: <video>.srt)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", default=None, help="Audio language code (e.g. en, es). Auto-detect if omitted.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Add offset (seconds) to all timestamps. Negative = earlier, positive = later")
    # Translation options
    parser.add_argument("--translate", action="store_true", help="Translate subtitles via Ollama")
    parser.add_argument("--source-lang", default="English", help="Source language name")
    parser.add_argument("--source-code", default="en", help="Source language code")
    parser.add_argument("--target-lang", default="Spanish", help="Target language name")
    parser.add_argument("--target-code", default="es", help="Target language code")
    parser.add_argument("--ollama-model", default="gemma", help="Ollama model for translation")
    # Subtitle timing options
    parser.add_argument("--min-duration", type=float, default=1.0,
                        help="Minimum subtitle duration in seconds (default: 1.0)")
    parser.add_argument("--max-duration", type=float, default=6.0,
                        help="Maximum subtitle duration in seconds (default: 6.0)")
    parser.add_argument("--chars-per-line", type=int, default=42,
                        help="Max characters per line (default: 42)")

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file '{args.video}' not found", file=sys.stderr)
        sys.exit(1)

    check_deps()

    if args.translate and not shutil.which("ollama"):
        print("Error: ollama not found (needed for translation).", file=sys.stderr)
        print("Install from https://ollama.com", file=sys.stderr)
        sys.exit(1)

    output = args.output or os.path.splitext(args.video)[0] + ".srt"

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.wav")
        duration = get_video_duration(args.video)

        extract_audio(args.video, audio_path, duration)

        result = transcribe(audio_path, args.whisper_model, args.language)

        chunks = build_subtitle_chunks(
            result["segments"],
            chars_per_line=args.chars_per_line,
            max_duration=args.max_duration,
            min_duration=args.min_duration
        )

        if args.delay != 0:
            for c in chunks:
                c["start"] = max(0, c["start"] + args.delay)
                c["end"] = max(0, c["end"] + args.delay)

        if args.translate:
            def translator(text):
                return translate_text(
                    text, args.source_lang, args.source_code,
                    args.target_lang, args.target_code, args.ollama_model
                )
        else:
            translator = None

        write_srt(chunks, output, translator)

    print(f"Done → {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
