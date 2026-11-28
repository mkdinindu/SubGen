#!/usr/bin/env python3
"""Read and write SubRip (``.srt``) subtitle files.

Cue numbers and timestamp strings are kept verbatim so that translating a
file never changes its timing or numbering.
"""

import re


TIMING_RE = re.compile(
    r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})"
)


class SRTError(ValueError):
    """Raised when an SRT file cannot be parsed."""


class Cue:
    """One SRT cue: its original number, timestamps, and text."""

    def __init__(self, index, start, end, text):
        self.index = index
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self):
        return f"Cue({self.index!r}, {self.start!r}, {self.end!r}, {self.text!r})"


def parse_srt(content):
    """Parse SRT text into a list of :class:`Cue` objects."""
    content = content.lstrip("\ufeff")
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues = []
    i = 0
    total = len(lines)

    while i < total:
        while i < total and not lines[i].strip():
            i += 1
        if i >= total:
            break

        first_line = lines[i].strip()
        i += 1
        if i >= total:
            raise SRTError(f"Cue {len(cues) + 1}: missing timing line")

        second_line = lines[i].strip()
        i += 1
        match = TIMING_RE.match(second_line)

        if match:
            index = first_line
            timing_line = second_line
        else:
            # Some SRT files omit cue numbers: the first line is the timing.
            match = TIMING_RE.match(first_line)
            if not match:
                raise SRTError(
                    f"Cue {len(cues) + 1}: invalid timing line {second_line!r}"
                )
            index = str(len(cues) + 1)
            timing_line = first_line
            i -= 1  # the line read as `second_line` is cue text

        text_lines = []
        while i < total and lines[i].strip():
            text_lines.append(lines[i].rstrip())
            i += 1

        cues.append(
            Cue(
                index=index,
                start=match.group(1),
                end=match.group(2),
                text="\n".join(text_lines),
            )
        )

    if not cues:
        raise SRTError("No subtitle cues found in the SRT file")
    return cues


def read_srt(path):
    """Read an SRT file, tolerating a UTF-8 BOM and Windows line endings."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return parse_srt(f.read())


def format_srt(cues):
    """Render cues as standard SRT text."""
    return (
        "\n\n".join(
            f"{cue.index}\n{cue.start} --> {cue.end}\n{cue.text}" for cue in cues
        )
        + "\n"
    )


def write_srt(cues, path):
    """Write cues to an SRT file using LF line endings."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(format_srt(cues))
