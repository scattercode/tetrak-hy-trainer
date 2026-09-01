#!/usr/bin/env python3
"""Diff a harvest's transcripts against the canonical charset.

Brief 012 Stage 1.3: run against every new source before anything trains
on it. A character the charset lacks is a character the training
pipeline silently filters and the model can never emit -- U+2024 cost
brief 011 a full training run this way. The output is a table of stray
characters with counts and codepoints; deciding which strays become
charset additions (a new model version) and which stay out of scope
(Cyrillic) is a recorded decision, not this script's.

Run from the repo root:

    python scripts/charset_diff.py runs/harvest/faustus-1968 [more dirs...]
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import charset  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for directory in map(Path, sys.argv[1:]):
        text = "\n".join(
            f.read_text(encoding="utf-8") for f in sorted((directory / "text").glob("*.txt"))
        )
        found = charset.strays(text)
        total = sum(found.values())
        print(f"\n{directory} — {len(found)} stray characters, {total} occurrences")
        for character, count in found.most_common(25):
            name = unicodedata.name(character, "?")
            print(f"  {count:>7}  U+{ord(character):04X}  {character!r}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
