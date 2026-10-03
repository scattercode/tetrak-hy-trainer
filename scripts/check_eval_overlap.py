#!/usr/bin/env python3
"""Find harvested pages that print an evaluation page's text.

Academic editions reprint and vary their texts, so a page of one volume can
carry an evaluation page of another almost word for word. Training on it --
as real crops, as synthetic text or in the word list -- lets the model
memorise text it is later scored on reading, and nothing fails: the numbers
just improve for the wrong reason. Brief 013 found 25 such pages, one of them
sharing 203 eight-word runs with the evaluation page it reprints, and they
are listed in ``tetrak_hy_trainer.heldout.OVERLAPPING_PAGES``.

This is that check. Every harvested page that is not itself an evaluation
page is compared with every evaluation transcript by eight-word shingles,
after ``wikisource.normalise_transcript`` and lower-casing; five or more
shared runs flags it. Pages already in ``OVERLAPPING_PAGES`` are reported as
known. It exits 1 when it finds an unlisted page, so it can gate a harvest.

Run it whenever a work with evaluation pages gains a volume, before training
on that volume, and add what it reports to ``OVERLAPPING_PAGES``.

Prerequisites: this repository's venv; harvests under ``runs/`` and the
evaluation sets under ``runs/eval/<register>/text/``.

Run:
    python scripts/check_eval_overlap.py
    python scripts/check_eval_overlap.py --min-shared 3      # more sensitive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from build_wordlist import harvest_dirs  # noqa: E402

from tetrak_hy_trainer import heldout, wikisource  # noqa: E402

SHINGLE = 8


def shingles(text: str) -> set[tuple[str, ...]]:
    words = wikisource.normalise_transcript(text).lower().split()
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", type=Path, default=REPO / "runs" / "eval")
    parser.add_argument("--min-shared", type=int, default=5)
    args = parser.parse_args()

    evaluation: dict[str, set[tuple[str, ...]]] = {}
    for text_file in sorted(args.eval_root.glob("*/text/*.txt")):
        evaluation[f"{text_file.parent.parent.name} p.{text_file.stem}"] = shingles(
            text_file.read_text(encoding="utf-8")
        )
    print(f"{len(evaluation)} evaluation pages")

    unlisted = known = 0
    for directory in harvest_dirs():
        index = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["index"]
        if heldout.is_held_out(index):
            continue
        evaluation_pages = heldout.held_out_pages(index) or frozenset()
        for text_file in sorted((directory / "text").glob("*.txt"), key=lambda p: int(p.stem)):
            page = int(text_file.stem)
            if page in evaluation_pages:
                continue
            page_shingles = shingles(text_file.read_text(encoding="utf-8"))
            for name, eval_shingles in evaluation.items():
                shared = len(page_shingles & eval_shingles)
                if shared < args.min_shared:
                    continue
                listed = heldout.page_is_held_out(index, page)
                known += listed
                unlisted += not listed
                status = "listed" if listed else "NOT LISTED"
                print(f"{status:10}  {index} p.{page}  shares {shared} runs with {name}")
    print(f"{known} known overlap(s), {unlisted} not in OVERLAPPING_PAGES")
    return 1 if unlisted else 0


if __name__ == "__main__":
    raise SystemExit(main())
