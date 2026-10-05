#!/usr/bin/env python3
"""Reuse a rendered dataset under a new run, with its labels re-normalised.

Re-rendering v5's 351,000 synthetic crops to change one character in
their labels would cost an hour and change nothing in the images. This
hard-links the images into the new run's folder (no extra disk; the
original folder is untouched) and writes a fresh ``labels.csv`` with every
label passed through ``wikisource.normalise_transcript``.

Brief 013 needed it for the full stop: real-crop labels now carry the
Armenian ``։`` wherever the transcribers typed ``:``, and a synthetic set
still teaching ``:`` for the same glyph in every other batch would pull
the other way.

Prerequisites: this repository's venv; source and destination on the same
filesystem (hard links cannot cross one).

Run:
    python scripts/relabel_dataset.py runs/v6/all_data/syn_train runs/v6-b013/all_data/syn_train
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import synth, wikisource  # noqa: E402


def read_labels(folder: Path) -> list[tuple[str, str]]:
    """``(filename, label)`` rows, split as the trainer splits them: first comma."""
    lines = (folder / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]
    return [tuple(line.split(",", 1)) for line in lines if line]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    if destination.exists():
        raise SystemExit(f"{destination} exists; refusing to mix two datasets in one folder")
    rows = read_labels(source)
    destination.mkdir(parents=True)
    for filename, _ in rows:
        os.link(source / filename, destination / filename)
    # A label has lost its source, and any angle bracket still in one
    # survived its source's normalisation on purpose (before charset v4 the
    # charset filter dropped them all), so it is kept rather than folded.
    relabelled = [
        (filename, wikisource.normalise_transcript(label, angle_brackets_printed=True))
        for filename, label in rows
    ]
    changed = sum(old != new for (_, old), (_, new) in zip(rows, relabelled, strict=True))
    synth.write_labels(destination, relabelled)
    print(f"{len(rows)} crops linked into {destination}; {changed} labels changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
