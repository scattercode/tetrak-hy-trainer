#!/usr/bin/env python3
"""Render brief 019's face-held-out test set: lines in faces no model trains on.

Brief 019 Stage 0. The question is whether the recogniser reads a letter
shape it has never been shown -- the 2021 epigraph whose forward-hooked
``ն`` v6 read as ``ճ`` -- and a real register cannot answer it, because no
register is set in a face we chose to keep out. This renders the same lines
twice:

- ``display/``: each line in one of the **held-out** display faces
  (``runs/fonts-display/heldout``: Kelvinch, Iosevka Etoile, Sharqayin,
  Catrinity, whole families that ``fetch_display_fonts.py`` keeps out of
  every training folder), the same number of lines per family;
- ``control/``: the same line in one of the book faces every model has
  trained on.

The control separates "this face is hard" from "this text is hard". Both
halves use the training sizes and the training degradations, and the same
seed, so a re-render is the same set crop for crop. Lines come from the
harvests through the same sampler as training, so held-out evaluation pages
are excluded here too.

Each folder holds ``labels.csv`` in the trainer's format and ``faces.tsv``
(file, family, face). ``evaluate_face_set.py`` scores a bundle on it.

Prerequisites: this repository's venv with the ``[train]`` extra, the book
faces (``fetch_fonts.py``), the display faces (``fetch_display_fonts.py``)
and the harvests.

Run from the repo root:

    python scripts/render_face_testset.py --out runs/b019/face_test \\
        --harvest-dirs runs/harvest/* runs/v0/harvest runs/v1/harvest-*
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from render_synthetic_set import families_beside  # noqa: E402
from train_synthetic import build_line_samples, find_fonts  # noqa: E402

from tetrak_hy_trainer import augment, charset, synth  # noqa: E402

SIZES = (18, 22, 28, 36, 48, 64)


def _faces(fonts: list[Path]) -> list[tuple[Path, frozenset[str]]]:
    full = charset.character_list()
    return [
        (path, synth.missing_glyphs(str(path), full) | synth.case_gaps(str(path))) for path in fonts
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "b019" / "face_test")
    parser.add_argument("--harvest-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--heldout-dir", type=Path, default=REPO / "runs" / "fonts-display" / "heldout"
    )
    parser.add_argument("--lines-per-family", type=int, default=400)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    from PIL import ImageFont

    rng = random.Random(args.seed)
    harvest_dirs = [d for d in args.harvest_dirs if (d / "text").is_dir()]
    families = families_beside(args.heldout_dir) or {}
    display = _faces(find_fonts(args.heldout_dir.resolve()))
    control = _faces(find_fonts())
    by_family: dict[str, list[tuple[Path, frozenset[str]]]] = {}
    for path, gaps in display:
        by_family.setdefault(families.get(path.name, path.name), []).append((path, gaps))
    print(f"held-out families: {sorted(by_family)}", flush=True)

    wanted = args.lines_per_family * len(by_family)
    samples = build_line_samples(harvest_dirs, wanted * 2, rng, tokens_max=3)
    allowed = set(charset.character_list())
    samples = [line for line in samples if set(line) <= allowed]

    rows: dict[str, list[tuple[str, str]]] = {"display": [], "control": []}
    faces_rows: dict[str, list[tuple[str, str, str]]] = {"display": [], "control": []}
    fonts_cache: dict[tuple[Path, int], ImageFont.FreeTypeFont] = {}
    lines = iter(samples)
    for family in sorted(by_family):
        done = 0
        while done < args.lines_per_family:
            line = next(lines)
            needed = set(line)
            shown = [p for p, gaps in by_family[family] if not gaps & needed]
            plain = [p for p, gaps in control if not gaps & needed]
            if not shown or not plain:
                continue
            name = f"{family.replace(' ', '')}_{done:04d}.png"
            for half, path in (("display", rng.choice(shown)), ("control", rng.choice(plain))):
                size = rng.choice(SIZES)
                font = fonts_cache.setdefault((path, size), ImageFont.truetype(str(path), size))
                image = augment.degrade(synth.render_word(line, font, jitter=rng), rng)
                folder = args.out / half
                folder.mkdir(parents=True, exist_ok=True)
                image.save(folder / name)
                rows[half].append((name, line))
                faces_rows[half].append((name, family, path.name))
            done += 1

    for half in rows:
        synth.write_labels(args.out / half, rows[half])
        with (args.out / half / "faces.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("file", "family", "face"))
            writer.writerows(faces_rows[half])
    print(f"rendered {len(rows['display'])} lines, twice, into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
