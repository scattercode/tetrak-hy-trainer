#!/usr/bin/env python3
"""Recompute the character confusion table -- the fine-tune's scorecard.

v1's error analysis reduced the model's remaining gap to a table of
character confusions and split it in two: homoglyphs (``հ``→``h``,
``։``→``:``), which ``tetrak_hy.fold_script`` fixes at serialisation time
with no retraining, and shape confusions (``հ``→``խ``, ``խ``→``ի``,
``տ``→``ո``, ``ճ``↔``ջ``), which only real crops reach. Brief 011 makes
that table the fine-tune's progress metric, because a single word-recall
number cannot say whether the shape cluster moved -- and the shape
cluster is the whole reason for Stage 3 step 2.

Runs over the evaluation pages, pairing each prediction against the
proofread transcript with the same alignment the crop harvester uses
(:mod:`tetrak_hy_trainer.align`), then counting character confusions
within the pairs (:mod:`tetrak_hy_trainer.confusion`). Reads a saved
predictions tarball where one exists, so v1's published numbers can be
reproduced without re-running inference, or a bundle to score a fresh
model.

Reading v1's saved predictions, which needs no OCR stack beyond the
alignment itself::

    PYTHONPATH=../tetrak/src ../tetrak-easyocr-armenian/.venv/bin/python \\
        scripts/confusion_report.py --predictions runs/eval/ase-vol2/predictions-v1.tar.gz

Scoring a model directly::

    ../tetrak-easyocr-armenian/.venv/bin/python scripts/confusion_report.py \\
        --bundle runs/v3/bundle --eval-dir runs/eval/ase-vol2

Pass ``--fold`` to count what remains *after* the homoglyph fold the
library ships, which is what the shipped pipeline actually emits: the
difference between the two runs is how much of the table the fold
already handles, and what is left is the fine-tune's target.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import align, confusion  # noqa: E402
from tetrak_hy_trainer.align import Detection, Tier  # noqa: E402

DEFAULT_EVAL_DIR = REPO / "runs" / "eval" / "ase-vol2"

# Confusions between these are the homoglyph half of the table -- the fold
# handles them and no retraining is involved. Everything else is a shape
# confusion, which is what a real-crop fine-tune has to move.
HOMOGLYPHS = {("հ", "h"), ("Հ", "H"), ("օ", "o"), ("Օ", "O"), ("։", ":")}


def load_predicted_pages(tarball: Path) -> dict[int, list[Detection]]:
    """Per-page detections from a saved predictions tarball.

    The JSON members carry the box and confidence per detection, which the
    plain text ones do not -- and the alignment needs the geometry to put
    the boxes into column order.
    """
    pages: dict[int, list[Detection]] = {}
    with tarfile.open(tarball) as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".json"):
                continue
            page = int(Path(member.name).stem)
            handle = archive.extractfile(member)
            pages[page] = [
                Detection(
                    text=entry["text"],
                    bbox=(
                        min(point[0] for point in entry["box"]),
                        min(point[1] for point in entry["box"]),
                        max(point[0] for point in entry["box"]),
                        max(point[1] for point in entry["box"]),
                    ),
                    confidence=entry.get("conf"),
                )
                for entry in json.loads(handle.read().decode("utf-8"))
            ]
    return pages


def read_pages_with_model(bundle: Path, eval_dir: Path, gpu: bool) -> dict[int, list[Detection]]:
    """Run a bundle over the evaluation scans."""
    sys.path.insert(0, str(REPO / "scripts"))
    from harvest_real_crops import build_reader, detect_page

    reader = build_reader(bundle, gpu=gpu)
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    pages = {}
    for entry in manifest["pages"]:
        image = eval_dir / "images" / f"{entry['page_number']}.jpg"
        if image.exists():
            pages[entry["page_number"]] = detect_page(reader, image)
            print(f"  read p{entry['page_number']}", flush=True)
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="a saved predictions tarball")
    source.add_argument("--bundle", type=Path, help="a tetrak_hy bundle to read the pages with")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--limit", type=int, default=25, help="rows in the table")
    parser.add_argument(
        "--fold",
        action="store_true",
        help="apply tetrak_hy.fold_script first, leaving only what it does not fix",
    )
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    fold = None
    if args.fold:
        from tetrak_hy import fold_script

        fold = fold_script

    if args.predictions:
        pages = load_predicted_pages(args.predictions)
    else:
        pages = read_pages_with_model(args.bundle, args.eval_dir, gpu=args.gpu)
    if not pages:
        raise SystemExit("no pages to score")

    word_pairs: list[tuple[str, str]] = []
    aligned = 0
    for page_number, detections in sorted(pages.items()):
        truth_path = args.eval_dir / "text" / f"{page_number}.txt"
        if not truth_path.exists():
            continue
        if fold is not None:
            detections = [
                Detection(text=fold(d.text), bbox=d.bbox, confidence=d.confidence)
                for d in detections
            ]
        crops = align.align_page(detections, truth_path.read_text(encoding="utf-8"))
        aligned += len(crops)
        # Only misreadings carry information; exact pairs contribute nothing
        # to a confusion table by construction.
        word_pairs.extend(
            (crop.predicted, crop.label) for crop in crops if crop.tier is not Tier.EXACT
        )

    confusions = confusion.character_confusions(word_pairs)
    shape = sum(count for key, count in confusions.items() if key not in HOMOGLYPHS)
    homoglyph = sum(count for key, count in confusions.items() if key in HOMOGLYPHS)

    print(f"\n{len(pages)} pages, {aligned} aligned boxes, {len(word_pairs)} misread")
    print(f"{homoglyph} homoglyph confusions (the fold's job), {shape} shape confusions\n")
    print(confusion.format_table(confusions, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
