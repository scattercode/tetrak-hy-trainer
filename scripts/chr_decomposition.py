#!/usr/bin/env python3
"""Break a model's character-similarity gap into named components.

Brief 012 Stage 0.2. Character similarity on the evaluation pages is one
opaque number sitting far below the Tesseract-family engines, and 011's
error analysis established that reading order dominated it — then fixed
reading order. What remains needed naming before anything else got
built, and the first run of this decomposition (2026-09-01) reordered
brief 012: **oracle ordering scores no better than the shipped column
serialisation**, so the residual gap is not order at all.

Five measurements per model, cumulative where marked:

  shipped     column-order serialisation of every box, homoglyph fold
              applied -- what the easyocr-hy backend emits, minus the
              engine differences of a live run
  oracle      the same boxes reordered by where their aligned text sits
              in the transcript; unalignable boxes kept at the end.
              The ceiling ordering alone can reach
  + dedupe    oracle, minus near-duplicate boxes: same reading, heavily
              overlapping area -- CRAFT double-detections that charge
              chr for insertions wherever they are placed
  + dehyphen  additionally rejoining a box ending in a dash with the
              box its text continues into, dropping the dash, the way
              the transcripts reflow line-break hyphenation
  truth-only  chr of the aligned labels themselves in truth order: what
              these boxes would score if every one were read perfectly.
              The detection ceiling -- everything above it is
              unreachable without better boxes

Needs the tetrak_hy package for the fold (the library's venv, or any
with it installed) and Tetrak's src on PYTHONPATH for the metric:

    PYTHONPATH=../tetrak/src ../tetrak-easyocr-armenian/.venv/bin/python \\
        scripts/chr_decomposition.py --predictions \\
        runs/eval/ase-vol2/predictions-v3.tar.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from confusion_report import load_predicted_pages  # noqa: E402

from tetrak_hy_trainer import align  # noqa: E402
from tetrak_hy_trainer.align import Detection  # noqa: E402

EVAL_DIR = REPO / "runs" / "eval" / "ase-vol2"

_DASHES = "-–—֊"


def overlap_fraction(a: Detection, b: Detection) -> float:
    """Intersection area over the smaller box's area."""
    left = max(a.bbox[0], b.bbox[0])
    top = max(a.bbox[1], b.bbox[1])
    right = min(a.bbox[2], b.bbox[2])
    bottom = min(a.bbox[3], b.bbox[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area = min(
        (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]),
        (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]),
    )
    return intersection / area if area else 0.0


def dedupe(ordered: list[Detection]) -> list[Detection]:
    """Drop boxes whose text a heavily-overlapping earlier box already gave."""
    kept: list[Detection] = []
    for detection in ordered:
        duplicate = any(
            overlap_fraction(detection, other) > 0.6 and detection.text == other.text
            for other in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def dehyphenate(texts: list[str]) -> list[str]:
    """Rejoin a fragment ending in a dash with the fragment that follows it."""
    joined: list[str] = []
    for text in texts:
        stripped = text.strip()
        if joined and joined[-1].rstrip() and joined[-1].rstrip()[-1] in _DASHES:
            previous = joined[-1].rstrip()
            joined[-1] = previous[:-1] + stripped
        else:
            joined.append(stripped)
    return joined


def oracle_order(detections: list[Detection], truth: str) -> tuple[list[Detection], list[str]]:
    """Boxes reordered by aligned truth position; plus their aligned labels."""
    ordered = align.column_order(detections)
    tokens: list[str] = []
    owner: list[int] = []
    for index, detection in enumerate(ordered):
        for token in detection.text.split():
            tokens.append(token)
            owner.append(index)
    truth_tokens = truth.split()
    first_position: dict[int, int] = {}
    labels: dict[int, list[str]] = {}
    for predicted_index, truth_index, _ in align.pair_tokens(tokens, truth_tokens):
        box = owner[predicted_index]
        first_position.setdefault(box, truth_index)
        labels.setdefault(box, []).append(truth_tokens[truth_index])
    matched = sorted(first_position, key=first_position.__getitem__)
    unmatched = [i for i in range(len(ordered)) if i not in first_position]
    reordered = [ordered[i] for i in matched + unmatched]
    aligned_labels = [" ".join(labels[i]) for i in matched]
    return reordered, aligned_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()

    from tetrak_hy import fold_script

    from tetrak_hy_trainer.accuracy import character_similarity

    pages = load_predicted_pages(args.predictions)
    stages: dict[str, list[float]] = {}

    def score(stage: str, text: str, truth: str) -> None:
        stages.setdefault(stage, []).append(character_similarity(text, truth))

    for page, detections in sorted(pages.items()):
        truth = (args.eval_dir / "text" / f"{page}.txt").read_text(encoding="utf-8")
        folded = [
            Detection(text=fold_script(d.text), bbox=d.bbox, confidence=d.confidence)
            for d in detections
        ]

        shipped = align.column_order(folded)
        score("shipped (column order + fold)", "\n".join(d.text for d in shipped), truth)

        reordered, aligned_labels = oracle_order(folded, truth)
        score("oracle ordering", "\n".join(d.text for d in reordered), truth)

        deduped = dedupe(reordered)
        score("+ duplicate suppression", "\n".join(d.text for d in deduped), truth)

        rejoined = dehyphenate([d.text for d in deduped])
        score("+ de-hyphenation", "\n".join(rejoined), truth)

        score("truth-only (detection ceiling)", "\n".join(aligned_labels), truth)

    print(f"{'stage':<34} {'chr':>7}")
    for stage, values in stages.items():
        print(f"{stage:<34} {sum(values) / len(values):>7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
