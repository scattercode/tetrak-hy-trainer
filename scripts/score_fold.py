#!/usr/bin/env python3
"""Re-score v1's saved predictions with the script-consistency fold.

The v1 error analysis (tetrak's
product/research/armenian-v1-error-analysis.md) found that roughly half
of v1's 0.500-vs-0.662 word-recall gap against `tesseract -l hye` is
mechanical rather than a recognition failure, and measured how much of
it two fixes recover on the *predicted* side alone, with no retraining:

  (a) fold-only -- tetrak_hy.fold_script's cross-script homoglyph fold
      (h->հ, o->օ, :->։; dashes are deliberately not folded, see the
      function's docstring), shipped in tetrak-easyocr-armenian and
      applied here exactly as a consumer would apply it to every
      recognised string. This is the number the shipped fold earns
      today, with v1's weights unchanged.
  (b) fold-plus-punctuation -- (a) plus folding U+2024 ONE DOT LEADER
      to ASCII '.' on *both* the prediction and the truth, as a ceiling
      estimate for what the v2 charset (task 3) is expected to reach.
      This is a measurement convenience only, not a real fix: v1's
      charset has no U+2024, so it can never emit that character at
      all, and folding it here does not change what v1 predicted --
      it only equates the two spellings for scoring. The real fix is
      widening the charset and retraining.

Reads runs/eval/ase-vol2/predictions-v1.tar.gz's pred/<page>.txt files:
already one predicted box's text per line, in detector order, exactly as
joined for the numbers in brief 011's decision log entry for 2026-08-31
("v1 trained"). No inference is re-run.

Scores each page against runs/eval/ase-vol2/text/<page>.txt with
Tetrak's word_recall (order-insensitive), then averages across the ten
pages the same way scripts/evaluate_baselines.py does: the mean of the
per-page scores, not a pooled word count -- so this reproduces the
0.5014 baseline as a sanity check before reporting (a) and (b).

Prerequisites: tetrak-easyocr-armenian's own venv, which has easyocr
installed (fold.py itself needs nothing heavier than the standard
library, but importing the tetrak_hy package pulls in easyocr's model
re-export regardless of which name is imported from it); Tetrak's
`src/` on PYTHONPATH, for tetrak_ocr.accuracy.word_recall (stdlib-only,
no heavy dependency of its own). Run from the trainer repo root:

    PYTHONPATH=../tetrak/src \
        ../tetrak-easyocr-armenian/.venv/bin/python scripts/score_fold.py
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from tetrak_hy.fold import fold_script
from tetrak_ocr.accuracy import word_recall

EVAL_DIR = Path(__file__).resolve().parent.parent / "runs" / "eval" / "ase-vol2"
PREDICTIONS_TARBALL = EVAL_DIR / "predictions-v1.tar.gz"
TEXT_DIR = EVAL_DIR / "text"

# U+2024 ONE DOT LEADER, the transcripts' abbreviation dot -- absent from
# v1's charset, so v1 can never emit it. See the module docstring: folding
# it here only equates the two spellings for measurement (b), it does not
# change v1's actual predictions.
ABBREVIATION_DOT = "․"


def punctuation_fold(text: str) -> str:
    return text.replace(ABBREVIATION_DOT, ".")


def load_predictions() -> dict[str, str]:
    """Page number -> the saved v1 prediction text, one box per line."""
    predictions = {}
    with tarfile.open(PREDICTIONS_TARBALL) as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".txt"):
                continue
            page = Path(member.name).stem
            handle = archive.extractfile(member)
            predictions[page] = handle.read().decode("utf-8")
    return predictions


def main() -> None:
    predictions = load_predictions()
    pages = sorted(predictions, key=int)

    baseline_scores = []
    fold_only_scores = []
    fold_plus_punctuation_scores = []

    print(f"{'page':>5}  {'baseline':>8}  {'fold-only':>9}  {'fold+punct':>10}")
    for page in pages:
        predicted = predictions[page]
        expected = (TEXT_DIR / f"{page}.txt").read_text(encoding="utf-8")

        baseline = word_recall(predicted, expected)
        folded = fold_script(predicted)
        fold_only = word_recall(folded, expected)
        fold_plus_punctuation = word_recall(punctuation_fold(folded), punctuation_fold(expected))

        baseline_scores.append(baseline)
        fold_only_scores.append(fold_only)
        fold_plus_punctuation_scores.append(fold_plus_punctuation)

        print(f"{page:>5}  {baseline:>8.4f}  {fold_only:>9.4f}  {fold_plus_punctuation:>10.4f}")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    print()
    print(f"AVERAGE baseline           wrd={mean(baseline_scores):.4f}  n={len(pages)}")
    print(f"AVERAGE fold-only          wrd={mean(fold_only_scores):.4f}  n={len(pages)}")
    print(
        f"AVERAGE fold+punctuation   wrd={mean(fold_plus_punctuation_scores):.4f}  n={len(pages)}"
    )


if __name__ == "__main__":
    main()
