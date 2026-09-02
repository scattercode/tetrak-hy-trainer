#!/usr/bin/env python3
"""Score a model on every register, not just the encyclopedia.

Brief 012 Stage 4. Until now one number stood for "how good is the
model": ten pages of ASE volume 2. That number cannot distinguish a
model that learnt Armenian better from one that learnt the ASE better,
and once the corpus took in Western Armenian literature, a bilingual
dictionary, a scholarly history and a second encyclopedia, the
difference is the whole point of the exercise.

This runs one model over every evaluation set under ``runs/eval/`` --
the ASE baseline plus the per-register sets ``build_eval_sets.py``
builds from the held-out registry -- and prints a table.

Both figures are reported for each set, and both matter:

* ``chr`` (character_similarity) is order-sensitive, so it collapses when
  the reading order is wrong. On a two-column page a perfect recogniser
  with no layout model still scores badly.
* ``wrd`` (word_recall) is order-insensitive, so it measures recognition
  almost independently of layout.

A register where ``wrd`` is healthy and ``chr`` is not has a layout
problem, not a recognition problem, and the two are fixed in different
places -- which is exactly the diagnosis the single-set evaluation could
never deliver.

``--fold`` additionally applies the shipped ``fold_script`` to both
sides. That is what a consumer of the model actually sees, since every
consumer folds; the raw figure is the model alone.

Prerequisites: needs easyocr and torch, so run it with the **sibling
library's venv**, the same interpreter ``harvest_real_crops.py`` needs:

    ../tetrak-easyocr-armenian/.venv/bin/python scripts/evaluate_registers.py \
        --bundle runs/v4/bundle

    # compare two models on every register
    ../tetrak-easyocr-armenian/.venv/bin/python scripts/evaluate_registers.py \
        --bundle runs/v3/bundle --bundle runs/v4/bundle --fold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Run under the sibling library's venv, which has easyocr but not this
# package installed -- the same arrangement harvest_real_crops.py needs.
sys.path.insert(0, str(REPO_ROOT / "src"))

from tetrak_hy_trainer import evaluate  # noqa: E402

EVAL_ROOT = REPO_ROOT / "runs" / "eval"


def evaluation_sets(eval_root: Path) -> list[Path]:
    """Every directory under *eval_root* shaped like an evaluation set."""
    return sorted(
        directory
        for directory in eval_root.iterdir()
        if (directory / "manifest.json").exists() and (directory / "images").is_dir()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        action="append",
        required=True,
        help="a packaged bundle directory; repeatable to compare models",
    )
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--set", action="append", help="only this evaluation set; repeatable")
    parser.add_argument(
        "--fold",
        action="store_true",
        help="also apply the shipped script fold, as every consumer does",
    )
    args = parser.parse_args(argv)

    transform = None
    if args.fold:
        # Imported lazily and only when asked: the fold ships in the
        # sibling library, so a checkout without it can still produce
        # the raw numbers.
        from tetrak_hy.fold import fold_script

        transform = fold_script

    sets = evaluation_sets(args.eval_root)
    if args.set:
        wanted = set(args.set)
        unknown = wanted - {directory.name for directory in sets}
        if unknown:
            parser.error(f"no evaluation set named: {', '.join(sorted(unknown))}")
        sets = [directory for directory in sets if directory.name in wanted]
    if not sets:
        print(f"no evaluation sets under {args.eval_root}", file=sys.stderr)
        return 1

    label = "fold" if args.fold else "raw"
    for bundle in args.bundle:
        print(f"\n=== {bundle} ({label}) ===")
        print(f"{'register':<24} {'pages':>5} {'chr':>7} {'wrd':>7}")
        reader = evaluate.load_reader(bundle)
        results = []
        for eval_dir in sets:
            result = evaluate.score_pages(bundle, eval_dir, reader=reader, transform=transform)
            if not len(result):
                print(f"{eval_dir.name:<24} {'-':>5}   no page images")
                continue
            results.append(result)
            print(
                f"{result.name:<24} {len(result):>5} "
                f"{result.mean_character_similarity:>7.4f} {result.mean_word_recall:>7.4f}"
            )
        if results:
            # An unweighted mean over registers, deliberately: each
            # register is one question about the model, and weighting by
            # page count would let the largest set speak for the rest --
            # which is the single-set problem this script exists to end.
            print(
                f"{'MEAN OVER REGISTERS':<24} {len(results):>5} "
                f"{sum(r.mean_character_similarity for r in results) / len(results):>7.4f} "
                f"{sum(r.mean_word_recall for r in results) / len(results):>7.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
