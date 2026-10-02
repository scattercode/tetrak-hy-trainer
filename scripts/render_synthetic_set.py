#!/usr/bin/env python3
"""Render one extra synthetic set into a run's ``all_data/``, beside ``syn_train``.

Brief 013 Stage 3's recipe lever: all-caps line samples. Encyclopedia
headwords are set in capitals, and v5's confusions there (``Հ``→``Վ``,
``Ս``→``Մ``, case flips) come from a pre-train whose rendered lines were
almost all lower case, because running text is. This renders lines from
the same harvested text, through the same fonts, sizes and degradations
as ``train_synthetic.py``, into a named set that ``finetune_real.py``
mixes in with ``--select-data`` and ``--batch-ratio``.

Held-out pages are excluded by the same sampler ``train_synthetic.py``
uses, so no evaluation transcript is rendered.

Prerequisites: this repository's venv with the ``[train]`` extra, the
fonts from ``scripts/fetch_fonts.py`` under ``runs/v0/fonts/``, and the
harvests to sample from.

Run:
    python scripts/render_synthetic_set.py --run-name v6-b013 --name syn_caps \\
        --upper --max-samples 30000 --repeats 2 \\
        --harvest-dirs runs/harvest/* runs/v0/harvest runs/v1/harvest-vol5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from train_synthetic import (  # noqa: E402
    build_line_samples,
    find_fonts,
    render_corpus,
    upper_case_line,
)

from tetrak_hy_trainer import charset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--name", required=True, help="train folder; validation is <name>_val")
    parser.add_argument("--harvest-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=30_000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--line-tokens-max", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=18)
    parser.add_argument("--upper", action="store_true", help="render every line in capitals")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    run_dir = (REPO / "runs" / args.run_name).resolve()
    harvest_dirs = [d for d in args.harvest_dirs if (d / "text").is_dir()]
    print(f"harvest dirs: {[str(d) for d in harvest_dirs]}", flush=True)

    started = time.time()
    rng = random.Random(args.seed)
    samples = build_line_samples(harvest_dirs, args.max_samples, rng, args.line_tokens_max)
    if args.upper:
        allowed = set(charset.character_list())
        upper = [upper_case_line(line) for line in samples]
        # A capital the charset lacks would be filtered by the trainer
        # without a word; say so here instead.
        strays = sorted({c for line in upper for c in line} - allowed)
        if strays:
            raise SystemExit(f"capitals outside the charset: {strays!r}")
        samples = upper
    print(f"line samples: {len(samples)}; e.g. {samples[:3]!r}", flush=True)

    sizes = tuple(s for s in (18, 22, 28, 36, 48, 64) if s >= args.min_size)
    root, crops = render_corpus(
        run_dir,
        samples,
        find_fonts(),
        sizes,
        args.repeats,
        use_augment=True,
        seed=args.seed,
        names=(args.name, f"{args.name}_val"),
    )
    print(f"rendered {crops} crops into {root / args.name} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
