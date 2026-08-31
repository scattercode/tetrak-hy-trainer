#!/usr/bin/env python3
"""Fine-tune a trained checkpoint on real crops, mixed with synthetic ones.

Brief 011 Stage 3 step 2, the second half: ``harvest_real_crops.py`` cuts
labelled crops from the archival scans, and this trains on them. It is
the lever v1's error analysis left standing -- the shape confusions
clustered on ``հ`` that no amount of rendering reaches, because synthetic
fonts draw an ascender the 1970s letterpress does not.

**Start from v2, not v1.** v2 widened the charset with U+2024, the
transcripts' abbreviation dot, which v1 could not emit at all -- 5.8% of
the evaluation pages' words unwinnable by construction. A fine-tune
inherits its parent's charset, so fine-tuning v1 would carry that hole
forward, and the CTC head's width would not match the v2 yaml either.

Real and synthetic are mixed **in every batch** rather than trained in
sequence, via the vendored trainer's ``select_data``/``batch_ratio``:
a few thousand real crops against v1's 175,500 synthetic ones is a small,
narrow distribution, and training on it alone is the classic recipe for
catastrophic forgetting -- the model would sharpen on volume 6's
typesetting and lose the breadth the synthetic pre-train bought. The
default 50/50 split by batch means each step sees both.

Both datasets must sit under **one root**: the trainer takes a single
``train_data`` and matches ``select_data`` names against directories
beneath it. So harvest the real crops into the v2 run's ``all_data/``,
where ``syn_train`` already is::

    python scripts/harvest_real_crops.py \\
        --harvest-dir runs/v1/harvest-vol6 --bundle runs/v2/bundle \\
        --out runs/v2/all_data

Then (this is an hour or two on MPS, not v2's eleven -- a fine-tune runs
far fewer iterations than a pre-train)::

    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -ims \\
        python scripts/finetune_real.py --device mps \\
        --data-root runs/v2/all_data \\
        --saved-model runs/v2/saved_models/v2/best_accuracy.pth \\
        --eval-dir runs/eval/ase-vol2

Validation is on **real** crops by default (``real_val``, split off by
page, never by crop). Synthetic validation accuracy is already 98.8% and
would say nothing about the thing being fixed.

Everything lands under ``runs/<run-name>/``, and the packaged bundle is a
drop-in ``tetrak_hy`` for the library, exactly as the pre-train's is.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "scripts"))

from train_synthetic import evaluate_pages, package  # noqa: E402

from tetrak_hy_trainer import train_config  # noqa: E402


def count_labels(folder: Path) -> int:
    """Rows in a dataset folder's labels.csv, excluding its header."""
    labels = folder / "labels.csv"
    if not labels.exists():
        return 0
    with labels.open(encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v3")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="dataset root holding real_train/ (and syn_train/ to mix with)",
    )
    parser.add_argument(
        "--saved-model",
        type=Path,
        required=True,
        help="checkpoint to fine-tune from -- v2's, not v1's; see the module docstring",
    )
    parser.add_argument("--valid-data", type=Path, default=None, help="default: <root>/real_val")
    parser.add_argument("--select-data", default="real_train-syn_train")
    parser.add_argument("--batch-ratio", default="0.5-0.5")
    parser.add_argument("--iters", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()

    # Resolved before training chdir()s into the run directory -- a relative
    # path would otherwise resolve against that and fail after the run, with
    # the number lost. The same bug the pre-train hit once already.
    data_root = args.data_root.resolve()
    saved_model = args.saved_model.resolve()
    valid_data = (args.valid_data or data_root / "real_val").resolve()
    eval_dir = args.eval_dir.resolve() if args.eval_dir else None

    run_dir = (REPO / "runs" / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.package_only:
        package(run_dir, args.run_name)
        return 0

    if not saved_model.exists():
        raise SystemExit(f"no checkpoint at {saved_model}")

    selected = args.select_data.split("-")
    ratios = args.batch_ratio.split("-")
    if len(selected) != len(ratios):
        raise SystemExit(
            f"--select-data names {len(selected)} dataset(s) but --batch-ratio gives "
            f"{len(ratios)} ratio(s); the trainer asserts they match"
        )

    for name in selected:
        folder = data_root / name
        count = count_labels(folder)
        if not count:
            raise SystemExit(
                f"{folder} holds no labelled crops. Both datasets must live under the "
                f"one root the trainer is given -- harvest real crops into {data_root} "
                f"so they sit beside the synthetic ones."
            )
        print(f"{name}: {count} crops", flush=True)
    print(f"validating on {valid_data} ({count_labels(valid_data)} crops)", flush=True)

    config = train_config.build_config(
        experiment_name=args.run_name,
        train_data=str(data_root),
        valid_data=str(valid_data),
        select_data=args.select_data,
        num_iter=args.iters,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        workers=0,
        batch_max_length=60,
        saved_model=str(saved_model),  # sets FT=True in the config
    )
    config["batch_ratio"] = args.batch_ratio
    config["imgW"] = 800  # room for four-token lines at 64 px height, as v1/v2

    os.chdir(run_dir)
    from tetrak_hy_trainer import trainer_compat

    trainer_compat.install()
    import torch
    import train as train_module  # vendored  # noqa: E402
    from train import train  # vendored  # noqa: E402
    from utils import AttrDict  # vendored  # noqa: E402

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps requested but MPS is unavailable")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise SystemExit("MPS needs PYTORCH_ENABLE_MPS_FALLBACK=1 (CTC loss has no MPS kernel)")
        train_module.device = torch.device("mps")

    opt = AttrDict(config)
    opt.character = opt.number + opt.symbol + opt.lang_char
    (run_dir / "saved_models" / opt.experiment_name).mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"fine-tuning {args.iters} iterations from {saved_model.name} on {args.device}...")
    try:
        train(opt, amp=False)
    except SystemExit:
        pass  # upstream train() sys.exit()s on completion

    bundle = package(run_dir, args.run_name)
    if eval_dir is not None:
        evaluate_pages(bundle, eval_dir)
    print(f"total wall time: {(time.time() - started) / 3600:.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
