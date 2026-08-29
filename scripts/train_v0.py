#!/usr/bin/env python3
"""The v0 overnight pre-train: real vocabulary, simple rendering, long run.

Builds a vocabulary from harvested proofread encyclopedia text (real
Armenian word forms at real frequencies, punctuation attached as printed),
renders each token across every provided font at several sizes with
``synth.render_word``, and trains for the requested iterations, packaging
the best checkpoint as a loadable ``tetrak_hy`` bundle at the end.

Honest scope: this is *v0*. Rendering is clean-page synthetic — the
archival degradation pipeline is Stage 2 — so expect a model that reads
clean print long before it reads bilevel scans. Its purpose is a first
measurable row for the benchmark and a full-length shakedown of the
training recipe.

Everything lands under ``--run-dir`` (default ``runs/v0``, gitignored):

    harvest/            input text (from tetrak_hy_trainer.harvest)
    all_data/           rendered crops + labels.csv (trainer format)
    saved_models/v0/    checkpoints; best_accuracy.pth updates throughout,
                        so an interrupted run still leaves a usable model
    bundle/             the packaged tetrak_hy.{yaml,py,pth} on completion
    train.log           stdout of the whole run (the caller redirects)

Run (overnight, Mac):
    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -ims \\
        python scripts/train_v0.py --device mps --iters 150000

Interrupted or crashed? The checkpoint survives; re-package it with
``--package-only``.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "training"))

from tetrak_hy_trainer import charset, packaging, synth, train_config  # noqa: E402


def build_vocabulary(harvest_dir: Path, max_words: int) -> list[str]:
    """Frequency-ranked charset-clean tokens from harvested page text."""
    allowed = set(charset.character_list())
    counts: collections.Counter[str] = collections.Counter()
    for text_file in sorted((harvest_dir / "text").glob("*.txt")):
        for token in text_file.read_text(encoding="utf-8").split():
            if 1 <= len(token) <= 24 and set(token) <= allowed:
                counts[token] += 1
    vocabulary = [word for word, _ in counts.most_common(max_words)]
    if len(vocabulary) < 500:
        raise SystemExit(
            f"only {len(vocabulary)} usable tokens in {harvest_dir} -- harvest more pages first"
        )
    return vocabulary


def render_corpus(
    run_dir: Path, vocabulary: list[str], fonts: list[Path], repeats: int, seed: int = 0
) -> tuple[Path, int]:
    """Render the train and val folders; return (data_root, crop_count).

    Each token is rendered `repeats` times, each time with a randomly
    drawn font and size, into one combined training folder. Every 50th
    token is additionally held out into the validation folder (rendered
    once per font), so validation words are seen forms but unseen
    renderings — the right check for a recogniser at this stage.
    """
    from PIL import ImageFont

    rng = random.Random(seed)
    sizes = (36, 44, 52, 64)
    faces = [ImageFont.truetype(str(path), size) for path in fonts for size in sizes]

    data_root = run_dir / "all_data"
    train_dir = data_root / "v0_train"
    val_dir = data_root / "v0_val"
    for directory in (train_dir, val_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    for index, word in enumerate(vocabulary):
        for repeat in range(repeats):
            name = f"{index:06d}_{repeat}.png"
            synth.render_word(word, rng.choice(faces), jitter=rng).save(train_dir / name)
            train_rows.append((name, word))
        if index % 50 == 0:
            for f_index, face in enumerate(faces[:: len(sizes)]):  # one size per font
                name = f"v{index:06d}_{f_index}.png"
                synth.render_word(word, face, jitter=rng).save(val_dir / name)
                val_rows.append((name, word))

    import csv

    for directory, rows in ((train_dir, train_rows), (val_dir, val_rows)):
        with (directory / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "words"])
            writer.writerows(rows)
    return data_root, len(train_rows)


def package(run_dir: Path) -> Path:
    """Package the best checkpoint as a tetrak_hy bundle; return its dir."""
    checkpoint = run_dir / "saved_models" / "v0" / "best_accuracy.pth"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    bundle = run_dir / "bundle"
    packaging.write_bundle(bundle)
    shutil.copyfile(checkpoint, bundle / f"{packaging.NETWORK_NAME}.pth")
    print(f"packaged: {bundle}")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs" / "v0")
    parser.add_argument("--iters", type=int, default=150_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--max-words", type=int, default=30_000)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-interval", type=int, default=2_000)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    if args.package_only:
        package(run_dir)
        return 0

    started = time.time()
    vocabulary = build_vocabulary(run_dir / "harvest", args.max_words)
    print(f"vocabulary: {len(vocabulary)} tokens", flush=True)

    fonts = sorted((run_dir / "fonts").glob("*.tt*"))
    system_font = Path("/System/Library/Fonts/Supplemental/Mshtakan.ttc")
    if system_font.exists():
        fonts.append(system_font)
    if not fonts:
        raise SystemExit(f"no fonts in {run_dir / 'fonts'}")
    print(f"fonts: {[f.name for f in fonts]}", flush=True)

    data_root, crops = render_corpus(run_dir, vocabulary, fonts, args.repeats)
    print(f"rendered {crops} training crops in {time.time() - started:.0f}s", flush=True)

    config = train_config.build_config(
        experiment_name="v0",
        train_data=str(data_root),
        valid_data=str(data_root / "v0_val"),
        select_data="v0_train",
        num_iter=args.iters,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        workers=0,
        batch_max_length=26,
    )
    train_config.write_config(
        run_dir / "v0_config.yaml",
        **{
            key: config[key]
            for key in ("experiment_name", "train_data", "valid_data", "select_data", "num_iter")
        }
        | dict(
            batch_size=args.batch_size,
            val_interval=args.val_interval,
            workers=0,
            batch_max_length=26,
        ),
    )

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

    print(f"training {args.iters} iterations on {args.device}...", flush=True)
    try:
        train(opt, amp=False)
    except SystemExit:
        pass  # upstream train() sys.exit()s on completion

    package(run_dir)
    print(f"total wall time: {(time.time() - started) / 3600:.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
