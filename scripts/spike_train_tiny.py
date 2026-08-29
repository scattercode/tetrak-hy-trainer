#!/usr/bin/env python3
"""Stage 1's finish line: train a tiny real model and read text back with it.

The loading spike proved a random-weight bundle loads through stock
EasyOCR; this proves the *training* half of the contract with a
deliberately tiny run: a couple of dozen Armenian words rendered a few
hundred times, the vendored trainer overfitting them for a few hundred
iterations on CPU, and the resulting checkpoint packaged and read back
through ``easyocr.Reader`` — asserting it actually recognises words it
was trained on. What it pins down:

  - the dataset contract (labels.csv, folder layout) really is what
    ``dataset.py`` reads;
  - the config generated from the canonical charset trains, and its CTC
    class order round-trips through the shipped yaml (a decoded word can
    only come out right if every index maps to the same character on
    both sides);
  - the trainer's DataParallel checkpoints carry the ``module.`` prefix
    EasyOCR strips, so a trained ``best_accuracy.pth`` is a valid
    ``tetrak_hy.pth`` as-is;
  - roughly what a CPU/MPS machine can train, before deciding whether
    real training rents a GPU.

Prerequisites: pip install -e '.[train]' plus easyocr, and an Armenian
font (defaults to macOS's Mshtakan).

Run:
    python scripts/spike_train_tiny.py [--iters 300] [--font PATH]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "training"))

from tetrak_hy_trainer import packaging, synth, train_config  # noqa: E402

# Real Armenian words, all inside the canonical charset, several carrying
# the և ligature so the round-trip exercises it.
WORDS = [
    "Հայաստան",
    "գիրք",
    "տպագրություն",
    "թանգարան",
    "արխիվ",
    "պատմություն",
    "լեզու",
    "մշակույթ",
    "գրադարան",
    "ձեռագիր",
    "քաղաք",
    "աշխարհ",
    "մարդ",
    "տարի",
    "նորություն",
    "մեծություն",
    "և",
    "երևույթ",
    "անձրև",
    "լույս",
    "գիշեր",
    "սարեր",
    "ծովափ",
    "արևելք",
]

DEFAULT_FONT = "/System/Library/Fonts/Supplemental/Mshtakan.ttc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps"],
        help="cpu (default) or Apple MPS. The vendored trainer binds a "
        "module-level device at import; the spike overrides it after import. "
        "DataParallel degrades to a passthrough wrapper off CUDA, so the "
        "module. checkpoint prefix survives either way.",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="tetrak_hy_train_spike_"))
    print(f"workdir: {workdir}")

    data_root = workdir / "all_data"
    synth.write_dataset(data_root, "hy_tiny_train", WORDS, args.font, repeats=args.repeats)
    synth.write_dataset(data_root, "hy_tiny_val", WORDS, args.font, repeats=2, seed=99)
    print(f"rendered {len(WORDS)} words x {args.repeats} train, x2 val")

    config = train_config.build_config(
        experiment_name="hy_tiny",
        train_data=str(data_root),
        valid_data=str(data_root / "hy_tiny_val"),
        select_data="hy_tiny_train",
        num_iter=args.iters,
        batch_size=32,
        val_interval=max(args.iters // 3, 50),
        workers=0,  # multiprocessing DataLoaders hang readily on macOS
    )

    # The vendored trainer composes opt the way its notebook does, and its
    # relative paths (saved_models/) resolve against the CWD.
    os.chdir(workdir)
    from tetrak_hy_trainer import trainer_compat

    trainer_compat.install()  # modern-torch shims; vendored files stay pristine
    from utils import AttrDict  # vendored  # noqa: E402

    opt = AttrDict(config)
    opt.character = opt.number + opt.symbol + opt.lang_char
    (workdir / "saved_models" / opt.experiment_name).mkdir(parents=True, exist_ok=True)

    import torch
    import train as train_module  # vendored  # noqa: E402
    from train import train  # vendored  # noqa: E402

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            print("FAILED: --device mps requested but MPS is unavailable", file=sys.stderr)
            return 1
        train_module.device = torch.device("mps")

    print(f"training {args.iters} iterations on {args.device}...")
    try:
        train(opt, amp=False)
    except SystemExit:
        pass  # upstream train() sys.exit()s on completion; we have more to do

    checkpoint = workdir / "saved_models" / "hy_tiny" / "best_accuracy.pth"
    if not checkpoint.exists():
        print("FAILED: trainer produced no best_accuracy.pth", file=sys.stderr)
        return 1

    # Package: the checkpoint is DataParallel-saved, so its keys already
    # carry the module. prefix EasyOCR strips. No rewriting.
    bundle = workdir / "bundle"
    packaging.write_bundle(bundle)
    shutil.copyfile(checkpoint, bundle / f"{packaging.NETWORK_NAME}.pth")

    import easyocr

    reader = easyocr.Reader(
        ["en"],  # see the loading spike: hy has no char file and needs none
        recog_network=packaging.NETWORK_NAME,
        user_network_directory=str(bundle),
        model_storage_directory=str(bundle),
        gpu=False,
        verbose=False,
    )

    # The criterion is mean edit-distance similarity, not exact match: what
    # this spike proves is CTC *alignment* -- misaligned class indices decode
    # as charset noise, aligned ones as near-misses of the trained word --
    # and a deliberately tiny model at a few hundred iterations produces
    # near-misses (measured: մշակույթ -> մշակույն), not perfection.
    probe_words = [
        "Հայաստան",
        "գրադարան",
        "երևույթ",
        "մշակույթ",
        "արխիվ",
        "գիշեր",
        "լեզու",
        "քաղաք",
        "թանգարան",
        "աշխարհ",
    ]
    import difflib

    from PIL import Image, ImageFont

    probe_font = ImageFont.truetype(args.font, 48)
    ratios = []
    for word in probe_words:
        crop = synth.render_word(word, probe_font)
        probe_path = workdir / "probe.png"
        # White margin around the crop so CRAFT has context to detect within.
        canvas = Image.new("L", (crop.width + 80, crop.height + 80), 245)
        canvas.paste(crop, (40, 40))
        canvas.convert("RGB").save(probe_path)
        decoded = " ".join(reader.readtext(str(probe_path), detail=0))
        ratio = difflib.SequenceMatcher(None, decoded, word).ratio()
        ratios.append(ratio)
        print(f"  {ratio:.2f}  trained {word!r} -> decoded {decoded!r}")

    mean = sum(ratios) / len(ratios)
    if mean < 0.45:
        print(
            f"\nSPIKE FAILED: mean similarity {mean:.3f} -- decoded text does not "
            "track the trained words, which points at the charset round-trip",
            file=sys.stderr,
        )
        return 1
    print(f"\nSPIKE PASSED: mean similarity {mean:.3f} over {len(probe_words)} trained words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
