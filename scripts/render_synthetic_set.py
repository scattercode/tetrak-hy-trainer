#!/usr/bin/env python3
"""Render one extra synthetic set into a run's ``all_data/``, beside ``syn_train``.

Brief 013 Stage 3's recipe lever: all-caps line samples. Encyclopedia
headwords are set in capitals, and v5's confusions there (``Հ``→``Վ``,
``Ս``→``Մ``, case flips) come from a pre-train whose rendered lines were
almost all lower case, because running text is. This renders lines from
the same harvested text, through the same fonts, sizes and degradations
as ``train_synthetic.py``, into a named set that ``finetune_real.py``
mixes in with ``--select-data`` and ``--batch-ratio``.

``--index`` renders the other shape brief 013's error analysis asked for:
index entries and page-number lists ("Կոգ (գավառ) — 159, 231, 288",
"113, 114, 115,"), with the digit 2 over-represented, and words carrying
an attached em dash ("մեք,—"). The Faustus of Byzantium index and notes
are set in bold italic, where v6 read 2 as Չ, շ, 8 or 7 -- a shape the
running text the pre-train sampled almost never contains. ``--faces``
restricts rendering to faces whose file names contain any of the given
substrings (``bld,blit,rit,bold,italic`` for the bold and italic faces).

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


def page_number(rng: random.Random) -> str:
    """A page reference, 1-3 digits, with 2 drawn twice as often as other digits."""
    digits = "0123456789" + "2"
    length = rng.choice((1, 2, 3, 3, 3))
    first = rng.choice(digits.replace("0", ""))
    return first + "".join(rng.choice(digits) for _ in range(length - 1))


def index_line(words: list[str], rng: random.Random, chars_max: int = 30) -> str:
    """One index-shaped line: an entry and its pages, a bare page list, or a word and a dash."""
    shape = rng.random()
    if shape < 0.45:
        line = f"{rng.choice(words)} — "
    elif shape < 0.85:
        line = ""
    else:
        return f"{rng.choice(words)},— {rng.choice(words)}"[:chars_max].rstrip()
    while True:
        nxt = f"{line}{page_number(rng)}, "
        if len(nxt.rstrip()) > chars_max:
            break
        line = nxt
    if rng.random() < 0.3:
        line = line.rstrip().rstrip(",")
    return line.rstrip()


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
    parser.add_argument("--index", action="store_true", help="render index entries and page lists")
    parser.add_argument("--faces", default=None, help="comma-separated file-name substrings")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    run_dir = (REPO / "runs" / args.run_name).resolve()
    harvest_dirs = [d for d in args.harvest_dirs if (d / "text").is_dir()]
    print(f"harvest dirs: {[str(d) for d in harvest_dirs]}", flush=True)

    started = time.time()
    rng = random.Random(args.seed)
    samples = build_line_samples(harvest_dirs, args.max_samples, rng, args.line_tokens_max)
    if args.index:
        # Single words from the same text are the entries; the shapes around
        # them are generated, since running text has almost no page lists.
        words = [line for line in samples if " " not in line and line[:1].isalpha()]
        samples = [index_line(words, rng) for _ in range(args.max_samples)]
    if args.upper:
        allowed = set(charset.character_list())
        upper = [upper_case_line(line) for line in samples]
        # A capital the charset lacks would be filtered by the trainer
        # without a word; say so here instead.
        strays = sorted({c for line in upper for c in line} - allowed)
        if strays:
            raise SystemExit(f"capitals outside the charset: {strays!r}")
        samples = upper
    allowed = set(charset.character_list())
    strays = sorted({c for line in samples for c in line} - allowed)
    if strays:
        raise SystemExit(f"characters outside the charset: {strays!r}")
    print(f"line samples: {len(samples)}; e.g. {samples[:3]!r}", flush=True)

    sizes = tuple(s for s in (18, 22, 28, 36, 48, 64) if s >= args.min_size)
    fonts = find_fonts()
    if args.faces:
        wanted = [needle.strip().lower() for needle in args.faces.split(",")]
        fonts = [font for font in fonts if any(needle in font.name.lower() for needle in wanted)]
        print(f"faces: {[font.name for font in fonts]}", flush=True)
    root, crops = render_corpus(
        run_dir,
        samples,
        fonts,
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
