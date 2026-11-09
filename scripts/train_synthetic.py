#!/usr/bin/env python3
"""The synthetic pre-train orchestrator (v0 was its first run; v1 default).

Samples *lines* of consecutive tokens from harvested proofread text (the
v0 lesson: CRAFT hands the recogniser multi-word line crops, and a model
trained on single words had never seen a space), renders them across the
provided fonts at sizes down to real scan scale, degrades them with the
Stage-2-lite pipeline (:mod:`tetrak_hy_trainer.augment`), trains, packages
the best checkpoint as a ``tetrak_hy`` bundle — and, when ``--eval-dir``
points at a harvested page set, scores the result against the real scans
so the honest number is in the log before anyone wakes up.

v0 (2026-08-30, recorded in Tetrak's benchmarks note) reproduces with:
``--line-tokens-max 1 --no-augment --min-size 36``.

Everything lands under ``runs/<run-name>/``:

    all_data/            rendered crops + labels.csv (trainer format)
    saved_models/<name>/ checkpoints; best_accuracy.pth updates throughout
    bundle/              packaged tetrak_hy.{yaml,py,pth} on completion
    train.log            stdout of the run (the caller redirects)

Run (overnight, Mac):
    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -ims \\
        python scripts/train_synthetic.py --device mps --iters 150000 \\
        --eval-dir <harvested-eval-pages>

Interrupted? The checkpoint survives; re-package with ``--package-only``.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "training"))

from tetrak_hy_trainer import (  # noqa: E402
    augment,
    charset,
    heldout,
    packaging,
    sampling,
    synth,
    train_config,
    wikisource,  # noqa: E402
)


def clean_token_runs(harvest_dirs: list[Path]) -> list[list[str]]:
    """Runs of consecutive charset-clean tokens, per stretch of page text.

    Held-out pages are skipped: rendering an evaluation page's transcript
    as synthetic training data would let the model memorise the exact
    text it is later scored on reading. The registry, not the filesystem,
    decides -- the text files may well be on disk.
    """
    import json

    allowed = set(charset.character_list())
    runs: list[list[str]] = []
    for harvest_dir in harvest_dirs:
        manifest = harvest_dir / "manifest.json"
        index_title = (
            json.loads(manifest.read_text(encoding="utf-8"))["index"] if manifest.exists() else ""
        )
        for text_file in sorted((harvest_dir / "text").glob("*.txt")):
            if index_title and heldout.page_is_held_out(index_title, int(text_file.stem)):
                continue
            current: list[str] = []
            # Normalised at read time, not only in clean_wikitext: several
            # thousand pages were harvested before the substitutions were
            # found, and re-fetching them to fix a character swap would be
            # discourteous. Idempotent, so new harvests are unaffected.
            page_text = wikisource.normalise_transcript(
                text_file.read_text(encoding="utf-8"), index_title
            )
            for token in page_text.split():
                # A token mixing Cyrillic with Armenian or Latin letters
                # breaks the run like a stray: brief 014's Stage 0.3
                # policy, see charset.mixes_cyrillic.
                if (
                    1 <= len(token) <= 24
                    and set(token) <= allowed
                    and not charset.mixes_cyrillic(token)
                ):
                    current.append(token)
                elif current:
                    runs.append(current)
                    current = []
            if current:
                runs.append(current)
    return runs


def build_line_samples(
    harvest_dirs: list[Path],
    max_samples: int,
    rng: random.Random,
    tokens_max: int,
    chars_max: int = 30,
    mixed_share: float = 0.0,
) -> list[str]:
    """Sample text lines: 1..tokens_max consecutive tokens, length-capped.

    One candidate per token position (with a random length draw), sampled
    down to *max_samples* — so common vocabulary appears at natural
    frequency and every page contributes. *mixed_share* lifts the lines
    that mix Armenian and Cyrillic tokens to that fraction of the sample
    (brief 014); zero keeps the natural rate and the draws unchanged.
    """
    candidates: list[str] = []
    for run in clean_token_runs(harvest_dirs):
        for start in range(len(run)):
            take = rng.randint(1, tokens_max)
            line = " ".join(run[start : start + take])
            if len(line) <= chars_max:
                candidates.append(line)
    if len(candidates) < 1000:
        raise SystemExit(f"only {len(candidates)} line candidates -- harvest more text first")
    samples, mixed = sampling.with_mixed_share(candidates, max_samples, rng, mixed_share)
    wanted = round(max_samples * mixed_share)
    print(
        f"mixed-script lines: {mixed} of {len(samples)}"
        + (f" ({wanted} wanted)" if mixed < wanted else ""),
        flush=True,
    )
    return samples


def upper_case_line(line: str) -> str:
    """*line* in capitals, as a headword or a title page prints it.

    ``str.upper`` maps the ligature ``և`` to ``ԵՒ``, the traditional
    spelling; the reformed-orthography editions this model reads print
    ``ԵՎ``, so the ligature is spelled out first.
    """
    return line.replace("և", "եվ").upper()


def render_corpus(
    run_dir: Path,
    samples: list[str],
    fonts: list[Path],
    sizes: tuple[int, ...],
    repeats: int,
    use_augment: bool,
    seed: int = 0,
    names: tuple[str, str] = ("syn_train", "syn_val"),
    families: dict[str, str] | None = None,
) -> tuple[Path, int]:
    """Render train/val folders in the trainer's format; return (root, count).

    *names* are the train and validation folder names under ``all_data/``,
    so a second synthetic set (brief 013's all-caps lines) can sit beside
    ``syn_train`` and be mixed in by ``select_data``.

    Validation gets the same rendering *and degradation* treatment —
    v0's crisp validation read 99.7% while real scans read 0.08, so a
    val set that never sees a degradation measures nothing useful.

    Font choice per line is glyph-coverage-aware: a font missing even one
    character in the line is excluded from that line's draw, not just
    charset-eligible. Membership in the canonical charset says nothing
    about whether a *specific* font can actually draw a character — found
    by checking whether Mshtakan, one of the three faces every crop is
    rendered in, could draw the charset's v2 additions (it cannot; see
    synth.missing_glyphs). Rendering through a face without the glyph
    would silently teach the model the wrong shape for whatever that face
    is missing, indistinguishable by eye from a genuine narrow glyph.

    A caps-only face (:func:`synth.draws_capitals_for_lowercase`) counts
    every lowercase letter as missing, so it draws only all-capital lines.

    *families* maps a font file's name to its family (brief 019's display
    faces, from ``faces.tsv``). Given, each line picks a family first and
    then a face within it, so a family published in eighteen styles is
    drawn as often as one with a single face. Left out, every face and size
    is equally likely, exactly as before: the random draws are unchanged,
    so a seeded corpus renders the same crops it always has.
    """
    from PIL import ImageFont

    rng = random.Random(seed)
    gaps_by_path = {
        path: synth.missing_glyphs(str(path), charset.character_list()) | synth.case_gaps(str(path))
        for path in fonts
    }
    faces = [
        (ImageFont.truetype(str(path), size), gaps_by_path[path], path.name)
        for path in fonts
        for size in sizes
    ]
    family_of = {} if families is None else {name: families.get(name, name) for _, _, name in faces}

    data_root = run_dir / "all_data"
    train_dir = data_root / names[0]
    val_dir = data_root / names[1]
    for directory in (train_dir, val_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    for index, line in enumerate(samples):
        needed = set(line)
        eligible = [(face, name) for face, gaps, name in faces if not (gaps & needed)]
        if not eligible:
            raise SystemExit(f"no font covers every character in {line!r}")
        is_val = index % 40 == 0
        for repeat in range(1 if is_val else repeats):
            if family_of:
                family = rng.choice(sorted({family_of[name] for _, name in eligible}))
                font = rng.choice([face for face, name in eligible if family_of[name] == family])
            else:
                font = rng.choice(eligible)[0]
            image = synth.render_word(line, font, jitter=rng)
            if use_augment:
                image = augment.degrade(image, rng)
            name = f"{index:06d}_{repeat}.png"
            if is_val:
                image.save(val_dir / name)
                val_rows.append((name, line))
            else:
                image.save(train_dir / name)
                train_rows.append((name, line))

    for directory, rows in ((train_dir, train_rows), (val_dir, val_rows)):
        # Not csv.writer: it quotes labels containing a comma, and the
        # trainer's regex split keeps the quotation marks. See
        # synth.write_labels -- this cost v1 21% of its crops.
        synth.write_labels(directory, rows)
    return data_root, len(train_rows)


def find_fonts(font_dir: Path | None = None) -> list[Path]:
    """Every rendering face, with each one's charset gaps printed.

    By default the book faces under ``runs/v0/fonts`` plus Mshtakan from the
    system. *font_dir* names another folder instead (brief 019's display
    faces), and then only its faces are used.
    """
    # .tt* catches TTF and TTC; the GHEA faces are OTF, which Pillow reads
    # just as happily and the original glob silently ignored.
    default = font_dir is None
    font_dir = REPO / "runs" / "v0" / "fonts" if default else font_dir
    fonts = sorted([*font_dir.glob("*.tt*"), *font_dir.glob("*.otf")])
    system_font = Path("/System/Library/Fonts/Supplemental/Mshtakan.ttc")
    if default and system_font.exists():
        fonts.append(system_font)
    if not fonts:
        raise SystemExit("no fonts found")
    print(f"fonts: {[f.name for f in fonts]}", flush=True)
    for font in fonts:
        gaps = synth.missing_glyphs(str(font), charset.character_list())
        if gaps:
            print(f"  {font.name} has no glyph for: {synth.describe_gaps(gaps)}", flush=True)
    return fonts


def package(run_dir: Path, experiment: str) -> Path:
    """Package the best checkpoint as a tetrak_hy bundle; return its dir."""
    checkpoint = run_dir / "saved_models" / experiment / "best_accuracy.pth"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    bundle = run_dir / "bundle"
    packaging.write_bundle(bundle)
    shutil.copyfile(checkpoint, bundle / f"{packaging.NETWORK_NAME}.pth")
    print(f"packaged: {bundle}", flush=True)
    return bundle


def evaluate_pages(bundle: Path, eval_dir: Path) -> None:
    """Score the packaged model against harvested real pages, in the log.

    Same loading and line-joining as Tetrak's easyocr backend, same
    metrics as the Armenian benchmarks note, so the number is comparable
    the moment it prints. The measurement itself lives in
    :mod:`tetrak_hy_trainer.evaluate`, shared with the per-register
    evaluation, so the two cannot drift apart; this only prints it.
    """
    from tetrak_hy_trainer import evaluate

    result = evaluate.score_pages(
        bundle,
        eval_dir,
        on_page=lambda page: print(
            f"EVAL p{page.page_number} chr={page.character_similarity:.3f} "
            f"wrd={page.word_recall:.3f}",
            flush=True,
        ),
    )
    if len(result):
        print(
            f"EVAL AVERAGE chr={result.mean_character_similarity:.4f} "
            f"wrd={result.mean_word_recall:.4f} n={len(result)}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--harvest-dirs", nargs="+", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=150_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--max-samples", type=int, default=60_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--line-tokens-max", type=int, default=4)
    parser.add_argument(
        "--mixed-share",
        type=float,
        default=0.0,
        help="this fraction of lines mix Armenian and Cyrillic tokens (brief 014); 0 is the natural rate",
    )
    parser.add_argument("--min-size", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-interval", type=int, default=2_000)
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()

    # Resolved here, not at the point of use: training chdir()s into the run
    # directory, so a relative --eval-dir would otherwise resolve against
    # that and fail -- after the whole run, with the number lost.
    if args.eval_dir is not None:
        args.eval_dir = args.eval_dir.resolve()

    run_dir = (REPO / "runs" / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.package_only:
        package(run_dir, args.run_name)
        return 0

    harvest_dirs = args.harvest_dirs or [
        directory
        for directory in [
            REPO / "runs" / "v0" / "harvest",
            *sorted((REPO / "runs" / "v1").glob("harvest-*")),
        ]
        if directory.is_dir()
    ]
    print(f"harvest dirs: {[str(d) for d in harvest_dirs]}", flush=True)

    started = time.time()
    rng = random.Random(0)
    samples = build_line_samples(
        harvest_dirs, args.max_samples, rng, args.line_tokens_max, mixed_share=args.mixed_share
    )
    print(f"line samples: {len(samples)}", flush=True)

    fonts = find_fonts()

    sizes = tuple(s for s in (18, 22, 28, 36, 48, 64) if s >= args.min_size)
    data_root, crops = render_corpus(run_dir, samples, fonts, sizes, args.repeats, args.augment)
    print(f"rendered {crops} training crops in {time.time() - started:.0f}s", flush=True)
    sampling.write_recipe(
        data_root / "syn_train",
        harvest_dirs=[str(d) for d in harvest_dirs],
        max_samples=args.max_samples,
        line_tokens_max=args.line_tokens_max,
        mixed_share=args.mixed_share,
        mixed_lines=sum(1 for line in samples if sampling.is_mixed_line(line)),
        lines=len(samples),
        repeats=args.repeats,
        seed=0,
    )

    config = train_config.build_config(
        experiment_name=args.run_name,
        train_data=str(data_root),
        valid_data=str(data_root / "syn_val"),
        select_data="syn_train",
        num_iter=args.iters,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        workers=0,
        batch_max_length=60,
    )
    config["imgW"] = 800  # room for four-token lines at 64 px height

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

    bundle = package(run_dir, args.run_name)
    if args.eval_dir is not None:
        evaluate_pages(bundle, args.eval_dir)
    print(f"total wall time: {(time.time() - started) / 3600:.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
