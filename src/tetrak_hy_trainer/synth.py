"""Render word crops in the trainer's dataset format.

The minimal synthesis: words rendered with Pillow onto paper-ish grounds,
written as ``<out>/<name>/labels.csv`` (columns ``filename,words``) plus
the image files beside it — exactly what the vendored trainer's
``OCRDataset`` reads. This is what the training spike runs on; the real
Stage 2 pipeline (SynthTIGER-class degradations, corpus text, many fonts)
replaces the *content* while keeping this output contract.

Words are filtered to the canonical charset, because the trainer drops
(or mis-encodes) labels containing characters outside ``opt.character``
and a silent filter is how a training set quietly shrinks.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tetrak_hy_trainer import charset


def render_word(
    word: str,
    font: ImageFont.FreeTypeFont,
    pad: int = 8,
    ink: int = 20,
    paper: int = 240,
    jitter: random.Random | None = None,
) -> Image.Image:
    """One greyscale crop: dark text on a light ground, slight variation.

    The variation (ground/ink level, a pixel of position noise) is the
    minimum that stops a tiny model treating pixel positions as the
    signal; it is not archival realism, which is Stage 2's job.
    """
    jitter = jitter or random.Random(0)
    left, top, right, bottom = font.getbbox(word)
    width = right - left + 2 * pad
    height = bottom - top + 2 * pad
    ground = paper + jitter.randint(-12, 12)
    image = Image.new("L", (width, height), ground)
    draw = ImageDraw.Draw(image)
    draw.text(
        (pad - left + jitter.randint(-1, 1), pad - top + jitter.randint(-1, 1)),
        word,
        font=font,
        fill=ink + jitter.randint(-10, 25),
    )
    return image


def write_dataset(
    out_dir: Path,
    name: str,
    words: list[str],
    font_path: str,
    repeats: int = 1,
    font_size: int = 48,
    seed: int = 0,
) -> Path:
    """Write ``out_dir/name/`` with images and labels.csv; return the folder.

    Args:
        out_dir: The trainer's ``train_data`` root.
        name: Folder name — what the config's ``select_data`` names.
        words: The vocabulary; anything containing characters outside the
            canonical charset is rejected loudly rather than filtered
            silently.
        font_path: A TTF/TTC with Armenian coverage.
        repeats: Renders per word (each with different jitter).
        font_size: Pixel size for the face.
        seed: Determinism for the jitter.
    """
    allowed = set(charset.character_list())
    strays = {c for w in words for c in w if c not in allowed}
    if strays:
        raise ValueError(f"words contain characters outside the charset: {sorted(strays)!r}")

    folder = Path(out_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_path, font_size)
    jitter = random.Random(seed)

    rows = []
    for repeat in range(repeats):
        for index, word in enumerate(words):
            filename = f"{repeat:03d}_{index:04d}.png"
            render_word(word, font, jitter=jitter).save(folder / filename)
            rows.append((filename, word))

    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "words"])
        writer.writerows(rows)
    return folder
