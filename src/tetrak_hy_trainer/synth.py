"""Render word crops in the trainer's dataset format.

The minimal synthesis: words rendered with Pillow onto paper-ish grounds,
written as ``<out>/<name>/labels.csv`` (columns ``filename,words``) plus
the image files beside it — exactly what the vendored trainer's
``OCRDataset`` reads. This is what the training spike runs on; the real
Stage 2 pipeline (SynthTIGER-class degradations, corpus text, many fonts)
replaces the *content* while keeping this output contract.

Words are filtered to the canonical charset, because the trainer drops
(or mis-encodes) labels containing characters outside ``opt.character``
and a silent filter is how a training set quietly shrinks. Charset
membership is necessary but not sufficient, though: :func:`missing_glyphs`
checks whether a *specific font* actually has a glyph for a character
that is otherwise in-charset, which membership alone cannot tell you.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tetrak_hy_trainer import charset


@lru_cache
def _cmap(font_path: str) -> frozenset[int]:
    """The Unicode code points *font_path* has a real glyph for.

    Cached per path: a font file is read once no matter how many sizes or
    render calls use it. ``fontTools`` reads the ``cmap`` table directly
    rather than going through Pillow/FreeType, which has no public,
    version-stable way to ask "does this font have a glyph for this
    character" -- it happily draws the ``.notdef`` fallback glyph for a
    missing one, visually indistinguishable from a genuine narrow glyph
    unless compared pixel-for-pixel against another missing character.
    That is not a hypothetical: Mshtakan, the macOS system font this
    trainer renders one third of its synthetic crops with, has no glyph
    for Latin ``A``/``x`` or for either of the charset's v2 additions
    (U+2024, ``°``) -- found by exactly this comparison while verifying
    those additions actually render (product/research/
    armenian-v1-error-analysis.md's charset fix). A corpus built without
    this check silently teaches the model the wrong shape for whatever a
    face is missing, for however many crops chose that face.
    """
    from fontTools.ttLib import TTFont

    font = TTFont(font_path, fontNumber=0, lazy=True)
    try:
        return frozenset(font.getBestCmap() or {})
    finally:
        font.close()


def write_labels(folder: Path, rows: Iterable[tuple[str, str]]) -> Path:
    """Write ``folder/labels.csv`` in the exact form the trainer parses.

    The vendored trainer does **not** read this file as CSV. It reads it
    with ``pd.read_csv(..., sep='^([^,]+),', engine='python')``, a regular
    expression that splits each line at its *first* comma and takes
    everything after it, verbatim, as the label. A label may therefore
    contain commas freely -- but it must not be quoted, and this is why
    the file cannot be written with :mod:`csv`.

    ``csv.writer`` quotes any field containing the delimiter, and those
    quotation marks are not stripped by a regex split: they become part of
    the label. v1 was trained this way, and **36,918 of its 175,500 crops
    -- 21% -- carry labels wrapped in quotation marks they do not show**,
    with a crop of a single comma labelled ``","``. The model learnt
    exactly what it was shown: inventing a quotation mark is the single
    commonest thing it does wrong on the evaluation pages, ahead of every
    genuine character confusion. Found while harvesting real crops, in
    time for v2's charset re-render but not for v1.

    Args:
        folder: The dataset folder; created if missing.
        rows: ``(filename, label)`` pairs.

    Returns:
        The path written.

    Raises:
        ValueError: A label contains a newline, which no regex split on a
            single line could ever recover.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "labels.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write("filename,words\n")
        for filename, label in rows:
            if "\n" in label or "\r" in label:
                raise ValueError(f"label for {filename!r} spans lines: {label!r}")
            handle.write(f"{filename},{label}\n")
    return destination


def missing_glyphs(font_path: str, characters: str) -> frozenset[str]:
    """The characters in *characters* that *font_path* has no glyph for.

    Args:
        font_path: A TTF/TTC/OTF path. ``.ttc`` collections are checked
            against their first font, matching how :func:`render_word`
            and ``ImageFont.truetype`` load them elsewhere in this module.
        characters: Any string; duplicates and order do not matter.

    Returns:
        The distinct characters in *characters* absent from the font's
        cmap. Empty means the font covers everything asked of it.
    """
    cmap = _cmap(font_path)
    return frozenset(character for character in characters if ord(character) not in cmap)


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

    write_labels(folder, rows)
    return folder
