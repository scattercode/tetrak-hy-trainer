"""Stage-2-lite degradations: make a clean render look scanned.

The v0 lesson (see Tetrak's Armenian OCR benchmarks note, reading 8): a
model at 99.7% on crisp renderings read real scans at 0.08, because the
pixels differ — real crops arrive blurred, aliased, JPEG-compressed, on
toned paper, slightly rotated. This module applies cheap PIL-only
approximations of those effects. The full SynthTIGER-class pipeline
remains Stage 2 proper; this is the subset that pays for itself tonight.

Every function takes and returns a greyscale (``L``) image and draws its
randomness from a caller-supplied ``random.Random``, so a seeded corpus
is reproducible crop for crop.
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter


def degrade(image: Image.Image, rng: random.Random, faint_share: float = 0.0) -> Image.Image:
    """The standard pipeline: each effect applied with its own probability.

    Tuned to *usually* apply something and *occasionally* stack harshly —
    matching a corpus where most pages are readable and some are rough.

    *faint_share* is the probability of :func:`faint` (brief 023). At the
    default of 0 nothing is drawn from *rng* for it, so every set rendered
    before the lever existed is still reproduced crop for crop.
    """
    if rng.random() < 0.6:
        image = downscale_cycle(image, rng)
    if rng.random() < 0.5:
        image = blur(image, rng)
    if rng.random() < 0.7:
        image = tone(image, rng)
    if rng.random() < 0.3:
        image = rotate(image, rng)
    if faint_share and rng.random() < faint_share:
        image = faint(image, rng)
    if rng.random() < 0.6:
        image = jpeg_cycle(image, rng)
    return image


def downscale_cycle(image: Image.Image, rng: random.Random) -> Image.Image:
    """Lose resolution the way a small scan does: shrink, then re-enlarge.

    Real crops are captured at ~18–30 px x-height and upscaled to the
    model's 64 px input; this teaches the aliased glyph forms that
    round-trip produces.
    """
    factor = rng.uniform(0.35, 0.8)
    small = image.resize(
        (max(8, int(image.width * factor)), max(8, int(image.height * factor))),
        Image.BILINEAR,
    )
    return small.resize(image.size, Image.BILINEAR)


def blur(image: Image.Image, rng: random.Random) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.4)))


def tone(image: Image.Image, rng: random.Random) -> Image.Image:
    """Paper tint and press variation: contrast and brightness jitter."""
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.55, 1.15))
    return ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.15))


def faint(image: Image.Image, rng: random.Random) -> Image.Image:
    """A grey photograph of a page: faint ink on toned paper, with grain.

    Brief 023. :func:`tone` jitters contrast around a page that stays
    black on white; at its harshest the ink is still about a hundred grey
    levels below the paper. A book photographed rather than scanned can be
    fainter than anything that produces: on the Abovyan evaluation pages
    (measured 16 November 2026) the paper sits near 160 and the ink only
    58 to 78 levels below it, and nothing the recogniser trained on looked
    like that. This maps white to a paper level and black to a level a
    chosen span below it, then blends in a little uniform grain, since a
    photograph that faint is never clean.

    The grain is drawn from *rng* (``randbytes``), not from Pillow's own
    noise generator, which cannot be seeded.
    """
    paper = rng.uniform(150, 215)
    span = rng.uniform(45, 100)
    table = [round(paper - (255 - level) * span / 255) for level in range(256)]
    image = image.point(table)
    grain = Image.frombytes("L", image.size, rng.randbytes(image.width * image.height))
    return Image.blend(image, grain, rng.uniform(0.02, 0.06))


def rotate(image: Image.Image, rng: random.Random) -> Image.Image:
    """A degree or two of skew, filled with the corner colour."""
    fill = image.getpixel((0, 0))
    return image.rotate(
        rng.uniform(-2.0, 2.0), resample=Image.BILINEAR, expand=False, fillcolor=fill
    )


def jpeg_cycle(image: Image.Image, rng: random.Random) -> Image.Image:
    """Round-trip through JPEG at archival-repository quality levels."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(40, 85))
    buffer.seek(0)
    return Image.open(buffer).convert("L")
