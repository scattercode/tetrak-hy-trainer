"""The degradation pipeline's contracts: deterministic, mode- and
size-preserving, and actually degrading. No font needed — the tests draw
their own test card."""

import random

from PIL import Image, ImageDraw

from tetrak_hy_trainer import augment


def test_card() -> Image.Image:
    image = Image.new("L", (200, 64), 240)
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 180, 44], fill=25)
    return image


def test_degrade_preserves_mode_and_size() -> None:
    result = augment.degrade(test_card(), random.Random(1))
    assert result.mode == "L"
    assert result.size == (200, 64)


def test_degrade_is_deterministic_under_a_seed() -> None:
    a = augment.degrade(test_card(), random.Random(7))
    b = augment.degrade(test_card(), random.Random(7))
    assert list(a.getdata()) == list(b.getdata())


def test_degrade_changes_the_pixels() -> None:
    """A pipeline that usually applies something must usually change
    something; across a handful of seeds at least one must differ."""
    original = list(test_card().getdata())
    assert any(
        list(augment.degrade(test_card(), random.Random(seed)).getdata()) != original
        for seed in range(5)
    )


def test_each_effect_stands_alone() -> None:
    for effect in (
        augment.downscale_cycle,
        augment.blur,
        augment.tone,
        augment.rotate,
        augment.jpeg_cycle,
    ):
        result = effect(test_card(), random.Random(3))
        assert result.mode == "L", effect.__name__
        assert result.size == (200, 64), effect.__name__
