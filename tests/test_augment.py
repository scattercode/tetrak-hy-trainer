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
        augment.faint,
        augment.jpeg_cycle,
    ):
        result = effect(test_card(), random.Random(3))
        assert result.mode == "L", effect.__name__
        assert result.size == (200, 64), effect.__name__


def test_faint_leaves_little_between_ink_and_paper() -> None:
    """The point of the effect: ink within about a hundred levels of a
    paper that is no longer white, as on a grey photograph of a page."""
    for seed in range(8):
        levels = sorted(augment.faint(test_card(), random.Random(seed)).getdata())
        paper = levels[len(levels) * 9 // 10]
        ink = levels[len(levels) // 100]
        assert paper < 225, seed
        assert 30 < paper - ink < 110, seed


def test_faint_is_deterministic_under_a_seed() -> None:
    """Its grain comes from the caller's generator, not Pillow's."""
    a = augment.faint(test_card(), random.Random(5))
    b = augment.faint(test_card(), random.Random(5))
    assert list(a.getdata()) == list(b.getdata())


def test_degrade_without_a_faint_share_draws_nothing_for_it() -> None:
    """A set rendered before the lever existed must still be reproduced:
    at a share of 0 the generator is left exactly where it was."""
    for seed in range(5):
        plain, with_lever = random.Random(seed), random.Random(seed)
        a = augment.degrade(test_card(), plain)
        b = augment.degrade(test_card(), with_lever, faint_share=0.0)
        assert list(a.getdata()) == list(b.getdata())
        assert plain.random() == with_lever.random()


def test_degrade_with_a_faint_share_applies_it() -> None:
    def contrast(image: Image.Image) -> int:
        levels = sorted(image.getdata())
        return levels[len(levels) * 9 // 10] - levels[len(levels) // 100]

    faint = [
        contrast(augment.degrade(test_card(), random.Random(seed), faint_share=1.0))
        for seed in range(8)
    ]
    assert max(faint) < 110
