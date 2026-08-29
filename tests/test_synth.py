"""The synthesiser's output contract with the vendored trainer.

Rendering needs a font with Armenian coverage, which CI's runner does not
have — those tests skip there and run on any mac (Mshtakan ships with the
OS). The charset guard needs no font and always runs.
"""

import csv
from pathlib import Path

import pytest

from tetrak_hy_trainer import synth

FONT = "/System/Library/Fonts/Supplemental/Mshtakan.ttc"

needs_font = pytest.mark.skipif(
    not Path(FONT).exists(), reason="needs an Armenian-capable system font"
)


def test_stray_characters_are_rejected_not_filtered(tmp_path: Path) -> None:
    """A silent filter is how a training set quietly shrinks; strays raise."""
    with pytest.raises(ValueError, match="outside the charset"):
        synth.write_dataset(tmp_path, "x", ["ok", "п-cyrillic"], font_path=FONT)


@needs_font
def test_dataset_layout_matches_the_trainer_contract(tmp_path: Path) -> None:
    folder = synth.write_dataset(tmp_path, "hy_t", ["գիրք", "և"], FONT, repeats=3)

    assert folder == tmp_path / "hy_t"
    with (folder / "labels.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["filename", "words"]
    assert len(rows) == 1 + 2 * 3
    for filename, word in rows[1:]:
        assert (folder / filename).exists()
        assert word in ("գիրք", "և")


@needs_font
def test_render_word_produces_a_grey_crop_with_margin(tmp_path: Path) -> None:
    from PIL import ImageFont

    image = synth.render_word("արխիվ", ImageFont.truetype(FONT, 48))
    assert image.mode == "L"
    assert image.width > image.height > 40
