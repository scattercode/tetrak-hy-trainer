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


class TestWriteLabels:
    """The trainer reads labels.csv with a regex that splits each line at
    its first comma, not as CSV -- so a label may contain commas but must
    never be quoted. csv.writer quotes them, and the quotation marks then
    become part of the label: 36,918 of v1's 175,500 crops were trained
    with labels wrapped in quote marks the images do not show."""

    def test_a_label_containing_a_comma_is_not_quoted(self, tmp_path: Path) -> None:
        synth.write_labels(tmp_path, [("a.png", "Շվեյցարիայում,")])
        assert (tmp_path / "labels.csv").read_text(encoding="utf-8").splitlines()[1] == (
            "a.png,Շվեյցարիայում,"
        )

    def test_a_label_that_is_only_a_comma_survives(self, tmp_path: Path) -> None:
        """v1 has crops of a single comma labelled '","'."""
        synth.write_labels(tmp_path, [("a.png", ",")])
        assert (tmp_path / "labels.csv").read_text(encoding="utf-8").splitlines()[1] == "a.png,,"

    def test_the_label_is_everything_after_the_first_comma(self, tmp_path: Path) -> None:
        """Mirrors the trainer's own split, so the round trip is pinned."""
        rows = [("a.png", "simple"), ("b.png", "Ե․, 1956։"), ("c.png", "two words")]
        synth.write_labels(tmp_path, rows)
        lines = (tmp_path / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]
        assert [tuple(line.split(",", 1)) for line in lines] == rows

    def test_the_header_the_trainer_reads_columns_by(self, tmp_path: Path) -> None:
        synth.write_labels(tmp_path, [])
        assert (tmp_path / "labels.csv").read_text(encoding="utf-8") == "filename,words\n"

    def test_quotation_marks_in_a_label_are_left_alone(self, tmp_path: Path) -> None:
        """A genuine quotation mark in the page must not be doubled or
        stripped -- « » are ordinary characters in this charset."""
        synth.write_labels(tmp_path, [("a.png", '«Ա» "b"')])
        assert (tmp_path / "labels.csv").read_text(encoding="utf-8").splitlines()[1] == (
            'a.png,«Ա» "b"'
        )

    def test_a_label_spanning_lines_is_refused(self, tmp_path: Path) -> None:
        """No line-based split could recover it, so fail rather than
        silently write a file the trainer will misread."""
        with pytest.raises(ValueError, match="spans lines"):
            synth.write_labels(tmp_path, [("a.png", "two\nlines")])

    def test_the_folder_is_created(self, tmp_path: Path) -> None:
        synth.write_labels(tmp_path / "new", [("a.png", "ա")])
        assert (tmp_path / "new" / "labels.csv").exists()


@needs_font
class TestMissingGlyphs:
    """Mshtakan is a real, reproducible example: it has no glyph for the
    charset's v2 additions (U+2024, degree sign), found while verifying
    those additions actually render rather than falling back to a
    fallback glyph indistinguishable by eye from a genuine one."""

    def test_finds_the_known_gap(self) -> None:
        assert synth.missing_glyphs(FONT, "․°") == {"․", "°"}

    def test_armenian_letters_are_covered(self) -> None:
        assert synth.missing_glyphs(FONT, "աբգդ") == frozenset()

    def test_only_reports_the_actually_missing_characters(self) -> None:
        # "․" is missing, "ա" is not -- exactly one of the two comes back
        assert synth.missing_glyphs(FONT, "ա․") == {"․"}

    def test_empty_input_has_nothing_missing(self) -> None:
        assert synth.missing_glyphs(FONT, "") == frozenset()


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
