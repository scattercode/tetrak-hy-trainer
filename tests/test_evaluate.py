"""The shared measurement behind every published figure.

``evaluate.score_pages`` moved out of ``train_synthetic.py`` so that the
training run and the per-register script cannot drift into scoring the
same model two slightly different ways. These tests pin the parts of it
that would change a reported number without failing anything: which
pages count, and what the transform is applied to.

No easyocr here -- the reader is injected, which is the other reason
``score_pages`` takes one.
"""

from __future__ import annotations

import json
from pathlib import Path

from tetrak_hy_trainer import evaluate


class FakeReader:
    """Returns canned box text per image, like easyocr's ``detail=0``."""

    def __init__(self, by_page: dict[str, list[str]]):
        self.by_page = by_page
        self.seen: list[str] = []

    def readtext(self, path, detail=0, paragraph=False):
        assert detail == 0 and paragraph is False, "must match Tetrak's backend call"
        self.seen.append(Path(path).name)
        return self.by_page[Path(path).stem]


def build_eval_dir(tmp_path: Path, pages: dict[int, str], with_images: set[int]) -> Path:
    eval_dir = tmp_path / "set"
    (eval_dir / "text").mkdir(parents=True)
    (eval_dir / "images").mkdir()
    manifest = {"index": "Ինդեքս:x.djvu", "min_quality": 3, "pages": []}
    for number, text in pages.items():
        (eval_dir / "text" / f"{number}.txt").write_text(text, encoding="utf-8")
        manifest["pages"].append({"page_number": number, "text": f"text/{number}.txt"})
        if number in with_images:
            (eval_dir / "images" / f"{number}.jpg").write_bytes(b"")
    (eval_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return eval_dir


def test_a_perfect_read_scores_one(tmp_path: Path) -> None:
    eval_dir = build_eval_dir(tmp_path, {1: "բարև աշխարհ"}, with_images={1})
    reader = FakeReader({"1": ["բարև աշխարհ"]})

    result = evaluate.score_pages(tmp_path / "bundle", eval_dir, reader=reader)

    assert len(result) == 1
    assert result.mean_character_similarity == 1.0
    assert result.mean_word_recall == 1.0


def test_a_page_without_a_scan_is_skipped_not_scored_zero(tmp_path: Path) -> None:
    """A missing image is a harvesting gap, not a recognition failure.

    Scoring it zero would drag the average down and report the model
    failing to read a page it was never shown -- and the manifest lists
    pages by transcript, so a text-only harvest has plenty of them.
    """
    eval_dir = build_eval_dir(
        tmp_path,
        {1: "բարև աշխարհ", 2: "անտեսանելի էջ"},
        with_images={1},
    )
    reader = FakeReader({"1": ["բարև աշխարհ"]})

    result = evaluate.score_pages(tmp_path / "bundle", eval_dir, reader=reader)

    assert [page.page_number for page in result.pages] == [1]
    assert result.mean_word_recall == 1.0
    assert reader.seen == ["1.jpg"], "the absent page must not reach the reader"


def test_the_transform_is_applied_to_both_sides(tmp_path: Path) -> None:
    """The fold has to touch prediction and truth, or it measures nothing.

    Applied to the prediction alone it would only ever equate spellings
    the truth does not use, which is the mistake that makes a fold look
    like it recovered nothing.
    """
    eval_dir = build_eval_dir(tmp_path, {1: "ՀԱՅ"}, with_images={1})
    reader = FakeReader({"1": ["HԱՅ"]})  # Latin H for Armenian Հ

    raw = evaluate.score_pages(tmp_path / "bundle", eval_dir, reader=reader)
    folded = evaluate.score_pages(
        tmp_path / "bundle",
        eval_dir,
        reader=FakeReader({"1": ["HԱՅ"]}),
        transform=lambda text: text.replace("H", "Հ"),
    )

    assert raw.mean_word_recall == 0.0
    assert folded.mean_word_recall == 1.0


def test_boxes_are_joined_with_newlines(tmp_path: Path) -> None:
    """Tetrak's backend joins detected boxes with newlines; so must this."""
    reader = FakeReader({"1": ["առաջին", "երկրորդ"]})

    assert evaluate.read_page(reader, Path("1.jpg")) == "առաջին\nերկրորդ"


def test_the_mean_is_over_pages(tmp_path: Path) -> None:
    eval_dir = build_eval_dir(tmp_path, {1: "ա բ", 2: "գ դ"}, with_images={1, 2})
    reader = FakeReader({"1": ["ա բ"], "2": ["ա բ"]})  # page 2 entirely wrong

    result = evaluate.score_pages(tmp_path / "bundle", eval_dir, reader=reader)

    assert len(result) == 2
    assert result.mean_word_recall == 0.5
