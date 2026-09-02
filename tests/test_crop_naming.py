"""Crops from different works must not write the same file.

Crop filenames were built from the ASE volume number, which was enough
while every source was a numbered ASE volume. Brief 012 added eight works
with no volume number, so ``heldout.volume_of`` returned None for all of
them and each named its crops ``v0_<page>_<index>.png``. Baronian page
100, Faustus page 100 and Otyan page 100 wrote the same file.

The damage is not a lost image. Every source still appends its own row to
``labels.csv``, so the surviving image ends up carrying two or three
contradictory labels, and a CTC model can only reduce loss across those
by emitting something short and noncommittal. It cost the first v5
fine-tune 27% of its 56,608 crops and showed up as dropped characters
mid-word.

These tests are about the naming, so they exercise ``load_pages`` and
``source_slug`` rather than the detector.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "harvest_real_crops.py"


def load_script():
    spec = importlib.util.spec_from_file_location("harvest_real_crops", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


def make_harvest(root: Path, name: str, index_title: str, page_numbers: list[int]) -> Path:
    directory = root / name
    (directory / "text").mkdir(parents=True)
    (directory / "images").mkdir()
    entries = []
    for number in page_numbers:
        (directory / "text" / f"{number}.txt").write_text("բովանդակություն", encoding="utf-8")
        (directory / "images" / f"{number}.jpg").write_bytes(b"")
        entries.append({"page_number": number, "text": f"text/{number}.txt", "revid": 1})
    (directory / "manifest.json").write_text(
        json.dumps({"index": index_title, "min_quality": 3, "pages": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return directory


def crop_name(script, page: dict, index: int) -> str:
    """The filename the cutter builds, kept in step with the script."""
    return f"{page['source']}_{page['page_number']:04d}_{index:04d}.png"


def test_same_page_number_in_different_works_gets_different_names(script, tmp_path) -> None:
    """The regression itself: three works, one shared page number."""
    works = {
        "baronian-vol10": "Ինդեքս:… Hagop Baronian ….djvu",
        "faustus-1968": "Ինդեքս:… Faustus of Byzantium ….djvu",
        "otyan-works": "Ինդեքս:… Yervand Otyan ….djvu",
    }
    directories = [make_harvest(tmp_path, name, title, [100]) for name, title in works.items()]

    _, pages = script.load_pages(directories)

    assert len(pages) == 3, "all three works contribute their page 100"
    names = {crop_name(script, page, 0) for page in pages}
    assert len(names) == 3, f"crops collided: {names}"


def test_volumeless_works_are_not_all_v0(script, tmp_path) -> None:
    """None of the names may fall back to the volume number.

    This is the specific shape of the bug: `volume_of` returns None, the
    old name used `volume or 0`, and every non-ASE work became `v0`.
    """
    directories = [
        make_harvest(tmp_path, "otyan-works", "Ինդեքս:… Yervand Otyan ….djvu", [100]),
        make_harvest(tmp_path, "totovents-works", "Ինդեքս:… Vahan Totovents ….djvu", [100]),
    ]

    _, pages = script.load_pages(directories)

    assert all(page["volume"] is None for page in pages), "these works have no volume number"
    assert {page["source"] for page in pages} == {"otyan-works", "totovents-works"}


def test_each_page_records_the_work_it_came_from(script, tmp_path) -> None:
    """Without this the audit trail cannot answer 'which work?'.

    Diagnosing a per-register regression means grouping crops by source,
    and page numbers overlap across works, so the source has to be
    recorded rather than inferred.
    """
    directories = [
        make_harvest(tmp_path, "faustus-1968", "Ինդեքս:… Faustus of Byzantium ….djvu", [100]),
        make_harvest(tmp_path, "otyan-works", "Ինդեքս:… Yervand Otyan ….djvu", [100]),
    ]

    _, pages = script.load_pages(directories)

    by_source = {page["source"]: page["index"] for page in pages}
    assert by_source == {
        "faustus-1968": "Ինդեքս:… Faustus of Byzantium ….djvu",
        "otyan-works": "Ինդեքս:… Yervand Otyan ….djvu",
    }


def test_two_directories_with_the_same_name_are_refused(script, tmp_path) -> None:
    """The slug is the crop-name prefix, so it has to be unique."""
    first = make_harvest(tmp_path / "a", "otyan-works", "Ինդեքս:… Yervand Otyan ….djvu", [100])
    second = make_harvest(tmp_path / "b", "otyan-works", "Ինդեքս:… Yervand Otyan ….djvu", [200])

    with pytest.raises(ValueError, match="same crop-name prefix"):
        script.load_pages([first, second])


def test_the_slug_is_safe_for_the_labels_file(script) -> None:
    """labels.csv is split on the first comma, so no commas in names."""
    slug = script.source_slug(Path("runs/harvest/a, awkward/name"))

    assert "," not in slug
    assert " " not in slug
