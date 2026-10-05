"""Harvest resume and the transcript-cleaning version.

Harvest text is saved after cleaning, and a re-run resumes past any page
whose text is already on disk. When a cleaning change throws away less
than the one before it -- cleaning 2 stopped folding printed angle
brackets to guillemets -- the text saved earlier is stale for good, so
each page records the cleaning it was written with, and
``refresh_stale`` fetches stale pages again. Checked here against a fake
client, with no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from tetrak_hy_trainer import wikisource
from tetrak_hy_trainer.harvest import harvest
from tetrak_hy_trainer.wikisource import PageRecord

INDEX = "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 9.djvu"
TITLE = "Էջ:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 9.djvu/100"


class FakeClient:
    """One page, whose wikitext prints the encyclopedia's 'derived from'."""

    def __init__(self) -> None:
        self.fetched: list[str] = []

    def pages_in_index(self, index_title: str, min_quality: int):
        yield PageRecord(title=TITLE, pageid=1, quality=4)

    def page_wikitext(self, title: str) -> tuple[str, int]:
        self.fetched.append(title)
        return "ԱՐՇԻՊԵԼԱԳ (<իտալ․ Arcipelago)", 42


def stale_harvest(out: Path) -> None:
    """A page saved by cleaning 1: its '<' already folded, no version."""
    (out / "text").mkdir(parents=True)
    (out / "text" / "100.txt").write_text("ԱՐՇԻՊԵԼԱԳ («իտալ․ Arcipelago)", encoding="utf-8")
    entry = {
        "title": TITLE,
        "pageid": 1,
        "quality": 4,
        "page_number": 100,
        "text": "text/100.txt",
        "revid": 7,
    }
    (out / "manifest.json").write_text(
        json.dumps({"index": INDEX, "min_quality": 3, "pages": [entry]}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_a_new_page_records_the_current_cleaning(tmp_path: Path) -> None:
    [entry] = harvest(FakeClient(), INDEX, tmp_path)

    assert entry["cleaning"] == wikisource.TRANSCRIPT_CLEANING
    assert "(<իտալ" in (tmp_path / "text" / "100.txt").read_text(encoding="utf-8")


def test_resuming_keeps_a_stale_page_visibly_stale(tmp_path: Path) -> None:
    stale_harvest(tmp_path)
    client = FakeClient()

    [entry] = harvest(client, INDEX, tmp_path)

    assert client.fetched == [], "a plain re-run resumes past existing text"
    assert entry["cleaning"] == 1


def test_refresh_stale_fetches_and_recleans(tmp_path: Path) -> None:
    stale_harvest(tmp_path)
    client = FakeClient()

    [entry] = harvest(client, INDEX, tmp_path, refresh_stale=True)

    assert client.fetched == [TITLE]
    assert entry["cleaning"] == wikisource.TRANSCRIPT_CLEANING
    assert "(<իտալ" in (tmp_path / "text" / "100.txt").read_text(encoding="utf-8")


def test_refresh_stale_leaves_current_pages_alone(tmp_path: Path) -> None:
    harvest(FakeClient(), INDEX, tmp_path)
    client = FakeClient()

    harvest(client, INDEX, tmp_path, refresh_stale=True)

    assert client.fetched == []
