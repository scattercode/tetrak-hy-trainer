"""Wikisource harvesting, tested without the network.

The cleaner is tested against a fragment captured verbatim from the live
API (volume 1, page 100, 2026-08-29), so the tests exercise the markup the
encyclopedia actually uses, not an invented approximation. The client is
tested through an injected fake session replaying canned API responses.
"""

from __future__ import annotations

import json

from tetrak_hy_trainer import wikisource
from tetrak_hy_trainer.wikisource import PageRecord, WikisourceClient, clean_wikitext

# Captured from the live API: the opening of Էջ:…1.djvu/100. The noinclude
# header carries the pagequality tag and the running header ("100
# ԱԶԱՏՈՒԹՅՈՒՆ"), which must not leak into the page text.
REAL_PAGE_OPENING = (
    '<noinclude><pagequality level="4" user="Ruzannahovhannisyan2005" />'
    "100 ԱԶԱՏՈՒԹՅՈՒՆ</noinclude>"
    '<section begin="Ազատություն և անհրաժեշտություն"/>'
    "և ա֊յան դիալեկտիկական փոխկապակցությունը, դրանք քննվում են "
    "իրական հարաբերություններից դուրս։"
)


class TestCleanWikitext:
    def test_strips_the_proofreadpage_header(self) -> None:
        cleaned = clean_wikitext(REAL_PAGE_OPENING)
        assert "pagequality" not in cleaned
        assert "ԱԶԱՏՈՒԹՅՈՒՆ" not in cleaned  # the running header
        assert "noinclude" not in cleaned

    def test_strips_section_markers_but_keeps_the_prose(self) -> None:
        cleaned = clean_wikitext(REAL_PAGE_OPENING)
        assert "section" not in cleaned
        assert cleaned.startswith("և ա֊յան դիալեկտիկական")
        assert cleaned.endswith("։")

    def test_preserves_armenian_typography(self) -> None:
        """The ֊ abbreviation hyphen and ։ full stop are page text, not
        markup — exactly the characters the recogniser must learn."""
        cleaned = clean_wikitext(REAL_PAGE_OPENING)
        assert "֊" in cleaned
        assert "։" in cleaned

    def test_removes_templates_including_nested(self) -> None:
        assert clean_wikitext("ա {{կաղապար|{{ներդիր}}}} բ") == "ա բ"

    def test_keeps_wikilink_display_text(self) -> None:
        assert clean_wikitext("[[Թիրախ|ցուցադրվող]] տեքստ") == "ցուցադրվող տեքստ"
        assert clean_wikitext("[[Պարզ հղում]]") == "Պարզ հղում"

    def test_collapses_whitespace_but_keeps_paragraphs(self) -> None:
        cleaned = clean_wikitext("առաջին  տող\n\n\n\nերկրորդ   տող")
        assert cleaned == "առաջին տող\n\nերկրորդ տող"


class FakeSession:
    """Replays canned JSON responses in order; records what was requested."""

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.requests: list[dict] = []
        self.headers: dict = {}

    def get(self, url, params=None, timeout=None):
        self.requests.append(params or {})
        return FakeResponse(self.payloads.pop(0))


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


def _pages_payload(pages: dict, cont: dict | None = None) -> dict:
    payload = {"query": {"pages": pages}}
    if cont:
        payload["continue"] = cont
    return payload


class TestPagesInIndex:
    def test_filters_below_the_quality_floor(self) -> None:
        session = FakeSession(
            [
                _pages_payload(
                    {
                        "1": {"pageid": 1, "title": "Էջ:Vol.djvu/1", "proofread": {"quality": 0}},
                        "2": {"pageid": 2, "title": "Էջ:Vol.djvu/2", "proofread": {"quality": 3}},
                        "3": {"pageid": 3, "title": "Էջ:Vol.djvu/3", "proofread": {"quality": 4}},
                    }
                )
            ]
        )
        client = WikisourceClient(session=session, pause=0)

        records = list(client.pages_in_index("Ինդեքս:Vol.djvu"))

        assert [r.pageid for r in records] == [2, 3]
        assert all(r.quality >= 3 for r in records)

    def test_follows_api_continuation(self) -> None:
        session = FakeSession(
            [
                _pages_payload(
                    {"1": {"pageid": 1, "title": "Էջ:Vol.djvu/1", "proofread": {"quality": 4}}},
                    cont={"gprppiicontinue": "x", "continue": "gprppii"},
                ),
                _pages_payload(
                    {"2": {"pageid": 2, "title": "Էջ:Vol.djvu/2", "proofread": {"quality": 4}}}
                ),
            ]
        )
        client = WikisourceClient(session=session, pause=0)

        records = list(client.pages_in_index("Ինդեքս:Vol.djvu"))

        assert len(records) == 2
        # The second request must carry the continuation token back.
        assert session.requests[1]["gprppiicontinue"] == "x"

    def test_page_number_comes_from_the_title(self) -> None:
        record = PageRecord(title="Էջ:Vol.djvu/307", pageid=9, quality=4)
        assert record.page_number == 307


class TestTitleMapping:
    def test_index_title_maps_to_file_title(self) -> None:
        assert wikisource.index_to_file_title("Ինդեքս:Vol 1.djvu") == "Պատկեր:Vol 1.djvu"

    def test_a_non_index_title_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            wikisource.index_to_file_title("Էջ:Vol 1.djvu/5")


class TestHarvest:
    def test_writes_text_and_manifest_with_provenance(self, tmp_path) -> None:
        from tetrak_hy_trainer.harvest import harvest

        session = FakeSession(
            [
                _pages_payload(
                    {"1": {"pageid": 10, "title": "Էջ:Vol.djvu/5", "proofread": {"quality": 4}}}
                ),
                {
                    "query": {
                        "pages": {
                            "10": {
                                "revisions": [
                                    {
                                        "revid": 777,
                                        "slots": {
                                            "main": {
                                                "*": "<noinclude>h</noinclude>Բովանդակություն։"
                                            }
                                        },
                                    }
                                ]
                            }
                        }
                    }
                },
            ]
        )
        client = WikisourceClient(session=session, pause=0)

        manifest = harvest(client, "Ինդեքս:Vol.djvu", tmp_path)

        assert (tmp_path / "text" / "5.txt").read_text(encoding="utf-8") == "Բովանդակություն։"
        saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert saved["min_quality"] == 3
        [entry] = saved["pages"]
        assert entry["revid"] == 777  # provenance: the exact revision harvested
        assert entry["quality"] == 4
        assert manifest == saved["pages"]

    def test_resumes_without_refetching_existing_pages(self, tmp_path) -> None:
        from tetrak_hy_trainer.harvest import harvest

        (tmp_path / "text").mkdir()
        (tmp_path / "text" / "5.txt").write_text("արդեն կա", encoding="utf-8")

        session = FakeSession(
            [
                _pages_payload(
                    {"1": {"pageid": 10, "title": "Էջ:Vol.djvu/5", "proofread": {"quality": 4}}}
                )
                # No second payload: fetching wikitext would exhaust the fake
                # and raise, so completing proves nothing was refetched.
            ]
        )
        client = WikisourceClient(session=session, pause=0)

        manifest = harvest(client, "Ինդեքս:Vol.djvu", tmp_path)

        assert len(manifest) == 1
        assert manifest[0]["revid"] is None  # kept, not refetched
        assert (tmp_path / "text" / "5.txt").read_text(encoding="utf-8") == "արդեն կա"
