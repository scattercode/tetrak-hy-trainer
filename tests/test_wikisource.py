"""Wikisource harvesting, tested without the network.

The cleaner is tested against a fragment captured verbatim from the live
API (volume 1, page 100, 2026-08-29), so the tests exercise the markup the
encyclopedia actually uses, not an invented approximation. The client is
tested through an injected fake session replaying canned API responses.
"""

from __future__ import annotations

import json

import pytest

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
    """Replays canned JSON responses in order; records what was requested.

    A request with no query parameters is a raw file download (the image
    fetch, which does not go through the API), so it is served from
    :attr:`image_bytes` rather than consuming a queued JSON payload --
    otherwise every image would eat the response meant for the next API
    call.
    """

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.requests: list[dict] = []
        self.headers: dict = {}
        self.image_bytes = b""
        self.downloaded: list[str] = []

    def get(self, url, params=None, timeout=None):
        if params is None:
            self.downloaded.append(url)
            return FakeResponse(content=self.image_bytes)
        self.requests.append(params)
        return FakeResponse(self.payloads.pop(0))


class FakeResponse:
    def __init__(self, payload: dict | None = None, content: bytes = b""):
        self.payload = payload
        self.content = content

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


class TestParsePageSpec:
    def test_a_range(self) -> None:
        from tetrak_hy_trainer.harvest import parse_page_spec

        assert parse_page_spec("20-24") == {20, 21, 22, 23, 24}

    def test_single_pages_and_ranges_together(self) -> None:
        from tetrak_hy_trainer.harvest import parse_page_spec

        assert parse_page_spec("5,20-22,99") == {5, 20, 21, 22, 99}

    def test_whitespace_and_empty_parts_are_tolerated(self) -> None:
        from tetrak_hy_trainer.harvest import parse_page_spec

        assert parse_page_spec(" 5 , 7 ,") == {5, 7}

    def test_a_backwards_range_is_rejected(self) -> None:
        from tetrak_hy_trainer.harvest import parse_page_spec

        with pytest.raises(ValueError, match="backwards"):
            parse_page_spec("80-20")

    def test_an_empty_spec_is_rejected(self) -> None:
        from tetrak_hy_trainer.harvest import parse_page_spec

        with pytest.raises(ValueError):
            parse_page_spec(",")


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

    def test_a_limited_run_keeps_pages_it_did_not_visit(self, tmp_path) -> None:
        """The manifest is provenance -- which revision of which transcript
        trained a published model. Writing only the visited pages would have
        cut volume 1's 717-page record down to the --limit."""
        from tetrak_hy_trainer.harvest import harvest

        (tmp_path / "text").mkdir()
        (tmp_path / "text" / "5.txt").write_text("արդեն կա", encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "index": "Ինդեքս:Vol.djvu",
                    "min_quality": 3,
                    "pages": [
                        {"page_number": 5, "text": "text/5.txt", "revid": 111},
                        {"page_number": 900, "text": "text/900.txt", "revid": 222},
                    ],
                }
            ),
            encoding="utf-8",
        )

        session = FakeSession(
            [
                _pages_payload(
                    {"1": {"pageid": 10, "title": "Էջ:Vol.djvu/5", "proofread": {"quality": 4}}}
                )
            ]
        )
        harvest(WikisourceClient(session=session, pause=0), "Ինդեքս:Vol.djvu", tmp_path, limit=1)

        saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert [entry["page_number"] for entry in saved["pages"]] == [5, 900]
        assert saved["pages"][1]["revid"] == 222  # untouched page kept its provenance

    def test_images_top_up_a_text_only_harvest(self, tmp_path) -> None:
        """v0's and v1's volumes were harvested text-only, and real-crop
        harvesting needs their scans. While the image fetch sat inside the
        "new page" branch, re-running with --images over them downloaded
        nothing and reported success."""
        from tetrak_hy_trainer.harvest import harvest

        (tmp_path / "text").mkdir()
        (tmp_path / "text" / "5.txt").write_text("արդեն կա", encoding="utf-8")

        session = FakeSession(
            [
                _pages_payload(
                    {"1": {"pageid": 10, "title": "Էջ:Vol.djvu/5", "proofread": {"quality": 4}}}
                ),
                # The imageinfo lookup; no wikitext payload, so refetching
                # the existing text would still exhaust the fake and raise.
                {"query": {"pages": {"20": {"imageinfo": [{"thumburl": "https://x/5.jpg"}]}}}},
            ]
        )
        session.image_bytes = b"\xff\xd8jpeg"
        client = WikisourceClient(session=session, pause=0)

        manifest = harvest(client, "Ինդեքս:Vol.djvu", tmp_path, images=True)

        assert (tmp_path / "images" / "5.jpg").read_bytes() == b"\xff\xd8jpeg"
        assert manifest[0]["image"] == "images/5.jpg"
        assert (tmp_path / "text" / "5.txt").read_text(encoding="utf-8") == "արդեն կա"


class TestNormaliseTranscript:
    """Transcriber substitutions found by diffing the widened corpus.

    Same class of habit as the ASCII colon standing in for the Armenian
    full stop, which v3's fine-tune faithfully learnt from its labels.
    """

    def test_angle_brackets_become_guillemets(self) -> None:
        assert wikisource.normalise_transcript("<Ազգ> և <Պահակ>") == "«Ազգ» և «Պահակ»"

    def test_minus_sign_becomes_an_en_dash(self) -> None:
        """U+2212 is a near-perfect homoglyph of the en dash, so it is
        normalised rather than admitted to the charset."""
        assert wikisource.normalise_transcript("acceleratio − արագացում") == (
            "acceleratio – արագացում"
        )

    def test_horizontal_bar_becomes_an_em_dash(self) -> None:
        assert wikisource.normalise_transcript("― Վա՜յ") == "— Վա՜յ"

    def test_a_byte_order_mark_is_stripped(self) -> None:
        assert wikisource.normalise_transcript("﻿մոտենում") == "մոտենում"

    def test_it_is_idempotent(self) -> None:
        """Applied by clean_wikitext and again at read time, so it must be."""
        once = wikisource.normalise_transcript("<Ա> − բ")
        assert wikisource.normalise_transcript(once) == once

    def test_genuine_print_is_untouched(self) -> None:
        """Ellipsis, brackets and the numero sign are printed, not
        substituted -- they belong in the charset, not in this table."""
        text = "…[13(15)․7․1852] № 6473 կմ²"
        assert wikisource.normalise_transcript(text) == text

    def test_clean_wikitext_applies_it(self) -> None:
        assert "«Ազգ»" in wikisource.clean_wikitext("<Ազգ>")


class TestHarvestProvenanceGuards:
    """A harvest may lose time; it may not write something untrue.

    Both guards here come from the same incident. Topping up Baronian's
    scans with the index title retyped from memory rather than read from
    the manifest wrote a title the wiki does not know over 739 correctly
    harvested pages -- and printed "Harvested 739 page(s)" while doing
    it. The pages survived, because a re-run merges rather than replaces;
    what was destroyed was the record of where they came from, which is
    what heldout keys on and what the weights release cites.
    """

    def test_a_title_matching_nothing_is_an_error_not_a_success(self, tmp_path) -> None:
        """An unknown index is not an API error -- it simply has no pages.

        So the harvest completes, writes a manifest naming an index that
        does not exist, and reports success.
        """
        from tetrak_hy_trainer.harvest import HarvestError, harvest

        session = FakeSession([_pages_payload({})])
        client = WikisourceClient(session=session, pause=0)

        with pytest.raises(HarvestError, match="no pages"):
            harvest(client, "Ինդեքս:Mistyped.djvu", tmp_path)

        assert not (tmp_path / "manifest.json").exists()

    def test_an_existing_harvest_is_not_restamped_with_another_title(self, tmp_path) -> None:
        """The pages are kept by the merge; the provenance would not be."""
        from tetrak_hy_trainer.harvest import HarvestError, harvest

        (tmp_path / "text").mkdir()
        (tmp_path / "text" / "5.txt").write_text("արդեն կա", encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "index": "Ինդեքս:Real title.djvu",
                    "min_quality": 3,
                    "pages": [{"page_number": 5, "text": "text/5.txt", "revid": 111}],
                }
            ),
            encoding="utf-8",
        )

        session = FakeSession(
            [
                _pages_payload(
                    {"1": {"pageid": 10, "title": "Էջ:Vol.djvu/5", "proofread": {"quality": 4}}}
                )
            ]
        )

        with pytest.raises(HarvestError, match="different work"):
            harvest(WikisourceClient(session=session, pause=0), "Ինդեքս:Typo.djvu", tmp_path)

        saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert saved["index"] == "Ինդեքս:Real title.djvu"
        assert saved["pages"][0]["revid"] == 111

    def test_the_same_title_still_tops_up(self, tmp_path) -> None:
        """The guard must not block the ordinary --images top-up."""
        from tetrak_hy_trainer.harvest import harvest

        (tmp_path / "text").mkdir()
        (tmp_path / "text" / "5.txt").write_text("արդեն կա", encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "index": "Ինդեքս:Vol.djvu",
                    "min_quality": 3,
                    "pages": [{"page_number": 5, "text": "text/5.txt", "revid": 111}],
                }
            ),
            encoding="utf-8",
        )

        session = FakeSession(
            [
                _pages_payload(
                    {"1": {"pageid": 10, "title": "Էջ:Vol.djvu/5", "proofread": {"quality": 4}}}
                )
            ]
        )
        manifest = harvest(WikisourceClient(session=session, pause=0), "Ինդեքս:Vol.djvu", tmp_path)

        assert [entry["page_number"] for entry in manifest] == [5]
