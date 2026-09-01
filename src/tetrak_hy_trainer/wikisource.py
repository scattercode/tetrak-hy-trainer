"""Harvest proofread page text and page scans from Armenian Wikisource.

The Armenian Soviet Encyclopedia (see the README's data table) lives on
hy.wikisource.org as ProofreadPage indexes: one ``Ինդեքս:`` page per DJVU
volume, one ``Էջ:<volume>.djvu/<n>`` page per scanned page, each carrying a
proofread quality level. This module speaks to the MediaWiki API to
enumerate those pages, filter by quality, fetch their wikitext, and resolve
a rendered image URL for the matching scan.

Every API shape here was verified against the live hy.wikisource.org API
(2026-08-29), not inferred from documentation — the ProofreadPage generator
prefix in particular (``gprppii``) is not guessable.

**Quality is the load-bearing filter.** Wikisource seeds unproofread pages
with machine OCR; training on those teaches the model another engine's
mistakes. The ProofreadPage levels are 0–4; the default floor of 3
("proofread") admits human-verified text only, and 4 ("validated") is
stricter still. Nothing in this module ever defaults below 3.

Network use is deliberately gentle: one shared session, a descriptive
User-Agent, ``maxlag`` set per Wikimedia etiquette, and a configurable
pause between requests.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

API_URL = "https://hy.wikisource.org/w/api.php"
USER_AGENT = "tetrak-hy-trainer/0.1 (https://github.com/scattercode/tetrak-hy-trainer)"

# ProofreadPage quality levels. 3 and 4 are human-verified; everything
# below is untrusted for training purposes.
QUALITY_PROOFREAD = 3
QUALITY_VALIDATED = 4


@dataclass(frozen=True)
class PageRecord:
    """One scanned page: its transcription page and proofread status."""

    title: str  # e.g. "Էջ:… 1.djvu/100"
    pageid: int
    quality: int

    @property
    def page_number(self) -> int:
        """The DJVU page number, from the title's trailing ``/<n>``."""
        return int(self.title.rsplit("/", 1)[1])


class WikisourceClient:
    """A thin, polite client for the hy.wikisource.org API.

    Args:
        session: Injectable for tests; defaults to a fresh
            :class:`requests.Session` with our User-Agent.
        pause: Seconds to sleep between successive API requests. Zero in
            tests; keep the default for real harvesting.
    """

    def __init__(self, session: requests.Session | None = None, pause: float = 0.5):
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            # A multi-thousand-page harvest will meet transient failures:
            # the first 5,500-page tranche died at page ~660 when the
            # server closed a connection mid-request. Retry the transport
            # errors and the throttling statuses with exponential backoff;
            # anything that survives four attempts is a real problem and
            # still raises. Only installed on the default session, so
            # tests injecting a fake see every request exactly once.
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry = Retry(
                total=4,
                connect=4,
                read=4,
                status=4,
                backoff_factor=2.0,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session = session
        self.pause = pause
        self._last_request = 0.0

    def _get(self, **params) -> dict:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.pause:
            time.sleep(self.pause - elapsed)
        response = self.session.get(
            API_URL,
            params={"format": "json", "maxlag": "5", **params},
            timeout=30,
        )
        self._last_request = time.monotonic()
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"API error: {payload['error']}")
        return payload

    def pages_in_index(self, index_title: str, min_quality: int = QUALITY_PROOFREAD):
        """Yield :class:`PageRecord` for every page at or above *min_quality*.

        Follows API continuation, so a 700-page volume arrives complete.

        Args:
            index_title: The full index title, e.g.
                ``"Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 1.djvu"``.
            min_quality: ProofreadPage floor; never set below
                :data:`QUALITY_PROOFREAD` for training data.
        """
        continuation: dict = {}
        while True:
            payload = self._get(
                action="query",
                generator="proofreadpagesinindex",
                gprppiititle=index_title,
                prop="proofread",
                **continuation,
            )
            for page in payload.get("query", {}).get("pages", {}).values():
                quality = page.get("proofread", {}).get("quality")
                if quality is None or quality < min_quality:
                    continue
                yield PageRecord(title=page["title"], pageid=page["pageid"], quality=quality)
            continuation = payload.get("continue", {})
            if not continuation:
                return

    def page_wikitext(self, title: str) -> tuple[str, int]:
        """Return ``(wikitext, revision_id)`` for one transcription page.

        The revision id is provenance: it pins exactly which state of the
        transcript a harvest saw, so a later re-harvest can tell whether
        anything changed.
        """
        payload = self._get(
            action="query",
            titles=title,
            prop="revisions",
            rvprop="content|ids",
            rvslots="main",
        )
        page = next(iter(payload["query"]["pages"].values()))
        revision = page["revisions"][0]
        return revision["slots"]["main"]["*"], revision["revid"]

    def page_image_url(self, file_title: str, page_number: int, width: int = 2048) -> str:
        """Return a URL rendering one DJVU page as a JPEG at *width* pixels.

        Args:
            file_title: The file page title, e.g.
                ``"Պատկեր:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 1.djvu"``.
            page_number: 1-based page within the DJVU.
            width: Rendered width in pixels; the source scans are ~3300 px
                wide, so the default keeps most of the resolution.
        """
        payload = self._get(
            action="query",
            titles=file_title,
            prop="imageinfo",
            iiprop="url",
            iiurlparam=f"page{page_number}-{width}px",
        )
        page = next(iter(payload["query"]["pages"].values()))
        return page["imageinfo"][0]["thumburl"]


# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------

_NOINCLUDE = re.compile(r"<noinclude>.*?</noinclude>", re.DOTALL)
_SECTION_TAG = re.compile(r"<section[^>]*/>")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
# Innermost-first template removal; applied repeatedly for nesting.
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_WIKILINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
_EMPHASIS = re.compile(r"'{2,}")


def clean_wikitext(text: str) -> str:
    """Reduce a transcription page's wikitext to its plain page text.

    Strips the ProofreadPage ``<noinclude>`` header/footer (quality tag and
    running header), ``<section>`` transclusion markers, references,
    templates, HTML tags and wiki markup, keeping link display text. The
    result is the page's prose — suitable as corpus text for synthesis and
    as the transcript side of a page/scan pair.

    Deliberately conservative: unfamiliar markup is stripped rather than
    interpreted, and whitespace is normalised per line, preserving the
    paragraph structure the transcribers encoded.
    """
    text = _NOINCLUDE.sub("", text)
    text = _SECTION_TAG.sub("", text)
    text = _REF.sub("", text)
    while _TEMPLATE.search(text):
        text = _TEMPLATE.sub("", text)
    text = _WIKILINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = _EMPHASIS.sub("", text)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def index_to_file_title(index_title: str) -> str:
    """Map an ``Ինդեքս:`` title to its ``Պատկեր:`` (file) title."""
    prefix = "Ինդեքս:"
    if not index_title.startswith(prefix):
        raise ValueError(f"not an index title: {index_title!r}")
    return "Պատկեր:" + index_title[len(prefix) :]
