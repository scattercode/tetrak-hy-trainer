"""Tell classical Armenian orthography from reformed, from the text alone.

Brief 014 Stage 0.1. The 1922 reform changed no letters, only where they
appear, so a classical page and a reformed page share a charset and differ
in distribution. Index titles cannot be trusted to say which a work is:
Wikisource titles are typed by the uploader, and Kajaznuni's classical
*Ազգ և հայրենիք* is titled with a reformed ``և``. Only the pages can say.

The discriminating marker is **ւ (U+0582) outside the ու digraph**. Reformed
spelling writes ``ւ`` only inside ``ու`` and ``և``; classical spelling uses it
freely (``նաւ``, ``իւր``, ``եւ``, ``հաւատք``). Measured on 2026-10-08 per
thousand Armenian letters, every reformed harvest under ``runs/harvest/``
sits between 0.04 and 1.00 (the bilingual dictionary aside), and sampled
classical pages sit between 16.8 and 21.4 (Toranian, Kajaznuni, Baronian's
*Ազգային Ջոջեր*, *Հայկական տպագրութիւն*). Nothing fell between 1 and 16.

Two markers that look useful and are not, kept here so nobody re-tries them:

- **``եւ`` against ``և``.** A per-press typographic habit, not an
  orthography. *Հայկական տպագրութիւն* is classical and sets the ``և``
  ligature (2 against 22 in the sample); Toranian sets both.
- **Inner ``է`` and final ``յ``** move the right way (classical 7–30 and
  2–4 per thousand against reformed 0–1) but overlap at the edges and add
  nothing once ``ւ`` has spoken. They are reported, not used to decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ARMENIAN_LETTER = re.compile(r"[Ա-Ֆա-և]")
#: ``ւ`` not preceded by ``ո``/``Ո`` -- that is, not the second half of ``ու``.
FREE_YIWN = re.compile(r"(?<![ոՈ])ւ")
TWO_LETTER_EW = re.compile(r"[եԵ]ւ")
LIGATURE_EW = re.compile(r"և")
#: A vowel followed by ``յ`` at the end of a token: the classical silent final.
FINAL_Y = re.compile(r"[աոեէիօ]յ(?![Ա-Ֆա-և])")
#: ``է`` after another lowercase letter, where reformed spelling writes ``ե``.
INNER_E = re.compile(r"(?<=[ա-և])է")

#: Free-standing ``ւ`` per thousand letters at or above this is classical.
CLASSICAL_YIWN = 8.0
#: At or below this is reformed; the band between is reported as mixed.
REFORMED_YIWN = 2.0
#: Below this many Armenian letters a sample says nothing.
MIN_LETTERS = 500

CLASSICAL = "classical"
REFORMED = "reformed"
MIXED = "mixed"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Markers:
    """Orthography marker rates for one text, per thousand Armenian letters."""

    letters: int
    free_yiwn: float
    final_y: float
    inner_e: float
    two_letter_ew: int
    ligature_ew: int

    @property
    def verdict(self) -> str:
        if self.letters < MIN_LETTERS:
            return UNKNOWN
        if self.free_yiwn >= CLASSICAL_YIWN:
            return CLASSICAL
        if self.free_yiwn <= REFORMED_YIWN:
            return REFORMED
        return MIXED

    def as_dict(self) -> dict:
        return {
            "letters": self.letters,
            "free_yiwn": round(self.free_yiwn, 2),
            "final_y": round(self.final_y, 2),
            "inner_e": round(self.inner_e, 2),
            "two_letter_ew": self.two_letter_ew,
            "ligature_ew": self.ligature_ew,
            "verdict": self.verdict,
        }


def markers(text: str) -> Markers:
    """Measure *text*'s orthography markers."""
    letters = len(ARMENIAN_LETTER.findall(text))
    per_thousand = 1000.0 / letters if letters else 0.0
    return Markers(
        letters=letters,
        free_yiwn=len(FREE_YIWN.findall(text)) * per_thousand,
        final_y=len(FINAL_Y.findall(text)) * per_thousand,
        inner_e=len(INNER_E.findall(text)) * per_thousand,
        two_letter_ew=len(TWO_LETTER_EW.findall(text)),
        ligature_ew=len(LIGATURE_EW.findall(text)),
    )


def classify(text: str) -> str:
    """One of :data:`CLASSICAL`, :data:`REFORMED`, :data:`MIXED`, :data:`UNKNOWN`."""
    return markers(text).verdict
