"""Text similarity metrics for scoring a model against a transcript.

**A deliberate copy of Tetrak's ``tetrak_ocr.accuracy``**, not an import of
it. This repository is public and Tetrak's is private, so importing it made
the training and scoring scripts depend on a package an outside contributor
cannot install -- and one this repository does not declare. That import ran in
``train_synthetic.py``'s evaluation path, so ``--eval-dir`` failed for anyone
who had not also checked out the private repository, at the end of an
eleven-hour run.

The two implementations must agree, because every published figure for this
model is quoted alongside Tetrak's own benchmark numbers and the comparison is
only meaningful if the metric is the same one. They are twenty lines of
standard library and have not changed since they were written; if either side
ever does change, change both, and say so in the release notes -- a metric
that moved silently reprices every historical number.

``scripts/evaluate_baselines.py`` still imports from ``tetrak_ocr`` and
should: it benchmarks Tetrak's *backends*, so it can only run in Tetrak's
environment anyway.

Two metrics, both normalising first so trivial formatting differences do not
register:

character_similarity
    difflib.SequenceMatcher over the two strings. Order-sensitive, which is
    what makes it catch a reading-order failure -- the same words down the
    wrong column score far lower.

word_recall
    The fraction of expected words present anywhere in the output.
    Order-insensitive and recall-oriented, so it rewards capturing the
    content and tolerates reordering.
"""

from __future__ import annotations

import difflib
import re

# Armenian punctuation -> the ASCII character it is visually identical to.
_PUNCTUATION_HOMOGLYPHS = str.maketrans({"։": ":", "․": "."})

# The two-letter եւ and the ligature և are one thing for scoring (brief 014,
# decided 2026-10-08). Classical-orthography presses split 21 to 16 on which
# they set, reformed presses set the ligature, and transcribers follow the
# page or do not; ground truth stays as printed so the model learns to read
# both, and the metric refuses to rank an engine on which form it emitted.
# Applied after lowercasing, so Եւ and ԵՒ fold too.
_EW_LIGATURE = ("եւ", "և")


def _is_word(token: str) -> bool:
    """A token with at least one letter or digit; ``-`` or ``…`` alone is not."""
    return any(character.isalnum() for character in token)


def normalise(text: str) -> str:
    """Lowercase, equate Armenian punctuation homoglyphs, collapse whitespace.

    Two Armenian marks are scored as the ASCII character they print
    identically to: the full stop ``։`` as the colon, and the abbreviation
    dot ``․`` (U+2024) as the full stop. Transcribers type the ASCII form
    so often (the medical encyclopedia's transcripts use the colon
    throughout; every register uses ``.`` for ``․``) that holding an engine
    to either form would rank engines on which habit their output happens
    to share with the transcript, not on what they read. Mapped towards
    ASCII so that text with no Armenian in it is unaffected. The two-letter
    ``եւ`` is scored as the ligature ``և`` for the same reason (brief 014).
    """
    lowered = text.lower().translate(_PUNCTUATION_HOMOGLYPHS).replace(*_EW_LIGATURE)
    return re.sub(r"\s+", " ", lowered).strip()


def character_similarity(actual: str, expected: str) -> float:
    """Return character-level similarity as a ratio 0.0-1.0.

    Args:
        actual:   The text produced by the model.
        expected: The reference (ground-truth) text.

    Returns:
        A float in [0.0, 1.0]. 1.0 means the texts are identical after
        normalisation.

    ``autojunk`` is off, as in Tetrak's copy (brief 013): with difflib's
    default, a page-length string has its common characters treated as
    junk and the ratio stops measuring how much of the text was read.
    ``align`` and ``confusion`` already turned it off for the same reason.
    """
    return difflib.SequenceMatcher(
        None, normalise(actual), normalise(expected), autojunk=False
    ).ratio()


def word_recall(actual: str, expected: str) -> float:
    """Return the fraction of expected words present in the output.

    Args:
        actual:   The text produced by the model.
        expected: The reference (ground-truth) text.

    Returns:
        A float in [0.0, 1.0]. 1.0 means every expected word was found.
    """
    # Punctuation-only tokens are not words. The medical encyclopedia's
    # index prints dot leaders that its transcripts type as a standalone
    # "-", and counting those charged every engine that read the page
    # correctly with a missed word per index entry (brief 013).
    expected_words = [word for word in normalise(expected).split() if _is_word(word)]
    if not expected_words:
        return 1.0
    actual_words = set(normalise(actual).split())
    found = sum(1 for word in expected_words if word in actual_words)
    return found / len(expected_words)
