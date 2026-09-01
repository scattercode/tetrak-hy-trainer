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


def normalise(text: str) -> str:
    """Lowercase and collapse all whitespace to a single space."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def character_similarity(actual: str, expected: str) -> float:
    """Return character-level similarity as a ratio 0.0-1.0.

    Args:
        actual:   The text produced by the model.
        expected: The reference (ground-truth) text.

    Returns:
        A float in [0.0, 1.0]. 1.0 means the texts are identical after
        normalisation.
    """
    return difflib.SequenceMatcher(None, normalise(actual), normalise(expected)).ratio()


def word_recall(actual: str, expected: str) -> float:
    """Return the fraction of expected words present in the output.

    Args:
        actual:   The text produced by the model.
        expected: The reference (ground-truth) text.

    Returns:
        A float in [0.0, 1.0]. 1.0 means every expected word was found.
    """
    expected_words = normalise(expected).split()
    if not expected_words:
        return 1.0
    actual_words = set(normalise(actual).split())
    found = sum(1 for word in expected_words if word in actual_words)
    return found / len(expected_words)
