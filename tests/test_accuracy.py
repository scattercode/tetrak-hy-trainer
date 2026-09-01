"""The scoring metrics, and the properties the published figures rely on.

`tetrak_hy_trainer.accuracy` is a deliberate copy of Tetrak's
`tetrak_ocr.accuracy` -- see that module's docstring for why a public
repository cannot import from a private one. These tests pin the behaviour
the copy has to keep, so a divergence is caught here rather than by two
figures quietly ceasing to be comparable.
"""

from __future__ import annotations

import pytest

from tetrak_hy_trainer.accuracy import character_similarity, normalise, word_recall

ARMENIAN = "Հայկական Սովետական Հանրագիտարան"


class TestNormalise:
    def test_case_and_whitespace_are_flattened(self):
        assert normalise("  The  QUICK\n\nbrown ") == "the quick brown"

    def test_armenian_lowercases(self):
        assert normalise("ՀԱՅԿԱԿԱՆ") == "հայկական"


class TestCharacterSimilarity:
    def test_identical_after_normalisation_scores_one(self):
        assert character_similarity(f"  {ARMENIAN}  ", ARMENIAN) == 1.0

    def test_nothing_in_common_scores_zero(self):
        assert character_similarity("aaaa", "bbbb") == 0.0

    def test_it_is_order_sensitive(self):
        """The property that makes it catch a reading-order failure.

        Two columns read across the gutter produce all the right words in the
        wrong order; word recall cannot see that and this must.
        """
        words = "alpha beta gamma delta"
        reversed_words = "delta gamma beta alpha"
        assert character_similarity(reversed_words, words) < 0.6


class TestWordRecall:
    def test_every_word_found_scores_one(self):
        assert word_recall("gamma alpha beta", "alpha beta gamma") == 1.0

    def test_it_is_order_insensitive(self):
        """Deliberate: the counterpart to character similarity."""
        assert word_recall("delta gamma beta alpha", "alpha beta gamma delta") == 1.0

    def test_half_the_words_scores_a_half(self):
        assert word_recall("alpha beta", "alpha beta gamma delta") == pytest.approx(0.5)

    def test_extra_words_are_not_penalised(self):
        """Recall, not precision -- a noisy transcript is not punished here."""
        assert word_recall("alpha beta gamma noise noise", "alpha beta gamma") == 1.0

    def test_an_empty_expectation_is_vacuously_met(self):
        assert word_recall("anything", "") == 1.0

    def test_an_empty_transcript_recalls_nothing(self):
        assert word_recall("", ARMENIAN) == 0.0
