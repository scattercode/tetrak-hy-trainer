"""The character confusion table -- the fine-tune's scorecard.

Entries read truth → predicted, the direction v1's error analysis used,
so the tables can be compared row for row.
"""

from __future__ import annotations

from tetrak_hy_trainer import confusion
from tetrak_hy_trainer.confusion import EMPTY


class TestCharacterConfusions:
    def test_a_single_substitution_is_counted_truth_first(self) -> None:
        """The analysis writes "հ → h 302": the page had հ, the model
        read h."""
        counts = confusion.character_confusions([("hայ", "հայ")])
        assert counts["հ", "h"] == 1

    def test_identical_pairs_contribute_nothing(self) -> None:
        assert confusion.character_confusions([("հայ", "հայ")]) == {}

    def test_counts_accumulate_across_pairs(self) -> None:
        counts = confusion.character_confusions([("hայ", "հայ")] * 5)
        assert counts["հ", "h"] == 5

    def test_several_substitutions_in_one_word(self) -> None:
        counts = confusion.character_confusions([("haյ", "հայ")])
        assert counts["հ", "h"] == 1
        assert counts["ա", "a"] == 1

    def test_a_dropped_character_is_counted_against_empty(self) -> None:
        counts = confusion.character_confusions([("հա", "հայ")])
        assert counts["յ", EMPTY] == 1

    def test_an_invented_character_is_counted_against_empty(self) -> None:
        counts = confusion.character_confusions([("հայք", "հայ")])
        assert counts[EMPTY, "ք"] == 1

    def test_an_unequal_run_is_kept_whole_rather_than_forced_into_pairs(self) -> None:
        """Honest beats tidy: there is no character-for-character reading
        of a run that changed length."""
        counts = confusion.character_confusions([("axb", "aիրb")])
        assert counts["իր", "x"] == 1

    def test_the_shape_cluster_the_finetune_targets(self) -> None:
        """հ→խ and խ→ի are the confusions real crops exist to fix."""
        counts = confusion.character_confusions([("խայ", "հայ"), ("իայ", "խայ")])
        assert counts["հ", "խ"] == 1
        assert counts["խ", "ի"] == 1

    def test_no_pairs_gives_an_empty_table(self) -> None:
        assert confusion.character_confusions([]) == {}


class TestFormatTable:
    def test_rows_are_ordered_by_count(self) -> None:
        counts = confusion.character_confusions([("hայ", "հայ")] * 3 + [("խայ", "հայ")])
        table = confusion.format_table(counts)
        assert table.index("հ → h") < table.index("հ → խ")

    def test_the_limit_is_respected(self) -> None:
        counts = confusion.character_confusions([("ax", "աբ"), ("by", "բգ"), ("cz", "գդ")])
        body = confusion.format_table(counts, limit=2).splitlines()[2:]
        assert len(body) == 2

    def test_an_empty_table_still_has_its_header(self) -> None:
        assert confusion.format_table(confusion.character_confusions([])).splitlines() == [
            "| confusion | count |",
            "|---|---|",
        ]
