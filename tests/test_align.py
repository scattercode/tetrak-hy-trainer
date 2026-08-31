"""Detector-box-to-transcript alignment.

The module fails closed by design -- see its docstring -- so most of what
matters here is what it *refuses* to pair, not what it pairs. A
mislabelled crop teaches the model the wrong shape, which is worse than
no crop at all.

Pure string and geometry handling; no OCR stack needed.
"""

from __future__ import annotations

from tetrak_hy_trainer import align
from tetrak_hy_trainer.align import Detection, Tier


def box(left: float, top: float, right: float, bottom: float) -> tuple:
    return (left, top, right, bottom)


def detection(text: str, left: float, top: float, width: float = 60, height: float = 20):
    return Detection(text=text, bbox=box(left, top, left + width, top + height))


class TestPairTokens:
    def test_identical_sequences_pair_exactly(self) -> None:
        tokens = ["ա", "բ", "գ"]
        pairs = align.pair_tokens(tokens, tokens)
        assert pairs == [(0, 0, Tier.EXACT), (1, 1, Tier.EXACT), (2, 2, Tier.EXACT)]

    def test_a_near_miss_between_anchors_is_paired_and_labelled_from_truth(self) -> None:
        """The whole point: a misread word keeps its correct label."""
        predicted = ["ա", "hայ", "գ"]
        truth = ["ա", "հայ", "գ"]
        pairs = align.pair_tokens(predicted, truth)
        assert (1, 1, Tier.NEAR) in pairs

    def test_gibberish_between_anchors_is_bracketed_not_near(self) -> None:
        """No similarity left to measure, but both sides agree one token
        sits in the gap and it is flanked by exact matches."""
        pairs = align.pair_tokens(["ա", "ձձքթյ", "գ"], ["ա", "պատմ", "գ"])
        assert (1, 1, Tier.BRACKETED) in pairs

    def test_an_unequal_run_is_dropped_entirely(self) -> None:
        """A split or merged box: no way to say which token is which."""
        pairs = align.pair_tokens(["ա", "հայ", "աստան", "գ"], ["ա", "հայաստան", "գ"])
        assert [tier for _, _, tier in pairs] == [Tier.EXACT, Tier.EXACT]
        assert {index for index, _, _ in pairs} == {0, 3}

    def test_a_long_dissimilar_run_is_dropped_even_between_anchors(self) -> None:
        """Position alone stops being evidence as the run grows."""
        predicted = ["ա", "xxxx", "yyyy", "zzzz", "գ"]
        truth = ["ա", "մեկը", "երկու", "երեք", "գ"]
        pairs = align.pair_tokens(predicted, truth, bracket_span=2)
        assert all(tier is Tier.EXACT for _, _, tier in pairs)

    def test_an_unflanked_mismatch_is_not_bracketed(self) -> None:
        """A dissimilar run at the very start has an anchor on one side
        only, so position is not corroborated."""
        pairs = align.pair_tokens(["xxxx", "բ", "գ"], ["առաջին", "բ", "գ"])
        assert all(tier is Tier.EXACT for _, _, tier in pairs)

    def test_deletions_and_insertions_are_dropped(self) -> None:
        pairs = align.pair_tokens(["ա", "գ"], ["ա", "բ", "գ"])
        assert [tier for _, _, tier in pairs] == [Tier.EXACT, Tier.EXACT]

    def test_indices_point_at_the_right_tokens(self) -> None:
        predicted = ["ա", "hայ", "գ"]
        truth = ["ա", "հայ", "գ"]
        for predicted_index, truth_index, _ in align.pair_tokens(predicted, truth):
            assert predicted_index == truth_index

    def test_empty_inputs_pair_nothing(self) -> None:
        assert align.pair_tokens([], []) == []
        assert align.pair_tokens(["ա"], []) == []
        assert align.pair_tokens([], ["ա"]) == []

    def test_common_tokens_still_anchor_on_a_long_page(self) -> None:
        """difflib's autojunk heuristic would treat a token appearing in
        more than 1% of a 200+ sequence as junk -- which on a page of
        Armenian prose is exactly the common words that anchor
        everything. It must stay off."""
        tokens = ["և", "որ"] * 150
        pairs = align.pair_tokens(tokens, tokens)
        assert len(pairs) == len(tokens)


class TestFindGutter:
    def test_two_columns_are_found(self) -> None:
        left = [detection("ա", 100, 100 + row * 30) for row in range(10)]
        right = [detection("բ", 500, 100 + row * 30) for row in range(10)]
        gutter = align.find_gutter(left + right)
        assert gutter is not None
        assert 160 < gutter < 500

    def test_a_single_column_page_reports_no_gutter(self) -> None:
        """Full-width lines cross any candidate, so nothing looks like a
        gutter."""
        spans = [detection("ա", 100, 100 + row * 30, width=500) for row in range(10)]
        assert align.find_gutter(spans) is None

    def test_too_few_boxes_to_judge(self) -> None:
        assert align.find_gutter([detection("ա", 100, 100)]) is None

    def test_a_lopsided_split_is_refused(self) -> None:
        """One stray box on the right is not a column."""
        spans = [detection("ա", 100, 100 + row * 30) for row in range(20)]
        spans.append(detection("բ", 600, 100))
        assert align.find_gutter(spans) is None


class TestColumnOrder:
    def test_columns_are_read_one_after_the_other(self) -> None:
        """Not straight across the gutter, which is what grouping by
        baseline alone would do."""
        detections = []
        for row in range(6):
            detections.append(detection(f"L{row}", 100, 100 + row * 30))
            detections.append(detection(f"R{row}", 500, 100 + row * 30))
        ordered = [d.text for d in align.column_order(detections)]
        assert ordered == [f"L{row}" for row in range(6)] + [f"R{row}" for row in range(6)]

    def test_a_box_straddling_the_gutter_is_dropped(self) -> None:
        """Running headers and page numbers span the columns and are
        absent from the transcripts, so they have nothing to pair with."""
        detections = []
        for row in range(6):
            detections.append(detection("L", 100, 200 + row * 30))
            detections.append(detection("R", 500, 200 + row * 30))
        detections.append(detection("HEADER", 100, 100, width=460))
        ordered = [d.text for d in align.column_order(detections)]
        assert "HEADER" not in ordered
        assert len(ordered) == 12

    def test_words_on_one_line_read_left_to_right(self) -> None:
        detections = [
            detection("երկու", 200, 100),
            detection("մեկ", 100, 100),
            detection("երեք", 300, 100),
        ]
        assert [d.text for d in align.column_order(detections)] == ["մեկ", "երկու", "երեք"]

    def test_a_descender_does_not_start_a_new_line(self) -> None:
        """Boxes are grouped on the bottom edge within half the median
        height, so ordinary glyph variation stays on one line."""
        detections = [
            detection("մեկ", 100, 100, height=20),
            detection("երկու", 200, 103, height=20),
        ]
        assert [d.text for d in align.column_order(detections)] == ["մեկ", "երկու"]


class TestAlignPage:
    def test_labels_come_from_the_transcript_never_the_recogniser(self) -> None:
        detections = [
            detection("ա", 100, 100),
            detection("hայ", 200, 100),
            detection("գ", 300, 100),
        ]
        crops = align.align_page(detections, "ա հայ գ")
        labels = {crop.label for crop in crops}
        assert "հայ" in labels
        assert "hայ" not in labels

    def test_the_misreading_is_kept_as_provenance(self) -> None:
        detections = [
            detection("ա", 100, 100),
            detection("hայ", 200, 100),
            detection("գ", 300, 100),
        ]
        crop = next(c for c in align.align_page(detections, "ա հայ գ") if c.label == "հայ")
        assert crop.predicted == "hայ"
        assert crop.tier is Tier.NEAR

    def test_the_bbox_travels_with_the_label(self) -> None:
        detections = [
            detection("ա", 100, 100),
            detection("hայ", 200, 100),
            detection("գ", 300, 100),
        ]
        crop = next(c for c in align.align_page(detections, "ա հայ գ") if c.label == "հայ")
        assert crop.bbox == (200, 100, 260, 120)

    def test_a_page_the_ordering_desynchronises_loses_recall_not_precision(self) -> None:
        """Whatever survives a bad alignment is still correctly labelled."""
        detections = [detection(text, 100 + i * 70, 100) for i, text in enumerate("աբգդե")]
        crops = align.align_page(detections, "ե դ գ բ ա")
        for crop in crops:
            assert crop.label == crop.predicted or crop.tier is not Tier.EXACT

    def test_an_empty_transcript_yields_nothing(self) -> None:
        assert align.align_page([detection("ա", 100, 100)], "") == []

    def test_no_detections_yields_nothing(self) -> None:
        assert align.align_page([], "ա բ գ") == []


class TestMultiWordBoxes:
    """CRAFT returns line fragments, not words -- roughly 550 boxes for a
    page's 900 tokens. Matching a whole box against one token kept only
    the short single-word boxes and discarded every line-shaped one,
    which is the shape the model is actually trained and used on."""

    def test_a_multi_word_box_is_labelled_with_the_whole_run(self) -> None:
        detections = [detection("ա", 100, 100), detection("hայ գիրք", 200, 100, width=140)]
        crops = align.align_page(detections, "ա հայ գիրք")
        assert [crop.label for crop in crops] == ["ա", "հայ գիրք"]

    def test_the_run_must_be_consecutive_in_the_transcript(self) -> None:
        """Two words that match far-apart transcript tokens are not a line."""
        detections = [detection("ա գ", 100, 100, width=140)]
        assert align.align_page(detections, "ա բ բ բ գ") == []

    def test_a_partly_placed_box_is_dropped_whole(self) -> None:
        """A label missing one of its words would teach the model to skip
        it, which is worse than not training on the crop at all."""
        detections = [
            detection("ա", 100, 100),
            detection("հայ քըստ", 200, 100, width=140),
            detection("գ", 400, 100),
        ]
        crops = align.align_page(detections, "ա հայ բոլորովին ուրիշ գ")
        assert all(" " not in crop.label for crop in crops)

    def test_the_weakest_token_sets_the_tier(self) -> None:
        """A crop is only as trustworthy as its least certain pairing."""
        detections = [
            detection("ա", 100, 100),
            detection("հայ hիրք", 200, 100, width=140),
            detection("գ", 400, 100),
        ]
        crop = next(c for c in align.align_page(detections, "ա հայ գիրք գ") if " " in c.label)
        assert crop.label == "հայ գիրք"
        assert crop.tier is Tier.NEAR

    def test_the_box_keeps_one_bbox_for_the_whole_run(self) -> None:
        detections = [detection("հայ գիրք", 200, 100, width=140)]
        crops = align.align_page(detections, "հայ գիրք")
        assert len(crops) == 1
        assert crops[0].bbox == (200, 100, 340, 120)

    def test_a_box_the_recogniser_read_as_nothing_is_skipped(self) -> None:
        detections = [detection("", 100, 100), detection("ա", 200, 100)]
        crops = align.align_page(detections, "ա")
        assert [crop.label for crop in crops] == ["ա"]


class TestTierCounts:
    def test_every_tier_is_reported_even_at_zero(self) -> None:
        counts = align.tier_counts([])
        assert set(counts) == set(Tier)
        assert all(count == 0 for count in counts.values())

    def test_counts_add_up(self) -> None:
        detections = [
            detection("ա", 100, 100),
            detection("hայ", 200, 100),
            detection("գ", 300, 100),
        ]
        counts = align.tier_counts(align.align_page(detections, "ա հայ գ"))
        assert counts[Tier.EXACT] == 2
        assert counts[Tier.NEAR] == 1
