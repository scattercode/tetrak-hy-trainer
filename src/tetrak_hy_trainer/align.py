"""Align detector boxes against a proofread transcript, conservatively.

Stage 3 step 2 of brief 011 fine-tunes on *real* crops cut from the
archival scans, because that is the only lever left for the shape
confusions v1's error analysis isolated (``հ``→``խ``, ``խ``→``ի``,
``տ``→``ո``, ``ճ``↔``ջ`` — clustered on ``հ``, one of the commonest
Armenian letters). Synthetic fonts render ``հ``'s ascender cleanly; 1970s
letterpress on yellowed paper does not, and no amount of rendering
teaches the difference.

Getting those crops means answering, for each box CRAFT found on a page,
"which word of the human transcript is this?" — the part brief 011 calls
"the genuinely fiddly part". This module is that answer, and it is
deliberately built to **fail closed**: a box it cannot place with
confidence is dropped, not guessed at. A few thousand correctly labelled
crops is the stated goal; a mislabelled crop is worse than no crop,
because it actively teaches the model the wrong shape.

How it works
------------
1. :func:`column_order` puts the boxes into transcript order. The ASE is
   set in two columns, and grouping boxes by baseline alone — what
   ``tetrak_ocr.layout`` does — reads straight across the gutter,
   interleaving the columns and desynchronising everything downstream.
   Boxes that *straddle* the gutter are dropped rather than placed:
   they are running headers, page numbers and rules, which the
   transcripts (correctly) omit, so they have no truth token to pair
   with anyway.
2. :func:`pair_tokens` aligns the ordered predictions against the
   transcript's tokens with :mod:`difflib`, and keeps a pair only where
   the alignment is trustworthy — see :class:`Tier`.

The label always comes from the **transcript**, never from the
recogniser: the recogniser's reading is kept only as provenance, for the
confusion table (:mod:`tetrak_hy_trainer.confusion`) and for anyone
auditing the crop set by hand.

Reading order is imperfect and that is fine. A page whose ordering goes
wrong loses *recall* — fewer crops — not *precision*, because a
desynchronised sequence stops matching and the pairs simply stop being
emitted. Every knob here is therefore tuned for precision.

Standard library only, so this is testable in CI with no OCR stack
installed. The heavy half — running the detector, cutting the images —
lives in ``scripts/harvest_real_crops.py``.

**The column logic here has a twin.** ``tetrak_ocr.layout`` (in the private
Tetrak repository) solves the same problem for transcript output, with its own
copy of the gutter detection, the baseline grouping and the three straddle
constants. The repositories cannot share code — this one is public and that
one is not — so the duplication is deliberate and the drift is not: a fix to
either is worth porting to the other. They differ in one respect only, and on
purpose: a box straddling the gutter is *dropped* here (the transcripts omit
running heads, so it has no truth token to pair with) and *placed* there
(dropping text is not an option when the output is the text).
"""

from __future__ import annotations

import difflib
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

# Fraction of a page's boxes allowed to straddle a candidate gutter before
# it stops looking like a gutter. On a genuinely two-column page almost
# nothing crosses it; on a single-column page most lines cross any x in
# the middle of the page, so this separates the two cases cleanly.
DEFAULT_MAX_STRADDLE_SHARE = 0.05

# ...but a handful of spanning elements -- a centred running header, a
# rule, a wide figure caption -- is normal on any page whatever its box
# count, so the share is floored at an absolute allowance. Without it a
# sparse page is judged by a threshold of less than one box, and a single
# header is enough to hide a perfectly good gutter.
DEFAULT_STRADDLE_ALLOWANCE = 3

# A column has to hold at least this share of the page's boxes for a split
# to be believed -- otherwise a page with one column and a stray marginal
# note could be "split" into a column and a rump.
DEFAULT_MIN_COLUMN_SHARE = 0.15

# difflib ratio below which a positionally-paired token is not trusted on
# content alone. "hայ" against "հայ" scores 0.67; unrelated words paired
# by a desynchronised alignment usually score well under 0.5.
DEFAULT_MIN_SIMILARITY = 0.6

# Longest run of consecutive mismatched tokens still eligible for
# :attr:`Tier.BRACKETED`. Kept short: the argument for trusting position
# over content weakens as the run grows.
DEFAULT_BRACKET_SPAN = 2


class Tier(StrEnum):
    """How a pairing was arrived at, strongest evidence first.

    Recorded per crop so the trade-off is auditable rather than buried in
    a threshold, and so a fine-tune can be run on a subset if one tier
    turns out to be noisy.
    """

    #: The recogniser read the token exactly as the transcript has it.
    #: Content and position agree; nothing to doubt. These crops teach the
    #: model little it does not already know, but they are what stops a
    #: fine-tune forgetting what it got right.
    EXACT = "exact"

    #: A misreading, paired positionally and corroborated by content --
    #: the two strings still look alike (see *min_similarity*). This is
    #: the tier the fine-tune actually wants: a real crop of a word the
    #: model currently gets wrong, carrying the correct label.
    NEAR = "near"

    #: A misreading paired on position alone, where content similarity is
    #: too low to corroborate -- but the run is short and *flanked on both
    #: sides by exact matches*, so both sequences agree on how many tokens
    #: sit in the gap. Badly-misread words land here (a word read as
    #: gibberish has no similarity left to measure), which makes this the
    #: most informative tier and the least self-evidently safe. Separated
    #: out so it can be excluded.
    BRACKETED = "bracketed"


@dataclass(frozen=True)
class Detection:
    """One box the detector found, and what the recogniser read in it.

    Attributes:
        text: The recogniser's reading. Never used as a label -- only to
            find the matching transcript token, and as provenance.
        bbox: ``(left, top, right, bottom)`` in pixels, top-left origin
            (the Pillow convention, which EasyOCR also speaks).
        confidence: The recogniser's own 0--1 score, where it reports one.
    """

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None = None

    @property
    def baseline(self) -> float:
        """The bottom edge, used to group boxes into lines.

        The bottom edge approximates the baseline, and words on one line
        share a baseline far more reliably than a top edge -- capitals,
        ascenders and descenders all move the top and leave the baseline
        alone. Same reasoning as ``tetrak_ocr.layout.group_lines``.
        """
        return self.bbox[3]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class AlignedCrop:
    """A box with the transcript token it was matched to.

    Attributes:
        bbox: Where to cut the page image.
        label: The text **from the transcript** -- what the crop is
            labelled with, and the only thing that reaches training. One
            or more tokens joined by single spaces, since a detector box
            usually holds a line fragment rather than a word.
        predicted: What the recogniser read there. Provenance and the
            input to the confusion table; never a label.
        tier: How the pairing was arrived at, taken from the box's
            weakest token. See :class:`Tier`.
        confidence: The recogniser's score for *predicted*, if any.
    """

    bbox: tuple[float, float, float, float]
    label: str
    predicted: str
    tier: Tier
    confidence: float | None = None


def similarity(left: str, right: str) -> float:
    """Character-level similarity of two tokens, 0.0--1.0."""
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def _line_tolerance(detections: Sequence[Detection]) -> float:
    """How far apart two baselines can sit and still be one line.

    Half the median box height, so the tolerance tracks the type size
    rather than assuming a page resolution.
    """
    heights = [d.height for d in detections if d.height > 0]
    return statistics.median(heights) * 0.5 if heights else 0.0


def _in_reading_order(detections: Sequence[Detection]) -> list[Detection]:
    """One column's boxes: lines top to bottom, each left to right."""
    if not detections:
        return []
    tolerance = _line_tolerance(detections)
    ordered = sorted(detections, key=lambda d: (d.baseline, d.bbox[0]))

    lines: list[list[Detection]] = [[ordered[0]]]
    for detection in ordered[1:]:
        if abs(detection.baseline - lines[-1][0].baseline) <= tolerance:
            lines[-1].append(detection)
        else:
            lines.append([detection])

    return [d for line in lines for d in sorted(line, key=lambda d: d.bbox[0])]


def find_gutter(
    detections: Sequence[Detection],
    max_straddle_share: float = DEFAULT_MAX_STRADDLE_SHARE,
    min_column_share: float = DEFAULT_MIN_COLUMN_SHARE,
    straddle_allowance: int = DEFAULT_STRADDLE_ALLOWANCE,
) -> float | None:
    """The x of the column gutter, or ``None`` if the page is one column.

    Scans candidate positions across the middle of the page and counts
    boxes crossing each one. A two-column page has a band almost nothing
    crosses; a single-column page has no such band, because most lines
    cross any x near the middle.

    Args:
        detections: The page's boxes. Fewer than four and no split is
            attempted -- there is nothing to be confident about.
        max_straddle_share: Largest share of boxes that may cross the
            gutter for it to count as one.
        min_column_share: Smallest share of boxes each side must hold.
        straddle_allowance: Absolute floor under *max_straddle_share*, so
            a few spanning elements never hide a real gutter on a sparse
            page.

    Returns:
        The gutter's x in pixels, or ``None``.
    """
    if len(detections) < 4:
        return None

    page_left = min(d.bbox[0] for d in detections)
    page_right = max(d.bbox[2] for d in detections)
    width = page_right - page_left
    if width <= 0:
        return None

    best: tuple[int, float] | None = None
    steps = 80
    low, high = page_left + 0.3 * width, page_left + 0.7 * width
    for step in range(steps + 1):
        candidate = low + (high - low) * step / steps
        straddling = sum(1 for d in detections if d.bbox[0] < candidate < d.bbox[2])
        if best is None or straddling < best[0]:
            best = (straddling, candidate)

    allowed = max(straddle_allowance, max_straddle_share * len(detections))
    if best is None or best[0] > allowed:
        return None

    gutter = best[1]
    left = sum(1 for d in detections if d.bbox[2] <= gutter)
    right = sum(1 for d in detections if d.bbox[0] >= gutter)
    minimum = min_column_share * len(detections)
    if left < minimum or right < minimum:
        return None
    return gutter


def column_order(detections: Sequence[Detection]) -> list[Detection]:
    """Order boxes as a reader would take them, column by column.

    Boxes straddling the gutter are **dropped**, not placed. On these
    pages they are running headers, page numbers and rules, which the
    transcripts omit -- so they have no token to pair with, and leaving
    them in only desynchronises the alignment.
    """
    gutter = find_gutter(detections)
    if gutter is None:
        return _in_reading_order(detections)

    left = [d for d in detections if d.bbox[2] <= gutter]
    right = [d for d in detections if d.bbox[0] >= gutter]
    return _in_reading_order(left) + _in_reading_order(right)


def pair_tokens(
    predicted: Sequence[str],
    truth: Sequence[str],
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    bracket_span: int = DEFAULT_BRACKET_SPAN,
) -> list[tuple[int, int, Tier]]:
    """Pair predicted tokens with transcript tokens, keeping only the safe ones.

    Runs :class:`difflib.SequenceMatcher` over the two token sequences and
    reads its opcodes:

    - ``equal`` -- the recogniser read these exactly right.
      :attr:`Tier.EXACT`, one pair per token.
    - ``replace`` of equal length -- both sequences agree on how many
      tokens sit here, so position pairs them. Kept as :attr:`Tier.NEAR`
      when the strings still resemble each other, or
      :attr:`Tier.BRACKETED` when they do not but the run is short and
      flanked by exact matches on both sides.
    - ``replace`` of unequal length, ``insert``, ``delete`` -- the
      detector split or merged a box, or the page and transcript genuinely
      disagree. Dropped: there is no way to say which token is which.

    ``autojunk`` is off. Its heuristic treats any element appearing in
    more than 1% of a sequence of 200+ as junk, which on a page of
    Armenian prose would discard exactly the common words that anchor the
    alignment.

    Args:
        predicted: The recogniser's readings, in transcript order.
        truth: The transcript's tokens.
        min_similarity: Content floor for :attr:`Tier.NEAR`.
        bracket_span: Longest mismatched run still eligible for
            :attr:`Tier.BRACKETED`.

    Returns:
        ``(predicted index, truth index, tier)``, in predicted order.
    """
    matcher = difflib.SequenceMatcher(None, predicted, truth, autojunk=False)
    opcodes = matcher.get_opcodes()
    pairs: list[tuple[int, int, Tier]] = []

    for position, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            pairs.extend((i1 + offset, j1 + offset, Tier.EXACT) for offset in range(i2 - i1))
            continue
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue

        flanked = (
            position > 0
            and opcodes[position - 1][0] == "equal"
            and position + 1 < len(opcodes)
            and opcodes[position + 1][0] == "equal"
        )
        short_enough = (i2 - i1) <= bracket_span

        for offset in range(i2 - i1):
            predicted_index, truth_index = i1 + offset, j1 + offset
            if similarity(predicted[predicted_index], truth[truth_index]) >= min_similarity:
                pairs.append((predicted_index, truth_index, Tier.NEAR))
            elif flanked and short_enough:
                pairs.append((predicted_index, truth_index, Tier.BRACKETED))

    return pairs


#: Riskiest tier wins when a box's tokens were matched at different
#: strengths -- a crop is only as trustworthy as its weakest pairing.
_TIER_RISK = {Tier.EXACT: 0, Tier.NEAR: 1, Tier.BRACKETED: 2}


def align_page(
    detections: Sequence[Detection],
    truth_text: str,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    bracket_span: int = DEFAULT_BRACKET_SPAN,
) -> list[AlignedCrop]:
    """Match one page's boxes to its transcript, a box at a time.

    Matching happens at the **token** level and the result is grouped back
    onto boxes, because CRAFT does not return one box per word: it returns
    line fragments, several words at a time. On a typical encyclopedia
    page it finds roughly 550 boxes for 900 transcript tokens. Comparing a
    whole box against a single token can therefore never match the
    multi-word ones, and an earlier cut of this function that did exactly
    that kept only the short single-word boxes -- 7 of 304 crops had so
    much as a space in them. That is the worst possible bias: it throws
    away every line-shaped crop, which is precisely the shape v1 was
    trained on (1--4 consecutive tokens, spaces included) and the shape
    the recogniser meets at inference.

    So each box's reading is split into tokens, all the page's tokens are
    aligned against the transcript in one sequence, and a box is emitted
    only if **every** one of its tokens was paired *and* those pairs land
    on a consecutive, in-order run of transcript tokens. Its label is
    that run, joined with single spaces. A box that is only partly
    placed is dropped: a label missing a word teaches the model to skip
    it.

    Args:
        detections: Every box the detector found, in any order.
        truth_text: The page's proofread transcript.
        min_similarity: See :func:`pair_tokens`.
        bracket_span: See :func:`pair_tokens`.

    Returns:
        One :class:`AlignedCrop` per confidently placed box, in reading
        order. Boxes that could not be placed are absent -- silently,
        because on a good page that is still a third of them and saying so
        per box would drown the log.
    """
    ordered = column_order(detections)

    tokens: list[str] = []
    owner: list[int] = []
    for index, detection in enumerate(ordered):
        for token in detection.text.split():
            tokens.append(token)
            owner.append(index)

    truth_tokens = truth_text.split()
    paired = {
        predicted_index: (truth_index, tier)
        for predicted_index, truth_index, tier in pair_tokens(
            tokens, truth_tokens, min_similarity=min_similarity, bracket_span=bracket_span
        )
    }

    per_box: list[list[int]] = [[] for _ in ordered]
    for token_index, box_index in enumerate(owner):
        per_box[box_index].append(token_index)

    crops: list[AlignedCrop] = []
    for box_index, token_indices in enumerate(per_box):
        if not token_indices or any(index not in paired for index in token_indices):
            continue
        matches = [paired[index] for index in token_indices]
        truth_indices = [truth_index for truth_index, _ in matches]
        first = truth_indices[0]
        if truth_indices != list(range(first, first + len(truth_indices))):
            continue  # the box's words are scattered through the transcript

        detection = ordered[box_index]
        crops.append(
            AlignedCrop(
                bbox=detection.bbox,
                label=" ".join(truth_tokens[first : first + len(truth_indices)]),
                predicted=detection.text,
                tier=max((tier for _, tier in matches), key=_TIER_RISK.__getitem__),
                confidence=detection.confidence,
            )
        )
    return crops


def tier_counts(crops: Iterable[AlignedCrop]) -> dict[Tier, int]:
    """How many crops came from each tier -- the yield report."""
    counts = dict.fromkeys(Tier, 0)
    for crop in crops:
        counts[crop.tier] += 1
    return counts
