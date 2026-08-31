"""Which characters the recogniser confuses, counted from paired words.

v1's error analysis (tetrak, ``product/research/
armenian-v1-error-analysis.md``) reduced the model's remaining gap to a
table of character confusions, and split it in two:

- **Homoglyphs** — ``հ``→``h``, ``։``→``:`` — the charset permitting
  visually identical twins from different scripts with no language model
  to prefer the Armenian one. Fixed at serialisation time by
  ``tetrak_hy.fold_script``; no retraining involved.
- **Shape confusions** — ``հ``→``խ``, ``խ``→``ի``, ``տ``→``ո``,
  ``ճ``↔``ջ`` — genuine misrecognition of degraded letterpress, clustered
  on ``հ``. These are what the real-crop fine-tune exists to fix.

That table is therefore the fine-tune's scorecard: recomputed on the
evaluation pages before and after, it says whether the shape cluster
actually moved, which a single word-recall number cannot. This module
computes it.

Counting convention: entries read **truth → predicted**, the direction
the analysis used, so a row says "the page had ``հ`` and the model read
``h``". Runs the model got wrong in unequal-length ways are kept as
multi-character entries (``աբ`` → ``x``) rather than being forced into
character pairs, and dropped or invented characters are counted against
``∅`` — an honest table beats a tidy one, and single-character
substitutions still dominate it.

Standard library only; testable with no OCR stack installed.
"""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Iterable

#: Stands in for "nothing" on either side of a confusion -- a character
#: the model dropped, or one it invented.
EMPTY = "∅"


def character_confusions(pairs: Iterable[tuple[str, str]]) -> Counter[tuple[str, str]]:
    """Count character confusions across ``(predicted, truth)`` word pairs.

    Args:
        pairs: Word pairs, each ``(what the model read, what the page
            says)``. Pairs that are equal contribute nothing, so it is
            harmless to pass every aligned word rather than only the
            misread ones.

    Returns:
        A counter keyed ``(truth, predicted)``. Keys are single characters
        for the ordinary case, longer runs where the model's error did not
        line up character-for-character, and :data:`EMPTY` on the side
        where there was nothing.
    """
    confusions: Counter[tuple[str, str]] = Counter()

    for predicted, truth in pairs:
        if predicted == truth:
            continue
        matcher = difflib.SequenceMatcher(None, truth, predicted, autojunk=False)
        for tag, truth_start, truth_end, pred_start, pred_end in matcher.get_opcodes():
            truth_run = truth[truth_start:truth_end]
            pred_run = predicted[pred_start:pred_end]
            if tag == "equal":
                continue
            if tag == "replace" and len(truth_run) == len(pred_run):
                confusions.update(zip(truth_run, pred_run, strict=True))
            elif tag == "replace":
                confusions[truth_run, pred_run] += 1
            elif tag == "delete":
                confusions[truth_run, EMPTY] += 1
            elif tag == "insert":
                confusions[EMPTY, pred_run] += 1

    return confusions


def format_table(confusions: Counter[tuple[str, str]], limit: int = 25) -> str:
    """Render the *limit* commonest confusions as a markdown table.

    The same shape as the table in the error analysis, so a fine-tune's
    before and after can be read side by side.
    """
    lines = ["| confusion | count |", "|---|---|"]
    lines.extend(
        f"| {truth} → {predicted} | {count} |"
        for (truth, predicted), count in confusions.most_common(limit)
    )
    return "\n".join(lines)
