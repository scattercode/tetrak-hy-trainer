"""Which scanned material is held out, and a guard that enforces it.

Every number this project quotes — v0's 0.0745/0.2742, v1's 0.1004/0.5014,
the baselines it is measured against — comes from ten pages of **volume 2**
of the Armenian Soviet Encyclopedia. v0 and v1 were both trained on volumes
1 and 3--6, and volume 2 was never harvested for training. That split is
what makes the evaluation mean anything.

Real-crop harvesting (brief 011 Stage 3 step 2) is the first thing in this
repository that turns *scans* into training data, which is also the first
time that split can be broken by accident: point the harvester at
``runs/eval/ase-vol2/`` — a directory that, unlike the training harvests,
does have page images sitting in it — and the evaluation quietly becomes
a training set. Nothing would fail; the numbers would just start
improving for the wrong reason, and every published figure would be
wrong.

Hence a guard with one job, in a module of its own so it is obvious and
cannot be diluted into some larger helper.

**The whole of volume 2 is held out, not merely pages 105--114.** The
narrow reading is defensible — a crop from page 300 shares no text with
page 105 — but the wider one is what the project has actually done since
v0, and the pages either side of the evaluation set share its typesetting,
its scanning session and its paper. Keeping the volume whole costs a
little data and keeps the evaluation honest against material the model
has never met.
"""

from __future__ import annotations

import re

#: The volume reserved for evaluation.
HELD_OUT_VOLUME = 2

#: The evaluation pages themselves, for error messages that say *why*.
EVALUATION_PAGES = range(105, 115)

# The Wikisource index titles carry the volume as a trailing "<n>.djvu", as in
# "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 2.djvu".
# The number is captured and compared as a number: the encyclopedia runs to
# thirteen volumes, so a plain endswith("2.djvu") would also condemn volume
# 12 and silently throw away a volume's worth of training data.
_VOLUME = re.compile(r"(\d+)\.djvu$")


class HeldOutDataError(RuntimeError):
    """Raised when held-out material was about to be used for training."""


def volume_of(index_title: str) -> int | None:
    """The volume number in *index_title*, or ``None`` if it has none."""
    match = _VOLUME.search(index_title.strip())
    return int(match.group(1)) if match else None


def is_held_out(index_title: str) -> bool:
    """Whether *index_title* names material reserved for evaluation.

    Args:
        index_title: A Wikisource ``Ինդեքս:`` title, as recorded in a
            harvest manifest's ``index`` field.

    Returns:
        True for volume 2 in its entirety. See the module docstring for
        why the whole volume rather than the ten evaluation pages.
    """
    return volume_of(index_title) == HELD_OUT_VOLUME


def assert_not_held_out(index_title: str, source: str = "") -> None:
    """Refuse to go on if *index_title* is held out.

    Args:
        index_title: The manifest's ``index`` field.
        source: Where it came from, for the message -- usually the
            manifest path, so the operator can see which directory they
            pointed at.

    Raises:
        HeldOutDataError: Always, when *index_title* is held out.
    """
    if not is_held_out(index_title):
        return
    where = f" ({source})" if source else ""
    raise HeldOutDataError(
        f"{index_title!r}{where} is volume {HELD_OUT_VOLUME}, held out for evaluation: "
        f"pages {EVALUATION_PAGES.start}-{EVALUATION_PAGES.stop - 1} are the set "
        f"every published figure for this model is measured on, and the volume is "
        f"kept whole so the evaluation stays honest. Harvest crops from volumes "
        f"1 and 3-6 instead. If this really is deliberate, change the split in "
        f"tetrak_hy_trainer.heldout and re-baseline everything -- do not work "
        f"around this check."
    )
