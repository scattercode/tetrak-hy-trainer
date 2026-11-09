"""Composing a synthetic line set from the sampled candidates.

Brief 014 Stage 1 adds two levers to the natural sample of lines:

- **Mixed-script lines.** Cyrillic in Soviet-era Armenian print is
  scattered words and short runs -- a Russian title in a bibliography, a
  name in brackets, a journal in a footnote -- so the recogniser has to
  learn the switch of alphabet *within* a line, not only Russian lines.
  The harvests supply such lines at about 0.27% of candidates (5,747 of
  2.1 million over the encyclopedia and Tumanyan volumes, measured
  2026-11-09): too few to be met often in a natural sample.
- **Classical-orthography lines.** Free-standing ``ւ`` and final ``յ`` are
  rare in reformed text and on every classical page, and the Institute's
  shelves hold far more classical print than Wikisource's proportions
  suggest. A line is classical by its *source*: the harvest it came from
  reads as classical to :mod:`tetrak_hy_trainer.orthography`, because a
  line of one to four tokens is too short to be read on its own.

:func:`compose` sets both shares. With both at zero it is a shuffle and a
slice, the draws every earlier seeded set made.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from tetrak_hy_trainer import charset

RECIPE_FILE = "recipe.json"


@dataclass(frozen=True)
class Candidate:
    """One candidate line and what its source says about it."""

    text: str
    classical: bool = False


def is_mixed_line(line: str) -> bool:
    """Whether *line* has both an Armenian token and a Cyrillic token."""
    found = frozenset().union(*(charset.scripts_of(token) for token in line.split()))
    return {"armenian", "cyrillic"} <= found


def compose(
    candidates: list[Candidate],
    max_samples: int,
    rng: random.Random,
    mixed_share: float = 0.0,
    classical_share: float = 0.0,
) -> tuple[list[str], dict[str, int]]:
    """Sample *max_samples* lines from *candidates* at the shares asked.

    With both shares at zero this is a shuffle and a slice -- the draws a
    seeded corpus has always made, so every earlier set renders unchanged.
    Otherwise each share with a value draws its lines first, as many as it
    asks for or the supply allows: mixed-script lines, then classical lines
    from what is left, then the rest from the other lines; the whole is
    shuffled. A share that is set is exact, not a floor: lines of that kind
    are not also drawn as "other". The caller compares the counts returned
    with what it asked for; a shortfall is said, not hidden.

    Args:
        candidates: Every candidate line; shuffled in place.
        max_samples: How many lines to return.
        rng: The seeded source of every draw.
        mixed_share: The fraction of lines that mix Armenian and Cyrillic
            tokens (:func:`is_mixed_line`).
        classical_share: The fraction of lines from classical-orthography
            sources (:attr:`Candidate.classical`).

    Returns:
        The sampled lines and the counts of mixed and classical lines
        among them, as ``{"mixed": n, "classical": n}``.
    """
    if mixed_share <= 0 and classical_share <= 0:
        rng.shuffle(candidates)
        chosen = candidates[:max_samples]
        return [c.text for c in chosen], _counts(chosen)

    mixed: list[Candidate] = []
    classical: list[Candidate] = []
    other: list[Candidate] = []
    for candidate in candidates:
        if mixed_share > 0 and is_mixed_line(candidate.text):
            mixed.append(candidate)
        elif classical_share > 0 and candidate.classical:
            classical.append(candidate)
        else:
            other.append(candidate)
    for pool in (mixed, classical, other):
        rng.shuffle(pool)
    chosen = mixed[: round(max_samples * mixed_share)]
    chosen += classical[: round(max_samples * classical_share)]
    chosen += other[: max_samples - len(chosen)]
    rng.shuffle(chosen)
    return [c.text for c in chosen], _counts(chosen)


def _counts(chosen: list[Candidate]) -> dict[str, int]:
    return {
        "mixed": sum(1 for c in chosen if is_mixed_line(c.text)),
        "classical": sum(1 for c in chosen if c.classical),
    }


def write_recipe(folder: Path, **fields: object) -> Path:
    """Record how a rendered set was composed, beside its labels.

    The shares are recipe levers like the all-caps share, and a lever
    nobody wrote down is one nobody can reproduce; the ``VERSIONS`` entry
    (training skill, step 10) copies from here.
    """
    path = Path(folder) / RECIPE_FILE
    path.write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
