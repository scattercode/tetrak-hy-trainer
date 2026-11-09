"""Composing a synthetic line set from the sampled candidates.

Brief 014 Stage 1: Cyrillic in Soviet-era Armenian print is scattered
words and short runs -- a Russian title in a bibliography, a name in
brackets, a journal in a footnote -- so the recogniser has to learn the
switch of alphabet *within* a line, not only Russian lines. The harvests
supply such lines at about 0.27% of candidates (5,747 of 2.1 million over
the encyclopedia and Tumanyan volumes, measured 2026-11-09): too few to be
met often in a natural sample, so :func:`with_mixed_share` sets the share.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tetrak_hy_trainer import charset

RECIPE_FILE = "recipe.json"


def is_mixed_line(line: str) -> bool:
    """Whether *line* has both an Armenian token and a Cyrillic token."""
    found = frozenset().union(*(charset.scripts_of(token) for token in line.split()))
    return {"armenian", "cyrillic"} <= found


def with_mixed_share(
    candidates: list[str], max_samples: int, rng: random.Random, mixed_share: float
) -> tuple[list[str], int]:
    """Sample *max_samples* lines from *candidates*, *mixed_share* of them mixed.

    With a share of zero this is a shuffle and a slice -- the draws a
    seeded corpus has always made, so every earlier set renders unchanged.
    Otherwise the mixed lines are drawn first, as many as the share asks
    for or the supply allows, the rest from the other lines, and the whole
    is shuffled. The caller compares the count returned with what it asked
    for: a shortfall is said, not hidden.

    Args:
        candidates: Every candidate line; shuffled in place.
        max_samples: How many lines to return.
        rng: The seeded source of every draw.
        mixed_share: The fraction of lines that should mix Armenian and
            Cyrillic tokens (:func:`is_mixed_line`).

    Returns:
        The sampled lines and how many of them are mixed.
    """
    if mixed_share <= 0:
        rng.shuffle(candidates)
        samples = candidates[:max_samples]
        return samples, sum(1 for line in samples if is_mixed_line(line))
    mixed = [line for line in candidates if is_mixed_line(line)]
    plain = [line for line in candidates if not is_mixed_line(line)]
    rng.shuffle(mixed)
    rng.shuffle(plain)
    taken = mixed[: round(max_samples * mixed_share)]
    samples = taken + plain[: max_samples - len(taken)]
    rng.shuffle(samples)
    return samples, len(taken)


def write_recipe(folder: Path, **fields: object) -> Path:
    """Record how a rendered set was composed, beside its labels.

    The share of mixed lines is a recipe lever like the all-caps share, and
    a lever nobody wrote down is one nobody can reproduce; the ``VERSIONS``
    entry (training skill, step 10) copies from here.
    """
    path = Path(folder) / RECIPE_FILE
    path.write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
