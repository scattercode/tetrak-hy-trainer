"""Composing a synthetic line set: the mixed-script and classical shares (brief 014)."""

import json
import random
from pathlib import Path

from tetrak_hy_trainer import sampling
from tetrak_hy_trainer.sampling import Candidate

MIXED = [Candidate(f"Գրկ․ Исраэлян {n}") for n in range(50)]
CLASSICAL = [Candidate(f"հաւատք {n}", classical=True) for n in range(60)]
PLAIN = [Candidate(f"բառ {n}") for n in range(890)]


def everything() -> list[Candidate]:
    return MIXED + CLASSICAL + PLAIN


def test_a_mixed_line_has_an_armenian_and_a_cyrillic_token() -> None:
    assert sampling.is_mixed_line("Գրկ․ Исраэлян")
    assert not sampling.is_mixed_line("Москва 1965")
    assert not sampling.is_mixed_line("Գրկ․ Lazarev")


def test_both_shares_are_honoured_and_exact() -> None:
    samples, counts = sampling.compose(
        everything(), 100, random.Random(0), mixed_share=0.2, classical_share=0.3
    )
    assert len(samples) == len(set(samples)) == 100
    assert counts == {"mixed": 20, "classical": 30}
    assert sum(1 for line in samples if sampling.is_mixed_line(line)) == 20
    assert sum(1 for line in samples if line.startswith("հաւատք")) == 30


def test_the_supply_caps_a_share_and_the_count_says_so() -> None:
    samples, counts = sampling.compose(everything(), 100, random.Random(0), mixed_share=0.9)
    assert len(samples) == 100
    assert counts["mixed"] == 50 < round(100 * 0.9)


def test_a_share_left_at_zero_is_not_separated() -> None:
    """With only the classical share set, mixed lines stay among the others
    at their natural rate rather than being held back."""
    candidates = MIXED + CLASSICAL
    samples, counts = sampling.compose(candidates, 100, random.Random(0), classical_share=0.5)
    assert counts == {"mixed": 50, "classical": 50}


def test_zero_shares_make_the_draws_a_seeded_corpus_always_made() -> None:
    """Earlier synthetic sets must render unchanged: shuffle, then slice."""
    expected = [c.text for c in everything()]
    random.Random(7).shuffle(expected)
    samples, counts = sampling.compose(everything(), 100, random.Random(7))
    assert samples == expected[:100]
    assert counts["classical"] == sum(1 for line in samples if line.startswith("հաւատք"))


def test_the_recipe_is_written_beside_the_labels(tmp_path: Path) -> None:
    path = sampling.write_recipe(tmp_path, classical_share=0.25, classical_lines=15000, lines=60000)
    assert path == tmp_path / "recipe.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "classical_share": 0.25,
        "classical_lines": 15000,
        "lines": 60000,
    }
