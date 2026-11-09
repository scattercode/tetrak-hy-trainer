"""The mixed-script share of a synthetic line set (brief 014 Stage 1)."""

import json
import random
from pathlib import Path

from tetrak_hy_trainer import sampling

MIXED = [f"Գրկ․ Исраэлян {n}" for n in range(50)]
PLAIN = [f"բառ {n}" for n in range(950)]


def test_a_mixed_line_has_an_armenian_and_a_cyrillic_token() -> None:
    assert sampling.is_mixed_line("Գրկ․ Исраэлян")
    assert not sampling.is_mixed_line("Москва 1965")
    assert not sampling.is_mixed_line("Գրկ․ Lazarev")


def test_the_share_is_honoured() -> None:
    samples, mixed = sampling.with_mixed_share(MIXED + PLAIN, 100, random.Random(0), 0.2)
    assert len(samples) == 100
    assert mixed == 20 == sum(1 for line in samples if sampling.is_mixed_line(line))
    assert len(set(samples)) == 100


def test_the_supply_caps_the_share_and_the_count_says_so() -> None:
    samples, mixed = sampling.with_mixed_share(MIXED + PLAIN, 100, random.Random(0), 0.9)
    assert len(samples) == 100
    assert mixed == 50 < round(100 * 0.9)


def test_a_zero_share_makes_the_draws_a_seeded_corpus_always_made() -> None:
    """Earlier synthetic sets must render unchanged: shuffle, then slice."""
    before = MIXED + PLAIN
    expected = list(before)
    random.Random(7).shuffle(expected)
    samples, mixed = sampling.with_mixed_share(list(before), 100, random.Random(7), 0.0)
    assert samples == expected[:100]
    assert mixed == sum(1 for line in samples if sampling.is_mixed_line(line))


def test_the_recipe_is_written_beside_the_labels(tmp_path: Path) -> None:
    path = sampling.write_recipe(tmp_path, mixed_share=0.05, mixed_lines=3000, lines=60000)
    assert path == tmp_path / "recipe.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "mixed_share": 0.05,
        "mixed_lines": 3000,
        "lines": 60000,
    }
