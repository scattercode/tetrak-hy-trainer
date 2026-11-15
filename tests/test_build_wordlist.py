"""The word list's orthography filter (brief 014, Stage 3).

A single list built from every harvest lets the decoder "correct" a
classical spelling to its reformed twin, so brief 014 ships a classical list
beside the modern one. ``--orthography`` judges each harvest whole, the way
``train_synthetic.py`` does, from the pages the held-out guard lets through.
Checked against fabricated harvests, so no real harvest is needed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tetrak_hy_trainer import orthography

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_wordlist.py"

# Enough letters to clear orthography.MIN_LETTERS, with the marker rates of
# each orthography: free-standing ւ about 17 per thousand in classical text,
# none in reformed.
CLASSICAL_LINE = "Յովհաննէս Թումանեանի հեքիաթներու գիրքը նոր տպագրութիւն մը ունեցաւ Պոլսոյ մէջ"
REFORMED_LINE = "Հովհաննես Թումանյանի հեքիաթների գիրքը նոր տպագրություն ունեցավ Երևանում"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("build_wordlist", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_harvest(root: Path, name: str, line: str, pages: int = 3) -> Path:
    directory = root / name
    (directory / "text").mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"index": f"Ինդեքս:{name}.djvu", "min_quality": 3, "pages": []}),
        encoding="utf-8",
    )
    for page in range(1, pages + 1):
        (directory / "text" / f"{page}.txt").write_text("\n".join([line] * 20), encoding="utf-8")
    return directory


@pytest.fixture
def harvests(tmp_path: Path) -> list[Path]:
    return [
        write_harvest(tmp_path, "classical-book", CLASSICAL_LINE),
        write_harvest(tmp_path, "reformed-book", REFORMED_LINE),
    ]


def test_the_fixtures_read_as_their_orthography() -> None:
    assert orthography.classify("\n".join([CLASSICAL_LINE] * 20)) == orthography.CLASSICAL
    assert orthography.classify("\n".join([REFORMED_LINE] * 20)) == orthography.REFORMED


def test_all_is_the_old_recipe(script, harvests) -> None:
    counts, read, _, verdicts = script.count_words(harvests, {}, script.ALL)
    assert read == 6
    assert verdicts == {"classical-book": "classical", "reformed-book": "reformed"}
    assert "տպագրութիւն" in counts and "տպագրություն" in counts


def test_classical_keeps_only_classical_harvests(script, harvests) -> None:
    counts, read, _, verdicts = script.count_words(harvests, {}, orthography.CLASSICAL)
    assert read == 3
    assert verdicts["reformed-book"] == "reformed", "the skipped harvest is still judged"
    assert "տպագրութիւն" in counts, "free-standing ւ"
    assert "պոլսոյ" in counts, "final յ"
    assert "տպագրություն" not in counts
    assert "երևանում" not in counts


def test_reformed_keeps_only_reformed_harvests(script, harvests) -> None:
    counts, read, _, _ = script.count_words(harvests, {}, orthography.REFORMED)
    assert read == 3
    assert "տպագրություն" in counts
    assert "տպագրութիւն" not in counts


def test_a_mixed_or_short_harvest_is_in_neither_list(script, tmp_path: Path) -> None:
    short = write_harvest(tmp_path, "too-short", CLASSICAL_LINE, pages=1)
    (short / "text" / "1.txt").write_text(CLASSICAL_LINE, encoding="utf-8")
    for wanted in (orthography.CLASSICAL, orthography.REFORMED):
        counts, read, _, verdicts = script.count_words([short], {}, wanted)
        assert verdicts == {"too-short": orthography.UNKNOWN}
        assert read == 0 and not counts
    counts, read, _, _ = script.count_words([short], {}, script.ALL)
    assert read == 1 and counts, "the default still takes everything"
