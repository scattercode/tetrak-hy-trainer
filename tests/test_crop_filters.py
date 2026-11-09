"""What the crop harvester keeps and drops under charset v5 (brief 014).

v5 and v6 dropped every crop with a Russian word because the charset had
no class for it. Under charset v5 an all-Cyrillic token is kept and a
token mixing Cyrillic with Armenian or Latin letters drops its crop, the
same per-token rule the synthetic sampler applies. These tests exercise
the label filters alone; no detector is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tetrak_hy_trainer import charset
from tetrak_hy_trainer.align import AlignedCrop, Tier

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "harvest_real_crops.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("harvest_real_crops", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop(label: str, predicted: str | None = None) -> AlignedCrop:
    return AlignedCrop(
        bbox=(0.0, 0.0, 200.0, 40.0), label=label, predicted=predicted or label, tier=Tier.EXACT
    )


class TestMixedScriptTokens:
    def test_an_all_cyrillic_token_is_not_mixed(self, script) -> None:
        assert not script.has_mixed_script_token("Редакция Нор-Дар")

    def test_cyrillic_inside_an_armenian_token_is_mixed(self, script) -> None:
        assert script.has_mixed_script_token("Նор")

    def test_cyrillic_inside_a_latin_token_is_mixed(self, script) -> None:
        assert script.has_mixed_script_token("Москва Mосква")

    def test_armenian_with_latin_is_not_the_cyrillic_rule(self, script) -> None:
        """Stage 0.3's rule is about Cyrillic; an Armenian-Latin token is
        left to the stray-character check as before."""
        assert not script.has_mixed_script_token("Հայx")


class TestCyrillicTokens:
    def test_counted_when_any_token_is_cyrillic(self, script) -> None:
        assert script.has_cyrillic_token("տես Правда 1961")
        assert not script.has_cyrillic_token("տես 1961")


class TestUsableUnderV5:
    def test_an_all_cyrillic_label_is_usable(self, script) -> None:
        allowed = set(charset.character_list())
        assert script.usable(crop("Редакция"), allowed, 60, 40.0)

    def test_a_greek_stray_is_still_refused(self, script) -> None:
        allowed = set(charset.character_list())
        assert not script.usable(crop("π-greek"), allowed, 60, 40.0)
