"""The yaml is the loading contract with EasyOCR: these tests pin the keys
easyocr 1.7.2's custom-model path actually reads, and that the file always
carries the charset module's list rather than a drifting copy."""

from pathlib import Path

import yaml

from tetrak_hy_trainer import charset, packaging


def test_config_carries_the_keys_easyocr_reads() -> None:
    config = packaging.build_config()
    assert set(config) == {"network_params", "imgH", "lang_list", "character_list"}


def test_character_list_comes_from_the_charset_module() -> None:
    """One source of truth: the yaml's charset is the module's, verbatim."""
    assert packaging.build_config()["character_list"] == charset.character_list()


def test_network_name_is_a_valid_module_name() -> None:
    """EasyOCR resolves the name with importlib.import_module, so it must
    be importable — underscores, no hyphens."""
    assert packaging.NETWORK_NAME.isidentifier()


def test_lang_list_includes_armenian() -> None:
    assert "hy" in packaging.build_config()["lang_list"]


def test_write_yaml_round_trips(tmp_path: Path) -> None:
    destination = packaging.write_yaml(tmp_path)
    assert destination == tmp_path / "tetrak_hy.yaml"

    with destination.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert loaded == packaging.build_config()
    # The Armenian characters must survive serialisation as themselves,
    # not as escaped code points.
    assert "Ա" in destination.read_text(encoding="utf-8")
