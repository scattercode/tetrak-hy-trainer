"""The training config is the other half of the CTC-order contract.

The vendored trainer composes its charset as ``number + symbol +
lang_char``; the shipped ``tetrak_hy.yaml`` carries
``charset.character_list()``. These tests pin the two to byte identity —
the property that makes a trained checkpoint decode correctly under the
shipped yaml — plus the architecture keys that make the state dict load
at all.
"""

from pathlib import Path

import yaml

from tetrak_hy_trainer import charset, packaging, train_config


def build(**overrides) -> dict:
    kwargs = dict(
        experiment_name="t",
        train_data="data",
        valid_data="data/val",
        select_data="train",
        num_iter=10,
    )
    kwargs.update(overrides)
    return train_config.build_config(**kwargs)


def test_composed_charset_is_byte_identical_to_the_shipped_one() -> None:
    config = build()
    composed = config["number"] + config["symbol"] + config["lang_char"]
    assert composed == charset.character_list()


def test_number_and_symbol_are_empty_by_design() -> None:
    """The whole charset rides in lang_char so ordering is owned by one
    module; anything in number/symbol would reorder the CTC classes."""
    config = build()
    assert config["number"] == ""
    assert config["symbol"] == ""


def test_architecture_matches_the_shipped_network_params() -> None:
    config = build()
    for key, value in packaging.DEFAULT_NETWORK_PARAMS.items():
        assert config[key] == value, key
    assert config["imgH"] == packaging.DEFAULT_IMG_H
    assert (config["Transformation"], config["Prediction"]) == ("None", "CTC")
    assert config["FeatureExtraction"] == "VGG"
    assert config["SequenceModeling"] == "BiLSTM"


def test_fine_tuning_flag_follows_saved_model() -> None:
    assert build()["FT"] is False
    assert build(saved_model="x.pth")["FT"] is True


def test_write_config_round_trips(tmp_path: Path) -> None:
    destination = train_config.write_config(
        tmp_path / "cfg.yaml",
        **dict(
            experiment_name="t",
            train_data="data",
            valid_data="data/val",
            select_data="train",
            num_iter=10,
        ),
    )
    with destination.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert loaded == build()
    # Armenian must survive as itself, not as escapes.
    assert "Ա" in destination.read_text(encoding="utf-8")
