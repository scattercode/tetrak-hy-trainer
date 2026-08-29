"""Generate the vendored trainer's config from the canonical charset.

The trainer composes its character set as ``number + symbol + lang_char``
(``training/VENDORED.md``), and CTC class indices are positional — so the
training charset must be **byte-identical** to the ``character_list`` the
shipped ``tetrak_hy.yaml`` carries, or the weights decode as the wrong
characters. This module guarantees that the only way to produce a training
config is from :mod:`tetrak_hy_trainer.charset`: the whole charset goes in
``lang_char`` with ``number`` and ``symbol`` empty, and the architecture
keys come from :mod:`tetrak_hy_trainer.packaging`'s defaults, so a model
trained with this config loads under the shipped yaml by construction.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tetrak_hy_trainer import charset, packaging


def build_config(
    experiment_name: str,
    train_data: str,
    valid_data: str,
    select_data: str,
    num_iter: int,
    batch_size: int = 32,
    val_interval: int = 500,
    workers: int = 2,
    batch_max_length: int = 40,
    saved_model: str = "",
) -> dict:
    """Return a trainer config dict for our architecture and charset.

    Args:
        experiment_name: Names the ``saved_models/<name>/`` output folder.
        train_data: The trainer's data root.
        valid_data: Path to the validation folder (holds a labels.csv).
        select_data: Training folder name(s) under *train_data*.
        num_iter: Training iterations.
        batch_size: Per-step batch size.
        val_interval: Iterations between validations (each saves
            ``best_accuracy.pth`` when improved).
        workers: DataLoader workers; keep low on MPS/CPU.
        batch_max_length: Longest label the encoder accepts.
        saved_model: Checkpoint to fine-tune from, or empty to start fresh.
    """
    params = packaging.DEFAULT_NETWORK_PARAMS
    return {
        # The whole charset rides in lang_char -- see the module docstring.
        "number": "",
        "symbol": "",
        "lang_char": charset.character_list(),
        "experiment_name": experiment_name,
        "train_data": train_data,
        "valid_data": valid_data,
        "manualSeed": 1111,
        "workers": workers,
        "batch_size": batch_size,
        "num_iter": num_iter,
        "valInterval": val_interval,
        "saved_model": saved_model,
        "FT": bool(saved_model),
        "optim": False,  # the trainer's default Adadelta
        "lr": 1.0,
        "beta1": 0.9,
        "rho": 0.95,
        "eps": 1e-8,
        "grad_clip": 5,
        "select_data": select_data,
        "batch_ratio": "1",
        "total_data_usage_ratio": 1.0,
        "batch_max_length": batch_max_length,
        "imgH": packaging.DEFAULT_IMG_H,
        "imgW": 600,
        "rgb": False,
        "contrast_adjust": 0.0,
        "sensitive": True,
        "PAD": True,
        "data_filtering_off": False,
        # The shipped architecture: must match packaging's network_params and
        # EasyOCR's generation2 class, or the state dict will not load.
        "Transformation": "None",
        "FeatureExtraction": "VGG",
        "SequenceModeling": "BiLSTM",
        "Prediction": "CTC",
        "num_fiducial": 20,
        "input_channel": params["input_channel"],
        "output_channel": params["output_channel"],
        "hidden_size": params["hidden_size"],
        "decode": "greedy",
        "new_prediction": False,
        "freeze_FeatureFxtraction": False,  # upstream's spelling
        "freeze_SequenceModeling": False,
    }


def write_config(destination: Path, **kwargs) -> Path:
    """Write a trainer config yaml and return its path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(build_config(**kwargs), handle, allow_unicode=True, sort_keys=False)
    return destination
