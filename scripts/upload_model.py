#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.27",
#     "torch>=2.0",
#     # safetensors converts torch tensors via numpy, which torch does not
#     # pull in on its own -- without it save_file raises ModuleNotFoundError.
#     "numpy>=1.24",
#     "safetensors>=0.4",
#     "pyyaml>=6.0",
# ]
# ///
"""Validate a packaged tetrak_hy bundle and upload it to the Hugging Face Hub.

Takes a bundle directory produced by the trainer's packaging step
(``runs/<run>/bundle``) and publishes ``tetrak/easyocr-armenian``:

- ``tetrak_hy.pth``, ``tetrak_hy.py``, ``tetrak_hy.yaml`` — the EasyOCR
  deliverable, byte-identical to the bundle. (``craft_mlt_25k.pth`` in
  the bundle is EasyOCR's own detector and is never uploaded.)
- ``model.safetensors`` — the same tensors converted from the ``.pth``,
  with the DataParallel ``module.`` prefix stripped.
- ``provenance.json`` — recipe, dataset revision, charset, checksums.
- ``README.md`` — from ``scripts/model_card.md``.

Before uploading, the weights are checked against the yaml: the CTC
head's output size must equal ``len(character_list) + 1`` (the blank),
so a charset/weights mismatch fails here rather than in the field.

The repository is created **private**; review it on the Hub, then flip
it public (``--make-public``). The upload is tagged with
``--version-tag`` so downstream code can pin ``revision=``.

Each version's recipe and scores live in the ``VERSIONS`` table below,
and a tag with no entry there is refused rather than published with
empty provenance.

Requires a Hugging Face login with write access to the ``tetrak`` org
(``hf auth login``). Run from the repo root:

    uv run scripts/upload_model.py --bundle-dir runs/v1/bundle \
        --version-tag v1 --dry-run   # validate + convert only
    uv run scripts/upload_model.py --bundle-dir runs/v1/bundle \
        --version-tag v1             # validate + push (private)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ID = "tetrak/easyocr-armenian"
DATASET_ID = "tetrak/armenian-ocr-crops"
ROOT = Path(__file__).resolve().parent.parent
CARD = Path(__file__).resolve().parent / "model_card.md"
BUNDLE_FILES = ("tetrak_hy.pth", "tetrak_hy.py", "tetrak_hy.yaml")

# Both released versions are scored on the same real pages, so the wording is
# shared rather than repeated per version.
REAL_EVAL_SOURCE = (
    "Armenian Soviet Encyclopedia vol. 2, pages 105-114 from Armenian "
    "Wikisource, proofread to quality level 4 (runs/eval/ase-vol2)"
)
REAL_EVAL_METRIC = "tetrak_ocr.accuracy; higher is better for both figures"

# Every face the renderer actually used, from the `fonts:` line both runs
# logged. Mshtakan ships with macOS; no font file is redistributed.
FONTS = ["NotoSansArmenian.ttf", "NotoSerifArmenian.ttf", "Mshtakan.ttc"]

# What each released version was trained on and what it scored, kept per
# version rather than as generic strings: a release whose numbers are not
# written down here is a release nobody can check. An unknown tag is refused
# rather than recorded as nulls -- see facts_for().
VERSIONS = {
    "v0": {
        # runs/v0/train.log
        "recipe": "scripts/train_synthetic.py --line-tokens-max 1 --no-augment --min-size 36",
        "dataset_config": "crops",
        "synthetic_validation": {
            "word_accuracy": 99.722,
            "norm_edit_distance": 0.9983,
            "note": (
                "the v0 validation crops are undegraded, so this figure says "
                "almost nothing about real scans"
            ),
        },
        "real_scan_evaluation": {
            "char_similarity": 0.0745,
            "word_recall": 0.2742,
            "pages": 10,
            "source": REAL_EVAL_SOURCE,
            "metric": REAL_EVAL_METRIC,
        },
    },
    "v1": {
        # runs/v1/train.log; the recipe is train_synthetic.py's defaults --
        # line samples of 1-4 consecutive tokens, sizes down to 18px,
        # degradations on both splits, imgW 800, batch_max_length 60.
        "recipe": "scripts/train_synthetic.py (defaults; 150,000 iterations)",
        "dataset_config": "crops-v1",
        "synthetic_validation": {
            "word_accuracy": 98.8,
            "norm_edit_distance": 0.9974,
            "note": (
                "the v1 validation crops are degraded like the training ones, "
                "so unlike v0 this figure is measured on realistic input"
            ),
        },
        "real_scan_evaluation": {
            "char_similarity": 0.1004,
            "word_recall": 0.5014,
            "pages": 10,
            "source": REAL_EVAL_SOURCE,
            "metric": REAL_EVAL_METRIC,
            "baselines": {
                "tesseract-hye": {"char_similarity": 0.6968, "word_recall": 0.6621},
                "tesseract-hye-auto": {"char_similarity": 0.1281, "word_recall": 0.6637},
                "marker": {"char_similarity": 0.2580, "word_recall": 0.7660},
                "tetrak-hy-v0": {"char_similarity": 0.0745, "word_recall": 0.2742},
                "easyocr-stock": {"char_similarity": 0.0348, "word_recall": 0.0314},
            },
            "note": (
                "char_similarity here is dominated by reading order on these "
                "two-column pages rather than by recognition: tesseract-hye-auto "
                "reads words as well as tesseract-hye (0.664 vs 0.662 word "
                "recall) yet scores 0.128 char_similarity, close to v1's, "
                "because the evaluation joins detected lines with a newline in "
                "detector order. v1's genuine recognition gap against "
                "tesseract-hye is word recall 0.50 vs 0.66."
            ),
        },
    },
}


def facts_for(version: str) -> dict:
    """The recorded facts for *version*, or a refusal to invent them."""
    try:
        return VERSIONS[version]
    except KeyError:
        raise SystemExit(
            f"No recorded facts for {version!r}. Add an entry to VERSIONS in "
            f"{Path(__file__).name} -- recipe, dataset config, synthetic "
            "validation and real-scan evaluation -- before publishing it."
        ) from None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_check(bundle: Path) -> dict:
    """Load the state dict, strip ``module.``, and check it against the yaml."""
    import torch

    for name in BUNDLE_FILES:
        if not (bundle / name).is_file():
            raise FileNotFoundError(bundle / name)

    config = yaml.safe_load((bundle / "tetrak_hy.yaml").read_text(encoding="utf-8"))
    for key in ("character_list", "lang_list", "imgH", "network_params"):
        if key not in config:
            raise ValueError(f"tetrak_hy.yaml is missing {key!r}")

    state = torch.load(bundle / "tetrak_hy.pth", map_location="cpu", weights_only=True)
    stripped = {k.removeprefix("module."): v for k, v in state.items()}

    num_class = stripped["Prediction.bias"].shape[0]
    expected = len(config["character_list"]) + 1  # + the CTC blank
    if num_class != expected:
        raise ValueError(
            f"CTC head predicts {num_class} classes but the yaml charset implies {expected}"
        )
    print(f"Bundle OK: {len(stripped)} tensors, {num_class} classes, imgH {config['imgH']}")
    return stripped


def write_safetensors(stripped: dict, target: Path) -> None:
    from safetensors.torch import save_file

    save_file(stripped, str(target))


def provenance(bundle: Path, version: str, dataset_revision: str | None) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    facts = facts_for(version)
    return {
        "model": "tetrak_hy",
        "version": version,
        "architecture": "EasyOCR generation2 (VGG + 2x BiLSTM + CTC)",
        "recipe": facts["recipe"],
        "fonts": FONTS,
        "synthetic_validation": facts["synthetic_validation"],
        "real_scan_evaluation": facts["real_scan_evaluation"],
        "dataset": {
            "repo": DATASET_ID,
            "config": facts["dataset_config"],
            "revision": dataset_revision,
        },
        "trainer_commit_at_upload": commit or None,
        "sha256": {name: sha256(bundle / name) for name in BUNDLE_FILES},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "runs" / "v0" / "bundle")
    parser.add_argument("--version-tag", default="v0", help="tag for this weights release")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, convert and print the provenance without uploading",
    )
    parser.add_argument(
        "--make-public",
        action="store_true",
        help="flip an already-reviewed repository to public and exit",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, ModelCard
    from huggingface_hub.errors import LocalTokenNotFoundError

    api = HfApi()
    try:
        account = api.whoami()
    except (LocalTokenNotFoundError, OSError) as error:
        print(f"Not logged in to the Hugging Face Hub ({error}).", file=sys.stderr)
        print("Run `hf auth login` with a write token for `tetrak`.", file=sys.stderr)
        return 1
    print(f"Logged in as {account['name']}")

    if args.make_public:
        api.update_repo_settings(args.repo_id, repo_type="model", private=False)
        print(f"{args.repo_id} is now public.")
        return 0

    stripped = load_and_check(args.bundle_dir)

    dataset_revision = None
    try:
        dataset_revision = api.dataset_info(DATASET_ID).sha
    except Exception as error:  # noqa: BLE001 — provenance is best-effort offline
        print(f"Could not resolve the dataset revision ({error}); recording null.")

    record = provenance(args.bundle_dir, args.version_tag, dataset_revision)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        write_safetensors(stripped, scratch_dir / "model.safetensors")
        (scratch_dir / "provenance.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, indent=2))

        if args.dry_run:
            print("Dry run — nothing uploaded.")
            return 0

        api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
        for name in BUNDLE_FILES:
            api.upload_file(
                path_or_fileobj=args.bundle_dir / name,
                path_in_repo=name,
                repo_id=args.repo_id,
                commit_message=f"Add {name} ({args.version_tag})",
            )
        for name in ("model.safetensors", "provenance.json"):
            api.upload_file(
                path_or_fileobj=scratch_dir / name,
                path_in_repo=name,
                repo_id=args.repo_id,
                commit_message=f"Add {name} ({args.version_tag})",
            )
        ModelCard(CARD.read_text(encoding="utf-8")).push_to_hub(
            args.repo_id, commit_message="Add the model card"
        )
        api.create_tag(args.repo_id, tag=args.version_tag, repo_type="model", exist_ok=True)

    print(f"Done: https://huggingface.co/{args.repo_id} (private, tagged {args.version_tag} —")
    print("review, then flip it public in the repo settings or rerun with --make-public).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
