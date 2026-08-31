#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.2",
#     "huggingface_hub>=0.27",
#     "pillow>=10.0",
# ]
# ///
"""Package the training data and upload it to the Hugging Face Hub.

Builds the ``tetrak/armenian-ocr-crops`` dataset from what lives under
``runs/`` and pushes it as three configurations:

- ``corpus`` — the harvested, proofread Armenian Soviet Encyclopedia
  pages (``runs/v0/harvest`` plus every ``runs/v1/harvest-vol*``), one
  row per page with its Wikisource provenance (page id, revision id,
  proofread quality) and plain text.
- ``crops`` — the v0 synthetic pre-training set
  (``runs/v0/all_data``), ``train`` and ``validation`` splits of
  ``image`` + ``text`` rows, images embedded in Parquet shards so the
  Hub's dataset viewer works.
- ``crops-v1`` — the v1 synthetic training set
  (``runs/v1/all_data``), the same columns and splits, line-shaped
  crops of 1-4 consecutive tokens with degradations applied to both
  splits. This is the set that trained the published v1 weights.

The repository is created **private**; review the card and viewer on
the Hub, then flip it public from the repo's settings page (or rerun
with ``--make-public``). The dataset card is kept in
``scripts/dataset_card.md`` and uploaded last, merged with the
metadata the push generates.

Requires a Hugging Face login with write access to the ``tetrak`` org
(``hf auth login``). Run from the repo root:

    uv run scripts/upload_dataset.py --dry-run   # build + summarise only
    uv run scripts/upload_dataset.py             # build + push (private)

The crop pushes move a lot of embedded images -- ~700 MB for ``crops``
and ~1.2 GB for ``crops-v1`` -- so expect them to take a while on a
domestic uplink. Use ``--only`` to push one configuration at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ID = "tetrak/armenian-ocr-crops"
ROOT = Path(__file__).resolve().parent.parent
CARD = Path(__file__).resolve().parent / "dataset_card.md"

_VOLUME = re.compile(r"(\d+)\.djvu$")

# Crop configuration -> its (train, validation) split directories under runs/.
# The mapping is explicit because the two sets are different data, not two
# renders of one recipe: `crops` is v0's single-word crops, `crops-v1` the
# line-shaped ones that trained v1.
CROP_CONFIGS = {
    "crops": ("v0/all_data/v0_train", "v0/all_data/v0_val"),
    "crops-v1": ("v1/all_data/syn_train", "v1/all_data/syn_val"),
}


def corpus_rows(runs: Path) -> list[dict]:
    """One row per harvested page, across every harvest directory."""
    harvest_dirs = sorted([runs / "v0" / "harvest", *sorted((runs / "v1").glob("harvest-vol*"))])
    rows: list[dict] = []
    for directory in harvest_dirs:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        match = _VOLUME.search(manifest["index"])
        if match is None:
            raise ValueError(f"cannot read a volume number from {manifest['index']!r}")
        volume = int(match.group(1))
        for page in manifest["pages"]:
            rows.append(
                {
                    "volume": volume,
                    "page_number": page["page_number"],
                    "title": page["title"],
                    "pageid": page["pageid"],
                    "revid": page["revid"],
                    "quality": page["quality"],
                    "index": manifest["index"],
                    "text": (directory / page["text"]).read_text(encoding="utf-8"),
                }
            )
    rows.sort(key=lambda row: (row["volume"], row["page_number"]))
    return rows


def crop_rows(split_dir: Path) -> list[dict]:
    """``image`` (path) + ``text`` rows from a trainer-format split."""
    rows: list[dict] = []
    with (split_dir / "labels.csv").open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            image = split_dir / record["filename"]
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append({"image": str(image), "text": record["words"]})
    return rows


def build_corpus(runs: Path):
    from datasets import Dataset, DatasetDict, Features, Value

    features = Features(
        {
            "volume": Value("int32"),
            "page_number": Value("int32"),
            "title": Value("string"),
            "pageid": Value("int64"),
            "revid": Value("int64"),
            "quality": Value("int32"),
            "index": Value("string"),
            "text": Value("string"),
        }
    )
    return DatasetDict({"train": Dataset.from_list(corpus_rows(runs), features=features)})


def build_crops(runs: Path, config: str):
    from datasets import Dataset, DatasetDict, Features, Image, Value

    features = Features({"image": Image(), "text": Value("string")})
    train, validation = CROP_CONFIGS[config]
    return DatasetDict(
        {
            "train": Dataset.from_list(crop_rows(runs / train), features=features),
            "validation": Dataset.from_list(crop_rows(runs / validation), features=features),
        }
    )


def push_card(repo_id: str) -> None:
    """Upload scripts/dataset_card.md, keeping the pushed config metadata."""
    from huggingface_hub import DatasetCard

    ours = DatasetCard(CARD.read_text(encoding="utf-8"))
    generated = DatasetCard.load(repo_id)
    merged = generated.data.to_dict()
    merged.update(ours.data.to_dict())
    for key, value in merged.items():
        setattr(ours.data, key, value)
    ours.push_to_hub(repo_id, commit_message="Add the dataset card")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--only",
        choices=["corpus", *CROP_CONFIGS],
        help="push a single configuration (default: all of them, corpus first)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the datasets and print a summary without uploading",
    )
    parser.add_argument(
        "--make-public",
        action="store_true",
        help="flip an already-reviewed repository to public and exit",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="re-push scripts/dataset_card.md to the existing repository and exit",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi
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
        api.update_repo_settings(args.repo_id, repo_type="dataset", private=False)
        print(f"{args.repo_id} is now public.")
        return 0

    if args.card_only:
        push_card(args.repo_id)
        print(f"Card updated: https://huggingface.co/datasets/{args.repo_id}")
        return 0

    builds = {}
    if args.only in (None, "corpus"):
        builds["corpus"] = build_corpus(args.runs_dir)
    for config in CROP_CONFIGS:
        if args.only in (None, config):
            builds[config] = build_crops(args.runs_dir, config)

    for name, dataset_dict in builds.items():
        counts = ", ".join(f"{split}: {len(ds):,}" for split, ds in dataset_dict.items())
        print(f"{name}: {counts}")

    if args.dry_run:
        print("Dry run — nothing uploaded.")
        return 0

    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    for name, dataset_dict in builds.items():
        dataset_dict.push_to_hub(
            args.repo_id,
            config_name=name,
            commit_message=f"Add the {name} configuration",
        )
        print(f"Pushed {name}.")
    push_card(args.repo_id)
    print(f"Done: https://huggingface.co/datasets/{args.repo_id} (private — review, then")
    print("flip it public in the repo settings or rerun with --make-public).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
