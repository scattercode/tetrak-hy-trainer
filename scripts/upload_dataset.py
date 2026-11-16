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
``runs/`` and pushes it as one configuration per synthetic crop set that
trained a released model, plus the corpus:

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
- ``crops-v4`` — the 351,000-crop pre-train set rendered from the
  fourteen-work corpus in fifteen faces (``runs/v4/all_data``), which
  pre-trained v4 and, through fine-tunes, v5 and v6 (v6 with the
  colon relabelled by ``scripts/relabel_dataset.py``).
- ``crops-v6-caps`` — v6's all-caps headword lines
  (``runs/v6-b013/all_data/syn_caps``).
- ``crops-v7``, ``crops-v7-caps``, ``crops-v7-latin`` — v7's pre-train
  set over all 58 works with a quarter classical orthography and a
  twentieth mixing Armenian with Russian, its all-caps set, and its
  English set from six Project Gutenberg novels (``runs/v7-b014/all_data``).

The real crops cut from scans (``real_train``/``real_val``) are not
published: they are fragments of the scanned pages themselves, and are
reproducible from the transcripts with ``scripts/harvest_real_crops.py``.

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
import json
import re
import sys
from pathlib import Path

REPO_ID = "tetrak/armenian-ocr-crops"
ROOT = Path(__file__).resolve().parent.parent
CARD = Path(__file__).resolve().parent / "dataset_card.md"

_VOLUME = re.compile(r"(\d+)\.djvu$")

# Crop configuration -> its (train, validation) split directories under runs/.
# The mapping is explicit because the sets are different data, not renders of
# one recipe: `crops` is v0's single-word crops, `crops-v1` the line-shaped
# ones that trained v1, and so on. Each set is published as the trainer read
# it, so a released model's provenance can name a configuration and mean it.
CROP_CONFIGS = {
    "crops": ("v0/all_data/v0_train", "v0/all_data/v0_val"),
    "crops-v1": ("v1/all_data/syn_train", "v1/all_data/syn_val"),
    "crops-v4": ("v4/all_data/syn_train", "v4/all_data/syn_val"),
    "crops-v6-caps": ("v6-b013/all_data/syn_caps", "v6-b013/all_data/syn_caps_val"),
    "crops-v7": ("v7-b014/all_data/syn_train", "v7-b014/all_data/syn_val"),
    "crops-v7-caps": ("v7-b014/all_data/syn_caps_train", "v7-b014/all_data/syn_caps_val"),
    "crops-v7-latin": ("v7-b014/all_data/syn_latin_train", "v7-b014/all_data/syn_latin_val"),
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
    """``image`` (path) + ``text`` rows from a trainer-format split.

    ``labels.csv`` is read the way the trainer reads it, not as CSV: each
    line splits at its first comma and the rest is the label verbatim
    (``synth.write_labels``). A label may contain commas and quotation
    marks -- the English set's do -- and a CSV reader would treat a
    leading quotation mark as quoting and swallow the lines that follow.
    """
    rows: list[dict] = []
    lines = (split_dir / "labels.csv").read_text(encoding="utf-8").splitlines()
    if lines[:1] != ["filename,words"]:
        raise ValueError(f"{split_dir / 'labels.csv'} does not start with the trainer's header")
    for line in lines[1:]:
        filename, _, text = line.partition(",")
        image = split_dir / filename
        if not image.is_file():
            raise FileNotFoundError(image)
        rows.append({"image": str(image), "text": text})
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
    parser.add_argument(
        "--tag",
        default=None,
        help="after pushing, tag the repository's head with this name (the weights version "
        "the pushed configurations trained, e.g. v7), so provenance can pin a revision",
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
    if args.tag:
        api.create_tag(args.repo_id, tag=args.tag, repo_type="dataset", exist_ok=True)
        print(f"Tagged {args.tag}.")
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}.")
    print(
        "A repository created by this run is private: review it, then rerun with "
        "--make-public. An existing public repository is live already."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
