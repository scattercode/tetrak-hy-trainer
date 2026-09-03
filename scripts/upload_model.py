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
REAL_EVAL_METRIC = "tetrak_hy_trainer.accuracy; higher is better for both figures"

# Every face the renderer actually used, from the `fonts:` line both runs
# logged. Mshtakan ships with macOS; no font file is redistributed.
FONTS = ["NotoSansArmenian.ttf", "NotoSerifArmenian.ttf", "Mshtakan.ttc"]

# Defects found after a version shipped. Recorded against the versions
# that carry them rather than quietly fixed forward, because the weights
# are published and people may be running them: a user seeing spurious
# quotation marks in v1's output deserves to find out why, and that the
# fix is to upgrade.
LABEL_QUOTING_DEFECT = (
    "21% of the training labels (36,918 of 175,500 for v1) were wrapped in "
    "quotation marks the images do not show. The trainer reads labels.csv "
    "with a regex splitting on the first comma rather than as CSV, so "
    "csv.writer's quoting of comma-bearing labels became part of the label. "
    "The model learnt it: inserting a quotation mark is the commonest single "
    "error in v1's output on the evaluation pages, ahead of every genuine "
    "character confusion. Fixed for v2, which does not have it. Found "
    "2026-09-01."
)

MISSING_ABBREVIATION_DOT = (
    "U+2024 ONE DOT LEADER, which the source transcripts use as the "
    "abbreviation dot, was absent from the charset, so the training pipeline "
    "silently dropped every crop containing it and the model has no class to "
    "emit it with. On the evaluation pages 518 words (5.8%) are therefore "
    "unwinnable, and the model emits the character zero times. Fixed for v2, "
    "whose charset includes it. Found 2026-08-31."
)

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
        "known_defects": [LABEL_QUOTING_DEFECT, MISSING_ABBREVIATION_DOT],
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
        "known_defects": [LABEL_QUOTING_DEFECT, MISSING_ABBREVIATION_DOT],
    },
    "v2": {
        # runs/v2/train.log; v1's recipe unchanged -- the differences are the
        # widened charset and two data fixes, not the training settings.
        "recipe": "scripts/train_synthetic.py (defaults; 150,000 iterations)",
        "dataset_config": None,  # rendered locally; see the note below
        "dataset_note": (
            "rendered locally from the harvested proofread pages rather than "
            "from a published dataset configuration: the v2 charset admits "
            "U+2024, so the crops differ from the crops-v1 configuration and "
            "a crops-v2 upload has not been made"
        ),
        "charset": {
            "num_class": 170,
            "added": ["U+2024 ONE DOT LEADER", "U+00B0 DEGREE SIGN"],
            "note": (
                "a charset change is a new model version by construction: CTC "
                "class indices are positional, so v2 weights cannot be loaded "
                "under v1's yaml or the reverse"
            ),
        },
        "synthetic_validation": {
            "word_accuracy": 99.333,
            "norm_edit_distance": 0.9989,
            "note": (
                "degraded validation crops, as v1's -- comparable with v1's "
                "98.8% rather than with v0's undegraded 99.722%"
            ),
        },
        "real_scan_evaluation": {
            "char_similarity": 0.1166,
            "word_recall": 0.6073,
            "word_recall_with_fold": 0.6919,
            "pages": 10,
            "source": REAL_EVAL_SOURCE,
            "metric": REAL_EVAL_METRIC,
            "baselines": {
                "tesseract-hye": {"char_similarity": 0.6968, "word_recall": 0.6621},
                "tesseract-hye-auto": {"char_similarity": 0.1281, "word_recall": 0.6637},
                "marker": {"char_similarity": 0.2580, "word_recall": 0.7660},
                "tetrak-hy-v1": {"char_similarity": 0.1004, "word_recall": 0.5014},
                "tetrak-hy-v0": {"char_similarity": 0.0745, "word_recall": 0.2742},
                "easyocr-stock": {"char_similarity": 0.0348, "word_recall": 0.0314},
            },
            "note": (
                "word_recall_with_fold applies tetrak_hy.fold_script from the "
                "tetrak-easyocr-armenian package, which folds cross-script "
                "homoglyphs (Latin h for հ, colon for ։) onto their Armenian "
                "forms in already-recognised text. That is the figure the "
                "shipped pipeline earns, and at 0.6919 it is the first of "
                "these models to pass tesseract-hye's 0.6621 word recall. The "
                "raw 0.6073 is the model alone. char_similarity remains "
                "dominated by reading order rather than recognition, for the "
                "reason recorded against v1."
            ),
        },
    },
    "v3": {
        # runs/v3/train.log. The first model here trained on real crops
        # rather than rendered ones -- brief 011 Stage 3 step 2.
        "recipe": (
            "scripts/finetune_real.py --iters 3000 --batch-ratio 0.5-0.5 "
            "(fine-tuned from v2; real and synthetic crops mixed in every batch)"
        ),
        "dataset_config": None,
        "dataset_note": (
            "fine-tuned on 6,097 crops cut from 30 human-proofread scans of "
            "volumes 5 and 6, labelled from the transcripts by "
            "detection-assisted alignment (tetrak_hy_trainer.align) and mixed "
            "50/50 in each batch with v2's 175,500 synthetic crops. Volume 2 "
            "is refused by tetrak_hy_trainer.heldout: it is the evaluation set"
        ),
        "charset": {
            "num_class": 170,
            "added": [],
            "note": "unchanged from v2; a fine-tune inherits its parent's charset",
        },
        "synthetic_validation": None,
        "real_crop_validation": {
            "word_accuracy": 95.0,
            "norm_edit_distance": 0.9936,
            "note": (
                "700 crops from pages held out of training -- split by page, "
                "never by crop, since crops from one page share its paper and "
                "scanning. Accuracy sits in a 93-95% band from iteration 500 "
                "onward, so this figure selects a plateau rather than a peak, "
                "and it is a poor proxy for page-level recall: a 10,000-"
                "iteration run reached the same 95% here while scoring 0.0564 "
                "worse on page word recall"
            ),
        },
        "real_scan_evaluation": {
            "char_similarity": 0.1470,
            "word_recall": 0.7356,
            "word_recall_with_fold": 0.7707,
            "pages": 10,
            "source": REAL_EVAL_SOURCE,
            "metric": REAL_EVAL_METRIC,
            "baselines": {
                "hye-calfa-n": {
                    "char_similarity": 0.8403,
                    "word_recall": 0.7889,
                    "note": "Calfa's Tesseract model, CC BY-NC 4.0 -- measured 2026-09-01 on the same pages",
                },
                "marker": {"char_similarity": 0.2580, "word_recall": 0.7660},
                "tesseract-hye": {"char_similarity": 0.6968, "word_recall": 0.6621},
                "tesseract-hye-auto": {"char_similarity": 0.1281, "word_recall": 0.6637},
                "tetrak-hy-v2": {"char_similarity": 0.1166, "word_recall": 0.6073},
                "tetrak-hy-v1": {"char_similarity": 0.1004, "word_recall": 0.5014},
                "tetrak-hy-v0": {"char_similarity": 0.0745, "word_recall": 0.2742},
                "easyocr-stock": {"char_similarity": 0.0348, "word_recall": 0.0314},
            },
            "note": (
                "word_recall_with_fold applies tetrak_hy.fold_script and is "
                "what the shipped pipeline earns. At 0.7707 it is ahead of "
                "marker's 0.7660 and tesseract-hye's 0.6621, and two points "
                "behind hye-calfa-n's 0.7889 -- Calfa's CC BY-NC model, "
                "measured on the same pages, is the strongest Armenian OCR "
                "here; this is the strongest permissively licensed one. "
                "char_similarity remains dominated by reading order rather "
                "than recognition, for the reason recorded against v1, and is "
                "not comparable across backends that serialise pages "
                "differently."
            ),
        },
    },
    "v5": {
        # runs/v4/train.log (pre-train) + runs/v6/train.log (fine-tune).
        # Two internal run numbers stand behind this one tag: the v4
        # pre-train was never published on its own, and the first
        # fine-tune attempt (internal v5, archived as
        # runs/v5-invalid-collided-crops) was discarded because a crop
        # filename collision gave 27% of its training images two or three
        # contradictory labels -- fixed in #9. The shipped weights are
        # internal run v6; the published sequence stays unbroken.
        "recipe": (
            "scripts/train_synthetic.py --iters 150000 --max-samples 120000 "
            "over 19 harvest dirs and 15 font faces (the unpublished v4 "
            "pre-train), then scripts/finetune_real.py --iters 12000 "
            "--batch-ratio 0.5-0.5 on real crops from 12 sources"
        ),
        "dataset_config": None,
        "dataset_note": (
            "pre-trained on 351,000 synthetic line crops rendered from "
            "~7,400 proofread Wikisource pages across the ASE (volumes 1, "
            "3-13), Western Armenian literature (Otyan, Totovents, "
            "Baronian), Tumanyan's collected works, Faustus of Byzantium "
            "1968, a popular medical encyclopedia and an Armenian-English "
            "dictionary; then fine-tuned on 51,078 real crops cut from 520 "
            "of those scans by detection-assisted alignment, mixed 50/50 "
            "in each batch with the synthetic set. Held-out material "
            "(ASE volume 2 whole, plus registered pages of each work -- "
            "tetrak_hy_trainer.heldout) is excluded from both stages"
        ),
        "charset": {
            "num_class": 175,
            "added": [
                "U+2026 HORIZONTAL ELLIPSIS",
                "U+005B/U+005D SQUARE BRACKETS",
                "U+2116 NUMERO SIGN",
                "U+00B2 SUPERSCRIPT TWO",
            ],
            "note": (
                "a charset change is a new model version by construction: CTC "
                "class indices are positional, so v5 weights cannot be loaded "
                "under v3's yaml or the reverse"
            ),
        },
        "synthetic_validation": {
            "word_accuracy": 95.2,
            "norm_edit_distance": 0.9949,
            "note": (
                "the pre-train's figure, on degraded crops drawn from 3.6x "
                "v2's vocabulary and 15 faces rather than 3 -- lower than "
                "v2's 99.3% because the validation set got harder, not the "
                "model worse"
            ),
        },
        "real_crop_validation": {
            "word_accuracy": 94.177,
            "norm_edit_distance": 0.9923,
            "note": (
                "5,530 crops from pages held out of the fine-tune, split by "
                "page, spanning all 12 sources. Accuracy climbed steadily "
                "from 89.4% at iteration 500 to 94.2% at 11,500 with "
                "validation loss flat from 8,500 on -- none of the "
                "overfitting that capped v3, because 51,078 real crops "
                "absorb 3.8 passes where v3's 6,097 could not"
            ),
        },
        "real_scan_evaluation": {
            "char_similarity": 0.1634,
            "word_recall": 0.8244,
            "word_recall_with_fold": 0.8295,
            "pages": 10,
            "source": REAL_EVAL_SOURCE,
            "metric": REAL_EVAL_METRIC,
            "baselines": {
                "hye-calfa-n": {
                    "char_similarity": 0.8403,
                    "word_recall": 0.7889,
                    "note": (
                        "Calfa's Tesseract model, CC BY-NC 4.0 -- measured "
                        "2026-09-01 on the same pages"
                    ),
                },
                "hye-paddle": {"char_similarity": 0.1627, "word_recall": 0.8073},
                "marker": {"char_similarity": 0.2580, "word_recall": 0.7660},
                "tesseract-hye": {"char_similarity": 0.6968, "word_recall": 0.6621},
                "tetrak-hy-v3": {"char_similarity": 0.1470, "word_recall": 0.7356},
                "tetrak-hy-v2": {"char_similarity": 0.1166, "word_recall": 0.6073},
            },
            "note": (
                "at 0.8244 raw this is the first of these models to lead "
                "every measured engine on word recall -- hye-paddle's 0.8073 "
                "and hye-calfa-n's 0.7889 included -- and the fold now adds "
                "only 0.005, because the model emits Armenian forms directly "
                "rather than Latin homoglyphs for the fold to repair. "
                "char_similarity remains dominated by reading order on these "
                "two-column pages, for the reason recorded against v1: "
                "hye-calfa-n's 0.84 reflects its layout analysis, not a "
                "recognition gap"
            ),
        },
        "per_register_evaluation": {
            "metric": "mean word recall (fold applied) over per-work held-out pages",
            "source": (
                "runs/eval/<work>/ built by scripts/build_eval_sets.py; "
                "scored by scripts/evaluate_registers.py"
            ),
            "registers": {
                "ase-vol2": 0.8295,
                "baronian-vol10": 0.8775,
                "dictionary-hy-en": 0.6009,
                "faustus-1968": 0.8215,
                "medical-encyclopedia": 0.7743,
                "otyan-works": 0.9003,
                "totovents-works": 0.9141,
                "tumanyan-elzh5": 0.8639,
            },
            "mean_over_registers": 0.8228,
            "note": (
                "v3 scored 0.4043 on this instrument and collapsed to "
                "0.23-0.33 on the Western Armenian literary sources; v5 "
                "holds 0.60-0.91 everywhere. Character similarity still "
                "splits by layout, not recognition -- 0.56-0.72 on "
                "single-column literary pages against 0.08-0.16 on "
                "multi-column encyclopedias -- so column-aware reading "
                "order is the remaining bottleneck (brief 012)"
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
    record = {
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
            **({"note": facts["dataset_note"]} if "dataset_note" in facts else {}),
        },
        "trainer_commit_at_upload": commit or None,
        "sha256": {name: sha256(bundle / name) for name in BUNDLE_FILES},
    }
    # Optional, and carried into the published record when present: a
    # defect found after a version shipped belongs beside its numbers,
    # not only in a commit message nobody downloading weights will read.
    for key in ("charset", "known_defects", "real_crop_validation"):
        if key in facts:
            record[key] = facts[key]
    return record


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
