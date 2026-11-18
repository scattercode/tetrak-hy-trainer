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
- ``wordlist.tsv.gz`` and, from v7, ``wordlist-classical.tsv.gz`` — the
  two word lists ``tetrak_hy.lexicon`` decodes with (``--wordlist`` and
  ``--classical-wordlist``), each with its attribution in the provenance.
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
import gzip
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

# Brief 013 corrected the metric itself. Figures recorded for v0-v5 used
# difflib with autojunk on, which on a page-length string treats most of
# the alphabet as junk, and counted the ASCII colon, the abbreviation dot
# and punctuation-only tokens against engines for what were transcription
# conventions. v6's figures, and the re-measured v5 figures beside them,
# use the corrected metric; the two sets are not comparable.
REAL_EVAL_METRIC_V6 = (
    "tetrak_hy_trainer.accuracy after brief 013: difflib autojunk off; "
    "։ scored as : and ․ as .; punctuation-only tokens not counted as words"
)
# Brief 014 added one equivalence: the two-letter եւ and the ligature և are
# one word printed two ways, so neither engine nor transcriber is scored on
# which it set. It moved no leader on any register.
REAL_EVAL_METRIC_V7 = REAL_EVAL_METRIC_V6 + "; եւ scored as և (brief 014)"

# The word lists tetrak_hy.lexicon decodes with, published beside the
# weights: one from v6 on, and from v7 a second built from the
# classical-orthography harvests, which the decoder reads as a union so a
# spelling in either is never rewritten into the other's (brief 014). Each
# list's attribution travels with it, read from the sidecar
# build_wordlist.py writes so it names the inputs actually used.
WORDLIST_NAME = "wordlist.tsv.gz"
CLASSICAL_WORDLIST_NAME = "wordlist-classical.tsv.gz"


def wordlist_record(wordlist: Path, name: str = WORDLIST_NAME) -> dict:
    """The provenance entry for a word list, refusing one that cannot ship.

    It is published as a ``.gz`` name, so plain text would be an asset that
    fails to decompress; and without its sidecar there is nothing true to
    say about its sources.
    """
    with wordlist.open("rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            raise SystemExit(f"{wordlist} is not gzip; build it with a .gz --out")
    with gzip.open(wordlist, "rt", encoding="utf-8") as handle:
        handle.readline()  # a truncated or corrupt stream fails here
    sidecar = wordlist.with_name(wordlist.name + ".json")
    if not sidecar.exists():
        raise SystemExit(f"{sidecar} is missing; rebuild the list with scripts/build_wordlist.py")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    return {
        "file": name,
        "sha256": sha256(wordlist),
        "words": meta["words"],
        "orthography": meta.get("orthography", "all"),
        "nayiri": meta["nayiri"],
        "licence": meta["licence"],
        "attribution": meta["attribution"],
    }


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

INHERITED_OVERLAP = (
    "v6 is fine-tuned from v5's weights, and its batches reuse v5's synthetic "
    "crops (relabelled, not re-rendered). v5 was trained before "
    "tetrak_hy_trainer.heldout.OVERLAPPING_PAGES existed, on text that "
    "included those 25 pages, which print evaluation pages' text: variants "
    "and reprints in Tumanyan's academic edition and Baronian's collected "
    "works vol. 10, two pages of Faustus of Byzantium and one of the "
    "Armenian Soviet Encyclopedia. The guard keeps them out of v6's new real "
    "crops, its all-caps set and the word list, but cannot remove the "
    "exposure from the starting weights or the reused synthetic crops, so "
    "v6's figures on those four registers may be flattered by it. A "
    "leakage-free model needs the synthetic set re-rendered without those "
    "pages and a pre-train from scratch."
)

SYN_CAPS_VALIDATION_TRAINED = (
    "The trainer selects dataset folders by substring, so selecting "
    "syn_caps also loaded syn_caps_val: v6 trained on those 750 synthetic "
    "validation crops. Nothing published is measured on them; the "
    "checkpoint was chosen on held-out real crops. "
    "render_synthetic_set.py now names its folders <name>_train and <name>_val."
)

OLD_REGISTER_LOSSES = (
    "Measured through Tetrak's pipeline on 2026-11-16, v7 reads five of the "
    "eight registers v6 was released on slightly worse by word recall "
    "(Faustus of Byzantium -0.018, medical encyclopedia -0.016, Otyan "
    "-0.014, Baronian vol. 10 -0.007, Totovents -0.003), the same size as "
    "the raw differences, so the losses are the recogniser's rather than "
    "the word lists' or the fold's. Shipped knowingly (brief 014 decision "
    "log, 2026-11-16) for the classical-orthography gains of +0.04 to +0.10 "
    "on eight of the nine new registers and for Cyrillic, which v6 cannot "
    "read at all. v6 stays tagged for a caller reading reformed print only."
)

V8_CYRILLIC_LOSS = (
    "v8 reads fewer of the Russian words in Tumanyan's apparatus than v7: 53 "
    "of the 81 on the evaluation pages against v7's 61, through Tetrak's "
    "pipeline on 2026-11-17. A control fine-tune with v7's own recipe for the "
    "same iterations kept 61, so the loss came with the italic crops, which "
    "carry almost no Cyrillic. Word recall on the register as a whole moved "
    "by -0.002. v7 stays tagged for a caller to whom the Russian matters more "
    "than the classical italics."
)
V8_THREE_DOTS = (
    "v8 writes three full stops where the page prints the ellipsis character "
    "more often than v7 does (25 times against once over the seventeen "
    "registers, with 23 ellipsis characters against v7's 60). The three "
    "italic books' transcripts type the dots, and the fine-tune learnt the "
    "transcribers' convention with the letterforms. Otyan's works, which "
    "print many ellipses, still gained word recall. To be folded at harvest "
    "before the next run."
)
V8_REGISTER_LOSSES = (
    "Against v7 through Tetrak's pipeline (2026-11-17), v8 reads the Armenian "
    "Soviet Encyclopedia slightly worse (word recall 0.917 to 0.909, nine "
    "pages of ten). The control fine-tune lost the same 0.008 on the same "
    "pages, so it is the cost of 3,000 more iterations on this recipe, not "
    "of the italic crops. Against v6, three of the eight registers v6 was "
    "released on are still read a little worse: the medical encyclopedia "
    "-0.008, Otyan -0.007 and Faustus of Byzantium -0.003, where v7 lost on "
    "five by up to 0.018. v6's figures are those measured on 2026-11-16, "
    "before Tetrak's faint-page rule, which v6 has not been re-read under."
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
    "v6": {
        # runs/v6-b013/ -- brief 013 Stage 3, fine-tuned from the released
        # v5 weights (runs/v6/saved_models/v6/best_accuracy.pth).
        "recipe": (
            "scripts/finetune_real.py --iters 6000 from v5, batches mixing "
            "real crops 0.5, v5's synthetic set relabelled with the "
            "Armenian full stop 0.4 (scripts/relabel_dataset.py) and "
            "all-caps lines 0.1 (scripts/render_synthetic_set.py --upper); "
            "iteration count fixed in advance, checkpoint chosen on held-out "
            "real crops"
        ),
        "dataset_config": None,
        "dataset_note": (
            "99,521 real crops (10,597 held out by page) cut with v5 from 862 "
            "scanned pages of 16 sources: v5's 12, plus Tumanyan's collected "
            "works vols. 7 and 10, Baronian's vols. 4 and 8, and 100 more "
            "pages of Faustus of Byzantium 1968. Transcript colons in "
            "Armenian words are labelled as the Armenian full stop. "
            "Evaluation pages, and 25 pages that print an evaluation page's "
            "text (tetrak_hy_trainer.heldout.OVERLAPPING_PAGES), are "
            "excluded from all of v6's new data; v5's exposure to the 25 is "
            "inherited (see known_defects)"
        ),
        "charset": {"num_class": 175, "added": [], "note": "unchanged from v5"},
        "known_defects": [INHERITED_OVERLAP, SYN_CAPS_VALIDATION_TRAINED],
        # The pre-train's faces, from runs/v4/train.log's `fonts:` line, and
        # the all-caps set's (runs/v6-b013/render-caps.log).
        "fonts": [
            "GHEAGpalatBld.otf",
            "GHEAGrapalatBlit.otf",
            "GHEAGrapalatRit.otf",
            "GHEAGrpalatReg.otf",
            "GHEAMariamBld.otf",
            "GHEAMariamBlit.otf",
            "GHEAMariamRIt.otf",
            "GHEAMariamReg.otf",
            "NotoSansArmenian.ttf",
            "NotoSerifArmenian.ttf",
            "arnamu.ttf",
            "arnamu_bold.ttf",
            "arnamu_italic.ttf",
            "arnamu_italic_bold.ttf",
            "Mshtakan.ttc",
        ],
        "synthetic_validation": {
            "word_accuracy": 95.2,
            "norm_edit_distance": 0.9949,
            "note": "inherited from the v4 pre-train; v6 is a fine-tune of v5",
        },
        "real_crop_validation": {
            "word_accuracy": 95.187,
            "norm_edit_distance": 0.9933,
            "note": "10,597 crops from pages held out of the fine-tune, all 16 sources",
        },
        "real_scan_evaluation": {
            "char_similarity": 0.3475,
            "word_recall": 0.8379,
            "pages": 10,
            "source": REAL_EVAL_SOURCE,
            "metric": REAL_EVAL_METRIC_V6,
            "baselines": {
                "tetrak-hy-v5": {"char_similarity": 0.3430, "word_recall": 0.8343},
                "hye-calfa-n": {"char_similarity": 0.9600, "word_recall": 0.8426},
                "hye-paddle": {"char_similarity": 0.3342, "word_recall": 0.8648},
                "marker": {"char_similarity": 0.5262, "word_recall": 0.8358},
                "tesseract-hye": {"char_similarity": 0.9456, "word_recall": 0.7728},
            },
            "note": (
                "the model alone, joined in detector order: char_similarity "
                "here measures reading order on three-column pages, not "
                "recognition. Tetrak's pipeline (layout, lexicon decoding, "
                "fold) scores 0.970 / 0.902 on the same pages"
            ),
        },
        "per_register_evaluation": {
            "metric": "mean word recall, model alone (no fold, no layout)",
            "source": (
                "runs/eval/<work>/ built by scripts/build_eval_sets.py; "
                "scored by scripts/evaluate_registers.py"
            ),
            "registers": {
                "ase-vol2": 0.8379,
                "baronian-vol10": 0.9110,
                "dictionary-hy-en": 0.6667,
                "faustus-1968": 0.8744,
                "medical-encyclopedia": 0.9260,
                "otyan-works": 0.9337,
                "totovents-works": 0.9289,
                "tumanyan-elzh5": 0.8958,
            },
            "mean_over_registers": 0.8718,
            "baseline_v5_mean": 0.8512,
            "note": (
                "higher than v5 on every register under the same metric. "
                "Not comparable with the figures recorded for v5 itself, "
                "which used the metric before brief 013 corrected it"
            ),
        },
    },
    "v7": {
        # runs/v7-b014/ -- brief 014: a pre-train from scratch on every
        # harvest with classical orthography a quarter of the lines, then a
        # fine-tune on twice v6's real crops with English lines in each batch.
        "recipe": (
            "scripts/train_synthetic.py --iters 150000 --classical-share 0.25 "
            "--mixed-share 0.05 over all 58 harvests (60,000 lines, 175,500 "
            "crops, charset v5; best synthetic validation at iteration "
            "132,000), then scripts/finetune_real.py --iters 6000 from that "
            "checkpoint, batches mixing real crops 0.5, the pre-train's "
            "synthetic set 0.3, all-caps lines 0.1 and English lines 0.1 "
            "(scripts/render_synthetic_set.py --upper, and --upper-share 0.1 "
            "over six Project Gutenberg novels); checkpoint chosen on held-out "
            "real crops (iteration 5,000)"
        ),
        "dataset_config": None,
        "dataset_note": (
            "217,362 real crops (21,574 held out by page) cut with v6 from "
            "2,136 scanned pages of 42 sources: v6's 16 plus 26 "
            "classical-orthography works harvested for brief 014, diaspora "
            "and pre-reform editions of Abovyan, Baronian, Kajaznuni, "
            "Khatisian, Khrimian, Toranian, Teodik and others. Three-column "
            "pages are split into their columns before alignment. Every "
            "evaluation page, and every page printing one's text "
            "(tetrak_hy_trainer.heldout), is excluded from the synthetic "
            "sample, the real crops and both word lists; nothing is inherited "
            "from an earlier model, the pre-train being from scratch. Crops "
            "whose label has a wholly Cyrillic token are kept (8); a token "
            "mixing Cyrillic with Armenian or Latin letters drops its crop"
        ),
        "charset": {
            "num_class": 246,
            "added": [
                "< > ³ ± (charset v4, 2026-10-04)",
                "՚ U+055A, the Armenian apostrophe of elision (charset v5)",
                "the Russian alphabet in both cases including ё/Ё, 66 letters (charset v5)",
            ],
            "note": (
                "245 characters plus the CTC blank, against v6's 174 plus "
                "blank. v4's angle brackets came with a label change: a "
                "printed '<' (the encyclopedia's derivation sign) is labelled "
                "as itself where earlier harvests folded it to «, so crops "
                "harvested before 2026-11-09 were re-cut. v5 (brief 014 "
                "Stage 1) adds the one printed character the classical "
                "harvests brought, and Russian for the apparatus of Soviet "
                "editions. Both are generated from Unicode ranges in "
                "tetrak_hy_trainer.charset; a charset change is a new model "
                "by construction, so take the yaml and the weights from the "
                "same revision"
            ),
        },
        "known_defects": [OLD_REGISTER_LOSSES],
        # The pre-train's faces (runs/v7-b014/train.log `fonts:`); the
        # all-caps and English sets used the same fifteen. The twelve GHEA
        # and Arnamu faces draw Cyrillic; the other three are kept off
        # lines with a Russian word.
        "fonts": [
            "GHEAGpalatBld.otf",
            "GHEAGrapalatBlit.otf",
            "GHEAGrapalatRit.otf",
            "GHEAGrpalatReg.otf",
            "GHEAMariamBld.otf",
            "GHEAMariamBlit.otf",
            "GHEAMariamRIt.otf",
            "GHEAMariamReg.otf",
            "NotoSansArmenian.ttf",
            "NotoSerifArmenian.ttf",
            "arnamu.ttf",
            "arnamu_bold.ttf",
            "arnamu_italic.ttf",
            "arnamu_italic_bold.ttf",
            "Mshtakan.ttc",
        ],
        "synthetic_validation": {
            "word_accuracy": 95.133,
            "norm_edit_distance": 0.9943,
            "note": (
                "the pre-train's own degraded validation crops, best at "
                "iteration 132,000 of 150,000"
            ),
        },
        "real_crop_validation": {
            "word_accuracy": 89.881,
            "norm_edit_distance": 0.9864,
            "note": (
                "21,574 crops from pages held out of the fine-tune, all 42 "
                "sources, at the chosen checkpoint (iteration 5,000). Lower "
                "than v6's 95.2 because the held-out set is twice the size "
                "and a third of it classical print and diaspora faces"
            ),
        },
        "real_scan_evaluation": {
            "char_similarity": 0.3493,
            "word_recall": 0.8457,
            "pages": 10,
            "source": REAL_EVAL_SOURCE,
            "metric": REAL_EVAL_METRIC_V7,
            "baselines": {
                "tetrak-hy-v6": {"char_similarity": 0.3475, "word_recall": 0.8379},
                "hye-calfa-n": {"char_similarity": 0.9600, "word_recall": 0.8426},
                "hye-paddle": {"char_similarity": 0.3342, "word_recall": 0.8648},
                "marker": {"char_similarity": 0.5262, "word_recall": 0.8358},
                "tesseract-hye": {"char_similarity": 0.9456, "word_recall": 0.7728},
            },
            "note": (
                "the model alone, joined in detector order: char_similarity "
                "here measures reading order on three-column pages, not "
                "recognition. Tetrak's pipeline (layout, both word lists, "
                "fold) scores 0.971 / 0.911 on the same pages"
            ),
        },
        "per_register_evaluation": {
            "metric": "mean word recall, model alone (no fold, no layout)",
            "source": (
                "runs/eval/<work>/ built by scripts/build_eval_sets.py; "
                "scored by scripts/evaluate_registers.py "
                "(runs/eval/registers/b014-v7-english-vs-v6.log)"
            ),
            "registers": {
                "abovyan-vol1": 0.7652,
                "aghbalyan-criticism": 0.7879,
                "aram-manukyan-documents": 0.8727,
                "ase-vol2": 0.8457,
                "baronian-azgayin-jojer": 0.8285,
                "baronian-vol10": 0.9010,
                "dictionary-hy-en": 0.6628,
                "faustus-1968": 0.8586,
                "haykakan-tpagrutyun": 0.8739,
                "kajaznuni-azg-ew-hayrenik": 0.8371,
                "khatisian-memoirs": 0.6719,
                "khrimian-works": 0.9286,
                "medical-encyclopedia": 0.9041,
                "mshak": 0.7421,
                "otyan-works": 0.9241,
                "toranian-erkirner-ew-marder": 0.9317,
                "totovents-works": 0.9231,
                "tumanyan-elzh5": 0.9122,
            },
            "mean_over_registers": 0.8428,
            "baseline_v6_mean": 0.8010,
            "note": (
                "eighteen registers: v6's eight, nine classical-orthography "
                "works added by brief 014 and two pages of the Mshak newspaper "
                "(1870s, not from Wikisource). v6 re-scored on the same pages "
                "under the same metric"
            ),
        },
        "full_pipeline_evaluation": {
            "metric": (
                "mean word recall through Tetrak's pipeline: layout, both word "
                "lists, fold with page context"
            ),
            "source": (
                "tetrak/tools/register_comparison.py over the trainer's "
                "runs/eval/<work>/, 2026-11-16 "
                "(runs/eval/registers/b014-v7-full-pipeline.log)"
            ),
            "registers": {
                "abovyan-vol1": 0.7939,
                "aghbalyan-criticism": 0.8239,
                "aram-manukyan-documents": 0.9340,
                "ase-vol2": 0.9109,
                "baronian-azgayin-jojer": 0.8880,
                "baronian-vol10": 0.9373,
                "dictionary-hy-en": 0.6874,
                "faustus-1968": 0.8979,
                "haykakan-tpagrutyun": 0.9269,
                "kajaznuni-azg-ew-hayrenik": 0.9005,
                "khatisian-memoirs": 0.7616,
                "khrimian-works": 0.9576,
                "medical-encyclopedia": 0.9513,
                "otyan-works": 0.9504,
                "toranian-erkirner-ew-marder": 0.9386,
                "totovents-works": 0.9649,
                "tumanyan-elzh5": 0.9366,
            },
            "mean_over_registers": 0.8919,
            "baseline_v6_mean": 0.8614,
            "cyrillic_words_tumanyan": "61 of 81 read (v6: 0 of 81)",
            "note": (
                "leads every engine measured on mean word recall (Calfa "
                "hye-paddle 0.880, hye-calfa-n 0.829, Marker 0.775, Tesseract "
                "hye 0.663 over the same seventeen registers; hye-paddle run "
                "on the classical ones 2026-11-16 under paddleocr 3.7) and on "
                "eleven of seventeen registers. hye-paddle leads the other "
                "six, five of them classical: Abovyan (0.910 against 0.794), "
                "Khatisian (0.856 against 0.762), Aghbalyan (0.846 against "
                "0.824), Faustus (0.922 against 0.898), Aram Manukyan (0.941 "
                "against 0.934) and Baronian's Azgayin jojer (0.894 against "
                "0.888). See known_defects for the five old registers it "
                "reads slightly worse than v6"
            ),
        },
    },
}

# runs/v7-b023-italic/ -- brief 023: v7 fine-tuned a further 3,000 iterations
# with crops from the three books set in italics, which v7 read worst.
# Defined after the table because it shares v7's charset, faces and
# pre-train, and repeating them would let the two drift.
VERSIONS["v8"] = {
    "recipe": (
        "scripts/finetune_real.py --iters 3000 from v7's released checkpoint, "
        "batches mixing v7's real crops 0.4, new crops from three italic "
        "books 0.1, v7's synthetic set 0.3, all-caps lines 0.1 and English "
        "lines 0.1 (v7's recipe with a fifth of the real half given to the "
        "italic crops); checkpoint chosen on held-out real crops (iteration "
        "2,500). A control with v7's own recipe for the same iterations "
        "separates the material from the extra training "
        "(runs/v7-b023/finetune.sh, finetune-italic.sh)"
    ),
    "dataset_config": None,
    "dataset_note": (
        "v7's real crops (195,582 in training, after 206 moved out with the "
        "pages the new pass holds for validation) and 28,331 more cut with "
        "v7 from 616 scanned pages of the three works it read worst, all "
        "set in italics: Khatisian's memoirs (13,927 crops from 391 pages, "
        "where v7 had trained on 1,278), Abovyan's collected works vol. 1 "
        "(12,365 from 146) and The First Love (2,039 from 79). 25,542 of "
        "them train and 2,789 validate, split by page; validation is 23,722 "
        "crops in all. v6 aligned 0.29 of Khatisian's boxes to the "
        "transcript on the scans v7 trained from; v7 aligns 0.43 over "
        "every proofread page. Every evaluation page, and every page printing one's text "
        "(tetrak_hy_trainer.heldout), is excluded as before; the synthetic "
        "sets and both word lists are v7's, unchanged"
    ),
    "charset": {
        **VERSIONS["v7"]["charset"],
        "added": [],
        "note": (
            "unchanged from v7: 245 characters plus the CTC blank. The yaml "
            "is identical, so v8's weights load wherever v7's do"
        ),
    },
    "known_defects": [V8_CYRILLIC_LOSS, V8_THREE_DOTS, V8_REGISTER_LOSSES],
    "fonts": VERSIONS["v7"]["fonts"],
    "synthetic_validation": {
        **VERSIONS["v7"]["synthetic_validation"],
        "note": (
            "v7's pre-train, which v8 inherits through v7's released "
            "weights; v8 rendered no synthetic set of its own"
        ),
    },
    "real_crop_validation": {
        "word_accuracy": 88.745,
        "norm_edit_distance": 0.9729,
        "note": (
            "23,722 crops from pages held out of the fine-tune, at the "
            "chosen checkpoint (iteration 2,500). Not comparable with v7's "
            "89.9: the set gained 2,789 crops from the italic books, the "
            "hardest print in it. Accuracy on it swings by a point between "
            "checkpoints of one run"
        ),
    },
    "real_scan_evaluation": {
        "char_similarity": 0.3491,
        "word_recall": 0.8418,
        "pages": 10,
        "source": REAL_EVAL_SOURCE,
        "metric": REAL_EVAL_METRIC_V7,
        "baselines": {
            "tetrak-hy-v7": {"char_similarity": 0.3493, "word_recall": 0.8457},
            "hye-calfa-n": {"char_similarity": 0.9600, "word_recall": 0.8426},
            "hye-paddle": {"char_similarity": 0.3342, "word_recall": 0.8648},
            "marker": {"char_similarity": 0.5262, "word_recall": 0.8358},
            "tesseract-hye": {"char_similarity": 0.9456, "word_recall": 0.7728},
        },
        "note": (
            "the model alone, joined in detector order: char_similarity "
            "here measures reading order on three-column pages, not "
            "recognition. Tetrak's pipeline (faint-page rule, layout, both "
            "word lists, fold) scores 0.975 / 0.909 on the same pages. This "
            "is the one register v8 reads worse than v7; see known_defects"
        ),
    },
    "per_register_evaluation": {
        "metric": "mean word recall, model alone (no fold, no layout)",
        "source": (
            "runs/eval/<work>/ built by scripts/build_eval_sets.py; "
            "scored by scripts/evaluate_registers.py "
            "(runs/eval/registers/b023-v7-b023-italic-raw.log)"
        ),
        "registers": {
            "abovyan-vol1": 0.8340,
            "aghbalyan-criticism": 0.7892,
            "aram-manukyan-documents": 0.8910,
            "ase-vol2": 0.8418,
            "baronian-azgayin-jojer": 0.8506,
            "baronian-vol10": 0.9091,
            "dictionary-hy-en": 0.6909,
            "faustus-1968": 0.8810,
            "haykakan-tpagrutyun": 0.8766,
            "kajaznuni-azg-ew-hayrenik": 0.8419,
            "khatisian-memoirs": 0.7725,
            "khrimian-works": 0.9306,
            "medical-encyclopedia": 0.9169,
            "mshak": 0.7923,
            "otyan-works": 0.9290,
            "toranian-erkirner-ew-marder": 0.9384,
            "totovents-works": 0.9294,
            "tumanyan-elzh5": 0.9137,
        },
        "mean_over_registers": 0.8627,
        "baseline_v7_mean": 0.8428,
        "control_mean": 0.8506,
        "note": (
            "the same eighteen registers and pages as v7. The control is "
            "v7's own recipe for the same 3,000 iterations"
        ),
    },
    "full_pipeline_evaluation": {
        "metric": (
            "mean word recall through Tetrak's pipeline: faint-page rule, "
            "layout, both word lists, fold with page context"
        ),
        "source": (
            "tetrak/tools/register_comparison.py over the trainer's "
            "runs/eval/<work>/, Tetrak 5.28.0, 2026-11-17 "
            "(runs/eval/registers/b023-v8-full-pipeline.log)"
        ),
        "registers": {
            "abovyan-vol1": 0.8558,
            "aghbalyan-criticism": 0.8242,
            "aram-manukyan-documents": 0.9413,
            "ase-vol2": 0.9095,
            "baronian-azgayin-jojer": 0.9008,
            "baronian-vol10": 0.9450,
            "dictionary-hy-en": 0.7078,
            "faustus-1968": 0.9137,
            "haykakan-tpagrutyun": 0.9327,
            "kajaznuni-azg-ew-hayrenik": 0.9109,
            "khatisian-memoirs": 0.8383,
            "khrimian-works": 0.9581,
            "medical-encyclopedia": 0.9602,
            "otyan-works": 0.9575,
            "toranian-erkirner-ew-marder": 0.9432,
            "totovents-works": 0.9687,
            "tumanyan-elzh5": 0.9346,
        },
        "mean_over_registers": 0.9060,
        "baseline_v7_mean": 0.8949,
        "control_mean": 0.8963,
        "cyrillic_words_tumanyan": "53 of 81 read (v7: 61 of 81)",
        "note": (
            "v7 re-read the same day under the same pipeline, whose "
            "faint-page rule (Tetrak 5.28.0) lifted v7 itself from the 0.892 "
            "published with it. Fifteen of seventeen registers gain, "
            "Khatisian most (0.762 to 0.838, every page). v8 leads every "
            "engine measured on mean word recall (Calfa hye-paddle 0.880, "
            "hye-calfa-n 0.829, Marker 0.775, Tesseract hye 0.663) and on "
            "twelve registers, and is level with hye-paddle on Aram "
            "Manukyan (0.941 each). hye-paddle leads the other four: "
            "Abovyan (0.910 against 0.856), Khatisian (0.856 against "
            "0.838), Aghbalyan (0.846 against 0.824) and Faustus (0.922 "
            "against 0.914)"
        ),
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


def provenance(
    bundle: Path,
    version: str,
    dataset_revision: str | None,
    wordlist: Path | None = None,
    classical_wordlist: Path | None = None,
) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    facts = facts_for(version)
    record = {
        "model": "tetrak_hy",
        "version": version,
        "architecture": "EasyOCR generation2 (VGG + 2x BiLSTM + CTC)",
        "recipe": facts["recipe"],
        "fonts": facts.get("fonts", FONTS),
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
    for key in (
        "charset",
        "known_defects",
        "real_crop_validation",
        "per_register_evaluation",
        "full_pipeline_evaluation",
    ):
        if key in facts:
            record[key] = facts[key]
    if wordlist is not None:
        record["wordlist"] = wordlist_record(wordlist)
    if classical_wordlist is not None:
        record["classical_wordlist"] = wordlist_record(classical_wordlist, CLASSICAL_WORDLIST_NAME)
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
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=None,
        help=f"word list for tetrak_hy.lexicon, uploaded as {WORDLIST_NAME} (v6 on)",
    )
    parser.add_argument(
        "--classical-wordlist",
        type=Path,
        default=None,
        help=(
            "the classical-orthography word list, uploaded as "
            f"{CLASSICAL_WORDLIST_NAME} beside the modern one (v7 on)"
        ),
    )
    args = parser.parse_args()
    if args.classical_wordlist is not None and args.wordlist is None:
        parser.error("--classical-wordlist needs --wordlist: the decoder reads the two as a union")

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

    record = provenance(
        args.bundle_dir,
        args.version_tag,
        dataset_revision,
        args.wordlist,
        args.classical_wordlist,
    )

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
        for wordlist, name in (
            (args.wordlist, WORDLIST_NAME),
            (args.classical_wordlist, CLASSICAL_WORDLIST_NAME),
        ):
            if wordlist is not None:
                api.upload_file(
                    path_or_fileobj=wordlist,
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

    print(f"Done: https://huggingface.co/{args.repo_id}, tagged {args.version_tag}.")
    print(
        "A repository created by this run is private: review it, then rerun with "
        "--make-public. An existing public repository is live already."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
