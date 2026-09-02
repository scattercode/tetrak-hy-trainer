#!/usr/bin/env python3
"""Cut labelled training crops from real scans, via detection-assisted alignment.

Brief 011 Stage 3 step 2. v1's error analysis (tetrak,
``product/research/armenian-v1-error-analysis.md``) left one lever that
no amount of rendering can pull: a cluster of *shape* confusions on
degraded 1970s letterpress -- ``հ``→``խ``, ``խ``→``ի``, ``տ``→``ո``,
``ճ``↔``ջ``, concentrated on ``հ``, one of the commonest Armenian
letters. Synthetic fonts draw ``հ``'s ascender cleanly and the
encyclopedia's printing does not, so the model has never seen the shape
it keeps getting wrong. Real crops are the fix.

The method, as the brief specifies it: run the stock CRAFT detector over
pages whose transcripts were proofread by humans, align its boxes against
those transcripts, and emit the boxes it can place with confidence as
labelled crops. The label always comes from the transcript; the
recogniser's own reading is kept only as provenance and as the input to
the confusion table. See :mod:`tetrak_hy_trainer.align` for the alignment
itself and why it fails closed -- a mislabelled crop teaches the wrong
shape, so precision beats recall throughout.

**Volume 2 is refused.** Every published figure for this model is
measured on ten of its pages, and it is the one directory in this repo
with scans already sitting in it. See :mod:`tetrak_hy_trainer.heldout`.

Input is one or more harvest directories that have page images. The
volumes taken for v0 and v1 were harvested text-only -- synthesis needed
corpus text and nothing else -- so top them up first (this re-fetches no
wikitext)::

    python -m tetrak_hy_trainer.harvest \\
        --index "Ինդեքս:… (Soviet Armenian Encyclopedia) 1.djvu" \\
        --out runs/v0/harvest --images --image-width 3840 --limit 60

3840 px matches the evaluation scans, so the crops carry the same detail
per character as the material the model is scored on.

Output is a trainer-format dataset plus its audit trail::

    <out>/real_train/{labels.csv, *.png}
    <out>/real_val/{labels.csv, *.png}   held-out crops, split by page
    <out>/crops.csv                      every crop: tier, label, reading, box
    <out>/summary.json                   settings, provenance, yield

The split is by **page**, not by crop: crops from one page share its
paper, ink and scanning, so splitting within a page would let the
validation set measure memorisation.

Prerequisites: easyocr and torch. The trainer's own venv has neither
(they are the ``[train]`` extra), so the sibling library's venv is the
convenient one::

    ../tetrak-easyocr-armenian/.venv/bin/python scripts/harvest_real_crops.py \\
        --harvest-dir runs/v1/harvest-vol5 runs/v1/harvest-vol6 \\
        --bundle runs/v2/bundle --out runs/v2/all_data

Pass every source in **one** run. A second run into the same ``--out``
truncates the first's ``labels.csv``. Page numbers repeat between
sources, so crops are named ``<harvest-dir>_<page>_<n>.png`` -- for the
directory, not the volume number, because eight of brief 012's works
have no volume number and would otherwise all be ``v0``. See
:func:`load_pages`.

Writing into the v2 run's ``all_data/`` puts ``real_train`` beside
``syn_train``, which is what lets ``finetune_real.py`` mix the two in one
batch -- the vendored trainer takes a single dataset root.

Read the model bundle you pass carefully: it decides which words align.
A better recogniser matches more of the transcript, so harvest with the
newest weights even though the fine-tune starts from them too.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import align, charset, heldout, synth, wikisource  # noqa: E402
from tetrak_hy_trainer.align import Detection, Tier  # noqa: E402

# Boxes shorter than this are detector noise -- rules, speckle, the edge of
# a figure -- not type, and resizing them to imgH would be inventing detail.
MIN_BOX_HEIGHT = 12

# ...and a box more than this many times the page's median box height is a
# detector blob that swallowed several lines. One turned up in the first
# 389 crops harvested: a five-line block labelled "(1961)", because that
# was the only part of it the alignment could place. Training on it would
# teach the model to read a paragraph as one short string.
MAX_BOX_HEIGHT_MULTIPLE = 2.0

# The vendored trainer refuses labels longer than batch_max_length, which
# v1 and v2 both set to 60. Filtering here keeps the dataset honest rather
# than letting the trainer drop rows silently at load time.
DEFAULT_MAX_LABEL_LENGTH = 60

# Dashes a line-broken word can end with in this material.
_TRAILING_DASHES = "-–—֊"

# Quotation marks the recogniser is fond of inventing around a reading;
# stripped before looking for a trailing dash.
_STRAY_QUOTES = "\"'«»"


def source_slug(harvest_dir: Path) -> str:
    """A filename-safe identifier for one harvest directory.

    Crops are named for their source, so this has to be unique across the
    run and safe to put in ``labels.csv``, which the vendored trainer
    parses by splitting on the first comma.
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", harvest_dir.name).strip("-")


def load_pages(harvest_dirs: list[Path]) -> tuple[list[str], list[dict]]:
    """Every page across *harvest_dirs* that has both a scan and a transcript.

    Several volumes in one run, rather than one run per volume, because
    the output is a single dataset: a second run writing into the same
    ``--out`` truncates the first's ``labels.csv``.

    Crop filenames drawn from page numbers alone would collide between
    sources, so each page carries an identifier its crops are named for.
    That identifier is the **harvest directory**, not the volume number.
    Volume was enough while every source was a numbered ASE volume; brief
    012 added eight works that have no volume number at all, so
    ``heldout.volume_of`` returns None for each and every one of them
    named its crops ``v0_<page>_<index>.png``. Baronian page 100, Faustus
    page 100 and Otyan page 100 then wrote the same file.

    That is not a cosmetic clash. The image is overwritten by whichever
    source is cut last, while every source still appends its own row to
    ``labels.csv`` -- so one image ends up carrying two or three
    contradictory labels, and the only way a CTC model can reduce loss
    across them is to emit something short and noncommittal. It cost the
    first v5 fine-tune 27% of its 56,608 crops.
    """
    index_titles: list[str] = []
    pages: list[dict] = []

    slugs = [source_slug(directory) for directory in harvest_dirs]
    duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
    if duplicates:
        raise ValueError(
            f"harvest directories collapse to the same crop-name prefix: {sorted(duplicates)}. "
            "Crops are named for their source directory, so two directories sharing a name "
            "would overwrite each other's crops and give one image several labels. "
            "Rename one, or pass them in separate runs with separate --out."
        )

    for harvest_dir in harvest_dirs:
        manifest_path = harvest_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_title = manifest["index"]

        # Before anything else, and against the manifest rather than the
        # directory name: a copied or renamed directory must not get past
        # this.
        heldout.assert_not_held_out(index_title, source=str(manifest_path))
        index_titles.append(index_title)

        volume = heldout.volume_of(index_title)
        for entry in manifest["pages"]:
            # Per-page hold-outs (brief 012's registry): an evaluation
            # slice inside an otherwise-trainable work never becomes crops.
            if heldout.page_is_held_out(index_title, entry["page_number"]):
                continue
            image = harvest_dir / "images" / f"{entry['page_number']}.jpg"
            text = harvest_dir / entry["text"]
            if image.exists() and text.exists():
                pages.append(
                    {
                        **entry,
                        "image_path": image,
                        "text_path": text,
                        "volume": volume,
                        "source": source_slug(harvest_dir),
                        "index": index_title,
                    }
                )

    pages.sort(key=lambda page: (page["source"], page["page_number"]))
    return index_titles, pages


def build_reader(bundle: Path, gpu: bool):
    """A stock EasyOCR reader wired to a tetrak_hy bundle.

    ``["en"]``, never ``["hy"]``: EasyOCR reads a per-language character
    file for each requested language and ships none for Armenian, and the
    setting is inert for a custom model anyway -- the decode filter is
    ``set(model charset) - set(lang_char)`` and ``lang_char`` carries the
    yaml's whole ``character_list``. Same quirk the library's ``reader()``
    hides and the baseline harness works around.
    """
    import easyocr

    return easyocr.Reader(
        ["en"],
        recog_network="tetrak_hy",
        user_network_directory=str(bundle),
        model_storage_directory=str(bundle),
        gpu=gpu,
        verbose=False,
    )


def detect_page(reader, image_path: Path) -> list[Detection]:
    """Every box on the page, with what the recogniser read in it."""
    detections = []
    for box, text, confidence in reader.readtext(str(image_path), detail=1, paragraph=False):
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        detections.append(
            Detection(
                text=text,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                confidence=float(confidence),
            )
        )
    return detections


def is_line_break_fragment(crop: align.AlignedCrop) -> bool:
    """Whether the crop's last word is half of a word broken across lines.

    A word hyphenated at the end of a line is one word to the transcript
    and a fragment to the detector, so the label claims more than the
    crop shows -- ``առըն-`` on the page against ``առըն`` in the label,
    which teaches the model to swallow the hyphen it can plainly see.
    Detected on the *reading* rather than the label, since the reading is
    what saw the page: a trailing dash the label does not share means the
    box stopped mid-word.

    This also drops the occasional good crop, where the recogniser merely
    misread a full stop as a dash. That is the trade this module makes
    everywhere: a wrong label costs more than a missing crop.
    """
    read = crop.predicted.strip().rstrip(_STRAY_QUOTES)
    return bool(read) and read[-1] in _TRAILING_DASHES and crop.label[-1] not in _TRAILING_DASHES


def usable(
    crop: align.AlignedCrop,
    allowed: set[str],
    max_label_length: int,
    median_height: float,
) -> bool:
    """Whether a crop can actually be trained on.

    Rejects labels carrying characters the model has no class for (the
    trainer would drop or mis-encode the row), labels past the encoder's
    length limit, boxes too small to hold type, detector blobs that
    swallowed several lines, and words broken across a line end.
    """
    if not crop.label or len(crop.label) > max_label_length:
        return False
    if not set(crop.label) <= allowed:
        return False
    if is_line_break_fragment(crop):
        return False
    left, top, right, bottom = crop.bbox
    height = bottom - top
    if height < MIN_BOX_HEIGHT or right <= left:
        return False
    return not (median_height and height > MAX_BOX_HEIGHT_MULTIPLE * median_height)


def cut(image, bbox, margin_fraction: float):
    """Crop *image* to *bbox* with a little air, in greyscale.

    The margin scales with the box height so it means the same thing at
    any scan resolution, and the mode matches the synthetic crops (the
    trainer runs ``rgb: False``, so colour would only be discarded later).
    """
    left, top, right, bottom = bbox
    margin = max(2, round((bottom - top) * margin_fraction))
    width, height = image.size
    window = (
        max(0, int(left) - margin),
        max(0, int(top) - margin),
        min(width, int(right) + margin),
        min(height, int(bottom) + margin),
    )
    return image.crop(window).convert("L")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harvest-dir",
        type=Path,
        required=True,
        nargs="+",
        help="one or more harvest directories holding images/ + text/; pass them "
        "together rather than in separate runs, which would truncate each "
        "other's labels.csv",
    )
    parser.add_argument(
        "--bundle", type=Path, required=True, help="tetrak_hy bundle to read the pages with"
    )
    parser.add_argument("--out", type=Path, required=True, help="dataset root to write into")
    parser.add_argument("--limit", type=int, default=None, help="stop after N pages")
    parser.add_argument("--val-every", type=int, default=10, help="every Nth page to validation")
    parser.add_argument("--min-similarity", type=float, default=align.DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--bracket-span", type=int, default=align.DEFAULT_BRACKET_SPAN)
    parser.add_argument(
        "--tiers",
        default="exact,near",
        help=(
            "which alignment tiers to emit (see tetrak_hy_trainer.align.Tier). "
            "'bracketed' is off by default: spot-checking the first pages harvested "
            "put its precision near half -- it collects line-break fragments and "
            "detector blobs along with the badly-misread words it is meant to catch. "
            "Add it only with the crops.csv review in hand"
        ),
    )
    parser.add_argument("--margin-fraction", type=float, default=0.1)
    parser.add_argument("--max-label-length", type=int, default=DEFAULT_MAX_LABEL_LENGTH)
    parser.add_argument("--gpu", action="store_true", help="let EasyOCR use a GPU")
    args = parser.parse_args()

    tiers = {Tier(name.strip()) for name in args.tiers.split(",") if name.strip()}
    allowed = set(charset.character_list())

    index_titles, pages = load_pages(args.harvest_dir)
    if not pages:
        raise SystemExit(
            f"no page has both an image and a transcript under "
            f"{[str(d) for d in args.harvest_dir]}. Harvest the scans first with "
            f"python -m tetrak_hy_trainer.harvest ... --images"
        )
    if args.limit is not None:
        pages = pages[: args.limit]
    print(f"{len(pages)} page(s) with scans from {len(index_titles)} volume(s)", flush=True)
    for title in index_titles:
        print(f"  {title}", flush=True)

    from PIL import Image

    reader = build_reader(args.bundle, gpu=args.gpu)

    folders = {
        "train": args.out / "real_train",
        "val": args.out / "real_val",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    rows = {"train": [], "val": []}
    review: list[dict] = []
    totals = dict.fromkeys(Tier, 0)
    started = time.time()

    for position, page in enumerate(pages):
        split = "val" if args.val_every and position % args.val_every == 0 else "train"
        detections = detect_page(reader, page["image_path"])
        truth = wikisource.normalise_transcript(page["text_path"].read_text(encoding="utf-8"))
        crops = align.align_page(
            detections,
            truth,
            min_similarity=args.min_similarity,
            bracket_span=args.bracket_span,
        )

        # The blob test is relative to this page: scans vary in resolution
        # between volumes (volume 1 is 1920 px wide where 2-6 are 3840), so
        # a height in pixels means nothing on its own.
        heights = [crop.bbox[3] - crop.bbox[1] for crop in crops]
        median_height = statistics.median(heights) if heights else 0.0

        image = Image.open(page["image_path"])
        kept = 0
        for index, crop in enumerate(crops):
            if crop.tier not in tiers:
                continue
            if not usable(crop, allowed, args.max_label_length, median_height):
                continue
            filename = f"{page['source']}_{page['page_number']:04d}_{index:04d}.png"
            cut(image, crop.bbox, args.margin_fraction).save(folders[split] / filename)
            rows[split].append((filename, crop.label))
            totals[crop.tier] += 1
            kept += 1
            review.append(
                {
                    "source": page["source"],
                    "index": page["index"],
                    "volume": page["volume"],
                    "page": page["page_number"],
                    "split": split,
                    "file": filename,
                    "tier": crop.tier.value,
                    "label": crop.label,
                    "predicted": crop.predicted,
                    "confidence": f"{crop.confidence:.4f}" if crop.confidence else "",
                    "bbox": " ".join(str(round(value)) for value in crop.bbox),
                }
            )
        image.close()

        counts = align.tier_counts(crops)
        print(
            f"  {page['source']} p{page['page_number']} [{split}] "
            f"{len(detections)} boxes -> "
            f"{kept} crops (exact {counts[Tier.EXACT]}, near {counts[Tier.NEAR]}, "
            f"bracketed {counts[Tier.BRACKETED]})",
            flush=True,
        )

    for split, folder in folders.items():
        # Not csv.writer: it quotes labels containing a comma, and the
        # trainer's regex split keeps the quotation marks. See
        # synth.write_labels -- this cost v1 21% of its crops.
        synth.write_labels(folder, rows[split])

    with (args.out / "crops.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "index",
                "volume",
                "page",
                "split",
                "file",
                "tier",
                "label",
                "predicted",
                "confidence",
                "bbox",
            ],
        )
        writer.writeheader()
        writer.writerows(review)

    summary = {
        "indexes": index_titles,
        "harvest_dirs": [str(d) for d in args.harvest_dir],
        "bundle": str(args.bundle),
        "pages": [f"{page['source']}/{page['page_number']}" for page in pages],
        "revids": {f"{page['source']}/{page['page_number']}": page.get("revid") for page in pages},
        "settings": {
            "min_similarity": args.min_similarity,
            "bracket_span": args.bracket_span,
            "tiers": sorted(tier.value for tier in tiers),
            "margin_fraction": args.margin_fraction,
            "max_label_length": args.max_label_length,
            "val_every": args.val_every,
        },
        "counts": {
            "train": len(rows["train"]),
            "val": len(rows["val"]),
            **{tier.value: count for tier, count in totals.items()},
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    total = len(rows["train"]) + len(rows["val"])
    print(
        f"\n{total} crops ({len(rows['train'])} train, {len(rows['val'])} val) "
        f"in {time.time() - started:.0f}s -> {args.out}",
        flush=True,
    )
    for tier, count in totals.items():
        print(f"  {tier.value}: {count}", flush=True)
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
