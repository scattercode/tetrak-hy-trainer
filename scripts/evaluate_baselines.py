"""Baseline every relevant OCR backend against a harvested page/transcript set.

Scores each backend's transcript of harvested (scan, human transcript)
pairs — see `tetrak_hy_trainer.harvest` — using Tetrak's metrics
(character similarity, order-sensitive; word recall, order-insensitive).
Produced the numbers in Tetrak's Armenian OCR benchmarks research note
(2026-08-29: 10 validated Armenian Soviet Encyclopedia pages).

Prerequisites: Tetrak's venv (the backends and metrics import from
tetrak_ocr, and the heavy engines need their extras); run from the Tetrak
repo root so the Claude backend finds its .env. Tesseract language
variants beyond the installed tessdata are selected by exporting
TESSDATA_PREFIX at a directory holding the .traineddata files.

Known systematic penalty, identical for every backend: the scans include
running headers/page numbers that the transcripts (correctly) omit, so
character similarity carries a small insertion penalty and word recall
none.

Run:
    python scripts/evaluate_baselines.py <harvest-dir>
"""

import csv
import json
import sys
import time
from pathlib import Path

from tetrak_ocr.accuracy import character_similarity, word_recall

EVAL_DIR = Path(sys.argv[1])
OUT_CSV = EVAL_DIR / "baselines.csv"

manifest = json.loads((EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
pages = [
    (
        EVAL_DIR / "images" / f"{entry['page_number']}.jpg",
        (EVAL_DIR / entry["text"]).read_text(encoding="utf-8"),
        entry["page_number"],
    )
    for entry in manifest["pages"]
]
print(f"{len(pages)} pages", flush=True)


def tesseract_eng(path):
    from tetrak_ocr.backends.tesseract import ocr_image

    return ocr_image(path)


def tesseract_hye(path):
    from tetrak_ocr.backends.tesseract import ocr_image

    return ocr_image(path, lang="hye")


def tesseract_hye_auto(path):
    from tetrak_ocr.backends.tesseract import ocr_image

    return ocr_image(path, lang="hye", auto=True)


def vision(path):
    from tetrak_ocr.backends.vision import ocr_image

    return ocr_image(path)


def easyocr_stock(path):
    from tetrak_ocr.backends.easyocr import ocr_image

    return ocr_image(path)


def paddle(path):
    from tetrak_ocr.backends.paddle import ocr_image

    return ocr_image(path)


def marker(path):
    from tetrak_ocr.backends.marker import ocr_image

    return ocr_image(path)


def claude(path):
    from tetrak_ocr.backends.claude import ocr_image

    return ocr_image(path)


BACKENDS = [
    ("tesseract-eng", tesseract_eng),
    ("tesseract-hye", tesseract_hye),
    ("tesseract-hye-auto", tesseract_hye_auto),
    ("vision", vision),
    ("easyocr", easyocr_stock),
    ("paddle", paddle),
    ("marker", marker),
    ("claude", claude),
]

rows = []
for name, fn in BACKENDS:
    sims, recs, secs = [], [], 0.0
    for image, expected, number in pages:
        started = time.perf_counter()
        try:
            text = fn(image)
        except Exception as exc:
            print(f"RESULT {name} p{number} FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append([name, number, "", "", ""])
            continue
        elapsed = time.perf_counter() - started
        sim = character_similarity(text, expected)
        rec = word_recall(text, expected)
        sims.append(sim)
        recs.append(rec)
        secs += elapsed
        rows.append([name, number, f"{sim:.4f}", f"{rec:.4f}", f"{elapsed:.1f}"])
        print(f"RESULT {name} p{number} chr={sim:.3f} wrd={rec:.3f} {elapsed:.0f}s", flush=True)
    if sims:
        print(
            f"AVERAGE {name} chr={sum(sims) / len(sims):.4f} "
            f"wrd={sum(recs) / len(recs):.4f} total={secs:.0f}s n={len(sims)}",
            flush=True,
        )

with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["backend", "page", "char_sim", "word_recall", "seconds"])
    writer.writerows(rows)
print(f"DONE wrote {OUT_CSV}", flush=True)
