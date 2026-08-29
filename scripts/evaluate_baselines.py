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
# A filtered run must not clobber the full table: it writes its own file,
# named for the selection.
_only_arg = sys.argv[2] if len(sys.argv) > 2 else None
OUT_CSV = EVAL_DIR / (
    f"baselines_{_only_arg.replace(',', '_')}.csv" if _only_arg else "baselines.csv"
)

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


def hye_paddle(path):
    """Calfa's paddle-calfa-tiny recognition model in a stock PaddleOCR
    pipeline. Requires HYE_PADDLE_DIR pointing at the model's inference/
    directory (huggingface.co/calfa-ai/hye-paddle, CC BY-NC 4.0 -- measure,
    never ship). Text is joined exactly as Tetrak's paddle backend joins its
    own, so the two rows are comparable."""
    import os

    from paddleocr import PaddleOCR

    global _hye_paddle_ocr
    if "_hye_paddle_ocr" not in globals():
        _hye_paddle_ocr = PaddleOCR(
            text_recognition_model_name="PP-OCRv6_tiny_rec",
            text_recognition_model_dir=os.environ["HYE_PADDLE_DIR"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    results = _hye_paddle_ocr.predict(str(path))
    texts = []
    for page in results:
        if isinstance(page, dict) and "rec_texts" in page:
            texts.extend(page["rec_texts"])
    return "\n".join(texts)


BACKENDS = [
    ("tesseract-eng", tesseract_eng),
    ("tesseract-hye", tesseract_hye),
    ("tesseract-hye-auto", tesseract_hye_auto),
    ("vision", vision),
    ("easyocr", easyocr_stock),
    ("paddle", paddle),
    ("marker", marker),
    ("claude", claude),
    ("hye-paddle", hye_paddle),
]

only = sys.argv[2].split(",") if len(sys.argv) > 2 else None

rows = []
for name, fn in BACKENDS:
    if only is not None and name not in only:
        continue
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
