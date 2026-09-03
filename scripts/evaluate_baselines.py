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


def hye_calfa_n(path):
    """Calfa's Armenian Tesseract model (CC BY-NC 4.0 -- measure, never ship).

    A first-class row rather than an environment trick. It was previously
    run by pointing TESSDATA_PREFIX at a directory holding
    hye-calfa-n.traineddata and passing lang="hye", which only works if
    the file is renamed, and leaves no record in the table of which model
    a "tesseract-hye" row actually used. Brief 012 requires this bar on
    every set published, so it needs a name of its own.

    Set TESSDATA_PREFIX to the directory holding hye-calfa-n.traineddata.
    """
    from tetrak_ocr.backends.tesseract import ocr_image

    return ocr_image(path, lang="hye-calfa-n")


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


def tetrak_hy_v0(path):
    """Our own model: a tetrak_hy bundle named by TETRAK_HY_BUNDLE, loaded
    through stock EasyOCR and joined line-by-line exactly as Tetrak's
    easyocr backend joins its output, so the rows are comparable."""
    import os

    import easyocr

    global _tetrak_hy_reader
    if "_tetrak_hy_reader" not in globals():
        bundle = os.environ["TETRAK_HY_BUNDLE"]
        _tetrak_hy_reader = easyocr.Reader(
            ["en"],  # no hy_char.txt ships with EasyOCR; inert for custom models
            recog_network="tetrak_hy",
            user_network_directory=bundle,
            model_storage_directory=bundle,
            verbose=False,
        )
    return "\n".join(_tetrak_hy_reader.readtext(str(path), detail=0, paragraph=False))


BACKENDS = [
    ("tesseract-eng", tesseract_eng),
    ("tesseract-hye", tesseract_hye),
    ("tesseract-hye-auto", tesseract_hye_auto),
    ("hye-calfa-n", hye_calfa_n),
    ("vision", vision),
    ("easyocr", easyocr_stock),
    ("paddle", paddle),
    ("marker", marker),
    ("claude", claude),
    ("hye-paddle", hye_paddle),
    ("tetrak-hy-v0", tetrak_hy_v0),
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
