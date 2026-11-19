#!/usr/bin/env python3
"""Measure the transcript screen on the evaluation pages, to set its limit.

The crop harvester leaves out a page whose transcript has too many words
our reading disputes (:mod:`tetrak_hy_trainer.transcript_check`). "Too
many" is a number, and this script is where it comes from. It reads every
evaluation page exactly as the harvester would -- the same detector and
recogniser, the same reading order, stock Tesseract as the second reader --
and prints the suspect rate per page beside an independent estimate of how
wrong the transcript is: the words where our reading and a saved
``hye-paddle`` reading agree with each other against the transcript. Two
unrelated engines agreeing is the test that was checked by hand on
Abovyan's page 232 (brief 023), where all seventeen such words were
transcript errors.

Evaluation pages are only *read* here, which is what evaluation does. No
crop is cut and nothing is written under ``runs/eval``. The hye-paddle
readings are Calfa's, CC BY-NC: they are a yardstick for this measurement
and never an input to the screen itself.

``--alternate register/page=path`` scores a second transcript for one page,
for example the page as it stood before it was corrected.

The same reading works on training harvests, to see what the screen would
leave out before a harvest is run: point ``--eval-root`` at ``runs/harvest``
and pass ``--training-pages``, which skips every held-out page. ``--limit``
caps the pages read per directory. Each row keeps the reading and the
second reader's words, so a change to the rule can be tried on the saved
file without reading the pages again.

Prerequisites: easyocr and torch (the sibling library's venv, as for
``harvest_real_crops.py``), the ``tesseract`` command with its stock
``hye`` model, and the word lists of the run. About six seconds a page.

Run:
    ../tetrak-easyocr-armenian/.venv/bin/python scripts/calibrate_transcript_screen.py \\
        --bundle runs/v7-b023-italic/bundle \\
        --wordlist runs/v7-b014/wordlist.tsv.gz \\
        --wordlist runs/v7-b014/wordlist-classical.tsv.gz \\
        --out runs/transcript-screen/calibration.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import heldout, transcript_check, wikisource  # noqa: E402
from tetrak_hy_trainer.transcript_check import aligned_words, letters  # noqa: E402

REFERENCE_ENGINE = "hye-paddle"


def load_harvester():
    spec = importlib.util.spec_from_file_location(
        "harvest_real_crops", REPO / "scripts" / "harvest_real_crops.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def two_engines_against(transcript: str, ours: str, theirs: str) -> int:
    """Words where both readings have the same other word in the transcript's place."""
    words = [word for word in transcript.split() if letters(word)]
    first = {id(t): r for t, r in aligned_words(words, ours.split())}
    second = {id(t): r for t, r in aligned_words(words, theirs.split())}
    return sum(
        1
        for word in words
        if id(word) in first
        and id(word) in second
        and letters(first[id(word)]) == letters(second[id(word)]) != letters(word)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wordlist", type=Path, action="append", required=True)
    parser.add_argument("--eval-root", type=Path, default=REPO / "runs" / "eval")
    parser.add_argument("--register", action="append", help="only this register; repeatable")
    parser.add_argument("--alternate", action="append", default=[], metavar="REGISTER/PAGE=PATH")
    parser.add_argument("--out", type=Path, required=True, help="where to write every page's row")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--training-pages",
        action="store_true",
        help="the root holds training harvests: skip every held-out page",
    )
    parser.add_argument("--limit", type=int, default=None, help="pages to read per directory")
    args = parser.parse_args()

    if not transcript_check.tesseract_available():
        raise SystemExit("the tesseract command is needed as the second reader")
    frequency = transcript_check.load_frequency(args.wordlist)
    harvester = load_harvester()
    reader = harvester.build_reader(args.bundle, gpu=args.gpu)
    alternates = dict(item.split("=", 1) for item in args.alternate)

    rows = []
    for manifest_path in sorted(args.eval_root.glob("*/manifest.json")):
        register = manifest_path.parent.name
        if args.register and register not in args.register:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_title = manifest.get("index", "")
        if args.training_pages and index_title and heldout.is_held_out(index_title):
            continue
        read = 0
        for entry in manifest["pages"]:
            page = entry["page_number"]
            if args.training_pages and heldout.page_is_held_out(index_title, page):
                continue
            if args.limit is not None and read >= args.limit:
                break
            image = manifest_path.parent / "images" / f"{page}.jpg"
            text = manifest_path.parent / entry["text"]
            if not (image.exists() and text.exists()):
                continue
            read += 1
            reading = harvester.reading_of(harvester.detect_page(reader, image))
            second = transcript_check.tesseract_words(image)
            reference_path = manifest_path.parent / "readings" / REFERENCE_ENGINE / f"{page}.txt"
            reference = (
                reference_path.read_text(encoding="utf-8") if reference_path.exists() else None
            )
            versions = [("", text)]
            if f"{register}/{page}" in alternates:
                versions.append((" (alternate)", Path(alternates[f"{register}/{page}"])))
            for label, path in versions:
                transcript = wikisource.normalise_transcript(
                    path.read_text(encoding="utf-8"), index_title
                )
                check = transcript_check.check_page(transcript, reading, second, frequency)
                agreed = (
                    two_engines_against(transcript, reading, reference)
                    if reference is not None
                    else None
                )
                row = {
                    "register": register,
                    "page": f"{page}{label}",
                    "quality": entry.get("quality"),
                    "words": check.words,
                    "suspects": len(check.suspects),
                    "per_hundred": round(check.per_hundred, 2),
                    "two_engines_per_hundred": (
                        round(100 * agreed / check.words, 2)
                        if agreed is not None and check.words
                        else None
                    ),
                    "examples": [f"{s.transcript}/{s.reader}" for s in check.suspects[:10]],
                    "transcript_path": str(path),
                    "reading": reading,
                    "second_reader": second,
                    "reference": reference,
                }
                rows.append(row)
                print(
                    f"{register:30} p{row['page']:<16} words {row['words']:5} "
                    f"screen {row['per_hundred']:5.2f}  two engines "
                    f"{row['two_engines_per_hundred']}",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    scored = [r for r in rows if r["two_engines_per_hundred"] is not None]
    if len(scored) > 2:
        xs = [r["per_hundred"] for r in scored]
        ys = [r["two_engines_per_hundred"] for r in scored]
        bad = [r for r in scored if r["two_engines_per_hundred"] >= 2.0]
        clean = [r for r in scored if r["two_engines_per_hundred"] < 1.0]
        print(f"\n{len(scored)} pages; correlation {statistics.correlation(xs, ys):.2f}")
        print("limit  left out  of the bad  of the clean   (bad: two engines >= 2, clean: < 1)")
        for limit in (0.75, 1.0, 1.25, 1.5, 2.0):
            print(
                f"{limit:5.2f}  {sum(r['per_hundred'] >= limit for r in scored):8}  "
                f"{sum(r['per_hundred'] >= limit for r in bad):4}/{len(bad):<5}  "
                f"{sum(r['per_hundred'] >= limit for r in clean):4}/{len(clean)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
