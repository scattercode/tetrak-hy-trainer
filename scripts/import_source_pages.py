#!/usr/bin/env python3
"""Turn downloaded page scans into an evaluation set the scorers can read.

Brief 014 Stage 0.5. The evaluation fixtures that matter most for classical
orthography are not on Wikisource -- Մշակ, the Tiflis daily of 1872-1921,
comes from the Armenian National Digital Library (arar.sci.am) as one PDF
per issue -- so there is no harvest to build a set from. This renders every
page of every PDF under ``runs/sources/<work>/`` into
``runs/eval/<work>/images/<page>.jpg`` and writes two records:

- ``pages.json``: every rendered page with its provenance (PDF, page within
  it, issue and edition id from ``selection.tsv`` when present, pixel size).
- ``manifest.json``: the harvester's shape (``index``, ``pages`` with
  ``page_number``, ``text``, ``image``), listing **only the pages that have
  a transcript** at ``text/<page>.txt``. Scorers read the manifest, so an
  untranscribed page is simply not evaluated yet. Re-run after adding a
  transcript and the manifest grows.

Page numbers are integers, as every scorer expects: ``<year><page>`` with
the page zero-padded to two digits, so 1882's page 3 is ``188203``. Rendering
is at ``--dpi`` (default 300, the scans' native resolution; the 100 dpi
issues are upsampled rather than left smaller, so a crop carries the same
pixels per line whatever the issue).

Transcripts say what the page prints, in reading order down each column and
across the columns, with line-end hyphenation joined and the masthead left
out, as the Wikisource transcripts do.

Prerequisites: poppler's ``pdftoppm`` and ``pdfinfo`` on PATH, this repo's
plain ``.venv``. Run from the repo root:

    .venv/bin/python scripts/import_source_pages.py mshak
    .venv/bin/python scripts/import_source_pages.py mshak --list
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = REPO / "runs" / "sources"
EVAL = REPO / "runs" / "eval"

_YEAR = re.compile(r"(\d{4})")


def page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if not match:
        raise RuntimeError(f"pdfinfo gave no page count for {pdf}")
    return int(match.group(1))


def render_page(pdf: Path, page: int, out: Path, dpi: int) -> None:
    """One page of *pdf* to *out* (JPEG), via pdftoppm."""
    stem = out.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-jpeg",
            "-r",
            str(dpi),
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            str(pdf),
            str(stem),
        ],
        check=True,
    )
    produced = stem.with_suffix(".jpg")
    if produced != out:
        produced.rename(out)


def selection(source_dir: Path) -> dict[str, dict]:
    """Issue metadata keyed by edition id, from selection.tsv if present."""
    path = source_dir / "selection.tsv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["edition_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def page_number_for(pdf: Path, page: int) -> int:
    match = _YEAR.search(pdf.stem)
    if not match:
        raise RuntimeError(f"no four-digit year in {pdf.name}; name the PDF <work>-<year>-<id>.pdf")
    return int(match.group(1)) * 100 + page


def import_work(work: str, dpi: int, index_title: str | None) -> list[dict]:
    source_dir = SOURCES / work
    eval_dir = EVAL / work
    images = eval_dir / "images"
    texts = eval_dir / "text"
    images.mkdir(parents=True, exist_ok=True)
    texts.mkdir(exist_ok=True)
    issues = selection(source_dir)

    pages: list[dict] = []
    for pdf in sorted(source_dir.glob("*.pdf")):
        edition = pdf.stem.rsplit("-", 1)[-1]
        issue = issues.get(edition, {})
        for page in range(1, page_count(pdf) + 1):
            number = page_number_for(pdf, page)
            image = images / f"{number}.jpg"
            if not image.exists():
                render_page(pdf, page, image, dpi)
                print(f"  rendered {pdf.name} p{page} -> {image.name}", flush=True)
            pages.append(
                {
                    "page_number": number,
                    "pdf": pdf.name,
                    "pdf_page": page,
                    "edition_id": edition,
                    "issue": issue.get("issue", ""),
                    "url": issue.get("url", ""),
                    "image": str(image.relative_to(eval_dir)),
                    "text": str((texts / f"{number}.txt").relative_to(eval_dir)),
                    "transcribed": (texts / f"{number}.txt").exists(),
                }
            )

    (eval_dir / "pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    manifest = {
        "index": index_title or f"{work} (runs/sources/{work})",
        "source": f"runs/sources/{work}",
        "pages": [
            {k: p[k] for k in ("page_number", "pdf", "pdf_page", "issue", "text", "image")}
            for p in pages
            if p["transcribed"]
        ],
    }
    (eval_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", help="directory name under runs/sources/ and runs/eval/")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--index", default=None, help="the manifest's index title")
    parser.add_argument("--list", action="store_true", help="print page status after importing")
    args = parser.parse_args()
    pages = import_work(args.work, args.dpi, args.index)
    done = sum(p["transcribed"] for p in pages)
    print(f"{len(pages)} pages rendered under runs/eval/{args.work}/images; {done} transcribed")
    if args.list:
        for p in pages:
            mark = "text" if p["transcribed"] else "    "
            print(f"  {p['page_number']}  {mark}  {p['pdf']} p{p['pdf_page']}  {p['issue']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
