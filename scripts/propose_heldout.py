#!/usr/bin/env python3
"""Propose held-out evaluation pages for a newly harvested work.

Brief 014 Stage 0.4, and the rule brief 012 applied by hand: a work's
evaluation pages are chosen by body-text density **before** anything trains
on it, so the registry in ``heldout.WORK_PAGES`` is fixed while the model is
still ignorant of the work. This makes the choice reproducible instead of a
judgement call: it ranks the harvest's pages by Armenian token count, drops
the front and back matter (the first and last tenth), and takes pages from
the denser half spread evenly through the work, so a set covers the whole
volume rather than one dense chapter.

Pages with any Cyrillic are skipped unless ``--allow-cyrillic`` is given, so
the Armenian figure for a classical register is not pulled down by a Russian
quotation the way Tumanyan's is.

It prints the ``WORK_PAGES`` entry to paste and the needle (a substring of
the index title) it would key on. It changes nothing: registering the pages
is a deliberate edit to ``heldout.py``, with its tests.

Run from the repo root, with the plain ``.venv``:

    .venv/bin/python scripts/propose_heldout.py runs/harvest/khatisian-memoirs
    .venv/bin/python scripts/propose_heldout.py runs/harvest/toranian-* --pages 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import heldout  # noqa: E402

ARMENIAN_TOKEN = re.compile(r"\S*[Ա-Ֆա-և]\S*")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def density(text: str) -> int:
    return len(ARMENIAN_TOKEN.findall(text))


def propose(harvest_dir: Path, count: int, allow_cyrillic: bool) -> tuple[str, list[int]]:
    manifest = json.loads((harvest_dir / "manifest.json").read_text(encoding="utf-8"))
    index_title = manifest["index"]
    scored = []
    for entry in manifest["pages"]:
        text = (harvest_dir / entry["text"]).read_text(encoding="utf-8")
        if not allow_cyrillic and CYRILLIC.search(text):
            continue
        scored.append((entry["page_number"], density(text)))
    scored.sort()
    if len(scored) < count * 2:
        raise SystemExit(f"{harvest_dir}: only {len(scored)} eligible pages; need {count * 2}")
    margin = len(scored) // 10
    body = scored[margin : len(scored) - margin] or scored
    threshold = sorted(d for _, d in body)[len(body) // 2]
    dense = [page for page, d in body if d >= threshold]
    step = max(1, len(dense) // count)
    picks = dense[::step][:count]
    return index_title, sorted(picks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harvests", nargs="+", type=Path, help="harvest directories")
    parser.add_argument("--pages", type=int, default=10, help="pages per work")
    parser.add_argument("--allow-cyrillic", action="store_true")
    args = parser.parse_args()
    for harvest_dir in args.harvests:
        index_title, picks = propose(harvest_dir, args.pages, args.allow_cyrillic)
        if heldout.held_out_pages(index_title) is not None:
            print(f"# {harvest_dir.name}: already registered -- {index_title}")
            continue
        needle = index_title.removeprefix("Ինդեքս:").removesuffix(".djvu").removesuffix(".pdf")
        print(f"# {harvest_dir.name}: {index_title}")
        print(f'    "{needle}": frozenset({{{", ".join(map(str, picks))}}}),')
    return 0


if __name__ == "__main__":
    sys.exit(main())
