#!/usr/bin/env python3
"""Build a per-register evaluation set for every work with held-out pages.

Until brief 012 the project had exactly one evaluation set — ten pages of
volume 2 of the Armenian Soviet Encyclopedia — and every published figure
came from it. That set answers one question well: how does the model read
*this* encyclopedia, in *this* face, at *this* scan width. It cannot say
whether a gain came from the model learning Armenian better or from it
learning the ASE better, and once brief 012 widened the corpus to Western
Armenian literature, a bilingual dictionary, a scholarly history and a
second encyclopedia, that distinction is the whole question.

So each work in :data:`tetrak_hy_trainer.heldout.WORK_PAGES` gets an
evaluation set of its own, built from the pages reserved for it before
anything trained on that work. The output matches the shape
``evaluate_baselines.py`` and ``train_synthetic.py --eval-dir`` already
expect, because it *is* a harvest — ``manifest.json``, ``text/<page>.txt``
and ``images/<page>.jpg`` — just one restricted to the held-out pages:

    runs/eval/<work>/

Which works get a set is not configured here. The script walks the
existing harvests under ``runs/harvest/`` and asks ``heldout`` whether
each one's index title has pages reserved; a work gains an evaluation set
by being added to ``WORK_PAGES``, never by being named in this file. That
keeps one registry authoritative for both halves of the split — the pages
this script downloads are exactly the pages the samplers refuse.

The ASE is deliberately absent: volume 2 is held out whole and its set
already exists at ``runs/eval/ase-vol2/``, which is the baseline every
published figure is measured against and must not be rebuilt.

**Scans come at 3840 px**, matching both the ASE evaluation pages and the
training harvests, so a crop carries the same detail per character as the
material the model saw.

Prerequisites: this repository's plain ``.venv`` (network and the standard
library; no torch). Harvesting is polite — one shared session, a pause
between requests — so a full build is a few minutes of mostly waiting.

    .venv/bin/python3 scripts/build_eval_sets.py            # build all
    .venv/bin/python3 scripts/build_eval_sets.py --dry-run  # just plan it
    .venv/bin/python3 scripts/build_eval_sets.py --work faustus-1968
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tetrak_hy_trainer import heldout
from tetrak_hy_trainer.harvest import harvest
from tetrak_hy_trainer.wikisource import WikisourceClient

REPO_ROOT = Path(__file__).resolve().parent.parent
HARVEST_ROOT = REPO_ROOT / "runs" / "harvest"
EVAL_ROOT = REPO_ROOT / "runs" / "eval"

# The ASE evaluation set predates the registry and is the baseline for
# every published figure. Rebuilding it would silently re-baseline the
# whole project, so the walk skips anything writing here.
PROTECTED = {"ase-vol2"}

# Matches the ASE evaluation pages and the training harvests.
IMAGE_WIDTH = 3840


def planned_sets(harvest_root: Path = HARVEST_ROOT) -> list[tuple[str, str, frozenset[int]]]:
    """Every ``(name, index_title, pages)`` an evaluation set is due for.

    Reads each harvest's manifest for its index title -- the same field
    the held-out guards key on -- and keeps the works the registry
    reserves pages for.
    """
    plans = []
    for manifest_path in sorted(harvest_root.glob("*/manifest.json")):
        name = manifest_path.parent.name
        if name in PROTECTED:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_title = manifest.get("index")
        if not index_title:
            continue
        # A whole-volume hold-out has no page list; its set is the
        # pre-existing ase-vol2 one, which PROTECTED already covers.
        if heldout.is_held_out(index_title):
            continue
        pages = heldout.held_out_pages(index_title)
        if pages:
            plans.append((name, index_title, pages))
    return plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", action="append", help="only this harvest name; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--image-width", type=int, default=IMAGE_WIDTH)
    args = parser.parse_args(argv)

    plans = planned_sets()
    if args.work:
        wanted = set(args.work)
        unknown = wanted - {name for name, _, _ in plans}
        if unknown:
            parser.error(f"no held-out pages registered for: {', '.join(sorted(unknown))}")
        plans = [plan for plan in plans if plan[0] in wanted]

    if not plans:
        print("nothing to build: no harvest has held-out pages registered", file=sys.stderr)
        return 1

    for name, _, pages in plans:
        print(f"{name:24} {len(pages):>3} pages  {sorted(pages)}")
    if args.dry_run:
        return 0

    client = WikisourceClient()
    for name, index_title, pages in plans:
        out_dir = EVAL_ROOT / name
        print(f"\n=== {name} -> {out_dir} ===")
        manifest = harvest(
            client,
            index_title=index_title,
            out_dir=out_dir,
            images=True,
            image_width=args.image_width,
            pages=set(pages),
        )
        missing = sorted(set(pages) - {entry["page_number"] for entry in manifest})
        print(f"harvested {len(manifest)}/{len(pages)} page(s) into {out_dir}")
        if missing:
            # A reserved page that will not download is a hole in the
            # register's evaluation, not a detail: say so loudly rather
            # than quietly scoring on a smaller set than the registry
            # claims.
            print(f"  MISSING {len(missing)}: {missing} -- below quality, or renumbered upstream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
