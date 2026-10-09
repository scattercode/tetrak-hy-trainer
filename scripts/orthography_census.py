#!/usr/bin/env python3
"""Classify every well-covered Wikisource index as classical or reformed.

Brief 014 Stage 0.1. The census (``wikisource_census.py``) says how many
proofread pages each index has; it cannot say which orthography they are
in, and titles mislead -- Kajaznuni's classical *Ազգ և հայրենիք* is titled
with a reformed ``և``, and the Soviet Yerevan editions of Western authors
are reset in reformed spelling whatever the author wrote. This samples a
few proofread pages from each index with enough of them, cleans the text
as a harvest would, and lets :mod:`tetrak_hy_trainer.orthography` read it.

Progress is cached per index into ``runs/census/orthography.json`` as it
goes, so an interrupted run resumes. ``--report`` prints the classical and
mixed indexes as a markdown table for the research note, with the page
counts and native resolution from the census cache.

Run from the repo root, with the plain ``.venv``:

    .venv/bin/python scripts/orthography_census.py             # the crawl
    .venv/bin/python scripts/orthography_census.py --report    # from the cache
    .venv/bin/python scripts/orthography_census.py --min-pages 20 --samples 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import orthography  # noqa: E402
from tetrak_hy_trainer.wikisource import WikisourceClient, clean_wikitext  # noqa: E402

CENSUS = REPO / "runs" / "census" / "census.json"
OUTPUT = REPO / "runs" / "census" / "orthography.json"


def sample_pages(records: list, samples: int) -> list:
    """*samples* pages spread through the work, skipping the front matter.

    The first tenth is title pages, prefaces and tables of contents, which
    are short and often in another hand; the body is what the model will
    meet. Evenly spaced picks after that cover a work whose orthography
    changes part-way (an anthology, a journal issue) rather than one run.
    """
    if not records:
        return []
    start = len(records) // 10
    body = records[start:] or records
    step = max(1, len(body) // samples)
    return body[::step][:samples]


def classify_index(client: WikisourceClient, title: str, samples: int) -> dict:
    records = list(client.pages_in_index(title))
    picks = sample_pages(records, samples)
    text = "\n".join(clean_wikitext(client.page_wikitext(r.title)[0], title) for r in picks)
    entry = {"title": title, "sampled_pages": [r.page_number for r in picks]}
    entry.update(orthography.markers(text).as_dict())
    return entry


def crawl(min_pages: int, samples: int) -> None:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    targets = [e["title"] for e in census if (e.get("proofread") or 0) >= min_pages]
    done: dict[str, dict] = {}
    if OUTPUT.exists():
        done = {e["title"]: e for e in json.loads(OUTPUT.read_text(encoding="utf-8"))}
    print(f"{len(targets)} indexes with >= {min_pages} proofread pages; {len(done)} cached")

    client = WikisourceClient(pause=0.3)
    for position, title in enumerate(targets, 1):
        if title in done:
            continue
        try:
            done[title] = classify_index(client, title, samples)
        except Exception as error:  # one broken index must not end the crawl
            done[title] = {"title": title, "error": str(error)[:200], "verdict": "error"}
        print(
            f"  [{position}/{len(targets)}] {done[title]['verdict']:9s} {title[7:70]}", flush=True
        )
        if position % 10 == 0:
            _save(done)
    _save(done)


def _save(done: dict[str, dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(list(done.values()), ensure_ascii=False, indent=1), encoding="utf-8"
    )


def report() -> None:
    census = {e["title"]: e for e in json.loads(CENSUS.read_text(encoding="utf-8"))}
    entries = json.loads(OUTPUT.read_text(encoding="utf-8"))
    by_verdict: dict[str, list[dict]] = {}
    for entry in entries:
        by_verdict.setdefault(entry.get("verdict", "error"), []).append(entry)
    counts = {k: len(v) for k, v in sorted(by_verdict.items())}
    pages = {
        k: sum(census[e["title"]].get("proofread", 0) for e in v) for k, v in by_verdict.items()
    }
    print(f"{len(entries)} indexes classified: {counts}")
    print(f"proofread pages by verdict: {pages}\n")
    for verdict in (orthography.CLASSICAL, orthography.MIXED):
        rows = sorted(
            by_verdict.get(verdict, []), key=lambda e: -census[e["title"]].get("proofread", 0)
        )
        if not rows:
            continue
        print(f"### {verdict}\n")
        print("| Index | proofread | q4 | native px | ւ/1000 | եւ/և |")
        print("|---|---:|---:|---:|---:|---:|")
        for e in rows:
            c = census[e["title"]]
            print(
                f"| {e['title'].removeprefix('Ինդեքս:')} | {c.get('proofread', 0)} "
                f"| {c.get('q4', 0)} | {c.get('native_px', '')} | {e['free_yiwn']} "
                f"| {e['two_letter_ew']}/{e['ligature_ew']} |"
            )
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print the table from the cache")
    parser.add_argument("--min-pages", type=int, default=40, help="proofread-page floor")
    parser.add_argument("--samples", type=int, default=4, help="pages to read per index")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        crawl(args.min_pages, args.samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
