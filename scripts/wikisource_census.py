#!/usr/bin/env python3
"""Census every ProofreadPage index on Armenian Wikisource.

Brief 012 Stage 0.1. The training corpus so far is one publication, and
the supply of others is documented only by spot checks: coverage varies
from 350 proofread pages (Faustus of Byzantium, 1968) to zero (the
Bakunts volumes), so nothing can be planned from titles alone. This
walks all indexes in the Ինդեքս namespace, counts pages at each
ProofreadPage quality level, and for any index with a useful amount of
proofread material also records the native scan resolution — volume 1
of the ASE taught us to check, being 1920 px where volumes 2–6 are
3840.

Progress is cached per index into the output JSON as it goes, so an
interrupted run resumes without re-asking, and re-runs only query
indexes it has not seen. One polite pass; the client sleeps between
requests per Wikimedia etiquette.

Output: ``runs/census/census.json`` -- one record per index with title,
page counts by quality, and (where probed) the native width in px.
``--top N`` prints the N best-covered indexes as a markdown table for
pasting into the research note.

Run from the repo root:

    python scripts/wikisource_census.py            # the crawl (~20 min)
    python scripts/wikisource_census.py --top 40   # report from the cache
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer.wikisource import WikisourceClient, index_to_file_title  # noqa: E402

OUTPUT = REPO / "runs" / "census" / "census.json"

# Only indexes with at least this many proofread pages get the extra
# imageinfo call for native resolution -- most of the 629 have none, and
# probing them all is discourteous for no information.
RESOLUTION_FLOOR = 15

_PX = re.compile(r"page\d+-(\d+)px")


def all_index_titles(client: WikisourceClient) -> list[str]:
    """Every page title in the Ինդեքս namespace (106), via allpages."""
    titles: list[str] = []
    continuation: dict = {}
    while True:
        payload = client._get(
            action="query", list="allpages", apnamespace=106, aplimit=500, **continuation
        )
        titles.extend(p["title"] for p in payload["query"]["allpages"])
        continuation = payload.get("continue", {})
        if not continuation:
            return titles


def census_index(client: WikisourceClient, title: str) -> dict:
    """Counts by quality for one index, plus resolution where warranted."""
    counts: dict[str, int] = {}
    for record in client.pages_in_index(title, min_quality=3):
        key = f"q{record.quality}"
        counts[key] = counts.get(key, 0) + 1
    proofread = sum(counts.values())

    entry: dict = {"title": title, "proofread": proofread, **counts}
    if proofread >= RESOLUTION_FLOOR:
        try:
            url = client.page_image_url(index_to_file_title(title), 1, 4000)
            match = _PX.search(url)
            if match:
                entry["native_px"] = int(match.group(1))
        except Exception as error:  # a malformed file page should not kill the crawl
            entry["resolution_error"] = str(error)[:120]
    return entry


def crawl() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if OUTPUT.exists():
        done = {e["title"]: e for e in json.loads(OUTPUT.read_text(encoding="utf-8"))}

    client = WikisourceClient(pause=0.25)
    titles = all_index_titles(client)
    print(f"{len(titles)} indexes; {len(done)} already cached", flush=True)

    for position, title in enumerate(titles, 1):
        if title in done:
            continue
        try:
            done[title] = census_index(client, title)
        except Exception as error:
            done[title] = {"title": title, "error": str(error)[:200]}
        if position % 25 == 0 or position == len(titles):
            OUTPUT.write_text(
                json.dumps(list(done.values()), ensure_ascii=False, indent=1), encoding="utf-8"
            )
            covered = sum(1 for e in done.values() if e.get("proofread", 0) > 0)
            print(f"  [{position}/{len(titles)}] {covered} with proofread pages", flush=True)

    OUTPUT.write_text(
        json.dumps(list(done.values()), ensure_ascii=False, indent=1), encoding="utf-8"
    )


def report(top: int) -> None:
    entries = json.loads(OUTPUT.read_text(encoding="utf-8"))
    ranked = sorted(entries, key=lambda e: -e.get("proofread", 0))
    total = sum(e.get("proofread", 0) for e in entries)
    covered = sum(1 for e in entries if e.get("proofread", 0) > 0)
    print(f"{len(entries)} indexes; {covered} with any proofread pages; {total} proofread pages\n")
    print("| Index | proofread | q4 | native px |")
    print("|---|---:|---:|---:|")
    for entry in ranked[:top]:
        title = entry["title"].removeprefix("Ինդեքս:")
        print(
            f"| {title} | {entry.get('proofread', 0)} | {entry.get('q4', 0)} "
            f"| {entry.get('native_px', '')} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=None, help="report from the cache instead")
    args = parser.parse_args()
    if args.top is not None:
        report(args.top)
    else:
        crawl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
