#!/usr/bin/env python3
"""Build the Armenian word list ``tetrak_hy.lexicon`` decodes with.

Counts every Armenian word in the harvested proofread transcripts, after
``wikisource.normalise_transcript``, and writes ``word<TAB>count`` lines,
most frequent first. Words are keyed as the decoder keys them: leading and
trailing punctuation removed, lower-cased.

**Held-out pages are excluded** through the trainer's registry, exactly as
the synthetic sampler excludes them, so no word is in the list because an
evaluation page contains it. Other pages of the evaluated works are
included -- the same exposure training already has. ``--exclude-crops``
additionally drops the pages a real-crop dataset was cut from, which is
what choosing the decoder's threshold on that dataset's validation crops
needs (brief 013 Stage 2).

``--nayiri`` adds every word form of the Nayiri Armenian Lexicon
(http://www.nayiri.com/nayiri-armenian-lexicon, CC BY 4.0: "Nayiri Armenian
Lexicon (c) Serouj Ourishian"), counted as ``--min-count`` so the decoder
keeps them. Its current release is Western Armenian in traditional
orthography, where it extends coverage well beyond the harvest.

The word list is derived from CC BY-SA 3.0 Wikisource text (and, with
``--nayiri``, CC BY 4.0 data); its attribution travels with it wherever it
is published, as the weights' does. A sidecar ``<out>.json`` records the
inputs actually used and the attribution they require; ``upload_model.py
--wordlist`` publishes that, so the attribution cannot claim a source the
list was not built from.

Prerequisites: this repository's venv; harvests under ``runs/harvest/``
and the others the defaults name.

Run:
    python scripts/build_wordlist.py --out runs/v6-b013/wordlist.tsv.gz \\
        [--nayiri nayiri-armenian-lexicon-2026-04-25-v3.json] \\
        [--exclude-crops runs/v6-b013/all_data/real_val]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import heldout, wikisource  # noqa: E402

_ARMENIAN_LETTER = re.compile(r"[Ա-Ֆա-և]")
# The decoder's key: Armenian letters only, so U+055A-U+055F (՝ ՛ ՜ ՞),
# which sit between the two letter ranges, are stripped as punctuation.
_EDGES = re.compile(r"^[^\wԱ-Ֆա-և]+|[^\wԱ-Ֆա-և]+$")


def source_slug(name: str) -> str:
    """The crop-name prefix for a harvest directory.

    Must match ``harvest_real_crops.source_slug``, which names crops; not
    imported from it because that script loads easyocr at import time.
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def harvest_dirs() -> list[Path]:
    return [
        d
        for d in [
            REPO / "runs" / "v0" / "harvest",
            *sorted((REPO / "runs" / "v1").glob("harvest-*")),
            *sorted((REPO / "runs" / "harvest").iterdir()),
        ]
        if (d / "manifest.json").exists()
    ]


def crop_pages(folder: Path) -> dict[str, set[int]]:
    """Pages a real-crop folder was cut from, by harvest directory name."""
    pages: dict[str, set[int]] = collections.defaultdict(set)
    for line in (folder / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
        source, page, _ = line.split(",", 1)[0].rsplit("_", 2)
        pages[source].add(int(page))
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nayiri", type=Path, default=None)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--exclude-crops", type=Path, default=None)
    args = parser.parse_args()

    excluded = crop_pages(args.exclude_crops) if args.exclude_crops else {}
    counts: collections.Counter[str] = collections.Counter()
    read = held_out = 0
    for directory in harvest_dirs():
        index = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["index"]
        for text_file in (directory / "text").glob("*.txt"):
            page = int(text_file.stem)
            if heldout.page_is_held_out(index, page):
                held_out += 1
                continue
            if page in excluded.get(source_slug(directory.name), ()):
                continue
            read += 1
            text = wikisource.normalise_transcript(text_file.read_text(encoding="utf-8"))
            for token in text.split():
                word = _EDGES.sub("", token).lower()
                if word and _ARMENIAN_LETTER.search(word):
                    counts[word] += 1
    print(f"{read} pages read, {held_out} held-out pages skipped, {len(counts)} word types")

    if args.nayiri:
        data = json.loads(args.nayiri.read_text(encoding="utf-8"))
        added = 0
        for lexeme in data["lexemes"]:
            for lemma in lexeme["lemmas"]:
                for form in lemma["wordForms"]:
                    word = form["s"].lower()
                    if counts[word] < args.min_count:
                        counts[word] = args.min_count
                        added += 1
        print(f"Nayiri: {added} forms added or raised to the minimum count")

    # Only the words the decoder keeps are written: below --min-count they
    # are dropped at load time anyway, and the list is downloaded by every
    # user of the library. A .gz name compresses it (34 MB -> 3.6 MB).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.out.suffix == ".gz" else open
    kept = 0
    with opener(args.out, "wt", encoding="utf-8") as handle:
        for word, count in counts.most_common():
            if count < args.min_count:
                break
            handle.write(f"{word}\t{count}\n")
            kept += 1
    print(f"wrote {args.out}: {kept} words at or above {args.min_count} of {len(counts)}")

    attribution = (
        "Word counts from proofread Armenian Wikisource transcripts (CC BY-SA "
        "3.0), evaluation pages and pages printing their text excluded"
    )
    nayiri = None
    if args.nayiri:
        release = args.nayiri.name.removeprefix("nayiri-armenian-lexicon-").removesuffix(".json")
        nayiri = {
            "file": args.nayiri.name,
            "sha256": hashlib.sha256(args.nayiri.read_bytes()).hexdigest(),
        }
        attribution += (
            f"; with every word form of the Nayiri Armenian Lexicon {release} "
            "(c) Serouj Ourishian, CC BY 4.0, added at the minimum count"
        )
    sidecar = args.out.with_name(args.out.name + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "file": args.out.name,
                "words": kept,
                "min_count": args.min_count,
                "pages_read": read,
                "held_out_pages_skipped": held_out,
                "crop_pages_excluded": str(args.exclude_crops) if args.exclude_crops else None,
                "nayiri": nayiri,
                "attribution": attribution + ". Built by scripts/build_wordlist.py.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
