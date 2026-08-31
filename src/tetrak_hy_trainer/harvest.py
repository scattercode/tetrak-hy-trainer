"""Harvest one Wikisource volume into a local page/scan dataset.

Writes, under ``--out``:

    manifest.json          one entry per harvested page: title, pageid,
                           revid, quality, page number, files written --
                           the provenance record a later training run and
                           the weights release both cite
    text/<n>.txt           cleaned transcript per page
    images/<n>.jpg         the rendered scan (only with --images)

Only pages at or above the quality floor are harvested; see
:mod:`tetrak_hy_trainer.wikisource` for why that floor exists and never
drops below "proofread".

Usage:
    python -m tetrak_hy_trainer.harvest \\
        --index "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 1.djvu" \\
        --out data/ase/vol1 --images [--limit 20] [--pages 20-80] [--min-quality 4]

Re-runs are incremental: a page whose text file already exists keeps it and
refetches no wikitext, so an interrupted harvest resumes where it stopped.
Adding ``--images`` to a harvest taken without them tops up the missing
scans without disturbing the text -- which is how the volumes harvested for
v0 and v1 acquire the page images real-crop harvesting needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tetrak_hy_trainer import wikisource
from tetrak_hy_trainer.wikisource import QUALITY_PROOFREAD, WikisourceClient


def parse_page_spec(spec: str) -> set[int]:
    """Parse ``"20-80,100,140-150"`` into the set of page numbers it names.

    Front matter is not training material: volume 1 opens with Russian
    title pages, a preface and editorial notes, and its encyclopedia
    entries only start around page 20. "The first N pages" is therefore
    the wrong sixty pages to fetch scans for, hence naming them.

    Raises:
        ValueError: The spec is malformed or describes a backwards range.
    """
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            first, _, last = part.partition("-")
            start, stop = int(first), int(last)
            if stop < start:
                raise ValueError(f"backwards page range: {part!r}")
            pages.update(range(start, stop + 1))
        else:
            pages.add(int(part))
    if not pages:
        raise ValueError(f"no pages in spec: {spec!r}")
    return pages


def harvest(
    client: WikisourceClient,
    index_title: str,
    out_dir: Path,
    min_quality: int = QUALITY_PROOFREAD,
    limit: int | None = None,
    images: bool = False,
    image_width: int = 2048,
    pages: set[int] | None = None,
) -> list[dict]:
    """Harvest *index_title* into *out_dir*; return the manifest entries.

    Args:
        pages: Only harvest these page numbers, as from
            :func:`parse_page_spec`. ``None`` takes them all, in order.
    """
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    if images:
        image_dir.mkdir(parents=True, exist_ok=True)

    file_title = wikisource.index_to_file_title(index_title)
    manifest: list[dict] = []
    harvested = 0

    for record in client.pages_in_index(index_title, min_quality=min_quality):
        if limit is not None and harvested >= limit:
            break
        if pages is not None and record.page_number not in pages:
            continue

        text_path = text_dir / f"{record.page_number}.txt"
        entry = {
            "title": record.title,
            "pageid": record.pageid,
            "quality": record.quality,
            "page_number": record.page_number,
            "text": str(text_path.relative_to(out_dir)),
        }

        if text_path.exists():
            # Resuming: keep the text as it is and fetch no wikitext.
            entry["revid"] = None
        else:
            wikitext, revid = client.page_wikitext(record.title)
            cleaned = wikisource.clean_wikitext(wikitext)
            if not cleaned:
                continue  # a blank or image-only page contributes nothing
            text_path.write_text(cleaned, encoding="utf-8")
            entry["revid"] = revid

        # Outside the resume branch on purpose. The volumes harvested for
        # v0 and v1 were taken text-only -- synthesis needs corpus text and
        # nothing else -- and real-crop harvesting (brief 011 Stage 3
        # step 2) later needs the scans for those same pages. While the
        # image fetch sat inside the "new page" branch, re-running with
        # --images over an existing harvest downloaded nothing at all and
        # said it had succeeded: every page was already resumed past.
        if images:
            image_path = image_dir / f"{record.page_number}.jpg"
            if not image_path.exists():
                url = client.page_image_url(file_title, record.page_number, image_width)
                response = client.session.get(url, timeout=60)
                response.raise_for_status()
                image_path.write_bytes(response.content)
            entry["image"] = str(image_path.relative_to(out_dir))

        manifest.append(entry)
        harvested += 1
        print(f"  [{harvested}] p{record.page_number} q{record.quality} {text_path.name}")

    manifest = _merged_with_existing(out_dir / "manifest.json", manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "index": index_title,
                "min_quality": min_quality,
                "pages": manifest,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return manifest


def _merged_with_existing(manifest_path: Path, entries: list[dict]) -> list[dict]:
    """Fold *entries* into any manifest already at *manifest_path*.

    A ``--limit``ed run visits only the first few pages, and the manifest
    is provenance -- it records which revision of which transcript trained
    a published model. Writing only the visited pages would quietly throw
    away the rest: topping up volume 1's scans with ``--limit 60`` would
    have cut its 717-page record down to 60. Pages visited this time win;
    pages not visited are kept as they were.
    """
    existing: dict[int, dict] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {entry["page_number"]: entry for entry in previous.get("pages", [])}

    existing.update({entry["page_number"]: entry for entry in entries})
    return sorted(existing.values(), key=lambda item: item["page_number"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="full Ինդեքս: title of the volume")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--min-quality",
        type=int,
        default=QUALITY_PROOFREAD,
        choices=[QUALITY_PROOFREAD, wikisource.QUALITY_VALIDATED],
        help="ProofreadPage floor: 3 proofread (default) or 4 validated",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N pages")
    parser.add_argument(
        "--pages",
        default=None,
        help='only these pages, e.g. "20-80,100" -- front matter is not training material',
    )
    parser.add_argument("--images", action="store_true", help="download page scans too")
    parser.add_argument("--image-width", type=int, default=2048)
    args = parser.parse_args(argv)

    client = WikisourceClient()
    manifest = harvest(
        client,
        index_title=args.index,
        out_dir=args.out,
        min_quality=args.min_quality,
        limit=args.limit,
        images=args.images,
        image_width=args.image_width,
        pages=parse_page_spec(args.pages) if args.pages else None,
    )
    print(f"Harvested {len(manifest)} page(s) into {args.out}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
