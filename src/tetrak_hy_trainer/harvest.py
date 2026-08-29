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
        --out data/ase/vol1 --images [--limit 20] [--min-quality 4]

Re-runs are incremental: a page whose text file already exists is skipped,
so an interrupted harvest resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tetrak_hy_trainer import wikisource
from tetrak_hy_trainer.wikisource import QUALITY_PROOFREAD, WikisourceClient


def harvest(
    client: WikisourceClient,
    index_title: str,
    out_dir: Path,
    min_quality: int = QUALITY_PROOFREAD,
    limit: int | None = None,
    images: bool = False,
    image_width: int = 2048,
) -> list[dict]:
    """Harvest *index_title* into *out_dir*; return the manifest entries."""
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

        text_path = text_dir / f"{record.page_number}.txt"
        entry = {
            "title": record.title,
            "pageid": record.pageid,
            "quality": record.quality,
            "page_number": record.page_number,
            "text": str(text_path.relative_to(out_dir)),
        }

        if text_path.exists():
            # Resuming: count it, keep it in the manifest, fetch nothing.
            entry["revid"] = None
            manifest.append(entry)
            harvested += 1
            continue

        wikitext, revid = client.page_wikitext(record.title)
        cleaned = wikisource.clean_wikitext(wikitext)
        if not cleaned:
            continue  # a blank or image-only page contributes nothing
        text_path.write_text(cleaned, encoding="utf-8")
        entry["revid"] = revid

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

    manifest.sort(key=lambda item: item["page_number"])
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
    )
    print(f"Harvested {len(manifest)} page(s) into {args.out}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
