#!/usr/bin/env python3
"""Fetch the rendering faces the synthetic corpus is drawn in.

The fonts live under ``runs/v0/fonts/`` -- gitignored, like everything
under ``runs/`` -- so a fresh clone needs this script before it can
render. It downloads each family, unpacks the faces, and then verifies
two things per file rather than trusting the download page:

- the **licence recorded inside the font** (name table entries 13/14),
  because provenance from an aggregator site is hearsay. Verified
  2026-09-01: the Noto faces and all four Arian AMU faces carry the SIL
  OFL; the eight GHEA faces carry "Armenian National Book Chamber" --
  the Republic of Armenia's free-use official faces, *not* OFL. That is
  fine for this pipeline, which only ever renders with fonts and never
  redistributes one (the Mshtakan precedent), and it corrects brief
  011's assumption that the GHEA faces were OFL.
- the **glyph coverage** against the canonical charset, printed per
  face. Gaps are expected and handled -- the renderer excludes a face
  from any line it cannot fully draw -- but they should be seen, not
  discovered. All eight GHEA faces lack U+2024, exactly the class of
  gap that made this guard necessary.

Mshtakan is not fetched: it ships with macOS and is picked up from the
system by the renderer.

Run from the repo root:

    python scripts/fetch_fonts.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import charset, synth  # noqa: E402

FONT_DIR = REPO / "runs" / "v0" / "fonts"

# family slug on fonter.am -> the faces expected inside its zip
FAMILIES = {
    "ghea-grapalat": (
        "GHEAGrpalatReg.otf",
        "GHEAGpalatBld.otf",
        "GHEAGrapalatRit.otf",
        "GHEAGrapalatBlit.otf",
    ),
    "ghea-mariam": (
        "GHEAMariamReg.otf",
        "GHEAMariamBld.otf",
        "GHEAMariamRIt.otf",
        "GHEAMariamBlit.otf",
    ),
    "arian-amu": ("arnamu.ttf", "arnamu_bold.ttf", "arnamu_italic.ttf", "arnamu_italic_bold.ttf"),
}

NOTO = {
    "NotoSansArmenian.ttf": (
        "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/"
        "NotoSansArmenian/full/ttf/NotoSansArmenian-Regular.ttf"
    ),
    "NotoSerifArmenian.ttf": (
        "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/"
        "NotoSerifArmenian/full/ttf/NotoSerifArmenian-Regular.ttf"
    ),
}


def licence_of(path: Path) -> str:
    from fontTools.ttLib import TTFont

    font = TTFont(str(path), fontNumber=0, lazy=True)
    try:
        name = font["name"]
        return (name.getDebugName(13) or name.getDebugName(14) or "(none recorded)").strip()
    finally:
        font.close()


def main() -> int:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "tetrak-hy-trainer (https://github.com/scattercode/tetrak-hy-trainer)"
    )

    for slug, faces in FAMILIES.items():
        missing = [f for f in faces if not (FONT_DIR / f).exists()]
        if missing:
            print(f"fetching {slug} ...", flush=True)
            response = session.get(f"https://fonter.am/fonts/download/{slug}", timeout=120)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                for member in archive.namelist():
                    basename = Path(member).name
                    if basename in faces:
                        (FONT_DIR / basename).write_bytes(archive.read(member))

    for filename, url in NOTO.items():
        if not (FONT_DIR / filename).exists():
            print(f"fetching {filename} ...", flush=True)
            response = session.get(url, timeout=120)
            response.raise_for_status()
            (FONT_DIR / filename).write_bytes(response.content)

    print(f"\n{'face':<26} {'licence (from the font itself)':<44} coverage")
    full = charset.character_list()
    for path in sorted([*FONT_DIR.glob("*.tt*"), *FONT_DIR.glob("*.otf")]):
        gaps = synth.missing_glyphs(str(path), full)
        coverage = "full" if not gaps else f"missing {''.join(sorted(gaps))!r}"
        print(f"{path.name:<26} {licence_of(path).splitlines()[0][:44]:<44} {coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
