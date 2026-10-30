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

The GHEA and Arian AMU faces came from fonter.am until October 2026, when
its download links began answering 303 to an HTML page behind a signed,
expiring link. They now come from the fonts-armenian-extra collection on
GitHub, pinned to a commit, and each file is checked against the SHA-256 of
the copy v0 to v6 were trained with: all twelve are byte-identical, so the
change of source changes nothing the renderer draws.

Mshtakan is not fetched: it ships with macOS and is picked up from the
system by the renderer.

Run from the repo root:

    python scripts/fetch_fonts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import charset, fontsource, synth  # noqa: E402
from tetrak_hy_trainer.fontsource import Face  # noqa: E402

FONT_DIR = REPO / "runs" / "v0" / "fonts"

MIRROR = (
    "https://codeload.github.com/lecramyajiv/fonts-armenian-extra/tar.gz/"
    "387f6ec024db5a8bf9a819deda146a8871dd64dc"
)
_ROOT = "fonts-armenian-extra-387f6ec024db5a8bf9a819deda146a8871dd64dc"


def _mirror(family: str, folder: str, filename: str, sha256: str) -> Face:
    return Face(family, filename, MIRROR, f"{_ROOT}/{folder}/{filename}", sha256)


# The faces every model since v0 has been rendered in, with the hashes of
# the files they were rendered from.
FACES = (
    _mirror(
        "GHEA Grapalat",
        "OTF",
        "GHEAGrpalatReg.otf",
        "61101112caaf07d575d6bd038e6661cbf9799daa1f364509d818db37eaf68700",
    ),
    _mirror(
        "GHEA Grapalat",
        "OTF",
        "GHEAGpalatBld.otf",
        "30c2fd93c127d2de488d4d21102c86242831f424739f1e9573f5b3891105abbe",
    ),
    _mirror(
        "GHEA Grapalat",
        "OTF",
        "GHEAGrapalatRit.otf",
        "054e3c1aa658905137d976314abc3935915a0a912ea5260a01591284e69813dd",
    ),
    _mirror(
        "GHEA Grapalat",
        "OTF",
        "GHEAGrapalatBlit.otf",
        "ec6cfe10e720c35af69ad81f9231d7e254aef4c94537863471c76a272e283980",
    ),
    _mirror(
        "GHEA Mariam",
        "OTF",
        "GHEAMariamReg.otf",
        "806ab65039449b157789dad32e8fb1efdc6131679454775c9003f5252541afc4",
    ),
    _mirror(
        "GHEA Mariam",
        "OTF",
        "GHEAMariamBld.otf",
        "0dc5eea38e0cf2d7e19d649f2996cb59726f2a569ce6b8abc4618979bb98e345",
    ),
    _mirror(
        "GHEA Mariam",
        "OTF",
        "GHEAMariamRIt.otf",
        "673045ee28bf37ba6051a6af0c3ba401dc04ae07b7a7d93cf606c6556e74b3b8",
    ),
    _mirror(
        "GHEA Mariam",
        "OTF",
        "GHEAMariamBlit.otf",
        "412a51e0338b660037fb82d5b19347f05aa99007bee49f56cdbbb30fe1357cce",
    ),
    _mirror(
        "Arian AMU",
        "TTF",
        "arnamu.ttf",
        "9733b1a04f52457839b301c1fca3e68e4132420001bb83c0ba8cf7d3212e6aa6",
    ),
    _mirror(
        "Arian AMU",
        "TTF",
        "arnamu_bold.ttf",
        "7197abb08ce1cc16ea3bcf35e920cf2707214a7a3e18fe4065988d59a728184f",
    ),
    _mirror(
        "Arian AMU",
        "TTF",
        "arnamu_italic.ttf",
        "86169da604999c3cf7d6fc0213518b4ed7d65ec5f822c1d65f3e557c8e37d7d2",
    ),
    _mirror(
        "Arian AMU",
        "TTF",
        "arnamu_italic_bold.ttf",
        "0c916213c18f49afcd33636238d52b39ef61f38ad49c59ed501396b87d015761",
    ),
)

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


def main() -> int:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "tetrak-hy-trainer (https://github.com/scattercode/tetrak-hy-trainer)"
    )

    for face in FACES:
        fontsource.fetch(face, FONT_DIR, FONT_DIR / ".cache", session)

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
        print(f"{path.name:<26} {fontsource.licence_of(path).splitlines()[0][:44]:<44} {coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
