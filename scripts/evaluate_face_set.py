#!/usr/bin/env python3
"""Score a bundle on brief 019's face-held-out test set, family by family.

Reads the set ``render_face_testset.py`` writes (``display/`` in the held-out
faces, ``control/`` with the same lines in the book faces) and recognises
each crop whole, greedily: the recogniser alone, no detector, no lexicon,
no fold, so the number is about letter shapes and nothing else.

Per held-out family and half it reports the share of lines read exactly and
the mean character similarity (Tetrak's metric, ``accuracy``), and then the
confusion table (``confusion``) for the display half and for the control
half. The gap between the halves is what the face costs; brief 019's
fine-tune should shrink it without moving the control.

The last line is ``RESULT {json}`` for comparing runs.

Prerequisites: needs easyocr and torch, so run it with the **sibling
library's venv**, as for ``evaluate_registers.py``:

    ../tetrak-easyocr-armenian/.venv/bin/python scripts/evaluate_face_set.py \\
        --bundle runs/v6-b013/bundle --set runs/b019/face_test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import accuracy, confusion, evaluate, packaging  # noqa: E402

#: The confusions brief 019 is about, reported whatever their rank.
WATCHED = (("ն", "ճ"), ("ճ", "ն"), ("ճ", "ջ"), ("ջ", "ճ"), ("ղ", "դ"), ("ք", "f"))


def read_half(folder: Path) -> list[tuple[str, str, str]]:
    """(file, label, family) for every crop in one half of the set."""
    with (folder / "faces.tsv").open(newline="") as handle:
        family = {row["file"]: row["family"] for row in csv.DictReader(handle, delimiter="\t")}
    rows = []
    for line in (folder / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
        name, _, label = line.partition(",")
        rows.append((name, label, family[name]))
    return rows


def recognise(reader, image: Path) -> str:
    import numpy as np
    from PIL import Image

    grey = np.array(Image.open(image).convert("L"))
    return "".join(reader.recognize(grey, detail=0, decoder="greedy"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--set", type=Path, default=REPO / "runs" / "b019" / "face_test")
    parser.add_argument("--limit", type=int, default=15, help="rows in each confusion table")
    args = parser.parse_args()

    import yaml

    reader = evaluate.load_reader(args.bundle.resolve())
    # A line with a character the bundle's charset lacks is unwinnable for
    # it, so it is left out rather than counted against an older model.
    config = yaml.safe_load((args.bundle / f"{packaging.NETWORK_NAME}.yaml").read_text())
    known = set(config["character_list"]) | {" "}
    result: dict[str, dict] = {}
    for half in ("display", "control"):
        exact: dict[str, list[bool]] = defaultdict(list)
        similar: dict[str, list[float]] = defaultdict(list)
        pairs = []
        for name, label, family in read_half(args.set / half):
            if not set(label) <= known:
                continue
            read = recognise(reader, args.set / half / name)
            exact[family].append(read == label)
            similar[family].append(accuracy.character_similarity(read, label))
            pairs.append((read, label))
        table: Counter = confusion.character_confusions(pairs)
        result[half] = {
            family: {
                "lines": len(exact[family]),
                "exact": round(sum(exact[family]) / len(exact[family]), 4),
                "chr": round(sum(similar[family]) / len(similar[family]), 4),
            }
            for family in sorted(exact)
        }
        result[half]["watched"] = {f"{a}->{b}": table[(a, b)] for a, b in WATCHED}
        print(f"\n== {half}")
        for family, score in result[half].items():
            if family != "watched":
                print(
                    f"{family:<16} {score['lines']:>5} lines  exact {score['exact']:.3f}  chr {score['chr']:.4f}"
                )
        print(confusion.format_table(table, args.limit))
        print("watched:", result[half]["watched"])
    print("RESULT " + json.dumps({"bundle": str(args.bundle), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
