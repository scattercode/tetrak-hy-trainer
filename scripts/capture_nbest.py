#!/usr/bin/env python3
"""Run a bundle over a crop set once and save every word's n-best readings.

Choosing the lexicon decoder's threshold tau (brief 013 Stage 2, re-checked
for each model since) needs the recogniser's alternatives for every word of
every held-out crop, scored under several word lists and several tau
values. The GPU pass is the slow part and does not depend on either, so
this script runs it once: one JSON line per crop with the file, the label,
the greedy text and, for every word of the greedy text, the beam
candidates ``[[text, log p], ...]`` that ``tetrak_hy.lexicon`` would choose
among. ``score_tau.py`` then applies the decoder's rule offline, in
seconds, for any list and any tau.

The crop set is a trainer dataset folder (``labels.csv`` beside the PNGs),
normally ``runs/<run>/all_data/real_val``: crops held out from training,
so that the lists scored against them can be rebuilt without their pages
(``build_wordlist.py --exclude-crops``) and no label is in a list by
construction.

Prerequisites: the sibling library's venv (it has easyocr; this
repository's venv does not) with ``tetrak_hy`` importable -- the editable
install, or a checkout passed as the last argument. Apple MPS is used when
available; 21,574 crops took under five minutes on an M-series laptop.

Run:
    ../tetrak-easyocr-armenian/.venv/bin/python -W ignore scripts/capture_nbest.py \\
        runs/v7-b014/bundle runs/v7-b014/all_data/real_val \\
        runs/v7-b014/tau/nbest-v7.jsonl [../tetrak-easyocr-armenian/src]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

bundle, crops, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if len(sys.argv) > 4:
    sys.path.insert(0, sys.argv[4])
import easyocr  # noqa: E402, I001
from tetrak_hy import lexicon  # noqa: E402, I001


class Capturer(lexicon.LexiconDecoder):
    """The decoder with an empty lexicon, recording each word's beam instead of correcting."""

    def __init__(self, character, ignore_idx) -> None:
        super().__init__(character, ignore_idx, frozenset())
        self.calls: list[list[tuple[str, list[tuple[str, float]]]]] = []

    def __call__(self, mat, beamWidth: int = 5) -> list[str]:  # noqa: N803 -- EasyOCR's name
        self.calls.append([])
        return [self.decode(m) for m in mat]

    def _correct(self, greedy: str, rows: list[list[float]]) -> str:
        self.calls[-1].append((greedy, self._beam(rows)))
        return greedy


# ["en"], never ["hy"]: see AGENTS.md.
reader = easyocr.Reader(
    ["en"],
    recog_network="tetrak_hy",
    user_network_directory=str(bundle),
    model_storage_directory=str(bundle),
    verbose=False,
    gpu=True,
)
cap = Capturer(reader.converter.character, reader.converter.ignore_idx)
reader.converter.decode_beamsearch = cap

labels = {}
for line in (crops / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
    name, label = line.split(",", 1)
    labels[name] = label

start = time.time()
done = 0
with out.open("w", encoding="utf-8") as handle:
    for name, label in labels.items():
        cap.calls = []
        texts = reader.recognize(str(crops / name), decoder="beamsearch", detail=0)
        # EasyOCR may re-run a low-contrast crop; the last call is the one it kept.
        words = cap.calls[-1] if cap.calls else []
        record = {
            "file": name,
            "label": label,
            "greedy": " ".join(texts),
            "words": [[g, [[t, round(lp, 4)] for t, lp in c]] for g, c in words],
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        done += 1
        if done % 1000 == 0:
            print(f"{done} crops, {time.time() - start:.0f}s", flush=True)
print(f"CAPTURE EXIT 0 ({done} crops, {time.time() - start:.0f}s)", flush=True)
