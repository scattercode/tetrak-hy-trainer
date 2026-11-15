#!/usr/bin/env python3
"""Score the lexicon decoder's correction rule offline, from captured n-best.

Reads the JSON lines ``capture_nbest.py`` wrote and applies
``tetrak_hy.lexicon.LexiconDecoder``'s rule -- a word in the lexicon is
left alone, a word outside it is replaced by the most probable listed
candidate when that candidate is within tau of the greedy reading -- for
every lexicon and every tau asked for. Reports exact-match crops, the
change against greedy, crops fixed and broken, and the number of
corrections; with ``--by-orthography`` also each figure per the
orthography of the crops' source harvests, read from a
``build_wordlist.py`` log's per-harvest verdicts, which is what shows a
single list over-correcting the other orthography's pages.

Each ``name=path[,path...]`` argument is a lexicon, the union of the files
named, so the modern list, the classical list and
their union can be compared in one run. Build the lists with
``--exclude-crops <the crop set>`` first, or the labels are in them.

Prerequisites: a venv with ``tetrak_hy`` importable (the sibling
library's, or this one with the library installed); no GPU, no easyocr.

Run:
    ../tetrak-easyocr-armenian/.venv/bin/python scripts/score_tau.py \\
        runs/v7-b014/tau/nbest-v7.jsonl \\
        modern=runs/v7-b014/tau/wordlist.tsv.gz \\
        classical=runs/v7-b014/tau/wordlist-classical.tsv.gz \\
        union=runs/v7-b014/tau/wordlist.tsv.gz,runs/v7-b014/tau/wordlist-classical.tsv.gz \\
        --taus 1,3,5,8,12 --by-orthography runs/v7-b014/tau/build-modern.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tetrak_hy import lexicon

_ARMENIAN_LETTER = re.compile(r"[Ա-Ֆա-և]")
_VERDICT = re.compile(r"\s+[+-] (\S+): (\w+)")


def correct(greedy: str, candidates: list, words: frozenset[str], tau: float) -> str:
    """``LexiconDecoder._correct`` on saved candidates instead of probability rows."""
    key = lexicon.core(greedy)
    if len(key) < 2 or not _ARMENIAN_LETTER.search(greedy) or key in words:
        return greedy
    greedy_logp = next((lp for t, lp in candidates if t == greedy), candidates[0][1])
    edges = lexicon._EDGES.match(greedy).group(1, 3)
    for text, logp in candidates:
        if lexicon.core(text) in words and lexicon._EDGES.match(text).group(1, 3) == edges:
            return text if greedy_logp - logp <= tau else greedy
    return greedy


def decode(crop: dict, words: frozenset[str], tau: float) -> tuple[str, int]:
    """The crop's corrected text and how many of its words changed."""
    if not crop["words"]:
        return crop["greedy"], 0
    out = [correct(g, c, words, tau) for g, c in crop["words"]]
    return " ".join(out), sum(n != g for n, (g, _) in zip(out, crop["words"], strict=True))


def score(crops: list[dict], words: frozenset[str], tau: float) -> tuple[int, int, int, int]:
    ok = fixed = broke = corrections = 0
    for crop in crops:
        text, changed = decode(crop, words, tau)
        was, now = crop["greedy"] == crop["label"], text == crop["label"]
        ok += now
        fixed += now and not was
        broke += was and not now
        corrections += changed
    return ok, fixed, broke, corrections


def harvest_verdicts(log: Path) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in map(_VERDICT.match, log.read_text().splitlines()) if m}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nbest", type=Path)
    parser.add_argument("lexicons", nargs="+", metavar="name=path[,path...]")
    parser.add_argument("--taus", default="1,3,5,8,12")
    parser.add_argument("--by-orthography", type=Path, metavar="BUILD_LOG", default=None)
    args = parser.parse_args()
    taus = [float(t) for t in args.taus.split(",")]

    crops = [json.loads(line) for line in args.nbest.open(encoding="utf-8")]
    greedy_ok = sum(c["greedy"] == c["label"] for c in crops)
    print(f"{len(crops)} crops from {args.nbest}; greedy exact-match {greedy_ok / len(crops):.4f}")
    lexicons = {}
    for spec in args.lexicons:
        name, _, paths = spec.partition("=")
        # The union by hand rather than lexicon.load_wordlists, so the script
        # also runs against a library older than the two-list release.
        lexicons[name] = frozenset().union(*(lexicon.load_wordlist(p) for p in paths.split(",")))
        print(f"lexicon {name}: {len(lexicons[name])} words")

    for name, words in lexicons.items():
        print(f"\n{name}")
        print(
            f"{'tau':>5} {'exact':>7} {'change':>7} {'fixed':>6} {'broke':>6} {'corrections':>12}"
        )
        for tau in taus:
            ok, fixed, broke, corrections = score(crops, words, tau)
            print(
                f"{tau:>5} {ok / len(crops):>7.4f} {ok - greedy_ok:>+7d} "
                f"{fixed:>6d} {broke:>6d} {corrections:>12d}"
            )

    if args.by_orthography:
        verdicts = harvest_verdicts(args.by_orthography)
        groups: dict[str, list[dict]] = {}
        for crop in crops:
            source = crop["file"].rsplit("_", 2)[0]
            groups.setdefault(verdicts.get(source, "unknown"), []).append(crop)
        for tau in taus:
            print(f"\nBy source orthography at tau = {tau:g}")
            print(
                f"{'lexicon':10} {'pages':10} {'crops':>6} {'greedy':>7} "
                f"{'exact':>7} {'change':>7} {'fixed':>6} {'broke':>6}"
            )
            for name, words in lexicons.items():
                for orthography, subset in sorted(groups.items()):
                    g_ok = sum(c["greedy"] == c["label"] for c in subset)
                    ok, fixed, broke, _ = score(subset, words, tau)
                    print(
                        f"{name:10} {orthography:10} {len(subset):>6} {g_ok / len(subset):>7.4f} "
                        f"{ok / len(subset):>7.4f} {ok - g_ok:>+7d} {fixed:>6d} {broke:>6d}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
