"""The character set the recogniser is trained on — the single source of truth.

The trainer and the packaged ``tetrak_hy.yaml`` both read this module;
nothing else may define its own copy of the character list. The order is
load-bearing: CTC class indices are positional, so any change to the
*content or order* of :func:`character_list` changes ``num_class`` and the
meaning of every index — which invalidates all previously trained weights.
Treat a charset change as a new model version, never a patch.

Composition
-----------
- Armenian uppercase Ա–Ֆ (U+0531–U+0556, 38 letters) and lowercase ա–ֆ
  (U+0561–U+0586, 38 letters), generated from the Unicode ranges rather
  than typed, so a typo cannot silently drop a letter.
- The և ligature (U+0587). Lowercase-only; it is ubiquitous in printed
  Armenian, so it is *in* the charset. **Policy, decided 2026-10-08 (brief
  014):** ground truth stays as the page prints it -- ``եւ`` where the press
  set two sorts, ``և`` where it set the ligature -- so the model learns to
  read both; the metric (``accuracy.normalise``) and ``tetrak_hy.fold_script``
  treat the two as one, so no engine is ranked on which form it emitted.
  Classical-orthography presses split 21 to 16 on the habit; reformed
  presses set the ligature almost without exception. Nothing is folded on
  the ground-truth side.
- v5 additions (:data:`V5_ADDITIONS`, brief 014 Stage 1): the Armenian
  apostrophe ՚ (U+055A), printed in classical orthography for elision
  (``կ՚ուզէ``) and absent from every reformed source -- the one printed
  addition the Stage 0 charset diff found across 33 classical harvests;
  and the Russian alphabet in both cases including ё/Ё (66 classes),
  generated from the Unicode ranges like the Armenian letters, with the
  pre-1918 letters (ѣ і ѳ ѵ) left out because no source supplies them.
  Ground truth for a Cyrillic word is decided per token by its letters;
  tokens mixing Cyrillic with Armenian or Latin letters (formulae,
  transcriber slips) are dropped from training rather than taught. The
  Latin/Cyrillic lookalikes (``а/a``, ``о/o``, ``с/c``, ``р/p``, ``е/e``,
  ``х/x``) and the Armenian ones (``Օ/О``, ``Ո/П``) are therefore distinct
  classes on purpose: the token's script, not the glyph, decides the label.
- Armenian punctuation and typography: ՝ ՛ ՞ ՜ ։ ֊ « ».
- Western digits and basic Latin, because 19th–20th century Armenian print
  mixes in Latin names, numerals and abbreviations.
- Common punctuation shared across scripts.
- v4 additions (:data:`V4_ADDITIONS`): the angle brackets, superscript
  three and the plus-minus sign -- the notation of the encyclopedias and
  the critical editions. v3 treated every angle bracket as a typed
  guillemet; checking the scans showed most are print: the encyclopedia's
  "derived from" sign in etymologies and sound changes, and the editorial
  brackets of Tumanyan's and Baronian's collected works. The guillemet fold
  in :func:`tetrak_hy_trainer.wikisource.normalise_transcript` now leaves
  those alone, so the charset needs a class for them.
- v3 additions (:data:`V3_ADDITIONS`): the ellipsis, square brackets,
  the numero sign and superscript two — found by diffing the corpus
  brief 012 widened to, and all genuinely printed. The same diff found
  transcriber *substitutions* (angle brackets for guillemets, a minus
  sign for a dash); those are normalised in
  :func:`tetrak_hy_trainer.wikisource.normalise_transcript` rather than
  admitted here, because a charset class for U+2212 would only give the
  model a new homoglyph to confuse with the en dash.
- v2 additions (:data:`V2_ADDITIONS`): U+2024 ONE DOT LEADER, the ASE
  transcripts' abbreviation dot (``Ա․``, ``Գրկ․``), and ``°``. v1's error
  analysis (design research 010) found U+2024 absent from v1's charset
  entirely, so the training pipeline's "drop any crop containing an
  out-of-charset character" step silently filtered every crop containing
  it and v1 could never emit it at all — 518 words (5.8% of the ten
  evaluation pages' expected words) unwinnable by construction. Appended
  at the end of :func:`character_list` rather than folded into
  :data:`ARMENIAN_PUNCTUATION`, so the existing charset's prefix and the
  ``ա`` index the v1 tests pin are untouched by the append itself — the
  charset is still a new, incompatible version (see below), just a
  minimal diff against v1's.

The space character is a flag (:data:`INCLUDE_SPACE`) rather than a fact,
because it was an open question when this was written. **It is settled: the
space is in.** The spike confirmed the EasyOCR trainer and inference path both
expect it in ``character_list``, and every released model (v0 onwards) carries
it — so the flag is now a knob nobody should turn, not a decision still to
make. Turning it off would change ``num_class`` and invalidate every weight
file, like any other charset change.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

ARMENIAN_UPPER = "".join(chr(code) for code in range(0x0531, 0x0556 + 1))  # Ա–Ֆ
ARMENIAN_LOWER = "".join(chr(code) for code in range(0x0561, 0x0586 + 1))  # ա–ֆ

# U+0587. Membership is settled (see module docstring); only the
# ground-truth normalisation policy is open.
ARMENIAN_EW_LIGATURE = "և"  # և

# ՝ but ՛ shesht ՞ hard question ՜ exclamation ։ full stop ֊ hyphen, plus
# the guillemets Armenian print quotes with.
ARMENIAN_PUNCTUATION = "՝՛՞՜։֊«»"

DIGITS = "0123456789"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
COMMON_PUNCTUATION = ".,:;!?'\"()-–—/%&+=*"

# U+2024 ONE DOT LEADER (the transcripts' abbreviation dot) and the degree
# sign — see the module docstring's "v2 additions" note for why these two
# and not more. Appended after COMMON_PUNCTUATION, not inserted earlier,
# to keep the diff against v1's charset a pure append.
V2_ADDITIONS = "․°"

# v3 additions, from diffing the whole widened corpus against the charset
# (brief 012 stage 1.3) -- every one genuinely printed, not a transcriber
# substitution:
#   …  2,930  the literary ellipsis, all through Tumanyan and Otyan
#   [ ] 3,410  encyclopedia editorial brackets, as in a birth/death note
#              "[13(15)․7․1852, Կ․ Պոլիս —1929, Փարիզ]"
#   №    833  the Soviet-era numero sign, in citations and shelfmarks
#   ²    263  superscript two, in areas and units ("30 կմ²")
# Appended, like V2_ADDITIONS, so the diff against v2's charset stays a
# pure append and the ordering tests keep their meaning.
V3_ADDITIONS = "…[]№²"

# v4 additions, from the scans rather than the transcripts alone:
#   < > 2,421  the encyclopedia's "derived from" sign ("(< լատ․ pulpa",
#              "արջ<արչ") and comparisons, and the critical editions'
#              editorial brackets ("օր<ինակ>", "<1 անընթ.>") -- counted
#              after the guillemet fold, so only the printed ones
#   ³    126  superscript three, in volumes and densities ("գ/սմ³")
#   ±     62  plus-minus, in measurements ("3500±450 գ")
# Appended, like V3_ADDITIONS. Every GHEA and Arnamu face draws all four;
# Noto lacks ³ and ± as it already lacks ², and the renderer excludes a
# face from any line it cannot draw.
V4_ADDITIONS = "<>³±"

# v5 additions (brief 014 Stage 1, decided at Stage 0.3 on 2026-10-08):
#   ՚      2,213  the Armenian apostrophe of elision ("կ՚ուզէ", "չ՚ուզեր"),
#                 printed in 25 of the 33 classical-orthography harvests and
#                 in no reformed source -- the only printed addition the
#                 Stage 0 charset diff found; the transcribers' ’ and ˈ for
#                 it are folded to it in wikisource.normalise_transcript
#   А–Я а–я     the Russian alphabet in both cases, 64 letters, generated
#                 from U+0410–U+042F and U+0430–U+044F so a typo cannot drop
#                 one; plus Ё (U+0401) and ё (U+0451), which the ranges skip
#   (not ѣ і ѳ ѵ) the pre-1918 letters are left out: the supply (the Soviet
#                 encyclopedias and Tumanyan's apparatus) never prints them
# Cyrillic is in because the Soviet-era harvests carry Russian words and
# short runs mid-line -- 72 tokens on the Tumanyan evaluation set alone --
# and v5/v6 could only drop every crop that held one. Appended, like
# V4_ADDITIONS, so every earlier charset stays a prefix of this one.
ARMENIAN_APOSTROPHE = "՚"  # U+055A
CYRILLIC_UPPER = "".join(chr(code) for code in range(0x0410, 0x042F + 1)) + "Ё"  # А–Я, Ё
CYRILLIC_LOWER = "".join(chr(code) for code in range(0x0430, 0x044F + 1)) + "ё"  # а–я, ё
V5_ADDITIONS = ARMENIAN_APOSTROPHE + CYRILLIC_UPPER + CYRILLIC_LOWER

# The letter sets the per-token script policy reads. Punctuation, digits
# and the ligature's neighbours decide nothing; letters do.
ARMENIAN_LETTERS = frozenset(ARMENIAN_UPPER + ARMENIAN_LOWER + ARMENIAN_EW_LIGATURE)
CYRILLIC_LETTERS = frozenset(CYRILLIC_UPPER + CYRILLIC_LOWER)
LATIN_LETTERS = frozenset(LATIN)

# Settled at spike time against what the EasyOCR trainer actually expects,
# and baked into every released model since v0. Not a switch to flip: see
# the module docstring.
INCLUDE_SPACE = True


def character_list(include_space: bool | None = None) -> str:
    """Return the full character list as the yaml's ``character_list`` string.

    Args:
        include_space: Override :data:`INCLUDE_SPACE`; ``None`` uses the
            module default.

    Returns:
        Every trainable character, in the fixed order described in the
        module docstring. Length of this string + 1 (CTC blank) is the
        model's ``num_class``.
    """
    include_space = INCLUDE_SPACE if include_space is None else include_space
    characters = (
        ARMENIAN_UPPER
        + ARMENIAN_LOWER
        + ARMENIAN_EW_LIGATURE
        + ARMENIAN_PUNCTUATION
        + DIGITS
        + LATIN
        + COMMON_PUNCTUATION
        + V2_ADDITIONS
        + V3_ADDITIONS
        + V4_ADDITIONS
        + V5_ADDITIONS
    )
    if include_space:
        characters += " "
    return characters


def num_class(include_space: bool | None = None) -> int:
    """The CTC output size: every character plus the blank token at index 0."""
    return len(character_list(include_space)) + 1


def scripts_of(token: str) -> frozenset[str]:
    """Which alphabets *token* has letters from.

    A subset of ``{"armenian", "cyrillic", "latin"}``; digits and
    punctuation count for nothing, so ``"1974․"`` is of no script.
    """
    letters = set(token)
    found = set()
    if letters & ARMENIAN_LETTERS:
        found.add("armenian")
    if letters & CYRILLIC_LETTERS:
        found.add("cyrillic")
    if letters & LATIN_LETTERS:
        found.add("latin")
    return frozenset(found)


def mixes_cyrillic(token: str) -> bool:
    """Whether *token* mixes Cyrillic with Armenian or Latin letters.

    The Stage 0.3 policy (brief 014, 2026-10-08): a token is Cyrillic if
    its letters are, and one that mixes Cyrillic with another alphabet --
    a formula, a transcriber's slip, a lookalike typed from the wrong
    keyboard (``Mосква`` with a Latin M) -- is dropped from training rather
    than taught, because no label for it is right. Across the harvests
    that is 770 tokens against 47,000 wholly Cyrillic ones.
    """
    found = scripts_of(token)
    return "cyrillic" in found and len(found) > 1


def strays(text: str) -> Counter[str]:
    """Count every character in *text* that the charset has no class for.

    The check brief 012 institutionalised after brief 011 paid for its
    absence: U+2024, the transcripts' abbreviation dot, was missing from
    v1's charset, so the training pipeline silently dropped every crop
    containing it and 5.8% of the evaluation words were unwinnable by
    construction. Five minutes of counting would have caught it before it
    cost a training run — so now it is a function, run against every new
    source before anything trains on it (``scripts/charset_diff.py``).

    Whitespace is ignored: the tokeniser splits on it, so it never needs
    a class of its own beyond the space :data:`INCLUDE_SPACE` governs.

    Args:
        text: Corpus text, typically a whole harvest's transcripts joined.

    Returns:
        Stray character -> occurrence count, most common first when
        iterated via ``most_common()``. Empty means the charset covers
        the material.
    """
    allowed = set(character_list(include_space=True))
    return Counter(c for c in text if c not in allowed and not c.isspace())


# Written beside a dataset's labels.csv by the code that labels it.
STAMP_FILE = "charset.txt"


def stamp(folder: Path) -> None:
    """Record the charset *folder*'s labels were made under.

    Real-crop labels are frozen at harvest: a later charset or
    normalisation change does not reach them, and ``relabel_dataset.py``
    cannot restore a character the old normaliser folded away. Charset v4
    is the case in point -- crops harvested under v3 carry « where the page
    prints <. The stamp lets the fine-tune refuse them rather than train
    on them unnoticed.
    """
    (folder / STAMP_FILE).write_text(character_list(), encoding="utf-8")


def stamp_matches(folder: Path) -> bool:
    """Whether *folder* was labelled under the current charset."""
    path = folder / STAMP_FILE
    return path.exists() and path.read_text(encoding="utf-8") == character_list()
