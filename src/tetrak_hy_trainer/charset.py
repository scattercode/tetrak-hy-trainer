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
  Armenian, so it is *in* the charset. The open policy question is
  normalisation on the ground-truth side (whether ԵՎ/Եւ forms in
  transcripts are folded to և), not membership here. Decided when real
  transcripts are in hand; record the decision in this docstring.
- Armenian punctuation and typography: ՝ ՛ ՞ ՜ ։ ֊ « ».
- Western digits and basic Latin, because 19th–20th century Armenian print
  mixes in Latin names, numerals and abbreviations.
- Common punctuation shared across scripts.
- v2 additions (:data:`V2_ADDITIONS`): U+2024 ONE DOT LEADER, the ASE
  transcripts' abbreviation dot (``Ա․``, ``Գրկ․``), and ``°``. v1's error
  analysis (``tetrak`` repo, ``product/research/
  armenian-v1-error-analysis.md``) found U+2024 absent from v1's charset
  entirely, so the training pipeline's "drop any crop containing an
  out-of-charset character" step silently filtered every crop containing
  it and v1 could never emit it at all — 518 words (5.8% of the ten
  evaluation pages' expected words) unwinnable by construction. Appended
  at the end of :func:`character_list` rather than folded into
  :data:`ARMENIAN_PUNCTUATION`, so the existing charset's prefix and the
  ``ա`` index the v1 tests pin are untouched by the append itself — the
  charset is still a new, incompatible version (see below), just a
  minimal diff against v1's.

The space character is a flag (:data:`INCLUDE_SPACE`) rather than a fact:
whether EasyOCR's trainer and inference path expect it in
``character_list`` is confirmed at spike time (stage 3 in the README), and
the default may flip then.
"""

from __future__ import annotations

from collections import Counter

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

# Confirmed at spike time against what the EasyOCR trainer actually
# expects; see the module docstring.
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
    )
    if include_space:
        characters += " "
    return characters


def num_class(include_space: bool | None = None) -> int:
    """The CTC output size: every character plus the blank token at index 0."""
    return len(character_list(include_space)) + 1


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
