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

The space character is a flag (:data:`INCLUDE_SPACE`) rather than a fact:
whether EasyOCR's trainer and inference path expect it in
``character_list`` is confirmed at spike time (stage 3 in the README), and
the default may flip then.
"""

from __future__ import annotations

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
    )
    if include_space:
        characters += " "
    return characters


def num_class(include_space: bool | None = None) -> int:
    """The CTC output size: every character plus the blank token at index 0."""
    return len(character_list(include_space)) + 1
