"""The charset is load-bearing: CTC indices are positional, so these tests
pin its composition and order-stability rather than merely its existence."""

from tetrak_hy_trainer import charset


def test_armenian_alphabet_is_complete() -> None:
    """38 uppercase and 38 lowercase letters — the full modern alphabet,
    generated from the Unicode ranges so no letter can be dropped by typo."""
    assert len(charset.ARMENIAN_UPPER) == 38
    assert len(charset.ARMENIAN_LOWER) == 38
    assert charset.ARMENIAN_UPPER[0] == "Ա"
    assert charset.ARMENIAN_UPPER[-1] == "Ֆ"
    assert charset.ARMENIAN_LOWER[0] == "ա"
    assert charset.ARMENIAN_LOWER[-1] == "ֆ"


def test_the_ew_ligature_is_present() -> None:
    """և (U+0587) is ubiquitous in printed Armenian; membership is settled,
    only the ground-truth normalisation policy remains open."""
    assert "և" in charset.character_list()


def test_no_duplicate_characters() -> None:
    """A duplicate would make two CTC classes for one glyph."""
    characters = charset.character_list()
    assert len(characters) == len(set(characters))


def test_armenian_punctuation_is_covered() -> None:
    characters = charset.character_list()
    for mark in "՝՛՞՜։֊«»":
        assert mark in characters, f"missing Armenian punctuation {mark!r}"


def test_num_class_counts_the_ctc_blank() -> None:
    assert charset.num_class() == len(charset.character_list()) + 1


def test_space_flag_is_respected() -> None:
    assert " " in charset.character_list(include_space=True)
    assert " " not in charset.character_list(include_space=False)


def test_order_is_stable() -> None:
    """The first characters must stay exactly where they are: changing
    order silently re-labels every CTC class and corrupts any trained
    weights. If this test fails, the change needs a new model version —
    see the module docstring."""
    characters = charset.character_list()
    assert characters.startswith("ԱԲԳԴ")
    assert characters.index("ա") == 38


def test_v2_additions_are_present() -> None:
    """U+2024 ONE DOT LEADER (the transcripts' abbreviation dot) and the
    degree sign — see the module docstring's "v2 additions" note."""
    characters = charset.character_list()
    assert "․" in characters
    assert "°" in characters


def test_additions_are_appended_in_version_order() -> None:
    """Each version's additions go on the end, so every earlier charset
    is a prefix of every later one and the ``ա`` index test_order_is_stable
    pins is never disturbed by an append."""
    characters = charset.character_list(include_space=False)
    assert characters.endswith(charset.V2_ADDITIONS + charset.V3_ADDITIONS)


def test_strays_counts_out_of_charset_characters() -> None:
    """The check that would have caught U+2024 before it cost a training
    run: count what the corpus contains that the charset cannot emit."""
    found = charset.strays("հայ п. § 5 «ok»")
    assert found["п"] == 1
    assert found["§"] == 1
    assert "հ" not in found


def test_strays_ignores_whitespace() -> None:
    assert charset.strays("ա\tբ\nգ") == {}


def test_covered_text_has_no_strays() -> None:
    assert charset.strays(charset.character_list()) == {}


def test_v3_additions_are_present() -> None:
    """Genuinely printed characters the corpus-wide diff found: the
    literary ellipsis, encyclopedia editorial brackets, the numero sign
    and superscript two."""
    characters = charset.character_list()
    for mark in "…[]№²":
        assert mark in characters, f"missing {mark!r}"


def test_transcriber_substitutions_stay_out_of_the_charset() -> None:
    """U+2212 would give the model a new homoglyph to confuse with the
    en dash; angle brackets are guillemets in disguise. Both are
    normalised in wikisource.normalise_transcript instead."""
    characters = charset.character_list()
    assert "−" not in characters
    assert "<" not in characters and ">" not in characters
