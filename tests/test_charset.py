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
