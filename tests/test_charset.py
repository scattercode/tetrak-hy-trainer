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
    assert characters.endswith(
        charset.V2_ADDITIONS + charset.V3_ADDITIONS + charset.V4_ADDITIONS + charset.V5_ADDITIONS
    )


def test_v4_additions_are_present() -> None:
    """The angle brackets the encyclopedias and critical editions print,
    and the scientific superscript three and plus-minus."""
    characters = charset.character_list()
    for character in "<>³±":
        assert character in characters


def test_v5_additions_are_present() -> None:
    """Brief 014 Stage 1: the apostrophe of elision the classical presses
    print, and the Russian alphabet in both cases including ё/Ё."""
    characters = charset.character_list()
    assert "՚" in characters
    for character in "АБВЯЁабвяё":
        assert character in characters


def test_v5_is_the_apostrophe_and_sixty_six_cyrillic_classes() -> None:
    """Pins the v5 suffix: 1 + 33 + 33 characters, num_class 179 -> 246.
    The pre-1918 letters stay out (Stage 0.3: no source prints them)."""
    assert len(charset.CYRILLIC_UPPER) == 33
    assert len(charset.CYRILLIC_LOWER) == 33
    assert charset.V5_ADDITIONS.startswith("՚")
    assert len(charset.V5_ADDITIONS) == 67
    assert charset.num_class() == 246
    for pre_reform in "ѣіѳѵѢІѲѴ":
        assert pre_reform not in charset.character_list()


def test_cyrillic_lookalikes_are_their_own_classes() -> None:
    """A Cyrillic а is not a Latin a: the token's script decides the label
    (Stage 0.3), so both glyphs need a class and must not collapse."""
    characters = charset.character_list()
    for latin, cyrillic in (("a", "а"), ("o", "о"), ("c", "с"), ("p", "р"), ("e", "е"), ("x", "х")):
        assert latin != cyrillic
        assert characters.index(latin) != characters.index(cyrillic)


class TestTokenScript:
    """Brief 014 Stage 0.3: a token's script is decided by its letters."""

    def test_scripts_are_read_from_letters_only(self) -> None:
        assert charset.scripts_of("Գրկ․") == {"armenian"}
        assert charset.scripts_of("Исраэлян") == {"cyrillic"}
        assert charset.scripts_of("Lazarev,") == {"latin"}
        assert charset.scripts_of("1974․") == frozenset()

    def test_a_wholly_cyrillic_token_is_kept(self) -> None:
        assert not charset.mixes_cyrillic("Москва")
        assert not charset.mixes_cyrillic("«Смена»")

    def test_a_token_mixing_cyrillic_with_another_alphabet_is_dropped(self) -> None:
        assert charset.mixes_cyrillic("Mосква")  # Latin M, the keyboard slip
        assert charset.mixes_cyrillic("Иսրաէլ")

    def test_armenian_with_latin_is_not_this_policy_s_business(self) -> None:
        assert not charset.mixes_cyrillic("Ե․-Yerevan")


def test_strays_counts_out_of_charset_characters() -> None:
    """The check that would have caught U+2024 before it cost a training
    run: count what the corpus contains that the charset cannot emit."""
    # Greek, not Cyrillic: Cyrillic joined the charset in v5.
    found = charset.strays("հայ π. § 5 «ok»")
    assert found["π"] == 1
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
    en dash, so it is normalised in wikisource.normalise_transcript
    instead. (Angle brackets were here until v4: the scans showed most of
    them are print, not guillemets in disguise.)"""
    characters = charset.character_list()
    assert "−" not in characters


def test_a_stamped_folder_matches_the_current_charset(tmp_path) -> None:
    charset.stamp(tmp_path)
    assert charset.stamp_matches(tmp_path)


def test_an_unstamped_or_older_folder_does_not_match(tmp_path) -> None:
    """Real crops harvested before v4 carry no stamp; one made under an
    older charset carries a different list. Both must be refused."""
    assert not charset.stamp_matches(tmp_path)
    (tmp_path / charset.STAMP_FILE).write_text(
        charset.character_list().removesuffix(charset.V5_ADDITIONS + " ") + " ",
        encoding="utf-8",
    )
    assert not charset.stamp_matches(tmp_path)
