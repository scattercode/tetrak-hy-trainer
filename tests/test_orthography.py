"""The orthography classifier reads the page, not the title."""

from tetrak_hy_trainer import orthography

# Reformed and classical spellings of the same sentence, padded to pass
# MIN_LETTERS by repetition. The classical line carries free-standing ւ in
# "իւր", "նաւ", "եւ" and "հաւատք"; the reformed line has ւ only inside ու.
REFORMED = "Նա իր նավով գնաց և հավատք ուներ, որ երևանի մեջ ամեն մարդ կարդում է։ "
CLASSICAL = "Նա իւր նաւով գնաց եւ հաւատք ունէր, որ Երեւանի մէջ ամէն մարդ կը կարդայ։ "


def test_reformed_text_is_reformed():
    text = REFORMED * 12
    m = orthography.markers(text)
    assert m.letters >= orthography.MIN_LETTERS
    assert m.free_yiwn <= orthography.REFORMED_YIWN
    assert m.verdict == orthography.REFORMED


def test_classical_text_is_classical():
    text = CLASSICAL * 12
    m = orthography.markers(text)
    assert m.free_yiwn >= orthography.CLASSICAL_YIWN
    assert m.verdict == orthography.CLASSICAL


def test_ou_digraph_is_not_a_free_yiwn():
    assert orthography.FREE_YIWN.findall("ուսուցիչ Ուրարտու") == []
    assert len(orthography.FREE_YIWN.findall("իւր եւ նաւ")) == 3


def test_ligature_habit_does_not_decide():
    # Classical spelling set with the և ligature, as Հայկական տպագրութիւն is.
    text = CLASSICAL.replace("եւ", "և") * 12
    m = orthography.markers(text)
    assert m.two_letter_ew == 0 and m.ligature_ew > 0
    assert m.verdict == orthography.CLASSICAL


def test_short_text_is_unknown():
    assert orthography.classify("Նաւ եւ իւր") == orthography.UNKNOWN


def test_as_dict_carries_the_verdict():
    d = orthography.markers(REFORMED * 12).as_dict()
    assert d["verdict"] == orthography.REFORMED
    assert set(d) >= {"letters", "free_yiwn", "final_y", "inner_e"}
