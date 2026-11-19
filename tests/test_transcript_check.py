"""The transcript screen: which words count as suspect, and when a page fails.

No detector and no Tesseract are needed: the screen is a pure function of
the transcript, what our reader read, the word counts and the words a
second reader saw. The examples are from Abovyan's first volume, page 232,
whose Wikisource transcript was the scanned file's OCR layer lightly tidied
(brief 023).
"""

from __future__ import annotations

import gzip

import pytest

from tetrak_hy_trainer import transcript_check as check

FREQUENCY = {
    "մեղմէին": 40,
    "քաղել": 120,
    "արդէն": 900,
    "արդին": 30,  # a real word too: the ի-for-է slips often are, so no rule may need it rare
    "մանաւանդ": 400,
    "զհոգի": 60,
    "իմ": 5000,
    "որ": 9000,
}


def test_letters_ignore_marks_case_and_the_ew_ligature():
    assert check.letters("Բարեկա՛մք,") == "բարեկամք"
    assert check.letters("և") == check.letters("Եւ") == "եւ"
    assert check.letters("1832.") == ""


def test_two_readers_against_the_transcript_make_a_suspect():
    page = check.check_page(
        "Ձայնք, որ մեղմիին զհոգի իմ",
        "Ձայնք, որ մեղմէին զհոգի իմ",
        second_reader=["Ձայնք,", "որ", "մեղմէին", "զհոգի", "իմ"],
        frequency=FREQUENCY,
    )
    assert [(s.transcript, s.reader) for s in page.suspects] == [("մեղմիին", "մեղմէին")]
    assert page.words == 5


def test_a_mark_or_a_comma_is_not_a_difference():
    page = check.check_page("Բարեկամք սիրունք,", "Բարեկա՛մք սիրունք.", ["Բարեկա՛մք", "սիրունք."])
    assert page.suspects == ()


def test_our_reader_alone_is_not_evidence():
    # The word list would have sided with the reading here. It is not asked:
    # a page our reader finds hard gives many misreadings that are words,
    # and counting them left out four of five dictionary pages.
    page = check.check_page(
        "Ձայնք, որ մեղմիին զհոգի իմ",
        "Ձայնք, որ մեղմէին զհոգի իմ",
        second_reader=["Ձայնք,", "որ", "զհոգի", "իմ"],
        frequency=FREQUENCY,
    )
    assert page.suspects == ()


def test_the_word_list_is_not_needed():
    page = check.check_page("Օրհասն արդին զիս", "Օրհասն արդէն զիս", ["Օրհասն", "արդէն", "զիս"])
    assert [s.reader for s in page.suspects] == ["արդէն"]


def test_a_second_reader_that_also_has_the_transcripts_word_decides_nothing():
    page = check.check_page("Օրհասն արդին զիս", "Օրհասն արդէն զիս", ["արդէն", "արդին"])
    assert page.suspects == ()


def test_two_readers_sharing_a_confusion_do_not_outvote_the_word_list():
    # Both engines read դ as գ; the word list knows which is the word.
    page = check.check_page(
        "մանաւանդ որ", "մանաւանգ որ", second_reader=["մանաւանգ", "որ"], frequency=FREQUENCY
    )
    assert page.suspects == ()
    # Without the word list the same two readers are believed.
    assert len(check.check_page("մանաւանդ որ", "մանաւանգ որ", ["մանաւանգ", "որ"]).suspects) == 1


def test_short_words_are_never_suspects():
    # և read as ն by both readers is the readers' habit, not a typo.
    page = check.check_page("քնար և սրինգ", "քնար ն սրինգ", second_reader=["ն"])
    assert page.suspects == ()


def test_only_one_for_one_replacements_are_paired():
    # The transcript split one printed word in two; no pair, no suspect.
    page = check.check_page("քնար մնա սրե՛լ, ինձ", "քնար մնասցեն ինձ", ["մնասցեն"])
    assert page.suspects == ()
    assert check.aligned_words(["ա", "բ", "գ"], ["ա", "դ", "գ"]) == [("բ", "դ")]


def test_verdict_needs_enough_words_and_enough_suspects():
    suspect = check.Suspect("մեղմիին", "մեղմէին")
    assert check.PageCheck(words=200, suspects=(suspect,) * 3).unchecked()
    assert not check.PageCheck(words=200, suspects=(suspect,) * 2).unchecked()
    # A title page: three suspects in twelve words is not a rate.
    assert not check.PageCheck(words=12, suspects=(suspect,) * 3).unchecked()
    assert check.PageCheck(words=200, suspects=(suspect,) * 2).unchecked(limit=1.0)
    assert check.PageCheck(words=0, suspects=()).per_hundred == 0.0


def test_load_frequency_sums_lists_and_keys_by_letters(tmp_path):
    first = tmp_path / "wordlist.tsv.gz"
    with gzip.open(first, "wt", encoding="utf-8") as handle:
        handle.write("և\t10\nԵւ\t5\nարդէն\t7\textra\n")
    second = tmp_path / "classical.tsv"
    second.write_text("արդէն\t3\nno-count\n", encoding="utf-8")
    frequency = check.load_frequency([first, second])
    assert frequency["եւ"] == 15
    assert frequency["արդէն"] == 10
    assert "" not in frequency


def test_tesseract_words_ignores_a_borrowed_tessdata_prefix(monkeypatch, tmp_path):
    # A baseline run leaves TESSDATA_PREFIX pointing at a non-commercial
    # model's directory; the screen's second reader must be the stock one.
    seen = {}

    class Result:
        returncode = 0
        stdout = "Ձայնք որ\nմեղմէին"
        stderr = ""

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs["env"]
        return Result()

    monkeypatch.setenv("TESSDATA_PREFIX", "/somewhere/else")
    monkeypatch.setattr(check.subprocess, "run", fake_run)
    words = check.tesseract_words(tmp_path / "page.jpg")
    assert words == ["Ձայնք", "որ", "մեղմէին"]
    assert "TESSDATA_PREFIX" not in seen["env"]
    assert seen["argv"][-2:] == ["-l", "hye"]


def test_tesseract_failure_is_an_error_not_an_empty_page(monkeypatch, tmp_path):
    # An empty second reading would silently turn the screen into the
    # word-list rule alone for that page.
    class Result:
        returncode = 1
        stdout = ""
        stderr = "Error opening data file"

    monkeypatch.setattr(check.subprocess, "run", lambda argv, **kwargs: Result())
    with pytest.raises(RuntimeError, match="tesseract failed"):
        check.tesseract_words(tmp_path / "page.jpg")
