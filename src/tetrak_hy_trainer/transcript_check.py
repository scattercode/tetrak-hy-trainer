"""Was this page's transcript checked against its scan?

A page marked proofread on Wikisource is supposed to have been read against
the scan by a person. Some were not. ProofreadPage fills a new page from the
OCR text layer inside the scanned file, and a contributor can tidy the
obvious damage, mark the page proofread and move on. What is left reads like
Armenian and is wrong: ``մեղմիին`` for ``մեղմէին``, ``չշաղել`` for ``քաղել``.
Page 232 of Abovyan's first volume carried about thirty such words in 216
and had been "proofread" since its only revision in 2017 (brief 023).

Such a page costs twice. As an evaluation page it charges every engine for
reading the print correctly. As a training page it is worse, because a crop
only teaches when the reader disagreed with the transcript and the
transcript's label was kept: exactly the words an unchecked page gets wrong.

This module measures a page from what the crop harvester already has in
hand. A transcript word is *suspect* when our reader read different letters
there and a second, unrelated reader -- stock Tesseract -- read our
reader's word on the page and not the transcript's. The word list has one
job, a veto: where the transcript's word is common and the readers' is
rare, the two readers share a confusion (``դ`` and ``գ``, ``զ`` and ``ղ``)
and the transcript stands.

That is not proof for a single word. The *rate* per page is what separates
pages. Measured on 167 evaluation pages against the count of words our
reader and a third engine agree on against the transcript -- the test
checked by hand on page 232, seventeen of seventeen -- it correlates 0.74,
and at the default limit it leaves out sixteen pages: fourteen of the
thirty-two worst, one of the eighty-five cleanest.

The word list is deliberately not evidence *for* the reader. A first
version counted a rare transcript word against a common reading, and it
left out four of the five dictionary pages: a page our reader finds hard
gives many misreadings that happen to be words, and those are the pages
the model most needs. Requiring the second reader removed that.

The verdict is for the whole page. A page that fails is unchecked, so its
other words cannot be trusted either: short words such as ``ի`` for ``է``
never show up as suspects at all.
"""

from __future__ import annotations

import difflib
import gzip
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Armenian letters only: U+0559 to U+055F are marks and punctuation, and a
# word must not differ from itself by an emphasis mark or a comma.
_NOT_A_LETTER = re.compile(r"[^Ա-Ֆա-և]")

# A word at or under RARE occurrences in the word lists is one the corpus
# barely knows; at or over COMMON it is established. The gap between them
# is deliberate: a word seen five times decides nothing.
RARE = 2
COMMON = 10

# The second reader's agreement counts only for words this long. Both
# readers confuse the short function words the same way (``և`` read as
# ``ն``, ``կը`` as ``կր``), and agreement there is the shared confusion.
MIN_SECOND_READER_LETTERS = 3

# Suspect words per hundred transcript words at which a page is treated as
# unchecked. Set from scripts/calibrate_transcript_screen.py (brief 023):
# at 1.0 the validated sources lose up to a fifth of their pages, which
# their transcripts do not deserve; at 1.5 they lose two or three in a
# hundred while Abovyan's first two volumes lose more than a third.
DEFAULT_MAX_SUSPECTS_PER_HUNDRED = 1.5

# Under this many words a rate means nothing: one suspect on a title page
# is not evidence about anything.
MIN_WORDS_FOR_VERDICT = 40


def letters(word: str) -> str:
    """The word's Armenian letters, lower case, with ``և`` spelt ``եւ``."""
    return _NOT_A_LETTER.sub("", word).lower().replace("և", "եւ")


@dataclass(frozen=True)
class Suspect:
    """A transcript word the evidence says the page does not print."""

    transcript: str
    reader: str


@dataclass(frozen=True)
class PageCheck:
    """What the screen found on one page."""

    words: int
    suspects: tuple[Suspect, ...]

    @property
    def per_hundred(self) -> float:
        return 100 * len(self.suspects) / self.words if self.words else 0.0

    def unchecked(self, limit: float = DEFAULT_MAX_SUSPECTS_PER_HUNDRED) -> bool:
        """True when the page has too many suspect words to train on."""
        return self.words >= MIN_WORDS_FOR_VERDICT and self.per_hundred >= limit


def load_frequency(paths: Iterable[Path]) -> Counter[str]:
    """Word counts from one or more ``word<TAB>count`` lists, keyed by letters."""
    frequency: Counter[str] = Counter()
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                word, _, rest = line.rstrip("\n").partition("\t")
                count = rest.split("\t")[0]
                if count.isdigit():
                    frequency[letters(word)] += int(count)
    frequency.pop("", None)
    return frequency


def aligned_words(transcript: Sequence[str], reader: Sequence[str]) -> list[tuple[str, str]]:
    """Pair each transcript word with the word the reader has in its place.

    Only one-for-one replacements are paired. Where the reader split a word,
    joined two or lost a line, there is no saying which word stands for
    which, and a guess would manufacture suspects.
    """
    left = [word for word in transcript if letters(word)]
    right = [word for word in reader if letters(word)]
    matcher = difflib.SequenceMatcher(
        None, [letters(w) for w in left], [letters(w) for w in right], autojunk=False
    )
    pairs = []
    for operation, a0, a1, b0, b1 in matcher.get_opcodes():
        if operation == "replace" and a1 - a0 == b1 - b0:
            pairs.extend(zip(left[a0:a1], right[b0:b1], strict=True))
    return pairs


def check_page(
    transcript: str,
    reader: str,
    second_reader: Iterable[str],
    frequency: Mapping[str, int] | None = None,
) -> PageCheck:
    """Count the transcript words two readers dispute.

    *transcript* and *reader* are the page's text in reading order.
    *second_reader* is the words another engine read anywhere on the page;
    its order is not used. *frequency* lets the word list veto a suspect
    where it knows the transcript's word and not the readers'.
    """
    frequency = frequency or {}
    transcript_words = [word for word in transcript.split() if letters(word)]
    second = {letters(word) for word in second_reader}
    found = []
    for theirs, ours in aligned_words(transcript_words, reader.split()):
        transcript_letters, reader_letters = letters(theirs), letters(ours)
        if (
            transcript_letters == reader_letters
            or len(reader_letters) < MIN_SECOND_READER_LETTERS
            or reader_letters not in second
            or transcript_letters in second
        ):
            continue
        shared_confusion = (
            frequency.get(transcript_letters, 0) >= COMMON
            and frequency.get(reader_letters, 0) <= RARE
        )
        if not shared_confusion:
            found.append(Suspect(theirs, ours))
    return PageCheck(words=len(transcript_words), suspects=tuple(found))


def tesseract_available() -> bool:
    """True when the ``tesseract`` command is on the path."""
    return shutil.which("tesseract") is not None


def tesseract_words(image_path: Path, language: str = "hye", timeout: float = 300) -> list[str]:
    """Every word stock Tesseract reads on the page.

    The second reader has to be independent of ours and free to use on
    training material, which the stock ``hye`` model is and the
    non-commercial models are not. Its order on the page is not trusted;
    callers use the words as a set.
    """
    environment = dict(os.environ)
    # Homebrew's own tessdata holds the stock model. A TESSDATA_PREFIX left
    # over from a baseline run would point at someone else's.
    environment.pop("TESSDATA_PREFIX", None)
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["tesseract", str(image_path), "stdout", "-l", language],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tesseract failed on {image_path}: {result.stderr.strip()[:200]}")
    return result.stdout.split()
