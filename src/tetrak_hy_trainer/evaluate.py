"""Score a packaged model against harvested real pages.

This is the measurement every published figure comes from, so it lives in
the package rather than in whichever script needed it first. It was
written inside ``scripts/train_synthetic.py`` -- fine while one script
scored one evaluation set at the end of a run, and a drift hazard the
moment a second caller wanted the same number, because "the same metric"
here means the same *reader configuration* and the same line joining as
Tetrak's easyocr backend, not merely the same two functions from
:mod:`tetrak_hy_trainer.accuracy`.

A second copy would keep agreeing right up until one of them changed
``paragraph`` or stopped joining boxes with newlines, at which point two
numbers in the same table would silently stop being comparable.

The fold belongs here for the same reason. ``fold_script`` ships in
``tetrak-easyocr-armenian`` and every consumer applies it, so raw output
is not what a reader of this model actually sees; :func:`score_pages`
takes a ``transform`` so the folded figure comes from this one code path
too, rather than from a second loop somewhere that reads the predictions
back and re-scores them slightly differently.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tetrak_hy_trainer import packaging
from tetrak_hy_trainer.accuracy import character_similarity, word_recall


@dataclass(frozen=True)
class PageScore:
    """One page's two metrics.

    ``character_similarity`` is order-sensitive and collapses when the
    reading order is wrong; ``word_recall`` is order-insensitive and does
    not. Reporting both is what makes a multi-column failure legible as a
    layout problem rather than a recognition one.
    """

    page_number: int
    character_similarity: float
    word_recall: float


@dataclass(frozen=True)
class EvaluationResult:
    """Every page scored, plus the average the project quotes."""

    name: str
    pages: tuple[PageScore, ...]

    @property
    def mean_character_similarity(self) -> float:
        return sum(page.character_similarity for page in self.pages) / len(self.pages)

    @property
    def mean_word_recall(self) -> float:
        return sum(page.word_recall for page in self.pages) / len(self.pages)

    def __len__(self) -> int:
        return len(self.pages)


def load_reader(bundle: Path):
    """An EasyOCR reader wired to a packaged bundle.

    ``["en"]`` rather than ``["hy"]`` is not a placeholder: EasyOCR ships
    no ``hy_char.txt``, so ``["hy"]`` raises FileNotFoundError, and the
    language list is inert for a custom recognition network anyway.
    """
    import easyocr

    return easyocr.Reader(
        ["en"],
        recog_network=packaging.NETWORK_NAME,
        user_network_directory=str(bundle),
        model_storage_directory=str(bundle),
        verbose=False,
    )


def read_page(reader, image: Path) -> str:
    """The model's text for one page, joined as Tetrak's backend joins it."""
    return "\n".join(reader.readtext(str(image), detail=0, paragraph=False))


def score_pages(
    bundle: Path,
    eval_dir: Path,
    reader=None,
    transform: Callable[[str], str] | None = None,
    on_page: Callable[[PageScore], None] | None = None,
) -> EvaluationResult:
    """Score every page of *eval_dir* that has an image.

    Args:
        bundle: A packaged bundle directory.
        eval_dir: A harvest laid out as ``manifest.json``, ``text/`` and
            ``images/`` -- what ``build_eval_sets.py`` produces.
        reader: An existing reader, so scoring several evaluation sets
            with one model does not reload the weights per set.
        transform: Applied to the prediction *and* the expected text
            before scoring. Used for the fold, which has to touch both
            sides for the comparison to mean anything.
        on_page: Called with each page's score as it is computed, for
            callers that stream progress into a log.

    Returns:
        The per-page scores and their averages. Pages whose image is
        absent are skipped rather than scored as zero -- a missing scan
        is a harvesting gap, and averaging a zero into it would report a
        recognition failure that never happened.
    """
    if reader is None:
        reader = load_reader(bundle)
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    scores: list[PageScore] = []
    for entry in manifest["pages"]:
        image = eval_dir / "images" / f"{entry['page_number']}.jpg"
        if not image.exists():
            continue
        expected = (eval_dir / entry["text"]).read_text(encoding="utf-8")
        text = read_page(reader, image)
        if transform is not None:
            text, expected = transform(text), transform(expected)
        score = PageScore(
            page_number=entry["page_number"],
            character_similarity=character_similarity(text, expected),
            word_recall=word_recall(text, expected),
        )
        scores.append(score)
        if on_page is not None:
            on_page(score)
    return EvaluationResult(name=eval_dir.name, pages=tuple(scores))
