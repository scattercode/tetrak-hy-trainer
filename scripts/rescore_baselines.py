"""Re-score saved baseline readings with the current metric, without re-running OCR.

``evaluate_baselines.py`` saves every backend's raw reading as
``<eval-dir>/readings/<backend>/<page>.txt`` beside its scores. When the
metric changes -- as it did twice in brief 013, for difflib's autojunk and
then for equating the Armenian full stop with the colon -- this rewrites
each ``baselines*.csv`` from those readings, keeping the recorded timings.
A metric change is then a few seconds' re-score rather than hours of
engines.

A row whose reading is missing (the backend failed on that page, or the
CSV predates saved readings) is left exactly as it was and named on
stdout, so a partly re-scored file never passes for a fully re-scored one.

Prerequisites: Tetrak's venv, for the same reason as
``evaluate_baselines.py`` -- the scores must come from the metric Tetrak
publishes.

Run:
    python scripts/rescore_baselines.py <eval-dir> [<eval-dir> ...]
"""

import csv
import sys
from pathlib import Path

from tetrak_ocr.accuracy import character_similarity, word_recall


def rescore(eval_dir: Path) -> None:
    for path in sorted(eval_dir.glob("baselines*.csv")):
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            # Taken from the header, not the first row: a backend filter that
            # matched nothing leaves a header-only file, which is valid.
            fieldnames = reader.fieldnames or []
        stale = 0
        for row in rows:
            reading = eval_dir / "readings" / row["backend"] / f"{row['page']}.txt"
            if not row["char_sim"] or not reading.exists():
                stale += 1
                continue
            text = reading.read_text(encoding="utf-8")
            expected = (eval_dir / "text" / f"{row['page']}.txt").read_text(encoding="utf-8")
            row["char_sim"] = f"{character_similarity(text, expected):.4f}"
            row["word_recall"] = f"{word_recall(text, expected):.4f}"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        note = f", {stale} row(s) left unscored or without a reading" if stale else ""
        print(f"rescored {path} ({len(rows) - stale} rows{note})")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        rescore(Path(argument))
