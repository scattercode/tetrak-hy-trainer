---
license: cc-by-sa-4.0
language:
- hy
task_categories:
- image-to-text
tags:
- ocr
- text-recognition
- armenian
- synthetic-data
- easyocr
pretty_name: Tetrak Armenian OCR crops
---

# Tetrak Armenian OCR crops

Training data for `tetrak_hy`, the Armenian text recogniser we are
building as an EasyOCR custom model in
[tetrak-hy-trainer](https://github.com/scattercode/tetrak-hy-trainer)
for [Tetrak](https://tetrak.dev/), an OCR pipeline for community
archives.

The dataset has three configurations:

- **`corpus`** — 1,190 proofread pages of the Armenian Soviet
  Encyclopedia, as plain text with full Wikisource provenance.
- **`crops`** — the v0 synthetic pre-training set: 181,800 rendered
  word crops with transcriptions.
- **`crops-v1`** — the v1 synthetic training set: 177,000 rendered
  line crops, degraded to look like scans. This is the set that
  trained the published v1 weights.

## The corpus configuration

Plain text of proofread pages from the Armenian Soviet Encyclopedia
(Հայկական սովետական հանրագիտարան, 1974–1987), harvested from
[Armenian Wikisource](https://hy.wikisource.org/) — volumes 1, 3, 4,
5 and 6 at the time of harvesting. Only pages proofread to Wikisource
quality level 3 or above were taken.

Fields per page: `volume`, `page_number`, `title` (the Wikisource page
title), `pageid`, `revid` (the exact revision harvested, so any page
can be traced back to what Wikisource served at the time), `quality`
(the Wikisource proofread level), `index` (the Wikisource index page
the harvest walked) and `text`.

## The crops configuration

The v0 synthetic pre-training set: 180,000 training and 1,800
validation greyscale word crops, each with its transcription (`image`
and `text` columns, in `train` and `validation` splits). Words are
sampled from the corpus above and rendered onto paper-like grounds,
down to real scan sizes, in three faces: Noto Sans Armenian and Noto
Serif Armenian (both under the SIL Open Font Licence 1.1) and
Mshtakan, which ships with macOS.

The recipe lives in tetrak-hy-trainer
(`scripts/train_synthetic.py`; v0 reproduces with
`--line-tokens-max 1 --no-augment --min-size 36`), so this
configuration can be regenerated from the corpus. We publish it so
that published weights can be reproduced and audited without
re-rendering.

## The crops-v1 configuration

The set that trained the v1 weights: 175,500 training and 1,500
validation greyscale line crops, same columns and splits as `crops`.
Three differences from v0, and they are the whole point of v1:

- **Lines, not words.** Each crop is 1 to 4 consecutive tokens taken
  from a corpus page, so the model sees spaces and line-shaped inputs
  rather than isolated words.
- **Scan scale.** Rendering sizes go down to 18 px, the x-height that
  real page renders actually produce, rather than stopping at 36 px.
- **Degradations on both splits.** A downscale cycle, blur, tone
  shift, small rotation and a JPEG round-trip are applied to the
  validation split as well as training. v0's crisp validation set read
  99.7% while real scans read far worse; a validation set that never
  sees a degradation measures nothing useful.

The 60,000 line samples are split before rendering — every 40th sample
is held out for validation and rendered once, the rest are rendered
three times each with independent degradations, giving 175,500 and
1,500. Fonts are the same three as `crops`. The recipe is
`scripts/train_synthetic.py` in tetrak-hy-trainer, which reproduces
this configuration on its defaults.

## Licences and attribution

We label the dataset as a whole `cc-by-sa-4.0`. Per configuration:

| Configuration | Licence | Why |
|---|---|---|
| `corpus` | CC BY-SA 3.0 | Verbatim text of the encyclopedia as hosted by Armenian Wikisource, which states CC BY-SA 3.0. |
| `crops` | CC BY-SA 4.0 | The rendered crops contain the corpus text verbatim, so share-alike carries through; CC BY-SA 3.0 permits adaptations to be shared under CC BY-SA 4.0. |
| `crops-v1` | CC BY-SA 4.0 | Same reasoning as `crops` — the same corpus text, rendered differently. |

Please attribute the Armenian Soviet Encyclopedia and the Armenian
Wikisource contributors who transcribed and proofread it. The fonts
are used for rendering only; no font file is redistributed here.

## Versioning

Dataset revisions that trained published weights are tagged. Pin a
revision when downloading — with `datasets`:

```python
from datasets import load_dataset

corpus = load_dataset("tetrak/armenian-ocr-crops", "corpus", revision="<tag or commit>")
```

## Acknowledgements

The Armenian Soviet Encyclopedia was
[Daniel Ohanian's](https://mastodon.social/@dohanian) suggestion — when we
asked on Mastodon for Armenian training material with trustworthy
transcriptions, he pointed us at the proofread volumes on Armenian
Wikisource. This dataset grew from that pointer. Thank you, Daniel.

## Related

- [tetrak-hy-trainer](https://github.com/scattercode/tetrak-hy-trainer)
  — the synthesis pipeline, trainer and packaging that consume this
  dataset (Apache 2.0).
- [Tetrak](https://tetrak.dev/) — the OCR pipeline the trained model
  ships in.
- `tetrak/easyocr-armenian` — the model repository for the trained
  weights (Apache 2.0).
