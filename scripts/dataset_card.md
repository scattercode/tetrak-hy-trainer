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

The dataset has two configurations:

- **`corpus`** — 1,190 proofread pages of the Armenian Soviet
  Encyclopedia, as plain text with full Wikisource provenance.
- **`crops`** — the v0 synthetic pre-training set: 181,800 rendered
  word crops with transcriptions.

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
sampled from the corpus above and rendered in Noto Sans Armenian and
Noto Serif Armenian (both under the SIL Open Font Licence 1.1) onto
paper-like grounds, down to real scan sizes.

The recipe lives in tetrak-hy-trainer
(`scripts/train_synthetic.py`; v0 reproduces with
`--line-tokens-max 1 --no-augment --min-size 36`), so this
configuration can be regenerated from the corpus. We publish it so
that published weights can be reproduced and audited without
re-rendering.

## Licences and attribution

We label the dataset as a whole `cc-by-sa-4.0`. Per configuration:

| Configuration | Licence | Why |
|---|---|---|
| `corpus` | CC BY-SA 3.0 | Verbatim text of the encyclopedia as hosted by Armenian Wikisource, which states CC BY-SA 3.0. |
| `crops` | CC BY-SA 4.0 | The rendered crops contain the corpus text verbatim, so share-alike carries through; CC BY-SA 3.0 permits adaptations to be shared under CC BY-SA 4.0. |

Please attribute the Armenian Soviet Encyclopedia and the Armenian
Wikisource contributors who transcribed and proofread it. The Noto
fonts are used for rendering only and are not redistributed here.

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
