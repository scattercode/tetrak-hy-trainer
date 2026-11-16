---
license: cc-by-sa-4.0
language:
- hy
- en
- ru
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

The dataset has one configuration per synthetic crop set that trained
a released model, plus the corpus the first two were rendered from:

| Configuration | Crops (train + validation) | Trained | Charset |
|---|---|---|---|
| `corpus` | 1,190 pages of text | — | — |
| `crops` | 180,000 + 1,800 word crops | v0 | 168 classes |
| `crops-v1` | 175,500 + 1,500 line crops | v1 | 168 classes |
| `crops-v4` | 351,000 + 3,000 line crops | v4, and v5 and v6 through fine-tunes | 175 classes |
| `crops-v6-caps` | 58,500 + 750 all-caps lines | v6 | 175 classes |
| `crops-v7` | 175,500 + 1,500 line crops | v7 | 246 classes |
| `crops-v7-caps` | 58,500 + 750 all-caps lines | v7 | 246 classes |
| `crops-v7-latin` | 58,500 + 750 English lines | v7 | 246 classes |

Every crop configuration has `image` and `text` columns in `train` and
`validation` splits, greyscale crops rendered onto paper-like grounds
at real scan sizes and degraded to look scanned. The class count is
the recogniser's CTC head: the charset the set was rendered under, plus
the blank. A set rendered under a wider charset carries characters a
narrower model has no class for, so take a set and a model of the same
charset together (the model repository's `tetrak_hy.yaml` lists it).

**The real crops are not published.** From v3 on, every released model is
fine-tuned on crops cut from the scanned pages themselves, aligned to
their transcripts; v7's 217,362 real crops come from 2,136 scans of 42
works. Those are fragments of the scans rather than renders of the
text, so they stay out of this dataset. They are reproducible from the
transcripts with the trainer's `scripts/harvest_real_crops.py`, and each
model's `provenance.json` says how many were used and from where.

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

## The crops-v4 configuration

The pre-train set behind v4, and so behind v5 and v6, which are
fine-tunes of it: 351,000 training and 3,000 validation line crops.
120,000 line samples of 1 to 4 consecutive tokens were drawn from about
7,400 proofread Wikisource pages of fourteen works — the Armenian Soviet
Encyclopedia (volumes 1 and 3 to 13), the collected works of Otyan,
Totovents, Baronian and Tumanyan, Faustus of Byzantium (1968), a popular
medical encyclopedia and an Armenian–English dictionary — split as
`crops-v1` is (every 40th sample held out and rendered once, the rest
rendered three times) and drawn in fifteen faces: the eight GHEA faces,
the four Arian AMU faces, Noto Sans and Noto Serif Armenian, and
Mshtakan. The charset is 174 characters plus the blank, 175 classes,
which added the ellipsis, square brackets, the numero sign and
superscript two to `crops-v1`'s. Volume 2 of the encyclopedia and the
registered held-out pages of every other work
(`tetrak_hy_trainer.heldout`) are excluded.

v6 trained on this set with one change of label: a colon closing an
Armenian word is relabelled as the Armenian full stop `։`, which is what
the page prints (`scripts/relabel_dataset.py`). The crops are the same;
the relabelled copy is not published separately.

## The crops-v6-caps configuration

58,500 training and 750 validation lines set entirely in capitals
(`scripts/render_synthetic_set.py --upper`), drawn from the same works
and faces as `crops-v4` under the same charset, so that v6 could read the
all-capital headwords of the encyclopedias. A tenth of each of v6's
fine-tune batches.

## The crops-v7 configuration

v7's pre-train set, rendered from scratch over every proofread page of
58 works on Armenian Wikisource — the fourteen above plus 44 harvested
for classical orthography, among them Abovyan, Khrimian, Kajaznuni,
Khatisian, Toranian, Teodik's history of Armenian printing and
Baronian's *National bigshots*. 60,000 line samples of 1 to 4 tokens,
held out and repeated as `crops-v4`'s, so 175,500 training and 1,500
validation crops. Two shares are set rather than left to chance: a
quarter of the lines (15,220) come from works in classical
orthography, which the renderer judges from each work's own text, and a
twentieth (3,000) are lines that mix Armenian with Russian, the
apparatus of Soviet editions. The charset is 245 characters plus the
blank, 246 classes: `crops-v4`'s plus `<`, `>`, `³`, `±`, the Armenian
apostrophe of elision `՚` and the Russian alphabet in both cases
including `ё`/`Ё`. The twelve GHEA and Arian AMU faces draw Cyrillic; the
other three are kept off any line with a Russian word. The recipe is
`scripts/train_synthetic.py --classical-share 0.25 --mixed-share 0.05`
over every harvest, and each rendered set records its recipe and seed in
a `recipe.json` beside its labels.

## The crops-v7-caps configuration

As `crops-v6-caps`, rendered over all 58 works under v7's charset:
58,500 training and 750 validation all-capital lines. A tenth of each of
v7's fine-tune batches.

## The crops-v7-latin configuration

58,500 training and 750 validation lines of English, so that the Latin
the bilingual dictionary prints is read as Latin rather than as the
Cyrillic it shares glyphs with. 30,000 lines of up to three tokens
were sampled from six public-domain novels in their Project Gutenberg
plain-text editions, with the Gutenberg header and licence stripped —
Austen's *Pride and Prejudice*, Dickens's *Great Expectations* and *A
Tale of Two Cities*, Conan Doyle's *The Adventures of Sherlock Holmes*,
Darwin's *On the Origin of Species* and Melville's *Moby-Dick* — one
line in ten set in capitals, rendered twice each in the same fifteen
faces under v7's charset. A tenth of each of v7's fine-tune batches.

## Licences and attribution

We label the dataset as a whole `cc-by-sa-4.0`. Per configuration:

| Configuration | Licence | Why |
|---|---|---|
| `corpus` | CC BY-SA 3.0 | Verbatim text of the encyclopedia as hosted by Armenian Wikisource, which states CC BY-SA 3.0. |
| `crops`, `crops-v1` | CC BY-SA 4.0 | The rendered crops contain the corpus text verbatim, so share-alike carries through; CC BY-SA 3.0 permits adaptations to be shared under CC BY-SA 4.0. |
| `crops-v4`, `crops-v6-caps`, `crops-v7`, `crops-v7-caps` | CC BY-SA 4.0 | The same reasoning: the text is proofread Armenian Wikisource transcripts of fourteen and then 58 works, all hosted under CC BY-SA 3.0. |
| `crops-v7-latin` | CC BY-SA 4.0 | The underlying text is public domain (six novels in their Project Gutenberg editions, with the Gutenberg header and licence removed, so no Project Gutenberg trademark use is made); the renders are released under the dataset's licence for uniformity, and no attribution is owed beyond the authors'. |

Please attribute the Armenian Soviet Encyclopedia, the other works'
authors and editions, and the Armenian Wikisource contributors who
transcribed and proofread them. The fonts are used for rendering only;
no font file is redistributed here. The Noto and Arian AMU faces are
under the SIL Open Font Licence 1.1; the GHEA faces are the Republic of
Armenia's free-use official faces (Armenian National Book Chamber);
Mshtakan ships with macOS.

## Versioning

Dataset revisions that trained published weights are tagged with the
weights' version (`v7` is the revision carrying the three v7 sets). Pin
a revision when downloading — with `datasets`:

```python
from datasets import load_dataset

corpus = load_dataset("tetrak/armenian-ocr-crops", "corpus", revision="v7")
crops = load_dataset("tetrak/armenian-ocr-crops", "crops-v7", revision="v7")
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
