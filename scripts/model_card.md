---
license: apache-2.0
language:
- hy
- en
pipeline_tag: image-to-text
tags:
- ocr
- text-recognition
- armenian
- easyocr
- ctc
datasets:
- tetrak/armenian-ocr-crops
---

# tetrak_hy — Armenian text recognition for EasyOCR

An Armenian text recogniser packaged as an
[EasyOCR](https://github.com/JaidedAI/EasyOCR) custom model, trained by
[tetrak-hy-trainer](https://github.com/scattercode/tetrak-hy-trainer)
for [Tetrak](https://tetrak.dev/), an OCR pipeline for community
archives. The architecture is EasyOCR's own generation2 recognition
network (VGG feature extractor, two BiLSTM layers, CTC head), so the
model drops into a stock EasyOCR install.

## Status: v3, alpha

These are the v3 weights: v2 fine-tuned on **real crops cut from
scanned pages**, not rendered ones. **Paired with the fold described
below, v3 reads 0.771 word recall on real scans — the highest measured
here, ahead of `tesseract -l hye` at 0.662 and of Marker at 0.766.**
The model alone reads 0.736; the difference is a one-line post-process
shipped in the companion package, and both figures are given below.

Character similarity is a different story and still a weak one. See
below: it measures reading order more than it measures recognition.

> **Use v3 or later. v0 and v1 carry two defects that v2 fixed.**
>
> - **21% of their training labels were wrapped in quotation marks the
>   images do not show** (36,918 of v1's 175,500 crops). The trainer
>   parses its label file by splitting on the first comma rather than as
>   CSV, so the quoting a CSV writer applies to comma-bearing labels
>   became part of the label. The models learnt it: inserting a
>   quotation mark is the single commonest error in v1's output, ahead
>   of every genuine character confusion. If you are seeing stray `"`
>   in v0 or v1 output, this is why.
> - **They cannot emit U+2024 ONE DOT LEADER**, the abbreviation dot
>   these transcripts use (`Ա․`, `Գրկ․`). It was missing from their
>   charset, so every training crop containing it was silently dropped
>   and the models have no class for it — 5.8% of the evaluation pages'
>   words are unwinnable, and v1 emits the character zero times in
>   6,672 detected boxes.
>
> Both were found by reading the training data rather than the scores,
> which is why they survived two releases. Neither is fixable in v0 or
> v1 without retraining, so those tags stay as they are, defects
> recorded, and later versions supersede them.

### Measured on real scans

Ten pages of the Armenian Soviet Encyclopedia (volume 2, pages
105-114) from Armenian Wikisource, proofread to quality level 4, with
their transcriptions as ground truth. Higher is better for both
figures.

| Backend | Char similarity | Word recall |
|---|---|---|
| **tetrak_hy v3 + `fold_script`** | **0.147** | **0.771** |
| marker | 0.258 | 0.766 |
| tetrak_hy v3 (raw) | 0.147 | 0.736 |
| tetrak_hy v2 + `fold_script` | 0.117 | 0.692 |
| `tesseract -l hye` (auto page mode) | 0.128 | 0.664 |
| `tesseract -l hye` | 0.697 | 0.662 |
| tetrak_hy v2 (raw) | 0.117 | 0.607 |
| tetrak_hy v1 | 0.100 | 0.501 |
| tetrak_hy v0 | 0.075 | 0.274 |
| stock EasyOCR | 0.035 | 0.031 |

**The fold is not retraining and not a trick.** The recognition head has
no language model, so inside an Armenian word it sometimes emits the
visually identical Latin twin of an Armenian character — `h` for `հ`, a
colon for the Armenian full stop `։`. `fold_script`, in the
[tetrak-easyocr-armenian](https://pypi.org/project/tetrak-easyocr-armenian/)
package, folds those back within any token that already contains an
Armenian letter. Applying it is one line, and it is worth +0.035 word
recall on v3:

```python
import tetrak_hy

reader = tetrak_hy.reader()
results = [
    (box, tetrak_hy.fold_script(text), confidence)
    for box, text, confidence in reader.readtext("page.png")
]
```

The char similarity column is mostly not about recognition. It is
dominated by reading order on these two-column pages: Tesseract in
automatic page mode reads words just as well as it does in the row
below (0.664 vs 0.662 word recall) yet its char similarity collapses
from 0.697 to 0.128, close to v3's, purely because the text comes out
in a different order. This evaluation joins detected lines with a
newline in detector order, so any backend that does not serialise
two-column pages into reading order is penalised the same way. Column
handling in the surrounding pipeline lifts that number without
retraining anything.

### What real crops changed

v3 is v2 fine-tuned on 6,097 crops cut from 30 human-proofread scans of
volumes 5 and 6, labelled from their transcripts by detection-assisted
alignment and mixed 50/50 in every batch with v2's synthetic crops, so
the model adapts to real print without forgetting the breadth the
synthetic pre-training bought. Volume 2 — the evaluation set — is
refused by the harvester outright.

This targeted the shape confusions synthetic fonts cannot teach: a
cleanly rendered `հ` looks nothing like a worn one on 1970s letterpress.
Against v2, misread words on the evaluation pages fall by a third and
`հ` read as Latin `h` — the commonest confusion in the whole table —
drops sharply.

A caveat worth stating: **held-out per-crop accuracy is a poor proxy for
page-level recall here.** A 10,000-iteration run of this same fine-tune
reached the same 95% crop accuracy as the 3,000-iteration run shipped
here, while scoring 0.056 *worse* on page word recall. Crop accuracy
plateaus early and then measures overfitting; the page metric is the
one to trust.

### Charset

The charset holds 169 characters plus the CTC blank, 170 classes. v2
added U+2024 ONE DOT LEADER and U+00B0 DEGREE SIGN to v1's set, and v3
inherits it unchanged — a fine-tune keeps its parent's charset.
**A charset change is a new model by construction** — CTC class indices
are positional — so v2 weights cannot be loaded under a v1 `tetrak_hy.yaml`
or the reverse. Always take the `.yaml` and the `.pth` from the same
revision.

### Validation

v3 is validated on **real** crops: 700 of them, from pages held out of
training entirely — split by page, never by crop, since crops from one
page share its paper and its scanning. 95.0% word accuracy, 0.9936
normalised edit distance. See the caveat above about what that figure
does and does not predict.

v2, the model v3 was fine-tuned from, scored 99.333% / 0.9989 on
degraded synthetic validation crops.

### What is next

More real crops, from more volumes. v3 learnt from 30 pages; the
alignment tooling scales to as many as there are proofread scans, and
the yield per page is roughly 200 crops. Reading order is the other
open front, and it is worth more than recognition for the char
similarity column — that belongs in the pipeline around the model
rather than in the weights.

## Files

- `tetrak_hy.pth` — the weights exactly as the trainer saved them
  (keys carry the `module.` prefix EasyOCR's loader expects to
  handle). This is the file EasyOCR loads.
- `model.safetensors` — the same tensors with the `module.` prefix
  stripped, for anything that isn't EasyOCR.
- `tetrak_hy.yaml` — charset, language list and network parameters.
- `tetrak_hy.py` — the architecture module EasyOCR imports by name.
- `provenance.json` — training recipe, dataset revision, charset and
  checksums for this release.

## Use with EasyOCR

Download the three EasyOCR files and place them where EasyOCR looks
for custom models:

```python
from huggingface_hub import hf_hub_download

for filename in ("tetrak_hy.pth", "tetrak_hy.py", "tetrak_hy.yaml"):
    hf_hub_download("tetrak/easyocr-armenian", filename, revision="v3")
```

- `tetrak_hy.yaml` and `tetrak_hy.py` go in the user network
  directory (by default `~/.EasyOCR/user_network/`).
- `tetrak_hy.pth` goes in the model directory (by default
  `~/.EasyOCR/model/`).

Then:

```python
import easyocr

reader = easyocr.Reader(["en"], recog_network="tetrak_hy")
results = reader.readtext("page.png")
```

Note the `["en"]`: with a custom `recog_network`, the language list
selects EasyOCR's dictionaries rather than the model — the recogniser
itself is chosen by `recog_network`, and this model's charset covers
Armenian plus basic Latin, digits and punctuation.

Pin `revision=` when downloading: each weights release is tagged, and
`provenance.json` records the exact dataset revision it was trained
from.

## Training data

v3 is a fine-tune of v2; the paragraph below describes v2's synthetic
pre-training, which v3 inherits. v3 adds 6,097 real crops, which are
not published: they are cut from Armenian Wikisource scans and are
reproducible from the transcripts with the trainer's harvester.

Trained on 175,500 synthetic line crops rendered locally from the same
source as
[tetrak/armenian-ocr-crops](https://huggingface.co/datasets/tetrak/armenian-ocr-crops).
v2's crops are not one of that dataset's published configurations: its
widened charset admits U+2024, so the crops differ from `crops-v1`, and
no `crops-v2` configuration has been uploaded. The recipe is otherwise
v1's — of 1 to 4 consecutive tokens, rendered
from proofread Armenian Soviet Encyclopedia text (Armenian Wikisource,
CC BY-SA) at sizes down to 18 px and degraded with a downscale cycle,
blur, tone shift, small rotation and a JPEG round-trip. Three faces
were used for rendering: Noto Sans Armenian and Noto Serif Armenian
(SIL Open Font Licence 1.1) and Mshtakan, which ships with macOS. No
font file is redistributed here.

`provenance.json` records the recipe, the charset and the checksums for
this release, along with any defects known against it. v1 was trained on
the `crops-v1` configuration and v0 on `crops` — single-word crops,
undegraded — both still published for reproducibility.

## Licence

The weights, like the trainer, are Apache 2.0. The training text is
CC BY-SA; we publish the text itself, share-alike, in the dataset
repository above, and take the position — shared by most of the
ecosystem, though not legally settled — that trained weights are not
a redistribution or adaptation of the training text.

## Related

- [tetrak-hy-trainer](https://github.com/scattercode/tetrak-hy-trainer)
  — synthesis, training and packaging (Apache 2.0).
- [tetrak/armenian-ocr-crops](https://huggingface.co/datasets/tetrak/armenian-ocr-crops)
  — the training data (CC BY-SA 4.0).
- [Tetrak](https://tetrak.dev/) — the OCR pipeline this model ships in.
