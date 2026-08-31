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

## Status: v1, alpha

These are the v1 weights, trained on synthetic line crops degraded to
look like scans. **v1 beats stock EasyOCR by roughly 16x on word
recall and nearly doubles v0, but it does not yet beat
`tesseract -l hye`.** If you need the best available Armenian OCR
today, use Tesseract's `hye` model. Use these weights if you are
working on EasyOCR-based pipelines, or want a base to fine-tune.

### Measured on real scans

Ten pages of the Armenian Soviet Encyclopedia (volume 2, pages
105-114) from Armenian Wikisource, proofread to quality level 4, with
their transcriptions as ground truth. Higher is better for both
figures.

| Backend | Char similarity | Word recall |
|---|---|---|
| `tesseract -l hye` | 0.697 | 0.662 |
| marker | 0.258 | 0.766 |
| `tesseract -l hye` (auto page mode) | 0.128 | 0.664 |
| **tetrak_hy v1** | **0.100** | **0.501** |
| tetrak_hy v0 | 0.075 | 0.274 |
| stock EasyOCR | 0.035 | 0.031 |

Read the two columns separately. **Word recall — 0.50 against
Tesseract's 0.66 — is the honest measure of recognition here**, and
the gap it shows is real.

The char similarity column is mostly not about recognition. It is
dominated by reading order on these two-column pages: Tesseract in
automatic page mode reads words just as well as it does in the row
above (0.664 vs 0.662 word recall) yet its char similarity collapses
from 0.697 to 0.128, close to v1's, purely because the text comes out
in a different order. This evaluation joins detected lines with a
newline in detector order, so any backend that does not serialise
two-column pages into reading order is penalised the same way. Column
handling in the surrounding pipeline lifts that number without
retraining anything.

### Synthetic validation

98.8% word accuracy, 0.9974 normalised edit distance. Unlike v0's,
the v1 validation crops are degraded exactly like the training ones,
so this figure is measured on realistic input rather than clean
renders — but it is still synthetic, and the real-scan table above is
the one that matters.

### What is next

Fine-tuning on real crops cut from scanned pages and aligned against
proofread transcripts is the lever expected to close the word-recall
gap. Treat v1 as a usable base, not a finished recogniser.

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
    hf_hub_download("tetrak/easyocr-armenian", filename, revision="v1")
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

Trained on the `crops-v1` configuration of
[tetrak/armenian-ocr-crops](https://huggingface.co/datasets/tetrak/armenian-ocr-crops):
175,500 synthetic line crops of 1 to 4 consecutive tokens, rendered
from proofread Armenian Soviet Encyclopedia text (Armenian Wikisource,
CC BY-SA) at sizes down to 18 px and degraded with a downscale cycle,
blur, tone shift, small rotation and a JPEG round-trip. Three faces
were used for rendering: Noto Sans Armenian and Noto Serif Armenian
(SIL Open Font Licence 1.1) and Mshtakan, which ships with macOS. No
font file is redistributed here.

The exact dataset revision is in `provenance.json`. v0 was trained on
the `crops` configuration — single-word crops, undegraded — which is
still published for reproducibility.

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
