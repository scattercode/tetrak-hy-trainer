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

## Status: v6, alpha

v6 is v5 fine-tuned on twice as many real crops: 99,521 cut from scanned pages of 16 works, not rendered ones. It reads eight kinds of Armenian print: an encyclopedia, a medical encyclopedia, a bilingual dictionary, a scholarly history, and literary editions in Eastern and Western Armenian.

With the companion package's post-processing (below), v6 leads every Armenian OCR engine we have measured on word recall on three of those eight registers, and is within 0.001 on a fourth. Those engines include Calfa's `hye-calfa-n` and `hye-paddle`, which are CC BY-NC. This model is Apache 2.0.

| Register (held-out pages) | v6 | v6 + fold + word list | best other engine |
|---|---|---|---|
| Otyan, Works (Western Armenian) | 0.934 | **0.944** | 0.934 hye-paddle |
| Totovents, Works | 0.929 | **0.945** | 0.941 hye-calfa-n |
| Baronian, Works vol. 10 | 0.911 | **0.920** | 0.918 hye-paddle |
| Tumanyan, academic edition vol. 5 | 0.895 | 0.905 | 0.913 hye-paddle |
| Faustus of Byzantium (1968) | 0.873 | 0.897 | 0.922 hye-paddle |
| Medical encyclopedia | 0.926 | 0.929 | 0.933 hye-paddle |
| Armenian Soviet Encyclopedia vol. 2 | 0.837 | 0.839 | 0.865 hye-paddle |
| Armenian–English dictionary | 0.667 | 0.677 | 0.678 marker |

Word recall: the share of the transcript's words found in the output. Ten pages per register (five for the dictionary), never trained on, proofread on Armenian Wikisource. Lines are joined in detector order, as EasyOCR returns them. Pages that print an evaluation page's text are kept out of training as well.

**Character similarity measures reading order here, not recognition.** In detector order, v6 scores 0.91–0.95 on the single-column registers, and 0.30–0.35 on the multi-column encyclopedias, where EasyOCR's output is not in column order. In [Tetrak](https://tetrak.dev/)'s pipeline, which orders columns before joining, the same weights score 0.970 on the encyclopedia and 0.965 on the medical encyclopedia.

> **These figures are not comparable with those published for v5 and earlier.** The metric was corrected for v6. Character similarity used difflib with `autojunk` on, which on a page of text ignores most of the alphabet. The ASCII colon and the Armenian full stop, and `.` and the abbreviation dot `․`, were also scored as different characters, though transcribers type one for the other. Under the corrected metric, v6 is ahead of v5 on every register (mean word recall 0.872 against 0.851).

> **Use v3 or later.** v0 and v1 carry two defects that v2 fixed: 21% of their training labels were wrapped in quotation marks the images do not show, and their charset has no U+2024 ONE DOT LEADER, so 5.8% of the evaluation words were unwinnable. Both are recorded in those versions' `provenance.json`.

### The fold and the word list

The recognition head has no language model, so two kinds of slip remain that a post-process can fix without retraining. Both ship in [tetrak-easyocr-armenian](https://pypi.org/project/tetrak-easyocr-armenian/):

- **`fold_script`** turns a Latin twin inside an Armenian word back into the Armenian letter (`h` → `հ`, `:` → `։`). It also reads `Չ` and `շ` as the digit 2 inside a number, which bold italic page numbers confuse.
- **`tetrak_hy.lexicon`** looks at each word whose reading is not in a word list. It takes the most probable listed reading from the model's own alternatives for that word, if it is nearly as probable. On the held-out pages it fixed 465 words and broke 13: proper nouns, and classical or edition spellings the list does not know.

`wordlist.tsv.gz` in this repository is that list. It holds 1.13 million word forms, counted from proofread Armenian Wikisource transcripts with the evaluation pages excluded. It adds the [Nayiri Armenian Lexicon](http://www.nayiri.com/nayiri-armenian-lexicon) (© Serouj Ourishian, CC BY 4.0).

```python
import tetrak_hy
from tetrak_hy import lexicon

reader = tetrak_hy.reader()
lexicon.use_lexicon(reader, lexicon.load_wordlist("wordlist.tsv.gz"))
results = [
    (box, tetrak_hy.fold_script(text), confidence)
    for box, text, confidence in reader.readtext("page.png", decoder="beamsearch")
]
```

`decoder="beamsearch"` is how the model's probabilities reach the word list. Without it, `readtext` decodes greedily as before.

### What is still lost, and why

- **Faustus of Byzantium:** the index and notes are set in a bold italic whose digits v6 confuses (`2` read as `8` or `7`).
- **Tumanyan:** the academic edition's apparatus quotes Russian, and the charset has no Cyrillic.

### Charset

174 characters plus the CTC blank, 175 classes, unchanged since v5. **A charset change is a new model by construction**, because CTC class indices are positional. Always take the `.yaml` and the `.pth` from the same revision.

### Validation

95.2% word accuracy (0.9933 normalised edit distance) on 10,597 real crops from pages held out of the fine-tune, split by page. Held-out crop accuracy is a poor proxy for page-level recall: it plateaus early and then measures overfitting. The page figures above are the ones to trust.

## Files

- `tetrak_hy.pth` — the weights exactly as the trainer saved them
  (keys carry the `module.` prefix EasyOCR's loader expects to
  handle). This is the file EasyOCR loads.
- `model.safetensors` — the same tensors with the `module.` prefix
  stripped, for anything that isn't EasyOCR.
- `tetrak_hy.yaml` — charset, language list and network parameters.
- `tetrak_hy.py` — the architecture module EasyOCR imports by name.
- `wordlist.tsv.gz` — the word list for `tetrak_hy.lexicon` (v6 on).
- `provenance.json` — training recipe, dataset revision, charset and
  checksums for this release.

## Use with EasyOCR

Download the three EasyOCR files and place them where EasyOCR looks
for custom models:

```python
from huggingface_hub import hf_hub_download

for filename in ("tetrak_hy.pth", "tetrak_hy.py", "tetrak_hy.yaml"):
    hf_hub_download("tetrak/easyocr-armenian", filename, revision="v6")
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

v6 is a fine-tune of v5, which was pre-trained on 351,000 synthetic line crops. Those crops were rendered in 15 Armenian faces from about 7,400 proofread Armenian Wikisource pages (CC BY-SA), then fine-tuned on real crops.

v6's real crops (99,521 for training) were cut from 862 scanned pages of 16 works by detection-assisted alignment against their transcripts. They are mixed in every batch with v5's synthetic crops and with all-caps lines for encyclopedia headwords. Transcript colons inside Armenian words are labelled as the Armenian full stop, which is what the page prints.

The real crops are not published. They are reproducible from the transcripts with the trainer's harvester. `provenance.json` records the recipe, the sources, the charset and the checksums. No font file is redistributed.

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
