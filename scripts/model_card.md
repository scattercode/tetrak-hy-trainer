---
license: apache-2.0
language:
- hy
- en
- ru
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

## Status: v8, alpha

v8 is v7 taught the print v7 read worst: books set in italics. It keeps what v7 brought, both Armenian orthographies and the Russian that Soviet editions quote, from a pre-train on every proofread work the trainer harvests from Armenian Wikisource, 58 of them, a third in the classical (pre-1922, diaspora) spelling. To that it adds 28,331 real crops cut from 616 pages of three italic books, Khatisian's memoirs above all, where v7 had trained on 1,278.

With the companion package's post-processing (below), v8 leads every Armenian OCR engine we have measured on mean word recall. It leads on twelve of seventeen registers and is level on one. The other four go to Calfa's `hye-paddle`: Abovyan by a distance, and Khatisian, Aghbalyan and Faustus by about 0.02 or less. Calfa's `hye-calfa-n` and `hye-paddle` are CC BY-NC; this model is Apache 2.0.

| Register (held-out pages) | v8 | v8 + fold + word lists | best other engine |
|---|---|---|---|
| Totovents, Works | 0.929 | **0.969** | 0.941 hye-calfa-n |
| Medical encyclopedia | 0.917 | **0.960** | 0.933 hye-paddle |
| Khrimian, Works (classical) | 0.931 | **0.958** | 0.943 hye-paddle |
| Otyan, Works (Western Armenian) | 0.929 | **0.958** | 0.934 hye-paddle |
| Baronian, Works vol. 10 | 0.909 | **0.945** | 0.918 hye-paddle |
| Toranian, *Countries and people* (classical, Aleppo) | 0.938 | **0.943** | 0.909 hye-paddle |
| Aram Manukyan, documents (classical) | 0.891 | 0.941 | 0.941 hye-paddle |
| Tumanyan, academic edition vol. 5 | 0.914 | **0.935** | 0.913 hye-paddle |
| Teodik, *Armenian printing* (classical) | 0.877 | **0.933** | 0.899 hye-paddle |
| Faustus of Byzantium (1968) | 0.881 | 0.914 | 0.922 hye-paddle |
| Kajaznuni, *Nation and homeland* (classical) | 0.842 | **0.911** | 0.886 hye-paddle |
| Armenian Soviet Encyclopedia vol. 2 | 0.842 | **0.909** | 0.865 hye-paddle |
| Baronian, *National bigshots* (classical) | 0.851 | **0.901** | 0.894 hye-paddle |
| Abovyan, Works vol. 1 (classical) | 0.834 | 0.856 | 0.910 hye-paddle |
| Khatisian, memoirs (classical) | 0.772 | 0.838 | 0.856 hye-paddle |
| Aghbalyan, criticism (classical) | 0.789 | 0.824 | 0.846 hye-paddle |
| Armenian–English dictionary | 0.691 | **0.708** | 0.678 marker |

Word recall: the share of the transcript's words found in the output. Ten pages per register (five for the dictionary), never trained on, proofread on Armenian Wikisource; the first column joins lines in detector order, as EasyOCR returns them, and the second is [Tetrak](https://tetrak.dev/)'s pipeline, which lifts a faint page, orders columns, decodes with the word lists and applies the fold. Mean word recall through the pipeline over the seventeen registers: 0.906, against v7's 0.895 on the same pages under the same pipeline and 0.880 for `hye-paddle`, the best other engine. (v7 was published at 0.892; the pipeline's faint-page rule has since lifted it.)

**What v8 gains over v7.** Fifteen of the seventeen registers, Khatisian most: 0.762 to 0.838, on every page, with the two confusions its heavy italic caused (`ռ` read as `ո`, `ւ` as `:`) almost gone. A control fine-tune with v7's own recipe for the same iterations moved the mean by 0.001, so the gain is the material and not the extra training.

**Cyrillic.** The academic edition of Tumanyan quotes Russian in its apparatus. v8 reads 53 of the 81 Russian words on its evaluation pages; v7 read 61, and v6, with no Cyrillic in its charset, none. The italic crops carry almost no Russian and the fine-tune cost those eight words.

**Where v8 is worse.** Against v7: the Russian above, and the Armenian Soviet Encyclopedia, 0.917 to 0.909, which the control lost equally and so is the price of training on. v8 also writes three full stops where the page prints `…` more often than v7; that came with the italic crops, and its cause is not yet established. Against v6, three of the eight registers v6 was released on are still read a little worse through the pipeline (the medical encyclopedia −0.008, Otyan −0.007, Faustus of Byzantium −0.003), where v7 lost on five by up to 0.018. `v6` and `v7` stay tagged.

**Character similarity measures reading order here, not recognition.** In detector order, v8 scores 0.85–0.98 on the single-column registers and 0.30–0.35 on the multi-column encyclopedias, where EasyOCR's output is not in column order. In Tetrak's pipeline the same weights score 0.975 on the encyclopedia and 0.955 on the medical encyclopedia.

> **Use v3 or later.** v0 and v1 carry two defects that v2 fixed: 21% of their training labels were wrapped in quotation marks the images do not show, and their charset has no U+2024 ONE DOT LEADER, so 5.8% of the evaluation words were unwinnable. Both are recorded in those versions' `provenance.json`. Figures for v5 and earlier were measured with a metric that brief 013 corrected and are not comparable with these.

### The fold and the word lists

The recognition head has no language model, so two kinds of slip remain that a post-process can fix without retraining. Both ship in [tetrak-easyocr-armenian](https://pypi.org/project/tetrak-easyocr-armenian/):

- **`fold_script`** reads each token in one script. A Latin or Cyrillic twin inside an Armenian word goes back to the Armenian letter (`h` → `հ`, `о` → `օ`, `:` → `։`); a token with no Armenian letter is written in the one script, Latin or Cyrillic, it belongs to, since the glyphs the two share (`а`/`a`, `р`/`p`, `с`/`c`) can come out in either. It also reads `Չ` and `շ` as the digit 2 inside a number.
- **`tetrak_hy.lexicon`** looks at each word whose reading is in neither word list. It takes the most probable listed reading from the model's own alternatives for that word, if it is nearly as probable.

Two word lists ship here, because Armenian is printed in two spellings and a single list built from reformed text would "correct" every classical page into the other orthography one word at a time:

- `wordlist.tsv.gz` — 130,762 word forms counted from the reformed-orthography harvests.
- `wordlist-classical.tsv.gz` — 1,055,306 forms counted from the classical-orthography harvests, with every word form of the [Nayiri Armenian Lexicon](http://www.nayiri.com/nayiri-armenian-lexicon) (© Serouj Ourishian, CC BY 4.0), which is Western Armenian in traditional spelling.

The decoder works on their union: a word in either list is accepted as written and never rewritten into the other's, and the two-letter `եւ` and the ligature `և` are read as one word. Both lists exclude the evaluation pages. Because they are derived from those sources, the lists are licensed **CC BY-SA 4.0**, not Apache 2.0 like the weights.

```python
import tetrak_hy
from tetrak_hy import lexicon

reader = tetrak_hy.reader()
lexicon.use_lexicon(reader, lexicon.load_wordlists("wordlist.tsv.gz", "wordlist-classical.tsv.gz"))
page = reader.readtext("page.png", decoder="beamsearch")
context = "\n".join(text for _, text, _ in page)
results = [
    (box, tetrak_hy.fold_script(text, context=context), confidence)
    for box, text, confidence in page
]
```

`decoder="beamsearch"` is how the model's probabilities reach the word lists. Without it, `readtext` decodes greedily as before. `tetrak_hy.reader(lexicon=True)` does the download and the wiring for you.

### What is still lost, and why

- **Abovyan:** the evaluation pages are Grabar verse set in a light italic and photographed faint, some of them out of focus. v8 reads them at 0.856 against `hye-paddle`'s 0.910. A quarter of the words it loses have the right letters and a wrong or missing intonation or punctuation mark.
- **Khatisian:** a heavy cursive italic from cover to cover. v8 closed most of the gap (0.838 against 0.856) but not all of it.
- **Faustus of Byzantium:** the index and notes are set in a bold italic whose digits the model confuses.
- **Mshak (1870s newspaper):** two transcribed pages so far, read at 0.83 word recall against 0.94 for `hye-paddle`; the first evaluation material not from Wikisource, and too little of it yet to publish a figure.

### Charset

245 characters plus the CTC blank, 246 classes. Over v6's 174: `<`, `>`, `³` and `±`; the Armenian apostrophe of elision `՚` (U+055A), the one printed character the classical harvests brought; and the Russian alphabet in both cases including `ё`/`Ё`, for the apparatus of Soviet editions. **A charset change is a new model by construction**, because CTC class indices are positional. Always take the `.yaml` and the `.pth` from the same revision.

### Validation

88.7% word accuracy (0.9729 normalised edit distance) on 23,722 real crops from pages held out of the fine-tune, split by page. Not comparable with v7's 89.9%: the held-out set gained 2,789 crops from the italic books, the hardest print in it. Held-out crop accuracy is a poor proxy for page-level recall, and swings by a point between checkpoints of one run; the page figures above are the ones to trust.

## Files

- `tetrak_hy.pth` — the weights exactly as the trainer saved them
  (keys carry the `module.` prefix EasyOCR's loader expects to
  handle). This is the file EasyOCR loads.
- `model.safetensors` — the same tensors with the `module.` prefix
  stripped, for anything that isn't EasyOCR.
- `tetrak_hy.yaml` — charset, language list and network parameters.
- `tetrak_hy.py` — the architecture module EasyOCR imports by name.
- `wordlist.tsv.gz` — the modern word list for `tetrak_hy.lexicon` (v6 on).
- `wordlist-classical.tsv.gz` — the classical word list (v7 on).
- `provenance.json` — training recipe, dataset revision, charset,
  evaluation figures and checksums for this release.

## Use with EasyOCR

Download the three EasyOCR files and place them where EasyOCR looks
for custom models:

```python
from huggingface_hub import hf_hub_download

for filename in ("tetrak_hy.pth", "tetrak_hy.py", "tetrak_hy.yaml"):
    hf_hub_download("tetrak/easyocr-armenian", filename, revision="v8")
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
Armenian, Russian, basic Latin, digits and punctuation.

Pin `revision=` when downloading: each weights release is tagged, and
`provenance.json` records the exact dataset revision it was trained
from.

## Training data

v7's pre-train rendered 175,500 synthetic line crops in 15 Armenian faces from 60,000 lines sampled over every proofread page of 58 works on Armenian Wikisource (CC BY-SA), 33 of them in classical orthography: a quarter of the lines are classical, and one in twenty mixes Armenian with Russian. Twelve of the faces draw Cyrillic; the three that do not are kept off lines with a Russian word.

v7's fine-tune mixes, in every batch, real crops (217,362, of which 21,574 held out by page; cut from 2,136 scanned pages of 42 works by detection-assisted alignment against their transcripts), the pre-train's synthetic crops, all-caps lines for encyclopedia headwords, and English lines rendered from six public-domain novels so that the Latin the bilingual dictionary prints is not read as Cyrillic.

v8 is v7's released weights fine-tuned for a further 3,000 iterations on the same mix, with a fifth of the real half given to new crops from three books set in italics: 28,331 (2,789 held out by page), cut with v7 from 616 scanned pages of Khatisian's memoirs, Abovyan's collected works vol. 1 and *The First Love*. v7 could align far more of those pages to their transcripts than v6 could, which is where the crops came from. The synthetic sets and both word lists are v7's, unchanged.

Every evaluation page, and every page that prints an evaluation page's text, is excluded from all of it and from both word lists. The real crops are not published; they are reproducible from the transcripts with the trainer's harvester. `provenance.json` records the recipe, the sources, the charset and the checksums. No font file is redistributed.

## Licence

The weights, like the trainer, are Apache 2.0. The training text is
CC BY-SA; we publish the text itself, share-alike, in the dataset
repository above, and take the position — shared by most of the
ecosystem, though not legally settled — that trained weights are not
a redistribution or adaptation of the training text.

The two word lists are different: they are counted directly from CC BY-SA 3.0
Wikisource text, and the classical one includes the CC BY 4.0 Nayiri
Armenian Lexicon, so they are licensed CC BY-SA 4.0, with the attribution
recorded in `provenance.json`.

## Related

- [tetrak-hy-trainer](https://github.com/scattercode/tetrak-hy-trainer)
  — synthesis, training and packaging (Apache 2.0).
- [tetrak/armenian-ocr-crops](https://huggingface.co/datasets/tetrak/armenian-ocr-crops)
  — the training data (CC BY-SA 4.0).
- [Tetrak](https://tetrak.dev/) — the OCR pipeline this model ships in.
