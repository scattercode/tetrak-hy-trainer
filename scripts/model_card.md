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

## Status: v7, alpha

v7 reads both Armenian orthographies and the Russian that Soviet editions quote. It is a pre-train from scratch on every proofread work the trainer harvests from Armenian Wikisource, 58 of them, a third in the classical (pre-1922, diaspora) spelling, fine-tuned on 217,362 real crops cut from 2,136 scanned pages of 42 works. v6 was trained on reformed-orthography print alone and read classical pages a word in four worse than it should.

With the companion package's post-processing (below), v7 leads every Armenian OCR engine we have measured on mean word recall, and leads on eleven of seventeen registers. The other six go to Calfa's `hye-paddle`, five of them classical, and on two, Abovyan and Khatisian, by a distance. Calfa's `hye-calfa-n` and `hye-paddle` are CC BY-NC; this model is Apache 2.0.

| Register (held-out pages) | v7 | v7 + fold + word lists | best other engine |
|---|---|---|---|
| Totovents, Works | 0.923 | **0.965** | 0.941 hye-calfa-n |
| Khrimian, Works (classical) | 0.929 | **0.958** | 0.943 hye-paddle |
| Medical encyclopedia | 0.904 | **0.951** | 0.933 hye-paddle |
| Otyan, Works (Western Armenian) | 0.924 | **0.950** | 0.934 hye-paddle |
| Toranian, *Countries and people* (classical, Aleppo) | 0.932 | **0.939** | 0.909 hye-paddle |
| Baronian, Works vol. 10 | 0.901 | **0.937** | 0.918 hye-paddle |
| Tumanyan, academic edition vol. 5 | 0.912 | **0.937** | 0.913 hye-paddle |
| Aram Manukyan, documents (classical) | 0.873 | 0.934 | 0.941 hye-paddle |
| Teodik, *Armenian printing* (classical) | 0.874 | **0.927** | 0.899 hye-paddle |
| Armenian Soviet Encyclopedia vol. 2 | 0.846 | **0.911** | 0.865 hye-paddle |
| Kajaznuni, *Nation and homeland* (classical) | 0.837 | **0.900** | 0.886 hye-paddle |
| Faustus of Byzantium (1968) | 0.859 | 0.898 | 0.922 hye-paddle |
| Baronian, *National bigshots* (classical) | 0.829 | 0.888 | 0.894 hye-paddle |
| Aghbalyan, criticism (classical) | 0.788 | 0.824 | 0.846 hye-paddle |
| Abovyan, Works vol. 1 (classical) | 0.765 | 0.794 | 0.910 hye-paddle |
| Khatisian, memoirs (classical) | 0.672 | 0.762 | 0.856 hye-paddle |
| Armenian–English dictionary | 0.663 | **0.687** | 0.678 marker |

Word recall: the share of the transcript's words found in the output. Ten pages per register (five for the dictionary), never trained on, proofread on Armenian Wikisource; the first column joins lines in detector order, as EasyOCR returns them, and the second is [Tetrak](https://tetrak.dev/)'s pipeline, which orders columns, decodes with the word lists and applies the fold. Mean word recall through the pipeline over the seventeen registers: 0.892, against v6's 0.861 on the same pages and Calfa's 0.829.

**Cyrillic.** The academic edition of Tumanyan quotes Russian in its apparatus. v7 reads 61 of the 81 Russian words on its evaluation pages; v6, with no Cyrillic in its charset, read none.

**Where v7 is worse than v6.** On five of the eight registers v6 was released on it loses a little word recall through the pipeline: Faustus of Byzantium −0.018, the medical encyclopedia −0.016, Otyan −0.014, Baronian vol. 10 −0.007, Totovents −0.003. The same losses appear in the model alone, so they are the recogniser's. They were accepted knowingly for the classical gains above; `v6` stays tagged for a caller who reads reformed print only.

**Character similarity measures reading order here, not recognition.** In detector order, v7 scores 0.83–0.98 on the single-column registers and 0.30–0.35 on the multi-column encyclopedias, where EasyOCR's output is not in column order. In Tetrak's pipeline the same weights score 0.971 on the encyclopedia and 0.960 on the medical encyclopedia.

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

- **Abovyan:** the collected works are set in a face and at a size v7 reads worse than Calfa does; the only classical register still behind.
- **Faustus of Byzantium:** the index and notes are set in a bold italic whose digits the model confuses.
- **Mshak (1870s newspaper):** two transcribed pages so far, read at 0.80 word recall against Calfa's 0.90; the first evaluation material not from Wikisource, and too little of it yet to publish a figure.

### Charset

245 characters plus the CTC blank, 246 classes. Over v6's 174: `<`, `>`, `³` and `±`; the Armenian apostrophe of elision `՚` (U+055A), the one printed character the classical harvests brought; and the Russian alphabet in both cases including `ё`/`Ё`, for the apparatus of Soviet editions. **A charset change is a new model by construction**, because CTC class indices are positional. Always take the `.yaml` and the `.pth` from the same revision.

### Validation

89.9% word accuracy (0.9864 normalised edit distance) on 21,574 real crops from pages held out of the fine-tune, split by page, over all 42 sources. Lower than v6's 95.2% because the held-out set is twice the size and a third of it is classical print in diaspora faces. Held-out crop accuracy is a poor proxy for page-level recall; the page figures above are the ones to trust.

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
    hf_hub_download("tetrak/easyocr-armenian", filename, revision="v7")
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

The fine-tune mixes, in every batch, real crops (217,362, of which 21,574 held out by page; cut from 2,136 scanned pages of 42 works by detection-assisted alignment against their transcripts), the pre-train's synthetic crops, all-caps lines for encyclopedia headwords, and English lines rendered from six public-domain novels so that the Latin the bilingual dictionary prints is not read as Cyrillic.

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
