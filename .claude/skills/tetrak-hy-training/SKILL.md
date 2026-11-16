---
name: tetrak-hy-training
description: Run the Armenian recogniser pipeline end to end in tetrak-hy-trainer — census and harvest Wikisource, check the charset and the evaluation overlap, render and pre-train on synthetic crops, harvest real crops, fine-tune, measure, build the word list, package a bundle and upload weights to Hugging Face. Covers the held-out evaluation rules and the traps that have each cost a training run. Use this whenever training, fine-tuning or evaluating the model, harvesting pages or crops, adding a corpus source, changing the charset, building the word list, packaging a bundle, or publishing weights.
---

# Training the Armenian recogniser

The whole pipeline, in the order it runs. Every step here has cost a run at
least once when skipped or done in the wrong order — the traps are marked.

## Before anything

`AGENTS.md` has the two maps this pipeline leans on: **which interpreter runs
which script** (picking the wrong one is the usual first five minutes lost),
and **the `runs/` layout** the scripts' defaults expect. It also carries the
hard rules, the held-out split above all. Read those first; they are not
repeated here.

The short version of the split: every evaluation page, and every harvested
page that reprints one (`OVERLAPPING_PAGES`), is excluded from real crops,
from the synthetic sampler and from the word list. `harvest_real_crops.py`
checks the harvest manifest before reading anything. If a guard fires, do not
work around it.

---

## 1. Find material — census

```bash
python scripts/wikisource_census.py            # the crawl, ~20 min, resumable
python scripts/wikisource_census.py --top 40   # report from the cache
```

Walks every ProofreadPage index on hy.wikisource.org and counts pages by
quality level. Coverage cannot be guessed from titles — it runs from 350
proofread pages down to zero. For promising indexes it also probes the native
scan width: **ASE volume 1 is 1920 px where volumes 2–6 are 3840**, which
matters for crop detail.

```bash
.venv/bin/python scripts/orthography_census.py            # read four pages per index
.venv/bin/python scripts/orthography_census.py --report   # classical and mixed indexes
```

Says which orthography each well-covered index is in, from the page text.
Titles cannot: Kajaznuni's classical text is titled with a reformed `և`, and
the Soviet editions of Western authors are reset in reformed spelling. The
marker is free-standing `ւ` outside `ու` (classical pages 15–30 per thousand
letters, reformed under 1); the `եւ`/`և` habit is per press and decides
nothing. Grabar verse in a modern edition reads as classical too — check the
author's century before harvesting a "classical" tagh collection.

## 2. Harvest pages

```bash
python -m tetrak_hy_trainer.harvest \
    --index "Ինդեքս:… (Soviet Armenian Encyclopedia) 1.djvu" \
    --out runs/v1/harvest-vol1 --images --image-width 3840 --pages 20-80
```

Only pages at ProofreadPage quality ≥ 3 are taken — Wikisource seeds
unproofread pages with machine OCR, and training on those teaches another
engine's mistakes.

- **`--pages`, not `--limit`, for front matter.** Volume 1 opens with Russian
  title pages and a preface; entries start around page 20.
- **Material that is not on Wikisource** (brief 014: Մշակ from arar.sci.am)
  goes under `runs/sources/<work>/` as downloaded, with a README naming the
  source and its terms. `scripts/import_source_pages.py <work>` renders every
  PDF page into `runs/eval/<work>/images/` and writes a manifest listing only
  the pages that have a transcript at `text/<page>.txt`, so the set grows as
  pages are proofread. Such a work has no `WORK_PAGES` entry: nothing can
  train on it, since nothing harvests it.
- **Verse pages are wrapped in `{{Poemx||<poem>…}}`.** Until cleaning 3
  (2026-10-08) the template stripper took the whole page with the wrapper and
  every poetry index harvested as empty, silently. Harvests taken before then
  have no verse: `--refresh-stale` refetches those pages.
- **`--image-width 3840`** matches the evaluation scans, so crops carry the
  same detail per character as the material the model is scored on.
- Re-runs are incremental. Adding `--images` to a harvest taken text-only tops
  up the scans without refetching wikitext — that is how the v0/v1 volumes
  acquire the images real-crop harvesting needs.

## 3. Check the charset before training on new material

```bash
python scripts/charset_diff.py runs/harvest/<new-source>
```

**Do not skip this on a new source.** A character the charset lacks is one the
pipeline silently filters and the model can never emit. U+2024 (the
transcripts' abbreviation dot) was missing from v1's charset, so every crop
containing it was dropped and 518 words — 5.8% of the evaluation pages — were
unwinnable by construction. Five minutes here would have saved a full run.

Adding a character is a **new model version, never a patch**: the charset is
positional, so `num_class` changes and every previous weight file is
invalidated.

## 4. Fonts

```bash
python scripts/fetch_fonts.py
```

Verifies the licence recorded *inside* each font (name table 13/14) rather
than trusting the download page, and prints glyph coverage against the
charset. Gaps are expected and handled — the renderer excludes a face from any
line it cannot fully draw — but they should be seen, not discovered. All eight
GHEA faces lack U+2024; Mshtakan (macOS system font) lacks Latin `A`/`x` and
both v2 additions.

**Cyrillic and the apostrophe (brief 014, checked 9 November 2026).** Of the
fifteen book faces, the eight GHEA and the four Arnamu faces draw the whole
Russian alphabet including `ё`/`Ё`; a rendered sheet confirmed the glyphs are
real Cyrillic, italics included, not fallback boxes. Noto Sans Armenian, Noto
Serif Armenian and Mshtakan have none, so the per-line guard keeps them off
any line with a Russian word; no face needed adding. Every book face draws the
apostrophe `՚`. Of the 128 display faces (brief 019), 68 have no Cyrillic and
19 lack the apostrophe (the Braind, Nicolo, Nortar and Yerevanyan families
and the two Grqalir bold faces); the same guard handles them, and a family
with no eligible face is simply never picked for that line. The reports from
`fetch_fonts.py` and `find_fonts` say "the Russian alphabet" rather than
listing 66 letters, so a single unexpected gap still stands out.

**Display faces (brief 019)** are a separate set, so the pre-train's faces
never change by accident:

```bash
python scripts/fetch_display_fonts.py     # runs/fonts-display/{train,heldout}
python scripts/render_synthetic_set.py --run-name <run> --name syn_display \
    --font-dir runs/fonts-display/train --upper-share 0.15 --harvest-dirs …
python scripts/render_face_testset.py --harvest-dirs …    # runs/b019/face_test
../tetrak-easyocr-armenian/.venv/bin/python scripts/evaluate_face_set.py \
    --bundle runs/<run>/bundle
```

- **Never move a face into `heldout/` or out of it** without re-baselining:
  the four held-out families are the only measure of reading a face the model
  has never seen.
- **Caps-only faces** (Oldtimer) draw capitals at lowercase code points. The
  renderer counts every lowercase letter as missing from them, so they draw
  only all-capital lines; `--upper-share` is what gives them any lines at all.
- **A full cmap is not a drawn glyph.** Handjet cut at element shape 0 drew
  nothing at all for every letter; check a new instance renders ink.
- **Variable fonts** are cut into fixed instances at fetch time, because Pillow
  draws only the default instance.
- `os.walk` does not follow symlinks, so **a symlinked dataset folder is
  silently skipped** by the trainer. Clone (`cp -cR`) rather than link.

## 5. Synthetic pre-train

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -ims \
    python scripts/train_synthetic.py --device mps --iters 150000 \
    --run-name v2 --eval-dir runs/eval/ase-vol2
```

Roughly eleven hours on MPS. `PYTORCH_ENABLE_MPS_FALLBACK=1` is required — CTC
loss has no MPS kernel — and the script refuses without it rather than failing
hours in.

- Samples **lines** of 1–4 consecutive tokens, not single words: CRAFT hands
  the recogniser multi-word line crops, and v0 had never seen a space.
- Degrades validation crops too. v0's crisp validation read 99.7% while real
  scans read 0.08.
- `--eval-dir` scores the packaged result against real pages before you wake
  up, so the honest number is in the log.
- Interrupted? The checkpoint survives — re-package with `--package-only`.
- **`--eval-dir` needs easyocr, which this venv does not have.** The v7
  pre-train ran its eleven hours, packaged, then crashed in the evaluation
  with `ModuleNotFoundError: easyocr` (14 November 2026). Until the script
  checks at start-up, leave `--eval-dir` off and score the bundle with
  `evaluate_registers.py` from the sibling venv (step 8).

**What v7's pre-train produced (brief 014, issue #44, 13–14 November 2026).**
60,000 lines from all 58 harvests (33 classical by their own text),
`--classical-share 0.25` (15,220 lines) and `--mixed-share 0.05` (3,000),
175,500 crops rendered in 195 s, 150,000 iterations in 11.2 h. Best
synthetic validation 95.13% at iteration 132,000; validation loss bottoms
near 70,000 and drifts up while accuracy still creeps, the usual CTC shape.
On the eight original registers the pre-train reads as v4 did (mean word
recall 0.621 against 0.619), the synthetic-only pre-train v6 was fine-tuned
from, so a wider head and a quarter classical lines cost nothing on reformed
text. The artefacts sit in `saved_models/v7-b014-pretrain/` and
`bundle-pretrain/` so the fine-tune could write `saved_models/v7-b014/` and
`bundle/` under the same run.

**Faint photographs (brief 023).** `--faint-share 0.2` on
`render_synthetic_set.py` degrades that fraction of crops to a grey
photograph of a page (`augment.faint`): paper at 150 to 215, ink only 45 to
100 grey levels below it, a little grain. `tone` never goes fainter than
about a hundred levels, and the Abovyan evaluation pages sit at 58 to 78
(measured 16 November 2026), which nothing before this lever rendered. At
`0`, the default, no random draw is made for it, so every earlier seeded
set is still reproduced crop for crop. The share is recorded in
`recipe.json`; choose it on validation crops, never on an evaluation page.

**Mixed-script lines (brief 014).** `--mixed-share 0.05` on either script
lifts the lines that mix Armenian and Cyrillic tokens to that fraction of the
sample; `0` (the default) keeps the natural rate and the draws of every
earlier seeded set. The supply is thin: 5,747 such candidate lines of 2.1
million over the encyclopedia and Tumanyan volumes (0.27%, measured
9 November 2026), so a share above about 10% of a 60,000-line set repeats
them. Tokens mixing Cyrillic with Armenian or Latin letters are dropped
(`charset.mixes_cyrillic`); wholly Cyrillic tokens stay. Each rendered set
writes `recipe.json` beside its labels with the share asked and the mixed
lines it got: copy both into the `VERSIONS` entry.

**Classical text (brief 014).** `--classical-share` sets the fraction of
lines drawn from classical-orthography harvests: `0.25` by default on the
pre-train, `0` on `render_synthetic_set.py` so brief 019's seeded sets stay
reproducible. A harvest is classical by its own text: the sampler runs
`orthography.classify` over its non-held-out pages at sampling time, prints
the verdict per harvest and records it in `recipe.json`, because a line of a
few tokens cannot be read on its own and titles mislead. On 9 November 2026
that is 33 classical harvests (6,440 pages), 3 mixed (which count as neither)
and 22 reformed. **Pass every harvest** (`runs/harvest/* runs/v0/harvest
runs/v1/harvest-*`): the script's default directories are the encyclopedia
volumes alone, all reformed, and a classical share with no classical source
is a printed shortfall, not an error. A share that is set is exact; with every
harvest passed the natural rate is already close to a quarter, and the share
holds it there whatever the mix of sources. Measured 9 November 2026 on a
60,000-line sample over every harvest: free-standing `ւ` 5.4 per thousand
Armenian letters at the natural rate (28% classical lines) and 4.8 at a
quarter, against about 0.3 in reformed text alone and 17 in classical, so a
sample's `orthography.markers` tells you at once whether the share took.

**The charset diff under v5 (9 November 2026).** Six of the 58 harvests are
clean; the residual in the rest is wiki and maths markup (`_ \ { } ^ | ~`,
mostly the v0 encyclopedia harvest), Greek, accented Latin (`é`), curly and
low quotation marks, bullets and `#`, all already left out at Stage 0.2 and
all dropped with their tokens. The only Cyrillic the corpus prints outside v5
is fifteen scattered non-Russian letters in encyclopedia names (`ғ ҹ ә ћ і` and
an accented `ѐ`), left out; no pre-1918 letter appears, as Stage 0.3 assumed.

## 6. Harvest real crops

```bash
../tetrak-easyocr-armenian/.venv/bin/python scripts/harvest_real_crops.py \
    --harvest-dir runs/v1/harvest-vol5 runs/v1/harvest-vol6 \
    --bundle runs/v2/bundle --out runs/v2/all_data
```

Runs CRAFT over pages with human transcripts and aligns boxes to the
transcript, emitting only what it can place confidently — a mislabelled crop
teaches the wrong shape, so it fails closed.

- **Pass every volume in one run.** A second run into the same `--out`
  truncates the first's `labels.csv`.
- **Write into the v2 run's `all_data/`**, beside `syn_train` — the trainer
  takes a single dataset root, and that is what lets the fine-tune mix both.
- **Harvest with the newest weights.** A better recogniser matches more of the
  transcript, so more crops survive alignment.
- `bracketed` tier is off by default: spot-checking put its precision near
  half. Add it only with the `crops.csv` review in hand.

**Cyrillic and mixed tokens (brief 014).** Under charset v5 a crop whose
label has an all-Cyrillic token is kept (v5 and v6 had no class for it and
dropped every such crop), and a crop with a token mixing Cyrillic with
Armenian or Latin letters is dropped whole, as the synthetic sampler drops
the token. Both are counted per page and per source in `summary.json`
(`cyrillic`, `mixed_dropped`), and the per-page log line says how many
columns the page split into.

**Classical sources (brief 014, issue #43).** The classical harvests were
taken text-only; `--images --image-width 3840 --limit 50` on each existing
harvest tops up fifty scans at the native width (the server clamps 3840 to
it) without refetching wikitext. `--limit` takes pages in the API's title
order (p1, p10, p100, …), which scatters them through the book rather than
taking the front matter. Sources under 1280 px stay text supply for the
sampler and contribute no crops: Gorts and the Bhagavad Gita (960 px by the
census) were not fetched, and of the thirty that were, three came back at
500 px (*Աղջկան մը յիշատակարանը*, Andersen's tales, Darwin) and *Zradasht*
at 960; `runs/v7-b014/harvest-crops.sh` checks the width of each
directory's first scan before passing it and lists the refusals in
`skipped-dirs.txt`. *Հայկական տպագրութիւն* came back at 1920 px, the
others at 1280. *Հայկական տպագրութիւն* is a history of printing whose specimen
pages vary more than a fine-tune wants: fifty pages of it is the sample the
brief asks for, not the whole.

**Columns (brief 014, issue #43; brief 013's Stage 4 note).** The
encyclopedia and the medical encyclopedia are set in *three* close-set
columns, and `align`'s single-gutter search on raw boxes found at most one
gutter on them (on a third of the pages, none), so every such page was read
partly straight across and the alignment kept 30% of its boxes against
80% on single-column books. `align.columns` now trims each box by a quarter
of a line height before looking for a gutter, sweeps the trimmed edges
rather than sampling, tries candidates emptiest and widest first, refuses a
split that leaves a sliver (a critical apparatus's line-number column), and
recurses, so a three-column page splits twice. Measured on identical cached
detections from 26 pages (`runs/v7-b014/gutter-comparison.txt`): the
seventeen encyclopedia pages went from 2,796 to 5,204 crops of 9,337 boxes
(0.30 → 0.56); the nine single- and two-column book and dictionary pages
kept exactly the same crops. This is Tetrak's `layout._vertical_cut`
ported; the two modules are twins and a fix to either belongs in both.

**What the re-harvest produced (brief 014, issue #43, 9–10 November 2026).**
2,136 pages from 42 sources in 2 h 22 min on MPS with the v6 bundle:
337,957 boxes, 217,362 crops (195,788 train, 21,574 val; 191,631 exact,
25,731 near), a yield of 0.64 against v6-b013's 110,118 crops from 862
pages. The encyclopedia volumes went from 0.39 and 0.40 to 0.61 and 0.58,
the medical encyclopedia to 0.48; the classical books sit between 0.50 and
0.90, where the Soviet books were. Three kinds of page yielded little, all
for the same reason — the v6 reader cannot read them, so the alignment
fails closed, and a second harvest with the v7 weights is how to recover
them, not a change to the harvester:

- *The First Love* (0.06) is set in an italic face from cover to cover.
  The transcript matches the scan.
- Russian-language pages in the two document collections (0.25 and 0.35)
  and the Cyrillic words everywhere else: 8 Cyrillic crops kept, 8 dropped
  as mixed, out of 217,362. v6 has no Cyrillic class.
- Khatisian's memoirs (0.29): small, worn Western Armenian type.

Columns: 1,891 pages read as one column, 25 as two, 207 as three. Thirteen
sparse pages (title and contents pages, 330 boxes between them) split into
four to seven columns and kept 84 crops; a minimum box count per column
would stop it, and is not worth doing for thirteen pages. No crop came from
a held-out page (`heldout.WORK_PAGES` and `OVERLAPPING_PAGES` checked
against `summary.json`'s page list). The per-source counts are in
`summary.json` under `per_source` and on issue #43.

## 7. Fine-tune on real crops

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -ims \
    python scripts/finetune_real.py --device mps --run-name <run> \
    --data-root runs/<run>/all_data \
    --saved-model runs/<parent>/saved_models/<parent>/best_accuracy.pth \
    --select-data real_train-syn_train --batch-ratio 0.5-0.5 \
    --iters 6000 --eval-dir runs/eval/<register>
```

An hour or two, not eleven.

- **Start from the latest weights with the charset you want.** A fine-tune
  inherits its parent's charset, so an older parent carries its gaps forward
  (v1's missing U+2024) and its CTC head will not match a newer yaml.
- Extra rendered sets (all-caps, index lines) come from
  `scripts/render_synthetic_set.py`, which names them `<name>_train` and
  `<name>_val`. Select the `_train` folder: the trainer matches folders by
  substring, so a bare `<name>` also selects the validation set (v6 did).
- Real and synthetic are mixed **in every batch** (50/50 by `batch_ratio`),
  not trained in sequence: a few thousand real crops against 175,500 synthetic
  is the classic recipe for catastrophic forgetting.
- Validation is on **real** crops, split by page. Synthetic validation is
  already 98.8% and says nothing about what is being fixed.

**Stop early, and trust the page metric over the crop metric.** v3 was first
run for 10,000 iterations and overfitted. Held-out per-crop accuracy sat in a
93.4–95.1% band from iteration 500 onward while validation loss climbed — and
on a 700-crop validation set that whole band is about four crops, so picking
the best of twenty checkpoints was selecting noise. It landed on iteration
8,000.

The 3,000-iteration re-run scored **0.7356 raw against the long run's
0.6792**: its *raw* output beat the long run's *folded* output, in a third of
the time, at the same 95% crop accuracy. A few thousand iterations is the
right order of magnitude for a fine-tune; the crop number will not tell you
that, and the page evaluation will.

**What v7's fine-tune produced (brief 014, issue #45, 14–15 November
2026).** Two runs from the same pre-train checkpoint. The first was brief
013's recipe on the re-harvested crops: 6,000 iterations,
`real_train-syn_train-syn_caps_train` at `0.5-0.4-0.1`, 4.3 h, held-out
real-crop accuracy 90.2% at 6,000 with validation loss still falling. The
21,574-crop validation set (twice v6's, with classical pages) is why it
took four hours rather than two — validation every 500 iterations is most
of the time — and why its 90% is not comparable with v6's 95%. Against v6
on all eighteen registers (`runs/eval/registers/b014-v7-vs-v6.log`): seven
of the eight original registers within ±0.02, all ten classical and
off-Wikisource registers up 0.05–0.19, mean over eighteen 0.801 → 0.847,
and the Russian in the Tumanyan notes read for the first time, 63 of 81
Cyrillic tokens. The one loss was the bilingual dictionary, −0.034.

**The dictionary finding, and what fixed it.** Not the Cyrillic lookalikes
it first looked like: of the 127 English words v6 read and the first run
did not, 122 were plain Latin misreads (`genlle` for *gentle*, `Ietter` for
*letter*) and five were lookalikes (`роог` for *poor*). The cause was the
pre-train's mix — 11,367 synthetic lines with Cyrillic against 708
Latin-only ones, and Latin in 1% of the real crops — so the shared glyphs
(`а/a`, `о/o`, `с/c`, `р/p`, `е/e`, `х/x`) had their coin loaded towards
Cyrillic and the model's Latin had been crowded. The lesson for the next
charset change: **when a model is first taught a script, give every script
it already reads a matching share of the synthetic set**, or the old one
regresses where the two share shapes. Two fixes, both kept:

- The lookalikes are a fold rule (tetrak-easyocr-armenian PR #27): a token
  with no Armenian letter is written in the one script its unshared letters
  show, and a token of shared glyphs alone follows the text around it.
  Re-scored from saved readings of all eighteen registers it never moves a
  register down (`b014-fold-latin-cyrillic.log`).
- The misreads are training. `syn_latin`: 30,000 English lines from six
  public-domain novels laid out as a harvest-style folder
  (`runs/latin-text/gutenberg-en/text/*.txt`; `clean_token_runs` needs no
  manifest), rendered with `render_synthetic_set.py --upper-share 0.1`
  (58,500 crops, 42 s). The second fine-tune was the first's recipe with
  the English set at a tenth of each batch, taken from the synthetic share:
  `--select-data real_train-syn_train-syn_caps_train-syn_latin_train
  --batch-ratio 0.5-0.3-0.1-0.1 --iters 6000`, 4.1 h, 89.6% at 6,000.

The second run is the v7 candidate (`runs/v7-b014/bundle`; the first is
kept as `bundle-ft-nolatin`). Against v6: dictionary 0.663 against 0.667,
seven of eight original registers within ±0.02 and the medical
encyclopedia −0.022 (−0.015 with the fold), the ten newer registers 0.824
against the first run's 0.833 and v6's 0.744, mean over eighteen 0.843. The
English set tilted the shared glyphs the other way on the Russian pages
(52 of 81 raw) and the fold rule brings 61 back — the two fixes need each
other. Tables: `b014-v7-english-vs-first-attempt.log` (three-way, from
saved readings) and `b014-v7-english-vs-v6.log` (official). A re-scoring
trick worth keeping: save each bundle's raw page readings once
(`evaluate.load_reader` + `read_page` per image) and score folds offline;
a register run costs half an hour of MPS, a re-score costs seconds.

## 8. Measure what actually moved

```bash
python scripts/confusion_report.py     # the character confusion table
```

Word recall alone cannot tell you whether the shape cluster on `հ` moved.
Recompute the confusion table before and after and read them side by side —
that is the fine-tune's scorecard.

For the full backend comparison, `evaluate_baselines.py` from Tetrak's venv;
when only the metric changes, `rescore_baselines.py` re-scores the saved
readings in seconds.

**Save the raw readings** (`evaluate_registers.py --save-readings DIR`)
whenever a model is scored on the registers: a fold rule or a metric
change is then measured on the saved text in seconds, with no model run
and no re-detection noise. Brief 014's two fold rules (the Latin/Cyrillic
token script, library PR #27; the three-way decision and the `եւ`/`և`
rule, PR #30) were both decided that way, and the readings they used are
kept under `runs/v7-b014/readings-registers/{v6,v7-ft-nolatin,v7}` with
the results in `runs/eval/registers/b014-fold-*.log`. A fold rule that
moves nothing on the saved readings is still worth shipping when it is
the policy (a Cyrillic `о` inside an Armenian token can never be meant);
one that moves a register down is not.

**The full-pipeline gate** is Tetrak's, not this repository's: brief 014's
"ours" is recogniser + lexicon + fold + layout, so a candidate bundle is
measured with `tetrak/tools/dump_layout_spans.py --bundle runs/<run>/bundle
--wordlist <modern> --wordlist <classical> --out evaluation/layout/spans-<run>`
and then `tools/register_comparison.py --spans … --ours-label 'tetrak-hy <run>
(ours)'`, both from Tetrak's venv (`uv sync --inexact --extra easyocr --extra
armenian` gives it easyocr and the library). Re-score the released model's
cached spans through the same tools first: the result must reproduce
`evaluation/ocr/registers.csv`, or the two columns are not comparable. The
trainer's raw figure for the same weights is lower and is not
interchangeable with the pipeline's. v7 on 15–16 November 2026
(`runs/eval/registers/b014-v7-full-pipeline.log`): mean word recall over
seventeen registers 0.861 → 0.892, down on five of the eight old registers
by 0.003–0.018 with raw deltas of the same size, so a pipeline loss that
matches the raw loss is the recogniser's; one that does not is the
lexicon's or the fold's. Externals come from the saved `baselines*.csv`;
Marker no longer runs in Tetrak's venv (withdrawn over Pillow 13), so a
new set gets Tesseract and Calfa only unless that changes.

## 9. Build the word list

```bash
python scripts/build_wordlist.py --out runs/<run>/wordlist.tsv.gz \
    --nayiri nayiri-armenian-lexicon-<release>.json
```

Counts words in the proofread transcripts, held-out pages excluded, and adds
the Nayiri Armenian Lexicon if given. It writes a `.json` sidecar naming the
inputs it actually used and the attribution and licence (CC BY-SA 4.0) they
require; the upload publishes that, so keep the two together. To choose the
decoder's threshold on validation crops, build a separate list with
`--exclude-crops runs/<run>/all_data/real_val` so those pages' words are not in
it.

**Two lists since v7 (brief 014, issue #46).** `--orthography reformed`
builds the modern list and `--orthography classical --nayiri …` the
classical one; the default `all` is the v6 recipe. The filter judges each
harvest whole, from its non-held-out pages, with `orthography.classify` as
the synthetic sampler does, and a harvest in the mixed band or too short
to say is in neither list. The Nayiri lexicon is Western Armenian in
traditional orthography, so it belongs with the classical list, where v6
had it in its only list. Built 15 November 2026: modern 130,762 words from
22 reformed harvests (9,138 pages); classical 1,055,306 from 33 classical
harvests (6,343 pages) plus 988,396 Nayiri forms. Free-standing `ւ` is in
11.8% of the classical list's words and 0.3% of the modern's, which is the
spot check that the filter took. The classical harvests carry a little
reformed apparatus (Soviet editions of classical authors), so a few
reformed forms sit in the classical list at low counts: fine for a decoder
that accepts a token in either list and never rewrites across them
(library issue #18), not for one that corrects towards a single list.

**Re-check τ with each model.** The decoder's threshold was chosen on
v5's held-out crops and has to be confirmed on the new model's, with the
lists rebuilt without those crops' pages (`--exclude-crops`). Two scripts
do it without re-running the model per list or per τ:

```bash
../tetrak-easyocr-armenian/.venv/bin/python -W ignore scripts/capture_nbest.py \
    runs/<run>/bundle runs/<run>/all_data/real_val runs/<run>/tau/nbest.jsonl
../tetrak-easyocr-armenian/.venv/bin/python scripts/score_tau.py runs/<run>/tau/nbest.jsonl \
    modern=… classical=… union=…,… --taus 1,3,5,8,12 --by-orthography <build log>
```

`capture_nbest.py` runs the bundle over the crops once and saves every
word's beam candidates (21,574 crops in under five minutes on MPS;
`-W ignore` keeps a per-crop MPS warning out of the log); `score_tau.py`
applies the decoder's rule offline for any lexicon and any τ, in seconds,
and with `--by-orthography` splits the figures by the orthography of each
crop's source harvest. On v7 (15 November 2026, `runs/v7-b014/tau/`): τ = 5
is again the peak for the union of the two lists, exact-match 0.900 → 0.916,
393 crops fixed for 46 broken, with 4 and 6 within 0.001 and 12 breaking as
many as it fixes. The split is the argument for two lists in one table: the
modern list alone is −19 net on the 8,547 crops from classical harvests
(178 fixed, 197 broken) and +127 on the 13,027 reformed; the union is +223
and +124.

## 10. Publish the weights

```bash
LISTS="--wordlist runs/<run>/wordlist.tsv.gz --classical-wordlist runs/<run>/wordlist-classical.tsv.gz"
uv run scripts/upload_model.py --bundle-dir runs/<run>/bundle --version-tag vN $LISTS --dry-run
uv run scripts/upload_model.py --repo-id tetrak/easyocr-armenian-staging \
    --bundle-dir runs/<run>/bundle --version-tag vN $LISTS
uv run scripts/upload_model.py --bundle-dir runs/<run>/bundle --version-tag vN $LISTS   # after review
```

Both lists go up since v7 (`--classical-wordlist` needs `--wordlist`: the
decoder reads them as a union), each with its sidecar's attribution in
`provenance.json`.

Needs `hf auth login` with write access to the `tetrak` org.

**`tetrak/easyocr-armenian` is public, so an upload to it is live at once.**
The script creates a repository private only when it does not exist yet. To
review a release first, upload it to a private staging repository with
`--repo-id` (it is created private), check the card and files there, then
upload to the real one. The script's closing "(private, …)" message is wrong
for an existing public repository.

Before uploading it checks the weights against the yaml — the CTC head's
output size must equal `len(character_list) + 1` — so a charset/weights
mismatch fails here rather than in the field.

**Add a `VERSIONS` entry before uploading.** A tag with no entry is refused
rather than published with empty provenance. Record the recipe, the synthetic
validation figures, the real-scan evaluation, and any `known_defects` —
defects are recorded against the versions that carry them rather than quietly
fixed forward, because people may be running those weights.

It also refuses a word list that is not gzip, or that has no sidecar.

Then hand the release to the library: see the `tetrak-hy-weights-release`
skill in `tetrak-easyocr-armenian`, which pins the Hub revision and the
checksums of the weights and the word list.

---

## Watching a run

**There are two logs, and the useful one is not the obvious one.**
`runs/<run>/train.log` is the orchestrating script's stdout: it goes quiet
immediately after "training N iterations" and stays quiet for the whole run,
so it looks hung when it is working. Per-iteration progress belongs to the
vendored trainer's own log:

```bash
tail -f runs/<run>/saved_models/<run>/log_train.txt
```

Worth knowing before starting an eleven-hour run rather than an hour into one.

## Traps, collected

- **`labels.csv` is not CSV.** The vendored trainer reads it with a regex
  splitting on the first comma, so `csv.writer`'s quoting becomes part of the
  label. This cost v1 21% of its crops — 36,918 labels wrapped in quotation
  marks the images do not show, and inventing a quotation mark became the
  commonest single error in v1's output. Always write labels through
  `synth.write_labels`.
- **Charset membership ≠ the font can draw it.** Pillow silently draws
  `.notdef` for a missing glyph, indistinguishable by eye. `synth.missing_glyphs`
  reads the cmap directly; the renderer excludes a face per line.
- **The vendored trainer is unmodified.** API drift against modern torch is
  repaired in `trainer_compat.install()`, called before importing `train` —
  never by editing `training/`.
- **Load EasyOCR with `["en"]`, never `["hy"]`.** EasyOCR ships no
  `hy_char.txt` so `["hy"]` raises FileNotFoundError; the setting is inert for
  a custom model anyway.
- **Real-crop labels inherit whatever the transcribers did.** Between v2 and
  v3 the `։`→`:` confusion moved the *wrong* way, because some transcripts
  type the Armenian full stop as an ASCII colon and the fine-tune learnt that
  from its labels. `wikisource.normalise_transcript` now labels a closing
  colon in an Armenian word as `։`, but the lesson stands: a real-crop
  fine-tune teaches the transcribers' conventions along with the page's.
  Check the confusion table for entries that got *worse*, not only for the
  cluster you were aiming at.
- **A new volume of an evaluated work can reprint an evaluation page.** Run
  `python scripts/check_eval_overlap.py` after harvesting it and before
  training on it; it exits 1 and names any page not yet in
  `OVERLAPPING_PAGES`.
