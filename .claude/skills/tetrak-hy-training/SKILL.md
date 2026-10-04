---
name: tetrak-hy-training
description: Run the Armenian recogniser pipeline end to end in tetrak-hy-trainer — census and harvest Wikisource, check the charset and the evaluation overlap, render and pre-train on synthetic crops, harvest real crops, fine-tune, measure, build the word list, package a bundle and upload weights to Hugging Face. Covers the held-out evaluation rules and the traps that have each cost a training run. Use this whenever training, fine-tuning or evaluating the model, harvesting pages or crops, adding a corpus source, changing the charset, building the word list, packaging a bundle, or publishing weights.
---

# Training the Armenian recogniser

The whole pipeline, in the order it runs. Every step here has cost a run at
least once when skipped or done in the wrong order — the traps are marked.

## Before anything

`CLAUDE.md` has the two maps this pipeline leans on: **which interpreter runs
which script** (picking the wrong one is the usual first five minutes lost),
and **the `runs/` layout** the scripts' defaults expect. It also carries the
hard rules, the held-out split above all. Read those first; they are not
repeated here.

The short version of the split: every evaluation page, and every harvested
page that reprints one (`OVERLAPPING_PAGES`), is excluded from real crops,
from the synthetic sampler and from the word list. `harvest_real_crops.py`
checks the harvest manifest before reading anything. If a guard fires, do not
work around it.

## Pending for the next run: charset v4

Charset v4 (`<`, `>`, `³`, `±`; `num_class` 175 → 179) and the per-source
angle-bracket fold landed on 2026-10-04 and have not been trained or
published yet. They ship with whatever the next run is for. Delete this
section once that model is released.

- [ ] **Fresh synthetic pre-train** (step 5). v6 and every earlier checkpoint
      has a 175-class head; `finetune_real.py` refuses them.
- [ ] **Re-harvest every real-crop set** (step 6). Crops harvested under v3
      carry `«` where the page prints `<`, and relabelling cannot restore it;
      `finetune_real.py` refuses an unstamped `real_*` folder.
- [ ] **Rebuild the word list** (step 9). It picks the change up on its own,
      but only if it is rebuilt rather than reused from v6.
- [ ] **Check the confusion table** (step 8) for `<` against `«`, and `³`
      against `²` and `3`.
- [ ] **`VERSIONS` entry** (step 10): record charset v4, the four additions,
      and that the `<` fold changed (see `charset.py` and
      `wikisource.PRINTED_ANGLE_BRACKETS`).
- [ ] **Library release** (`tetrak-hy-weights-release` in
      `tetrak-easyocr-armenian`): a new `num_class` is a new yaml, so the
      packaged yaml and the pins move together.

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

## 10. Publish the weights

```bash
uv run scripts/upload_model.py --bundle-dir runs/<run>/bundle --version-tag vN \
    --wordlist runs/<run>/wordlist.tsv.gz --dry-run
uv run scripts/upload_model.py --repo-id tetrak/easyocr-armenian-staging \
    --bundle-dir runs/<run>/bundle --version-tag vN --wordlist runs/<run>/wordlist.tsv.gz
uv run scripts/upload_model.py --bundle-dir runs/<run>/bundle --version-tag vN \
    --wordlist runs/<run>/wordlist.tsv.gz              # after review
```

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
