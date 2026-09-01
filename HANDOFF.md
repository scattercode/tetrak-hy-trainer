# Handoff: act on the v1 error analysis — fold, v2 charset, real-crop fine-tune

For a Claude Code session. Written 31 August 2026, replacing the completed
publish-v1 handoff (all four of its tasks landed: weights wired and
checksummed in the library, `upload_model.py` v1-aware, decisions 002/003
recorded). This file is untracked — delete it when the work is done. Work
spans three sibling repos; each task names its repo. Read each repo's
CLAUDE.md before working in it.

**Status**: tasks 0–4 done. Task 5 (the `easyocr-hy` backend in tetrak)
is the only one left, and it is Sonnet work.

**v3, the real-crop fine-tune, is trained and published.** Stage 3
step 2 delivered, and it is now the most accurate Armenian word recall
of any backend this harness measures:

| model | chr | wrd (raw) | wrd (+ fold) |
|---|---|---|---|
| v1 | 0.1004 | 0.5014 | 0.5499 |
| v2 | 0.1166 | 0.6073 | 0.6919 |
| **v3** | **0.1470** | **0.7356** | **0.7707** |
| `marker` | 0.258 | 0.766 | — |
| `tesseract -l hye` | 0.697 | 0.662 | — |

6,097 crops from 30 proofread pages across volumes 5 and 6, harvested
with the v2 bundle as reader; fine-tuned from v2 for 3,000 iterations
(0.3 h on MPS), real and synthetic mixed 50/50 in every batch,
validating on real crops split off by page.

Shipped: weights tagged `v3` on the Hub (commit `54f37fcd`),
`tetrak-easyocr-armenian` **0.5.0** on PyPI pinning it, verified from
the published wheel.

**The methodological trap worth not repeating.** This fine-tune was
first run for 10,000 iterations and overfitted. Held-out per-crop
accuracy sat in a 93.4–95.1% band from iteration 500 onward while
validation loss climbed — and on a 700-crop set that band is about four
crops, so picking the best of twenty checkpoints selected noise, landing
on iteration 8,000. The 3,000-iteration re-run scores **0.7356 raw
against the long run's 0.6792**: its raw output beats the long run's
*folded* output, in a third of the time, at the same 95% crop accuracy.
Trust the page metric, and stop early. The archived long run is at
`runs/v3-10k` with its predictions beside the others.

**`։→:` moved the wrong way** between v2 and v3, because some ASE
transcripts write the Armenian full stop as an ASCII colon and the
fine-tune learnt that from its labels. Harmless in the shipped path
(the fold maps it back), but the clearest evidence that real-crop
labels inherit whatever the transcribers did.

**v2 is trained, published and released.** 10.8 h on MPS.

| model | chr | wrd (raw) | wrd (+ fold) |
|---|---|---|---|
| v1 | 0.1004 | 0.5014 | 0.5499 |
| **v2** | **0.1166** | **0.6073** | **0.6919** |
| `tesseract -l hye` | 0.697 | 0.662 | — |

**v2 with the fold reaches 0.6919 word recall, past `tesseract -l hye`'s
0.662** — the bar this brief has been measured against since v0.
Synthetic validation 99.333% / 0.9989. Character similarity still
trails and still for the established reason: it is dominated by reading
order, which is task 5's job, not a training problem.

Shipped:

- Weights at `tetrak/easyocr-armenian`, tagged `v2`
  (commit `a9b8f56a`), public, with `provenance.json`.
- `tetrak-easyocr-armenian` **0.4.0** on PyPI, pinning that commit and
  its checksum, carrying `fold_script` and v2's 170-class yaml. Weights
  mirrored onto the GitHub release. Verified from the published wheel.
- The model card records **both v0/v1 defects** and tells users to
  upgrade; `provenance.json` carries them under `known_defects`.
- Trainer **0.3.0** released.
- tetrak's product docs are in PR #59 (its `main` requires one).

Both fixes confirmed by measurement, not assumed:

- **Charset** — v1 emitted U+2024 zero times in 6,672 boxes; v2 emits it
  221 times against 523 in the transcripts.
- **Labels** — "inserted a quotation mark", v1's commonest single error,
  has vanished from v2's confusion table.

**Two logs, and the useful one is not the obvious one.** `train.log` is
the script's stdout and goes quiet for the whole run after "training
150000 iterations"; per-iteration progress belongs to the vendored
trainer's own log (`runs/<run>/saved_models/<run>/log_train.txt`). Worth
knowing before the fine-tune run.

**Pushing the trainer needs `.venv` on PATH** — the pre-push hook runs
`pytest` bare and cannot collect without it.

**What is left**: task 5's `easyocr-hy` backend, deciding whether to
publish v3, and merging PR #59 to close out the docs. See "Which model"
at the end — the remaining code is Sonnet work.

## Where things stand

The v1 error analysis is complete:
`tetrak/product/research/armenian-v1-error-analysis.md`. **Read it in
full before starting** — every task below implements one of its ranked
conclusions.

Headline, on the ten real ASE vol-2 eval pages (105–114,
`runs/eval/ase-vol2/`): v1 word recall 0.500 vs `tesseract -l hye` 0.662,
and roughly half that gap is mechanical, not model capability:

- **Finding 1**: U+2024 ONE DOT LEADER (the transcripts' abbreviation
  dot, `ա․`, `Գրկ․`) is missing from the charset — 518 words (5.8%) are
  unwinnable, every time, and the training pipeline silently filtered
  every synthetic crop containing it.
- **Finding 2**: cross-script homoglyphs — the model writes Latin `h` for
  `հ` (302), `:` for `։` (141), en dash for hyphen (23). Distinct from
  the genuine shape confusions (հ→խ, խ→ի, տ→ո, ճ↔ջ), which cluster on
  `հ` and are the real-crop fine-tune's target.
- Folding those equivalences on both sides lifts v1 to **0.599** — seven
  points off Tesseract, without retraining. (Measured precisely in task
  2: the shipped fold reaches 0.5499, and a measurement-only ceiling
  0.5832 — both land under this rough estimate; dash folding turned out
  net harmful and was dropped. See task 2 below.)
- **Findings 3–4**: detection is fine (3.3% unaligned); chr is dominated
  by reading order — do **not** chase it with eval-side sorting (a naive
  midline split hurts pages 108 and 114). Layout serialisation in
  Stage 4 is the honest fix; per-word metrics track recognition until
  then.

## Task 0 — commit the analysis and log it (repo: tetrak) — DONE

- ~~Commit `product/research/armenian-v1-error-analysis.md` as `docs:`.~~
  Done (`docs(product): record the v1 error analysis and its findings`).
  `.tetrak-review.tgz` left alone, as instructed — untouched, presumably
  another session's artefact.
- ~~Add a dated entry to brief 011's decision log~~ Done, same commit.

## Task 1 — script-consistency fold (repo: tetrak-easyocr-armenian) — DONE

Shipped as `src/tetrak_hy/fold.py`, exported as `tetrak_hy.fold_script`.
Folds `h`→`հ`/`H`→`Հ`, `o`→`օ`/`O`→`Օ`, `:`→`։` within any whitespace
token that already contains an Armenian letter (U+0530–U+058F). Does
**not** fold ASCII `.`→`․` (finding 1's charset problem, not a homoglyph
one — waits for v2) or dashes.

**Dash folding was tried and reverted** — this deviates from the
original plan below, on measured evidence, not a guess. The first cut
folded en/em dash to hyphen-minus per the analysis's confusion table (23
occurrences of truth `-` read back as `–`). Re-scoring v1's saved
predictions with it in place (task 2's script) showed the harvested ASE
transcripts use en dash as a genuine orthographic convention — attaching
a grammatical case suffix to an abbreviated headword, e.g. `Ա–ի` — 184
times on the ten eval pages. An unconditional swap broke 74 of those
correctly-predicted words for every 66 it recovered: net harmful, on top
of the h/o/colon folds' net +449 with zero regressions. Dropped in a
follow-up `fix:` commit; see `fold.py`'s docstring for the full
reasoning and `tests/test_fold.py`'s `test_does_not_fold_en_dash`.

Punctuation-only tokens (e.g. a stray `363)` next to Armenian text) are
a documented, accepted gap — same reasoning as originally planned:
folding by line instead of by token risks rewriting genuine Latin or
Cyrillic text sharing a detected region with Armenian words.

Two commits: `feat: fold cross-script homoglyphs onto Armenian in
recognised text`, then `fix: stop folding en/em dash in fold_script,
measured net harmful`. Both shipped in 0.4.0.

## Task 2 — measure the fold honestly (repo: tetrak-hy-trainer) — DONE

`scripts/score_fold.py`: re-scores v1's saved predictions
(`predictions-v1.tar.gz`, no inference re-run), fold applied to the
prediction side only. Reproduces the reported 0.5004 baseline exactly
(sanity check), then:

- **fold-only 0.5499** — the number the shipped fold earns today.
- **fold-plus-punctuation 0.5832** — folding U+2024→`.` on *both* sides
  too, as a measurement-only ceiling for what the v2 charset (task 3) is
  expected to reach. Not a real fix: v1 can never emit U+2024 at all, so
  this only equates the two spellings for scoring.

Both land ~1.6 points under the analysis's rough "0.599" estimate —
recorded as a correction in brief 011's decision log, not silently
smoothed over. These measured numbers are now authoritative over that
estimate. Committed `chore:` (scripts/ is outside the shipped wheel).

## Task 3 — v2 charset, re-render, retrain (repo: tetrak-hy-trainer) — CODE DONE, TRAINING NOT STARTED

Everything up to the actual training run is done and verified at real
scale. **The ~11 h run itself has not been launched** — it needs
Stevie's terminal (MPS, real wall-clock); the exact command is below.

Done:

- `src/tetrak_hy_trainer/charset.py`: added `V2_ADDITIONS = "․°"`
  (U+2024 and the degree sign), appended after `COMMON_PUNCTUATION` so
  the diff against v1's charset is a pure append and the existing
  ordering tests (`characters.startswith("ԱԲԳԴ")`, `"ա"` at index 38)
  are untouched. `num_class` moves 168 → 170. Decision made (not asked):
  **yes to `°`**, per the analysis's own recommendation and this being
  a natural moment to pay for it. Cyrillic stays out, as planned —
  recorded as a known limitation. Committed `feat:`.
- Verified the harvested corpus (vols 1, 3–6, already on disk) actually
  contains these characters: 2,422 raw U+2024 occurrences across 384
  pages, and confirmed `clean_token_runs` (the charset-membership filter
  in `scripts/train_synthetic.py`) now keeps 2,223 of them as valid
  tokens instead of silently dropping every crop containing one — v1's
  exact bug, now fixed by the charset widening alone, no code change to
  that function needed.
- **Found and fixed a second, more consequential bug while doing that
  verification**: charset membership says nothing about whether a
  *specific font* can actually draw a character. Mshtakan — the macOS
  system font, one of the three faces every synthetic crop is rendered
  in (v0's and **already-published v1's** included) — has no glyph for
  Latin `A`–`Z`/`a`–`z` **or** for either v2 addition. Pillow silently
  draws its fallback glyph instead, indistinguishable by eye from a
  genuine narrow glyph — confirmed by comparing rendered pixel output
  across several definitely-missing characters, all identical. Added
  `synth.missing_glyphs()` (reads the font's cmap via `fonttools`, a new
  `dev`+`train` dependency) and made `render_corpus` exclude a font from
  a line's draw when it's missing even one of that line's characters,
  rather than choosing uniformly across all three regardless of
  coverage. Two commits: `feat: widen the v2 charset …` and
  `fix: skip fonts missing a glyph when choosing a face to render with`.
  This does not retroactively change v1's shipped weights — only the v2
  re-render inherits the fix.
- Rendered a real 5,000-line sample with the v1 recipe's actual
  parameters (all six sizes down to 18 px, three fonts) against the real
  harvested corpus: 24 lines containing U+2024, 10 containing `°`, none
  raised, and the rendered crops were visually inspected — genuine dot
  and degree-sign glyphs, not tofu. `°` correctly renders in a
  Noto face (the only one that has it), never Mshtakan.

Still to do:

- **Launch the training run** (Stevie's terminal — same reasoning as HF
  pushes: real wall-clock, and MPS lives on his Mac). Fresh synthetic
  pre-train, not a warm start: the charset resize changes the CTC head's
  output width, there is no code in this repo for surgically resizing an
  existing head, and v0→v1 was also a fresh pre-train each time — same
  precedent. **Pass `--run-name v2` explicitly** — the flag's default is
  still `"v1"`, and `runs/v1/` holds the real, already-published v1
  bundle and harvest dirs; omitting it would render into and risk
  clobbering that directory.

  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -ims \
      python scripts/train_synthetic.py --run-name v2 --device mps \
      --iters 150000 --eval-dir runs/eval/ase-vol2
  ```

  Everything else matches v1's recipe by using the script's defaults
  (`--line-tokens-max 4 --min-size 18 --max-samples 60000 --repeats 3`).
  Interrupted? The checkpoint survives; `--package-only` re-packages it.
- Evaluate v2 through the same harness with the task-1 fold applied
  (the script's `--eval-dir` does this automatically, pre-fold — apply
  `tetrak_hy.fold_script` to its output the same way task 2's script
  does, or re-run `scripts/score_fold.py` against v2's predictions once
  captured). The 518 abbreviation words become winnable; expect word
  recall meaningfully above the fold-only 0.5499 measured on v1.
- Publish via the v-aware `scripts/upload_model.py` (Stevie's terminal),
  updating the model card. **Decision point, ask**: publish v2, or hold
  until the real-crop fine-tune (task 4) lands too?
- Log the run's numbers in brief 011's decision log, matching the style
  of the existing v0/v1 entries.

## Task 4 — real-crop fine-tune, Stage 3 step 2 (repo: tetrak-hy-trainer) — TOOLING DONE, RUN BLOCKED ON v2

The genuinely model-limited remainder: shape confusions on degraded
1970s letterpress, clustered on `հ`, which synthetic fonts render too
cleanly.

All the tooling is built, tested and exercised on real scans. The
fine-tune itself cannot run until v2 exists, because it must inherit v2's
widened charset — fine-tuning v1 would carry the U+2024 hole forward and
the CTC head would not match the v2 yaml.

Done:

- **`tetrak_hy_trainer.align`** — the detection-assisted alignment brief
  011 calls "the genuinely fiddly part". Matches at token level and
  groups back onto boxes, because CRAFT returns *line fragments*, not
  words (~550 boxes for a page's ~900 tokens); an earlier cut that
  compared a whole box against one token kept only short single-word
  boxes, 7 of 304 crops having so much as a space — throwing away
  exactly the line-shaped crops v1 was trained on. Column-aware
  ordering keeps two-column pages in step, and gutter-straddling boxes
  are dropped (running headers and page numbers, which the transcripts
  omit anyway). Pairings are **tiered** — `exact`, `near`, `bracketed`
  — so the precision/recall trade is auditable instead of hidden in a
  threshold. It fails closed throughout: a box it cannot place is
  dropped, never guessed.
- **`tetrak_hy_trainer.heldout`** — refuses volume 2 outright, checked
  against the manifest rather than the directory name. Verified:
  pointing the harvester at `runs/eval/ase-vol2` raises before it opens
  anything or creates an output directory. Note it condemns the whole
  volume, not just pages 105–114, matching what the project has done
  since v0. (Its first cut used `endswith("2.djvu")`, which would also
  have condemned volume 12 — the encyclopedia runs to thirteen.)
- **`tetrak_hy_trainer.confusion`** + **`scripts/confusion_report.py`** —
  the progress metric. Reproduces the analysis's table shape from v1's
  saved predictions with no inference re-run. With `--fold` it shows the
  division of labour the analysis argued for: homoglyph confusions fall
  77 → 1 while the shape cluster stands.
- **`scripts/harvest_real_crops.py`** — CRAFT + align + cut, with an
  audit trail (`crops.csv`, `summary.json`). Two filters came from
  actually looking at its output rather than from the algorithm: a box
  over twice the page's median height is a detector blob (one arrived
  labelled `(1961)` for a five-line block), and a reading ending in a
  dash the label lacks is a word broken across a line end, where the
  label claims more than the crop shows — that one was mislabelling
  ~11 crops per hundred *in the `near` tier*. `bracketed` ships off by
  default: spot checks put its precision near half.
- **`scripts/finetune_real.py`** — fine-tunes from a checkpoint, mixing
  real and synthetic in **every batch** (`select_data`/`batch_ratio`,
  50/50 by default) rather than in sequence. A few thousand real crops
  against 175,500 synthetic ones is a narrow distribution and training
  on it alone is how a model forgets everything else. Validates on real
  crops, split off **by page** — crops from one page share its paper and
  scanning, so a per-crop split would measure memorisation.

Measured, on ten volume-6 pages with v1 as the alignment reader:
**1,575 crops (1,385 train / 190 val), 1,164 exact + 411 near**, about
158 per page — so the brief's "a few thousand" needs roughly 20–30
pages, not hundreds. Labels were spot-checked against the images and are
correct; the misreadings they capture are squarely the target set
(`տեղաՌանվել`→`տեղահանվել`, `hայ`→`հայ`, `ղեսել`→`տեսել`).

Two findings that change how to run it:

- **Volume 1 is scanned at 1920 px; volumes 2–6 at 3840.** The
  evaluation volume is 3840, so crops should come from volumes 3–6 or
  they arrive at half the detail the model meets at inference. Wikimedia
  will not upscale past the source, so `--image-width 3840` silently
  yields 1920 on volume 1.
- **Harvest with the v2 bundle, not v1.** v1 cannot emit U+2024 at all,
  so abbreviation words never align: exactly one crop in 1,575 carried
  the abbreviation dot. v2 can, and should recover them.

Still to do, in order:

1. Train v2 (task 3).
2. Top up scans for more volume 3–6 pages — 20–30 dense body pages is
   the target. Front matter is not training material, hence `--pages`:

   ```bash
   python -m tetrak_hy_trainer.harvest \
       --index "Ինդեքս:… (Soviet Armenian Encyclopedia) 6.djvu" \
       --out runs/v1/harvest-vol6 --images --image-width 3840 \
       --pages 35-80
   ```

3. Harvest crops **into the v2 run's `all_data/`**, so `real_train` sits
   beside `syn_train` — the trainer takes one dataset root:

   ```bash
   ../tetrak-easyocr-armenian/.venv/bin/python \
       scripts/harvest_real_crops.py \
       --harvest-dir runs/v1/harvest-vol6 --bundle runs/v2/bundle \
       --out runs/v2/all_data
   ```

4. Read `crops.csv` before training on it. The labels are only as good
   as the transcripts, and transcript quality is uneven even at
   quality 4 — volume 6 page 35 has `առըն չությունների` split by a
   space and `հայպարսկական` with a real hyphen dropped. This is what
   brief 011 meant by "correct by hand".
5. Fine-tune (an hour or two on MPS, not v2's eleven):

   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -ims \
       python scripts/finetune_real.py --device mps \
       --data-root runs/v2/all_data \
       --saved-model runs/v2/saved_models/v2/best_accuracy.pth \
       --eval-dir runs/eval/ase-vol2
   ```

6. Score it: `confusion_report.py --bundle runs/v3/bundle` against the
   same report for v2, and `score_fold.py` for word recall. The shape
   cluster moving is the thing to look for; a single recall number
   cannot show it.
- Nothing derived from the CC BY-NC `portmind-armenian-ocr` fork enters
  any repo, as ever.

## Task 5 — the easyocr-hy backend (repo: tetrak)

Analysis conclusion 4. Independent of tasks 3–4 — it works against v1
today and picks up v2/v3 weights for free via the library.

- New `src/tetrak_ocr/backends/armenian.py`, registry name `easyocr-hy`,
  per brief 011 Stage 4: wraps `tetrak_hy.reader()` with the same
  singleton, TIFF-conversion and multi-page guards as
  `src/tetrak_ocr/backends/easyocr.py`.
- Output path: apply the task-1 fold to each recognised string, then
  serialise through `tetrak_ocr.layout` — this is what lifts chr
  honestly. No naive midline sorting (finding 4).
- Score `easyocr-hy` through the harness on the eval pages: expect chr
  well above raw v1's 0.100 and word recall matching task 2's fold-only
  number.
- Commit as `feat:`.

## Order and dependencies

Tasks 0–2 are done. Tasks 3 and 4 are code-complete; **the v2 training
run is the next action and everything else waits on it** — task 4's
fine-tune needs v2's charset, and task 4's crop harvest wants v2 as its
alignment reader. Task 5 needs only task 1 and can run any time.

## Decision points — ask Stevie, don't guess

1. ~~`°` into the v2 charset~~ — decided yes, done (task 3).
2. ~~Publish v2, or hold until the real-crop fine-tune?~~ — published:
   weights tagged `v2` on the Hub, `tetrak-easyocr-armenian` 0.4.0 on
   PyPI, v0 and v1's defects recorded on the model card.
3. ~~Canonical dash for the fold~~ — superseded: measurement showed
   folding either dash variant is net harmful on this corpus (task 1),
   so neither is folded. Not Stevie's call to make; settled by data.
4. How many pages to harvest crops from, and whether to hand-review
   `crops.csv` before the fine-tune. 20–30 pages reaches the brief's
   "few thousand"; the labels are only as good as the transcripts, and
   those are uneven even at quality 4.
5. Whether v0's and v1's model cards should record that 21% of their
   training labels carried spurious quotation marks. It does not change
   their measured numbers — those were honest measurements of the model
   as trained — but it does explain a defect visible in their output.

## Which model

The original split was Sonnet for tasks 0–3 and 5, Opus for task 4, and
it held up: tasks 0–3 were tightly specified with an existing pattern to
copy and a measurable acceptance number, while task 4 was open-ended
tool-building where the judgement calls only surfaced by looking at real
output — the box-versus-token mismatch, the line-break fragments, the
labels.csv quoting. Brief 011 calls the alignment "the genuinely fiddly
part" for good reason.

What is left needs less of that. **Task 5 is Sonnet work**: it has the
existing easyocr backend to copy and a number to hit. The remaining
parts of tasks 3 and 4 are training runs and review, not code.

## Standing cautions

- Never edit CHANGELOG.md, never tag or release by hand — automation
  owns both in all three repos.
- Conventional Commits, hook-enforced; commit types gate releases
  (trainer `scripts/` changes are `chore:`).
- British English throughout.
- No weights, datasets or crops in git; `runs/` is gitignored.
- HF pushes and training runs happen only in Stevie's terminal (write
  token and the 11 h of wall-clock live there).
- Trainer repo: run scripts with `uv run`; the pre-push pytest hook
  needs `.venv` on PATH.
