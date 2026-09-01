# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repository is

**tetrak-hy-trainer** trains an Armenian text-recognition model and packages
it as an EasyOCR custom model. It is the training half of a two-repo
arrangement:

- **This repo** (public, Apache 2.0) owns the synthetic-data pipeline, the
  trainer, the charset, the training configs, and the packaging step that
  emits the deliverable.
- **Tetrak** (the OCR pipeline behind [tetrak.dev](https://tetrak.dev/))
  ships the inference files and downloads the trained weights from this
  repo's GitHub Releases. It also holds the product-level plan and the
  provenance records for published weights.

The deliverable is exactly three files, named for the network:
`tetrak_hy.yaml` (charset + network params), `tetrak_hy.py` (the
architecture module) and `tetrak_hy.pth` (weights), loaded by
`easyocr.Reader(['hy'], recog_network='tetrak_hy', ...)`.

## Hard rules

### The Portmind quarantine

A CC BY-NC 4.0 repository, `portmind-armenian-ocr`, sits in this same
workspace (`../portmind-armenian-ocr`). It validated the approach this
project uses, and **nothing from it may enter this repository — ever**:

- Never copy code from it, however small the fragment. Its licence is
  non-commercial; this repo is Apache 2.0 and public. A copied line here
  is a published licence violation.
- Never copy its test annotations or reference its (defunct) weights.
- Never add it as a dependency, submodule, or vendored path.
- Re-implementing *ideas* from it (dual-threshold detection
  post-processing, small-box joining, confidence gating) is lawful and
  fine — copyright protects expression, not method — but write such code
  fresh, from the idea, not with the fork open beside you.

### The EasyOCR delivery contract

Verified against easyocr 1.7.2 source, not docs. Breaking any of these
breaks loading in the field:

- **CTC head only.** EasyOCR's custom-model path constructs
  `CTCLabelConverter` unconditionally. Attention-head models cannot be
  delivered; do not train one for release.
- **The network name is a Python module name.** EasyOCR calls
  `importlib.import_module(recog_network)`, so the name stays
  `tetrak_hy` — underscores, never hyphens — and the module must expose
  `Model(num_class=..., **network_params)`.
- The yaml must carry `character_list`, `lang_list`, `imgH` and
  `network_params`. Its `lang_list` gates which languages a `Reader` may
  request.
- A missing `hy` dictionary in EasyOCR is tolerated (greedy decode needs
  none); an Armenian wordlist is a later beam-search upgrade, not a
  launch requirement.
- **Load with `Reader(['en'], recog_network='tetrak_hy')`, not `['hy']`.**
  A spike finding (2026-08-29, `scripts/spike_easyocr_loading.py`):
  `setLanguageList` reads `easyocr/character/<lang>_char.txt` for every
  requested language and no `hy_char.txt` ships with EasyOCR, so `['hy']`
  raises FileNotFoundError. The file is irrelevant for a custom model —
  the decode filter is `set(model charset) − set(lang_char)` and
  `lang_char` always includes the yaml's full `character_list`, so the
  filter is empty whichever language is requested. The yaml's own
  `lang_list` keeps `hy` first for self-description; the *loading*
  incantation uses `en`.

### The held-out evaluation split

Every published figure for this model — v0's 0.0745/0.2742, v1's
0.1004/0.5014, the baselines they are measured against — comes from ten pages
of **volume 2** of the Armenian Soviet Encyclopedia. Volume 2 is held out
**whole**, not merely pages 105–114, and has never been harvested for
training.

`runs/eval/ase-vol2/` is also the one directory in this repository with page
images already sitting in it, which makes it precisely the thing a harvester
gets pointed at by accident. Nothing would fail; the numbers would simply
start improving for the wrong reason, and every published figure would become
wrong.

`src/tetrak_hy_trainer/heldout.py` enforces this and is deliberately a module
of its own so it cannot be diluted into some larger helper. It checks the
harvest *manifest*, not the directory name, so a copied or renamed directory
cannot get past it. Brief 012 widened it from one rule to a registry
(`WORK_PAGES`): each new work contributes held-out pages of its own, chosen
before anything trains on that work.

Held-out material is excluded from **both** uses of a harvest — real crops
obviously, but also the synthetic sampler, since rendering an evaluation
page's transcript would let the model memorise the text it is later scored on
reading.

**If a guard fires, do not work around it.** Change the split deliberately in
`heldout.py` and re-baseline everything.

### The charset is a single source of truth

`src/tetrak_hy_trainer/charset.py` defines the character list. The
trainer and the packaged yaml both read it; nothing else may define its
own copy. Changing it changes `num_class` and **invalidates every
previously trained weight file** — treat any charset change as a new
model version, never a patch.

### Artefact discipline

- **No weights, no datasets, no crops in git.** Synthetic data is
  regenerated from committed recipes; weights are published as GitHub
  Release assets with a checksum and a provenance record (data recipe,
  font list, real-crop counts, config, git SHA).
- **Record every data source's licence.** Corpus texts (e.g. Armenian
  Wikisource — public domain; Armenian Wikipedia — CC BY-SA) and fonts
  (OFL) are listed with their licences in the README as they are adopted.
- Code derived from EasyOCR's trainer keeps its Apache 2.0 headers, and
  the NOTICE file is updated when such code lands.

## Conventions

- **British English** throughout; sentence case for headings.
- **Conventional Commits**, enforced by the shared hook at
  `.githooks/commit-msg` (activate with
  `git config core.hooksPath .githooks`, or `lefthook install`).
- **Ruff** with the explicit rule set in `pyproject.toml` — do not fall
  back to defaults; they differ between ruff versions and CI and local
  machines then disagree.
- Write docs in the first person plural — this is a collaborative
  project.
- Public docs here must not link into the Tetrak repository's paths
  (it is private); link to tetrak.dev where a pointer is needed.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest                 # fast, no OCR stack or GPU needed
ruff check src tests scripts
ruff format src tests scripts
uv lock --check        # the lockfile matches pyproject.toml
```

The pre-push hook runs the suite through `.venv/bin/python` when that exists,
so it no longer needs `.venv` on PATH by hand.

### Which interpreter runs which script

There are three, and picking the wrong one is the usual first five minutes
lost. Each script's own docstring says so too; this is the map.

| Script | Run with | Why |
|---|---|---|
| `upload_model.py`, `upload_dataset.py` | `uv run scripts/<name>.py` | PEP 723 inline dependencies in the file header |
| `train_synthetic.py`, `finetune_real.py` | this repo's venv with the **`[train]`** extra | torch and the vendored trainer |
| `harvest_real_crops.py` | **`../tetrak-easyocr-armenian/.venv/bin/python`** | needs easyocr *and* torch; `[train]` has no easyocr |
| `score_fold.py` | the same sibling venv | imports the published `tetrak_hy` package |
| `evaluate_baselines.py` | **Tetrak's venv, from Tetrak's repo root** | imports `tetrak_ocr` backends; the Claude backend needs Tetrak's `.env` |
| everything else | this repo's plain `.venv` | standard library plus core deps |

`evaluate_baselines.py` is the only script that may import from Tetrak. It
benchmarks Tetrak's backends, so it can only run there anyway. **Nothing else
may**: this repository is public and Tetrak is not, so an import of it is a
dependency an outside contributor cannot satisfy. The scoring metrics are
therefore a deliberate copy at `src/tetrak_hy_trainer/accuracy.py` — see its
docstring for the obligation that copy carries.

### The `runs/` layout the defaults expect

`runs/` is gitignored in full, but the scripts' default paths assume this
shape:

```text
runs/
├── v0/fonts/            rendering faces (fetch_fonts.py)
├── v0/harvest/          first text-only harvest
├── v1/harvest-vol*/     per-volume harvests
├── v2/all_data/         syn_train, syn_val, real_train, real_val — one root
├── v2/saved_models/v2/  checkpoints; best_accuracy.pth
├── v2/bundle/           packaged tetrak_hy.{yaml,py,pth}
├── eval/ase-vol2/       the held-out evaluation set — never train on it
└── census/census.json   the Wikisource census cache
```

For the pipeline itself — census, harvest, charset check, pre-train, real
crops, fine-tune, evaluate, upload — load the **`tetrak-hy-training`** skill
rather than reassembling it from the script docstrings.

## Dependencies and the lockfile

Ranges live in `pyproject.toml`; the fully resolved tree (all extras, torch
included) lives in `uv.lock`, which exists so security scanning — Trivy in
CI and on a weekly schedule, plus Dependabot — sees real versions rather
than ranges. After any dependency change run `uv lock` and commit the lock
alongside `pyproject.toml`; both the pre-commit hook and CI fail on a stale
one. The lock is universal (platform-independent), so it never needs
regenerating just because the machine changed.

## Releases and changelog

Releases are automated, the same arrangement as Tetrak — do not perform
them by hand.

- Every push to `main` runs `release.yml`: the shared `next-version` action
  (`scattercode/release-pipelines`, pinned to `@v1`) computes the next
  semantic version from the Conventional Commit history, git-cliff prepends
  the new section to `CHANGELOG.md`, and the workflow tags and publishes a
  GitHub Release. The computation and its tests live in that repository
  because they were byte-identical in three of ours and nothing detected
  drift; `cliff.toml` and `.githooks/commit-msg` are copies from the same
  place, checked by `sync.sh --check`.
- Never edit `CHANGELOG.md` by hand — change the commit messages or the
  `commit_parsers` in `cliff.toml` instead.
- Never create tags or Releases manually. Trained weights are *attached* to
  the automatically created releases as assets, with checksum and
  provenance record.
- **The package version comes from the git tag**, via `hatch-vcs`. Do not
  add a `version = "..."` literal back to `pyproject.toml` — nothing
  updates it, so it silently goes stale.
  `src/tetrak_hy_trainer/_version.py` is generated at build time and
  gitignored.
- `fix` → patch, `feat` → minor, `!` → major. Choose types accordingly.
- Tooling commits use `chore(release):` so the changelog parser skips them.

## Where decisions live

The staged plan (baselines gate, spike, synthetic pipeline, training,
packaging) and its decision log live in Tetrak's product zone, not here —
this repo does not restate them. Tetrak's ADRs are cited by number from both
satellite repositories ("ADR 001", "decision 003"), so they are load-bearing
here even though they are not published.

Decisions that directly affect this code:

1. **Whether `character_list` includes a space** — *settled*: it does. The
   spike confirmed the EasyOCR trainer and inference path both expect it, and
   every released model carries it. `INCLUDE_SPACE` remains a flag, but
   turning it off would change `num_class` and invalidate every weight file,
   like any other charset change.
2. **և / ԵՎ normalisation policy** — still open. To be decided when enough
   real transcripts are in hand, and recorded in `charset.py` when made.
   Membership of `և` itself is not in question; only whether ԵՎ/Եւ forms in
   ground truth are folded to it.
