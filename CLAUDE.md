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
ruff check src tests
ruff format src tests
```

## Where decisions live

The staged plan (baselines gate, spike, synthetic pipeline, training,
packaging) and its decision log live in Tetrak's product zone, not here —
this repo does not restate them. Two decisions currently open there that
directly affect this code:

1. **և / ԵՎ normalisation policy** — decided when real transcripts are in
   hand; recorded in `charset.py` when made.
2. **Whether `character_list` includes a space** — confirmed at spike
   time against what the EasyOCR trainer and inference path actually
   expect; `charset.py` carries the flag.
