# tetrak-hy-trainer

Training pipeline for an Armenian text recogniser — synthetic data
generation, CTC model training, and packaging as an
[EasyOCR](https://github.com/JaidedAI/EasyOCR) custom model.

## Why

No mainstream local OCR engine reads Armenian well. EasyOCR and PaddleOCR do
not list the language at all; Tesseract ships `hye` traineddata of
unmeasured quality on archival material. Yet the architecture EasyOCR
already uses — CRAFT text detection feeding a compact CTC recogniser — is
proven on Armenian: a National Library of Armenia-adjacent system built on
exactly these components reported character error rates better than Google
Cloud Vision on dense newsprint.

Detection needs no training (CRAFT is script-agnostic). All the
Armenian-specific work concentrates in one small trainable model, and
EasyOCR has a documented custom-model mechanism to load it. This repository
builds that model.

## What it produces

Three files, loadable by stock EasyOCR:

| File | Contents |
|---|---|
| `tetrak_hy.yaml` | Character list, language list, image height, network parameters |
| `tetrak_hy.py` | The recognition network module (`Model(num_class, **network_params)`) |
| `tetrak_hy.pth` | Trained weights — published to the Hugging Face model repository [tetrak/easyocr-armenian](https://huggingface.co/tetrak/easyocr-armenian) and mirrored onto GitHub Releases, never committed |

```python
import easyocr

reader = easyocr.Reader(
    ["en"],  # not ["hy"]: EasyOCR ships no hy_char.txt and would raise; the
    # language list is inert for a custom model, whose charset comes from the yaml
    recog_network="tetrak_hy",
    user_network_directory="path/holding/yaml/and/py",
    model_storage_directory="path/holding/pth",
)
reader.readtext("scan.png")
```

The name is a Python module name (EasyOCR imports it), hence the
underscore. The model uses a CTC head — EasyOCR's custom-model inference
path is CTC-only.

**Distribution happens through
[tetrak-easyocr-armenian](https://github.com/scattercode/tetrak-easyocr-armenian)**,
the installable library whose import package *is* the `tetrak_hy` network:
this trainer's output — the yaml, the weights and their provenance — lands
there as a pull request, and a weights release on that repo is what users
install. The bundle-writing here remains the tool for local spikes and for
producing that PR.

## Status

Trained, released and measured. The current model is **v6** (October
2026), shipped with its word list in
[tetrak-easyocr-armenian](https://github.com/scattercode/tetrak-easyocr-armenian)
and consumed by Tetrak as its `easyocr-hy` backend.

What v6 is: v5, a CTC recogniser pre-trained on synthetic line crops in
fifteen faces from proofread Armenian Wikisource text, fine-tuned again on
99,521 real crops cut from the scans of sixteen works. Its charset admits
174 characters plus the CTC blank, unchanged since v5.

How it reads, on pages held out from training: over eight evaluation
registers (ten pages each from seven works plus volume 2 of the Armenian
Soviet Encyclopedia), Tetrak's pipeline built on v6 leads every engine we
have measured on both mean word recall and mean character similarity, and
leads both metrics on five of the eight registers. It still trails on
Faustus of Byzantium's bold-italic back matter, on Tumanyan's Russian
apparatus (the charset has no Cyrillic) and on the medical encyclopedia's
index dashes, which EasyOCR's detector never boxes. The live tables are on
[tetrak.dev](https://tetrak.dev/articles/2026/10/beating-the-baselines/),
read from the comparison CSV rather than copied here. The raw recogniser's
figures are in each release's provenance record.

Figures published for v5 and earlier used a metric that brief 013 found
to be wrong: difflib's `autojunk`, which on a page of text ignores most of
the alphabet, and transcription conventions scored as errors. They are not
comparable with v6's.

The pipeline stages, all in use:

1. **Charset** — `src/tetrak_hy_trainer/charset.py`, the single source of
   truth read by both the trainer and the packaging step, with a
   corpus-wide stray-character diff run before any new source is admitted.
2. **Census and harvest** — every ProofreadPage index on Armenian
   Wikisource counted, then proofread pages and their scans fetched per
   work; a per-work held-out registry (`heldout.py`) reserves evaluation
   pages before training sees them.
3. **Synthetic data** — corpus text rendered in the fetched faces with
   archival degradations, glyph coverage checked per face.
4. **Training** — CTC pre-training on synthetic crops, then fine-tuning on
   real crops aligned from detector boxes to transcripts.
5. **Evaluation** — every register scored on every run, and the external
   engines run on the same pages, with their raw readings saved so a metric
   change is a re-score (`scripts/rescore_baselines.py`).
6. **Packaging and release** — the three-file bundle, the provenance
   record, and the upload to Hugging Face that a weights pull request to
   the library then pins.

Next: real crops from bold-italic index pages for the Faustus back matter,
and Cyrillic and classical orthography, which are a new charset and so a
new model.

## Data and font licences

Recorded as sources are adopted:

| Source | Use | Licence |
|---|---|---|
| [Armenian Soviet Encyclopedia](https://hy.wikisource.org/wiki/Հայկական_սովետական_հանրագիտարան) on Armenian Wikisource (13 volumes, 1974–1987) | Corpus text for synthesis; paired page scans + transcripts for fine-tuning crops and evaluation | CC BY-SA 3.0, as stated by the hosting Wikisource page |
| Further proofread works on Armenian Wikisource, adopted for v4/v5: Faustus of Byzantium (1968), the Armenian–English practical dictionary, Totovents, Otyan and Baronian (vol. 10) — Western Armenian — Tumanyan's academic edition (vol. 5), and the Popular Medical Encyclopedia | Corpus text for synthesis; page scans + transcripts for fine-tuning crops and per-register evaluation | CC BY-SA, on Wikisource's hosting terms; per-page revision provenance is recorded in each harvest manifest and the dataset card |
| Noto Sans Armenian, Noto Serif Armenian | Rendering faces | SIL Open Font Licence 1.1, read from the font files |
| Arian AMU (four faces) | Rendering faces | SIL Open Font Licence 1.1, read from the font files |
| GHEA Grapalat, GHEA Mariam (eight faces) | Rendering faces | Armenian National Book Chamber free-use terms — **not** OFL; used for rendering only, never redistributed |
| Mshtakan | Rendering face | Ships with macOS; picked up from the system, never redistributed |

Two disciplines attach to the encyclopedia source:

- **Only proofread pages.** Wikisource seeds unproofread pages with
  machine OCR; training on those would teach the model another engine's
  mistakes. Harvesting filters on ProofreadPage quality status
  (proofread/validated only), via the API.
- **Attribution and share-alike.** BY is satisfied by this table and the
  provenance records shipped with weights. Whether SA obligations
  propagate to trained weights is a genuinely unsettled question; we
  record the source and licence with every release so the position is
  auditable either way.

`scripts/fetch_fonts.py` fetches the faces and prints the licence each
font file declares about itself, which is how the GHEA correction above was
found. Further candidates: Armenian Wikisource's public-domain period texts
and Armenian Wikipedia (CC BY-SA 4.0) for corpus text.

## Licence

Apache License 2.0 — see
[LICENSE](https://github.com/scattercode/tetrak-hy-trainer/blob/main/LICENSE)
and
[NOTICE](https://github.com/scattercode/tetrak-hy-trainer/blob/main/NOTICE).
The vendored trainer derives from EasyOCR's trainer (Apache 2.0), itself
derived from NAVER's
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
(Apache 2.0).

**A deliberate exclusion:** this project was informed by studying
[portmind/armenian-ocr](https://github.com/portmind/armenian-ocr)
(CC BY-NC 4.0), whose approach it independently reproduces from
permissively-licensed parts. No code, annotations or weights from that
project are included here, and contributions derived from it cannot be
accepted — its non-commercial licence is incompatible with this one.

## Relationship to Tetrak

This is a satellite of [Tetrak](https://tetrak.dev/), a local-first
transcription pipeline for archival material. Tetrak ships the inference
files and consumes the released weights as its `easyocr-hy` backend;
benchmark results against its evaluation corpus are published there.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check src tests && ruff format --check src tests
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/),
enforced by the hook in `.githooks/` (`git config core.hooksPath .githooks`
after cloning, or `lefthook install`). Releases and `CHANGELOG.md` are
generated from those commits automatically on every push to `main`.

See
[CONTRIBUTING.md](https://github.com/scattercode/tetrak-hy-trainer/blob/main/CONTRIBUTING.md)
for the full workflow — checks, the dependency lockfile, and what the
automation expects — and
[SECURITY.md](https://github.com/scattercode/tetrak-hy-trainer/blob/main/SECURITY.md)
for how to report a vulnerability.
