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
| `tetrak_hy.pth` | Trained weights — published as GitHub Release assets, never committed |

```python
import easyocr

reader = easyocr.Reader(
    ["hy"],
    recog_network="tetrak_hy",
    user_network_directory="path/holding/yaml/and/py",
    model_storage_directory="path/holding/pth",
)
reader.readtext("scan.png")
```

The name is a Python module name (EasyOCR imports it), hence the
underscore. The model uses a CTC head — EasyOCR's custom-model inference
path is CTC-only.

## Status

Early scaffold. The pipeline stages, in order:

1. **Charset** — `src/tetrak_hy_trainer/charset.py`, the single source of
   truth read by both the trainer and the packaging step. ✔ (two decisions
   deliberately open; see the module)
2. **Packaging** — emit a valid `tetrak_hy.yaml` from the charset. ✔
3. **Spike** — train a deliberately tiny model and prove the EasyOCR
   loading contract end to end. Not started.
4. **Synthetic data** — Armenian corpus text rendered in Armenian fonts
   with archival degradations. Not started.
5. **Training** — CTC pre-training on synthetic crops, fine-tuning on
   human-verified real crops. Not started.

## Data and font licences

Recorded as sources are adopted:

| Source | Use | Licence |
|---|---|---|
| *(none adopted yet)* | | |

Planned candidates: Armenian Wikisource (public domain) and Armenian
Wikipedia (CC BY-SA 4.0) for corpus text; the Noto Armenian family, GHEA
faces and Arian AMU (all OFL) for fonts.

## Licence

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Training
code will derive in part from EasyOCR's trainer (Apache 2.0), itself
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
after cloning, or `lefthook install`).
