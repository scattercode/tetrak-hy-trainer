# Vendored: the EasyOCR trainer

The Python files in this directory (and `modules/`) are vendored from
[JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)'s `trainer/`
directory (latest upstream trainer commit at vendoring time:
`4c4de08c5d01`, 2023-03-29), under the Apache License 2.0 — see the
repository NOTICE file. They are a deep-text-recognition-benchmark
derivative (NAVER Corp., Apache 2.0).

Vendored rather than pip-installed because upstream does not publish the
trainer as a package, and vendored **unmodified** so diffing against
upstream stays trivial. Everything Tetrak-specific — config generation
from the canonical charset, synthetic data, orchestration — lives in
`src/tetrak_hy_trainer/` and `scripts/`, never here.

Training data layout (from upstream's convention, confirmed in
`dataset.py` and the notebook): `train_data/<select_data>/labels.csv`
with columns `filename,words`, images in the same folder. The charset is
composed as `number + symbol + lang_char`; our config generator passes
the entire canonical charset as `lang_char` with the other two empty, so
the CTC class order is byte-identical to the shipped
`tetrak_hy.yaml`'s `character_list`.
