# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-09-02

### Fixed

- Refuse to write a harvest manifest that misstates its provenance (#8)

## [0.5.0] - 2026-09-02

### Changed

- Run the pre-push suite through the project's own interpreter
- Build a per-register evaluation set for every held-out work (#6)
- Score every register, not just the encyclopedia (#7)

## [0.4.0] - 2026-09-01

### Added

- Add hye-calfa-n to v3's baselines and drop the "highest" claim
- Add the census and chr-decomposition scripts (brief 012 stage 0)

### Changed

- Harvest crops from several volumes in one run
- Make the model upload v3-aware
- Per-work held-out registry, and a charset diff for new sources
- Fetch script for the rendering faces, licences verified per file
- Write down the held-out split, the interpreters and the run layout
- Take the version computation from the shared release pipeline
- Charset v3 and transcript normalisation, from the corpus-wide diff
- Retire the handoff, keeping the lessons that outlive it

### Fixed

- Retry transient failures in the Wikisource client
- Stop the public scripts importing the private Tetrak package

## [0.3.0] - 2026-09-01

### Added

- Add Hugging Face dataset packaging and upload
- Add Hugging Face model packaging and upload
- Add the crops-v1 dataset configuration
- Add the real-crop harvest, fine-tune and confusion-report scripts

### Changed

- Resolve --eval-dir before the run and re-push the card alone
- Make the model upload v1-aware
- Score the fold against v1's saved predictions
- Widen the v2 charset with the abbreviation dot and degree sign
- Align detector boxes to transcripts for real-crop harvesting
- Score any model's saved predictions, not just v1's
- Make the model upload v2-aware and record v0/v1's defects

### Fixed

- Skip fonts missing a glyph when choosing a face to render with
- Let --images top up a harvest without losing text or manifest pages
- Stop csv.writer wrapping comma-bearing labels in quotation marks

## [0.2.0] - 2026-08-30

### Added

- Add manual TestPyPI dry-run workflow (#4)

### Changed

- Bump actions/setup-python from 5 to 7 (#1)
- Bump actions/checkout from 4 to 7 (#2)
- Bump actions/upload-artifact from 4 to 7 (#3)
- Publish wheels to PyPI via Trusted Publisher (#5)

## [0.1.0] - 2026-08-30

### Added

- Add the baseline evaluation script
- Add the v0 overnight pre-train orchestrator
- Add contributing and security policies

### Changed

- Scaffold the trainer with its charset and packaging modules
- Adopt the Armenian Soviet Encyclopedia as a data source
- Prove the EasyOCR loading contract with a random-weight spike
- Harvest proofread pages and scans from Armenian Wikisource
- Point distribution at tetrak-easyocr-armenian
- Measure Calfa's paddle-calfa-tiny in the baseline script
- Vendor the EasyOCR trainer
- Complete Stage 1 -- a trained tiny model reads back through EasyOCR
- Train on Apple MPS, measured and verified
- Evaluate a tetrak_hy bundle in the baseline script
- The v1 recipe -- lines, scan scale, degradations, honest eval
- Derive the package version from the git tag via hatch-vcs
- Pin the full dependency tree in uv.lock
- Automate releases and the changelog from the commit history
- Scan dependencies with Trivy and keep them current with Dependabot

### Fixed

- Carry pillow in the dev extra so test collection succeeds

