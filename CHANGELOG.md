# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2026-11-09

### Changed

- Report Cyrillic coverage per render face for charset v5 (#73)

## [0.10.1] - 2026-11-09

### Changed

- Bump fonttools from 4.67.0 to 4.68.0 (#69)
- Bump pandas from 3.1.0 to 3.1.1 (#70)

### Fixed

- Keep Abovyan vol. 1 p. 69 out of training (#72)

## [0.10.0] - 2026-11-09

### Changed

- V5 appends the Armenian apostrophe and Russian alphabet (#71)

## [0.9.0] - 2026-11-09

### Changed

- Brief 014 Stage 0 — orthography census, poem fix, held-out proposer (#30)

## [0.8.0] - 2026-11-04

### Changed

- Bump ruff from 0.16.8 to 0.16.9 (#27)
- Bump torchvision from 0.29.0 to 0.29.1 (#26)
- Bump fonttools from 4.66.0 to 4.66.1 (#25)
- Bump torch from 2.14.0 to 2.14.1 (#28)
- Move the house rules to AGENTS.md for any coding agent (#31)
- Bump ruff from 0.16.9 to 0.16.10 (#32)
- Bump fonttools from 4.66.1 to 4.67.0 (#53)
- Bump ruff from 0.16.10 to 0.17.0 (#54)
- Move the lock past the pillow, filelock and certifi advisories (#60)
- Bump setuptools from 84.0.0 to 84.1.0 (#57)
- Bump ruff from 0.17.0 to 0.17.2 (#58)
- Bump nltk from 3.10.3 to 3.10.4 (#59)
- Bump pandas from 3.0.6 to 3.1.0 (#65)
- Bump torch from 2.14.1 to 2.15.0 (#64)
- Bump pytest from 9.1.1 to 9.2.0 (#63)
- Bump ruff from 0.17.2 to 0.17.3 (#62)
- Bump torchvision from 0.29.1 to 0.29.2 (#61)
- Grant permissions per job and pass the branch through env (#67)
- Display faces and a face-held-out test set for modern print (#68)

## [0.7.0] - 2026-10-05

### Changed

- Bump fonttools from 4.63.0 to 4.66.0 (#21)
- Bump pandas from 3.0.5 to 3.0.6 (#20)
- Bump ruff from 0.16.5 to 0.16.8 (#18)
- Bump torch from 2.13.0 to 2.14.0 (#15)
- Bump actions/download-artifact from 7 to 8 (#13)
- Bring CLAUDE.md and the training skill up to date (#24)
- Charset v4 keeps the angle brackets the page prints (#29)

## [0.6.1] - 2026-10-03

### Changed

- Make hye-calfa-n a named backend rather than an environment trick (#12)

### Fixed

- Score Armenian pages fairly and teach the Armenian full stop (#22)

## [0.6.0] - 2026-09-04

### Changed

- Record the v5 release facts (#10)
- Make hye-calfa-n a named backend rather than an environment trick (#11)

## [0.5.2] - 2026-09-02

### Fixed

- Name crops for their source work, not the ASE volume number (#9)

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

