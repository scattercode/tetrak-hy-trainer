# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

