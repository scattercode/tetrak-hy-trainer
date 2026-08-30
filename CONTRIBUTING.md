# Contributing

Thanks for your interest in tetrak-hy-trainer. This document covers how we
work in this repository — the checks, the commit style, and the automation
they feed. It applies to us as much as to anyone sending a pull request.

## Before you start: the licensing quarantine

This project is Apache 2.0. A CC BY-NC 4.0 repository
(`portmind-armenian-ocr`) validated the approach we use here, and **nothing
derived from it can be accepted** — no code, however small the fragment, no
test annotations, no weights. Re-implementing *ideas* from it fresh is fine;
copying expression is not. If your contribution draws on that repository, we
have to decline it, so please say so up front if you are unsure.

Relatedly: every data source and font used in training must arrive with its
licence recorded (see the README's data sources section), and code derived
from EasyOCR's Apache 2.0 trainer keeps its headers, with the NOTICE file
updated when such code lands.

## Getting set up

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Then activate the git hooks — one of:

```bash
lefthook install                          # if you have lefthook (brew install lefthook)
git config core.hooksPath .githooks       # dependency-free fallback (commit-msg only)
```

The lefthook hooks run Ruff and the lockfile check before each commit, the
test suite before each push, and the Conventional Commits check on every
commit message.

## The checks

Everything the hooks and CI run, runnable by hand:

```bash
ruff check src tests tools scripts
ruff format --check src tests tools scripts
pytest                  # fast — no OCR stack, no torch, no GPU
uv lock --check         # the lockfile matches pyproject.toml
```

All of these must pass before a pull request can merge. The Ruff rule set is
declared explicitly in `pyproject.toml` — do not fall back to defaults; they
differ between Ruff versions.

## Dependencies and the lockfile

Dependency ranges live in `pyproject.toml`; the fully resolved tree — every
transitive package, all extras, torch included — lives in `uv.lock`. The
lockfile exists so that security scanning (Trivy in CI, Dependabot) sees
real resolved versions rather than ranges.

After any change to `[project.dependencies]` or an extra:

```bash
uv lock
```

and commit `uv.lock` alongside `pyproject.toml`. CI fails on a stale lock.

## Commit style

Commits follow [Conventional Commits](https://www.conventionalcommits.org/),
enforced by the shared hook in `.githooks/`:

```
<type>[(scope)][!]: <description>
```

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
  `build`, `ci`, `chore`, `revert`.
- The description is non-empty, has no trailing full stop, and the header is
  100 characters or fewer (72 or fewer preferred).

This is load-bearing, not cosmetic: the release automation computes the next
semantic version from the commit types (`fix` → patch, `feat` → minor,
`!` → major) and generates `CHANGELOG.md` from the messages. Write the
message for the changelog reader.

## Releases and the changelog

Releases are fully automated — never perform one by hand:

- Every push to `main` runs `release.yml`, which computes the next version
  from the commit history, updates `CHANGELOG.md`, tags, and publishes a
  GitHub Release.
- Never edit `CHANGELOG.md` manually and never create tags or Releases —
  fix the commit messages (or `cliff.toml`) instead.
- The package version comes from the git tag via `hatch-vcs`; there is no
  version literal to bump.

Trained weights are attached to these releases as assets, each with a
checksum and a provenance record — see CLAUDE.md's artefact discipline.
Weights, datasets and crops are never committed to git.

## Conventions

- British English throughout; sentence case for headings.
- Docs are written in the first person plural — this is a collaborative
  project.
- Public docs must not link into the private Tetrak repository's paths;
  link to [tetrak.dev](https://tetrak.dev/) where a pointer is needed.

## Security

Please report suspected vulnerabilities privately — see
[SECURITY.md](SECURITY.md), not the issue tracker.
