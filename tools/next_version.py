#!/usr/bin/env python3
"""Compute the next semantic version from the commits since the last tag.

Why this exists rather than `git cliff --bumped-version`:

    git-cliff answers "what is the next version?" by first bucketing the
    *entire* history into releases, and that bucketing goes wrong on merge
    topology. On 2026-08-22 a branch cut before 4.3.0 and merged after 4.4.1
    had all six of its commits -- including a `feat!` with a BREAKING CHANGE
    footer -- filed under release 4.3.0, whose tag is not even an ancestor of
    them. With nothing left unreleased, git-cliff reported "nothing to bump"
    and the 5.0.0 release was silently skipped. See
    `product/design-research/` for the investigation.

    The version only ever depended on one commit range. Reading that range
    directly is deterministic, independent of topology, and testable. git-cliff
    still writes the changelog, where a mis-bucketed old release is a cosmetic
    wart rather than a blocked release.

Run:
    python tools/next_version.py              # print the version, or nothing
    python tools/next_version.py --explain    # and say how it was decided
    python tools/next_version.py --github     # emit GITHUB_OUTPUT lines

Exit codes:
    0  a version was computed, or there is legitimately nothing to release
    1  the history is inconsistent -- see the message. Never silent.

Standard library only: this runs on a CI runner with no project install.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# type(scope)!: subject -- the `!` is what marks a breaking change inline.
_HEADER = re.compile(
    r"^(?P<kind>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<subject>.+)$"
)

# Types that justify cutting a release at all. Anything else (docs, chore,
# style, test, ci, build, refactor) rides along in the changelog without
# forcing a version of its own.
_MINOR_TYPES = {"feat"}
_PATCH_TYPES = {"fix", "perf", "revert"}

_SEPARATOR = "\x1e"
_FIELD = "\x1f"


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def latest_tag() -> str:
    """The most recent tag reachable from HEAD, or empty for a fresh repository."""
    try:
        return _git("describe", "--tags", "--abbrev=0").strip()
    except RuntimeError:
        return ""


def commits_since(tag: str) -> list[tuple[str, str]]:
    """(subject, body) for every non-merge commit after *tag*.

    Merges are excluded with --no-merges rather than by matching "Merge ",
    which would also drop a legitimate `fix: merge...` subject.
    """
    span = f"{tag}..HEAD" if tag else "HEAD"
    raw = _git("log", "--no-merges", f"--format={_SEPARATOR}%s{_FIELD}%b", span)
    out = []
    for entry in raw.split(_SEPARATOR):
        if not entry.strip():
            continue
        subject, _, body = entry.partition(_FIELD)
        out.append((subject.strip(), body))
    return out


def classify(commits: list[tuple[str, str]]) -> dict:
    """Decide the bump, and record why so the decision can be printed."""
    breaking: list[str] = []
    features: list[str] = []
    fixes: list[str] = []
    other = 0
    unconventional: list[str] = []

    for subject, body in commits:
        match = _HEADER.match(subject)
        if not match:
            unconventional.append(subject)
            continue
        kind = match.group("kind").lower()
        # A breaking change is marked either inline with `!` or by a footer.
        # Both are Conventional Commits; accepting only one silently loses
        # releases, which is the failure this file exists to prevent.
        if match.group("bang") or re.search(r"^BREAKING[ -]CHANGE:", body, re.M):
            breaking.append(subject)
        elif kind in _MINOR_TYPES:
            features.append(subject)
        elif kind in _PATCH_TYPES:
            fixes.append(subject)
        else:
            other += 1

    if breaking:
        bump = "major"
    elif features:
        bump = "minor"
    elif fixes:
        bump = "patch"
    else:
        bump = ""

    return {
        "bump": bump,
        "breaking": breaking,
        "features": features,
        "fixes": fixes,
        "other": other,
        "unconventional": unconventional,
    }


def apply_bump(tag: str, bump: str) -> str:
    """Raise *tag* by *bump*, preserving any `v` prefix it carries."""
    if not bump:
        return ""
    prefix = "v" if tag.startswith("v") else ""
    core = tag[1:] if prefix else tag
    parts = (core or "0.0.0").split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise RuntimeError(
            f"latest tag {tag!r} is not a three-part semantic version, so the "
            f"next one cannot be derived from it"
        )
    major, minor, patch = (int(p) for p in parts)
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f"{prefix}{major}.{minor}.{patch}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explain", action="store_true", help="show how the bump was decided")
    parser.add_argument("--github", action="store_true", help="write GITHUB_OUTPUT lines")
    args = parser.parse_args(argv)

    tag = latest_tag()
    commits = commits_since(tag)
    verdict = classify(commits)
    version = apply_bump(tag, verdict["bump"])

    if args.explain or args.github:
        span = f"{tag}..HEAD" if tag else "HEAD (no tags yet)"
        print(f"range:          {span}", file=sys.stderr)
        print(f"commits:        {len(commits)} (merges excluded)", file=sys.stderr)
        for label in ("breaking", "features", "fixes"):
            for subject in verdict[label]:
                print(f"  {label[:5]:<5}      {subject[:72]}", file=sys.stderr)
        if verdict["other"]:
            print(f"  other       {verdict['other']} commit(s), no version impact", file=sys.stderr)
        for subject in verdict["unconventional"]:
            print(f"  IGNORED     not a conventional commit: {subject[:56]}", file=sys.stderr)
        print(f"bump:           {verdict['bump'] or 'none'}", file=sys.stderr)
        print(f"next version:   {version or '<no release>'}", file=sys.stderr)

    # The failure this script exists to make impossible: commits that should
    # have produced a release, and no release. It cannot happen by
    # construction, so if it ever does the logic above is wrong and the run
    # must fail rather than pass quietly the way the old one did.
    if verdict["bump"] and not version:
        print(
            f"error: {verdict['bump']} change detected but no version computed from tag {tag!r}",
            file=sys.stderr,
        )
        return 1

    # Unconventional subjects on main are worth surfacing: the commit-msg hook
    # should make them impossible, so one appearing means the hook was bypassed
    # and a release-worthy change may be invisible here.
    if verdict["unconventional"]:
        print(
            f"warning: {len(verdict['unconventional'])} commit(s) since {tag or 'the start'} "
            f"are not conventional commits and were ignored when deciding the version",
            file=sys.stderr,
        )

    if args.github and (path := os.environ.get("GITHUB_OUTPUT")):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"next={version}\n")
            fh.write(f"release={'true' if version else 'false'}\n")
            fh.write(f"bump={verdict['bump']}\n")

    if version:
        print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
