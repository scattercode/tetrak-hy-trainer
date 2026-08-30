"""Tests for the release version computation.

This replaced `git cliff --bumped-version`, which on 2026-08-22 filed a
`feat!` with a BREAKING CHANGE footer under an *older* release and reported
"nothing to bump" -- silently skipping 5.0.0. The workflow went green while
the release never happened.

The topology that caused it is reproduced below in a real repository, because
the whole point is that this must not depend on how branches were merged.
"""

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "next_version.py"


def git(repo: Path, *args: str, **kw) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, **kw
    ).stdout


_counter = itertools.count()


def commit(repo: Path, message: str, body: str = "") -> None:
    """Each commit touches its own file, so branches merge without conflicting."""
    name = f"f{next(_counter)}.txt"
    (repo / name).write_text(message, encoding="utf-8")
    git(repo, "add", name)
    full = f"{message}\n\n{body}" if body else message
    git(repo, "commit", "-q", "-m", full)


def release(repo: Path, version: str) -> None:
    """What the release workflow does: a chore(release) commit, then the tag.

    Tagging a separate commit rather than the merge matters -- it is part of
    what makes the failure below reproduce.
    """
    (repo / "CHANGELOG.md").write_text(version, encoding="utf-8")
    git(repo, "add", "CHANGELOG.md")
    git(repo, "commit", "-q", "-m", f"chore(release): {version}")
    git(repo, "tag", "-a", version, "-m", version)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "Test")
    commit(r, "feat: initial")
    git(r, "tag", "-a", "1.0.0", "-m", "1.0.0")
    return r


def run(repo: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(TOOL), *args], cwd=repo, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


class TestBumpSelection:
    def test_a_bang_gives_a_major(self, repo: Path) -> None:
        commit(repo, "feat(pipeline)!: move the working directories")
        assert run(repo)[1] == "2.0.0"

    def test_a_breaking_change_footer_gives_a_major(self, repo: Path) -> None:
        """The footer form is equally valid, and was the one that got lost."""
        commit(
            repo,
            "feat(pipeline): move the working directories",
            "BREAKING CHANGE: scans/ now lives under workspace/.",
        )
        assert run(repo)[1] == "2.0.0"

    def test_a_feature_gives_a_minor(self, repo: Path) -> None:
        commit(repo, "feat: add an output format")
        assert run(repo)[1] == "1.1.0"

    @pytest.mark.parametrize("kind", ["fix", "perf", "revert"])
    def test_a_correction_gives_a_patch(self, repo: Path, kind: str) -> None:
        commit(repo, f"{kind}: close the pdfium handles")
        assert run(repo)[1] == "1.0.1"

    def test_docs_and_chores_alone_release_nothing(self, repo: Path) -> None:
        commit(repo, "docs: write up the investigation")
        commit(repo, "chore(corpus): add fixtures")
        code, out = run(repo)
        assert code == 0
        assert out == ""

    def test_nothing_at_all_releases_nothing(self, repo: Path) -> None:
        code, out = run(repo)
        assert (code, out) == (0, "")

    def test_the_highest_signal_wins(self, repo: Path) -> None:
        """A breaking change among features and fixes still forces a major."""
        commit(repo, "fix: a correction")
        commit(repo, "feat: an addition")
        commit(repo, "refactor!: a breaking rearrangement")
        assert run(repo)[1] == "2.0.0"


class TestTheTopologyThatBrokeIt:
    def test_a_branch_that_merges_main_in_before_merging_back(self, repo: Path) -> None:
        """The 2026-08-22 failure, reproduced exactly.

        Confirmed to distinguish the two implementations: on this topology
        `git cliff --bumped-version` returns the *existing* tag (no release)
        while this returns the major. A simpler version -- branch, tag twice
        on main, merge back -- is not enough; git-cliff handles that one
        correctly. The trigger is the branch merging the tagged mainline into
        itself first, which is what updating a long-lived branch before
        opening its PR does.
        """
        git(repo, "checkout", "-q", "-b", "feat/long-lived")
        commit(repo, "feat(pdf)!: a breaking change made early")

        # main takes two PRs while the branch is open, each tagged afterwards
        # by the release workflow -- so the tags sit on chore(release) commits
        # rather than on the merges.
        for n, (subject, version) in enumerate(
            [("feat: analytics", "1.1.0"), ("fix: a 404 page", "1.1.1")], start=1
        ):
            git(repo, "checkout", "-q", "main")
            git(repo, "checkout", "-q", "-b", f"tmp{n}")
            commit(repo, subject)
            git(repo, "checkout", "-q", "main")
            git(repo, "merge", "-q", "--no-ff", "-m", f"Merge pull request #{n}", f"tmp{n}")
            release(repo, version)

        # the branch is brought up to date, then merged -- this is the trigger
        git(repo, "checkout", "-q", "feat/long-lived")
        git(repo, "merge", "-q", "--no-ff", "-m", "Merge main into branch", "main")
        git(repo, "checkout", "-q", "main")
        git(repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #40", "feat/long-lived")

        code, out = run(repo)
        assert code == 0
        assert out == "2.0.0", "the breaking change on the branch must still cut a major"

    def test_merge_commits_do_not_count_as_releasable(self, repo: Path) -> None:
        """A merge subject must not be parsed as a conventional commit."""
        git(repo, "checkout", "-q", "-b", "side")
        commit(repo, "docs: nothing releasable")
        git(repo, "checkout", "-q", "main")
        git(repo, "merge", "-q", "--no-ff", "-m", "feat: merge that looks like a feature", "side")

        assert run(repo)[1] == "", "the merge itself must not force a release"


class TestTagHandling:
    def test_a_v_prefix_is_preserved(self, repo: Path) -> None:
        commit(repo, "feat: something to tag")
        git(repo, "tag", "-a", "v2.0.0", "-m", "v2.0.0")
        commit(repo, "fix: a correction")
        assert run(repo)[1] == "v2.0.1"

    def test_a_repository_with_no_tags_starts_from_zero(self, tmp_path: Path) -> None:
        r = tmp_path / "fresh"
        r.mkdir()
        git(r, "init", "-q", "-b", "main")
        git(r, "config", "user.email", "t@example.com")
        git(r, "config", "user.name", "Test")
        commit(r, "feat: the first thing")

        assert run(r)[1] == "0.1.0"

    def test_a_non_semver_tag_fails_loudly(self, repo: Path) -> None:
        """Better to stop than to invent a version from something unparseable."""
        commit(repo, "feat: something to tag")
        git(repo, "tag", "-a", "release-candidate", "-m", "rc")
        commit(repo, "feat: an addition")

        code, out = run(repo)
        assert code != 0
        assert out == ""


class TestItIsNeverSilent:
    def test_unconventional_commits_are_reported(self, repo: Path) -> None:
        """The commit-msg hook should stop these, so one here means it was bypassed."""
        commit(repo, "just some words with no type")
        p = subprocess.run([sys.executable, str(TOOL)], cwd=repo, capture_output=True, text=True)
        assert "not conventional commits" in p.stderr

    def test_explain_shows_the_deciding_commit(self, repo: Path) -> None:
        commit(repo, "feat(api)!: the deciding change")
        p = subprocess.run(
            [sys.executable, str(TOOL), "--explain"], cwd=repo, capture_output=True, text=True
        )
        assert "the deciding change" in p.stderr
        assert "bump:           major" in p.stderr
