"""Which works get an evaluation set, and which must never be rebuilt.

``scripts/build_eval_sets.py`` downloads exactly the pages the samplers
refuse, so its selection is the mirror image of the held-out guards. Two
ways for that mirror to crack matter more than the rest:

* a work in the registry silently missing its set, which leaves a
  register unmeasured while the build still reports success; and
* the ASE volume 2 set being rebuilt, which would re-baseline every
  published figure the project has quoted.

Both are checked here against a fabricated ``runs/harvest`` tree, so the
test needs no network and no real harvest.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tetrak_hy_trainer import heldout

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_eval_sets.py"

ASE_INDEX = "Ինդեքս:Հայկական Սովետական Հանրագիտարան (Soviet Armenian Encyclopedia) 2.djvu"


def load_script():
    spec = importlib.util.spec_from_file_location("build_eval_sets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


def write_harvest(root: Path, name: str, index_title: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"index": index_title, "min_quality": 3, "pages": []}),
        encoding="utf-8",
    )


def test_every_registered_work_is_planned(script, tmp_path: Path) -> None:
    """A work with reserved pages gets a set, with exactly those pages.

    The registry is the single source of truth: the script names no work
    of its own, so adding one to WORK_PAGES is what earns it a set.
    """
    root = tmp_path / "harvest"
    for position, needle in enumerate(heldout.WORK_PAGES):
        # The manifest index only has to *contain* the registry's needle,
        # which is how the real titles match.
        write_harvest(root, f"work-{position}", f"Ինդեքս:… {needle} ….djvu")

    plans = script.planned_sets(root)

    assert len(plans) == len(heldout.WORK_PAGES)
    for _, index_title, pages in plans:
        assert pages == heldout.held_out_pages(index_title)
        assert pages, "a planned set with no pages would score on nothing"


def test_the_ase_evaluation_set_is_never_rebuilt(
    script, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Volume 2 is held out whole and its set is the published baseline.

    Today the ASE is excluded twice over -- once by name, once by the
    whole-volume rule -- and also, incidentally, by having no registry
    entry at all. That last one is why this test registers pages against
    volume 2 before asking: without it the assertion passes whether the
    guards are present or not, and a vacuous guard test is worse than
    none, because it is the thing you would point at when deciding the
    behaviour is safe.

    Volume 2 reaches the walk two ways -- as a directory literally named
    ``ase-vol2``, and as any harvest whose manifest says volume 2 -- and
    neither may be planned.
    """
    monkeypatch.setitem(heldout.WORK_PAGES, "Soviet Armenian Encyclopedia", frozenset({105, 106}))
    root = tmp_path / "harvest"
    write_harvest(root, "ase-vol2", ASE_INDEX)
    write_harvest(root, "innocently-renamed", ASE_INDEX)

    assert script.planned_sets(root) == []


def test_the_protected_name_holds_when_the_manifest_does_not_say_volume_two(
    script, tmp_path: Path
) -> None:
    """The name guard is what is left when the volume rule cannot fire.

    ``PROTECTED`` is belt-and-braces over the whole-volume rule for any
    manifest that identifies as volume 2, so this is the case that
    isolates it: a directory sitting at the evaluation set's name whose
    manifest says something else entirely -- a copied tree, a
    hand-edited manifest, a rename. The output path is what would be
    overwritten, and the output path is chosen from the directory name.
    """
    root = tmp_path / "harvest"
    write_harvest(root, "ase-vol2", "Ինդեքս:… Faustus of Byzantium ….djvu")

    assert script.planned_sets(root) == []


def test_a_work_with_no_reserved_pages_is_skipped(script, tmp_path: Path) -> None:
    """Most harvests are training material and get no set of their own."""
    root = tmp_path / "harvest"
    write_harvest(root, "ase-vol9", "Ինդեքս:… (Soviet Armenian Encyclopedia) 9.djvu")

    assert script.planned_sets(root) == []


def test_a_harvest_without_an_index_is_ignored(script, tmp_path: Path) -> None:
    """A manifest missing its index cannot be matched, and must not raise.

    ``runs/`` accumulates hand-made directories; one of them lacking the
    field should not take the whole build down.
    """
    root = tmp_path / "harvest"
    directory = root / "improvised"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")

    assert script.planned_sets(root) == []
