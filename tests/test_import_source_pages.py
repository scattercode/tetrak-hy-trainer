"""Downloaded scans become an evaluation set; only transcribed pages count."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("isp", REPO / "scripts" / "import_source_pages.py")
isp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isp)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(isp, "SOURCES", tmp_path / "sources")
    monkeypatch.setattr(isp, "EVAL", tmp_path / "eval")
    source = tmp_path / "sources" / "paper"
    source.mkdir(parents=True)
    pages = [Image.new("RGB", (40, 60), "white"), Image.new("RGB", (40, 60), "white")]
    pages[0].save(source / "paper-1882-1.pdf", save_all=True, append_images=pages[1:])
    (source / "selection.tsv").write_text(
        "year\tedition_id\tsize\tissue\turl\n1882\t1\t1MB\tԹիւ 123\thttps://x/1\n",
        encoding="utf-8",
    )
    return "paper"


def test_pages_render_and_only_transcribed_ones_enter_the_manifest(work):
    pages = isp.import_work(work, dpi=50, index_title="Paper")
    assert [p["page_number"] for p in pages] == [188201, 188202]
    assert all((isp.EVAL / work / p["image"]).exists() for p in pages)
    assert pages[0]["issue"] == "Թիւ 123"
    manifest = json.loads((isp.EVAL / work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["index"] == "Paper" and manifest["pages"] == []

    (isp.EVAL / work / "text" / "188202.txt").write_text("Մշակ", encoding="utf-8")
    isp.import_work(work, dpi=50, index_title="Paper")
    manifest = json.loads((isp.EVAL / work / "manifest.json").read_text(encoding="utf-8"))
    assert [p["page_number"] for p in manifest["pages"]] == [188202]
    assert manifest["pages"][0]["text"] == "text/188202.txt"


def test_a_pdf_without_a_year_is_refused(work):
    (isp.SOURCES / work / "nodate.pdf").write_bytes(b"")
    with pytest.raises(RuntimeError):
        isp.page_number_for(isp.SOURCES / work / "nodate.pdf", 1)
