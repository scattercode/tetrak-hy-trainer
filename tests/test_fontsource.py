"""Pinned font fetching: a changed source must fail, not slip through.

fonter.am's download links began answering 303 to an HTML page in October
2026; without a pinned hash the fetcher would have handed the renderer
whatever came back. These run with no network: the session is a stand-in.
"""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tetrak_hy_trainer import fontsource
from tetrak_hy_trainer.fontsource import Face


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


class _Session:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls = 0

    def get(self, url: str, timeout: int) -> _Response:
        self.calls += 1
        return _Response(self.content)


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _tar_gz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize("pack", [_zip, _tar_gz])
def test_a_member_comes_out_of_either_archive_kind(pack) -> None:
    archive = pack({"root/TTF/a.ttf": b"font a", "root/TTF/b.ttf": b"font b"})
    assert fontsource.member_of(archive, "root/TTF/b.ttf") == b"font b"


def test_a_pinned_face_is_saved(tmp_path: Path) -> None:
    data = b"the real font"
    face = Face("F", "f.ttf", "https://example/f.zip", "f.ttf", fontsource.sha256_of(data))
    path = fontsource.fetch(
        face, tmp_path / "out", tmp_path / "cache", _Session(_zip({"f.ttf": data}))
    )
    assert path.read_bytes() == data


def test_a_changed_source_is_refused(tmp_path: Path) -> None:
    face = Face("F", "f.ttf", "https://example/f", None, fontsource.sha256_of(b"what we pinned"))
    with pytest.raises(SystemExit, match="is not the pinned"):
        fontsource.fetch(face, tmp_path / "out", tmp_path / "cache", _Session(b"<html>303</html>"))
    assert not (tmp_path / "out" / "f.ttf").exists()


def test_a_cached_archive_is_not_downloaded_twice(tmp_path: Path) -> None:
    data = _zip({"a.ttf": b"a", "b.ttf": b"b"})
    session = _Session(data)
    for name in ("a.ttf", "b.ttf"):
        face = Face(
            "F", name, "https://example/all.zip", name, fontsource.sha256_of(name[0].encode())
        )
        fontsource.fetch(face, tmp_path / "out", tmp_path / "cache", session)
    assert session.calls == 1
