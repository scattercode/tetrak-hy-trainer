"""Fetch font files from pinned archives and check them byte for byte.

Both font fetchers (``scripts/fetch_fonts.py`` for the book faces,
``scripts/fetch_display_fonts.py`` for brief 019's display faces) name each
face by a source, a member path inside it, and the SHA-256 of the file they
expect. A source is a URL to a zip, a gzipped tarball or a single file, and
every URL is pinned: a commit SHA for a repository, a version for a release.
A download page that changes under us (fonter.am's links began answering
303 to an HTML page in October 2026) then fails loudly here instead of
handing the renderer something else.

Archives are cached under the caller's directory, so re-running a fetch
after adding one face downloads nothing that is already there.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Face:
    """One font file: where it comes from and what it must hash to.

    ``member`` is the path inside the source archive, or ``None`` when the
    source URL is the font file itself. ``filename`` is what it is saved as.
    """

    family: str
    filename: str
    source: str
    member: str | None
    sha256: str


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, cache: Path, session) -> bytes:
    """The bytes at *url*, from *cache* when fetched before."""
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / sha256_of(url.encode())
    if cached.exists():
        return cached.read_bytes()
    response = session.get(url, timeout=300)
    response.raise_for_status()
    cached.write_bytes(response.content)
    return response.content


def member_of(archive: bytes, member: str) -> bytes:
    """One file out of a zip or a gzipped tarball, by its path inside it."""
    if archive[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            return handle.read(member)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        extracted = handle.extractfile(member)
        if extracted is None:
            raise KeyError(member)
        return extracted.read()


def fetch(face: Face, dest: Path, cache: Path, session) -> Path:
    """Save *face* under *dest*, refusing it if its hash is not the pinned one."""
    target = dest / face.filename
    if target.exists() and sha256_of(target.read_bytes()) == face.sha256:
        return target
    data = download(face.source, cache, session)
    if face.member is not None:
        data = member_of(data, face.member)
    got = sha256_of(data)
    if got != face.sha256:
        raise SystemExit(
            f"{face.filename}: SHA-256 {got} is not the pinned {face.sha256}. "
            f"The source changed ({face.source}); check the new file before re-pinning."
        )
    dest.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def licence_of(path: Path) -> str:
    """The licence the font records about itself (name table 13, else 14)."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(path), fontNumber=0, lazy=True)
    try:
        name = font["name"]
        return (name.getDebugName(13) or name.getDebugName(14) or "(none recorded)").strip()
    finally:
        font.close()
