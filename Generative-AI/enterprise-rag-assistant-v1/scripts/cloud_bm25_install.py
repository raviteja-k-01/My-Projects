"""Install a SHA-256-locked public demo BM25 asset from an HTTPS archive.

No embedding/reindexing/model calls. Intended for ephemeral free-host instances.
The archive must contain only files under bm25/ and is never trusted before
SHA-256 and path/symlink checks. No secrets are stored in the archive URL.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen
from zipfile import ZipFile

MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 160 * 1024 * 1024
MAX_FILES = 128


def install_archive(data: bytes, *, expected_sha256: str, destination: Path) -> int:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Invalid BM25 archive SHA-256")
    if hashlib.sha256(data).hexdigest() != expected_sha256.lower():
        raise ValueError("BM25 archive SHA-256 mismatch")
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("BM25 archive exceeds size limit")
    if destination.exists():
        raise ValueError("Destination already exists; refusing to overwrite an index")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".bm25-install-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "bm25"
        stage.mkdir()
        with ZipFile(BytesIO(data)) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members or len(members) > MAX_FILES:
                raise ValueError("Empty or oversized BM25 archive file list")
            if sum(info.file_size for info in members) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("BM25 archive expands beyond allowed size")
            seen: set[str] = set()
            for info in members:
                parts = PurePosixPath(info.filename).parts
                if (len(parts) < 2 or parts[0] != "bm25"
                    or any(part in {"", ".", ".."} for part in parts)
                    or info.filename.startswith("/") or "\\" in info.filename):
                    raise ValueError("Unsafe BM25 archive path")
                mode = info.external_attr >> 16
                if stat.S_IFMT(mode) == stat.S_IFLNK:
                    raise ValueError("BM25 archive must not contain symlinks")
                relative = Path(*parts[1:])
                if relative.as_posix() in seen:
                    raise ValueError("Duplicate BM25 archive path")
                seen.add(relative.as_posix())
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
        stage.rename(destination)
        return len(members)


def download_archive(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("BM25 archive URL must use HTTPS")
    if parsed.username or parsed.password:
        raise ValueError("BM25 archive URL must not embed credentials")
    with urlopen(url, timeout=90) as response:
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("BM25 archive download exceeds size limit")
    return data


def main() -> None:
    destination = Path("data/bm25")
    if destination.exists():
        print("BM25 archive install skipped: local destination already present", flush=True)
        return
    url = os.getenv("RAG_BM25_ARCHIVE_URL", "").strip()
    digest = os.getenv("RAG_BM25_ARCHIVE_SHA256", "").strip()
    if not url or not digest:
        raise SystemExit(
            "BM25_DEPLOY_FAILED: configure RAG_BM25_ARCHIVE_URL and "
            "RAG_BM25_ARCHIVE_SHA256, or provide existing data/bm25"
        )
    try:
        count = install_archive(download_archive(url), expected_sha256=digest, destination=destination)
    except Exception as exc:
        # Fail closed; no URL, provider response, or credential is logged.
        raise SystemExit(f"BM25_DEPLOY_FAILED: {type(exc).__name__}") from None
    print(f"BM25_ARCHIVE_INSTALLED: {count} index files, SHA-256 verified", flush=True)


if __name__ == "__main__":
    main()
