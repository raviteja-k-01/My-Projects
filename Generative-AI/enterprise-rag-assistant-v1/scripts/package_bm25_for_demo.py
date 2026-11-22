"""Package the EXISTING compatible local BM25 index for a public demo.

Never rebuilds, re-embeds, or modifies the source BM25 directory. Output is
a separate, gitignored archive, suitable for a GitHub Release asset ONLY if
the fictional source corpus is approved for public access.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.cloud_bm25_install import MAX_ARCHIVE_BYTES, MAX_FILES


def package(directory: Path, output: Path) -> dict:
    if not directory.is_dir():
        raise ValueError("Original BM25 artifact directory is missing")
    if output.exists():
        raise ValueError("Output exists; refusing overwrite")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files or len(files) > MAX_FILES or any(p.is_symlink() for p in files):
        raise ValueError("Invalid BM25 file count or symlink in input")
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Symlinks prohibited")
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Archive output must be OUTSIDE original BM25 directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for file in files:
                archive.write(file, arcname="bm25/" + file.relative_to(directory).as_posix())
        if output.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds public demo size limit")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return {"file": str(output), "sha256": digest, "bytes": output.stat().st_size,
                "files": len(files)}
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-dir", type=Path, default=Path("data/bm25"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/bm25_demo.zip"))
    args = parser.parse_args()
    try:
        result = package(args.bm25_dir, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"BM25_PACKAGE_FAILED: {exc}\n")
    import json
    print(json.dumps(result, indent=2))
    print("DO NOT upload index to a public Release unless its content may be public.")


if __name__ == "__main__":
    main()
