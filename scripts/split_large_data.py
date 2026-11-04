#!/usr/bin/env python3
"""Split generated data files above 90 MiB and restore them before the next cycle.

The original bytes are preserved exactly. A manifest records chunk order, size,
and SHA-256 so restoration can detect missing, reordered, or damaged parts.
Only files under data/ are managed; source code is never split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LIMIT = 90 * 1024 * 1024
MANIFEST = DATA / ".large_files_manifest.json"
PARTS_DIR = DATA / ".large_file_parts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore() -> int:
    if not MANIFEST.exists():
        print("No split data files to restore.")
        return 0
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    restored = 0
    for item in manifest.get("files", []):
        destination = ROOT / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".restoring")
        digest = hashlib.sha256()
        total = 0
        with temporary.open("wb") as output:
            for part_name in item["parts"]:
                part = ROOT / part_name
                if not part.is_file():
                    raise FileNotFoundError(f"Missing split part: {part_name}")
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        total += len(block)
        if total != item["size"] or digest.hexdigest() != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Integrity check failed while restoring {item['path']}")
        temporary.replace(destination)
        restored += 1
        print(f"Restored {item['path']} ({total} bytes)")
    MANIFEST.unlink()
    if PARTS_DIR.exists():
        for path in sorted(PARTS_DIR.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        PARTS_DIR.rmdir()
    print(f"Restored {restored} file(s).")
    return restored


def split() -> int:
    if MANIFEST.exists():
        restore()
    entries = []
    for path in sorted(DATA.rglob("*")):
        if not path.is_file() or PARTS_DIR in path.parents or path == MANIFEST:
            continue
        size = path.stat().st_size
        if size <= LIMIT:
            continue
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256_file(path)
        part_dir = PARTS_DIR / path.relative_to(DATA).parent
        part_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        with path.open("rb") as source:
            index = 1
            while block := source.read(LIMIT):
                part_path = part_dir / f"{path.name}.part{index:04d}"
                part_path.write_bytes(block)
                parts.append(part_path.relative_to(ROOT).as_posix())
                index += 1
        entries.append({
            "path": relative,
            "size": size,
            "sha256": digest,
            "parts": parts,
        })
        path.unlink()
        print(f"Split {relative} ({size} bytes) into {len(parts)} part(s)")
    if entries:
        MANIFEST.write_text(
            json.dumps({"format": 1, "chunk_limit_bytes": LIMIT, "files": entries},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        print("No data files exceed 90 MiB.")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("split", "restore"))
    args = parser.parse_args()
    (split if args.action == "split" else restore)()


if __name__ == "__main__":
    main()
