"""Verify, restore, or create the lossless research archive (standard library only).

Creation requires an explicit JSON list of project-relative files/directories.
Pruning is a separate command and rechecks both archived and loose file hashes.
Restoration never overwrites a different existing file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "_ARCHIVE_MANIFEST.json"


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def safe_path(name):
    relative = Path(name)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("Expected a project-relative path: " + name)
    candidate = ROOT / relative
    if any(path.is_symlink() for path in (candidate, *candidate.parents) if path != ROOT):
        raise ValueError("Symlink path is not supported: " + name)
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("Path escapes project: " + name)
    if any(part in (".git", "archives") for part in path.relative_to(ROOT).parts):
        raise ValueError("Protected path: " + name)
    return path


def verify(archive):
    with zipfile.ZipFile(archive) as bundle:
        records = json.loads(bundle.read(MANIFEST))
        names = [item["path"] for item in records]
        members = bundle.namelist()
        if (len(set(names)) != len(names) or len(members) != len(names) + 1 or
                set(members) != set(names + [MANIFEST])):
            raise ValueError("Archive member inventory mismatch")
        for item in records:
            safe_path(item["path"])
            with bundle.open(item["path"]) as stream:
                actual = digest(stream)
            if actual != item["sha256"] or bundle.getinfo(item["path"]).file_size != item["bytes"]:
                raise ValueError("Archive hash mismatch: " + item["path"])
    print("Verified %d archived files / %d original bytes" %
          (len(records), sum(item["bytes"] for item in records)), flush=True)
    return records


def create(archive, selection):
    paths = set()
    for name in json.loads(selection.read_text(encoding="utf-8-sig")):
        path = safe_path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        paths.update(path.rglob("*") if path.is_dir() else [path])
    records = []
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(paths):
            if not path.is_file():
                continue
            name = path.relative_to(ROOT).as_posix()
            if path.is_symlink() or safe_path(name) != path:
                raise ValueError("Symlink is not archivable: " + name)
            with path.open("rb") as stream:
                checksum = digest(stream)
            records.append({"path": name, "bytes": path.stat().st_size, "sha256": checksum})
            bundle.write(path, name)
            if len(records) % 100 == 0:
                print("Archived %d files" % len(records), flush=True)
        bundle.writestr(MANIFEST, json.dumps(records, indent=2) + "\n")
    verify(archive)


def prune(archive):
    records = verify(archive)
    # Validate the entire source set before removing any loose file.
    for item in records:
        path = safe_path(item["path"])
        if path.exists():
            with path.open("rb") as stream:
                if digest(stream) != item["sha256"]:
                    raise ValueError("Loose source changed: " + item["path"])
    parents = set()
    for item in records:
        path = safe_path(item["path"])
        if path.exists():
            with path.open("rb") as stream:
                if digest(stream) != item["sha256"]:
                    raise ValueError("Loose source changed before removal: " + item["path"])
        path.unlink(missing_ok=True)
        parents.update(parent for parent in path.parents if parent != ROOT and parent.is_relative_to(ROOT))
    for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    print("Removed verified loose originals; archive retains every byte", flush=True)


def restore(archive):
    records = verify(archive)
    for item in records:
        path = safe_path(item["path"])
        if path.exists():
            with path.open("rb") as stream:
                if digest(stream) != item["sha256"]:
                    raise ValueError("Refusing to overwrite changed file: " + item["path"])
    with zipfile.ZipFile(archive) as bundle:
        for item in records:
            path = safe_path(item["path"])
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item["path"]) as source, path.open("xb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
    print("Restored original relative paths", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify", "prune", "restore"))
    parser.add_argument("--archive", type=Path, default=ROOT / "archives/research_20261019.zip")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        if args.selection is None:
            parser.error("create requires --selection")
        create(args.archive, args.selection)
    else:
        {"verify": verify, "prune": prune, "restore": restore}[args.action](args.archive)


if __name__ == "__main__":
    main()
