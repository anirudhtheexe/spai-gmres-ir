"""Create a source-only Teams ZIP using an explicit allowlist (Python 3.10+)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src", "include", "cmake", "cuda", "tests", "scripts", "docs", ".codex/agents")
ROOT_FILES = ("README.md", "CMakeLists.txt", ".clang-format", ".gitignore",
              "AGENTS.md", "PROJECT_HANDOFF.md", "CODEBASE_GUIDE.md", "PAPER_INDEX.md")
SUFFIXES = {".cpp", ".cu", ".hpp", ".h", ".cmake", ".txt", ".mtx", ".py", ".cmd", ".md", ".toml"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission/SPAI_GMRES_IR_Submission.zip")
    args = parser.parse_args()
    paths = {ROOT / name for name in ROOT_FILES}
    for directory in DIRECTORIES:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts:
                paths.add(path)
    manifest = []
    # Fail before opening a destination if a required source is absent.
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Unexpected source path: " + str(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(suffix=".zip", dir=args.output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        write_archive(temporary, paths, manifest)
        with zipfile.ZipFile(temporary) as bundle:
            for item in manifest:
                if hashlib.sha256(bundle.read("SPAI_GMRES_IR/" + item["path"])).hexdigest() != item["sha256"]:
                    raise ValueError("Submission archive verification failed")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print("%s: %d files, %d bytes compressed, %d bytes source" %
          (args.output, len(manifest), args.output.stat().st_size, sum(item["bytes"] for item in manifest)))


def write_archive(destination, paths, manifest):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(paths):
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError("Unexpected source path: " + str(path))
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            manifest.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            bundle.writestr("SPAI_GMRES_IR/" + name, data)
        bundle.writestr("SPAI_GMRES_IR/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
