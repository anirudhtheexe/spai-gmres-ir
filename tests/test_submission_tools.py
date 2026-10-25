"""Safety checks for archival deletion and transactional submission packaging."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def temporary_workspace():
    directory = (ROOT / "build/tool-tests").resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("Test workspace escapes project")
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=directory)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveSafety(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_workspace()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = load("research_archive")
        self.archive.ROOT = self.root
        (self.root / "evidence.txt").write_text("original research bytes")
        selection = self.root / "selection.json"
        selection.write_text(json.dumps(["evidence.txt"]))
        self.bundle = self.root / "sample.zip"
        self.archive.create(self.bundle, selection)

    def test_changed_source_prevents_prune(self):
        source = self.root / "evidence.txt"
        source.write_text("new bytes must survive")
        with self.assertRaises(ValueError):
            self.archive.prune(self.bundle)
        self.assertEqual(source.read_text(), "new bytes must survive")

    def test_round_trip_and_overwrite_refusal(self):
        self.archive.prune(self.bundle)
        self.assertFalse((self.root / "evidence.txt").exists())
        self.archive.restore(self.bundle)
        self.assertEqual((self.root / "evidence.txt").read_text(), "original research bytes")
        (self.root / "evidence.txt").write_text("changed")
        with self.assertRaises(ValueError):
            self.archive.restore(self.bundle)

    def test_duplicate_member_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.bundle, "a") as bundle:
                bundle.writestr("evidence.txt", "ambiguous duplicate")
        with self.assertRaises(ValueError):
            self.archive.verify(self.bundle)

    def test_protected_paths_rejected(self):
        for name in ("../escape", str(self.root / "absolute"), ".git/config", "archives/data"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.archive.safe_path(name)


class PackageSafety(unittest.TestCase):
    def test_missing_required_source_preserves_previous_zip(self):
        module = load("package_submission")
        with temporary_workspace() as temporary:
            module.ROOT = Path(temporary)
            destination = module.ROOT / "submission.zip"
            destination.write_bytes(b"previous completed submission")
            with patch.object(sys, "argv", ["package_submission", "--output", str(destination)]):
                with self.assertRaises(FileNotFoundError):
                    module.main()
            self.assertEqual(destination.read_bytes(), b"previous completed submission")


if __name__ == "__main__":
    unittest.main()
