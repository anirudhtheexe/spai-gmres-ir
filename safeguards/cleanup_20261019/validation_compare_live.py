import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = Path(__file__).resolve().parent / 'source'
DIRECTORIES = ('src', 'include', 'cmake', 'cuda', 'tests', 'scripts', '.codex/agents')
ROOT_FILES = ('CMakeLists.txt',)
SUFFIXES = {'.cpp', '.cu', '.hpp', '.h', '.cmake', '.txt', '.mtx', '.py', '.cmd', '.toml'}

names = set(ROOT_FILES)
for directory in DIRECTORIES:
    for path in (ROOT / directory).rglob('*'):
        if path.is_file() and path.suffix in SUFFIXES and '__pycache__' not in path.parts:
            names.add(path.relative_to(ROOT).as_posix())
different = []
missing = []
for name in sorted(names):
    live = ROOT / name
    packaged = EXTRACTED / name
    if not packaged.is_file():
        missing.append(name)
        continue
    if hashlib.sha256(live.read_bytes()).digest() != hashlib.sha256(packaged.read_bytes()).digest():
        different.append(name)
print('live_relevant_files=' + str(len(names)))
print('matching_extracted=' + str(len(names) - len(missing) - len(different)))
print('missing_from_tested_zip=' + repr(missing))
print('different_live_vs_extracted=' + repr(different))
assert missing == ['tests/test_submission_tools.py'], missing
assert different == [], different
