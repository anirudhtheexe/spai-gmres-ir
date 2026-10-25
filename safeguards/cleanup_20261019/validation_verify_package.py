import hashlib
import json
import pathlib
import stat
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZIP = ROOT / 'submission' / 'SPAI_GMRES_IR_Submission.zip'
OUT = pathlib.Path(__file__).resolve().parent / 'source'
PREFIX = 'SPAI_GMRES_IR/'
MANIFEST = PREFIX + 'SUBMISSION_MANIFEST.json'

assert not OUT.exists(), 'fresh extraction target must not exist'
OUT.mkdir()
zip_hash = hashlib.sha256(ZIP.read_bytes()).hexdigest()
with zipfile.ZipFile(ZIP) as bundle:
    infos = bundle.infolist()
    names = [info.filename for info in infos]
    assert len(names) == len(set(names)), 'duplicate ZIP member'
    records = json.loads(bundle.read(MANIFEST))
    recorded = {PREFIX + row['path']: row for row in records}
    assert len(recorded) == len(records), 'duplicate manifest path'
    assert set(names) == set(recorded) | {MANIFEST}, 'member inventory mismatch'
    for info in infos:
        name = info.filename
        assert name.startswith(PREFIX), name
        relative = pathlib.PurePosixPath(name[len(PREFIX):])
        assert relative.parts and all(part not in ('', '.', '..') for part in relative.parts), name
        assert not relative.is_absolute(), name
        assert relative.parts[0] not in ('.git', 'archives', 'build', 'safeguards', 'submission', 'results', 'data'), name
        assert not stat.S_ISLNK(info.external_attr >> 16), name
        if name == MANIFEST:
            continue
        target = (OUT / pathlib.Path(*relative.parts)).resolve()
        assert target.is_relative_to(OUT), name
        data = bundle.read(info)
        row = recorded[name]
        assert len(data) == row['bytes'], name
        assert hashlib.sha256(data).hexdigest() == row['sha256'], name
        source = ROOT / pathlib.Path(*relative.parts)
        assert source.exists(), 'package path absent in live source: ' + str(relative)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256'], 'package/live source mismatch: ' + str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
print('zip_sha256=' + zip_hash)
print('files=' + str(len(records)))
print('source_bytes=' + str(sum(row['bytes'] for row in records)))
print('archive_bytes=' + str(ZIP.stat().st_size))
print('all_manifest_hashes_and_live_source=match')
print('exclusions_and_containment=pass')
