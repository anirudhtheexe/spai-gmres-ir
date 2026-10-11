"""Untimed, sequential screening with lossless provenance and external watchdog.

Canonical hash: little-endian uint64 rows/cols, offsets, indices, then binary64
values, in that order. No timing measurements or parameter fallback.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(a):
    data = struct.pack('<QQ', a['rows'], a['cols'])
    for key in ('offsets', 'indices'):
        data += b''.join(struct.pack('<Q', x) for x in a[key])
    data += b''.join(struct.pack('<d', x) for x in a['values'])
    return sha(data)


def mm_metadata(raw):
    lines = raw.decode('utf-8-sig').splitlines()
    banner = lines[0] if lines else ''
    comments = [x for x in lines[1:] if x.lstrip().startswith('%')]
    body = [x.strip() for x in lines[1:] if x.strip() and not x.lstrip().startswith('%')]
    result = {'banner': banner, 'comments': comments}
    try:
        rows, cols, entries = map(int, body[0].split())
        seen, duplicates, zeros = set(), 0, 0
        for line in body[1:]:
            fields = line.split()
            key = (int(fields[0]), int(fields[1]))
            duplicates += key in seen
            seen.add(key)
            zeros += float(fields[2]) == 0
        result.update(declared_rows=rows, declared_cols=cols, declared_entries=entries,
                      raw_explicit_zero_count=zeros, raw_duplicate_count=duplicates,
                      entry_count_policy='stored coordinate entries before symmetry expansion')
    except (ValueError, IndexError) as error:
        result['metadata_parse_error'] = str(error)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runner', required=True)
    p.add_argument('--matrix', required=True)
    p.add_argument('--provenance', required=True)
    p.add_argument('--phase', choices=['metadata', 'construct', 'solve'], default='solve')
    p.add_argument('--backend', choices=['cpu_native', 'gpu_native', 'cpu_fixed_p', 'gpu_fixed_p'], required=True)
    p.add_argument('--rhs', default='equal_unit2')
    p.add_argument('--epsilon', default='0.3')
    p.add_argument('--alpha', required=True)
    p.add_argument('--beta', default='8')
    p.add_argument('--innercap', required=True)
    p.add_argument('--outercap', default='20')
    p.add_argument('--P', default='-')
    p.add_argument('--reference', default='-')
    p.add_argument('--output', help='New JSONL artifact; exclusive creation, never overwrite')
    args = p.parse_args()
    raw = Path(args.matrix).read_bytes()
    provenance_raw = Path(args.provenance).read_bytes()
    record = {'schema_version': 1, 'untimed': True, 'validation_outcome': 'unvalidated',
              'watchdog_seconds': 300, 'artificial_memory_cap': None,
              'supplied_provenance': json.loads(provenance_raw),
              'supplied_provenance_raw_base64': base64.b64encode(provenance_raw).decode('ascii'),
              'supplied_provenance_sha256': sha(provenance_raw),
              'derived_provenance': {'raw_path': str(Path(args.matrix).resolve()),
                                     'raw_sha256': sha(raw), 'raw_bytes': len(raw),
                                     'MatrixMarket': mm_metadata(raw)},
              'runner_sha256': sha(Path(args.runner).read_bytes()),
              'environment': {'CUBLAS_EMULATE_DOUBLE_PRECISION': os.environ.get('CUBLAS_EMULATE_DOUBLE_PRECISION')},
              'source_hashes': {}}
    root = Path(__file__).resolve().parents[1]
    for folder in ('src', 'include', 'cuda'):
        for file in sorted((root / folder).rglob('*')):
            if file.is_file() and file.suffix in ('.cpp', '.cu', '.hpp', '.txt'):
                record['source_hashes'][file.relative_to(root).as_posix()] = sha(file.read_bytes())
    command = [args.runner, args.matrix, args.phase, args.backend, args.rhs, args.epsilon,
               args.alpha, args.beta, args.innercap, args.outercap, args.P, args.reference]
    record['command'] = command
    try:
        child = subprocess.run(command, capture_output=True, timeout=300)
        record['returncode'] = child.returncode
        record['raw_stdout'] = child.stdout.decode('utf-8', errors='replace')
        record['raw_stderr'] = child.stderr.decode('utf-8', errors='replace')
        try:
            result = json.loads(record['raw_stdout'])
            record['result'] = result
            record['classification'] = result['classification']
            if 'A' in result:
                a = result['A']
                record['derived_provenance'].update(canonical_rows=a['rows'], canonical_cols=a['cols'],
                    canonical_nnz=a['nnz'], canonical_sha256=canonical_hash(a),
                    canonicalization_notes='CPU loader: symmetry expansion, ascending unique columns; duplicate sum; exact-zero removal. No permutation/scaling of original A.')
            for name in ('b', 'reference'):
                if name in result:
                    record['derived_provenance'][name + '_sha256'] = sha(b''.join(struct.pack('<d', x) for x in result[name]))
        except (ValueError, KeyError, TypeError, struct.error) as error:
            record.update(classification='resource_abort' if child.returncode else 'unsupported_input',
                          substatus='process_or_record_failure', message=str(error))
    except subprocess.TimeoutExpired as error:
        record.update(classification='resource_abort', substatus='external_watchdog_timeout',
                      message='300-second watchdog: terminated process; no fallback or retry',
                      raw_stdout=(error.stdout or b'').decode('utf-8', errors='replace'),
                      raw_stderr=(error.stderr or b'').decode('utf-8', errors='replace'))
    except OSError as error:
        record.update(classification='unsupported_input', substatus='process_launch_failure', message=str(error))
    line = json.dumps(record, allow_nan=False, separators=(',', ':'))
    if args.output:
        with open(args.output, 'x', encoding='utf-8', newline='\n') as artifact:
            artifact.write(line + '\n')
    else:
        print(line)


if __name__ == '__main__':
    main()
