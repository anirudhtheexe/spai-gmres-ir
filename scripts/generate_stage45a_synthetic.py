"""Deterministic validation matrices; no solver changes or performance clocks.

The fixed manifest is declared before screening. All weights, diagonal sums,
and manufactured solution entries are dyadic and exactly representable here.
"""
import argparse
import hashlib
import json
from pathlib import Path

VERSION = 'stage45a-dyadic-v1'


def cases():
    return ([{'id': 'stencil_%d' % (side * side), 'family': 'stencil', 'n': side * side,
              'side': side} for side in (8, 16, 32)] +
            [{'id': 'irregular_%d' % n, 'family': 'irregular', 'n': n} for n in (64, 256, 1024)] +
            [{'id': 'arrowhead_%d' % n, 'family': 'arrowhead', 'n': n, 'long_row_degree': degree}
             for n, degree in ((64, 16), (256, 32), (1024, 64))])


def rows_for(case):
    n = case['n']
    rows = [{} for _ in range(n)]
    family = case['family']
    if family == 'stencil':
        side = case['side']
        if n != side * side:
            raise ValueError('Invalid stencil dimensions')
        for i, row in enumerate(rows):
            y, x = divmod(i, side)
            for valid, j, weight in ((x > 0, i - 1, .25), (x + 1 < side, i + 1, .125),
                                      (y > 0, i - side, .5), (y + 1 < side, i + side, .0625)):
                if valid:
                    row[j] = -weight
    elif family == 'irregular':
        # Uneven block sizes 3,5,8,13 repeated, with final block truncated.
        start, block = 0, 0
        while start < n:
            width = min((3, 5, 8, 13)[block % 4], n - start)
            for local in range(width):
                i = start + local
                for shift in range(1, 1 + min(width - 1, 1 + i % 5)):
                    j = start + (local + shift) % width
                    rows[i][j] = -((1 + (i + shift) % 4) / 16)
                # Directed sparse connection across blocks prevents block diagonal form.
                j = (i + width + 1) % n
                if j != i:
                    rows[i][j] = -.0625
            start += width
            block += 1
    elif family == 'arrowhead':
        degree = case['long_row_degree']
        if not 1 <= degree < n:
            raise ValueError('Invalid long-row degree')
        # Hub supports spread evenly; one controlled long row, not a dense matrix.
        hubs = [1 + (k * (n - 1)) // degree for k in range(degree)]
        for k, j in enumerate(hubs):
            rows[0][j] = -(1 + k % 4) / 32
            rows[j][0] = -(1 + k % 3) / 16
        for i in range(1, n):
            if i > 1:
                rows[i][i - 1] = -.125
            if i + 1 < n:
                rows[i][i + 1] = -.0625
    else:
        raise ValueError('Unknown family')
    for i, row in enumerate(rows):
        row[i] = sum(abs(x) for x in row.values()) + .5
    return rows


def matrix_bytes(case):
    rows = rows_for(case)
    lines = ['%%MatrixMarket matrix coordinate real general',
             '%% Generator %s; case %s; no reordering or regularization' % (VERSION, case['id']),
             '%d %d %d' % (case['n'], case['n'], sum(map(len, rows)))]
    lines.extend('%d %d %.17g' % (i + 1, j + 1, value)
                 for i, row in enumerate(rows) for j, value in sorted(row.items()))
    return ('\n'.join(lines) + '\n').encode('ascii')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', required=True, help='New directory only; never overwrite')
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {'schema_version': 1, 'generator': VERSION, 'generator_sha256': source_hash,
                'untimed': True, 'watchdog_seconds': 300, 'artificial_memory_cap': None,
                'formula_notes': {
                    'stencil': 'row-major grid, west/east/north/south -1/4,-1/8,-1/2,-1/16',
                    'irregular': 'blocks3,5,8,13 repeated; shifts1..min(width-1,1+i%5), weights -(1+(i+shift)%4)/16; cross-edge(i+width+1)%n weight-1/16',
                    'arrowhead': 'hub columns1+floor(k*(n-1)/degree), weight-(1+k%4)/32; reverse-(1+k%3)/16; chainlower-1/8 upper-1/16',
                    'diagonal': 'every row sum(abs(offdiagonal)) + 1/2; strictly row diagonally dominant',
                    'RHS': 'periodic7: x_i=(i%7-3)/4; b computed/stored by existing binary64 CPU CSR multiplication; x is diagnostic, not asserted exact for rounded b'},
                'cases': []}
    for case in cases():
        raw = matrix_bytes(case)
        name = case['id'] + '.mtx'
        with (destination / name).open('xb') as file:
            file.write(raw)
        provenance = dict(case, source_id='controlled/' + case['id'], source_url=None,
                          license_url=None, acquired_at=None, raw_path=name,
                          raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw),
                          generator=VERSION, generator_sha256=source_hash,
                          generator_method=manifest['formula_notes'][case['family']],
                          diagonal_method=manifest['formula_notes']['diagonal'],
                          rhs_method='periodic7', reference_method='periodic7 dyadic manufactured diagnostic',
                          options={'epsilon': .3, 'alpha': (case['n'] + 7) // 8, 'beta': 8,
                                   'tau': 1e-8, 'innercap': case['n'], 'outercap': 20,
                                   'backward_tolerance': 10 * 2**-53})
        provenance_name = case['id'] + '.provenance.json'
        with (destination / provenance_name).open('x', encoding='utf-8', newline='\n') as file:
            file.write(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
        manifest['cases'].append(dict(provenance, provenance_path=provenance_name))
    with (destination / 'MANIFEST.json').open('x', encoding='utf-8', newline='\n') as file:
        file.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'generator': VERSION, 'cases': len(manifest['cases']),
                      'manifest': str(destination / 'MANIFEST.json')}, separators=(',', ':')))


if __name__ == '__main__':
    main()
