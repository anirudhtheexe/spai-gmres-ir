"""Independent Decimal80 Release-admission audit of cpu_release_snapshot.

The C++ snapshot emits every binary64 datum as a C99 hex-float string.  This
checker reconstructs those exact stored values before auditing SPAI, Arnoldi /
Givens residual estimates, recomputed GMRES residuals, and IR diagnostics.
It is a validation oracle only; it never supplies values back to the solver.
"""
import argparse
from decimal import Decimal as D, getcontext
import json
import struct
import subprocess
import sys

getcontext().prec = 80
U = D(2) ** -53
TINY = D.from_float(float.fromhex('0x0.0000000000001p-1022'))


def val(text):
    if not isinstance(text, str) or text.count('@') != 1:
        raise ValueError('snapshot number is not a finite hex-float string')
    literal, payload = text.split('@')
    if not literal.lower().replace('-', '').startswith('0x'):
        raise ValueError('snapshot number has malformed hex-float literal')
    x = float.fromhex(literal)
    if not x == x or x in (float('inf'), float('-inf')):
        raise ValueError('snapshot number is nonfinite')
    try:
        expected_bits = int(payload, 16)
    except ValueError as error:
        raise ValueError('snapshot number has malformed payload') from error
    actual_bits = struct.unpack('>Q', struct.pack('>d', x))[0]
    if actual_bits != expected_bits:
        raise ValueError('hex-float does not round-trip to stored binary64 payload')
    # Decimal.from_float receives the reconstructed binary64, not a decimal
    # approximation of the text.  This is the independently checked bridge.
    return D.from_float(x)


def binary64(text):
    """Parse only after the bit-payload validation used by val()."""
    val(text)
    return float.fromhex(text.split('@', 1)[0])


def vector(items):
    return [val(x) for x in items]


def csr(item):
    offsets, indices, values = item['offsets'], item['indices'], vector(item['values'])
    if len(offsets) != item['rows'] + 1 or offsets[0] != 0 or offsets[-1] != len(indices) or len(indices) != len(values):
        raise ValueError('malformed CSR snapshot')
    return [[(indices[k], values[k]) for k in range(offsets[i], offsets[i + 1])] for i in range(item['rows'])]


def mv(rows, x):
    return [sum((v * x[j] for j, v in row), D(0)) for row in rows]


def norm(x):
    return sum((v * v for v in x), D(0)).sqrt()


def inf(x):
    return max(map(abs, x), default=D(0))


def close(got, expected, operations=4096, scale=None):
    scale = max(D(1), abs(expected) if scale is None else scale)
    return abs(got - expected) <= D(operations) * U * scale + TINY


class Audit:
    def __init__(self, name):
        self.name, self.checks, self.failures = name, 0, []

    def check(self, ok, label):
        self.checks += 1
        if not ok:
            self.failures.append(label)


def b_columns(b, n):
    """Convert CSR rows to sparse B columns: maps column -> {row:value}."""
    result = [dict() for _ in range(n)]
    for row_index, row in enumerate(b):
        for column, value in row:
            result[column][row_index] = value
    return result


def mgs_fit(bcols, shadow, pattern, target):
    q, r = [], [[D(0) for _ in pattern] for _ in pattern]
    for j, col in enumerate(pattern):
        w = [bcols[col].get(row, D(0)) for row in shadow]
        for k in range(j):
            r[k][j] = sum((q[k][i] * w[i] for i in range(len(w))), D(0))
            w = [w[i] - r[k][j] * q[k][i] for i in range(len(w))]
        r[j][j] = norm(w)
        if not r[j][j]:
            raise ValueError('Decimal MGS found exact rank deficiency')
        q.append([z / r[j][j] for z in w])
    rhs = [sum((qj[i] for i, row in enumerate(shadow) if row == target), D(0)) for qj in q]
    x = [D(0) for _ in pattern]
    for j in range(len(pattern) - 1, -1, -1):
        x[j] = (rhs[j] - sum((r[j][k] * x[k] for k in range(j + 1, len(pattern))), D(0))) / r[j][j]
    return x


def assess_construction(a, construction, epsilon, beta, audit, near_dependent=False):
    b = csr(construction['B'])
    n = len(a)
    bcols = b_columns(b, n)
    scales = vector(construction['scaling'])
    audit.check(len(scales) == n, 'scaling dimension')
    for k, row in enumerate(a):
        rowmax = max((abs(v) for _, v in row), default=D(0))
        audit.check(rowmax != 0 and close(scales[k], D(1) / rowmax), 'B=A^T D row scaling')
    for i, row in enumerate(b):
        for j, value in row:
            a_value = dict(a[j]).get(i, D(0))
            audit.check(close(value, a_value * scales[j]), 'B orientation/value')
    columns = construction['columns']
    audit.check(len(columns) == n, 'column diagnostic dimension')
    expected_n, expected_p = [dict() for _ in range(n)], [dict() for _ in range(n)]
    for k, column in enumerate(columns):
        history = column['history']
        if not history:
            continue
        audit.check(history[0]['J'] == [k], 'identity initial pattern')
        audit.check(column['pattern'] == history[-1]['J'], 'published solved pattern')
        for h, fit in enumerate(history):
            shadow, pattern = fit['I'], fit['J']
            audit.check(shadow == sorted(set().union(*(set(bcols[col]) for col in pattern))), 'complete sparse shadow')
            coefficients = vector(fit['coefficients'])
            residual = []
            for row in shadow:
                residual.append(sum((bcols[col].get(row, D(0)) * x for col, x in zip(pattern, coefficients)), D(0)) - D(row == k))
            rn = norm(residual)
            allowance = D(8192) * U * max(D(1), rn)
            if near_dependent:
                # This deliberately ill-conditioned original-test fixture validates
                # the no-arbitrary-rank-cutoff policy.  Its acceptance contract is
                # epsilon=1e-5, not agreement of a rounded binary64 residual with
                # a Decimal recomputation to ordinary-condition precision.
                if h + 1 == len(history):
                    audit.check(val(fit['residual']) <= epsilon and val(fit['full_residual']) <= epsilon,
                                'near-dependent final recorded residual meets frozen epsilon')
            else:
                audit.check(abs(rn - val(fit['residual'])) <= allowance, 'Decimal reduced fit residual')
                audit.check(abs(rn - val(fit['full_residual'])) <= allowance, 'complete-shadow full residual')
            try:
                expected = mgs_fit(bcols, shadow, pattern, k)
                for got, want in zip(coefficients, expected):
                    if near_dependent:
                        audit.check(abs(got-want) <= D('0.000002') * max(D(1), abs(want)),
                                    'near-dependent QR coefficient relative bound')
                    else:
                        audit.check(close(got, want, 8192), 'Decimal QR coefficient')
            except ValueError:
                audit.check(column['status'] != 'success', 'rank result cannot be successful')
            candidates = fit['candidates']
            candidate_set = set()
            for row in sorted(set(shadow) | {k}):
                candidate_set.update(col for col, _ in b[row])
            expected_candidates = [col for col in sorted(candidate_set) if col not in pattern]
            # The implementation checks its binary64 residual and growth cap
            # before it enumerates candidates; only a following fit proves
            # this fit entered the candidate-selection branch.
            if h + 1 < len(history):
                audit.check(candidates == expected_candidates, 'restricted sorted candidates')
            else:
                audit.check(not candidates, 'terminal fit has no candidate scoring')
            scores = vector(fit['scores'])
            audit.check(len(scores) == len(candidates), 'candidate score dimension')
            for candidate, score in zip(candidates, scores):
                v = [bcols[candidate].get(row, D(0)) for row in shadow]
                den = sum((z * z for z in v), D(0))
                dot = sum((x * y for x, y in zip(residual, v)), D(0))
                expected_squared = max(D(0), rn * rn - dot * dot / den)
                audit.check(abs(score * score - expected_squared) <= D(8192) * U * max(D(1), rn * rn), 'Decimal restricted score')
            if candidates:
                # The production mean and selection are binary64 decisions.  Reproduce
                # that sequence from the exact serialized binary64 scores.
                binary_scores = [binary64(x) for x in fit['scores']]
                mean = sum(binary_scores) / len(binary_scores)
                expected_selected = [c for score, c in sorted(zip(binary_scores, candidates)) if score <= mean][:beta]
                audit.check(val(fit['mean_score']) == D.from_float(mean), 'binary64 score mean')
                audit.check(fit['selected'] == expected_selected, 'binary64 score selection/order')
            if h + 1 < len(history):
                audit.check(history[h + 1]['J'] == sorted(set(pattern) | set(fit['selected'])), 'adaptive pattern growth')
        for j, coefficient in zip(column['pattern'], vector(column['coefficients'])):
            if coefficient:
                expected_n[j][k] = coefficient
                p = coefficient * scales[j]
                if p:
                    expected_p[k][j] = p
    for name, expected in (('N', expected_n), ('P', expected_p)):
        actual = csr(construction[name])
        actual_dicts = [dict(row) for row in actual]
        audit.check([set(row) for row in actual_dicts] == [set(row) for row in expected], 'assembled ' + name + ' support/orientation')
        for got_row, want_row in zip(actual_dicts, expected):
            for key, want in want_row.items():
                audit.check(key in got_row and close(got_row[key], want, 64, max(D(1), abs(want))), 'assembled ' + name + ' Decimal rounding envelope')
    if near_dependent:
        # Original stress oracle: analytic 2x2 inverse relative bound.  Do not
        # replace it with an ordinary-condition QR-equality assertion.
        aa = [[dict(row).get(j, D(0)) for j in range(2)] for row in a]
        det = aa[0][0]*aa[1][1] - aa[0][1]*aa[1][0]
        audit.check(det != 0, 'near-dependent analytic determinant nonzero')
        expected_inverse = [[aa[1][1]/det, -aa[0][1]/det], [-aa[1][0]/det, aa[0][0]/det]]
        actual = [dict(row) for row in csr(construction['P'])]
        for i in range(2):
            for j in range(2):
                audit.check(abs(actual[i].get(j,D(0))-expected_inverse[i][j]) <= D('0.000002') * max(D(1),abs(expected_inverse[i][j])),
                            'near-dependent analytic inverse relative bound')
    return b


def assess_gmres(a, p, b, gm, tau, audit):
    basis = [vector(x) for x in gm['basis']]
    h = [vector(x) for x in gm['H']]
    history = gm['history']
    audit.check(len(history) == gm['iterations'], 'GMRES history count')
    audit.check(len(h) == len(b) + 1, 'Hessenberg row count')
    pb, beta = mv(p, b), norm(mv(p, b))
    audit.check(close(val(gm['initial_preconditioned']), D(1)), 'zero-guess preconditioned normalization')
    rotations, g = [], [beta] + [D(0) for _ in b]
    for j, entry in enumerate(history):
        audit.check(entry['iteration'] == j + 1, 'GMRES ordinal')
        audit.check(len(basis) >= j + 1, 'Arnoldi basis availability')
        q = basis[j]
        audit.check(close(norm(q), D(1)), 'Arnoldi basis normalization')
        w = mv(p, mv(a, q))
        reconstruction = [D(0) for _ in b]
        for k in range(j + 1):
            for i in range(len(b)): reconstruction[i] += h[k][j] * basis[k][i]
        if j + 1 < len(basis):
            for i in range(len(b)): reconstruction[i] += h[j + 1][j] * basis[j + 1][i]
        audit.check(norm([x-y for x, y in zip(w, reconstruction)]) <= D(16384) * U * max(D(1), norm(w)), 'Decimal Arnoldi relation')
        column = [h[row][j] for row in range(j + 2)]
        for k, (c, s) in enumerate(rotations):
            upper, lower = column[k], column[k + 1]
            column[k], column[k + 1] = c * upper + s * lower, -s * upper + c * lower
        radius = (column[j] * column[j] + column[j + 1] * column[j + 1]).sqrt()
        audit.check(radius != 0, 'Givens radius')
        c, s = column[j] / radius, column[j + 1] / radius
        rotations.append((c, s))
        old = g[j]; g[j], g[j + 1] = c * old, -s * old
        estimate = abs(g[j + 1]) / beta
        audit.check(abs(estimate - val(entry['estimate'])) <= D(16384) * U * max(D(1), estimate), 'Givens projected residual estimate')
        x = vector(entry['solution'])
        ax = mv(a, x); pax = mv(p, ax)
        relative = norm([u-v for u, v in zip(pb, pax)]) / beta
        original = norm([u-v for u, v in zip(b, ax)])
        audit.check(abs(relative - val(entry['recomputed'])) <= D(16384) * U * max(D(1), relative), 'recomputed preconditioned residual')
        audit.check(abs(original - val(entry['original'])) <= D(16384) * U * max(D(1), original), 'recomputed original residual')
        if entry['accepted']:
            audit.check(entry['verified'], 'acceptance requires verification')
            audit.check(val(entry['recomputed']) <= tau, 'recorded GMRES acceptance')
            audit.check(relative <= tau + D(16384) * U, 'Decimal GMRES acceptance')


def assess_ir(a, p, b, ir, audit):
    history = ir['history']
    audit.check(bool(history), 'IR has initial state')
    if not history:
        return
    x0 = vector(history[0]['x']); expected_x0 = mv(p, b)
    audit.check(all(close(got, want, 32, max(D(1), abs(want))) for got, want in zip(x0, expected_x0)), 'IR x0=P*b Decimal envelope')
    for index, state in enumerate(history):
        x = vector(state['x']); d = state['diagnostics']; residual = [u-v for u, v in zip(b, mv(a, x))]
        recorded = vector(d['residual'])
        scale = [abs(bi) + sum((abs(v)*abs(x[j]) for j, v in row), D(0)) for bi, row in zip(b, a)]
        allowance = max((D(8192)*U*max(D(1), s) + TINY for s in scale), default=TINY)
        audit.check(all(abs(u-v) <= allowance for u, v in zip(residual, recorded)), 'IR residual components')
        rinf, r2 = inf(residual), norm(residual)
        anorm, xnorm, bnorm = max((sum((abs(v) for _, v in row), D(0)) for row in a), default=D(0)), inf(x), inf(b)
        nbe = rinf / (anorm*xnorm + bnorm) if anorm*xnorm + bnorm else D(0)
        cbe = max((abs(r)/s if s else D(0) for r, s in zip(residual, scale)), default=D(0))
        for got, expected, label in ((val(d['inf']),rinf,'IR infinity residual'),(val(d['norm2']),r2,'IR 2 residual'),(val(d['nbe']),nbe,'IR normwise BE'),(val(d['cbe']),cbe,'IR componentwise BE')):
            audit.check(abs(got-expected) <= D(16384)*U*max(D(1),abs(expected))+allowance, label)
        if index and index-1 < len(ir['updates']) and ir['updates'][index-1]:
            rhs = vector(ir['rhs'][index-1])
            prior = vector(history[index-1]['diagnostics']['residual'])
            audit.check(all(close(x, y, 4) for x, y in zip(rhs, prior)), 'IR correction RHS uses previous original residual')
            audit.check(index-1 < len(ir['inner_history']), 'IR update has recorded inner solve')
            if index-1 < len(ir['inner_history']):
                inner = ir['inner_history'][index-1]
                audit.check(inner['converged'], 'accepted IR correction inner solve converged')
                correction = vector(inner['solution'])
                expected_x = [u+v for u,v in zip(vector(history[index-1]['x']), correction)]
                audit.check(all(close(x,y,32) for x,y in zip(x,expected_x)), 'IR accepted correction update')
                audit.check(inner['initial_preconditioned'] is not None, 'IR inner records zero-initial solve diagnostics')
                assess_gmres(a, p, rhs, inner, D('1e-8'), audit)
        elif index:
            audit.check(False, 'IR state beyond initial must have an applied correction')
    for i, inner in enumerate(ir.get('inner_history', [])):
        audit.check(i < len(ir['rhs']), 'IR inner has correction RHS')
        audit.check(i < len(ir['updates']), 'IR inner has update decision')
        if i < len(ir['updates']) and not ir['updates'][i]:
            audit.check(not inner['converged'], 'IR rejected correction is nonconverged')
    if ir['converged']:
        audit.check(val(history[-1]['diagnostics']['nbe']) <= D(10)*U, 'IR frozen outer acceptance')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable')
    args = parser.parse_args()
    run = subprocess.run([args.executable], text=True, capture_output=True)
    audits = []
    for line in run.stdout.splitlines():
        if not line.startswith('RELEASE_CPU_SNAPSHOT '):
            print(line)
            continue
        item = json.loads(line[len('RELEASE_CPU_SNAPSHOT '):])
        audit = Audit(item['case']); audits.append(audit)
        if item.get('schema') != 2:
            audit.check(False, 'snapshot schema 2')
            continue
        if 'expected_status' in item:
            audit.check(item['actual_status'] == item['expected_status'], 'explicit boundary/failure status')
            continue
        a, b, epsilon = csr(item['A']), vector(item['b']), val(item['epsilon'])
        if 'construction' in item:
            construction = item['construction']
            audit.check(construction['status'] in ('success', 'iteration_limit'), 'expected construction status')
            assess_construction(a, construction, epsilon, int(item['beta']), audit,
                                item['case'] == 'spai_near_dependent')
            if item['case'] == 'spai_gmres_ir':
                assess_ir(a, csr(construction['P']), b, item['ir'], audit)
            elif 'ir' in item and item['ir'].get('history'):
                assess_ir(a, csr(item.get('fixed_P', construction['P'])), b, item['ir'], audit)
        elif 'ir' in item and item['ir'].get('history'):
            assess_ir(a, csr(item['fixed_P']), b, item['ir'], audit)
        if 'gmres' in item:
            assess_gmres(a, csr(item['fixed_P']), b, item['gmres'], D('1e-8'), audit)
    failures = sum(len(a.failures) for a in audits)
    checks = sum(a.checks for a in audits)
    if run.returncode:
        failures += 1
        print('FAIL: snapshot executable returned %d' % run.returncode, file=sys.stderr)
    if not audits:
        failures += 1
        print('FAIL: no Release snapshots emitted', file=sys.stderr)
    for audit in audits:
        for failure in audit.failures:
            print('FAIL: %s: %s' % (audit.name, failure), file=sys.stderr)
    if run.stderr:
        print(run.stderr, end='', file=sys.stderr)
    print('Release Decimal80 snapshots=%d checks=%d failures=%d' % (len(audits), checks, failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
