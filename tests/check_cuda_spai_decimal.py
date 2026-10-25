"""Independent 80-digit reduced QR/residual oracle for GPU SPAI snapshots."""
import argparse
from decimal import Decimal as D, getcontext
import json
import subprocess
import sys
from cuda_test_env import cuda_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    args = parser.parse_args()
    getcontext().prec = 80
    value = lambda x: D.from_float(float(x))
    u = D(2) ** -53
    tiny = value(float.fromhex("0x0.0000000000001p-1022"))
    env = cuda_environment()
    run = subprocess.run([args.executable], capture_output=True, text=True, env=env)
    checks = failures = snapshots = 0
    maximum_coefficient_difference = maximum_residual_difference = D(0)
    maximum_ordinary_coefficient_difference = maximum_stress_relative_difference = D(0)
    maximum_ordinary_residual_difference = D(0)
    maximum_accepted_stress_full_residual = D(0)

    def check(ok, name):
        nonlocal checks, failures
        checks += 1
        if not ok:
            failures += 1
            print(f"FAIL: fit {snapshots}: {name}", file=sys.stderr)

    def norm(v):
        return sum((x*x for x in v), D(0)).sqrt()

    for line in run.stdout.splitlines():
        if not line.startswith("SPAI_ORACLE "):
            print(line)
            continue
        snapshots += 1
        item = json.loads(line[len("SPAI_ORACLE "):])
        b = [[value(v) for v in row] for row in item["B"]]
        rows, columns = item["I"], item["J"]
        m, n = len(rows), len(columns)
        q, r = [], [[D(0)]*n for _ in range(n)]
        for j, column in enumerate(columns):
            v = [b[i][column] for i in rows]
            for k in range(j):
                r[k][j] = sum((qi*vi for qi, vi in zip(q[k], v)), D(0))
                v = [vi-r[k][j]*qi for vi, qi in zip(v, q[k])]
            r[j][j] = norm(v)
            if not r[j][j]:
                raise RuntimeError("independent oracle encountered exact rank deficiency")
            q.append([vi/r[j][j] for vi in v])
        rhs = [sum((qi for row, qi in zip(rows, qj) if row == item["target"]), D(0)) for qj in q]
        expected = [D(0)]*n
        for j in reversed(range(n)):
            expected[j] = (rhs[j]-sum((r[j][k]*expected[k] for k in range(j+1,n)), D(0)))/r[j][j]
        coefficients = [value(v) for v in item["coefficients"]]
        check(len(coefficients) == n, "coefficient shape")
        for got, reference in zip(coefficients, expected):
            allowance = (D("2e-6") if item["stress"] else 1536*u) * max(D(1), abs(reference))
            difference = abs(got-reference)
            maximum_coefficient_difference = max(maximum_coefficient_difference, difference)
            if item["stress"]:
                maximum_stress_relative_difference = max(maximum_stress_relative_difference, difference/max(D(1),abs(reference)))
            else:
                maximum_ordinary_coefficient_difference = max(maximum_ordinary_coefficient_difference, difference)
            check(difference <= allowance, "80-digit independent QR coefficient")
        s = [sum((b[i][j]*x for j, x in zip(columns, coefficients)), D(0)) - D(i == item["target"]) for i in rows]
        full = [sum((b[i][j]*x for j, x in zip(columns, coefficients)), D(0)) - D(i == item["target"]) for i in range(len(b))]
        scales = [sum((abs(b[i][j]*x) for j, x in zip(columns, coefficients)), D(0)) + D(i == item["target"]) for i in rows]
        allowance = norm([64*u*scale+tiny for scale in scales]) + 64*u*norm(s)
        residual_difference = abs(value(item["residual"])-norm(s))
        maximum_residual_difference = max(maximum_residual_difference, residual_difference)
        if not item["stress"]:
            maximum_ordinary_residual_difference = max(maximum_ordinary_residual_difference,residual_difference)
        check(residual_difference <= allowance, "80-digit recomputed reduced residual")
        check(abs(value(item["full"])-norm(full)) <= allowance, "80-digit recomputed full residual")
        check(norm(s) == norm(full), "complete-shadow reduced/full equality")
        if item["accepted"]:
            check(value(item["residual"]) <= value(item["epsilon"]), "unchanged construction acceptance threshold")
            check(norm(full) <= value(item["epsilon"])+allowance, "true full residual within declared rounding bound")
            if item["stress"]:
                maximum_accepted_stress_full_residual = max(maximum_accepted_stress_full_residual,norm(full))
                check(norm(full) <= value(item["epsilon"]), "stress independent full residual meets unchanged1e-5 threshold")
        for candidate, score in zip(item["candidates"], item["scores"]):
            col = [b[i][candidate] for i in rows]
            denominator = sum((v*v for v in col), D(0))
            dot = sum((si*ci for si, ci in zip(s, col)), D(0))
            radicand = norm(s)**2-dot*dot/denominator
            check(abs(value(score)**2-max(D(0),radicand)) <= 1536*u*max(D(1),norm(s)**2), "80-digit restricted squared candidate score")
    check(snapshots > 0, "fit snapshots present")
    check(run.returncode == 0, "GPU SPAI test executable passed")
    if run.stderr:
        print(run.stderr, file=sys.stderr, end="")
    print(f"80-digit SPAI fits={snapshots}, checks={checks}, failures={failures}")
    print(f"Maximum 80-digit coefficient difference={maximum_coefficient_difference}")
    print(f"Maximum 80-digit residual difference={maximum_residual_difference}")
    print(f"Maximum ordinary 80-digit coefficient difference={maximum_ordinary_coefficient_difference}")
    print(f"Maximum ordinary 80-digit residual difference={maximum_ordinary_residual_difference}")
    print(f"Maximum stress 80-digit relative coefficient difference={maximum_stress_relative_difference}")
    print(f"Maximum accepted stress 80-digit full residual={maximum_accepted_stress_full_residual}")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
