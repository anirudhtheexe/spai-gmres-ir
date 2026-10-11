"""Validation-only 80-digit original-system audit; no solver fallback."""
import argparse
from decimal import Decimal as D, getcontext
import json
import math
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("executable")
    args = p.parse_args()
    getcontext().prec = 80
    env = os.environ.copy()
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64" + os.pathsep + env.get("PATH", "")
    env["CUBLAS_EMULATE_DOUBLE_PRECISION"] = "0"
    run = subprocess.run([args.executable], capture_output=True, text=True, env=env)
    value = lambda x: D.from_float(float(x))
    u = D(2) ** -53
    tiny = value(float.fromhex("0x0.0000000000001p-1022"))
    checks = failures = snapshots = 0
    maximum_nbe_difference = D(0)

    def check(ok, name):
        nonlocal checks, failures
        checks += 1
        if not ok:
            failures += 1
            print("FAIL:", name, file=sys.stderr)

    for line in run.stdout.splitlines():
        if not line.startswith("IR_ORACLE "):
            print(line)
            continue
        snapshots += 1
        j = json.loads(line[len("IR_ORACLE "):])
        a = [[value(v) for v in row] for row in j["A"]]
        b, x = [value(v) for v in j["b"]], [value(v) for v in j["x"]]
        r = [bi - sum((v * xi for v, xi in zip(row, x)), D(0)) for bi, row in zip(b, a)]
        scales = [abs(bi) + sum((abs(v) * abs(xi) for v, xi in zip(row, x)), D(0)) for bi, row in zip(b, a)]
        allowances = [64 * u * s + tiny for s in scales]
        for ri, stored, allowance in zip(r, j["residual"], allowances):
            check(abs(ri - value(stored)) <= allowance, "80-digit residual component rounding bound")
        ri = max(map(abs, r), default=D(0))
        norm = sum((v * v for v in r), D(0)).sqrt()
        norm_allowance = sum((v * v for v in allowances), D(0)).sqrt()
        check(abs(ri - value(j["inf"])) <= max(allowances, default=tiny), "80-digit original infinity norm")
        check(abs(norm - value(j["norm2"])) <= norm_allowance + 64 * u * norm, "80-digit original 2-norm")
        denominator = max(sum(map(abs, row), D(0)) for row in a) * max(map(abs, x)) + max(map(abs, b))
        eta = ri / denominator if denominator else D(0)
        cbe = max((abs(v) / s if s else D(0) for v, s in zip(r, scales)), default=D(0))
        nbe_allowance = max(allowances, default=tiny) / denominator + 64 * u * eta if denominator else tiny
        cbe_allowance = max((v / s if s else D(0) for v, s in zip(allowances, scales)), default=tiny) + 64 * u * cbe
        check(abs(eta - value(j["nbe"])) <= nbe_allowance, "80-digit normwise backward error")
        check(abs(cbe - value(j["cbe"])) <= cbe_allowance, "80-digit componentwise backward error")
        maximum_nbe_difference = max(maximum_nbe_difference, abs(eta - value(j["nbe"])))
        if j["accepted"]:
            check(value(j["nbe"]) <= 10 * u, "unchanged recorded outer acceptance threshold")
            check(eta <= 10 * u + nbe_allowance, "accepted true BE within independently bounded binary64 rounding")
    check(snapshots > 0, "outer snapshots present")
    check(run.returncode == 0, "hybrid executable success")
    if run.stderr:
        print(run.stderr, file=sys.stderr, end="")
    print(f"80-digit IR snapshots={snapshots}, checks={checks}, failures={failures}")
    print(f"Maximum 80-digit normwise BE difference={maximum_nbe_difference}")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
