"""Independent 80-decimal-digit residual audit of emitted GPU snapshots.

Runs the validation executable without creating files. Input JSON doubles are
converted with Decimal.from_float, preserving their exact stored binary64 value.
This is a validation-only oracle, not a production fallback or timing benchmark.
"""
import argparse
import decimal
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    args = parser.parse_args()
    decimal.getcontext().prec = 80
    D = decimal.Decimal
    eps = D(2) ** -52

    def value(x):
        return D.from_float(float(x))

    def product(matrix, vector):
        return [sum((value(a) * x for a, x in zip(row, vector)), D(0)) for row in matrix]

    def norm(vector):
        return sum((x * x for x in vector), D(0)).sqrt()

    env = os.environ.copy()
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64" + os.pathsep + env.get("PATH", "")
    env["CUBLAS_EMULATE_DOUBLE_PRECISION"] = "0"
    process = subprocess.run([args.executable], capture_output=True, text=True, env=env)
    checks = failures = snapshots = 0
    maximum_original_difference = maximum_relative_difference = D(0)
    norm_snapshots = 0
    for line in process.stdout.splitlines():
        if line.startswith("NORM_ORACLE "):
            norm_snapshots += 1
            item = json.loads(line[len("NORM_ORACLE "):])
            data = [value(v) for v in item["values"]]
            expected = norm(data)
            n = D(3 * len(data) + 1)
            u = D(2) ** -53
            gamma = n * u / (1 - n * u)
            allowance = 8 * gamma * expected + 8 * value(float.fromhex("0x0.0000000000001p-1022"))
            conditions = [(abs(expected - value(item["norm"])) <= allowance, "80-digit scaled norm bound")]
            if item["exact"]:
                conditions.append((float(expected) == item["norm"], "correctly rounded exact singleton/dyadic norm"))
            for ok, name in conditions:
                checks += 1
                if not ok:
                    failures += 1
                    print(f"FAIL: norm snapshot {norm_snapshots}: {name}", file=sys.stderr)
            continue
        if not line.startswith("ORACLE "):
            print(line)
            continue
        snapshots += 1
        item = json.loads(line[len("ORACLE "):])
        b, x = [value(v) for v in item["b"]], [value(v) for v in item["x"]]
        ax = product(item["A"], x)
        pb, pax = product(item["P"], b), product(item["P"], ax)
        original = norm([u - v for u, v in zip(b, ax)])
        denominator = norm(pb)
        numerator = norm([u - v for u, v in zip(pb, pax)])
        relative = D(0) if numerator == denominator == 0 else numerator / denominator
        original_difference = abs(original - value(item["original"]))
        relative_difference = abs(relative - value(item["relative"]))
        maximum_original_difference = max(maximum_original_difference, original_difference)
        maximum_relative_difference = max(maximum_relative_difference, relative_difference)
        scale = max(value(float.fromhex("0x0.0000000000001p-1022")), norm(b) + norm(ax) + norm(pb) + norm(pax))
        conditions = [
            (original_difference <= 768 * eps * scale, "original residual rounding bound"),
            (relative_difference <= 768 * eps * max(D(1), relative), "preconditioned residual rounding bound"),
        ]
        if item["accepted"]:
            conditions.extend([
                (value(item["relative"]) <= value(item["tau"]), "recorded GPU acceptance threshold"),
                (relative <= value(item["tau"]) + 768 * eps, "80-digit accepted residual bound"),
            ])
        for ok, name in conditions:
            checks += 1
            if not ok:
                failures += 1
                print(f"FAIL: snapshot {snapshots}: {name}", file=sys.stderr)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    if process.returncode:
        failures += 1
        print(f"FAIL: GPU test executable returned {process.returncode}", file=sys.stderr)
    checks += 1
    if not snapshots:
        failures += 1
        print("FAIL: no GPU snapshot records were observed", file=sys.stderr)
    print(f"80-digit Decimal snapshots={snapshots}, checks={checks}, failures={failures}")
    print(f"80-digit norm snapshots={norm_snapshots}")
    print(f"Maximum 80-digit original residual difference={maximum_original_difference}")
    print(f"Maximum 80-digit relative preconditioned residual difference={maximum_relative_difference}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
