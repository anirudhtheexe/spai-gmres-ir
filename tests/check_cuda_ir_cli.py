"""Independent runnable Matrix Market hybrid CLI contract checks."""
import argparse
from decimal import Decimal as D, getcontext
import math
import re
import subprocess
import sys
from cuda_test_env import cuda_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("matrix")
    args = parser.parse_args()
    env = cuda_environment()
    checks = failures = 0

    def check(ok, name):
        nonlocal checks, failures
        checks += 1
        if not ok:
            failures += 1
            print("FAIL:", name, file=sys.stderr)

    def run(arguments, code):
        r = subprocess.run([args.executable] + arguments, capture_output=True, text=True, env=env)
        check(r.returncode == code, f"exit={code} for {arguments}: actual={r.returncode}")
        return r.stdout

    output = run([args.matrix, ".5"], 0)
    print(output, end="")
    check("status=success converged=1" in output, "CLI solver success")
    check("cpu_spai=true cpu_outer=true gpu_gmres=true rhs=equal_unit2" in output, "CLI explicit hybrid/precision/RHS policy")
    check("epsilon=0.5 beta=8 alpha=1" in output, "CLI explicit epsilon with fixed beta/alpha policy")
    check("arnoldi_uploads=0 arnoldi_downloads=0" in output, "CLI inner residency diagnostics")
    values = [float(v) for _, v in re.findall(r"x\[(\d+)\]=([^\s]+)", output)]
    check(len(values) == 3, "CLI complete solution")
    getcontext().prec = 80
    if len(values) == 3:
        b = D.from_float(1 / math.sqrt(3))
        a = [[D(4), D(1), D(0)], [D(2), D(5), D(1)], [D(0), D(1), D(3)]]
        x = list(map(D.from_float, values))
        truth = [12 * b / 50, 2 * b / 50, 16 * b / 50]
        check(max(abs(v - t) for v, t in zip(x, truth)) <= D("1e-12"), "CLI exact rational inverse agreement")
        r = [b - sum((v * xi for v, xi in zip(row, x)), D(0)) for row in a]
        eta = max(map(abs, r)) / (D(8) * max(map(abs, x)) + b)
        check(eta <= D(10) * D(2) ** -53 + D(64) * D(2) ** -53, "CLI independent original backward error rounding bound")
    run([], 2)
    run([args.matrix + ".missing"], 2)
    for epsilon in ["0", "nan", "inf", "0.3junk"]:
        run([args.matrix, epsilon], 2)
    run([args.matrix, ".3", "21"], 2)
    run([args.matrix, ".3", "20", "4"], 2)
    run([args.matrix, ".3", "-1"], 2)
    cap = run([args.matrix, ".5", "0"], 1)
    check("status=iteration_limit" in cap and "inner_attempts=0" in cap, "CLI cap0 terminal reason and no inner attempt")
    construction = run([args.matrix, "1e-12"], 1)
    check("status=construction_failure" in construction, "CLI tight epsilon alpha1 construction failure")
    default = run([args.matrix], 1)
    check("status=construction_failure" in default, "CLI default epsilon.3 alpha1 fails transparently on fixture")
    print(f"Independent hybrid CLI validation: {checks} checks, {failures} failures")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
