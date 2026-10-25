"""Validation-only GPU-construction Matrix Market CLI probes."""
import argparse
from decimal import Decimal as D, getcontext
import math
import re
import subprocess
import sys
from cuda_test_env import cuda_environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("executable")
    p.add_argument("matrix")
    args = p.parse_args()
    env = cuda_environment()
    checks = failures = 0

    def check(ok, name):
        nonlocal checks, failures
        checks += 1
        if not ok:
            failures += 1
            print("FAIL:", name, file=sys.stderr)

    def run(arguments, code):
        out = subprocess.run([args.executable]+arguments, capture_output=True, text=True, env=env)
        check(out.returncode == code, f"exit={code} for {arguments}, actual={out.returncode}")
        return out.stdout

    output = run([args.matrix, ".5"], 0)
    print(output, end="")
    check("cpu_spai=false cpu_outer=true gpu_gmres=true rhs=equal_unit2 gpu_spai=true" in output, "GPU numeric construction declared, CPU outer staging explicit")
    check("status=success converged=1" in output and "spai_status=success" in output, "CLI GPU SPAI/IR success")
    check("arnoldi_uploads=0 arnoldi_downloads=0" in output, "GPU P inner vector residency")
    check("construction_streams=3 max_fit_batch=3" in output and "max_qr_before_collection=3" in output, "three independent QR submissions before collection")
    values = [float(v) for _, v in re.findall(r"x\[(\d+)\]=([^\s]+)", output)]
    check(len(values) == 3, "GPU CLI complete solution")
    if len(values) == 3:
        getcontext().prec = 80
        b = D.from_float(1/math.sqrt(3))
        truth = [12*b/50,2*b/50,16*b/50]
        check(max(abs(D.from_float(v)-t) for v,t in zip(values,truth)) <= D("1e-12"), "GPU CLI exact rational-inverse reference")
    default = run([args.matrix], 1)
    check("status=construction_failure" in default and "contexts=0" in default, "GPU default epsilon.3/alpha1 failure prevents partialP use")
    capped = run([args.matrix, ".5", "0"], 1)
    check("status=iteration_limit" in capped and "inner_attempts=0" in capped, "GPU CLI cap0 original criterion failure")
    limited = run([args.matrix, ".5", "20", "1"], 1)
    check("status=inner_failure" in limited and "update_applied=0" in limited, "GPU CLI failed inner correction not applied")
    run([args.matrix, "nan"], 2)
    run([args.matrix, ".5", "21"], 2)
    print(f"Independent GPU SPAI CLI validation: {checks} checks, {failures} failures")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
