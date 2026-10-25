"""Exercise all four submission CLIs and independently reconstruct backward error."""
import argparse
from decimal import Decimal as D, getcontext
import math
from pathlib import Path
import re
import struct
import subprocess
from cuda_test_env import cuda_environment

ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executables", nargs="+", type=Path)
    args = parser.parse_args()
    getcontext().prec = 80
    matrix = ROOT / "tests/fixtures/tiny_nonsymmetric.mtx"
    checks = 0
    for executable in args.executables:
        def run(arguments, expected):
            nonlocal checks
            result = subprocess.run([str(executable.resolve()), *map(str, arguments)],
                                    capture_output=True, text=True, env=cuda_environment(), timeout=60)
            assert result.returncode == expected, (executable, arguments, result.returncode,
                                                   result.stdout, result.stderr)
            checks += 1
            return result.stdout

        output = run([matrix, ".5"], 0)
        assert "converged=1" in output
        single = "precision=FP32" in output
        values = [float(value) for _, value in re.findall(r"x\[(\d+)\]=([^\s]+)", output)]
        assert len(values) == 3
        # Round-trip max_digits10 text to the actual stored binary32/binary64.
        x = [D.from_float(f32(value) if single else value) for value in values]
        rhs = f32(1 / f32(math.sqrt(3))) if single else 1 / math.sqrt(3)
        b = D.from_float(rhs)
        rows = [[4, 1, 0], [2, 5, 1], [0, 1, 3]]
        residual = [b - sum((D(a) * v for a, v in zip(row, x)), D(0)) for row in rows]
        error = max(map(abs, residual)) / (D(8) * max(map(abs, x)) + abs(b))
        target = D(10) * D(2) ** (-24 if single else -53)
        assert error <= target, (executable, "independent backward error", error, target)
        checks += 1
        run([], 2)
        run([matrix, "nan"], 2)
        run([matrix, ".5", "21"], 2)
        run([matrix, ".5", "20", "4"], 2)
        run([matrix, ".5", "-1"], 2)
        run([matrix, "0.00001", "0"], 1)
        print("%s: Decimal80 backward error=%s" % (executable.name, error))
    print("CLI checks=%d failures=0" % checks)


if __name__ == "__main__":
    main()
