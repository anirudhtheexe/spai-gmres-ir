import hashlib
import json
import statistics
import struct
from pathlib import Path

root = Path(__file__).parent
for mode in ("", "pipeline_"):
    all_solutions = set()
    all_diagnostics = set()
    for variant in ("baseline_1", "context_1", "context_2", "baseline_2"):
        path = root / f"{mode}{variant}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-16").splitlines()]
        elapsed = [row["elapsed_ms"] for row in rows if row["phase"] == "measured"]
        for row in rows:
            bits = b"".join(struct.pack("<f", value) for value in row["solution"])
            all_solutions.add(hashlib.sha256(bits).hexdigest())
            all_diagnostics.add((row["status"], row["outer_steps"],
                                 row["inner_attempts"], row["inner_iterations_total"],
                                 row["original_residual_2"], row["original_residual_inf"],
                                 row["normwise_backward_error"]))
        print(mode or "fixed_ir", variant, "median_ms", statistics.median(elapsed),
              "min_ms", min(elapsed), "max_ms", max(elapsed),
              "raw_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    print(mode or "fixed_ir", "distinct_solution_hashes", all_solutions,
          "distinct_diagnostics", len(all_diagnostics), all_diagnostics)
