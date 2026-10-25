"""Build a compact, traceable presentation source pack without running experiments."""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation/SPAI_GMRES_IR_Presentation_Sources.zip"

REPORTS = {
    "IEEE_FINAL_BENCHMARK_EVIDENCE.md", "STAGE45B_EXPLORATORY_BENCHMARK.md",
    "STAGE45B_FP64_SCALABILITY.md", "STAGE45B_FP64_12000_RESULTS.md",
    "STAGE45B_RELEASE_ADMISSION.md", "STAGE45A_NUMERICAL_VALIDATION.md",
    "STAGE45A_EXTENSION.md", "EXPERIMENT2A_FP32_VS_FP64_TIMING.md",
    "EXPERIMENT2A_FP32_12000_NUMERICAL_RESULTS.md",
    "EXPERIMENT2A_GPU_SPAI_ADAPTIVE_OVERHEAD.md",
    "EXPERIMENT2A_GPU_SPAI_MEMORY_OPTIMIZATION.md", "EXPERIMENT2A_GPU_SPAI_STAGE_A.md",
    "CODE_OVERVIEW.md", "CUDA_SPAI_DIAGRAM_AUDIT.md",
    "CUDA_SPAI_PRECONDITIONER_ASSEMBLY_AUDIT.md",
}
GROUPS = (
    "report_package/PCAP_IEEE_REPORT_EVIDENCE_20261012/",
    "results/ieee_final_evidence_20261012/",
    "results/stage45b_selected_aggregation_20261011/",
    "results/stage45b_fp64_scalability_aggregation_20261011/",
    "results/stage45b_fp64_12000_clean_aggregation_20261011/",
)
REPRODUCTION_SOURCE = {
    "tests/stage45b_benchmark.cpp", "tests/experiment2a_fp32_benchmark.cpp",
    "tests/check_experiment2a_timing_decimal.py", "scripts/run_ieee_final_scalability.py",
    "scripts/generate_stage45b_scalability.py", "scripts/benchmark_experiment_lock.py",
    "scripts/aggregate_ieee_final_evidence.py",
}
OLD_FIGURES = {
    "report_package/PCAP_IEEE_REPORT_EVIDENCE_20261012/figures/architecture.svg",
    "report_package/PCAP_IEEE_REPORT_EVIDENCE_20261012/figures/adaptive_fit_workflow.svg",
}
CONTRACTS = (
    "PHASE2A_ALGORITHM_SPEC.md", "EXPERIMENT2A_FP32_SPEC.md",
    "PHASE2A_VALIDATION_PLAN.md", "PHASE2B_VALIDATION_RECORD.md",
    "CPU_MILESTONE1.md", "CPU_MILESTONE2.md", "CPU_MILESTONE3.md", "CPU_MILESTONE4.md",
    "CUDA_MILESTONE5A.md", "CUDA_STAGE1_GMRES.md", "CUDA_STAGE2_HYBRID_IR.md",
    "CUDA_STAGE3_GPU_SPAI.md", "CUDA_STAGE4_SCALABLE_SPAI.md",
    "STAGE4_5A_INDEPENDENT_VALIDATION.md",
)

GUIDE = """# Presentation source pack: SPAI-GMRES-IR

This is source material for preparing a presentation, not a finished deck. It
contains implementation, methods, results, editable figures, references and
supporting validation. Large profiles, Git history and solver snapshots are
excluded. No benchmark or solver was rerun to create this pack.

## Start here

1. Read `implementation/README.md` for the implemented method, build/run
   instructions, current correctness results, and limitations.
2. Read `context/CODEBASE_GUIDE.md` and
   `evidence/CODE_OVERVIEW.md` for the CPU/GPU operation split.
3. Use `evidence/IEEE_FINAL_BENCHMARK_EVIDENCE.md` for the main performance story.
4. Use `evidence/results/ieee_final_evidence_20261012/` for final CSVs and SVGs.
5. Consult `evidence/STAGE45B_EXPLORATORY_BENCHMARK.md` for historical application
   matrices and `evidence/results/ieee_final_evidence_20261012/inventory/` for
   which historical runs were accepted, excluded or incomplete.
6. Use `evidence/EXPERIMENT2A_GPU_SPAI_ADAPTIVE_OVERHEAD.md` and
   `evidence/EXPERIMENT2A_GPU_SPAI_STAGE_A.md` for optimization evidence.

The final controlled campaign is the preferred series for the main scaling
graphs. Do not splice its points together with earlier runs as if they were
one unchanged experiment. Older numbered report-package chapters and graphs
are background sources; later final-evidence reports take precedence.

## Material for the presentation story

| Topic | Sources |
| --- | --- |
| Title, objective, problem and scope | implementation/README.md; contracts/PHASE2A_ALGORITHM_SPEC.md |
| SPAI, GMRES and iterative refinement | contracts/; numbered methodology/algorithm chapters in evidence/report_package/ |
| CPU implementation | implementation/src/{cpu,spai,gmres,ir,fp32_cpu}.cpp; CPU milestone reports |
| CUDA implementation and transfers | evidence/CODE_OVERVIEW.md; CUDA stage reports; implementation/src/ |
| Actual CPU/GPU boundary | figures/architecture_verified.mmd; evidence/CUDA_SPAI_DIAGRAM_AUDIT.md |
| Correctness and precision | implementation/README.md; contracts/; current_validation/; saved Decimal80 sidecars |
| Experimental setup | final reproducibility manifest, command matrix, source harnesses, matrix specifications |
| Historical CPU/GPU benchmarks | historical report, selected aggregation JSON/figures, historical inventory |
| Final FP32/FP64 scaling | final raw-sample CSV, medians, speedup CSV, final SVGs |
| Synchronization and transfer optimizations | optimization reports, profiler CSV summaries, campaign summaries |
| Interpretation, limitations and future work | final evidence report, contracts, missing-information notes |
| References | references/REFERENCES.md; bibliography in evidence/report_package/ |

## Primary graph/table locations

Under `evidence/results/ieee_final_evidence_20261012/`:

- `raw/fresh_all_samples.csv`: all exported timing samples, including warm-ups.
- `summaries/fresh_median_results.csv`: all 120 configurations with measured
  trials, medians, statuses and diagnostics.
- `summaries/fresh_cpu_gpu_speedups.csv`: CPU/GPU ratios.
- `validation/decimal80_snapshot_validation.csv`: numerical acceptance summary.
- `figures/fp64_cpu_gpu_time.svg` and `figures/fp64_speedup.svg`.
- `figures/fp32_fp64_pipeline.svg`.
- `figures/cuda_api_counts.svg` and `figures/stage_a_walltime.svg`.

Historical selected JSON and SVGs are under
`evidence/results/stage45b_selected_aggregation_20261011/`.
Profiler CSVs retain their original campaign directories under `evidence/results/`.
SVGs are editable vector graphics; the CSVs can be used to redraw charts.
For Stage A, only the `stage_a_profile_*_retry_20261012` directories are valid
after-change profiles: the earlier non-retry traces used pre-edit executables
and are excluded here. Profile manifests accompany included CSVs. For batching,
use the before/after classification in its report rather than directory age.

## Interpretation rules

- Implemented: uniform FP32 and FP64 SPAI-GMRES-IR on CPU and CUDA.
  BSPAI bucketing and mixed precision are future work, not completed features.
- GPU SPAI performs scaling, QR fits, residual/scoring and packed numerical P
  assembly on the device. CPU handles adaptive metadata/decisions, packs N data,
  canonicalizes downloaded P, and uploads canonical P into GMRES.
- GPU GMRES has device vector arithmetic and sparse products with host scalar
  control. FP64 Givens rotations and projected back-substitution run on CPU.
  The outer IR residual, backward error, update and stopping decision are CPU staged.
- Two older workflow SVGs were deliberately omitted: the architecture labels
  were misleading and the adaptive-fit graphic predates Stage A transfer packing.
  Use the verified Mermaid dataflow and the diagram audit when drawing slides.
- Ratios are CPU time / GPU time: greater than one means the GPU is faster.
  Keep SPAI construction, same-P GMRES, and native pipeline scopes separate.
- Final campaign: two matrix families, five sizes (1024 through 12000), both
  precisions, six modes, three warm-ups and three retained trials/configuration.
  All 120 configurations and 360 measured assessments were accepted.
- Final FP64 SPAI is faster on stencil at 8192 and both families at 12000;
  this is family-dependent. Fixed-P GPU GMRES remains slower at tested sizes.
  FP32 GPU does not cross CPU through 12000. No universal GPU speedup claim.
- Batching reduced synchronization call counts by 65.17%. Stage A reduced async
  copy calls by 11.688%. These are API-count reductions; wall times were mixed.
- Native CPU/GPU pipelines may construct different P or take different numbers
  of iterations. Their timings are not guaranteed equal-work comparisons.
- Historical MSVC CPU Release tests had a wider-long-double oracle limitation;
  keep historical timings exploratory. Later correctness checks do not
  retroactively turn old timings into newly controlled experiments.
- Cite the input family, precision, size, scope and campaign beside any number.
  Do not use resource preflight aborts as zero timings or failed solver samples.

## Known gaps and conflicting historical descriptions

There is no supplied presentation rubric, duration, slide limit, team/title
information or required design template. Those must come from the presenter.
Personal motivations and undocumented decisions should not be invented.

The references contain links and metadata, not full-text paper PDFs or local
MATLAB reference repositories. Check source papers before adding new literature
claims. No genuine terminal/Nsight screenshots are included. If screenshots are
required, capture real views of retained evidence rather than fabricate them.

An older benchmark report names an AMD Ryzen 7 7745HX and a Windows build; a
later evidence report says CPU/RAM/OS details were not reliably recorded. Keep
that discrepancy explicit and verify campaign-specific manifests before using
those host details as a common setup. The documented GPU is RTX 4060 Laptop;
there is no RTX 4080 experiment in this pack.

Historical reports contain original paths/line numbers and sometimes obsolete
authorization/status text. Treat those as dated provenance, not instructions.
Names of local files in old reports can be located using PACKAGE_MANIFEST.json.
Evidence paths are preserved below `evidence/`; current source is below
`implementation/`. Source line numbers changed during formatting; search by
function name. Links to omitted raw snapshots/large matrices point to the full
research archive and are intentionally not a complete reproduction environment.

## Package integrity

PACKAGE_MANIFEST.json records each file's SHA-256, size and original source.
Archived material was checked against the archive's per-file SHA-256 manifest.
Implementation files are the exact tested submission payload, without changes.
The package includes small benchmark source/validation scripts for explanation;
it does not contain all datasets or binaries needed to rerun historical campaigns.
No PowerPoint file is included.
"""

ARCHITECTURE = """flowchart TB
  A["Input: host CSR A and b"] --> S["GPU: row-max scaling and sparse B = transpose(A) D"]
  S --> H["CPU: patterns, shadows and candidate metadata"]
  H --> Q["GPU streams: gather reduced systems; QR; triangular solve"]
  Q --> R["GPU: fitted residuals and candidate scores"]
  R --> C["CPU: acceptance, score ordering and adaptive growth"]
  C -->|grow| H
  C -->|all columns accepted| N["CPU: pack N metadata and fitted coefficients"]
  N --> P["GPU: packed numerical P = transpose(N) D assembly"]
  P --> K["CPU: download P values and canonicalize CSR"]
  K --> U["Upload canonical A and P to persistent GPU GMRES context"]
  U --> Z["CPU: initial guess x0 = P b"]
  Z --> I["CPU outer IR: original residual and backward error"]
  I -->|correction required; upload r and zero initial guess| G["GPU GMRES: P(A v), vector updates, dot products and norms"]
  G <--> T["CPU: small projected-system and convergence control"]
  G --> D["Download correction and diagnostic histories"]
  D --> X["CPU: accepted update; recompute original backward error"]
  X -->|next correction| I
  I -->|target satisfied| F["Return solution and diagnostics"]
  X -->|target satisfied| F
"""


def main():
    payload = {}
    origins = {}

    def add(name, data, origin):
        if name in payload:
            raise ValueError("Duplicate package path: " + name)
        payload[name] = data
        origins[name] = origin

    submission = ROOT / "submission/SPAI_GMRES_IR_Submission.zip"
    original_submission_hash = hashlib.sha256(submission.read_bytes()).hexdigest()
    with zipfile.ZipFile(submission) as bundle:
        records = json.loads(bundle.read("SPAI_GMRES_IR/SUBMISSION_MANIFEST.json"))
        for record in records:
            data = bundle.read("SPAI_GMRES_IR/" + record["path"])
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
            add("implementation/" + record["path"], data,
                "submission/SPAI_GMRES_IR_Submission.zip::" + record["path"])

    with zipfile.ZipFile(ROOT / "archives/research_20261019.zip") as archive:
        archive_manifest = {r["path"]: r for r in json.loads(archive.read("_ARCHIVE_MANIFEST.json"))}
        for item in archive.infolist():
            name = item.filename
            selected = name in REPORTS or name.startswith(GROUPS) or name in REPRODUCTION_SOURCE
            if name.startswith("results/experiment2a"):
                valid_profile = "profile" in name and ("stage_a_profile" not in name or "_retry_" in name)
                selected |= name.endswith("/SUMMARY.json") or (valid_profile and
                            (name.endswith(".csv") or name.endswith("/MANIFEST.json")))
            if name.startswith(("results/ieee_final_scalability_fp32_stage_a_20261012/",
                                "results/ieee_final_scalability_fp64_stage_a_20261012/")):
                selected |= name.endswith("_decimal80.jsonl") or name.endswith("/RUN_MANIFEST.json")
            if not selected or name in OLD_FIGURES:
                continue
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == archive_manifest[name]["sha256"], name
            add("evidence/" + name, data, "archives/research_20261019.zip::" + name)

    for name in CONTRACTS:
        path = ROOT / "docs" / name
        add("contracts/" + name, path.read_bytes(), "docs/" + name)
    add("context/CODEBASE_GUIDE.md", (ROOT / "CODEBASE_GUIDE.md").read_bytes(), "CODEBASE_GUIDE.md")
    add("references/REFERENCES.md", (ROOT / "PAPER_INDEX.md").read_bytes(), "PAPER_INDEX.md")
    logs = ("package_cpu_ctest_amended.log", "package_cuda_amended_refresh.log",
            "package_cpu_decimal_amended.log", "package_msvc_decimal_host.log", "package_all_cli_amended.log")
    for name in logs:
        path = ROOT / "safeguards/cleanup_20261019" / name
        add("current_validation/" + name, path.read_bytes(), path.relative_to(ROOT).as_posix())
    add("README.md", GUIDE.encode(), "package reading guide; existing project evidence only")
    add("figures/architecture_verified.mmd", ARCHITECTURE.encode(),
        "derived from CODEBASE_GUIDE.md and CUDA_SPAI_DIAGRAM_AUDIT.md")

    # Independently check the small exported timing table, without running solvers.
    key = "evidence/results/ieee_final_evidence_20261012/summaries/fresh_median_results.csv"
    rows = list(csv.DictReader(io.StringIO(payload[key].decode())))
    assert len(rows) == 120
    for row in rows:
        trials = [float(value) for value in row["trial_ms"].split(";")]
        assert len(trials) == 3 and row["accepted"] == "True"
        assert math.isclose(statistics.median(trials), float(row["median_ms"]), rel_tol=1e-8, abs_tol=1e-6)
    verdict_key = "evidence/results/ieee_final_evidence_20261012/validation/decimal80_snapshot_validation.csv"
    verdicts = list(csv.DictReader(io.StringIO(payload[verdict_key].decode())))
    assert len(verdicts) == 120
    assert all(r["accepted"] == "True" and int(r["failure_count"]) == 0 for r in verdicts)
    assert sum(int(r["checked_measured_trials"]) for r in verdicts) == 360
    assert sum(name.endswith("_decimal80.jsonl") for name in payload) == 120
    assert not any(name.endswith((".sqlite", ".nsys-rep", ".exe", ".pptx")) for name in payload)

    manifest = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                 "origin": origins[name]} for name, data in sorted(payload.items())]
    OUT.parent.mkdir(exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(suffix=".zip", dir=OUT.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, data in sorted(payload.items()):
                bundle.writestr("SPAI_GMRES_IR_Presentation_Sources/" + name, data)
            bundle.writestr("SPAI_GMRES_IR_Presentation_Sources/PACKAGE_MANIFEST.json",
                            json.dumps(manifest, indent=2) + "\n")
        with zipfile.ZipFile(temporary) as bundle:
            assert bundle.testzip() is None
            assert len(bundle.namelist()) == len(set(bundle.namelist())) == len(manifest) + 1
            for item in manifest:
                data = bundle.read("SPAI_GMRES_IR_Presentation_Sources/" + item["path"])
                assert hashlib.sha256(data).hexdigest() == item["sha256"]
        temporary.replace(OUT)
    finally:
        temporary.unlink(missing_ok=True)
    assert hashlib.sha256(submission.read_bytes()).hexdigest() == original_submission_hash
    report = {"archive": OUT.name, "bytes": OUT.stat().st_size,
              "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "source_files": len(manifest),
              "uncompressed_bytes": sum(item["bytes"] for item in manifest),
              "final_configurations_checked": len(rows), "accepted_measured_assessments": 360,
              "saved_decimal80_sidecars": 120, "submission_unchanged": True,
              "checks": "archive source hashes, ZIP CRC, manifest, unique paths, CSV medians and acceptance"}
    (OUT.parent / "package_integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
