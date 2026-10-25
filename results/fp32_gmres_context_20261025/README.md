# FP32 GMRES context experiment, 25 October 2026

`fp32_cuda_gmres_before_context.cu` is the immediate prechange FP32 source,
copied before implementation (SHA-256 `19020F72E354237B201382E20C12D366B57ACC23D3EE79069FE1207297BDB2D9`).
`context_benchmark.cpp` is the existing Experiment 2A harness extended with
`gpu_ir_fixed`, which times only outer refinement with fixed A and P.
`prepare_benchmark.py` records the harness extension. `build_benchmark.cmd`
builds the baseline and context executables with matching flags and libraries.
`run_benchmark.ps1` and `run_pipeline.ps1` run serial ABBA trials. The eight
JSONL files retain raw samples, including full solutions. PowerShell 5 wrote
them as UTF-16; `summarize.py` reads that encoding and reports medians,
solution-bit hashes and final diagnostics. The fixed P was generated once by
the harness's `prepare_fixed_p` mode and kept under
`build/fp32-context-bench/stencil_8192_P.mtx` (a reproducible build artifact).
`run_regression.cmd` rebuilds and runs the full CUDA CTest suite.

The input `build/spai-row-validation/stencil_8192.mtx` is the archived matrix
from `archives/research_20261019.zip` at
`data/stage45b_fp64_scalability_v1/stencil_8192.mtx`; its SHA-256 is
`90d25d6332cc3729030c13e89deb74169b13aef8c1fe89b1ffa4af6709f09b1f`.
The fixed P was prepared with:

```text
build/fp32-context-bench/current.exe --mode prepare_fixed_p --matrix build/spai-row-validation/stencil_8192.mtx --rhs periodic7 --epsilon 0.3 --alpha 1024 --beta 8 --innercap 8192 --outercap 20 --warmups 0 --trials 1 --p-output build/fp32-context-bench/stencil_8192_P.mtx
```

The resulting P SHA-256 was
`5475F5B0EA644479B85089CE5AA1103B7051F5708D5900B14216E8589A011C8E`.

The validated interpretation, numerical checks, environment and limits are
in `docs/FP32_GMRES_CONTEXT_VALIDATION.md`.
