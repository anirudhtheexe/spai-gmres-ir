$ErrorActionPreference = 'Stop'
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;' + $env:PATH
$base = 'results/fp32_gmres_context_20261025'
$common = @('--mode', 'gpu_ir_fixed', '--matrix', 'build/spai-row-validation/stencil_8192.mtx',
    '--p', 'build/fp32-context-bench/stencil_8192_P.mtx', '--rhs', 'periodic7',
    '--epsilon', '0.3', '--alpha', '1024', '--beta', '8', '--innercap', '8192',
    '--outercap', '20', '--warmups', '2', '--trials', '12')
foreach ($run in @('baseline_1', 'context_1', 'context_2', 'baseline_2')) {
    $which = if ($run.StartsWith('baseline')) { 'baseline' } else { 'current' }
    & "build/fp32-context-bench/$which.exe" @common > "$base/$run.jsonl"
    if ($LASTEXITCODE -ne 0) { throw "$run exited $LASTEXITCODE" }
}
