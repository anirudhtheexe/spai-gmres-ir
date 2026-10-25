# Recovery checklist -- repository organization checkpoint

## Do not lose

1. Preserve `.git`, the dirty working tree, `src/`, `include/`, `tests/`,
   `scripts/`, both CMake files, `data/`, `docs/`, root reports, `manuscript/`,
   `.codex/agents/`, and `results/`.
2. Keep every experiment directory as a unit with its raw JSONL, Decimal80
   sidecars, manifests, progress logs, generated fixed-P matrices, reports, and
   profile exports. Treat all raw result and validation files as irreplaceable.
3. Keep each immutable matrix together with its manifest and provenance sidecar.
4. Preserve Nsight `.nsys-rep` traces, CSV summaries, and SQLite exports through
   final-document review. SQLite exports may be regenerable later, but are not
   safe to remove at this checkpoint.
5. Preserve the separate backup at
   `C:\Users\Anirudh\Desktop\pcapproj_checkpoint_20261012T041000`.

## Regenerable only after explicit approval

- `build/` is machine-generated CMake/Ninja/CUDA output and can be regenerated
  from the preserved source/configuration. It was excluded from the backup.
- `validation_fp32_cuda_probe.obj` is a root generated object file.
- Python `__pycache__` files are generated, but leave them untouched until a
  later approved cleanup transaction.

## Safe resume procedure

1. Confirm the sibling backup exists and its file count is 2,097.
2. Read `AGENTS.md`, `PROJECT_HANDOFF.md`, `CODEBASE_GUIDE.md`, and
   `docs/EXPERIMENT2A_LIVE_HANDOFF.md`.
3. Read this directory's `manifest.json` and raw inventory files, then run a
   fresh `git status --short --untracked-files=all`; do not reset, clean, or
   overwrite differences.
4. Check `results/STAGE45B_EXPERIMENT_LOCK.json`, any historical lock's owning
   process only if it is current, and active Codex agents before execution.
5. Resume only the explicitly authorized research or implementation activity.
   BSPAI remains unimplemented; the separate literature review is in another
   ChatGPT conversation.

## Restore procedure

If the working tree is damaged, copy required paths from the sibling backup into
a new recovery worktree or a user-approved destination. Do not copy over the
current tree in place. Recover raw results and data before attempting any build.
Use the preserved `.git` metadata in that backup to inspect the captured branch
and uncommitted state.
