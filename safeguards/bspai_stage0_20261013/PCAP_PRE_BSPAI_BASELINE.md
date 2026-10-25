# PCAP pre-BSPAI baseline record

**Recorded:** 13 October 2026 (Asia/Kolkata)  
**Status:** preservation attempt blocked; this is not a verified full recovery snapshot.

## Current repository state

- Repository: `C:\Users\Anirudh\Desktop\pcapproj`
- Branch / HEAD: `stage4-scalable-spai` / `8eac10eec719bf24d55e219884d7a35bcd25aaaa`
- The tree is intentionally non-clean: eight tracked files are modified and substantial Stage 4.5 / Experiment 2A source, reports, data, profiling, and result artifacts are untracked. No reset, stash, clean, commit, or numerical source edit occurred.
- No relevant build, solver, benchmark, profiler, Python experiment process, or root experiment lock was active at pre-copy inspection.

## Preservation status

The attempted complete destination, `C:\Users\Anirudh\Desktop\pcapproj_pre_bspai_20261013T002333`, failed with Windows error 112 (insufficient disk space) and robocopy exit 9. It included `.git`, `build`, results, and untracked work until disk exhaustion. The incomplete directory was removed immediately; it is not a backup.

Before Stage 0 documentation, the source inventory captured **3,274 regular files** and **2,503,416,551 bytes**. Its SHA-256 is `8E1098C48C74F3DFFA2EAEA2E760D55C97D761CE708CE13C68DE527516482387`; see `safeguards/bspai_stage0_20261013/PCAP_PRE_BSPAI_FILE_MANIFEST.csv`. It is source-side evidence only: destination comparison did not occur.

An earlier recovery copy exists at `C:\Users\Anirudh\Desktop\pcapproj_checkpoint_20261012T041000` (2,097 regular files; 1,316,551,917 bytes), but its record excludes `build` and it predates this inventory. It is not a substitute for the requested complete current snapshot.

## Required next action and recovery procedure

Provide storage with at least 3 GiB free, create a complete external snapshot, hash and compare every regular source/destination file, and verify `.git`, untracked artifacts, `results/ieee_final_evidence_20261012`, and Stage A evidence in the copy. Then copy this record and the manifest into the verified snapshot. Restore only by copying/extracting that snapshot to a new directory, verifying the manifest plus `git status --short` and `git rev-parse HEAD`; never overwrite the original repository.
