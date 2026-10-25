# Independent submission package validation

Date: 19 October 2026. Reviewer owned validation; no production source edits or benchmarks. Tested the source ZIP at `submission/SPAI_GMRES_IR_Submission.zip` by extracting into a fresh `build/submission-check/source` directory. The tested original ZIP was 306,391 bytes, SHA-256 `2b89328df2a1e6c8349f795b6d5543b70a98b47a28bc106ae4f9de415ca9c1`. It contained 97 source files and 862,661 uncompressed source bytes. Every member matched its SHA-256 manifest and then-current live source. The independent extractor rejected duplicate members, paths outside the fixed `SPAI_GMRES_IR/` prefix, traversal, symlinks, and `.git`, `archives`, `build`, `safeguards`, `submission`, `results`, and `data` trees before writing any extracted file. See `safeguards/cleanup_20261019/package_manifest_validation.log` and validation-only script `build/submission-check/verify_package.py`.

## Fresh extracted-tree results

The environment was Windows x64, GCC 6.3.0 at `C:/MinGW/bin/g++.exe`, MSVC 19.44.35228, CMake 3.31, Ninja, CUDA 13.4.59, and NVIDIA RTX 4060 Laptop GPU (architecture 89). Both GCC and MSVC/CUDA builds used Debug. The first sandbox GCC configure failed before compilation with `Illegal System DLL Relocation`; host execution with `require_escalated` succeeded. That failure is retained in `package_cpu_configure.log` and is not a solver result.

| Extracted-tree check | Result | Evidence log |
| --- | --- | --- |
| GCC CPU configure/build, then CTest | 25 original build steps; 5/5 suites passed | `package_cpu_configure_host.log`, `package_cpu_build_host.log`, `package_cpu_ctest_host.log` |
| GCC CPU exact-bit snapshot checked by Decimal80 | 16 snapshots, 500 checks, zero failures | `package_cpu_decimal_host.log` |
| Standalone `scripts/build_windows.cmd cuda 89 Debug` | 60 build steps; CUDA CTest 8/8 passed serially | `package_cuda_build_test_host.log` |
| Four CLIs with `tests/check_solver_cli.py` and `tests/fixtures/tiny_nonsymmetric.mtx`, epsilon `0.5` | 32 checks, zero failures; all four Decimal80 backward errors below frozen targets | `package_all_cli_host.log` |
| MSVC CPU snapshot from standalone CUDA build checked by Decimal80 | 16 snapshots, 500 checks, zero failures | `package_msvc_decimal_host.log` |
| Documented `scripts/build_windows.cmd cpu 89 Debug` | selected MSVC CTest 2/2 plus 16 snapshots/500 Decimal80 checks passed | `package_windows_cpu_helper_host.log` |
| GCC CPU `BUILD_TESTING=OFF` | configure/build passed; CTest inventory 0 | `package_cpu_no_tests_configure.log`, `package_cpu_no_tests_build.log`, `package_cpu_no_tests_inventory.log` |
| Standalone CUDA `BUILD_TESTING=OFF` | configure/build passed; CTest inventory 0; no test, Gate 3 reference, or snapshot targets emitted | `package_cuda_no_tests_build.log`, `package_cuda_no_tests_inventory.log`, `package_cuda_no_tests_excluded_targets.log` |

The CLI checker exercised each executable's success and six non-success/invalid-input paths, and independently reconstructed its final normwise backward error at Decimal80 precision. Its first complete run reported CPU FP32 `2.53746e-8`, CPU FP64 `5.40158e-17`, GPU FP32 `1.44998e-8`, and GPU FP64 `2.02559e-17`. The GCC snapshot checker and the MSVC snapshot checker each passed all 500 checks, confirming the repaired exact-bit serializer on both compilers. These are small-fixture correctness results; no timing claim follows.

## Staged build-file amendment and repeat

During package validation, the parent corrected the consolidated root `CMakeLists.txt` so the historical MinGW 32-bit SSE2 flags apply only to `bspai_cpu`, as in the original FP64 build, rather than also to `bspai_fp32_cpu`. No numerical source was changed. I copied only that corrected CMake file into the extracted tree for an explicit staged test: SHA-256 changed from `27ABCEC5E1413B0DA2A8FD2452260686389B1271CB752416D17EACEC4B804DD5` to `BF24C213E8F6DEAEFCCDF14A21F29ABBC36EBD498169BCCA3B605AA91DA75021`, matching the live file. See `package_staged_cmake_amendment.log`.

After that amendment, extracted GCC CPU reconfiguration/rebuild and CTest 5/5 passed; GCC Decimal80 again passed 16 snapshots/500 checks. The CPU CLI check passed 16/16. The complete four-CLI check passed 32/32 with CPU FP32 backward error `4.34993e-8`, CPU FP64 `5.40158e-17`, GPU FP32 `1.44998e-8`, and GPU FP64 `2.02559e-17`. Extracted standalone CUDA reconfigured with **no build work** and again passed 8/8 CTest suites. The amended `BUILD_TESTING=OFF` GCC CPU build passed with zero tests, and a fresh standalone CUDA `BUILD_TESTING=OFF` build passed with zero tests. Logs carry the `_amended` suffix, `package_all_cli_amended.log`, and the `package_cuda_no_tests_*` names.

An independent live/extracted comparison after the amendment found 67 matching source, test, build, script, fixture, and role-definition files; the only live relevant file missing from the originally tested ZIP is newly added `tests/test_submission_tools.py`. There were no differing relevant files. This five-test archive/package safety suite was run independently by the parent and reported 5/5; it is not part of CTest. See `package_live_code_comparison.log`. Documentation was edited during validation and is not included in this 67-file code/build comparison. The parent must generate the final ZIP from the settled tree and compare its relevant member hashes against this staged tested tree; the final ZIP should additionally contain `tests/test_submission_tools.py` and the final documentation.

## Prior review findings

The packager now preflights required files, writes and verifies a temporary ZIP, then replaces the destination; the missing-README/truncated-prior-ZIP risk was fixed and is covered by the new safety suite. It now includes `.codex/agents/*.toml`; README explicitly describes this as a source-only package and explains omitted historical data/evidence. The research archive verifier now checks duplicate member counts and rejects absolute, traversal, and symlink paths. Prune rehashes each loose source immediately before unlinking. The parent recorded full archive verification, source hash checks, a second verification, and a six-category sample restore. The old archive risk is resolved within the documented no-active-writer boundary; this package check did not rerun or alter the large research archive.

The full CUDA build emitted MSVC/CUDA vendor-header deprecation warnings and generated CUDA stub warnings but no build failures. The primary remaining delivery step is final ZIP regeneration and hash comparison after documentation settles; the original tested ZIP hash above must not be presented as the final regenerated ZIP hash.
