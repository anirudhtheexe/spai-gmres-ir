# Production source formatting record

Date: 19 October 2026. Scope is exactly the 28 C++/CUDA and public header files
under `src/` and `include/bspai/` present in
`safeguards/cleanup_20261019/source_before.zip`. The separately added
`src/solver_main.cpp` is outside this checkpoint and was not formatted here.

Later cleanup corrections added finite-input/output and size-overflow guards to
three FP32 source files. Their final hashes therefore differ from the
formatting-only hashes below. Those behavior changes and their independent tests
are recorded separately in CLEANUP_RECORD.md; the token-identity claim applies
to the formatting stage before those corrections.

`clang-format` 19.1.7 applied `.clang-format` with four-space indentation,
100-column guidance, no include sorting, no comment reflow, no namespace-comment
insertion, no string-literal splitting, and preserved template angle spacing.
The changes are layout only. Existing expressions, braces, identifiers, comments,
preprocessor directives, include order, and literals retain their original token order.

## Verification

- Before formatting, each current source file matched its checkpoint bytes.
- A C++/CUDA lexical scan compared each checkpoint file with its formatted
  counterpart while ignoring only whitespace outside literals and comments.
  It compared identifiers, preprocessing tokens, numbers, punctuators, literal
  contents, and comment contents in order: **28/28 files and 47,661/47,661
  tokens matched**. The scanner handled escaped and raw string literals, both
  comment forms, identifiers, numeric tokens, and longest-match punctuators.
- `clang-format --style=file --dry-run --Werror` passed on the exact 28-file
  checkpoint list. `git diff --check -- src include/bspai` passed.
- No solver, GPU, benchmark, or numerical test was run for this formatting-only
  change; the parent cleanup validation covers builds and numerical checks.

To repeat the formatting check without including later new source files:

```powershell
$files = @(
    "include/bspai/cpu.hpp",
    "include/bspai/cuda_gmres.hpp",
    "include/bspai/cuda_ir.hpp",
    "include/bspai/cuda_spai.hpp",
    "include/bspai/cuda_spmv.hpp",
    "include/bspai/fp32.hpp",
    "include/bspai/fp32_cuda_gmres.hpp",
    "include/bspai/fp32_cuda_spai.hpp",
    "include/bspai/fp32_cuda_spmv.hpp",
    "include/bspai/gmres.hpp",
    "include/bspai/ir.hpp",
    "include/bspai/spai.hpp",
    "src/cpu.cpp",
    "src/cuda_environment.cpp",
    "src/cuda_gmres.cu",
    "src/cuda_hybrid_main.cpp",
    "src/cuda_ir.cpp",
    "src/cuda_norm_probe.cpp",
    "src/cuda_spai.cu",
    "src/cuda_spmv.cu",
    "src/fp32_cpu.cpp",
    "src/fp32_cuda_gmres.cu",
    "src/fp32_cuda_spai.cu",
    "src/fp32_cuda_spmv.cu",
    "src/gmres.cpp",
    "src/ir.cpp",
    "src/main.cpp",
    "src/spai.cpp",
)
& build/cleanup-tools/clang_format/data/bin/clang-format.exe --style=file --dry-run --Werror $files
```

Aggregate SHA-256 uses checkpoint-list order and, for each file, UTF-8 path, a
NUL byte, an eight-byte big-endian content length, then content bytes.

- Before: `d5826e72f44b2d74d7d64ed961676ff7d3ce57646440be4bd508cc9e9cdd8558`
- After: `673ecc5da728b927e9bdf861582c22d630ba6070877cec8feed5f50cd4df812f`

| File | Before SHA-256 | After SHA-256 |
|---|---|---|
| `include/bspai/cpu.hpp` | `c498dffb61d9807674ce2d88b1d0fb62dada35ed9eb0fccc8a60c30eded5a8e1` | `d734268768049d67ca57781567762e859476b10e6695442234182fa2f9e4de49` |
| `include/bspai/cuda_gmres.hpp` | `8327475f724ca0d1a18b6432c40d2b6b9b86709d115053fd94117d0f157e786c` | `6203158c19bb8f9d7502dcd5f4f4097cad7859c792ac330afd5e355d0dc2b312` |
| `include/bspai/cuda_ir.hpp` | `dde7b6bfb299ebf5b6cf853ecafef82efada82ee30df911d250a8261949bbbb0` | `8eec2c57d46591bd89cc9f1f6b0ed3df86ac679cfd878f25e3742f9a56635dd4` |
| `include/bspai/cuda_spai.hpp` | `09b212274b248d05ea5df68ac8f88ef264a097439d8ca974d62aac865a456c15` | `012258731354878f4047b931e1ff6361582e07dc11dc74c59d2aa7245e45d0bc` |
| `include/bspai/cuda_spmv.hpp` | `5eeb4870cc4f3dc5dc703c56e1758e75618017c330279317e15f04ce12ccdd6b` | `12fe90594cbf6b75cf154cff78b11fa26ca78fa260a014e09cf9462c4e6c57e0` |
| `include/bspai/fp32.hpp` | `cad974c0e4f781926040a81a46250e9fad2a49cb26f2e705b54b0ae11534d008` | `e27eb24b5b94d7dfd284d9e9f97c2c002bdd5f3dcd30b976540f6e63653d69d1` |
| `include/bspai/fp32_cuda_gmres.hpp` | `efbc08c10bafd6a5927df42e253eac543d14ee5250f70b207c6eba5125b680c9` | `5e1893d84436a91a5d996052826e9b99a92ab116a7b6292599b31d086bb8e92b` |
| `include/bspai/fp32_cuda_spai.hpp` | `17ae7cc08a33a7bfbc212b7ae8cd4d00238f85034f4768ca4b6bf4b9399f15bb` | `2940297ffb02b89eec692e5066c7307ee3ac7f6e2894683cb91047f0f1de2d46` |
| `include/bspai/fp32_cuda_spmv.hpp` | `e9c62d306db5a5307ce3d4e192ad91bd114664099ccebf4f210d1a7f347b490c` | `e1386605a61dff7bcbfd39559892314a95648810a61700c278e4c75025fcc9c5` |
| `include/bspai/gmres.hpp` | `4ed7598923ac3ddc5a5e15cd5bea2fd64b31ea6eb029c09a7711891c70919e0d` | `789ff1ac2dfac691f19515dd687ab13a5d71a1938a4fd90792d7cdc7cbc1c941` |
| `include/bspai/ir.hpp` | `f5ac2caed34b9839df984df90f5d0bac4fb8e69c4aa85bf4dfa45a0ee811eb5b` | `4b51a58c66fdc2981ad77aa2c2feeaa05dc1bb9c5233e537ff72baea426fd486` |
| `include/bspai/spai.hpp` | `085ef3d365bc2662dbda6f4d039d2749138c9375396242c8fe7ebb9e3825d6b1` | `fbafe9a47c99465f4d58abe3a84d73e558b6ffb1893774b631d8f7f19751c8ab` |
| `src/cpu.cpp` | `071d6614723dff930bbad66f7e7c0ca9a0265b3620ee38fd607367fbf47ef37c` | `88ab693cba5cf8e4036ec34575e98a2aeabf45331e08b048134a06151ee04372` |
| `src/cuda_environment.cpp` | `e53cf79d26edefd072b927fc202caa2314d6e2ae10ff6ac324d1d1a69ea769ba` | `63956ede9cb3324688e95269b24c9ad5c1abfa957729eba05b2640416b869426` |
| `src/cuda_gmres.cu` | `a208711a73c13a58244b3a3d79f88446e000d9566c2aa47e186b3f541f22c0c8` | `e43cf0df4ddced33aa35735816d2a38b1f396087eec773724fe2294a48f490f9` |
| `src/cuda_hybrid_main.cpp` | `afeccf7aa0ef66aa5995aac28928419eb24b482640dfd4a5de426a3048d0db4d` | `c6a16517be3304cce582ecc965d9faa158ab9ea4df172b7e494d46c8493e9258` |
| `src/cuda_ir.cpp` | `8dc6b5492370c21d9995ae32fa43fda585bb138a5c891ef664edd5d4c01b8077` | `9bed4792f58e309e804f2fabfeadf2689749f94eac17b41e1bb8a2d52ea30cb1` |
| `src/cuda_norm_probe.cpp` | `4b804c336d0005f7ddffaaa052e2fa8ca4f9c6078582234b82edfe12f45ea95c` | `653dbc53706935d8376804f323b38319d838a55440895108c63ca3ca6650bd19` |
| `src/cuda_spai.cu` | `a096a2449c04caa7b9e89dfbb8ac1614a2e3953d9d590676cf9322e3e93f05be` | `2f1899e356047d6d5e9ca0efdec23ba6342a501549aa6a195e71a40671db6a93` |
| `src/cuda_spmv.cu` | `efd9ccc5056f4f9c15abca4afea10c4e0cb01443cdaa347aa02e7040f18ef901` | `6fab7fb6a5f26ccb32ab25c70111ff9bd7f7b32405d7263d0ba73e6ae6bc0158` |
| `src/fp32_cpu.cpp` | `60999d91c5ff26fb4b76c732ba611ccdc1c121fd4751d3396a3787e3e06f6768` | `84fff2979ef57aac2f78eb4b7a3deef797833fc8ae4d9a6842f59aaedfc6e02d` |
| `src/fp32_cuda_gmres.cu` | `565ed3fec863ff2fc7bd78a0b89c5b0f225afde1efdb1d401a51c7115620fd3b` | `42bffa329f6938871835f68fddb959fa1e5cb40be5fb9a80efd503b7f8b48659` |
| `src/fp32_cuda_spai.cu` | `55dc153952c06199e8060b00fa05a6a7fa260bcc40ca72b467474bcb2ac8d815` | `11d7451df966ffaa3f18649af1db93d7fb73f84696f13a238ee9de36bd75e604` |
| `src/fp32_cuda_spmv.cu` | `aab8e6e5ff6b18d9b34627cd56ddcc8d37d44057e222b6e7cd4d9c19ce674511` | `76d914580a158da1e19259a04d534b860c61faf31c4b37d1eb343cf32d0fb9f4` |
| `src/gmres.cpp` | `78ae262b6e90a6a901686c66d7ececf6f2fa68dd1e186e759d913cf4f7b7ba5f` | `5b32b7cc14fdb79962178ac581acaba3fc3c128c1efd634ac7e1d779783f3a59` |
| `src/ir.cpp` | `823309f4329fc3a394605b8896b046966b27df804a50d6f3ae616006280f65ff` | `e08c3f07030957be93bcdb2d8aef1635147dcc6607661ed0afce042531a51cc3` |
| `src/main.cpp` | `4094a23c7dd3d7804aefc9638f4dd21abccb4c901090aa02ac51c251622e5552` | `2128a55ffc8759b17c6ce2be0c323cf17ecd65b976f9ceb6236a2d9b5ee2f16e` |
| `src/spai.cpp` | `fbc42843a3c52529217d5e2870d94e74d385afa2c37fae90b2a0517a424bf8eb` | `9671a068becf1dcd8e55d019393d5fbb94953a0217b3bf9058ecd8489446a32d` |
