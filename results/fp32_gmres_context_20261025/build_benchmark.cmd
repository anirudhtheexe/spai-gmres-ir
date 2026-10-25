@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin\x64;%PATH%"
if not exist build\fp32-context-bench mkdir build\fp32-context-bench
"%CUDA_PATH%\bin\nvcc.exe" -Iinclude -DNDEBUG -arch=sm_89 --fmad=false -Xcompiler=/O2,/MD,/EHsc,/fp:strict,/Z7 -c results\fp32_gmres_context_20261025\fp32_cuda_gmres_before_context.cu -o build\fp32-context-bench\baseline.obj
if errorlevel 1 exit /b %errorlevel%
cl /O2 /MD /Z7 /EHsc /std:c++17 /Iinclude /I"%CUDA_PATH%\include" results\fp32_gmres_context_20261025\context_benchmark.cpp build\fp32-context-bench\baseline.obj build\cuda-NoPdb\solver\bspai_fp32_cuda_spai.lib build\cuda-NoPdb\solver\bspai_fp32_cuda_spmv.lib build\cuda-NoPdb\solver\bspai_fp32_cpu.lib build\cuda-NoPdb\solver\bspai_cpu.lib /Fe:build\fp32-context-bench\baseline.exe /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cublas.lib cusparse.lib cusolver.lib
if errorlevel 1 exit /b %errorlevel%
cl /O2 /MD /Z7 /EHsc /std:c++17 /Iinclude /I"%CUDA_PATH%\include" results\fp32_gmres_context_20261025\context_benchmark.cpp build\cuda-NoPdb\solver\bspai_fp32_cuda_gmres.lib build\cuda-NoPdb\solver\bspai_fp32_cuda_spai.lib build\cuda-NoPdb\solver\bspai_fp32_cuda_spmv.lib build\cuda-NoPdb\solver\bspai_fp32_cpu.lib build\cuda-NoPdb\solver\bspai_cpu.lib /Fe:build\fp32-context-bench\current.exe /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cublas.lib cusparse.lib cusolver.lib
if errorlevel 1 exit /b %errorlevel%
