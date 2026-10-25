@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%"
set "CUBLAS_EMULATE_DOUBLE_PRECISION=0"
set "TASK_CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
"%TASK_CMAKE%\cmake.exe" -S cuda -B build/cleanup-cuda-baseline -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=cl "-DCMAKE_MAKE_PROGRAM=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" -DCMAKE_CUDA_ARCHITECTURES=89 > safeguards/cleanup_20261019/cuda_baseline_configure.log 2>&1
if errorlevel 1 exit /b %errorlevel%
"%TASK_CMAKE%\cmake.exe" --build build/cleanup-cuda-baseline > safeguards/cleanup_20261019/cuda_baseline_build.log 2>&1
if errorlevel 1 exit /b %errorlevel%
"%TASK_CMAKE%\ctest.exe" --test-dir build/cleanup-cuda-baseline --output-on-failure > safeguards/cleanup_20261019/cuda_baseline_tests.log 2>&1
exit /b %errorlevel%
