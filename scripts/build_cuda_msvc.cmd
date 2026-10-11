@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%"
set "CUBLAS_EMULATE_DOUBLE_PRECISION=0"
set "BSPAI_CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
"%BSPAI_CMAKE%\cmake.exe" -S cuda -B build/cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=cl "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe" "-DCMAKE_MAKE_PROGRAM=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" -DCMAKE_CUDA_ARCHITECTURES=89
if errorlevel 1 exit /b %errorlevel%
"%BSPAI_CMAKE%\cmake.exe" --build build/cuda-ninja --verbose
if errorlevel 1 exit /b %errorlevel%
"%BSPAI_CMAKE%\ctest.exe" --test-dir build/cuda-ninja --output-on-failure
exit /b %errorlevel%
