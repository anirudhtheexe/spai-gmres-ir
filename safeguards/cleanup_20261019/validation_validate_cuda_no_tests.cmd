@echo off
setlocal
set "TASK_VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%TASK_VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "PATH=%TASK_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%TASK_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
set "CUBLAS_EMULATE_DOUBLE_PRECISION=0"
cmake -S "%~dp0source\cuda" -B "%~dp0cuda-no-tests" -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=OFF
if errorlevel 1 exit /b %errorlevel%
cmake --build "%~dp0cuda-no-tests"
exit /b %errorlevel%
