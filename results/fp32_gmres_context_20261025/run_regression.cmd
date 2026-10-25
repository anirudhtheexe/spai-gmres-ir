@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
set "TASK_VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "PATH=%TASK_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%TASK_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%"
set "CUBLAS_EMULATE_DOUBLE_PRECISION=0"
cmake --build build\cuda-NoPdb -j 2
if errorlevel 1 exit /b %errorlevel%
ctest --test-dir build\cuda-NoPdb --output-on-failure
if errorlevel 1 exit /b %errorlevel%
