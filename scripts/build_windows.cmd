@echo off
setlocal
if /i "%~1"=="cpu" goto valid
if /i "%~1"=="cuda" goto valid
echo Usage: %~nx0 cpu^|cuda [CUDA architecture, default native] [Debug^|Release]
exit /b 2
:valid
set "TASK_MODE=%~1"
set "TASK_ARCH=%~2"
if not defined TASK_ARCH set "TASK_ARCH=native"
set "TASK_CONFIG=%~3"
if not defined TASK_CONFIG set "TASK_CONFIG=Debug"
set "TASK_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%TASK_VSWHERE%" (
  echo Visual Studio C++ Build Tools with CMake/Ninja are required.
  exit /b 2
)
for /f "usebackq tokens=*" %%I in (`"%TASK_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "TASK_VS=%%I"
if not defined TASK_VS exit /b 2
call "%TASK_VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "PATH=%TASK_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%TASK_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
pushd "%~dp0.."
if errorlevel 1 exit /b %errorlevel%
set "TASK_BUILD=build\%TASK_MODE%-%TASK_CONFIG%"
if /i "%TASK_MODE%"=="cuda" goto cuda
cmake -S . -B "%TASK_BUILD%" -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=%TASK_CONFIG%
if errorlevel 1 exit /b %errorlevel%
cmake --build "%TASK_BUILD%"
if errorlevel 1 exit /b %errorlevel%
rem MSVC long double is binary64; original wider-oracle suites run with GCC.
ctest --test-dir "%TASK_BUILD%" --output-on-failure -R "^(cpu_core_tests|fp32_cpu_tests)$"
if errorlevel 1 exit /b %errorlevel%
python tests\check_release_cpu_decimal.py "%TASK_BUILD%\cpu_release_snapshot.exe"
exit /b %errorlevel%
:cuda
if not defined CUDA_PATH (
  echo CUDA_PATH must point to the installed CUDA Toolkit.
  exit /b 2
)
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
set "CUBLAS_EMULATE_DOUBLE_PRECISION=0"
cmake -S cuda -B "%TASK_BUILD%" -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=%TASK_CONFIG% -DCMAKE_CUDA_ARCHITECTURES=%TASK_ARCH%
if errorlevel 1 exit /b %errorlevel%
cmake --build "%TASK_BUILD%"
if errorlevel 1 exit /b %errorlevel%
ctest --test-dir "%TASK_BUILD%" --output-on-failure
exit /b %errorlevel%
