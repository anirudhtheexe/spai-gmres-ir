@echo off
rem Compatibility wrapper. Extra arguments select architecture and configuration.
call "%~dp0build_windows.cmd" cuda %*
exit /b %errorlevel%
