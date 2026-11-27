@echo off
setlocal EnableDelayedExpansion
set "PYCMD=py"
where py >nul 2>&1 || set "PYCMD=python"

set "FIRST=%~1"
set "TARGET=%~dp0scripts\serve.py"
set "IS_SUB=0"

if /I "%FIRST%"=="--web" (
    set "TARGET=%~dp0scripts\web_ui.py"
    set "IS_SUB=1"
) else if /I "%FIRST%"=="agent" (
    set "TARGET=%~dp0scripts\agent_cli.py"
    set "IS_SUB=1"
) else if /I "%FIRST%"=="--agent" (
    set "TARGET=%~dp0scripts\agent_cli.py"
    set "IS_SUB=1"
)

if "%IS_SUB%"=="0" (
    %PYCMD% "%TARGET%" %*
    goto :eof
)

shift
set "REST="
:collect_args
if "%~1"=="" goto :run_sub
set "REST=!REST! %1"
shift
goto :collect_args

:run_sub
%PYCMD% "%TARGET%"!REST!
endlocal
