@echo off
setlocal
set "PYCMD=py"
where py >nul 2>&1 || set "PYCMD=python"
if /I "%~1"=="--web" (
    shift
    %PYCMD% "%~dp0scripts\web_ui.py" %*
) else if /I "%~1"=="agent" (
    shift
    %PYCMD% "%~dp0scripts\agent_cli.py" %*
) else if /I "%~1"=="--agent" (
    shift
    %PYCMD% "%~dp0scripts\agent_cli.py" %*
) else (
    %PYCMD% "%~dp0scripts\serve.py" %*
)
endlocal
