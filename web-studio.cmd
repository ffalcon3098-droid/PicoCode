@echo off
setlocal
set "PYCMD=py"
where py >nul 2>&1 || set "PYCMD=python"
%PYCMD% "%~dp0scripts\web_ui.py" %*
endlocal
