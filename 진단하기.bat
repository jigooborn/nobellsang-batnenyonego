@echo off
rem ---- ASCII only. Korean messages are printed by setup_and_run.py ----
cd /d "%~dp0"
title Diagnose

set "PY="
python --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
  py --version >nul 2>nul
  if not errorlevel 1 set "PY=py"
)

if not defined PY (
  echo.
  echo  [!] Python not found / 파이썬이 설치되어 있지 않습니다.
  echo      https://www.python.org/downloads/release/python-3127/
  echo.
  pause
  exit /b 1
)

%PY% "%~dp0setup_and_run.py" --diagnose
echo.
pause
