@echo off
rem ---- ASCII only. Korean messages are printed by setup_and_run.py ----
cd /d "%~dp0"
title Stellar Spectral Type Classifier

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
  echo.
  echo      Install Python 3.12 from:
  echo      https://www.python.org/downloads/release/python-3127/
  echo.
  echo      Check "Add Python to PATH" during setup.
  echo.
  pause
  exit /b 1
)

%PY% "%~dp0setup_and_run.py"
echo.
pause
