@echo off
chcp 949 >nul 2>nul
setlocal
cd /d "%~dp0"
title 항성 분광형 AI 자동 분류 프로그램

echo.
echo  ==================================================
echo    항성 분광형 AI 자동 분류 프로그램
echo    목천고등학교  이한결 . 맹진호
echo  ==================================================
echo.

rem ---------- 1. 파이썬 찾기 ----------
set "PY="
python --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
  py --version >nul 2>nul
  if not errorlevel 1 set "PY=py"
)
if not defined PY (
  echo  [!] 파이썬이 설치되어 있지 않습니다.
  echo.
  echo      https://www.python.org/downloads/
  echo      위 주소에서 내려받아 설치해 주세요.
  echo      설치 화면에서 "Add Python to PATH" 를 꼭 체크해야 합니다.
  echo.
  pause
  exit /b 1
)
for /f "tokens=*" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo  [확인] %PYVER%
echo.

rem ---------- 2. 라이브러리 설치 ----------
%PY% -c "import numpy, scipy, pandas, astropy, matplotlib, torch" >nul 2>nul
if not errorlevel 1 goto RUN

echo  [1/2] 필요한 라이브러리를 설치합니다.
echo        처음 한 번만 5~10분 정도 걸립니다. 창을 닫지 마세요.
echo.
%PY% -m pip install --upgrade pip

rem 가벼운 패키지 먼저 (하나가 실패해도 나머지는 설치되도록 단계를 나눔)
%PY% -m pip install --timeout 120 --retries 5 numpy scipy pandas astropy matplotlib
%PY% -c "import numpy, scipy, pandas, astropy, matplotlib" >nul 2>nul
if errorlevel 1 %PY% -m pip install --user --timeout 120 --retries 5 numpy scipy pandas astropy matplotlib

rem 파이토치는 용량이 커서 따로 (실패하면 CPU 전용 저장소로 재시도)
%PY% -c "import torch" >nul 2>nul
if not errorlevel 1 goto CHECK
echo.
echo  [설치] 파이토치(AI 엔진)를 내려받습니다. 용량이 커서 시간이 걸립니다.
echo.
%PY% -m pip install --timeout 180 --retries 5 torch
%PY% -c "import torch" >nul 2>nul
if not errorlevel 1 goto CHECK
echo.
echo  [재시도] CPU 전용 파이토치로 다시 시도합니다...
echo.
%PY% -m pip install --timeout 180 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu

:CHECK
%PY% -c "import numpy, scipy, pandas, astropy, matplotlib, torch" >nul 2>nul
if not errorlevel 1 goto RUN

echo.
echo  [!] 라이브러리 설치에 실패했습니다. 아래를 확인해 주세요.
echo.
%PY% -c "import numpy" 2>nul || echo      - numpy 설치 안 됨
%PY% -c "import scipy" 2>nul || echo      - scipy 설치 안 됨
%PY% -c "import pandas" 2>nul || echo      - pandas 설치 안 됨
%PY% -c "import astropy" 2>nul || echo      - astropy 설치 안 됨
%PY% -c "import matplotlib" 2>nul || echo      - matplotlib 설치 안 됨
%PY% -c "import torch" 2>nul || echo      - torch 설치 안 됨
echo.
echo      * 인터넷 연결과 방화벽을 확인해 주세요.
echo      * torch 만 실패한다면 파이썬 버전이 너무 최신일 수 있습니다.
echo        파이썬 3.12 버전을 설치하면 대부분 해결됩니다.
echo        https://www.python.org/downloads/release/python-3127/
echo.
echo      직접 설치하려면 명령 프롬프트에서 아래를 실행하세요.
echo        %PY% -m pip install --timeout 120 --retries 5 -r requirements.txt
echo.
pause
exit /b 1

rem ---------- 3. 실행 ----------
:RUN
echo  [2/2] 프로그램을 실행합니다.
echo.
%PY% classify_gui_v5.py
if errorlevel 1 (
  echo.
  echo  [!] 프로그램이 오류로 종료되었습니다. 위 메시지를 확인해 주세요.
)
echo.
pause
