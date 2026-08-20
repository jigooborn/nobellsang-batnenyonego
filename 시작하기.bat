@echo off
chcp 65001 >nul
title 항성 분광형 AI 분류 프로그램
echo.
echo  ============================================
echo   항성 분광형 AI 자동 분류 프로그램
echo  ============================================
echo.
where python >/dev/null 2>nul
if errorlevel 1 (
  echo  [!] Python이 설치되어 있지 않습니다.
  echo      https://www.python.org/downloads/ 에서 설치할 때
  echo      "Add Python to PATH" 체크박스를 꼭 선택하세요.
  echo.
  pause
  exit /b
)
echo  [1/2] 필요한 라이브러리 확인 중... (처음 한 번만 몇 분 걸립니다)
pip install -r requirements.txt --quiet
echo  [2/2] 프로그램 실행!
echo.
python classify_gui_v5.py
pause
