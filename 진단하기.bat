@echo off
chcp 949 >nul 2>nul
setlocal
cd /d "%~dp0"
title 설치 문제 진단

echo.
echo  진단 중입니다... 30초쯤 걸립니다.
echo.
call :BODY > 진단결과.txt 2>&1
type 진단결과.txt
echo.
echo  ==================================================
echo   위 내용이 진단결과.txt 파일로도 저장되었습니다.
echo   해결이 안 되면 이 파일을 그대로 보내 주세요.
echo  ==================================================
echo.
pause
exit /b 0

:BODY
echo ============ 설치 진단 ============
echo 시각: %date% %time%
echo 폴더: %cd%
echo.
echo --- [1] 이 폴더가 최신 버전인가 ---
if exist requirements-full.txt (
  echo   최신 버전 폴더가 맞습니다.
) else (
  echo   [!] 옛 버전 폴더입니다. 새로 내려받아야 합니다.
  echo       https://github.com/jigooborn/nobellsang-batnenyonego/archive/refs/heads/main.zip
)
echo.
echo --- [2] 파이썬 ---
where python
where py
python --version
python -c "import sys;print('실행파일:', sys.executable)"
python -c "import sys;print('비트:', sys.maxsize > 2**32 and '64bit' or '32bit')"
echo.
echo --- [3] pip ---
python -m pip --version
echo.
echo --- [4] 이미 설치된 패키지 ---
python -c "import numpy;print('numpy', numpy.__version__)"
python -c "import scipy;print('scipy', scipy.__version__)"
python -c "import pandas;print('pandas', pandas.__version__)"
python -c "import astropy;print('astropy', astropy.__version__)"
python -c "import matplotlib;print('matplotlib', matplotlib.__version__)"
python -c "import torch;print('torch', torch.__version__)"
echo.
echo --- [5] 인터넷 연결 (PyPI) ---
curl -s -o nul -m 20 -w "PyPI 응답코드 %%{http_code}  (200 이면 정상)" https://pypi.org/simple/torch/
echo.
echo.
echo --- [6] C 드라이브 남은 용량 ---
powershell -NoProfile -Command "'{0:N1} GB' -f ((Get-PSDrive C).Free/1GB)"
echo.
echo --- [7] torch 설치 가능 여부 확인 (실제 설치는 안 함) ---
python -m pip install --dry-run --ignore-installed --timeout 60 --retries 2 torch
echo.
echo ============ 진단 끝 ============
exit /b 0
