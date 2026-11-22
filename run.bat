ذ@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo Starting LocalRAG from "%CD%" with "%PY%"

"%PY%" -c "import webview, uvicorn, fastapi" >nul 2>&1
if errorlevel 1 (
  echo Dependencies are missing; installing requirements.txt...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)

set "RC=0"
if "%~1"=="" (
  "%PY%" -m localrag desktop
) else (
  "%PY%" -m localrag %*
)
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :failed
exit /b 0

:failed
echo.
echo LocalRAG could not start. Read the error above and check README.md.
REM Exit code 2 means "another instance is already running"; no human action needed.
if not "%RC%"=="2" pause
exit /b %RC%
