@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo Pipeline Sentinel requires uv for the source-checkout launcher.
  echo Install uv, then double-click this file again.
  pause
  exit /b 1
)

echo Preparing Pipeline Sentinel operator console...
echo Development launcher includes the optional Ultralytics YOLO backend.
uv sync --extra operator --extra yolo --group dev
if errorlevel 1 (
  echo.
  echo Dependency setup failed.
  pause
  exit /b 1
)

echo.
echo Starting local operator console at http://127.0.0.1:8765/
echo Close this window or press Ctrl+C to stop the service.
uv run pipeline-sentinel-operator --open-browser
if errorlevel 1 (
  echo.
  echo Operator console exited with an error.
  pause
)
