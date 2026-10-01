@echo off
cd /d "%~dp0backend"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run RUN_BACKEND.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import requests; print(requests.get('http://127.0.0.1:8000/api/model/status').json())"
pause
