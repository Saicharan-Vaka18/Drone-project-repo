@echo off
cd /d "%~dp0backend"
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
