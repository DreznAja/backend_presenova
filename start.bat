@echo off
echo Presenova Face API - Starting...
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies (first run may take a while)...
pip install -r requirements.txt -q

REM Copy .env if not exists
if not exist ".env" (
    copy .env.example .env
    echo.
    echo [!] Please edit face_api\.env with your Supabase credentials!
    echo     SUPABASE_URL=your_url
    echo     SUPABASE_KEY=your_key
    pause
)

echo.
echo Starting Face Recognition API on http://localhost:8000
echo Press Ctrl+C to stop
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
