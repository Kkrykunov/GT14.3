@echo off
REM GT14 WhaleTracker v14.3 - Launch Script for Windows
REM This script sets up the environment and runs the WhaleTracker application

REM Set UTF-8 encoding for proper Unicode support
chcp 65001 >nul 2>&1

echo ========================================
echo GT14 WhaleTracker v14.3
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and add it to PATH.
    pause
    exit /b 1
)

REM Display Python version
echo Python version:
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo Installing/updating requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Set Python UTF-8 mode
set PYTHONUTF8=1

REM Run the main application
echo.
echo Starting GT14 WhaleTracker...
echo ========================================
python GT14_v14_3_FINAL.py %*

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

pause