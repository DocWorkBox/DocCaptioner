@echo off
title DocCaptioner v1.0
if not exist "venv" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b
)

call venv\Scripts\activate
python web_app_ng.py
pause