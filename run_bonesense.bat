@echo off
echo ======================================================================
echo           BONESENSE - AI FRACTURE DETECTION & ANATOMY SYSTEM
echo ======================================================================
echo Starting server on http://127.0.0.1:5000 ...
cd /d "%~dp0"
start http://127.0.0.1:5000
python src/app.py
pause
