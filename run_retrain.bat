@echo off
echo ======================================================================
echo           BONESENSE - AI FRACTURE MODEL RETRAINING PIPELINE
echo ======================================================================
echo.
echo Select retraining target:
echo [1] Retrain All Models (FracAtlas Unified + Clinical Paired-View) [Recommended]
echo [2] Retrain FracAtlas Unified Model only (Web App Detection + Localization)
echo [3] Retrain Clinical Paired-View Model only (Dual AP/Lat + 5 Subtypes)
echo [4] Quick Smoke Test (2 epochs)
echo.
set /p opt="Enter selection [1-4] (default is 1): "

if "%opt%"=="2" (
    echo Retraining FracAtlas Unified Model...
    python retrain.py --model fracatlas
) else if "%opt%"=="3" (
    echo Retraining Clinical Paired-View Model...
    python retrain.py --model paired
) else if "%opt%"=="4" (
    echo Running Quick Smoke Test...
    python retrain.py --model all --epochs 2
) else (
    echo Retraining All Models...
    python retrain.py --model all
)

echo.
echo Retraining session completed.
pause
