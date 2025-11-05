@echo off
cd /d %~dp0
set PYTHON=..\\.venv\Scripts\python.exe
set STREAMLIT=..\\.venv\Scripts\streamlit.exe

REM Capture start time
set START_TIME=%TIME%

echo ====================================
echo Starting Complete RAG Pipeline
echo ====================================

echo.
echo Step 1: Running data_pipeline_pdf.py
echo ------------------------------------
%PYTHON% data_pipeline_pdf.py
if %errorlevel% neq 0 (
    echo ERROR: data_pipeline_pdf.py failed with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo Step 1 completed successfully.

echo.
echo Step 2: Running pre_chunking.py
echo ------------------------------------
%PYTHON% pre_chunking.py
if %errorlevel% neq 0 (
    echo ERROR: pre_chunking.py failed with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo Step 2 completed successfully.

echo.
echo Step 3: Running chunk_qwen3_0_6B.py
echo ------------------------------------
%PYTHON% chunk_qwen3_0_6B.py
if %errorlevel% neq 0 (
    echo ERROR: chunk_qwen3_0_6B.py failed with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo Step 3 completed successfully.

echo.
echo ====================================
echo Pipeline Complete
echo ====================================

REM Calculate and display total time
set END_TIME=%TIME%
echo.
echo Start Time: %START_TIME%
echo End Time:   %END_TIME%

REM Calculate elapsed time
for /F "tokens=1-4 delims=:.," %%a in ("%START_TIME%") do (
    set /A "start_h=%%a, start_m=%%b, start_s=%%c, start_cs=%%d"
)
for /F "tokens=1-4 delims=:.," %%a in ("%END_TIME%") do (
    set /A "end_h=%%a, end_m=%%b, end_s=%%c, end_cs=%%d"
)

set /A "elapsed_cs=%end_cs%-%start_cs%"
set /A "elapsed_s=%end_s%-%start_s%"
set /A "elapsed_m=%end_m%-%start_m%"
set /A "elapsed_h=%end_h%-%start_h%"

if %elapsed_cs% lss 0 set /A "elapsed_cs+=100, elapsed_s-=1"
if %elapsed_s% lss 0 set /A "elapsed_s+=60, elapsed_m-=1"
if %elapsed_m% lss 0 set /A "elapsed_m+=60, elapsed_h-=1"

echo Total Time: %elapsed_h%h %elapsed_m%m %elapsed_s%s
echo.

pause
