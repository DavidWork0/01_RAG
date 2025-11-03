@echo off
cd /d %~dp0
set PYTHON=..\\.venv\Scripts\python.exe
set STREAMLIT=..\\.venv\Scripts\streamlit.exe

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
echo Step 4: Starting Streamlit application
echo ------------------------------------
echo Press Ctrl+C to stop the application when done.
%STREAMLIT% run streamlit_modern_multiuser.py

echo.
echo ====================================
echo Pipeline Complete
echo ====================================
pause
