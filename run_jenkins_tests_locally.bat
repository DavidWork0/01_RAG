@echo off
echo Running Jenkins Test Suite locally...
echo.

REM Set path to Python in virtual environment
set PYTHON=.venv\Scripts\python.exe

REM Check if venv exists
if not exist "%PYTHON%" (
    echo Virtual environment not found at .venv\Scripts\python.exe
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

echo 1. Running test_full.py...
"%PYTHON%" tests\test_full.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ test_full.py FAILED
    goto :end
) else (
    echo ✅ test_full.py PASSED
)

echo.
echo 2. Running test_reranking.py...
"%PYTHON%" tests\test_reranking.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ test_reranking.py FAILED
    goto :end
) else (
    echo ✅ test_reranking.py PASSED
)

echo.
echo 3. Running test_similarity_filter.py...
"%PYTHON%" tests\test_similarity_filter.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ test_similarity_filter.py FAILED
    goto :end
) else (
    echo ✅ test_similarity_filter.py PASSED
)

echo.
echo 4. Running test_inference.py (Quick Mode)...
"%PYTHON%" tests\test_inference.py --model Qwen3-8B-Q5_K_M --mode quick
if %ERRORLEVEL% NEQ 0 (
    echo ❌ test_inference.py FAILED
    goto :end
) else (
    echo ✅ test_inference.py PASSED
)

echo.
echo ==========================================
echo 🎉 All Jenkins tests passed successfully!
echo ==========================================

:end
pause
