@echo off
REM Hybrid RAG Command-Line Launcher
REM This batch file makes it easy to run the hybrid RAG system

echo ========================================
echo Hybrid RAG System Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo.
) else (
    echo Warning: Virtual environment not found at .venv
    echo Using system Python...
    echo.
)

REM Check if database exists
if not exist "data\output\chroma_db_fixed_size_Qwen_Qwen3-Embedding-0.6B_1024" (
    if not exist "data\output\chroma_db_by_sentence_Qwen_Qwen3-Embedding-0.6B_1024" (
        echo ========================================
        echo ERROR: Vector database not found!
        echo ========================================
        echo.
        echo Please create the database first by running:
        echo   python src\chunk_qwen3_0_6B.py
        echo.
        pause
        exit /b 1
    )
)

REM Run the hybrid RAG system
python src\hybrid_rag_cmd_generated.py %*

pause
