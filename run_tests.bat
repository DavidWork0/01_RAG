@echo off
REM Run the test suite with suppressed Streamlit warnings
REM This batch file redirects stderr to filter out Streamlit warnings

echo Running RAG Pipeline Tests...
echo.

powershell -Command "$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe tests\test_generated.py 2>&1 | Where-Object { $_ -notmatch 'WARNING streamlit|Warning: to view|Thread.*missing ScriptRunContext|streamlit run' }"

echo.
echo Tests completed.
pause
