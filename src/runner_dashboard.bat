@echo off
cd /d %~dp0
set STREAMLIT=..\\.venv\Scripts\streamlit.exe

echo ====================================
echo Starting RAG Dashboard
echo ====================================

echo.
echo Press Ctrl+C to stop the application when done.
%STREAMLIT% run streamlit_modern_multiuser.py
