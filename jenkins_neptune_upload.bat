@echo off
REM Jenkins Neptune Upload Script (Windows)
REM =========================================
REM Automatically upload test results to Neptune.ai after Jenkins test execution.
REM This script should be run as a post-build step in your Jenkinsfile.
REM
REM Usage:
REM   jenkins_neptune_upload.bat
REM
REM Environment Variables (set in Jenkins):
REM   NEPTUNE_API_TOKEN: Your Neptune.ai API token (required)
REM   NEPTUNE_PROJECT: Your Neptune.ai project name (required)
REM   NEPTUNE_UPLOAD_MODE: Upload mode - latest|all|inference (default: latest)
REM   NEPTUNE_TAGS: Space-separated tags (optional)
REM   BUILD_NUMBER: Jenkins build number (automatically set by Jenkins)
REM   JOB_NAME: Jenkins job name (automatically set by Jenkins)
REM

echo ==============================================================================
echo Neptune.ai Upload Script
echo ==============================================================================
echo.

REM Check if Neptune credentials are set
if "%NEPTUNE_API_TOKEN%"=="" (
    echo ERROR: NEPTUNE_API_TOKEN environment variable is not set
    echo    Please configure it in Jenkins credentials or environment variables
    exit /b 1
)

if "%NEPTUNE_PROJECT%"=="" (
    echo ERROR: NEPTUNE_PROJECT environment variable is not set
    echo    Example: username/project-name
    exit /b 1
)

echo Neptune credentials configured
echo    Project: %NEPTUNE_PROJECT%
echo.

REM Set default upload mode
if "%NEPTUNE_UPLOAD_MODE%"=="" set NEPTUNE_UPLOAD_MODE=latest
echo Upload mode: %NEPTUNE_UPLOAD_MODE%

REM Prepare tags
set TAGS=jenkins
if not "%JOB_NAME%"=="" set TAGS=%TAGS% jenkins-%JOB_NAME%
if not "%BUILD_NUMBER%"=="" set TAGS=%TAGS% build-%BUILD_NUMBER%
if not "%NEPTUNE_TAGS%"=="" set TAGS=%TAGS% %NEPTUNE_TAGS%

echo Tags: %TAGS%
echo.

REM Change to project root
cd /d %~dp0

REM Check Python environment
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

REM Install neptune if not already installed
echo Checking neptune installation...
%PYTHON% -c "import neptune" 2>nul
if errorlevel 1 (
    echo    Installing neptune...
    %PYTHON% -m pip install neptune --quiet
    echo    Neptune installed
) else (
    echo    Neptune already installed
)
echo.

REM Run the uploader
echo Starting upload to Neptune.ai...
echo.

if "%NEPTUNE_UPLOAD_MODE%"=="latest" (
    %PYTHON% src\neptune_uploader.py --upload-latest --tags %TAGS%
) else if "%NEPTUNE_UPLOAD_MODE%"=="all" (
    %PYTHON% src\neptune_uploader.py --upload-all --limit 10 --tags %TAGS%
) else if "%NEPTUNE_UPLOAD_MODE%"=="inference" (
    %PYTHON% src\neptune_uploader.py --upload-inference-logs --tags %TAGS%
) else (
    echo ERROR: Invalid NEPTUNE_UPLOAD_MODE: %NEPTUNE_UPLOAD_MODE%
    echo    Valid options: latest, all, inference
    exit /b 1
)

echo.
echo ==============================================================================
echo Neptune upload completed successfully!
echo ==============================================================================
