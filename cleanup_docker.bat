@echo off
REM ============================================================================
REM Docker Cleanup and VHDX Compaction Script
REM ============================================================================
REM This script must be run as Administrator
REM It will clean up Docker and compact the VHDX file to free up disk space
REM ============================================================================

echo.
echo ============================================================================
echo Docker Cleanup and VHDX Compaction
echo ============================================================================
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo Step 1: Stopping Docker containers...
docker-compose down
if %errorLevel% neq 0 (
    echo Warning: docker-compose down failed or no containers were running
)

echo.
echo Step 2: Stopping all Docker processes...
taskkill /F /IM "Docker Desktop.exe" 2>nul
taskkill /F /IM "com.docker.backend.exe" 2>nul
taskkill /F /IM "com.docker.build.exe" 2>nul

echo Waiting for Docker to fully stop...
timeout /t 5 /nobreak >nul

echo.
echo Step 3: Cleaning up Docker system (images, containers, volumes, cache)...
docker system prune -a --volumes -f
if %errorLevel% neq 0 (
    echo Warning: Docker system prune failed - Docker may already be stopped
)

echo.
echo Step 4: Shutting down WSL...
wsl --shutdown

echo Waiting for WSL to fully shutdown...
timeout /t 10 /nobreak >nul

echo.
echo Step 5: Checking VHDX file size BEFORE compaction...
powershell -Command "Get-ChildItem -Path \"$env:LOCALAPPDATA\Docker\wsl\disk\" -Filter '*.vhdx' | Select-Object Name, @{Name='Size(GB)';Expression={[math]::Round($_.Length/1GB,2)}}"

echo.
echo Step 6: Compacting Docker VHDX file...
echo This may take several minutes depending on file size...
powershell -Command "Optimize-VHD -Path \"$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx\" -Mode Full"

if %errorLevel% neq 0 (
    echo.
    echo ERROR: VHDX compaction failed!
    echo This could be because:
    echo   - Docker or WSL is still running
    echo   - The VHDX file is locked by another process
    echo   - Hyper-V features are not enabled
    echo.
    pause
    exit /b 1
)

echo.
echo Step 7: Checking VHDX file size AFTER compaction...
powershell -Command "Get-ChildItem -Path \"$env:LOCALAPPDATA\Docker\wsl\disk\" -Filter '*.vhdx' | Select-Object Name, @{Name='Size(GB)';Expression={[math]::Round($_.Length/1GB,2)}}"

echo.
echo ============================================================================
echo CLEANUP COMPLETE!
echo ============================================================================
echo.
echo You can now start Docker Desktop manually from the Start menu.
echo After Docker starts, you can rebuild your containers with:
echo   docker-compose build
echo.
pause
