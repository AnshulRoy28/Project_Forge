@echo off
REM =============================================================================
REM Forge Docker Auto-Detect Script (Windows)
REM Detects GPU architecture and runs the appropriate container
REM =============================================================================

echo.
echo 🔥 Forge Docker Runner
echo.

REM Check for NVIDIA GPU
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ✗ nvidia-smi not found. Is NVIDIA driver installed?
    exit /b 1
)

REM Get GPU info
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format^=csv^,noheader') do set GPU_NAME=%%i
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=compute_cap --format^=csv^,noheader') do set COMPUTE_CAP=%%i

echo   GPU: %GPU_NAME%
echo   Compute: %COMPUTE_CAP%

REM Parse compute capability
for /f "tokens=1,2 delims=." %%a in ("%COMPUTE_CAP%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

REM Determine architecture
set ARCH=ampere
set PROFILE=ampere

if %MAJOR% GEQ 12 (
    set ARCH=blackwell
    set PROFILE=blackwell
    echo   Architecture: Blackwell ^(RTX 50-series^)
) else if %MAJOR% EQU 9 (
    set ARCH=hopper
    set PROFILE=hopper
    echo   Architecture: Hopper ^(H100/H200^)
) else if %MAJOR% EQU 8 (
    if %MINOR% GEQ 9 (
        set ARCH=ada
        set PROFILE=ada
        echo   Architecture: Ada Lovelace ^(RTX 40-series^)
    ) else (
        set ARCH=ampere
        set PROFILE=ampere
        echo   Architecture: Ampere ^(RTX 30-series^)
    )
) else (
    echo   Architecture: Legacy ^(using Ampere image^)
)

echo.

REM Build if needed
docker image inspect forge:%ARCH% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Building forge:%ARCH%...
    docker compose -f docker\docker-compose.yml --profile %PROFILE% build
)

REM Run
echo Running Forge...
echo.

docker compose -f docker\docker-compose.yml --profile %PROFILE% run --rm forge-%ARCH% %*
