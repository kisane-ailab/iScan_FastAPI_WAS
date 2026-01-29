@echo off
setlocal enabledelayedexpansion

echo ========================================
echo iScan WAS Manager Deploy Script
echo ========================================
echo.

:: Load environment variables from .env.deploy
set "ENV_FILE=.env.deploy"
if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%a"=="" set "%%a=%%b"
        )
    )
    echo Loaded environment from %ENV_FILE%
) else (
    echo Error: %ENV_FILE% not found!
    pause
    exit /b 1
)
echo.

:: 1. Build
echo [1/4] Building...
call npm run build
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)
echo Build successful!
echo.

:: 2. Create archive
echo [2/4] Creating archive...
tar -czf iscan-was-manager-standalone.tar.gz .next/standalone .next/static public
echo Archive created!
echo.

:: 3. Check ssh2 package
echo [3/4] Checking ssh2 package...
if not exist "node_modules\ssh2" (
    echo Installing ssh2...
    call npm install ssh2 --save-dev
)
echo.

:: 4. Run deploy script
echo [4/4] Running deploy script...
node deploy.js
if errorlevel 1 (
    echo Deployment failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Done! %APP_URL%
echo ========================================
pause
