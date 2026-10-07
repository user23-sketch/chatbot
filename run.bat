@echo off
setlocal
cd /d "%~dp0"

set "UV_CACHE_DIR=%~dp0.uv-cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv-python"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

if exist "%~dp0.venv\Scripts\streamlit.exe" goto :run_app

set "UV_EXE="
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if defined UV_EXE goto :install_deps
where uv.exe >nul 2>&1
if errorlevel 1 goto :missing_uv
set "UV_EXE=uv.exe"

:install_deps
echo Installing project dependencies...
call "%UV_EXE%" sync
if errorlevel 1 goto :error

:run_app
if not exist "%~dp0.venv\Scripts\streamlit.exe" goto :error
echo Starting the chatbot. Close this window to stop it.
echo.|call "%~dp0.venv\Scripts\streamlit.exe" run "%~dp0app.py"
if errorlevel 1 goto :error
exit /b 0

:missing_uv
echo uv was not found. Install uv and run this file again.
goto :error

:error
echo The chatbot could not start. Review the error above.
pause
exit /b 1
