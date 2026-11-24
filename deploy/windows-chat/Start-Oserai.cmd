@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
echo Starting the OserAI chat host for your Windows laptop.
echo First launch finds Python 3.12 or 3.13, then installs the chat-service dependencies.
echo The host downloads Cloudflare's tunnel connector if it is not installed.
echo.

set "OSERAI_STEP=Checking the local Python environment"
call :try_python "%~dp0.venv\Scripts\python.exe"
if not errorlevel 1 goto dependencies
call :find_python
if not errorlevel 1 goto create_environment

:install_python
set "OSERAI_STEP=Checking Windows Package Manager (winget)"
echo Checking whether WinGet can actually run...
winget --version >nul 2>nul
if errorlevel 1 goto missing_python
set "OSERAI_STEP=Installing Python through WinGet"
echo Installing the official Python 3.13 package for this Windows user...
winget install --id Python.Python.3.13 --exact --scope user --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto missing_python
call :find_python
if errorlevel 1 goto missing_python

:create_environment
set "OSERAI_STEP=Creating the local Python environment"
echo Using Python: "%OSERAI_PYTHON%"
"%OSERAI_PYTHON%" -m venv .venv
if errorlevel 1 goto failed
call :try_python "%~dp0.venv\Scripts\python.exe"
if errorlevel 1 goto failed

:dependencies
set "OSERAI_STEP=Installing the chat-service dependencies"
echo Using Python: "%OSERAI_PYTHON%"
"%OSERAI_PYTHON%" -m pip install --disable-pip-version-check -r requirements-chat.txt
if errorlevel 1 goto failed
set "OSERAI_STEP=Starting the chat gateway and tunnel"
"%OSERAI_PYTHON%" -m chat_service.windows_host %*
if errorlevel 1 goto failed
popd
exit /b 0

:missing_python
echo.
echo No working 64-bit Python 3.12 or 3.13 was found.
echo WinGet could not install it. You can bypass WinGet entirely:
echo 1. Open https://www.python.org/downloads/release/python-31316/
echo 2. Download the Windows installer (64-bit), run it, and choose Install Now.
echo 3. Close this window and run Start-Oserai.cmd again.
goto failed

:failed
echo.
echo OserAI stopped at: %OSERAI_STEP%
echo Read the message above before closing this window.
pause
popd
exit /b 1

:try_python
if not exist "%~1" exit /b 1
"%~1" -c "import sys; assert sys.version_info[:2] in ((3,12),(3,13)) and sys.maxsize > 2**32" >nul 2>nul
if errorlevel 1 exit /b 1
set "OSERAI_PYTHON=%~1"
exit /b 0

:find_python
if not defined OSERAI_PYTHON goto common_paths
call :try_python "%OSERAI_PYTHON%"
if not errorlevel 1 exit /b 0
:common_paths
call :try_python "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not errorlevel 1 exit /b 0
call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not errorlevel 1 exit /b 0
call :try_python "%ProgramFiles%\Python313\python.exe"
if not errorlevel 1 exit /b 0
call :try_python "%ProgramFiles%\Python312\python.exe"
if not errorlevel 1 exit /b 0
call :from_py 3.13
if not errorlevel 1 exit /b 0
call :from_py 3.12
exit /b %errorlevel%

:from_py
set "OSERAI_PYTHON_PROBE=%TEMP%\oserai-python-%RANDOM%-%RANDOM%.txt"
py -%~1 -c "import sys; assert sys.version_info[:2] in ((3,12),(3,13)) and sys.maxsize > 2**32; print(sys.executable)" >"%OSERAI_PYTHON_PROBE%" 2>nul
if errorlevel 1 goto launcher_missing
set /p "OSERAI_PYTHON=" <"%OSERAI_PYTHON_PROBE%"
del /q "%OSERAI_PYTHON_PROBE%" >nul 2>nul
call :try_python "%OSERAI_PYTHON%"
exit /b %errorlevel%
:launcher_missing
del /q "%OSERAI_PYTHON_PROBE%" >nul 2>nul
exit /b 1
