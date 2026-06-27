@echo off
chcp 65001 >nul
title 🛰️ 遥感图像分析平台 - 环境配置

echo ============================================
echo   🛰️  遥感图像分析平台 - 环境配置
echo ============================================
echo.

where bash >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set BASH=bash
) else (
    if exist "%ProgramFiles%\Git\bin\bash.exe" set BASH="%ProgramFiles%\Git\bin\bash.exe"
    if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set BASH="%ProgramFiles(x86)%\Git\bin\bash.exe"
    if exist "%LocalAppData%\Programs\Git\bin\bash.exe" set BASH="%LocalAppData%\Programs\Git\bin\bash.exe"
)

if "%BASH%"=="" (
    echo.
    echo [错误] 未找到 Git Bash
    echo.
    echo 解决方法:
    echo   1. 下载 Git for Windows:
    echo      https://git-scm.com/download/win
    echo.
    echo   2. 安装后重新运行本脚本
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Git Bash: %BASH%
echo.
echo 正在启动 setup.sh ...
"%BASH%" -c "cd \"%~dp0\" && bash setup.sh"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [失败] 错误码: %ERRORLEVEL%
    echo        请查看 setup_*.log
    pause
)

echo.
pause
