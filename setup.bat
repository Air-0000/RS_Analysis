@echo off
chcp 65001 >nul
title 🛰️ 遥感图像分析平台 - 环境配置

REM =============================================
REM  遥感图像分析平台 — Windows 启动入口
REM  自动检测 Git Bash，找不到则引导安装
REM =============================================

echo ============================================
echo   🛰️  遥感图像分析平台 — 环境配置
echo ============================================
echo.

REM 尝试定位 Git Bash
set "BASH="
where bash >nul 2>&1
if %ERRORLEVEL% EQU 0 set "BASH=bash"

REM 检查常见安装路径
if "%BASH%"=="" (
    if exist "%ProgramFiles%\Git\bin\bash.exe" set "BASH=%ProgramFiles%\Git\bin\bash.exe"
)
if "%BASH%"=="" (
    if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set "BASH=%ProgramFiles(x86)%\Git\bin\bash.exe"
)
if "%BASH%"=="" (
    if exist "%LocalAppData%\Programs\Git\bin\bash.exe" set "BASH=%LocalAppData%\Programs\Git\bin\bash.exe"
)

if "%BASH%"=="" (
    echo ❌ 未找到 Git Bash
    echo.
    echo 本脚本需要 Git Bash 环境来运行。
    echo.
    echo 解决方法：
    echo   1. 下载 Git for Windows:
    echo      https://git-scm.com/download/win
    echo.
    echo   2. 安装时保持默认设置（确保选中"Git Bash"组件）
    echo.
    echo   3. 安装完成后重新运行本脚本
    echo.
    echo 或者直接用 Git Bash 手动运行:
    echo   bash setup.sh
    echo.
    pause
    exit /b 1
)

echo ✅ Git Bash: %BASH%
echo.
echo 🚀 启动 setup.sh ...
"%BASH%" -c "cd \"%~dp0\" && bash setup.sh"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 运行失败，错误码: %ERRORLEVEL%
    echo   请查看 setup_*.log 获取详细信息
    pause
)

echo.
echo 按任意键退出...
pause >nul
