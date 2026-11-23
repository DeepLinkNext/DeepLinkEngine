@echo off
chcp 65001 >nul
call "%~dp0launcher\启动验收实验台.cmd" %*
exit /b %errorlevel%
