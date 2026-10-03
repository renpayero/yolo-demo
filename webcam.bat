@echo off
rem Publica la webcam en el servidor de video de la demo (ver scripts\webcam.ps1).
setlocal
rem Si se lanza desde PowerShell 7, su PSModulePath rompe los cmdlets de Windows PowerShell 5.1.
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\webcam.ps1" %*
