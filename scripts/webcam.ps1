<#
.SYNOPSIS
    Publica la webcam de esta PC como si fuera una cámara IP (RTSP) para la demo.

.DESCRIPTION
    Docker Desktop en Windows no puede leer webcams USB. Este script usa ffmpeg portable
    (se descarga una sola vez a .\tools, sin instalar nada) para capturar la webcam y publicarla
    en el servidor de video de la demo: rtsp://127.0.0.1:8554/webcam
    Si ffmpeg se corta, lo vuelve a lanzar. Ctrl+C para terminar.

.EXAMPLE
    .\webcam.bat                         # primera cámara encontrada, 1280x720 a 30 fps
    .\webcam.bat -List                   # lista las cámaras disponibles
    .\webcam.bat -Device "UVC Camera" -Size 640x480
#>
param(
    [string]$Device = "",
    [string]$Size = "1280x720",
    [int]$Fps = 30,
    # 127.0.0.1 y no "localhost": en Windows localhost resuelve primero a ::1 (IPv6) y ffmpeg se queda colgado,
    # porque Docker publica el puerto solo en IPv4.
    [string]$Url = "rtsp://127.0.0.1:8554/webcam",
    [switch]$List
)

# "Continue": en Windows PowerShell 5.1, con "Stop" cualquier línea que ffmpeg escriba por stderr corta el script.
$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root "tools"
$Ffmpeg = Join-Path $Tools "ffmpeg.exe"
$FfmpegZipUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip"
$FfmpegSha256 = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"

function Get-Sha256([string]$Path) {
    # .NET directo: Get-FileHash no siempre está disponible en Windows PowerShell 5.1.
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try { return -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("x2") }) }
    finally { $stream.Dispose(); $sha.Dispose() }
}

function Install-Ffmpeg {
    if (Test-Path $Ffmpeg) { return }
    Write-Host "Descargando ffmpeg portable (una sola vez, ~115 MB)..."
    New-Item -ItemType Directory -Force $Tools | Out-Null
    $zip = Join-Path $env:TEMP "ffmpeg-essentials-9.0.2.zip"
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $FfmpegZipUrl -OutFile $zip -ErrorAction Stop
    if ((Get-Sha256 $zip) -ne $FfmpegSha256) {
        Remove-Item $zip
        throw "La descarga de ffmpeg no coincide con el checksum esperado; no se usa."
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
    try {
        $entry = $archive.Entries | Where-Object { $_.FullName -like "*/bin/ffmpeg.exe" } | Select-Object -First 1
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Ffmpeg, $true)
    }
    finally {
        $archive.Dispose()
        Remove-Item $zip
    }
    Write-Host "ffmpeg listo en $Ffmpeg"
}

function Get-VideoDevices {
    # ffmpeg informa los dispositivos por stderr:  [dshow @ ...] "UVC Camera" (video)
    $lines = & $Ffmpeg -hide_banner -list_devices true -f dshow -i dummy 2>&1 | ForEach-Object { "$_" }
    $lines | Where-Object { $_ -match '"(.+)" \(video\)' } | ForEach-Object { $Matches[1] }
}

function Test-MjpegSupport([string]$Name) {
    # Muchas webcams USB solo dan 720p a 30 fps en MJPEG; en formato crudo bajan a 5-10 fps.
    $lines = & $Ffmpeg -hide_banner -list_options true -f dshow -i "video=$Name" 2>&1 | ForEach-Object { "$_" }
    return [bool]($lines | Where-Object { $_ -match "vcodec=mjpeg" -and $_ -match "s=$Size" })
}

Install-Ffmpeg
$devices = @(Get-VideoDevices)

if ($List) {
    if ($devices.Count -eq 0) { Write-Host "No se encontró ninguna cámara." }
    $devices | ForEach-Object { Write-Host "  - $_" }
    exit 0
}
if ($devices.Count -eq 0) {
    Write-Host "No se encontró ninguna cámara. Revisá que esté conectada y que Windows permita el acceso a la cámara a las aplicaciones de escritorio."
    exit 1
}
if (-not $Device) { $Device = $devices[0] }

$inputArgs = @("-f", "dshow", "-rtbufsize", "100M")
if (Test-MjpegSupport $Device) { $inputArgs += @("-vcodec", "mjpeg") }
$inputArgs += @("-video_size", $Size, "-framerate", "$Fps", "-i", "video=$Device")

# H.264 de baja latencia: sin B-frames, un keyframe por segundo, bitrate acotado.
$encodeArgs = @(
    "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
    "-g", "$Fps", "-bf", "0", "-b:v", "2500k", "-maxrate", "2500k", "-bufsize", "1250k",
    "-f", "rtsp", "-rtsp_transport", "tcp", $Url
)

Write-Host "Publicando '$Device' ($Size @ $Fps fps) en $Url"
Write-Host "Ctrl+C para terminar."

$quickFailures = 0
while ($true) {
    $started = Get-Date
    & $Ffmpeg -hide_banner -loglevel warning @inputArgs @encodeArgs
    $code = $LASTEXITCODE
    $ranSeconds = ((Get-Date) - $started).TotalSeconds

    if ($ranSeconds -lt 5) { $quickFailures++ } else { $quickFailures = 0 }
    if ($quickFailures -eq 3 -and $inputArgs -contains "-video_size") {
        Write-Host "La cámara no acepta $Size @ $Fps fps; pruebo con su formato por defecto."
        $inputArgs = @("-f", "dshow", "-rtbufsize", "100M", "-i", "video=$Device")
    }
    Write-Host "ffmpeg terminó (código $code). Reintento en 3 s..."
    Write-Host "  Si se repite: ¿está levantado 'docker compose up'? ¿La cámara la usa otra aplicación (Teams, OBS)?"
    Start-Sleep -Seconds 3
}
