# Build dist\slopcad-windows.zip and run the headless selftest on it.
# Used by CI and by packaging/release.sh (over ssh to a Windows box).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build_windows.ps1 [-Python py] [-NoSmoke]
param(
    [string]$Python = "python",
    [switch]$NoSmoke
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Run([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "$exe $($argv -join ' ') failed ($LASTEXITCODE)" }
}

$venv = ".build-venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    $pyArgs = @()
    if ($Python -eq "py") { $pyArgs = @("-3.12") }
    Run $Python ($pyArgs + @("-m", "venv", $venv))
}
$py = "$venv\Scripts\python.exe"
Run $py @("-m", "pip", "install", "-q", "--upgrade", "pip")
Run $py @("-m", "pip", "install", "-q", "-r", "requirements.txt", "-r", "packaging\requirements-build.txt")

# slopcad.spec picks up slopcad.ico if present (multi-res for taskbar/Explorer)
Run $py @("-c", "from PIL import Image; Image.open('brand/avatar-640.png').save('slopcad.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])")
Run "$venv\Scripts\pyinstaller.exe" @("--noconfirm", "--clean", "slopcad.spec")

if (Test-Path dist\slopcad-windows.zip) { Remove-Item dist\slopcad-windows.zip }
Compress-Archive -Path dist\slopcad\* -DestinationPath dist\slopcad-windows.zip
Write-Host "built dist\slopcad-windows.zip"

if (-not $NoSmoke) {
    # The exe is windowed (no stdout): selftest writes its JSON to a file, and
    # Start-Process -Wait is needed to get its exit code.
    $out = Join-Path $env:TEMP "slopcad-selftest.json"
    $p = Start-Process -FilePath "dist\slopcad\slopcad.exe" -ArgumentList "--selftest", "`"$out`"" -Wait -PassThru -NoNewWindow
    Get-Content $out -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0) { throw "selftest failed (exit $($p.ExitCode))" }
}
