param([switch]$InstallOllama)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'local_windows_helpers.ps1')
Set-LocalProcessEnvironment
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RuntimeDir = Join-Path $ProjectRoot 'local_runtime_win'
$PythonExe = Join-Path $RuntimeDir 'python.exe'
$CompleteMarker = Join-Path $RuntimeDir '.ready'
$BuildDir = Join-Path ([IO.Path]::GetTempPath()) ('ballons-local-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $BuildDir | Out-Null
Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        if (Test-Path -LiteralPath $RuntimeDir) { throw 'local_runtime_win exists without python.exe. Keep a backup and move that folder before retrying; it was not deleted.' }
        Write-Host 'Downloading isolated Python 3.12.10 runtime...'
        $PythonZip = Join-Path $BuildDir 'python.zip'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile $PythonZip
        $StagingRuntime = Join-Path $BuildDir 'runtime'
        Expand-Archive -LiteralPath $PythonZip -DestinationPath $StagingRuntime
        $PthFile = Join-Path $StagingRuntime 'python312._pth'
        [IO.File]::WriteAllText($PthFile, "python312.zip`n.`n..`nLib\site-packages`nimport site`n", [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $StagingRuntime -Destination $RuntimeDir
    }
    if (-not (Test-Path -LiteralPath $CompleteMarker)) {
        Write-Host 'Installing Python packages. Existing models and settings are preserved.'
        $GetPip = Join-Path $BuildDir 'get-pip.py'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $GetPip
        Invoke-CheckedNative $PythonExe @($GetPip, '--no-warn-script-location')
        Invoke-CheckedNative $PythonExe @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', '-r', (Join-Path $ProjectRoot 'requirements-local.txt'))
        $ProfileScript = Join-Path $BuildDir 'torch_profile.py'
        $ProfileSource = @'
import json
from ballontranslator.utils.torch_install_helper import select_torch_install_profile_for_device
profile, device = select_torch_install_profile_for_device(None)
print(json.dumps({'requirements': list(profile.requirements), 'index': profile.index_url, 'device': device}))
'@
        [IO.File]::WriteAllText($ProfileScript, $ProfileSource, [Text.UTF8Encoding]::new($false))
        $ProfileJson = & $PythonExe $ProfileScript
        if ($LASTEXITCODE -ne 0) { throw 'GPU package selection failed. No ready marker was written.' }
        $TorchProfile = $ProfileJson | ConvertFrom-Json
        Write-Host "Installing image processing backend for $($TorchProfile.device)..."
        $TorchArgs = @('-m', 'pip', 'install', '--no-warn-script-location', '--index-url', $TorchProfile.index) + @($TorchProfile.requirements)
        Invoke-CheckedNative $PythonExe $TorchArgs
        Invoke-CheckedNative $PythonExe @('-c', 'import torch, torchvision, transformers, jaconv, fugashi; from PyQt6 import QtWidgets')
        [IO.File]::WriteAllText($CompleteMarker, "Python 3.12.10; local requirements installed`n", [Text.UTF8Encoding]::new($false))
    }
    if (-not (Find-LocalOllama)) {
        if (-not $InstallOllama) { throw 'Ollama is missing. Run this installer with -InstallOllama or install it from https://ollama.com/download/windows.' }
        $Installer = Join-Path $BuildDir 'OllamaSetup.exe'
        Write-Host 'Downloading the official Ollama installer...'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile $Installer
        $Signature = Get-AuthenticodeSignature -LiteralPath $Installer
        if ($Signature.Status -ne 'Valid' -or -not $Signature.SignerCertificate -or $Signature.SignerCertificate.Subject -notmatch 'Ollama') {
            throw 'The Ollama installer signature or publisher could not be verified. The installer was not run.'
        }
        # Interactive official installer: preserves its normal consent and installation UI.
        # The official setup can start the tray app. Keep it off our managed port.
        $InstallerPreviousHost = $env:OLLAMA_HOST
        try {
            $env:OLLAMA_HOST = '127.0.0.1:11434'
            $Setup = Start-Process -FilePath $Installer -Wait -PassThru
        } finally {
            $env:OLLAMA_HOST = $InstallerPreviousHost
        }
        if ($Setup.ExitCode -ne 0) { throw "Ollama installer returned $($Setup.ExitCode)." }
        if (-not (Find-LocalOllama)) { throw 'Ollama was not found after installation. Reopen launch_local.bat after completing the installer.' }
    }
    Write-Host 'Runtime is ready. Start launch_local.bat and use the preparation screen to download the selected models.'
} finally {
    Pop-Location
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
