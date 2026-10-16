Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-CheckedNative {
    param([string]$Executable, [string[]]$NativeArguments)
    & $Executable @NativeArguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE : $Executable" }
}

# The Windows embeddable Python uses python312._pth and ignores PYTHONPATH.
# pip's build-isolation backend would consequently be unreachable. Install
# the build backend into this runtime and build source dependencies in-place.
function Initialize-EmbeddedPythonPackaging {
    param([string]$PythonExe, [string]$GetPipScript)
    Invoke-CheckedNative $PythonExe @($GetPipScript, '--no-warn-script-location')
    Invoke-CheckedNative $PythonExe @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', '--upgrade', 'setuptools>=70,<81', 'wheel')
    Invoke-CheckedNative $PythonExe @('-c', 'import setuptools.build_meta; import wheel')
}

function Install-EmbeddedPythonRequirements {
    param([string]$PythonExe, [string]$RequirementsFile)
    Invoke-CheckedNative $PythonExe @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', '--no-build-isolation', '--prefer-binary', '-r', $RequirementsFile)
}

function Set-LocalProcessEnvironment {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $script:OutputEncoding = [Console]::OutputEncoding
    $env:PYTHONUTF8 = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:OLLAMA_NO_CLOUD = '1'
    $env:OLLAMA_HOST = '127.0.0.1:11435'
    $env:OLLAMA_NUM_PARALLEL = '1'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    $env:DO_NOT_TRACK = '1'
}

function Find-LocalOllama {
    $Command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Installed = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $Installed) { return $Installed }
    return $null
}

function Start-ManagedOllama {
    param([string]$Executable, [string]$ProjectRoot)
    # Do not attach to or stop an unknown service on our dedicated port.
    $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 11435)
    try { $Listener.Start() } catch { throw 'Port 11435 is already occupied. Close the other local translator window and retry. No existing process was changed.' } finally { $Listener.Stop() }
    $LogDir = Join-Path $ProjectRoot 'logs'
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $RunId = [guid]::NewGuid().ToString('N')
    $Process = Start-Process -FilePath $Executable -ArgumentList 'serve' -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "ollama-$RunId.log") -RedirectStandardError (Join-Path $LogDir "ollama-$RunId.err.log")
    try {
        for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
            if ($Process.HasExited) { throw 'The local Ollama server stopped. See logs/ollama-*.err.log.' }
            try {
                $Response = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 2 -MaximumRedirection 0
                if ($Response.version) { return $Process }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        throw 'The local Ollama server did not become ready in time.'
    } catch {
        if (-not $Process.HasExited) { $Process.Kill() }
        throw
    }
}
