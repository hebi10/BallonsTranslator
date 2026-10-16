param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArguments)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'local_windows_helpers.ps1')
Set-LocalProcessEnvironment
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RuntimeDir = Join-Path $ProjectRoot 'local_runtime_win'
if (-not (Test-Path -LiteralPath (Join-Path $RuntimeDir '.ready')) -or -not (Find-LocalOllama)) {
    & (Join-Path $PSScriptRoot 'install_local.ps1') -InstallOllama
}
$PythonExe = Join-Path $RuntimeDir 'python.exe'
$env:PATH = "$RuntimeDir;$RuntimeDir\Scripts;$env:PATH"
$env:BALLOONTRANS_APP_ROOT = $ProjectRoot
$OllamaProcess = $null
Push-Location $ProjectRoot
try {
    $OllamaProcess = Start-ManagedOllama -Executable (Find-LocalOllama) -ProjectRoot $ProjectRoot
    Invoke-CheckedNative $PythonExe (@('-m', 'ballontranslator', '--config', (Join-Path $ProjectRoot 'config\hebi-local.json')) + @($AppArguments))
} finally {
    if ($OllamaProcess -and -not $OllamaProcess.HasExited) { $OllamaProcess.Kill() }
    Pop-Location
}
