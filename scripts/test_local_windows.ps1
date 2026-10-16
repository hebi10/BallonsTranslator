$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Files = @('install_local.ps1', 'start_local.ps1', 'local_windows_helpers.ps1')
foreach ($File in $Files) {
    $Tokens = $null
    $ParseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $File), [ref]$Tokens, [ref]$ParseErrors)
    if ($ParseErrors.Count -gt 0) { throw ($ParseErrors | Out-String) }
}
. (Join-Path $PSScriptRoot 'local_windows_helpers.ps1')
$Shell = (Get-Process -Id $PID).Path
Invoke-CheckedNative $Shell @('-NoProfile', '-Command', 'exit 0')
$Failed = $false
try { Invoke-CheckedNative $Shell @('-NoProfile', '-Command', 'exit 23') } catch { $Failed = $_.Exception.Message -match '23' }
if (-not $Failed) { throw 'Native command failure was not propagated.' }
$env:OLLAMA_NO_CLOUD = '0'
Set-LocalProcessEnvironment
if ($env:OLLAMA_NO_CLOUD -ne '1' -or $env:OLLAMA_HOST -ne '127.0.0.1:11435') { throw 'Managed server environment is incorrect.' }
if ($env:PYTHONUTF8 -ne '1' -or $env:PYTHONNOUSERSITE -ne '1') { throw 'Python isolation environment is incorrect.' }
if ($env:HF_HUB_OFFLINE -ne '1' -or $env:TRANSFORMERS_OFFLINE -ne '1') { throw 'Model loading is not offline.' }
Write-Host 'Windows script parsing and native-exit/environment checks passed.'
# GitHub Actions propagates LASTEXITCODE; the expected exit-23 probe must not leak.
exit 0
