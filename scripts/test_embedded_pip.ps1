$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'local_windows_helpers.ps1')
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$TempRoot = Join-Path ([IO.Path]::GetTempPath()) ('ballons-embedded-pip-probe-' + [guid]::NewGuid().ToString('N'))
$Runtime = Join-Path $TempRoot 'local_runtime_win'
$PythonExe = Join-Path $Runtime 'python.exe'
$ProbeProject = Join-Path $TempRoot 'backend-probe'
New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
New-Item -ItemType Directory -Path $ProbeProject -Force | Out-Null

try {
    $PythonArchive = Join-Path $TempRoot 'python.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile $PythonArchive
    Expand-Archive -LiteralPath $PythonArchive -DestinationPath $Runtime
    [IO.File]::WriteAllText((Join-Path $Runtime 'python312._pth'), "python312.zip`n.`n..`nLib\site-packages`nimport site`n", [Text.UTF8Encoding]::new($false))

    $GetPipScript = Join-Path $TempRoot 'get-pip.py'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $GetPipScript
    Initialize-EmbeddedPythonPackaging -PythonExe $PythonExe -GetPipScript $GetPipScript

    $Pyproject = @'
[build-system]
requires = ["setuptools>=70,<81"]
build-backend = "setuptools.build_meta"

[project]
name = "ballons-embedded-pip-probe"
version = "0.0.1"

[tool.setuptools]
py-modules = ["ballons_embedded_probe"]
'@
    [IO.File]::WriteAllText((Join-Path $ProbeProject 'pyproject.toml'), $Pyproject, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText((Join-Path $ProbeProject 'ballons_embedded_probe.py'), "OK = True`n", [Text.UTF8Encoding]::new($false))
    $RequirementsFile = Join-Path $TempRoot 'requirements.txt'
    [IO.File]::WriteAllText($RequirementsFile, "$ProbeProject`n", [Text.UTF8Encoding]::new($false))

    Install-EmbeddedPythonRequirements -PythonExe $PythonExe -RequirementsFile $RequirementsFile
    Invoke-CheckedNative $PythonExe @('-c', 'import ballons_embedded_probe; assert ballons_embedded_probe.OK')
    Write-Host 'Embeddable Python setuptools.build_meta package installation passed.'
} finally {
    Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
