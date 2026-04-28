param(
  [string]$PythonExe = ".venv311\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "build\backend"
$work = Join-Path $root "build\backend-work"
$spec = Join-Path $root "packaging\trace_aml_backend.spec"
$pythonPath = $PythonExe

if (-not [System.IO.Path]::IsPathRooted($pythonPath)) {
  $pythonPath = Join-Path $root $pythonPath
}

$pythonPath = [System.IO.Path]::GetFullPath($pythonPath)

if (-not (Test-Path $pythonPath)) {
  throw "Python executable not found: $pythonPath"
}

if (-not (Test-Path $spec)) {
  throw "PyInstaller spec not found: $spec"
}

Remove-Item -LiteralPath $dist -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue

& $pythonPath -m PyInstaller --noconfirm --distpath $dist --workpath $work $spec
