$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$electronRoot = Join-Path $repoRoot "electron"

Push-Location $electronRoot
try {
    if (-not (Test-Path "node_modules")) {
        npm install
    }

    npm test
    npm start
}
finally {
    Pop-Location
}
