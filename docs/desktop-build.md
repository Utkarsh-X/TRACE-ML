# TRACE-AML Desktop Build Guide

## Goal

Build a Windows Electron app that starts its own local backend without requiring a separate terminal window.

## Architecture

The desktop build is a two-stage package:

1. Build a standalone Python backend artifact from `.venv311`
2. Package that backend artifact into the Electron app

This is the supported path for future iterations. Do not bundle `.venv` or `.venv311` directly into Electron.

## Prerequisites

Use the working Python environment:

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\python.exe -m pip install pytest pyinstaller
```

## Quick Verification

Before packaging, confirm the desktop-share config starts correctly:

```powershell
cd "D:\github FORK\TRACE-ML"
.\.venv311\Scripts\trace-aml.exe --config config/config.desktop.yaml service run --host 127.0.0.1 --port 18080
```

Then verify:

```powershell
Invoke-WebRequest http://127.0.0.1:18080/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:18080/api/v1/live/snapshot -UseBasicParsing
```

## Build Backend

```powershell
cd "D:\github FORK\TRACE-ML"
.\scripts\build_backend.ps1
```

Expected output:

```text
build\backend\trace_aml_backend\trace-aml-backend.exe
```

## Build Electron

```powershell
cd "D:\github FORK\TRACE-ML\electron"
npm install
npm test
npm run dist
```

`npm run dist` now calls the backend build script first, then runs `electron-builder`.

## Shareable Config

Packaged mode uses:

```text
config/config.desktop.yaml
```

That profile:
- keeps the current demo recognition tuning
- disables outbound email
- disables outbound WhatsApp
- disables PDF report generation
- keeps action logging enabled

## Portable Data Behavior

Electron sets `TRACE_DATA_ROOT` before launching the backend, so packaged mode writes runtime data under the user-specific app-data directory instead of inside the install folder.

Important data categories:
- LanceDB vectors
- encrypted vault blobs and indexes
- screenshots
- logs
- exports

## Smoke Test After Build

Use the unpacked app first:

```powershell
cd "D:\github FORK\TRACE-ML\electron\dist\win-unpacked"
.\TRACE-AML.exe
```

Then verify:
- splash screen appears
- welcome screen appears
- main workspace loads
- `/health` responds on `127.0.0.1:18080`
- no manual backend terminal is required

## Current Blocking Requirement

If the build toolchain is missing, install these into `.venv311`:

```powershell
.\.venv311\Scripts\python.exe -m pip install pytest pyinstaller
```

Without `PyInstaller`, the Electron package cannot yet switch to the stable compiled-backend architecture.
