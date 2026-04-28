# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH).resolve().parent
entrypoint = project_root / "src" / "trace_aml" / "desktop_backend.py"

datas = [
    (str(project_root / "config"), "config"),
    (str(project_root / "models"), "models"),
    (str(project_root / "src" / "frontend"), "src/frontend"),
]

env_file = project_root / ".env"
if env_file.exists():
    datas.append((str(env_file), "."))

binaries = []
hiddenimports = [
    "trace_aml.desktop_backend",
    "trace_aml.cli",
    "dotenv",
    "insightface.app",
    "insightface.model_zoo",
    "insightface.utils",
    "cv2",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
]

for package_name in ("trace_aml", "insightface", "dotenv", "skimage", "cryptography"):
    try:
        hiddenimports += collect_submodules(package_name)
    except Exception:
        pass

for package_name in ("insightface", "skimage", "cryptography"):
    try:
        datas += collect_data_files(package_name)
    except Exception:
        pass

for package_name in ("cv2", "onnxruntime", "numpy", "scipy", "pyarrow", "duckdb", "cryptography"):
    try:
        binaries += collect_dynamic_libs(package_name)
    except Exception:
        pass


a = Analysis(
    [str(entrypoint)],
    pathex=[str(project_root), str(project_root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "playwright",
        "playwright.sync_api",
        "pytest",
        "_pytest",
        "sklearn",
        "tkinter",
        "_tkinter",
        "Cython",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="trace-aml-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="trace_aml_backend",
)
