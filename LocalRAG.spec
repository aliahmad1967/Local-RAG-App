# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for LocalRAG.

The onefile mode is deliberately avoided: torch/sentence-transformers unpack
hundreds of MB on every launch. Build with:
    pyinstaller --noconfirm --clean LocalRAG.spec
Run the result from dist/LocalRAG/LocalRAG.exe. User data is written beside the
executable, not into the temporary bundle directory.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [
    ("config.yaml", "."),
    ("localrag/db/schema.sql", "localrag/db"),
    ("localrag/ui/static", "localrag/ui/static"),
]
datas += collect_data_files("certifi", includes=["cacert.pem"])

hiddenimports = [
    "uvicorn",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "webview",
    "webview.platforms.edgechromium",
    "pystray._win32",
    "fitz",
    "PIL.Image",
    "numpy",
    "httpx",
    "yaml",
    "sqlite3",
    "sentence_transformers",
    "transformers",
    "torch",
]
for package in ("uvicorn", "sentence_transformers", "transformers"):
    hiddenimports += collect_submodules(package)

excludes = [
    "pytest",
    "tkinter",
    "PyQt5",
    "PySide2",
    "matplotlib",
    "jupyter",
    "IPython",
]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalRAG",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LocalRAG",
)
