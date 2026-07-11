# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for SlopCAD.
#
# build123d / OCP load a lot of modules dynamically, so we list the key
# hidden imports explicitly.  The collect_* helpers grab all the data files
# (shaders, fonts, etc.) that PyInstaller would otherwise miss.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

import os, sys
venv_sp = next(p for p in sys.path if "site-packages" in p and os.path.isdir(p))

datas = []
datas += collect_data_files("build123d")
datas += collect_data_files("ocp_tessellate")

hiddenimports = []
hiddenimports += collect_submodules("OCP")
hiddenimports += collect_submodules("build123d")
hiddenimports += collect_submodules("ocp_tessellate")
hiddenimports += [
    "PyQt6.QtOpenGL",
    "PyQt6.QtOpenGLWidgets",
    "OpenGL",
    "OpenGL.GL",
    "python_solvespace",
]
if sys.platform == "win32":
    hiddenimports += ["OpenGL.platform.win32"]

lib3mf_ext = ".so" if sys.platform != "win32" else ".dll"
lib3mf_bin = os.path.join(venv_sp, "lib3mf", f"lib3mf{lib3mf_ext}")
binaries = [(lib3mf_bin, "lib3mf")] if os.path.exists(lib3mf_bin) else []

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["vtk", "matplotlib", "ipython"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="slopcad",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # no terminal window
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="slopcad",
)
