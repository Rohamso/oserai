# PyInstaller spec for the OserAI Trainer desktop app.
#
#   pyinstaller desktop_app/build.spec
#
# Produces a single-folder app bundle under dist/OserAI-Trainer/ containing
# a native launcher -- no separate Python install required on the
# contributor's machine. Run from the repo root so relative paths resolve.
#
# Only verified on macOS in this repo so far (the only OS available to
# build/test on). Windows/Linux builds should work unchanged with the same
# command on those platforms (PyInstaller itself is cross-platform, but a
# build always targets the OS it runs on -- there's no cross-compiling a
# Windows .exe from macOS or vice versa), but haven't been run here.

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
import sys

# Run from anywhere (repo root or desktop_app/ -- `pyinstaller build.spec`
# is typically run from inside desktop_app/, per the README) and still find
# the repo root, since main.py imports sibling packages (agent, training,
# common) that live one level up from this spec file, not inside
# desktop_app/ itself. Without the repo root on pathex, PyInstaller's
# static import analysis can't discover those packages at all -- the build
# succeeds but the frozen app fails at startup with
# "ModuleNotFoundError: No module named 'agent'".
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))
ASSETS_DIR = os.path.join(REPO_ROOT, "desktop_app", "assets")

# icon= wants a platform-native format: .icns on macOS (ignored on other
# platforms), .ico on Windows -- this spec is the same file used for both
# the local macOS build and the windows-latest CI build, so pick by
# sys.platform rather than hardcoding one.
if sys.platform == "darwin":
    APP_ICON = os.path.join(ASSETS_DIR, "icon.icns")
elif sys.platform == "win32":
    APP_ICON = os.path.join(ASSETS_DIR, "icon.ico")
else:
    APP_ICON = None

a = Analysis(
    ["main.py"],
    pathex=[REPO_ROOT],
    binaries=[],
    datas=[
        (os.path.join(ASSETS_DIR, "logo_tray.png"), "assets"),
        # training/dataset.py's DATA_DIR resolves against sys._MEIPASS when
        # frozen (see its comment) -- without this, a contributor's machine
        # crashes loading the training data right after the base model
        # download finishes, since it never ran prepare_training_data.py
        # itself and has no repo checkout to find these files in.
        (os.path.join(REPO_ROOT, "training", "data"), os.path.join("training", "data")),
    ],
    # torch/transformers/peft do a lot of dynamic/lazy importing that
    # PyInstaller's static analysis can miss -- collect their submodules
    # explicitly rather than debugging missing-module errors one at a time.
    hiddenimports=[
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "agent",
        "agent.coordinator_client",
        "agent.train_agent",
        "agent.train_config",
        "training",
        "training.chat_dataset",
        "training.checkpoint_io",
        "training.config",
        "training.dataset",
        "training.lora",
        "training.model",
        "training.outer_optim",
        "common",
        "common.logging_setup",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OserAI-Trainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=APP_ICON,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="OserAI-Trainer",
)

# macOS-only: wraps COLLECT's output in a proper .app bundle (Info.plist,
# NSApplication activation policy). Without this, running the raw
# executable directly doesn't register as a normal foreground GUI app with
# the window server -- it launches, but no window ever becomes visible.
app = BUNDLE(
    coll,
    name="OserAI-Trainer.app",
    icon=APP_ICON,
    bundle_identifier="com.oserai.trainer",
)
