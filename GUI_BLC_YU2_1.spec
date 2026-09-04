# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

# 1. 收集 MediaPipe 的所有二進位檔與數據
mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all('mediapipe')

# 2. 分析主程式 GUI_BLC_YU2_1.py
a_main = Analysis(
    ['GUI_BLC_YU2_1.py'],
    pathex=[],
    binaries=mediapipe_binaries,
    datas=mediapipe_datas,
    hiddenimports=mediapipe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# 3. 分析 UI.py
a_ui = Analysis(
    ['UI.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# 4. 分析 Action3_0.py
a_action = Analysis(
    ['Action3_0.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# 合併三個程式的依賴關係（共用 _internal）
MERGE(
    (a_main, 'GUI_BLC_YU2_1', 'GUI_BLC_YU2_1'),
    (a_ui, 'UI', 'UI'),
    (a_action, 'Action3_0', 'Action3_0')
)

pyz_main = PYZ(a_main.pure)
pyz_ui = PYZ(a_ui.pure)
pyz_action = PYZ(a_action.pure)

# 建立 GUI_BLC_YU2_1.exe
exe_main = EXE(
    pyz_main,
    a_main.scripts,
    [],
    exclude_binaries=True,
    name='GUI_BLC_YU2_1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# 建立 UI.exe
exe_ui = EXE(
    pyz_ui,
    a_ui.scripts,
    [],
    exclude_binaries=True,
    name='UI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# 建立 Action3_0.exe
exe_action = EXE(
    pyz_action,
    a_action.scripts,
    [],
    exclude_binaries=True,
    name='Action3_0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# 將所有內容統一輸出至同一個目錄
coll = COLLECT(
    exe_main,
    exe_ui,
    exe_action,
    a_main.binaries,
    a_main.datas,
    a_ui.binaries,
    a_ui.datas,
    a_action.binaries,
    a_action.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GUI_BLC_YU2_1',
)