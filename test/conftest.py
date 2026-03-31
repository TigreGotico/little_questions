"""pytest configuration: stub out optional / unavailable heavy dependencies.

JarbasModelZoo and xdg (pyxdg) are optional packages not installed in all
environments.  Stubbing them at import time prevents ImportError when the
language feature modules are loaded.
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

# Ensure repo root is on sys.path so `train` package is importable from tests.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# --- JarbasModelZoo stub ---
if "JarbasModelZoo" not in sys.modules:
    _zoo_stub = MagicMock()
    _zoo_stub.LOG = MagicMock()
    sys.modules["JarbasModelZoo"] = _zoo_stub

# --- xdg / pyxdg stub ---
# pyxdg exposes `xdg.BaseDirectory`; stub the whole package if missing.
try:
    import xdg  # noqa: F401
except ModuleNotFoundError:
    import os
    _xdg_mod = types.ModuleType("xdg")
    _base_dir_mod = types.ModuleType("xdg.BaseDirectory")

    _home = os.path.expanduser("~")
    _xdg_data_home = os.path.join(_home, ".local", "share")

    class _BaseDirectory:
        xdg_data_home = _xdg_data_home

        @staticmethod
        def save_data_path(app: str) -> str:
            path = os.path.join(_xdg_data_home, app)
            os.makedirs(path, exist_ok=True)
            return path

    _base_dir_mod.BaseDirectory = _BaseDirectory()  # type: ignore[attr-defined]
    _xdg_mod.BaseDirectory = _BaseDirectory()  # type: ignore[attr-defined]

    sys.modules["xdg"] = _xdg_mod
    sys.modules["xdg.BaseDirectory"] = _base_dir_mod
