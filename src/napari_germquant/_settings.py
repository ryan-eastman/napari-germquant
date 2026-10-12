"""Per-machine settings of the Run dock, kept in QSettings (docs/DESIGN_system_packs.md, section 8.1): the engine
interpreter, the interpreter of single recipes, the device choice and the last pack picked.

The engine interpreter is either the path of a Python executable (the subprocess runner) or the word
``inprocess`` (this napari environment, chosen explicitly). Nothing here guesses: with no value stored the
dock asks. QSettings keeps the values for this user on this computer (the registry on Windows, a plist on a
Mac, an ini file on Linux). ``$NAPARI_GERMQUANT_SETTINGS`` names an ini file to use instead: the tests set it, so
they never touch the real settings, and a script driving the docks can do the same.
"""
from __future__ import annotations

import os

from qtpy.QtCore import QSettings

SETTINGS_ENV = "NAPARI_GERMQUANT_SETTINGS"
ORG = "napari-germquant"
APP = "napari-germquant"
INPROCESS = "inprocess"

_ENGINE = "engine/interpreter"
_DEVICE = "engine/device"
_RECIPES = "recipe_interpreters"
_LAST_PACK = "ui/last_pack"
_LAST_RECIPE = "ui/last_recipe"


def qsettings() -> QSettings:
    path = os.environ.get(SETTINGS_ENV, "").strip()
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, ORG, APP)


def _get(key: str) -> str | None:
    value = qsettings().value(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _set(key: str, value: str | None) -> None:
    s = qsettings()
    if value is None or str(value).strip() == "":
        s.remove(key)
    else:
        s.setValue(key, str(value))
    s.sync()


def engine_choice() -> str | None:
    """The stored engine interpreter: a Python path, ``inprocess``, or None (not chosen yet)."""
    return _get(_ENGINE)


def set_engine_choice(value: str | None) -> None:
    _set(_ENGINE, value)


def _recipe_key(pack_id: str, recipe_id: str) -> str:
    return f"{_RECIPES}/{pack_id}/{recipe_id}"


def recipe_interpreter(pack_id: str | None, recipe_id: str | None) -> str | None:
    """The interpreter stored for one pack recipe (keyed by pack id and recipe id), or None."""
    if not pack_id or not recipe_id:
        return None
    return _get(_recipe_key(pack_id, recipe_id))


def set_recipe_interpreter(pack_id: str, recipe_id: str, value: str | None) -> None:
    _set(_recipe_key(pack_id, recipe_id), value)


def recipe_interpreters() -> dict[str, str]:
    """Every stored recipe interpreter, ``{"<pack id>/<recipe id>": interpreter}``."""
    s = qsettings()
    s.beginGroup(_RECIPES)
    try:
        return {k: str(s.value(k)) for k in s.allKeys() if s.value(k)}
    finally:
        s.endGroup()


def device() -> str:
    """The device asked for: auto (the engine picks CUDA, Apple Metal, then the CPU), cuda, mps or cpu."""
    value = _get(_DEVICE)
    return value if value in ("auto", "cuda", "mps", "cpu") else "auto"


def set_device(value: str) -> None:
    _set(_DEVICE, value)


def last_pack() -> tuple[str | None, str | None]:
    return _get(_LAST_PACK), _get(_LAST_RECIPE)


def set_last_pack(pack_id: str | None, recipe_id: str | None) -> None:
    _set(_LAST_PACK, pack_id)
    _set(_LAST_RECIPE, recipe_id)
