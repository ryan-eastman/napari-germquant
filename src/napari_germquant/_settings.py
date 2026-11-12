"""Settings of the Run dock, kept in QSettings (docs/DESIGN_system_packs.md, section 8.1): the engine
interpreter and the interpreter of single recipes (per machine), the device choice and the last pack picked
(per user).

The engine interpreter is either the path of a Python executable (the subprocess runner) or the word
``inprocess`` (this napari environment, chosen explicitly). Nothing here guesses: with no value stored the
dock asks. QSettings keeps the values for this user (the registry on Windows, a plist on a Mac, an ini file on
Linux). A Windows roaming profile or a home folder shared by many computers (an NFS home on a cluster) carries
those values to every computer the user logs in to, so the interpreter keys include this computer's host name:
an interpreter chosen on one computer is never used on another, where the same path may hold another engine or
nothing. ``$NAPARI_GERMQUANT_SETTINGS`` names an ini file to use instead: the tests set it, so they never touch
the real settings, and a script driving the docks can do the same.

The project docks (tranche 5 Train B) add the recent projects, kept per computer like the interpreter (a project
folder's path, a UNC path kept verbatim, means nothing on another computer), and the annotator, the short name a
spot-tuning truth-point file carries (annotations/<image_id>__points_<annotator>.csv), kept per user.
"""
from __future__ import annotations

import json
import os
import re
import socket

from qtpy.QtCore import QSettings

SETTINGS_ENV = "NAPARI_GERMQUANT_SETTINGS"
ORG = "napari-germquant"
APP = "napari-germquant"
INPROCESS = "inprocess"

_ENGINE = "engine/{host}/interpreter"
_DEVICE = "engine/device"
_RECIPES = "recipe_interpreters/{host}"
_LAST_PACK = "ui/last_pack"
_LAST_RECIPE = "ui/last_recipe"
_RECENT = "projects/{host}/recent"
_ANNOTATOR = "tuning/annotator"
RECENT_MAX = 10
# the annotator of a truth-point file name (letters, digits, dash and underscore)
ANNOTATOR_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def host_key() -> str:
    """This computer's host name as a settings key part (lower case; anything but letters, digits, dot, dash
    and underscore becomes an underscore)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname().strip().lower()) or "unknown_host"


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
    """The engine interpreter stored for this computer: a Python path, ``inprocess``, or None (not chosen)."""
    return _get(_ENGINE.format(host=host_key()))


def set_engine_choice(value: str | None) -> None:
    _set(_ENGINE.format(host=host_key()), value)


def _recipe_key(pack_id: str, recipe_id: str) -> str:
    return f"{_RECIPES.format(host=host_key())}/{pack_id}/{recipe_id}"


def recipe_interpreter(pack_id: str | None, recipe_id: str | None) -> str | None:
    """The interpreter stored on this computer for one pack recipe (keyed by pack id and recipe id), or None."""
    if not pack_id or not recipe_id:
        return None
    return _get(_recipe_key(pack_id, recipe_id))


def set_recipe_interpreter(pack_id: str, recipe_id: str, value: str | None) -> None:
    _set(_recipe_key(pack_id, recipe_id), value)


def recipe_interpreters() -> dict[str, str]:
    """Every recipe interpreter stored on this computer, ``{"<pack id>/<recipe id>": interpreter}``."""
    s = qsettings()
    s.beginGroup(_RECIPES.format(host=host_key()))
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
    """The pack and recipe the user last picked (never one the dock picked on its own)."""
    return _get(_LAST_PACK), _get(_LAST_RECIPE)


def set_last_pack(pack_id: str | None, recipe_id: str | None) -> None:
    _set(_LAST_PACK, pack_id)
    _set(_LAST_RECIPE, recipe_id)


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def recent_projects() -> list[str]:
    """The project folders opened on this computer, the most recent first (at most RECENT_MAX), as given."""
    raw = _get(_RECENT.format(host=host_key()))
    try:
        items = json.loads(raw) if raw else []
    except ValueError:
        items = []
    if not isinstance(items, list):
        return []
    return [p for p in items if isinstance(p, str) and p.strip()][:RECENT_MAX]


def add_recent_project(path) -> list[str]:
    """Put ``path`` first in this computer's recent projects (once: an earlier entry of the same folder goes);
    returns the list."""
    p = str(path)
    items = [p, *[q for q in recent_projects() if not _same_path(q, p)]][:RECENT_MAX]
    _set(_RECENT.format(host=host_key()), json.dumps(items))
    return items


def remove_recent_project(path) -> list[str]:
    items = [q for q in recent_projects() if not _same_path(q, str(path))]
    _set(_RECENT.format(host=host_key()), json.dumps(items) if items else None)
    return items


def annotator() -> str | None:
    """The annotator name stored for this user, or None (the Spot tuning dock asks)."""
    value = _get(_ANNOTATOR)
    return value if value and ANNOTATOR_RE.fullmatch(value) else None


def set_annotator(name: str | None) -> None:
    """Store the annotator name (letters, digits, dash and underscore, 1 to 32 of them); None forgets it."""
    if name is None:
        _set(_ANNOTATOR, None)
        return
    if not isinstance(name, str) or not ANNOTATOR_RE.fullmatch(name):
        raise ValueError(f"not an annotator name: {name!r} (letters, digits, dash and underscore, at most 32)")
    _set(_ANNOTATOR, name)
