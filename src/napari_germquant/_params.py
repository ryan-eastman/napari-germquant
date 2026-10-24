"""What the Recipe tab of the Project dock shows and sends: the parameter form's rows (the pack recipe's
tunables) and the role editor's roles (docs/DESIGN_system_packs.md, sections 7.3 and 8.5; tranche 5 Train B plan,
decisions D19 and D39). No Qt: every rule here is tested headless.

Where it reads and where it writes. The plugin reads a project's files in this napari environment: the manifest
and the hash through germquant.project.open_project, the pinned pack recipe (its tunables, its values and its
channel map) through germquant.packs with the project's packs/ searched first, recipe.yaml for the edits, and
the reference image's header (``read_view``). Every change goes to the engine instead, as ``project edit`` through
the engine runner (``_runner``), the code that also runs the project: its dry run (``--dry-run --json``) gives
the role preview and the config_hash an edit would give, its write validates by origin (only declared tunables,
of the declared type, inside min, max and choices; roles by exact channel name) and refuses anything else. The
form's own checks here (``Tunable.check``) only keep a widget inside what the pack declares; the engine's
answer is the last word. No resolver is copied here: how a role binds on an image is what the engine's dry run
says.

recipe.yaml holds only ``extends: pack:<id>@<version>/<recipe>`` and the edits (design 7.3), so its keys are the
project's edits; the manifest says where each came from (``recipe.origins``: form, tuning, model or manual) and
which roles the role editor bound (``recipe.roles``: {role: exact channel name}).
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import numbers
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class _Sentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


# a value the plugin could not read (the pack recipe did not load here)
MISSING: Any = _Sentinel("MISSING")
# a pending form change that drops the project's edit of a key: back to the pack's value
RESET: Any = _Sentinel("RESET")

# the tunable types the form writes (the engine's form origin takes these; "string" is read as "str")
EDITABLE_TYPES = ("float", "int", "bool", "choice", "str")
ORIGIN_TEXT = {
    "form": "the parameter form",
    "tuning": "a tuning run (Spot tuning, Apply)",
    "model": "Use in project",
    "manual": "a hand edit of recipe.yaml",
}
RECIPE_NAME = "recipe.yaml"
CHANNEL_MAP_NAME = "channel_map.yaml"
MANIFEST_NAME = "germquant_project.yaml"
# the preview cell states, worst first: a collision and a required role unbound block Apply (red); the index
# fallback is amber (the run flags it); an optional role unbound and a name match are plain
STATES = ("collision", "missing", "index", "unbound", "name")
HOW_TEXT = {"name": "by name", "index": "by index", "unbound": "unbound"}


class ProjectsMissing(RuntimeError):
    """This napari environment's germquant cannot read projects (it predates germquant.project)."""


def _same(a, b) -> bool:
    """Value equality as the engine's form compares it: true and false are never numbers, numbers compare by
    value (1 equals 1.0), anything else by type and value."""
    if a is MISSING or b is MISSING:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return float(a) == float(b)
    return type(a) is type(b) and a == b


def value_text(value) -> str:
    """A value as the form prints it (YAML words for true and false)."""
    if value is MISSING:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _bound(b):
    return b if isinstance(b, numbers.Real) and not isinstance(b, bool) and math.isfinite(float(b)) else None


@dataclasses.dataclass(frozen=True)
class Tunable:
    """One row of the form: a tunable the pack recipe declares (pack.yaml ``recipes[].tunables``), the pack
    recipe's value and the project's (its recipe.yaml edit, else the pack's), and where the edit came from
    (``origin``: form, tuning, model, manual; None when the project does not set the key)."""

    key: str
    label: str | None
    type: str
    unit: str | None
    lo: Any
    hi: Any
    choices: tuple
    pack_value: Any
    value: Any
    origin: str | None

    @classmethod
    def from_declaration(cls, decl: dict, *, pack_value=MISSING, value=MISSING, origin=None) -> "Tunable":
        kind = str(decl.get("type") or "").strip().lower()
        kind = "str" if kind == "string" else kind
        choices = decl.get("choices")
        return cls(key=str(decl.get("key")), label=str(decl["label"]) if decl.get("label") else None, type=kind,
                   unit=str(decl["unit"]) if decl.get("unit") else None, lo=_bound(decl.get("min")),
                   hi=_bound(decl.get("max")), choices=tuple(choices) if isinstance(choices, (list, tuple)) else (),
                   pack_value=pack_value, value=pack_value if value is MISSING else value,
                   origin=str(origin) if origin else None)

    @property
    def title(self) -> str:
        return self.label or self.key

    @property
    def editable(self) -> bool:
        return self.type in EDITABLE_TYPES

    @property
    def edited(self) -> bool:
        return self.origin is not None

    @staticmethod
    def same(a, b) -> bool:
        return _same(a, b)

    def check(self, value):
        """The value to write for this key, or ValueError naming the key and the rule (the engine's form rules:
        one scalar of the declared type, inside min, max and choices; numbers are finite)."""
        k = self.key
        if not self.editable:
            raise ValueError(f"{k}: tunables of type {self.type or '(none declared)'} are shown read only; the form "
                             f"does not write them")
        if value is None or isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"{k}: the form sets one value (a number, true or false, or a word), got {value!r}")
        if self.type in ("float", "int"):
            is_num = isinstance(value, numbers.Real) and not isinstance(value, bool)
            if self.type == "int" and not (is_num and math.isfinite(float(value)) and float(value).is_integer()):
                raise ValueError(f"{k} takes a whole number (int), got {value!r}")
            if self.type == "float" and not is_num:
                raise ValueError(f"{k} takes a float (a number), got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{k} takes a finite number, got {value!r}")
            v = int(value) if self.type == "int" else float(value)
            if self.lo is not None and v < self.lo:
                raise ValueError(f"{k} must be at least {self.lo:g}, got {v:g}")
            if self.hi is not None and v > self.hi:
                raise ValueError(f"{k} must be at most {self.hi:g}, got {v:g}")
            return v
        if self.type == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{k} takes true or false (bool), got {value!r}")
            return value
        if self.type == "choice":
            for c in self.choices:
                if _same(value, c):
                    return c
            raise ValueError(f"{k} must be one of the choices {list(self.choices)}, got {value!r}")
        if not isinstance(value, str):
            raise ValueError(f"{k} takes a word (str), got {value!r}")
        return value

    def pack_text(self) -> str:
        return "pack value unknown" if self.pack_value is MISSING else f"pack {value_text(self.pack_value)}"

    def note_text(self) -> str:
        """Where the project's value comes from, or why the row is read only."""
        if not self.editable:
            return f"shown read only (type {self.type or 'not declared'})"
        if self.origin is None:
            return ""
        if self.origin == "form":
            return "edited (the parameter form)"
        return f"set by {ORIGIN_TEXT.get(self.origin, self.origin)}"


@dataclasses.dataclass(frozen=True)
class OtherEdit:
    """A key recipe.yaml sets that the form does not edit (a tuning result, Use in project, a hand edit): shown
    read only, with where it came from."""

    key: str
    value: Any
    origin: str

    def text(self) -> str:
        return f"{self.key} = {value_text(self.value)} ({ORIGIN_TEXT.get(self.origin, self.origin)})"


@dataclasses.dataclass(frozen=True)
class Role:
    """A role of the pack recipe's channel map: its marker, whether it is required, the pack map's name tokens
    and index fallback, and the project's role edit (an exact channel name, or None: the pack map)."""

    role: str
    marker: str
    required: bool
    names: tuple
    index: int | None
    edit: str | None

    def with_edit(self, channel: str | None) -> "Role":
        return dataclasses.replace(self, edit=str(channel) if channel else None)

    def pack_text(self) -> str:
        """The combo's first item: the pack map's own binding (no edit)."""
        if self.names:
            return "pack map: " + ", ".join(str(n) for n in self.names)
        if self.index is not None:
            return f"pack map: channel index {self.index}"
        return "pack map: no channel"


@dataclasses.dataclass
class RecipeView:
    """A project's recipe as the tab shows it (``read_view``). ``config_hash`` is the project's hash as this
    environment computes it on open (the engine's dry run confirms it), ``validated`` the manifest's record,
    ``digest`` the sha256 of the manifest and the two recipe files (a preview made for another digest is
    stale), ``saved`` the manifest's saved stamp (every write changes it)."""

    root: Path
    name: str
    pack: str
    recipe: str
    pack_hash: str | None
    config_hash: str | None
    edited: bool
    validated: dict | None
    tunables: list
    others: list
    roles: list
    match_by: str
    images: list
    reference: str | None
    channels: list
    channels_error: str | None
    warnings: list
    digest: str
    saved: dict | None = None

    def role(self, name: str) -> Role:
        for r in self.roles:
            if r.role == name:
                return r
        raise KeyError(f"{name} is not a role of the pack recipe's channel map (its roles: "
                       f"{', '.join(r.role for r in self.roles) or 'none'})")

    def tunable(self, key: str) -> Tunable:
        for t in self.tunables:
            if t.key == key:
                return t
        raise ValueError(f"{key} is not a tunable of the pack recipe {self.pack}/{self.recipe}: the parameter form "
                         f"edits only its tunables ({', '.join(t.key for t in self.tunables) or 'none declared'})")

    def saved_roles(self) -> dict[str, str | None]:
        return {r.role: r.edit for r in self.roles}

    def image_ids(self) -> list[str]:
        return [str(e["id"]) for e in self.images if not e.get("excluded")]


# ---------------------------------------------------------------------------------------------------
# reading a project in this environment
# ---------------------------------------------------------------------------------------------------


def _long(path):
    try:
        from germquant.fsutil import long_path
    except ImportError:
        return path
    return long_path(path)


def _read_bytes(path: Path) -> bytes | None:
    try:
        with open(_long(path), "rb") as fh:            # read and closed at once: no handle on a project file
            return fh.read()
    except FileNotFoundError:
        return None


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def recipe_edits(root: Path, manifest: dict) -> tuple[dict, str | None]:
    """``({dotted key: value}, problem)``: what the project's recipe.yaml sets, the extends line left out, and
    io.channel_map left out when it names the role editor's own channel_map.yaml. ``problem`` says why the file
    could not be read (the edits are then empty)."""
    import yaml

    rec = manifest.get("recipe") or {}
    if not rec.get("file"):
        return {}, None
    p = Path(root) / str(rec["file"])
    data = _read_bytes(p)
    if data is None:
        return {}, f"{p}: the manifest names it but it is missing"
    try:
        doc = yaml.safe_load(data.decode("utf-8", errors="replace")) or {}
    except yaml.YAMLError as e:
        return {}, f"{p}: not readable as YAML ({str(e).splitlines()[0]})"
    if not isinstance(doc, dict):
        return {}, f"{p}: not a mapping"
    doc.pop("extends", None)
    flat = _flatten(doc)
    if rec.get("roles") and flat.get("io.channel_map") == CHANNEL_MAP_NAME:
        flat.pop("io.channel_map")
    return flat, None


def image_path(root: Path, stored: str) -> Path:
    """The file a manifest path names: absolute (of any system, UNC included) as it is, relative under root."""
    s = str(stored)
    if PureWindowsPath(s).is_absolute() or PurePosixPath(s).is_absolute():
        return Path(s)
    return Path(root) / PurePosixPath(s)


def header_channels(path: Path) -> list[str]:
    """The channel names of an image, read with the engine's light header reader (germquant.project.headers, the
    reader the engine's role preview uses), else germquant.api's."""
    try:
        from germquant.project.headers import read_header
    except ImportError:
        from germquant import api

        return [str(n) for n in api.read_image_metadata(path).get("channel_names") or []]
    return [str(n) for n in read_header(path).get("channel_names") or []]


def _project_module():
    try:
        import germquant.project as gp
    except ImportError as e:
        raise ProjectsMissing(_missing_message()) from e
    if not hasattr(gp, "open_project"):
        raise ProjectsMissing(_missing_message())
    return gp


def _missing_message() -> str:
    from . import _engine

    return ("The Recipe tab reads the project in this napari environment. " +
            _engine.feature_missing_message("projects"))


def state_digest(root: Path) -> str:
    """sha256 over the manifest, recipe.yaml and channel_map.yaml bytes (absent files counted as absent)."""
    h = hashlib.sha256()
    for name in (MANIFEST_NAME, RECIPE_NAME, CHANNEL_MAP_NAME):
        data = _read_bytes(Path(root) / name)
        h.update(name.encode("ascii") + (b"\x00" if data is None else b"\x01" + hashlib.sha256(data).digest()))
    return h.hexdigest()


def read_view(root, *, reference: str | None = None) -> RecipeView:
    """The project's recipe as the tab shows it, read in this environment (module docstring). ``reference``: the
    image whose channels the role combos list (default: the project's first image that is not excluded).
    ProjectsMissing when this germquant cannot read projects; the project's own refusals (not a project, a
    manifest that does not load) as they come."""
    gp = _project_module()
    from germquant import packs
    from germquant.config import load_config

    root = Path(os.path.abspath(os.fspath(root)))
    digest = state_digest(root)
    proj = gp.open_project(root)
    m = proj.manifest
    pin = m["pack"]
    ref = f"{pin['id']}@{pin['version']}"
    warnings = [str(w) for w in proj.warnings]
    pack = packs.find_pack(ref, project=root)
    recipe = pack.recipe(pin["recipe"])
    base = None
    try:
        with packs.use_project(root):
            base = load_config(packs.recipe_profile_path(pack, recipe))
    except Exception as e:  # noqa: BLE001 - the form still shows its rows, without the pack values
        warnings.append(f"the pack recipe {ref}/{recipe.id} does not load in this napari environment "
                        f"({type(e).__name__}: {e}): the form shows no pack values and the role editor no roles")
    edits, problem = recipe_edits(root, m)
    if problem:
        warnings.append(problem)
    origins = dict((m.get("recipe") or {}).get("origins") or {})
    tunables, declared = [], set()
    for decl in recipe.tunables or []:
        if not isinstance(decl, dict) or not decl.get("key"):
            continue
        key = str(decl["key"])
        declared.add(key)
        pv = base.get(key, MISSING) if base is not None else MISSING
        tunables.append(Tunable.from_declaration(decl, pack_value=pv, value=edits.get(key, pv),
                                                 origin=(origins.get(key) or "manual") if key in edits else None))
    others = [OtherEdit(k, v, str(origins.get(k) or "manual")) for k, v in sorted(edits.items()) if k not in declared]
    role_edits = dict((m.get("recipe") or {}).get("roles") or {})
    roles, match_by = [], "name"
    if base is not None:
        cm = base.channel_map
        match_by = str(getattr(cm, "match_by", "name"))
        roles = [Role(str(name), str(spec.marker), bool(spec.required), tuple(str(n) for n in spec.names),
                      spec.index, role_edits.get(name) or None) for name, spec in cm.roles.items()]
    images = [{"id": str(e["id"]), "path": str(e["path"]), "excluded": bool(e.get("excluded"))}
              for e in m.get("images") or []]
    usable = [e for e in images if not e["excluded"]]
    chosen = next((e for e in usable if e["id"] == reference), usable[0] if usable else None)
    channels, channels_error = [], None
    if chosen is not None:
        try:
            channels = header_channels(image_path(root, chosen["path"]))
        except Exception as e:  # noqa: BLE001 - an unreachable image (a NAS down) must not hide the tab
            channels_error = f"image {chosen['id']}: header unreadable ({type(e).__name__}: {e})"
    return RecipeView(root=root, name=str(m.get("name") or root.name), pack=ref, recipe=str(recipe.id),
                      pack_hash=getattr(recipe, "expected_config_hash", None), config_hash=proj.config_hash,
                      edited=bool((m.get("recipe") or {}).get("file")), validated=m.get("validated"),
                      tunables=tunables, others=others, roles=roles, match_by=match_by, images=images,
                      reference=chosen["id"] if chosen else None, channels=channels, channels_error=channels_error,
                      warnings=warnings, digest=digest,
                      saved=dict(m["saved"]) if isinstance(m.get("saved"), dict) else None)


# ---------------------------------------------------------------------------------------------------
# what the tab sends, and what it makes of the answers
# ---------------------------------------------------------------------------------------------------


def role_choices(roles, choices: dict) -> dict[str, str]:
    """The ``roles=`` of ``project edit`` for these choices: every role of the map, its exact channel name or ""
    for the pack map (the engine then binds exactly these, whatever the manifest recorded before)."""
    return {r.role: str(choices.get(r.role) or "") for r in roles}


def combo_items(role: Role, channels, reference: str | None, chosen: str | None = None) -> list[tuple[str, Any]]:
    """The items of a role's combo: the pack map (None), then the reference image's channels, then an edit or a
    choice that names no channel of the reference image (made on another image)."""
    items: list[tuple[str, Any]] = [(role.pack_text(), None)]
    items += [(f"{i}: {c}", str(c)) for i, c in enumerate(channels)]
    extra = [c for c in dict.fromkeys([role.edit, chosen]) if c and c not in channels]
    items += [(f"{c} (not a channel of {reference or 'the reference image'})", c) for c in extra]
    return items


def form_change(tunables, pending: dict) -> tuple[dict, list[str]]:
    """``(edits, unset)`` of ``project edit`` for the pending form changes: each value (checked) as an edit, each
    RESET of a key the project sets as a key to unset (a RESET of a key it does not set is nothing)."""
    by_key = {t.key: t for t in tunables}
    edits, unset = {}, []
    for key, value in pending.items():
        t = by_key.get(key)
        if t is None:
            raise ValueError(f"{key} is not a tunable of the pack recipe: the parameter form edits only its tunables")
        if value is RESET:
            if t.edited:
                unset.append(key)
        else:
            edits[key] = t.check(value)
    return edits, unset


def preview_states(record: dict | None, roles) -> dict[str, dict[str, str]]:
    """``{image_id: {role: state}}`` of a dry run's role preview: name, index, unbound (an optional role), missing
    (a required role unbound) or collision (two roles bound to one channel of that image)."""
    preview = (record or {}).get("preview") or {}
    required = {r.role: r.required for r in roles}
    out: dict[str, dict[str, str]] = {}
    for image_id, binds in (preview.get("bindings") or {}).items():
        row: dict[str, str] = {}
        seen: dict[str, list[str]] = {}
        for role, b in (binds or {}).items():
            how = str((b or {}).get("how") or "unbound")
            if how == "unbound" or (b or {}).get("channel") is None:
                row[role] = "missing" if required.get(role) else "unbound"
            else:
                row[role] = "index" if how == "index" else "name"
                seen.setdefault(str(b["channel"]), []).append(role)
        for members in seen.values():
            if len(members) > 1:
                for role in members:
                    row[role] = "collision"
        out[str(image_id)] = row
    return out


def cell_text(binding: dict | None) -> str:
    b = binding or {}
    if b.get("channel") is None:
        return "(none)"
    return f"{b['channel']} ({HOW_TEXT.get(str(b.get('how')), b.get('how'))})"


def validated_badge(validated: dict | None) -> tuple[str, str]:
    """``(text, level)`` of the validated badge (level ok or warn): "validated (record ...)", "validated: false
    (edited)", "not validated (provisional pack)" or "not validated (<the reason>)"."""
    if not isinstance(validated, dict):
        return "validated: unknown", "warn"
    why = str(validated.get("why") or "")
    if validated.get("value"):
        return f"validated (record {validated.get('record')})", "ok"
    if why.startswith("edited"):
        return "validated: false (edited)", "warn"
    if "is provisional" in why:
        return "not validated (provisional pack)", "warn"
    reason = why.split(":", 1)[1].strip() if why.startswith("not validated:") else why
    return f"not validated ({reason or 'no record'})", "warn"


def written_text(record: dict) -> str:
    """What a write did, in words: the recipe files written and removed, and the new config_hash."""
    parts = []
    files = [str(n) for n in record.get("written") or [] if str(n) != MANIFEST_NAME]
    if files:
        parts.append("wrote " + ", ".join(files))
    if record.get("removed"):
        parts.append("removed " + ", ".join(str(n) for n in record["removed"]))
    if not parts:
        parts.append("no recipe file changed")
    return "; ".join(parts) + f"; config_hash {record.get('config_hash')}"
