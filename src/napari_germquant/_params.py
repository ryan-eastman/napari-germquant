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

Post-stages (design 5.6 and 8.5; docs/POST_STAGES.md in the engine). The post-stages a run can switch on are those
installed in the environment that runs it, so their list comes from the engine runner (`python -m
germquant.post_stages list --json`, ``post_stages_from_listing``), and post_stages.yaml is written there too
(`python -m germquant.post_stages write`, which validates the entries against them; D39). The tab reads the list a
run of the project uses in this environment (``post_list``): the project's own post_stages.yaml, else the
post_stages/<recipe>.yaml its pack ships, else none. post_stages.yaml is not part of config_hash: switching a
post-stage on is no recipe edit, and changes neither the hash nor validated (design 7.3). What Apply writes holds,
for each post-stage switched on, only the parameters off their declared defaults (``post_entries``), as recipe.yaml
holds only the edits.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import numbers
import os
import re
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
# the recipe key that names the role editor's channel_map.yaml; `project edit --unset io.channel_map` drops every
# role edit (docs/PROJECTS.md), with no role preview, so no role problem refuses it
CHANNEL_MAP_KEY = "io.channel_map"
# the preview cell states, worst first: a collision and a required role unbound block Apply (red: both are the
# engine's problems); the index fallback is amber (the run flags it); an optional role unbound and a name match are
# plain
STATES = ("collision", "missing", "index", "unbound", "name")
HOW_TEXT = {"name": "by name", "index": "by index", "unbound": "unbound"}
# the post-stage list of a project (its root) and of a pack (post_stages/<recipe>.yaml), as the engine names them
# (germquant.post_stages.POST_STAGES_FILE and PACK_POST_STAGES_DIR)
POST_STAGES_NAME = "post_stages.yaml"
PACK_POST_STAGES_DIR = "post_stages"
POST_ENTRY_KEYS = ("name", "params")
UNCHANGED_TEXT = "unchanged (post_stages.yaml is not part of config_hash)"


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


@dataclasses.dataclass(frozen=True)
class PostList:
    """The post-stage list a run of the project uses (``post_list``): ``origin`` "project" (its own
    post_stages.yaml), "pack" (the post_stages/<recipe>.yaml its pack ships) or None (none in force); ``path`` the
    file; ``entries`` the entries as read ({"name", "params"} each; empty when none is in force or the file cannot
    be read); ``problem`` why the file cannot be read."""

    origin: str | None = None
    path: Path | None = None
    entries: tuple = ()
    problem: str | None = None

    def where(self) -> str:
        """The list's name in a message."""
        if self.origin == "project":
            return f"the project's {POST_STAGES_NAME}"
        if self.origin == "pack" and self.path is not None:
            return f"the pack's {PACK_POST_STAGES_DIR}/{Path(self.path).name}"
        return "no post-stage list"

    def summary(self) -> str:
        """What is in force, in words (the section's first line)."""
        if self.origin is None:
            return (f"none: the project has no {POST_STAGES_NAME}, and its pack ships no {PACK_POST_STAGES_DIR}/"
                    f"<recipe>.yaml for its recipe")
        if self.problem:
            return f"{self.where()}, which cannot be read ({self.problem})"
        return f"{self.where()}: {entries_text(self.entries)}"


@dataclasses.dataclass(frozen=True)
class PostStage:
    """A post-stage the engine environment offers (a row of its listing, `python -m germquant.post_stages list
    --json`, I12): its name, version, distribution and title, and its declared parameters as form rows
    (``params``: Tunables whose pack value is the declared default; the types are the tunables' own)."""

    name: str
    version: str | None
    dist: str | None
    title: str
    params: tuple
    reads: dict = dataclasses.field(default_factory=dict)
    writes: dict = dataclasses.field(default_factory=dict)

    @property
    def defaults(self) -> dict:
        return {p.key: p.pack_value for p in self.params}

    def param(self, key: str) -> Tunable:
        for p in self.params:
            if p.key == key:
                return p
        raise ValueError(f"post-stage {self.name} declares no parameter {key} (its parameters: "
                         f"{', '.join(p.key for p in self.params) or 'none'})")

    def label(self) -> str:
        """The switch's text: name, version and distribution."""
        return f"{self.name} {self.version or '?'}" + (f" ({self.dist})" if self.dist else "")


@dataclasses.dataclass
class RecipeView:
    """A project's recipe as the tab shows it (``read_view``). ``config_hash`` is the project's hash as this
    environment computes it on open (the engine's dry run confirms it), ``validated`` the manifest's record,
    ``digest`` the sha256 of the manifest and the two recipe files (a preview made for another digest is
    stale), ``saved`` the manifest's saved stamp (every write changes it), ``overrides`` the stage switches the
    project turns off (the manifest's recipe overrides, ``{"spots.enabled": False}``: they enter the hash),
    ``manifest`` the manifest as read (a host's later reading equal to it holds no write since: the saved stamp
    alone is to the second, so two writes in one second share it), and ``post`` the post-stage list a run of the
    project uses (not part of the hash, nor of ``digest``: a post-stage write leaves every preview current)."""

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
    overrides: dict = dataclasses.field(default_factory=dict)
    manifest: dict = dataclasses.field(default_factory=dict)
    post: PostList = dataclasses.field(default_factory=PostList)

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
    if rec.get("roles") and flat.get(CHANNEL_MAP_KEY) == CHANNEL_MAP_NAME:
        flat.pop(CHANNEL_MAP_KEY)
    return flat, None


def image_path(root: Path, stored: str) -> Path:
    """The file a manifest path names: absolute (of any system, UNC included) as it is, relative under root."""
    s = str(stored)
    if PureWindowsPath(s).is_absolute() or PurePosixPath(s).is_absolute():
        return Path(s)
    return Path(root) / PurePosixPath(s)


def header_channels(path: Path) -> list[str]:
    """The channel names of an image, read with the engine's light header reader (germquant.project.headers, the
    reader the engine's role preview uses; no bioio, D28). read_view calls it only after germquant.project has
    imported, which ships with headers."""
    from germquant.project.headers import read_header

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
    """The project's recipe as the tab shows it, read in this environment (module docstring), with the post-stage
    list its runs use (``post_list``: its post_stages.yaml, else its pack's for the recipe). ``reference``: the
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
                      saved=dict(m["saved"]) if isinstance(m.get("saved"), dict) else None,
                      overrides=dict((m.get("recipe") or {}).get("overrides") or {}), manifest=copy.deepcopy(m),
                      post=post_list(root, getattr(pack, "folder", None), str(recipe.id)))


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


# the engine's problem for two roles on one channel (project/recipe.py bindings): "image <id>: the roles <a> and
# <b> bind the same channel <index> (<name>)"
_COLLISION = re.compile(r"the roles (?P<a>.+?) and (?P<b>.+?) bind the same channel \d")


def collision_roles(problems, image_id) -> set[str]:
    """The roles that the engine's problems say bind one channel of ``image_id`` (it judges by channel index, so
    two channels that share a name are no collision)."""
    prefix = f"image {image_id}: "
    out: set[str] = set()
    for p in problems or ():
        text = str(p)
        if text.startswith(prefix):
            m = _COLLISION.match(text, len(prefix))
            if m:
                out.update(m.group("a", "b"))
    return out


def preview_states(record: dict | None, roles) -> dict[str, dict[str, str]]:
    """``{image_id: {role: state}}`` of a dry run's role preview: name, index, unbound (an optional role), missing
    (a required role unbound) or collision (two roles on one channel of that image, as the engine's problems name
    them: never judged here by channel names, which two channels may share)."""
    preview = (record or {}).get("preview") or {}
    problems = [str(p) for p in preview.get("problems") or []]
    required = {r.role: r.required for r in roles}
    out: dict[str, dict[str, str]] = {}
    for image_id, binds in (preview.get("bindings") or {}).items():
        row: dict[str, str] = {}
        for role, b in (binds or {}).items():
            how = str((b or {}).get("how") or "unbound")
            if how == "unbound" or (b or {}).get("channel") is None:
                row[role] = "missing" if required.get(role) else "unbound"
            else:
                row[role] = "index" if how == "index" else "name"
        for role in collision_roles(problems, image_id):
            if role in row:
                row[role] = "collision"
        out[str(image_id)] = row
    return out


def cell_text(binding: dict | None) -> str:
    b = binding or {}
    if b.get("channel") is None:
        return "(none)"
    return f"{b['channel']} ({HOW_TEXT.get(str(b.get('how')), b.get('how'))})"


def hash_note(view: RecipeView) -> str:
    """What the project's config_hash is the hash of, for the hash line: the edited recipe (recipe.yaml) or the
    pack recipe, and the stage switches the project turns off (Run tab, ``--no-<stage>``), which enter the hash
    too; any other override is named with its value."""
    over = dict(view.overrides or {})
    switch = {k: v for k, v in over.items() if str(k).endswith(".enabled") and v is False}
    off = sorted(str(k)[:-len(".enabled")] for k in switch)
    other = sorted(f"{k}={value_text(v)}" for k, v in over.items() if k not in switch)
    parts = ["edited" if view.edited else "the pack recipe"]
    if off:
        parts.append(f"with the stage switch{'es' if len(off) > 1 else ''} {', '.join(off)} off")
    if other:
        parts.append("with the overrides " + ", ".join(other))
    return ", ".join(parts)


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


# ---------------------------------------------------------------------------------------------------
# post-stages: the list in force, the engine environment's post-stages, and what Apply writes
# ---------------------------------------------------------------------------------------------------


def _entries_problems(doc) -> list[str]:
    """The shape problems of a post-stage list, in the engine's words (germquant.post_stages.config's
    entries_problems): a list of mappings, each with a name and, optionally, params."""
    if isinstance(doc, (str, bytes)) or not isinstance(doc, (list, tuple)):
        return [f"post-stages must be a list of {{name, params}} entries, got {type(doc).__name__}"]
    out = []
    for i, e in enumerate(doc):
        if not isinstance(e, dict):
            out.append(f"entry {i} must be a mapping {{name, params}}, got {type(e).__name__}")
            continue
        unknown = sorted(str(k) for k in e if k not in POST_ENTRY_KEYS)
        if unknown:
            out.append(f"entry {i}: unknown key(s) {unknown} (an entry takes name and params)")
        if "name" not in e:
            out.append(f"entry {i} has no name")
        elif not isinstance(e["name"], str) or not e["name"].strip():
            out.append(f"entry {i}: name must be the post-stage's name as text, got {e['name']!r}")
        params = e.get("params")
        if params is not None and not isinstance(params, dict):
            out.append(f"entry {i}: params must be a mapping of parameter keys to values, got {type(params).__name__}")
    return out


def read_post_entries(path) -> tuple[list[dict] | None, str | None]:
    """``(entries, problem)`` of a post_stages.yaml, read as the engine's load_post_stages_file reads it: UTF-8 with
    or without a byte-order mark, YAML, an empty file an empty list, a list of {name, params}. ``entries`` is None
    when the file cannot be read, and ``problem`` says why. The names and values are not checked here: only the
    environment that runs has the post-stages to check them against (post_state names what its listing lacks)."""
    import yaml

    p = Path(path)
    try:
        data = _read_bytes(p)
    except OSError as e:
        return None, f"{p} cannot be opened ({e.strerror or e})"
    if data is None:
        return None, f"{p} is missing"
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        return None, f"{p} is not UTF-8 text (byte {e.start})"
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        return None, f"{p} cannot be read as YAML" + (f" (line {mark.line + 1})" if mark is not None else "")
    if doc is None:
        return [], None
    problems = _entries_problems(doc)
    if problems:
        return None, f"{p}: " + "; ".join(problems)
    return [{"name": str(e["name"]), "params": dict(e.get("params") or {})} for e in doc], None


def _is_file(path) -> bool:
    try:
        return os.path.isfile(_long(path))
    except OSError:
        return False


def post_list(root, pack_folder=None, recipe: str | None = None) -> PostList:
    """The post-stage list a run of the project in ``root`` uses (the engine's post_stages_in_force, which project
    run follows): the project's own post_stages.yaml; else the post_stages/<recipe>.yaml that the pack in
    ``pack_folder`` ships for ``recipe``; else none. A project's file replaces the pack's, an empty one included."""
    own = Path(root) / POST_STAGES_NAME
    if _is_file(own):
        entries, problem = read_post_entries(own)
        return PostList("project", own, tuple(entries or ()), problem)
    if pack_folder is not None and recipe:
        shipped = Path(pack_folder) / PACK_POST_STAGES_DIR / f"{recipe}.yaml"
        if _is_file(shipped):
            entries, problem = read_post_entries(shipped)
            return PostList("pack", shipped, tuple(entries or ()), problem)
    return PostList()


def post_stages_from_listing(rows) -> list[PostStage]:
    """The post-stages of an engine environment's listing (`python -m germquant.post_stages list --json`, or
    api.list_post_stages in process: [{"name", "version", "dist", "title", "reads", "writes", "params",
    "source"}]), in its order (by name); a row without a name is left out."""
    out = []
    for r in rows or ():
        if not isinstance(r, dict) or not r.get("name"):
            continue
        params = []
        for spec in r.get("params") or ():
            if isinstance(spec, dict) and spec.get("key"):
                default = spec.get("default", MISSING)
                params.append(Tunable.from_declaration(spec, pack_value=default, value=default))
        out.append(PostStage(name=str(r["name"]), version=str(r["version"]) if r.get("version") else None,
                             dist=str(r["dist"]) if r.get("dist") else None, title=str(r.get("title") or ""),
                             params=tuple(params), reads=dict(r.get("reads") or {}),
                             writes=dict(r.get("writes") or {})))
    return out


def post_value_problem(spec: Tunable, value) -> str | None:
    """Why ``value`` cannot be the value of the declared post-stage parameter ``spec``, in the engine's words (its
    param_problem: a float takes any finite number but true or false; an int an integer, never a float; a choice
    one of its choices of the same type; ranges inclusive), or None when it can."""
    kind, key = spec.type, spec.key
    is_num = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if kind == "float":
        if not is_num or not math.isfinite(float(value)):
            return f"parameter {key} must be a number, got {value!r}"
    elif kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"parameter {key} must be an integer, got {value!r}"
    elif kind == "bool":
        if not isinstance(value, bool):
            return f"parameter {key} must be true or false, got {value!r}"
    elif kind == "choice":
        if not any(type(c) is type(value) and c == value for c in spec.choices):
            return f"parameter {key} must be one of {list(spec.choices)}, got {value!r}"
    elif kind == "str":
        if not isinstance(value, str):
            return f"parameter {key} must be text, got {value!r}"
    else:
        return f"parameter {key} has the unknown type {kind!r}"
    if kind in ("float", "int"):
        if spec.lo is not None and value < spec.lo:
            return f"parameter {key} must be at least {spec.lo}, got {value!r}"
        if spec.hi is not None and value > spec.hi:
            return f"parameter {key} must be at most {spec.hi}, got {value!r}"
    return None


def post_same(spec: Tunable, a, b) -> bool:
    """Whether a post-stage parameter's two values give a run the same value (the engine stores a float parameter
    as a float, so 1 equals 1.0 there; any other type is compared with its type)."""
    if a is MISSING or b is MISSING:
        return a is b
    nums = all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in (a, b))
    if spec.type == "float" and nums:
        return float(a) == float(b)
    return type(a) is type(b) and a == b


def post_state(stages, plist: PostList) -> tuple[dict, list[str]]:
    """``({name: {"on": bool, "values": {key: value}}}, problems)``: what the form shows of the list in force, for
    each post-stage of the listing ``stages`` (on when the list names it; each parameter the list's value, else the
    default). What the engine would refuse in a run of the list, or cannot be shown, is a problem, each saying what
    Apply writes instead: a list that cannot be read, a post-stage the engine environment does not offer, one
    listed twice, an undeclared parameter, a value outside its declaration (the form then shows the default)."""
    by = {s.name: s for s in stages}
    state = {s.name: {"on": False, "values": dict(s.defaults)} for s in stages}
    where = plist.where()
    if plist.problem:
        return state, [f"{where} cannot be read ({plist.problem}): a run of the project is refused while it is so, "
                       f"and Apply writes a new {POST_STAGES_NAME}"]
    problems: list[str] = []
    seen: set[str] = set()
    for e in plist.entries:
        name = str(e.get("name"))
        if name in seen:
            problems.append(f"{where} lists {name} more than once: Apply writes it once")
            continue
        seen.add(name)
        stage = by.get(name)
        if stage is None:
            problems.append(f"{where} switches on {name}, which the engine environment does not offer: a run of the "
                            f"project is refused while the list names it, and Apply writes the list without it")
            continue
        state[name]["on"] = True
        for key, value in dict(e.get("params") or {}).items():
            try:
                spec = stage.param(str(key))
            except ValueError:
                problems.append(f"{where}: {name} declares no parameter {key}: Apply leaves it out")
                continue
            why = post_value_problem(spec, value)
            if why:
                problems.append(f"{where}: {name}: {why}: the form shows the default {value_text(spec.pack_value)}, "
                                f"which Apply writes")
                continue
            state[name]["values"][spec.key] = float(value) if spec.type == "float" else value
    return state, problems


def post_entries(stages, on: dict, values: dict) -> list[dict]:
    """The post_stages.yaml Apply writes: each post-stage of ``stages`` that ``on`` switches on, in the listing's
    order, with the parameters of ``values`` that are off their declared defaults (post_stages.yaml then holds
    only what was changed, as recipe.yaml holds only the edits). An empty list switches every post-stage off."""
    out = []
    for s in stages:
        if not on.get(s.name):
            continue
        given = values.get(s.name) or {}
        params = {}
        for p in s.params:
            v = given.get(p.key, p.pack_value)
            if v is MISSING or post_same(p, v, p.pack_value):
                continue
            params[p.key] = v
        out.append({"name": s.name, "params": params})
    return out


def post_list_changed(stages, plist: PostList, entries) -> bool:
    """Whether writing ``entries`` changes what a run gets from the list in force: which post-stages are on, and
    each of their parameters' values with the defaults filled in, whatever the order of the entries or a default
    written out. A list in force with a problem (post_state) is always a change: Apply writes one a run takes."""
    state, problems = post_state(stages, plist)
    if problems:
        return True
    by = {s.name: s for s in stages}
    current = {name: st["values"] for name, st in state.items() if st["on"]}
    wanted = {}
    for e in entries:
        s = by.get(str(e.get("name")))
        if s is None:
            return True
        wanted[s.name] = {**s.defaults, **dict(e.get("params") or {})}
    if current.keys() != wanted.keys():
        return True
    return any(not post_same(p, wanted[name].get(p.key, MISSING), current[name].get(p.key, MISSING))
               for name in wanted for p in by[name].params)


def entries_text(entries) -> str:
    """A post-stage list in words: "fixture_nuclei (factor 2.5, statistic max)", "ring_intensity (its defaults)",
    or "every post-stage off" for an empty list."""
    parts = []
    for e in entries or ():
        params = dict(e.get("params") or {})
        inner = ", ".join(f"{k} {value_text(v)}" for k, v in params.items()) or "its defaults"
        parts.append(f"{e.get('name')} ({inner})")
    return "; ".join(parts) or "every post-stage off"
