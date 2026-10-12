"""The pack picker's data: entries of an engine's pack listing and the text of a pack card. No Qt.

The listing is what ``python -m germquant.packs list --json --all`` prints in the engine environment (or the
same document built in process): one row per pack with its recipes, models, licence and citation, the state
(ok, unusable or shadowed) and, for a broken pack, its problems. A recipe's availability comes from
``pack check --json`` (germquant.packs.pack_status): usable, blockers and warnings, each with its fix, and the
state of every model the recipe needs. The card (docs/DESIGN_system_packs.md, section 8.2) shows the title,
system, object kind, dims, version and engine range; the models with their state; the backends and modules,
available or missing with the extra to install; the validation status with dataset and metric, or "not
validated"; the licence and citation; and the example a "Get example" button fetches.

Two things the listing leaves out are read from the pack folder the engine reported, as data
(yaml.safe_load and json, never code): the examples the pack declares (pack.yaml) and a recipe's validation
record (validation/<recipe>.json).
"""
from __future__ import annotations

import dataclasses
import html
import json
from pathlib import Path

import yaml

# states of a model in pack check: usable as they are, or a reason the recipe cannot run
MODEL_OK = ("trusted", "verified", "unverified")
MODEL_BAD = ("missing", "wrong_size", "mismatch")

RED = "#c83c3c"
AMBER = "#c8901e"
GREEN = "#3c8c50"
GREY = "#808080"

OBJECT_WORDS = {"nucleus": ("nucleus", "nuclei"), "cell": ("cell", "cells")}


def object_words(kind: str | None) -> tuple[str, str]:
    """``(singular, plural)`` for a pack's object_kind (one row of the nuclei table)."""
    return OBJECT_WORDS.get(str(kind or "nucleus"), (str(kind), f"{kind}s"))


@dataclasses.dataclass
class PackEntry:
    """One row of an engine's pack listing."""

    data: dict

    @property
    def id(self) -> str:
        return str(self.data.get("id", ""))

    @property
    def version(self) -> str:
        return str(self.data.get("version") or "")

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def title(self) -> str:
        return str(self.data.get("title") or self.id)

    @property
    def state(self) -> str:
        return str(self.data.get("state", "ok"))

    @property
    def usable_manifest(self) -> bool:
        return self.state == "ok" and not self.problems

    @property
    def problems(self) -> list[str]:
        return [str(p) for p in self.data.get("problems") or []]

    @property
    def folder(self) -> Path | None:
        f = self.data.get("folder")
        return Path(f) if f else None

    @property
    def object_kind(self) -> str:
        return str(self.data.get("object_kind") or "nucleus")

    @property
    def recipes(self) -> list[dict]:
        return [r for r in self.data.get("recipes") or [] if isinstance(r, dict)]

    @property
    def models(self) -> list[dict]:
        return [m for m in self.data.get("models") or [] if isinstance(m, dict)]

    @property
    def engine_range(self) -> str:
        return str((self.data.get("requires") or {}).get("germquant") or "any")

    @property
    def label(self) -> str:
        tail = "" if self.state == "ok" else f"  [{self.state}]"
        return f"{self.id} {self.version}: {self.title}{tail}"

    def recipe(self, recipe_id: str | None) -> dict | None:
        for r in self.recipes:
            if r.get("id") == recipe_id:
                return r
        return None

    def default_recipe_id(self) -> str | None:
        for r in self.recipes:
            if r.get("default"):
                return str(r.get("id"))
        return str(self.recipes[0].get("id")) if self.recipes else None

    def profile_path(self, recipe_id: str) -> Path | None:
        """The recipe's profile file as the engine sees it: inside the pack (root: pack), or under the engine
        checkout the built-in pack lives in (root: checkout)."""
        r = self.recipe(recipe_id)
        if r is None or self.folder is None or not r.get("profile"):
            return None
        if str(self.data.get("root", "pack")) == "checkout":
            if self.data.get("source") != "builtin":
                return None                        # a vendored checkout pack: the engine resolves it, not us
            return self.folder.parent.parent / str(r["profile"])
        return self.folder / str(r["profile"])

    def manifest(self) -> dict:
        """The pack's pack.yaml, read as data (empty when it cannot be read)."""
        if self.folder is None:
            return {}
        try:
            data = yaml.safe_load((self.folder / "pack.yaml").read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}

    def examples(self) -> list[dict]:
        """The examples the pack declares, each with ``kind`` remote (a public image fetched on demand) or
        local (inside the pack), remote first (the one `fetch-example` takes by default)."""
        out = []
        for ex in self.manifest().get("examples") or []:
            if not isinstance(ex, dict) or not ex.get("id"):
                continue
            rec = dict(ex)
            rec["kind"] = "remote" if ex.get("base_url") and ex.get("sources") else "local"
            out.append(rec)
        return sorted(out, key=lambda e: e["kind"] != "remote")

    def validation_record(self, recipe_id: str) -> dict | None:
        """The recipe's validation record (validation/<recipe>.json), when the recipe names one and it reads."""
        r = self.recipe(recipe_id) or {}
        rel = r.get("validation")
        if not rel or self.folder is None:
            return None
        try:
            data = json.loads((self.folder / str(rel)).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


def listing_entries(doc: dict | None) -> list[PackEntry]:
    """The pack rows of a listing document, usable packs first, in the engine's search order."""
    rows = [PackEntry(r) for r in (doc or {}).get("packs") or [] if isinstance(r, dict) and r.get("id")]
    return sorted(rows, key=lambda e: e.state != "ok")


def pack_folders(doc: dict | None) -> list[Path]:
    """Every pack folder and pack root an engine listing names: an output folder must not lie inside one."""
    out = [Path(r["path"]) for r in (doc or {}).get("roots") or [] if isinstance(r, dict) and r.get("path")]
    out += [e.folder for e in listing_entries(doc) if e.folder is not None]
    return out


def inside(path: str | Path, folder: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except (ValueError, OSError):
        return False


def manifest_status(entry: PackEntry) -> dict:
    """The status of a pack whose manifest is unusable (the listing's problems, no check needed)."""
    blockers = [{"what": f"pack manifest: {p}", "fix": "use a corrected pack version (germquant pack list --all "
                                                         "names the problem)"} for p in entry.problems]
    if not blockers:
        blockers = [{"what": f"pack {entry.ref} is listed as {entry.state}", "fix": "germquant pack list --all"}]
    return {"pack": entry.id, "version": entry.version, "recipe": None, "usable": False, "blockers": blockers,
            "warnings": [], "models": []}


def models_need_fetch(status: dict | None) -> bool:
    return any(m.get("state") in MODEL_BAD for m in (status or {}).get("models") or [])


def need_rows(recipe: dict | None, status: dict | None) -> list[dict]:
    """The recipe's backends and modules: ``{"what", "name", "available", "fix"}`` (available None while no
    check has run). A need is missing when a blocker of the check names it."""
    needs = (recipe or {}).get("needs") or {}
    blockers = (status or {}).get("blockers") or []

    def blocker(phrase: str, name: str) -> dict | None:
        return next((b for b in blockers if f"{phrase} '{name}'" in str(b.get("what", ""))), None)

    rows = []
    for key, what, phrase in (("segment_backends", "segmentation backend", "segmentation backend"),
                              ("spot_backends", "spot backend", "spot backend"),
                              ("modules", "Python module", "Python module")):
        for name in needs.get(key) or []:
            hit = blocker(phrase, str(name))
            if hit is None and key != "modules":
                # a built-in backend name is always declared; its Python module (cellpose, spotmax) is what
                # the environment may lack, and then the backend cannot run either
                hit = blocker("Python module", str(name))
            rows.append({"what": what, "name": str(name),
                         "available": None if status is None else hit is None,
                         "fix": str(hit.get("fix", "")) if hit else ""})
    return rows


def _e(x) -> str:
    return html.escape(str(x))


def _mb(n) -> str:
    try:
        return f"{int(n) / 1e6:.1f} MB"
    except (TypeError, ValueError):
        return "size unknown"


def validation_unreadable(entry: PackEntry, recipe_id: str) -> bool:
    """The recipe names a validation record that cannot be read (its status then has nothing behind it)."""
    r = entry.recipe(recipe_id) or {}
    return bool(r.get("validation")) and entry.validation_record(recipe_id) is None


def validation_text(entry: PackEntry, recipe_id: str) -> str:
    """'not validated', or the validation record's dataset and first metric (section 3.6: a record written by
    ``germquant extbench report --final``; its exact keys are settled by package P7, so the dataset is read as
    ``dataset.id`` or ``dataset_id`` and a metric as a number or ``{value or estimate, ci_low or lo, ci_high or
    hi}``)."""
    r = entry.recipe(recipe_id) or {}
    rec = entry.validation_record(recipe_id)
    status = str(r.get("status") or "provisional")
    if rec is None:
        if r.get("validation"):
            return f"status {status}, but its validation record {r.get('validation')} cannot be read"
        return f"not validated (status {status})"
    ds = rec.get("dataset")
    dataset = ds.get("id") if isinstance(ds, dict) else (ds or rec.get("dataset_id") or "dataset not named")
    metrics = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
    metric = ""
    if metrics:
        name, value = next(iter(metrics.items()))
        if isinstance(value, dict):
            point = value.get("value", value.get("estimate"))
            lo, hi = value.get("ci_low", value.get("lo")), value.get("ci_high", value.get("hi"))
            metric = f"{name} {point}" + (f" (95% interval {lo} to {hi})" if lo is not None and hi is not None else "")
        else:
            metric = f"{name} {value}"
    gate = rec.get("gate") if isinstance(rec.get("gate"), dict) else {}
    verdict = rec.get("verdict") or gate.get("verdict")
    return f"{status} on {dataset}" + (f": {metric}" if metric else "") + (f"; verdict {verdict}" if verdict else "")


def card_html(entry: PackEntry, recipe_id: str | None, status: dict | None, *, engine_text: str = "",
              pending: bool = False) -> str:
    """The pack card as rich text for a QLabel (``pending``: its check is running in ``engine_text``)."""
    d = entry.data
    r = entry.recipe(recipe_id) or {}
    sysd = d.get("system") or {}
    singular, _plural = object_words(entry.object_kind)
    out = [f"<b>{_e(entry.title)}</b> &nbsp;({_e(entry.id)} {_e(entry.version)})"]
    if d.get("summary"):
        out.append(_e(d["summary"]))
    if isinstance(sysd, dict) and sysd:
        out.append("system: " + "; ".join(f"{_e(k)} {_e(v)}" for k, v in sysd.items()))
    out.append(f"object kind: {_e(entry.object_kind)} (one row of the nuclei table is one {_e(singular)})")
    if r:
        out.append(f"recipe {_e(r.get('id'))}: {_e(r.get('title', ''))}; dims {_e(r.get('dims', '3d'))}; "
                   f"config_hash {_e(r.get('expected_config_hash', ''))}")
    out.append(f"version {_e(entry.version)}; engine range {_e(entry.engine_range)}; found: "
               f"{_e(d.get('source', ''))} {_e(d.get('folder', ''))}")
    # models
    need = [str(m) for m in ((r.get("needs") or {}).get("models") or [])]
    states = {m.get("name"): m for m in (status or {}).get("models") or []}
    specs = {m.get("name"): m for m in entry.models}
    if need:
        parts = []
        for name in need:
            spec, st = specs.get(name, {}), states.get(name)
            state = st.get("state") if st else "not checked"
            color = RED if state in MODEL_BAD else (GREEN if state in MODEL_OK else GREY)
            detail = f" ({_e(st.get('detail'))})" if st and st.get("detail") else ""
            lic = f"; licence {_e(spec.get('licence'))}" if spec.get("licence") else ""
            parts.append(f"{_e(name)}: <span style='color:{color}'>{_e(state)}</span>{detail}, "
                         f"{_mb(spec.get('size'))}{lic}")
        out.append("<b>Models</b>: " + "<br>&nbsp;&nbsp;".join(parts))
    else:
        out.append("<b>Models</b>: none needed")
    # backends and modules
    rows = need_rows(r, status)
    if rows:
        parts = []
        for row in rows:
            if row["available"] is None:
                state = f"<span style='color:{GREY}'>not checked</span>"
            elif row["available"]:
                state = f"<span style='color:{GREEN}'>available</span>"
            else:
                state = f"<span style='color:{RED}'>missing</span>: {_e(row['fix'])}"
            parts.append(f"{_e(row['what'])} {_e(row['name'])}: {state}")
        out.append("<b>Backends</b>: " + "<br>&nbsp;&nbsp;".join(parts))
        if any(row["available"] is False for row in rows):
            out.append("<i>Install what is missing into the engine's environment, or run this recipe in another "
                       "environment that has it (Recipe interpreter...).</i>")
    # validation, licence, citation, example
    if recipe_id:
        text = _e(validation_text(entry, recipe_id))
        if validation_unreadable(entry, recipe_id):
            text = f"<span style='color:{AMBER}'>{text}</span>"
        out.append(f"<b>Validation</b>: {text}")
    lic = [f"pack {_e(d.get('licence') or 'not stated')}"]
    lic += [f"{_e(m.get('name'))}: {_e(m.get('licence'))}" for m in entry.models if m.get("licence")]
    out.append("<b>Licence</b>: " + "; ".join(lic))
    cit = d.get("citation") or {}
    if isinstance(cit, dict) and (cit.get("text") or cit.get("doi")):
        doi = f" (doi {_e(cit.get('doi'))})" if cit.get("doi") else ""
        out.append(f"<b>Cite</b>: {_e(cit.get('text', ''))}{doi}")
    exs = entry.examples()
    if exs:
        ex = exs[0]
        if ex["kind"] == "remote":
            out.append(f"<b>Example</b>: {_e(ex['id'])}, public data fetched on demand; licence "
                       f"{_e(ex.get('licence', 'not stated'))}; {_e(ex.get('attribution', ''))}")
        else:
            out.append(f"<b>Example</b>: {_e(ex['id'])}, a small synthetic image inside the pack")
    else:
        out.append("<b>Example</b>: none")
    # availability
    if pending:
        out.append(f"<span style='color:{GREY}'>checking in {_e(engine_text)} ...</span>")
        return "<br>".join(out)
    if engine_text:
        out.append(f"checked in: {_e(engine_text)}")
    if status is None:
        out.append(f"<span style='color:{GREY}'>not checked yet</span>")
    elif status.get("usable"):
        out.append(f"<b><span style='color:{GREEN}'>usable here</span></b>")
    else:
        lines = [f"<b><span style='color:{RED}'>NOT usable here</span></b>"]
        for b in status.get("blockers") or []:
            lines.append(f"<span style='color:{RED}'>{_e(b.get('what'))}</span><br>&nbsp;&nbsp;fix: "
                         f"{_e(b.get('fix'))}")
        out.append("<br>".join(lines))
    for w in (status or {}).get("warnings") or []:
        out.append(f"<span style='color:{AMBER}'>warning: {_e(w.get('what'))}</span> ({_e(w.get('fix'))})")
    return "<br>".join(out)


def blocker_text(status: dict | None) -> str:
    """Plain text of a check's blockers (for tooltips and the status line)."""
    if not status:
        return ""
    return "\n".join(f"{b.get('what')}\n  fix: {b.get('fix')}" for b in status.get("blockers") or [])
