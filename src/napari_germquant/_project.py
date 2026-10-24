"""The Project dock's model: no Qt, so every rule here is tested headless (docs/DESIGN_system_packs.md, sections 7
and 8.4; the tranche-5 Train B plan, unit P6-PROJECT).

Where things come from (the train's decision D39): every write that can change a project's config_hash or its
manifest goes through the engine runner, the code that also runs the project (``project new``, ``add``, ``edit``,
``unlock``; the dock's Save is one ``project edit``, D42), and only the engine environment answers ``project
status`` and ``project run``. The plugin reads the project's own files in process: the manifest
(germquant_project.yaml), the image headers, the run logs, the results and the sample sheet. Nothing here writes a
project file.

* The status board is the engine's status record (``germquant project status --json``, which equals
  api.project_status; PLAN section 9.3): ``board_rows`` turns it into one row per image, with the objects and
  spots of each finished image counted from its results folder (a table is read again only when a run rewrote
  it). While a run goes, ``apply_events`` lays the
  events of the run's own log over the rows (run_start marks the images it will run as waiting; image_done,
  image_failed and image_skipped land as they come); when the run ends the dock asks for a fresh status record.
* ``header_text`` is the Images tab's header summary (sizes, the voxel line, the channel names), from the light
  header read of germquant.project.headers (``read_header``, in process) or, with an engine before it,
  germquant.api.read_image_metadata. A path that cannot be reached (a NAS share without the VPN) is named as such
  (``probe_image``) and never stops the rest.
* ``badge``, ``lock_text`` and ``setting_notes`` are the header's words for validation, a held lock and what the
  project records differently from this computer (device, runner, interpreter: the computer's choice wins at run
  time, the project's record is a provenance note).
* ``recorded_settings`` and ``settings_changes`` are what Save writes: only what differs from the manifest, as
  ``project edit`` options (``--device``, ``--runner``, ``--interpreter RECIPE=PYTHON``, ``--xy-stride``,
  ``--no-<stage>``, and ``--unset`` of a stage key the project had turned off).
* ``SampleTable`` is the Samples tab: one row per image, the design columns of design 7.4 plus any extra column,
  read from the project's sheet or prefilled from the file name with the recipe's metadata.filename_regex
  (germquant's parse_sample, only where the rule matches). The sheet itself is written by the engine's own writer,
  germquant.io.sample_sheet.write_sample_sheet (ASCII, LF, atomic), and recorded with ``project edit --samples``.
* ``mark_running`` and ``run_in_progress`` tell every dock (Training's Use in project, Spot tuning's Apply, the
  Recipe tab) that a run of a project is in progress, so no recipe edit starts under it.
"""
from __future__ import annotations

# germquant.api first (through _engine): the torch-before-CuPy import order of the workstation
from . import _engine  # noqa: F401,I001

import copy
import csv
import dataclasses
import io
import math
import os
import re
import threading
from pathlib import Path, PurePosixPath

import yaml

from ._reader import header_spacing_text

MANIFEST_NAME = "germquant_project.yaml"
LOCK_NAME = "germquant_project.lock"
SAMPLES_NAME = "samples.csv"
# the record germquant.timelapse writes next to one t<NNN>/ folder per timepoint (an image with timepoints)
SERIES_RECORD_SUFFIX = "__series.json"
# the design columns of a sample sheet (design 7.4), in the order the tab shows them
DESIGN_COLUMNS = ("image_id", "genotype", "sex", "germ_cell", "treatment", "replicate", "acquisition_date")
# an extra column's name: what a CSV header and a batch table column can both carry
COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# the stale reasons of the status record (PLAN 9.3 reason_code), in words, for a row without the engine's own
REASONS = {
    "config_hash": "the recipe changed (config hash)",
    "stages": "the stages changed",
    "geometry": "the run geometry changed (xy stride or z range)",
    "planar_flag": "the single-plane setting changed",
    "segment_hash": "the segmentation input changed (model or labels file)",
    "sample_row": "its sample sheet row changed",
    "failed_stages": "some stages failed last time",
}
UNREACHABLE = "not reachable: is the NAS or VPN connected?"
NOT_FOUND = "file not found"
HEADER_PENDING = "header not read yet"
# the statuses of the status record, plus the board's own: waiting (in a run, not reported yet), unknown (no
# status record yet: the engine has not answered, or lacks projects)
STATUSES = ("done", "stale", "failed", "missing", "excluded", "unreachable", "waiting", "unknown")


class ProjectFileError(ValueError):
    """A folder that is no project, or a manifest that cannot be read; the message says which and why."""


def _long(path) -> str:
    from germquant.fsutil import long_path

    return long_path(path)


# ---------------------------------------------------------------------------------------------------
# the manifest and the images, read in process
# ---------------------------------------------------------------------------------------------------


def manifest_path(root) -> Path:
    return Path(root) / MANIFEST_NAME


def is_project(root) -> bool:
    """Whether ``root`` holds a project manifest (one stat; call it in a worker for a network path)."""
    try:
        return os.path.isfile(_long(manifest_path(root)))
    except (OSError, ValueError):
        return False


def read_manifest(root) -> dict:
    """germquant_project.yaml of ``root`` (format 1, design 7.2), read once and closed at once (an open handle
    would make the engine's os.replace of it fail on Windows). ProjectFileError for a folder that is no project, a
    file that is not YAML or not a mapping, and another format."""
    p = manifest_path(root)
    try:
        with open(_long(p), "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        raise ProjectFileError(f"{root} is not a germquant project (it holds no {MANIFEST_NAME})") from None
    except OSError as e:
        raise ProjectFileError(f"{p} cannot be read: {e}") from e
    try:
        m = yaml.safe_load(data.decode("utf-8", errors="replace"))
    except yaml.YAMLError as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        raise ProjectFileError(f"{p} is not YAML ({first})") from e
    if not isinstance(m, dict):
        raise ProjectFileError(f"{p} is not a project manifest (not a mapping)")
    if m.get("format") != 1:
        raise ProjectFileError(f"{p} is format {m.get('format')}; this plugin reads format 1: update napari-germquant")
    for key, empty in (("images", []), ("models", []), ("recipe", {}), ("run", {}), ("pack", {})):
        if not isinstance(m.get(key), type(empty)):
            m[key] = copy.copy(empty)
    return m


def _is_unc(text: str) -> bool:
    return text.startswith(("\\\\", "//"))


def resolve_image(root, path) -> Path:
    """An image entry's path: inside the project it is relative (POSIX), any other is absolute and taken as it is
    (a UNC path verbatim)."""
    text = str(path)
    p = Path(text)
    if _is_unc(text) or p.is_absolute():
        return p
    return Path(root) / PurePosixPath(text)


def _remote(path: Path) -> bool:
    """Whether a missing file sits on a share that is not there (a UNC path, or a drive or mount root that does
    not answer): a NAS without the VPN, rather than a file that was moved."""
    text = str(path)
    if _is_unc(text):
        return True
    anchor = path.anchor
    try:
        return bool(anchor) and not os.path.exists(anchor)
    except (OSError, ValueError):
        return True


def read_header(path) -> dict:
    """An image's header, in process: germquant.project.headers.read_header (the light read of the Train B engine:
    sizes, the OME image count, channel names, single plane, the yx spacing) when this germquant has it, else
    germquant.api.read_image_metadata (whose ``n_scenes`` becomes ``n_images``). Raises what the reader raises."""
    try:
        from germquant.project.headers import read_header as light
    except ImportError:
        light = None
    if light is not None:
        return dict(light(path))
    from germquant import api

    meta = dict(api.read_image_metadata(path))
    meta.setdefault("n_images", int(meta.get("n_scenes", 1) or 1))
    return meta


def probe_image(root, entry: dict) -> dict:
    """The header of one manifest image, or ``{"error": ...}``: UNREACHABLE for a share that is not there,
    NOT_FOUND for a missing file, or why the header cannot be read. Blocks on a network path: call it in a
    worker."""
    path = resolve_image(root, entry.get("path", ""))
    try:
        present = os.path.exists(_long(path))
    except (OSError, ValueError):
        present = False
    if not present:
        return {"error": UNREACHABLE if _remote(path) else NOT_FOUND, "reachable": False}
    try:
        return read_header(path)
    except Exception as e:  # noqa: BLE001 - the reader's refusal is what the tab shows
        return {"error": f"cannot read the header: {type(e).__name__}: {e}"}


def header_text(header: dict | None) -> str:
    """The Images tab's header summary: the sizes, the voxel line (``_reader.header_spacing_text``: "single plane,
    voxel um dy=.. dx=.." for a one-plane file) and the channel names; the error for an image that cannot be
    read."""
    if header is None:
        return HEADER_PENDING
    if header.get("error"):
        return str(header["error"])
    sizes = dict(header.get("sizes") or {})
    parts = [" ".join(f"{k}={v}" for k, v in sizes.items())] if sizes else []
    meta = {"sizes": sizes, "spacing": tuple(header.get("spacing") or (1, 1, 1)),
            "spacing_ok": header.get("spacing_ok", True), "spacing_yx_um": header.get("spacing_yx_um")}
    if isinstance(header.get("single_plane"), bool):
        meta["single_plane"] = header["single_plane"]
    parts.append(header_spacing_text(meta))
    names = header.get("channel_names") or []
    if names:
        parts.append("channels: " + ", ".join(str(n) for n in names))
    n_images = header.get("n_images")
    if isinstance(n_images, int) and n_images > 1:
        parts.append(f"{n_images} images in the file")
    return "; ".join(p for p in parts if p)


def _pack_manifest(root, manifest: dict) -> dict:
    """The pack.yaml of the project's copy of its pinned pack (packs/<id>/<version>/), or {}."""
    pack = manifest.get("pack") or {}
    p = Path(root) / "packs" / str(pack.get("id")) / str(pack.get("version")) / "pack.yaml"
    try:
        with open(_long(p), "rb") as fh:
            data = yaml.safe_load(fh.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def object_kind(root, manifest: dict) -> str:
    """What one row of the nuclei table is in this project's pack: nucleus or cell (pack.yaml object_kind)."""
    return str(_pack_manifest(root, manifest).get("object_kind") or "nucleus")


def recipe_status(root, manifest: dict) -> str | None:
    """The status pack.yaml gives the project's recipe (provisional, validated), or None."""
    rid = (manifest.get("pack") or {}).get("recipe")
    for r in _pack_manifest(root, manifest).get("recipes") or []:
        if isinstance(r, dict) and r.get("id") == rid:
            return str(r.get("status")) if r.get("status") else None
    return None


# ---------------------------------------------------------------------------------------------------
# the status board
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BoardRow:
    """One image on the status board: the status record's fields (PLAN 9.3), the objects and spots counted from
    its results, and ``skipped`` for an image a resumed run passed over."""

    image_id: str
    path: str = ""
    status: str = "unknown"
    reason_code: str | None = None
    reason: str | None = None
    qc_pass: bool | None = None
    objects: int | None = None
    spots: int | None = None
    last_run_id: str | None = None
    device_used: str | None = None
    error: str | None = None
    out_dir: str | None = None
    skipped: bool = False

    @property
    def status_text(self) -> str:
        s = self.status
        if s == "done":
            return "done (skipped: already done)" if self.skipped else "done"
        if s == "stale":
            return f"stale: {self.reason or REASONS.get(str(self.reason_code), self.reason_code or 'out of date')}"
        if s == "failed":
            return f"failed: {self.error}" if self.error else "failed"
        if s == "missing":
            return "missing (not run yet)"
        if s == "excluded":
            return f"excluded: {self.reason}" if self.reason else "excluded"
        if s == "unreachable":
            return UNREACHABLE
        if s == "waiting":
            return "waiting (in this run)"
        if s == "unknown":
            return "unknown (no status from the engine yet)"
        return s


def _count_lines(path: Path) -> int | None:
    try:
        with open(_long(path), "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    n = data.count(b"\n")
    if data and not data.endswith(b"\n"):
        n += 1
    return max(0, n - 1)                              # the header line


def _summary_objects(path: Path) -> int | None:
    """n_nuclei of an image_summary table (its one row), or None."""
    try:
        with open(_long(path), "rb") as fh:
            text = fh.read().decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
        if rows and rows[0].get("n_nuclei") not in (None, ""):
            value = float(rows[0]["n_nuclei"])
            return int(value) if math.isfinite(value) else None
    except (OSError, ValueError):
        pass
    return None


# what summary_counts read, per table file: {normcased path: ((st_mtime_ns, st_size), value)}. A run rewrites the
# tables of the images it runs, so a refresh reads only those again (a project on the NAS would otherwise read
# every spots table over the VPN at every refresh). At most _COUNTS_MAX files are kept.
_COUNTS: dict[str, tuple[tuple[int, int], int | None]] = {}
_COUNTS_LOCK = threading.Lock()
_COUNTS_MAX = 4096


def _cached_read(path: Path, read) -> int | None:
    """``read(path)``, or what it gave when the file had this modification time and size; None (nothing read or
    kept) for a file that is not there."""
    try:
        st = os.stat(_long(path))
    except (OSError, ValueError):
        return None
    key, stamp = os.path.normcase(str(path)), (st.st_mtime_ns, st.st_size)
    with _COUNTS_LOCK:
        hit = _COUNTS.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    value = read(path)
    with _COUNTS_LOCK:
        if len(_COUNTS) >= _COUNTS_MAX:
            _COUNTS.clear()
        _COUNTS[key] = (stamp, value)
    return value


def summary_counts(out_dir, image_id: str) -> tuple[int | None, int | None]:
    """``(objects, spots)`` of a finished image: n_nuclei of its image_summary table and the rows of its spots
    table (the image summary has no spot total), or None for what is not there. Each table is read again only
    when its modification time or size changed. Reads files: call it in a worker for a network path."""
    folder = Path(out_dir)
    objects = _cached_read(folder / f"{image_id}__image_summary.csv", _summary_objects)
    spots = _cached_read(folder / f"{image_id}__spots.csv", _count_lines)
    return objects, spots


def series_result(out_dir, image_id: str) -> bool:
    """Whether ``out_dir`` holds the results of an image with timepoints (``<id>__series.json`` next to one
    t<NNN>/ folder per timepoint) rather than one image's tables. Reads the folder: call it in a worker."""
    try:
        return os.path.isfile(_long(Path(out_dir) / f"{image_id}{SERIES_RECORD_SUFFIX}"))
    except (OSError, ValueError):
        return False


def _qc(value) -> bool | None:
    if isinstance(value, bool) or value is None:
        return value
    text = str(value).strip().lower()
    return True if text == "true" else False if text == "false" else None


def board_rows(root, manifest: dict, status: dict | None = None) -> list[BoardRow]:
    """The board: one row per image of the status record (its order and fields), else of the manifest with the
    status unknown (excluded for an image the manifest excludes). The objects and spots of every image with a
    results folder are counted (``summary_counts``)."""
    root = Path(root)
    entries = {str(e.get("id")): e for e in manifest.get("images") or [] if isinstance(e, dict)}
    rows: list[BoardRow] = []
    if isinstance(status, dict) and isinstance(status.get("images"), list):
        for s in status["images"]:
            if not isinstance(s, dict):
                continue
            iid = str(s.get("id"))
            out_dir = str(s.get("out_dir") or root / "results" / iid)
            objects, spots = summary_counts(out_dir, iid)
            rows.append(BoardRow(image_id=iid, path=str(s.get("path") or entries.get(iid, {}).get("path") or ""),
                                 status=str(s.get("status") or "unknown"), reason_code=s.get("reason_code"),
                                 reason=s.get("reason"), qc_pass=_qc(s.get("qc_pass")), objects=objects, spots=spots,
                                 last_run_id=s.get("last_run_id"), device_used=s.get("device_used"),
                                 error=s.get("error"), out_dir=out_dir))
        return rows
    for iid, e in entries.items():
        out_dir = str(root / "results" / iid)
        objects, spots = summary_counts(out_dir, iid)
        excluded = bool(e.get("excluded"))
        rows.append(BoardRow(image_id=iid, path=str(e.get("path") or ""), status="excluded" if excluded else "unknown",
                             reason=e.get("reason") if excluded else None, objects=objects, spots=spots,
                             out_dir=out_dir))
    return rows


def apply_events(rows: list[BoardRow], events: list[dict]) -> list[BoardRow]:
    """``rows`` with a run's log events laid over them (the event fields of PLAN 9.3): run_start makes the images
    it lists waiting; image_done, image_failed and image_skipped set the status, qc_pass, error, the run id and
    the device used. Nothing here reads a file; the end of the run brings a fresh status record."""
    out = list(rows)
    index = {r.image_id: i for i, r in enumerate(out)}
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        kind = str(ev.get("event") or "")
        if kind == "run_start":
            for item in ev.get("images") or []:
                iid = str(item.get("image_id") if isinstance(item, dict) else item)
                if iid in index:
                    i = index[iid]
                    out[i] = dataclasses.replace(out[i], status="waiting", skipped=False)
        elif kind in ("image_done", "image_failed", "image_skipped"):
            iid = str(ev.get("image_id"))
            if iid not in index:
                continue
            i = index[iid]
            word = str(ev.get("status") or kind.removeprefix("image_"))
            skipped = word == "skipped" or kind == "image_skipped"
            status = "done" if skipped else ("failed" if word == "failed" else word)
            changes = {"status": status, "skipped": skipped, "error": ev.get("error") if status == "failed" else None,
                       "reason_code": None, "reason": None}
            if "qc_pass" in ev and not skipped:
                changes["qc_pass"] = _qc(ev.get("qc_pass"))
            if not skipped:
                changes["last_run_id"] = ev.get("run_id") or out[i].last_run_id
                if ev.get("segment_device_used"):
                    changes["device_used"] = ev.get("segment_device_used")
            if ev.get("out_dir"):
                changes["out_dir"] = str(ev["out_dir"])
            out[i] = dataclasses.replace(out[i], **changes)
    return out


def counts_text(counts: dict | None) -> str:
    c = dict(counts or {})
    return f"{int(c.get('done', 0) or 0)} done, {int(c.get('failed', 0) or 0)} failed, " \
           f"{int(c.get('skipped', 0) or 0)} skipped"


def collect_text(record) -> str:
    """The words of ``project collect``'s record (the engine's collect_project: root, results, outlier_factor,
    images, counts {table: rows}, summary): how many images, the summary table it wrote, the rows per table."""
    rec = record if isinstance(record, dict) else {}
    counts = rec.get("counts") if isinstance(rec.get("counts"), dict) else {}
    rows = ", ".join(f"{table} {n}" for table, n in counts.items() if n)
    where = rec.get("summary") or rec.get("results") or "results/"
    return f"collected {rec.get('images', 0)} image(s) into {where}" + (f" (rows: {rows})" if rows else "")


def run_text(result: dict) -> tuple[str, str]:
    """``(text, level)`` of a finished project run (the runner's project_run_result): level ok, warn or bad."""
    rid, state = result.get("run_id"), result.get("state")
    counts = counts_text(result.get("counts"))
    if state == "refused":
        return f"run refused by the engine before any image: {result.get('message') or 'no reason given'}", "bad"
    if state == "interrupted":
        return f"run {rid} interrupted: {counts}; the image in progress has no results", "warn"
    if state == "failed":
        return f"run {rid}: {counts} (the board says which failed and why)", "bad"
    return f"run {rid} done: {counts}", "ok"


# ---------------------------------------------------------------------------------------------------
# the header: validation, the lock, what the project records differently from this computer
# ---------------------------------------------------------------------------------------------------


def _edited(status: dict | None, manifest: dict | None) -> bool:
    recipe = (manifest or {}).get("recipe") or {}
    return bool((status or {}).get("edited")) or bool(recipe.get("file")) or bool(recipe.get("overrides"))


def badge(status: dict | None, manifest: dict | None, *, root=None) -> tuple[str, str]:
    """``(text, level)``: "validated (record ...)" (ok), "validated: false (edited)" (edited) after any recipe edit
    or stage switch, "not validated (provisional pack)" (warn) for a provisional pack recipe, else "not
    validated" (warn). The engine's reason (validated.why) is the tooltip."""
    validated = (status or {}).get("validated") or (manifest or {}).get("validated") or {}
    if isinstance(validated, dict) and validated.get("value") is True:
        return f"validated (record {validated.get('record')})", "ok"
    if _edited(status, manifest):
        return "validated: false (edited)", "edited"
    provisional = root is not None and manifest is not None and recipe_status(root, manifest) == "provisional"
    if provisional or "provisional" in str((validated or {}).get("why") or ""):
        return "not validated (provisional pack)", "warn"
    return "not validated", "warn"


def lock_holder(lock: dict | None) -> tuple[str, int] | None:
    """``(host, pid)`` of a held lock as the status record gives it, or None: not held, or a lock file the engine
    cannot read (its lock_report then gives no host or pid), which only a person removes, by hand."""
    if not isinstance(lock, dict) or not lock.get("held") or lock.get("host") is None:
        return None
    try:
        return str(lock["host"]), int(lock.get("pid"))
    except (TypeError, ValueError):
        return None


def unreadable_lock_text(root) -> str:
    """The engine's own advice for a lock file it cannot read (project/lock.py, unlock_hint)."""
    return (f"the lock file cannot be read (a command may have stopped while it was making it): if no germquant "
            f"command is writing this project, delete {Path(root) / LOCK_NAME} by hand")


def lock_text(lock: dict | None, *, root=None) -> str | None:
    """What the header says about a held lock (status record ``lock``), or None when the project is not locked.
    ``root`` names the lock file of a lock the engine cannot read."""
    if not isinstance(lock, dict) or not lock.get("held"):
        return None
    if lock_holder(lock) is None:
        return unreadable_lock_text(root if root is not None else ".")
    who = f"{lock.get('host')} (pid {lock.get('pid')}) since {lock.get('since')}"
    if lock.get("stale"):
        return (f"locked by {who}: older than 12 h, so the command that took it has probably stopped. Once that "
                f"process is gone, remove the lock (Remove stale lock); it is never removed silently.")
    return f"locked by {who}: a command is writing the project; edits wait until it has finished"


def _switch_keys() -> dict[str, str]:
    """{config key: stage name} of the stages with a --no-<stage> switch (germquant.api.cli_switches)."""
    from germquant import api

    return {str(st.config_key): str(st.name) for st in api.cli_switches()}


def _same_python(a, b) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


def recorded_settings(manifest: dict) -> dict:
    """What the manifest records of the run settings Save writes: device, runner, xy_stride, the stages it turned
    off (``disabled``, names, sorted) and the interpreters ({recipe: python})."""
    run = manifest.get("run") or {}
    keys = _switch_keys()
    over = (manifest.get("recipe") or {}).get("overrides") or {}
    disabled = tuple(sorted(keys[k] for k, v in over.items() if k in keys and v is False))
    interpreters = {}
    for recipe, rec in (run.get("interpreters") or {}).items():
        python = rec.get("python") if isinstance(rec, dict) else rec
        if python:
            interpreters[str(recipe)] = str(python)
    try:
        stride = int(run.get("xy_stride") or 1)
    except (TypeError, ValueError):
        stride = 1
    return {"device": run.get("device"), "runner": run.get("runner"), "xy_stride": stride, "disabled": disabled,
            "interpreters": interpreters}


def settings_changes(manifest: dict, *, device: str | None = None, runner: str | None = None,
                     xy_stride: int | None = None, disabled=None, interpreter: tuple | None = None) -> dict:
    """The ``project edit`` keywords that record what differs from the manifest (an empty dict: nothing to save).
    ``disabled``: the stage names the dock has switched off (a stage the project had off and the dock turns back
    on becomes ``unset`` of its config key); ``interpreter``: ``(recipe, python)`` of this computer."""
    rec = recorded_settings(manifest)
    out: dict = {}
    if device and device != rec["device"]:
        out["device"] = device
    if runner and runner != rec["runner"]:
        out["runner"] = runner
    if xy_stride is not None and int(xy_stride) != rec["xy_stride"]:
        out["xy_stride"] = int(xy_stride)
    if disabled is not None:
        keys = _switch_keys()
        order = list(keys.values())
        now, want = set(rec["disabled"]), {str(n) for n in disabled}
        add = [n for n in order if n in want - now]
        drop = [k for k, n in keys.items() if n in now - want]
        if add:
            out["disable"] = add
        if drop:
            out["unset"] = drop
    if interpreter is not None:
        recipe, python = interpreter
        if recipe and python and not _same_python(rec["interpreters"].get(str(recipe)), python):
            out["interpreters"] = {str(recipe): str(python)}
    return out


def setting_notes(manifest: dict, *, device: str | None, runner: str | None, interpreter: tuple | None) -> list[str]:
    """What the project records differently from this computer: the computer's choice wins at run time, and Save
    records it (design 8.4: a hint and a provenance note)."""
    rec = recorded_settings(manifest)
    notes = []
    if rec["device"] and device and rec["device"] != device:
        notes.append(f"the project records device {rec['device']}; this computer asks for {device} (the choice on "
                     f"this computer wins; Save records it)")
    if rec["runner"] and runner and rec["runner"] != runner:
        notes.append(f"the project records the {rec['runner']} runner; this computer runs it with the {runner} "
                     f"runner")
    if interpreter is not None:
        recipe, python = interpreter
        had = rec["interpreters"].get(str(recipe))
        if had and python and not _same_python(had, python):
            notes.append(f"the project records the interpreter {had} for recipe {recipe}; this computer runs it with "
                         f"{python}")
    return notes


def own_config_hash(root) -> str | None:
    """The config_hash this napari environment's germquant computes for the project (germquant.project.
    project_config, a read), or None when it has no projects or cannot load the recipe. A value that differs from
    the engine's means two different germquant checkouts (design 15e)."""
    try:
        from germquant.project import project_config
    except ImportError:
        return None
    try:
        return str(project_config(Path(root)).hash)
    except Exception:  # noqa: BLE001 - only a cross-check: the engine's hash is the one that counts
        return None


def recipe_config(root, manifest: dict):
    """The project's Config without its stage switches, loaded in process (germquant.project.project_config on a
    copy of the manifest with no overrides), for the Run tab's stage rows. Raises ImportError when this germquant
    has no projects, and what loading raises."""
    from germquant.project import Project, project_config

    m = copy.deepcopy(manifest)
    m.setdefault("recipe", {})["overrides"] = {}
    return project_config(Project(root=Path(root), manifest=m))


# ---------------------------------------------------------------------------------------------------
# the sample sheet
# ---------------------------------------------------------------------------------------------------


def read_sheet(path) -> tuple[list[str], list[dict], list[str]]:
    """``(columns, rows, warnings)`` of a sample sheet CSV, read as the engine reads it: UTF-8 with or without a
    byte-order mark, else cp1252 with a warning (Excel's plain CSV); cells stripped."""
    warnings = []
    with open(_long(path), "rb") as fh:
        data = fh.read()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
        warnings.append(f"{Path(path).name} is not UTF-8: it was read as cp1252 (Excel's plain CSV); save it as "
                        f"'CSV UTF-8' to keep every character")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = [str(c).strip() for c in (reader.fieldnames or [])]
    rows = []
    for raw in reader:
        rows.append({str(k).strip(): ("" if v is None else str(v).strip()) for k, v in raw.items() if k is not None})
    return columns, rows, warnings


def _prefill(image_id: str, path, regex: str | None, defaults: dict | None) -> dict:
    """The design columns from the file name (germquant's parse_sample), only where the recipe's rule matches;
    blank otherwise (an empty cell falls back to the file name in the run)."""
    row = {c: "" for c in DESIGN_COLUMNS}
    row["image_id"] = image_id
    if not regex:
        return row
    try:
        if re.search(regex, image_id, flags=re.IGNORECASE) is None:
            return row
        from germquant.io import parse_sample

        fields = parse_sample(path, regex, defaults)
    except Exception:  # noqa: BLE001 - a rule that does not compile prefills nothing
        return row
    for c in DESIGN_COLUMNS[1:]:
        value = fields.get(c)
        row[c] = "" if value is None else str(value)
    return row


class SampleTable:
    """The Samples tab's table (module docstring). ``ids`` follow the manifest's images; ``columns`` are the
    design columns, then any extra column (from the sheet, or added). ``saved`` holds the sheet's rows as read
    ({image_id: row}) or None when the project has no sheet; ``dirty`` is set by the user's edits since then;
    ``changed_ids`` names the images whose row differs from the saved sheet (a new row included): saving
    restales exactly those."""

    def __init__(self, ids: list[str], columns: list[str], rows: dict[str, dict], saved: dict | None,
                 warnings: list[str] | None = None) -> None:
        self.ids = list(ids)
        self.columns = list(columns)
        self._rows = {i: dict(rows[i]) for i in self.ids}
        self.saved = None if saved is None else {i: dict(r) for i, r in saved.items()}
        self.warnings = list(warnings or [])
        self.dirty = False

    @classmethod
    def from_project(cls, root, manifest: dict, *, regex: str | None = None, defaults: dict | None = None) -> SampleTable:
        """The table of a project: its sheet (the file the manifest's ``samples`` names, else samples.csv when it
        is there) for the images it holds, the file-name prefill for the others."""
        root = Path(root)
        entries = [e for e in manifest.get("images") or [] if isinstance(e, dict)]
        ids = [str(e.get("id")) for e in entries]
        name = manifest.get("samples") or SAMPLES_NAME
        columns, saved, warnings = list(DESIGN_COLUMNS), None, []
        sheet = root / str(name)
        if os.path.isfile(_long(sheet)):
            cols, rows, warnings = read_sheet(sheet)
            if not manifest.get("samples"):
                warnings.append(f"{name} is in the project folder but the manifest does not name it, so runs do not use "
                                f"it: Save records it")
            from germquant import api

            saved = {}
            for r in rows:
                iid = r.get("image_id") or (api.image_stem(Path(r["file"])) if r.get("file") else "")
                if iid:
                    saved[str(iid)] = {**r, "image_id": str(iid)}
            columns += [c for c in cols if c not in columns]
        rows = {}
        for e, iid in zip(entries, ids):
            if saved is not None and iid in saved:
                rows[iid] = {c: saved[iid].get(c, "") for c in columns}
            else:
                pre = _prefill(iid, resolve_image(root, e.get("path", "")), regex, defaults)
                rows[iid] = {c: pre.get(c, "") for c in columns}
        return cls(ids, columns, rows, saved, warnings)

    def row(self, image_id: str) -> dict:
        return dict(self._rows[image_id])

    def records(self) -> list[dict]:
        """Every row, in the manifest's order, with every column: what write_sample_sheet is given."""
        return [{c: self._rows[i].get(c, "") for c in self.columns} for i in self.ids]

    def not_in_sheet(self) -> list[str]:
        return [i for i in self.ids if self.saved is None or i not in self.saved]

    def changed_ids(self) -> list[str]:
        out = []
        for i in self.ids:
            old = (self.saved or {}).get(i)
            if old is None or any(str(old.get(c, "")) != str(self._rows[i].get(c, "")) for c in self.columns):
                out.append(i)
        return out

    def set(self, image_id: str, column: str, value) -> None:
        """Set one cell (stripped). The image_id column names the image and is not edited; the sheet is written as
        ASCII, so a non-ASCII value is refused here, naming its cell."""
        if image_id not in self._rows:
            raise KeyError(f"no image {image_id!r} in the project")
        if column not in self.columns:
            raise KeyError(f"no column {column!r} (add it first)")
        if column == "image_id":
            raise ValueError("the image_id column names the image and cannot be edited")
        text = "" if value is None else str(value).strip()
        if not text.isascii():
            raise ValueError(f"{image_id}, column {column}: the sample sheet is written as ASCII, and {text!r} is not "
                             f"(write the name without accents, or as the lab writes it in ASCII)")
        if self._rows[image_id].get(column, "") != text:
            self._rows[image_id][column] = text
            self.dirty = True

    def add_column(self, name: str) -> None:
        """An extra column (letters, digits and underscore, a letter or underscore first, at most 64)."""
        name = str(name or "").strip()
        if not COLUMN_RE.fullmatch(name):
            raise ValueError(f"not a column name: {name!r} (letters, digits and underscore, a letter first, at most 64)")
        if name in self.columns:
            raise ValueError(f"the sheet already has a column {name!r}")
        self.columns.append(name)
        for r in self._rows.values():
            r[name] = ""
        self.dirty = True

    def mark_saved(self) -> None:
        self.saved = {i: dict(self._rows[i]) for i in self.ids}
        self.dirty = False


def sheet_module():
    """germquant.io.sample_sheet (the engine's sheet reader and writer, tranche 5 Train B), or None when this
    napari environment's germquant has none."""
    try:
        from germquant.io import sample_sheet
    except ImportError:
        return None
    return sample_sheet if hasattr(sample_sheet, "write_sample_sheet") else None


# ---------------------------------------------------------------------------------------------------
# runs in progress, known to every dock of this napari
# ---------------------------------------------------------------------------------------------------

_RUNS: dict[str, str | None] = {}
_RUNS_LOCK = threading.Lock()


def _key(root) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(root))))


def mark_running(root, run_id: str | None = None) -> None:
    with _RUNS_LOCK:
        _RUNS[_key(root)] = run_id


def mark_finished(root) -> None:
    with _RUNS_LOCK:
        _RUNS.pop(_key(root), None)


def run_in_progress(root) -> bool:
    """Whether a dock of this napari runs this project now: recipe edits wait until it has finished."""
    with _RUNS_LOCK:
        return _key(root) in _RUNS


def running_run_id(root) -> str | None:
    with _RUNS_LOCK:
        return _RUNS.get(_key(root))
