"""Helpers of the Project dock tests (tranche 5 Train B, unit P6-PROJECT): project folders written by hand, the
status record of the train's contract (PLAN section 9.3), a minimal results folder, and a fake engine.

The fake engine (FakeRunner, with FakeSession around it) answers the runner's project methods in the dock's own
worker thread, as the Train B engine does through the subprocess runner: it writes a real manifest
(germquant_project.yaml, YAML as the engine writes it) for new, add and edit, so the dock's in-process reads see
what an engine left; ``project status`` is a 9.3 record built from that manifest and the statuses a test sets;
``project run`` plays a scenario of run-log events, handing them to ``progress`` (and writing them to
runs/<run_id>.jsonl) as the subprocess runner's log tailing does, honours ``should_stop`` between events and
raises RunCancelled then, and ends with the exit codes 0, 1 and 2. Every call is recorded in ``calls`` with its
keywords, so a test can assert what the dock asked for (the plugin version of every write, the run settings of
Save). No process is started, so these tests are cheap on every system.

Only the real engine can show that the dock's records are the engine's: the tests of acceptance 1 to 3 run against
it, gated with conftest.need_features.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from napari_germquant import _engine
from napari_germquant._engine_session import EngineSession
from napari_germquant._runner import (
    EngineRefused,
    EngineRunner,
    FeatureMissing,
    RunCancelled,
    check_run_id,
    new_run_id,
    plugin_version,
    project_run_result,
)

MANIFEST = "germquant_project.yaml"
PACK = {"id": "test_pack", "version": "0.1.0", "recipe": "basic", "sha256": "0" * 64}
BASE_HASH = "02bb83cf74f9"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def stamp(plugin: str | None = None) -> dict:
    return {"at": utc_now(), "germquant": "0.2.0.dev0", "git_sha": "f" * 40, "plugin": plugin}


def image_entry(path: str | Path, image_id: str | None = None, **extra) -> dict:
    p = Path(path)
    iid = image_id or p.name.split(".")[0]
    return {"id": iid, "path": str(path), "size": 1234, "mtime": 0.0, **extra, "excluded": False, "reason": None}


def write_manifest(root: str | Path, *, name: str = "study", pack: dict | None = None, images=(),
                   run: dict | None = None, overrides: dict | None = None, samples: str | None = None,
                   recipe_file: str | None = None, expected: str = BASE_HASH, plugin: str | None = None) -> Path:
    """A project folder as the engine leaves it: germquant_project.yaml (format 1, the engine's key order) and the
    pinned pack's copy under packs/<id>/<version>/pack.yaml (object_kind nucleus)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pack = dict(pack or PACK)
    m = {"format": 1, "name": name, "id": "00000000-0000-4000-8000-000000000000", "created": stamp(plugin),
         "saved": stamp(plugin), "pack": pack,
         "recipe": {"file": recipe_file, "expected_config_hash": expected, "overrides": dict(overrides or {}),
                    "roles": {}, "origins": {}},
         "run": {"xy_stride": 1, "z_range": None, "device": None, "runner": None, "interpreters": {},
                 **(run or {})},
         "models": [], "images": [dict(e) for e in images], "samples": samples,
         "validated": {"value": False, "why": f"not validated: the pack recipe {pack['id']}@{pack['version']}/"
                                              f"{pack['recipe']} is provisional", "record": None}}
    (root / MANIFEST).write_bytes(yaml.safe_dump(m, sort_keys=False, allow_unicode=False).encode("ascii"))
    copy = root / "packs" / pack["id"] / pack["version"]
    copy.mkdir(parents=True, exist_ok=True)
    (copy / "pack.yaml").write_text(f"format: 1\nid: {pack['id']}\nversion: {pack['version']}\nobject_kind: nucleus\n",
                                    encoding="ascii")
    return root / MANIFEST


def read_manifest(root: str | Path) -> dict:
    return yaml.safe_load((Path(root) / MANIFEST).read_text(encoding="ascii"))


def status_record(root: str | Path, manifest: dict | None = None, *, statuses: dict | None = None,
                  config_hash: str | None = None, warnings=(), lock: dict | None = None, runs=(),
                  models=()) -> dict:
    """The status record of PLAN section 9.3 for a project folder: every image of the manifest, with the status a
    test gives it in ``statuses`` ({image_id: {"status", "reason_code", ...}}, default missing)."""
    root = Path(root)
    m = manifest or read_manifest(root)
    pinned = f"{m['pack']['id']}@{m['pack']['version']}"
    images = []
    for e in m["images"]:
        given = dict((statuses or {}).get(e["id"]) or {})
        images.append({"id": e["id"], "path": e["path"], "status": given.get("status", "missing"),
                       "reason_code": given.get("reason_code"), "reason": given.get("reason"),
                       "out_dir": str(root / "results" / e["id"]), "qc_pass": given.get("qc_pass"),
                       "last_run_id": given.get("last_run_id"), "device_used": given.get("device_used"),
                       "error": given.get("error")})
    edited = bool(m["recipe"].get("file")) or bool(m["recipe"].get("overrides"))
    return {"format": 1, "root": str(root), "name": m["name"],
            "config_hash": config_hash or m["recipe"]["expected_config_hash"],
            "expected_config_hash": m["recipe"]["expected_config_hash"], "edited": edited,
            "validated": dict(m["validated"]),
            "pack": {"id": m["pack"]["id"], "pinned": pinned, "installed": [m["pack"]["version"]],
                     "project_copy": True},
            "models": list(models), "images": images, "warnings": list(warnings),
            "lock": lock or {"held": False, "host": None, "pid": None, "since": None, "stale": False},
            "runs": list(runs)}


def write_results(folder: str | Path, image_id: str, *, n_nuclei: int = 3, n_spots: int = 5,
                  spacing=(0.4, 0.2, 0.2), source: str | Path | None = None, config_hash: str = BASE_HASH) -> Path:
    """A minimal results folder of one image, as process_image leaves it: a ZYX label image with n_nuclei labels,
    the nuclei, spots and image_summary tables (the shared voxel columns), and the done marker."""
    import pandas as pd
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((2, 16, 24), np.int32)
    for i in range(n_nuclei):
        labels[:, 2 + i * 4:5 + i * 4, 3:9] = i + 1
    tifffile.imwrite(str(folder / f"{image_id}__nuclei_labels.tif"), labels)
    dz, dy, dx = spacing
    voxel = {"voxel_dz_um": dz, "voxel_dy_um": dy, "voxel_dx_um": dx, "config_hash": config_hash}
    pd.DataFrame([{"nucleus_id": i + 1, "volume_um3": 1.0 + i, **voxel} for i in range(n_nuclei)]).to_csv(
        folder / f"{image_id}__nuclei.csv", index=False)
    pd.DataFrame([{"spot_id": j + 1, "nucleus_id": 1 + j % max(1, n_nuclei), "z_um": 0.4, "y_um": 1.0 + j * 0.2,
                   "x_um": 1.0} for j in range(n_spots)]).to_csv(folder / f"{image_id}__spots.csv", index=False)
    row = {"n_nuclei": n_nuclei, "mean_spots": n_spots / max(1, n_nuclei), "qc_pass": True, "qc_flags": "",
           "file_path": str(source) if source else "", **voxel}
    pd.DataFrame([row]).to_csv(folder / f"{image_id}__image_summary.csv", index=False)
    (folder / f"{image_id}__done.json").write_text(json.dumps({"image_id": image_id, "config_hash": config_hash,
                                                                 "xy_stride": 1, "failed_stages": []}),
                                                   encoding="ascii")
    return folder


def listing(folder: str | Path | None = None) -> dict:
    """A pack listing with one usable pack (what `python -m germquant.packs list --json --all` prints)."""
    return {"version": {"germquant": "0.2.0.dev0", "git_sha": "f" * 40}, "builtin_root_found": True, "message": None,
            "roots": [], "packs": [{
                "id": "test_pack", "version": "0.1.0", "title": "Test pack", "state": "ok", "problems": [],
                "folder": str(folder) if folder else None, "object_kind": "nucleus", "root": "pack",
                "recipes": [{"id": "basic", "title": "Classical nuclei", "default": True, "dims": "3d",
                             "profile": "config/profiles/basic.yaml", "tunables": []},
                            {"id": "with_model", "title": "Cellpose nuclei", "default": False, "dims": "3d",
                             "profile": "config/profiles/with_model.yaml", "tunables": []}],
                "models": []}]}


def _fake_hash(overrides: dict, edits: dict) -> str:
    if not overrides and not edits:
        return BASE_HASH
    text = json.dumps([sorted(overrides.items()), sorted(edits.items())], default=str)
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:12]


class FakeRunner(EngineRunner):
    """The fake engine of the module docstring. ``features`` the version document lists; ``script`` the events of
    the next ``project run`` ([{"event", "image_id", "status", "error", ...}, ...], each after ``pause`` seconds),
    ``run_exit`` its exit code, ``refuse_run`` a refusal text (exit 2 before any event), ``results`` image ids whose
    image_done also writes a results folder; ``statuses`` feeds ``project status``; ``headers`` the records of
    ``project add``; ``gate`` (a threading.Event) holds a run before its first event until a test sets it."""

    kind = "subprocess"
    via = "gui-subprocess"
    cancellable = True

    def __init__(self, *, features=_engine.FEATURES, python: str | None = None) -> None:
        super().__init__()
        self.python = python or sys.executable
        self._features = frozenset(features)
        self.calls: list[tuple[str, dict]] = []
        self.statuses: dict[str, dict] = {}
        self.status_warnings: list[str] = []
        self.lock: dict | None = None
        self.headers: dict[str, dict] = {}
        self.script: list[dict] = []
        self.pause = 0.05
        self.run_exit = 0
        self.refuse_run: str | None = None
        self.refuse: dict[str, str] = {}
        self.results: set[str] = set()
        self.gate: threading.Event | None = None
        self.example: Path | None = None
        self.edits: dict = {}
        self._lock = threading.Lock()
        self._released = threading.Event()

    def release(self) -> None:
        """Let a run that waits (``gate``, a step's ``wait``) go on: a test calls it at teardown, so a test that
        failed before setting them never leaves a worker thread blocked (napari would wait for it at exit)."""
        self._released.set()
        for ev in [self.gate, *[s.get("wait") for s in self.script]]:
            if ev is not None:
                ev.set()

    # ---- the plumbing ----
    def _call(self, _what: str, **kw) -> None:
        with self._lock:
            self.calls.append((_what, dict(kw)))
        if _what in self.refuse:
            raise EngineRefused(f"`germquant project {_what}` was refused by the engine (exit 2):\n"
                                f"REFUSED: {self.refuse[_what]}", output=self.refuse[_what])

    def called(self, name: str) -> list[dict]:
        with self._lock:
            return [kw for n, kw in self.calls if n == name]

    def describe(self) -> str:
        return f"{self.python} (a fake engine)"

    def version(self, refresh: bool = False, should_stop=None) -> dict:
        return {"germquant": "0.2.0.dev0", "git_sha": "f" * 40, "python": "3.11", "executable": self.python,
                "features": sorted(self._features)}

    def cached_version(self) -> dict | None:
        return self.version()

    def features(self, should_stop=None) -> frozenset[str]:
        return self._features

    def feature_message(self, feature: str) -> str:
        return _engine.feature_missing_message(feature, info=self.version(), interpreter=self.python)

    def runtime(self, prefer=None, should_stop=None) -> dict:
        return {"device_requested": self.device, "device_selected": "cpu", "available": {"cuda": False, "mps": False}}

    def list_packs(self, should_stop=None, project=None) -> dict:
        self._call("list_packs", project=project)
        return listing()

    def check(self, pack, recipe=None, image=None, should_stop=None, project=None) -> dict:
        self._call("check", pack=pack, recipe=recipe, image=image, project=project)
        return {"pack": pack, "recipe": recipe, "usable": True, "blockers": [], "warnings": [], "models": []}

    def fetch_models(self, pack, recipe=None, say=None, should_stop=None, project=None) -> dict:
        self._call("fetch_models", pack=pack, recipe=recipe, project=project)
        return {"ok": True, "models": [], "lines": []}

    def fetch_example(self, pack, example=None, out=None, say=None, should_stop=None) -> dict:
        self._call("fetch_example", pack=pack, example=example, out=out)
        return {"id": "synthetic", "pack": pack, "kind": "local", "status": "local", "image": str(self.example)}

    # ---- the project writes (the torch-free entry) ----
    def _check(self, _what: str, kw: dict, should_stop) -> None:
        if should_stop is not None and should_stop():
            raise RunCancelled("cancelled by the user before it started")
        self.require("projects")
        self._call(_what, **kw)

    def project_new(self, root, pack, *, recipe=None, images=(), name=None, plugin_version=None,
                    should_stop=None) -> dict:
        self._check("project_new", {"root": root, "pack": pack, "recipe": recipe, "images": images, "name": name,
                                    "plugin_version": plugin_version}, should_stop)
        pid, _, ver = str(pack).partition("@")
        entry = {"id": pid, "version": ver or "0.1.0", "recipe": recipe or "basic", "sha256": "0" * 64}
        write_manifest(root, name=name or Path(root).name, pack=entry, plugin=plugin_version or _plugin())
        res = self._add(root, [images] if isinstance(images, (str, Path)) else list(images or ()), plugin_version)
        m = read_manifest(root)
        return {"root": str(root), "name": m["name"], "id": m["id"], "pack": m["pack"], "recipe": m["recipe"],
                "config_hash": BASE_HASH, "expected_config_hash": BASE_HASH, "edited": False,
                "validated": m["validated"], "run": m["run"], "samples": None, "images": m["images"],
                "warnings": [], "added": res["added"], "refused": res["refused"]}

    def _add(self, root, paths, plugin) -> dict:
        m = read_manifest(root)
        added, refused, entries, headers = [], [], [], {}
        for p in paths:
            text = str(p)
            if "refuse" in Path(text).name:
                refused.append({"path": text, "reason": "holds 2 images (a multi-scene file): split it into one file "
                                                        "per scene"})
                continue
            entry = image_entry(text)
            if any(e["id"] == entry["id"] for e in m["images"]):
                refused.append({"path": text, "reason": f"duplicate image id {entry['id']}"})
                continue
            m["images"].append(entry)
            entries.append(entry)
            added.append(entry["id"])
            headers[entry["id"]] = self.headers.get(entry["id"], {"sizes": {"C": 2, "Z": 3, "Y": 16, "X": 24},
                                                                  "channel_names": ["DAPI", "GFP"], "n_images": 1,
                                                                  "spacing": [0.4, 0.2, 0.2], "spacing_ok": True,
                                                                  "spacing_yx_um": [0.2, 0.2],
                                                                  "single_plane": False, "reader": "tifffile"})
        if added:
            m["saved"] = stamp(plugin or _plugin())
            (Path(root) / MANIFEST).write_bytes(yaml.safe_dump(m, sort_keys=False, allow_unicode=False)
                                                .encode("ascii"))
        return {"added": added, "refused": refused, "images": entries, "headers": headers}

    def project_add(self, root, paths, *, plugin_version=None, should_stop=None) -> dict:
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        self._check("project_add", {"root": root, "paths": paths, "plugin_version": plugin_version}, should_stop)
        return self._add(root, paths, plugin_version)

    def project_edit(self, root, *, edits=None, unset=(), roles=None, revert=False, device=None, xy_stride=None,
                     z_range=None, runner=None, interpreters=None, samples=None, disable=(), dry_run=False,
                     plugin_version=None, should_stop=None) -> dict:
        kw = {"root": root, "edits": edits, "unset": list(unset or ()), "roles": roles, "revert": revert,
              "device": device, "xy_stride": xy_stride, "z_range": z_range, "runner": runner,
              "interpreters": interpreters, "samples": samples, "disable": list(disable or ()), "dry_run": dry_run,
              "plugin_version": plugin_version}
        self._check("project_edit", kw, should_stop)
        from germquant.stages import BY_NAME

        m = read_manifest(root)
        overrides = dict(m["recipe"]["overrides"])
        for name in disable or ():
            overrides[BY_NAME[name].config_key] = False
        for key in unset or ():
            overrides.pop(key, None)
        edits_now = {} if revert else {**self.edits, **(edits or {})}
        run = dict(m["run"])
        if device is not None:
            run["device"] = None if device == "none" else device
        if runner is not None:
            run["runner"] = None if runner == "none" else runner
        if xy_stride is not None:
            run["xy_stride"] = int(xy_stride)
        for recipe, python in (interpreters or {}).items():
            run["interpreters"] = {**run["interpreters"], recipe: {"python": str(python), "germquant": "0.2.0.dev0",
                                                                    "git_sha": "f" * 40}}
        new_hash = _fake_hash(overrides, edits_now)
        if not dry_run:
            self.edits = edits_now
            m["recipe"]["overrides"] = overrides
            m["recipe"]["expected_config_hash"] = new_hash
            m["recipe"]["file"] = "recipe.yaml" if edits_now else None
            if edits_now or overrides:
                m["validated"] = {"value": False, "why": "edited: the project changed the pack recipe", "record": None}
            m["run"] = run
            if samples is not None:
                m["samples"] = None if samples == "none" else str(samples)
            m["saved"] = stamp(plugin_version or _plugin())
            (Path(root) / MANIFEST).write_bytes(yaml.safe_dump(m, sort_keys=False, allow_unicode=False)
                                                .encode("ascii"))
        return {"dry_run": bool(dry_run), "changed": True, "config_hash": new_hash,
                "previous_config_hash": read_manifest(root)["recipe"]["expected_config_hash"],
                "edited": bool(edits_now), "validated": m["validated"], "recipe_yaml": None, "channel_map_yaml": None,
                "preview": None, "written": [] if dry_run else [MANIFEST], "removed": [], "run": run,
                "samples": m["samples"], "overrides": overrides, "warnings": []}

    def project_unlock(self, root, *, host, pid, should_stop=None) -> dict:
        self._check("project_unlock", {"root": root, "host": host, "pid": pid}, should_stop)
        if self.lock is not None and (self.lock.get("host"), self.lock.get("pid")) != (host, pid):
            raise EngineRefused("REFUSED: the lock is held by another process", exit_code=2)
        removed, self.lock = self.lock is not None, None
        return {"removed": removed, "exit_code": 0, "lines": []}

    # ---- the full command line ----
    def project_status(self, root, *, should_stop=None) -> dict:
        self._check("project_status", {"root": root}, should_stop)
        m = read_manifest(root)
        return status_record(root, m, statuses=self.statuses, warnings=self.status_warnings,
                             config_hash=m["recipe"]["expected_config_hash"],
                             lock=self.lock and {"held": True, "stale": True, "since": "2026-10-20T00:00:00+00:00",
                                                 **self.lock})

    def project_run(self, root, *, images=None, resume=True, device=None, shard=None, run_id=None, say=None,
                    progress=None, should_stop=None) -> dict:
        self._check("project_run", {"root": root, "images": images, "resume": resume, "device": device,
                                    "shard": shard, "run_id": run_id}, should_stop)
        run_id = check_run_id(run_id or new_run_id())
        log = Path(root) / "runs" / f"{run_id}.jsonl"
        if self.refuse_run:
            return project_run_result(run_id, log, [], 2, [f"REFUSED: {self.refuse_run}"])
        log.parent.mkdir(parents=True, exist_ok=True)
        m = read_manifest(root)
        ids = list(images) if images else [e["id"] for e in m["images"]]
        events: list[dict] = []

        def emit(kind: str, **fields) -> None:
            ev = {"event": kind, "run_id": run_id, "time": utc_now(), **fields}
            events.append(ev)
            with open(log, "ab") as fh:
                fh.write((json.dumps(ev) + "\n").encode("ascii"))
            if progress is not None:
                progress([ev])

        emit("run_start", via=self.via, device_requested=device, project=str(root), shard=shard,
             images=[{"image_id": i, "out_dir": str(Path(root) / "results" / i)} for i in ids])
        if self.gate is not None:
            while not self.gate.wait(0.02):
                if should_stop is not None and should_stop():
                    break
        for step in self.script:
            hold = step.get("wait")                   # a threading.Event the test sets to let this event come
            while hold is not None and not hold.wait(0.02):
                if should_stop is not None and should_stop():
                    break
            if should_stop is not None and should_stop():
                err = RunCancelled("cancelled by the user: the fake engine stopped between images")
                err.run_id, err.run_log, err.events = run_id, str(log), events
                raise err
            time.sleep(float(step.get("pause", self.pause)))
            kind = step["event"]
            iid = step["image_id"]
            if kind == "image_done" and iid in self.results:
                write_results(Path(root) / "results" / iid, iid)
            emit(kind, image_id=iid, out_dir=str(Path(root) / "results" / iid),
                 status=step.get("status") or kind.removeprefix("image_"), qc_pass=step.get("qc_pass", True),
                 segment_device_used=step.get("device", "cpu"), error=step.get("error"))
            if say is not None:
                say(f"{kind} {iid}")
        counts = {"done": 0, "failed": 0, "skipped": 0}
        for ev in events:
            if ev["event"] in ("image_done", "image_failed", "image_skipped"):
                counts[ev["status"]] = counts.get(ev["status"], 0) + 1
        emit("run_end", counts=counts, interrupted=False)
        return project_run_result(run_id, log, events, self.run_exit, [])

    def project_collect(self, root, *, should_stop=None) -> dict:
        self._check("project_collect", {"root": root}, should_stop)
        return {"out": str(Path(root) / "results"), "tables": ["batch_nuclei.csv", "batch_image_summary.csv"]}

    def project_export_commands(self, root, *, shell="sh", shards=None, out=None, should_stop=None) -> dict:
        self._check("project_export_commands", {"root": root, "shell": shell, "shards": shards, "out": out},
                    should_stop)
        text = f"germquant project run {root}\n"
        if out is not None:
            Path(out).write_text(text, encoding="ascii")
        return {"text": text, "out": str(out) if out is not None else None}


def _plugin() -> str:
    return plugin_version()


class FakeSession(EngineSession):
    """An engine session whose engine is ``runner`` (no settings read, no version check): the dock's seam for the
    fake engine. ``missing_feature`` makes every require fail as an engine without that feature would."""

    def __init__(self, runner: EngineRunner, *, missing_feature: str | None = None) -> None:
        super().__init__()
        self.fake = runner
        self.missing_feature = missing_feature

    def runner(self):
        return self.fake

    def ensure(self, should_stop=None):
        return self.fake

    def runner_for(self, pack_id=None, recipe_id=None, should_stop=None):
        return self.fake

    def require(self, *features, should_stop=None):
        if self.missing_feature is not None and self.missing_feature in features:
            raise FeatureMissing(self.fake.feature_message(self.missing_feature), self.missing_feature)
        self.fake.require(*features, should_stop=should_stop)
        return self.fake

    def describe(self) -> str:
        return f"{self.fake.describe()}; device {self.device()}"

    def info(self):
        return self.fake.version()

    def devices(self, should_stop=None) -> list[str]:
        return ["auto", "cpu"]
