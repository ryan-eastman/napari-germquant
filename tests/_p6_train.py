"""Helpers of the Training dock's project-mode tests (tranche 5 Train B, unit P6-TRAIN): project folders and
finished runs written by hand as the engine writes them, and a fake engine.

The fake engine (FakeTrainRunner, with StaticSession around it) answers the runner methods the Training dock
calls (project rounds and use-model, pack derive and the pack listing with a project, import-labels,
export-training and train-nuclei) in the dock's own worker thread, as the Train B engine does through the
subprocess runner: it writes what the engine would leave on disk (round folders, slice pairs with their
manifest.csv, a model file and its card, a derived pack's folder) so the dock's in-process reads see real files,
refuses what the engine refuses (a _seg.npy without allow_pickle, exit 2: EngineRefused), and records every call
with its keywords, so a test can assert what the dock asked for. ``gates`` hold a call until the test lets it go,
and a held call honours ``should_stop`` as the subprocess runner does (RunCancelled). No process is started, so
these tests are cheap on every system and run against every engine the plugin CI pins.

Only the real engine can show that the dock's calls are the engine's commands: the acceptance tests of
tests/test_training.py run against it, gated with conftest.need_features.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from qtpy.QtCore import QObject, Signal

from napari_germquant import _engine
from napari_germquant._runner import (
    EngineRefused,
    FeatureMissing,
    RunCancelled,
    RunnerError,
    device_choices,
)

MANIFEST = "germquant_project.yaml"
ROUND_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
ALL_FEATURES = frozenset(_engine.FEATURES)
NEW_HASH = "5e1f00ddc0de"        # the config_hash the fake gives after use-model
DERIVED_ID = "test_pack_derived_1a2b3c4d"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def stamp(plugin: str | None = None) -> dict:
    return {"at": utc_now(), "germquant": "0.2.0.dev0", "git_sha": "f" * 40, "plugin": plugin}


def write_project(root: str | Path, *, name: str = "study", pack_id: str = "test_pack", version: str = "0.1.0",
                  recipe: str = "basic", images=(), models=(), object_kind: str = "nucleus", pack_models=(),
                  rounds=()) -> Path:
    """A project folder as the engine leaves it: germquant_project.yaml (format 1, the engine's key order), the
    pinned pack's copy under packs/<id>/<version>/pack.yaml (its object kind and the recipe's needs.models), and
    the round folders training/<round>/. ``images``: (image id, path) pairs; a relative path is a POSIX path inside
    the project, as the engine records an image there."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pack = {"id": pack_id, "version": version, "recipe": recipe, "sha256": "0" * 64}
    m = {"format": 1, "name": name, "id": "00000000-0000-4000-8000-000000000000", "created": stamp(),
         "saved": stamp(), "pack": pack,
         "recipe": {"file": None, "expected_config_hash": "02bb83cf74f9", "overrides": {}, "roles": {},
                    "origins": {}},
         "run": {"xy_stride": 1, "z_range": None, "device": None, "runner": None, "interpreters": {}},
         "models": [dict(x) for x in models],
         "images": [{"id": iid, "path": str(path), "size": 1234, "mtime": 0.0, "excluded": False, "reason": None}
                    for iid, path in images],
         "samples": None,
         "validated": {"value": False, "why": "not validated: the pack recipe is provisional", "record": None}}
    (root / MANIFEST).write_bytes(yaml.safe_dump(m, sort_keys=False, allow_unicode=False).encode("ascii"))
    copy = root / "packs" / pack_id / version
    copy.mkdir(parents=True, exist_ok=True)
    doc = {"format": 1, "id": pack_id, "version": version, "object_kind": object_kind,
           "recipes": [{"id": recipe, "needs": {"models": list(pack_models)}}]}
    (copy / "pack.yaml").write_bytes(yaml.safe_dump(doc, sort_keys=False).encode("ascii"))
    for r in rounds:
        (root / "training" / r).mkdir(parents=True, exist_ok=True)
    return root


def read_manifest(root: str | Path) -> dict:
    return yaml.safe_load((Path(root) / MANIFEST).read_text(encoding="ascii"))


def write_model(folder: str | Path, name: str | None = None) -> tuple[Path, Path]:
    """A model file and its card as train-nuclei writes them: ``<folder>/<name>`` and ``<name>.card.json``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = name or folder.name
    model = folder / name
    model.write_bytes(b"fake cellpose weights " * 64)
    card = folder / f"{name}.card.json"
    card.write_text(json.dumps({"name": name, "kind": "cellpose", "dims": "3d", "object_kind": "nucleus",
                                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}), encoding="ascii")
    return model, card


def write_run(folder: str | Path, image_id: str, *, kind: str = "3d", source: str | Path | None = None,
              batch: bool = False) -> Path:
    """A finished run as process_image leaves it, for the geometry the export follows: ``kind`` "3d" (a ZYX label
    image), "projected" (segment_projection in the record, the 2D labels copied through z) or "planar" (a YX label
    image, ``"planar": true`` in the record). The record is run_manifest.json, or with ``batch`` the run's
    ``<id>__stages.json`` with the geometry nested under run_geometry, as a batch writes it."""
    import pandas as pd
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lab = np.zeros((4, 24, 24), np.int32)
    for i in range(3):
        lab[:, 2 + 7 * i:7 + 7 * i, 4:10] = i + 1
    if kind == "planar":
        lab = lab[0]
    tifffile.imwrite(str(folder / f"{image_id}__nuclei_labels.tif"), lab)
    pd.DataFrame([{"image_id": image_id, "file_path": str(source or folder / f"{image_id}.ome.tif"), "n_nuclei": 3,
                   "voxel_dz_um": 0.5, "voxel_dy_um": 0.2, "voxel_dx_um": 0.2,
                   "channel_map_json": json.dumps({"dna": 0})}]).to_csv(folder / f"{image_id}__image_summary.csv",
                                                                       index=False)
    geometry: dict = {"xy_stride": 1, "z_range": None, "t_index": None}
    if kind == "projected":
        geometry["segment_projection"] = {"mode": "max", "plane": None, "z": None}
    elif kind == "planar":
        geometry["planar"] = True
        geometry["spacing_source"] = "file_yx"
    elif kind != "3d":
        raise ValueError(kind)
    if batch:
        rec = {"image_id": image_id, "config_hash": "0" * 12, "run_geometry": geometry}
        (folder / f"{image_id}__stages.json").write_text(json.dumps(rec), encoding="ascii")
    else:
        (folder / "run_manifest.json").write_text(json.dumps({**geometry, "config": {}}), encoding="ascii")
    return folder


def write_pairs(pairs_dir: str | Path, image_id: str, n: int = 2, *, orientation: str = "xy",
                source: str = "") -> list[str]:
    """``n`` slice pairs of ``image_id`` written into a pairs folder as export-training writes them (image/ uint16,
    labels/ int32, manifest.csv merged by image id); returns the pair names."""
    import pandas as pd
    import tifffile

    pairs_dir = Path(pairs_dir)
    (pairs_dir / "image").mkdir(parents=True, exist_ok=True)
    (pairs_dir / "labels").mkdir(parents=True, exist_ok=True)
    rows, names = [], []
    for i in range(n):
        stem = f"{image_id}_{orientation}_{i:04d}.tif"
        lab = np.zeros((16, 16), np.int32)
        lab[2:6, 2 + i:6 + i] = 1
        lab[9:13, 8:12] = 2
        tifffile.imwrite(str(pairs_dir / "image" / stem), (lab * 100 + i).astype(np.uint16))
        tifffile.imwrite(str(pairs_dir / "labels" / stem), lab)
        rows.append({"image_id": image_id, "series_id": image_id, "t_index": "", "orientation": orientation,
                     "slice_index": i, "n_labels": 2, "zone": "", "image": f"image/{stem}",
                     "labels": f"labels/{stem}", "source_file": "", "xy_stride": 1, "z_range": "",
                     "channel_role": "dna", "channel_index": 0, "pixel_um": 0.2, "z_zoom": 1.0, "height": 16,
                     "width": 16, **({"source": source} if source else {})})
        names.append(stem)
    manifest = pairs_dir / "manifest.csv"
    df = pd.DataFrame(rows)
    if manifest.is_file():
        old = pd.read_csv(manifest)
        df = pd.concat([old[old["image_id"] != image_id], df], ignore_index=True)
    df.to_csv(manifest, index=False)
    return names


class FakeTrainRunner:
    """A fake engine runner for the Training dock (module docstring). ``features`` the engine lists (all of them by
    default), ``available`` its accelerators (the runtime probe), ``channel`` whether its import_labels takes the
    channel (a runner of a later P6-RUNNER), ``refuse`` {method: message} refusals (EngineRefused, exit 2)."""

    kind = "fake"
    cancellable = True

    def __init__(self, *, features=ALL_FEATURES, available=None, refuse=None, describe: str = "a fake engine") -> None:
        self.python = sys.executable
        self.device = "auto"
        self._features = frozenset(features)
        self.available = dict(available or {"cuda": False, "mps": False})
        self.refuse = dict(refuse or {})
        self._describe = describe
        self.calls: list[tuple[str, tuple, dict]] = []
        self.gates: dict[str, threading.Event] = {}
        self.entered: dict[str, threading.Event] = {}
        self.derived: list[dict] = []
        self.lock = threading.Lock()

    # ---- what a runner is ----
    @property
    def key(self) -> str:
        return f"{self.kind}:{self.python}"

    def describe(self) -> str:
        return self._describe

    def version(self, refresh: bool = False, should_stop=None) -> dict:
        return {"germquant": "0.2.0.dev0", "git_sha": "f" * 40, "python": "3.11", "executable": self.python,
                "features": sorted(self._features)}

    def cached_version(self) -> dict:
        return self.version()

    def features(self, should_stop=None) -> frozenset:
        return self._features

    def feature_message(self, feature: str) -> str:
        return _engine.feature_missing_message(feature, info=self.version(), interpreter=self.python)

    def require(self, *features: str, should_stop=None) -> None:
        for f in features:
            if f not in self._features:
                raise FeatureMissing(self.feature_message(f), f)

    def runtime(self, prefer=None, should_stop=None) -> dict:
        return {"device_requested": "auto", "device_selected": "cpu", "available": dict(self.available)}

    # ---- the plumbing of a fake call ----
    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        with self.lock:
            self.calls.append((name, args, {k: v for k, v in kwargs.items() if k not in ("say", "should_stop")}))

    def _hold(self, name: str, should_stop) -> None:
        """Wait at the gate of ``name`` (when a test set one), honouring should_stop as the subprocess runner
        does."""
        if name in self.entered:
            self.entered[name].set()
        gate = self.gates.get(name)
        if gate is None:
            return
        deadline = time.monotonic() + 60
        while not gate.wait(0.05):
            if should_stop is not None and should_stop():
                raise RunCancelled("cancelled by the user: the fake engine and every process it started were ended")
            if time.monotonic() > deadline:
                raise RunnerError(f"the test never opened the gate of {name}")

    def _start(self, name: str, args: tuple, kwargs: dict, *features: str) -> None:
        should_stop = kwargs.get("should_stop")
        if should_stop is not None and should_stop():
            raise RunCancelled("cancelled by the user before it started")
        self.require(*features)
        self._record(name, args, kwargs)
        self._hold(name, should_stop)
        if name in self.refuse:
            raise EngineRefused(f"`germquant {name}` was refused by the engine (exit 2):\nREFUSED: {self.refuse[name]}",
                                output=f"REFUSED: {self.refuse[name]}")

    def called(self, name: str) -> list[tuple[tuple, dict]]:
        with self.lock:
            return [(a, k) for n, a, k in self.calls if n == name]

    def last(self, name: str) -> tuple[tuple, dict]:
        got = self.called(name)
        assert got, f"{name} was never called; calls: {[n for n, _a, _k in self.calls]}"
        return got[-1]

    # ---- the commands the Training dock calls ----
    def project_rounds(self, root, *, new=None, plugin_version=None, should_stop=None) -> dict:
        self._start("project_rounds", (root,), {"new": new, "plugin_version": plugin_version,
                                                 "should_stop": should_stop}, "projects")
        d = Path(root) / "training"
        created = path = None
        if new:
            taken = {p.name.lower() for p in d.iterdir()} if d.is_dir() else set()
            if isinstance(new, str):
                name = new
            else:
                n = 1
                while f"r{n:02d}" in taken:
                    n += 1
                name = f"r{n:02d}"
            if name.lower() in taken:
                raise EngineRefused(f"REFUSED: the training round {name} already exists",
                                    output="REFUSED: exists")
            (d / name).mkdir(parents=True)
            created, path = name, str(d / name)
        rounds = sorted(p.name for p in d.iterdir() if p.is_dir() and ROUND_RE.fullmatch(p.name)) if d.is_dir() else []
        return {"rounds": rounds, "created": created, "path": path}

    def project_use_model(self, root, model, *, card=None, plugin_version=None, should_stop=None) -> dict:
        self._start("project_use_model", (root, model), {"card": card, "plugin_version": plugin_version,
                                                          "should_stop": should_stop}, "projects")
        return {"config_hash": NEW_HASH, "expected_config_hash": NEW_HASH,
                "validated": {"value": False, "why": "not validated: the recipe was edited", "record": None}}

    def pack_derive(self, from_ref, recipe, model, card, out, *, new_id=None, should_stop=None) -> dict:
        self._start("pack_derive", (from_ref, recipe, model, card, out), {"new_id": new_id,
                                                                           "should_stop": should_stop},
                    "pack_derive")
        pid = new_id or DERIVED_ID
        folder = Path(out) / pid / "0.1.0"
        folder.mkdir(parents=True, exist_ok=True)
        rec = {"ok": True, "status": "derived", "pack": f"{pid}@0.1.0", "id": pid, "version": "0.1.0",
               "folder": str(folder), "recipe": recipe, "extends": f"pack:{from_ref}/{recipe}",
               "config_hash": "d371ed0c0de1", "parent": {"pack": from_ref, "folder": None, "copied": False},
               "where": "project"}
        self.derived.append(rec)
        return rec

    def list_packs(self, should_stop=None, project=None) -> dict:
        self._start("list_packs", (), {"project": project, "should_stop": should_stop},
                    *(("projects",) if project is not None else ()))
        packs = [{"id": "test_pack", "version": "0.1.0", "source": "project" if project else "builtin"}]
        if project is not None:
            packs += [{"id": d["id"], "version": d["version"], "source": "project",
                       "recipes": [{"id": d["recipe"]}]} for d in self.derived]
        return {"packs": packs}

    def import_labels(self, src, fmt, image, pairs, *, allow_pickle=False, frame=None, should_stop=None) -> dict:
        self._start("import_labels", (src, fmt, image, pairs), {"allow_pickle": allow_pickle, "frame": frame,
                                                                "should_stop": should_stop}, "import_labels")
        return self._import(src, fmt, image, pairs, allow_pickle=allow_pickle, frame=frame, channel=None)

    def _import(self, src, fmt, image, pairs, *, allow_pickle, frame, channel) -> dict:
        if fmt == "cellpose-npy" and not allow_pickle:
            raise EngineRefused("`germquant import-labels` was refused by the engine (exit 2):\nREFUSED: "
                                f"{Path(src).name}: a Cellpose _seg.npy is a Python pickle, which can run code when it "
                                "is loaded, so it is read only with --allow-pickle",
                                output="REFUSED: pickle")
        iid = Path(image).name.split(".")[0]
        tool = {"tiff": "tiff", "png": "png", "cellacdc-npz": "cellacdc", "cellpose-npy": "cellpose"}[fmt]
        names = write_pairs(pairs, iid, 1, source=f"import:{tool}")
        warnings = ([f"{Path(image).name} has 2 channels and none was named: the pairs take channel 0 (405); name "
                     f"the channel the labels were drawn on with --channel"] if channel is None else [])
        if fmt == "cellpose-npy":
            warnings.append(f"{Path(src).name}: a _seg.npy file is a Python pickle: loading it can run code; allow "
                            "only files you made")
        return {"image_id": iid, "series_id": iid, "t_index": None, "image": str(image), "image_sha256": "a" * 64,
                "pixels_sha256": "b" * 64, "labels": str(src), "labels_sha256": "c" * 64, "format": fmt,
                "tool": tool, "source": f"import:{tool}", "dims": "2d", "channel_index": 0 if channel is None else 1,
                "channel_name": "405" if channel is None else str(channel), "n_objects": 2, "pairs": names,
                "n_pairs": len(names), "replaced": 0, "pairs_dir": str(pairs),
                "manifest": str(Path(pairs) / "manifest.csv"), "splits_checked": [], "warnings": warnings}

    def export_training(self, results, out, *, orientations=None, slices_per_image=None, seed=None, zone_column=None,
                        min_labels=None, max_gb=None, say=None, should_stop=None) -> dict:
        kw = {"orientations": orientations, "slices_per_image": slices_per_image, "seed": seed,
              "zone_column": zone_column, "min_labels": min_labels, "max_gb": max_gb, "should_stop": should_stop}
        self._start("export_training", (results, out), kw)
        write_pairs(out, "img_a", 2)
        line = f"2 slice pair(s) from 1 run(s) -> {out}"
        if say is not None:
            say(line)
        return {"exit_code": 0, "lines": [line]}

    def train_nuclei(self, pairs, out, *, base="cpsam", name=None, epochs=None, split_by=None, val_frac=None, seed=None,
                     device=None, system=None, kind=None, dataset=None, say=None, should_stop=None) -> dict:
        kw = {"base": base, "name": name, "epochs": epochs, "split_by": split_by, "val_frac": val_frac, "seed": seed,
              "device": device, "system": system, "kind": kind, "dataset": dataset, "should_stop": should_stop}
        self._start("train_nuclei", (pairs, out), kw)
        if say is not None:
            say("0, train_loss=0.5000, test_loss=0.6000, LR=0.000010, time 0.01s")
        name = name or Path(out).name
        model, card = write_model(out, name)
        lines = [f"{name}: trained on 3 slice(s) from 2 image group(s), 1 held out from 1, device cpu",
                 "  hold-out at IoU 0.5: F1 1.000  precision 1.000  recall 1.000  miss rate 0.000  (TP 2 FP 0 FN 0)",
                 f"  model -> {model} (sha256 0123456789ab...)", f"  card  -> {card}"]
        for ln in lines:
            if say is not None:
                say(ln)
        return {"exit_code": 0, "lines": ["loading the pairs ...", *lines], "model_path": str(model),
                "card_path": str(card)}


class ChannelTrainRunner(FakeTrainRunner):
    """A fake engine whose runner passes the import channel (``--channel``): the runner of handoff/
    P6-TRAIN_blocker.md's patch."""

    def import_labels(self, src, fmt, image, pairs, *, allow_pickle=False, frame=None, channel=None,
                      should_stop=None) -> dict:
        self._start("import_labels", (src, fmt, image, pairs), {"allow_pickle": allow_pickle, "frame": frame,
                                                                "channel": channel, "should_stop": should_stop},
                    "import_labels")
        return self._import(src, fmt, image, pairs, allow_pickle=allow_pickle, frame=frame, channel=channel)


class StaticSession(QObject):
    """An engine session around one runner (any object with the runner's methods): what the docks ask a session
    for, without a settings file or a version check."""

    changed = Signal()

    def __init__(self, runner, parent=None) -> None:
        super().__init__(parent)
        self._runner = runner
        self.ensured = 0

    def runner(self):
        return self._runner

    def ensure(self, should_stop=None):
        self.ensured += 1
        return self._runner

    def runner_for(self, pack_id=None, recipe_id=None, should_stop=None):
        return self._runner

    def require(self, *features: str, should_stop=None):
        self._runner.require(*features)
        return self._runner

    def features(self) -> frozenset:
        return self._runner.features()

    def missing(self, *features: str) -> str | None:
        try:
            self._runner.require(*features)
        except FeatureMissing as e:
            return str(e)
        return None

    def devices(self, should_stop=None) -> list[str]:
        return device_choices(self._runner.runtime())

    def device(self) -> str:
        return "auto"

    def describe(self) -> str:
        return f"{self._runner.describe()}; germquant 0.2.0.dev0 (git ffffffffffff), python 3.11; device auto"

    def warning(self) -> str:
        return ""

    def info(self) -> dict:
        return self._runner.version()

    def error(self) -> str | None:
        return None


class FakeHub(QObject):
    """The Project dock's hub (napari_germquant._project_widget.ProjectHub): its signals and attributes, and
    ``reloads`` counting request_reload."""

    opened = Signal(str)
    closed = Signal()
    changed = Signal()
    running_changed = Signal(bool)
    reload_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root: Path | None = None
        self.state = None
        self.running = False
        self.reloads = 0

    def request_reload(self) -> None:
        self.reloads += 1
        self.reload_requested.emit()

    def open(self, root) -> None:
        self.root = Path(root)
        self.opened.emit(str(root))

    def close(self) -> None:
        self.root = None
        self.closed.emit()
