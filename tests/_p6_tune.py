"""Helpers of the Spot tuning dock tests (tranche 5 Train B, unit P6-TUNE; tests/test_tuning.py and
tests/test_tuning_dock.py), kept out of conftest.py (plan section 5.2).

Folders written by hand, shaped as the engine writes them (plan section 9.3; docs/PROJECTS.md and docs/TUNING.md
of the engine):

* ``write_project``: germquant_project.yaml (format 1, the engine's key order) with its images;
* ``write_finished_run``: results/<image_id>/ of one finished run: the label image (int32, the run's grid after
  its stride and z range), the image summary (the voxel columns with the stride in dy and dx, file_path,
  n_nuclei), the spots table, the done marker (config_hash, xy_stride, z_range) and run_manifest.json (the run's
  config: the log backend);
* ``write_source``: the image file itself, a small two-channel OME-TIFF (dna, foci);
* ``write_tuning_run``: tuning/<run_id>/ with ranked.csv, best.yaml, grid.yaml, trials.jsonl and run.json.

The fake engine, ``FakeTuneRunner`` inside ``StaticSession``, answers the runner's tune-spots methods in the dock's
own worker thread: ``tune_spots`` writes a tuning run folder (ranked.csv in the engine's columns) and returns the
command's record, ``tune_apply`` writes recipe.yaml and the manifest's new hash as an apply does and returns the
hash; both can be held on a gate (Cancel, a run that goes) and refuse or fail on request. Every call is recorded
with its keywords. No process is started.

``FakeHub`` has the signals and attributes of the Project dock's hub (napari_germquant._project_widget.ProjectHub,
unit P6-PROJECT): opened(str), closed(), changed(), running_changed(bool), reload_requested(); root, state and
running; request_reload(). The dock is tested against it until the Project dock is in the same branch.

``StaticSession`` wraps any runner, a real SubprocessRunner of this interpreter included (the acceptance test), so a
test sees the engine calls the dock makes. Every file here is ASCII."""
from __future__ import annotations

import datetime as _dt
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qtpy.QtCore import QObject, Signal

from napari_germquant._engine_session import EngineSession
from napari_germquant._runner import (
    EngineRefused,
    EngineRunner,
    FeatureMissing,
    RunCancelled,
    RunnerError,
    SubprocessRunner,
    new_run_id,
)

UM = chr(0xB5) + "m"                      # the OME unit the engine's writers use
MANIFEST = "germquant_project.yaml"
PACK = {"id": "test_pack", "version": "0.1.0", "recipe": "basic", "sha256": "0" * 64}
BASE_HASH = "02bb83cf74f9"
TUNED_HASH = "805dad578390"
SPACING = (0.5, 0.2, 0.2)                 # (dz, dy, dx) of the source files, stride 1
SHAPE = (6, 32, 48)                       # (Z, Y, X) of the source files
# (label, centre z, y, x, radius) in source pixels: two nuclei
NUCLEI = ((1, 2.5, 10, 12, 6), (2, 2.5, 20, 34, 7))
OBJECTIVE = ("per-object Lin's CCC over the annotated objects, rounded to 3 decimals, then point F1 (higher is "
             "better; a missing value ranks last)")
GRID_KEY = "spots.log.min_snr"
GRID_VALUES = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def stamp(plugin: str | None = None) -> dict:
    return {"at": utc_now(), "germquant": "0.2.0.dev0", "git_sha": "f" * 40, "plugin": plugin}


def image_entry(image_id: str, path: str, **extra) -> dict:
    """A manifest images[] entry: {id, path, size, mtime[, timepoints], excluded, reason}."""
    entry = {"id": image_id, "path": str(path), "size": 1234, "mtime": 0.0, **extra}
    entry.setdefault("excluded", False)
    entry.setdefault("reason", None)
    return entry


def write_project(root, images, *, expected: str = BASE_HASH, recipe_file: str | None = None,
                  origins: dict | None = None, name: str = "study") -> Path:
    """germquant_project.yaml as the engine writes it (format 1, its key order), with ``images`` (entries of
    ``image_entry``)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    m = {"format": 1, "name": name, "id": "00000000-0000-4000-8000-000000000000", "created": stamp(),
         "saved": stamp(), "pack": dict(PACK),
         "recipe": {"file": recipe_file, "expected_config_hash": expected, "overrides": {}, "roles": {},
                    "origins": dict(origins or {})},
         "run": {"xy_stride": 1, "z_range": None, "device": None, "runner": None, "interpreters": {}},
         "models": [], "images": [dict(e) for e in images], "samples": None,
         "validated": {"value": False, "why": "not validated: the pack recipe test_pack@0.1.0/basic is provisional",
                       "record": None}}
    (root / MANIFEST).write_bytes(yaml.safe_dump(m, sort_keys=False, allow_unicode=False).encode("ascii"))
    return root


def read_manifest(root) -> dict:
    return yaml.safe_load((Path(root) / MANIFEST).read_text(encoding="ascii"))


def source_stack(shape=SHAPE, nuclei=NUCLEI, spacing=SPACING) -> np.ndarray:
    """(C, Z, Y, X) uint16: a dna channel with ``nuclei`` (balls in microns) and a foci channel with one bright
    voxel near each nucleus's centre."""
    zz, yy, xx = np.indices(shape).astype(np.float32)
    dz, dy, dx = spacing
    dna = np.zeros(shape, np.uint16)
    foci = np.zeros(shape, np.uint16)
    for _lab, cz, cy, cx, r in nuclei:
        d2 = ((zz - cz) * dz / dy) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        dna[d2 <= r * r] = 1000
        foci[int(round(cz)), int(cy), int(cx)] = 4000
    return np.stack([dna, foci], axis=0)


def write_source(path, *, shape=SHAPE, spacing=SPACING) -> Path:
    """A two-channel OME-TIFF (dna, foci) as a microscope export would hold it."""
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dz, dy, dx = spacing
    tifffile.imwrite(str(path), source_stack(shape, spacing=spacing), ome=True, photometric="minisblack",
                     metadata={"axes": "CZYX", "PhysicalSizeX": dx, "PhysicalSizeXUnit": UM, "PhysicalSizeY": dy,
                               "PhysicalSizeYUnit": UM, "PhysicalSizeZ": dz, "PhysicalSizeZUnit": UM,
                               "Channel": {"Name": ["dna", "foci"]}})
    return path


def run_labels(*, shape=SHAPE, nuclei=NUCLEI, xy_stride: int = 1, z_range=None, spacing=SPACING) -> np.ndarray:
    """The label image a run on the source makes, on the run's grid (its stride and z range)."""
    zz, yy, xx = np.indices(shape).astype(np.float32)
    dz, dy, _dx = spacing
    full = np.zeros(shape, np.int32)
    for lab, cz, cy, cx, r in nuclei:
        full[((zz - cz) * dz / dy) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = lab
    z0, z1 = (0, shape[0]) if z_range is None else (int(z_range[0]), int(z_range[1]))
    return np.ascontiguousarray(full[z0:z1, ::xy_stride, ::xy_stride])


def write_finished_run(root, image_id: str, source, *, config_hash: str = BASE_HASH, xy_stride: int = 1,
                       z_range=None, backend: str = "log", spacing=SPACING, planar: bool = False,
                       labels: np.ndarray | None = None) -> Path:
    """results/<image_id>/ of a finished run of ``source`` (see the module docstring). The tables record the
    run's own voxel size: dy and dx times the stride (a single-plane run: voxel_dz_um NaN)."""
    import tifffile

    out = Path(root) / "results" / image_id
    out.mkdir(parents=True, exist_ok=True)
    lab = run_labels(xy_stride=xy_stride, z_range=z_range, spacing=spacing) if labels is None else labels
    tifffile.imwrite(str(out / f"{image_id}__nuclei_labels.tif"), lab.astype(np.int32))
    dz, dy, dx = spacing
    rdz, rdy, rdx = (float("nan") if planar else dz), dy * xy_stride, dx * xy_stride
    n = int(lab.max())
    pd.DataFrame([{"n_nuclei": n, "mean_spots": 1.0, "qc_pass": True, "image_id": image_id,
                   "voxel_dz_um": rdz, "voxel_dy_um": rdy, "voxel_dx_um": rdx, "file_path": str(source),
                   "segmentation_method": "classical"}]).to_csv(out / f"{image_id}__image_summary.csv", index=False)
    pd.DataFrame([{"nucleus_id": i, "volume_um3": 10.0, "image_id": image_id} for i in range(1, n + 1)]).to_csv(
        out / f"{image_id}__nuclei.csv", index=False)
    spots = []
    for k, (labv, cz, cy, cx, _r) in enumerate(NUCLEI[:n]):
        z0 = 0 if z_range is None else int(z_range[0])
        spots.append({"spot_id": k, "nucleus_id": labv, "marker": "foci",
                      "z_um": 0.0 if planar else (round(cz) - z0) * dz, "y_um": cy * dy, "x_um": cx * dx,
                      "effect_size": 30.0, "detector": backend, "image_id": image_id})
    pd.DataFrame(spots).to_csv(out / f"{image_id}__spots.csv", index=False)
    marker = {"image_id": image_id, "config_hash": config_hash, "enabled_stages": ["read", "segment", "spots"],
              "stage_hashes": {"read": "0" * 12, "segment": "1" * 12, "spots": "2" * 12}, "xy_stride": xy_stride,
              "z_range": list(z_range) if z_range else None, "failed_stages": [],
              "finished": "2026-10-24T12:00:00+00:00"}
    if planar:
        marker["planar"] = True
    (out / f"{image_id}__done.json").write_text(json.dumps(marker, indent=1), encoding="ascii")
    config = {"spots": {"backend": backend, "spot_radius_um": 0.3, "log": {"min_snr": 2.0, "diameter_um": 0.5}}}
    (out / "run_manifest.json").write_text(json.dumps({"config_hash": config_hash, "config": config,
                                                       "xy_stride": xy_stride}, indent=1), encoding="ascii")
    return out


def ranked_rows(values=GRID_VALUES, key: str = GRID_KEY) -> list[dict]:
    """ranked.csv rows of a grid of one key, best first, as the engine writes them."""
    rows = []
    for r, v in enumerate(values, start=1):
        rows.append({"rank": r, "objective": "object_ccc", "objective_value": round(1.0 - 0.1 * (r - 1), 3),
                     "tie_break": "point_f1", "tie_break_value": round(0.95 - 0.05 * (r - 1), 3), key: v})
    return rows


def write_tuning_run(root, run_id: str | None = None, *, rows=None, best: dict | None = None, backend: str = "log",
                     key: str = GRID_KEY, warnings=()) -> Path:
    """tuning/<run_id>/ as a search leaves it (docs/TUNING.md, "The run folder"); ``best`` None writes no
    best.yaml, as a search that scored nothing."""
    run_id = run_id or new_run_id()
    out = Path(root) / "tuning" / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows = ranked_rows(key=key) if rows is None else rows
    values = [r[key] for r in rows]
    (out / "grid.yaml").write_text(yaml.safe_dump({key: sorted(values)}, sort_keys=False), encoding="ascii")
    cols = ["rank", "objective", "objective_value", "tie_break", "tie_break_value", key]
    (out / "ranked.csv").write_text(pd.DataFrame(rows, columns=cols).to_csv(index=False, lineterminator="\n"),
                                    encoding="ascii")
    if best is not None:
        (out / "best.yaml").write_text(yaml.safe_dump({"objective": OBJECTIVE, "backend": backend, "keys": best},
                                                      sort_keys=False), encoding="ascii")
    (out / "trials.jsonl").write_text("".join(json.dumps({"trial": i, "params": {key: v}}) + "\n"
                                              for i, v in enumerate(values)), encoding="ascii")
    (out / "run.json").write_text(json.dumps({"format": 1, "run_id": run_id, "backend": backend,
                                              "warnings": list(warnings)}, indent=1), encoding="ascii")
    return out


class FakeTuneRunner(EngineRunner):
    """A fake engine for the Spot tuning dock (module docstring). ``features`` the engine has; ``tune_refusal`` or
    ``apply_refusal`` (a text) make the call refused (EngineRefused, exit 2), ``tune_error`` a failed search
    (RunnerError); ``scored_nothing`` a search that ran but scored nothing (best None); ``gate`` (a
    threading.Event) holds tune_spots until it is set, or until should_stop asks (then RunCancelled);
    ``new_hash`` is the config hash an apply writes."""

    kind = "fake"
    via = "gui-subprocess"
    python = "fake-engine-python"
    cancellable = True

    def __init__(self, *, features=("packs", "projects", "tune_spots"), new_hash: str = TUNED_HASH) -> None:
        super().__init__()
        self.have = frozenset(features)
        self.new_hash = new_hash
        self.calls: list[tuple[str, dict]] = []
        self.tune_refusal: str | None = None
        self.apply_refusal: str | None = None
        self.tune_error: str | None = None
        self.scored_nothing = False
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.threads: list[str] = []
        self.rows = ranked_rows()
        self.best = {GRID_KEY: 1.5}

    def describe(self) -> str:
        return "a fake engine (tests/_p6_tune.py)"

    def version(self, refresh: bool = False, should_stop=None) -> dict:
        return {"germquant": "0.2.0.dev0", "git_sha": "f" * 40, "python": "3.11", "executable": self.python,
                "features": sorted(self.have)}

    def cached_version(self) -> dict:
        return self.version()

    def feature_message(self, feature: str) -> str:
        from napari_germquant import _engine

        return _engine.feature_missing_message(feature, info=self.version(), interpreter=self.python)

    def tune_spots(self, project, *, points=None, grid=None, max_runs=None, out=None, say=None, should_stop=None):
        self.calls.append(("tune_spots", {"project": str(project), "points": points, "grid": grid,
                                          "max_runs": max_runs, "out": out}))
        self.threads.append(threading.current_thread().name)
        self.require("tune_spots")
        self.started.set()
        if say is not None:
            say("INFO germquant.tune.spots: tune: img_a (1/1), set 1 of 6: 2 spots (0.1 s)")
        if self.gate is not None:
            while not self.gate.wait(0.05):
                if should_stop is not None and should_stop():
                    raise RunCancelled("cancelled by the user: the fake engine was ended")
        if self.tune_refusal:
            raise EngineRefused(f"`germquant tune-spots` was refused by the engine (exit 2):\nERROR: {self.tune_refusal}",
                                output=self.tune_refusal)
        if self.tune_error:
            raise RunnerError(f"`germquant tune-spots` failed (exit 1): {self.tune_error}")
        run = write_tuning_run(project, rows=self.rows, best=None if self.scored_nothing else self.best,
                               warnings=["a warning of the search"])
        best = None if self.scored_nothing else {"objective": OBJECTIVE, "backend": "log", "keys": dict(self.best)}
        warnings = ["a warning of the search"] + (["no parameter set could be scored"] if self.scored_nothing else [])
        return {"run_dir": str(run), "best": best, "ranked": list(self.rows), "warnings": warnings}

    def tune_apply(self, project, run_dir, *, plugin_version=None, should_stop=None):
        self.calls.append(("tune_apply", {"project": str(project), "run_dir": str(run_dir),
                                          "plugin_version": plugin_version}))
        self.require("tune_spots")
        if self.apply_refusal:
            raise EngineRefused(f"`germquant tune-spots` was refused by the engine (exit 2):\nERROR: {self.apply_refusal}",
                                output=self.apply_refusal)
        best = yaml.safe_load((Path(run_dir) / "best.yaml").read_text(encoding="ascii"))
        root = Path(project)
        (root / "recipe.yaml").write_text("extends: pack:test_pack@0.1.0/basic\n" + yaml.safe_dump(
            _nested(best["keys"]), sort_keys=False), encoding="ascii")
        m = read_manifest(root)
        m["recipe"].update({"file": "recipe.yaml", "expected_config_hash": self.new_hash,
                            "origins": {k: "tuning" for k in best["keys"]}})
        m["saved"] = stamp(plugin_version)
        (root / MANIFEST).write_bytes(yaml.safe_dump(m, sort_keys=False, allow_unicode=False).encode("ascii"))
        return {"config_hash": self.new_hash, "exit_code": 0, "lines": [f"applied {run_dir}: config_hash {self.new_hash}"]}

    def release(self) -> None:
        if self.gate is not None:
            self.gate.set()


def _nested(keys: dict) -> dict:
    out: dict = {}
    for dotted, value in keys.items():
        node = out
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


class RecordingRunner(SubprocessRunner):
    """A real subprocess engine (this interpreter) that also records the tuning calls the dock makes with their
    keywords (``calls``), so a test sees what the dock asked of the engine besides the command lines
    (``history``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[str, dict]] = []

    def tune_spots(self, project, **kw):
        self.calls.append(("tune_spots", {"project": str(project), **{k: v for k, v in kw.items()
                                                                      if k not in ("say", "should_stop")}}))
        return super().tune_spots(project, **kw)

    def tune_apply(self, project, run_dir, **kw):
        self.calls.append(("tune_apply", {"project": str(project), "run_dir": str(run_dir),
                                          **{k: v for k, v in kw.items() if k != "should_stop"}}))
        return super().tune_apply(project, run_dir, **kw)


class StaticSession(EngineSession):
    """An engine session whose engine is ``runner`` (no settings read, no version check): the dock's seam for
    the fake engine and for a recording engine. ``missing_feature`` makes every require of it fail as an engine
    without that feature would."""

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

    def missing(self, *features):
        if self.missing_feature is not None and self.missing_feature in features:
            return self.fake.feature_message(self.missing_feature)
        return super().missing(*features)

    def describe(self) -> str:
        return f"{self.fake.describe()}; device {self.device()}"

    def info(self):
        return self.fake.version()


class FakeHub(QObject):
    """The Project dock's hub as napari_germquant._project_widget.ProjectHub has it (module docstring)."""

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
        self.changed.emit()

    def close(self) -> None:
        self.root = None
        self.closed.emit()

    def set_running(self, running: bool) -> None:
        self.running = bool(running)
        self.running_changed.emit(bool(running))


def standard_project(root, *, stride: int = 2, z_range=(1, 5)) -> dict:
    """A project of five images: img_a finished (at ``stride`` and ``z_range``) with tester's truth points saved,
    img_b finished without points, img_c not run yet, img_x excluded and img_t a time series. Returns the paths."""
    root = Path(root)
    images = root / "images"
    src = {i: write_source(images / f"{i}.ome.tif") for i in ("img_a", "img_b", "img_c")}
    entries = [image_entry("img_a", "images/img_a.ome.tif"), image_entry("img_b", "images/img_b.ome.tif"),
               image_entry("img_c", "images/img_c.ome.tif"),
               image_entry("img_x", "images/img_x.ome.tif", excluded=True, reason="out of focus"),
               image_entry("img_t", "images/img_t.ome.tif", timepoints=5)]
    write_project(root, entries)
    write_finished_run(root, "img_a", src["img_a"], xy_stride=stride, z_range=z_range)
    write_finished_run(root, "img_b", src["img_b"])
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    header = "point_id,z_um,y_um,x_um,zi,yi,xi,object_id,projected\n"
    (ann / "img_a__points_tester.csv").write_text(
        header + "0,1.0,2.0,2.4,2,10,12,1,false\n1,1.0,4.0,6.8,2,20,34,2,false\n", encoding="ascii")
    return {"root": root, "sources": src}
