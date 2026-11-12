"""The Spot tuning dock's model (no Qt): a project's images with their finished runs and truth points, an image
opened on its run's grid, the truth-point files, what a search means, and the tuning runs a project holds
(docs/DESIGN_system_packs.md 8.7, 6.8 and 10 step 9; the engine's docs/TUNING.md; tranche 5 Train B plan, unit
P6-TUNE, decisions D24, D38, D39 and D42).

Spots are tuned, not trained: the engine's ``germquant tune-spots`` runs the spot detection again on finished
runs, once per setting of a grid, scores each setting against truth points and ranks them; ``--apply`` writes
the best setting into the project's recipe.yaml. The dock runs both through the engine runner (the numbers are
made in the engine's environment). What it does in napari's own process are file reads and one file write:

* the project's manifest (germquant_project.yaml) and each image's results (results/<image_id>/), read and
  closed at once;
* the truth-point file, annotations/<image_id>__points_<annotator>.csv, written through the engine's own writer,
  ``germquant.tune.points.write_points`` (plan D39: annotation points are written in napari's process through the
  engine's writers; importing germquant.tune loads nothing heavy, and the plugin never imports germquant.benchmark
  itself). A germquant without germquant.tune (``points_problem``) cannot save points, and says so;
* the tuning runs, tuning/<run_id>/ (ranked.csv, best.yaml, run.json), read to show the top five settings.

An image is opened on its run's own grid: the run's label image, spots and spots image (the Results loader,
``_results.load_results_layers``), the source image read as the run read it (its stride and z range, so plane 0 of
every layer is the run's first plane), and one editable Points layer, "<image_id> truth points (<annotator>)", in
the run's micrometres (napari's world coordinates of those layers; the layer has no scale). The points of an
existing file are put on the run's grid with ``germquant.tune.points.to_run_frame``; Save gives the layer's points
to ``write_points`` with the run's labels, voxel size and geometry, which fills every column of the file (its own
grid and frame, the run's object under each point, projected).

Save refuses what would put a wrong file on disk: the points of other planes in a run of one plane (they are left
out of the layer, and a Save would remove them from the file), and a run whose grid changed since the layer was
opened (the image was run again with another stride, z range or voxel size: ``RunChanged``). The label image is
decoded once for the Saves of one image (``run_labels``).
"""
from __future__ import annotations

from . import _engine  # noqa: F401,I001  (import order: germquant.api before anything else)

import csv
import functools
import importlib
import io
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._reader import check_single_timepoint, layers_from_stack
from ._results import ResultsSet, load_results_layers, spot_marker_size_um

MANIFEST_NAME = "germquant_project.yaml"
RESULTS_DIR = "results"
POINTS_DIR = "annotations"
TUNING_DIR = "tuning"
FEATURE = "tune_spots"
TOP_N = 5
MAX_RUNS = 50
# the metadata key of a truth-point layer: which project, image, annotator and file it belongs to, and the run's
# grid it was opened on (what Save needs)
LAYER_KEY = "napari_germquant_tuning"
# a colour no layer of a run uses for its detections (yellow spots; cyan, magenta, orange and lime for the extra
# spot instances, _results.INSTANCE_COLORS), so a truth point never looks like a detection
TRUTH_COLOR = "red"
BACKEND_WORDS = {"log": ("LoG detection", "LoG detections"), "spotmax": ("SpotMAX run", "SpotMAX runs")}
# the engine's file names (germquant.tune.points and germquant.tune.spots)
POINTS_NAME_RE = re.compile(r"(?P<image_id>.+)__points_(?P<annotator>[A-Za-z0-9_-]{1,32})\.csv")
RUN_ID_RE = re.compile(r"\d{8}T\d{6}Z_[0-9a-f]{8}")


class TuningError(RuntimeError):
    """What the dock cannot do, with the reason and the fix (shown as it is)."""


class RunChanged(TuningError):
    """The image's run changed its grid (stride, z range, label image or voxel size) since its truth layer was
    opened: the layer's points sit on the old grid, so they are not saved on the new one."""


@functools.lru_cache(maxsize=None)
def tune_module(name: str):
    """``germquant.tune.<name>`` of this napari environment's germquant, or None when it has no germquant.tune
    (an engine from before tranche 5 Train B). Kept once looked up: the buttons ask at every refresh, and Python
    keeps no record of an import that failed."""
    try:
        return importlib.import_module(f"germquant.tune.{name}")
    except ImportError:
        return None


def points_problem() -> str | None:
    """Why this napari environment cannot write or read truth points (its germquant has no germquant.tune), or
    None when it can: the one message of the missing feature."""
    if tune_module("points") is None:
        return _engine.feature_missing_message(FEATURE)
    return None


def _long(path) -> str:
    from germquant.fsutil import long_path

    return long_path(Path(path))


def _isfile(path) -> bool:
    try:
        return os.path.isfile(_long(path))
    except OSError:                                   # a share that is not there
        return False


def _isdir(path) -> bool:
    try:
        return os.path.isdir(_long(path))
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------------
# the project
# ---------------------------------------------------------------------------------------------------


def read_manifest(root) -> dict:
    """germquant_project.yaml of ``root``, read and closed at once (an open handle would make the engine's
    os.replace of it fail on Windows). TuningError for a folder that is no germquant project."""
    import yaml

    path = Path(root) / MANIFEST_NAME
    try:
        with open(_long(path), encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise TuningError(f"{root} is not a germquant project (no {MANIFEST_NAME}): open the project's folder, or "
                          f"make one in the Project dock (germquant project new)") from None
    except (OSError, UnicodeDecodeError) as e:
        raise TuningError(f"{path} cannot be read: {e}") from None
    except yaml.YAMLError as e:
        raise TuningError(f"{path} is not YAML: {' '.join(str(e).split())}") from None
    if not isinstance(data, dict) or data.get("format") != 1 or not isinstance(data.get("images", []), list):
        raise TuningError(f"{path} is not a germquant project manifest of format 1")
    return data


def image_path(root, entry: dict) -> Path | None:
    """An image's file: a path inside the project is stored relative with forward slashes, any other absolute
    (a UNC path verbatim)."""
    raw = entry.get("path")
    if not raw:
        return None
    p = Path(str(raw))
    return p if p.is_absolute() or str(raw).startswith(("\\\\", "//")) else Path(root) / p


def pack_text(manifest: dict) -> str:
    p = manifest.get("pack") or {}
    return f"{p.get('id')}@{p.get('version')} {p.get('recipe')}" if p.get("id") else "no pack recorded"


def project_hash(manifest: dict) -> str | None:
    """The config hash the project had when it was last written (the manifest's recipe.expected_config_hash)."""
    value = (manifest.get("recipe") or {}).get("expected_config_hash")
    return str(value) if value else None


def points_files(root) -> dict[str, dict[str, Path]]:
    """{image_id: {annotator: file}} of the truth-point files in the project's annotations/ folder."""
    folder = Path(root) / POINTS_DIR
    out: dict[str, dict[str, Path]] = {}
    if not _isdir(folder):
        return out
    try:
        names = sorted(os.listdir(_long(folder)))
    except OSError:
        return out
    for name in names:
        m = POINTS_NAME_RE.fullmatch(name)
        if m and _isfile(folder / name):
            out.setdefault(m.group("image_id"), {})[m.group("annotator")] = folder / name
    return out


def count_points(path) -> int:
    """How many points a truth-point file holds (its data rows); 0 for a file that cannot be read."""
    try:
        with open(_long(path), encoding="utf-8-sig", newline="") as fh:
            return max(0, sum(1 for row in csv.reader(fh) if row) - 1)
    except OSError:
        return 0


@dataclass
class ImageRow:
    """One image of the project, as the dock lists it."""

    image_id: str
    path: Path | None
    excluded: bool = False
    reason: str | None = None
    timepoints: int = 1
    finished: bool = False
    run_hash: str | None = None
    project_hash: str | None = None
    points: int = 0                                   # the points of the dock's annotator
    annotators: list[str] = field(default_factory=list)    # every annotator with a file for this image
    text: str = ""

    @property
    def usable(self) -> bool:
        """Spot tuning takes this image: not excluded, not a time series."""
        return not self.excluded and self.timepoints <= 1

    @property
    def stale(self) -> bool:
        """The run was made with another config than the project's now (an Apply since, or another edit)."""
        return bool(self.finished and self.run_hash and self.project_hash and self.run_hash != self.project_hash)


def result_dir(root, image_id: str) -> Path:
    return Path(root) / RESULTS_DIR / image_id


def finished_run(root, image_id: str) -> dict | None:
    """The done marker of an image's run when its run finished with its label image, else None."""
    rs = ResultsSet(result_dir(root, image_id), image_id)
    marker = rs.done_marker()
    if not isinstance(marker, dict) or not _isfile(rs.file("nuclei_labels.tif")):
        return None
    return marker


def _row_text(row: ImageRow, annotator: str | None) -> str:
    if row.excluded:
        return f"{row.image_id}: excluded ({row.reason or 'no reason recorded'})"
    if row.timepoints > 1:
        return f"{row.image_id}: a time series ({row.timepoints} time points): spot tuning does not take time series in " \
               f"this release"
    if not row.finished:
        return f"{row.image_id}: no finished run: run the project first (Project dock, or germquant project run)"
    run = "finished"
    if row.stale:
        run = f"finished, stale (run with config {row.run_hash}; the project is now {row.project_hash})"
    if annotator and row.points:
        pts = f"{row.points} truth point{'s' if row.points != 1 else ''} ({annotator})"
    elif annotator:
        pts = f"no truth points yet ({annotator})"
    else:
        pts = "no annotator chosen"
    others = [a for a in row.annotators if a != annotator]
    if others:
        pts += f"; also annotated by {', '.join(others)}"
    return f"{row.image_id}: {run}; {pts}"


def project_rows(root, manifest: dict, annotator: str | None) -> list[ImageRow]:
    """The project's images in the manifest's order, each with its run (finished or not, stale or not) and its
    truth points (the annotator's count, and every annotator with a file)."""
    files = points_files(root)
    now = project_hash(manifest)
    rows = []
    for entry in manifest.get("images") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        image_id = str(entry["id"])
        row = ImageRow(image_id=image_id, path=image_path(root, entry), excluded=bool(entry.get("excluded")),
                       reason=entry.get("reason"), timepoints=int(entry.get("timepoints") or 1), project_hash=now)
        marker = finished_run(root, image_id) if row.usable else None
        if marker is not None:
            row.finished = True
            row.run_hash = str(marker.get("config_hash") or "") or None
        mine = files.get(image_id, {})
        row.annotators = sorted(mine)
        if annotator and annotator in mine:
            row.points = count_points(mine[annotator])
        row.text = _row_text(row, annotator)
        rows.append(row)
    return rows


def annotated(rows: list[ImageRow]) -> list[str]:
    """The images a search of annotations/ takes: usable, with a finished run and a truth-point file."""
    return [r.image_id for r in rows if r.usable and r.finished and r.annotators]


def several_annotators(rows: list[ImageRow]) -> list[str]:
    """Images with truth points of more than one annotator in annotations/ (a search refuses them: it scores
    against one set of points)."""
    return [r.image_id for r in rows if len(r.annotators) > 1]


# ---------------------------------------------------------------------------------------------------
# one finished run: its grid, and the image opened on it
# ---------------------------------------------------------------------------------------------------


@dataclass
class RunInfo:
    """What the dock needs of an image's finished run: its folder, the image's file, the voxel size of its
    layers (``spacing``: a single plane's dz is 1.0, as the Results loader shows it) and as its tables record it
    (``table_spacing``: a single plane's dz NaN), its stride and z range, whether it is one plane, its config and
    the spot backend that config names."""

    root: Path
    image_id: str
    result_dir: Path
    source: Path | None
    spacing: tuple[float, float, float]
    table_spacing: tuple[float, float, float]
    xy_stride: int
    z_range: tuple[int, int] | None
    planar: bool
    config: dict | None
    config_hash: str | None

    @property
    def backend(self) -> str | None:
        spots = (self.config or {}).get("spots")
        return str(spots.get("backend") or "spotmax") if isinstance(spots, dict) else None

    def meta(self, annotator: str, file: Path, projected: bool, *, labels_shape=None, other_planes: int = 0,
             run_stamp=None) -> dict:
        """The truth layer's record (LAYER_KEY): enough to save it again, and to tell at Save whether the run
        still has the grid the layer was opened on (``labels_shape``, the label image's (Z, Y, X)).
        ``other_planes`` counts the file's points that a one-plane run cannot hold (``read_truth``); ``run_stamp``
        names the run the layer was opened on (``run_stamp``); ``box``, set by the dock, is the "placed on a
        projection" box as the user left it for this layer (``projected`` is what the file holds)."""
        return {"root": str(self.root), "image_id": self.image_id, "annotator": annotator, "file": str(file),
                "result_dir": str(self.result_dir), "table_spacing": [float(s) for s in self.table_spacing],
                "xy_stride": int(self.xy_stride), "z_range": list(self.z_range) if self.z_range else None,
                "labels_shape": [int(v) for v in labels_shape] if labels_shape is not None else None,
                "planar": bool(self.planar), "projected": bool(projected), "other_planes": int(other_planes),
                "run_stamp": list(run_stamp) if run_stamp is not None else None}


def _table_spacing(rs: ResultsSet) -> tuple[float, float, float]:
    df = rs.table("image_summary")
    cols = ("voxel_dz_um", "voxel_dy_um", "voxel_dx_um")
    if df is not None and len(df) and set(cols) <= set(df.columns):
        try:
            return tuple(float(df[c].iloc[0]) for c in cols)   # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    return rs.spacing()


def run_info(root, row: ImageRow) -> RunInfo:
    """The finished run of ``row``'s image; TuningError when it has none (or no label image)."""
    if not row.usable:
        raise TuningError(_row_text(row, None))
    marker = finished_run(root, row.image_id)
    if marker is None:
        raise TuningError(f"{row.image_id} has no finished run with its label image in {result_dir(root, row.image_id)}: "
                          f"run the project first")
    rs = ResultsSet(result_dir(root, row.image_id), row.image_id)
    zr = marker.get("z_range")
    z_range = (int(zr[0]), int(zr[1])) if isinstance(zr, (list, tuple)) and len(zr) == 2 else None
    table = _table_spacing(rs)
    planar = bool(marker.get("planar")) or (math.isnan(table[0]) if isinstance(table[0], float) else False)
    source = row.path if row.path is not None else rs.source_path()
    return RunInfo(root=Path(root), image_id=row.image_id, result_dir=rs.folder, source=source, spacing=rs.spacing(),
                   table_spacing=table, xy_stride=int(marker.get("xy_stride") or 1), z_range=z_range, planar=planar,
                   config=rs.run_config(), config_hash=rs.config_hash())


def labels_file(result_dir, image_id: str) -> Path:
    return Path(result_dir) / f"{image_id}__nuclei_labels.tif"


def tiff_shape(path) -> tuple[int, int, int]:
    """The (Z, Y, X) shape of a label image, from its header (no pixels read; a single plane's YX image is one
    plane)."""
    import tifffile

    with tifffile.TiffFile(_long(path)) as tf:
        shape = tuple(int(s) for s in tf.series[0].shape)
    return (1, *shape) if len(shape) == 2 else shape   # type: ignore[return-value]


def labels_shape(info: RunInfo) -> tuple[int, int, int]:
    """The (Z, Y, X) shape of the run's label image, from its header."""
    return tiff_shape(labels_file(info.result_dir, info.image_id))


def run_stamp(info: RunInfo, shape=None) -> list:
    """What tells one run of an image from another for the layers it opens with: its stride, z range and label
    image shape, the label image file's modification time and size (a run made again writes it again) and the
    source file. A dock shows the run's source and results layers once for all its annotators only while the
    image's run has the same stamp."""
    st = os.stat(_long(labels_file(info.result_dir, info.image_id)))
    shape = labels_shape(info) if shape is None else shape
    return [int(info.xy_stride), list(info.z_range) if info.z_range else None, [int(v) for v in shape],
            int(st.st_mtime_ns), int(st.st_size), str(info.source) if info.source is not None else None]


# The run's label image, decoded once for the Saves of one image: the engine writes it zlib-compressed, so it
# cannot be memory-mapped, and at stride 1 a large gonad decodes to more than a gigabyte. One image is kept,
# keyed by its file's path, size and modification time, so a run made again since is read again.
_LABELS_LOCK = threading.Lock()
_LABELS: dict[tuple, np.ndarray] = {}


def run_labels(path) -> np.ndarray:
    """The label image ``path`` (read only), decoded once while its file stays the same."""
    import tifffile

    lp = _long(path)
    st = os.stat(lp)
    key = (os.path.normcase(os.path.abspath(str(path))), st.st_mtime_ns, st.st_size)
    with _LABELS_LOCK:
        hit = _LABELS.get(key)
    if hit is not None:
        return hit
    labels = np.asarray(tifffile.imread(lp))
    labels.setflags(write=False)
    with _LABELS_LOCK:
        _LABELS.clear()
        _LABELS[key] = labels
    return labels


def forget_labels() -> None:
    """Let the kept label image go (the dock's project closed, or another one was taken)."""
    with _LABELS_LOCK:
        _LABELS.clear()


def source_layers(info: RunInfo) -> list:
    """The source image read as the run read it, at its stride and on its planes, placed on the run's grid (the
    labels' scale), one Image layer per channel. TuningError when the file is not there or cannot be read."""
    from germquant import api

    src = info.source
    if src is None:
        raise TuningError(f"{info.image_id}: the project records no file for this image")
    if not (_isfile(src) or _isdir(src)):
        raise TuningError(f"{info.image_id}: the source image {src} was not found (is the NAS or VPN connected?); "
                          f"the run's layers are open without it")
    meta = api.read_image_metadata(src)
    check_single_timepoint(src, meta)
    stack = api.read_image(src, xy_stride=info.xy_stride, z_range=info.z_range)
    return layers_from_stack(stack, src, meta, scale=info.spacing)


def truth_name(image_id: str, annotator: str) -> str:
    return f"{image_id} truth points ({annotator})"


def points_file(root, image_id: str, annotator: str) -> Path:
    """annotations/<image_id>__points_<annotator>.csv (the engine's points_file_name checks both names when this
    environment has it)."""
    pts = tune_module("points")
    name = pts.points_file_name(image_id, annotator) if pts is not None else f"{image_id}__points_{annotator}.csv"
    return Path(root) / POINTS_DIR / name


def other_planes_text(image_id: str, n: int, file_name: str) -> str:
    """Why the dock does not save a one-plane run's truth layer when the file holds points of other planes."""
    many = n != 1
    return (f"{n} truth point{'s' if many else ''} of {file_name} lie{'' if many else 's'} on "
            f"{'planes' if many else 'a plane'} this one-plane run did not read, so {'they are' if many else 'it is'} "
            f"not shown, and saving here would remove {'them' if many else 'it'} from the file: annotate {image_id} "
            f"on a run of its planes, or remove {'those points' if many else 'that point'} from the file first")


def read_truth(info: RunInfo, annotator: str, shape=None) -> tuple[np.ndarray, bool, str, int]:
    """The annotator's truth points of the image on the run's frame: ``(points (N, 3) in um, projected, note,
    other_planes)``. No file: no points. A file this environment cannot read (no germquant.tune) or that is
    malformed: no points, and the note says why. ``shape`` is the run's label image (Z, Y, X) (read from its
    header when None).

    A run of one plane puts every point on its plane (the engine's to_run_frame and write_points), so the points
    of the file's other planes are left out of the layer and counted in ``other_planes``: shown on this plane
    they would be in the wrong place, and a Save would move them there (``save_truth`` refuses instead)."""
    path = points_file(info.root, info.image_id, annotator)
    empty = np.zeros((0, 3), float)
    if not _isfile(path):
        return empty, False, "", 0
    pts = tune_module("points")
    if pts is None:
        return empty, False, f"{path.name} is there, but {points_problem()}", 0
    try:
        shape = labels_shape(info) if shape is None else tuple(int(v) for v in shape)
        table = pts.read_points(path)
        frame = pts.to_run_frame(table, shape=shape, spacing=info.table_spacing, xy_stride=info.xy_stride,
                                 z_range=info.z_range)
    except (ValueError, OSError) as e:
        return empty, False, f"{path.name} cannot be read ({e}); fix or remove it before saving over it", 0
    n = len(table)
    keep = np.ones(n, bool)
    if shape[0] == 1 and n:
        keep = np.asarray(frame["zr"]) == 0             # the points of the run's own plane
    other = int((~keep).sum())
    data = (np.column_stack([frame["z_um"], frame["y_um"], frame["x_um"]]).astype(float)[keep] if n else empty)
    flags = table["projected"].to_numpy(bool)[keep] if n else np.zeros(0, bool)
    projected = bool(len(flags) and flags.all())
    notes = []
    if other:
        notes.append(f"{other} of its points {'lies' if other == 1 else 'lie'} on planes this one-plane run did not "
                     f"read: {'it is' if other == 1 else 'they are'} not shown, and Save is refused while the file "
                     f"holds {'it' if other == 1 else 'them'} (annotate the image on a run of its planes)")
    outside = int((~np.asarray(frame["inside"], bool) & keep).sum()) if n else 0
    if outside:
        notes.append(f"{outside} of its points {'lies' if outside == 1 else 'lie'} outside this run's field (planes "
                     f"or pixels the run did not read)")
    if len(flags) and flags.any() and not flags.all():
        notes.append("some of its points are marked projected and some not; Save marks every point as the "
                     "'placed on a projection' box says")
    return data, projected, "; ".join(notes), other


def truth_layer(info: RunInfo, annotator: str, data: np.ndarray, projected: bool, *, labels_shape=None,
                other_planes: int = 0, run_stamp=None) -> tuple:
    """The editable Points layer of the annotator's truth points on the run's frame (no scale: its coordinates
    are the run's micrometres, napari's world coordinates of the run's layers)."""
    size = spot_marker_size_um(info.config, "spots")
    file = points_file(info.root, info.image_id, annotator)
    record = info.meta(annotator, file, projected, labels_shape=labels_shape, other_planes=other_planes,
                       run_stamp=run_stamp)
    kwargs = {"name": truth_name(info.image_id, annotator), "ndim": 3, "size": size, "face_color": TRUTH_COLOR,
              "border_color": "white", "metadata": {LAYER_KEY: record}}
    return (np.asarray(data, float).reshape(-1, 3), kwargs, "points")


def image_layers(info: RunInfo, annotator: str, *, have=None) -> dict:
    """The layers an image opens with, bottom first: the source image (when its file is there), the run's
    results and the truth layer. ``have`` names the run (``run_stamp``) of the image's source and results layers
    a dock has in the viewer already, for another annotator ({"source": stamp, "results": stamp}): a part whose
    stamp is this run's is not read again. Returns {"source": [layer tuples], "results": [...], "truth": layer
    tuple, "data": the truth points it starts with, "note": notes, "stamp": this run's stamp, "read_source" and
    "read_results": whether those parts were read}."""
    have = have or {}
    shape = labels_shape(info)
    stamp = run_stamp(info, shape)
    read_source, read_results = have.get("source") != stamp, have.get("results") != stamp
    notes = []
    src: list = []
    if read_source:
        try:
            src = source_layers(info)
        except TuningError as e:
            notes.append(str(e))
        except Exception as e:  # noqa: BLE001 - an unreadable source never stops the run's own layers
            notes.append(f"{info.image_id}: the source image {info.source} cannot be read ({type(e).__name__}: {e})")
    res = (load_results_layers(ResultsSet(info.result_dir, info.image_id), include_source=False)
           if read_results else [])
    data, projected, note, other = read_truth(info, annotator, shape)
    if note:
        notes.append(note)
    truth = truth_layer(info, annotator, data, projected, labels_shape=shape, other_planes=other, run_stamp=stamp)
    return {"source": src, "results": res, "truth": truth, "data": data, "note": "; ".join(notes), "stamp": stamp,
            "read_source": read_source, "read_results": read_results}


def _z_range(value) -> tuple[int, int] | None:
    return (int(value[0]), int(value[1])) if isinstance(value, (list, tuple)) and len(value) == 2 else None


def _same_spacing(a, b) -> bool:
    try:
        return all((math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
                   for x, y in zip((float(v) for v in a), (float(v) for v in b), strict=True))
    except (TypeError, ValueError):
        return False


def run_changed(meta: dict) -> str | None:
    """What changed in the grid of the image's run since its truth layer was opened (``meta``, the layer's
    record): its stride, z range, label image or voxel size; None when nothing did. TuningError when the image
    has no finished run now (it is being run again, or its run failed)."""
    rdir, image_id = Path(meta["result_dir"]), str(meta["image_id"])
    rs = ResultsSet(rdir, image_id)
    marker = rs.done_marker()
    lab = labels_file(rdir, image_id)
    if not isinstance(marker, dict) or not _isfile(lab):
        raise TuningError(f"results/{image_id} has no finished run now (it is being run again, or its run failed): "
                          f"save its points when its run has finished")
    what = []
    stride, was = int(marker.get("xy_stride") or 1), int(meta.get("xy_stride") or 1)
    if stride != was:
        what.append(f"xy stride {was} to {stride}")
    zr, zr_was = _z_range(marker.get("z_range")), _z_range(meta.get("z_range"))
    if zr != zr_was:
        what.append(f"z range {list(zr_was) if zr_was else 'all planes'} to {list(zr) if zr else 'all planes'}")
    if meta.get("labels_shape") is not None:
        shape, shape_was = tiff_shape(lab), tuple(int(v) for v in meta["labels_shape"])
        if shape != shape_was:
            what.append(f"label image {list(shape_was)} to {list(shape)}")
    spacing = _table_spacing(rs)
    if meta.get("table_spacing") is not None and not _same_spacing(spacing, meta["table_spacing"]):
        what.append(f"voxel size {list(meta['table_spacing'])} to {[float(s) for s in spacing]} um")
    return "; ".join(what) or None


def save_truth(meta: dict, data, *, projected: bool) -> Path:
    """Write the truth layer's points (``data``, the run's frame) to its file through the engine's writer,
    ``germquant.tune.points.write_points``, with the run's labels, voxel size and geometry (``meta``, the layer's
    LAYER_KEY record). TuningError when this environment has no writer, when the file holds points of planes a
    one-plane run did not read (a Save would remove them), when the image has no finished run now, or when the
    writer refuses the points; RunChanged when the run's grid changed since the layer was opened."""
    pts = tune_module("points")
    if pts is None:
        raise TuningError(points_problem() or "no germquant.tune in this napari environment")
    image_id = str(meta["image_id"])
    other = int(meta.get("other_planes") or 0)
    if other:
        raise TuningError(other_planes_text(image_id, other, Path(meta["file"]).name))
    changed = run_changed(meta)
    if changed:
        raise RunChanged(f"the run of {image_id} changed since its truth layer was opened ({changed}), so its "
                         f"points sit on the old grid and were not saved")
    labels = run_labels(labels_file(meta["result_dir"], image_id))
    zr = meta.get("z_range")
    try:
        return pts.write_points(Path(meta["file"]), np.asarray(data, float).reshape(-1, 3), labels=labels,
                                spacing=tuple(meta["table_spacing"]), xy_stride=int(meta.get("xy_stride") or 1),
                                z_range=tuple(zr) if zr else None, projected=bool(projected))
    except ValueError as e:
        raise TuningError(f"the points of {image_id} were not saved: {e}") from None


# ---------------------------------------------------------------------------------------------------
# a search: the backend, the grid and what it means
# ---------------------------------------------------------------------------------------------------


def project_backend(root, rows: list[ImageRow]) -> tuple[str | None, str]:
    """The spot backend a search tunes, and where it was read: the project's recipe, through this environment's
    germquant.project when it loads (the engine tunes the recipe's backend), else the config of the annotated
    runs. (None, why) when neither says."""
    try:
        from germquant.project import project_config

        cfg = project_config(Path(root))
        return str(cfg.get("spots.backend", "spotmax")), "the project's recipe"
    except Exception:  # noqa: BLE001 - another germquant, a pack that does not resolve here: the runs then say
        pass
    seen = []
    for r in rows:
        if r.image_id in annotated(rows):
            cfg = ResultsSet(result_dir(root, r.image_id), r.image_id).run_config()
            spots = (cfg or {}).get("spots")
            if isinstance(spots, dict):
                seen.append(str(spots.get("backend") or "spotmax"))
    if seen:
        return seen[0], "the annotated runs" + (f" (they differ: {', '.join(sorted(set(seen)))})"
                                                if len(set(seen)) > 1 else "")
    return None, "no annotated finished run"


def search_grid(backend: str | None, grid_file=None) -> dict:
    """The grid a search takes: the file's (germquant.tune.grid.load_grid) or the engine's default for the
    backend (default_grid). TuningError with the engine's reason when it cannot be read, or without
    germquant.tune here."""
    grid_mod = tune_module("grid")
    if grid_mod is None:
        raise TuningError("the engine counts the detections when the search starts (this napari environment's "
                          "germquant has no germquant.tune)")
    try:
        if grid_file is not None:
            return grid_mod.load_grid(Path(grid_file))
        if backend is None:
            raise TuningError("no annotated finished run says which spot backend the search tunes")
        return grid_mod.default_grid(backend)
    except FileNotFoundError:
        raise TuningError(f"the grid file {grid_file} was not found") from None
    except ValueError as e:                           # GridError is a ValueError (TuneRefused)
        raise TuningError(str(e)) from None


def grid_text(grid: dict | None, grid_file=None) -> str:
    keys = "; ".join(f"{k} in {', '.join(_value_text(v) for v in vals)}" for k, vals in (grid or {}).items())
    if grid_file is not None:
        return f"the grid file {grid_file}" + (f": {keys}" if keys else "")
    return "the engine's default for the recipe's spot backend" + (f": {keys}" if keys else "")


def _value_text(v) -> str:
    """A grid or best.yaml value as Python writes it (2.0 stays 2.0, as the files hold it)."""
    return str(v)


def count_text(grid: dict, n_images: int, backend: str, max_runs: int) -> str:
    """What a search means (germquant.tune.grid.n_runs, what --max-runs bounds): "6 LoG detections on 1 annotated
    image", and a word when that is above the limit."""
    grid_mod = tune_module("grid")
    n = int(grid_mod.n_runs(grid, n_images, backend)) if grid_mod is not None else 0
    one, many = BACKEND_WORDS.get(backend, (f"{backend} detection", f"{backend} detections"))
    text = f"{n} {one if n == 1 else many} on {n_images} annotated image{'s' if n_images != 1 else ''}"
    if n > int(max_runs):
        text += f": above Max runs {int(max_runs)}, so the engine refuses it; narrow the grid or raise Max runs"
    return text


# ---------------------------------------------------------------------------------------------------
# tuning runs
# ---------------------------------------------------------------------------------------------------


def tuning_runs(root) -> list[Path]:
    """The project's tuning runs (tuning/<run_id>/ with a ranked.csv), newest first: run ids begin with their UTC
    time, so their names sort by time."""
    folder = Path(root) / TUNING_DIR
    if not _isdir(folder):
        return []
    try:
        names = os.listdir(_long(folder))
    except OSError:
        return []
    runs = [folder / n for n in names if _isfile(folder / n / "ranked.csv")]
    return sorted(runs, key=lambda p: (bool(RUN_ID_RE.fullmatch(p.name)), p.name), reverse=True)


def ranked_rows(run_dir, n: int = TOP_N) -> tuple[list[str], list[list[str]]]:
    """The header and the first ``n`` rows of a tuning run's ranked.csv, as the file holds them (text)."""
    with open(_long(Path(run_dir) / "ranked.csv"), encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(io.StringIO(fh.read())) if r]
    return (rows[0] if rows else []), rows[1:n + 1]


def read_best(run_dir) -> dict | None:
    """best.yaml of a tuning run ({objective, backend, keys}), or None when the search scored nothing (no file)
    or the file is not such a record."""
    import yaml

    p = Path(run_dir) / "best.yaml"
    if not _isfile(p):
        return None
    try:
        with open(_long(p), encoding="utf-8-sig") as fh:
            best = yaml.safe_load(fh)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    return best if isinstance(best, dict) and isinstance(best.get("keys"), dict) and best["keys"] else None


def run_record(run_dir) -> dict:
    """run.json of a tuning run, or {} when it cannot be read."""
    try:
        with open(_long(Path(run_dir) / "run.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def run_view(run_dir) -> dict:
    """What the dock shows of a tuning run: ``head`` and ``rows`` (the top TOP_N of ranked.csv, as text), ``best``
    (best.yaml, or None) and ``record`` (run.json, or {})."""
    try:
        head, rows = ranked_rows(run_dir)
    except OSError:
        head, rows = [], []
    return {"run_dir": str(run_dir), "head": head, "rows": rows, "best": read_best(run_dir),
            "record": run_record(run_dir)}


def grid_state(backend: str | None, grid_file=None) -> tuple[dict | None, str | None]:
    """(grid, None) for the grid a search would take, or (None, why) when it cannot be told here."""
    try:
        return search_grid(backend, grid_file), None
    except TuningError as e:
        return None, str(e)


def read_project(root, annotator: str | None, grid_file=None) -> dict:
    """Everything the dock shows of a project, read in one go (the dock's file worker): the manifest, the image
    rows, the spot backend and where it was read, the grid a search would take, and the tuning runs with their
    views, newest first. TuningError for a folder that is no project."""
    root = Path(root)
    manifest = read_manifest(root)
    rows = project_rows(root, manifest, annotator)
    backend, source = project_backend(root, rows)
    grid, grid_error = grid_state(backend, grid_file)
    runs = tuning_runs(root)
    return {"root": root, "manifest": manifest, "rows": rows, "backend": backend, "backend_source": source,
            "grid": grid, "grid_error": grid_error, "runs": [str(r) for r in runs],
            "views": {str(r): run_view(r) for r in runs}}


def best_text(best: dict | None) -> str:
    if not best:
        return "no setting could be scored: nothing to apply (see the warnings)"
    keys = ", ".join(f"{k} = {_value_text(v)}" for k, v in best["keys"].items())
    return f"best: {keys} ({best.get('backend')})"


def objective_text(best: dict | None, record: dict) -> str:
    text = (best or {}).get("objective") or record.get("objective") or ""
    return f"ranked by {text}" if text else ""


def validated_text(validated) -> str:
    """The manifest's validated record ({value, why, record}) as the dock says it."""
    if not isinstance(validated, dict) or not isinstance(validated.get("value"), bool):
        return "validated: not read (germquant_project.yaml could not be read after the write)"
    if validated["value"]:
        return "validated: true" + (f", record {validated['record']}" if validated.get("record") else "")
    return "validated: false"


def applied_text(new_hash: str | None, old_hash: str | None, validated=None) -> str:
    """What an Apply did, as the dock says it: the new config hash, whether the project reads validated (the
    manifest's record after the write, ``validated``) and which images turn stale."""
    if new_hash and old_hash and new_hash == old_hash:
        return f"applied: config_hash {new_hash}, unchanged (the recipe already holds this setting)"
    stale = f"; the images run with {old_hash} are stale until they are run again" if old_hash else ""
    return f"applied: config_hash {new_hash or 'not reported'} ({validated_text(validated)}){stale}"


def apply_warnings(rec) -> list[str]:
    """The warnings of an Apply's record, each once and in order: the command's ``warnings`` (its --json
    record) and its ``WARNING:`` lines (the engine prints each TuningWarning so: the recipe changed after the
    tuning run's images were run, and the write still happened)."""
    if not isinstance(rec, dict):
        return []
    raw = rec.get("warnings") or []
    notes = [str(w).strip() for w in ([raw] if isinstance(raw, str) else raw)]
    for line in rec.get("lines") or []:
        text = str(line).strip()
        if text.startswith("WARNING:"):
            notes.append(text.split(":", 1)[1].strip())
    return list(dict.fromkeys(n for n in notes if n))


# ---------------------------------------------------------------------------------------------------
# runs in progress (the Project dock's registry)
# ---------------------------------------------------------------------------------------------------


def run_in_progress(root) -> bool:
    """Whether a dock of this napari runs this project now (napari_germquant._project.run_in_progress, unit
    P6-PROJECT): a search and an Apply wait until it has finished."""
    from ._project import run_in_progress as running

    try:
        return bool(running(root))
    except Exception:  # noqa: BLE001 - the registry is a hint; the hub's running flag is the other
        return False
