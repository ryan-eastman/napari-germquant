"""A finished time-lapse series folder as napari layers, plus the Register and Link steps. Pure python, no Qt.

The folder is the one ``germquant run SERIES.nd2 --t ...`` writes (``<stem>__series.json`` next to one
``t<NNN>/`` folder per timepoint); ``germquant.timelapse.layout.find_series`` reads it, so the dock sees
exactly the timepoints the tracking commands see. The layers:

* the per-timepoint nucleus label images as one (T, Z, Y, X) Labels layer, lazy: a dask array whose blocks
  are single planes, read from the TIFF page only when napari shows that plane (nothing is read when the
  layer is built). A timepoint the record lists as done but whose files are missing is an empty frame, so
  the time slider always shows the real t_index;
* the reference volumes the series loop kept for registration (``t<NNN>/<id>__ref_stride<k>.npy``, the
  segmentation channel at xy stride k) as a (T, Z, Y/k, X/k) Image layer, lazy the same way;
* ``<stem>__track_points.csv`` as a Tracks layer (time_track_id, t, z, y, x) with the point table as
  features, and the points without a track (backend ``none``, or nuclei a tracker left out) as a 4D Points
  layer.

Coordinates, raw or registered (one toggle for every layer). Raw is the microscope frame the tables and
label images are written in. Registered is the shared contract of the time-lapse packages: raw minus the
timepoint's cumulative shift from ``<stem>__register_shifts.csv``. The tracks take the table's own
registered columns (reg_z_um, reg_y_um, reg_x_um); the label and reference frames are displaced by that
shift rounded to whole voxels, the grid ``germquant track-export`` uses (``track.place_frame``). This is a
view only: nothing is resampled or written.

After a link with the piecewise registration (``germquant register --mode piecewise``, then ``track-link
--registration piecewise``) every nucleus has its own displacement, so one shift per frame cannot place the
frames. The registered labels then move label by label, each by its nucleus's displacement rounded to whole
voxels (``track.place_frame_labels``, the rule the engine's overlap linking uses; the displacement is the
points table's raw minus registered centroid). Labels without a track point are left out, and the reference
volumes stay in the raw frame. This view reads a whole label image per timepoint, because a displacement in
z moves a nucleus across planes.

Register and Link run ``germquant.timelapse.register.register_series`` (``germquant register``), or
``germquant.timelapse.piecewise.register_series_piecewise`` for ``--mode piecewise``, and
``germquant.timelapse.track.link_series`` (``germquant track-link``) with the arguments those commands pass.
"""
from __future__ import annotations

# germquant.api first (through _engine): the torch-before-CuPy import order of the workstation
from . import _engine  # noqa: F401,I001

import dataclasses
import importlib.util
import json
import threading
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from germquant import schema
from germquant.fsutil import long_path
from germquant.timelapse import layout as gq_layout
from germquant.timelapse import piecewise as gq_piecewise
from germquant.timelapse import register as gq_register
from germquant.timelapse import track as gq_track

from ._engine import _quote

LayerData = tuple
META_KEY = "napari_germquant_timelapse"
BACKENDS = tuple(gq_track.BACKENDS)
REGISTRATIONS = tuple(gq_track.REGISTRATIONS)     # how track-link removes the motion: rigid | piecewise
REGISTER_MODES = ("rigid", "piecewise")             # germquant register --mode
REGISTERED_COLUMNS = (("reg_z_um", "reg_y_um", "reg_x_um"), ("zr_um", "yr_um", "xr_um"))
RAW_COLUMNS = ("z_um", "y_um", "x_um")


# ---------------------------------------------------------------------------------------------------
# the series folder
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class SeriesView:
    """Everything the dock shows about one series folder, read once (``open_series``)."""
    layout: gq_layout.SeriesLayout
    spacing: tuple[float, float, float]
    shifts: pd.DataFrame                    # load_shifts: t_index, dz_um, dy_um, dx_um (zeros when unregistered)
    shift_source: str                       # "register_shifts" or "none"
    shift_table: pd.DataFrame | None        # <stem>__register_shifts.csv as written
    report: dict | None                     # <stem>__register_report.json
    points: pd.DataFrame | None             # <stem>__track_points.csv
    link_record: dict | None                # <stem>__track_link.json
    label_shapes: dict[int, tuple[int, int, int]]
    label_dtype: np.dtype
    refs: dict[int, tuple[Path, int]]       # t -> (reference volume, its stride)
    notes: list[str]
    piecewise_report: dict | None = None    # <stem>__register_piecewise_report.json
    has_piecewise: bool = False             # <stem>__register_piecewise.csv is in the folder

    @property
    def linked_piecewise(self) -> bool:
        """The tracks were linked with the piecewise registration (track-link --registration piecewise)."""
        return str((self.link_record or {}).get("registration") or "").strip().lower() == "piecewise"

    @property
    def folder(self) -> Path:
        return Path(self.layout.out_dir)

    @property
    def stem(self) -> str:
        return self.layout.stem

    @property
    def timepoints(self) -> list[int]:
        return list(self.layout.timepoints)

    @property
    def t_first(self) -> int:
        return int(self.layout.timepoints[0])

    @property
    def t_last(self) -> int:
        return int(self.layout.timepoints[-1])

    @property
    def n_frames(self) -> int:
        return self.t_last - self.t_first + 1


def _read_json(path: Path) -> dict | None:
    try:
        with open(long_path(path), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _label_dtype(path: Path) -> np.dtype:
    import tifffile

    with tifffile.TiffFile(long_path(path)) as tf:
        return np.dtype(tf.series[0].dtype)


def open_series(folder: str | Path, stem: str | None = None) -> SeriesView:
    """Read a series folder: its layout, voxel size, shifts, registration report, track points and the
    shapes of its label images and reference volumes (headers only, no pixels)."""
    layout = gq_layout.find_series(folder, stem)
    spacing = tuple(float(v) for v in layout.spacing())
    shifts, source = gq_layout.load_shifts(layout)
    shift_path = layout.series_table_path(gq_layout.SHIFTS_TABLE)
    shift_table = pd.read_csv(long_path(shift_path), low_memory=False) if shift_path.is_file() else None
    report = _read_json(layout.out_dir / f"{layout.stem}{gq_register.REPORT_SUFFIX}")
    points = None
    if layout.series_table_path("track_points").is_file():
        points = gq_track.read_track_points(layout)
    link_record = _read_json(layout.out_dir / f"{layout.stem}__{gq_track.LINK_RECORD}.json")
    shapes = {int(t): tuple(int(v) for v in layout.labels_shape(t)) for t in layout.timepoints}
    dtype = _label_dtype(layout.labels_path(layout.timepoints[0]))
    refs = {}
    for t in layout.timepoints:
        found = layout.ref_path(t)
        if found is not None:
            refs[int(t)] = (Path(found[0]), int(found[1]))
    pw_path = layout.series_table_path(gq_piecewise.TABLE)
    has_pw = pw_path.is_file()
    pw_report = _read_json(layout.out_dir / f"{layout.stem}{gq_piecewise.REPORT_SUFFIX}") if has_pw else None
    linked_pw = str((link_record or {}).get("registration") or "").strip().lower() == "piecewise"
    notes = []
    if layout.missing:
        notes.append(f"timepoint(s) {layout.missing} are in the series record but their files are missing: empty frames")
    tp = layout.series_table_path("track_points")
    if points is not None and source == "register_shifts" and not linked_pw:
        linked_with = (link_record or {}).get("shift_source")
        if linked_with == "none" or (shift_path.stat().st_mtime > tp.stat().st_mtime):
            notes.append("the tracks were linked before the latest registration: press Link again so their registered "
                         "coordinates use the new shifts")
    if points is not None and linked_pw:
        if has_pw and pw_path.stat().st_mtime > tp.stat().st_mtime:
            notes.append("the tracks were linked before the latest piecewise registration: press Link again so their "
                         "registered coordinates use it")
        notes.append("registered view: the tracks were linked with the piecewise registration, so each nucleus label "
                     "moves by its own displacement, labels without a track point are left out, and the reference "
                     "volumes stay in the raw frame")
    return SeriesView(layout=layout, spacing=spacing, shifts=shifts, shift_source=source, shift_table=shift_table,
                      report=report, points=points, link_record=link_record, label_shapes=shapes, label_dtype=dtype,
                      refs=refs, notes=notes, piecewise_report=pw_report, has_piecewise=has_pw)


def info_text(view: SeriesView) -> str:
    """A few lines for the dock: timepoints, voxel size, label grid, reference volumes, registration, tracks."""
    lay = view.layout
    period = f", period {lay.period_s:g} s" if lay.period_s else ""
    lines = [f"{view.stem}: {len(view.timepoints)} timepoint(s) on disk, t{view.t_first:03d} to t{view.t_last:03d}{period}"]
    dz, dy, dx = view.spacing
    zyx = Counter(view.label_shapes.values()).most_common(1)[0][0]
    lines.append(f"voxel um dz={dz:.3f} dy={dy:.3f} dx={dx:.3f}; label images (Z, Y, X) = {zyx} {view.label_dtype}")
    if view.refs:
        k = _common_stride(view)
        lines.append(f"reference volumes: {sum(1 for _, s in view.refs.values() if s == k)} timepoint(s) at xy stride {k}")
    else:
        lines.append("reference volumes: none (Register needs them; rerun the series with "
                     "timelapse.keep_reference_volume on, the code default)")
    if view.shift_source == "register_shifts":
        rep = view.report or {}
        lines.append(f"registration: {view.stem}__register_shifts.csv"
                     + (f" ({rep.get('method')}, {rep.get('n_registered')} step(s) registered)" if rep else ""))
    elif not view.has_piecewise:
        lines.append("registration: none yet (registered coordinates equal raw until you press Register)")
    if view.has_piecewise:
        rep = view.piecewise_report or {}
        par = rep.get("parameters") or {}
        lines.append(f"piecewise registration: {view.stem}__register_piecewise.csv"
                     + (f" ({par.get('segments')} segment(s), {rep.get('n_registered')} step(s) registered)" if rep else ""))
    if view.points is not None:
        n_linked = int(view.points["time_track_id"].notna().sum())
        n_tracks = int(view.points["time_track_id"].dropna().nunique())
        backend = (view.link_record or {}).get("backend") or "?"
        how = ", piecewise registration" if view.linked_piecewise else ""
        lines.append(f"tracks: {n_tracks} track(s) over {n_linked} of {len(view.points)} nucleus-timepoints "
                     f"(backend {backend}{how})")
    else:
        lines.append("tracks: none yet (press Link)")
    lines += view.notes
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# lazy frames
# ---------------------------------------------------------------------------------------------------


def _common_shape(shapes) -> tuple[int, int, int]:
    arr = np.array([tuple(s) for s in shapes], dtype=int)
    return tuple(int(v) for v in arr.max(axis=0))  # type: ignore[return-value]


def place_plane(get_plane, shape_t, z: int, sv, stride: int, out_shape, dtype) -> np.ndarray:
    """One plane of a frame on the display grid: display voxel (z, i, j) holds raw voxel
    (z + sv_z, i * stride + sv_y, j * stride + sv_x), zero where that falls outside the raw frame. ``sv`` is
    the frame's whole-voxel cumulative shift (zeros for raw coordinates); the rule is ``track.place_frame``'s,
    one plane at a time, so the registered view sits on the grid ``germquant track-export`` writes.
    ``get_plane(z)`` returns raw plane z (only called when it is inside the frame)."""
    Zt, Yt, Xt = (int(v) for v in shape_t)
    out = np.zeros(tuple(int(v) for v in out_shape), dtype=dtype)
    zr = int(z) + int(sv[0])
    if not 0 <= zr < Zt:
        return out
    s = max(1, int(stride))
    yi = np.arange(out.shape[0]) * s + int(sv[1])
    xi = np.arange(out.shape[1]) * s + int(sv[2])
    vy = (yi >= 0) & (yi < Yt)
    vx = (xi >= 0) & (xi < Xt)
    if vy.any() and vx.any():
        plane = np.asarray(get_plane(zr))
        out[np.ix_(vy, vx)] = plane[np.ix_(yi[vy], xi[vx])]
    return out


def read_tiff_plane(path: str | Path, z: int, n_planes: int) -> np.ndarray:
    """Plane ``z`` of a (Z, Y, X) label TIFF, reading only that page (the engine writes one page per plane);
    a file stored otherwise is read whole and sliced."""
    import tifffile

    with tifffile.TiffFile(long_path(path)) as tf:
        series = tf.series[0]
        pages = series.pages
        if len(pages) == int(n_planes) and pages[int(z)] is not None:
            return np.asarray(pages[int(z)].asarray())
        vol = np.asarray(series.asarray())
    return vol[int(z)] if vol.ndim == 3 else vol


def _lazy_stack(n_frames: int, n_planes: int, plane_shape, dtype, block_fn, name: str):
    """A (T, Z, Y, X) dask array with one block per plane; ``block_fn(frame_index, z)`` returns the plane."""
    import dask.array as da

    def block(block_id=None):
        ti, zi = int(block_id[0]), int(block_id[1])
        return block_fn(ti, zi)[None, None]

    chunks = ((1,) * int(n_frames), (1,) * int(n_planes), (int(plane_shape[0]),), (int(plane_shape[1]),))
    return da.map_blocks(block, dtype=dtype, chunks=chunks, meta=np.empty((0, 0, 0, 0), dtype=dtype),
                         name=f"{name}-{uuid.uuid4().hex}")


def frame_shifts(view: SeriesView, spacing, registered: bool) -> dict[int, np.ndarray]:
    """``{t: whole-voxel shift}`` on a grid of ``spacing`` (zeros for raw coordinates, and for a series linked
    with the piecewise registration, which has no single shift per frame: its labels move label by label in
    labels_stack and its reference volumes stay raw)."""
    if not registered or view.linked_piecewise:
        return {}
    return gq_layout.voxel_shifts(view.shifts, spacing)


def label_shifts_from_points(view: SeriesView) -> dict[int, dict[int, np.ndarray]]:
    """``{t_index: {nucleus_id: whole-voxel displacement (z, y, x)}}`` from the points table: the raw minus the
    registered centroid, rounded to voxels, the displacement the engine's ``track.label_voxel_shifts`` rounds.
    Gap-closing points and points without a nucleus or a registered centroid are skipped."""
    pts = view.points
    if pts is None or not len(pts):
        return {}
    rz, ry, rx = coordinate_columns(pts, True)
    raw = pts[list(RAW_COLUMNS)].to_numpy(float)
    reg = pts[[rz, ry, rx]].to_numpy(float)
    nid = pd.to_numeric(pts["nucleus_id"], errors="coerce").to_numpy(float)
    gap = (pts["is_gap"].astype(str).str.strip().str.lower().isin(("true", "1")).to_numpy(bool) if "is_gap" in pts
           else np.zeros(len(pts), dtype=bool))
    ok = np.isfinite(raw).all(axis=1) & np.isfinite(reg).all(axis=1) & np.isfinite(nid) & ~gap
    vox = np.rint((raw - reg) / np.asarray(view.spacing, dtype=float)).astype(int)
    out: dict[int, dict[int, np.ndarray]] = {}
    for t, n, v in zip(pts["t_index"].to_numpy()[ok].astype(int), nid[ok].astype(int), vox[ok]):
        out.setdefault(int(t), {})[int(n)] = v
    return out


def _piecewise_labels_stack(view: SeriesView, stride: int, common_shape, out_shape):
    """labels_stack for a series linked with the piecewise registration: every frame placed label by label
    (track.place_frame_labels) with the displacements of label_shifts_from_points, one whole frame per block
    request (kept until another timepoint is asked for)."""
    shifts = label_shifts_from_points(view)
    Zc, Yc, Xc = (int(v) for v in common_shape)
    box = {"start": (0, 0, 0), "stop": (Zc, Yc, Xc)}
    dtype = view.label_dtype
    t0 = view.t_first
    on_disk = {int(t) for t in view.timepoints}
    cache: dict[int, np.ndarray] = {}
    lock = threading.Lock()

    def frame(t: int) -> np.ndarray:
        with lock:
            hit = cache.get(t)
        if hit is not None:
            return hit
        sh = shifts.get(t, {})
        ids = np.array(sorted(sh), dtype=np.int64)
        vol = gq_track.place_frame_labels(view.layout.read_labels(t), sh, box, stride, ids, dtype=dtype)
        with lock:
            cache.clear()                     # one frame: a frame is as large as a label image
            cache[t] = vol
        return vol

    def plane(ti: int, zi: int) -> np.ndarray:
        t = t0 + ti
        if t not in on_disk:
            return np.zeros(out_shape, dtype=dtype)
        return frame(t)[zi]

    return _lazy_stack(view.n_frames, Zc, out_shape, dtype, plane, f"germquant-labels-{view.stem}")


def labels_stack(view: SeriesView, *, registered: bool = False, stride: int = 1):
    """The label images as a lazy (T, Z, Y/stride, X/stride) dask array covering t_first..t_last."""
    s = max(1, int(stride))
    Zc, Yc, Xc = _common_shape(view.label_shapes.values())
    out_shape = (len(range(0, Yc, s)), len(range(0, Xc, s)))
    if registered and view.linked_piecewise:
        return _piecewise_labels_stack(view, s, (Zc, Yc, Xc), out_shape)
    zero = np.zeros(3, dtype=int)
    vsh = frame_shifts(view, view.spacing, registered)
    paths = {int(t): view.layout.labels_path(t) for t in view.timepoints}
    shapes = dict(view.label_shapes)
    dtype = view.label_dtype
    t0 = view.t_first

    def plane(ti: int, zi: int) -> np.ndarray:
        t = t0 + ti
        if t not in paths:
            return np.zeros(out_shape, dtype=dtype)
        Zt = shapes[t][0]
        return place_plane(lambda z: read_tiff_plane(paths[t], z, Zt), shapes[t], zi, vsh.get(t, zero), s,
                           out_shape, dtype)

    return _lazy_stack(view.n_frames, Zc, out_shape, dtype, plane, f"germquant-labels-{view.stem}")


def _common_stride(view: SeriesView) -> int | None:
    if not view.refs:
        return None
    counts = Counter(k for _, k in view.refs.values())
    return max(counts, key=lambda k: (counts[k], -k))     # the stride most timepoints have, smallest on a tie


def reference_stack(view: SeriesView, *, registered: bool = False):
    """The reference volumes as a lazy (T, Z, Y/k, X/k) dask array (uint16) and their voxel size, or
    (None, None) when the series kept none. Volumes at another stride than the common one are left out."""
    k = _common_stride(view)
    if k is None:
        return None, None
    files = {t: p for t, (p, kk) in view.refs.items() if kk == k}
    shapes = {}
    for t, p in files.items():
        shapes[t] = tuple(int(v) for v in np.load(long_path(p), mmap_mode="r").shape)
    files = {t: p for t, p in files.items() if len(shapes[t]) == 3}
    if not files:
        return None, None
    Zc, Yc, Xc = _common_shape(shapes[t] for t in files)
    dz, dy, dx = view.spacing
    spacing = (dz, dy * k, dx * k)
    zero = np.zeros(3, dtype=int)
    vsh = frame_shifts(view, spacing, registered)
    t0 = view.t_first

    def plane(ti: int, zi: int) -> np.ndarray:
        t = t0 + ti
        if t not in files:
            return np.zeros((Yc, Xc), dtype=np.uint16)
        arr = np.load(long_path(files[t]), mmap_mode="r")
        return place_plane(lambda z: arr[z], shapes[t], zi, vsh.get(t, zero), 1, (Yc, Xc), np.uint16)

    return _lazy_stack(view.n_frames, Zc, (Yc, Xc), np.dtype(np.uint16), plane, f"germquant-ref-{view.stem}"), spacing


def _contrast(view: SeriesView) -> tuple[float, float] | None:
    """Display range of the reference layer from one volume (the first timepoint's, a few MB), so napari
    never has to compute the range of the whole lazy stack."""
    if not view.refs:
        return None
    k = _common_stride(view)
    t = next(t for t in sorted(view.refs) if view.refs[t][1] == k)
    vol = np.asarray(np.load(long_path(view.refs[t][0]), mmap_mode="r"))
    if not vol.size:
        return None
    lo, hi = (float(v) for v in np.percentile(vol, (0.5, 99.9)))
    return (lo, hi) if hi > lo else (float(vol.min()), float(vol.max()) + 1.0)


# ---------------------------------------------------------------------------------------------------
# tracks
# ---------------------------------------------------------------------------------------------------


def coordinate_columns(points: pd.DataFrame, registered: bool) -> tuple[str, str, str]:
    if not registered:
        return RAW_COLUMNS
    for cols in REGISTERED_COLUMNS:
        if set(cols) <= set(points.columns):
            return cols
    raise ValueError("the track_points table has no registered coordinate columns (reg_z_um, ...)")


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """The point table minus the series-level block every row repeats (series_id, file_path, versions, ...)."""
    drop = [c for c in schema.SERIES_META if c in df.columns]
    return df.drop(columns=drop).reset_index(drop=True)


def tracks_data(points: pd.DataFrame, registered: bool) -> tuple[np.ndarray, pd.DataFrame, int]:
    """(data, features, n_dropped): the linked points as napari Tracks rows (time_track_id, t_index, z, y, x),
    sorted by track then time, coordinates in microns, raw or registered; points with a non-finite
    coordinate are dropped and counted."""
    cols = coordinate_columns(points, registered)
    linked = points[points["time_track_id"].notna()].copy()
    for c in ("time_track_id", "t_index", *cols):
        linked[c] = pd.to_numeric(linked[c], errors="coerce")
    ok = np.isfinite(linked[["t_index", *cols]].to_numpy(float)).all(axis=1)
    n_dropped = int((~ok).sum())
    linked = linked[ok].sort_values(["time_track_id", "t_index"], kind="stable")
    data = np.column_stack([linked["time_track_id"].to_numpy(float), linked["t_index"].to_numpy(float),
                            *(linked[c].to_numpy(float) for c in cols)]) if len(linked) else np.zeros((0, 5))
    return data, _features(linked), n_dropped


def unlinked_data(points: pd.DataFrame, registered: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """The points without a track as (t_index, z, y, x) rows in microns, with their features."""
    cols = coordinate_columns(points, registered)
    free = points[points["time_track_id"].isna()].copy()
    for c in ("t_index", *cols):
        free[c] = pd.to_numeric(free[c], errors="coerce")
    free = free[np.isfinite(free[["t_index", *cols]].to_numpy(float)).all(axis=1)]
    data = free[["t_index", *cols]].to_numpy(float) if len(free) else np.zeros((0, 4))
    return data, _features(free)


# ---------------------------------------------------------------------------------------------------
# layer tuples
# ---------------------------------------------------------------------------------------------------


def _meta(view: SeriesView, kind: str, registered: bool, **extra) -> dict:
    return {META_KEY: {"folder": str(view.folder), "stem": view.stem, "layer": kind,
                       "coordinates": "registered" if registered else "raw", "t_first": view.t_first, **extra}}


def labels_layer(view: SeriesView, *, registered: bool = False, stride: int = 1) -> LayerData:
    s = max(1, int(stride))
    dz, dy, dx = view.spacing
    kwargs = {"name": f"{view.stem} nuclei", "scale": (1.0, dz, dy * s, dx * s),
              "translate": (float(view.t_first), 0.0, 0.0, 0.0),
              "metadata": _meta(view, "labels", registered, xy_stride=s)}
    return (labels_stack(view, registered=registered, stride=s), kwargs, "labels")


def reference_layer(view: SeriesView, *, registered: bool = False) -> LayerData | None:
    data, spacing = reference_stack(view, registered=registered)
    if data is None:
        return None
    kwargs = {"name": f"{view.stem} reference", "scale": (1.0, *spacing),
              "translate": (float(view.t_first), 0.0, 0.0, 0.0), "colormap": "gray", "blending": "additive",
              "metadata": _meta(view, "reference", registered, stride=_common_stride(view))}
    limits = _contrast(view)
    if limits is not None:
        kwargs["contrast_limits"] = limits
    return (data, kwargs, "image")


def tracks_layer(view: SeriesView, *, registered: bool = False) -> LayerData | None:
    if view.points is None:
        return None
    data, features, dropped = tracks_data(view.points, registered)
    if not len(data):
        return None
    kwargs = {"name": f"{view.stem} tracks", "features": features, "tail_length": max(1, view.n_frames),
              "metadata": _meta(view, "tracks", registered, n_dropped=dropped)}
    return (data, kwargs, "tracks")


def unlinked_layer(view: SeriesView, *, registered: bool = False) -> LayerData | None:
    if view.points is None:
        return None
    data, features = unlinked_data(view.points, registered)
    if not len(data):
        return None
    kwargs = {"name": f"{view.stem} nuclei without a track", "features": features, "size": 1.0,
              "face_color": "gray", "border_color": "white",
              "metadata": _meta(view, "unlinked", registered)}
    return (data, kwargs, "points")


# ---------------------------------------------------------------------------------------------------
# the register_shifts summary
# ---------------------------------------------------------------------------------------------------

SHIFT_COLUMNS = (("t_index", "t"), ("dz_um", "dz um"), ("dy_um", "dy um"), ("dx_um", "dx um"),
                 ("step_um", "step um"), ("capped", "capped"), ("quality_before_um", "NN before um"),
                 ("quality_after_um", "NN after um"), ("error", "error"))


def shift_rows(view: SeriesView) -> list[list[str]]:
    """One row of text per timepoint of the shift table: t, the cumulative shift, the step length, capped,
    the nearest-neighbour quality before and after the step, and the error (empty without a table)."""
    df = view.shift_table
    if df is None or not len(df):
        return []
    rows = []
    for r in df.to_dict("records"):
        step = [r.get(f"step_d{a}_um", r.get(f"shift_{a}_um")) for a in "zyx"]
        try:
            step_len = float(np.linalg.norm(np.asarray(step, dtype=float)))
        except (TypeError, ValueError):
            step_len = float("nan")
        values = {**r, "step_um": step_len}
        cells = []
        for key, _label in SHIFT_COLUMNS:
            v = values.get(key)
            if key == "t_index":
                cells.append(str(int(v)) if pd.notna(v) else "")
            elif key == "capped":
                cells.append("yes" if str(v).strip().lower() in ("true", "1") else "")
            elif key == "error":
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            else:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    f = float("nan")
                signed = key in ("dz_um", "dy_um", "dx_um")
                cells.append("" if not np.isfinite(f) else (f"{f:+.3f}" if signed else f"{f:.3f}"))
        rows.append(cells)
    return rows


def shifts_summary(view: SeriesView) -> str:
    """The lines ``germquant register`` prints, from the registration report (or the table alone)."""
    if view.shift_source != "register_shifts":
        return "not registered yet: no register_shifts table (registered coordinates equal raw)"
    rep = view.report
    if not rep:
        return (f"{view.stem}__register_shifts.csv: {len(view.shift_table) if view.shift_table is not None else 0} row(s) "
                f"(no register_report.json next to it)")
    par = rep.get("parameters", {})
    lines = [f"{rep.get('n_timepoints')} timepoint(s), {rep.get('n_registered')} step(s) registered on "
             f"{par.get('channel_role')} ({rep.get('method')}, stride {par.get('stride')}, upsample "
             f"{par.get('upsample_factor')}, reference {par.get('reference')})"]
    last = rep.get("cumulative_last_um")
    if last:
        cap = f"{par['max_step_um']:g} um" if par.get("max_step_um") else "none"
        largest = rep.get("max_step_um")
        largest = f"{largest:.2f} um" if largest is not None else "n/a"
        lines.append(f"cumulative shift at t{int(rep.get('last_t_index') or 0):03d}: dz={last[0]:+.2f} dy={last[1]:+.2f} "
                     f"dx={last[2]:+.2f} um; largest step {largest}; {rep.get('n_capped')} capped (cap {cap})")
    q = rep.get("quality") or {}
    if q.get("median_before_um") is not None and q.get("median_after_um") is not None:
        lines.append(f"quality: median nearest-neighbour displacement {q['median_before_um']:.2f} -> "
                     f"{q['median_after_um']:.2f} um (before -> after the step, over {q.get('n_steps')} step(s))")
    if par.get("cross_check_role"):
        med = rep.get("median_disagreement_um")
        lines.append(f"{par['cross_check_role']} cross-check: median disagreement "
                     f"{f'{med:.2f} um' if med is not None else 'n/a'}, {rep.get('n_disagreement')} step(s) above "
                     f"{par.get('disagreement_um')} um")
    for k, why in (rep.get("errors") or {}).items():
        lines.append(f"t{int(k):03d}: {why}")
    return "\n".join(lines)


def piecewise_summary(view: SeriesView) -> str:
    """The lines ``germquant register --mode piecewise`` prints, from its report ("" without the table)."""
    if not view.has_piecewise:
        return ""
    rep = view.piecewise_report
    if not rep:
        return f"{view.stem}__register_piecewise.csv (no register_piecewise_report.json next to it)"
    par = rep.get("parameters") or {}

    def num(v, fmt):
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "n/a"

    lines = [f"piecewise: {par.get('segments')} segment(s) along the axis ({par.get('axis_source')}), "
             f"{rep.get('n_registered')} step(s) registered on {par.get('channel_role')} (stride {par.get('stride')}, "
             f"margin {num(par.get('margin_um'), 'g')} um, taper {num(par.get('taper'), 'g')})"]
    cum_all = rep.get("cumulative_last_um") or []
    by_seg = (rep.get("quality") or {}).get("by_segment") or []
    last = rep.get("last_t_index")
    for i, s in enumerate(rep.get("segments") or []):
        seg = s.get("segment", i)
        cum = cum_all[seg] if isinstance(cum_all, list) and 0 <= int(seg) < len(cum_all) else None
        q = by_seg[i] if i < len(by_seg) else {}
        ctxt = (f"cumulative at t{int(last):03d} dz={num(cum[0], '+.2f')} dy={num(cum[1], '+.2f')} "
                f"dx={num(cum[2], '+.2f')} um" if cum is not None and last is not None else "no cumulative displacement")
        qtxt = (f"; nearest-neighbour displacement {num(q.get('median_before_um'), '.2f')} -> "
                f"{num(q.get('median_after_um'), '.2f')} um" if q.get("median_after_um") is not None else "")
        lines.append(f"segment {seg} (axis {num(s.get('axis_start_um'), '.1f')} to {num(s.get('axis_end_um'), '.1f')} um, "
                     f"{s.get('n_nuclei_t0')} nuclei): {ctxt}{qtxt}")
    lines.append(f"{len(rep.get('capped') or [])} segment step(s) capped, {len(rep.get('filled') or [])} filled "
                 f"from neighbours")
    for k, why in (rep.get("errors") or {}).items():
        lines.append(f"t{int(k):03d}: {why}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# Register and Link, as the commands run them
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RegisterSettings:
    """The ``germquant register`` options the dock sets (the same defaults); the channel, the taper and the
    experimental rigid fit keep theirs. ``mode`` piecewise adds ``segments`` and ``margin_um`` and always
    registers against the previous timepoint, as the command does."""
    reference: str = gq_register.DEFAULTS["reference"]
    upsample: int = int(gq_register.DEFAULTS["upsample"])
    max_step_um: float = float(gq_register.DEFAULTS["max_step_um"])
    normalization: str = gq_register.DEFAULTS["normalization"]
    mode: str = "rigid"
    segments: int = int(gq_piecewise.DEFAULTS["segments"])
    margin_um: float = float(gq_piecewise.DEFAULTS["margin_um"])

    def __post_init__(self) -> None:
        if self.mode not in REGISTER_MODES:
            raise ValueError(f"mode must be one of {REGISTER_MODES}, got {self.mode!r}")
        if self.mode == "piecewise" and self.reference != "previous":
            raise ValueError("the piecewise registration registers every timepoint against the previous one "
                             "(reference previous)")

    def argv(self, folder: str | Path) -> list[str]:
        if self.mode == "piecewise":
            return ["register", str(folder), "--mode", "piecewise", "--segments", str(int(self.segments)),
                    "--margin-um", f"{float(self.margin_um):g}", "--upsample", str(int(self.upsample)),
                    "--max-step-um", f"{float(self.max_step_um):g}", "--normalization", self.normalization]
        return ["register", str(folder), "--reference", self.reference, "--upsample", str(int(self.upsample)),
                "--max-step-um", f"{float(self.max_step_um):g}", "--normalization", self.normalization]

    def command(self, folder: str | Path) -> str:
        return "germquant " + " ".join(_quote(a) for a in self.argv(folder))


def register(folder: str | Path, settings: RegisterSettings = RegisterSettings()) -> dict:
    """``register_series`` (or ``register_series_piecewise`` for mode piecewise) with the arguments ``germquant
    register`` hands it; the piecewise taper comes from the series config or the code default, as on the
    command line without --taper."""
    if settings.mode == "piecewise":
        return gq_piecewise.register_series_piecewise(folder, segments=int(settings.segments),
                                                      margin_um=float(settings.margin_um), taper=None,
                                                      upsample=int(settings.upsample),
                                                      max_step_um=float(settings.max_step_um), channel=None,
                                                      normalization=settings.normalization, cfg=series_config(folder))
    return gq_register.register_series(folder, reference=settings.reference, upsample=int(settings.upsample),
                                       max_step_um=float(settings.max_step_um), channel=None, rigid=False,
                                       normalization=settings.normalization,
                                       taper=float(gq_register.DEFAULTS["taper"]))


@dataclasses.dataclass(frozen=True)
class LinkSettings:
    """The ``germquant track-link`` options the dock sets; the backend parameters come from the series'
    run_manifest.json config (timelapse.track.*) or the engine's code defaults, as on the command line."""
    backend: str = "overlap"
    all_nuclei: bool = False
    registration: str = "rigid"

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.registration not in REGISTRATIONS:
            raise ValueError(f"registration must be one of {REGISTRATIONS}, got {self.registration!r}")

    def argv(self, folder: str | Path) -> list[str]:
        return (["track-link", str(folder), "--backend", self.backend] + (["--all-nuclei"] if self.all_nuclei else [])
                + ["--registration", self.registration])

    def command(self, folder: str | Path) -> str:
        return "germquant " + " ".join(_quote(a) for a in self.argv(folder))


def series_config(folder: str | Path) -> dict | None:
    """The config the timelapse.* keys are read from when no --config or --profile is given: the series'
    own run_manifest.json config, else None (code defaults). The rule of ``germquant.cli._track_cfg``."""
    data = _read_json(Path(folder) / "run_manifest.json")
    return (data or {}).get("config") or None


def link(folder: str | Path, settings: LinkSettings = LinkSettings()) -> dict:
    """``link_series`` with the arguments ``germquant track-link`` hands it."""
    return gq_track.link_series(folder, backend=settings.backend, gap=None, max_dist_um=None, min_overlap=None,
                                stride=None, germline_only=not settings.all_nuclei, stem=None,
                                cfg=series_config(folder), registration=settings.registration)


def series_defaults(folder: str | Path) -> dict:
    """The dock's starting values from the series config, as the commands read them without options: the
    piecewise segments and margin (timelapse.register.piecewise_*) and the link registration
    (timelapse.track.registration), each with its code default."""
    cfg = series_config(folder)
    try:
        pw = gq_piecewise.piecewise_settings(cfg)
    except ValueError:
        pw = dict(gq_piecewise.DEFAULTS)
    reg = str(gq_track.cfg_get(cfg, "timelapse.track.registration", gq_track.DEFAULTS["registration"])).strip().lower()
    return {"segments": int(pw["segments"]), "margin_um": float(pw["margin_um"]),
            "registration": reg if reg in REGISTRATIONS else "rigid"}


def link_summary(res: dict) -> str:
    """The lines ``germquant track-link`` prints."""
    rec = res["record"]
    lines = [f"{rec['series_id']}: backend {rec['backend']} ({rec['link_source']}): {rec['n_objects']} nucleus-timepoints "
             f"over {len(rec['timepoints'])} timepoint(s) -> {rec['n_tracks']} track(s), {rec['n_full_length']} full "
             f"length; shifts from {rec['shift_source']}"
             + (f" ({rec['registration']} registration)" if rec.get("registration") else "")]
    if rec.get("note"):
        lines.append(f"note: {rec['note']}")
    flipped = [t for t, s in rec.get("axis_sign_by_t", {}).items() if int(s) < 0]
    if flipped:
        lines.append(f"axis orientation flipped relative to t0 at timepoint(s) {', '.join(flipped)} (aligned)")
    return "\n".join(lines)


def backend_choices() -> list[tuple[str, bool, str]]:
    """(backend, usable here, what it is) for the Link backend list. Nothing is imported to find out."""
    acdc = importlib.util.find_spec("cellacdc") is not None
    lap = importlib.util.find_spec("laptrack") is not None
    return [
        ("overlap", True, "Cell-ACDC's overlap tracker, headless" if acdc else
         "Cell-ACDC is not installed here: the engine runs its in-house overlap rule instead and says so"),
        ("overlap_inhouse", True, "the same majority-overlap rule in NumPy"),
        ("laptrack", lap, "LAP linking of the registered centroids with gap closing" if lap else
         "laptrack is not installed in this environment: uv pip install laptrack==0.17.1 (the engine's timelapse extra)"),
        ("none", True, "no linking: the points table only, every nucleus-timepoint without a track"),
    ]
