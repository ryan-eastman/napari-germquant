"""The Time-lapse dock and its glue on a synthetic three-timepoint series written straight into a results folder
the way ``germquant run --t`` lays one out (``t<NNN>/<stem>_t<NNN>__nuclei.csv``, ``__nuclei_labels.tif``,
``__image_summary.csv``, ``__done.json``, ``__ref_stride2.npy``, the series record and the root
run_manifest.json): six germline nuclei that flow one voxel along x per timepoint, a drift of (0, 1, 2) voxels
per timepoint, one off-gonad nucleus, and nucleus ids shuffled per timepoint.

Checked: the lazy time-stacked labels (nothing read until a plane is shown, then the pixels of the TIFFs), the
reference volumes, the registered view on the grid the engine's own export uses (track.place_frame), the
Tracks layer in raw and registered coordinates, Register and Link run in thread workers with the arguments the
commands pass, and the files a Link from the dock writes equal those of `germquant track-link`."""
from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_germquant import _timelapse as TL
from napari_germquant._timelapse_widget import TimelapseWidget
from germquant import cli
from germquant.timelapse import layout as gq_layout
from germquant.timelapse import register as gq_register
from germquant.timelapse import track as gq_track

STEM = "20261005_dlw93_nohs_herm_001"
SPACING = (0.4, 0.2, 0.2)
SHAPE = (8, 56, 96)                    # (Z, Y, X) voxels
T = 3
PERIOD_S = 300.0
DRIFT_PX = np.array([0, 1, 2])         # the sample moves by this many voxels per timepoint
REF_STRIDE = 2
R_UM = 1.0
GERMLINE = {i + 1: np.array([4.0, y, x]) for i, (y, x) in
            enumerate([(18, 18), (18, 40), (18, 62), (36, 18), (36, 40), (36, 62)])}
JUNK = {7: np.array([4.0, 46.0, 84.0])}


def _centre(identity: int, k: int) -> np.ndarray:
    if identity in JUNK:
        return JUNK[identity] + k * DRIFT_PX
    return GERMLINE[identity] + np.array([0, 0, k]) + k * DRIFT_PX     # one voxel of flow along x per timepoint


def write_series(root: Path, *, missing: int | None = None) -> Path:
    """The series folder; ``missing``: a timepoint the record lists as done whose label image is deleted."""
    out = root / STEM
    zz, yy, xx = np.indices(SHAPE).astype(np.float32)
    rng = np.random.default_rng(7)
    entries = []
    for k in range(T):
        idents = list(GERMLINE) + list(JUNK)
        order = rng.permutation(len(idents))
        nid = {ident: int(order[j]) + 1 for j, ident in enumerate(idents)}
        lab = np.zeros(SHAPE, np.int32)
        ref = np.zeros(SHAPE, np.float32)
        rows = []
        for ident in idents:
            c = _centre(ident, k)
            d2 = ((zz - c[0]) * SPACING[0]) ** 2 + ((yy - c[1]) * SPACING[1]) ** 2 + ((xx - c[2]) * SPACING[2]) ** 2
            lab[d2 <= R_UM ** 2] = nid[ident]
            ref[d2 <= R_UM ** 2] = 800.0 + 60.0 * ident
            vox = np.argwhere(lab == nid[ident])
            cz, cy, cx = (vox.mean(axis=0) * np.array(SPACING)).tolist()
            rows.append({"nucleus_id": nid[ident], "volume_um3": float(len(vox)) * float(np.prod(SPACING)),
                         "n_voxels": int(len(vox)), "centroid_z_um": cz, "centroid_y_um": cy, "centroid_x_um": cx,
                         "in_germline": ident in GERMLINE, "central_element_mean_intensity": 1000.0 + ident})
        sub = out / f"t{k:03d}"
        sub.mkdir(parents=True)
        iid = f"{STEM}_t{k:03d}"
        tifffile.imwrite(sub / f"{iid}__nuclei_labels.tif", lab, compression="zlib")
        np.save(sub / f"{iid}__ref_stride{REF_STRIDE}.npy", ref[:, ::REF_STRIDE, ::REF_STRIDE].astype(np.uint16))
        meta = {"image_id": iid, "file_path": "x.nd2", "genotype": "DLW93", "sex": "herm", "voxel_dz_um": SPACING[0],
                "voxel_dy_um": SPACING[1], "voxel_dx_um": SPACING[2],
                "channel_map_json": json.dumps({"central_element": 1, "brightfield": 0}), "config_hash": "cafe"}
        nuclei = pd.DataFrame(rows).sort_values("nucleus_id")
        for key, value in meta.items():
            nuclei[key] = value
        nuclei.to_csv(sub / f"{iid}__nuclei.csv", index=False)
        pd.DataFrame([{"n_nuclei": len(rows), "n_germline_nuclei": len(GERMLINE), "qc_pass": True, "qc_flags": "",
                       "total_germline_length_um": 20.0, **meta}]).to_csv(sub / f"{iid}__image_summary.csv", index=False)
        (sub / f"{iid}__done.json").write_text(json.dumps({"image_id": iid, "t_index": k, "t_seconds": k * PERIOD_S}))
        entries.append({"t_index": k, "t_seconds": k * PERIOD_S, "image_id": iid, "out_dir": sub.name, "status": "done"})
    record = {"series_id": STEM, "file_path": "x.nd2", "config_hash": "cafe", "n_timepoints": T, "period_s": PERIOD_S,
              "t_seconds": [k * PERIOD_S for k in range(T)], "requested": list(range(T)), "done": list(range(T)),
              "failed": [], "timepoints": entries, "complete": True}
    (out / f"{STEM}__series.json").write_text(json.dumps(record, indent=1))
    (out / "run_manifest.json").write_text(json.dumps({"config": {"timelapse": {"track": {"max_dist_um": 2.5}}}}))
    if missing is not None:
        (out / f"t{missing:03d}" / f"{STEM}_t{missing:03d}__nuclei_labels.tif").unlink()
    return out


@pytest.fixture
def series(tmp_path) -> Path:
    return write_series(tmp_path)


def _labels(folder: Path, k: int) -> np.ndarray:
    return tifffile.imread(folder / f"t{k:03d}" / f"{STEM}_t{k:03d}__nuclei_labels.tif")


def _ref(folder: Path, k: int) -> np.ndarray:
    return np.load(folder / f"t{k:03d}" / f"{STEM}_t{k:03d}__ref_stride{REF_STRIDE}.npy")


# ---- the glue ------------------------------------------------------------------------------------------------

def test_open_series_and_lazy_label_stack(series, monkeypatch):
    view = TL.open_series(series)
    assert view.stem == STEM and view.timepoints == [0, 1, 2] and view.spacing == SPACING
    assert view.label_shapes == {k: SHAPE for k in range(T)} and view.label_dtype == np.int32
    assert view.shift_source == "none" and view.points is None and set(view.refs) == {0, 1, 2}
    text = TL.info_text(view)
    assert "3 timepoint(s) on disk, t000 to t002, period 300 s" in text and "at xy stride 2" in text
    assert "registration: none yet" in text and "tracks: none yet" in text

    reads = []
    real = TL.read_tiff_plane
    monkeypatch.setattr(TL, "read_tiff_plane", lambda *a: reads.append(a[1]) or real(*a))
    stack = TL.labels_stack(view)
    assert stack.shape == (T, *SHAPE) and stack.dtype == np.int32 and reads == []     # lazy: nothing read yet
    assert np.array_equal(np.asarray(stack[1, 4]), _labels(series, 1)[4]) and reads == [4]   # one page for one plane
    for k in range(T):
        assert np.array_equal(np.asarray(stack[k]), _labels(series, k))
    half = TL.labels_stack(view, stride=2)
    assert half.shape == (T, 8, 28, 48) and np.array_equal(np.asarray(half[2]), _labels(series, 2)[:, ::2, ::2])

    ref, ref_spacing = TL.reference_stack(view)
    assert ref.shape == (T, 8, 28, 48) and ref.dtype == np.uint16 and ref_spacing == (0.4, 0.4, 0.4)
    assert np.array_equal(np.asarray(ref[1]), _ref(series, 1))

    data, kw, kind = TL.labels_layer(view, stride=2)
    assert kind == "labels" and kw["scale"] == (1.0, 0.4, 0.4, 0.4) and kw["translate"] == (0.0, 0.0, 0.0, 0.0)
    assert kw["metadata"][TL.META_KEY]["coordinates"] == "raw" and kw["metadata"][TL.META_KEY]["xy_stride"] == 2
    _, rkw, rkind = TL.reference_layer(view)
    assert rkind == "image" and rkw["scale"] == (1.0, 0.4, 0.4, 0.4) and rkw["contrast_limits"][1] > rkw["contrast_limits"][0]
    assert TL.tracks_layer(view) is None and TL.unlinked_layer(view) is None


def test_registered_frames_use_the_engine_export_grid(series):
    shifts = pd.DataFrame({"t_index": range(T), "dz_um": [k * DRIFT_PX[0] * SPACING[0] for k in range(T)],
                           "dy_um": [k * DRIFT_PX[1] * SPACING[1] for k in range(T)],
                           "dx_um": [k * DRIFT_PX[2] * SPACING[2] for k in range(T)]})
    shifts.to_csv(series / f"{STEM}__register_shifts.csv", index=False)
    view = TL.open_series(series)
    assert view.shift_source == "register_shifts"
    vsh = gq_layout.voxel_shifts(view.shifts, SPACING)
    assert [vsh[k].tolist() for k in range(T)] == [(k * DRIFT_PX).tolist() for k in range(T)]
    reg = TL.labels_stack(view, registered=True)
    box = {"start": [0, 0, 0], "stop": list(SHAPE)}
    for k in range(T):
        expected = gq_track.place_frame(_labels(series, k), vsh[k], box, 1)
        assert np.array_equal(np.asarray(reg[k]), expected), k
    reg2 = TL.labels_stack(view, registered=True, stride=2)
    assert np.array_equal(np.asarray(reg2[2]), gq_track.place_frame(_labels(series, 2), vsh[2], box, 2))
    # the drift is gone in the registered view: a germline nucleus moves only by its own flow
    ident_t0 = _labels(series, 0)[4, 18, 18]
    assert ident_t0 > 0 and np.asarray(reg[2])[4, 18, 18 + 2] > 0
    ref, ref_spacing = TL.reference_stack(view, registered=True)
    rsh = gq_layout.voxel_shifts(view.shifts, ref_spacing)
    assert np.array_equal(np.asarray(ref[2]), gq_track.place_frame(_ref(series, 2), rsh[2],
                                                                   {"start": [0, 0, 0], "stop": [8, 28, 48]}, 1))


def test_a_missing_timepoint_is_an_empty_frame(tmp_path):
    folder = write_series(tmp_path, missing=1)
    view = TL.open_series(folder)
    assert view.timepoints == [0, 2] and view.layout.missing == [1] and view.n_frames == 3
    stack = TL.labels_stack(view)
    assert stack.shape[0] == 3 and not np.asarray(stack[1]).any()
    assert np.array_equal(np.asarray(stack[2]), _labels(folder, 2))
    assert "[1] are in the series record but their files are missing" in TL.info_text(view)


def test_tracks_data_raw_and_registered():
    pts = pd.DataFrame({"time_track_id": pd.array([2, 1, 1, pd.NA, 2], dtype="Int64"), "t_index": [0, 1, 0, 1, 1],
                        "nucleus_id": [5, 6, 7, 8, 9], "z_um": [1.0, 2.0, 3.0, 4.0, 5.0], "y_um": [1.0] * 5,
                        "x_um": [0.5, 1.5, 2.5, 3.5, 4.5], "reg_z_um": [1.1, 2.1, 3.1, 4.1, 5.1], "reg_y_um": [0.0] * 5,
                        "reg_x_um": [0.0, 0.1, 0.2, 0.3, 0.4], "series_id": "s", "config_hash": "c"})
    data, feats, dropped = TL.tracks_data(pts, registered=False)
    assert dropped == 0 and data[:, :2].tolist() == [[1, 0], [1, 1], [2, 0], [2, 1]]
    assert data[:, 2].tolist() == [3.0, 2.0, 1.0, 5.0] and list(feats["nucleus_id"]) == [7, 6, 5, 9]
    assert "series_id" not in feats.columns and "config_hash" not in feats.columns
    reg, _, _ = TL.tracks_data(pts, registered=True)
    assert reg[:, 2].tolist() == [3.1, 2.1, 1.1, 5.1] and reg[:, 4].tolist() == [0.2, 0.1, 0.0, 0.4]
    free, free_feats = TL.unlinked_data(pts, registered=False)
    assert free.tolist() == [[1.0, 4.0, 1.0, 3.5]] and list(free_feats["nucleus_id"]) == [8]


def test_settings_reproduce_the_command_line_arguments(series, monkeypatch):
    """Register and Link call the engine functions with exactly the arguments `germquant register` and
    `germquant track-link` pass for the same choices (the series' run_manifest config included)."""
    class Stop(BaseException):
        pass

    def recorder(fn, sink):
        sig = inspect.signature(fn)

        def rec(*a, **k):
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            sink.append({key: (str(v) if isinstance(v, Path) else v) for key, v in bound.arguments.items()})
            raise Stop
        return rec

    seen: list = []
    monkeypatch.setattr(gq_register, "register_series", recorder(gq_register.register_series, seen))
    settings = TL.RegisterSettings(reference="t0", upsample=4, max_step_um=2.5, normalization="phase")
    with pytest.raises(Stop):
        TL.register(series, settings)
    with pytest.raises(Stop):
        cli.main(settings.argv(series))
    assert seen[0] == seen[1] and seen[0]["reference"] == "t0" and seen[0]["max_step_um"] == 2.5
    assert TL.RegisterSettings().argv(series)[2:] == ["--reference", "previous", "--upsample", "10", "--max-step-um", "5",
                                                      "--normalization", "none"]

    seen.clear()
    monkeypatch.setattr(gq_track, "link_series", recorder(gq_track.link_series, seen))
    for ls in (TL.LinkSettings("overlap_inhouse"), TL.LinkSettings("laptrack", all_nuclei=True)):
        with pytest.raises(Stop):
            TL.link(series, ls)
        with pytest.raises(Stop):
            cli.main(ls.argv(series))
    assert seen[0] == seen[1] and seen[2] == seen[3]
    assert seen[0]["cfg"] == {"timelapse": {"track": {"max_dist_um": 2.5}}} and seen[0]["germline_only"] is True
    assert seen[2]["backend"] == "laptrack" and seen[2]["germline_only"] is False
    assert TL.series_config(series) == cli._track_cfg(type("A", (), {"series_dir": str(series), "config": None,
                                                                     "profile": None})())
    with pytest.raises(ValueError):
        TL.LinkSettings("trackmate")


def test_backend_choices_say_what_is_installed():
    choices = {b: (usable, what) for b, usable, what in TL.backend_choices()}
    assert list(choices) == list(gq_track.BACKENDS)
    assert choices["overlap"][0] and choices["overlap_inhouse"][0] and choices["none"][0]
    import importlib.util

    assert choices["laptrack"][0] == (importlib.util.find_spec("laptrack") is not None)


# ---- the dock -----------------------------------------------------------------------------------------------

def _wait(qtbot, w, attr):
    qtbot.waitUntil(lambda: getattr(w, attr) is not None or w.last_error is not None, timeout=120_000)
    qtbot.waitUntil(lambda: not w.runner.busy, timeout=30_000)
    assert w.last_error is None, w.last_error


def test_dock_opens_registers_links_and_toggles(make_napari_viewer, qtbot, series, tmp_path):
    copy = tmp_path / "cli_copy" / STEM
    shutil.copytree(series, copy)
    viewer = make_napari_viewer()
    w = TimelapseWidget(viewer)
    assert not w.register_btn.isEnabled()
    view = w.open_folder(str(series))
    assert view is not None and w.register_btn.isEnabled() and "t000 to t002" in w.info_label.text()
    names = [layer.name for layer in viewer.layers]
    assert names == [f"{STEM} reference", f"{STEM} nuclei"]                 # no tracks before Link
    lab = w.layers["labels"]
    assert type(lab).__name__ == "Labels" and lab.data.shape == (T, *SHAPE) and type(lab.data).__module__.startswith("dask")
    assert tuple(np.round(lab.scale, 6)) == (1.0, 0.4, 0.2, 0.2)
    assert np.array_equal(np.asarray(lab.data[2]), _labels(series, 2))
    assert "not registered yet" in w.shifts_label.text() and w.shifts_table.rowCount() == 0

    # Register in a thread worker: the table and the summary appear, and they are the files on disk
    assert w.register() and not w.register_btn.isEnabled()
    _wait(qtbot, w, "last_register")
    table = pd.read_csv(series / f"{STEM}__register_shifts.csv")
    assert w.shifts_table.rowCount() == T and len(table) == T
    assert w.shifts_table.item(2, 0).text() == "2"
    assert w.shifts_table.item(2, 3).text() == f"{table['dx_um'].iloc[2]:+.3f}"
    assert "step(s) registered" in w.shifts_label.text() and "germquant register" in w.status_label.text()
    assert w.view.shift_source == "register_shifts"

    # Link (overlap: Cell-ACDC is not installed here, so the engine's in-house overlap rule runs and says so)
    w.select_backend("overlap")
    assert w.link()
    _wait(qtbot, w, "last_link")
    rec = w.last_link["record"]
    assert rec["backend"] in ("overlap", "overlap_inhouse") and rec["n_tracks"] == len(GERMLINE)
    assert rec["n_full_length"] == len(GERMLINE) and "track(s)" in w.link_label.text()
    tracks = w.layers["tracks"]
    assert type(tracks).__name__ == "Tracks" and tracks.name == f"{STEM} tracks"
    pts = pd.read_csv(series / f"{STEM}__track_points.csv")
    linked = pts[pts["time_track_id"].notna()].sort_values(["time_track_id", "t_index"])
    assert np.allclose(tracks.data[:, 1:], linked[["t_index", "z_um", "y_um", "x_um"]].to_numpy(float))
    assert np.array_equal(tracks.data[:, 0], linked["time_track_id"].to_numpy(float))
    assert "nucleus_id" in tracks.features.columns and "unlinked" not in w.layers

    # the toggle: tracks take the registered columns, labels and reference volumes move by the whole-voxel shift
    w.set_coordinates(True)
    assert w.registered
    assert np.allclose(w.layers["tracks"].data[:, 2:], linked[["reg_z_um", "reg_y_um", "reg_x_um"]].to_numpy(float))
    assert list(w.layers["tracks"].features["nucleus_id"]) == linked["nucleus_id"].tolist()
    vsh = gq_layout.voxel_shifts(w.view.shifts, SPACING)
    box = {"start": [0, 0, 0], "stop": list(SHAPE)}
    assert np.array_equal(np.asarray(w.layers["labels"].data[2]), gq_track.place_frame(_labels(series, 2), vsh[2], box, 1))
    assert w.layers["labels"].metadata[TL.META_KEY]["coordinates"] == "registered"
    w.stride_spin.setValue(2)
    assert w.layers["labels"].data.shape == (T, 8, 28, 48)
    assert tuple(np.round(w.layers["labels"].scale, 6)) == (1.0, 0.4, 0.4, 0.4)
    w.set_coordinates(False)
    assert np.allclose(w.layers["tracks"].data[:, 2:], linked[["z_um", "y_um", "x_um"]].to_numpy(float))
    assert len([layer for layer in viewer.layers if layer.name.startswith(STEM)]) == 3      # updated in place

    # the same steps on the copy through the command line write the same tables (timestamps aside)
    assert cli.main(["register", str(copy)]) == 0
    assert cli.main(["track-link", str(copy), "--backend", "overlap"]) == 0
    for name in ("register_shifts", "track_points", "tracks"):
        a = pd.read_csv(series / f"{STEM}__{name}.csv").drop(columns=["run_timestamp"])
        b = pd.read_csv(copy / f"{STEM}__{name}.csv").drop(columns=["run_timestamp"])
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=name)

    # backend none: no tracks, every nucleus-timepoint as a point
    w.select_backend("none")
    assert w.link()
    _wait(qtbot, w, "last_link")
    assert "tracks" not in w.layers and f"{STEM} tracks" not in [layer.name for layer in viewer.layers]
    free = w.layers["unlinked"]
    assert type(free).__name__ == "Points" and len(free.data) == len(GERMLINE) * T and free.ndim == 4


def test_dock_reports_a_folder_that_is_not_a_series(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    w = TimelapseWidget(viewer)
    assert w.open_folder(str(tmp_path)) is None
    assert "cannot open" in w.info_label.text() and not w.link_btn.isEnabled() and len(viewer.layers) == 0
    assert not w.register()


def test_register_without_reference_volumes_fails_in_the_dock(make_napari_viewer, qtbot, series):
    for p in series.glob("t*/*__ref_stride*.npy"):
        p.unlink()
    viewer = make_napari_viewer()
    w = TimelapseWidget(viewer)
    w.open_folder(series)
    assert "reference volumes: none" in w.info_label.text() and "reference" not in w.layers
    assert w.register()
    qtbot.waitUntil(lambda: w.last_error is not None, timeout=60_000)
    qtbot.waitUntil(lambda: not w.runner.busy, timeout=30_000)
    assert isinstance(w.last_error, gq_register.ReferenceVolumesMissing)
    assert "register FAILED" in w.status_label.text() and w.register_btn.isEnabled()


def test_dock_through_the_plugin_menu_updates_layers_in_place(make_napari_viewer, series):
    """napari hands a plugin dock a public-only proxy of the viewer; the toggle must still update the same
    layers instead of adding new ones."""
    viewer = make_napari_viewer()
    _dock, w = viewer.window.add_plugin_dock_widget("napari-germquant", "Time-lapse")
    w.open_folder(series)
    names = [layer.name for layer in viewer.layers]
    w.set_coordinates(True)
    w.set_coordinates(False)
    w.load_layers()
    assert [layer.name for layer in viewer.layers] == names == [f"{STEM} reference", f"{STEM} nuclei"]
