"""Single-plane results and images in napari (docs/DESIGN_system_packs.md 8.8; P6 acceptance 8 of the
tranche 5 Train B plan, with its decision D37): a planar run's layers at (1.0, dy, dx) and its YX spots image
given a z axis, the Points size from the run's own spot detector, a batch's config found in the batch root's
manifest, the reader's scale for a one-plane stack (Stack.spacing_yx), and the single-plane header line of
the Run dock. The folders and stacks are hand-made (tests/_p6_2d.py), so every engine this suite runs against
gives the same answers; 3D folders keep exactly what they showed before."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from _p6_2d import (
    DX,
    DY,
    PLANAR_ID,
    PLANE,
    RADIUS_UM,
    SP3,
    STACK_ID,
    plane_stack,
    write_cyx_image,
    write_manifest,
    write_planar_results,
    write_stack_results,
    write_summary_only,
)

from napari_germquant import _reader, _results, _run_widget
from napari_germquant._reader import header_spacing_text, is_single_plane, layers_from_stack, open_image
from napari_germquant._results import load_results_layers, spot_marker_size_um
from napari_germquant._results_widget import ResultsWidget
from napari_germquant._run_widget import RunWidget

PLANAR_SCALE = (1.0, DY, DX)


def _by_kind(layers) -> dict:
    out: dict = {}
    for data, kw, kind in layers:
        out.setdefault(kind, []).append((data, kw))
    return out


def test_planar_scale_spots_axis_and_points_size(make_napari_viewer, tmp_path):
    rs = write_planar_results(tmp_path / "plane")
    assert rs.spacing() == PLANAR_SCALE           # voxel_dz_um is NaN: z gets 1.0, y and x the plane's 0.1
    layers = load_results_layers(rs)
    assert [kind for *_, kind in layers] == ["labels", "image", "points"]
    (labels, lkw, _), (blob, bkw, _), (pts, pkw, _) = layers
    assert labels.shape == (1, *PLANE) and tuple(lkw["scale"]) == PLANAR_SCALE
    assert labels.max() == 4 and lkw["features"].shape[0] == 5
    assert blob.shape == (1, *PLANE) and tuple(bkw["scale"]) == PLANAR_SCALE   # the YX TIFF gains its z axis
    assert pts.shape == (5, 3) and np.all(pts[:, 0] == 0.0)
    assert pkw["size"] == pytest.approx(2 * RADIUS_UM)        # 2 x spots.spot_radius_um of the run (spotmax)
    # in the viewer, through the Results dock: every spot lies on the plane, inside its own nucleus
    viewer = make_napari_viewer()
    w = ResultsWidget(viewer)
    assert w.set_folder(str(rs.folder)) == [rs]
    w.load_selected()
    lab = next(layer for layer in viewer.layers if layer.__class__.__name__ == "Labels")
    points = next(layer for layer in viewer.layers if layer.__class__.__name__ == "Points")
    spots_img = next(layer for layer in viewer.layers if layer.__class__.__name__ == "Image")
    assert tuple(np.round(lab.scale, 6)) == PLANAR_SCALE
    assert tuple(np.round(spots_img.scale, 6)) == PLANAR_SCALE
    assert lab.data.shape == (1, *PLANE) and spots_img.data.shape == (1, *PLANE)
    assert np.allclose(points.size, 2 * RADIUS_UM)
    for coord, nucleus in zip(points.data, points.features["nucleus_id"]):
        index = tuple(int(round(float(c))) for c in lab.world_to_data(coord))
        assert index[0] == 0 and lab.data[index] == nucleus, (coord, index)
    # a 3D folder without a manifest keeps its (dz, dy, dx), its ZYX images and the 0.6 um marker
    rs3 = write_stack_results(tmp_path / "stack")
    by = _by_kind(load_results_layers(rs3))
    assert rs3.spacing() == SP3
    assert by["labels"][0][0].shape == (3, *PLANE) and tuple(by["labels"][0][1]["scale"]) == SP3
    assert by["image"][0][0].shape == (3, *PLANE) and tuple(by["image"][0][1]["scale"]) == SP3
    assert by["points"][0][1]["size"] == pytest.approx(0.6)


def test_spacing_keeps_the_old_rule_unless_only_dz_is_nan(tmp_path):
    nan, inf = float("nan"), float("inf")
    cases = [((0.4, 0.2, 0.2), (0.4, 0.2, 0.2)),        # a stack, as before
             ((nan, 0.25, 0.3), (1.0, 0.25, 0.3)),      # a single-plane run
             ((nan, nan, 0.3), (1.0, 1.0, 1.0)),        # no calibrated y either: 1 um isotropic, as before
             ((nan, 0.2, 0.0), (1.0, 1.0, 1.0)),
             ((nan, -0.2, 0.2), (1.0, 1.0, 1.0)),
             ((inf, 0.2, 0.2), (1.0, 1.0, 1.0)),        # only a missing dz (NaN) means one plane
             ((0.0, 0.2, 0.2), (1.0, 1.0, 1.0)),
             ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))]        # a legacy single plane: 1.0 with voxel:UNREADABLE
    for i, (spacing, want) in enumerate(cases):
        assert write_summary_only(tmp_path / f"case{i}", spacing).spacing() == want, spacing


def test_points_size_follows_the_backend():
    def spots(**section):
        return {"spots": section}

    def inst(name="b", **keys):
        return [{"name": name, "role": "x", **keys}]

    cases = [
        (spots(backend="spotmax", spot_radius_um=0.5), "spots", 1.0),
        (spots(spot_radius_um=0.25), "spots", 0.5),                      # spotmax is the default backend
        (spots(), "spots", 0.6),                                         # 2 x the code default 0.3
        (spots(backend="log", log={"diameter_um": 0.8}), "spots", 0.8),
        (spots(backend="log", log={"min_snr": 2.0}), "spots", 0.5),      # the LoG code default diameter
        (spots(backend="log", log=None), "spots", 0.5),
        (spots(backend="log", spot_radius_um=0.45), "spots", 0.5),       # LoG does not read spot_radius_um
        (spots(backend="net3d", spot_radius_um=0.4), "spots", 0.8),      # any other backend: 2 x the radius
        # an instance (spots_<name>): its own keys over the shared section, as the engine resolves them
        (spots(spot_radius_um=0.3, instances=inst("cosa1", spot_radius_um=0.45)), "spots_cosa1", 0.9),
        (spots(spot_radius_um=0.35, instances=inst("cosa1")), "spots_cosa1", 0.7),
        (spots(backend="log", log={"diameter_um": 0.6}, instances=inst(log={"min_snr": 3})), "spots_b", 0.6),
        (spots(instances=inst(backend="log", log={"diameter_um": 0.7})), "spots_b", 0.7),
        (spots(backend="log", log={"diameter_um": 0.7},
               instances=inst(backend="spotmax", spot_radius_um=0.2)), "spots_b", 0.4),
        (spots(spot_radius_um=0.5, instances=inst()), "spots", 1.0),     # the main table
        (spots(instances=inst()), "spots_other", 0.6),                   # an instance the config lacks
        (spots(instances={"b": {}}), "spots_b", 0.6),
        (None, "spots", 0.6), ({}, "spots", 0.6), ({"spots": None}, "spots", 0.6), ("text", "spots", 0.6),
        (spots(spot_radius_um="wide"), "spots", 0.6), (spots(spot_radius_um=-1), "spots", 0.6),
        (spots(spot_radius_um=float("nan")), "spots", 0.6),
        (spots(backend="log", log={"diameter_um": 0}), "spots", 0.6),
    ]
    for cfg, table, want in cases:
        assert spot_marker_size_um(cfg, table) == pytest.approx(want), (cfg, table)


def test_points_size_defaults_are_the_engines():
    from germquant.spots.detect import SPOTMAX_DEFAULTS
    from germquant.spots.log_detect import LOG_DEFAULTS

    assert _results.SPOT_RADIUS_DEFAULT_UM == SPOTMAX_DEFAULTS["spot_radius_um"]
    assert _results.LOG_DIAMETER_DEFAULT_UM == LOG_DEFAULTS["diameter_um"]
    assert _results.DEFAULT_POINT_SIZE_UM == pytest.approx(0.6)


def _sizes(rs) -> dict:
    return {kw["name"]: kw["size"] for _, kw, kind in load_results_layers(rs) if kind == "points"}


def test_points_size_of_a_batch_comes_from_the_batch_root(tmp_path):
    root = tmp_path / "batch"
    rs = write_stack_results(root / "day1" / STACK_ID, config_hash="h1", instances=("cosa1",))
    assert rs.manifest() is None                             # a batch writes no per-image manifest
    config = {"spots": {"backend": "log", "log": {"diameter_um": 0.8}, "instances": [
        {"name": "cosa1", "role": "co", "backend": "spotmax", "spot_radius_um": 0.2}]}}
    write_manifest(root, config_hash="h1", config=config, n_files=1, input_root="C:/in", xy_stride=1)
    assert rs.run_config() == config
    assert _sizes(rs) == {f"{STACK_ID} spots": pytest.approx(0.8),
                          f"{STACK_ID} spots_cosa1": pytest.approx(0.4)}
    # without a done marker the image's config_hash comes from its tables
    (rs.folder / f"{STACK_ID}__done.json").unlink()
    assert rs.config_hash() == "h1" and rs.run_config() == config
    # the root manifest of another config (a later batch into the same root): the run's config is unknown
    write_manifest(root, config_hash="h2", config=config)
    assert rs.run_config() is None
    assert list(_sizes(rs).values()) == [pytest.approx(0.6)] * 2
    # a nearer manifest of another config (another batch's root, below this one) is passed over
    write_manifest(root, config_hash="h1", config=config)
    write_manifest(root / "day1", config_hash="h3", config={"spots": {"spot_radius_um": 0.1}})
    assert rs.run_config() == config
    (root / "day1" / "run_manifest.json").unlink()
    assert rs.run_config() == config
    # a planar folder inside a batch: its points size from the root, its scale from its tables
    plane = write_planar_results(root / "day2" / PLANAR_ID, manifest=False, config_hash="h1")
    by = _by_kind(load_results_layers(plane))
    assert by["points"][0][1]["size"] == pytest.approx(0.8)
    assert tuple(by["labels"][0][1]["scale"]) == PLANAR_SCALE
    # an unreadable manifest is no config
    (root / "run_manifest.json").write_text("{not json", encoding="utf-8")
    assert plane.run_config() is None


def test_points_size_from_the_folders_own_manifest(tmp_path):
    root = tmp_path / "out"
    rs = write_stack_results(root / STACK_ID, config_hash="h1")
    own = {"spots": {"spot_radius_um": 0.45}}
    # a single-image or project run writes its manifest into the image's folder
    write_manifest(rs.folder, config_hash="h1", config=own)
    assert rs.run_config() == own and _sizes(rs) == {f"{STACK_ID} spots": pytest.approx(0.9)}
    # an earlier run's manifest left in a folder that a batch wrote later is passed over for the batch root's
    write_manifest(rs.folder, config_hash="h0", config=own)
    assert rs.run_config() is None
    batch = {"spots": {"spot_radius_um": 0.2}}
    write_manifest(root, config_hash="h1", config=batch)
    assert rs.run_config() == batch and _sizes(rs) == {f"{STACK_ID} spots": pytest.approx(0.4)}
    # an own manifest of the image's hash without a config mapping is no answer either
    write_manifest(rs.folder, config_hash="h1", config=None)
    assert rs.run_config() == batch
    # an image whose config_hash is not recorded (no done marker, no config_hash column): its own manifest
    write_manifest(rs.folder, config_hash="h0", config=own)
    (rs.folder / f"{STACK_ID}__done.json").unlink()
    summary = rs.folder / f"{STACK_ID}__image_summary.csv"
    pd.read_csv(summary).drop(columns=["config_hash"]).to_csv(summary, index=False)
    assert rs.config_hash() is None and rs.run_config() == own
    (rs.folder / "run_manifest.json").unlink()
    assert rs.run_config() is None                         # and no search above without a hash to match


def test_a_done_marker_that_is_not_an_object_still_loads(tmp_path, monkeypatch):
    # valid JSON that is not an object is no done marker: the image's config_hash comes from its tables and
    # every layer loads, the source image at stride 1 too (the Results dock said "could not load" before)
    root = tmp_path / "batch"
    rs = write_stack_results(root / STACK_ID, config_hash="h1")
    write_manifest(root, config_hash="h1", config={"spots": {"spot_radius_um": 0.2}})
    src = write_cyx_image(tmp_path / "cells_001.ome.tif", np.zeros((2, *PLANE), np.uint16), ["DNA", "foci"])
    summary = rs.file("image_summary.csv")
    pd.read_csv(summary).assign(file_path=str(src)).to_csv(summary, index=False)
    stack = plane_stack(np.zeros((2, 3, *PLANE), np.uint16), SP3, spacing_ok=True)
    monkeypatch.setattr(_reader.api, "read_image", lambda p, xy_stride=1: stack if xy_stride == 1 else None)
    for text in ("[1, 2]", "7", '"done"'):
        (rs.folder / f"{STACK_ID}__done.json").write_text(text, encoding="utf-8")
        assert rs.config_hash() == "h1"
        layers = load_results_layers(rs, include_source=True)
        assert [kind for *_, kind in layers] == ["image", "image", "labels", "image", "points"], text
        assert layers[-1][1]["size"] == pytest.approx(0.4)


def test_reader_scales_a_single_plane(tmp_path, monkeypatch):
    path = tmp_path / "cells_001.ome.tif"
    data = np.arange(2 * 1 * 8 * 12, dtype=np.uint16).reshape(2, 1, 8, 12)
    # the engine's bioio reader on a calibrated plane (P2b): no z step, so spacing_ok False, and spacing_yx
    plane = plane_stack(data, (1.0, 1.0, 1.0), spacing_ok=False, spacing_yx=(0.11, 0.12), path=path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # no "voxel size unreadable" warning
        layers = layers_from_stack(plane, path)
    assert [tuple(kw["scale"]) for _, kw, _ in layers] == [(1.0, 0.11, 0.12)] * 2
    assert layers[1][1]["metadata"]["napari_germquant"]["spacing_um"] == (1.0, 0.11, 0.12)
    assert np.array_equal(layers[1][0], data[1])
    # a strided read: spacing_yx is on the stack's own grid, as Stack.spacing is
    strided = plane_stack(data[:, :, ::4, ::4], (1.0, 4.0, 4.0), spacing_ok=False, spacing_yx=(0.44, 0.48),
                          xy_stride=4, path=path)
    assert tuple(layers_from_stack(strided, path)[0][1]["scale"]) == (1.0, 0.44, 0.48)
    # no plane spacing (an engine before P2b, or an uncalibrated plane): 1 um per pixel and the warning
    bare = plane_stack(data, (1.0, 1.0, 1.0), spacing_ok=False, path=path)
    with pytest.warns(UserWarning, match="voxel size unreadable"):
        assert tuple(layers_from_stack(bare, path)[0][1]["scale"]) == (1.0, 1.0, 1.0)
    for bad in ((0.0, 0.1), (float("nan"), 0.1), (0.1,), "0.1"):
        odd = plane_stack(data, (1.0, 1.0, 1.0), spacing_ok=False, spacing_yx=bad, path=path)
        with pytest.warns(UserWarning, match="voxel size unreadable"):
            assert tuple(layers_from_stack(odd, path)[0][1]["scale"]) == (1.0, 1.0, 1.0)
    # a calibrated one-plane stack: (1.0, dy, dx) from spacing_yx; without it the file's (dz, dy, dx)
    ok = plane_stack(data, (0.5, 0.11, 0.12), spacing_ok=True, spacing_yx=(0.11, 0.12), path=path)
    assert tuple(layers_from_stack(ok, path)[0][1]["scale"]) == (1.0, 0.11, 0.12)
    ok_bare = plane_stack(data, (0.5, 0.11, 0.12), spacing_ok=True, path=path)
    assert tuple(layers_from_stack(ok_bare, path)[0][1]["scale"]) == (0.5, 0.11, 0.12)
    # a stack of several planes keeps its (dz, dy, dx), whatever spacing_yx holds
    stack = plane_stack(np.zeros((1, 3, 8, 8), np.uint16), (0.4, 0.2, 0.2), spacing_ok=True,
                        spacing_yx=(0.1, 0.1))
    assert tuple(layers_from_stack(stack, path)[0][1]["scale"]) == (0.4, 0.2, 0.2)
    # open_image: the header check, then the stack the engine reads
    header = {"sizes": {"T": 1, "C": 2, "Z": 1, "Y": 8, "X": 12}, "spacing": (1.0, 1.0, 1.0),
              "spacing_ok": False, "spacing_yx_um": (0.11, 0.12), "channel_names": ["ch0", "ch1"],
              "dtype": "uint16", "is_2d": True}
    monkeypatch.setattr(_reader.api, "read_image_metadata", lambda p: header)
    monkeypatch.setattr(_reader.api, "read_image", lambda p, xy_stride=1: plane)
    assert [tuple(kw["scale"]) for _, kw, _ in open_image(path)] == [(1.0, 0.11, 0.12)] * 2


def test_source_image_overlays_the_run(tmp_path, monkeypatch):
    # "also open the source image" puts the source on the run's own grid, whatever pixel size the reader
    # finds: a calibrated plane run without io.planar (tables at 1 um, voxel:UNREADABLE), and a planar run
    # whose pixel size came from io.voxel_size_um on an uncalibrated plane; the reader's "voxel size
    # unreadable" warning comes only where the layers sit at its 1 um guess
    src = write_cyx_image(tmp_path / "cells_001.ome.tif", np.zeros((2, *PLANE), np.uint16), ["DNA", "foci"])
    data = np.zeros((2, 1, *PLANE), np.uint16)
    calibrated = plane_stack(data, (1.0, 1.0, 1.0), spacing_ok=False, spacing_yx=(DY, DX))
    bare = plane_stack(data, (1.0, 1.0, 1.0), spacing_ok=False)
    cases = (((1.0, 1.0, 1.0), calibrated, 1),       # the run measured at 1 um, so the source sits there too
             ((float("nan"), DY, DX), bare, 0),       # the override's grid, not the reader's guess
             ((1.0, 1.0, 1.0), bare, 1))              # an uncalibrated plane at 1 um: as before
    for i, (run_spacing, stack, n_warned) in enumerate(cases):
        monkeypatch.setattr(_reader.api, "read_image", lambda p, xy_stride=1, s=stack: s)
        rs = write_planar_results(tmp_path / f"run{i}")
        for name in ("image_summary", "nuclei", "spots"):
            table = rs.file(f"{name}.csv")
            df = pd.read_csv(table)
            df["voxel_dz_um"], df["voxel_dy_um"], df["voxel_dx_um"] = run_spacing
            df["file_path"] = str(src)
            df.to_csv(table, index=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layers = load_results_layers(rs, include_source=True)
        assert [kind for *_, kind in layers[:3]] == ["image", "image", "labels"]
        assert [tuple(kw["scale"]) for _, kw, _ in layers[:3]] == [rs.spacing()] * 3, i
        infos = [kw["metadata"]["napari_germquant"] for _, kw, _ in layers[:2]]
        assert [info["spacing_um"] for info in infos] == [rs.spacing()] * 2
        unreadable = [str(w.message) for w in caught if "voxel size unreadable" in str(w.message)]
        assert len(unreadable) == n_warned, (i, unreadable)


def test_header_line_says_single_plane():
    unreadable = "single plane, voxel size UNREADABLE (1 um assumed)"
    plane = {"sizes": {"T": 1, "C": 2, "Z": 1, "Y": 256, "X": 256}, "spacing": (1.0, 1.0, 1.0),
             "spacing_ok": False, "spacing_yx_um": (0.1, 0.1)}
    assert is_single_plane(plane)
    assert header_spacing_text(plane) == "single plane, voxel um dy=0.100 dx=0.100"
    assert header_spacing_text({**plane, "spacing_yx_um": None}) == unreadable
    assert header_spacing_text({**plane, "spacing_yx_um": [0.0, 0.1]}) == unreadable
    # an nd2 plane: no Z axis in the sizes, the file's own dy and dx
    nd2 = {"sizes": {"C": 1, "Y": 64, "X": 64}, "spacing": (1.0, 0.108, 0.109), "spacing_ok": True}
    assert header_spacing_text(nd2) == "single plane, voxel um dy=0.108 dx=0.109"
    # a project's header record says single_plane itself (germquant.project.headers.read_header)
    record = {"sizes": {"C": 1, "Y": 64, "X": 64}, "single_plane": True, "spacing": [1.0, 1.0, 1.0],
              "spacing_ok": False, "spacing_yx_um": [0.2, 0.25]}
    assert header_spacing_text(record) == "single plane, voxel um dy=0.200 dx=0.250"
    assert not is_single_plane({**record, "single_plane": False})
    # a stack: the line as before
    stack = {"sizes": {"Z": 20, "C": 3, "Y": 80, "X": 320}, "spacing": (0.4, 0.2, 0.2), "spacing_ok": True}
    assert not is_single_plane(stack)
    assert header_spacing_text(stack) == "voxel um dz=0.400 dy=0.200 dx=0.200"
    assert header_spacing_text({**stack, "spacing_ok": False}) == "voxel size UNREADABLE (1 um assumed)"
    assert header_spacing_text({"sizes": {"Z": 4}}) == "voxel um dz=1.000 dy=1.000 dx=1.000"
    assert not is_single_plane({}) and not is_single_plane({"sizes": {}})


def test_run_dock_header_says_single_plane(qtbot, tmp_path, monkeypatch):
    p = write_cyx_image(tmp_path / "cells_001.ome.tif", np.zeros((2, 32, 32), np.uint16), ["DNA", "foci"])
    w = RunWidget(None)
    qtbot.addWidget(w)
    w.set_image(p)
    text = w.image_info.text()
    assert "Z=1" in text and "\nsingle plane, " in text and "channels: ['DNA', 'foci']" in text
    # an engine whose header carries the plane's pixel size (P2b's spacing_yx_um)
    real = _run_widget.api.read_image_metadata
    monkeypatch.setattr(_run_widget.api, "read_image_metadata",
                        lambda path: {**real(path), "spacing_yx_um": (0.1, 0.1)})
    w.set_image(p)
    assert "\nsingle plane, voxel um dy=0.100 dx=0.100\n" in w.image_info.text()


def test_loading_a_planar_folder_writes_nothing(tmp_path):
    rs = write_planar_results(tmp_path / "plane")
    before = {p.name: p.read_bytes() for p in rs.folder.iterdir()}
    layers = load_results_layers(rs)
    assert {p.name: p.read_bytes() for p in rs.folder.iterdir()} == before
    pts_meta = layers[-1][1]["metadata"]["napari_germquant_results"]
    assert pts_meta["spacing_um"] == PLANAR_SCALE and pts_meta["table"] == "spots"
    assert json.loads((rs.folder / f"{PLANAR_ID}__done.json").read_text(encoding="utf-8"))["planar"] is True
