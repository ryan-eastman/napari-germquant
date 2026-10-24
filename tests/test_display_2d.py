"""Single-plane results and images in napari (docs/DESIGN_system_packs.md 8.8; P6 acceptance 8 of the
tranche 5 Train B plan, with its decision D37): a planar run's layers at (1.0, dy, dx) and its YX spots image
given a z axis, the Points size from the run's own spot detector, a batch's config found in the batch root's
manifest, the reader's scale for a one-plane stack (Stack.spacing_yx), and the single-plane header line of
the Run dock. The folders and stacks are hand-made (tests/_p6_2d.py), so every engine this suite runs against
gives the same answers; 3D folders keep exactly what they showed before."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from _p6_2d import plane_stack, write_cyx_image

from napari_germquant import _reader, _run_widget
from napari_germquant._reader import header_spacing_text, is_single_plane, layers_from_stack, open_image
from napari_germquant._run_widget import RunWidget


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
