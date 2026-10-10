"""The reader contribution: which paths it claims, one Image layer per channel with the physical scale, the
same pixels the engine reads, the stride option, and the refusal of time-lapse and multi-scene files from
the header alone (no pixel is read)."""
from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import CHANNELS, IMAGE_ID, LOCAL_ND2, NAS_ND2, SPACING, write_ome

from napari_germquant import _reader
from napari_germquant._reader import check_single_timepoint, napari_get_reader, open_image, read_layers
from germquant import api


def test_get_reader_claims_the_engine_suffixes_only():
    for p in ("x.nd2", "a.ome.tif", "a.tif", "b.czi", "c.lif", "d.zarr", ["a.nd2", "b.czi"]):
        assert callable(napari_get_reader(p)), p
    for p in ("x.png", "notes.txt", ["a.nd2", "b.png"]):
        assert napari_get_reader(p) is None, p


def test_open_image_gives_one_scaled_layer_per_channel(synthetic_image):
    layers = open_image(synthetic_image)
    assert len(layers) == 3
    stack = api.read_image(synthetic_image)
    for i, ((data, kw, kind), ch) in enumerate(zip(layers, CHANNELS)):
        assert kind == "image" and data.shape == (20, 80, 320)
        assert tuple(kw["scale"]) == SPACING and kw["name"] == f"{IMAGE_ID} [{ch}]"
        info = kw["metadata"]["napari_germquant"]
        assert info["path"] == str(synthetic_image) and info["channel_index"] == i and info["channel_name"] == ch
        assert info["spacing_um"] == SPACING and info["xy_stride"] == 1 and info["spacing_ok"]
        assert np.array_equal(data, stack.data[i])              # the pixels the pipeline measures
    assert read_layers(str(synthetic_image))[2][1]["name"].endswith("[545]")
    assert len(read_layers([str(synthetic_image), str(synthetic_image)])) == 6


def test_open_image_with_stride_scales_the_layer(synthetic_image):
    (data, kw, _), *_ = open_image(synthetic_image, xy_stride=4)
    assert data.shape == (20, 20, 80) and tuple(kw["scale"]) == (0.4, 0.8, 0.8)
    assert kw["metadata"]["napari_germquant"]["xy_stride"] == 4


def test_viewer_opens_through_the_plugin(make_napari_viewer, synthetic_image):
    viewer = make_napari_viewer()
    viewer.open(str(synthetic_image), plugin="napari-germquant")
    assert [layer.name for layer in viewer.layers] == [f"{IMAGE_ID} [{c}]" for c in CHANNELS]
    assert tuple(np.round(viewer.layers[0].scale, 6)) == SPACING
    assert _reader.image_info(viewer.layers[1])["channel_index"] == 1
    assert _reader.image_info(viewer.add_points()) is None


def test_refuses_a_timelapse_file_from_its_header(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    p = write_ome(tmp_path / "series.ome.tif", rng.integers(0, 1000, (2, 3, 4, 8, 8), dtype=np.uint16),
                  ["405", "477", "545"], axes="TCZYX")
    assert api.read_image_metadata(p)["sizes"]["T"] == 2
    with pytest.raises(api.UnsupportedImageError, match="time-lapse") as ei:
        open_image(p)
    assert "2 time points" in str(ei.value) and "DESIGN_timelapse.md" in str(ei.value)
    # the header of the lab's live series as nd2 reports it: refused with the count, no pixel read
    live = {"sizes": {"T": 25, "Z": 76, "C": 2, "Y": 2720, "X": 2720}, "spacing": (0.2, 0.108, 0.108),
            "spacing_ok": True, "channel_names": ["Brightfield", "EGFP"], "dtype": "uint16", "is_2d": False}
    with pytest.raises(api.UnsupportedImageError, match="25 time points"):
        check_single_timepoint("20261005_DLW93_nohs_gfpsyp2_5min_2hr_herm.nd2", live)
    monkeypatch.setattr(_reader.api, "read_image_metadata", lambda path: live)
    monkeypatch.setattr(_reader.api, "read_image", lambda *a, **k: pytest.fail("pixels were read"))
    with pytest.raises(api.UnsupportedImageError, match="time-lapse"):
        read_layers("live.nd2")


def test_refuses_positions_and_scenes():
    base = {"spacing": (1, 1, 1), "spacing_ok": True, "channel_names": [], "dtype": "uint16", "is_2d": False}
    with pytest.raises(api.UnsupportedImageError, match="3 positions or scenes"):
        check_single_timepoint("multi.nd2", {**base, "sizes": {"P": 3, "C": 1, "Z": 2, "Y": 4, "X": 4}})
    with pytest.raises(api.UnsupportedImageError, match="2 positions or scenes"):
        check_single_timepoint("multi.czi", {**base, "sizes": {"C": 1, "Z": 2, "Y": 4, "X": 4}, "n_scenes": 2})
    check_single_timepoint("ok.nd2", {**base, "sizes": {"T": 1, "P": 1, "C": 1, "Z": 2, "Y": 4, "X": 4}})


def test_viewer_reports_the_refusal(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    p = write_ome(tmp_path / "series.ome.tif", np.zeros((2, 1, 2, 4, 4), np.uint16), ["405"], axes="TCZYX")
    with pytest.raises(Exception) as ei:
        viewer.open(str(p), plugin="napari-germquant")
    chain = str(ei.value) + str(getattr(ei.value, "__cause__", ""))
    assert "time-lapse" in chain and len(viewer.layers) == 0


@pytest.mark.realdata
def test_local_nd2_header_is_a_single_fixed_stack():
    if not LOCAL_ND2.is_file():
        pytest.skip(f"{LOCAL_ND2} not present")
    meta = api.read_image_metadata(LOCAL_ND2)            # the header only: the file is 2.9 GB
    check_single_timepoint(LOCAL_ND2, meta)
    assert meta["spacing_ok"] and len(meta["channel_names"]) == meta["sizes"]["C"] == 3
    assert callable(napari_get_reader(str(LOCAL_ND2)))


@pytest.mark.nas
def test_nas_timelapse_header_is_refused():
    if os.environ.get("NAPARI_GERMQUANT_NAS_TEST") != "1":
        pytest.skip("set NAPARI_GERMQUANT_NAS_TEST=1 to read the live-series header from the NAS")
    if not NAS_ND2.is_file():
        pytest.skip(f"{NAS_ND2} not reachable")
    meta = api.read_image_metadata(NAS_ND2)              # the header only, never the 56 GB of pixels
    assert meta["sizes"]["T"] == 25 and meta["sizes"]["C"] == 2
    with pytest.raises(api.UnsupportedImageError, match="25 time points"):
        check_single_timepoint(NAS_ND2, meta)
