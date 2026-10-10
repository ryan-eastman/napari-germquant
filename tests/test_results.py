"""The results loader and the Results dock on a hand-made results folder shaped like the engine's output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_germquant._results import ResultsSet, find_results, labels_features, load_results_layers, summary_text
from napari_germquant._results_widget import ResultsWidget

ID = "20250101_n2_hs_MALE_7"
SP = (0.4, 0.2, 0.2)
META = {"image_id": ID, "file_path": "C:/nowhere/x.nd2", "voxel_dz_um": SP[0], "voxel_dy_um": SP[1], "voxel_dx_um": SP[2],
        "config_hash": "abc123def456"}


def _write_results(folder: Path, image_id: str = ID) -> ResultsSet:
    import tifffile

    folder.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((4, 16, 16), np.int32)
    labels[1:3, 2:6, 2:6] = 1
    labels[1:3, 8:12, 2:6] = 2
    labels[0:2, 8:12, 9:13] = 4                                # label 3 does not exist
    tifffile.imwrite(str(folder / f"{image_id}__nuclei_labels.tif"), labels, photometric="minisblack")
    nuclei = pd.DataFrame({"nucleus_id": [1, 2, 4], "volume_um3": [10.0, 20.0, 40.0], "n_spots": [1, 0, 2], **META})
    nuclei.to_csv(folder / f"{image_id}__nuclei.csv", index=False)
    spots = pd.DataFrame({"spot_id": [1, 2, 3], "nucleus_id": [1, 4, 4], "z_um": [0.6, 0.2, 0.4],
                          "y_um": [0.8, 2.0, 2.2], "x_um": [0.8, 2.2, 2.4], "effect_size": [5.0, 3.5, 4.1], **META})
    spots.to_csv(folder / f"{image_id}__spots.csv", index=False)
    pd.DataFrame([{"n_nuclei": 3, "n_germline_nuclei": 3, "mean_spots": 1.0, "qc_pass": True, "qc_flags": np.nan,
                   **META}]).to_csv(folder / f"{image_id}__image_summary.csv", index=False)
    (folder / f"{image_id}__stages.json").write_text(json.dumps({
        "read": {"status": "ran"}, "segment": {"status": "ran"}, "spots": {"status": "ran"},
        "coloc": {"status": "skipped", "reason": "disabled"}}), encoding="utf-8")
    (folder / f"{image_id}__done.json").write_text(json.dumps({"image_id": image_id, "config_hash": "abc123def456",
                                                              "xy_stride": 2}), encoding="utf-8")
    return ResultsSet(folder, image_id)


@pytest.fixture
def results(tmp_path) -> ResultsSet:
    return _write_results(tmp_path / "res")


def test_find_results_in_a_folder_and_one_level_below(tmp_path, results):
    assert find_results(results.folder) == [results]
    batch = tmp_path / "batch"
    a = _write_results(batch / "day1" / "img_a", "20250101_n2_hs_MALE_1")
    b = _write_results(batch / "day1" / "img_b", "20250101_n2_hs_MALE_2")
    assert find_results(batch) == []                          # two levels down
    assert find_results(batch / "day1") == [a, b]
    with pytest.raises(FileNotFoundError):
        find_results(tmp_path / "missing")
    assert results.spacing() == SP and results.source_path() is None
    assert results.done_marker()["xy_stride"] == 2 and results.stages()["coloc"]["status"] == "skipped"


def test_labels_features_put_row_i_on_label_i():
    table = pd.DataFrame({"nucleus_id": [4, 1, 2], "volume_um3": [40.0, 10.0, 20.0]})
    feats = labels_features(table, "nucleus_id", 4)
    assert feats.shape[0] == 5 and list(feats.index) == [0, 1, 2, 3, 4]
    assert feats["volume_um3"].tolist()[1:3] == [10.0, 20.0] and feats["volume_um3"].iloc[4] == 40.0
    assert np.isnan(feats["volume_um3"].iloc[0]) and np.isnan(feats["volume_um3"].iloc[3])
    assert labels_features(table, "nucleus_id", 9).shape[0] == 10         # the image has more labels than rows
    assert labels_features(pd.DataFrame({"x": [1]}), "nucleus_id", 2).shape == (3, 0)


def test_load_results_layers(results):
    layers = load_results_layers(results)
    kinds = [kind for _, _, kind in layers]
    assert kinds == ["labels", "points"]
    labels, kw, _ = layers[0]
    assert labels.shape == (4, 16, 16) and labels.max() == 4 and tuple(kw["scale"]) == SP
    assert kw["name"] == f"{ID} nuclei" and kw["features"].shape[0] == 5
    assert kw["features"]["volume_um3"].iloc[4] == 40.0 and kw["metadata"]["napari_germquant_results"]["table"] == "nuclei"
    pts, pkw, _ = layers[1]
    spots = results.table("spots")
    assert np.array_equal(pts, spots[["z_um", "y_um", "x_um"]].to_numpy(float)) and "scale" not in pkw
    assert pkw["features"].shape[0] == 3 and list(pkw["features"]["effect_size"]) == [5.0, 3.5, 4.1]
    assert pkw["size"] == pytest.approx(0.6)
    text = summary_text(results)
    assert "nuclei: 3" in text and "flags: none" in text and "stages ran: read, segment, spots" in text
    assert "stages skipped: coloc" in text and "completion marker: present" in text and "xy_stride: 2" in text
    # include_source with a file_path that does not exist: no image layer, no error
    assert len(load_results_layers(results, include_source=True)) == 2


def test_results_widget_opens_a_folder_as_layers(make_napari_viewer, results):
    viewer = make_napari_viewer()
    w = ResultsWidget(viewer)
    sets = w.set_folder(str(results.folder))
    assert sets == [results] and w.image_combo.count() == 1 and w.selected() == results
    assert "nuclei: 3" in w.summary.text()
    layers = w.load_selected()
    assert [layer.__class__.__name__ for layer in layers] == ["Labels", "Points"]
    assert [layer.name for layer in viewer.layers] == [f"{ID} nuclei", f"{ID} spots"]
    lab, pts = viewer.layers
    assert lab.features.shape[0] == 5 and lab.features["volume_um3"].iloc[4] == 40.0
    assert len(pts.data) == 3 and list(pts.features["nucleus_id"]) == [1, 4, 4]
    assert "loaded 2 layer(s)" in w.summary.text()
    assert w.set_folder(str(results.folder.parent)) == [results]       # the parent sees it one level down
    assert w.set_folder(str(results.folder / "nothing_here_yet")) == []
