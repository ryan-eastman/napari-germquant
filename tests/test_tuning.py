"""Spot tuning against the real engine (tranche 5 Train B, unit P6-TUNE; plan 5.3 acceptance 7, D24, D39 and D42;
plugin_specs.md 4.5 and 6.2): truth points saved in napari's process through the engine's own writer
(germquant.tune.points), a real LoG tune of the test pack's example run in the engine's process
(``germquant tune-spots``, no SpotMAX needed), the top five rows of its ranked.csv, and Apply through the engine,
after which the project's images read stale.

Each test needs the Train B features it names (conftest.need_features): it skips, naming them, on an engine
without them, and fails instead when $NAPARI_GERMQUANT_REQUIRE_FEATURES requires them (the final CI runs). The
dock's flow on a fake engine is tests/test_tuning_dock.py."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from _p6_tune import (
    BASE_HASH,
    FakeTuneRunner,
    RecordingRunner,
    StaticSession,
    FakeHub,
    image_entry,
    project_module_stand_in,
    read_manifest,
    run_labels,
    standard_project,
    write_finished_run,
    write_project,
    write_source,
)
from conftest import need_features

from napari_germquant import _tuning
from napari_germquant._runner import InProcessRunner, plugin_version
from napari_germquant._tuning_widget import SpotTuningWidget

WAIT = 60_000
ENGINE_WAIT = 600_000                 # a command of the engine's process imports torch first


@pytest.fixture
def tproj():
    root = Path(tempfile.mkdtemp(prefix="gqt", dir=os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _settle(qtbot, w, timeout: int = WAIT) -> None:
    qtbot.waitUntil(lambda: w.idle, timeout=timeout)


def test_points_saved_tuned_top_five_and_applied(make_napari_viewer, qtbot, project_env, tmp_path):
    """Acceptance 7: points clicked on a finished run of the test pack's basic recipe are saved in the truth-point
    format, tune-spots runs in the engine's process (its command line asserted), the dock shows the top five rows
    of ranked.csv, and Apply writes recipe.yaml through the engine, after which the done image reads stale."""
    need_features("projects", "tune_spots")
    from germquant.tune.points import POINT_COLUMNS, write_points

    runner = RecordingRunner(sys.executable)         # the engine under test, in a process of its own
    P = project_env.new_dir()
    made = runner.project_new(P, "test_pack@0.1.0", recipe="basic", images=[project_env.image])
    (image_id,) = made["added"]
    assert made["config_hash"] == BASE_HASH
    ran = InProcessRunner().project_run(P)            # the finished run that tuning reads
    assert ran["state"] == "done", ran
    viewer = make_napari_viewer()
    hub = FakeHub()
    hub.root = Path(P)
    w = SpotTuningWidget(viewer, session=StaticSession(runner), hub=hub)
    qtbot.addWidget(w)
    w.set_annotator("tester")
    w.set_project(P)
    _settle(qtbot, w)
    assert w.select_image(image_id) and w.open_image()
    _settle(qtbot, w)
    truth = w.truth_layer(image_id)
    spots = viewer.layers[f"{image_id} spots"]
    assert len(spots.data) == 10
    truth.data = np.array(spots.data, float)          # a click on every spot the run found (two per nucleus)
    assert w.unsaved_images() == [image_id]
    assert w.save_points()
    _settle(qtbot, w)
    saved = Path(P) / "annotations" / f"{image_id}__points_tester.csv"
    table = pd.read_csv(saved)
    assert list(table.columns) == POINT_COLUMNS and len(table) == 10
    assert sorted(set(table["object_id"])) == [1, 2, 3, 4, 5] and not table["projected"].any()
    labels = tifffile.imread(str(Path(P) / "results" / image_id / f"{image_id}__nuclei_labels.tif"))
    expect = write_points(tmp_path / "expect.csv", np.array(spots.data, float), labels=labels,
                          spacing=(0.5, 0.2, 0.2))
    assert saved.read_bytes() == expect.read_bytes()  # the engine's writer made it, from the dock's points
    assert w.unsaved_images() == [] and "10 points saved" in w.points_label.text()
    # before the search: what the default LoG grid means
    assert "6 LoG detections on 1 annotated image" in w.count_label.text()
    assert w.tune()
    _settle(qtbot, w, ENGINE_WAIT)
    tune_argv = next(a for a in reversed(runner.history) if "tune-spots" in a)
    assert tune_argv[:3] == [sys.executable, "-m", "germquant"]
    assert tune_argv[3:6] == ["tune-spots", "--project", str(P)] and tune_argv[-1] == "--json"
    run = Path(w.current_run)
    assert run.parent == Path(P) / "tuning", w.last_message
    ranked = pd.read_csv(run / "ranked.csv", dtype=str, keep_default_na=False)
    assert len(ranked) == 6 and list(ranked.columns)[-1] == "spots.log.min_snr"
    t = w.ranked_table
    head = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    cells = [[t.item(r, c).text() for c in range(t.columnCount())] for r in range(t.rowCount())]
    assert head == list(ranked.columns) and cells == ranked.head(5).values.tolist()
    best = yaml.safe_load((run / "best.yaml").read_text(encoding="ascii"))
    assert best["backend"] == "log" and list(best["keys"]) == ["spots.log.min_snr"]
    value = best["keys"]["spots.log.min_snr"]
    # every default min_snr finds the same ten spots on this example, so the grid's first value ranks first:
    # not the recipe's own 2.0, so Apply is an edit (a value equal to the recipe's is none: docs/TUNING.md)
    assert value != 2.0
    assert f"spots.log.min_snr = {value}" in w.best_label.text()
    # Apply, through the engine, with the plugin version
    assert w.apply()
    _settle(qtbot, w, ENGINE_WAIT)
    assert runner.calls[-1] == ("tune_apply", {"project": str(P), "run_dir": str(run),
                                               "plugin_version": plugin_version()})
    apply_argv = next(a for a in reversed(runner.history) if "--apply" in a)
    assert apply_argv[3:8] == ["tune-spots", "--project", str(P), "--apply", str(run)]
    recipe = yaml.safe_load((Path(P) / "recipe.yaml").read_text(encoding="ascii"))
    assert recipe["extends"] == "pack:test_pack@0.1.0/basic" and recipe["spots"]["log"]["min_snr"] == value
    m = read_manifest(P)
    assert m["recipe"]["origins"] == {"spots.log.min_snr": "tuning"}
    assert m["saved"]["plugin"] == plugin_version()   # the write is stamped with the plugin version (D42)
    assert apply_argv[-3:] == ["--plugin-version", plugin_version(), "--json"]
    new_hash = m["recipe"]["expected_config_hash"]
    assert new_hash != BASE_HASH
    assert new_hash in w.last_message and "validated: false" in w.last_message
    assert hub.reloads == 1
    status = InProcessRunner().project_status(P)      # what the Project dock's board shows
    (row,) = status["images"]
    assert row["status"] == "stale" and row["reason_code"] == "config_hash"
    assert {r.image_id: r for r in w.rows}[image_id].stale


def test_truth_points_round_trip_on_the_runs_grid(make_napari_viewer, qtbot, tproj, tmp_path):
    """The truth layer of a run made at stride 2 on planes 1 to 4 shows a points file of the image's own grid on
    the run's layers, and Save writes it back through the engine's writer: the file's grid and frame (zi, yi, xi
    at stride 1, z from the file's first plane), the run's object under each point, projected as the box says. A
    single-plane run puts every point on its plane, z_um 0.0."""
    need_features("tune_spots")
    from germquant.tune.points import read_points, write_points

    standard_project(tproj, stride=2, z_range=(1, 5))
    viewer = make_napari_viewer()
    w = SpotTuningWidget(viewer, session=StaticSession(FakeTuneRunner()))
    qtbot.addWidget(w)
    w.set_annotator("tester")
    w.set_project(tproj)
    _settle(qtbot, w)
    assert w.select_image("img_a") and w.open_image()
    _settle(qtbot, w)
    truth = w.truth_layer("img_a")
    # the file's points (z 1.0 um of the file is plane 2, plane 1 of the run) on the run's frame
    assert np.allclose(truth.data, [[0.5, 2.0, 2.4], [0.5, 4.0, 6.8]])
    assert w.unsaved_images() == [] and "2 points saved" in w.points_label.text()
    w.set_projected(True)                             # the box alone changes the file: unsaved, and kept per layer
    assert w.unsaved_images() == ["img_a"] and "box changed, not saved" in w.points_label.text()
    assert w.select_image("img_b") and not w.projected_check.isChecked()
    assert w.select_image("img_a") and w.projected_check.isChecked()
    truth.add(np.array([[1.0, 3.2, 4.0]]))
    assert w.save_points()
    _settle(qtbot, w)
    assert w.unsaved_images() == []
    path = tproj / "annotations" / "img_a__points_tester.csv"
    expect = write_points(tmp_path / "expect.csv", np.array(truth.data, float),
                          labels=run_labels(xy_stride=2, z_range=(1, 5)), spacing=(0.5, 0.4, 0.4), xy_stride=2,
                          z_range=(1, 5), projected=True)
    assert path.read_bytes() == expect.read_bytes()
    pts = read_points(path)
    assert pts["zi"].tolist() == [2, 2, 3] and pts["yi"].tolist() == [10, 20, 16] and pts["xi"].tolist() == [12, 34, 20]
    assert np.allclose(pts["z_um"], [1.0, 1.0, 1.5]) and pts["projected"].all()
    assert pts["object_id"].tolist()[:2] == [1, 2]
    # opened again (a new dock), the three points come back where they were placed
    viewer.layers.remove(truth)
    w2 = SpotTuningWidget(viewer, session=StaticSession(FakeTuneRunner()))
    qtbot.addWidget(w2)
    w2.set_project(tproj)
    _settle(qtbot, w2)
    assert w2.select_image("img_a") and w2.open_image()
    _settle(qtbot, w2)
    again = w2.truth_layer("img_a")
    assert np.allclose(again.data, truth.data) and w2.projected_check.isChecked()
    # a single plane: one plane of labels, voxel_dz_um NaN, the done marker's planar extra
    plane = tproj / "plane"
    flat_nuclei = ((1, 0, 10, 12, 6), (2, 0, 20, 34, 7))
    src = write_source(plane / "images" / "img_p.ome.tif", shape=(1, 32, 48), nuclei=flat_nuclei)
    write_project(plane, [image_entry("img_p", "images/img_p.ome.tif")])
    write_finished_run(plane, "img_p", src, planar=True, labels=run_labels(shape=(1, 32, 48), nuclei=flat_nuclei))
    w3 = SpotTuningWidget(viewer, session=StaticSession(FakeTuneRunner()))
    qtbot.addWidget(w3)
    w3.set_annotator("tester")
    w3.set_project(plane)
    _settle(qtbot, w3)
    assert w3.select_image("img_p") and w3.open_image()
    _settle(qtbot, w3)
    flat = w3.truth_layer("img_p")
    flat.add(np.array([[0.0, 2.0, 2.4]]))
    assert w3.save_points()
    _settle(qtbot, w3)
    one = read_points(plane / "annotations" / "img_p__points_tester.csv")
    assert one["z_um"].tolist() == [0.0] and one["zi"].tolist() == [0] and one["object_id"].tolist() == [1]


def test_a_one_plane_run_leaves_the_points_of_other_planes_alone(make_napari_viewer, qtbot, tproj):
    """A run of one plane (here a z range of one plane of a stack) holds no other plane: the file's points on
    other planes are not shown on its plane, and Save is refused (the engine's writer would put every point on
    the run's plane), so the file keeps them where they are."""
    need_features("tune_spots")
    from germquant.tune.points import read_points

    standard_project(tproj)
    write_finished_run(tproj, "img_b", tproj / "images" / "img_b.ome.tif", z_range=(2, 3),
                       labels=run_labels(z_range=(2, 3)))
    ann = tproj / "annotations" / "img_b__points_tester.csv"
    ann.write_text("point_id,z_um,y_um,x_um,zi,yi,xi,object_id,projected\n"
                   "0,1.0,2.0,2.4,2,10,12,1,false\n1,2.0,4.0,6.8,4,20,34,2,false\n", encoding="ascii")
    before = ann.read_bytes()
    viewer = make_napari_viewer()
    w = SpotTuningWidget(viewer, session=StaticSession(FakeTuneRunner()))
    qtbot.addWidget(w)
    w.set_annotator("tester")
    w.set_project(tproj)
    _settle(qtbot, w)
    assert w.select_image("img_b") and w.open_image()
    _settle(qtbot, w)
    layer = w.truth_layer("img_b")
    assert np.allclose(layer.data, [[0.0, 2.0, 2.4]])                 # the point of plane 2, the run's plane
    assert "1 of its points lies on planes this one-plane run did not read" in w.last_message
    layer.add(np.array([[0.0, 3.0, 3.0]]))
    assert not w.save_points()
    assert "1 truth point of img_b__points_tester.csv lies on a plane this one-plane run did not read" \
        in w.last_message
    assert ann.read_bytes() == before and read_points(ann)["zi"].tolist() == [2, 4]
    with pytest.raises(_tuning.TuningError, match="would remove it from the file"):
        _tuning.save_truth(dict(layer.metadata[_tuning.LAYER_KEY]), layer.data, projected=False)


def test_a_run_made_again_on_another_grid_is_not_saved_over(make_napari_viewer, qtbot, tproj):
    """An image run again with another stride and z range after its truth layer was opened: the layer's points
    sit on the old grid, so Save refuses them and the file stays as it was; the layer is kept, renamed, and Open
    image opens the new run with the points of the file."""
    need_features("tune_spots")
    from germquant.tune.points import read_points

    made = standard_project(tproj, stride=2, z_range=(1, 5))
    viewer = make_napari_viewer()
    w = SpotTuningWidget(viewer, session=StaticSession(FakeTuneRunner()))
    qtbot.addWidget(w)
    w.set_annotator("tester")
    w.set_project(tproj)
    _settle(qtbot, w)
    assert w.select_image("img_a") and w.open_image()
    _settle(qtbot, w)
    old = w.truth_layer("img_a")
    old.add(np.array([[1.0, 3.2, 4.0]]))
    ann = tproj / "annotations" / "img_a__points_tester.csv"
    before = ann.read_bytes()
    write_finished_run(tproj, "img_a", made["sources"]["img_a"])      # run again: stride 1, every plane
    assert w.save_points()
    _settle(qtbot, w)
    assert "changed since its truth layer was opened" in w.last_message and "xy stride 2 to 1" in w.last_message
    assert ann.read_bytes() == before
    assert w.truth_layer("img_a") is None and old.name == "img_a truth points (tester) (old run)"
    assert "img_a nuclei (old run)" in [layer.name for layer in viewer.layers]
    assert w.open_image("img_a")                                       # the new run, the file's two points on it
    _settle(qtbot, w)
    new = w.truth_layer("img_a")
    assert new is not old and new.name == "img_a truth points (tester)"
    assert np.allclose(new.data, [[1.0, 2.0, 2.4], [1.0, 4.0, 6.8]])
    assert viewer.layers["img_a nuclei"].data.shape == (6, 32, 48)
    new.add(np.array([[1.5, 3.2, 4.0]]))
    assert w.save_points()
    _settle(qtbot, w)
    assert read_points(ann)["zi"].tolist() == [2, 2, 3] and w.unsaved_images() == []


def test_the_count_line_names_the_detections_the_grid_means(qtbot, tproj, monkeypatch):
    """What a search means before it starts (germquant.tune.grid.n_runs): the default grid of the runs' backend or
    a grid file, on the images that have truth points and a finished run, against Max runs."""
    need_features("tune_spots")
    standard_project(tproj)
    w = SpotTuningWidget(None, session=StaticSession(FakeTuneRunner()))
    qtbot.addWidget(w)
    w.set_annotator("tester")
    w.set_project(tproj)
    _settle(qtbot, w)
    assert "6 LoG detections on 1 annotated image" in w.count_label.text()
    assert "spots.log.min_snr" in w.grid_label.text() and "default" in w.grid_label.text()
    grid = tproj / "grid.yaml"
    grid.write_text("spots.log.min_snr: [2, 3]\nspots.log.diameter_um: [0.4, 0.5]\n", encoding="ascii")
    w.set_grid_file(grid)                             # the file is read in the dock's file worker
    _settle(qtbot, w)
    assert "4 LoG detections on 1 annotated image" in w.count_label.text()
    w.set_max_runs(3)
    assert "above Max runs 3" in w.count_label.text()
    bad = tproj / "bad.yaml"
    bad.write_text("spots.backend: [log]\n", encoding="ascii")
    w.set_grid_file(bad)
    _settle(qtbot, w)
    assert "never tuned" in w.count_label.text()
    # a SpotMAX recipe: one run per image for every effect-size minimum
    write_finished_run(tproj, "img_b", tproj / "images" / "img_b.ome.tif", backend="spotmax")
    (tproj / "annotations" / "img_b__points_tester.csv").write_text(
        "point_id,z_um,y_um,x_um,zi,yi,xi,object_id,projected\n0,1.0,2.0,2.4,2,10,12,1,false\n", encoding="ascii")
    write_finished_run(tproj, "img_a", tproj / "images" / "img_a.ome.tif", backend="spotmax")
    w.set_grid_file(None)
    _settle(qtbot, w)
    w.set_max_runs(50)
    w.refresh()
    _settle(qtbot, w)
    assert "2 SpotMAX runs on 2 annotated images" in w.count_label.text()
    assert "(the backend from the annotated runs)" in w.count_label.text()   # the pack does not resolve here
    # the project's recipe decides when napari's germquant resolves it (the engine tunes the recipe's backend)
    monkeypatch.setitem(sys.modules, "germquant.project", project_module_stand_in("log"))
    w.refresh()
    _settle(qtbot, w)
    assert "12 LoG detections on 2 annotated images" in w.count_label.text()
    assert "the backend from" not in w.count_label.text()


def test_saving_points_needs_the_engine_writer_in_this_environment(make_napari_viewer, qtbot, tproj, monkeypatch):
    """Points are written in napari's own process through germquant.tune.points (D39): a germquant without it gets
    the one message of a missing feature, and Save stays off."""
    standard_project(tproj)
    monkeypatch.setattr(_tuning, "tune_module", lambda name: None)
    viewer = make_napari_viewer()
    w = SpotTuningWidget(viewer, session=StaticSession(FakeTuneRunner()))
    qtbot.addWidget(w)
    w.set_annotator("tester")
    w.set_project(tproj)
    _settle(qtbot, w)
    assert w.select_image("img_b") and w.open_image()
    _settle(qtbot, w)
    w.truth_layer("img_b").add(np.array([[1.0, 2.0, 2.4]]))
    assert not w.save_btn.isEnabled()
    assert not w.save_points()
    assert w.last_message.startswith("Spot tuning needs the tranche-5 Train B engine")
    assert "this napari environment" in w.last_message
    assert not (tproj / "annotations" / "img_b__points_tester.csv").exists()
