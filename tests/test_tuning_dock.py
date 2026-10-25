"""The Spot tuning dock with a fake engine (tranche 5 Train B, unit P6-TUNE; plugin_specs.md 4.5; plan 5.1, D24,
D38, D39 and D42): the project's images with their runs and truth points, an image opened on its run's grid with
an editable truth-point layer, tune-spots and its Apply through the engine runner (in a thread worker, with the
plugin version), the top five rows of ranked.csv, Cancel, refusals, the one message of an engine without spot
tuning, the annotator, the wait while a run of the project goes, and the Project dock's hub.

The fake engine (tests/_p6_tune.py) answers in the dock's worker as the engine's command does, so these tests start
no process and need no Train B feature: they run against every engine the suite runs on. The real engine (a real
LoG tune of the test pack's example, the engine's own points writer) is tests/test_tuning.py."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from _p6_tune import (
    BASE_HASH,
    GRID_KEY,
    OBJECTIVE,
    TUNED_HASH,
    FakeHub,
    FakeTuneRunner,
    StaticSession,
    ranked_rows,
    read_manifest,
    standard_project,
    write_finished_run,
    write_tuning_run,
)

from napari_germquant import _settings, _tuning
from napari_germquant._runner import plugin_version
from napari_germquant._tuning_widget import SpotTuningWidget

WAIT = 60_000


@pytest.fixture
def tproj():
    """A short project folder (the system temporary folder, or $RUNNER_TEMP on a CI runner): a project nests
    results/<image id>/ and tuning/<run id>/ below its root."""
    root = Path(tempfile.mkdtemp(prefix="gqt", dir=os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def fake():
    runner = FakeTuneRunner()
    try:
        yield runner
    finally:
        runner.release()


def _dock(qtbot, viewer=None, *, runner=None, hub=None, missing=None) -> SpotTuningWidget:
    w = SpotTuningWidget(viewer, session=StaticSession(runner or FakeTuneRunner(), missing_feature=missing), hub=hub)
    qtbot.addWidget(w)
    return w


def _settle(qtbot, w) -> None:
    qtbot.waitUntil(lambda: w.idle, timeout=WAIT)


def _opened(qtbot, w, root) -> None:
    w.set_project(root)
    _settle(qtbot, w)
    assert w.root == Path(root), w.last_message


def _table(w) -> tuple[list[str], list[list[str]]]:
    t = w.ranked_table
    head = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    return head, [[t.item(r, c).text() for c in range(t.columnCount())] for r in range(t.rowCount())]


def test_the_dock_lists_the_projects_images_with_their_runs_and_points(qtbot, tproj, fake):
    standard_project(tproj)
    w = _dock(qtbot, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    assert [r.image_id for r in w.rows] == ["img_a", "img_b", "img_c", "img_x", "img_t"]
    by = {r.image_id: r for r in w.rows}
    assert by["img_a"].finished and by["img_a"].points == 2 and by["img_a"].annotators == ["tester"]
    assert "finished" in by["img_a"].text and "2 truth points (tester)" in by["img_a"].text
    assert by["img_b"].finished and by["img_b"].points == 0 and "no truth points yet" in by["img_b"].text
    assert not by["img_c"].finished and "no finished run" in by["img_c"].text
    assert not by["img_x"].usable and "excluded (out of focus)" in by["img_x"].text
    assert not by["img_t"].usable and "time series" in by["img_t"].text
    assert w.image_list.count() == 5 and w.annotated_images() == ["img_a"]
    # the header names the project and the pack, and the engine line is the session's
    assert str(tproj) in w.project_label.text() and "test_pack@0.1.0 basic" in w.project_label.text()
    assert "a fake engine" in w.engine_label.text()
    # a folder that is no project is refused, and the project that was open stays open
    other = tproj / "not_a_project"
    other.mkdir()
    w.set_project(other)
    _settle(qtbot, w)
    assert w.root == tproj and "not a germquant project" in w.last_message
    # nothing ran in the engine to list a project
    assert fake.calls == []


def test_opening_an_image_puts_the_source_results_and_a_truth_layer_on_the_runs_grid(make_napari_viewer, qtbot,
                                                                                     tproj, fake):
    """Through napari's public proxy of the viewer, what a dock opened from the Plugins menu gets."""
    from napari.utils._proxies import PublicOnlyProxy

    standard_project(tproj, stride=2, z_range=(1, 5))
    viewer = make_napari_viewer()
    w = _dock(qtbot, PublicOnlyProxy(viewer), runner=fake)
    w.set_annotator("other")                         # no truth points of "other" yet
    _opened(qtbot, w, tproj)
    assert w.select_image("img_a") and w.open_image()
    _settle(qtbot, w)
    names = [layer.name for layer in viewer.layers]
    assert "img_a nuclei" in names and "img_a spots" in names
    assert "img_a [dna]" in names and "img_a [foci]" in names
    grid = (0.5, 0.4, 0.4)                           # the run's own voxel: dy and dx times the stride of 2
    labels = viewer.layers["img_a nuclei"]
    assert labels.data.shape == (4, 16, 24) and tuple(np.round(labels.scale, 6)) == grid
    for ch in ("dna", "foci"):                       # the source read as the run read it: planes 1 to 4, stride 2
        layer = viewer.layers[f"img_a [{ch}]"]
        assert layer.data.shape == (4, 16, 24) and tuple(np.round(layer.scale, 6)) == grid
    truth = w.truth_layer("img_a")
    assert truth is not None and truth.name == "img_a truth points (other)"
    assert truth.ndim == 3 and len(truth.data) == 0 and truth.mode == "add" and truth.editable
    info = truth.metadata[_tuning.LAYER_KEY]
    assert info["image_id"] == "img_a" and info["annotator"] == "other" and Path(info["root"]) == tproj
    assert Path(info["file"]) == tproj / "annotations" / "img_a__points_other.csv"
    assert viewer.layers.selection.active is truth
    # the dock says what to do, and opening the same image again adds nothing
    assert "click" in w.points_label.text().lower()
    n = len(viewer.layers)
    assert w.open_image("img_a")
    _settle(qtbot, w)
    assert len(viewer.layers) == n and "already open" in w.last_message
    # an image without a finished run cannot be opened
    assert w.select_image("img_c") and not w.open_image()
    assert "no finished run" in w.last_message and not w.open_btn.isEnabled()
    assert fake.calls == []


def test_tune_runs_through_the_engine_and_shows_the_top_five_of_ranked_csv(qtbot, tproj, fake):
    standard_project(tproj)
    w = _dock(qtbot, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    assert w.tune_btn.isEnabled() and not w.apply_btn.isEnabled()
    assert w.tune()
    _settle(qtbot, w)
    name, kw = fake.calls[-1]
    assert name == "tune_spots" and kw == {"project": str(tproj), "points": "annotations", "grid": None,
                                           "max_runs": 50, "out": None}
    assert fake.threads[-1] != threading.current_thread().name          # in the dock's worker, not the GUI thread
    run = Path(w.current_run)
    assert run.parent == tproj / "tuning" and (run / "ranked.csv").is_file()
    head, cells = _table(w)
    ranked = pd.read_csv(run / "ranked.csv", dtype=str, keep_default_na=False)
    assert head == list(ranked.columns) == ["rank", "objective", "objective_value", "tie_break", "tie_break_value",
                                            GRID_KEY]
    assert len(ranked) == 6 and cells == ranked.head(5).values.tolist()  # the top five rows, as the file holds them
    assert OBJECTIVE in w.objective_label.text()
    assert f"{GRID_KEY} = 1.5" in w.best_label.text() and "log" in w.best_label.text()
    assert "a warning of the search" in w.warnings_label.text()
    assert "tune: img_a (1/1)" in w.progress_text                      # the engine's lines, as they came
    assert w.runs_combo.currentData() == str(run) and w.apply_btn.isEnabled()
    # a grid file and a limit go to the command as they are
    grid = tproj / "my_grid.yaml"
    grid.write_text(f"{GRID_KEY}: [2.0, 3.0]\n", encoding="ascii")
    w.set_grid_file(grid)
    w.set_max_runs(12)
    assert str(grid) in w.grid_label.text()
    assert w.tune()
    _settle(qtbot, w)
    assert fake.calls[-1][1]["grid"] == str(grid) and fake.calls[-1][1]["max_runs"] == 12
    w.set_grid_file(None)
    assert "default" in w.grid_label.text()


def test_apply_goes_through_the_engine_with_the_plugin_version_and_the_project_reads_again(qtbot, tproj, fake):
    standard_project(tproj)
    hub = FakeHub()
    w = _dock(qtbot, runner=fake, hub=hub)
    w.set_annotator("tester")
    hub.open(tproj)                                  # the Project dock opens the project; the tuning dock follows
    _settle(qtbot, w)
    assert w.root == tproj
    assert w.tune()
    _settle(qtbot, w)
    run = w.current_run
    assert w.apply()
    _settle(qtbot, w)
    name, kw = fake.calls[-1]
    assert name == "tune_apply"
    assert kw == {"project": str(tproj), "run_dir": str(run), "plugin_version": plugin_version()}
    assert TUNED_HASH in w.last_message and "validated: false" in w.last_message
    assert BASE_HASH in w.last_message                       # the images run with the earlier config are named stale
    assert hub.reloads == 1                                   # the Project dock reads the project again
    assert read_manifest(tproj)["recipe"]["expected_config_hash"] == TUNED_HASH
    by = {r.image_id: r for r in w.rows}                     # the dock read it again too
    assert by["img_a"].stale and "stale" in by["img_a"].text and TUNED_HASH in by["img_a"].text
    # a refused apply writes nothing and shows the engine's reason
    fake.apply_refusal = "best.yaml sets ['spots.log.diameter_um'], which the run's grid does not search"
    before = (tproj / "recipe.yaml").read_bytes()
    assert w.apply()
    _settle(qtbot, w)
    assert "which the run's grid does not search" in w.last_message and hub.reloads == 1
    assert (tproj / "recipe.yaml").read_bytes() == before


def test_tune_and_apply_wait_while_a_run_of_the_project_goes(qtbot, tproj, fake):
    standard_project(tproj)
    write_tuning_run(tproj, "20261024T100000Z_0000aaaa", best={GRID_KEY: 2.5})
    hub = FakeHub()
    w = _dock(qtbot, runner=fake, hub=hub)
    w.set_annotator("tester")
    hub.open(tproj)
    _settle(qtbot, w)
    assert w.tune_btn.isEnabled() and w.apply_btn.isEnabled()
    hub.set_running(True)
    assert not w.tune_btn.isEnabled() and not w.apply_btn.isEnabled()
    assert not w.tune() and "a run of this project is in progress" in w.last_message
    assert not w.apply() and "a run of this project is in progress" in w.last_message
    assert fake.calls == []
    hub.set_running(False)
    assert w.tune_btn.isEnabled() and w.apply_btn.isEnabled()


def test_cancel_ends_a_tuning_search(qtbot, tproj, fake):
    standard_project(tproj)
    fake.gate = threading.Event()
    w = _dock(qtbot, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    assert not w.cancel_btn.isEnabled()
    assert w.tune()
    qtbot.waitUntil(fake.started.is_set, timeout=WAIT)
    assert w.cancel_btn.isEnabled() and not w.tune_btn.isEnabled() and not w.apply_btn.isEnabled()
    assert not w.tune()                                       # one search at a time
    assert w.cancel()
    _settle(qtbot, w)
    assert "cancelled" in w.last_message and w.current_run is None
    assert not (tproj / "tuning").exists() and w.tune_btn.isEnabled() and not w.cancel_btn.isEnabled()


def test_refusals_failures_and_a_search_that_scored_nothing_are_shown(qtbot, tproj, fake):
    standard_project(tproj)
    w = _dock(qtbot, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    fake.tune_refusal = "no truth point lies in an object of its run: nothing can be scored"
    assert w.tune()
    _settle(qtbot, w)
    assert "nothing can be scored" in w.last_message and w.current_run is None
    fake.tune_refusal, fake.tune_error = None, "ERROR: RuntimeError: the backend crashed"
    assert w.tune()
    _settle(qtbot, w)
    assert "the backend crashed" in w.last_message and w.current_run is None
    fake.tune_error, fake.scored_nothing = None, True
    assert w.tune()
    _settle(qtbot, w)
    assert w.current_run is not None and w.ranked_table.rowCount() == 5
    assert "no setting could be scored" in w.best_label.text() and not w.apply_btn.isEnabled()
    assert "no parameter set could be scored" in w.warnings_label.text()
    assert not w.apply() and "no best setting" in w.last_message


def test_an_engine_without_spot_tuning_gets_one_message(qtbot, tproj):
    standard_project(tproj)
    runner = FakeTuneRunner(features=("packs",))
    w = _dock(qtbot, runner=runner, missing="tune_spots")
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    message = runner.feature_message("tune_spots")
    assert message.startswith("Spot tuning needs the tranche-5 Train B engine")
    assert message in w.notice_label.text()
    assert w.tune()
    _settle(qtbot, w)
    assert w.last_message == message and runner.calls == []


def test_the_annotator_is_checked_and_kept_per_user(qtbot, tproj, fake):
    standard_project(tproj)
    w = _dock(qtbot, runner=fake)
    assert w.annotator_edit.text() == ""
    with pytest.raises(ValueError, match="letters, digits"):
        w.set_annotator("Ryan E.")
    w.set_annotator("rje")
    assert _settings.annotator() == "rje"
    w2 = _dock(qtbot, runner=fake)
    assert w2.annotator_edit.text() == "rje"                  # per user, read back by the next dock
    w2.annotator_edit.setText("no spaces please")
    w2.annotator_edit.editingFinished.emit()
    assert _settings.annotator() == "rje" and "letters, digits" in w2.last_message
    _opened(qtbot, w2, tproj)
    assert w2.select_image("img_b") and not w2.save_btn.isEnabled()


def test_unsaved_points_must_be_saved_before_a_search(make_napari_viewer, qtbot, tproj, fake):
    standard_project(tproj)
    viewer = make_napari_viewer()
    w = _dock(qtbot, viewer, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    assert w.select_image("img_b") and w.open_image()
    _settle(qtbot, w)
    truth = w.truth_layer("img_b")
    truth.add(np.array([[1.0, 2.0, 2.4]]))
    assert w.unsaved_images() == ["img_b"]
    assert "1 point, not saved" in w.points_label.text()
    assert not w.tune() and "save the truth points of img_b first" in w.last_message.lower()
    assert fake.calls == []
    # a point removed again leaves nothing unsaved
    truth.data = np.zeros((0, 3))
    assert w.unsaved_images() == [] and w.tune()
    _settle(qtbot, w)


def test_the_dock_follows_the_project_dock(qtbot, tproj, fake):
    standard_project(tproj)
    hub = FakeHub()
    w = _dock(qtbot, runner=fake, hub=hub)
    w.set_annotator("tester")
    assert w.root is None and "Project dock" in w.project_label.text()
    hub.open(tproj)
    _settle(qtbot, w)
    assert w.root == tproj
    # a reading of the Project dock (after a run, say) reads the image list again
    write_finished_run(tproj, "img_c", tproj / "images" / "img_c.ome.tif")
    hub.changed.emit()
    _settle(qtbot, w)
    assert {r.image_id: r for r in w.rows}["img_c"].finished
    hub.close()
    _settle(qtbot, w)
    assert w.root is None and w.image_list.count() == 0
    # a project opened in this dock itself stays open when the Project dock closes another one
    other = tproj / "other"
    standard_project(other)
    w.set_project(other)
    _settle(qtbot, w)
    hub.open(tproj)
    _settle(qtbot, w)
    assert w.root == tproj                                    # the Project dock's project comes first
    w.set_project(other)
    _settle(qtbot, w)
    hub.close()
    _settle(qtbot, w)
    assert w.root == other


def test_a_reading_that_returns_after_close_leaves_the_project_closed(make_napari_viewer, qtbot, tproj, fake,
                                                                      monkeypatch):
    """A reading of the project, or an image being opened, that is still in the file worker when the project is
    closed (here, or by the Project dock) changes nothing when it returns: the project stays closed and no layer
    is added."""
    standard_project(tproj)
    gate = threading.Event()
    real_read, real_layers = _tuning.read_project, _tuning.image_layers

    def held(fn):
        def run(*a, **k):
            assert gate.wait(30)
            return fn(*a, **k)
        return run

    viewer = make_napari_viewer()
    w = _dock(qtbot, viewer, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    monkeypatch.setattr(_tuning, "read_project", held(real_read))
    w.refresh()
    qtbot.waitUntil(lambda: w.io_worker.busy, timeout=WAIT)
    w.close_project()
    gate.set()
    _settle(qtbot, w)
    assert w.root is None and w.image_list.count() == 0
    # an image that opens while its project is closed adds nothing
    monkeypatch.setattr(_tuning, "read_project", real_read)
    gate.clear()
    _opened(qtbot, w, tproj)
    monkeypatch.setattr(_tuning, "image_layers", held(real_layers))
    n = len(viewer.layers)
    assert w.select_image("img_b") and w.open_image()
    qtbot.waitUntil(lambda: w.io_worker.busy, timeout=WAIT)
    w.close_project()
    gate.set()
    _settle(qtbot, w)
    assert len(viewer.layers) == n and "closed meanwhile" in w.last_message and w.root is None


def test_earlier_tuning_runs_are_listed_newest_first_and_can_be_applied(qtbot, tproj, fake):
    standard_project(tproj)
    old = write_tuning_run(tproj, "20261020T090000Z_0000aaaa", rows=ranked_rows((2.0, 3.0, 1.5)),
                           best={GRID_KEY: 2.0})
    new = write_tuning_run(tproj, "20261023T090000Z_0000bbbb", best={GRID_KEY: 1.5})
    (tproj / "tuning" / "not_a_run").mkdir()
    w = _dock(qtbot, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    assert [w.runs_combo.itemData(i) for i in range(w.runs_combo.count())] == [str(new), str(old)]
    assert w.current_run == str(new) and w.ranked_table.rowCount() == 5
    assert w.select_run(old)
    head, cells = _table(w)
    assert len(cells) == 3 and [c[-1] for c in cells] == ["2.0", "3.0", "1.5"]
    assert f"{GRID_KEY} = 2.0" in w.best_label.text()
    assert w.apply()
    _settle(qtbot, w)
    assert fake.calls[-1][1]["run_dir"] == str(old)
    assert yaml.safe_load((tproj / "recipe.yaml").read_text(encoding="ascii"))["spots"]["log"]["min_snr"] == 2.0


class _StubViewer:
    """Stands for a viewer where no layer is added: the hub and the status line are all the dock asks of it."""

    status = ""


def test_the_project_docks_own_hub_is_followed(qtbot, tproj, fake):
    """With the Project dock in the same plugin (unit P6-PROJECT, merged into feat/t5-p6 before this unit, plan
    5.1), the tuning dock takes the viewer's own hub, napari_germquant._project_widget.project_hub."""
    widget_module = pytest.importorskip("napari_germquant._project_widget",
                                        reason="the Project dock (unit P6-PROJECT) is not in this branch yet")
    standard_project(tproj)
    w0 = _dock(qtbot, runner=fake)
    assert w0.hub is None                                     # no viewer, no hub
    viewer = _StubViewer()
    hub = widget_module.project_hub(viewer)
    w = SpotTuningWidget(viewer, session=StaticSession(fake))
    qtbot.addWidget(w)
    assert w.hub is hub
    reloads = []
    hub.reload_requested.connect(lambda: reloads.append(1))
    w.set_annotator("tester")
    hub.root = tproj
    hub._emit(hub.opened, str(tproj))
    _settle(qtbot, w)
    assert w.root == tproj
    assert w.tune()
    _settle(qtbot, w)
    assert w.apply()
    _settle(qtbot, w)
    assert reloads == [1]


def test_a_source_that_is_not_there_still_opens_the_results(make_napari_viewer, qtbot, tproj, fake):
    standard_project(tproj)
    (tproj / "images" / "img_b.ome.tif").unlink()
    viewer = make_napari_viewer()
    w = _dock(qtbot, viewer, runner=fake)
    w.set_annotator("tester")
    _opened(qtbot, w, tproj)
    assert w.select_image("img_b") and w.open_image()
    _settle(qtbot, w)
    names = [layer.name for layer in viewer.layers]
    assert "img_b nuclei" in names and "img_b truth points (tester)" in names
    assert not any(n.startswith("img_b [") for n in names)
    assert "source image" in w.last_message and "not found" in w.last_message
