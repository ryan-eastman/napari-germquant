"""The Training dock's project mode and engine route (tranche 5 Train B, unit P6-TRAIN; plugin_specs.md 4.4,
decisions D36 and D39), against a fake engine (tests/_p6_train.py): where the dock computes (the engine chosen
for this computer, else this napari environment, said in amber), the project followed from the Project dock's hub
or opened here, its rounds from ``project rounds`` (the engine names a new one), the folders that follow the round,
the orientations that follow each run's geometry, the import of labels made elsewhere (the format from the file
name, the pickle box only for a Cellpose _seg.npy), training through the engine with its devices and the card
fields, Use in project and Derive pack through the engine, the starting weights of the project, and the jobs that
return after their project was closed. No engine process is started here: the acceptance tests of
tests/test_training.py run the real engine."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from _p6_train import (
    DERIVED_ID,
    NEW_HASH,
    ChannelTrainRunner,
    FakeHub,
    FakeTrainRunner,
    StaticSession,
    write_model,
    write_pairs,
    write_project,
    write_run,
)

from napari_germquant import _training as T
from napari_germquant._runner import plugin_version
from napari_germquant._training_widget import TrainingWidget

AMBER = "rgb(225, 160, 40)"
RED = "rgb(200, 60, 60)"


def _settle(qtbot, w, timeout: int = 30_000) -> None:
    qtbot.waitUntil(lambda: w.idle, timeout=timeout)


def _dock(make_napari_viewer, runner=None, hub=None, *, viewer: bool = False):
    """The dock on a fake engine (StaticSession) and a hub. A napari viewer only for a test that reads layers or the
    status bar: a viewer's teardown costs about half a second, twice that on the Windows runner."""
    v = make_napari_viewer() if viewer else None
    session = StaticSession(runner) if runner is not None else None
    w = TrainingWidget(v, session=session, hub=hub if hub is not None else FakeHub())
    return v, w


def _items(combo) -> list:
    return [combo.itemData(i) for i in range(combo.count())]


def _texts(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


# ---- where the dock computes ------------------------------------------------------------------------------------

def test_without_an_engine_the_dock_computes_in_this_napari_environment(make_napari_viewer):
    viewer = make_napari_viewer()
    w = TrainingWidget(viewer)
    assert w.idle                                     # no engine is stored: nothing was started
    text = w.engine_label.text()
    assert "No engine chosen" in text and "napari environment" in text and sys.executable in text
    assert AMBER in w.engine_label.styleSheet() and w.engine_route() == "napari"
    assert _items(w.device_combo) == ["cuda", "mps", "cpu"]
    assert w.root is None and not w.round_combo.isEnabled() and not w.new_round_btn.isEnabled()
    assert not w.use_btn.isEnabled() and not w.derive_btn.isEnabled()


def test_an_engine_chosen_in_another_dock_is_shown_before_the_next_step(make_napari_viewer, qtbot, tmp_path,
                                                                       monkeypatch):
    """The engine choice is per computer: stored by another dock while this one is open, the next step goes there,
    and the engine line says so first (never a silent switch)."""
    from napari_germquant import _settings

    tree = tmp_path / "results"
    write_run(tree / "a", "a", kind="3d")
    seen: list = []

    def export_training(self, results, out, **kw):
        seen.append(kw)
        write_pairs(out, "a", 2)
        return {"exit_code": 0, "lines": []}

    from napari_germquant._runner import InProcessRunner

    here: list = []

    def export_slices(source, pairs_dir, settings):
        here.append(settings)
        write_pairs(pairs_dir, "a", 2)
        return T.read_manifest_csv(pairs_dir)

    monkeypatch.setattr(InProcessRunner, "export_training", export_training)
    monkeypatch.setattr(T, "export_slices", export_slices)
    viewer = make_napari_viewer()
    w = TrainingWidget(viewer)
    assert w.engine_route() == "napari" and "No engine chosen" in w.engine_label.text()
    _settings.set_engine_choice(_settings.INPROCESS)       # chosen in the Run dock, say
    w.set_source(tree)
    w.set_pairs_dir(tmp_path / "pairs")
    _settle(qtbot, w)
    assert w.export()
    assert "No engine chosen" not in w.engine_label.text()           # said before the step runs
    _settle(qtbot, w)
    assert w.engine_route() == "engine" and len(seen) == 1           # the engine runner's export, not the dock's
    assert "No engine chosen" not in w.engine_label.text() and "in process" in w.engine_label.text()
    assert len(w.pairs) == 2
    # the choice removed again: the next step runs here, and the engine line says so before it starts
    _settings.set_engine_choice(None)
    assert w.export(overwrite_corrections=True)
    assert w.engine_route() == "napari" and "No engine chosen" in w.engine_label.text()
    _settle(qtbot, w)
    assert len(seen) == 1 and len(here) == 1 and w.last_error is None, w.status_label.text()


def test_a_step_that_returns_after_another_project_opened_leaves_its_folders(make_napari_viewer, qtbot, tmp_path):
    a = write_project(tmp_path / "a", rounds=("r01",))
    b = write_project(tmp_path / "b")
    write_run(a / "results" / "img", "img", kind="3d")
    runner = FakeTrainRunner()
    runner.gates["export_training"] = threading.Event()
    runner.entered["export_training"] = threading.Event()
    viewer, w = _dock(make_napari_viewer, runner)
    w.set_project(a)
    _settle(qtbot, w)
    assert Path(w.pairs_edit.text()) == a / "training" / "r01"
    assert w.export()
    qtbot.waitUntil(lambda: runner.entered["export_training"].is_set(), timeout=30_000)
    w.set_project(b)                                       # the files worker reads b while a's export waits
    qtbot.waitUntil(lambda: w.root == b, timeout=30_000)
    runner.gates["export_training"].set()
    _settle(qtbot, w)
    assert w.root == b and w.pairs_edit.text() == "" and w.pairs == []
    assert "Another project was opened meanwhile" in w.status_label.text() and str(a) in w.status_label.text()


def test_with_an_engine_the_dock_offers_the_engines_devices(make_napari_viewer, qtbot):
    runner = FakeTrainRunner(available={"cuda": True, "mps": False})
    viewer, w = _dock(make_napari_viewer, runner)
    _settle(qtbot, w)
    assert w.engine_route() == "engine" and "a fake engine" in w.engine_label.text()
    assert AMBER not in w.engine_label.styleSheet()
    assert _items(w.device_combo) == ["auto", "cuda", "cpu"] and w.device_combo.currentData() == "auto"


# ---- the project and its rounds ---------------------------------------------------------------------------------

def test_project_mode_follows_the_hub_and_the_round_folders(make_napari_viewer, qtbot, tmp_path):
    root = write_project(tmp_path / "study", images=[("img_a", "images/img_a.ome.tif")], rounds=("r01",))
    hub, runner = FakeHub(), FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner, hub)
    _settle(qtbot, w)
    hub.open(root)
    _settle(qtbot, w)
    assert w.root == root and w.project is not None
    head = w.project_label.text()
    assert "study" in head and "test_pack@0.1.0" in head and "basic" in head and str(root) in head
    # the rounds come from `project rounds` (a listing: no --new, no plugin version)
    assert runner.called("project_rounds")[0] == ((root,), {"new": None, "plugin_version": None})
    assert w.rounds == ["r01"] and w.round_combo.currentText() == "r01" and w.round_combo.isEnabled()
    assert Path(w.source_edit.text()) == root / "results"
    assert Path(w.pairs_edit.text()) == root / "training" / "r01"
    assert Path(w.model_dir_edit.text()) == root / "training" / "r01" / "models" / "r01_nuclei"

    # New round: the engine names it (r02) and stamps the write with the plugin version
    assert w.new_round()
    _settle(qtbot, w)
    assert runner.last("project_rounds")[1] == {"new": True, "plugin_version": plugin_version()}
    assert w.rounds == ["r01", "r02"] and w.round_combo.currentText() == "r02"
    assert Path(w.pairs_edit.text()) == root / "training" / "r02"
    assert Path(w.model_dir_edit.text()) == root / "training" / "r02" / "models" / "r02_nuclei"
    assert w.new_round("fine_tune")
    _settle(qtbot, w)
    assert runner.last("project_rounds")[1]["new"] == "fine_tune" and w.round_combo.currentText() == "fine_tune"
    # a name the engine's rule refuses is refused before anything is asked
    n = len(runner.called("project_rounds"))
    assert not w.new_round("no spaces here") and len(runner.called("project_rounds")) == n
    assert "round name" in w.status_label.text() and RED in w.status_label.styleSheet()

    # another round moves the folders
    w.select_round("r01")
    assert Path(w.pairs_edit.text()) == root / "training" / "r01"

    # the Project dock closes the project: folder mode again, the folders as they were
    hub.close()
    assert w.root is None and w.project is None and w.round_combo.count() == 0
    assert not w.round_combo.isEnabled() and not w.new_round_btn.isEnabled()
    assert Path(w.pairs_edit.text()) == root / "training" / "r01"


def test_the_project_docks_own_hub_is_followed(make_napari_viewer, qtbot, tmp_path):
    """The Project dock's real hub (unit P6-PROJECT), which a dock opened on the same viewer finds by itself: it
    runs from P6-INT's merge of that unit into feat/t5-p6 on (a sibling plugin module, not an engine feature)."""
    hubs = pytest.importorskip("napari_germquant._project_widget",
                               reason="the Project dock (unit P6-PROJECT) is not in this branch yet")
    root = write_project(tmp_path / "study", rounds=("r01",))
    viewer = make_napari_viewer()
    hub = hubs.project_hub(viewer)
    w = TrainingWidget(viewer, session=StaticSession(FakeTrainRunner()))
    assert w.hub is hub
    hub.root = root
    hub.opened.emit(str(root))
    _settle(qtbot, w)
    assert w.root == root and w.rounds == ["r01"]
    w.model_edit.setText(str(write_model(root / "training" / "r01" / "models" / "m1")[0]))
    assert w.use_btn.isEnabled()
    hub.running = True
    hub.running_changed.emit(True)
    assert not w.use_btn.isEnabled()
    hub.running = False
    hub.root = None
    hub.closed.emit()
    assert w.root is None


def test_a_new_round_made_elsewhere_is_listed_at_the_next_reading(make_napari_viewer, qtbot, tmp_path):
    """Every reading of the Project dock reads the project here again; the engine lists the rounds again only when
    the round folders on disk are not the ones it listed (a round made in a terminal, say)."""
    root = write_project(tmp_path / "study", rounds=("r01",))
    hub, runner = FakeHub(), FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner, hub)
    hub.open(root)
    _settle(qtbot, w)
    assert w.rounds == ["r01"] and len(runner.called("project_rounds")) == 1
    hub.changed.emit()
    _settle(qtbot, w)
    assert len(runner.called("project_rounds")) == 1           # nothing changed: no engine call
    (root / "training" / "r02").mkdir()                         # `germquant project rounds STUDY --new`
    hub.changed.emit()
    _settle(qtbot, w)
    assert len(runner.called("project_rounds")) == 2 and w.rounds == ["r01", "r02"]
    assert w.round_combo.currentText() == "r01"                 # the round chosen stays chosen


def test_a_project_opened_here_survives_the_hub_closing_another(make_napari_viewer, qtbot, tmp_path):
    root = write_project(tmp_path / "mine")
    hub, runner = FakeHub(), FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner, hub)
    w.set_project(root)
    _settle(qtbot, w)
    assert w.root == root and w.rounds == []
    assert "New round" in w.notice_label.text()             # no round yet: the dock says how to make one
    hub.close()                                             # the Project dock closes something else
    assert w.root == root
    w.close_project()
    assert w.root is None
    # a folder that is no project is refused, and the dock stays as it was
    w.set_project(tmp_path / "nothing_here")
    _settle(qtbot, w)
    assert w.root is None and "no project" in w.status_label.text().lower()


def test_an_engine_without_projects_gets_one_message_and_the_folders_stay_free(make_napari_viewer, qtbot, tmp_path):
    root = write_project(tmp_path / "study", rounds=("r01",))
    runner = FakeTrainRunner(features={"packs", "import_labels", "train_curve"})
    viewer, w = _dock(make_napari_viewer, runner)
    w.set_project(root)
    _settle(qtbot, w)
    assert w.root == root and w.rounds == [] and not w.round_combo.isEnabled()
    assert "Projects need the tranche-5 Train B engine" in w.notice_label.text()
    assert not w.new_round_btn.isEnabled() and not w.use_btn.isEnabled()
    # the folders stay the user's to choose: the results folder is the project's
    assert Path(w.source_edit.text()) == root / "results" and w.pairs_edit.isEnabled()


def test_a_step_the_engine_cannot_do_gets_its_one_message(make_napari_viewer, qtbot, tmp_path):
    runner = FakeTrainRunner(features={"packs"})
    viewer, w = _dock(make_napari_viewer, runner)
    _settle(qtbot, w)
    w.set_labels_file(tmp_path / "masks.tif")
    w.set_import_image(tmp_path / "img.ome.tif")
    w.set_pairs_dir(tmp_path / "pairs")
    assert w.import_labels()
    _settle(qtbot, w)
    text = w.status_label.text()
    assert "Importing labels needs the tranche-5 Train B engine" in text and "FAILED" not in text
    assert w.last_import is None and not runner.called("import_labels")


def test_a_reading_that_returns_after_the_project_closed_changes_nothing(make_napari_viewer, qtbot, tmp_path):
    root = write_project(tmp_path / "study", rounds=("r01",))
    runner = FakeTrainRunner()
    runner.gates["project_rounds"] = threading.Event()
    runner.entered["project_rounds"] = threading.Event()
    viewer, w = _dock(make_napari_viewer, runner)
    w.set_project(root)
    qtbot.waitUntil(lambda: runner.entered["project_rounds"].is_set(), timeout=30_000)
    w.close_project()
    runner.gates["project_rounds"].set()
    _settle(qtbot, w)
    assert w.root is None and w.rounds == [] and w.round_combo.count() == 0


# ---- the orientations follow each run -------------------------------------------------------------------------

def test_orientations_follow_each_runs_geometry(make_napari_viewer, qtbot, tmp_path):
    tree = tmp_path / "results"
    write_run(tree / "a", "a", kind="3d")
    write_run(tree / "b", "b", kind="projected")
    write_run(tree / "c", "c", kind="planar", batch=True)
    runner = FakeTrainRunner()                          # an engine with train-curve: it exports "auto"
    viewer, w = _dock(make_napari_viewer, runner)
    w.set_source(tree)
    w.set_pairs_dir(tmp_path / "pairs")
    _settle(qtbot, w)
    assert [(g.image_id, g.kind) for g in w.geometry] == [("a", "3d"), ("b", "projected"), ("c", "planar")]
    text = w.geometry_label.text()
    assert "1 3D run" in text and "xy, xz and yz" in text
    assert "1 projected run" in text and "proj" in text and "1 single-plane run" in text
    assert all(cb.isEnabled() for cb in w.orient_checks.values())
    assert w.export()
    _settle(qtbot, w)
    args, kw = runner.last("export_training")
    assert args == (str(tree), str(tmp_path / "pairs")) and kw["orientations"] == "auto"
    assert (kw["slices_per_image"], kw["seed"], kw["min_labels"], kw["max_gb"]) == (3, 0, 3, 2.0)
    assert "export-training" in w.status_label.text() and "--orientations auto" in w.status_label.text()
    w.orient_checks["yz"].setChecked(False)              # a subset names the slices of the 3D runs
    assert w.export(overwrite_corrections=True)
    _settle(qtbot, w)
    assert runner.last("export_training")[1]["orientations"] == ("xy", "xz")

    # only runs of one plane or a projection: there is nothing to choose, the export follows them
    flat = tmp_path / "flat"
    write_run(flat / "b", "b", kind="projected")
    w.set_source(flat)
    _settle(qtbot, w)
    assert not any(cb.isEnabled() for cb in w.orient_checks.values())
    assert "3D" not in w.geometry_label.text() and "proj" in w.geometry_label.text()
    assert w.export(overwrite_corrections=True)
    _settle(qtbot, w)
    assert runner.last("export_training")[1]["orientations"] == "auto"


def test_an_engine_from_before_the_proj_export_gets_the_list_and_a_warning(make_napari_viewer, qtbot, tmp_path):
    tree = tmp_path / "results"
    write_run(tree / "a", "a", kind="3d")
    write_run(tree / "b", "b", kind="projected")
    old = FakeTrainRunner(features={"packs"})           # no train_curve: no proj pairs, no "auto"
    viewer, w = _dock(make_napari_viewer, old)
    _settle(qtbot, w)
    w.set_source(tree)
    w.set_pairs_dir(tmp_path / "pairs")
    _settle(qtbot, w)
    assert "proj" in w.geometry_label.text() and AMBER in w.geometry_label.styleSheet()
    assert "slices of its copied labels" in w.geometry_label.text()
    assert w.export()
    _settle(qtbot, w)
    assert old.last("export_training")[1]["orientations"] == ("xy", "xz", "yz")


def test_without_an_engine_the_export_runs_here_and_follows_the_runs(make_napari_viewer, qtbot, tmp_path,
                                                                     monkeypatch):
    tree = tmp_path / "results"
    write_run(tree / "a", "a", kind="3d")
    seen: list = []

    def export(source, pairs_dir, settings):
        seen.append(settings)
        write_pairs(pairs_dir, "a", 2)
        import pandas as pd

        return pd.read_csv(Path(pairs_dir) / "manifest.csv")

    monkeypatch.setattr(T, "export_slices", export)
    viewer = make_napari_viewer()
    w = TrainingWidget(viewer)
    w.set_source(tree)
    _settle(qtbot, w)
    assert w.export()
    _settle(qtbot, w)
    # this napari environment's export: "auto" when its germquant has it, else the ticked list as before
    assert seen[-1].orientations == ("auto" if T.export_auto_supported() else ("xy", "xz", "yz"))
    assert w.last_manifest is not None and len(w.pairs) == 2


# ---- export, import and training through the engine -----------------------------------------------------------

def test_export_and_training_go_through_the_engine(make_napari_viewer, qtbot, tmp_path):
    tree = tmp_path / "results"
    write_run(tree / "a", "a", kind="3d")
    runner = FakeTrainRunner(available={"cuda": True})
    viewer, w = _dock(make_napari_viewer, runner, viewer=True)
    statuses: list[str] = []
    viewer.events.status.connect(lambda e: statuses.append(str(e.value)))
    _settle(qtbot, w)
    w.set_source(tree)
    w.set_pairs_dir(tmp_path / "pairs")
    assert w.export()
    _settle(qtbot, w)
    assert w.last_manifest is not None and len(w.pairs) == 2 and w.index == 0
    assert "2 slice pair(s) from 1 run(s)" in w.status_label.text()

    w.model_dir_edit.setText(str(tmp_path / "models" / "m1"))
    w.base_combo.setCurrentText("none")
    w.epochs_spin.setValue(4)
    w.split_combo.setCurrentText("image")
    w.select_device("cuda")
    w.kind_combo.setCurrentIndex(w.kind_combo.findData("cell"))
    w.system_edit.setText("budding yeast")
    assert w.train()
    _settle(qtbot, w)
    args, kw = runner.last("train_nuclei")
    assert args == (str(tmp_path / "pairs"), str(tmp_path / "models" / "m1"))
    assert kw == {"base": "none", "name": None, "epochs": 4, "split_by": "image", "val_frac": 0.2, "seed": 0,
                  "device": "cuda", "system": "budding yeast", "kind": "cell", "dataset": None}
    assert any("epoch 0/4" in s for s in statuses)               # the engine's epoch line on the status bar
    model = tmp_path / "models" / "m1" / "m1"
    assert Path(w.model_edit.text()) == model and Path(w.card_edit.text()) == model.with_name("m1.card.json")
    assert w.reg_name_edit.text() == "m1" and w.last_train["model_path"] == str(model)
    text = w.status_label.text()
    assert "m1: trained on 3 slice(s)" in text and "hold-out at IoU 0.5" in text and "train-nuclei --pairs" in text
    assert "loading the pairs" not in text                       # the engine's summary, not its whole output
    assert "--kind cell" in text and "--system" in text
    # auto: the engine chooses; nothing more is passed
    w.select_device("auto")
    w.kind_combo.setCurrentIndex(w.kind_combo.findData("auto"))
    w.system_edit.setText("")
    assert w.train()
    _settle(qtbot, w)
    kw = runner.last("train_nuclei")[1]
    assert kw["device"] == "auto" and kw["kind"] is None and kw["system"] is None


def test_cancel_ends_a_training_through_the_engine(make_napari_viewer, qtbot, tmp_path):
    pairs = tmp_path / "pairs"
    write_pairs(pairs, "a", 2)
    runner = FakeTrainRunner()
    runner.gates["train_nuclei"] = threading.Event()
    runner.entered["train_nuclei"] = threading.Event()
    viewer, w = _dock(make_napari_viewer, runner)
    _settle(qtbot, w)
    w.open_pairs(pairs)
    w.model_dir_edit.setText(str(tmp_path / "models" / "m1"))
    assert not w.cancel_btn.isEnabled()
    assert w.train()
    qtbot.waitUntil(lambda: runner.entered["train_nuclei"].is_set(), timeout=30_000)
    assert w.cancel_btn.isEnabled() and w.cancel()
    _settle(qtbot, w)
    assert "cancelled" in w.status_label.text() and w.last_train is None
    assert w.train_btn.isEnabled() and not w.cancel_btn.isEnabled()


def test_import_guesses_the_format_and_guards_the_pickle(make_napari_viewer, qtbot, tmp_path):
    pairs = tmp_path / "pairs"
    image = tmp_path / "img_a.ome.tif"
    image.write_bytes(b"not read by the fake engine")
    runner = FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner)
    _settle(qtbot, w)
    for name, fmt in (("cells_segm.npz", "cellacdc-npz"), ("img_a_seg.npy", "cellpose-npy"),
                      ("masks.tif", "tiff"), ("masks.ome.tiff", "tiff"), ("masks.png", "png")):
        w.set_labels_file(tmp_path / name)
        assert w.format_combo.currentData() == fmt, name
        assert w.pickle_check.isEnabled() == (fmt == "cellpose-npy"), name
        assert w.pickle_label.isHidden() == (fmt != "cellpose-npy"), name
    assert "a _seg.npy file is a Python pickle: loading it can run code; allow only files you made" == T.PICKLE_WARNING
    w.set_labels_file(tmp_path / "img_a_seg.npy")
    assert T.PICKLE_WARNING in w.pickle_label.text() and not w.pickle_check.isChecked()
    w.set_import_image(image)
    w.set_pairs_dir(pairs)
    # without the box the engine refuses, and nothing is written
    assert w.import_labels()
    _settle(qtbot, w)
    assert "--allow-pickle" in w.status_label.text() and RED in w.status_label.styleSheet()
    assert w.last_import is None and not (pairs / "manifest.csv").exists()
    assert runner.last("import_labels")[1] == {"allow_pickle": False, "frame": None}
    # with the box: read, with the engine's warnings in amber, and the pairs open
    w.pickle_check.setChecked(True)
    w.frame_spin.setValue(2)
    assert w.import_labels()
    _settle(qtbot, w)
    args, kw = runner.last("import_labels")
    assert args == (str(tmp_path / "img_a_seg.npy"), "cellpose-npy", str(image), str(pairs))
    assert kw == {"allow_pickle": True, "frame": 2}
    assert w.last_import["n_pairs"] == 1 and len(w.pairs) == 1
    text = w.status_label.text()
    assert "imported 2 object(s) of img_a" in text and "1 pair(s)" in text and AMBER in w.status_label.styleSheet()
    assert "none was named: the pairs take channel 0" in text and "import-labels" in text
    # another format unticks the box and turns it off
    w.set_labels_file(tmp_path / "cells_segm.npz")
    assert not w.pickle_check.isChecked() and not w.pickle_check.isEnabled()


def test_the_channel_is_offered_once_the_runner_can_pass_it(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    image = tmp_path / "img_a.ome.tif"
    image.write_bytes(b"x")
    monkeypatch.setattr(T, "image_channels", lambda path: ["405", "477 SYP-3"])
    # the plugin's runner takes no channel yet (handoff/P6-TRAIN_blocker.md): the engine's default, said so
    viewer, w = _dock(make_napari_viewer, FakeTrainRunner())
    _settle(qtbot, w)
    w.set_import_image(image)
    _settle(qtbot, w)
    assert not w.channel_combo.isEnabled() and "channel 0" in w.channel_combo.toolTip()
    # a runner that takes it: the image's channels, passed by name when one is chosen
    runner = ChannelTrainRunner()
    viewer2, w2 = _dock(make_napari_viewer, runner)
    _settle(qtbot, w2)
    w2.set_labels_file(tmp_path / "masks.tif")
    w2.set_import_image(image)
    w2.set_pairs_dir(tmp_path / "pairs")
    _settle(qtbot, w2)
    assert w2.channel_combo.isEnabled() and _texts(w2.channel_combo)[1:] == ["0: 405", "1: 477 SYP-3"]
    assert w2.import_labels()
    _settle(qtbot, w2)
    assert runner.last("import_labels")[1]["channel"] is None          # the engine's default
    w2.select_channel("477 SYP-3")
    assert w2.import_labels()
    _settle(qtbot, w2)
    assert runner.last("import_labels")[1]["channel"] == "477 SYP-3"


def test_in_a_project_the_import_takes_its_images_and_the_round(make_napari_viewer, qtbot, tmp_path):
    root = write_project(tmp_path / "study", images=[("img_a", "images/img_a.ome.tif"),
                                                       ("img_b", str(tmp_path / "far" / "img_b.ome.tif"))],
                         rounds=("r01",))
    runner = FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner)
    w.set_project(root)
    _settle(qtbot, w)
    assert _texts(w.image_combo)[:2] == ["img_a", "img_b"]
    assert w.image_combo.itemData(0) == str(root / "images" / "img_a.ome.tif")
    assert w.image_combo.itemData(1) == str(tmp_path / "far" / "img_b.ome.tif")
    w.set_labels_file(tmp_path / "img_b_segm.npz")
    w.image_combo.setCurrentIndex(1)
    assert w.import_labels()
    _settle(qtbot, w)
    args, _kw = runner.last("import_labels")
    assert args == (str(tmp_path / "img_b_segm.npz"), "cellacdc-npz", str(tmp_path / "far" / "img_b.ome.tif"),
                    str(root / "training" / "r01"))


# ---- the starting weights, Use in project and Derive pack -------------------------------------------------------

def test_starting_weights_offer_the_packs_and_the_projects_models(make_napari_viewer, qtbot, tmp_path):
    root = write_project(tmp_path / "study", recipe="with_model", pack_models=("tinynet@f5055f8a",), rounds=("r01",),
                         object_kind="cell")
    model, _card = write_model(root / "training" / "r01" / "models" / "m1")
    runner = FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner)
    before = _texts(w.base_combo)
    assert w.kind_combo.currentData() == "auto"
    w.set_project(root)
    _settle(qtbot, w)
    texts = _texts(w.base_combo)
    assert texts[0] == "tinynet@f5055f8a" and texts[1] == str(model) and w.base_combo.currentIndex() == 0
    assert texts[2:4] == ["cpsam", "none"]
    assert w.kind_combo.currentData() == "cell"           # the pack's object kind goes on the card (--kind)
    w.close_project()
    assert _texts(w.base_combo) == before and w.kind_combo.currentData() == "auto"


def test_use_in_project_and_derive_pack_go_through_the_engine(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    root = write_project(tmp_path / "study", recipe="with_model", pack_models=("tinynet@f5055f8a",), rounds=("r01",))
    model, card = write_model(root / "training" / "r01" / "models" / "r01_nuclei")
    hub, runner = FakeHub(), FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner, hub)
    hub.open(root)
    _settle(qtbot, w)
    w.model_edit.setText(str(model))
    w.card_edit.setText(str(card))
    assert w.use_btn.isEnabled() and w.derive_btn.isEnabled()

    # Use in project: the model by its path inside the project, stamped with the plugin version
    assert w.use_in_project()
    _settle(qtbot, w)
    args, kw = runner.last("project_use_model")
    assert args == (root, "training/r01/models/r01_nuclei/r01_nuclei")
    assert kw == {"card": "training/r01/models/r01_nuclei/r01_nuclei.card.json", "plugin_version": plugin_version()}
    text = w.status_label.text()
    assert NEW_HASH in text and "validated: false" in text and "stale" in text
    assert hub.reloads == 1 and w.last_use["config_hash"] == NEW_HASH

    # never while a run of the project goes (the Project dock's run)
    hub.running = True
    hub.running_changed.emit(True)
    assert not w.use_btn.isEnabled()
    n = len(runner.called("project_use_model"))
    assert not w.use_in_project() and len(runner.called("project_use_model")) == n
    assert "run" in w.status_label.text()
    hub.running = False
    hub.running_changed.emit(False)
    # a model outside the project cannot be named by the project
    outside, _ = write_model(tmp_path / "elsewhere" / "m2")
    w.model_edit.setText(str(outside))
    assert not w.use_in_project() and "inside the project" in w.status_label.text()
    w.model_edit.setText(str(model))

    # Derive pack: from the project's pinned pack and recipe, into its packs/, then listed with the project
    assert w.derive_pack()
    _settle(qtbot, w)
    args, kw = runner.last("pack_derive")
    assert args == ("test_pack@0.1.0", "with_model", str(model), str(card), str(root / "packs"))
    assert kw == {"new_id": None}
    assert runner.last("list_packs")[1] == {"project": root}
    text = w.status_label.text()
    assert DERIVED_ID in text and "listed" in text and w.last_derive["id"] == DERIVED_ID


def test_derive_and_use_say_what_the_engine_refused(make_napari_viewer, qtbot, tmp_path):
    root = write_project(tmp_path / "study", recipe="basic", rounds=("r01",))
    model, card = write_model(root / "training" / "r01" / "models" / "r01_nuclei")
    runner = FakeTrainRunner(refuse={"pack_derive": "the parent recipe test_pack@0.1.0/basic never runs a Cellpose "
                                                    "model (segmentation.nuclei.method classical)"})
    viewer, w = _dock(make_napari_viewer, runner)
    w.set_project(root)
    _settle(qtbot, w)
    w.model_edit.setText(str(model))
    assert w.derive_pack()
    _settle(qtbot, w)
    assert "never runs a Cellpose model" in w.status_label.text() and RED in w.status_label.styleSheet()
    assert w.last_derive is None and not runner.called("list_packs")


@pytest.mark.parametrize("missing", ["model", "card"])
def test_use_in_project_needs_the_model_and_its_card(make_napari_viewer, qtbot, tmp_path, missing):
    root = write_project(tmp_path / "study", rounds=("r01",))
    model, card = write_model(root / "training" / "r01" / "models" / "r01_nuclei")
    (model if missing == "model" else card).unlink()
    runner = FakeTrainRunner()
    viewer, w = _dock(make_napari_viewer, runner)
    w.set_project(root)
    _settle(qtbot, w)
    w.model_edit.setText(str(model))
    w.card_edit.setText("")
    ok = w.use_in_project()
    _settle(qtbot, w)
    if missing == "model":
        assert not ok and "model file" in w.status_label.text() and not runner.called("project_use_model")
    else:
        # no card next to the model: the engine is asked without one and says what it needs
        assert ok and runner.last("project_use_model")[1]["card"] is None
