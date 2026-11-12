"""The Project dock (napari_germquant._project_widget; tranche 5 Train B, unit P6-PROJECT; design 8.4,
plugin_specs 4.1): New, Open, Save and Close, the Images, Samples and Run tabs, the status board and its live
updates from the engine's run log, Cancel, refusals, the lock, collect and export-commands, and results opened
from the board.

Most tests drive the dock against the fake engine of tests/_p6_project.py (no process starts); one runs the real
subprocess runner against the stand-in engine of tests/_p6_runner.py, whose log the dock tails. Acceptance 2
(Run all through both runners writes what `germquant project run` writes) and 3 (the codeless round) run the
real engine and skip, naming the feature, until it has projects (conftest.need_features)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _p6_project import (
    BASE_HASH,
    FakeRunner,
    FakeSession,
    image_entry,
    read_manifest,
    status_record,
    write_manifest,
    write_results,
)
from _p6_runner import run_spec, standin_engine
from conftest import CHANNELS, need_features, settle, write_ome
from qtpy.QtCore import Qt

from napari_germquant import _engine, _project, _settings
from napari_germquant import _project_widget as PW
from napari_germquant._engine_session import EngineSession
from napari_germquant._project_widget import ProjectTab, ProjectWidget, project_hub
from napari_germquant._runner import (
    RunCancelled,
    RunnerError,
    check_run_id,
    engine_too_old,
    plugin_engine_info,
    plugin_version,
    user_runs_dir,
)
from napari_germquant._settings import INPROCESS


_DOCKS: list[tuple[ProjectWidget, FakeRunner]] = []


@pytest.fixture(autouse=True)
def _no_run_left_behind(qtbot):
    """After each test: every fake run released and every dock's run cancelled and finished, so a test that failed
    part way never leaves a worker thread blocked (napari's thread pool would wait for it at exit)."""
    yield
    for w, fake in _DOCKS:
        fake.release()
        w.cancel()
    for w, _fake in _DOCKS:
        qtbot.waitUntil(lambda w=w: not w.running, timeout=30_000)
    _DOCKS.clear()


def _dock(qtbot, runner: FakeRunner | None = None, *, viewer=None, **session_kw) -> tuple[ProjectWidget, FakeRunner]:
    runner = runner or FakeRunner()
    w = ProjectWidget(viewer, session=FakeSession(runner, **session_kw))
    qtbot.addWidget(w)
    _DOCKS.append((w, runner))
    return w, runner


def _make_project(tmp_path: Path, ids=("a", "b"), **kw) -> Path:
    """A project folder whose manifest records this computer's choices as the fake engine and the default device
    give them (so the dock opens it with nothing unsaved)."""
    root = tmp_path / "study"
    kw.setdefault("run", {"device": "auto", "runner": "subprocess", "interpreters": {"basic": {"python": sys.executable}}})
    write_manifest(root, images=[image_entry(tmp_path / f"{i}.ome.tif", i) for i in ids], **kw)
    return root


def _board(w: ProjectWidget) -> list[tuple[str, str]]:
    return [(r.image_id, r.status) for r in w.rows]


def _cells(table, column: int) -> list[str]:
    return [table.item(i, column).text() if table.item(i, column) else "" for i in range(table.rowCount())]


# ---------------------------------------------------------------------------------------------------
# the fake engine
# ---------------------------------------------------------------------------------------------------


def test_new_open_and_save_go_through_the_engine_with_the_plugin_version(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    assert w.root is None and not w.save_btn.isEnabled() and not w.run_all_btn.isEnabled()
    img = tmp_path / "a.ome.tif"
    root = tmp_path / "study"
    w.new_project(root, "test_pack@0.1.0", recipe="basic", images=[img], name="study")
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    (new,) = fake.called("project_new")
    assert (new["root"], new["pack"], new["recipe"], new["name"]) == (root, "test_pack@0.1.0", "basic", "study")
    assert [str(p) for p in new["images"]] == [str(img)] and new["plugin_version"] == plugin_version()
    assert fake.called("project_status")                      # the board is the engine's status
    # this computer's device, runner and interpreter recorded at once (D42), stamped with the plugin version
    (here,) = fake.called("project_edit")
    assert (here["device"], here["runner"], here["interpreters"]) == ("auto", "subprocess", {"basic": fake.python})
    assert here["plugin_version"] == plugin_version() and here["xy_stride"] is None and not here["disable"]
    assert w.root == root and w.manifest == read_manifest(root) and _board(w) == [("a", "missing")]
    assert w.name_label.text() == "study" and str(root) in w.folder_label.text()
    assert "test_pack@0.1.0" in w.pack_label.text() and "basic" in w.pack_label.text()
    assert BASE_HASH in w.hash_label.text() and w.badge_label.text() == "not validated (provisional pack)"
    assert _settings.recent_projects()[0] == str(root)
    assert w.save_btn.isEnabled() and w.run_all_btn.isEnabled() and not w.unsaved
    # the Run tab's settings are unsaved until Save, which records them through `project edit`
    w.set_xy_stride(3)
    w.set_stage("spots", False)
    w.set_device("cpu")
    assert w.unsaved and "unsaved" in w.save_btn.text().lower()
    w.save()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    _here, edit = fake.called("project_edit")
    assert edit["xy_stride"] == 3 and edit["disable"] == ["spots"] and edit["device"] == "cpu"
    assert edit["runner"] is None and edit["interpreters"] is None        # recorded already: only what differs
    assert edit["plugin_version"] == plugin_version() and not edit["dry_run"]
    m = read_manifest(root)
    assert m["run"]["xy_stride"] == 3 and m["recipe"]["overrides"] == {"spots.enabled": False}
    assert w.manifest == m and not w.unsaved and m["recipe"]["expected_config_hash"] in w.hash_label.text()
    assert w.badge_label.text() == "validated: false (edited)"
    w.save()                                                  # nothing changed: nothing written
    settle(w, qtbot)
    assert len(fake.called("project_edit")) == 2 and "nothing to save" in w.status_label.text()
    # Close, then Open: the manifest read in process, the status asked of the engine again
    w.close_project()
    assert w.root is None and w.rows == [] and w.images_table.rowCount() == 0 and not w.save_btn.isEnabled()
    w.open_project(root)
    settle(w, qtbot)
    assert w.root == root and w.manifest == m and len(fake.called("project_status")) == 3
    assert w.stride_spin.value() == 3 and not w.stage_checks["spots"].isChecked() and not w.unsaved
    assert w.recent_combo.itemData(0) == str(root)
    # Open of a folder that is no project says so, and the project that is open stays open
    w.open_project(tmp_path / "nothing")
    settle(w, qtbot)
    assert w.root == root and "not a germquant project" in w.status_label.text()


def test_the_images_tab_shows_headers_refusals_and_unreachable_paths(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    good = write_ome(tmp_path / "good.ome.tif", np.zeros((3, 2, 8, 8), np.uint16), CHANNELS)
    gone = tmp_path / "gone" / "far.ome.tif"
    unc = "\\\\nas-not-there.invalid\\SpermBank\\x.nd2"
    root = tmp_path / "study"
    write_manifest(root, images=[image_entry(good, "good"), image_entry(gone, "far"), image_entry(unc, "x")])
    w.open_project(root)
    settle(w, qtbot)
    assert _cells(w.images_table, 0) == ["good", "far", "x"]
    header = _cells(w.images_table, 3)
    assert "C=3 Z=2 Y=8 X=8" in header[0] and "channels: 405, 477, 545" in header[0]
    assert header[1] == "file not found"
    assert header[2] == _project.UNREACHABLE                  # a share that is not there never blocks the dock
    assert w.images_table.item(2, 3).foreground().color() == PW.AMBER
    assert w.headers["good"]["sizes"]["Z"] == 2
    # Add: the engine's refusals per path, the added images with the header the engine read
    fake.headers["h"] = {"sizes": {"C": 2, "Z": 1, "Y": 40, "X": 48}, "channel_names": ["DAPI", "GFP"],
                         "spacing": [1, 1, 1], "spacing_ok": False, "spacing_yx_um": [0.1, 0.1], "single_plane": True}
    w.add_paths([str(tmp_path / "h.ome.tif"), str(tmp_path / "refuse_me.ome.tif")])
    settle(w, qtbot)
    (add,) = fake.called("project_add")
    assert add["paths"] == [str(tmp_path / "h.ome.tif"), str(tmp_path / "refuse_me.ome.tif")]
    assert add["plugin_version"] == plugin_version()
    assert "refuse_me.ome.tif" in w.refused_label.text() and "multi-scene" in w.refused_label.text()
    assert _cells(w.images_table, 0) == ["good", "far", "x", "h"]
    assert "single plane, voxel um dy=0.100 dx=0.100" in _cells(w.images_table, 3)[3]
    assert [r.image_id for r in w.rows] == ["good", "far", "x", "h"]
    # Paste path keeps the text as it is: a UNC path is never rewritten
    w.paste_edit.setText(unc + "  ")
    w.paste_btn.click()
    settle(w, qtbot)
    assert fake.called("project_add")[-1]["paths"] == [unc]


def test_the_board_follows_the_run_log_live_and_refreshes_at_the_end(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    assert _board(w) == [("a", "missing"), ("b", "missing")] and w.board.horizontalHeaderItem(3).text() == "nuclei"
    fake.gate, second = threading.Event(), threading.Event()
    fake.script = [{"event": "image_done", "image_id": "a"},
                   {"event": "image_failed", "image_id": "b", "error": "RuntimeError: no nuclei", "wait": second}]
    fake.results, fake.run_exit = {"a"}, 1
    w.run_all()
    qtbot.waitUntil(lambda: w.running and _board(w) == [("a", "waiting"), ("b", "waiting")], timeout=10_000)
    assert _project.run_in_progress(root) and not w.save_btn.isEnabled() and not w.add_files_btn.isEnabled()
    assert not w.run_all_btn.isEnabled() and w.cancel_btn.isEnabled() and not w.idle
    fake.gate.set()
    qtbot.waitUntil(lambda: _board(w) == [("a", "done"), ("b", "waiting")], timeout=10_000)    # row by row
    assert w.rows[0].last_run_id is not None and w.rows[0].device_used == "cpu"
    fake.statuses = {"a": {"status": "done", "qc_pass": True, "last_run_id": w.rows[0].last_run_id},
                     "b": {"status": "failed", "error": "RuntimeError: no nuclei"}}
    second.set()
    qtbot.waitUntil(lambda: not w.running, timeout=10_000)
    settle(w, qtbot)
    res = w.last_result
    assert res["state"] == "failed" and res["exit_code"] == 1 and check_run_id(res["run_id"])
    assert "1 done, 1 failed" in w.run_label.text()
    assert _board(w) == [("a", "done"), ("b", "failed")] and w.rows[1].error == "RuntimeError: no nuclei"
    assert (w.rows[0].objects, w.rows[0].spots) == (3, 5)        # from the run's image_summary and spots tables
    assert "failed: RuntimeError: no nuclei" in _cells(w.board, 1)[1]
    assert len(fake.called("project_status")) == 2               # at Open, and once the run ended
    (run,) = fake.called("project_run")
    assert run["images"] is None and run["resume"] is True and run["run_id"] == res["run_id"]
    assert run["device"] == "auto"                               # this computer's choice, always passed
    assert not _project.run_in_progress(root) and w.save_btn.isEnabled() and w.run_all_btn.isEnabled()


def test_cancel_ends_the_run_and_the_board_says_interrupted(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    fake.gate = threading.Event()                                # never set: the run waits for Cancel
    fake.script = [{"event": "image_done", "image_id": "a"}]
    w.run_all()
    qtbot.waitUntil(lambda: w.running and _board(w) == [("a", "waiting"), ("b", "waiting")], timeout=10_000)
    assert w.cancel()
    qtbot.waitUntil(lambda: not w.running, timeout=10_000)
    settle(w, qtbot)
    assert isinstance(w.last_error, RunCancelled) and w.last_result is None
    assert "interrupted" in w.run_label.text()
    assert _board(w) == [("a", "missing"), ("b", "missing")]   # the engine's status after the stop
    assert not w.cancel()                                        # nothing left to cancel
    assert not _project.run_in_progress(root)


def test_a_refused_run_shows_the_engines_reason(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    w.open_project(_make_project(tmp_path))
    settle(w, qtbot)
    fake.refuse_run = "the model file models/m1 differs from the sha256 the project recorded"
    w.run_all()
    settle(w, qtbot)
    assert w.last_result["state"] == "refused" and w.last_result["exit_code"] == 2
    assert "refused" in w.run_label.text() and "differs from the sha256" in w.run_label.text()
    assert w.run_label.styleSheet() == PW.css(PW.RED)


def test_run_records_pending_settings_first_and_runs_the_selection(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    w.open_project(_make_project(tmp_path, ids=("a", "b", "c")))
    settle(w, qtbot)
    w.set_xy_stride(2)
    w.run_selected(["b", "c"])
    settle(w, qtbot)
    names = [n for n, _ in fake.calls if n in ("project_edit", "project_run")]
    assert names == ["project_edit", "project_run"] and fake.called("project_edit")[0]["xy_stride"] == 2
    assert fake.called("project_run")[0]["images"] == ["b", "c"] and fake.called("project_run")[0]["resume"]
    # the board's selection is what Run selected and Rerun selected take; Rerun runs without resume
    w.board.selectRow(0)
    w.rerun_selected()
    settle(w, qtbot)
    last = fake.called("project_run")[-1]
    assert last["images"] == ["a"] and last["resume"] is False
    assert len(fake.called("project_edit")) == 1                 # nothing new to record
    # nothing selected: Run selected says so and runs nothing
    w.board.clearSelection()
    w.run_selected()
    settle(w, qtbot)
    assert len(fake.called("project_run")) == 2 and "select" in w.status_label.text().lower()


class _ProbeTab(ProjectTab):
    """A tab registered the way unit P6-RECIPE registers its Recipe tab (one line in EXTRA_TABS)."""

    def __init__(self, dock):
        super().__init__(dock)
        self.seen: list = []

    def project_changed(self, state) -> None:
        self.seen.append(("project", None if state is None else state.root))

    def running_changed(self, running: bool) -> None:
        self.seen.append(("running", running))


def test_recipe_and_sample_edits_are_off_while_a_run_goes(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(PW, "EXTRA_TABS", [("Probe", f"{__name__}:_ProbeTab"),
                                           ("Broken", "napari_germquant._no_such_module:Tab")])
    w, fake = _dock(qtbot)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == ["Images", "Samples", "Run", "Probe", "Broken"]
    tab = w.tab("Probe")
    assert isinstance(tab, _ProbeTab) and tab.dock is w
    assert "could not be built" in w.tab("Broken").text() and "Broken" not in w.extra_tabs   # the dock still works
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    assert tab.seen[-1] == ("project", root)
    fake.gate = threading.Event()
    fake.script = [{"event": "image_done", "image_id": "a"}]
    w.run_all()
    qtbot.waitUntil(lambda: w.running, timeout=10_000)
    assert ("running", True) in tab.seen
    # every recipe edit is off: the dock's own edit route, the stage switches, the samples, Save
    w.edit(edits={"spots.log.min_snr": 3.0})
    assert not fake.called("project_edit") and "a run of this project is in progress" in w.status_label.text()
    assert not w.stage_checks["spots"].isEnabled() and not w.stride_spin.isEnabled()
    assert not w.samples_table.isEnabled() and not w.save_btn.isEnabled()
    fake.gate.set()
    qtbot.waitUntil(lambda: not w.running, timeout=10_000)
    settle(w, qtbot)
    assert tab.seen[-1] == ("running", False) or ("running", False) in tab.seen
    assert w.stage_checks["spots"].isEnabled() and w.samples_table.isEnabled()
    w.edit(edits={"spots.log.min_snr": 3.0})
    settle(w, qtbot)
    (edit,) = fake.called("project_edit")
    assert edit["edits"] == {"spots.log.min_snr": 3.0} and edit["plugin_version"] == plugin_version()
    assert w.badge_label.text() == "validated: false (edited)" and BASE_HASH not in w.hash_label.text()
    w.close_project()
    assert tab.seen[-1] == ("project", None)


def test_a_stale_lock_is_shown_and_removed_only_with_its_holder(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    fake.lock = {"host": "LABPC", "pid": 4242}
    w.open_project(_make_project(tmp_path))
    settle(w, qtbot)
    assert "LABPC" in w.lock_label.text() and "4242" in w.lock_label.text() and w.unlock_btn.isEnabled()
    w.remove_stale_lock()
    settle(w, qtbot)
    (unlock,) = fake.called("project_unlock")
    assert (unlock["host"], unlock["pid"]) == ("LABPC", 4242)
    assert w.lock_label.text() == "" and not w.unlock_btn.isEnabled()


def test_collect_and_export_commands_go_through_the_engine(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    w.collect()
    settle(w, qtbot)
    # the engine's record in words (images, the summary table, the rows per table), never the raw record
    text = w.status_label.text()
    assert fake.called("project_collect") == [{"root": root}] and "collected 2 image(s)" in text
    assert "batch_summary.csv" in text and "nuclei 6" in text and "{" not in text
    out = tmp_path / "run_study.ps1"
    w.export_commands(out, shell="ps1")
    settle(w, qtbot)
    (exp,) = fake.called("project_export_commands")
    assert (exp["shell"], exp["out"]) == ("ps1", out) and out.is_file() and str(out) in w.status_label.text()


def test_double_click_opens_the_results_with_their_source(make_napari_viewer, qtbot, tmp_path, monkeypatch):
    """Through napari's public proxy of the viewer (what a dock opened from the Plugins menu gets). The results
    and the source image are read in a worker, never in the GUI thread (a NAS path over VPN)."""
    from napari.utils._proxies import PublicOnlyProxy

    from napari_germquant import _results

    threads = []
    real_load = _results.load_results_layers

    def load(rs, **kw):
        threads.append(threading.current_thread())
        return real_load(rs, **kw)

    monkeypatch.setattr(_results, "load_results_layers", load)
    viewer = make_napari_viewer()
    src = write_ome(tmp_path / "a.ome.tif", np.zeros((3, 2, 16, 24), np.uint16), CHANNELS)
    root = tmp_path / "study"
    write_manifest(root, images=[image_entry(src, "a")])
    write_results(root / "results" / "a", "a", source=src)
    w, fake = _dock(qtbot, viewer=PublicOnlyProxy(viewer))
    fake.statuses = {"a": {"status": "done"}}
    w.open_project(root)
    settle(w, qtbot)
    w.board.cellDoubleClicked.emit(0, 0)
    settle(w, qtbot)
    assert threads and all(t is not threading.main_thread() for t in threads)
    names = [layer.name for layer in viewer.layers]
    assert "a nuclei" in names and "a spots" in names and len(names) >= 5     # three source channels too
    labels = next(layer for layer in viewer.layers if layer.name == "a nuclei")
    assert tuple(np.round(labels.scale, 6)) == (0.4, 0.2, 0.2) and int(labels.data.max()) == 3
    assert "opened" in w.status_label.text()
    # an image without results says so
    write_manifest(root, images=[image_entry(src, "a"), image_entry(tmp_path / "b.ome.tif", "b")])
    w.reload()
    settle(w, qtbot)
    assert w.open_results("b")
    settle(w, qtbot)
    assert "no results" in w.status_label.text()


def test_an_engine_without_projects_gets_one_message(qtbot, tmp_path):
    w, fake = _dock(qtbot, FakeRunner(features=("packs",)))
    message = _engine.feature_missing_message("projects", info=fake.version(), interpreter=fake.python)
    w.new_project(tmp_path / "study", "test_pack@0.1.0")
    settle(w, qtbot)
    assert not fake.called("project_new") and w.root is None and message in w.status_label.text()
    root = _make_project(tmp_path)
    w.open_project(root)                                         # the manifest still opens, read in process
    settle(w, qtbot)
    assert w.root == root and _board(w) == [("a", "unknown"), ("b", "unknown")]
    assert message in w.status_label.text() and not w.run_all_btn.isEnabled()


def test_the_project_hub_tells_the_other_docks(qtbot, tmp_path):
    class Viewer:
        pass

    viewer = Viewer()
    hub = project_hub(viewer)
    assert project_hub(viewer) is hub and hub.root is None and not hub.running
    w, fake = _dock(qtbot, viewer=viewer)
    root = _make_project(tmp_path)
    with qtbot.waitSignal(hub.opened, timeout=10_000):
        w.open_project(root)
    settle(w, qtbot)
    assert hub.root == root and hub.state.manifest["name"] == "study"
    # another dock that wrote the project asks the Project dock to read it again
    write_manifest(root, name="renamed elsewhere", images=read_manifest(root)["images"], run=read_manifest(root)["run"])
    with qtbot.waitSignal(hub.changed, timeout=10_000):
        hub.request_reload()
    settle(w, qtbot)
    assert w.name_label.text() == "renamed elsewhere" and len(fake.called("project_status")) == 2
    fake.gate = threading.Event()
    with qtbot.waitSignal(hub.running_changed, timeout=10_000):
        w.run_all()
    assert hub.running
    fake.gate.set()
    qtbot.waitUntil(lambda: not hub.running, timeout=10_000)
    settle(w, qtbot)
    with qtbot.waitSignal(hub.closed, timeout=5_000):
        w.close_project()
    assert hub.root is None and project_hub(None) is not project_hub(None)


def test_the_new_dialog_lists_the_engines_packs_and_makes_the_project(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    fake.example = tmp_path / "example.ome.tif"
    dlg = w.new_dialog()
    settle(w, qtbot)
    assert fake.called("list_packs") and dlg.picker.current_pack_id() == "test_pack"
    dlg.picker.select("test_pack", "with_model")
    settle(w, qtbot)
    assert fake.called("check")[-1]["recipe"] == "with_model"
    dlg.get_example()
    settle(w, qtbot)
    assert dlg.images() == [str(fake.example)]
    dlg.set_folder(tmp_path / "codeless")
    dlg.name_edit.setText("codeless study")
    assert dlg.ok_button().isEnabled()
    dlg.accept()
    settle(w, qtbot)
    (new,) = fake.called("project_new")
    assert (new["pack"], new["recipe"], new["name"]) == ("test_pack@0.1.0", "with_model", "codeless study")
    assert [str(p) for p in new["images"]] == [str(fake.example)] and w.root == tmp_path / "codeless"


def _stand_in_sheet(written: list) -> SimpleNamespace:
    """germquant.io.sample_sheet as far as the dock uses it (write_sample_sheet: ASCII, LF, the rows as given, every
    column), for the dock's flow on an engine without P5-SHEET; the engine's own writer runs where it exists."""
    def write_sample_sheet(path, rows):
        rows = [dict(r) for r in rows]
        written.append(rows)
        cols = list(rows[0])
        text = ",".join(cols) + "\n" + "".join(",".join(str(r[c]) for c in cols) + "\n" for r in rows)
        Path(path).write_bytes(text.encode("ascii"))
        return Path(path)

    return SimpleNamespace(write_sample_sheet=write_sample_sheet)


def test_the_samples_tab_prefills_edits_and_saves_through_the_engines_writer(qtbot, tmp_path, monkeypatch):
    engine_sheet = _project.sheet_module()
    written: list = []
    monkeypatch.setattr(_project, "sheet_module", lambda: _stand_in_sheet(written))
    w, fake = _dock(qtbot)
    worm = "20251105_n2_nohs_HERM_001"
    root = tmp_path / "study"
    write_manifest(root, images=[image_entry(tmp_path / f"{worm}.ome.tif", worm), image_entry(tmp_path / "f.nd2", "f")])
    w.open_project(root)
    settle(w, qtbot)
    assert w.samples is not None and w.samples.ids == [worm, "f"]
    assert _cells(w.samples_table, 0) == [worm, "f"] and w.state.sheet_problem is None
    w.add_sample_column("lane")
    w.set_sample("f", "genotype", "CB4856")
    w.set_sample("f", "lane", "2")
    assert w.unsaved and w.samples.dirty
    with pytest.raises(ValueError, match="ASCII"):
        w.set_sample("f", "genotype", "caf" + chr(0xE9))
    w.save()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    (rows,) = written
    assert [r["image_id"] for r in rows] == [worm, "f"] and rows[1]["genotype"] == "CB4856" and rows[1]["lane"] == "2"
    edit = fake.called("project_edit")[-1]
    assert edit["samples"] == "samples.csv" and edit["plugin_version"] == plugin_version()
    assert read_manifest(root)["samples"] == "samples.csv" and not w.unsaved and not w.samples.dirty
    assert w.samples.row("f")["lane"] == "2"                     # the sheet read back from the project
    # the engine's own writer (P5-SHEET): ASCII with LF line endings
    need_features("sample_sheet")
    monkeypatch.setattr(_project, "sheet_module", lambda: engine_sheet)
    w.reload()
    settle(w, qtbot)
    w.set_sample("f", "genotype", "JK574")
    w.save()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    data = (root / "samples.csv").read_bytes()
    assert data.decode("ascii") and b"\r\n" not in data and b"JK574" in data and b"lane" in data
    assert not w.unsaved


def test_a_sheet_keyed_by_file_keeps_its_key_when_the_dock_saves_it(qtbot, tmp_path):
    """reviews/P6_package.md N2 through the dock, with the engine's own reader and writer: the Samples tab shows a
    sheet keyed by file (written on Windows: a backslash between folders) as runs read it, its key column is never
    edited, and Save writes it keyed by file with its cells as they were, then records it (`project edit
    --samples`, D39). A sheet the engine refuses is named in the tab and never written over."""
    need_features("sample_sheet")
    w, fake = _dock(qtbot)
    root = tmp_path / "study"
    write_manifest(root, images=[image_entry("sub/x.ome.tif", "x"), image_entry(tmp_path / "y.nd2", "y")],
                   run={"device": "auto", "runner": "subprocess", "interpreters": {"basic": {"python": sys.executable}}})
    (root / "samples.csv").write_bytes(b"file,genotype,lane\r\nsub\\x.ome.tif,CB4856,3\r\ny.nd2,N2,4\r\n")
    w.open_project(root)
    settle(w, qtbot)
    assert w.state.sheet_problem is None and w.samples.row("x")["genotype"] == "CB4856"
    w.set_sample("y", "genotype", "JK574")
    assert w.save()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert (root / "samples.csv").read_bytes() == (b"file,genotype,sex,germ_cell,treatment,replicate,acquisition_date,"
                                                   b"lane\nsub\\x.ome.tif,CB4856,,,,,,3\ny.nd2,JK574,,,,,,4\n")
    assert fake.called("project_edit")[-1]["samples"] == "samples.csv" and read_manifest(root)["samples"] == "samples.csv"
    assert _cells(w.samples_table, 1) == ["sub\\x.ome.tif", "y.nd2"] and not w.samples.dirty
    with pytest.raises(ValueError, match="file"):
        w.set_sample("x", "file", "x.ome.tif")
    w.samples_table.item(0, 1).setText("x.ome.tif")            # typed into the table: the cell goes back
    assert _cells(w.samples_table, 1)[0] == "sub\\x.ome.tif" and "sheet's key" in w.status_label.text()
    assert not w.samples.dirty
    # a sheet the engine refuses (two rows name y): the tab says why, and Save never writes over it
    bad = b"file,genotype\r\ny.nd2,N2\r\nsub\\y.nd2,CB4856\r\n"
    (root / "samples.csv").write_bytes(bad)
    w.reload()
    settle(w, qtbot)
    assert w.samples is None and "twice" in w.samples_note.text() and not w.save_sheet_btn.isEnabled()
    w.save(sheet=True)
    settle(w, qtbot)
    assert (root / "samples.csv").read_bytes() == bad


def test_device_and_interpreter_records_that_differ_from_this_computer_are_shown(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path, run={"device": "cuda", "runner": "inprocess",
                                   "interpreters": {"basic": {"python": "D:/elsewhere/python.exe"}}})
    w.open_project(root)
    settle(w, qtbot)
    notes = w.warnings_label.text()
    assert "records device cuda" in notes and "this computer asks for auto" in notes
    assert "D:/elsewhere/python.exe" in notes and "subprocess" in notes
    assert w.unsaved                                             # Save records this computer's choices
    w.save()
    settle(w, qtbot)
    edit = fake.called("project_edit")[-1]
    assert (edit["device"], edit["runner"], edit["interpreters"]) == ("auto", "subprocess", {"basic": fake.python})
    assert "records device" not in w.warnings_label.text()


def _status_calls(fake: FakeRunner) -> int:
    return len(fake.called("project_status"))


def test_save_then_close_leaves_the_project_closed(qtbot, tmp_path):
    """A write that comes back after Close does not open the project again: the save is made all the same, and its
    message says the project was closed meanwhile."""
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    w.set_xy_stride(2)
    fake.status_gate = threading.Event()                         # the save's reading waits until the test lets it go
    assert w.save()
    qtbot.waitUntil(lambda: _status_calls(fake) == 2, timeout=10_000)
    assert w.close_project() and w.root is None and w.hub.root is None
    fake.status_gate.set()
    settle(w, qtbot)
    assert w.root is None and w.rows == [] and w.hub.root is None and w.hub.state is None
    assert read_manifest(root)["run"]["xy_stride"] == 2           # written before the project was closed
    assert "saved" in w.status_label.text() and "closed meanwhile" in w.status_label.text()


def test_close_right_after_a_run_leaves_the_project_closed(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    fake.script = [{"event": "image_done", "image_id": "a"}]
    fake.status_gate = threading.Event()                         # the end-of-run reading waits
    w.run_all()
    qtbot.waitUntil(lambda: not w.running and _status_calls(fake) == 2, timeout=10_000)
    assert w.close_project() and w.root is None
    fake.status_gate.set()
    settle(w, qtbot)
    assert w.root is None and w.rows == [] and w.board.rowCount() == 0


def test_the_settings_controls_are_off_while_a_write_is_in_flight(qtbot, tmp_path):
    """A write in flight counts like a run for the Run tab's settings and the sample table, and a setting changed
    meanwhile (through the controls' programmatic twin) is kept as unsaved, never set back by the write's reading."""
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    controls = (w.stride_spin, w.samples_table, w.save_btn, w.stage_checks["spots"], w.device_combo,
                w.interpreter_btn)
    assert not w.writing and all(c.isEnabled() for c in controls)
    # a Save waiting behind a reading in flight: the controls are off from the click on
    fake.status_gate, fake.edit_gate = threading.Event(), threading.Event()
    w.reload()
    qtbot.waitUntil(lambda: _status_calls(fake) == 2, timeout=10_000)
    assert not w.writing and all(c.isEnabled() for c in controls)
    w.set_xy_stride(2)
    assert w.save() and w.writing and not any(c.isEnabled() for c in controls) and w.close_btn.isEnabled()
    # the Save in flight (its project edit held)
    fake.status_gate.set()
    qtbot.waitUntil(lambda: len(fake.called("project_edit")) == 1, timeout=10_000)
    assert w.writing and not any(c.isEnabled() for c in controls)
    with pytest.raises(ValueError, match="in progress"):
        w.set_sample("a", "genotype", "N2")
    w.set_xy_stride(3)
    fake.edit_gate.set()
    settle(w, qtbot)
    assert not w.writing and all(c.isEnabled() for c in controls)
    assert read_manifest(root)["run"]["xy_stride"] == 2 and w.stride_spin.value() == 3 and w.unsaved


def test_a_young_or_unreadable_lock_is_never_removed_from_the_dock(qtbot, tmp_path):
    """Remove stale lock is for a lock older than 12 h with its holder's host and pid (D18); a lock file the
    engine cannot read is removed by hand, and the dock says where it is."""
    w, fake = _dock(qtbot)
    fake.lock = {"host": "LABPC", "pid": 4242, "stale": False}
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    assert "LABPC" in w.lock_label.text() and "a command is writing" in w.lock_label.text()
    assert not w.unlock_btn.isEnabled() and not w.remove_stale_lock()
    settle(w, qtbot)
    assert not fake.called("project_unlock")
    # the engine's lock_report of a lock file it cannot read: held, without a host or pid
    fake.lock = {"host": None, "pid": None, "since": None, "stale": True}
    w.reload()
    settle(w, qtbot)
    assert "cannot be read" in w.lock_label.text() and "None" not in w.lock_label.text()
    assert str(root / "germquant_project.lock") in w.lock_label.text() and not w.unlock_btn.isEnabled()
    assert not w.remove_stale_lock()
    settle(w, qtbot)
    assert not fake.called("project_unlock") and "by hand" in w.status_label.text()


def test_cancel_before_a_run_never_kills_the_settings_write(qtbot, tmp_path):
    """The settings a run records first are written to their end (a write killed while it holds the project's
    lock would leave a lock no one removes for 12 h); Cancel then stops the run before it starts."""
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    w.set_xy_stride(2)
    fake.edit_gate = threading.Event()
    w.run_all()
    qtbot.waitUntil(lambda: len(fake.called("project_edit")) == 1, timeout=10_000)
    assert w.cancel()
    qtbot.wait(300)                         # long enough for a write that polls Cancel to be ended (20 ms polls)
    fake.edit_gate.set()
    qtbot.waitUntil(lambda: not w.running, timeout=10_000)
    settle(w, qtbot)
    assert read_manifest(root)["run"]["xy_stride"] == 2 and not fake.called("project_run")
    assert isinstance(w.last_error, RunCancelled) and "before it started" in w.run_label.text()
    assert not _project.run_in_progress(root) and not w.unsaved


def test_two_germquant_checkouts_with_different_hashes_are_named_in_red(qtbot, tmp_path, monkeypatch):
    """plugin_specs 3: the config_hash this napari environment's germquant computes is checked against the
    engine's, and a difference (two checkouts) shows red with the fix."""
    monkeypatch.setattr(_project, "own_config_hash", lambda root: "ffffffffffff")
    w, fake = _dock(qtbot)
    w.open_project(_make_project(tmp_path))
    settle(w, qtbot)
    assert "ffffffffffff" in w.hash_warn.text() and BASE_HASH in w.hash_warn.text()
    assert "same commit" in w.hash_warn.text() and w.hash_warn.styleSheet() == PW.css(PW.RED)
    assert "ffffffffffff" not in w.warnings_label.text()
    monkeypatch.setattr(_project, "own_config_hash", lambda root: BASE_HASH)
    w.reload()
    settle(w, qtbot)
    assert w.hash_warn.text() == ""


def test_a_failing_sheet_check_never_stops_a_reading(qtbot, tmp_path, monkeypatch):
    """The sheet check of every reading is only a hint for the Samples tab: an engine error there leaves the table
    editable (Save says why when it is tried) and the project opens."""
    monkeypatch.setattr(_project, "sheet_module", lambda: _stand_in_sheet([]))
    w, fake = _dock(qtbot)
    real = fake.require

    def require(*features, should_stop=None):
        if "sample_sheet" in features:
            raise RunnerError("the engine's version check failed")
        return real(*features, should_stop=should_stop)

    monkeypatch.setattr(fake, "require", require)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    assert w.root == root and w.last_error is None and w.state.sheet_problem is None


def test_sample_edits_need_a_sheet_writer_and_never_block_a_run(qtbot, tmp_path, monkeypatch):
    """Without germquant's sample sheet writer the table is read-only and says why; edits made before the writer
    went away never stop a run, which then uses the sheet on disk and says so."""
    monkeypatch.setattr(_project, "sheet_module", lambda: None)
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path)
    w.open_project(root)
    settle(w, qtbot)
    assert w.state.sheet_problem and "sample sheet" in w.samples_note.text() and "read-only" in w.samples_note.text()
    assert not (w.samples_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
    assert not w.save_sheet_btn.isEnabled() and not w.add_column_btn.isEnabled()
    with pytest.raises(ValueError, match="sample sheet"):
        w.set_sample("a", "genotype", "N2")
    with pytest.raises(ValueError, match="sample sheet"):
        w.add_sample_column("lane")
    written: list = []
    monkeypatch.setattr(_project, "sheet_module", lambda: _stand_in_sheet(written))
    w.reload()
    settle(w, qtbot)
    w.set_sample("a", "genotype", "N2")
    monkeypatch.setattr(_project, "sheet_module", lambda: None)  # another germquant now: no writer
    w.reload()
    settle(w, qtbot)
    assert w.samples.dirty and w.state.sheet_problem
    fake.script = [{"event": "image_done", "image_id": "a"}]
    w.run_all()
    qtbot.waitUntil(lambda: not w.running, timeout=10_000)
    settle(w, qtbot)
    assert w.last_result["state"] == "done" and not written and fake.called("project_run")
    assert "sample edits were not written" in w.run_label.text() and w.samples.dirty
    # the engine loses the sample sheet between the reading and the write (another engine chosen): Save writes the
    # settings all the same and says the sheet was not written; the edits stay for a later Save
    monkeypatch.setattr(_project, "sheet_module", lambda: _stand_in_sheet(written))
    w.reload()
    settle(w, qtbot)
    assert w.state.sheet_problem is None and w.samples.dirty
    fake._features = frozenset(f for f in _engine.FEATURES if f != "sample_sheet")
    w.set_xy_stride(2)
    assert w.save()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert read_manifest(root)["run"]["xy_stride"] == 2 and not written and w.samples.dirty
    assert "the sample sheet was not written" in w.status_label.text() and w.state.sheet_problem


def test_a_new_project_opens_when_this_computers_settings_cannot_be_recorded(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    fake.refuse["project_edit"] = "the project is locked by LABPC pid 4242"
    root = tmp_path / "study"
    w.new_project(root, "test_pack@0.1.0", recipe="basic", images=[tmp_path / "a.ome.tif"])
    settle(w, qtbot)
    assert fake.called("project_new") and fake.called("project_edit") and w.root == root
    assert "not recorded" in w.status_label.text() and w.status_label.styleSheet() == PW.css(PW.AMBER)
    assert w.unsaved                                             # Save records them


def test_a_time_series_result_is_named_as_such(qtbot, tmp_path):
    w, fake = _dock(qtbot)
    root = _make_project(tmp_path, ids=("movie",))
    (root / "results" / "movie").mkdir(parents=True)
    (root / "results" / "movie" / "movie__series.json").write_text("{}", encoding="ascii")
    fake.statuses = {"movie": {"status": "done"}}
    w.open_project(root)
    settle(w, qtbot)
    assert w.open_results("movie")
    settle(w, qtbot)
    assert "time series" in w.status_label.text() and "Time-lapse" in w.status_label.text()


def test_a_relative_project_folder_is_kept_absolute(qtbot, tmp_path, monkeypatch):
    w, fake = _dock(qtbot)
    _make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    here = Path(os.path.abspath("study"))
    w.open_project("study")
    settle(w, qtbot)
    assert w.root == here and w.root.is_absolute() and _settings.recent_projects()[0] == str(here)
    w.close_project()
    w.new_project("fresh", "test_pack@0.1.0", recipe="basic")
    settle(w, qtbot)
    assert fake.called("project_new")[0]["root"] == Path(os.path.abspath("fresh")) == w.root


# ---------------------------------------------------------------------------------------------------
# the stand-in engine: the real subprocess runner tails the engine's own log
# ---------------------------------------------------------------------------------------------------


def test_a_stand_in_engine_run_updates_the_board_row_by_row_from_its_log(qtbot, tmp_path, monkeypatch):
    own = plugin_engine_info()
    version = _engine.MIN_ENGINE if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    eng = standin_engine(tmp_path / "eng", version=version, git_sha=str(own["git_sha"]), features=list(_engine.FEATURES))
    _settings.set_engine_choice(sys.executable)
    root = _make_project(tmp_path)
    monkeypatch.setenv("FAKE_ENGINE_STDOUT", json.dumps(status_record(root)))
    w = ProjectWidget(None, session=EngineSession(env={"PYTHONPATH": str(eng)}))
    qtbot.addWidget(w)
    w.open_project(root)
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert _board(w) == [("a", "missing"), ("b", "missing")]
    monkeypatch.setenv("FAKE_PROJECT_RUN", run_spec(ids=["a", "b"], exit=1, steps=[
        {"event": "image_done", "image_id": "a", "sleep": 0.3},
        {"event": "image_failed", "image_id": "b", "sleep": 4.0, "error": "RuntimeError: the stand-in failed"}]))
    w.run_all()
    qtbot.waitUntil(lambda: _board(w) == [("a", "done"), ("b", "waiting")], timeout=60_000)   # a, while b runs
    monkeypatch.setenv("FAKE_ENGINE_STDOUT", json.dumps(status_record(root, statuses={
        "a": {"status": "done"}, "b": {"status": "failed", "error": "RuntimeError: the stand-in failed"}})))
    qtbot.waitUntil(lambda: not w.running, timeout=60_000)
    settle(w, qtbot)
    res = w.last_result
    assert res["state"] == "failed" and res["exit_code"] == 1 and res["counts"] == {"done": 1, "failed": 1,
                                                                                   "skipped": 0}
    assert _board(w) == [("a", "done"), ("b", "failed")]
    start = json.loads((root / "runs" / f"{res['run_id']}.jsonl").read_text(encoding="ascii").splitlines()[0])
    assert start["via"] == "gui-subprocess" and start["run_id"] == res["run_id"]
    assert "--device" in start["argv"] and "--run-id" in start["argv"]


# ---------------------------------------------------------------------------------------------------
# acceptance 2 and 3, against the real engine
# ---------------------------------------------------------------------------------------------------


def _tables(out: Path, image_id: str) -> dict:
    tables = {f.name: pd.read_csv(f, low_memory=False, float_precision="round_trip")
              for f in sorted(out.glob(f"{image_id}__*.csv"))}
    tables.update({f.name: pd.read_parquet(f) for f in sorted(out.glob(f"{image_id}__*.parquet"))})
    return {name: df.drop(columns=["run_timestamp"], errors="ignore") for name, df in tables.items()}


def _same_results(a: Path, b: Path, image_id: str) -> None:
    """Section 9 of the design: the same file list, cell-identical tables (run_timestamp aside), done markers
    equal in every key but finished."""
    from germquant import provenance

    assert sorted(f.name for f in a.iterdir()) == sorted(f.name for f in b.iterdir())
    ta, tb = _tables(a, image_id), _tables(b, image_id)
    assert ta.keys() == tb.keys() and len(ta) >= 5
    for name in ta:
        pd.testing.assert_frame_equal(ta[name], tb[name], check_exact=True, obj=name)
    ma, mb = provenance.read_done_marker(a, image_id), provenance.read_done_marker(b, image_id)
    assert set(ma) == set(mb) and "finished" in ma
    for key in ma:
        if key != "finished":
            assert ma[key] == mb[key], key


def _engine_dock(qtbot, runner: str, viewer=None) -> ProjectWidget:
    w = ProjectWidget(viewer)
    qtbot.addWidget(w)
    w.set_engine(sys.executable if runner == "subprocess" else INPROCESS)
    settle(w, qtbot)
    assert w.engine_error is None, w.engine_error
    return w


def _run_all(w: ProjectWidget, qtbot) -> dict:
    w.run_all()
    qtbot.waitUntil(lambda: not w.running, timeout=600_000)
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    return w.last_result


@pytest.mark.parametrize("runner", ["inprocess", "subprocess"])
def test_run_all_writes_the_same_files_as_project_run(qtbot, project_env, runner):
    """Acceptance 2 (and design section 9, routes 1, 2 and 4): a project made in the dock, copied before any run;
    Run all in the dock through the runner under test, `germquant project run` on the copy; the same files. The
    run log of the dock's run carries its run id and via, lies under the project's runs/, and the plugin writes
    no log of its own."""
    need_features("projects")
    from germquant import cli

    w = _engine_dock(qtbot, runner)
    gui = project_env.new_dir("gui")
    w.new_project(gui, "test_pack@0.1.0", recipe="basic", images=[project_env.image])
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    headless = project_env.new_dir("cli")
    shutil.copytree(gui, headless)
    res = _run_all(w, qtbot)
    image_id = "test_pack_synthetic"
    assert res["state"] == "done" and res["exit_code"] == 0 and _board(w) == [(image_id, "done")]
    assert cli.main(["project", "run", str(headless)]) == 0
    _same_results(gui / "results" / image_id, headless / "results" / image_id, image_id)
    logs = sorted((gui / "runs").iterdir())
    assert [p.name for p in logs] == [f"{res['run_id']}.jsonl"]
    start = json.loads(logs[0].read_text(encoding="ascii").splitlines()[0])
    assert start["event"] == "run_start" and start["run_id"] == res["run_id"]
    assert start["via"] == ("gui-subprocess" if runner == "subprocess" else "gui-inprocess")
    assert not (project_env.root / "runs").exists() and not user_runs_dir().exists()
    # Collect tables: the engine's record in words
    w.collect()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert (gui / "results" / "batch_summary.csv").is_file() and "batch_summary.csv" in w.status_label.text()
    assert "collected 1 image(s)" in w.status_label.text() and "{" not in w.status_label.text()


def test_codeless_project_from_the_test_pack(make_napari_viewer, qtbot, served_model_pack, project_env):
    """Acceptance 3 (with unit P6-RECIPE, whose parameter form makes the same edit): New from served_model_pack's
    basic in the New dialog, its card saying the model is missing until Get models fetches it; Get example; Run
    all; the labels and points as layers; an edit of the declared tunable spots.spot_radius_um, which changes the
    hash and shows validated: false; Save, Close, Open; then the dock's run and `germquant project run` on a copy
    give the same files."""
    need_features("projects")
    from germquant import cli

    viewer = make_napari_viewer()
    w = _engine_dock(qtbot, "subprocess", viewer=viewer)
    dlg = w.new_dialog()
    settle(w, qtbot)
    dlg.picker.select(served_model_pack.id, served_model_pack.recipe)
    settle(w, qtbot)
    assert dlg.recipe_usable() is False and "missing" in dlg.picker.card.text().lower()
    dlg.get_models()                                             # the served 64-byte stand-in
    settle(w, qtbot)
    assert dlg.recipe_usable() is True, dlg.status_text()
    dlg.get_example()
    settle(w, qtbot)
    (example,) = dlg.images()
    root = project_env.new_dir()
    dlg.set_folder(root)
    dlg.accept()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    image_id = w.rows[0].image_id
    assert w.manifest["pack"]["id"] == served_model_pack.id and Path(example).is_file()
    assert _run_all(w, qtbot)["state"] == "done"
    assert w.open_results(image_id)
    settle(w, qtbot)
    kinds = {type(layer).__name__ for layer in viewer.layers}
    assert {"Labels", "Points"} <= kinds
    before = w.status["config_hash"]
    tab = w.extra_tabs["Recipe"]                    # the parameter form (P6-RECIPE), where a user makes the edit
    qtbot.waitUntil(lambda: tab.idle, timeout=120_000)
    tab.set_value(served_model_pack.tunable, 0.4)
    assert tab.apply_form()
    qtbot.waitUntil(lambda: tab.idle, timeout=120_000)
    settle(w, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    assert w.status["config_hash"] == tab.config_hash
    assert tab.view.tunable(served_model_pack.tunable).value == pytest.approx(0.4)   # recipe.yaml's, read again
    assert w.last_error is None, w.last_error
    assert w.status["config_hash"] != before and w.badge_label.text() == "validated: false (edited)"
    assert _board(w) == [(image_id, "stale")] and w.rows[0].reason_code == "config_hash"
    w.save()
    settle(w, qtbot)
    w.close_project()
    w.open_project(root)
    settle(w, qtbot)
    assert w.manifest["recipe"]["file"] == "recipe.yaml" and served_model_pack.tunable.split(".")[-1] in \
        (root / "recipe.yaml").read_text(encoding="ascii")
    copy = project_env.new_dir("copy")
    shutil.copytree(root, copy)
    assert _run_all(w, qtbot)["state"] == "done" and _board(w) == [(image_id, "done")]
    assert cli.main(["project", "run", str(copy)]) == 0
    _same_results(root / "results" / image_id, copy / "results" / image_id, image_id)


def test_a_sheet_row_saved_in_the_dock_makes_only_its_image_stale(qtbot, project_env):
    """plugin_specs 4.1: Save writes samples.csv, and the images whose row changed then show stale (sample row),
    the others stay done (D21: row_sha256 per image). Through the in-process runner, on a project the dock made."""
    need_features("projects", "sample_sheet")
    w = _engine_dock(qtbot, "inprocess")
    second = project_env.images / "second_field.ome.tif"
    shutil.copyfile(project_env.image, second)
    root = project_env.new_dir()
    w.new_project(root, "test_pack@0.1.0", recipe="basic", images=[project_env.image, second])
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    a, b = (r.image_id for r in w.rows)
    for iid in (a, b):
        w.set_sample(iid, "genotype", "N2")
    assert w.save(sheet=True)
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert read_manifest(root)["samples"] == "samples.csv" and not w.unsaved
    assert _run_all(w, qtbot)["state"] == "done" and _board(w) == [(a, "done"), (b, "done")]
    w.set_sample(b, "genotype", "CB4856")
    assert w.save(sheet=True)
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert _board(w) == [(a, "done"), (b, "stale")] and w.rows[1].reason_code == "sample_row"
