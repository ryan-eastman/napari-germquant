"""The Project dock's model (napari_germquant._project, no Qt) and acceptance 1 of the tranche-5 Train B plugin
work: New, Save, Open round trip keeps the manifest, and the status board equals api.project_status, through
both runners (PLAN section 5.3; plugin_specs 4.1 and 6.2).

The model tests need no engine feature: they read project folders written as the engine writes them
(tests/_p6_project.py). The acceptance tests run the real engine and skip, naming the feature, until it has
projects (conftest.need_features)."""
from __future__ import annotations

import shutil
import sys

import pytest
from _p6_project import (
    BASE_HASH,
    image_entry,
    read_manifest,
    status_record,
    write_manifest,
    write_results,
)
from conftest import need_features, settle

from napari_germquant import _project as P
from napari_germquant._runner import plugin_version
from napari_germquant._settings import INPROCESS

RID = "20261024T120000Z_0123abcd"
RID2 = "20261024T130000Z_89abcdef"
# the worm pack's file name rule (config/config.yaml of the engine, metadata.filename_regex)
WORM_REGEX = (r"(?P<date>\d{6,8})_(?P<genotype>[A-Za-z0-9]+)_(?:[a-z0-9]+_)*?(?P<treatment>nohs|hs)_"
              r"(?:[a-z0-9]+_)*?(?P<sex>HERM|MALE|H|M)\s*_?(?P<replicate>\d+)?\s*$")
WORM_DEFAULTS = {"genotype": "unknown", "sex": "unknown", "treatment": "unknown", "replicate": "NA"}


# ---------------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------------


def test_a_project_folder_is_read_in_process_and_refused_when_it_is_none(tmp_path):
    with pytest.raises(P.ProjectFileError, match="not a germquant project"):
        P.read_manifest(tmp_path)
    assert not P.is_project(tmp_path)
    (tmp_path / P.MANIFEST_NAME).write_text("format: 2\nname: x\n", encoding="ascii")
    with pytest.raises(P.ProjectFileError, match="format 2"):
        P.read_manifest(tmp_path)
    (tmp_path / P.MANIFEST_NAME).write_text("format: 1\nname: [x\n", encoding="ascii")
    with pytest.raises(P.ProjectFileError, match="germquant_project.yaml"):
        P.read_manifest(tmp_path)
    root = tmp_path / "study"
    write_manifest(root, images=[image_entry("images/a.ome.tif"), image_entry(tmp_path / "b.nd2")])
    m = P.read_manifest(root)
    assert P.is_project(root) and m["format"] == 1 and [e["id"] for e in m["images"]] == ["a", "b"]
    # a path inside the project is relative (POSIX), any other absolute; both resolve
    assert P.resolve_image(root, "images/a.ome.tif") == root / "images" / "a.ome.tif"
    assert P.resolve_image(root, str(tmp_path / "b.nd2")) == tmp_path / "b.nd2"
    assert P.object_kind(root, m) == "nucleus"


def test_the_board_follows_the_status_record_and_then_the_run_log(tmp_path):
    root = tmp_path / "study"
    write_manifest(root, images=[image_entry(tmp_path / "a.ome.tif"), image_entry(tmp_path / "b.ome.tif"),
                                 image_entry(tmp_path / "c.nd2")])
    write_results(root / "results" / "a", "a", n_nuclei=4, n_spots=9)
    st = status_record(root, statuses={
        "a": {"status": "done", "qc_pass": True, "last_run_id": RID, "device_used": "cuda"},
        "b": {"status": "stale", "reason_code": "sample_row", "reason": "its sample sheet row changed"},
        "c": {"status": "failed", "error": "RuntimeError: boom", "last_run_id": RID}})
    rows = P.board_rows(root, read_manifest(root), st)
    assert [(r.image_id, r.status) for r in rows] == [("a", "done"), ("b", "stale"), ("c", "failed")]
    a, b, c = rows
    assert (a.objects, a.spots, a.qc_pass, a.last_run_id, a.device_used) == (4, 9, True, RID, "cuda")
    assert a.status_text == "done" and a.out_dir == str(root / "results" / "a")
    assert b.reason_code == "sample_row" and b.status_text == "stale: its sample sheet row changed"
    assert (b.objects, b.spots) == (None, None)               # never run: no image_summary to count
    assert c.status_text == "failed: RuntimeError: boom"
    # a stale row without the engine's words gets the reason code's
    st["images"][1]["reason"] = None
    assert P.board_rows(root, read_manifest(root), st)[1].status_text == f"stale: {P.REASONS['sample_row']}"
    # live, from the run's own log: run_start lists the images it will run, then each image event lands
    live = P.apply_events(rows, [{"event": "run_start", "run_id": RID2, "images": [{"image_id": "b"},
                                                                                  {"image_id": "c"}]}])
    assert [r.status for r in live] == ["done", "waiting", "waiting"] and rows[1].status == "stale"
    live = P.apply_events(live, [{"event": "image_done", "run_id": RID2, "image_id": "b", "status": "done",
                                  "qc_pass": False, "segment_device_used": "cpu", "error": None}])
    assert (live[1].status, live[1].qc_pass, live[1].last_run_id, live[1].device_used) == ("done", False, RID2, "cpu")
    live = P.apply_events(live, [{"event": "image_failed", "run_id": RID2, "image_id": "c", "status": "failed",
                                  "error": "ValueError: no nuclei"}])
    assert live[2].status == "failed" and live[2].status_text == "failed: ValueError: no nuclei"
    skipped = P.apply_events(rows, [{"event": "image_skipped", "run_id": RID2, "image_id": "a", "status": "skipped"}])
    assert skipped[0].status == "done" and skipped[0].status_text == "done (skipped: already done)"
    # an event of an image the board does not hold changes nothing
    assert P.apply_events(rows, [{"event": "image_done", "image_id": "zz", "status": "done"}]) == rows
    # without a status record: the manifest's images, status unknown until the engine answers
    rows0 = P.board_rows(root, read_manifest(root), None)
    assert [(r.image_id, r.status) for r in rows0] == [("a", "unknown"), ("b", "unknown"), ("c", "unknown")]
    assert rows0[0].objects == 4                              # the results on disk are counted all the same


def test_the_header_text_names_sizes_channels_and_a_single_plane():
    plane = {"sizes": {"C": 2, "Z": 1, "Y": 40, "X": 48}, "channel_names": ["DAPI", "Cy3 RAD-51"], "n_images": 1,
             "spacing": [1.0, 1.0, 1.0], "spacing_ok": False, "spacing_yx_um": [0.1, 0.1], "single_plane": True}
    text = P.header_text(plane)
    assert text == "C=2 Z=1 Y=40 X=48; single plane, voxel um dy=0.100 dx=0.100; channels: DAPI, Cy3 RAD-51"
    stack = {"sizes": {"C": 3, "Z": 10, "Y": 64, "X": 64}, "channel_names": ["405", "477", "545"],
             "spacing": [0.4, 0.2, 0.2], "spacing_ok": True, "single_plane": False}
    assert P.header_text(stack) == "C=3 Z=10 Y=64 X=64; voxel um dz=0.400 dy=0.200 dx=0.200; channels: 405, 477, 545"
    assert P.header_text(None) == "header not read yet"
    assert P.header_text({"error": "cannot read the header: boom"}) == "cannot read the header: boom"


def test_the_badge_lock_and_settings_say_what_the_project_records(tmp_path):
    root = tmp_path / "study"
    write_manifest(root, images=[image_entry(tmp_path / "a.ome.tif")])
    (root / "packs" / "test_pack" / "0.1.0" / "pack.yaml").write_text(
        "format: 1\nid: test_pack\nversion: 0.1.0\nobject_kind: cell\nrecipes:\n  - {id: basic, status: provisional}\n",
        encoding="ascii")
    m = read_manifest(root)
    assert P.object_kind(root, m) == "cell"
    st = status_record(root, m)
    assert P.badge(st, m, root=root) == ("not validated (provisional pack)", "warn")
    st["edited"], st["validated"] = True, {"value": False, "why": "edited: the stage switches", "record": None}
    assert P.badge(st, m, root=root) == ("validated: false (edited)", "edited")
    st["edited"], st["validated"] = False, {"value": True, "why": "validated", "record": "validation/basic.json"}
    assert P.badge(st, m, root=root) == ("validated (record validation/basic.json)", "ok")
    assert P.badge(None, m, root=root) == ("not validated (provisional pack)", "warn")   # from the manifest alone
    held = {"held": True, "host": "LABPC", "pid": 4242, "since": "2026-10-20T08:00:00+00:00", "stale": True}
    text = P.lock_text(held)
    assert "LABPC" in text and "4242" in text and "older than 12 h" in text
    assert "a command is writing" in P.lock_text({**held, "stale": False})
    assert P.lock_text({"held": False}) is None and P.lock_text(None) is None
    # Save writes only what differs from the manifest (the run settings and the stage switches of design 7.2)
    py, py2 = sys.executable, str(tmp_path / "other" / "python.exe")
    write_manifest(root, images=m["images"], run={"device": "cpu", "runner": "subprocess", "xy_stride": 1,
                                                  "interpreters": {"basic": {"python": py}}})
    m = read_manifest(root)
    assert P.recorded_settings(m) == {"device": "cpu", "runner": "subprocess", "xy_stride": 1, "disabled": (),
                                      "interpreters": {"basic": py}}
    same = dict(device="cpu", runner="subprocess", xy_stride=1, disabled=(), interpreter=("basic", py))
    assert P.settings_changes(m, **same) == {}
    assert P.settings_changes(m, device="auto", runner="inprocess", xy_stride=4, disabled=("spots",),
                              interpreter=("basic", py2)) == \
        {"device": "auto", "runner": "inprocess", "xy_stride": 4, "disable": ["spots"], "interpreters": {"basic": py2}}
    write_manifest(root, images=m["images"], run=m["run"], overrides={"spots.enabled": False, "axis.enabled": False})
    m = read_manifest(root)
    assert P.recorded_settings(m)["disabled"] == ("axis", "spots")
    # a stage the project turned off and the dock turns back on is an --unset of its key
    assert P.settings_changes(m, **{**same, "disabled": ("axis",)}) == {"unset": ["spots.enabled"]}
    assert P.settings_changes(m, **{**same, "disabled": ("axis", "spots", "germline")}) == {"disable": ["germline"]}


def test_the_sample_table_prefills_from_file_names_and_keeps_extra_columns(tmp_path):
    root = tmp_path / "study"
    worm, field = "20251105_n2_nohs_HERM_001", "field_07"
    write_manifest(root, images=[image_entry(tmp_path / f"{worm}.ome.tif", worm), image_entry(tmp_path / f"{field}.nd2",
                                                                                            field)])
    m = read_manifest(root)
    t = P.SampleTable.from_project(root, m, regex=WORM_REGEX, defaults=WORM_DEFAULTS)
    assert t.columns == list(P.DESIGN_COLUMNS) and t.ids == [worm, field] and t.saved is None and not t.dirty
    assert t.row(worm) == {"image_id": worm, "genotype": "N2", "sex": "herm", "germ_cell": "oocyte",
                           "treatment": "control", "replicate": "001", "acquisition_date": "20251105"}
    assert t.row(field) == {c: (field if c == "image_id" else "") for c in P.DESIGN_COLUMNS}   # no match: blank
    assert t.not_in_sheet() == [worm, field]
    t.add_column("lane")
    t.set(field, "lane", "3")
    t.set(field, "genotype", " CB4856 ")
    assert t.dirty and t.row(field)["genotype"] == "CB4856" and t.columns[-1] == "lane"
    assert t.records()[1] == {**t.row(field), "lane": "3"}
    for bad in ("genotype", "image_id", "a,b", "", "x" * 65):
        with pytest.raises(ValueError):
            t.add_column(bad)
    with pytest.raises(ValueError, match=f"{field}.*genotype.*ASCII"):
        t.set(field, "genotype", "caf" + chr(0xE9))
    with pytest.raises(KeyError):
        t.set("nope", "genotype", "N2")
    with pytest.raises(ValueError, match="image_id"):
        t.set(field, "image_id", "other")
    # a saved sheet (Excel's UTF-8 CSV: a byte-order mark and CRLF) is what the table shows; an extra column is
    # kept, and an image the sheet leaves out is prefilled and named as not in the sheet
    (root / "samples.csv").write_bytes(b"\xef\xbb\xbfimage_id,genotype,lane\r\nfield_07,CB4856,3\r\n")
    write_manifest(root, images=m["images"], samples="samples.csv")
    t2 = P.SampleTable.from_project(root, read_manifest(root), regex=WORM_REGEX, defaults=WORM_DEFAULTS)
    assert t2.columns == [*P.DESIGN_COLUMNS, "lane"] and t2.row(field)["genotype"] == "CB4856"
    assert t2.row(field)["lane"] == "3" and t2.row(worm)["genotype"] == "N2" and t2.row(worm)["lane"] == ""
    assert t2.not_in_sheet() == [worm] and not t2.dirty and t2.changed_ids() == [worm]
    t2.set(field, "lane", "4")
    assert t2.changed_ids() == [worm, field]
    # a cp1252 sheet (Excel's plain CSV) reads too, with the warning the engine gives
    (root / "samples.csv").write_bytes("image_id,genotype\r\nfield_07,caf\xe9\r\n".encode("cp1252"))
    t3 = P.SampleTable.from_project(root, read_manifest(root), regex=WORM_REGEX, defaults=WORM_DEFAULTS)
    assert t3.row(field)["genotype"] == "caf" + chr(0xE9) and "cp1252" in " ".join(t3.warnings)


def test_a_run_in_progress_is_known_to_every_dock(tmp_path):
    root = tmp_path / "study"
    assert not P.run_in_progress(root) and P.running_run_id(root) is None
    P.mark_running(root, RID)
    try:
        assert P.run_in_progress(root) and P.run_in_progress(str(root) + "/") and P.running_run_id(root) == RID
        assert not P.run_in_progress(tmp_path / "other")
    finally:
        P.mark_finished(root)
    assert not P.run_in_progress(root)


# ---------------------------------------------------------------------------------------------------
# acceptance 1, against the real engine
# ---------------------------------------------------------------------------------------------------


def _dock_on_engine(qtbot, runner: str):
    from napari_germquant._project_widget import ProjectWidget

    w = ProjectWidget(None)
    qtbot.addWidget(w)
    w.set_engine(sys.executable if runner == "subprocess" else INPROCESS)
    settle(w, qtbot)
    assert w.engine_error is None, w.engine_error
    return w


@pytest.mark.parametrize("runner", ["inprocess", "subprocess"])
def test_new_save_open_round_trip_keeps_the_manifest(qtbot, project_env, runner):
    """New through the dock (the engine's `project new`, stamped with the plugin version), the Run tab's settings
    recorded with Save (`project edit --device --runner --interpreter --xy-stride --no-spots`), Close, Open: the
    manifest keeps every key but the saved stamp and what Save changed on purpose, and Open writes nothing."""
    need_features("projects")
    w = _dock_on_engine(qtbot, runner)
    root = project_env.new_dir()
    w.new_project(root, "test_pack@0.1.0", recipe="basic", images=[project_env.image], name="round trip")
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert w.root == root and w.manifest["name"] == "round trip"
    assert [r.image_id for r in w.rows] == ["test_pack_synthetic"] and w.rows[0].status == "missing"
    created = read_manifest(root)
    assert created["created"]["plugin"] == created["saved"]["plugin"] == plugin_version()
    assert w.status["config_hash"] == created["recipe"]["expected_config_hash"] == BASE_HASH
    w.set_device("cpu")
    w.set_xy_stride(2)
    w.set_stage("spots", False)
    assert w.unsaved
    w.save()
    settle(w, qtbot)
    assert w.last_error is None, w.last_error
    assert not w.unsaved
    saved = read_manifest(root)
    assert saved["run"]["device"] == "cpu" and saved["run"]["xy_stride"] == 2 and saved["run"]["runner"] == runner
    assert saved["run"]["interpreters"]["basic"]["python"] == sys.executable
    assert saved["recipe"]["overrides"] == {"spots.enabled": False} and saved["saved"]["plugin"] == plugin_version()
    assert saved["recipe"]["expected_config_hash"] != BASE_HASH         # a stage switch enters the hash
    changed = {"run", "recipe", "validated", "saved"}
    assert {k: v for k, v in saved.items() if k not in changed} == {k: v for k, v in created.items() if k not in changed}
    w.close_project()
    assert w.root is None and w.rows == []
    w.open_project(root)
    settle(w, qtbot)
    assert w.manifest == saved == read_manifest(root)                    # every key: Open writes nothing
    assert w.status["config_hash"] == saved["recipe"]["expected_config_hash"]
    assert w.stride_spin.value() == 2 and not w.stage_checks["spots"].isChecked() and not w.unsaved
    assert w.badge_label.text() == "validated: false (edited)"


@pytest.mark.parametrize("runner", ["inprocess", "subprocess"])
def test_status_board_equals_project_status(qtbot, project_env, runner):
    """The board's rows are api.project_status's images (status, reason code, qc_pass, last run, device, error):
    one image run, one not, then an edit that makes the run one stale."""
    need_features("projects")
    from germquant import api

    w = _dock_on_engine(qtbot, runner)
    second = project_env.images / "second_field.ome.tif"
    shutil.copyfile(project_env.image, second)
    root = project_env.new_dir()
    w.new_project(root, "test_pack@0.1.0", recipe="basic", images=[project_env.image, second])
    settle(w, qtbot)
    assert w.last_error is None, w.last_error

    def compare() -> list[dict]:
        engine = api.project_status(api.open_project(root))["images"]
        keys = ("status", "reason_code", "qc_pass", "last_run_id", "device_used", "error")
        assert [(r.image_id, *(getattr(r, k) for k in keys)) for r in w.rows] == \
            [(e["id"], *(e.get(k) for k in keys)) for e in engine]
        return engine

    assert [e["status"] for e in compare()] == ["missing", "missing"]
    w.run_selected(["test_pack_synthetic"])
    qtbot.waitUntil(lambda: not w.running, timeout=600_000)
    settle(w, qtbot)
    assert w.last_result is not None and w.last_result["state"] == "done", w.last_message
    assert [e["status"] for e in compare()] == ["done", "missing"]
    assert w.rows[0].objects is not None and w.rows[0].objects >= 1
    w.set_stage("spots", False)
    w.save()
    settle(w, qtbot)
    engine = compare()
    assert engine[0]["status"] == "stale" and engine[0]["reason_code"] == "config_hash"
