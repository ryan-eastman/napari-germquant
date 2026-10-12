"""The Run dock: the roles table and stage checkboxes it builds, the plan and export, and the run itself,
which must write exactly the files `germquant run` writes with the same settings, through both engine runners
(docs/DESIGN_system_packs.md, section 9)."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import CHANNELS, IMAGE_ID, TEST_IMAGE, write_ome

from napari_germquant._engine import Recipe
from napari_germquant._run_widget import RED, RunWidget
from napari_germquant._runner import VIA_INPROCESS, VIA_SUBPROCESS, read_run_log
from napari_germquant._settings import INPROCESS
from germquant import api, cli, provenance


def _widget(viewer, image, profile="rad51_foci"):
    w = RunWidget(viewer)
    w.select_profile(profile)
    w.set_image(image)
    return w


def test_roles_table_and_stage_checks_for_the_gonad(make_napari_viewer, synthetic_image, tmp_path):
    viewer = make_napari_viewer()
    w = _widget(viewer, synthetic_image)
    assert w.recipe == Recipe.from_profile("rad51_foci") and w.config is not None
    assert "C=3" in w.image_info.text() and "dz=0.400" in w.image_info.text()
    assert w.profile_combo.count() >= 5
    # one row per role of the channel map, channels bound by name, nothing red
    assert [w.roles_table.item(i, 0).text() for i in range(w.roles_table.rowCount())] == \
        ["dna *", "central_element *", "foci", "granule"]
    assert [w.roles_table.item(i, 1).text() for i in range(3)] == [f"{i}: {c}" for i, c in enumerate(CHANNELS)]
    assert w.roles_table.item(3, 1).text() == "(none)"
    assert not any(r.problem for r in w.role_rows)
    assert all(w.roles_table.item(i, 0).background().color() != RED for i in range(w.roles_table.rowCount()))
    assert w.unassigned_label.text() == "every channel has a role"
    # the stage checkboxes come from cli_switches(), live for stages the profile has on with their roles bound
    assert list(w.stage_checks) == [s.name for s in api.cli_switches()]
    for name in ("spots", "germline", "axis", "signal_qc"):
        assert w.stage_checks[name].isEnabled() and w.stage_checks[name].isChecked(), name
    for name in ("coloc", "granule", "partition", "granule_tail", "envelope", "audit", "staging", "sc_trace",
                 "acquisition"):
        assert not w.stage_checks[name].isEnabled() and not w.stage_checks[name].isChecked(), name
    assert "off in this profile" in w.stage_checks["coloc"].text()
    assert w.disabled_stages() == ()
    w.stage_checks["spots"].setChecked(False)
    assert w.disabled_stages() == ("spots",)
    plan = w.plan()
    assert plan.image == synthetic_image and plan.out == api.default_out_dir(synthetic_image)
    assert plan.xy_stride == 1 and plan.disabled == ("spots",)
    overlay, commands = w.export_to(tmp_path / "exp")
    assert overlay.is_file() and commands.is_file() and w.last_export == (overlay, commands)
    assert "--no-spots" in commands.read_text(encoding="utf-8")
    # changing the recipe keeps the user's ticks for stages that stay live
    w.select_profile("segmentation_only")
    assert not w.stage_checks["spots"].isEnabled()           # off in that profile
    assert w.stage_checks["germline"].isChecked()


def test_one_channel_image_marks_the_required_role_red_and_greys_stages(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    p = write_ome(tmp_path / "20251105_n2_nohs_HERM_002.ome.tif", np.zeros((1, 4, 8, 8), np.uint16), ["DAPI"])
    w = _widget(viewer, p)
    by = {r.role: r for r in w.role_rows}
    assert by["dna"].index == 0 and by["central_element"].problem
    row = [r.role for r in w.role_rows].index("central_element")
    assert w.roles_table.item(row, 0).background().color() == RED
    assert w.roles_table.item(row, 1).text() == "(none)" and "UNBOUND" in w.roles_table.item(row, 3).text()
    assert not w.stage_checks["germline"].isEnabled() and "central_element" in w.stage_checks["germline"].text()
    assert not w.stage_checks["spots"].isEnabled() and "foci" in w.stage_checks["spots"].text()
    assert w.stage_checks["axis"].isEnabled()
    with pytest.raises(ValueError, match="central_element"):
        w.plan()
    w.run()                                                   # refused, no worker, message shown
    assert w.last_result is None and "central_element" in w.status_label.text()


def test_timelapse_header_blocks_the_run(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    p = write_ome(tmp_path / "series.ome.tif", np.zeros((2, 3, 2, 4, 4), np.uint16), CHANNELS, axes="TCZYX")
    w = _widget(viewer, p)
    assert "REFUSED" in w.image_info.text() and "2 time points" in w.image_info.text()
    with pytest.raises(ValueError, match="time-lapse"):
        w.plan()


def _tables(out: Path, image_id: str = IMAGE_ID) -> dict[str, pd.DataFrame]:
    """Every table of the image, CSV and parquet, without the run_timestamp column (section 9)."""
    tables = {f.name: pd.read_csv(f, low_memory=False, float_precision="round_trip")
              for f in sorted(out.glob(f"{image_id}__*.csv"))}
    tables.update({f.name: pd.read_parquet(f) for f in sorted(out.glob(f"{image_id}__*.parquet"))})
    return {name: df.drop(columns=["run_timestamp"], errors="ignore") for name, df in tables.items()}


@pytest.mark.parametrize("runner", ["inprocess", "subprocess"])
@pytest.mark.parametrize("route", ["config", "pack"])
def test_gui_run_writes_the_same_files_as_the_cli(make_napari_viewer, qtbot, synthetic_image, classical_overlay,
                                                  tmp_path, request, route, runner):
    """The Run button against `germquant run` with the exported argv, through the in-process runner and the
    subprocess runner (sys.executable as the engine interpreter, as in the plugin CI): the same file list,
    cell-identical tables (run_timestamp aside) and equal completion markers (every key except finished).
    The config route runs the synthetic gonad with the classical segmenter and spots off; the pack route runs
    the test pack's basic recipe on its synthetic example. Only the config route in process runs with an
    engine older than 0.2.0.dev0 (no packs, no `python -m germquant`)."""
    if (route == "pack" or runner == "subprocess") and importlib.util.find_spec("germquant.packs") is None:
        pytest.skip("the pack route and the subprocess runner need germquant 0.2.0.dev0 or later")
    if route == "pack":
        request.getfixturevalue("pack_env")
    viewer = make_napari_viewer()
    w = RunWidget(viewer)
    assert w.engine is None                                   # nothing chosen yet: the dock would ask
    assert w.set_engine(sys.executable if runner == "subprocess" else INPROCESS)
    assert w.engine.via == (VIA_SUBPROCESS if runner == "subprocess" else VIA_INPROCESS)
    out_gui, out_cli = tmp_path / "gui", tmp_path / "cli"
    if route == "config":
        image, image_id = synthetic_image, IMAGE_ID
        w.set_image(str(image))
        w.select_config_file(classical_overlay)               # the --config route
        assert w.config is not None and w.config.get("segmentation.nuclei.method") == "classical"
        w.stage_checks["spots"].setChecked(False)             # the --no-spots route
        recipe_args, tail = ["--config", str(classical_overlay)], ["--no-spots"]
    else:
        image = tmp_path / "image" / TEST_IMAGE.name
        image.parent.mkdir()
        shutil.copyfile(TEST_IMAGE, image)
        image_id = api.image_stem(image)
        w.select_pack("test_pack", "basic")                   # the --pack route
        w.set_image(str(image))
        assert w.route == "pack" and w.recipe == Recipe.from_pack("test_pack@0.1.0", "basic")
        assert w.pack_status["usable"], w.pack_status
        assert w.config.get("segmentation.nuclei.method") == "classical"
        recipe_args, tail = ["--pack", "test_pack@0.1.0", "--recipe", "basic"], []
    w.out_edit.setText(str(out_gui))
    plan = w.plan()
    assert plan.run_argv() == ["run", str(image), *recipe_args, "--out", str(out_gui), *tail]

    w.run_btn.click()
    assert not w.run_btn.isEnabled()                          # a run is in progress
    qtbot.waitUntil(lambda: w.last_result is not None or w.last_error is not None, timeout=600_000)
    qtbot.waitUntil(lambda: w.run_btn.isEnabled(), timeout=30_000)
    assert w.last_error is None, w.last_error
    res = w.last_result
    assert res["image_id"] == image_id and res["n_nuclei"] >= 1 and Path(res["out_dir"]) == out_gui
    assert "done:" in w.status_label.text() and str(res["n_nuclei"]) in w.status_label.text()
    # the results came back as layers: nuclei labels with the nuclei table as features (row i = label i)
    labels = [layer for layer in viewer.layers if layer.__class__.__name__ == "Labels"]
    assert len(labels) == 1 and labels[0].name == f"{image_id} nuclei"
    feats = labels[0].features
    assert feats.shape[0] == int(labels[0].data.max()) + 1 and "volume_um3" in feats.columns
    nuclei = pd.read_csv(out_gui / f"{image_id}__nuclei.csv")
    nid = int(nuclei["nucleus_id"].iloc[0])
    assert feats["volume_um3"].iloc[nid] == pytest.approx(float(nuclei["volume_um3"].iloc[0]))
    points = [layer for layer in viewer.layers if layer.__class__.__name__ == "Points"]
    if route == "config":
        assert "spots" not in provenance.read_done_marker(out_gui, image_id)["enabled_stages"]
        assert tuple(np.round(labels[0].scale, 6)) == (0.4, 0.2, 0.2)
        assert not points                                     # spots were off
    else:
        assert [p.name for p in points] == [f"{image_id} spots"] and len(points[0].data) >= 1
        assert "pack: test_pack@0.1.0   recipe: basic" in w.status_label.text()
        assert "WARNING" not in w.status_label.text()         # basic expects the classical segmenter

    # the same settings through the command line
    argv = plan.run_argv()
    argv[argv.index("--out") + 1] = str(out_cli)
    assert cli.main(argv) == 0

    assert sorted(f.name for f in out_gui.iterdir()) == sorted(f.name for f in out_cli.iterdir())
    a, b = _tables(out_gui, image_id), _tables(out_cli, image_id)
    assert a.keys() == b.keys() and len(a) >= 10
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True, obj=name)
    ma = provenance.read_done_marker(out_gui, image_id)
    mb = provenance.read_done_marker(out_cli, image_id)
    assert set(ma) == set(mb) and "finished" in ma
    for k in ma:
        if k != "finished":                                   # stamped with the time of writing
            assert ma[k] == mb[k], k
    assert a[f"{image_id}__nuclei.csv"]["config_hash"].iloc[0] == mb["config_hash"]
    assert not ma["failed_stages"]

    # the run log sits next to the output folder, never inside it (the file lists above are equal)
    log = w.last_run_log
    assert log is not None and log.parent == tmp_path / "runs" and log.suffix == ".jsonl"
    start, done, end = read_run_log(log)
    assert [start["event"], done["event"], end["event"]] == ["run_start", "image_done", "run_end"]
    assert start["via"] == w.engine.via and start["interpreter"] == w.engine.python
    assert start["config_hash"] == mb["config_hash"] == done["config_hash"]
    assert start["device_requested"] == "auto" and start["device_selected"] == "cpu"
    assert start["engine"]["germquant"] == api.__version__ and start["engine"]["git_sha"]
    assert start["images"] == [{"path": str(image), "image_id": image_id, "out_dir": str(out_gui)}]
    assert done["done_sha256"] and done["stage_hashes"] == mb["stage_hashes"] and end["counts"]["done"] == 1
