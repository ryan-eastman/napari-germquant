"""The Run dock: the roles table and stage checkboxes it builds, the plan and export, and the run itself,
which must write exactly the files `germquant run` writes with the same settings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import CHANNELS, IMAGE_ID, write_ome

from napari_germquant._engine import Recipe
from napari_germquant._run_widget import RED, RunWidget
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


def _tables(out: Path) -> dict[str, pd.DataFrame]:
    return {f.name: pd.read_csv(f, low_memory=False, float_precision="round_trip").drop(columns=["run_timestamp"])
            for f in sorted(out.glob(f"{IMAGE_ID}__*.csv"))}


def test_gui_run_writes_the_same_files_as_the_cli(make_napari_viewer, qtbot, synthetic_image, classical_overlay,
                                                  tmp_path):
    """The Run button on the synthetic gonad with the classical segmenter and spots off, against
    `germquant run` with the exported settings: same file list, cell-identical tables, same completion marker."""
    viewer = make_napari_viewer()
    w = RunWidget(viewer)
    w.set_image(str(synthetic_image))
    w.select_config_file(classical_overlay)                   # the --config route
    assert w.config is not None and w.config.get("segmentation.nuclei.method") == "classical"
    w.stage_checks["spots"].setChecked(False)                 # the --no-spots route
    out_gui, out_cli = tmp_path / "gui", tmp_path / "cli"
    w.out_edit.setText(str(out_gui))
    plan = w.plan()
    assert plan.run_argv() == ["run", str(synthetic_image), "--config", str(classical_overlay), "--out", str(out_gui),
                               "--no-spots"]

    w.run_btn.click()
    assert not w.run_btn.isEnabled()                          # a run is in progress
    qtbot.waitUntil(lambda: w.last_result is not None or w.last_error is not None, timeout=600_000)
    qtbot.waitUntil(lambda: w.run_btn.isEnabled(), timeout=30_000)
    assert w.last_error is None, w.last_error
    res = w.last_result
    assert res["image_id"] == IMAGE_ID and res["n_nuclei"] >= 1 and Path(res["out_dir"]) == out_gui
    assert "spots" not in provenance.read_done_marker(out_gui, IMAGE_ID)["enabled_stages"]
    assert "done:" in w.status_label.text() and str(res["n_nuclei"]) in w.status_label.text()
    # the results came back as layers: nuclei labels with the nuclei table as features (row i = label i)
    labels = [layer for layer in viewer.layers if layer.__class__.__name__ == "Labels"]
    assert len(labels) == 1 and labels[0].name == f"{IMAGE_ID} nuclei"
    feats = labels[0].features
    assert feats.shape[0] == int(labels[0].data.max()) + 1 and "volume_um3" in feats.columns
    nuclei = pd.read_csv(out_gui / f"{IMAGE_ID}__nuclei.csv")
    nid = int(nuclei["nucleus_id"].iloc[0])
    assert feats["volume_um3"].iloc[nid] == pytest.approx(float(nuclei["volume_um3"].iloc[0]))
    assert tuple(np.round(labels[0].scale, 6)) == (0.4, 0.2, 0.2)
    assert not [layer for layer in viewer.layers if layer.__class__.__name__ == "Points"]   # spots were off

    # the same settings through the command line
    argv = plan.run_argv()
    argv[argv.index("--out") + 1] = str(out_cli)
    assert cli.main(argv) == 0

    assert sorted(f.name for f in out_gui.iterdir()) == sorted(f.name for f in out_cli.iterdir())
    a, b = _tables(out_gui), _tables(out_cli)
    assert a.keys() == b.keys() and len(a) >= 5
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True, obj=name)
    ma = provenance.read_done_marker(out_gui, IMAGE_ID)
    mb = provenance.read_done_marker(out_cli, IMAGE_ID)
    for k in ("config_hash", "enabled_stages", "stage_hashes", "xy_stride", "z_range", "failed_stages"):
        assert ma[k] == mb[k], k
    assert a[f"{IMAGE_ID}__nuclei.csv"]["config_hash"].iloc[0] == mb["config_hash"]
    assert not ma["failed_stages"]
