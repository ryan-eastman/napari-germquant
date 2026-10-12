"""The Run dock's system packs (docs/DESIGN_system_packs.md, sections 8.2 and 8.3; package P4 acceptance 1, 2, 4
and 5): the pack list and card through both engine runners, a recipe that cannot run greyed with its reason and
fix, Get models and Get example through the runner, results named after the pack's object kind with the
fallback banner and the Provenance box, the export with --pack, the refusals (a broken pack, an output folder
inside a pack), the pack the dock shows before the user picks one, and the validation line of the card. Every
pack root and store is private to the test (conftest.pack_env)."""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import (
    FIXTURE_PACKS,
    TEST_IMAGE,
    TEST_PACK,
    TINYNET,
    TINYNET_BYTES,
    TINYNET_SHA,
    copy_pack,
    engine_device,
    settle,
)
from qtpy.QtCore import Qt

from napari_germquant import _settings
from napari_germquant._engine import Recipe
from napari_germquant._packs import AMBER, PackEntry, card_html, validation_text
from napari_germquant._results_widget import ResultsWidget
from napari_germquant._run_widget import RunWidget
from napari_germquant._settings import INPROCESS
from germquant import api, provenance

pytestmark = pytest.mark.skipif(importlib.util.find_spec("germquant.packs") is None,
                                reason="system packs need germquant 0.2.0.dev0 or later")

BUILT_IN = {"celegans_germline", "cultured_cells", "budding_yeast"}


def _engine(runner: str) -> str:
    return sys.executable if runner == "subprocess" else INPROCESS


def _dock(make_napari_viewer, qtbot, runner: str) -> tuple[object, RunWidget]:
    viewer = make_napari_viewer()
    w = RunWidget(viewer)
    w.set_engine(_engine(runner))
    settle(w, qtbot)
    assert w.engine is not None and w.engine_ok, w.engine_box.runs_label.text()
    return viewer, w


def _image_copy(tmp_path: Path) -> Path:
    image = tmp_path / "image" / TEST_IMAGE.name
    image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEST_IMAGE, image)
    return image


def _run(w: RunWidget, qtbot) -> dict:
    w.run_btn.click()
    qtbot.waitUntil(lambda: w.last_result is not None or w.last_error is not None, timeout=600_000)
    qtbot.waitUntil(lambda: w.run_btn.isEnabled(), timeout=30_000)
    assert w.last_error is None, w.last_error
    return w.last_result


def _wait_pack_op(w: RunWidget, qtbot) -> dict:
    qtbot.waitUntil(lambda: w.last_pack_result is not None or w.last_pack_error is not None, timeout=300_000)
    qtbot.waitUntil(lambda: not w.pack_worker.busy, timeout=30_000)
    settle(w, qtbot)                                  # the card is checked again after a fetch
    assert w.last_pack_error is None, w.last_pack_error
    return w.last_pack_result


@pytest.mark.parametrize("runner", ["inprocess", "subprocess"])
def test_dock_lists_the_packs_and_shows_the_card(make_napari_viewer, qtbot, pack_env, runner):
    """Acceptance 1: the built-in packs and the one under $GERMQUANT_PACKS, through both runners; the card of
    the test pack; the bench profiles hidden until "show all" is ticked."""
    _viewer, w = _dock(make_napari_viewer, qtbot, runner)
    entries = {e.id: e for e in w.pack_entries}
    assert BUILT_IN | {"test_pack"} <= set(entries)
    assert entries["test_pack"].data["source"] == "GERMQUANT_PACKS"
    assert all(entries[i].data["source"] == "builtin" for i in BUILT_IN)
    labels = [w.pack_picker.pack_combo.itemText(i) for i in range(w.pack_picker.pack_combo.count())]
    assert any(t.startswith("test_pack 0.1.0: Test pack") for t in labels)
    assert w.route == "profile"                    # a later engine choice does not take the route away

    w.select_pack("test_pack", "basic")
    assert "checking in" in w.pack_picker.card.text() and not w.run_btn.isEnabled()    # until the check is back
    settle(w, qtbot)
    assert w.route == "pack" and w.recipe == Recipe.from_pack("test_pack@0.1.0", "basic")
    assert w.pack_status["usable"] and w.run_btn.isEnabled()
    card = w.pack_picker.card.text()
    for piece in ("Test pack: synthetic nuclei and puncta", "system: organism none (synthetic)",
                  "object kind: nucleus (one row of the nuclei table is one nucleus)", "dims 3d",
                  "version 0.1.0; engine range &gt;=0.2.0.dev0,&lt;0.3", "<b>Models</b>: none needed",
                  "segmentation backend classical: <span style='color:#3c8c50'>available</span>",
                  "spot backend log", "not validated (status provisional)", "<b>Licence</b>: pack MIT",
                  "germquant test pack (synthetic data, not citable)", "<b>Example</b>: synthetic",
                  "usable here", w.engine.describe()):
        assert piece in card, piece
    assert w.pack_picker.get_example_btn.isEnabled() and not w.pack_picker.get_models_btn.isEnabled()
    # the second recipe: its model is not in the (empty) store, so it cannot run here
    w.select_recipe("with_model")
    settle(w, qtbot)
    card = w.pack_picker.card.text()
    assert f"{TINYNET}: <span style='color:#c83c3c'>missing</span>" in card and "NOT usable here" in card
    assert "fix: germquant pack fetch-models test_pack --recipe with_model" in card
    assert w.pack_picker.get_models_btn.isEnabled() and not w.run_btn.isEnabled()
    # an outside pack shows its needs; spotmax is missing from the plugin environment (and from CI)
    w.select_pack("budding_yeast")
    settle(w, qtbot)
    card = w.pack_picker.card.text()
    assert "object kind: cell (one row of the nuclei table is one cell)" in card and "dims projected" in card
    assert "cpsam_v2@0f1cc3f7" in card and "<b>Example</b>: sbiad3750_sut509_p1, public data" in card
    if importlib.util.find_spec("spotmax") is None:
        missing = "<span style='color:#c83c3c'>missing</span>: install the &#x27;spots&#x27; extra"
        assert f"Python module spotmax: {missing}" in card and f"spot backend spotmax: {missing}" in card
        assert not w.pack_status["usable"] and not w.run_btn.isEnabled()

    # the bench profiles sit in Advanced, hidden until "show all" is ticked
    names = [w.profile_combo.itemText(i) for i in range(w.profile_combo.count())]
    assert "rad51_foci" in names and not [n for n in names if n.startswith("bench_")]
    w.show_all_check.setChecked(True)
    names = [w.profile_combo.itemText(i) for i in range(w.profile_combo.count())]
    assert {"bench_log_tuned", "bench_spotmax_tuned"} <= set(names)
    w.select_profile("rad51_foci")
    assert w.route == "profile" and w.recipe == Recipe.from_profile("rad51_foci") and w.run_btn.isEnabled()


@pytest.mark.parametrize("runner", ["subprocess", "inprocess"])
def test_unusable_recipe_get_models_and_get_example(make_napari_viewer, qtbot, served_pack, caplog, runner):
    """Acceptance 2: a recipe that cannot run is greyed with its reason and fix and Run is off; Get models
    through the runner (a 64-byte stand-in served by the test) makes it usable; Get example (two per-channel
    TIFFs served by the test) assembles the image, opens it and makes it the run's image. Through the subprocess
    runner both go to the engine interpreter (its argument lists are checked), and the example's progress lines
    reach the status bar while it is fetched."""
    caplog.set_level(logging.INFO, logger="napari_germquant.engine")
    viewer, w = _dock(make_napari_viewer, qtbot, runner)
    ids = [e.id for e in w.pack_entries]
    assert "served_pack" in ids and "test_pack" not in ids and BUILT_IN <= set(ids)
    w.select_pack("served_pack", "with_model")
    settle(w, qtbot)
    assert w.recipe == Recipe.from_pack("served_pack@0.1.0", "with_model")      # the recipe picked, not the default
    st = w.pack_status
    assert not st["usable"]
    assert any(f"model {TINYNET} is not usable" in b["what"] and
               b["fix"] == "germquant pack fetch-models served_pack --recipe with_model" for b in st["blockers"])
    combo = w.pack_picker.recipe_combo
    assert w.pack_picker.is_greyed(combo, combo.findData("with_model"))
    assert "fetch-models served_pack" in combo.itemData(combo.findData("with_model"), Qt.ItemDataRole.ToolTipRole)
    assert not w.run_btn.isEnabled() and "cannot run here" in w.run_btn.toolTip()
    assert "NOT usable here" in w.pack_picker.card.text()

    assert w.pack_picker.get_models_btn.isEnabled()
    w.pack_picker.get_models_btn.click()
    assert w.pack_picker.cancel_btn.isEnabled() == (runner == "subprocess")      # an engine process can be ended
    res = _wait_pack_op(w, qtbot)
    assert res["ok"] and [m["status"] for m in res["models"]] == ["downloaded"]
    blob = served_pack.env.blobs / TINYNET_SHA / "tinynet"
    assert blob.read_bytes() == TINYNET_BYTES
    assert ("models/tinynet", None) in served_pack.log
    assert w.pack_status["usable"] and w.run_btn.isEnabled()
    assert not w.pack_picker.is_greyed(combo, combo.findData("with_model"))
    assert f"{TINYNET}: <span style='color:#3c8c50'>trusted</span>" in w.pack_picker.card.text()
    assert not w.pack_picker.get_models_btn.isEnabled() and not w.pack_picker.cancel_btn.isEnabled()

    w.select_recipe("basic")
    settle(w, qtbot)
    w.last_pack_result = None
    w.pack_picker.get_example_btn.click()
    rec = _wait_pack_op(w, qtbot)
    image = Path(rec["image"])
    assert rec["status"] == "fetched" and rec["id"] == "standin" and rec["kind"] == "remote"
    assert image == served_pack.env.examples / "served_pack" / "standin" / "standin.ome.tif"
    assert w.image_edit.text() == str(image) and w.out_edit.text() == str(api.default_out_dir(image))
    opened = [layer for layer in viewer.layers if layer.metadata.get("napari_germquant", {}).get("path") == str(image)]
    assert [layer.name for layer in opened] == [f"standin [{n}]" for n in served_pack.names]
    for i, layer in enumerate(opened):
        assert np.array_equal(np.asarray(layer.data), served_pack.data[i])
    assert w.pack_status["usable"] and "example standin (fetched)" in w.status_label.text()
    plan = w.plan()
    assert plan.image == image and plan.recipe == Recipe.from_pack("served_pack@0.1.0", "basic")
    progress = [r.getMessage() for r in caplog.records if r.name == "napari_germquant.engine"]
    assert any("assembling standin.ome.tif" in line for line in progress), progress
    if runner == "subprocess":                     # the engine environment made both calls, not this one
        packs = [sys.executable, "-m", "germquant.packs"]
        assert [*packs, "fetch-models", "served_pack@0.1.0", "--recipe", "with_model"] in w.engine.history
        assert [*packs, "fetch-example", "served_pack@0.1.0"] in w.engine.history
        assert [*packs, "fetch-example", "served_pack@0.1.0", "--no-download", "--json"] in w.engine.history


def test_cell_pack_names_cells_and_warns_about_the_segmenter(make_napari_viewer, qtbot, pack_env, tmp_path,
                                                            monkeypatch):
    """Acceptance 4: with object_kind cell the Labels layer is "<id> cells" and the summary says cells; the
    banner shows because the labels came from the classical segmenter while the pack declares cellpose. A pack
    run whose germline stage did not run has no germline count in the summary."""
    from germquant import packs

    def edit(d):
        d["id"] = "cell_pack"
        d["object_kind"] = "cell"
        d["recipes"][0]["needs"]["segment_backends"] = ["cellpose"]     # basic still runs the classical segmenter
    root = tmp_path / "cell_packs"
    copy_pack(TEST_PACK, root / "cell_pack", edit=edit)
    monkeypatch.setenv("GERMQUANT_PACKS", str(root))
    packs.refresh()
    viewer, w = _dock(make_napari_viewer, qtbot, "inprocess")
    w.select_pack("cell_pack", "basic")
    image = _image_copy(tmp_path)
    w.set_image(str(image))
    settle(w, qtbot)
    out = tmp_path / "out"
    w.out_edit.setText(str(out))
    res = _run(w, qtbot)
    image_id = res["image_id"]
    labels = [layer for layer in viewer.layers if layer.__class__.__name__ == "Labels"]
    assert [layer.name for layer in labels] == [f"{image_id} cells"]
    text = w.status_label.text()
    assert "5 cells, 10 spots, mean spots per cell 2.00" in text
    assert ("WARNING: These labels were made by classical, but pack cell_pack@0.1.0 recipe basic expects cellpose"
            in text)

    rw = ResultsWidget(make_napari_viewer())                         # a fresh viewer: no layer names taken
    rw.set_folder(str(out))
    assert rw.summary_line.text().startswith("5 cells, 10 spots, mean spots per cell 2.00")
    assert not rw.banner.isHidden() and "expects cellpose" in rw.banner.text()
    prov = rw.provenance.text()
    for piece in ("pack: cell_pack@0.1.0   recipe: basic", "config_hash: 02bb83cf74f9", "model sha256: unused",
                  f"git sha: {provenance.git_sha()}", f"device used: {engine_device()} (asked for auto)"):
        assert piece in prov, piece
    assert "cells: 5   mean spots per cell: 2.00" in rw.summary.text()
    assert "germline cells" not in rw.summary.text()                 # the germline stage is off in this pack
    assert "germline" in rw.summary.text().split("stages skipped: ")[1]
    layers = rw.load_selected()
    assert f"{image_id} cells" in [layer.name for layer in layers]
    # the pristine test pack expects the classical segmenter: no banner for the same tables
    monkeypatch.setenv("GERMQUANT_PACKS", str(FIXTURE_PACKS))
    packs.refresh()
    from napari_germquant._results import ResultsSet, fallback_banner

    assert fallback_banner(ResultsSet(out, image_id), ["classical"]) is None


def test_export_writes_pack_commands_and_a_portable_overlay(make_napari_viewer, qtbot, pack_env, tmp_path):
    """Acceptance 5: --pack commands and an overlay that extends pack:test_pack@0.1.0/basic, in ASCII, with no
    absolute path; the engine loads the overlay through the pack form of extends."""
    _viewer, w = _dock(make_napari_viewer, qtbot, "subprocess")
    w.select_pack("test_pack", "basic")
    image = _image_copy(tmp_path)
    w.set_image(str(image))
    settle(w, qtbot)
    out = tmp_path / "res"
    w.out_edit.setText(str(out))
    w.stage_checks["spots"].setChecked(False)
    overlay, commands = w.export_to(tmp_path / "export")
    assert overlay.name == "test_pack_basic_test_pack_synthetic.yaml"
    raw = overlay.read_bytes()
    text = raw.decode("ascii")                                     # ASCII, or this raises
    assert chr(0x2013) not in text and chr(0x2014) not in text            # no en or em dash
    # no absolute path anywhere, comments included: no drive letter, no POSIX root after anything but a word
    # character, @ or a dot (pack:test_pack@0.1.0/basic is a pack reference, not a path)
    assert not re.search(r"[A-Za-z]:[\\/]|(?<![\w@.])/[A-Za-z]", text), text
    for p in (tmp_path, Path.home(), TEST_PACK, FIXTURE_PACKS):
        assert str(p) not in text and p.as_posix() not in text, p
    assert yaml.safe_load(text) == {"validated": False, "extends": "pack:test_pack@0.1.0/basic",
                                    "spots": {"enabled": False}}
    cfg = api.load_config(overlay)
    base = api.resolve_config(pack="test_pack@0.1.0", recipe="basic")
    assert cfg.get("spots.enabled") is False and cfg.get("segmentation.nuclei.method") == "classical"
    assert cfg.get("validated") is False and cfg.hash != base.hash
    body = commands.read_text(encoding="utf-8")
    plan = w.plan()
    assert plan.run_command() in body and plan.batch_command() in body
    assert f"--pack test_pack@0.1.0 --recipe basic --out {out} --no-spots" in plan.run_command()
    assert "batch" in plan.batch_command() and "--pack test_pack@0.1.0 --recipe basic" in plan.batch_command()
    assert "-m germquant run" in body and sys.executable in body
    overlay_line = [ln for ln in body.splitlines() if "--config" in ln][-1]
    assert str(overlay) in overlay_line and "--no-spots" not in overlay_line


def test_the_export_guard_finds_a_path_in_a_comment():
    """The absolute-path check of the export test, against what it must catch: a Windows or POSIX path in
    parentheses, after a space or in quotes; and what it must let pass (the pack reference)."""
    guard = re.compile(r"[A-Za-z]:[\\/]|(?<![\w@.])/[A-Za-z]")
    for bad in ("# (C:/Users/x/basic.yaml)", "# (/home/x/basic.yaml)", "# Base recipe: X (C:\\Users\\x)",
                "extends: '/home/x/basic.yaml'", "# see /opt/packs"):
        assert guard.search(bad), bad
    for good in ("extends: pack:test_pack@0.1.0/basic", "# Base recipe: pack test_pack@0.1.0, recipe basic.",
                 "#   segmentation:", "spots:\n  enabled: false"):
        assert not guard.search(good), good


def test_a_broken_pack_is_greyed_with_its_problem(make_napari_viewer, qtbot, pack_env, tmp_path, monkeypatch):
    """A pack whose manifest is broken is listed greyed with the problem and its fix, and Run stays off."""
    import os

    from germquant import packs

    def edit(d):
        d["id"] = "broken_pack"
        d["models"][0]["name"] = "tinynet@00000000"                  # the suffix no longer matches the digest
    root = tmp_path / "broken"
    copy_pack(TEST_PACK, root / "broken_pack", edit=edit)
    monkeypatch.setenv("GERMQUANT_PACKS", os.pathsep.join([str(root), str(FIXTURE_PACKS)]))
    packs.refresh()
    _viewer, w = _dock(make_napari_viewer, qtbot, "subprocess")
    entries = {e.id: e for e in w.pack_entries}
    assert entries["broken_pack"].state == "unusable" and entries["broken_pack"].problems
    combo = w.pack_picker.pack_combo
    i = [combo.itemData(k) for k in range(combo.count())].index(w.pack_entries.index(entries["broken_pack"]))
    assert w.pack_picker.is_greyed(combo, i) and "[unusable]" in combo.itemText(i)
    w.select_pack("broken_pack")
    settle(w, qtbot)
    assert not w.pack_status["usable"] and "pack manifest:" in w.pack_status["blockers"][0]["what"]
    assert "NOT usable here" in w.pack_picker.card.text() and not w.run_btn.isEnabled()
    w.select_pack("test_pack", "basic")
    settle(w, qtbot)
    assert w.pack_status["usable"] and w.run_btn.isEnabled()


def test_output_folder_inside_a_pack_is_refused(make_napari_viewer, qtbot, pack_env):
    """The synthetic example lives inside the pack: a run must never write there (the pack would fail its
    SHA256SUMS); Get example opens it and points the output folder at the examples folder instead."""
    _viewer, w = _dock(make_napari_viewer, qtbot, "inprocess")
    w.select_pack("test_pack", "basic")
    w.set_image(str(TEST_IMAGE))                                      # the default output folder is next to it
    settle(w, qtbot)
    with pytest.raises(ValueError, match="inside the pack folder"):
        w.plan()
    w.out_edit.setText("")
    w.pack_picker.get_example_btn.click()
    rec = _wait_pack_op(w, qtbot)
    assert rec["status"] == "local" and Path(rec["image"]) == TEST_IMAGE
    want = pack_env.examples / "test_pack" / "synthetic" / "test_pack_synthetic_results"
    assert Path(w.out_edit.text()) == want
    assert w.plan().out == want
    assert not (TEST_PACK / "example" / "test_pack_synthetic_results").exists()


def test_run_after_the_engine_prompt_never_runs_a_pack_the_dock_picked(make_napari_viewer, qtbot, pack_env,
                                                                       tmp_path):
    """With no engine, Run on the pack route asks for one; the picker is empty then. Once the engine lists its
    packs the dock shows its first look (the C. elegans pack), but it does not run it: Run was pressed with no
    recipe, so the dock says to pick one. Nothing the dock picked on its own is stored as the user's last pack,
    so the next dock opens on the first look too; a pick the user makes is stored and wins."""
    viewer = make_napari_viewer()
    w = RunWidget(viewer)
    assert w.engine is None
    w.advanced_box.setChecked(False)                              # the pack route, with an empty picker
    assert w.route == "pack" and w.recipe is None and not w.pack_entries
    w.set_image(str(_image_copy(tmp_path)))
    out = tmp_path / "out"
    w.out_edit.setText(str(out))
    w.run_btn.click()
    dlg = w.interpreter_dialog
    assert dlg is not None and dlg.isVisible()
    dlg.choose_inprocess()
    dlg.accept()
    settle(w, qtbot)
    assert w.engine is not None and w.pack_entries
    assert w._worker is None and w.last_result is None and w.last_error is None and not out.exists()
    assert "Engine set. Pick a pack and recipe, then press Run." in w.status_label.text()
    assert w.pack_picker.current_pack_id() == "celegans_germline"
    assert _settings.last_pack() == (None, None)

    w2 = RunWidget(make_napari_viewer())                          # the stored engine, the first look again
    settle(w2, qtbot)
    assert w2.route == "pack" and w2.pack_picker.current_pack_id() == "celegans_germline"
    assert _settings.last_pack() == (None, None)
    w2.select_pack("test_pack", "with_model")
    settle(w2, qtbot)
    assert _settings.last_pack() == ("test_pack", "with_model")
    w3 = RunWidget(make_napari_viewer())
    settle(w3, qtbot)
    assert w3.route == "pack" and w3.recipe == Recipe.from_pack("test_pack@0.1.0", "with_model")


def test_the_card_shows_a_validation_record_and_an_unreadable_one_in_amber(tmp_path):
    """The card's validation line (section 8.2): the dataset and first metric with its interval and the gate
    verdict of a record shaped as section 3.6 describes; a record the recipe names but that cannot be read is
    said in amber; no record is "not validated". The record's exact keys come with package P7."""
    folder = tmp_path / "vpack"
    (folder / "validation").mkdir(parents=True)
    record = {"pack": {"id": "vpack", "version": "0.1.0", "recipe": "basic"}, "config_hash": "0123456789ab",
              "dataset": {"id": "S-BIAD1494", "licence": "CC0-1.0"},
              "metrics": {"object_f1": {"value": 0.91, "ci_low": 0.88, "ci_high": 0.94, "n": 40}},
              "gate": {"rule": "object_f1 lower bound >= 0.85", "verdict": "pass"}}
    (folder / "validation" / "basic.json").write_text(json.dumps(record), encoding="utf-8")
    entry = PackEntry({"id": "vpack", "version": "0.1.0", "state": "ok", "folder": str(folder), "recipes": [
        {"id": "basic", "status": "validated", "validation": "validation/basic.json"},
        {"id": "lost", "status": "validated", "validation": "validation/lost.json"},
        {"id": "plain"}]})
    assert validation_text(entry, "basic") == ("validated on S-BIAD1494: object_f1 0.91 (95% interval 0.88 to "
                                               "0.94); verdict pass")
    assert validation_text(entry, "lost") == ("status validated, but its validation record validation/lost.json "
                                              "cannot be read")
    assert validation_text(entry, "plain") == "not validated (status provisional)"
    assert (f"<b>Validation</b>: <span style='color:{AMBER}'>status validated, but its validation record"
            in card_html(entry, "lost", None))
    assert "<b>Validation</b>: validated on S-BIAD1494" in card_html(entry, "basic", None)


def test_the_test_pack_copy_matches_its_sums_and_the_engines():
    """The plugin's copy of the engine's test pack (section 8.10): every file matches its SHA256SUMS, and the
    SHA256SUMS equals the engine checkout's copy when that checkout is visible (CI runs the same comparison)."""
    import germquant
    from germquant import packs

    assert packs.verify_sums(TEST_PACK) == []
    engine_sums = Path(germquant.__file__).resolve().parents[2] / "tests" / "fixtures" / "packs" / "test_pack" / \
        "SHA256SUMS"
    if not engine_sums.is_file():
        pytest.skip(f"the engine checkout's test pack is not visible from {germquant.__file__}")

    def norm(p: Path) -> bytes:
        return p.read_bytes().replace(b"\r\n", b"\n")

    assert norm(TEST_PACK / "SHA256SUMS") == norm(engine_sums)
