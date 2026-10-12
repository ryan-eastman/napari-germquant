"""The Run dock's system packs (docs/DESIGN_system_packs.md, sections 8.2 and 8.3; package P4 acceptance 1, 2, 4
and 5): the pack list and card through both engine runners, a recipe that cannot run greyed with its reason and
fix, Get models and Get example through the runner, results named after the pack's object kind with the
fallback banner and the Provenance box, the export with --pack, and the refusals (a broken pack, an output
folder inside a pack). Every pack root and store is private to the test (conftest.pack_env)."""
from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import FIXTURE_PACKS, TEST_IMAGE, TEST_PACK, TINYNET, TINYNET_BYTES, TINYNET_SHA, copy_pack
from qtpy.QtCore import Qt

from napari_germquant._engine import Recipe
from napari_germquant._results_widget import ResultsWidget
from napari_germquant._run_widget import RunWidget
from napari_germquant._settings import INPROCESS
from germquant import api, provenance

pytestmark = pytest.mark.skipif(importlib.util.find_spec("germquant.packs") is None,
                                reason="system packs need germquant 0.2.0.dev0 or later")

BUILT_IN = {"celegans_germline", "cultured_cells", "budding_yeast"}


def _engine(runner: str) -> str:
    return sys.executable if runner == "subprocess" else INPROCESS


def _dock(make_napari_viewer, runner: str) -> tuple[object, RunWidget]:
    viewer = make_napari_viewer()
    w = RunWidget(viewer)
    assert w.set_engine(_engine(runner)), w.engine_box.runs_label.text()
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
    assert w.last_pack_error is None, w.last_pack_error
    return w.last_pack_result


@pytest.mark.parametrize("runner", ["inprocess", "subprocess"])
def test_dock_lists_the_packs_and_shows_the_card(make_napari_viewer, pack_env, runner):
    """Acceptance 1: the built-in packs and the one under $GERMQUANT_PACKS, through both runners; the card of
    the test pack; the bench profiles hidden until "show all" is ticked."""
    _viewer, w = _dock(make_napari_viewer, runner)
    entries = {e.id: e for e in w.pack_entries}
    assert BUILT_IN | {"test_pack"} <= set(entries)
    assert entries["test_pack"].data["source"] == "GERMQUANT_PACKS"
    assert all(entries[i].data["source"] == "builtin" for i in BUILT_IN)
    labels = [w.pack_picker.pack_combo.itemText(i) for i in range(w.pack_picker.pack_combo.count())]
    assert any(t.startswith("test_pack 0.1.0: Test pack") for t in labels)
    assert w.route == "profile"                    # a later engine choice does not take the route away

    w.select_pack("test_pack", "basic")
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
    card = w.pack_picker.card.text()
    assert f"{TINYNET}: <span style='color:#c83c3c'>missing</span>" in card and "NOT usable here" in card
    assert "fix: germquant pack fetch-models test_pack --recipe with_model" in card
    assert w.pack_picker.get_models_btn.isEnabled() and not w.run_btn.isEnabled()
    # an outside pack shows its needs; spotmax is missing from the plugin environment (and from CI)
    w.select_pack("budding_yeast")
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
def test_unusable_recipe_get_models_and_get_example(make_napari_viewer, qtbot, served_pack, runner):
    """Acceptance 2: a recipe that cannot run is greyed with its reason and fix and Run is off; Get models
    through the runner (a 64-byte stand-in served by the test) makes it usable; Get example (two per-channel
    TIFFs served by the test) assembles the image, opens it and makes it the run's image."""
    viewer, w = _dock(make_napari_viewer, runner)
    ids = [e.id for e in w.pack_entries]
    assert "served_pack" in ids and "test_pack" not in ids and BUILT_IN <= set(ids)
    w.select_pack("served_pack", "with_model")
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
    res = _wait_pack_op(w, qtbot)
    assert res["ok"] and [m["status"] for m in res["models"]] == ["downloaded"]
    blob = served_pack.env.blobs / TINYNET_SHA / "tinynet"
    assert blob.read_bytes() == TINYNET_BYTES
    assert ("models/tinynet", None) in served_pack.log
    assert w.pack_status["usable"] and w.run_btn.isEnabled()
    assert not w.pack_picker.is_greyed(combo, combo.findData("with_model"))
    assert f"{TINYNET}: <span style='color:#3c8c50'>trusted</span>" in w.pack_picker.card.text()
    assert not w.pack_picker.get_models_btn.isEnabled()

    w.select_recipe("basic")
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


def test_cell_pack_names_cells_and_warns_about_the_segmenter(make_napari_viewer, qtbot, pack_env, tmp_path,
                                                            monkeypatch):
    """Acceptance 4: with object_kind cell the Labels layer is "<id> cells" and the summary says cells; the
    banner shows because the labels came from the classical segmenter while the pack declares cellpose."""
    from germquant import packs

    def edit(d):
        d["id"] = "cell_pack"
        d["object_kind"] = "cell"
        d["recipes"][0]["needs"]["segment_backends"] = ["cellpose"]     # basic still runs the classical segmenter
    root = tmp_path / "cell_packs"
    copy_pack(TEST_PACK, root / "cell_pack", edit=edit)
    monkeypatch.setenv("GERMQUANT_PACKS", str(root))
    packs.refresh()
    viewer, w = _dock(make_napari_viewer, "inprocess")
    w.select_pack("cell_pack", "basic")
    image = _image_copy(tmp_path)
    w.set_image(str(image))
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
                  f"git sha: {provenance.git_sha()}", "device used: cpu (asked for auto)"):
        assert piece in prov, piece
    assert "cells: 5   germline cells: 5" in rw.summary.text()
    layers = rw.load_selected()
    assert f"{image_id} cells" in [layer.name for layer in layers]
    # the pristine test pack expects the classical segmenter: no banner for the same tables
    monkeypatch.setenv("GERMQUANT_PACKS", str(FIXTURE_PACKS))
    packs.refresh()
    from napari_germquant._results import ResultsSet, fallback_banner

    assert fallback_banner(ResultsSet(out, image_id), ["classical"]) is None


def test_export_writes_pack_commands_and_a_portable_overlay(make_napari_viewer, pack_env, tmp_path):
    """Acceptance 5: --pack commands and an overlay that extends pack:test_pack@0.1.0/basic, in ASCII, with no
    absolute path; the engine loads the overlay through the pack form of extends."""
    _viewer, w = _dock(make_napari_viewer, "subprocess")
    w.select_pack("test_pack", "basic")
    image = _image_copy(tmp_path)
    w.set_image(str(image))
    out = tmp_path / "res"
    w.out_edit.setText(str(out))
    w.stage_checks["spots"].setChecked(False)
    overlay, commands = w.export_to(tmp_path / "export")
    assert overlay.name == "test_pack_basic_test_pack_synthetic.yaml"
    raw = overlay.read_bytes()
    text = raw.decode("ascii")                                     # ASCII, or this raises
    assert chr(0x2013) not in text and chr(0x2014) not in text            # no en or em dash
    assert str(tmp_path) not in text and str(Path.home()) not in text
    assert not re.search(r"(^|[\s'\"])(/[A-Za-z]|[A-Za-z]:[\\/])", text), text   # no absolute path, comments too
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


def test_a_broken_pack_is_greyed_with_its_problem(make_napari_viewer, pack_env, tmp_path, monkeypatch):
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
    _viewer, w = _dock(make_napari_viewer, "subprocess")
    entries = {e.id: e for e in w.pack_entries}
    assert entries["broken_pack"].state == "unusable" and entries["broken_pack"].problems
    combo = w.pack_picker.pack_combo
    i = [combo.itemData(k) for k in range(combo.count())].index(w.pack_entries.index(entries["broken_pack"]))
    assert w.pack_picker.is_greyed(combo, i) and "[unusable]" in combo.itemText(i)
    w.select_pack("broken_pack")
    assert not w.pack_status["usable"] and "pack manifest:" in w.pack_status["blockers"][0]["what"]
    assert "NOT usable here" in w.pack_picker.card.text() and not w.run_btn.isEnabled()
    w.select_pack("test_pack", "basic")
    assert w.pack_status["usable"] and w.run_btn.isEnabled()


def test_output_folder_inside_a_pack_is_refused(make_napari_viewer, qtbot, pack_env):
    """The synthetic example lives inside the pack: a run must never write there (the pack would fail its
    SHA256SUMS); Get example opens it and points the output folder at the examples folder instead."""
    _viewer, w = _dock(make_napari_viewer, "inprocess")
    w.select_pack("test_pack", "basic")
    w.set_image(str(TEST_IMAGE))                                      # the default output folder is next to it
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
