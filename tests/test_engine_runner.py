"""The engine runners and the engine choice (docs/DESIGN_system_packs.md, section 8.1; package P4 acceptance 6, 9
and 10): the version check and the warning when the engine differs from the plugin's germquant, the one message
an engine without system packs gets (in this environment and through a subprocess), the dock asking for an
engine instead of running in process, a recipe mapped to a second interpreter running there with run_start
recording it, a pack card refresh through the subprocess runner that loads no torch in the engine process and
does not re-hash a verified blob, and the run log's place and format."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import HEAVY, TEST_IMAGE, TINYNET_SHA

from napari_germquant import _engine, _settings
from napari_germquant._engine import Recipe
from napari_germquant._run_widget import RunWidget
from napari_germquant._runner import (
    VIA_INPROCESS,
    VIA_SUBPROCESS,
    EngineTooOld,
    InProcessRunner,
    RunLog,
    SubprocessRunner,
    detect_interpreters,
    mismatch_warning,
    plugin_engine_info,
    read_run_log,
    runs_folder_for,
    user_runs_dir,
    version_mismatch,
)
from napari_germquant._settings import INPROCESS
from germquant import api

HAS_PACKS = importlib.util.find_spec("germquant.packs") is not None
needs_packs = pytest.mark.skipif(not HAS_PACKS, reason="system packs need germquant 0.2.0.dev0 or later")

# runs `python -m germquant.packs ARGS` with germquant.datafetch.sha256_file counted, then reports whether torch
# (or another heavy module) was loaded and which files were hashed
COUNTING_WRAPPER = (
    "import json, runpy, sys\n"
    "import germquant.datafetch as D\n"
    "calls = []\n"
    "orig = D.sha256_file\n"
    "def counted(*a, **k):\n"
    "    calls.append(str(a[0]) if a else '')\n"
    "    return orig(*a, **k)\n"
    "D.sha256_file = counted\n"
    "sys.argv = ['germquant.packs', *sys.argv[1:]]\n"
    "code = 0\n"
    "try:\n"
    "    runpy.run_module('germquant.packs', run_name='__main__', alter_sys=True)\n"
    "except SystemExit as e:\n"
    "    code = e.code\n"
    "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'cupy', 'spotmax', 'cellpose'))\n"
    "print('WRAPPER ' + json.dumps({'heavy': heavy, 'sha256_calls': calls, 'exit': code}))\n"
)


def _fake_engine(root: Path, *, version: str, git_sha: str = "feedfacecafebeef", with_packs: bool = True) -> Path:
    """A stand-in germquant package: ``__version__`` only (an engine from before system packs), or with a
    ``germquant.packs`` entry that answers ``--version --json`` (and every other command) with the given
    version and git sha."""
    pkg = root / "germquant"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="ascii")
    if with_packs:
        (pkg / "packs").mkdir()
        (pkg / "packs" / "__init__.py").write_text("", encoding="ascii")
        (pkg / "packs" / "__main__.py").write_text(
            "import json, platform, sys\n"
            f"print(json.dumps({{'germquant': {version!r}, 'git_sha': {git_sha!r}, "
            "'python': platform.python_version(), 'executable': sys.executable}))\n", encoding="ascii")
    return root


def _same_path(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _imported(stderr: str) -> set[str]:
    """Module names from Python's -X importtime log."""
    return {ln.rsplit("|", 1)[1].strip() for ln in stderr.splitlines() if ln.startswith("import time:") and "|" in ln}


def _run(w: RunWidget, qtbot, timeout: int = 600_000) -> dict:
    try:
        qtbot.waitUntil(lambda: w.last_result is not None or w.last_error is not None, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - say what the dock says
        raise AssertionError(f"no run result; the dock says: {w.status_label.text()!r}") from e
    qtbot.waitUntil(lambda: w.run_btn.isEnabled(), timeout=30_000)
    assert w.last_error is None, w.last_error
    return w.last_result


# ---------------------------------------------------------------------------------------------------
# versions and listing
# ---------------------------------------------------------------------------------------------------


@needs_packs
def test_both_runners_list_and_check_through_the_engine(pack_env):
    """The version check, the listing and the checks give the same answers through both runners; the
    subprocess commands are the torch-free entry, as lists of arguments."""
    own = plugin_engine_info()
    sub, inproc = SubprocessRunner(sys.executable), InProcessRunner()
    for r in (inproc, sub):
        v = r.version()
        assert v["germquant"] == api.__version__ and v["git_sha"] == own["git_sha"] and not version_mismatch(own, v)
        listing = r.list_packs()
        ids = {p["id"]: p for p in listing["packs"] if p["state"] == "ok"}
        assert {"celegans_germline", "cultured_cells", "budding_yeast", "test_pack"} <= set(ids)
        assert listing["builtin_root_found"] is True
        assert "GERMQUANT_PACKS" in {root["kind"] for root in listing["roots"]}
        basic = r.check("test_pack@0.1.0", "basic")
        assert basic["usable"] and basic["recipe"] == "basic" and basic["blockers"] == []
        model = r.check("test_pack@0.1.0", "with_model")
        assert not model["usable"] and model["models"][0]["state"] == "missing"
        assert any(b["fix"] == "germquant pack fetch-models test_pack --recipe with_model" for b in model["blockers"])
        unknown = r.check("no_such_pack")
        assert not unknown["usable"] and "no pack 'no_such_pack'" in unknown["blockers"][0]["what"]
    assert all(a[1:3] == ["-m", "germquant.packs"] and isinstance(a, list) for a in sub.history)
    assert sub.describe() == f"{sys.executable} (a separate process)" and sub.via == VIA_SUBPROCESS
    assert inproc.via == VIA_INPROCESS
    found = [c["python"] for c in detect_interpreters()]
    assert sys.executable in found                       # always offered, never chosen without asking


def test_version_mismatch_words():
    own = {"germquant": "0.2.0.dev0", "git_sha": "aaa"}
    assert version_mismatch(own, dict(own)) == [] and mismatch_warning(own, dict(own)) == ""
    diffs = version_mismatch(own, {"germquant": "0.2.1", "git_sha": "bbb"})
    assert diffs == ["the engine is germquant 0.2.1, the plugin's own germquant is 0.2.0.dev0",
                     "the engine's git sha is bbb, the plugin's is aaa"]
    assert mismatch_warning(own, {"germquant": "0.2.0.dev0", "git_sha": "bbb"}).startswith("Warning: the engine's git")


@needs_packs
def test_a_different_engine_version_or_git_sha_shows_a_warning(make_napari_viewer, tmp_path):
    """Acceptance 9: the dock warns when the engine's germquant differs from the plugin's own."""
    own = plugin_engine_info()
    fake = _fake_engine(tmp_path / "other", version=str(own["germquant"]), git_sha="feedfacecafebeef")
    w = RunWidget(make_napari_viewer())
    assert w.set_engine(sys.executable, env={"PYTHONPATH": str(fake)})
    warn = w.engine_box.warn_label
    assert not warn.isHidden()
    assert "the engine's git sha is feedfacecafebeef" in warn.text() and str(own["git_sha"]) in warn.text()
    assert "git feedfacecafe" in w.engine_box.runs_label.text()
    newer = _fake_engine(tmp_path / "newer", version="0.2.7", git_sha=str(own["git_sha"]))
    w2 = RunWidget(make_napari_viewer())
    assert w2.set_engine(sys.executable, env={"PYTHONPATH": str(newer)})
    assert "the engine is germquant 0.2.7" in w2.engine_box.warn_label.text()
    w3 = RunWidget(make_napari_viewer())                      # the same engine: no warning
    assert w3.set_engine(sys.executable) and w3.engine_box.warn_label.isHidden()


# ---------------------------------------------------------------------------------------------------
# an engine older than 0.2.0.dev0 (acceptance 6)
# ---------------------------------------------------------------------------------------------------


def test_an_engine_without_packs_through_the_subprocess_runner_gets_one_message(make_napari_viewer, tmp_path):
    old = _fake_engine(tmp_path / "old", version="0.1.0", with_packs=False)
    r = SubprocessRunner(sys.executable, env={"PYTHONPATH": str(old)})
    with pytest.raises(EngineTooOld) as e:
        r.version()
    msg = str(e.value)
    assert msg.startswith(f"The engine at {sys.executable} has a germquant without system packs (older than "
                          f"0.2.0.dev0).") and "git pull" in msg
    numbered = _fake_engine(tmp_path / "numbered", version="0.1.9")
    with pytest.raises(EngineTooOld, match="is germquant 0.1.9, older than 0.2.0.dev0"):
        SubprocessRunner(sys.executable, env={"PYTHONPATH": str(numbered)}).version()
    w = RunWidget(make_napari_viewer())
    assert not w.set_engine(sys.executable, env={"PYTHONPATH": str(old)})
    assert w.engine is None and _settings.engine_choice() is None       # not stored: Run asks again
    assert w.engine_box.runs_label.text() == f"The engine chosen for this computer cannot be used: {msg}"
    if HAS_PACKS:
        assert w.pack_picker.message.text() == msg


def test_an_engine_without_packs_in_this_environment_gets_one_message(make_napari_viewer, monkeypatch,
                                                                     synthetic_image):
    """hasattr(api, "list_packs") gates every pack feature; the profile route keeps working."""
    monkeypatch.delattr(api, "list_packs", raising=False)
    assert not _engine.packs_supported()
    msg = _engine.packs_unavailable_message()
    assert msg.startswith("System packs need germquant 0.2.0.dev0 or later.") and str(api.__version__) in msg
    w = RunWidget(make_napari_viewer())
    assert w.pack_picker.message.text() == msg and not w.pack_picker.isEnabled()
    assert not w.advanced_box.isCheckable() and w.route == "profile"
    with pytest.raises(RuntimeError, match="System packs need germquant 0.2.0.dev0"):
        w.select_pack("test_pack")
    with pytest.raises(EngineTooOld) as e:
        InProcessRunner().list_packs()
    assert str(e.value) == msg
    with pytest.raises(RuntimeError, match="0.2.0.dev0"):
        Recipe.from_pack("test_pack@0.1.0", "basic").load()
    w.select_profile("rad51_foci")
    w.set_image(str(synthetic_image))
    assert w.plan().recipe == Recipe.from_profile("rad51_foci")
    assert w.set_engine(INPROCESS) and w.pack_entries == []


# ---------------------------------------------------------------------------------------------------
# the engine choice (acceptance 9)
# ---------------------------------------------------------------------------------------------------


def test_with_no_engine_the_dock_asks_instead_of_running_here(make_napari_viewer, qtbot, synthetic_image,
                                                              classical_overlay, tmp_path):
    viewer = make_napari_viewer()
    w = RunWidget(viewer)
    assert w.engine is None and _settings.engine_choice() is None
    assert w.engine_box.runs_label.text().startswith("Not chosen yet. Run asks for it")
    w.set_image(str(synthetic_image))
    w.select_config_file(classical_overlay)
    w.stage_checks["spots"].setChecked(False)
    out = tmp_path / "out"
    w.out_edit.setText(str(out))
    w.run_btn.click()
    dlg = w.interpreter_dialog
    assert dlg is not None and dlg.isVisible() and dlg.windowTitle() == "germquant engine for this computer"
    assert w._worker is None and w.last_result is None and not out.exists()     # nothing ran, here or elsewhere
    assert "Choose the engine environment first" in w.status_label.text()
    assert sys.executable in dlg.candidates() and dlg.choice() == dlg.candidates()[0]
    dlg.reject()
    assert _settings.engine_choice() is None and not out.exists() and "nothing ran" in w.status_label.text()
    if HAS_PACKS:                                              # the pack route asks too
        w.advanced_box.setChecked(False)
        assert w.route == "pack"
        w.run_btn.click()
        assert w.interpreter_dialog is not dlg and w.interpreter_dialog.isVisible() and w._worker is None
        w.interpreter_dialog.reject()
        w.select_config_file(classical_overlay)                # the same file again: still the --config route
        assert w.recipe == Recipe.from_file(classical_overlay) and w.config is not None
        w.stage_checks["spots"].setChecked(False)
    # asked again, the user explicitly picks this napari environment: then the run starts
    previous = w.interpreter_dialog
    w.run_btn.click()
    dlg = w.interpreter_dialog
    assert dlg is not previous and dlg.isVisible()
    dlg.choose_inprocess()
    assert dlg.choice() == INPROCESS
    dlg.accept()
    res = _run(w, qtbot, timeout=180_000)
    assert res["n_nuclei"] >= 1 and _settings.engine_choice() == INPROCESS and w.engine.via == VIA_INPROCESS
    start = read_run_log(w.last_run_log)[0]
    assert start["via"] == VIA_INPROCESS and start["interpreter"] == sys.executable
    # a new dock on this computer starts with the stored engine
    w2 = RunWidget(make_napari_viewer())
    assert w2.engine is not None and w2.engine.via == VIA_INPROCESS


@needs_packs
def test_a_recipe_mapped_to_a_second_interpreter_runs_there(make_napari_viewer, qtbot, pack_env, second_python,
                                                            tmp_path):
    viewer = make_napari_viewer()
    w = RunWidget(viewer)
    assert w.set_engine(sys.executable)
    w.select_pack("test_pack", "basic")
    dlg = w.ask_for_recipe_interpreter()
    assert dlg.windowTitle() == "interpreter for test_pack basic"
    dlg.select_path(str(second_python))
    dlg.accept()
    assert _settings.recipe_interpreter("test_pack", "basic") == str(second_python)
    runner = w.engine_runner("test_pack", "basic")
    assert runner is not w.engine and runner.python == str(second_python)
    assert w.engine_runner("test_pack", "with_model") is w.engine
    assert _same_path(runner.version()["executable"], second_python)
    assert f"recipe test_pack basic runs in its own interpreter: {second_python}" in w.engine_box.runs_label.text()
    assert w.pack_status["usable"] and any(a[3:5] == ["check", "test_pack@0.1.0"] for a in runner.history)
    image = tmp_path / "image" / TEST_IMAGE.name
    image.parent.mkdir()
    shutil.copyfile(TEST_IMAGE, image)
    w.set_image(str(image))
    w.out_edit.setText(str(tmp_path / "out"))
    w.run_btn.click()
    res = _run(w, qtbot)
    assert res["interpreter"] == str(second_python) and res["via"] == VIA_SUBPROCESS
    assert runner.history[-1][:4] == [str(second_python), "-m", "germquant", "run"]
    start, done, end = read_run_log(w.last_run_log)
    assert start["interpreter"] == str(second_python) and start["interpreter_source"] == "recipe"
    assert start["via"] == VIA_SUBPROCESS and start["recipe"] == {"kind": "pack", "pack": "test_pack@0.1.0",
                                                                 "recipe": "basic"}
    assert done["event"] == "image_done" and end["counts"] == {"done": 1, "failed": 0, "skipped": 0}
    assert api.read_done_marker(tmp_path / "out", res["image_id"])["config_hash"] == "02bb83cf74f9"
    # back to the dock's engine
    dlg = w.ask_for_recipe_interpreter()
    dlg.choose_default()
    assert dlg.choice() == ""
    dlg.accept()
    assert _settings.recipe_interpreter("test_pack", "basic") is None
    assert w.engine_runner("test_pack", "basic") is w.engine


# ---------------------------------------------------------------------------------------------------
# a card refresh is torch-free and trusts a verified blob (acceptance 10)
# ---------------------------------------------------------------------------------------------------


@needs_packs
def test_card_refresh_loads_no_torch_and_does_not_rehash_a_verified_blob(make_napari_viewer, qtbot, served_pack):
    w = RunWidget(make_napari_viewer())
    assert w.set_engine(sys.executable)
    w.select_pack("served_pack", "with_model")
    assert not w.pack_status["usable"]
    w.pack_picker.get_models_btn.click()
    qtbot.waitUntil(lambda: w.last_pack_result is not None or w.last_pack_error is not None, timeout=120_000)
    qtbot.waitUntil(lambda: not w.pack_worker.busy, timeout=30_000)
    assert w.last_pack_result["ok"], w.last_pack_result
    blob = served_pack.env.blobs / TINYNET_SHA / "tinynet"
    record = blob.parent / f"{TINYNET_SHA}.json"
    rec0, mtime0 = json.loads(record.read_text(encoding="utf-8")), blob.stat().st_mtime_ns

    # the card refresh, through the dock's own subprocess runner with Python's import log switched on
    w.engine.interpreter_options = ("-X", "importtime")
    st = w.refresh_card(force=True)
    argv = w.engine.history[-1]
    assert argv[:5] == [sys.executable, "-X", "importtime", "-m", "germquant.packs"]
    assert argv[5:] == ["check", "served_pack@0.1.0", "--recipe", "with_model", "--json"]
    assert st["usable"] and [m["state"] for m in st["models"]] == ["trusted"]        # trusted: not re-hashed
    imported = _imported(w.engine.last_stderr)
    assert "germquant.packs" in imported and "yaml" in imported                       # the log was read
    assert not [m for m in imported if m.split(".")[0] in HEAVY], sorted(imported)
    assert "germquant.pipeline" not in imported
    rec1 = json.loads(record.read_text(encoding="utf-8"))
    assert rec1["verified_at"] == rec0["verified_at"] and blob.stat().st_mtime_ns == mtime0

    # the same command once more with the hashing counted
    plain = [a for a in argv if a not in ("-X", "importtime")]
    proc = subprocess.run([sys.executable, "-c", COUNTING_WRAPPER, *plain[3:]], capture_output=True, text=True,
                          env=w.engine.child_env(), timeout=300)
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("WRAPPER "))
    info = json.loads(line[len("WRAPPER "):])
    assert info == {"heavy": [], "sha256_calls": [], "exit": 0}, (info, proc.stderr[-2000:])
    # touching the blob (a new mtime) makes the next check re-hash it, and say so
    os.utime(blob, ns=(mtime0 + 10_000_000_000, mtime0 + 10_000_000_000))
    proc = subprocess.run([sys.executable, "-c", COUNTING_WRAPPER, *plain[3:]], capture_output=True, text=True,
                          env=w.engine.child_env(), timeout=300)
    info = json.loads(next(ln for ln in proc.stdout.splitlines() if ln.startswith("WRAPPER "))[len("WRAPPER "):])
    assert len(info["sha256_calls"]) == 1 and Path(info["sha256_calls"][0]).name == "tinynet" and info["heavy"] == []


# ---------------------------------------------------------------------------------------------------
# run logs (section 7.6)
# ---------------------------------------------------------------------------------------------------


def test_run_log_location_and_format(tmp_path, monkeypatch):
    out = tmp_path / "data" / "img_results"
    folder, where = runs_folder_for(out)
    assert folder == tmp_path / "data" / "runs" and where == "beside the output folder"
    log = RunLog(folder, "20261011T000000Z_abcdef12")
    log.event("run_start", via="gui-subprocess", note="plain")
    log.event("run_end", counts={"done": 1})
    raw = log.path.read_bytes()
    raw.decode("ascii")
    assert raw.count(b"\n") == 2 and [e["event"] for e in read_run_log(log.path)] == ["run_start", "run_end"]
    assert all(e["run_id"] == "20261011T000000Z_abcdef12" and e["time"] for e in read_run_log(log.path))
    # a folder that cannot hold runs/ sends the log to the per-user folder
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "runs").write_text("a file, not a folder", encoding="ascii")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder2, where2 = runs_folder_for(blocked / "img_results")
    assert folder2 == user_runs_dir() and folder2.is_dir() and where2.startswith("the per-user folder")
