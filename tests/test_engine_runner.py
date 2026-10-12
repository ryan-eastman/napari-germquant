"""The engine runners and the engine choice (docs/DESIGN_system_packs.md, section 8.1; package P4 acceptance 6, 9
and 10): the version check and the warning when the engine differs from the plugin's germquant, the one message
an engine without system packs gets (in this environment and through a subprocess), the dock asking for an
engine instead of running in process (and preselecting only a detected engine checkout), a recipe mapped to a
second interpreter running there with run_start recording it, a pack card refresh through the subprocess runner
that loads no torch in the engine process and does not re-hash a verified blob, the run log's place and format,
the device choice reaching the engine, a GUI thread that never waits for the engine, Cancel and napari quitting
ending a hung engine, the interpreter kept per computer, and executables that are not Python refused."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import HEAVY, TEST_IMAGE, TINYNET_SHA, settle
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QDialogButtonBox

from napari_germquant import _engine, _settings
from napari_germquant._engine import Recipe, RunPlan
from napari_germquant._engine_widgets import InterpreterDialog
from napari_germquant._run_widget import RunWidget
from napari_germquant._runner import (
    MIN_ENGINE,
    VIA_INPROCESS,
    VIA_SUBPROCESS,
    EngineRunner,
    EngineTooOld,
    InProcessRunner,
    RunCancelled,
    RunLog,
    RunnerError,
    SubprocessRunner,
    detect_interpreters,
    engine_too_old,
    execute_run,
    mismatch_warning,
    plugin_engine_info,
    read_run_log,
    runs_folder_for,
    user_runs_dir,
    version_mismatch,
)
from napari_germquant._settings import INPROCESS
from germquant import api, cli

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

# a stand-in engine's `python -m germquant.packs`: --version answers at once; list and check answer after the
# given sleeps, with one pack (slow_pack) that is usable; with FAKE_ENGINE_PID_FILE set, fetch-models writes its
# pid and never ends; anything else gets the version document
FAKE_PACKS_MAIN = """\
import json, os, platform, sys, time
args = sys.argv[1:]
info = {{"germquant": {version!r}, "git_sha": {git_sha!r}, "python": platform.python_version(),
        "executable": sys.executable}}
if "--version" in args:
    print(json.dumps(info))
    sys.exit(0)
cmd = args[0] if args else ""
if cmd == "fetch-models" and os.environ.get("FAKE_ENGINE_PID_FILE"):
    print("downloading slowly", flush=True)
    with open(os.environ["FAKE_ENGINE_PID_FILE"], "w", encoding="ascii") as fh:
        fh.write(str(os.getpid()))
    time.sleep(600)
time.sleep(float({slow!r}.get(cmd, 0)))
if cmd == "list":
    pack = {{"id": "slow_pack", "version": "0.1.0", "state": "ok", "title": "Slow pack", "source": "GERMQUANT_PACKS",
            "recipes": [{{"id": "r1", "default": True, "dims": "3d", "title": "only recipe"}}], "models": []}}
    print(json.dumps({{"packs": [pack], "roots": [], "builtin_root_found": True, "message": None}}))
elif cmd == "check":
    print(json.dumps({{"pack": args[1].split("@")[0], "recipe": "r1", "usable": True, "blockers": [],
                      "warnings": [], "models": []}}))
else:
    print(json.dumps(info))
"""

# a stand-in engine's `python -m germquant` (a run that never ends): it starts a child process of its own (as an
# engine may), writes both pids, then sleeps; ending the engine must end the child too
FAKE_HUNG_RUN = """\
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
Path(os.environ["FAKE_ENGINE_PID_FILE"]).write_text(f"{os.getpid()} {child.pid}", encoding="ascii")
time.sleep(600)
"""
# the interpreter behind this one: a venv's python.exe on Windows is a launcher that already ends its own child
# when it is killed, which would hide whether the runner ends a whole process tree
BASE_PYTHON = getattr(sys, "_base_executable", None) or sys.executable


def _pids(pid_file: Path) -> list[int]:
    return [int(p) for p in pid_file.read_text(encoding="ascii").split()]


def _accepted(version) -> str:
    """napari's own germquant version for a stand-in engine, or the oldest the runner accepts when napari's
    germquant is older (the suite also runs against an engine without packs)."""
    return MIN_ENGINE if engine_too_old(str(version)) else str(version)


def _fake_engine(root: Path, *, version: str, git_sha: str = "feedfacecafebeef", with_packs: bool = True,
                 slow: dict | None = None, hang_run: bool = False) -> Path:
    """A stand-in germquant package: ``__version__`` only (an engine from before system packs), or with a
    ``germquant.packs`` entry (FAKE_PACKS_MAIN) answering with the given version and git sha; ``hang_run`` adds a
    ``python -m germquant`` that never ends (FAKE_HUNG_RUN)."""
    pkg = root / "germquant"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="ascii")
    if with_packs:
        (pkg / "packs").mkdir()
        (pkg / "packs" / "__init__.py").write_text("", encoding="ascii")
        (pkg / "packs" / "__main__.py").write_text(
            FAKE_PACKS_MAIN.format(version=version, git_sha=git_sha, slow=dict(slow or {})), encoding="ascii")
    if hang_run:
        (pkg / "__main__.py").write_text(FAKE_HUNG_RUN, encoding="ascii")
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


def _engine_set(w: RunWidget, qtbot, choice, env: dict | None = None) -> RunWidget:
    w.set_engine(choice, env=env)
    settle(w, qtbot)
    return w


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
        assert r.cached_version() == v
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
    assert sub.cancellable and not inproc.cancellable and inproc.via == VIA_INPROCESS
    found = {c["python"]: c for c in detect_interpreters()}
    assert sys.executable in found                       # always offered, never chosen without asking
    own_python = found[sys.executable]
    assert own_python["why"].startswith("this napari environment's own Python") and not own_python["detected"]
    if importlib.util.find_spec("spotmax") is None:      # the plugin environment (and CI) lack SpotMAX
        assert "no SpotMAX" in own_python["why"] and "the spots stage cannot run here" in own_python["why"]


def test_version_mismatch_words():
    own = {"germquant": "0.2.0.dev0", "git_sha": "aaa"}
    assert version_mismatch(own, dict(own)) == [] and mismatch_warning(own, dict(own)) == ""
    diffs = version_mismatch(own, {"germquant": "0.2.1", "git_sha": "bbb"})
    assert diffs == ["the engine is germquant 0.2.1, the plugin's own germquant is 0.2.0.dev0",
                     "the engine's git sha is bbb, the plugin's is aaa"]
    assert mismatch_warning(own, {"germquant": "0.2.0.dev0", "git_sha": "bbb"}).startswith("Warning: the engine's git")


@needs_packs
def test_a_different_engine_version_or_git_sha_shows_a_warning(make_napari_viewer, qtbot, tmp_path):
    """Acceptance 9: the dock warns when the engine's germquant differs from the plugin's own."""
    own = plugin_engine_info()
    fake = _fake_engine(tmp_path / "other", version=str(own["germquant"]), git_sha="feedfacecafebeef")
    w = _engine_set(RunWidget(make_napari_viewer()), qtbot, sys.executable, {"PYTHONPATH": str(fake)})
    warn = w.engine_box.warn_label
    assert not warn.isHidden()
    assert "the engine's git sha is feedfacecafebeef" in warn.text() and str(own["git_sha"]) in warn.text()
    assert "git feedfacecafe" in w.engine_box.runs_label.text()
    newer = _fake_engine(tmp_path / "newer", version="0.2.7", git_sha=str(own["git_sha"]))
    w2 = _engine_set(RunWidget(make_napari_viewer()), qtbot, sys.executable, {"PYTHONPATH": str(newer)})
    assert "the engine is germquant 0.2.7" in w2.engine_box.warn_label.text()
    w3 = _engine_set(RunWidget(make_napari_viewer()), qtbot, sys.executable)     # the same engine: no warning
    assert w3.engine is not None and w3.engine_box.warn_label.isHidden()


# ---------------------------------------------------------------------------------------------------
# an engine older than 0.2.0.dev0 (acceptance 6)
# ---------------------------------------------------------------------------------------------------


def test_an_engine_without_packs_through_the_subprocess_runner_gets_one_message(make_napari_viewer, qtbot,
                                                                               tmp_path):
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
    w = _engine_set(RunWidget(make_napari_viewer()), qtbot, sys.executable, {"PYTHONPATH": str(old)})
    assert w.engine is None and w.engine_ok is False and _settings.engine_choice() is None   # Run asks again
    assert w.engine_box.runs_label.text() == f"The engine chosen for this computer cannot be used: {msg}"
    if HAS_PACKS:
        assert w.pack_picker.message.text() == msg
    # with a working engine chosen, trying an old one keeps the working one (and stores nothing new)
    w2 = _engine_set(RunWidget(make_napari_viewer()), qtbot, INPROCESS)
    assert w2.engine is not None and _settings.engine_choice() == INPROCESS
    _engine_set(w2, qtbot, sys.executable, {"PYTHONPATH": str(old)})
    assert w2.engine_ok is False
    assert w2.engine is not None and w2.engine.via == VIA_INPROCESS and _settings.engine_choice() == INPROCESS
    assert "that engine cannot be used, so the dock keeps this napari environment" in w2.status_label.text()


def test_an_engine_without_packs_in_this_environment_gets_one_message(make_napari_viewer, qtbot, monkeypatch,
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
    _engine_set(w, qtbot, INPROCESS)
    assert w.engine is not None and w.pack_entries == []


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
    assert sys.executable in dlg.candidates()
    # only a detected engine checkout .venv is preselected; napari's own Python never is
    detected = [c["python"] for c in detect_interpreters() if c["detected"]]
    ok = dlg.buttons.button(QDialogButtonBox.StandardButton.Ok)
    if detected:
        assert dlg.choice() == detected[0] and ok.isEnabled()
    else:
        assert dlg.choice() is None and not ok.isEnabled()
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
    settle(w2, qtbot)
    assert w2.engine is not None and w2.engine.via == VIA_INPROCESS


def test_the_interpreter_dialog_preselects_only_a_detected_checkout(qtbot, tmp_path):
    """One OK click must never make napari's own Python the engine: with no detected checkout .venv nothing is
    selected and OK is off until a row is picked."""
    a, b = str(tmp_path / "engine" / "python.exe"), str(tmp_path / "napari" / "python.exe")
    dlg = InterpreterDialog([{"python": a, "why": "the .venv", "detected": True}, {"python": b, "why": "own"}],
                            title="t", intro="i")
    qtbot.addWidget(dlg)
    assert dlg.choice() == a
    dlg2 = InterpreterDialog([{"python": b, "why": "own", "detected": False}], title="t", intro="i")
    qtbot.addWidget(dlg2)
    ok = dlg2.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert dlg2.choice() is None and not ok.isEnabled()
    dlg2.list.setCurrentRow(0)                                 # the user picks the row
    assert dlg2.choice() == b and ok.isEnabled()


@needs_packs
def test_a_recipe_mapped_to_a_second_interpreter_runs_there(make_napari_viewer, qtbot, pack_env, second_python,
                                                            tmp_path):
    viewer = make_napari_viewer()
    w = _engine_set(RunWidget(viewer), qtbot, sys.executable)
    w.select_pack("test_pack", "basic")
    settle(w, qtbot)
    dlg = w.ask_for_recipe_interpreter()
    assert dlg.windowTitle() == "interpreter for test_pack basic"
    dlg.select_path(str(second_python))
    dlg.accept()
    settle(w, qtbot)                                           # its version check, then the card through it
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
    settle(w, qtbot)
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
    settle(w, qtbot)
    assert _settings.recipe_interpreter("test_pack", "basic") is None
    assert w.engine_runner("test_pack", "basic") is w.engine


def test_the_engine_choice_is_kept_per_computer(make_napari_viewer, qtbot, monkeypatch):
    """The interpreter keys carry the host name: a roaming profile or a home folder shared by many computers
    never hands one computer's engine path to another; the device and the last pack stay per user."""
    monkeypatch.setattr(socket, "gethostname", lambda: "Lab PC/1")
    _settings.set_engine_choice(INPROCESS)
    _settings.set_recipe_interpreter("test_pack", "basic", "C:/elsewhere/python.exe")
    _settings.set_device("cpu")
    assert _settings.engine_choice() == INPROCESS and _settings.recipe_interpreters() == {
        "test_pack/basic": "C:/elsewhere/python.exe"}
    keys = _settings.qsettings().allKeys()
    assert "engine/lab_pc_1/interpreter" in keys and "recipe_interpreters/lab_pc_1/test_pack/basic" in keys
    monkeypatch.setattr(socket, "gethostname", lambda: "laptop")
    assert _settings.engine_choice() is None and _settings.recipe_interpreter("test_pack", "basic") is None
    assert _settings.recipe_interpreters() == {} and _settings.device() == "cpu"
    w = RunWidget(make_napari_viewer())                        # another computer: the dock asks
    assert w.idle and w.engine is None and w.engine_box.runs_label.text().startswith("Not chosen yet")


def test_what_is_not_a_python_interpreter_is_refused_with_the_fix(tmp_path):
    """A folder, an environment's germquant or napari command, and an executable that does not answer as Python
    get one message naming the fix, never a confusing engine error (or a second napari)."""
    with pytest.raises(RunnerError, match="is a folder, not a Python interpreter. Choose python.exe"):
        SubprocessRunner(tmp_path).version()
    for name in ("germquant.exe", "napari.exe", "germquant", "napari"):
        exe = tmp_path / name
        exe.write_bytes(b"")
        with pytest.raises(RunnerError, match=f"is not a Python interpreter but the {name.split('.')[0]} command"):
            SubprocessRunner(exe).version()
    other = shutil.which("where" if os.name == "nt" else "true")
    if other is not None:
        r = SubprocessRunner(other)
        with pytest.raises(RunnerError, match="is not a Python interpreter \\(it answered: "):
            r.version()
        assert r.history[0][1:3] == ["-I", "-c"] and len(r.history) == 1    # never asked for germquant.packs
    with pytest.raises(RunnerError, match="cannot be started"):
        SubprocessRunner(tmp_path / "missing" / "python.exe").version()


# ---------------------------------------------------------------------------------------------------
# the GUI thread never waits for the engine; Cancel and napari quitting end it
# ---------------------------------------------------------------------------------------------------


@needs_packs
def test_engine_calls_never_block_the_gui_thread(make_napari_viewer, qtbot, tmp_path):
    """The version check, the pack listing and the card check run in the check worker: set_engine and
    refresh_card return at once while a slow engine answers, the event loop keeps running, and the card says it
    is checking (Run off) until the answer is back."""
    own = plugin_engine_info()
    fake = _fake_engine(tmp_path / "slow", version=str(own["germquant"]), git_sha=str(own["git_sha"]),
                        slow={"list": 2.0, "check": 2.0})
    w = RunWidget(make_napari_viewer())
    ticks: list[float] = []
    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    try:
        t0 = time.monotonic()
        w.set_engine(sys.executable, env={"PYTHONPATH": str(fake)})
        assert time.monotonic() - t0 < 0.5 and not w.idle
        qtbot.waitUntil(lambda: bool(w.pack_entries), timeout=60_000)
        assert len(ticks) >= 15                                # the event loop ran through the 2 s listing
        assert w.engine is not None and [e.id for e in w.pack_entries] == ["slow_pack"]
        w.select_pack("slow_pack", "r1")
        assert "checking in" in w.pack_picker.card.text() and not w.run_btn.isEnabled()
        settle(w, qtbot)
        assert w.pack_status["usable"]
        n, t0 = len(ticks), time.monotonic()
        assert w.refresh_card(force=True) is None              # not known yet: being checked
        assert time.monotonic() - t0 < 0.5 and not w.run_btn.isEnabled()
        settle(w, qtbot)
        assert len(ticks) - n >= 15 and w.pack_status["usable"]
    finally:
        timer.stop()


def test_cancel_and_napari_quitting_end_a_hung_engine(make_napari_viewer, qtbot, synthetic_image,
                                                      classical_overlay, tmp_path):
    """A run in a separate process that never ends: Cancel ends the engine and the process it started (the base
    interpreter runs the stand-in, so no venv launcher ends the child for us) and the run log says it was
    cancelled; napari quitting, which asks every thread worker to quit and waits for them, ends the engine too
    instead of waiting forever; Cancel ends a model download the same way."""
    psutil = pytest.importorskip("psutil")
    from superqt.utils import WorkerBase

    def gone(pids):
        return lambda: not any(psutil.pid_exists(p) for p in pids)

    own = plugin_engine_info()
    fake = _fake_engine(tmp_path / "hung", version=_accepted(own["germquant"]), git_sha=str(own["git_sha"]),
                        hang_run=True)
    pid_file = tmp_path / "engine.pid"
    w = _engine_set(RunWidget(make_napari_viewer()), qtbot, BASE_PYTHON,
                    {"PYTHONPATH": str(fake), "FAKE_ENGINE_PID_FILE": str(pid_file)})
    assert w.engine is not None
    w.set_image(str(synthetic_image))
    w.select_config_file(classical_overlay)
    w.stage_checks["spots"].setChecked(False)
    w.out_edit.setText(str(tmp_path / "out"))
    assert not w.cancel_btn.isEnabled()
    w.run_btn.click()
    qtbot.waitUntil(lambda: pid_file.exists() and len(pid_file.read_text(encoding="ascii").split()) == 2,
                    timeout=60_000)
    pids = _pids(pid_file)
    assert all(psutil.pid_exists(p) for p in pids) and w.cancel_btn.isEnabled() and not w.run_btn.isEnabled()
    t0 = time.monotonic()
    w.cancel_btn.click()
    qtbot.waitUntil(lambda: w.last_error is not None and w.run_btn.isEnabled(), timeout=30_000)
    assert time.monotonic() - t0 < 20
    assert isinstance(w.last_error, RunCancelled) and w.status_label.text().startswith("run cancelled")
    qtbot.waitUntil(gone(pids), timeout=15_000)                # the engine and the child it started
    start, failed, end = read_run_log(w.last_run_log)
    assert [start["event"], failed["event"], end["event"]] == ["run_start", "image_failed", "run_end"]
    assert failed["error"].startswith("RunCancelled: cancelled by the user") and not w.cancel_btn.isEnabled()
    assert end["counts"] == {"done": 0, "failed": 1, "skipped": 0}

    pid_file.unlink()
    w.run_btn.click()
    qtbot.waitUntil(lambda: pid_file.exists() and len(pid_file.read_text(encoding="ascii").split()) == 2,
                    timeout=60_000)
    pids = _pids(pid_file)
    t0 = time.monotonic()
    WorkerBase.await_workers(msecs=20_000)                     # what napari does when it quits; raises on a timeout
    assert time.monotonic() - t0 < 20
    qtbot.waitUntil(gone(pids), timeout=15_000)
    qtbot.waitUntil(lambda: w.run_btn.isEnabled(), timeout=30_000)

    # Cancel stops a model download in the engine environment the same way
    pid_file.unlink()
    if HAS_PACKS:
        w.select_pack("slow_pack", "r1")
        settle(w, qtbot)
        assert w.get_models() and w.pack_picker.cancel_btn.isEnabled()
        qtbot.waitUntil(pid_file.exists, timeout=60_000)
        pids = _pids(pid_file)
        w.pack_picker.cancel_btn.click()
        qtbot.waitUntil(lambda: w.last_pack_error is not None and not w.pack_worker.busy, timeout=30_000)
        assert isinstance(w.last_pack_error, RunCancelled) and w.status_label.text().startswith("cancelled")
        assert not w.pack_picker.cancel_btn.isEnabled()
        qtbot.waitUntil(gone(pids), timeout=15_000)


STANDIN_NAPARI = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "from napari_germquant._runner import SubprocessRunner\n"
    "Path(sys.argv[3]).write_text(str(os.getpid()), encoding='ascii')\n"
    "r = SubprocessRunner(sys.executable, env={'PYTHONPATH': sys.argv[1], 'FAKE_ENGINE_PID_FILE': sys.argv[2]})\n"
    "r._stream([sys.executable, '-m', 'germquant', 'run'])\n"
)


@pytest.mark.skipif(os.name != "nt", reason="the job object that ends the engine with napari is Windows only")
def test_killing_napari_ends_its_engine_on_windows(tmp_path):
    """napari killed from Task Manager while an engine runs: the engine sits in a job object that ends with
    napari's handles, so it does not go on writing into the output folder."""
    psutil = pytest.importorskip("psutil")
    own = plugin_engine_info()
    fake = _fake_engine(tmp_path / "hung", version=_accepted(own["germquant"]), git_sha=str(own["git_sha"]),
                        hang_run=True)
    engine_pid, standin_pid = tmp_path / "engine.pid", tmp_path / "napari.pid"
    proc = subprocess.Popen([sys.executable, "-c", STANDIN_NAPARI, str(fake), str(engine_pid), str(standin_pid)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 120
        while not engine_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        deadline = time.monotonic() + 60
        while len(engine_pid.read_text(encoding="ascii").split()) < 2 and time.monotonic() < deadline:
            time.sleep(0.2)
        pids = [int(standin_pid.read_text(encoding="ascii")), *_pids(engine_pid)]
        psutil.Process(pids[0]).kill()                         # the real interpreter of the stand-in napari
        deadline = time.monotonic() + 15
        while any(psutil.pid_exists(p) for p in pids[1:]) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not any(psutil.pid_exists(p) for p in pids[1:]), "the engine outlived the napari that started it"
    finally:
        for pid in [*pids, proc.pid]:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        proc.wait(timeout=30)


# ---------------------------------------------------------------------------------------------------
# a card refresh is torch-free and trusts a verified blob (acceptance 10)
# ---------------------------------------------------------------------------------------------------


@needs_packs
def test_card_refresh_loads_no_torch_and_does_not_rehash_a_verified_blob(make_napari_viewer, qtbot, served_pack):
    w = _engine_set(RunWidget(make_napari_viewer()), qtbot, sys.executable)
    w.select_pack("served_pack", "with_model")
    settle(w, qtbot)
    assert not w.pack_status["usable"]
    w.pack_picker.get_models_btn.click()
    qtbot.waitUntil(lambda: w.last_pack_result is not None or w.last_pack_error is not None, timeout=120_000)
    qtbot.waitUntil(lambda: not w.pack_worker.busy, timeout=30_000)
    settle(w, qtbot)
    assert w.last_pack_result["ok"], w.last_pack_result
    blob = served_pack.env.blobs / TINYNET_SHA / "tinynet"
    record = blob.parent / f"{TINYNET_SHA}.json"
    rec0, mtime0 = json.loads(record.read_text(encoding="utf-8")), blob.stat().st_mtime_ns

    # the card refresh, through the dock's own subprocess runner with Python's import log switched on
    w.engine.interpreter_options = ("-X", "importtime")
    w.refresh_card(force=True)
    settle(w, qtbot)
    st = w.pack_status
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
# run logs (section 7.6) and the device
# ---------------------------------------------------------------------------------------------------


def test_a_failed_run_is_logged_and_its_log_named(make_napari_viewer, qtbot, synthetic_image, classical_overlay,
                                                  tmp_path):
    """A run that fails in the engine (here: its output folder is a file) writes run_start, image_failed and
    run_end, and the dock names the log with the error."""
    w = _engine_set(RunWidget(make_napari_viewer()), qtbot, INPROCESS)
    w.set_image(str(synthetic_image))
    w.select_config_file(classical_overlay)
    w.stage_checks["spots"].setChecked(False)
    blocker = tmp_path / "results_is_a_file"
    blocker.write_text("not a folder", encoding="ascii")
    w.out_edit.setText(str(blocker))
    w.run_btn.click()
    qtbot.waitUntil(lambda: w.last_error is not None or w.last_result is not None, timeout=300_000)
    qtbot.waitUntil(lambda: w.run_btn.isEnabled(), timeout=30_000)
    assert w.last_result is None and w.last_error is not None
    start, failed, end = read_run_log(w.last_run_log)
    assert w.last_run_log.parent == tmp_path / "runs"
    assert [start["event"], failed["event"], end["event"]] == ["run_start", "image_failed", "run_end"]
    assert failed["status"] == "failed" and failed["error"] and end["counts"] == {"done": 0, "failed": 1, "skipped": 0}
    assert "run FAILED" in w.status_label.text() and f"run log {w.last_run_log}" in w.status_label.text()


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
    # an output folder named runs, or a drive root, would get its log inside itself: the per-user folder instead
    named = [tmp_path / "runs"] + ([tmp_path / "proj" / "Runs"] if os.name == "nt" else [])   # NTFS ignores case
    for named_runs in named:
        folder3, where3 = runs_folder_for(named_runs)
        assert folder3 == user_runs_dir() and "itself named runs" in where3
        assert not (named_runs / "runs").exists()
    folder4, _ = runs_folder_for(Path(tmp_path.anchor))
    assert folder4 == user_runs_dir()


def test_run_start_records_napari_s_config_hash_only_for_the_same_engine(tmp_path, synthetic_image,
                                                                        classical_overlay):
    """run_start's config_hash comes from napari's germquant: kept when the engine is the same germquant (version
    and git sha), recorded as null when they differ (image_done carries the run's own), with its source said."""

    class _Stub(EngineRunner):
        kind, via, python = "stub", VIA_SUBPROCESS, "stub-python"

        def __init__(self, info):
            super().__init__()
            self.info = info

        def version(self, refresh=False, should_stop=None):
            return dict(self.info)

        def runtime(self, prefer=None, should_stop=None):
            return {"device_requested": "auto", "device_selected": "cpu"}

        def run(self, plan, say=None, should_stop=None):
            raise RunnerError("a stand-in engine does not run")

    plan = RunPlan(image=synthetic_image, recipe=Recipe.from_file(classical_overlay), out=tmp_path / "out")
    own = plugin_engine_info()
    for name, info, want in (("same", own, "abcdef123456"), ("other", {**own, "git_sha": "0" * 40}, None)):
        with pytest.raises(RunnerError, match="does not run"):
            execute_run(_Stub(info), plan, config_hash="abcdef123456", log_dir=tmp_path / name)
        start = read_run_log(next((tmp_path / name).glob("*.jsonl")))[0]
        assert start["config_hash"] == want, name
        assert start["config_hash_source"].startswith("napari's germquant" if want else "none: napari's germquant "
                                                                                         "is not the engine's"), name


@needs_packs
def test_the_device_choice_reaches_the_engine_and_a_runs_folder_keeps_only_engine_files(
        make_napari_viewer, qtbot, pack_env, tmp_path, monkeypatch):
    """The device combo sets GERMQUANT_DEVICE for the engine process: run_start and the engine's own runtime
    record say cpu was asked for. The output folder is named runs, so its log goes to the per-user folder and
    the folder holds exactly what `germquant run` writes."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))       # the per-user folder on a Mac lives under HOME
    w = _engine_set(RunWidget(make_napari_viewer()), qtbot, sys.executable)
    combo = w.engine_box.device_combo
    combo.setCurrentIndex(combo.findData("cpu"))
    assert w.engine.device == "cpu" and _settings.device() == "cpu"
    w.select_pack("test_pack", "basic")
    image = tmp_path / "image" / TEST_IMAGE.name
    image.parent.mkdir()
    shutil.copyfile(TEST_IMAGE, image)
    w.set_image(str(image))
    settle(w, qtbot)
    out = tmp_path / "runs"
    w.out_edit.setText(str(out))
    w.run_btn.click()
    res = _run(w, qtbot)
    start = read_run_log(w.last_run_log)[0]
    assert start["device_requested"] == "cpu" and start["device_selected"] == "cpu"
    stages = json.loads((out / f"{res['image_id']}__stages.json").read_text(encoding="utf-8"))
    assert stages["runtime"]["device_requested"] == "cpu", stages.get("runtime")
    assert w.last_run_log.parent == user_runs_dir() and "itself named runs" in start["log_location"]
    argv = w.plan().run_argv()
    argv[argv.index("--out") + 1] = str(tmp_path / "cli")
    assert cli.main(argv) == 0
    assert sorted(f.name for f in out.iterdir()) == sorted(f.name for f in (tmp_path / "cli").iterdir())
