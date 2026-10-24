"""The engine session the project docks share (tranche 5 Train B, unit P6-RUNNER), the settings it adds (recent
projects per computer, the annotator per user), the tests' one feature gate (skip, or fail when
$NAPARI_GERMQUANT_REQUIRE_FEATURES requires the feature) and the shared fixtures project_env, seams_runner and
served_model_pack."""
from __future__ import annotations

import gc
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from _p6_runner import SEAMS_WRAPPER, standin_engine
from conftest import FIXTURE_PACKS, REQUIRE_FEATURES_ENV, TINYNET, need_features

from napari_germquant import _engine, _engine_session, _settings
from napari_germquant._engine import feature_missing_message
from napari_germquant._engine_session import EngineNotChosen, EngineSession, session_for
from napari_germquant._runner import (
    VIA_INPROCESS,
    FeatureMissing,
    InProcessRunner,
    RunnerError,
    SubprocessRunner,
    engine_too_old,
    plugin_engine_info,
)
from napari_germquant._settings import INPROCESS
from germquant import api


def _standin(tmp_path: Path, name: str, features) -> Path:
    own = plugin_engine_info()
    version = _engine.MIN_ENGINE if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    return standin_engine(tmp_path / name, version=version, git_sha=str(own["git_sha"]), features=features)


class _Viewer:
    """Stands in for a napari viewer: the session is keyed on the object (napari's public proxy unwrapped)."""


def test_one_engine_session_per_viewer_shared_by_the_docks(make_napari_viewer):
    from napari.utils._proxies import PublicOnlyProxy

    viewer = make_napari_viewer()
    s = session_for(viewer)
    assert session_for(viewer) is s and session_for(PublicOnlyProxy(viewer)) is s   # the Plugins menu route
    other = _Viewer()
    assert session_for(other) is not s and session_for(other) is session_for(other)
    assert session_for(None) is not session_for(None)                  # no viewer, no sharing
    key = id(other)
    assert key in _engine_session._SESSIONS
    del other
    gc.collect()
    assert key not in _engine_session._SESSIONS                        # dropped with its viewer


def test_the_session_checks_the_stored_engine_and_gates_features(tmp_path, qtbot):
    s = EngineSession()
    with pytest.raises(EngineNotChosen, match="Choose the engine"):
        s.ensure()
    assert s.runner() is None and s.describe().startswith("Not chosen yet")
    assert s.missing("projects") == s.describe()
    # an engine from before the features list, stored for this computer (what the Run dock stores)
    old = _standin(tmp_path, "trainA", None)
    _settings.set_engine_choice(sys.executable)
    s.set_env({"PYTHONPATH": str(old)})
    with qtbot.waitSignal(s.changed, timeout=30_000):
        r = s.ensure()
    assert isinstance(r, SubprocessRunner) and s.runner() is r and s.ensure() is r     # checked once
    assert r.python == sys.executable and r.env["PYTHONPATH"] == str(old)
    assert s.features() == frozenset({"packs"}) and s.missing("packs") is None
    msg = feature_missing_message("projects", info=r.cached_version(), interpreter=sys.executable)
    assert s.missing("projects") == msg and s.missing("packs", "projects") == msg
    with pytest.raises(FeatureMissing) as e:
        s.require("projects")
    assert str(e.value) == msg
    assert s.describe().startswith(f"{sys.executable} (a separate process); germquant ") and "device auto" in \
        s.describe()
    # a new environment (every feature): the next check makes a new runner
    new = _standin(tmp_path, "trainB", list(_engine.FEATURES))
    s.set_env({"PYTHONPATH": str(new)})
    assert s.runner() is None
    r2 = s.require("projects", "tune_spots")
    assert r2 is not r and s.missing("projects", "post_stages") is None
    # the device is the per-user choice the Run dock also uses
    with qtbot.waitSignal(s.changed, timeout=5_000):
        s.set_device("cpu")
    assert _settings.device() == "cpu" and s.device() == "cpu" and r2.device == "cpu" and "device cpu" in \
        s.describe()
    with pytest.raises(ValueError, match="device"):
        s.set_device("gpu")
    # a pack recipe with its own interpreter on this computer runs there; others go to the engine
    _settings.set_recipe_interpreter("test_pack", "basic", sys.executable)
    rr = s.runner_for("test_pack", "basic")
    assert rr is not r2 and rr.python == sys.executable and s.runner_for("test_pack", "basic") is rr
    assert s.runner_for("test_pack", "with_model") is r2 and s.runner_for() is r2 and rr.device == "cpu"
    # a choice stored elsewhere (the Run dock's dialog) is picked up at the next check
    _settings.set_engine_choice(INPROCESS)
    assert s.runner() is None
    r3 = s.ensure()
    assert isinstance(r3, InProcessRunner) and r3.via == VIA_INPROCESS and r3.device == "cpu"


def test_set_engine_stores_the_choice_only_after_its_check(tmp_path):
    s = EngineSession()
    with pytest.raises(RunnerError, match="is a folder, not a Python interpreter"):
        s.set_engine(tmp_path)
    assert _settings.engine_choice() is None and s.runner() is None
    s.set_env({"PYTHONPATH": str(_standin(tmp_path, "eng", list(_engine.FEATURES)))})
    r = s.set_engine(sys.executable)
    assert _settings.engine_choice() == sys.executable and s.runner() is r and s.error() is None
    assert s.info()["git_sha"] == plugin_engine_info()["git_sha"] and s.warning() == ""


def test_recent_projects_are_kept_per_computer_and_the_annotator_per_user(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "Lab PC/1")
    assert _settings.recent_projects() == []
    unc = "\\\\129.82.125.105\\SpermBank\\projects\\p2"
    for p in ("C:/data/p1", unc, "C:/data/p3"):
        _settings.add_recent_project(p)
    assert _settings.recent_projects() == ["C:/data/p3", unc, "C:/data/p1"]        # newest first, UNC verbatim
    assert _settings.add_recent_project("C:/data/p1") == ["C:/data/p1", "C:/data/p3", unc]
    for i in range(20):
        _settings.add_recent_project(f"C:/data/many/{i}")
    assert len(_settings.recent_projects()) == _settings.RECENT_MAX and _settings.recent_projects()[0] == \
        "C:/data/many/19"
    _settings.remove_recent_project("C:/data/many/19")
    assert "C:/data/many/19" not in _settings.recent_projects()
    assert _settings.annotator() is None
    _settings.set_annotator("ryan_e-1")
    for bad in ("ry an", "", "x" * 33, "r" + chr(0x00E9), "a/b"):
        with pytest.raises(ValueError, match="annotator"):
            _settings.set_annotator(bad)
    monkeypatch.setattr(socket, "gethostname", lambda: "laptop")
    assert _settings.recent_projects() == [] and _settings.annotator() == "ryan_e-1"   # per computer, per user
    _settings.set_annotator(None)
    assert _settings.annotator() is None


def test_a_missing_feature_skips_unless_required(monkeypatch):
    monkeypatch.delenv(REQUIRE_FEATURES_ENV, raising=False)
    need_features("packs", have={"packs"})                            # present: nothing happens
    with pytest.raises(pytest.skip.Exception, match="the engine lacks the feature projects"):
        need_features("projects", have={"packs"})
    with pytest.raises(pytest.skip.Exception, match="the engine lacks the features projects, tune_spots"):
        need_features("projects", "tune_spots", have={"packs"})
    monkeypatch.setenv(REQUIRE_FEATURES_ENV, "projects, tune_spots")
    with pytest.raises(pytest.fail.Exception, match="NAPARI_GERMQUANT_REQUIRE_FEATURES requires projects"):
        need_features("projects", have={"packs"})
    with pytest.raises(pytest.skip.Exception):                        # not listed: still a skip
        need_features("post_stages", have={"packs"})
    need_features("projects", have=set(_engine.FEATURES))
    monkeypatch.setenv(REQUIRE_FEATURES_ENV, "projets")
    with pytest.raises(pytest.fail.Exception, match="unknown features"):
        need_features("projects", have={"packs"})
    with pytest.raises(ValueError, match="not engine features"):
        need_features("no_such_feature", have=set())


@pytest.mark.skipif(not hasattr(api, "list_packs"), reason="system packs need germquant 0.2.0.dev0 or later")
def test_served_model_pack_needs_its_served_model_and_declares_the_radius(served_model_pack):
    from germquant import packs

    p = packs.find_pack(served_model_pack.ref)
    assert packs.validate_pack(p) == [] and packs.verify_pack(p.folder) == []
    basic = p.recipe("basic")
    assert served_model_pack.tunable in [t["key"] for t in basic.tunables]
    ids = [m["id"] for m in packs.list_packs()]
    assert "served_model_pack" in ids and "test_pack" not in ids             # the copy replaces the test pack
    r = InProcessRunner()
    st = r.check(served_model_pack.ref, "basic")
    assert not st["usable"] and [m["state"] for m in st["models"]] == ["missing"]
    res = r.fetch_models(served_model_pack.ref, "basic")
    assert res["ok"] and [m["name"] for m in res["models"]] == [TINYNET]
    assert any(path == "models/tinynet" for path, _rng in served_model_pack.log)    # fetched from the server
    assert r.check(served_model_pack.ref, "basic")["usable"]
    cfg = api.resolve_config(pack=served_model_pack.ref, recipe="basic")
    assert cfg.hash == basic.expected_config_hash == "02bb83cf74f9"                  # pack.yaml is not hashed
    assert cfg.get("spots.spot_radius_um") == 0.3


def test_project_env_gives_a_short_project_root(project_env):
    assert project_env.root.is_dir() and len(str(project_env.root)) < 90
    assert project_env.image.is_file() and project_env.image.parent == project_env.images
    p = project_env.new_dir()
    assert not p.exists() and p.parent == project_env.root and project_env.new_dir() != p
    assert os.environ["GERMQUANT_PACKS"] == str(FIXTURE_PACKS) and Path(os.environ["GERMQUANT_MODELS"]) == \
        project_env.models


def _pairs(folder: Path) -> Path:
    """Two images' worth of 2D slice pairs (nine box nuclei each), named as export-training names them."""
    for d in ("image", "labels"):
        (folder / d).mkdir(parents=True)
    for i, iid in enumerate(("20250101_n2_nohs_HERM_001", "20250101_n2_nohs_HERM_002")):
        for k in range(2):
            lab = np.zeros((48, 48), np.int32)
            n = 1
            for cy in (10, 24, 38):
                for cx in (10, 24, 38):
                    lab[cy - 4 + i:cy + 4 + i, cx - 4 + k:cx + 4 + k] = n
                    n += 1
            img = ((lab > 0) * 1000 + 20 * (i + k)).astype(np.uint16)
            tifffile.imwrite(folder / "image" / f"{iid}_xy_{k}.tif", img)
            tifffile.imwrite(folder / "labels" / f"{iid}_xy_{k}.tif", lab)
    return folder


def test_the_seams_runner_trains_in_an_engine_child_with_the_seams_replaced(seams_runner, tmp_path):
    """train-nuclei through the seams runner: the command starts in a child process through the wrapper, the
    stand-in seams write the "model" and the engine writes its card around it; nothing is trained for real."""
    pairs = _pairs(tmp_path / "pairs")
    out = tmp_path / "models" / "m1"
    said: list[str] = []
    res = seams_runner.train_nuclei(pairs, out, base="none", epochs=2, split_by="image", device="cpu",
                                    say=said.append)
    argv = seams_runner.history[-1]
    assert argv[:3] == [sys.executable, "-c", SEAMS_WRAPPER] and argv[3:5] == ["train-nuclei", "--pairs"]
    assert res["exit_code"] == 0, res["lines"][-20:]
    assert (out / "m1").read_bytes() == b"fake cellpose weights " * 64
    card = json.loads((out / "m1.card.json").read_text(encoding="utf-8"))
    assert Path(res["model_path"]) == out / "m1" and Path(res["card_path"]) == out / "m1.card.json"
    assert card["base_model"]["kind"] == "random_init"
    assert any("train_loss=0.5000" in line for line in said)            # the stand-in's Cellpose epoch line
    assert len(seams_runner.history) == 1                                # training needs no feature check
