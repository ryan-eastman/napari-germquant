"""The engine runners' commands of tranche 5 Train B (unit P6-RUNNER; the interface of section 9 of the Train B
plan): one method per command, each building the command line the engine documents (stand-in engines from
tests/_p6_runner.py answer with what they were given), the torch-free entries used where they exist, the
plugin version stamped on every project write, ``project=`` on the pack calls, the device list of the runtime
probe, the refusal forms of the Train B commands, and the in-process runner: the same commands through
germquant.api, each error a refusal or a failure as that command answers it, but its project writes through the
torch-free command of this environment (and, on an engine with projects, both runners writing the same project
files, an Apply's record with the engine's warning, and the same refusal of a project file that is not YAML). The
two post-stage commands, ``python -m germquant.post_stages list --json`` and ``write``: their exit codes through
the subprocess runner, the in-process write step by step as the command takes it, and, on an engine with
post-stages, both runners listing, writing, refusing and failing alike."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import types
import warnings
from pathlib import Path

import pytest
from _p6_runner import echo, standin_engine
from conftest import TEST_PACK, copy_pack, engine_device, need_features

from napari_germquant import _engine
from napari_germquant._runner import (
    VIA_INPROCESS,
    EngineRefused,
    EngineRunner,
    FeatureMissing,
    InProcessRunner,
    RunCancelled,
    RunLog,
    RunnerError,
    SubprocessRunner,
    device_choices,
    engine_too_old,
    plugin_engine_info,
    plugin_version,
    set_value_text,
)
from germquant import api

MIN = _engine.MIN_ENGINE
RID = "20261023T120000Z_0123abcd"
GAMMA = chr(0x03B3)          # a channel named with a Greek gamma (gammaH2AX), kept out of this file's bytes


@pytest.fixture
def standin(tmp_path):
    """A subprocess runner on a stand-in engine with every Train B feature."""
    own = plugin_engine_info()
    version = MIN if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    root = standin_engine(tmp_path / "engine", version=version, git_sha=str(own["git_sha"]),
                          features=list(_engine.FEATURES))
    return SubprocessRunner(sys.executable, env={"PYTHONPATH": str(root)})


def test_one_method_per_command_builds_the_documented_command_line(standin, tmp_path):
    r, py, v = standin, sys.executable, plugin_version()
    P = str(tmp_path / "proj")

    def last(module: str) -> list[str]:
        argv = r.history[-1]
        assert argv[:3] == [py, "-m", module], argv
        return argv[3:]

    # the torch-free project entry: every write stamps the plugin version (D42)
    doc = r.project_new(P, "test_pack@0.1.0", recipe="basic", images=["a.ome.tif", "b.nd2"], name="Lab project")
    assert last("germquant.project") == ["new", P, "--pack", "test_pack@0.1.0", "--recipe", "basic", "--images",
                                         "a.ome.tif", "b.nd2", "--name", "Lab project", "--plugin-version", v,
                                         "--json"]
    assert doc["root"] == P and doc["argv"] == r.history[-1][3:]
    r.project_add(P, ["c.ome.tif", tmp_path / "folder"])
    assert last("germquant.project") == ["add", P, "c.ome.tif", str(tmp_path / "folder"), "--plugin-version", v,
                                         "--json"]
    r.project_add(P, "C:/data/d.ome.tif")                    # one path, not a string split into letters
    assert last("germquant.project") == ["add", P, "C:/data/d.ome.tif", "--plugin-version", v, "--json"]
    r.project_new(P, "test_pack@0.1.0", images=tmp_path / "e.nd2")
    assert last("germquant.project") == ["new", P, "--pack", "test_pack@0.1.0", "--images", str(tmp_path / "e.nd2"),
                                         "--plugin-version", v, "--json"]
    doc = r.project_edit(P, edits={"spots.log.min_snr": 2.5, "segmentation.nuclei.method": "classical",
                                 "qc.flag": True, "spots.text": "1.0"}, unset=["spots.log.diameter_um"],
                         roles={"foci": GAMMA + "H2AX", "dna": "H2B mCherry"}, revert=False, device="cuda",
                         xy_stride=2, z_range=(3, 9), runner="subprocess",
                         interpreters={"basic": "C:/envs/py311/python.exe"}, samples="samples.csv",
                         disable=["spots"], dry_run=True)
    assert last("germquant.project") == [
        "edit", P, "--set", "spots.log.min_snr=2.5", "--set", "segmentation.nuclei.method=classical",
        "--set", "qc.flag=true", "--set", 'spots.text="1.0"', "--unset", "spots.log.diameter_um",
        "--role", "foci=" + GAMMA + "H2AX", "--role", "dna=H2B mCherry", "--device", "cuda", "--xy-stride", "2",
        "--z-range", "3", "9", "--runner", "subprocess", "--interpreter", "basic=C:/envs/py311/python.exe",
        "--samples", "samples.csv", "--no-spots", "--plugin-version", v, "--dry-run", "--json"]
    assert doc["argv"] == r.history[-1][3:]               # a non-ASCII channel name reached the engine intact
    r.project_edit(P, revert=True)
    assert last("germquant.project") == ["edit", P, "--revert", "--plugin-version", v, "--json"]
    r.project_use_model(P, "training/r01/models/m1/m1", card="training/r01/models/m1/m1.card.json")
    assert last("germquant.project") == ["use-model", P, "--model", "training/r01/models/m1/m1", "--card",
                                         "training/r01/models/m1/m1.card.json", "--plugin-version", v, "--json"]
    r.project_rounds(P)
    assert last("germquant.project") == ["rounds", P, "--json"]
    r.project_rounds(P, new=True)
    assert last("germquant.project") == ["rounds", P, "--new", "--plugin-version", v, "--json"]
    r.project_rounds(P, new="r07")
    assert last("germquant.project") == ["rounds", P, "--new", "r07", "--plugin-version", v, "--json"]
    with pytest.raises(ValueError, match="round name"):
        r.project_rounds(P, new="round 7")
    res = r.project_unlock(P, host="LAB-PC", pid=4242)
    assert last("germquant.project") == ["unlock", P, "--host", "LAB-PC", "--pid", "4242"]
    assert res["removed"] is True and res["exit_code"] == 0

    # the full command line
    r.project_status(P)
    assert last("germquant") == ["project", "status", P, "--json"]
    r.project_collect(P)
    assert last("germquant") == ["project", "collect", P, "--json"]
    out = r.project_export_commands(P, shell="ps1", shards=3, out=tmp_path / "cmds.ps1")
    assert last("germquant") == ["project", "export-commands", P, "--shell", "ps1", "--shards", "3", "--out",
                                 str(tmp_path / "cmds.ps1")]
    # with --out the commands are the file the engine wrote, not the line it printed ("wrote FILE")
    assert out["out"] == str(tmp_path / "cmds.ps1") and echo(out["text"].splitlines())["module"] == "germquant"
    assert out["text"] == (tmp_path / "cmds.ps1").read_bytes().decode("ascii") and "wrote" not in out["text"]
    assert echo(r.project_export_commands(P)["text"].splitlines())["argv"][-2:] == ["--shell", "sh"]
    r.device = "cpu"
    res = r.project_run(P, images=["a", "b"], resume=False, shard="2/3", run_id=RID)
    assert last("germquant") == ["project", "run", P, "--images", "a", "b", "--no-resume", "--device", "cpu",
                                 "--shard", "2/3", "--run-id", RID, "--via", "gui-subprocess"]
    assert res["run_id"] == RID and res["state"] == "done"
    with pytest.raises(ValueError, match="run id"):
        r.project_run(P, run_id="not-a-run-id")
    r.device = "auto"
    r.import_labels("seg.npz", "cellacdc-npz", "img.ome.tif", "pairs", allow_pickle=True, frame=2)
    assert last("germquant") == ["import-labels", "seg.npz", "--format", "cellacdc-npz", "--image", "img.ome.tif",
                                 "--pairs", "pairs", "--allow-pickle", "--frame", "2", "--json"]
    r.import_labels("seg.npz", "cellacdc-npz", "img.ome.tif", "pairs", channel="Cy3 RAD-51")
    assert last("germquant") == ["import-labels", "seg.npz", "--format", "cellacdc-npz", "--image", "img.ome.tif",
                                 "--pairs", "pairs", "--channel", "Cy3 RAD-51", "--json"]
    with pytest.raises(ValueError, match="format"):
        r.import_labels("seg.zip", "zip", "img.ome.tif", "pairs")
    said: list[str] = []
    res = r.export_training("results", "pairs", orientations=("xy", "xz"), slices_per_image=3, seed=0,
                            min_labels=3, max_gb=2.0, zone_column="zone", say=said.append)
    assert last("germquant") == ["export-training", "results", "--out", "pairs", "--orientations", "xy,xz",
                                 "--slices-per-image", "3", "--seed", "0", "--zone-column", "zone",
                                 "--min-labels", "3", "--max-gb", "2"]
    assert res["exit_code"] == 0 and echo(said)["argv"] == r.history[-1][3:]
    r.export_training("results", "pairs", orientations="auto")
    assert last("germquant") == ["export-training", "results", "--out", "pairs", "--orientations", "auto"]
    res = r.train_nuclei("pairs", "models/m1", base="cpsam_v2@0f1cc3f7", name="m1", epochs=2, split_by="image",
                         val_frac=0.25, seed=1, device="cuda", system="C. elegans germline", kind="nucleus",
                         dataset="lab_set_1")
    assert last("germquant") == ["train-nuclei", "--pairs", "pairs", "--out", "models/m1", "--base",
                                 "cpsam_v2@0f1cc3f7", "--name", "m1", "--epochs", "2", "--split-by", "image",
                                 "--val-frac", "0.25", "--seed", "1", "--system", "C. elegans germline", "--kind",
                                 "nucleus", "--dataset", "lab_set_1"]
    assert echo(res["lines"])["device_env"] == "cuda"     # an accelerator goes through GERMQUANT_DEVICE
    res = r.train_nuclei("pairs", "models/m2", device="cpu")
    assert last("germquant") == ["train-nuclei", "--pairs", "pairs", "--out", "models/m2", "--base", "cpsam",
                                 "--cpu"]
    assert echo(res["lines"])["device_env"] is None
    res = r.train_curve("pairs", "curve", sizes=[1, 2, 4, "all"], repeats=3, seed=7, base="tinynet@f5055f8a",
                        keep_models="best")
    assert last("germquant") == ["train-curve", "--pairs", "pairs", "--out", "curve", "--sizes", "1,2,4,all",
                                 "--repeats", "3", "--seed", "7", "--base", "tinynet@f5055f8a", "--keep-models",
                                 "best", "--json"]
    assert res["device_env"] is None                          # auto: the engine chooses
    # every model of a curve trains on the device asked for, else the runner's, through GERMQUANT_DEVICE
    assert r.train_curve("pairs", "curve", sizes="all", repeats=1, seed=0, base="none",
                         device="cuda")["device_env"] == "cuda"
    r.device = "cpu"
    assert r.train_curve("pairs", "curve", sizes="all", repeats=1, seed=0, base="none")["device_env"] == "cpu"
    res = r.train_nuclei("pairs", "models/m3")                 # train-nuclei follows the runner's device too
    assert last("germquant")[-1] == "--cpu" and echo(res["lines"])["device_env"] is None
    r.device = "auto"
    r.tune_spots(P, points="annotations", grid="grid.yaml", max_runs=20, out="tuning/t1")
    assert last("germquant") == ["tune-spots", "--project", P, "--points", "annotations", "--grid", "grid.yaml",
                                 "--max-runs", "20", "--out", "tuning/t1", "--json"]
    res = r.tune_apply(P, "tuning/t1")
    assert last("germquant") == ["tune-spots", "--project", P, "--apply", "tuning/t1", "--plugin-version",
                                 plugin_version(), "--json"]
    assert res["argv"] == last("germquant")       # the stand-in echoes its arguments as the --json record
    res = r.info("img.ome.tif")
    assert last("germquant") == ["info", "img.ome.tif"] and echo(res["lines"])["argv"] == ["info", "img.ome.tif"]
    r.collect("results", samples="samples.csv")
    assert last("germquant") == ["collect", "results", "--samples", "samples.csv"]
    r.batch("folder", options=["--profile", "rad51_foci", "--out", "out"], samples="s.csv", post_stages="p.yaml")
    assert last("germquant") == ["batch", "folder", "--profile", "rad51_foci", "--out", "out", "--samples", "s.csv",
                                 "--post-stages", "p.yaml"]

    # the torch-free packs entry: derive, and --project on list, check and fetch-models (I1)
    r.pack_derive("test_pack@0.1.0", "basic", "models/m1/m1", "models/m1/m1.card.json", P + "/packs",
                  new_id="lab_basic")
    assert last("germquant.packs") == ["derive", "--from", "test_pack@0.1.0", "--recipe", "basic", "--model",
                                       "models/m1/m1", "--card", "models/m1/m1.card.json", "--out", P + "/packs",
                                       "--id", "lab_basic", "--json"]
    r.list_packs(project=P)
    assert last("germquant.packs") == ["list", "--json", "--all", "--project", P]
    r.check("test_pack@0.1.0", "basic", image="img.ome.tif", project=P)
    assert last("germquant.packs") == ["check", "test_pack@0.1.0", "--recipe", "basic", "--image", "img.ome.tif",
                                       "--project", P, "--json"]
    r.fetch_models("test_pack@0.1.0", "basic", project=P)
    assert last("germquant.packs") == ["fetch-models", "test_pack@0.1.0", "--recipe", "basic", "--project", P]
    r.list_packs()
    assert last("germquant.packs") == ["list", "--json", "--all"]              # P4's listing, unchanged

    # the torch-free post-stage entry
    r.list_post_stages()
    assert last("germquant.post_stages") == ["list", "--json"]
    entries = [{"name": "fixture_post", "params": {"radius_um": 1.5}}]
    doc = r.write_post_stages(P, entries)
    assert last("germquant.post_stages") == ["write", P, "--entries", '[{"name":"fixture_post","params":'
                                                                       '{"radius_um":1.5}}]', "--json"]
    assert json.loads(doc["argv"][3]) == entries
    assert r.history[0][3:] == ["--version", "--json"] and sum(a[3:] == ["--version", "--json"]
                                                               for a in r.history) == 1   # one version check
    for name in ("list_packs", "check", "fetch_models"):      # the interface names project= as both runners take it
        assert "project" in inspect.signature(getattr(EngineRunner, name)).parameters, name


def test_set_values_are_written_so_the_engine_reads_them_back_with_their_type():
    """``--set KEY=VALUE``: YAML (and, for every quoted string, JSON too) reads VALUE back as the same value of the
    same type; the ASCII escape keeps a non-ASCII value out of the command line's bytes."""
    import yaml

    for value, text in ((2.5, "2.5"), (1e-5, "1.0e-05"), (3, "3"), (True, "true"), (False, "false"),
                        (None, "null"), ("classical", "classical"), ("1.0", '"1.0"'), ("yes", '"yes"'),
                        ("", '""'), ("a: b", '"a: b"'), (" padded", '" padded"'),
                        (GAMMA + "H2AX", '"\\u03b3H2AX"')):
        assert set_value_text(value) == text, value
        assert yaml.safe_load(text) == value and type(yaml.safe_load(text)) is type(value), value
        if text.startswith('"'):
            assert json.loads(text) == value
    for bad in (float("nan"), float("inf"), [1, 2], {"a": 1}):
        with pytest.raises(ValueError, match="scalar"):
            set_value_text(bad)


def test_every_call_takes_should_stop(standin, tmp_path):
    """A call asked to stop before it starts runs nothing (as every P4 call does)."""
    n = len(standin.history)
    for call in (lambda s: standin.project_status(tmp_path, should_stop=s),
                 lambda s: standin.project_new(tmp_path / "p", "test_pack@0.1.0", should_stop=s),
                 lambda s: standin.train_nuclei("pairs", "m", should_stop=s),
                 lambda s: standin.write_post_stages(tmp_path, [], should_stop=s)):
        with pytest.raises(RunCancelled):
            call(lambda: True)
    assert len(standin.history) == n


def test_the_runtime_probe_lists_the_devices():
    """runtime() adds germquant.device.available() for the Training dock's device list, through both runners;
    the device the engine picks is unchanged."""
    for r in (InProcessRunner(), SubprocessRunner(sys.executable)):
        rt = r.runtime()
        assert set(rt["available"]) == {"cuda", "mps"} and rt["device_selected"] == engine_device()
        choices = device_choices(rt)
        assert choices[0] == "auto" and choices[-1] == "cpu"
        assert [d for d in ("cuda", "mps") if d in choices] == [d for d in ("cuda", "mps") if rt["available"][d]]
    assert device_choices({"available": {"cuda": True, "mps": False}}) == ["auto", "cuda", "cpu"]
    assert device_choices(None) == ["auto", "cpu"] == device_choices({"device_selected": "cpu"})


@pytest.mark.skipif(not hasattr(api, "list_packs"), reason="system packs need germquant 0.2.0.dev0 or later")
def test_project_packs_through_the_in_process_runner(pack_env, tmp_path, monkeypatch):
    """list_packs, check and fetch_models with ``project=`` search <project>/packs first in this environment
    (packs.use_project in the calling thread, I11); without it the project's pack is not seen."""
    monkeypatch.setattr(api, "open_project", lambda root: None, raising=False)      # the projects gate
    project = tmp_path / "proj"
    copy_pack(TEST_PACK, project / "packs" / "lab_pack" / "0.1.0", edit=lambda d: d.update(id="lab_pack"))
    r = InProcessRunner()
    listing = r.list_packs(project=project)
    assert {"kind": "project", "path": str(project / "packs")} in listing["roots"]
    assert [p["source"] for p in listing["packs"] if p["id"] == "lab_pack"] == ["project"]
    assert "lab_pack" not in {p["id"] for p in r.list_packs()["packs"]}
    st = r.check("lab_pack@0.1.0", "basic", project=project)
    assert st["usable"] and st["pack"] == "lab_pack"
    assert not r.check("lab_pack@0.1.0", "basic")["usable"]
    res = r.fetch_models("lab_pack@0.1.0", "basic", project=project)
    assert res["ok"] and res["models"] == []                         # basic needs no model
    with pytest.raises(RunnerError, match="no pack"):
        r.fetch_models("lab_pack@0.1.0", "basic")


class _FakeProject:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = {"name": "fake", "format": 1}
        self.images: list = []
        self.pack = {"id": "test_pack", "version": "0.1.0"}
        self.warnings: list = []


def test_the_in_process_runner_calls_the_project_api(project_env, monkeypatch):
    """The commands that are no project write, in this napari environment: germquant.api's project functions,
    with the project passed explicitly (I11), and project runs through api.run_project with via=gui-inprocess,
    the run id, a progress callback that tails the engine's log, and should_stop. (The six project writes run
    the command of this environment: test_in_process_project_writes_run_the_command_of_this_environment.)"""
    calls: list[tuple] = []

    def fake(name, ret=None):
        def f(*a, **k):
            calls.append((name, a, k))
            return ret(*a, **k) if callable(ret) else ret
        return f

    def run_project(project, images=None, resume=True, device=None, shard=None, progress=None, *, via="cli",
                    run_id=None, should_stop=None):
        calls.append(("run_project", (project,), {"images": images, "resume": resume, "device": device,
                                                  "shard": shard, "via": via, "run_id": run_id}))
        log = RunLog(Path(project.root) / "runs", run_id)
        log.path.parent.mkdir(parents=True, exist_ok=True)
        log.event("run_start", via=via, project=str(project.root), shard=shard, images=[])
        progress({"image_id": "a"})                                   # whatever it passes is not read
        log.event("image_done", image_id="a", status="done")
        progress({"image_id": "a"})
        log.event("run_end", counts={"done": 1, "failed": 0, "skipped": 0}, interrupted=False)
        return {"run_id": run_id, "log": str(log.path), "counts": {"done": 1, "failed": 0, "skipped": 0},
                "exit_code": 0}

    for name, ret in {"open_project": _FakeProject, "project_status": {"format": 1, "images": []},
                      "collect_project": {"tables": []}, "export_commands": "germquant project run X\n",
                      "import_labels": {"pairs": 2}, "tune_spots": {"run_dir": "tuning/t1", "best": {}, "ranked": []},
                      "apply_tuning": "fedcba987654",
                      "list_post_stages": [{"name": "fixture_post"}]}.items():
        monkeypatch.setattr(api, name, fake(name, ret), raising=False)
    monkeypatch.setattr(api, "run_project", run_project, raising=False)
    writes: list = []
    fake_post = types.ModuleType("germquant.post_stages")          # what the post-stage write takes of it
    fake_post.POST_STAGES_FILE = "post_stages.yaml"
    fake_post.validate_post_stages = lambda entries, declarations=None: []
    fake_post.write_post_stages_file = lambda path, entries, **k: writes.append((Path(path), entries)) or Path(path)
    fake_post.post_stages_sha256 = lambda source, declarations=None: "ab" * 32
    fake_post.load_post_stages_file = lambda path: [{"name": "fixture_post", "params": {}}]
    monkeypatch.setitem(sys.modules, "germquant.post_stages", fake_post)
    try:                                   # the warning class an Apply records (9.4 #7)
        from germquant.tune import TuningWarning
    except ImportError:                    # an engine before P5-TUNE: a stand-in germquant.tune
        TuningWarning = type("TuningWarning", (UserWarning,), {})
        fake_tune = types.ModuleType("germquant.tune")
        fake_tune.TuningWarning = TuningWarning
        monkeypatch.setitem(sys.modules, "germquant.tune", fake_tune)

    r = InProcessRunner()
    assert {"projects", "import_labels", "tune_spots", "post_stages"} <= r.features()
    P = project_env.new_dir()
    assert r.project_status(P) == {"format": 1, "images": []}
    assert calls[-2][0] == "open_project" and calls[-2][1] == (P,)
    assert isinstance(calls[-1][1][0], _FakeProject) and calls[-1][1][0].root == Path(P)
    assert r.project_collect(P) == {"tables": []}
    out = r.project_export_commands(P, shell="ps1", out=project_env.root / "cmds.ps1")
    assert out["text"] == "germquant project run X\n"
    assert (project_env.root / "cmds.ps1").read_bytes() == b"\xef\xbb\xbf" + out["text"].encode("ascii")   # ps1: a BOM
    assert [f.name for f in project_env.root.iterdir() if f.name.endswith(".part")] == []    # written atomically
    seen: list[dict] = []
    res = r.project_run(P, images=["a"], resume=False, device="cpu", shard="1/2", run_id=RID, progress=seen.extend)
    rp = next(c for c in reversed(calls) if c[0] == "run_project")
    assert rp[2] == {"images": ["a"], "resume": False, "device": "cpu", "shard": "1/2", "via": VIA_INPROCESS,
                     "run_id": RID}
    assert [e["event"] for e in seen] == ["run_start", "image_done", "run_end"]
    assert res["state"] == "done" and res["counts"]["done"] == 1 and res["log"] == str(Path(P) / "runs" /
                                                                                         f"{RID}.jsonl")
    assert r.import_labels("seg.npz", "cellacdc-npz", "img.tif", "pairs", frame=1) == {"pairs": 2}
    assert calls[-1] == ("import_labels", ("seg.npz", "cellacdc-npz", "img.tif", "pairs"),
                         {"allow_pickle": False, "frame": 1})
    assert r.import_labels("seg.npz", "cellacdc-npz", "img.tif", "pairs", channel=1) == {"pairs": 2}
    assert calls[-1] == ("import_labels", ("seg.npz", "cellacdc-npz", "img.tif", "pairs"),
                         {"allow_pickle": False, "channel": 1})
    # the text the command line reads as an index is an index here too (the dock sends "1" for a channel whose
    # name is not unique); any other text is a channel name
    r.import_labels("seg.npz", "cellacdc-npz", "img.tif", "pairs", channel=" 1 ")
    assert calls[-1][2] == {"allow_pickle": False, "channel": 1}
    r.import_labels("seg.npz", "cellacdc-npz", "img.tif", "pairs", channel="Cy3 RAD-51")
    assert calls[-1][2] == {"allow_pickle": False, "channel": "Cy3 RAD-51"}
    assert r.tune_spots(P, grid="g.yaml")["run_dir"] == "tuning/t1"
    assert calls[-1][2] == {"points": "annotations", "grid": "g.yaml", "out": None, "max_runs": 50}
    assert r.tune_apply(P, "tuning/t1") == {"config_hash": "fedcba987654", "warnings": []}
    assert calls[-1][1][1:] == ("tuning/t1",) and calls[-1][2] == {"plugin_version": plugin_version()}
    # the engine's TuningWarning is the record's, as in the command's --json record (9.4 #7; reviews/P6-TUNE.md
    # S2), also under a filter of this process that ignores warnings; any other warning is shown as it would be
    note = "t1: its images were run with config 02bb83cf74f9, and the recipe is now fedcba987654"

    def apply_warned(project, run_dir, *, plugin_version=None):
        warnings.warn(note, TuningWarning, stacklevel=2)
        warnings.warn("an unrelated note", RuntimeWarning, stacklevel=2)
        return "805dad578390"

    monkeypatch.setattr(api, "apply_tuning", apply_warned)
    with pytest.warns(RuntimeWarning, match="an unrelated note"):
        assert r.tune_apply(P, "tuning/t1") == {"config_hash": "805dad578390", "warnings": [note]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert r.tune_apply(P, "tuning/t1") == {"config_hash": "805dad578390", "warnings": [note]}
    assert r.list_post_stages() == [{"name": "fixture_post"}]
    Path(P).mkdir(exist_ok=True)                                   # the write refuses a root that is not a folder
    assert r.write_post_stages(P, [{"name": "fixture_post", "params": {}}]) == {
        "written": True, "path": str(Path(P) / "post_stages.yaml"), "sha256": "ab" * 32,
        "entries": [{"name": "fixture_post", "params": {}}]}
    assert writes == [(Path(P) / "post_stages.yaml", [{"name": "fixture_post", "params": {}}])]
    # an API refusal of a one-shot command reads as the engine's exit 2
    monkeypatch.setattr(api, "import_labels", fake("import_labels", _raise(ValueError("a test-split image"))))
    with pytest.raises(EngineRefused, match="a test-split image"):
        r.import_labels("seg.npz", "cellacdc-npz", "img.tif", "pairs")
    # so does a folder that is no project (open_project's ProjectError, a ValueError), as the commands' exit 2
    monkeypatch.setattr(api, "open_project", fake("open_project", _raise(ValueError(f"no project at {P}"))))
    for call in (r.project_status, r.project_collect, r.project_export_commands):
        with pytest.raises(EngineRefused, match="REFUSED: no project at"):
            call(P)


def _raise(exc):
    def f(*a, **k):
        raise exc
    return f


def test_the_in_process_runner_exports_and_trains_through_the_engine_functions(tmp_path, monkeypatch):
    """export-training and train-nuclei in this environment: segment.training's functions with the arguments
    the commands hand them; an accelerator through GERMQUANT_DEVICE for the call only; the card fields under the
    names the engine's function takes, and a clear error from an engine without them."""
    from germquant.segment import training as T

    seen: dict = {}

    def export(results, out, **kw):
        seen["export"] = (results, out, kw)
        return [1, 2, 3]

    def train(pairs_dir, out_dir, *, base="cpsam", name=None, epochs=150, split_by="series", val_frac=0.2, seed=0,
              cpu=False, system=None, object_kind=None, dataset=None):
        seen["train"] = {"base": base, "name": name, "epochs": epochs, "split_by": split_by, "cpu": cpu,
                         "system": system, "object_kind": object_kind, "dataset": dataset,
                         "device_env": os.environ.get("GERMQUANT_DEVICE")}
        return {"model_path": str(Path(out_dir) / "m1"), "card_path": str(Path(out_dir) / "m1.card.json")}

    monkeypatch.setattr(T, "export_training_pairs", export)
    monkeypatch.setattr(T, "train_nuclei", train)
    monkeypatch.delenv("GERMQUANT_DEVICE", raising=False)
    r = InProcessRunner()
    said: list[str] = []
    res = r.export_training("results", "pairs", orientations=("xy", "xz"), slices_per_image=2, max_gb=1.5,
                            zone_column="", say=said.append)
    assert seen["export"] == ("results", "pairs", {"orientations": ("xy", "xz"), "slices_per_image": 2,
                                                   "zone_column": None, "max_bytes": 1.5e9})
    assert res == {"exit_code": 0, "lines": ["3 slice pair(s) -> pairs"]} and said == res["lines"]
    r.export_training("results", "pairs", orientations="auto")
    assert seen["export"][2] == {"orientations": "auto"}
    res = r.train_nuclei("pairs", tmp_path / "m", base="none", epochs=2, device="cuda", system="worm",
                         kind="nucleus", dataset="d1")
    assert seen["train"] == {"base": "none", "name": None, "epochs": 2, "split_by": "series", "cpu": False,
                             "system": "worm", "object_kind": "nucleus", "dataset": "d1", "device_env": "cuda"}
    assert "GERMQUANT_DEVICE" not in os.environ                      # put back after the call
    assert res["model_path"] == str(tmp_path / "m" / "m1") and res["exit_code"] == 0
    r.train_nuclei("pairs", tmp_path / "m", device="cpu")
    assert seen["train"]["cpu"] is True and seen["train"]["device_env"] is None

    def old_train(pairs_dir, out_dir, *, base="cpsam", name=None, cpu=False):
        return {}

    monkeypatch.setattr(T, "train_nuclei", old_train)
    with pytest.raises(RunnerError, match="takes no kind"):
        r.train_nuclei("pairs", tmp_path / "m", kind="nucleus")

    # a learning curve trains every model on the device asked for, else the runner's (GERMQUANT_DEVICE for the
    # call); train_nuclei follows the runner's device too
    def curve(pairs, out, **kw):
        seen["curve"] = (kw, os.environ.get("GERMQUANT_DEVICE"))
        return {"models": []}

    monkeypatch.setattr(T, "train_curve", curve, raising=False)
    monkeypatch.setattr(T, "train_nuclei", train)
    monkeypatch.setattr(InProcessRunner, "features", lambda self, should_stop=None: frozenset(_engine.FEATURES))
    assert r.train_curve("pairs", "curve", sizes=[1, "all"], repeats=2, seed=3, base="none",
                         device="cuda") == {"models": []}
    assert seen["curve"] == ({"sizes": [1, "all"], "repeats": 2, "seed": 3, "base": "none"}, "cuda")
    r.device = "cpu"
    r.train_curve("pairs", "curve", sizes="all", repeats=1, seed=0, base="none")
    assert seen["curve"][1] == "cpu" and "GERMQUANT_DEVICE" not in os.environ
    r.train_nuclei("pairs", tmp_path / "m")
    assert seen["train"]["cpu"] is True and seen["train"]["device_env"] is None


def test_a_missing_entry_is_named_and_add_keeps_its_record_on_exit_1(tmp_path):
    """An engine part way through the train (projects listed, no germquant.project entry yet) gets a message
    naming the entry, not P4's "no germquant installed"; ``project add`` returns its record when it exits 1
    (some paths refused, the others added), while another command's exit 1 is an error."""
    own = plugin_engine_info()
    version = MIN if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    feats = list(_engine.FEATURES)
    early = standin_engine(tmp_path / "early", version=version, git_sha=str(own["git_sha"]), features=feats,
                           entries=("packs",))
    r = SubprocessRunner(sys.executable, env={"PYTHONPATH": str(early)})
    with pytest.raises(RunnerError, match="has no germquant.project entry") as e:
        r.project_new(tmp_path / "p", "test_pack@0.1.0")
    assert "no germquant installed" not in str(e.value)
    with pytest.raises(RunnerError, match="has no germquant.post_stages entry"):
        r.list_post_stages()
    one = standin_engine(tmp_path / "one", version=version, git_sha=str(own["git_sha"]), features=feats)
    r1 = SubprocessRunner(sys.executable, env={"PYTHONPATH": str(one), "FAKE_ENGINE_EXIT": "1"})
    assert r1.project_add(tmp_path / "p", ["a.tif"])["argv"][:2] == ["add", str(tmp_path / "p")]
    with pytest.raises(RunnerError, match=r"`germquant project status` failed \(exit 1\)"):
        r1.project_status(tmp_path / "p")


def test_the_engines_answers_to_refusals_partial_adds_and_a_lock_that_is_not_there(tmp_path):
    """As the project command line answers: exit 2 is a refusal whose reason is the engine's REFUSED line (or the
    error of its JSON record); new and add exit 1 with their record when some files were refused; unlock exits 0
    when there was no lock, and that is no removal, and exit 2 when someone else holds it, which is a refusal as
    for every other command. A revert with edits is refused before anything starts."""
    own = plugin_engine_info()
    version = MIN if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    root = standin_engine(tmp_path / "eng", version=version, git_sha=str(own["git_sha"]),
                          features=list(_engine.FEATURES))
    P = str(tmp_path / "proj")

    def runner(**env) -> SubprocessRunner:
        return SubprocessRunner(sys.executable, env={"PYTHONPATH": str(root), **env})

    refused = runner(FAKE_ENGINE_EXIT="2", FAKE_ENGINE_STDERR="REFUSED: the project is locked by LAB-PC pid 42",
                     FAKE_ENGINE_STDOUT='{"error": "the project is locked by LAB-PC pid 42", "kind": "ProjectError"}')
    with pytest.raises(EngineRefused) as e:
        refused.project_edit(P, edits={"spots.log.min_snr": 2.0})
    assert str(e.value) == ("`germquant.project edit` was refused by the engine (exit 2):\n"
                            "REFUSED: the project is locked by LAB-PC pid 42")
    json_only = runner(FAKE_ENGINE_EXIT="2", FAKE_ENGINE_STDOUT='{"error": "no project at X", "kind": "ProjectError"}')
    with pytest.raises(EngineRefused, match="REFUSED: no project at X$"):
        json_only.project_use_model(P, "training/r01/models/m/m")
    partial = runner(FAKE_ENGINE_EXIT="1", FAKE_ENGINE_STDOUT='{"root": "R", "added": ["a"], "refused": '
                                                             '[{"path": "b.czi", "reason": "multi-scene"}]}')
    assert partial.project_new(P, "test_pack@0.1.0", images=["a.tif", "b.czi"])["refused"][0]["reason"] == \
        "multi-scene"
    assert partial.project_add(P, ["b.czi"])["added"] == ["a"]
    with pytest.raises(RunnerError, match="exit 1"):
        partial.project_edit(P, edits={"a.b": 1})                       # exit 1 is no answer of edit
    gone = runner(FAKE_ENGINE_STDOUT="the project is not locked")
    assert gone.project_unlock(P, host="h", pid=1)["removed"] is False
    assert runner(FAKE_ENGINE_STDOUT="removed the lock of h pid 1").project_unlock(P, host="h", pid=1)["removed"]
    held = runner(FAKE_ENGINE_EXIT="2", FAKE_ENGINE_STDERR="REFUSED: the lock of P is held by h pid 2, not by h pid 1")
    with pytest.raises(EngineRefused) as e:
        held.project_unlock(P, host="h", pid=1)
    assert str(e.value) == ("`germquant.project unlock` was refused by the engine (exit 2):\n"
                            "REFUSED: the lock of P is held by h pid 2, not by h pid 1") and e.value.exit_code == 2
    n = len(gone.history)
    with pytest.raises(ValueError, match="a revert cannot be combined"):
        gone.project_edit(P, revert=True, roles={"foci": "x"})
    assert len(gone.history) == n                                       # nothing ran
    plain = runner()
    plain.project_edit(P, z_range="none", device="none")
    assert plain.history[-1][3:] == ["edit", P, "--device", "none", "--z-range", "none", "--plugin-version",
                                     plugin_version(), "--json"]


def test_each_engine_refusal_form_reads_as_a_refusal_with_its_reason(tmp_path):
    """The refusal forms of the Train B commands, each read as EngineRefused whose reason is the engine's own
    words, not the whole output: REFUSED: (the project commands, train-curve), ERROR: (tune-spots), refused:
    lines and a {"written": false, "problems"} record over many lines (post_stages write), an indented
    {"error"} record alone, and pack derive's exit 1 with a REFUSED line; a crash of derive (exit 1, no reason)
    stays an error. A tune-spots search that scored nothing (exit 1) keeps its record."""
    own = plugin_engine_info()
    version = MIN if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    root = standin_engine(tmp_path / "eng", version=version, git_sha=str(own["git_sha"]),
                          features=list(_engine.FEATURES))
    P = str(tmp_path / "proj")

    def runner(**env) -> SubprocessRunner:
        return SubprocessRunner(sys.executable, env={"PYTHONPATH": str(root), **env})

    def reason(call, **env) -> EngineRefused:
        with pytest.raises(EngineRefused) as e:
            call(runner(**env))
        return e.value

    e = reason(lambda r: r.tune_spots(P), FAKE_ENGINE_EXIT="2",
               FAKE_ENGINE_STDERR="ERROR: cannot open the project P: ProjectError: no project")
    assert str(e) == ("`germquant tune-spots` was refused by the engine (exit 2):\n"
                      "ERROR: cannot open the project P: ProjectError: no project")
    problems = ["fixture_post: radius_um 9.0 is above its max 5.0", "no_such: not installed in this environment"]
    record = json.dumps({"written": False, "problems": problems}, indent=1)
    write = lambda r: r.write_post_stages(P, [{"name": "fixture_post", "params": {"radius_um": 9.0}}])  # noqa: E731
    e = reason(write, FAKE_ENGINE_EXIT="2", FAKE_ENGINE_STDOUT=record)
    assert str(e).splitlines()[1:] == [f"REFUSED: {p}" for p in problems]
    e = reason(write, FAKE_ENGINE_EXIT="2", FAKE_ENGINE_STDOUT=record,
               FAKE_ENGINE_STDERR="\n".join(f"refused: {p}" for p in problems))
    assert str(e).splitlines()[1:] == [f"refused: {p}" for p in problems]
    e = reason(lambda r: r.project_edit(P, z_range=(5, 2)), FAKE_ENGINE_EXIT="2",
               FAKE_ENGINE_STDOUT=json.dumps({"error": "z range [5, 2] must be two planes", "kind": "ProjectError"},
                                             indent=1))
    assert str(e).endswith("\nREFUSED: z range [5, 2] must be two planes")
    derive = lambda r: r.pack_derive("test_pack@0.1.0", "basic", "m", "m.card.json", "out")  # noqa: E731
    e = reason(derive, FAKE_ENGINE_EXIT="1", FAKE_ENGINE_STDERR="REFUSED: the card names no base model",
               FAKE_ENGINE_STDOUT=json.dumps({"ok": False, "error": "the card names no base model"}, indent=1))
    assert e.exit_code == 1 and str(e) == ("`germquant.packs derive` was refused by the engine (exit 1):\n"
                                           "REFUSED: the card names no base model")
    with pytest.raises(RunnerError, match=r"`germquant.packs derive` failed \(exit 1\)") as crash:
        derive(runner(FAKE_ENGINE_EXIT="1", FAKE_ENGINE_STDERR="Traceback (most recent call last):"))
    assert not isinstance(crash.value, EngineRefused)
    scored_nothing = {"run_dir": "tuning/t1", "best": None, "ranked": [], "warnings": ["no truth points scored"]}
    t = runner(FAKE_ENGINE_EXIT="1", FAKE_ENGINE_STDOUT=json.dumps(scored_nothing, indent=1),
               FAKE_ENGINE_STDERR="ERROR: no parameter set could be scored; see the warnings and docs/TUNING.md")
    assert t.tune_spots(P) == scored_nothing


def test_in_process_project_writes_run_the_command_of_this_environment(tmp_path, monkeypatch):
    """The six project writes of the in-process runner (new, add, edit, use-model, rounds, unlock) run the
    torch-free ``-m germquant.project`` of this environment's own Python (here a stand-in engine first on its
    PYTHONPATH): the command line the subprocess runner sends, the command's own record back, its refusals as
    the subprocess runner raises them, and no version check (the child is this germquant). So an edit is the
    command's one planned write, never API writes in a row that a refusal could stop half way (review S1), and
    new and add answer with the command's record (S3). This environment's feature gate still decides first."""
    own = plugin_engine_info()
    version = MIN if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    eng = standin_engine(tmp_path / "eng", version=version, git_sha=str(own["git_sha"]),
                         features=list(_engine.FEATURES))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in (str(eng), os.environ.get("PYTHONPATH")) if p))
    monkeypatch.setattr(api, "open_project", lambda root: None, raising=False)    # this environment's projects gate
    started: list[list[str]] = []
    run_proc = SubprocessRunner._run_proc

    def spy(self, argv, **kw):
        started.append(list(argv))
        return run_proc(self, argv, **kw)

    monkeypatch.setattr(SubprocessRunner, "_run_proc", spy)
    inproc, sub = InProcessRunner(), SubprocessRunner(sys.executable)
    sub.version()
    P = str(tmp_path / "proj")
    calls = {
        "new": lambda r: r.project_new(P, "test_pack@0.1.0", recipe="basic", images=tmp_path / "a.ome.tif",
                                       name="N"),
        "add": lambda r: r.project_add(P, [tmp_path / "b.ome.tif", str(tmp_path / "folder")]),
        "edit": lambda r: r.project_edit(P, edits={"spots.log.min_snr": 2.5}, unset=["segmentation.x"],
                                         roles={"foci": GAMMA + "H2AX", "dna": ""}, device="none", xy_stride=2,
                                         z_range=(5, 2), runner="subprocess", interpreters={"basic": "none"},
                                         disable=["spots"], dry_run=True),
        "use-model": lambda r: r.project_use_model(P, "training/r01/models/m/m"),
        "rounds": lambda r: r.project_rounds(P, new="r02"),
        "unlock": lambda r: r.project_unlock(P, host="LAB-PC", pid=42),
    }
    for name, call in calls.items():
        n = len(started)
        theirs = call(sub)
        sent = started[n:]
        ours = call(inproc)
        assert started[n + len(sent):] == sent and len(sent) == 1, name
        assert sent[0][:4] == [sys.executable, "-m", "germquant.project", name], name
        assert ours == theirs, name
    # a refusal: the same EngineRefused, with the engine's reason
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "2")
    monkeypatch.setenv("FAKE_ENGINE_STDERR", "REFUSED: z range [5, 2] must be two planes Z0 Z1 with 0 <= Z0 < Z1")
    said = []
    for r in (sub, inproc):
        with pytest.raises(EngineRefused) as e:
            r.project_edit(P, edits={"spots.log.min_snr": 2.5}, z_range=(5, 2))
        said.append(str(e.value))
    assert said[0] == said[1] and said[0].endswith("\nREFUSED: z range [5, 2] must be two planes Z0 Z1 with 0 <= Z0 < Z1")
    # without projects in this environment, nothing starts
    monkeypatch.delattr(api, "open_project")
    n = len(started)
    with pytest.raises(FeatureMissing, match="this napari environment"):
        inproc.project_edit(P, edits={"spots.log.min_snr": 2.5})
    assert len(started) == n


def _project_state(root: Path) -> tuple:
    """What a project edit can change: recipe.yaml and channel_map.yaml (their bytes, or None), the manifest
    without what names this one project (its name, id and stamps), and the config_hash the engine computes."""
    import yaml

    files = tuple((root / n).read_bytes() if (root / n).is_file() else None
                  for n in ("recipe.yaml", "channel_map.yaml"))
    manifest = yaml.safe_load((root / "germquant_project.yaml").read_bytes().decode("ascii"))
    return files, {k: v for k, v in manifest.items() if k not in ("name", "id", "created", "saved")}, \
        api.open_project(root).config_hash


def test_a_project_edit_through_both_runners_writes_the_same_files(project_env):
    """The same project commands through the subprocess runner and the in-process runner, on projects the engine
    makes from the test pack: new and add answer with the same record; a dry run writes nothing and gives the
    config_hash the edit then gives; the edit writes the same recipe.yaml, manifest and config_hash; and an edit
    the engine refuses (a recipe edit together with a z range it refuses) leaves both projects exactly as they
    were (review S1, S3)."""
    need_features("projects")
    runners = {"subprocess": SubprocessRunner(sys.executable), "inprocess": InProcessRunner()}
    roots = {k: project_env.new_dir(k) for k in runners}
    image_id = api.image_stem(project_env.image)
    new = {k: r.project_new(roots[k], "test_pack@0.1.0", images=project_env.image) for k, r in runners.items()}
    assert sorted(new["subprocess"]) == sorted(new["inprocess"])
    assert all(doc["added"] == [image_id] and doc["refused"] == [] for doc in new.values()), new
    again = {k: r.project_add(roots[k], project_env.image) for k, r in runners.items()}       # a duplicate id
    assert sorted(again["subprocess"]) == sorted(again["inprocess"])
    assert [len(doc["refused"]) for doc in again.values()] == [1, 1]
    start = {k: _project_state(roots[k]) for k in runners}
    assert start["subprocess"] == start["inprocess"]
    edit = {"edits": {"spots.log.min_snr": 2.5}, "xy_stride": 2}
    dry = {k: r.project_edit(roots[k], dry_run=True, **edit) for k, r in runners.items()}
    assert {k: _project_state(roots[k]) for k in runners} == start                    # a dry run writes nothing
    done = {k: r.project_edit(roots[k], **edit) for k, r in runners.items()}
    after = {k: _project_state(roots[k]) for k in runners}
    assert after["subprocess"] == after["inprocess"] != start["subprocess"]
    for k in runners:
        assert dry[k]["config_hash"] == done[k]["config_hash"] == after[k][2] != start[k][2], k
    for k, r in runners.items():
        with pytest.raises(EngineRefused, match="z range"):
            r.project_edit(roots[k], edits={"spots.log.min_snr": 4.0}, z_range=(5, 2))
    assert {k: _project_state(roots[k]) for k in runners} == after                    # none of it was written


def test_an_apply_gives_the_engines_tuning_warning_through_both_runners(project_env):
    """Apply (plan 9.4 #7) through both runners on the engine, of a tuning run whose images were run with another
    config (its run.json says so): the same recipe.yaml, manifest and config_hash, the save stamped with the
    plugin version (D42), and the engine's TuningWarning in the record: the command's --json record through the
    subprocess runner, apply_tuning's warning recorded in process (reviews/P6-TUNE.md S2 and S3)."""
    need_features("projects", "tune_spots")
    import yaml

    rid, made_with = "20261030T120000Z_0123abcd", "0123456789ab"
    runners = {"subprocess": SubprocessRunner(sys.executable), "inprocess": InProcessRunner()}
    roots = {k: project_env.new_dir(k) for k in runners}
    for root in roots.values():
        api.new_project(root, "test_pack@0.1.0", recipe="basic")
        run = root / "tuning" / rid                    # the files apply reads (9.3), as a search writes them
        run.mkdir(parents=True)
        (run / "grid.yaml").write_bytes(b"spots.log.min_snr:\n- 1.5\n- 3.0\n")
        (run / "best.yaml").write_bytes(yaml.safe_dump({"objective": "pooled F1", "backend": "log",
                                                        "keys": {"spots.log.min_snr": 3.0}}).encode("ascii"))
        (run / "run.json").write_bytes(json.dumps({"items": [{"image_id": "x", "config_hash": made_with}]})
                                       .encode("ascii"))

    def saved_plugin(root: Path):
        return yaml.safe_load((root / "germquant_project.yaml").read_bytes().decode("ascii"))["saved"]["plugin"]

    start = {k: _project_state(roots[k]) for k in runners}
    assert start["subprocess"] == start["inprocess"]
    assert [saved_plugin(root) for root in roots.values()] == [None, None]
    recs = {k: r.tune_apply(roots[k], f"tuning/{rid}") for k, r in runners.items()}
    after = {k: _project_state(roots[k]) for k in runners}
    assert after["subprocess"] == after["inprocess"] != start["subprocess"]
    assert [saved_plugin(root) for root in roots.values()] == [plugin_version()] * 2
    sub, inproc = recs["subprocess"], recs["inprocess"]
    assert inproc == {"config_hash": after["inprocess"][2], "warnings": inproc["warnings"]}
    assert sub == {"config_hash": inproc["config_hash"], "run_dir": f"tuning/{rid}", "warnings": inproc["warnings"]}
    (note,) = inproc["warnings"]                       # the run, the config it was made with and the recipe's now
    assert rid in note and made_with in note and start["inprocess"][2] in note


# ---- each in-process call refuses what its command refuses (reviews/P6_package.md N1) ----

DISK_FULL = "No space left on device"
# what `germquant project ...` says of a project file that is not YAML (cli_projects.dispatch), for the error
# _yaml_error raises
YAML_REFUSAL = ("a YAML file of the project does not load (expected ',' or ']', but got '<stream end>', line 2 "
                "column 1): if recipe.yaml or channel_map.yaml was edited by hand, fix it, or go back to the pack "
                "recipe with `germquant project edit DIR --revert`")


def _yaml_error() -> Exception:
    """The error PyYAML raises for a hand edit gone wrong: a list left open on line 1."""
    import yaml

    try:
        yaml.safe_load("spots: [unclosed\n")
    except yaml.YAMLError as e:
        return e
    raise AssertionError("the text read as YAML")


def _part_way(E) -> Exception:
    """germquant.project's ProjectError for a write that stopped part way: raised from the OSError (9.4 #7)."""
    e = E.ProjectError("the write stopped part way: recipe.yaml was written, germquant_project.yaml was not")
    e.__cause__ = OSError(28, DISK_FULL)
    return e


def _engine_errors(monkeypatch) -> types.SimpleNamespace:
    """The engine's exception classes the commands name: germquant.tune's TuneRefused (and TuningWarning, which an
    Apply records), germquant.project's ProjectError, germquant.models_registry's ModelPinError; on an engine
    before P5 (the CI pin), stand-in modules with classes of the same bases."""
    from germquant.models_registry import ModelPinError

    try:
        from germquant.tune import TuneRefused
    except ImportError:
        tune = types.ModuleType("germquant.tune")
        tune.TuneRefused = TuneRefused = type("TuneRefused", (ValueError,), {})
        tune.TuningWarning = type("TuningWarning", (UserWarning,), {})
        monkeypatch.setitem(sys.modules, "germquant.tune", tune)
    try:
        from germquant.project import ProjectError
    except ImportError:
        project = types.ModuleType("germquant.project")
        project.ProjectError = ProjectError = type("ProjectError", (ValueError,), {})
        monkeypatch.setitem(sys.modules, "germquant.project", project)
    return types.SimpleNamespace(TuneRefused=TuneRefused, ProjectError=ProjectError, ModelPinError=ModelPinError)


def _fake_api(monkeypatch, raising: dict) -> None:
    """germquant.api's project, import and tuning functions and segment.training.train_curve as fakes that answer,
    or raise the error ``raising`` gives for their name; this environment has every Train B feature."""
    from germquant.segment import training as T

    def fake(name, value):
        def f(*a, **k):
            if name in raising:
                raise raising[name]
            return value(*a, **k) if callable(value) else value
        f.__name__ = name
        return f

    answers = {"open_project": _FakeProject, "project_status": {"format": 1, "images": []},
               "collect_project": {"tables": []}, "export_commands": "germquant project run X\n",
               "run_project": {"exit_code": 0}, "import_labels": {"pairs": 2},
               "tune_spots": {"run_dir": "tuning/t1", "best": None, "ranked": []}, "apply_tuning": "fedcba987654",
               "list_post_stages": [{"name": "fixture_post"}]}
    for name, value in answers.items():
        monkeypatch.setattr(api, name, fake(name, value), raising=False)
    monkeypatch.setattr(T, "train_curve", fake("train_curve", {"models": []}), raising=False)
    monkeypatch.setattr(InProcessRunner, "features", lambda self, should_stop=None: frozenset(_engine.FEATURES))


IN_PROCESS_CALLS = {
    "import-labels": lambda r, P: r.import_labels("seg.npz", "cellacdc-npz", "img.tif", "pairs"),
    "train-curve": lambda r, P: r.train_curve("pairs", "curve", sizes=[1, "all"], repeats=1, seed=0, base="none"),
    "tune-spots": lambda r, P: r.tune_spots(P),
    "tune-spots --apply": lambda r, P: r.tune_apply(P, "tuning/t1"),
    "project run": lambda r, P: r.project_run(P, run_id=RID),
    "project status": lambda r, P: r.project_status(P),
    "project collect": lambda r, P: r.project_collect(P),
    "project export-commands": lambda r, P: r.project_export_commands(P),
    "post_stages list": lambda r, P: r.list_post_stages(),
}

# (the command, the API function that raises, the error, what the command answers), as the commands are built at
# 39a0a84: cli_training._import_labels and _train_curve, cli_tune.dispatch and cli_projects.dispatch; and at P5b's
# WIRE draft eef86bc: germquant.post_stages.__main__._list
ANSWERS = [
    # import-labels: ValueError, FileNotFoundError and ImportError are refused; any other OSError fails
    ("import-labels", "import_labels", lambda E: ValueError("a test-split image"), "refused"),
    ("import-labels", "import_labels", lambda E: FileNotFoundError("labels file not found: seg.npz"), "refused"),
    ("import-labels", "import_labels", lambda E: ModuleNotFoundError("No module named 'bioio_ome_tiff'"), "refused"),
    ("import-labels", "import_labels", lambda E: OSError(28, DISK_FULL), "failed"),
    ("import-labels", "import_labels", lambda E: PermissionError(13, "Permission denied"), "failed"),
    # train-curve: ValueError, FileNotFoundError and ModelPinError (a RuntimeError) are refused; anything else fails
    ("train-curve", "train_curve", lambda E: ValueError("the pairs form a single series group"), "refused"),
    ("train-curve", "train_curve", lambda E: FileNotFoundError("no manifest.csv in pairs"), "refused"),
    ("train-curve", "train_curve", lambda E: E.ModelPinError("cpsam_v2@00000000 is not a pinned model"), "refused"),
    ("train-curve", "train_curve", lambda E: OSError(28, DISK_FULL), "failed"),
    ("train-curve", "train_curve", lambda E: RuntimeError("CUDA out of memory"), "failed"),
    # tune-spots, the search: TuneRefused and FileNotFoundError are refused, and any error opening the project;
    # anything else fails (a detection that failed once the search ran)
    ("tune-spots", "tune_spots", lambda E: E.TuneRefused("a split named test is refused"), "refused"),
    ("tune-spots", "tune_spots", lambda E: FileNotFoundError("no annotations/ folder"), "refused"),
    ("tune-spots", "tune_spots", lambda E: ValueError("a detection failed"), "failed"),
    ("tune-spots", "tune_spots", lambda E: OSError(28, DISK_FULL), "failed"),
    ("tune-spots", "open_project", lambda E: TypeError("the manifest's images are not a list"), "refused"),
    # tune-spots --apply: TuneRefused, FileNotFoundError and ValueError are refused, and any error opening the
    # project, but a write that stopped part way fails, as anything else does
    ("tune-spots --apply", "apply_tuning", lambda E: E.TuneRefused("keys outside the run's grid"), "refused"),
    ("tune-spots --apply", "apply_tuning", lambda E: FileNotFoundError("no best.yaml in tuning/t1"), "refused"),
    ("tune-spots --apply", "apply_tuning", lambda E: E.ProjectError("the project is locked by LAB-PC pid 42"),
     "refused"),
    ("tune-spots --apply", "apply_tuning", _part_way, "failed"),
    ("tune-spots --apply", "apply_tuning", lambda E: OSError(28, DISK_FULL), "failed"),
    ("tune-spots --apply", "open_project", lambda E: TypeError("the manifest's images are not a list"), "refused"),
    # the project commands: ValueError (ProjectError), OSError and PackInstallError are refused, and a project file
    # that is not YAML, with the engine's reason and way back; anything else fails
    ("project run", "run_project", lambda E: _yaml_error(), "refused"),
    ("project run", "run_project", lambda E: ValueError("model digest mismatch for cpsam_v2@0f1cc3f7"), "refused"),
    ("project run", "run_project", lambda E: RuntimeError("boom"), "failed"),
    ("project status", "project_status", lambda E: _yaml_error(), "refused"),
    ("project status", "project_status", lambda E: OSError("the share went away"), "refused"),
    ("project status", "open_project", lambda E: TypeError("the manifest's images are not a list"), "failed"),
    ("project collect", "collect_project", lambda E: _yaml_error(), "refused"),
    ("project collect", "collect_project", lambda E: RuntimeError("boom"), "failed"),
    ("project export-commands", "export_commands", lambda E: _yaml_error(), "refused"),
    # post_stages list: the command has no handler, so every error fails it (a traceback, exit 1), also the classes
    # the other commands refuse
    ("post_stages list", "list_post_stages", lambda E: ValueError("a declaration file that cannot be read"),
     "failed"),
    ("post_stages list", "list_post_stages", lambda E: FileNotFoundError("no such distribution"), "failed"),
    ("post_stages list", "list_post_stages", lambda E: PermissionError(13, "Permission denied"), "failed"),
]


@pytest.mark.parametrize(("command", "raiser", "error", "answer"), ANSWERS,
                         ids=[f"{i:02d}-{c.replace(' ', '_')}-{a}" for i, (c, _r, _e, a) in enumerate(ANSWERS)])
def test_each_in_process_call_refuses_what_its_command_refuses(command, raiser, error, answer, tmp_path,
                                                                 monkeypatch):
    """In process, an error of the API call is a refusal (EngineRefused, and for a project run the refused state)
    when the engine's command refuses it (exit 2: nothing ran or was written), else a failure (a RunnerError
    naming the error), as the command fails (exit 1), so the dock says what the command line would say: each
    command's own rule, not the project commands' for every call (reviews/P6_package.md N1)."""
    import yaml

    E = _engine_errors(monkeypatch)
    exc = error(E)
    _fake_api(monkeypatch, raising={raiser: exc})
    call = IN_PROCESS_CALLS[command]
    r, P = InProcessRunner(), tmp_path / "proj"
    if answer == "failed":
        with pytest.raises(RunnerError) as e:
            call(r, P)
        assert not isinstance(e.value, EngineRefused), str(e.value)
        assert f"{type(exc).__name__}: {exc}" in str(e.value) and e.value.__cause__ is exc
        return
    reason = YAML_REFUSAL if isinstance(exc, yaml.YAMLError) else str(exc)
    if command == "project run":
        res = call(r, P)
        assert (res["state"], res["exit_code"], res["events"], res["message"]) == ("refused", 2, [],
                                                                                    f"REFUSED: {reason}")
        return
    with pytest.raises(EngineRefused) as e:
        call(r, P)
    assert str(e.value) == f"REFUSED: {reason}" and e.value.output == reason and e.value.__cause__ is exc


def test_an_in_process_export_that_cannot_write_its_file_is_refused(tmp_path, monkeypatch):
    """`project export-commands --out FILE` refuses a FILE it cannot write (exit 2: cli_projects.dispatch refuses
    an OSError of its handler, the write included), so in process too, and no part file is left."""
    _engine_errors(monkeypatch)
    _fake_api(monkeypatch, raising={})
    out = tmp_path / "no such folder" / "cmds.ps1"
    with pytest.raises(EngineRefused, match="^REFUSED: ") as e:
        InProcessRunner().project_export_commands(tmp_path / "proj", shell="ps1", out=out)
    assert isinstance(e.value.__cause__, FileNotFoundError) and not out.parent.exists()


def test_an_in_process_export_writes_the_commands_bytes(tmp_path, monkeypatch):
    """`project export-commands --out FILE` writes ps1 with a UTF-8 byte-order mark (Windows PowerShell 5.1 reads a
    script without one in the ANSI code page, so a project folder named with a letter outside ASCII would be
    misread) and sh without one (cli_projects._export_commands): in process too (reviews/P6_package_fixes_delta.md
    N1)."""
    _engine_errors(monkeypatch)
    _fake_api(monkeypatch, raising={})
    for shell, head in (("ps1", b"\xef\xbb\xbf"), ("sh", b"")):
        out = tmp_path / f"cmds.{shell}"
        rec = InProcessRunner().project_export_commands(tmp_path / "proj", shell=shell, out=out)
        assert out.read_bytes() == head + b"germquant project run X\n" and rec["text"] == "germquant project run X\n"


def test_a_project_file_that_is_not_yaml_is_refused_alike_through_both_runners(project_env):
    """A recipe.yaml broken by hand: `project run` and `project collect` refuse the project (exit 2) with one
    REFUSED line, the engine's reason and way back (cli_projects.dispatch); the in-process runner gives that line
    for a run (the refused state) and refuses collect and export-commands with it, where it used to fail with the
    YAML parser's own error (reviews/P6_package.md N1). Nothing runs."""
    need_features("projects")
    root = project_env.new_dir()
    api.new_project(root, "test_pack@0.1.0", images=[project_env.image])
    api.write_recipe(api.open_project(root), edits={"spots.log.min_snr": 2.5})
    recipe = root / "recipe.yaml"
    recipe.write_bytes(recipe.read_bytes() + b"spots: [unclosed\n")
    sub, inproc = SubprocessRunner(sys.executable), InProcessRunner()
    ran = sub.project_run(root, device="cpu")
    assert (ran["state"], ran["exit_code"], ran["events"]) == ("refused", 2, [])
    (line,) = [ln for ln in ran["message"].splitlines() if ln.startswith("REFUSED: ")]
    assert line.startswith("REFUSED: a YAML file of the project does not load (") and line.endswith(" --revert`")
    with pytest.raises(EngineRefused) as e:
        sub.project_collect(root)
    assert str(e.value).splitlines()[1:] == [line]
    res = inproc.project_run(root, device="cpu")
    assert (res["state"], res["exit_code"], res["events"], res["message"]) == ("refused", 2, [], line)
    for call in (inproc.project_collect, inproc.project_export_commands):
        with pytest.raises(EngineRefused) as e:
            call(root)
        assert str(e.value) == line, call.__name__
    assert list(root.glob("runs/*.jsonl")) == []


# ---- the two post-stage commands: `python -m germquant.post_stages list --json` and `write DIR --entries JSON
# --json` (germquant.post_stages.__main__, P5b; plan 9.2 and 9.4 #9; handoff/P6-RUNNER.md, round 3) ----

POST = "fixture_post"


def _fake_post_stages(monkeypatch, *, problems=(), raising: dict | None = None) -> types.SimpleNamespace:
    """A stand-in germquant.post_stages with the names the write takes (POST_STAGES_FILE, validate_post_stages,
    write_post_stages_file, post_stages_sha256, load_post_stages_file), each call recorded in ``calls``.
    validate_post_stages answers ``problems``; the writer refuses a list with problems as the engine's does (a
    ValueError, nothing written); ``raising`` maps "validate", "write", "sha256" or "load" to the error that call
    raises. This environment has every Train B feature."""
    fake = types.SimpleNamespace(calls=[], problems=list(problems), raising=dict(raising or {}))

    def call(name, *args):
        fake.calls.append((name, *args))
        if name in fake.raising:
            raise fake.raising[name]

    def validate_post_stages(entries, declarations=None):
        call("validate", entries)
        return list(fake.problems)

    def write_post_stages_file(path, entries, *, declarations=None):
        call("write", Path(path), entries)
        if fake.problems:
            raise ValueError("post_stages.yaml not written: " + "; ".join(fake.problems))
        Path(path).write_bytes(b"- name: fixture_post\n  params:\n    radius_um: 1.5\n")
        return Path(path)

    def post_stages_sha256(source, declarations=None):
        call("sha256", Path(source))
        return "ab" * 32

    def load_post_stages_file(path):
        call("load", Path(path))
        return [{"name": POST, "params": {"radius_um": 1.5}}]

    post = types.ModuleType("germquant.post_stages")
    post.POST_STAGES_FILE = "post_stages.yaml"
    for f in (validate_post_stages, write_post_stages_file, post_stages_sha256, load_post_stages_file):
        setattr(post, f.__name__, f)
    monkeypatch.setitem(sys.modules, "germquant.post_stages", post)
    monkeypatch.setattr(InProcessRunner, "features", lambda self, should_stop=None: frozenset(_engine.FEATURES))
    return fake


def test_an_in_process_post_stage_write_gives_the_commands_record(tmp_path, monkeypatch):
    """`post_stages write DIR --entries JSON --json` answers {"written": true, "path", "sha256", "entries"} (plan 9.4
    #9); in process too, through the engine's own functions (the file's digest, post_stages_sha256, and its entries
    as written, load_post_stages_file), where the record was {"path", "entries"} with the entries as given
    (reviews/P6_package.md). The engine gets the entries as the command parses them, the JSON list the subprocess
    runner sends (a tuple of entries is a list), checks them, then writes once."""
    fake = _fake_post_stages(monkeypatch)
    root = tmp_path / "proj"
    root.mkdir()
    given = ({"name": POST, "params": {"radius_um": 1.5}},)
    rec = InProcessRunner().write_post_stages(root, given)
    target = root / "post_stages.yaml"
    assert rec == {"written": True, "path": str(target), "sha256": "ab" * 32,
                   "entries": [{"name": POST, "params": {"radius_um": 1.5}}]}
    assert fake.calls == [("validate", [given[0]]), ("write", target, [given[0]]), ("sha256", target),
                          ("load", target)]
    assert type(fake.calls[0][1]) is list and fake.calls[0][1][0] is not given[0]     # the command's parse


def test_an_in_process_post_stage_write_refuses_each_problem_on_its_line(tmp_path, monkeypatch):
    """The command checks the entries before it writes (validate_post_stages) and refuses every problem, one line
    each (exit 2, nothing written); in process EngineRefused with one "REFUSED: <problem>" line per problem, where
    the writer's ValueError was one line, "REFUSED: post_stages.yaml not written: <problem>; <problem>"."""
    problems = ["entry 0: post-stage fixture_post: parameter radius_um must be at most 5.0, got 9.0",
                "entry 1: post-stage no_such is not installed in this environment (installed: fixture_post)"]
    fake = _fake_post_stages(monkeypatch, problems=problems)
    root = tmp_path / "proj"
    root.mkdir()
    with pytest.raises(EngineRefused) as e:
        InProcessRunner().write_post_stages(root, [{"name": POST, "params": {"radius_um": 9.0}}, {"name": "no_such"}])
    assert str(e.value) == "\n".join(f"REFUSED: {p}" for p in problems)
    assert e.value.output == "\n".join(problems) and e.value.exit_code == 2
    assert [c[0] for c in fake.calls] == ["validate"] and list(root.iterdir()) == []      # nothing written


def test_an_in_process_post_stage_write_refuses_a_root_that_is_not_a_folder(tmp_path, monkeypatch):
    """`post_stages write DIR` refuses a DIR that is not a folder ("<DIR> is not a folder", exit 2) before it checks
    the entries; in process the same line, where the write was tried and its error read as the refusal."""
    fake = _fake_post_stages(monkeypatch, problems=["never checked"])
    a_file = tmp_path / "a_file.txt"
    a_file.write_bytes(b"x")
    for root in (tmp_path / "missing", a_file, str(tmp_path / "missing" / "deeper")):
        with pytest.raises(EngineRefused) as e:
            InProcessRunner().write_post_stages(root, [{"name": POST, "params": {}}])
        assert str(e.value) == f"REFUSED: {Path(root)} is not a folder" and e.value.exit_code == 2, root
    assert fake.calls == [] and not (tmp_path / "missing").exists() and a_file.read_bytes() == b"x"


WRITE_FAILURES = [
    ("write", lambda: OSError(28, DISK_FULL)),
    ("write", lambda: PermissionError(13, "Permission denied")),
    ("write", lambda: ValueError("post_stages.yaml not written: fixture_post was uninstalled meanwhile")),
    ("write", lambda: RuntimeError("boom")),
    ("validate", lambda: ValueError("a declaration of this environment cannot be read")),
    ("sha256", lambda: OSError(13, "Permission denied")),
    ("load", lambda: ValueError("post_stages.yaml cannot be read as YAML (line 3)")),
]


@pytest.mark.parametrize(("where", "error"), WRITE_FAILURES,
                         ids=[f"{i}-{w}-{e().__class__.__name__}" for i, (w, e) in enumerate(WRITE_FAILURES)])
def test_an_in_process_post_stage_write_that_fails_is_a_failure(where, error, tmp_path, monkeypatch):
    """Past its two refusals the command fails on every error (exit 1): the write's OSError or ValueError ("error:
    <path> could not be written (...)"), anything else, and an error of the check or of the record (no handler
    around them, a traceback); in process a RunnerError naming the error, never EngineRefused, where an OSError or
    a ValueError of the write read as refused (the dock's "refused by the engine", nothing written, it implies)."""
    exc = error()
    fake = _fake_post_stages(monkeypatch, raising={where: exc})
    root = tmp_path / "proj"
    root.mkdir()
    with pytest.raises(RunnerError) as e:
        InProcessRunner().write_post_stages(root, [{"name": POST, "params": {}}])
    assert not isinstance(e.value, EngineRefused), str(e.value)
    assert f"{type(exc).__name__}: {exc}" in str(e.value) and e.value.__cause__ is exc
    order = ["validate", "write", "sha256", "load"]
    assert [c[0] for c in fake.calls] == order[:order.index(where) + 1]


def test_the_subprocess_runner_reads_the_post_stage_commands_exit_codes(tmp_path):
    """Both post-stage commands through the subprocess runner, as germquant.post_stages.__main__ answers: `list
    --json` exits 0 with the listing, or 1 with a traceback (it has no handler: a failure); `write` exits 0 with
    its record, 2 when it refused before writing (the reason its "refused:" lines), and 1 when the file could not
    be written (a failure, not a refusal, though its line starts with "error:"). Exit 2 is a refusal as for every
    command, a usage error of the listing too."""
    own = plugin_engine_info()
    version = MIN if engine_too_old(str(own["germquant"])) else str(own["germquant"])
    root = standin_engine(tmp_path / "eng", version=version, git_sha=str(own["git_sha"]),
                          features=list(_engine.FEATURES))
    P = str(tmp_path / "proj")

    def runner(**env) -> SubprocessRunner:
        return SubprocessRunner(sys.executable, env={"PYTHONPATH": str(root), **env})

    listing = [{"name": POST, "version": "1.0.0", "dist": "lab-post-stages", "params": []}]
    assert runner(FAKE_ENGINE_STDOUT=json.dumps(listing, indent=1)).list_post_stages() == listing
    crash = runner(FAKE_ENGINE_EXIT="1", FAKE_ENGINE_STDERR="Traceback (most recent call last):\nValueError: bad")
    with pytest.raises(RunnerError, match=r"`germquant.post_stages list` failed \(exit 1\)") as e:
        crash.list_post_stages()
    assert not isinstance(e.value, EngineRefused)
    usage = runner(FAKE_ENGINE_EXIT="2",
                   FAKE_ENGINE_STDERR="python -m germquant.post_stages: error: unrecognized arguments: --json")
    with pytest.raises(EngineRefused) as e:
        usage.list_post_stages()
    assert e.value.exit_code == 2
    record = {"written": True, "path": str(Path(P) / "post_stages.yaml"), "sha256": "ab" * 32, "entries": []}
    assert runner(FAKE_ENGINE_STDOUT=json.dumps(record, indent=1)).write_post_stages(P, []) == record
    line = f"error: {Path(P) / 'post_stages.yaml'} could not be written (PermissionError: Access is denied)"
    with pytest.raises(RunnerError, match=r"`germquant.post_stages write` failed \(exit 1\)") as e:
        runner(FAKE_ENGINE_EXIT="1", FAKE_ENGINE_STDERR=line).write_post_stages(P, [])
    assert not isinstance(e.value, EngineRefused) and str(e.value).endswith(line)
    refused = runner(FAKE_ENGINE_EXIT="2", FAKE_ENGINE_STDERR=f"refused: {P} is not a folder",
                     FAKE_ENGINE_STDOUT=json.dumps({"written": False, "problems": [f"{P} is not a folder"]}, indent=1))
    with pytest.raises(EngineRefused) as e:
        refused.write_post_stages(P, [])
    assert str(e.value).splitlines()[1:] == [f"refused: {P} is not a folder"] and e.value.exit_code == 2


def test_both_runners_list_write_refuse_and_fail_post_stages_alike(tmp_path, monkeypatch):
    """On an engine with post-stages, the fixture post-stage installed from metadata where each runner looks (the
    child's PYTHONPATH, this process's sys.path): both runners list the same post-stages; write the same
    post_stages.yaml with the same record (an empty list too, which switches every post-stage off); refuse the
    same problems, one line each, and a root that is not a folder, writing nothing; and fail, not refuse, a write
    the file system stops (a folder where post_stages.yaml goes: the command's exit 1)."""
    need_features("post_stages")
    from germquant.post_stages import validate_post_stages
    from germquant.post_stages.testing import FIXTURE_NAME, write_fixture_distribution

    dist = write_fixture_distribution(tmp_path / "dist")
    monkeypatch.syspath_prepend(str(dist))
    child_path = os.pathsep.join(p for p in (str(dist), os.environ.get("PYTHONPATH")) if p)
    runners = {"subprocess": SubprocessRunner(sys.executable, env={"PYTHONPATH": child_path}),
               "inprocess": InProcessRunner()}
    listed = {k: r.list_post_stages() for k, r in runners.items()}
    assert listed["subprocess"] == listed["inprocess"]
    assert FIXTURE_NAME in [row["name"] for row in listed["inprocess"]]

    def problems_of(e: EngineRefused) -> list[str]:      # each refused line's problem, whatever the head and case
        return [ln.split(": ", 1)[1] for ln in str(e).splitlines() if ln.lower().startswith("refused: ")]

    roots = {k: tmp_path / k for k in runners}
    for entries, written in (([{"name": FIXTURE_NAME, "params": {"factor": 3, "statistic": "max"}}],
                              [{"name": FIXTURE_NAME, "params": {"factor": 3.0, "statistic": "max"}}]), ([], [])):
        recs = {}
        for k, r in runners.items():
            roots[k].mkdir(exist_ok=True)
            recs[k] = r.write_post_stages(roots[k], entries)
        data = {k: (roots[k] / "post_stages.yaml").read_bytes() for k in runners}
        assert data["subprocess"] == data["inprocess"]
        for k in runners:
            assert recs[k] == {"written": True, "path": str(roots[k] / "post_stages.yaml"),
                               "sha256": hashlib.sha256(data[k]).hexdigest(), "entries": written}, k
            # the entries as written: the float parameter given as 3 reads back as 3.0
            assert [type(v) for e in recs[k]["entries"] for v in e["params"].values()] == \
                [type(v) for e in written for v in e["params"].values()], k
    bad = [{"name": FIXTURE_NAME, "params": {"factor": 99.0, "nope": 1}}, {"name": "not_there"}]
    problems = validate_post_stages(bad)
    assert len(problems) == 3
    a_file = tmp_path / "a_file.txt"
    a_file.write_bytes(b"x")
    for k, r in runners.items():
        with pytest.raises(EngineRefused) as e:
            r.write_post_stages(roots[k], bad)
        assert problems_of(e.value) == problems and e.value.exit_code == 2, k
        for root in (tmp_path / "missing", a_file):
            with pytest.raises(EngineRefused) as e:
                r.write_post_stages(root, bad)
            assert problems_of(e.value) == [f"{root} is not a folder"], (k, root)
    assert {k: (roots[k] / "post_stages.yaml").read_bytes() for k in runners} == data      # nothing written
    assert not (tmp_path / "missing").exists() and a_file.read_bytes() == b"x"
    for k, r in runners.items():
        blocked = tmp_path / f"blocked_{k}"
        (blocked / "post_stages.yaml").mkdir(parents=True)
        with pytest.raises(RunnerError) as e:
            r.write_post_stages(blocked, [{"name": FIXTURE_NAME, "params": {}}])
        assert not isinstance(e.value, EngineRefused), (k, str(e.value))
        if k == "subprocess":
            assert "failed (exit 1)" in str(e.value) and "could not be written" in str(e.value)
        else:
            assert isinstance(e.value.__cause__, OSError), repr(e.value.__cause__)
        assert [p.name for p in blocked.iterdir()] == ["post_stages.yaml"] and (blocked / "post_stages.yaml").is_dir()
