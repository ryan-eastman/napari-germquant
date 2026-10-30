"""The engine runners' commands of tranche 5 Train B (unit P6-RUNNER; the interface of section 9 of the Train B
plan): one method per command, each building the command line the engine documents (stand-in engines from
tests/_p6_runner.py answer with what they were given), the torch-free entries used where they exist, the
plugin version stamped on every project write, ``project=`` on the pack calls, the device list of the runtime
probe, the refusal forms of the Train B commands, and the in-process runner: the same commands through
germquant.api, but its project writes through the torch-free command of this environment (and, on an engine
with projects, both runners writing the same project files)."""
from __future__ import annotations

import inspect
import json
import os
import sys
import types
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
    assert last("germquant") == ["tune-spots", "--project", P, "--apply", "tuning/t1"] and res["exit_code"] == 0
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
    fake_post = types.ModuleType("germquant.post_stages")
    fake_post.write_post_stages_file = lambda path, entries, **k: writes.append((Path(path), entries)) or Path(path)
    monkeypatch.setitem(sys.modules, "germquant.post_stages", fake_post)

    r = InProcessRunner()
    assert {"projects", "import_labels", "tune_spots", "post_stages"} <= r.features()
    P = project_env.new_dir()
    assert r.project_status(P) == {"format": 1, "images": []}
    assert calls[-2][0] == "open_project" and calls[-2][1] == (P,)
    assert isinstance(calls[-1][1][0], _FakeProject) and calls[-1][1][0].root == Path(P)
    assert r.project_collect(P) == {"tables": []}
    out = r.project_export_commands(P, shell="ps1", out=project_env.root / "cmds.ps1")
    assert out["text"] == "germquant project run X\n" and (project_env.root / "cmds.ps1").read_text() == out["text"]
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
    assert r.tune_apply(P, "tuning/t1")["config_hash"] == "fedcba987654"
    assert r.list_post_stages() == [{"name": "fixture_post"}]
    r.write_post_stages(P, [{"name": "fixture_post", "params": {}}])
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
