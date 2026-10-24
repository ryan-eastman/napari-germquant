"""The engine runners' commands of tranche 5 Train B (unit P6-RUNNER; the interface of section 9 of the Train B
plan): one method per command, each building the command line the engine documents (stand-in engines from
tests/_p6_runner.py answer with what they were given), the torch-free entries used where they exist, the
plugin version stamped on every project write, ``project=`` on the pack calls, the device list of the runtime
probe, and the in-process runner calling the same commands through germquant.api."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from _p6_runner import echo, standin_engine
from conftest import TEST_PACK, copy_pack, engine_device

from napari_germquant import _engine
from napari_germquant._runner import (
    VIA_INPROCESS,
    EngineRefused,
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
from germquant.stages import BY_NAME

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
    assert out["out"] == str(tmp_path / "cmds.ps1") and echo(out["text"].splitlines())["module"] == "germquant"
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
    r.train_curve("pairs", "curve", sizes=[1, 2, 4, "all"], repeats=3, seed=7, base="tinynet@f5055f8a",
                  keep_models="best")
    assert last("germquant") == ["train-curve", "--pairs", "pairs", "--out", "curve", "--sizes", "1,2,4,all",
                                 "--repeats", "3", "--seed", "7", "--base", "tinynet@f5055f8a", "--keep-models",
                                 "best", "--json"]
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
    """The same commands in this napari environment: germquant.api's project functions, with the project passed
    explicitly (I11), the plugin version on every write, and project runs through api.run_project with
    via=gui-inprocess, the run id, a progress callback that tails the engine's log, and should_stop."""
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

    for name, ret in {"open_project": _FakeProject, "new_project": lambda root, *a, **k: _FakeProject(root),
                      "add_images": {"added": ["x"], "refused": []}, "write_recipe": "abc123abc123",
                      "revert_recipe": "02bb83cf74f9", "save_project": None,
                      "preview_roles": {"yaml": "roles: {}\n", "bindings": {}, "problems": []},
                      "use_model": "def456def456", "list_rounds": ["r01"],
                      "new_round": lambda p, name=None, **k: Path(p.root) / "training" / (name or "r02"),
                      "break_lock": True, "project_status": {"format": 1, "images": []},
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

    r, v = InProcessRunner(), plugin_version()
    assert {"projects", "import_labels", "tune_spots", "post_stages"} <= r.features()
    P = project_env.new_dir()
    doc = r.project_new(P, "test_pack@0.1.0", recipe="basic", images=["a.tif"], name="N")
    assert calls[-1] == ("new_project", (P, "test_pack@0.1.0", "basic", ["a.tif"]), {"name": "N",
                                                                                      "plugin_version": v})
    assert doc["root"] == str(P) and doc["manifest"] == {"name": "fake", "format": 1}
    assert r.project_add(P, ["b.tif"]) == {"added": ["x"], "refused": []}
    assert calls[-1][0] == "add_images" and calls[-1][1][1] == ["b.tif"] and calls[-1][2] == {"plugin_version": v}
    assert isinstance(calls[-1][1][0], _FakeProject) and calls[-1][1][0].root == Path(P)
    res = r.project_edit(P, edits={"spots.log.min_snr": 2.5}, unset=["a.b"], roles={"foci": "Rad52-GFP"},
                         device="cpu", xy_stride=2, disable=["spots"])
    # the recipe write first, then the run settings on the project as it is after that write
    assert [c[0] for c in calls[-4:]] == ["open_project", "write_recipe", "open_project", "save_project"]
    assert calls[-3][2] == {"edits": {"spots.log.min_snr": 2.5}, "unset": ("a.b",), "roles": {"foci": "Rad52-GFP"},
                            "origin": "form", "plugin_version": v}
    assert calls[-1][2] == {"plugin_version": v, "device": "cpu", "xy_stride": 2,
                            "overrides": {BY_NAME["spots"].config_key: False}}     # what --no-spots sets
    assert res == {"config_hash": "abc123abc123", "dry_run": False}
    n = len(calls)
    preview = r.project_edit(P, roles={"foci": "Rad52-GFP"}, edits={"a.b": 1}, dry_run=True)
    assert preview == {"config_hash": None, "dry_run": True,
                       "preview": {"yaml": "roles: {}\n", "bindings": {}, "problems": []}}
    assert [c[0] for c in calls[n:]] == ["open_project", "preview_roles"]            # a dry run writes nothing
    assert r.project_edit(P, revert=True)["config_hash"] == "02bb83cf74f9"
    assert r.project_use_model(P, "training/r01/models/m/m")["config_hash"] == "def456def456"
    assert r.project_rounds(P) == {"rounds": ["r01"], "created": None, "path": None}
    assert r.project_rounds(P, new="r09") == {"rounds": ["r01"], "created": "r09",
                                              "path": str(Path(P) / "training" / "r09")}
    with pytest.raises(ValueError, match="a revert cannot be combined"):
        r.project_edit(P, revert=True, edits={"spots.log.min_snr": 2.0})
    assert r.project_unlock(P, host="h", pid=7)["removed"] is True
    assert calls[-1][2] == {"host": "h", "pid": 7}
    assert r.project_status(P) == {"format": 1, "images": []}
    assert r.project_collect(P) == {"tables": []}
    out = r.project_export_commands(P, shell="ps1", out=project_env.root / "cmds.ps1")
    assert out["text"] == "germquant project run X\n" and (project_env.root / "cmds.ps1").read_text() == out["text"]
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
    when there was no lock, and that is no removal. A revert with edits is refused before anything starts."""
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
    n = len(gone.history)
    with pytest.raises(ValueError, match="a revert cannot be combined"):
        gone.project_edit(P, revert=True, roles={"foci": "x"})
    assert len(gone.history) == n                                       # nothing ran
    plain = runner()
    plain.project_edit(P, z_range="none", device="none")
    assert plain.history[-1][3:] == ["edit", P, "--device", "none", "--z-range", "none", "--plugin-version",
                                     plugin_version(), "--json"]


def test_an_in_process_edit_means_what_the_command_means(project_env, monkeypatch):
    """--role adds to the project's role edits (an empty channel drops one), --no-<stage> adds to its switched-off
    stages, --unset of a stage key turns that stage back on, and none clears a run setting: the in-process edit
    hands the API the same result the command line computes."""
    from germquant.stages import BY_NAME

    one, two = [s for s in BY_NAME.values() if s.cli_switchable][:2]
    calls: list[tuple] = []

    class Recorded(_FakeProject):
        def __init__(self, root):
            super().__init__(root)
            self.manifest = {"recipe": {"roles": {"dna": "DAPI", "foci": "old"},
                                        "overrides": {one.config_key: False}}}

    monkeypatch.setattr(api, "open_project", Recorded, raising=False)
    for name, ret in (("write_recipe", "aaaaaaaaaaaa"), ("save_project", None), ("preview_roles", {})):
        monkeypatch.setattr(api, name, lambda *a, _n=name, _r=ret, **k: calls.append((_n, a, k)) or _r,
                            raising=False)
    r, P = InProcessRunner(), project_env.new_dir()
    res = r.project_edit(P, roles={"foci": "Rad52-GFP", "dna": ""}, unset=[one.config_key, "spots.log.min_snr"],
                         disable=[two.name], device="none", z_range="none", runner="none",
                         interpreters={"basic": "none"})
    (_, _, wr), (_, _, sp) = calls
    assert wr["roles"] == {"foci": "Rad52-GFP"} and wr["unset"] == ("spots.log.min_snr",)
    assert sp["overrides"] == {two.config_key: False}                  # one is back on, two is off
    assert (sp["device"], sp["runner"], sp["z_range"], sp["interpreters"]) == ("", "", (), {"basic": None})
    assert res["config_hash"] == "aaaaaaaaaaaa"
    calls.clear()
    r.project_edit(P, roles={"foci": "x"}, dry_run=True)
    assert calls == [("preview_roles", (calls[0][1][0], {"dna": "DAPI", "foci": "x"}), {})]
