"""The pure-python glue: profiles, recipes, channel-role rows, stage rows and the RunPlan (its overrides, the
command text and the exported overlay), all without Qt."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from napari_germquant import _engine
from napari_germquant._engine import Recipe, RunPlan, role_rows, stage_rows
from germquant import api

KNOWN_PROFILES = {"rad51_foci", "generic", "segmentation_only", "n2_sc", "ccw77_partition"}
THREE = ["405", "477", "545"]


def test_profiles_come_from_the_checkout(engine_root):
    profs = _engine.profiles()
    names = {p.name for p in profs}
    assert KNOWN_PROFILES <= names
    for p in profs:
        assert p.path.is_file() and p.path.parent == (engine_root / "config" / "profiles").resolve()
    assert len(names) == len(profs)                     # one entry per name


def test_recipe_is_a_profile_or_a_file(engine_root, tmp_path):
    r = Recipe.from_profile("rad51_foci")
    assert r.cli_args() == ["--profile", "rad51_foci"] and r.label == "rad51_foci"
    assert r.path() == (engine_root / "config" / "profiles" / "rad51_foci.yaml").resolve()
    assert r.load().get("coloc.enabled") is False      # the profile's own switch
    f = tmp_path / "study.yaml"
    f.write_text(f"extends: {r.path().as_posix()}\n", encoding="utf-8")
    rf = Recipe.from_file(f)
    assert rf.cli_args() == ["--config", str(f)] and rf.label == "study.yaml" and rf.path() == f.resolve()
    assert rf.load().get("coloc.enabled") is False
    with pytest.raises(ValueError):
        Recipe()
    with pytest.raises(ValueError):
        Recipe(profile="a", config=Path("b"))


def test_role_rows_three_channel_gonad():
    cfg = Recipe.from_profile("rad51_foci").load()
    rows = role_rows(cfg, THREE)
    by = {r.role: r for r in rows}
    assert list(by) == ["dna", "central_element", "foci", "granule"]
    assert (by["dna"].index, by["dna"].how, by["dna"].required, by["dna"].channel_name) == (0, "name", True, "405")
    assert (by["central_element"].index, by["central_element"].required) == (1, True)
    assert by["foci"].index == 2 and by["foci"].marker == "RAD-51"
    assert by["granule"].index is None and not by["granule"].problem and "optional" in by["granule"].status
    assert not any(r.problem for r in rows)
    assert _engine.unassigned_channels(rows, THREE) == []
    assert _engine.role_to_index(rows) == {"dna": 0, "central_element": 1, "foci": 2, "granule": None}


def test_role_rows_one_channel_marks_the_required_role():
    cfg = Recipe.from_profile("rad51_foci").load()
    by = {r.role: r for r in role_rows(cfg, ["DAPI"])}
    assert by["dna"].index == 0 and by["dna"].how == "name"
    assert by["central_element"].problem and by["central_element"].status == "UNBOUND (required)"
    assert by["foci"].index is None and not by["foci"].problem


def test_role_rows_index_fallback_is_amber_not_red():
    cfg = Recipe.from_profile("generic").load()
    rows = role_rows(cfg, ["DAPI", "488", "640"])
    by = {r.role: r for r in rows}
    assert by["dna"].how == "name" and by["foci"].how == "index" and by["granule"].how == "index"
    assert not any(r.problem for r in rows)
    assert "position" in by["foci"].status
    assert _engine.unassigned_channels(rows, ["DAPI", "488", "640", "extra"]) == [(3, "extra")]


def test_stage_rows_follow_the_profile_and_the_roles():
    cfg = Recipe.from_profile("rad51_foci").load()
    rows = stage_rows(cfg, {"dna": 0, "central_element": 1, "foci": 2, "granule": None})
    assert [r.name for r in rows] == [s.name for s in api.cli_switches()]
    by = {r.name: r for r in rows}
    for name in ("spots", "germline", "axis", "signal_qc"):
        assert by[name].selectable and by[name].reason == "", name
    assert not by["coloc"].on_in_profile and not by["coloc"].selectable and "off in this profile" in by["coloc"].reason
    assert not by["envelope"].on_in_profile and not by["sc_trace"].on_in_profile
    # one channel: the stages that need SYP or RAD-51 are greyed with the role named
    one = {r.name: r for r in stage_rows(cfg, {"dna": 0, "central_element": None, "foci": None, "granule": None})}
    assert one["germline"].missing == ("central_element",) and "central_element" in one["germline"].reason
    assert one["spots"].missing == ("foci",) and not one["spots"].selectable
    assert one["axis"].selectable                      # needs no role


def test_run_plan_overrides_argv_and_commands():
    image = Path("C:/data/gonad 1.nd2")
    out = Path("C:/out/gonad 1_results")
    plan = RunPlan(image=image, recipe=Recipe.from_profile("rad51_foci"), out=out, xy_stride=2,
                   disabled=("spots", "signal_qc"))
    assert plan.overrides() == {"spots.enabled": False, "signal_qc.enabled": False}
    assert plan.run_argv() == ["run", str(image), "--profile", "rad51_foci", "--out", str(out), "--xy-stride", "2",
                               "--no-spots", "--no-signal_qc"]
    cmd = plan.run_command()
    assert cmd.startswith('germquant run "') and f'"{image}"' in cmd and f'"{out}"' in cmd
    assert cmd.endswith("--xy-stride 2 --no-spots --no-signal_qc")
    batch = plan.batch_argv()
    assert batch[:2] == ["batch", str(image.parent)] and batch[2:4] == ["--profile", "rad51_foci"]
    assert batch[batch.index("--out") + 1] == str(out.parent / "data_results")
    assert plan.batch_command(Path("D:/all"), Path("D:/all_results")).startswith("germquant batch D:")
    plain = RunPlan(image=Path("g.nd2"), recipe=Recipe.from_profile("generic"), out=Path("o"))
    assert plain.run_argv() == ["run", "g.nd2", "--profile", "generic", "--out", "o"] and plain.overrides() == {}
    with pytest.raises(ValueError, match="switchable"):
        RunPlan(image=image, recipe=Recipe.from_profile("generic"), out=out, disabled=("segment",))
    with pytest.raises(ValueError, match="xy_stride"):
        RunPlan(image=image, recipe=Recipe.from_profile("generic"), out=out, xy_stride=0)


def test_export_writes_a_loadable_overlay_and_the_commands(tmp_path, engine_root):
    plan = RunPlan(image=tmp_path / "g.nd2", recipe=Recipe.from_profile("rad51_foci"), out=tmp_path / "g_results",
                   disabled=("spots",))
    overlay, commands = plan.export(tmp_path / "export")
    assert overlay.name == "rad51_foci_g.yaml" and commands.name == "rad51_foci_g_commands.txt"
    text = overlay.read_text(encoding="utf-8")
    assert text.startswith("# Profile overlay exported by napari-germquant")
    mapping = yaml.safe_load(text)
    assert list(mapping) == ["validated", "extends", "spots"]
    assert mapping["validated"] is False and mapping["spots"] == {"enabled": False}
    assert Path(mapping["extends"]) == (engine_root / "config" / "profiles" / "rad51_foci.yaml").resolve()
    # the engine loads it: the switch took, the base profile's own switches came through, the hash is its own
    cfg = api.load_config(overlay)
    base = plan.recipe.load()
    assert cfg.get("spots.enabled") is False and cfg.get("coloc.enabled") is False and cfg.get("validated") is False
    assert cfg.channel_map.roles.keys() == base.channel_map.roles.keys()
    assert cfg.get("io.channel_map") and Path(cfg.get("io.channel_map")).is_file()   # anchored to the checkout
    assert cfg.hash != base.hash
    body = commands.read_text(encoding="utf-8")
    assert plan.run_command() in body and plan.batch_command() in body
    overlay_line = [ln for ln in body.splitlines() if "--config" in ln][-1]
    assert str(overlay) in overlay_line and "--no-spots" not in overlay_line
    assert "run_timestamp" in body and str(api.__version__) in body


def test_the_runner_line_of_the_commands_file_runs_in_powershell(monkeypatch):
    """PowerShell reads a line that starts with a quoted string as an expression and rejects `-m`: on Windows a
    quoted interpreter path gets the call operator & (with a note for cmd); an unquoted one, or another system,
    gets none."""
    plan = RunPlan(image=Path("D:/data/a b.nd2"), recipe=Recipe.from_profile("rad51_foci"), out=Path("D:/data/a b_r"))
    tail = ' -m germquant run "' + str(Path("D:/data/a b.nd2")) + '" --profile rad51_foci --out "' + \
        str(Path("D:/data/a b_r")) + '"'
    for windows, spaced, want_amp in ((True, True, True), (True, False, False), (False, True, False)):
        monkeypatch.setattr(_engine, "_on_windows", lambda w=windows: w)
        python = "C:/Program Files/Python311/python.exe" if spaced else "C:/py311/python.exe"
        body = plan.commands_text(engine={"python": python, "line": "0.2.0.dev0"})
        lines = body.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("# The same, as the dock's engine runner starts it"))
        quoted = f'"{python}"' if spaced else python
        assert lines[i + 1] == ("& " if want_amp else "") + quoted + tail, (windows, spaced)
        assert ("PowerShell needs the & before a quoted interpreter path; in cmd leave the & out" in lines[i]) == \
            want_amp
