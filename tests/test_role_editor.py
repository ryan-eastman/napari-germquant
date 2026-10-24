"""The role editor of the Recipe tab (tranche 5 Train B, unit P6-RECIPE; design 7.3 and 8.5, plugin_specs 4.2, P6
acceptance 4): one combo per role of the pack recipe's channel map, listing the channels of a reference image;
a preview for every image of the project, from `project edit --dry-run --json` (the engine code that writes the
map, never a plugin copy of the resolver); Apply through `project edit` (the engine runner, D39), which writes
channel_map.yaml and recipe.yaml in ASCII; red (a required role unbound, two roles on one channel) blocks Apply.

The acceptance tests run the engine's project commands, so they need the projects feature (need_features); the
widget tests further down use a fake engine and run against any engine."""
from __future__ import annotations

import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml
from _p6_recipe import (
    GAMMA,
    RENAMED,
    FakeRunner,
    FakeSession,
    engine_edit,
    fake_view,
    no_machine_path,
    nofallback_pack,
    preview_record,
    recipe_files,
    renamed_example,
)
from conftest import need_features

from napari_germquant import _settings
from napari_germquant._recipe_widget import AMBER, RED, RecipeTab
from napari_germquant._runner import EngineRefused, InProcessRunner, SubprocessRunner

PACK_HASH = "02bb83cf74f9"           # test_pack@0.1.0/basic, unedited


def _settle(tab, qtbot, timeout: int = 180_000) -> None:
    qtbot.waitUntil(lambda: tab.idle, timeout=timeout)


def _tab(qtbot, root=None, **kwargs) -> RecipeTab:
    tab = RecipeTab(None, **kwargs)
    qtbot.addWidget(tab)
    if root is not None:
        tab.set_project(root)
        _settle(tab, qtbot)
    return tab


def _items(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def _results(root, image_id) -> dict:
    """Every file a run left under results/<image_id>/: the tables as text frames (run_timestamp dropped), the
    parquet tables, the done marker without finished, and the names of the other files."""
    import json

    folder = root / "results" / image_id
    out = {}
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = f.relative_to(folder).as_posix()
        if f.suffix == ".csv":
            out[rel] = pd.read_csv(f, dtype=str, keep_default_na=False).drop(columns=["run_timestamp"],
                                                                             errors="ignore")
        elif f.suffix == ".parquet":
            out[rel] = pd.read_parquet(f).drop(columns=["run_timestamp"], errors="ignore")
        elif f.name.endswith("__done.json"):
            out[rel] = {k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if k != "finished"}
        else:
            out[rel] = None
    return out


# ---------------------------------------------------------------------------------------------------
# acceptance 4: the engine's project commands (need the projects feature)
# ---------------------------------------------------------------------------------------------------


def test_role_edit_writes_ascii_maps_and_runs_headless(project_env, qtbot):
    """A study whose channel names no token of the pack map matches (H2B-mCherry, Rad52-GFP): the pack map binds
    both roles by the index fallback (amber). Binding each role to its channel by name through the tab writes
    channel_map.yaml and recipe.yaml, both ASCII with no machine path, recipe.yaml extending the pinned pack
    recipe; the tab shows the engine's new config_hash and validated: false; and the dock's run of the edited
    project and a headless `germquant project run` of a copy give the same files and tables."""
    need_features("projects")
    _settings.set_engine_choice(sys.executable)
    runner = SubprocessRunner(sys.executable)
    image = renamed_example(project_env.images, RENAMED)
    root = project_env.new_dir()
    new = runner.project_new(root, "test_pack@0.1.0", images=image)
    (image_id,) = new["added"]

    tab = _tab(qtbot, root)
    assert tab.last_error is None, tab.status_label.text()
    assert tab.config_hash == PACK_HASH and "not validated (provisional pack)" in tab.badge_label.text()
    assert list(tab.role_combos) == ["dna", "foci"]
    items = _items(tab.role_combos["dna"])
    assert items[0] == "pack map: DAPI, Hoechst, DNA" and items[1:] == ["0: H2B-mCherry", "1: Rad52-GFP"]
    assert tab.role_choice("dna") is None and tab.role_choice("foci") is None
    assert tab.preview_states() == {image_id: {"dna": "index", "foci": "index"}}
    assert tab.preview_table.item(0, 1).background().color() == AMBER
    assert not tab.apply_roles_btn.isEnabled()                     # nothing to apply yet

    tab.choose_channel("dna", "H2B-mCherry")
    tab.choose_channel("foci", "Rad52-GFP")
    _settle(tab, qtbot)
    assert tab.role_choices() == {"dna": "H2B-mCherry", "foci": "Rad52-GFP"}
    assert tab.preview_states() == {image_id: {"dna": "name", "foci": "name"}}
    assert tab.role_problems == [] and tab.apply_roles_btn.isEnabled()
    after = tab.roles_hash
    assert after not in (None, PACK_HASH)
    assert recipe_files(root) == {"recipe.yaml": None, "channel_map.yaml": None}   # a preview writes nothing

    applied = []
    tab.applied.connect(applied.append)
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    assert len(applied) == 1 and applied[0]["config_hash"] == after
    files = recipe_files(root)
    for name, data in files.items():
        assert data is not None, name
        text = data.decode("ascii")                                # ASCII, with escapes
        assert no_machine_path(text, root), (name, text)
    assert yaml.safe_load(files["recipe.yaml"]) == {"extends": "pack:test_pack@0.1.0/basic",
                                                    "io": {"channel_map": "channel_map.yaml"}}
    roles = yaml.safe_load(files["channel_map.yaml"])["roles"]
    assert roles["dna"] == {"names": ["H2B-mCherry"], "index": 0, "required": True, "marker": "DNA"}
    assert roles["foci"] == {"names": ["Rad52-GFP"], "index": 1, "required": False, "marker": "foci"}
    # the tab shows the hash the engine computes for the project now, and validated: false
    assert engine_edit(root, "--dry-run")["config_hash"] == after == tab.config_hash
    assert after in tab.hash_label.text() and "validated: false (edited)" in tab.badge_label.text()
    assert tab.role_choice("dna") == "H2B-mCherry" and tab.role_choice("foci") == "Rad52-GFP"
    assert tab.preview_states() == {image_id: {"dna": "name", "foci": "name"}}
    assert not tab.apply_roles_btn.isEnabled()                     # applied: nothing left to apply
    assert tab.others_label.text() == ""                           # its own io.channel_map is no hand edit

    # the dock's run (its in-process route, gui-inprocess: no second torch start in CI) against a headless
    # `germquant project run` of a copy
    cli_root = project_env.new_dir("cli")
    shutil.copytree(root, cli_root)
    res = InProcessRunner().project_run(root)
    assert res["exit_code"] == 0 and res["state"] == "done", res
    proc = subprocess.run([sys.executable, "-m", "germquant", "project", "run", str(cli_root)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    gui, cli = _results(root, image_id), _results(cli_root, image_id)
    assert sorted(gui) == sorted(cli) and any(k.endswith("__done.json") for k in gui)
    for name in gui:
        if isinstance(gui[name], pd.DataFrame):
            pd.testing.assert_frame_equal(gui[name], cli[name], check_exact=True, obj=name)
        else:
            assert gui[name] == cli[name], name
    marker = next(v for k, v in gui.items() if k.endswith("__done.json"))
    assert marker["config_hash"] == after
    summary = next(v for k, v in gui.items() if k.endswith("__image_summary.csv"))
    assert "name_unmatched_used_index" not in ";".join(summary["qc_flags"])   # the run bound the roles by name


def test_collision_and_unbound_block_apply(project_env, qtbot, tmp_path, monkeypatch):
    """Two roles on one channel, and a required role that binds no channel on an image, show red in the preview,
    list their problems, and keep Apply off; the engine refuses the same edit, and nothing is written."""
    need_features("projects")
    _settings.set_engine_choice(sys.executable)
    runner = SubprocessRunner(sys.executable)
    root = project_env.new_dir()
    new = runner.project_new(root, "test_pack@0.1.0", images=project_env.image)
    (image_id,) = new["added"]
    tab = _tab(qtbot, root)
    assert tab.preview_states() == {image_id: {"dna": "name", "foci": "name"}}
    tab.choose_channel("foci", "DAPI")                             # dna binds DAPI by name too
    _settle(tab, qtbot)
    assert tab.preview_states() == {image_id: {"dna": "collision", "foci": "collision"}}
    assert tab.preview_table.item(0, 1).background().color() == RED
    assert tab.preview_table.item(0, 2).background().color() == RED
    assert len(tab.role_problems) == 1 and "bind the same channel 0" in tab.role_problems[0]
    assert tab.role_problems[0] in tab.problems_label.text()
    assert not tab.apply_roles_btn.isEnabled()
    assert not tab.apply_roles() and "problem" in tab.status_label.text()
    with pytest.raises(EngineRefused, match="bind the same channel"):
        runner.project_edit(root, roles=tab.role_choices())
    assert recipe_files(root) == {"recipe.yaml": None, "channel_map.yaml": None}
    tab.choose_channel("foci", None)                               # back to the pack map: clean again
    _settle(tab, qtbot)
    assert tab.preview_states() == {image_id: {"dna": "name", "foci": "name"}} and tab.role_problems == []

    # a pack whose required role dna has no index fallback: an image with other names leaves it unbound
    from germquant import packs

    monkeypatch.setenv("GERMQUANT_PACKS", str(nofallback_pack(tmp_path / "packs")))
    packs.refresh()
    other = renamed_example(project_env.images, RENAMED, name="other.ome.tif")
    root2 = project_env.new_dir()
    new2 = runner.project_new(root2, "nofallback_pack@0.1.0", images=[project_env.image, other])
    assert new2["added"] == [image_id, "other"], new2
    tab2 = _tab(qtbot, root2)
    states = tab2.preview_states()
    assert states["other"]["dna"] == "missing" and states[image_id]["dna"] == "name"
    tab2.set_reference_image("other")
    _settle(tab2, qtbot)
    assert _items(tab2.role_combos["dna"])[1:] == ["0: H2B-mCherry", "1: Rad52-GFP"]
    tab2.choose_channel("dna", "H2B-mCherry")                      # binds on other, unbinds on the first image
    _settle(tab2, qtbot)
    states = tab2.preview_states()
    assert states == {image_id: {"dna": "missing", "foci": "name"}, "other": {"dna": "name", "foci": "index"}}
    row = [tab2.preview_table.item(r, 0).text() for r in range(tab2.preview_table.rowCount())].index(image_id)
    assert tab2.preview_table.item(row, 1).background().color() == RED
    assert any("required role dna binds no channel" in p for p in tab2.role_problems)
    assert not tab2.apply_roles_btn.isEnabled() and not tab2.apply_roles()
    with pytest.raises(EngineRefused, match="binds no channel"):
        runner.project_edit(root2, roles=tab2.role_choices())
    assert recipe_files(root2) == {"recipe.yaml": None, "channel_map.yaml": None}


def test_revert_roles_when_the_pack_map_leaves_a_problem(project_env, qtbot, tmp_path, monkeypatch):
    """Revert roles goes back to the pack map also when that map leaves a required role unbound on an image,
    which is when a role edit is needed (review S2): it drops the role edits (`project edit --unset
    io.channel_map`), which no role problem refuses, and keeps the form's edits."""
    need_features("projects")
    from germquant import packs

    _settings.set_engine_choice(sys.executable)
    runner = SubprocessRunner(sys.executable)
    monkeypatch.setenv("GERMQUANT_PACKS", str(nofallback_pack(tmp_path / "packs")))
    packs.refresh()
    other = renamed_example(project_env.images, RENAMED, name="other.ome.tif")
    root = project_env.new_dir()
    runner.project_new(root, "nofallback_pack@0.1.0", images=[other])
    pack_hash = engine_edit(root, "--dry-run")["config_hash"]
    tab = _tab(qtbot, root)
    assert tab.preview_states() == {"other": {"dna": "missing", "foci": "index"}}
    tab.choose_channel("dna", "H2B-mCherry")
    _settle(tab, qtbot)
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert tab.last_error is None and recipe_files(root)["channel_map.yaml"] is not None
    tab.set_value("spots.log.min_snr", 3.5)
    assert tab.apply_form()
    _settle(tab, qtbot)
    runner.project_add(root, [project_env.image])                  # the saved map leaves dna unbound on this one
    tab.reload()
    _settle(tab, qtbot)
    assert any("required role dna binds no channel" in p for p in tab.role_problems)
    assert tab.revert_roles_btn.isEnabled() and tab.revert_roles()
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    assert "REFUSED" not in tab.status_label.text() and "removed channel_map.yaml" in tab.status_label.text()
    files = recipe_files(root)
    assert files["channel_map.yaml"] is None
    assert yaml.safe_load(files["recipe.yaml"]) == {"extends": "pack:nofallback_pack@0.1.0/basic",
                                                    "spots": {"log": {"min_snr": 3.5}}}
    assert tab.role_choice("dna") is None and tab.config_hash == engine_edit(root, "--dry-run")["config_hash"]
    assert tab.config_hash != pack_hash                            # the form's edit stays
    assert tab.preview_states()["other"]["dna"] == "missing"       # the pack map's own problem, shown again


def test_non_ascii_channel_name_round_trips(project_env, qtbot):
    """A channel named with a Greek gamma (gammaH2AX): the combo offers it, the edit binds foci to it by name,
    channel_map.yaml holds it as a YAML escape (ASCII) that reads back as the same name, the reopened tab shows
    it chosen, and Revert goes back to the pack map and the pack's config_hash."""
    need_features("projects")
    _settings.set_engine_choice(sys.executable)
    runner = SubprocessRunner(sys.executable)
    name = GAMMA + "H2AX"
    image = renamed_example(project_env.images, ["DAPI", name])
    root = project_env.new_dir()
    (image_id,) = runner.project_new(root, "test_pack@0.1.0", images=image)["added"]
    tab = _tab(qtbot, root)
    assert _items(tab.role_combos["foci"])[1:] == ["0: DAPI", f"1: {name}"]
    assert tab.preview_states() == {image_id: {"dna": "name", "foci": "index"}}
    tab.choose_channel("foci", name)
    _settle(tab, qtbot)
    assert tab.preview_states() == {image_id: {"dna": "name", "foci": "name"}}
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    data = recipe_files(root)["channel_map.yaml"]
    text = data.decode("ascii")
    assert "\\u03B3H2AX" in text or "\\u03b3H2AX" in text
    assert yaml.safe_load(text)["roles"]["foci"]["names"] == [name]
    edited = tab.config_hash
    assert edited != PACK_HASH

    again = _tab(qtbot, root)                                      # the project opened again
    assert again.role_choice("foci") == name and again.role_choice("dna") is None
    assert again.config_hash == edited
    assert again.preview_states() == {image_id: {"dna": "name", "foci": "name"}}
    assert again.revert_roles()
    _settle(again, qtbot)
    assert again.last_error is None, again.status_label.text()
    assert again.config_hash == PACK_HASH and recipe_files(root) == {"recipe.yaml": None, "channel_map.yaml": None}
    assert again.role_choice("foci") is None and "not validated (provisional pack)" in again.badge_label.text()


# ---------------------------------------------------------------------------------------------------
# the widget against a fake engine (any engine)
# ---------------------------------------------------------------------------------------------------


def _bind(dna=("DAPI", "name"), foci=("foci", "name"), image="img1") -> dict:
    return {image: {"dna": {"channel": dna[0], "how": dna[1]}, "foci": {"channel": foci[0], "how": foci[1]}}}


def test_the_role_editor_sends_every_role_and_shows_the_engines_preview(qtbot, tmp_path):
    """Without an engine: the combos list the pack map, then the reference image's channels; every preview and
    write sends one choice per role ("" for the pack map); the preview cells take the engine's bindings and
    problems; a collision is red and keeps Apply off; Apply writes once and says so."""
    def answer(kw):
        roles = kw.get("roles") or {}
        if roles.get("foci") == "DAPI":
            return preview_record(_bind(foci=("DAPI", "name")), ["image img1: the roles dna and foci bind the same "
                                                                 "channel 0 (DAPI)"], config_hash="cccccccccccc")
        if kw.get("dry_run"):
            changed = any(roles.values())
            return preview_record(_bind(), config_hash="bbbbbbbbbbbb" if changed else "02bb83cf74f9",
                                  changed=changed)
        return {**preview_record(_bind(), config_hash="bbbbbbbbbbbb"), "dry_run": False,
                "written": ["channel_map.yaml", "recipe.yaml", "germquant_project.yaml"]}

    fake = FakeRunner(answer)
    views = [fake_view(tmp_path)]
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake), reader=lambda root, reference=None: views[-1])
    assert tab.last_error is None, tab.status_label.text()
    assert fake.calls == [{"root": str(tmp_path), "roles": {"dna": "", "foci": ""}, "dry_run": True}]
    assert _items(tab.role_combos["foci"]) == ["pack map: foci, puncta, spots", "0: DAPI", "1: foci"]
    assert tab.role_combos["dna"].itemText(0) == "pack map: DAPI, Hoechst, DNA"
    assert "dna *" in [tab.preview_table.horizontalHeaderItem(c).text() for c in range(3)]
    assert tab.preview_states() == {"img1": {"dna": "name", "foci": "name"}}
    tab.choose_channel("foci", "DAPI")
    _settle(tab, qtbot)
    assert fake.calls[-1]["roles"] == {"dna": "", "foci": "DAPI"} and fake.calls[-1]["dry_run"] is True
    assert tab.preview_states() == {"img1": {"dna": "collision", "foci": "collision"}}
    assert tab.preview_table.item(0, 2).background().color() == RED and not tab.apply_roles_btn.isEnabled()
    assert "bind the same channel" in tab.problems_label.text()
    tab.choose_channel("foci", "foci")
    _settle(tab, qtbot)
    assert tab.preview_states() == {"img1": {"dna": "name", "foci": "name"}} and tab.role_problems == []
    assert tab.apply_roles_btn.isEnabled() and tab.roles_hash == "bbbbbbbbbbbb"
    assert "bbbbbbbbbbbb" in tab.roles_hash_label.text()
    applied = []
    tab.applied.connect(applied.append)
    views.append(fake_view(tmp_path, config_hash="bbbbbbbbbbbb", digest="d1",
                           roles=[r.with_edit("foci" if r.role == "foci" else None) for r in views[0].roles]))
    n = len(fake.calls)
    tab.apply_roles_btn.click()
    _settle(tab, qtbot)
    writes = [c for c in fake.calls[n:] if not c.get("dry_run")]
    assert writes == [{"root": str(tmp_path), "roles": {"dna": "", "foci": "foci"}}]
    assert [a["config_hash"] for a in applied] == ["bbbbbbbbbbbb"]
    assert tab.config_hash == "bbbbbbbbbbbb" and tab.role_choice("foci") == "foci"
    assert "wrote channel_map.yaml, recipe.yaml" in tab.status_label.text()
    assert not tab.apply_roles_btn.isEnabled()


def test_collisions_are_the_engines_not_shared_names(qtbot, tmp_path):
    """Two roles bound to two channels that share a name are no collision: the engine judges by channel index and
    reports none, so the cells are not red (review N4). A collision the engine reports is red on the roles it
    names, on that image only."""
    from napari_germquant._params import collision_roles, preview_states

    roles = fake_view(tmp_path).roles
    same_name = {"img1": {"dna": {"channel": "Cy5", "how": "index"}, "foci": {"channel": "Cy5", "how": "index"}}}
    assert preview_states(preview_record(same_name), roles) == {"img1": {"dna": "index", "foci": "index"}}
    clash = {**_bind(foci=("DAPI", "name")), **_bind(image="img2")}
    rec = preview_record(clash, ["image img1: the roles dna and foci bind the same channel 0 (DAPI)"])
    assert preview_states(rec, roles) == {"img1": {"dna": "collision", "foci": "collision"},
                                          "img2": {"dna": "name", "foci": "name"}}
    assert collision_roles(["image img10: the roles dna and foci bind the same channel 0 (DAPI)"], "img1") == set()
    assert collision_roles(["image img1: the required role dna binds no channel"], "img1") == set()

    fake = FakeRunner(lambda kw: preview_record(same_name, changed=False, config_hash=PACK_HASH))
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake),
               reader=lambda root, reference=None: fake_view(tmp_path, channels=("Cy5", "Cy5")))
    assert tab.preview_states() == {"img1": {"dna": "index", "foci": "index"}}
    assert [tab.preview_table.item(0, c).background().color() for c in (1, 2)] == [AMBER, AMBER]
    assert tab.role_problems == [] and tab.problems_label.text() == ""


def test_a_choice_the_pack_map_already_names_is_no_role_change(qtbot, tmp_path):
    """A role choice that the pack map's role already names, alone, is no edit by the engine's rule (its dry run
    says changed: false, with the same config_hash): the tab says so and keeps Apply off, instead of a write that
    rewrites only the manifest (review N2)."""
    from napari_germquant._params import Role
    from napari_germquant._recipe_widget import NO_ROLE_CHANGE_TEXT

    def answer(kw):
        roles = kw.get("roles") or {}
        real = {k: v for k, v in roles.items() if v and not (k == "foci" and v == "foci")}   # the engine's rule
        if kw.get("dry_run"):
            return preview_record(_bind(), config_hash="bbbbbbbbbbbb" if real else PACK_HASH, changed=bool(real))
        return {**preview_record(_bind(), config_hash="bbbbbbbbbbbb"), "dry_run": False,
                "written": ["channel_map.yaml", "recipe.yaml", "germquant_project.yaml"]}

    fake = FakeRunner(answer)
    roles = [Role("dna", "DNA", True, ("DAPI", "Hoechst", "DNA"), 0, None), Role("foci", "foci", False, ("foci",), 1,
                                                                                None)]
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake),
               reader=lambda root, reference=None: fake_view(tmp_path, roles=roles))
    tab.choose_channel("foci", "foci")
    _settle(tab, qtbot)
    assert tab.role_choices() == {"dna": "", "foci": "foci"} and tab.roles_hash == PACK_HASH
    assert tab.roles_hash_label.text() == NO_ROLE_CHANGE_TEXT and not tab.apply_roles_btn.isEnabled()
    assert not tab.apply_roles() and "no role change" in tab.status_label.text()
    assert all(c.get("dry_run") for c in fake.calls)
    tab.choose_channel("dna", "DAPI")                              # narrowing dna's names is an edit
    _settle(tab, qtbot)
    assert "with these roles: config_hash bbbbbbbbbbbb" in tab.roles_hash_label.text()
    assert tab.apply_roles_btn.isEnabled()


def test_revert_roles_sends_one_unset_and_previews_again(qtbot, tmp_path):
    """Revert roles with role edits saved is one `project edit --unset io.channel_map` (no --role: a role
    write is refused while the pack map leaves a problem, review S2); the preview is then asked for again. With
    no edit saved it only drops the unapplied choices, and writes nothing."""
    def answer(kw):
        if kw.get("dry_run"):
            changed = any((kw.get("roles") or {}).values())
            return preview_record(_bind(), config_hash="bbbbbbbbbbbb" if changed else PACK_HASH, changed=changed)
        return {**preview_record(_bind(), config_hash=PACK_HASH), "preview": None, "dry_run": False,
                "written": ["germquant_project.yaml"], "removed": ["recipe.yaml", "channel_map.yaml"]}

    fake = FakeRunner(answer)
    edited = fake_view(tmp_path, config_hash="bbbbbbbbbbbb", digest="d1")
    edited.roles[1] = edited.roles[1].with_edit("foci")
    views = [edited]
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake), reader=lambda root, reference=None: views[-1])
    assert tab.role_choice("foci") == "foci" and tab.revert_roles_btn.isEnabled()
    views.append(fake_view(tmp_path, digest="d2"))
    n = len(fake.calls)
    assert tab.revert_roles()
    _settle(tab, qtbot)
    assert fake.calls[n] == {"root": str(tmp_path), "unset": ["io.channel_map"]}
    assert fake.calls[n + 1:] == [{"root": str(tmp_path), "roles": {"dna": "", "foci": ""}, "dry_run": True}]
    assert tab.role_choice("foci") is None and tab.config_hash == PACK_HASH
    assert "removed recipe.yaml, channel_map.yaml" in tab.status_label.text()
    tab.choose_channel("dna", "DAPI")                              # nothing saved: the choice is dropped here
    _settle(tab, qtbot)
    n = len(fake.calls)
    assert tab.revert_roles() and tab.role_choice("dna") is None
    _settle(tab, qtbot)
    assert all(c.get("dry_run") for c in fake.calls[n:])


def test_amber_and_unreadable_headers_and_a_required_role_unbound(qtbot, tmp_path):
    """The preview cell states: by index (amber), a required role unbound (red), an optional one unbound (no
    colour), an image whose header the engine could not read (left out, its warning shown); and an edit naming a
    channel the reference image does not have still shows as chosen."""
    record = preview_record({"img1": {"dna": {"channel": "H2B", "how": "index"}, "foci": {"channel": None,
                                                                                         "how": "unbound"}},
                             "img2": {"dna": {"channel": None, "how": "unbound"}, "foci": {"channel": "x",
                                                                                          "how": "name"}}},
                            ["image img2: the required role dna binds no channel"],
                            warnings=["image img3: header unreadable (OSError: gone); its bindings are not shown"])
    fake = FakeRunner(lambda kw: record)
    images = [{"id": i, "path": f"images/{i}.ome.tif", "excluded": False} for i in ("img1", "img2", "img3")]
    view = fake_view(tmp_path, images=images, channels=("H2B", "x"))
    view.roles[1] = view.roles[1].with_edit("Cy5 RAD-51")             # an edit made on another image
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake), reader=lambda root, reference=None: view)
    assert tab.preview_states() == {"img1": {"dna": "index", "foci": "unbound"},
                                    "img2": {"dna": "missing", "foci": "name"}}
    assert tab.preview_table.item(0, 1).background().color() == AMBER
    assert tab.preview_table.item(1, 1).background().color() == RED
    assert tab.preview_table.item(0, 2).background().color() not in (RED, AMBER)
    assert "img3: header unreadable" in tab.warnings_label.text()
    assert tab.role_choice("foci") == "Cy5 RAD-51"
    assert _items(tab.role_combos["foci"])[-1] == "Cy5 RAD-51 (not a channel of img1)"
    assert not tab.apply_roles_btn.isEnabled()                     # a problem is left


def test_a_run_in_progress_and_a_refused_write_leave_the_project_as_it_was(qtbot, tmp_path):
    """While the host says a run of the project goes, the role editor and the form take no edit; a write the
    engine refuses shows its reason, emits nothing and keeps the choices, and the project is read again (a write
    that stops part way has changed files, review N10)."""
    def answer(kw):
        if not kw.get("dry_run"):
            return EngineRefused("`germquant.project edit` was refused by the engine (exit 2):\nREFUSED: the project "
                                 "is locked by host pc pid 1 (project run)")
        return preview_record(_bind(), config_hash="bbbbbbbbbbbb" if any((kw.get("roles") or {}).values())
                              else "02bb83cf74f9", changed=any((kw.get("roles") or {}).values()))

    reads = []

    def reader(root, reference=None):
        reads.append(str(root))
        return fake_view(tmp_path)

    fake = FakeRunner(answer)
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake), reader=reader)
    tab.set_running(True)
    assert not tab.role_combos["dna"].isEnabled() and not tab.apply_roles_btn.isEnabled()
    assert "a run of this project is in progress" in tab.status_label.text()
    tab.choose_channel("foci", "DAPI")                             # the programmatic twin is refused too
    assert tab.role_choice("foci") is None and not tab.apply_roles()
    tab.set_running(False)
    assert tab.role_combos["dna"].isEnabled()
    tab.choose_channel("dna", "DAPI")
    _settle(tab, qtbot)
    applied = []
    tab.applied.connect(applied.append)
    n = len(reads)
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert applied == [] and "the project is locked" in tab.status_label.text()
    assert tab.role_choice("dna") == "DAPI" and tab.apply_roles_btn.isEnabled()    # the choice is kept
    assert len(reads) == n + 1                                     # the project read again after the refusal
