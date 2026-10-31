"""The parameter form of the Recipe tab (tranche 5 Train B, unit P6-RECIPE; design 7.3 and 8.5, plugin_specs 4.3,
P6 acceptance 5 and 10): built only from the pack recipe's tunables, one widget per declared type with its unit
and range, the project's value next to the pack's, Reset per row; keys set outside the form (tuning, Use in
project, a hand edit) are shown read only; Apply writes through `project edit` (the engine runner, D39), which
validates by origin, and every Apply shows the new config_hash; with every edit reset the project hashes as the
pack recipe again (I7).

The post-stages (acceptance 10): the form lists the post-stages the engine environment offers, each with its
declared parameters, a switch and parameter widgets; Apply writes post_stages.yaml through the engine runner
(`python -m germquant.post_stages write`, D39), which leaves config_hash and validated as they are.

The acceptance tests run the engine's project commands, so they need the projects feature, and the post-stage one
the post_stages feature too (need_features); the widget tests and the rule tests further down run against any
engine."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest
import yaml
from _p6_recipe import (
    FIXTURE,
    RING,
    FakePostRunner,
    FakeRunner,
    FakeSession,
    engine_edit,
    fake_view,
    post_rows,
    preview_record,
    recipe_files,
)
from conftest import need_features
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from napari_germquant import _engine, _settings
from napari_germquant._params import (
    MISSING,
    RESET,
    PostList,
    Tunable,
    entries_text,
    form_change,
    hash_note,
    post_entries,
    post_list,
    post_list_changed,
    post_stages_from_listing,
    post_state,
    read_post_entries,
    recipe_edits,
    validated_badge,
)
from napari_germquant._recipe_widget import RUNNING_TEXT, WRITING_TEXT, RecipeTab
from napari_germquant._runner import EngineRefused, FeatureMissing, InProcessRunner, SubprocessRunner

PACK_HASH = "02bb83cf74f9"           # test_pack@0.1.0/basic, unedited
MIN_VOL, MIN_SNR = "segmentation.nuclei.min_volume_um3", "spots.log.min_snr"


def _settle(tab, qtbot, timeout: int = 180_000) -> None:
    qtbot.waitUntil(lambda: tab.idle, timeout=timeout)


def _tab(qtbot, root=None, **kwargs) -> RecipeTab:
    tab = RecipeTab(None, **kwargs)
    qtbot.addWidget(tab)
    if root is not None:
        tab.set_project(root)
        _settle(tab, qtbot)
    return tab


def _project(project_env, pack="test_pack@0.1.0"):
    """A project the engine makes (the in-process runner: this environment's own project command) on ``pack``,
    with the test pack's example; the Recipe tab then runs on the same engine."""
    _settings.set_engine_choice(_settings.INPROCESS)
    runner = InProcessRunner()
    root = project_env.new_dir()
    runner.project_new(root, pack, images=project_env.image)
    return runner, root


def _edits(root) -> dict:
    data = recipe_files(root)["recipe.yaml"]
    return {} if data is None else yaml.safe_load(data.decode("ascii"))


# ---------------------------------------------------------------------------------------------------
# acceptance 5: the engine's project commands (need the projects feature)
# ---------------------------------------------------------------------------------------------------


def test_form_has_only_declared_tunables(project_env, qtbot):
    """The test pack's basic recipe declares two tunables: the form has exactly those two rows, as spin boxes
    with the declared range and unit, the project's value next to the pack's. An edit writes that one key into
    recipe.yaml through the engine, and the tab shows the engine's new config_hash and validated: false. A key
    set outside the form (a hand edit of recipe.yaml) is shown read only and gets no widget."""
    need_features("projects")
    runner, root = _project(project_env)
    tab = _tab(qtbot, root)
    assert tab.last_error is None, tab.status_label.text()
    assert list(tab.form_rows) == [MIN_VOL, MIN_SNR]
    vol, snr = tab.form_rows[MIN_VOL], tab.form_rows[MIN_SNR]
    assert isinstance(vol.widget, QDoubleSpinBox) and isinstance(snr.widget, QDoubleSpinBox)
    assert (vol.widget.minimum(), vol.widget.maximum(), vol.widget.suffix()) == (0.0, 50.0, " um3")
    assert (snr.widget.minimum(), snr.widget.maximum(), snr.widget.suffix()) == (1.0, 10.0, "")
    assert vol.widget.value() == 2.0 and snr.widget.value() == 2.0      # basic.yaml's values
    assert vol.label.text() == "Smallest nucleus" and MIN_VOL in vol.label.toolTip()
    assert vol.pack_label.text() == "pack 2.0" and vol.note_label.text() == ""
    assert not vol.reset_btn.isEnabled() and tab.others_label.text() == ""
    assert tab.config_hash == PACK_HASH and not tab.apply_form_btn.isEnabled()

    tab.set_value(MIN_SNR, 3.5)
    assert tab.pending() == {MIN_SNR: 3.5} and tab.apply_form_btn.isEnabled()
    _settle(tab, qtbot)
    after = tab.form_hash
    assert after not in (None, PACK_HASH) and after in tab.form_hash_label.text()
    assert recipe_files(root)["recipe.yaml"] is None                   # the preview writes nothing
    applied = []
    tab.applied.connect(applied.append)
    assert tab.apply_form()
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    assert _edits(root) == {"extends": "pack:test_pack@0.1.0/basic", "spots": {"log": {"min_snr": 3.5}}}
    assert [a["config_hash"] for a in applied] == [after]
    assert tab.config_hash == after == engine_edit(root, "--dry-run")["config_hash"]
    assert after in tab.hash_label.text() and "validated: false (edited)" in tab.badge_label.text()
    snr = tab.form_rows[MIN_SNR]
    assert snr.widget.value() == 3.5 and snr.pack_label.text() == "pack 2.0"
    assert snr.note_label.text() == "edited (the parameter form)" and snr.reset_btn.isEnabled()
    assert tab.pending() == {} and not tab.apply_form_btn.isEnabled()

    # a key the form does not edit, set by a hand edit of recipe.yaml: read only, no widget
    p = root / "recipe.yaml"
    p.write_bytes(p.read_bytes() + b"  effect_size_min: 4.0\n")
    tab.reload()
    _settle(tab, qtbot)
    assert list(tab.form_rows) == [MIN_VOL, MIN_SNR]
    assert "spots.effect_size_min = 4.0 (a hand edit of recipe.yaml)" in tab.others_label.text()
    assert "changed outside germquant" in tab.warnings_label.text()    # open_project's warning, shown


def test_undeclared_key_is_refused(project_env, qtbot):
    """The form has no widget for a key the recipe does not declare, and its programmatic twin refuses one, and a
    value outside the declared range; the engine refuses both edits too, and nothing is written."""
    need_features("projects")
    runner, root = _project(project_env)
    tab = _tab(qtbot, root)
    assert "spots.spot_radius_um" not in tab.form_rows
    with pytest.raises(ValueError, match=r"spots\.spot_radius_um is not a tunable of the pack recipe"):
        tab.set_value("spots.spot_radius_um", 0.5)
    with pytest.raises(ValueError, match="must be at most 10"):
        tab.set_value(MIN_SNR, 12.0)
    with pytest.raises(ValueError, match="takes a float"):
        tab.set_value(MIN_SNR, "high")
    assert tab.pending() == {} and not tab.apply_form_btn.isEnabled()
    with pytest.raises(EngineRefused, match="is not a tunable of the pack recipe"):
        runner.project_edit(root, edits={"spots.spot_radius_um": 0.5})
    with pytest.raises(EngineRefused, match="must be at most 10"):
        runner.project_edit(root, edits={MIN_SNR: 12.0})
    assert recipe_files(root) == {"recipe.yaml": None, "channel_map.yaml": None}
    tab.reload()
    _settle(tab, qtbot)
    assert tab.config_hash == PACK_HASH

    # a recipe.yaml the manifest does not record (put there by hand): the form's dry run warns that the write
    # would refuse to replace it, the form shows the warning (review N8), and the write is refused
    (root / "recipe.yaml").write_bytes(b"extends: pack:test_pack@0.1.0/basic\n")
    tab.reload()
    _settle(tab, qtbot)
    tab.set_value(MIN_SNR, 3.0)
    _settle(tab, qtbot)
    assert "\nwarning: " in tab.form_hash_label.text() and "not recorded in the manifest" in tab.form_hash_label.text()
    assert tab.apply_form()
    _settle(tab, qtbot)
    assert "REFUSED" in tab.status_label.text() and "not recorded in the manifest" in tab.status_label.text()
    assert recipe_files(root)["recipe.yaml"] == b"extends: pack:test_pack@0.1.0/basic\n"


def test_the_served_model_pack_declares_the_spot_radius(project_env, served_model_pack, qtbot):
    """The codeless round's edit (P6 acceptance 3, D38): served_model_pack's basic recipe also declares
    spots.spot_radius_um, so the form offers it with the pack's value 0.3; editing it changes the config_hash and
    shows validated: false."""
    need_features("projects")
    runner, root = _project(project_env, pack=served_model_pack.ref)
    tab = _tab(qtbot, root)
    assert tab.last_error is None, tab.status_label.text()
    assert served_model_pack.tunable in tab.form_rows
    row = tab.form_rows[served_model_pack.tunable]
    assert row.widget.value() == pytest.approx(0.3) and row.pack_label.text() == "pack 0.3"
    tab.set_value(served_model_pack.tunable, 0.5)
    assert tab.apply_form()
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    assert _edits(root)["spots"] == {"spot_radius_um": 0.5}
    assert tab.config_hash not in (None, PACK_HASH) and "validated: false (edited)" in tab.badge_label.text()


def test_reset_returns_to_the_pack_hash(project_env, qtbot):
    """Two edits, then Reset on each in turn: with the last edit reset recipe.yaml is gone and the project hashes
    as the pack recipe again; a value set back to the pack's own is no edit; and Back to the pack recipe drops a
    role edit and a form edit together."""
    need_features("projects")
    runner, root = _project(project_env)
    tab = _tab(qtbot, root)
    tab.set_value(MIN_SNR, 3.5)
    tab.set_value(MIN_VOL, 5.0)
    assert tab.apply_form()
    _settle(tab, qtbot)
    two = tab.config_hash
    assert _edits(root) == {"extends": "pack:test_pack@0.1.0/basic", "segmentation": {"nuclei": {"min_volume_um3":
                                                                                                5.0}},
                            "spots": {"log": {"min_snr": 3.5}}}
    tab.reset(MIN_SNR)
    assert tab.pending() == {MIN_SNR: RESET} and tab.form_rows[MIN_SNR].widget.value() == 2.0
    assert tab.apply_form()
    _settle(tab, qtbot)
    one = tab.config_hash
    assert one not in (two, PACK_HASH)
    assert _edits(root) == {"extends": "pack:test_pack@0.1.0/basic", "segmentation": {"nuclei": {"min_volume_um3":
                                                                                                5.0}}}
    assert not tab.form_rows[MIN_SNR].reset_btn.isEnabled() and tab.form_rows[MIN_VOL].reset_btn.isEnabled()
    tab.set_value(MIN_VOL, 2.0)                                       # the pack's own value: no edit
    assert tab.apply_form()
    _settle(tab, qtbot)
    assert recipe_files(root)["recipe.yaml"] is None and tab.config_hash == PACK_HASH
    assert "not validated (provisional pack)" in tab.badge_label.text()
    assert not tab.form_rows[MIN_VOL].reset_btn.isEnabled()

    tab.choose_channel("dna", "DAPI")
    _settle(tab, qtbot)
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert tab.others_label.text() == ""                           # the role editor's own io.channel_map
    tab.set_value(MIN_SNR, 4.0)
    assert tab.apply_form()
    _settle(tab, qtbot)
    assert all(v is not None for v in recipe_files(root).values()) and tab.config_hash != PACK_HASH
    assert tab.revert_all(confirm=False)
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    assert recipe_files(root) == {"recipe.yaml": None, "channel_map.yaml": None} and tab.config_hash == PACK_HASH
    assert tab.role_choice("dna") is None and tab.form_rows[MIN_SNR].widget.value() == 2.0
    assert tab.hash_label.text() == f"config_hash {PACK_HASH} (the pack recipe)"

    # a stage switch off (the Run tab's, `--no-spots`) enters the hash: the line names it (review N7)
    switched = engine_edit(root, "--no-spots")["config_hash"]
    tab.reload()
    _settle(tab, qtbot)
    assert tab.config_hash == switched != PACK_HASH
    assert tab.hash_label.text() == f"config_hash {switched} (the pack recipe, with the stage switch spots off)"


# ---------------------------------------------------------------------------------------------------
# acceptance 10: the post-stages (need the projects and post_stages features)
# ---------------------------------------------------------------------------------------------------

ENTRIES = [{"name": FIXTURE, "params": {"factor": 2.5, "statistic": "max"}}]
UNCHANGED = f"config_hash {PACK_HASH} unchanged (post_stages.yaml is not part of config_hash)"


def _switch_the_fixture_on(tab, qtbot) -> None:
    """Through the form: the fixture post-stage on, its factor 2.5 and its statistic max; then Apply."""
    tab.set_post_stage(FIXTURE, True)
    tab.set_post_param(FIXTURE, "factor", 2.5)
    tab.set_post_param(FIXTURE, "statistic", "max")
    assert tab.post_entries() == ENTRIES and tab.post_changed and tab.apply_post_btn.isEnabled()
    assert tab.post_hash_label.text() == f"with these post-stages: {UNCHANGED}"
    assert tab.apply_post_stages()
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()


def test_post_stage_switch_keeps_the_hash_and_matches_headless(project_env, qtbot, tmp_path, monkeypatch,
                                                                 request):
    """P6 acceptance 10 (design 8.5 and 7.3, plugin_specs 4.3; I12 to I14; D39). The engine's fixture post-stage is
    installed as an on-disk distribution (germquant.post_stages.testing.write_fixture_distribution, I14) on the
    engine's PYTHONPATH and on this process's sys.path. The form lists it through the engine runner with its
    declared parameters, typed like tunables; switching it on with two parameters changed writes post_stages.yaml
    through `python -m germquant.post_stages write` (the subprocess runner: the engine environment validates,
    D39), byte for byte the engine's canonical file, and leaves config_hash, validated, the manifest and the recipe
    files as they were, on the tab and in the engine's own answer. The in-process runner lists the same and writes
    the same bytes. The dock's run of the project (the subprocess runner's project run, as the Project dock's Run
    all) and a headless `germquant project run` of a copy give the same post table, made with the form's
    parameters."""
    need_features("projects", "post_stages")
    from germquant import cli
    from germquant.post_stages import dump_post_stages
    from germquant.post_stages.testing import FIXTURE_PACKAGE, FIXTURE_TABLE, write_fixture_distribution

    dist = write_fixture_distribution(tmp_path / "post_dist")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in (str(dist), os.environ.get("PYTHONPATH")) if p))
    monkeypatch.syspath_prepend(str(dist))                       # this process: in process, and the headless run
    request.addfinalizer(lambda: sys.modules.pop(FIXTURE_PACKAGE, None))
    _settings.set_engine_choice(sys.executable)                  # the engine: this interpreter, in its own process
    engine = SubprocessRunner(sys.executable)
    root = project_env.new_dir("gui")
    engine.project_new(root, "test_pack@0.1.0", images=project_env.image)
    twin = project_env.new_dir("twin")
    shutil.copytree(root, twin)                                  # for the in-process half
    manifest = (root / "germquant_project.yaml").read_bytes()

    tab = _tab(qtbot, root)
    assert tab.last_error is None and tab.post_error is None, tab.post_label.text()
    assert tab.config_hash == PACK_HASH and tab.post_stage_names() == [FIXTURE]
    row = tab.post_rows[FIXTURE]
    assert not row.switch.isChecked() and list(row.params) == ["factor", "min_voxels", "statistic", "fail"]
    factor = row.params["factor"].widget
    assert isinstance(factor, QDoubleSpinBox) and (factor.minimum(), factor.maximum(), factor.value()) == (0.1, 10.0,
                                                                                                           1.0)
    assert isinstance(row.params["min_voxels"].widget, QSpinBox) and isinstance(row.params["fail"].widget, QCheckBox)
    choice = row.params["statistic"].widget
    assert [choice.itemData(i) for i in range(choice.count())] == ["mean", "max"]
    assert tab.post_label.text().startswith("In force: none") and not tab.apply_post_btn.isEnabled()
    hash_line = tab.hash_label.text()
    written = []
    tab.post_written.connect(written.append)
    _switch_the_fixture_on(tab, qtbot)
    text = (root / "post_stages.yaml").read_bytes()
    assert text == dump_post_stages(ENTRIES).encode("ascii")                 # the engine's own file
    assert [Path(w["path"]) for w in written] == [root / "post_stages.yaml"]
    assert (root / "germquant_project.yaml").read_bytes() == manifest        # no recipe edit (design 7.3)
    assert recipe_files(root) == {"recipe.yaml": None, "channel_map.yaml": None}
    assert tab.config_hash == PACK_HASH == engine_edit(root, "--dry-run")["config_hash"]
    assert tab.hash_label.text() == hash_line and "not validated (provisional pack)" in tab.badge_label.text()
    assert UNCHANGED in tab.status_label.text()
    assert tab.post_switch(FIXTURE) and tab.post_value(FIXTURE, "factor") == 2.5     # the file, read back
    assert tab.post_value(FIXTURE, "statistic") == "max" and not tab.apply_post_btn.isEnabled()
    assert tab.post_label.text().startswith("In force: the project's post_stages.yaml: fixture_nuclei (factor 2.5, "
                                            "statistic max)")

    _settings.set_engine_choice(_settings.INPROCESS)                          # in process: the same file
    tab2 = _tab(qtbot, twin)
    assert tab2.post_error is None and tab2.post_stage_names() == [FIXTURE], tab2.post_label.text()
    _switch_the_fixture_on(tab2, qtbot)
    assert (twin / "post_stages.yaml").read_bytes() == text

    headless = project_env.new_dir("cli")
    shutil.copytree(root, headless)                                           # switched on, never run
    res = engine.project_run(root)                                            # the Project dock's Run all
    assert res["state"] == "done", res.get("message")
    assert cli.main(["project", "run", str(headless)]) == 0
    image_id = "test_pack_synthetic"
    table = f"{image_id}__post_{FIXTURE}_{FIXTURE_TABLE}.csv"
    dock, cmd = (p / "results" / image_id for p in (root, headless))
    ta, tb = (pd.read_csv(d / table, float_precision="round_trip") for d in (dock, cmd))
    pd.testing.assert_frame_equal(ta, tb, check_exact=True)
    assert len(ta) > 0
    pd.testing.assert_series_equal(ta["foci_value_scaled"], ta["foci_value"] * 2.5, check_names=False)
    for d in (dock, cmd):
        stage = json.loads((d / f"{image_id}__post.json").read_text(encoding="utf-8"))["stages"][FIXTURE]
        assert stage["done"] and stage["error"] is None and stage["tables"] == [table]
        assert stage["params"] == {"factor": 2.5, "min_voxels": 0, "statistic": "max", "fail": False}


# ---------------------------------------------------------------------------------------------------
# the widget against a fake engine (any engine)
# ---------------------------------------------------------------------------------------------------

DECLS = [
    ({"key": "a.float", "label": "A float", "type": "float", "unit": "um", "min": 0.0, "max": 5.0}, 1.25, 1.25, None),
    ({"key": "a.int", "label": "An int", "type": "int", "min": 1, "max": 9}, 3, 4, "form"),
    ({"key": "a.bool", "type": "bool"}, True, True, None),
    ({"key": "a.choice", "label": "A choice", "type": "choice", "choices": ["max", 2, 0.5]}, "max", 2, "tuning"),
    ({"key": "a.word", "label": "A word", "type": "str"}, "x", "x", None),
    ({"key": "a.list", "label": "A list", "type": "list"}, [1, 2], [1, 2], None),
]


def _form_tab(qtbot, tmp_path, answer=None, views=None):
    fake = FakeRunner(answer or (lambda kw: preview_record({}, config_hash="ffffffffffff")))
    views = views if views is not None else [fake_view(tmp_path, tunables=DECLS)]
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake), reader=lambda root, reference=None: views[-1])
    return tab, fake, views


def test_the_form_has_one_widget_per_declared_type(qtbot, tmp_path):
    tab, fake, _ = _form_tab(qtbot, tmp_path)
    rows = tab.form_rows
    assert list(rows) == ["a.float", "a.int", "a.bool", "a.choice", "a.word", "a.list"]
    assert isinstance(rows["a.float"].widget, QDoubleSpinBox) and rows["a.float"].widget.suffix() == " um"
    assert (rows["a.float"].widget.minimum(), rows["a.float"].widget.maximum()) == (0.0, 5.0)
    assert rows["a.float"].widget.value() == 1.25
    assert isinstance(rows["a.int"].widget, QSpinBox) and rows["a.int"].widget.value() == 4
    assert (rows["a.int"].widget.minimum(), rows["a.int"].widget.maximum()) == (1, 9)
    assert isinstance(rows["a.bool"].widget, QCheckBox) and rows["a.bool"].widget.isChecked()
    combo = rows["a.choice"].widget
    assert isinstance(combo, QComboBox) and [combo.itemData(i) for i in range(combo.count())] == ["max", 2, 0.5]
    assert combo.currentData() == 2 and isinstance(combo.currentData(), int)
    assert isinstance(rows["a.word"].widget, QLineEdit) and rows["a.word"].widget.text() == "x"
    assert rows["a.list"].widget is None and "shown read only" in rows["a.list"].note_label.text()
    assert rows["a.bool"].label.text() == "a.bool"                       # no label declared: the key
    assert rows["a.int"].note_label.text() == "edited (the parameter form)" and rows["a.int"].reset_btn.isEnabled()
    assert rows["a.choice"].note_label.text() == "set by a tuning run (Spot tuning, Apply)"
    assert rows["a.int"].pack_label.text() == "pack 3" and rows["a.choice"].pack_label.text() == "pack max"
    assert not rows["a.float"].reset_btn.isEnabled()
    with pytest.raises(ValueError, match="shown read only"):
        tab.set_value("a.list", [3])
    # the widget's own changes become pending edits (the programmatic twin goes through the widgets)
    rows["a.float"].widget.setValue(2.5)
    rows["a.bool"].widget.setChecked(False)
    combo.setCurrentIndex(0)
    rows["a.word"].widget.setText("y")
    rows["a.word"].widget.editingFinished.emit()
    assert tab.pending() == {"a.float": 2.5, "a.bool": False, "a.choice": "max", "a.word": "y"}
    rows["a.float"].widget.setValue(1.25)                                # back to the project's value
    assert "a.float" not in tab.pending()
    _settle(tab, qtbot)


def test_form_edits_and_resets_become_one_project_edit(qtbot, tmp_path):
    """Pending values and resets are previewed and then written in one `project edit`: the values as edits, the
    resets of edited keys as unset keys; the write's record is announced and the form reloads. A tunable binds no
    channel, so the role preview carries over the write with the new hash (no second role dry run)."""
    binds = {"img1": {"dna": {"channel": "DAPI", "how": "name"}, "foci": {"channel": "foci", "how": "index"}}}

    def answer(kw):
        if kw.get("roles") is not None:
            return preview_record(binds, config_hash="02bb83cf74f9", changed=False)
        if kw.get("dry_run"):
            return {**preview_record({}, config_hash="eeeeeeeeeeee"), "preview": None}
        return {**preview_record({}, config_hash="dddddddddddd"), "preview": None, "dry_run": False,
                "written": ["recipe.yaml", "germquant_project.yaml"]}

    tab, fake, views = _form_tab(qtbot, tmp_path, answer)
    n = len(fake.calls)
    tab.set_value("a.float", 3.0)
    tab.reset("a.int")                                                  # an edited key: unset
    tab.reset("a.float")                                                # never edited: just dropped
    tab.set_value("a.float", 0.5)
    assert tab.pending() == {"a.float": 0.5, "a.int": RESET}
    assert tab.form_rows["a.int"].widget.value() == 3                   # the pack value shown
    _settle(tab, qtbot)
    dry = [c for c in fake.calls[n:] if c.get("dry_run")]
    assert dry and dry[-1] == {"root": str(tmp_path), "edits": {"a.float": 0.5}, "unset": ["a.int"],
                               "dry_run": True}
    assert tab.form_hash == "eeeeeeeeeeee" and "eeeeeeeeeeee" in tab.form_hash_label.text()
    applied = []
    tab.applied.connect(applied.append)
    views.append(fake_view(tmp_path, tunables=[(d, p, 0.5 if d["key"] == "a.float" else (p if d["key"] == "a.int"
                                                                                         else v),
                                                "form" if d["key"] == "a.float" else (None if d["key"] == "a.int"
                                                                                     else o))
                                               for d, p, v, o in DECLS], config_hash="dddddddddddd", digest="d1"))
    role_runs = len([c for c in fake.calls if "roles" in c])
    assert role_runs == 1 and tab.preview_states() == {"img1": {"dna": "name", "foci": "index"}}
    assert tab.apply_form()
    _settle(tab, qtbot)
    writes = [c for c in fake.calls if not c.get("dry_run")]
    assert writes == [{"root": str(tmp_path), "edits": {"a.float": 0.5}, "unset": ["a.int"]}]
    assert [a["config_hash"] for a in applied] == ["dddddddddddd"] and tab.pending() == {}
    assert tab.config_hash == "dddddddddddd" and tab.form_rows["a.float"].widget.value() == 0.5
    assert "wrote recipe.yaml" in tab.status_label.text()
    assert len([c for c in fake.calls if "roles" in c]) == role_runs                # carried over
    assert tab.preview_states() == {"img1": {"dna": "name", "foci": "index"}} and tab.roles_hash == "dddddddddddd"


def test_no_edit_is_taken_while_a_write_runs(qtbot, tmp_path):
    """While a write is queued or running, every edit control is off and every edit is refused with the reason
    (review S1): the write's answer resets the pending values and the role choices, so an edit taken then would
    be lost without a word. The form's and the role editor's own widgets are put back as well."""
    binds = {"img1": {"dna": {"channel": "DAPI", "how": "name"}, "foci": {"channel": "foci", "how": "name"}}}

    def answer(kw):
        roles = kw.get("roles")
        if roles is not None:
            changed = any(roles.values())
            rec = preview_record(binds, changed=changed, config_hash="bbbbbbbbbbbb" if changed else PACK_HASH)
            return rec if kw.get("dry_run") else {**rec, "dry_run": False,
                                                  "written": ["channel_map.yaml", "recipe.yaml",
                                                              "germquant_project.yaml"]}
        if kw.get("dry_run"):
            return {**preview_record({}, config_hash="eeeeeeeeeeee"), "preview": None}
        return {**preview_record({}, config_hash="dddddddddddd"), "preview": None, "dry_run": False,
                "written": ["recipe.yaml", "germquant_project.yaml"]}

    def writes():
        return [c for c in fake.calls if not c.get("dry_run")]

    tab, fake, _views = _form_tab(qtbot, tmp_path, answer)
    tab.set_value("a.float", 2.5)
    _settle(tab, qtbot)
    fake.gate = threading.Event()                                       # hold the write
    assert tab.apply_form()
    qtbot.waitUntil(lambda: len(writes()) == 1, timeout=30_000)
    int_box = tab.form_rows["a.int"].widget
    assert not int_box.isEnabled() and not tab.role_combos["foci"].isEnabled()
    assert not tab.form_rows["a.int"].reset_btn.isEnabled() and not tab.apply_form_btn.isEnabled()
    tab.set_value("a.int", 5)
    assert tab.pending() == {"a.float": 2.5} and "a write is in progress" in tab.status_label.text()
    int_box.setValue(6)                                                 # the widget's own signal: put back
    assert tab.pending() == {"a.float": 2.5} and int_box.value() == 4
    tab.choose_channel("foci", "DAPI")
    assert tab.role_choice("foci") is None and not tab.revert_all(confirm=False)
    fake.gate.set()
    fake.gate = None
    _settle(tab, qtbot)
    assert writes() == [{"root": str(tmp_path), "edits": {"a.float": 2.5}}]
    assert tab.form_rows["a.int"].widget.isEnabled() and tab.role_combos["foci"].isEnabled()   # rebuilt
    assert tab.pending() == {}

    tab.choose_channel("dna", "DAPI")                                   # the same for a role write
    _settle(tab, qtbot)
    fake.gate = threading.Event()
    assert tab.apply_roles()
    qtbot.waitUntil(lambda: len(writes()) == 2, timeout=30_000)
    combo = tab.role_combos["foci"]
    assert not combo.isEnabled() and not tab.apply_roles_btn.isEnabled()
    tab.choose_channel("foci", "DAPI")
    combo.setCurrentIndex(1)                                            # the combo's own signal: put back
    assert tab.role_choice("foci") is None and combo.currentData() is None
    tab.set_value("a.int", 5)
    assert tab.pending() == {} and "a write is in progress" in tab.status_label.text()
    fake.gate.set()
    fake.gate = None
    _settle(tab, qtbot)
    assert writes()[-1] == {"root": str(tmp_path), "roles": {"dna": "DAPI", "foci": ""}}
    assert tab.role_combos["foci"].isEnabled() and tab.form_rows["a.int"].widget.isEnabled()


def test_values_the_widgets_cannot_show_are_named(qtbot, tmp_path):
    """A hand edit can leave a value outside the declared range or choices: the row says what the project holds
    instead of showing the spin box's limit or another choice as if it were the value."""
    view = fake_view(tmp_path, tunables=[
        ({"key": "a.float", "type": "float", "min": 0, "max": 5}, 1.0, 7.5, "manual"),
        ({"key": "a.choice", "type": "choice", "choices": ["max", "mean"]}, "max", "median", "manual")])
    tab, fake, _ = _form_tab(qtbot, tmp_path, views=[view])
    f, c = tab.form_rows["a.float"], tab.form_rows["a.choice"]
    assert f.widget.value() == 5.0
    assert f.note_label.text() == "set by a hand edit of recipe.yaml; now 7.5, outside the declared range"
    assert c.widget.currentText() == "median (not a declared choice)" and c.widget.currentData() == "median"
    assert tab.pending() == {}
    c.widget.setCurrentIndex(1)                                         # a declared choice: a change
    assert tab.pending() == {"a.choice": "mean"}
    c.widget.setCurrentIndex(2)                                         # back to what the project holds
    assert tab.pending() == {}
    _settle(tab, qtbot)


def test_a_form_preview_shows_the_engines_warnings(qtbot, tmp_path):
    """The form's dry run can warn (PLAN 9.4 #6: a recipe file the write would refuse to replace): the warnings are
    shown under the new config_hash, in amber (review N8)."""
    text = f"{tmp_path / 'recipe.yaml'} is not recorded in the manifest, so runs do not use it"

    def answer(kw):
        if kw.get("edits"):
            return {**preview_record({}, config_hash="eeeeeeeeeeee", warnings=[text]), "preview": None}
        return preview_record({}, config_hash="ffffffffffff")

    view = fake_view(tmp_path, tunables=[({"key": "a.float", "type": "float"}, 1.0, 1.0, None)])
    tab, fake, _ = _form_tab(qtbot, tmp_path, answer, views=[view])
    tab.set_value("a.float", 2.0)
    _settle(tab, qtbot)
    assert tab.form_hash == "eeeeeeeeeeee"
    want = f"with these values: config_hash eeeeeeeeeeee (now {PACK_HASH})\nwarning: {text}"
    assert tab.form_hash_label.text() == want
    assert "rgb(225, 160, 40)" in tab.form_hash_label.styleSheet() and tab.apply_form_btn.isEnabled()


def test_a_fine_step_gets_the_decimals_it_needs(qtbot, tmp_path):
    """A float declared on a narrow range steps by a hundredth of it: the spin box shows the decimals its step
    needs, so a step up moves the value (review N5)."""
    view = fake_view(tmp_path, tunables=[({"key": "a.float", "type": "float", "min": 0, "max": 0.1}, 0.05, 0.05,
                                          None)])
    tab, fake, _ = _form_tab(qtbot, tmp_path, views=[view])
    box = tab.form_rows["a.float"].widget
    assert box.singleStep() == pytest.approx(0.001) and box.decimals() == 3
    box.stepUp()
    assert box.value() == pytest.approx(0.051) and tab.pending() == {"a.float": pytest.approx(0.051)}
    _settle(tab, qtbot)


def test_a_refused_form_preview_keeps_apply_off(qtbot, tmp_path):
    def answer(kw):
        if kw.get("edits"):
            return EngineRefused("`germquant.project edit` was refused by the engine (exit 2):\nREFUSED: a.float "
                                 "must be at most 5, got 6")
        return preview_record({}, config_hash="ffffffffffff")

    view = fake_view(tmp_path, tunables=[({"key": "a.float", "type": "float"}, 1.0, 1.0, None)])
    tab, fake, _ = _form_tab(qtbot, tmp_path, answer, views=[view])
    tab.set_value("a.float", 6.0)                                       # no range declared: the engine decides
    _settle(tab, qtbot)
    assert "must be at most 5" in tab.form_hash_label.text() and not tab.apply_form_btn.isEnabled()
    tab.set_value("a.float", 1.0)
    assert tab.pending() == {} and not tab.apply_form_btn.isEnabled()
    _settle(tab, qtbot)


# ---- the post-stages, against a fake engine that lists two and writes post_stages.yaml into the folder ----

BINDS = {"img1": {"dna": {"channel": "DAPI", "how": "name"}, "foci": {"channel": "foci", "how": "name"}}}


def _post_tab(qtbot, tmp_path, *, rows=None, post=None):
    """A tab on a fake engine that lists ``rows`` (post_rows() by default) and writes post_stages.yaml into
    ``tmp_path``; the view's list in force is read from that folder, as read_view reads it, unless ``post`` is
    given."""
    fake = FakePostRunner(lambda kw: preview_record(BINDS, config_hash=PACK_HASH, changed=False), rows=rows)

    def reader(root, reference=None):
        return fake_view(root, post=post if post is not None else post_list(root))

    return _tab(qtbot, tmp_path, session=FakeSession(fake), reader=reader), fake


def _writes(fake) -> list:
    return [c[1] for c in fake.post_calls if isinstance(c, tuple)]


def test_the_post_stages_are_listed_switched_and_written_through_the_engine(qtbot, tmp_path):
    """The post-stages the engine lists, each a switch with its declared parameters as typed widgets (units,
    ranges, labels, defaults); a switched-off post-stage's parameters wait. Apply writes, through the engine, each
    post-stage switched on with the parameters off their defaults (as recipe.yaml holds only the edits); the tab
    reads the file back, and config_hash, the role preview and the form are left as they were. Every post-stage
    off is an empty list (which also replaces a pack's list)."""
    tab, fake = _post_tab(qtbot, tmp_path)
    assert fake.post_calls == ["list"] and tab.post_error is None and tab.post_stage_names() == [FIXTURE, RING]
    fx, ring = tab.post_rows[FIXTURE], tab.post_rows[RING]
    assert fx.switch.text() == "fixture_nuclei 1.0.0 (germquant-fixture-post-stage)"
    assert fx.title_label.text() == "Fixture: per-nucleus voxels, volume, foci statistic and spot count"
    assert list(fx.params) == ["factor", "min_voxels", "statistic", "fail"] and list(ring.params) == ["radius_um",
                                                                                                       "note"]
    radius = ring.params["radius_um"]
    assert isinstance(radius.widget, QDoubleSpinBox) and radius.widget.suffix() == " um"
    assert (radius.widget.minimum(), radius.widget.maximum(), radius.widget.value()) == (0.1, 5.0, 1.0)
    assert radius.label.text() == "Shell width" and radius.pack_label.text() == "default 1.0"
    assert isinstance(ring.params["note"].widget, QLineEdit) and ring.params["note"].widget.text() == "none"
    assert ring.params["note"].label.text() == "note"                   # no label declared: the key
    assert not radius.widget.isEnabled() and not fx.params["factor"].widget.isEnabled()   # off: they wait
    assert tab.post_label.text().startswith("In force: none")
    assert tab.post_entries() == [] and not tab.post_changed and not tab.apply_post_btn.isEnabled()
    assert tab.post_hash_label.text() == ""

    tab.set_post_stage(RING, True)
    assert radius.widget.isEnabled() and tab.post_entries() == [{"name": RING, "params": {}}]
    assert tab.apply_post_btn.isEnabled() and tab.post_hash_label.text() == f"with these post-stages: {UNCHANGED}"
    tab.set_post_param(RING, "radius_um", 0.8)
    note = ring.params["note"].widget
    note.setText("x")
    note.editingFinished.emit()                                         # the widget's own change
    assert tab.post_entries() == [{"name": RING, "params": {"radius_um": 0.8, "note": "x"}}]
    assert ring.params["note"].note_label.text() == "changed: x at Apply"
    tab.reset_post_param(RING, "note")
    assert note.text() == "none" and tab.post_entries() == [{"name": RING, "params": {"radius_um": 0.8}}]
    with pytest.raises(ValueError, match="must be at most 5"):
        tab.set_post_param(RING, "radius_um", 7.0)
    with pytest.raises(ValueError, match="declares no parameter width"):
        tab.set_post_param(RING, "width", 1.0)
    with pytest.raises(ValueError, match="not offered by the engine environment"):
        tab.set_post_stage("absent_stage", True)
    _settle(tab, qtbot)
    n = len(fake.calls)
    written = []
    tab.post_written.connect(written.append)
    applied = []
    tab.applied.connect(applied.append)
    assert tab.apply_post_stages()
    _settle(tab, qtbot)
    path = tmp_path / "post_stages.yaml"
    assert _writes(fake) == [[{"name": RING, "params": {"radius_um": 0.8}}]]
    assert [Path(w["path"]) for w in written] == [path] and applied == []
    assert len(fake.calls) == n                         # no project edit, no dry run: the file is not hashed
    assert tab.config_hash == PACK_HASH and tab.post_switch(RING) and tab.post_value(RING, "radius_um") == 0.8
    assert not tab.post_changed and not tab.apply_post_btn.isEnabled()
    assert tab.post_rows[RING].params["radius_um"].note_label.text() == ""          # applied: no change left
    assert tab.status_label.text() == f"wrote {path}: ring_intensity (radius_um 0.8); {UNCHANGED}"
    assert tab.post_label.text().startswith("In force: the project's post_stages.yaml: ring_intensity (radius_um "
                                            "0.8)")
    tab.set_post_stage(RING, False)                                     # every post-stage off
    assert tab.post_entries() == [] and tab.post_changed
    assert tab.apply_post_stages()
    _settle(tab, qtbot)
    assert _writes(fake)[-1] == [] and yaml.safe_load(path.read_text(encoding="ascii")) == []
    assert tab.post_label.text().startswith("In force: the project's post_stages.yaml: every post-stage off")
    assert fake.post_calls.count("list") == 1           # the engine environment's list: a write does not ask again
    tab.reload()
    _settle(tab, qtbot)
    assert fake.post_calls.count("list") == 2           # an explicit reload does


def test_post_stage_edits_wait_for_a_run_and_a_write(qtbot, tmp_path):
    """No post-stage edit while a run of the project goes or a write is queued or running, as for the other recipe
    edits: the switches and widgets are off, every edit call is refused with the reason, and a widget's own change
    is put back. A post-stage write keeps the role choices and form values not applied yet."""
    tab, fake = _post_tab(qtbot, tmp_path)
    tab.set_running(True)
    fx = tab.post_rows[FIXTURE]
    assert not fx.switch.isEnabled() and not tab.apply_post_btn.isEnabled()
    tab.set_post_stage(FIXTURE, True)
    assert not tab.post_switch(FIXTURE) and tab.status_label.text() == RUNNING_TEXT
    fx.switch.setChecked(True)                                          # the switch's own signal: put back
    assert not fx.switch.isChecked() and not tab.post_switch(FIXTURE)
    assert not tab.apply_post_stages()
    tab.set_running(False)
    tab.set_post_stage(FIXTURE, True)
    tab.choose_channel("dna", "DAPI")                                   # chosen before the write, not applied
    _settle(tab, qtbot)
    fake.post_gate = threading.Event()                                  # hold the write
    assert tab.apply_post_stages()
    qtbot.waitUntil(lambda: len(_writes(fake)) == 1, timeout=30_000)
    assert not fx.switch.isEnabled() and not fx.params["factor"].widget.isEnabled()
    assert not tab.role_combos["dna"].isEnabled()
    tab.set_post_param(FIXTURE, "factor", 3.0)
    assert tab.post_value(FIXTURE, "factor") == 1.0 and tab.status_label.text() == WRITING_TEXT
    fx.params["factor"].widget.setValue(4.0)                            # the widget's own signal: put back
    assert fx.params["factor"].widget.value() == 1.0 and tab.post_value(FIXTURE, "factor") == 1.0
    fake.post_gate.set()
    fake.post_gate = None
    _settle(tab, qtbot)
    assert _writes(fake) == [[{"name": FIXTURE, "params": {}}]]
    assert tab.post_switch(FIXTURE) and tab.post_rows[FIXTURE].switch.isEnabled()
    assert tab.role_choice("dna") == "DAPI" and tab.role_combos["dna"].currentData() == "DAPI"


def test_a_list_in_force_the_engine_cannot_run_is_named(qtbot, tmp_path):
    """The project's post_stages.yaml, written by hand: a value outside its range, a float for an int, an undeclared
    parameter, a post-stage the engine environment does not offer, and one listed twice. Each is named, with what
    Apply writes instead (the engine refuses a run of the file as it is); the form shows the defaults, and Apply
    writes the list without what cannot run. A pack's own list is in force until the project has its own."""
    (tmp_path / "post_stages.yaml").write_bytes(b"- name: fixture_nuclei\n  params: {factor: 20.0, min_voxels: 3.0,"
                                                b" width: 2}\n- name: absent_stage\n- name: fixture_nuclei\n")
    tab, fake = _post_tab(qtbot, tmp_path)
    where = "the project's post_stages.yaml"
    assert tab.post_problems == [
        f"{where}: fixture_nuclei: parameter factor must be at most 10.0, got 20.0: the form shows the default 1.0, "
        f"which Apply writes",
        f"{where}: fixture_nuclei: parameter min_voxels must be an integer, got 3.0: the form shows the default 0, "
        f"which Apply writes",
        f"{where}: fixture_nuclei declares no parameter width: Apply leaves it out",
        f"{where} switches on absent_stage, which the engine environment does not offer: a run of the project is "
        f"refused while the list names it, and Apply writes the list without it",
        f"{where} lists fixture_nuclei more than once: Apply writes it once"]
    assert tab.post_problems_label.text() == "\n".join(f"warning: {p}" for p in tab.post_problems)
    assert tab.post_switch(FIXTURE) and tab.post_value(FIXTURE, "factor") == 1.0
    assert tab.post_entries() == [{"name": FIXTURE, "params": {}}] and tab.post_changed
    assert tab.apply_post_stages()
    _settle(tab, qtbot)
    assert _writes(fake) == [[{"name": FIXTURE, "params": {}}]] and tab.post_problems == []

    pack = tmp_path / "pack" / "post_stages"
    pack.mkdir(parents=True)
    (pack / "basic.yaml").write_bytes(b"- name: ring_intensity\n  params: {radius_um: 0.5}\n")
    shipped = post_list(tmp_path / "nothing_here", tmp_path / "pack", "basic")
    tab2, fake2 = _post_tab(qtbot, tmp_path, post=shipped)
    assert tab2.post_label.text().startswith("In force: the pack's post_stages/basic.yaml: ring_intensity "
                                             "(radius_um 0.5)")
    assert tab2.post_switch(RING) and tab2.post_value(RING, "radius_um") == 0.5 and not tab2.post_changed
    tab2.set_post_param(RING, "radius_um", 0.6)
    assert tab2.post_hash_label.text() == (f"with these post-stages: {UNCHANGED}. The project's own post_stages.yaml "
                                           f"then replaces the pack's post_stages/basic.yaml.")


def test_the_post_stages_are_those_of_the_environment_that_runs_the_recipe(qtbot, tmp_path):
    """Post-stages are installed where the project runs (D39; design R6: .venv_packs or a lab's own environment):
    the listing and the write go to the runner of the recipe's own interpreter when this computer stores one
    (session.runner_for, which the Project dock's runs take), while the role and form previews stay the engine's."""
    def answer(kw):
        return preview_record(BINDS, config_hash=PACK_HASH, changed=False)

    engine, recipe_env = FakePostRunner(answer, rows=[]), FakePostRunner(answer)
    asked = []
    session = FakeSession(engine)

    def runner_for(pack_id=None, recipe_id=None, should_stop=None):
        asked.append((pack_id, recipe_id))
        return recipe_env

    session.runner_for = runner_for
    pin = {"pack": {"id": "test_pack", "version": "0.1.0", "recipe": "basic"}}
    tab = _tab(qtbot, tmp_path, session=session,
               reader=lambda root, reference=None: fake_view(root, manifest=pin, post=post_list(root)))
    assert asked == [("test_pack", "basic")] and tab.post_stage_names() == [FIXTURE, RING]
    assert engine.post_calls == [] and recipe_env.post_calls == ["list"]
    tab.set_post_stage(RING, True)
    assert tab.apply_post_stages()
    _settle(tab, qtbot)
    assert _writes(recipe_env) == [[{"name": RING, "params": {}}]] and engine.post_calls == []
    assert asked == [("test_pack", "basic")] * 2
    assert len(engine.calls) == 1 and recipe_env.calls == []            # the role preview: the engine's


def test_without_post_stages_the_section_says_why_and_the_rest_works(qtbot, tmp_path):
    """An engine runner that cannot list post-stages, or an engine without the post_stages feature (its one
    message): the section says why and shows the list in force, read only; the role editor and the form work as
    before."""
    view = fake_view(tmp_path, tunables=[({"key": "a.float", "type": "float", "min": 0, "max": 9}, 1.0, 1.0, None)],
                     post=PostList("project", tmp_path / "post_stages.yaml",
                                   ({"name": FIXTURE, "params": {"factor": 2.5}},)))
    fake = FakeRunner(lambda kw: preview_record(BINDS, config_hash="eeeeeeeeeeee" if kw.get("edits") else PACK_HASH,
                                                changed=False))                           # no post-stage methods
    tab = _tab(qtbot, tmp_path, session=FakeSession(fake), reader=lambda root, reference=None: view)
    assert tab.post_error == "this engine runner cannot list post-stages" and tab.post_rows == {}
    assert tab.post_label.text() == ("In force: the project's post_stages.yaml: fixture_nuclei (factor 2.5).\n"
                                     "No post-stage list: this engine runner cannot list post-stages")
    assert not tab.apply_post_btn.isEnabled() and not tab.apply_post_stages()
    with pytest.raises(ValueError, match="no post-stage list"):
        tab.set_post_stage(FIXTURE, False)
    tab.set_value("a.float", 2.0)
    _settle(tab, qtbot)
    assert tab.form_hash == "eeeeeeeeeeee" and tab.apply_form_btn.isEnabled() and tab.role_combos["dna"].isEnabled()

    message = _engine.feature_missing_message("post_stages", info={"germquant": "0.2.0.dev0", "git_sha": "x" * 40},
                                              interpreter=sys.executable)
    fake2 = FakePostRunner(lambda kw: preview_record(BINDS, config_hash=PACK_HASH, changed=False))
    fake2.list_error = FeatureMissing(message, "post_stages")
    tab2 = _tab(qtbot, tmp_path, session=FakeSession(fake2), reader=lambda root, reference=None: view)
    assert tab2.post_error == message and "Post-stages need the tranche-5 Train B engine" in tab2.post_label.text()
    assert tab2.role_combos["dna"].isEnabled() and tab2.last_error is None and tab2.engine_error is None


# ---------------------------------------------------------------------------------------------------
# the rules (no Qt, no engine)
# ---------------------------------------------------------------------------------------------------


def _t(decl, pack=1.0, value=None, origin=None) -> Tunable:
    return Tunable.from_declaration(decl, pack_value=pack, value=pack if value is None else value, origin=origin)


def test_the_form_rules_follow_the_engines():
    f = _t({"key": "k", "type": "float", "min": 0.5, "max": 2})
    assert f.check(1) == 1.0 and isinstance(f.check(1), float)
    for bad, why in ((True, "takes a float"), ("1", "takes a float"), (0.25, "at least 0.5"), (3, "at most 2"),
                     (math.nan, "a finite number"), (math.inf, "a finite number")):
        with pytest.raises(ValueError, match=why):
            f.check(bad)
    i = _t({"key": "k", "type": "INT", "min": 1}, pack=2)
    assert i.check(3.0) == 3 and isinstance(i.check(3.0), int) and i.type == "int"
    with pytest.raises(ValueError, match="whole number"):
        i.check(2.5)
    b = _t({"key": "k", "type": "bool"}, pack=False)
    assert b.check(True) is True
    with pytest.raises(ValueError, match="true or false"):
        b.check(1)
    c = _t({"key": "k", "type": "choice", "choices": [1, "max", True]}, pack=1)
    assert c.check(1.0) == 1 and c.check("max") == "max" and c.check(True) is True
    with pytest.raises(ValueError, match="one of the choices"):
        c.check("min")
    assert c.same(1, 1.0) and not c.same(True, 1) and not c.same("1", 1)
    s = _t({"key": "k", "type": "string"}, pack="a")
    assert s.check("b") == "b" and s.type == "str"
    with pytest.raises(ValueError, match="a word"):
        s.check(3)
    u = _t({"key": "k", "type": "vector"})
    assert not u.editable
    with pytest.raises(ValueError, match="shown read only"):
        u.check(1)
    assert _t({"key": "k", "type": "float"}, pack=MISSING).pack_text() == "pack value unknown"
    assert _t({"key": "k", "label": "Lbl", "type": "float", "unit": "um3"}).title == "Lbl"


def test_the_hash_line_names_the_stage_switches_off(tmp_path):
    """The hash line says what the config_hash is the hash of: the pack recipe or an edited one, and the stage
    switches the project turns off, which enter the hash too (review N7)."""
    v = fake_view(tmp_path)
    assert hash_note(v) == "the pack recipe"
    v.overrides = {"spots.enabled": False}
    assert hash_note(v) == "the pack recipe, with the stage switch spots off"
    v.edited, v.overrides = True, {"spots.enabled": False, "coloc.enabled": False, "x.y": 3}
    assert hash_note(v) == "edited, with the stage switches coloc, spots off, with the overrides x.y=3"
    v.overrides = {}
    assert hash_note(v) == "edited"


def test_recipe_edits_leave_out_the_role_editors_own_map(tmp_path):
    """recipe.yaml's keys are the project's edits, the extends line left out, and io.channel_map left out when it
    names the role editor's own channel_map.yaml (the manifest records role choices), as the engine's read_edits
    does; a map the roles did not write is a hand edit, shown as one."""
    (tmp_path / "recipe.yaml").write_bytes(b"extends: pack:test_pack@0.1.0/basic\nio:\n  channel_map: "
                                           b"channel_map.yaml\nspots:\n  log:\n    min_snr: 3.5\n")
    rec = {"file": "recipe.yaml", "roles": {"dna": "H2B"}}
    assert recipe_edits(tmp_path, {"recipe": rec}) == ({"spots.log.min_snr": 3.5}, None)
    assert recipe_edits(tmp_path, {"recipe": {**rec, "roles": {}}}) == (
        {"io.channel_map": "channel_map.yaml", "spots.log.min_snr": 3.5}, None)
    (tmp_path / "recipe.yaml").write_bytes(b"extends: pack:test_pack@0.1.0/basic\nio:\n  channel_map: mine.yaml\n")
    assert recipe_edits(tmp_path, {"recipe": rec}) == ({"io.channel_map": "mine.yaml"}, None)
    assert recipe_edits(tmp_path, {"recipe": {"file": None}}) == ({}, None)
    edits, problem = recipe_edits(tmp_path, {"recipe": {"file": "gone.yaml"}})
    assert edits == {} and "the manifest names it but it is missing" in problem


def test_form_change_and_the_badge():
    rows = [_t({"key": "a", "type": "float"}, 1.0, 2.0, "form"), _t({"key": "b", "type": "int"}, 1, 1)]
    assert form_change(rows, {"a": RESET, "b": 3}) == ({"b": 3}, ["a"])
    assert form_change(rows, {"b": RESET}) == ({}, [])                  # b is not edited: nothing to unset
    with pytest.raises(ValueError, match="not a tunable"):
        form_change(rows, {"c": 1})
    assert validated_badge({"value": True, "why": "validated: record ok", "record": "packs/p/1/v.json"}) == \
        ("validated (record packs/p/1/v.json)", "ok")
    assert validated_badge({"value": False, "why": "edited: recipe.yaml changes the pack recipe"}) == \
        ("validated: false (edited)", "warn")
    assert validated_badge({"value": False, "why": "not validated: the pack recipe x@1/r is provisional"}) == \
        ("not validated (provisional pack)", "warn")
    assert validated_badge({"value": False, "why": "not validated: the record fails its check"})[0] == \
        "not validated (the record fails its check)"
    assert validated_badge(None) == ("validated: unknown", "warn")


def test_the_post_stage_rules(tmp_path):
    """post_stages.yaml as the engine reads it (UTF-8 with or without a byte-order mark, YAML, an empty file an empty
    list, a list of {name, params}); the list in force (the project's own, else the pack's for the recipe, else
    none); what Apply writes (the listing's order, only the parameters off their defaults); and when that is a
    change (by the values a run gets, not by the order or a default written out)."""
    p = tmp_path / "a.yaml"
    p.write_bytes(b"\xef\xbb\xbf- name: ring_intensity\n  params: {radius_um: 1}\n")
    assert read_post_entries(p) == ([{"name": RING, "params": {"radius_um": 1}}], None)
    for text in (b"", b"[]\n", b"# a comment only\n"):
        p.write_bytes(text)
        assert read_post_entries(p) == ([], None)
    for text, why in ((b"- name: [\n", "cannot be read as YAML (line "), (b"name: x\n", "must be a list"),
                      (b"- params: {}\n", "entry 0 has no name"), (b"- {name: x, other: 1}\n", "unknown key"),
                      (b"- {name: x, params: [1]}\n", "params must be a mapping"), (b"\xff\xfe", "not UTF-8 text")):
        p.write_bytes(text)
        entries, problem = read_post_entries(p)
        assert entries is None and why in problem, (text, problem)

    pack, proj = tmp_path / "pack", tmp_path / "proj"
    (pack / "post_stages").mkdir(parents=True)
    proj.mkdir()
    (pack / "post_stages" / "basic.yaml").write_bytes(b"- name: ring_intensity\n")
    assert post_list(proj, pack, "other") == PostList()
    shipped = post_list(proj, pack, "basic")
    assert (shipped.origin, shipped.path, shipped.entries) == ("pack", pack / "post_stages" / "basic.yaml",
                                                               ({"name": RING, "params": {}},))
    (proj / "post_stages.yaml").write_bytes(b"[]\n")
    own = post_list(proj, pack, "basic")
    assert (own.origin, own.path, own.entries, own.problem) == ("project", proj / "post_stages.yaml", (), None)
    assert own.summary() == "the project's post_stages.yaml: every post-stage off"

    stages = post_stages_from_listing(post_rows())
    assert [s.name for s in stages] == [FIXTURE, RING] and stages[0].defaults == {"factor": 1.0, "min_voxels": 0,
                                                                                    "statistic": "mean", "fail": False}
    want = post_entries(stages, {RING: True, FIXTURE: True}, {FIXTURE: {"factor": 1.0, "statistic": "max"},
                                                             RING: {"radius_um": 0.8}})
    assert want == [{"name": FIXTURE, "params": {"statistic": "max"}}, {"name": RING, "params": {"radius_um": 0.8}}]
    assert entries_text(want) == "fixture_nuclei (statistic max); ring_intensity (radius_um 0.8)"
    assert entries_text([{"name": RING, "params": {}}]) == "ring_intensity (its defaults)" and \
        entries_text([]) == "every post-stage off"
    same = PostList("project", proj / "post_stages.yaml", ({"name": RING, "params": {"radius_um": 0.8, "note": "none"}},
                                                           {"name": FIXTURE, "params": {"statistic": "max", "factor": 1}}))
    assert post_state(stages, same)[1] == [] and not post_list_changed(stages, same, want)
    assert post_list_changed(stages, same, want[:1]) and post_list_changed(stages, PostList(), want)
    assert not post_list_changed(stages, PostList(), []) and not post_list_changed(stages, own, [])
    broken = PostList("project", proj / "post_stages.yaml", (), "it cannot be read as YAML (line 1)")
    assert post_list_changed(stages, broken, [])                        # Apply writes a list a run can read
    state, problems = post_state(stages, broken)
    assert not state[FIXTURE]["on"] and problems == [
        "the project's post_stages.yaml cannot be read (it cannot be read as YAML (line 1)): a run of the project is "
        "refused while it is so, and Apply writes a new post_stages.yaml"]
