"""The parameter form of the Recipe tab (tranche 5 Train B, unit P6-RECIPE; design 7.3 and 8.5, plugin_specs 4.3,
P6 acceptance 5): built only from the pack recipe's tunables, one widget per declared type with its unit and
range, the project's value next to the pack's, Reset per row; keys set outside the form (tuning, Use in project,
a hand edit) are shown read only; Apply writes through `project edit` (the engine runner, D39), which validates
by origin, and every Apply shows the new config_hash; with every edit reset the project hashes as the pack
recipe again (I7).

The acceptance tests run the engine's project commands, so they need the projects feature (need_features); the
widget tests and the rule tests further down run against any engine."""
from __future__ import annotations

import math

import pytest
import yaml
from _p6_recipe import FakeRunner, FakeSession, engine_edit, fake_view, preview_record, recipe_files
from conftest import need_features
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from napari_germquant import _settings
from napari_germquant._params import MISSING, RESET, Tunable, form_change, validated_badge
from napari_germquant._recipe_widget import RecipeTab
from napari_germquant._runner import EngineRefused, InProcessRunner

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
    tab.set_value(MIN_SNR, 4.0)
    assert tab.apply_form()
    _settle(tab, qtbot)
    assert all(v is not None for v in recipe_files(root).values()) and tab.config_hash != PACK_HASH
    assert tab.revert_all(confirm=False)
    _settle(tab, qtbot)
    assert tab.last_error is None, tab.status_label.text()
    assert recipe_files(root) == {"recipe.yaml": None, "channel_map.yaml": None} and tab.config_hash == PACK_HASH
    assert tab.role_choice("dna") is None and tab.form_rows[MIN_SNR].widget.value() == 2.0


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
