"""The Recipe tab as a part of the Project dock (tranche 5 Train B, unit P6-RECIPE): the host's one line
(``RecipeTab.for_dock``) and the signals it listens to, the order of its engine calls when the project shown
changes under them, and the read-only states (no engine chosen, an engine without projects, this environment's
germquant without projects). A fake engine and a hand-made view: these run against any engine."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

from _p6_recipe import FakeRunner, FakeSession, fake_view, preview_record
from qtpy.QtCore import QObject, Signal

from napari_germquant import _engine, _params
from napari_germquant._engine_session import NOT_CHOSEN, EngineSession
from napari_germquant._recipe_widget import RecipeTab
from napari_germquant._runner import FeatureMissing


def _settle(tab, qtbot, timeout: int = 60_000) -> None:
    qtbot.waitUntil(lambda: tab.idle, timeout=timeout)


def _bindings(image="img1") -> dict:
    return {image: {"dna": {"channel": "DAPI", "how": "name"}, "foci": {"channel": "foci", "how": "name"}}}


class _Host(QObject):
    """A Project dock as far as the tab sees it."""

    project_changed = Signal(object)
    image_selected = Signal(str)
    run_state_changed = Signal(bool)

    def __init__(self, session):
        super().__init__()
        self.viewer = None
        self.session = session
        self.records: list[dict] = []

    def recipe_applied(self, record: dict) -> None:
        self.records.append(record)


def test_the_project_dock_adds_the_tab_with_one_line(qtbot, tmp_path):
    def answer(kw):
        if kw.get("dry_run"):
            return preview_record(_bindings(), config_hash="bbbbbbbbbbbb" if any((kw.get("roles") or {}).values())
                                  else "02bb83cf74f9", changed=any((kw.get("roles") or {}).values()))
        return {**preview_record(_bindings(), config_hash="bbbbbbbbbbbb"), "dry_run": False,
                "written": ["channel_map.yaml", "recipe.yaml", "germquant_project.yaml"]}

    fake = FakeRunner(answer)
    host = _Host(FakeSession(fake))
    images = [{"id": i, "path": f"images/{i}.ome.tif", "excluded": False} for i in ("img1", "img2")]
    channels = {"img1": ["DAPI", "foci"], "img2": ["Hoechst", "spots"]}

    def reader(root, reference=None):
        ref = reference or "img1"
        return fake_view(root, images=images, reference=ref, channels=channels[ref])

    tab = RecipeTab.for_dock(host)                     # the one line (the reader is the tests' own)
    tab._reader = reader
    qtbot.addWidget(tab)
    assert tab.session is host.session
    other = _Host(host.session)
    assert tab.connect_host(other) == ["project_changed", "image_selected", "run_state_changed", "recipe_applied"]
    host.project_changed.emit(str(tmp_path))
    _settle(tab, qtbot)
    assert tab.root == Path(tmp_path) and tab.view.reference == "img1"
    assert [tab.reference_combo.itemText(i) for i in range(tab.reference_combo.count())] == ["img1", "img2"]
    items = [tab.role_combos["dna"].itemText(i) for i in range(tab.role_combos["dna"].count())]
    assert items[1:] == ["0: DAPI", "1: foci"]
    host.image_selected.emit("img2")                   # the Images tab's selection is the reference image
    _settle(tab, qtbot)
    assert tab.view.reference == "img2" and tab.reference_combo.currentText() == "img2"
    assert [tab.role_combos["dna"].itemText(i) for i in range(1, 3)] == ["0: Hoechst", "1: spots"]
    host.run_state_changed.emit(True)
    assert tab.running and not tab.role_combos["dna"].isEnabled()
    host.run_state_changed.emit(False)
    tab.choose_channel("dna", "Hoechst")
    _settle(tab, qtbot)
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert [r["config_hash"] for r in host.records] == ["bbbbbbbbbbbb"] == [r["config_hash"] for r in other.records]
    host.project_changed.emit(None)
    assert tab.root is None and tab.view is None and tab.header_label.text() == "No project."


def test_answers_for_a_project_no_longer_shown_are_dropped_and_a_write_keeps_its_folder(qtbot, tmp_path):
    """The engine answers late: a preview made for project A never lands in project B, and a write queued for A
    is made on A (and announced) even when B is shown by the time it runs."""
    a, b = tmp_path / "a", tmp_path / "b"
    fake = FakeRunner(lambda kw: preview_record(_bindings(), config_hash="aaaaaaaaaaaa" if kw.get("dry_run")
                                                else "cccccccccccc", changed=True))
    names = {str(a): "project a", str(b): "project b"}

    def reader(root, reference=None):
        v = fake_view(root, digest=str(root))
        v.name = names[str(Path(root))]
        return v

    tab = RecipeTab(None, session=FakeSession(fake), reader=reader)
    qtbot.addWidget(tab)
    tab.set_project(a)
    _settle(tab, qtbot)
    assert tab.view.name == "project a"
    tab.choose_channel("dna", "DAPI")
    _settle(tab, qtbot)
    fake.gate = threading.Event()                       # hold the next engine call
    assert tab.apply_roles()
    qtbot.waitUntil(lambda: len([c for c in fake.calls if not c.get("dry_run")]) == 1, timeout=30_000)
    applied = []
    tab.applied.connect(applied.append)
    tab.set_project(b)                                  # B is shown while A's write is in flight
    fake.gate.set()
    fake.gate = None
    _settle(tab, qtbot)
    writes = [c for c in fake.calls if not c.get("dry_run")]
    assert [w["root"] for w in writes] == [str(a)]
    assert [r["config_hash"] for r in applied] == ["cccccccccccc"]
    assert tab.root == b and tab.view.name == "project b" and tab.role_choice("dna") is None
    assert all(c["root"] == str(b) for c in fake.calls[-1:])


def test_without_an_engine_the_tab_shows_the_recipe_read_only(qtbot, tmp_path):
    """No engine chosen on this computer: the project's recipe is shown, every edit is off, and the status names
    what to do. An engine without the projects feature gives its one message the same way."""
    view = fake_view(tmp_path, tunables=[({"key": "a.float", "type": "float", "min": 0, "max": 9}, 1.0, 1.0, None)])
    tab = RecipeTab(None, session=EngineSession(), reader=lambda root, reference=None: view)
    qtbot.addWidget(tab)
    tab.set_project(tmp_path)
    _settle(tab, qtbot)
    assert tab.engine_error == NOT_CHOSEN and NOT_CHOSEN in tab.status_label.text()
    assert list(tab.form_rows) == ["a.float"] and not tab.form_rows["a.float"].widget.isEnabled()
    assert not tab.role_combos["dna"].isEnabled() and not tab.apply_roles_btn.isEnabled()
    assert tab.roles_hash_label.text() == "no role preview: there is no engine to ask"
    tab.set_value("a.float", 2.0)
    assert tab.pending() == {} and "read only" in tab.status_label.text()
    assert not tab.apply_form() and not tab.revert_all(confirm=False)

    message = _engine.feature_missing_message("projects", info={"germquant": "0.2.0.dev0", "git_sha": "x" * 40},
                                              interpreter=sys.executable)
    tab2 = RecipeTab(None, session=FakeSession(error=FeatureMissing(message, "projects")),
                     reader=lambda root, reference=None: view)
    qtbot.addWidget(tab2)
    tab2.set_project(tmp_path)
    _settle(tab2, qtbot)
    assert tab2.engine_error == message and "Projects need the tranche-5 Train B engine" in tab2.status_label.text()
    assert not tab2.form_rows["a.float"].widget.isEnabled()


def test_a_germquant_without_projects_here_gets_one_message(qtbot, tmp_path, monkeypatch):
    """The tab reads the project in this napari environment: a germquant there that predates projects gives the
    one message, which says so."""
    monkeypatch.setitem(sys.modules, "germquant.project", None)       # import germquant.project fails
    try:
        _params.read_view(tmp_path)
    except _params.ProjectsMissing as e:
        text = str(e)
    else:
        raise AssertionError("read_view did not refuse")
    assert text.startswith("The Recipe tab reads the project in this napari environment.")
    assert "Projects need the tranche-5 Train B engine" in text
    tab = RecipeTab(None, session=FakeSession(FakeRunner(lambda kw: preview_record({}))))
    qtbot.addWidget(tab)
    tab.set_project(tmp_path)
    _settle(tab, qtbot)
    assert tab.view is None and tab.last_error == text and text in tab.status_label.text()
    assert not tab.apply_form_btn.isEnabled() and not tab.revert_all_btn.isEnabled()
