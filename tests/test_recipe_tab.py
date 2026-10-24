"""The Recipe tab as a part of the Project dock (tranche 5 Train B, unit P6-RECIPE): the reference image, the
dock's tab protocol (RecipeProjectTab, the class the dock's EXTRA_TABS line names), the order of its engine calls
when the project shown changes under them, and the read-only states (no engine chosen, an engine without
projects, this environment's germquant without projects). A fake engine and a hand-made view: these run against
any engine."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

from types import SimpleNamespace

from _p6_recipe import FakeRunner, FakeSession, fake_view, preview_record
from qtpy.QtWidgets import QWidget

from napari_germquant import _engine, _params
from napari_germquant._engine_session import NOT_CHOSEN, EngineSession
from napari_germquant._recipe_widget import RecipeProjectTab, RecipeTab
from napari_germquant._runner import FeatureMissing


def _settle(tab, qtbot, timeout: int = 60_000) -> None:
    qtbot.waitUntil(lambda: tab.idle, timeout=timeout)


def _bindings(image="img1") -> dict:
    return {image: {"dna": {"channel": "DAPI", "how": "name"}, "foci": {"channel": "foci", "how": "name"}}}


def _items(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def test_the_reference_image_fills_the_role_lists(qtbot, tmp_path):
    """The role combos list the channels of the reference image: the project's first image, or the one chosen in
    the tab's own list or by a host (set_reference_image). A choice made on one image stays chosen on another,
    named as no channel of it, and Apply writes it."""
    def answer(kw):
        changed = any((kw.get("roles") or {}).values())
        if kw.get("dry_run"):
            return preview_record(_bindings(), config_hash="bbbbbbbbbbbb" if changed else "02bb83cf74f9",
                                  changed=changed)
        return {**preview_record(_bindings(), config_hash="bbbbbbbbbbbb"), "dry_run": False,
                "written": ["channel_map.yaml", "recipe.yaml", "germquant_project.yaml"]}

    fake = FakeRunner(answer)
    images = [{"id": i, "path": f"images/{i}.ome.tif", "excluded": False} for i in ("img1", "img2")]
    channels = {"img1": ["DAPI", "foci"], "img2": ["Hoechst", "spots"]}

    def reader(root, reference=None):
        ref = reference or "img1"
        return fake_view(root, images=images, reference=ref, channels=channels[ref])

    tab = RecipeTab(None, session=FakeSession(fake), reader=reader)
    qtbot.addWidget(tab)
    tab.set_project(tmp_path)
    _settle(tab, qtbot)
    assert tab.root == Path(tmp_path) and tab.view.reference == "img1"
    assert _items(tab.reference_combo) == ["img1", "img2"]
    assert _items(tab.role_combos["dna"])[1:] == ["0: DAPI", "1: foci"]
    tab.set_reference_image("img2")                    # a host's call
    _settle(tab, qtbot)
    assert tab.view.reference == "img2" and tab.reference_combo.currentText() == "img2"
    assert _items(tab.role_combos["dna"])[1:] == ["0: Hoechst", "1: spots"]
    tab.choose_channel("dna", "Hoechst")
    _settle(tab, qtbot)
    tab.reference_combo.setCurrentIndex(0)             # the tab's own list
    _settle(tab, qtbot)
    assert tab.view.reference == "img1" and tab.role_choice("dna") == "Hoechst"
    assert _items(tab.role_combos["dna"])[1:] == ["0: DAPI", "1: foci", "Hoechst (not a channel of img1)"]
    assert tab.role_combos["dna"].currentData() == "Hoechst"
    applied = []
    tab.applied.connect(applied.append)
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert [r["config_hash"] for r in applied] == ["bbbbbbbbbbbb"]
    assert [c for c in fake.calls if not c.get("dry_run")] == [{"root": str(tmp_path),
                                                                "roles": {"dna": "Hoechst", "foci": ""}}]
    tab.set_project(None)
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


class _Dock(QWidget):
    """The Project dock as its tab protocol shows a tab (unit P6-PROJECT's EXTRA_TABS: the class is made with the
    dock, which tells it of every reading of the project and of runs, and reads the project again on reload)."""

    def __init__(self, session):
        super().__init__()
        self.viewer = None
        self.session = session
        self.reloads = 0

    def reload(self) -> None:
        self.reloads += 1


def test_the_tab_follows_the_project_docks_tab_protocol(qtbot, tmp_path):
    """RecipeProjectTab, the class the Project dock's EXTRA_TABS line names: it follows the dock's readings of the
    project (a reading with no write since its own is not read again), its runs, and after a write it asks the
    dock to read the project again. While it is hidden it reads the project, but its role preview (a dry run that
    reads every image's header) waits until it is shown (review N12)."""
    def answer(kw):
        changed = any((kw.get("roles") or {}).values())
        if kw.get("dry_run"):
            return preview_record(_bindings(), config_hash="bbbbbbbbbbbb" if changed else "02bb83cf74f9",
                                  changed=changed)
        return {**preview_record(_bindings(), config_hash="bbbbbbbbbbbb"), "dry_run": False,
                "written": ["channel_map.yaml", "recipe.yaml", "germquant_project.yaml"]}

    fake = FakeRunner(answer)
    dock = _Dock(FakeSession(fake))
    qtbot.addWidget(dock)
    views = [fake_view(tmp_path, saved={"at": "t0"})]
    reads = []

    def reader(root, reference=None):
        reads.append(str(root))
        return views[-1]

    tab = RecipeProjectTab(dock)
    tab._reader = reader
    assert tab.session is dock.session and tab.dock is dock
    tab.project_changed(SimpleNamespace(root=Path(tmp_path), manifest={"saved": {"at": "t0"}}))
    _settle(tab, qtbot)
    assert tab.root == Path(tmp_path) and tab.config_hash == "02bb83cf74f9" and len(reads) == 1
    assert fake.calls == [] and tab.preview_states() == {}            # hidden: no role preview yet
    dock.show()                                                       # the dock shows the tab
    _settle(tab, qtbot)
    assert len(fake.calls) == 1 and tab.preview_states() == {"img1": {"dna": "name", "foci": "name"}}
    tab.project_changed(SimpleNamespace(root=Path(tmp_path), manifest={"saved": {"at": "t0"}}))
    _settle(tab, qtbot)
    assert len(fake.calls) == 1                         # read again with no write since: nothing to do
    tab.running_changed(True)
    assert tab.running and not tab.role_combos["dna"].isEnabled()
    tab.running_changed(False)
    tab.choose_channel("dna", "DAPI")
    _settle(tab, qtbot)
    views.append(fake_view(tmp_path, saved={"at": "t1"}, digest="d1", config_hash="bbbbbbbbbbbb",
                           roles=[r.with_edit("DAPI" if r.role == "dna" else None) for r in views[0].roles]))
    assert tab.apply_roles()
    _settle(tab, qtbot)
    assert dock.reloads == 1 and tab.config_hash == "bbbbbbbbbbbb"
    n = len(fake.calls)
    tab.project_changed(SimpleNamespace(root=Path(tmp_path), manifest={"saved": {"at": "t1"}}))   # the dock's
    _settle(tab, qtbot)
    assert len(fake.calls) == n                         # what the tab wrote and read already
    views.append(fake_view(tmp_path, saved={"at": "t2"}, digest="d2"))
    tab.project_changed(SimpleNamespace(root=Path(tmp_path), manifest={"saved": {"at": "t2"}}))   # a write elsewhere
    _settle(tab, qtbot)
    assert len(fake.calls) == n + 1 and tab.view.saved == {"at": "t2"}
    # a write elsewhere in the same second (the saved stamp is to the second): its manifest differs all the same
    tuned = {"saved": {"at": "t2"}, "recipe": {"origins": {"spots.log.min_snr": "tuning"}}}
    views.append(fake_view(tmp_path, saved={"at": "t2"}, digest="d3", manifest=tuned))
    tab.project_changed(SimpleNamespace(root=Path(tmp_path), manifest=dict(tuned)))
    _settle(tab, qtbot)
    assert len(fake.calls) == n + 2 and tab.view.manifest == tuned
    dock.hide()                                                       # another tab of the dock, say
    views.append(fake_view(tmp_path, saved={"at": "t3"}, digest="d4"))
    k = len(reads)
    tab.project_changed(SimpleNamespace(root=Path(tmp_path), manifest={"saved": {"at": "t3"}}))
    _settle(tab, qtbot)
    assert len(reads) == k + 1 and tab.view.saved == {"at": "t3"} and len(fake.calls) == n + 2
    dock.show()
    _settle(tab, qtbot)
    assert len(fake.calls) == n + 3 and tab.preview_states() == {"img1": {"dna": "name", "foci": "name"}}
    tab.project_changed(None)
    assert tab.root is None and tab.view is None
