"""The Project dock (docs/DESIGN_system_packs.md, section 8.4; the tranche-5 Train B plan, unit P6-PROJECT;
plugin_specs 4.1): one study as a project folder, made, edited and run through the engine.

What the dock shows:

* the header: the project's name and folder; the engine line (where it runs, from the engine session shared by
  the viewer's project docks); the pack id@version and recipe; the config_hash; the validation badge
  ("validated: false (edited)", "not validated (provisional pack)", "validated (record ...)"); the warnings of the
  engine's status (a hash that differs from the one the project was saved with, the project's copy of its pack
  missing, a model digest mismatch, an environment mix), what the project records differently from this
  computer, a held lock (with Remove stale lock for one older than 12 h); and New, Open (with the recent projects
  of this computer), Save and Close;
* the Images tab: id, path, size, the header summary, exclusion; Add files, Add folder and a pasted path (a UNC
  path kept as it is), each through ``project add``, its refusals named per path; an image that cannot be reached
  is amber and never blocks the dock;
* the Samples tab: one row per image with the design columns and any extra column, prefilled from the file name
  where the recipe's rule matches; Save writes samples.csv with the engine's own writer and records it
  (``project edit --samples``); images whose row changed then show stale (sample row);
* the Run tab: the stage switches (untick = ``--no-<stage>``, greyed as the Run dock greys them), the xy stride,
  the device (the per-user choice of this computer), the recipe's own interpreter (P4's dialog), Run selected,
  Run all, Rerun selected (without resume), Cancel, Collect tables and Export commands, and the project's runs;
* the status board: one row per image, the engine's status (done; stale with its reason; failed with the error;
  missing), qc_pass, the objects (nuclei or cells, by the pack's object_kind) and spots of its results, the last
  run and the device used. While a run goes the rows follow the run's own log (``project run`` tailed by the
  runner); when it ends the engine's status is asked again. Double-click opens the image's results as layers
  with its source image.

Where things happen (the train's decision D39): every write that can change the config_hash or the manifest goes
through the engine runner, with this plugin's version (``--plugin-version``) on every one: ``project new`` (then
``project edit`` recording this computer's device, runner and interpreter, D42), ``add``, ``edit`` (Save, the
stage switches, the run settings), ``unlock``. The status, runs, collect and export-commands are the engine's
too. The dock reads the manifest, the image headers, the run logs, the results and the sample sheet in process.
The GUI thread never waits: engine calls run one at a time in a thread worker (``engine_worker``), a project run
in its own (``run_worker``), header reads in a third (``header_worker``). Every recipe edit and sample edit is off
while a run of the project goes (``_project.run_in_progress`` tells the viewer's other docks too).

Other docks and tabs: ``project_hub(viewer)`` says which project the viewer's Project dock has open and whether it
runs (Training's project mode, Spot tuning). A tab of this dock is a ProjectTab subclass named in EXTRA_TABS
(unit P6-RECIPE registers its Recipe tab there with one line); it reads ``dock.state``, runs engine calls with
``dock.engine_call`` or ``dock.edit`` and is told of every change (``project_changed``) and run
(``running_changed``).
"""
from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import sys
import threading
import time
import weakref
from pathlib import Path

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import _packs, _settings
from . import _project as P
from ._engine_session import NOT_CHOSEN, EngineNotChosen, EngineSession, session_for
from ._engine_widgets import DEVICE_TEXT, InterpreterDialog, PackPicker
from ._qt_common import WorkerRunner, add_layer_tuples, format_exception, set_viewer_status
from ._runner import (
    DEVICES,
    FeatureMissing,
    InProcessRunner,
    RunCancelled,
    RunnerError,
    SubprocessRunner,
    detect_interpreters,
    new_run_id,
    plugin_version,
    run_outcome,
)
from ._settings import INPROCESS

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)
LEVELS = {"ok": GREEN, "warn": AMBER, "edited": AMBER, "bad": RED, None: None}

# The dock's tabs after Images, Samples and Run: (title, "module:Class"), each class a ProjectTab subclass made with
# the dock. Unit P6-RECIPE registers its Recipe tab (role editor and parameter form) here with one line.
EXTRA_TABS: list[tuple[str, str]] = [
]

BOARD_COLUMNS = ("Image", "Status", "qc_pass", "nuclei", "spots", "Last run", "Device")
IMAGE_COLUMNS = ("Image id", "Path", "Size", "Header", "Excluded")
STATUS_COLORS = {"done": GREEN, "stale": AMBER, "failed": RED, "unreachable": AMBER, "missing": None,
                 "waiting": None, "unknown": None, "excluded": None}

engine_log = logging.getLogger("napari_germquant.engine")


def css(color: QColor | None) -> str:
    return "" if color is None else f"color: rgb({color.red()}, {color.green()}, {color.blue()});"


def _say(line: str) -> None:
    """An engine line, to the status bar (the worker's LogBridge listens to napari_germquant)."""
    engine_log.info(line)


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, (RunnerError, P.ProjectFileError, ValueError, KeyError)):
        return str(exc)
    return format_exception(exc, 1200)


def _size_text(size) -> str:
    try:
        n = float(size)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


# ---------------------------------------------------------------------------------------------------
# what the dock knows of the open project, and who else is told
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class ProjectState:
    """The open project, as the dock last read it: the folder, the manifest (in process), the engine's status
    record (None until the engine answered, or when it cannot), the board rows, the image headers read so far
    ({image_id: header or {"error"}}), the pack's object kind, the recipe's Config without the stage switches
    (None when this germquant cannot load it: ``config_error`` says why), the config_hash this germquant computes
    (``own_hash``), the sample table, and why the status could not be had (``status_error``)."""

    root: Path
    manifest: dict
    status: dict | None = None
    rows: list = dataclasses.field(default_factory=list)
    headers: dict = dataclasses.field(default_factory=dict)
    kind: str = "nucleus"
    config: object | None = None
    config_error: str | None = None
    own_hash: str | None = None
    samples: P.SampleTable | None = None
    samples_error: str | None = None
    status_error: BaseException | None = None

    @property
    def pack_id(self) -> str | None:
        return (self.manifest.get("pack") or {}).get("id")

    @property
    def pack_ref(self) -> str:
        pack = self.manifest.get("pack") or {}
        return f"{pack.get('id')}@{pack.get('version')}"

    @property
    def recipe_id(self) -> str | None:
        return (self.manifest.get("pack") or {}).get("recipe")

    @property
    def config_hash(self) -> str | None:
        if self.status is not None and self.status.get("config_hash"):
            return str(self.status["config_hash"])
        return (self.manifest.get("recipe") or {}).get("expected_config_hash")


class ProjectHub(QObject):
    """Which project the viewer's Project dock has open (``root``, ``state``) and whether a run of it goes
    (``running``). ``opened`` (the folder) and ``closed`` fire as the dock opens and closes projects, ``changed`` at
    every new reading of it, ``running_changed`` when a run starts or ends. Another dock that wrote the project
    (Training's Use in project, Spot tuning's Apply) calls ``request_reload()``: the Project dock reads it again."""

    opened = Signal(str)
    closed = Signal()
    changed = Signal()
    running_changed = Signal(bool)
    reload_requested = Signal()

    def request_reload(self) -> None:
        self._emit(self.reload_requested)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root: Path | None = None
        self.state: ProjectState | None = None
        self.running = False

    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:                  # the hub's QObject is gone (napari closing)
            pass


_HUBS: dict[int, ProjectHub] = {}
_HUBS_LOCK = threading.Lock()


def project_hub(viewer) -> ProjectHub:
    """The one hub of ``viewer`` (napari's public proxy counts as the viewer); a new, unshared hub for None."""
    if viewer is None:
        return ProjectHub()
    target = getattr(viewer, "__wrapped__", viewer)
    key = id(target)
    with _HUBS_LOCK:
        hub = _HUBS.get(key)
        if hub is None:
            hub = ProjectHub()
            _HUBS[key] = hub
            try:
                weakref.finalize(target, _drop_hub, key, hub)
            except TypeError:
                pass
    return hub


def _drop_hub(key: int, hub: ProjectHub) -> None:
    with _HUBS_LOCK:
        if _HUBS.get(key) is hub:
            del _HUBS[key]


class ProjectTab(QWidget):
    """A tab of the Project dock (EXTRA_TABS). ``dock`` is the ProjectWidget: ``dock.state`` the open project (None
    when none is), ``dock.engine_call(job, label, returned, errored)`` an engine call in the dock's worker
    (``job(runner, should_stop)``), ``dock.edit(...)`` a ``project edit`` that reloads the project, ``dock.running``
    whether a run goes. The dock calls ``project_changed(state)`` at every new reading of the project (None when it
    closes) and ``running_changed(running)`` when a run starts and ends: recipe edits are off while one goes."""

    def __init__(self, dock: ProjectWidget) -> None:
        super().__init__(dock)
        self.dock = dock

    def project_changed(self, state: ProjectState | None) -> None:
        pass

    def running_changed(self, running: bool) -> None:
        pass


class _Bridge(QObject):
    """Run-log events from the run worker's thread to the GUI thread (a signal emitted elsewhere is queued to the
    thread this object lives in)."""

    events = Signal(object)


def _tab_class(spec: str):
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)


# ---------------------------------------------------------------------------------------------------
# the New dialog
# ---------------------------------------------------------------------------------------------------


class NewProjectDialog(QDialog):
    """New project: P4's pack picker listing the engine's packs (the card checked in the engine environment, Get
    models and Get example there too), a name, a folder and the first images. The dock makes the engine calls;
    ``accept`` asks it for ``project new``. Opened with ``open()`` (never blocking the event loop); every control
    has a programmatic twin (set_folder, add_image, get_example, get_models)."""

    def __init__(self, dock: ProjectWidget) -> None:
        super().__init__(dock)
        self.dock = dock
        self.setWindowTitle("New germquant project")
        self.setObjectName("germquant_new_project_dialog")
        self._checks: dict[tuple[str, str], dict] = {}
        lay = QVBoxLayout(self)
        intro = QLabel("A project keeps one study: the pack version (copied into the project), its images, any "
                       "recipe edits and every result. The pack list, its checks and the models come from the "
                       "engine environment.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.picker = PackPicker()
        self.picker.interpreter_btn.setVisible(False)        # a recipe's own interpreter is set in the Run tab
        self.picker.selection_changed.connect(self._on_selection)
        self.picker.refresh_clicked.connect(lambda: self.dock._list_packs(self))
        self.picker.get_models_clicked.connect(self.get_models)
        self.picker.get_example_clicked.connect(self.get_example)
        lay.addWidget(self.picker)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("default: the folder's name")
        form.addRow("Name", self.name_edit)
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("a new or empty folder (a short path: results nest below it)")
        self.folder_edit.textChanged.connect(lambda _t: self._update())
        row.addWidget(self.folder_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_folder)
        row.addWidget(browse)
        form.addRow("Folder", row)
        self.images_list = QListWidget()
        form.addRow("Images", self.images_list)
        irow = QHBoxLayout()
        add = QPushButton("Add files...")
        add.clicked.connect(self._browse_images)
        irow.addWidget(add)
        drop = QPushButton("Remove")
        drop.clicked.connect(self._remove_image)
        irow.addWidget(drop)
        irow.addStretch(1)
        form.addRow("", irow)
        lay.addLayout(form)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._update()

    # ---- what the dock fills in ----
    def set_listing(self, listing: dict | None, error: str | None = None) -> None:
        if error is not None or listing is None:
            self.picker.show_message(f"could not list the packs: {error}", RED)
            self.picker.set_entries([])
        else:
            entries = _packs.listing_entries(listing)
            usable = [e for e in entries if e.usable_manifest]
            self.picker.show_message(f"{len(usable)} pack(s) found by the engine", None if usable else AMBER)
            self.picker.blockSignals(True)
            try:
                self.picker.set_entries(entries, select=(usable[0].id, None) if usable else None)
            finally:
                self.picker.blockSignals(False)
            self._on_selection()
        self._update()

    def show_check(self, ref: str, rid: str, status: dict) -> None:
        self._checks[(ref, rid)] = status
        entry = self.picker.current_entry()
        if entry is not None and entry.ref == ref and self.picker.current_recipe_id() == rid:
            self.picker.show_status(entry, rid, status, engine_text=self.dock.session.describe())
        self._update()

    def _on_selection(self) -> None:
        entry, rid = self.picker.current_entry(), self.picker.current_recipe_id()
        if entry is None or not rid:
            self._update()
            return
        st = self._checks.get((entry.ref, rid))
        if st is None:
            self.picker.show_status(entry, rid, None, engine_text=self.dock.session.describe(), pending=True)
            self.dock._check_recipe(self, entry.ref, rid)
        else:
            self.picker.show_status(entry, rid, st, engine_text=self.dock.session.describe())
        self._update()

    # ---- the controls and their twins ----
    def choice(self) -> tuple[str | None, str | None]:
        entry = self.picker.current_entry()
        return (entry.ref if entry is not None else None), self.picker.current_recipe_id()

    def recipe_usable(self) -> bool | None:
        """The engine's check of the chosen recipe: usable or not, None while it is being checked."""
        ref, rid = self.choice()
        st = self._checks.get((ref, rid)) if ref and rid else None
        return None if st is None else bool(st.get("usable"))

    def set_folder(self, path) -> None:
        self.folder_edit.setText(str(path))

    def images(self) -> list[str]:
        return [self.images_list.item(i).text() for i in range(self.images_list.count())]

    def add_image(self, path) -> None:
        text = str(path)
        if text and text not in self.images():
            self.images_list.addItem(text)

    def get_example(self) -> bool:
        ref, _rid = self.choice()
        if not ref:
            return False
        self.status.setText(f"getting the example of {ref} ...")
        return self.dock._fetch_example(self, ref)

    def get_models(self) -> bool:
        ref, rid = self.choice()
        if not ref or not rid:
            return False
        self.status.setText(f"getting the models of {ref} {rid} ...")
        return self.dock._fetch_models(self, ref, rid)

    def status_text(self) -> str:
        return self.status.text()

    def ok_button(self) -> QPushButton:
        return self.buttons.button(QDialogButtonBox.StandardButton.Ok)

    def _update(self) -> None:
        ref, rid = self.choice()
        self.ok_button().setEnabled(bool(ref and rid and self.folder_edit.text().strip()))

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder of the new project")
        if path:
            self.set_folder(path)

    def _browse_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Images", "", _image_filter())
        for p in paths:
            self.add_image(p)

    def _remove_image(self) -> None:
        for item in self.images_list.selectedItems():
            self.images_list.takeItem(self.images_list.row(item))

    def accept(self) -> None:
        ref, rid = self.choice()
        folder = self.folder_edit.text().strip().strip('"')
        if not ref or not rid or not folder:
            return
        self.dock.new_project(folder, ref, recipe=rid, name=self.name_edit.text().strip() or None,
                              images=self.images())
        super().accept()


def _image_filter() -> str:
    from germquant import api

    return "Images (" + " ".join(f"*{s}" for s in api.SUPPORTED_SUFFIXES) + ");;All files (*)"


# ---------------------------------------------------------------------------------------------------
# the dock
# ---------------------------------------------------------------------------------------------------


class ProjectWidget(QWidget):
    """napari dock widget (module docstring). ``napari_viewer`` is injected by napari; None builds the dock
    without a viewer. ``session`` replaces the viewer's engine session (tests)."""

    def __init__(self, napari_viewer=None, parent=None, *, session: EngineSession | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session = session if session is not None else session_for(napari_viewer)
        self.hub = project_hub(napari_viewer)
        self._state: ProjectState | None = None
        self._jobs: list[dict] = []
        self._loaded_settings: dict | None = None
        self._run_root: Path | None = None
        self._run_id: str | None = None
        self._run_sheet: tuple = (None, {})
        self.last_result: dict | None = None
        self.last_error: BaseException | None = None
        self.last_message = ""
        self.engine_error: str | None = None
        self.last_example: str | None = None
        self.new_dialog_ref: NewProjectDialog | None = None
        self.interpreter_dialog: InterpreterDialog | None = None
        self.engine_worker = WorkerRunner(napari_viewer, self)
        self.run_worker = WorkerRunner(napari_viewer, self)
        self.header_worker = WorkerRunner(None, self)
        self._bridge = _Bridge(self)
        self._bridge.events.connect(self._on_events)
        self.extra_tabs: dict[str, ProjectTab] = {}
        self._build()
        try:
            self.session.changed.connect(self._on_session_changed)
        except (RuntimeError, AttributeError):
            pass
        self.hub.reload_requested.connect(self.reload)
        self._show_project()
        self._check_engine()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        head = QGroupBox("Project")
        hform = QFormLayout(head)
        brow = QHBoxLayout()
        self.new_btn = QPushButton("New...")
        self.new_btn.setToolTip("make a project: a pack recipe, a folder and images (germquant project new)")
        self.new_btn.clicked.connect(self.new_dialog)
        brow.addWidget(self.new_btn)
        self.open_btn = QPushButton("Open...")
        self.open_btn.clicked.connect(self.open_dialog)
        brow.addWidget(self.open_btn)
        self.recent_combo = QComboBox()
        self.recent_combo.setToolTip("the projects opened on this computer")
        self.recent_combo.activated.connect(self._on_recent)
        brow.addWidget(self.recent_combo, 1)
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("record the Run tab's settings and the sample sheet in the project (germquant project "
                                 "edit, with this plugin's version)")
        self.save_btn.clicked.connect(self.save)
        brow.addWidget(self.save_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_project)
        brow.addWidget(self.close_btn)
        hform.addRow(brow)
        self.name_label = QLabel("")
        hform.addRow("Name", self.name_label)
        self.folder_label = QLabel("")
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hform.addRow("Folder", self.folder_label)
        erow = QHBoxLayout()
        self.engine_label = QLabel("")
        self.engine_label.setWordWrap(True)
        self.engine_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        erow.addWidget(self.engine_label, 1)
        self.engine_btn = QPushButton("Choose engine...")
        self.engine_btn.setToolTip("the Python environment that runs germquant on this computer (shared with the Run "
                                   "dock)")
        self.engine_btn.clicked.connect(self.ask_for_engine)
        erow.addWidget(self.engine_btn)
        hform.addRow("Engine", erow)
        self.engine_warn = QLabel("")
        self.engine_warn.setWordWrap(True)
        self.engine_warn.setStyleSheet(css(AMBER))
        hform.addRow("", self.engine_warn)
        self.pack_label = QLabel("")
        hform.addRow("Pack", self.pack_label)
        self.hash_label = QLabel("")
        self.hash_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hform.addRow("config_hash", self.hash_label)
        self.badge_label = QLabel("")
        hform.addRow("Validation", self.badge_label)
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet(css(AMBER))
        hform.addRow("Notes", self.warnings_label)
        lrow = QHBoxLayout()
        self.lock_label = QLabel("")
        self.lock_label.setWordWrap(True)
        lrow.addWidget(self.lock_label, 1)
        self.unlock_btn = QPushButton("Remove stale lock")
        self.unlock_btn.setToolTip("germquant project unlock with the holder's host and pid, as shown")
        self.unlock_btn.clicked.connect(self.remove_stale_lock)
        lrow.addWidget(self.unlock_btn)
        hform.addRow("Lock", lrow)
        outer.addWidget(head)

        split = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._images_tab(), "Images")
        self.tabs.addTab(self._samples_tab(), "Samples")
        self.tabs.addTab(self._run_tab(), "Run")
        for title, spec in EXTRA_TABS:
            tab = _tab_class(spec)(self)
            self.extra_tabs[title] = tab
            self.tabs.addTab(tab, title)
        split.addWidget(self.tabs)
        board_box = QGroupBox("Status board (double-click an image to open its results)")
        blay = QVBoxLayout(board_box)
        self.run_label = QLabel("")
        self.run_label.setWordWrap(True)
        blay.addWidget(self.run_label)
        self.board = QTableWidget(0, len(BOARD_COLUMNS))
        self.board.setHorizontalHeaderLabels(list(BOARD_COLUMNS))
        self.board.horizontalHeader().setStretchLastSection(True)
        self.board.verticalHeader().setVisible(False)
        self.board.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.board.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.board.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.board.cellDoubleClicked.connect(self._on_board_double_click)
        blay.addWidget(self.board)
        split.addWidget(board_box)
        outer.addWidget(split, 1)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.status_label)

    def _images_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.images_table = QTableWidget(0, len(IMAGE_COLUMNS))
        self.images_table.setHorizontalHeaderLabels(list(IMAGE_COLUMNS))
        self.images_table.horizontalHeader().setStretchLastSection(True)
        self.images_table.verticalHeader().setVisible(False)
        self.images_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.images_table)
        row = QHBoxLayout()
        self.add_files_btn = QPushButton("Add files...")
        self.add_files_btn.clicked.connect(self._browse_add_files)
        row.addWidget(self.add_files_btn)
        self.add_folder_btn = QPushButton("Add folder...")
        self.add_folder_btn.setToolTip("the engine searches it for the recipe's io.input_suffixes (every format germquant "
                                       "reads when the recipe names none); germquant outputs are skipped")
        self.add_folder_btn.clicked.connect(self._browse_add_folder)
        row.addWidget(self.add_folder_btn)
        lay.addLayout(row)
        prow = QHBoxLayout()
        self.paste_edit = QLineEdit()
        self.paste_edit.setPlaceholderText("or paste a path, a folder or a glob (a NAS path such as \\\\server\\share\\...)")
        self.paste_edit.returnPressed.connect(self._add_pasted)
        prow.addWidget(self.paste_edit, 1)
        self.paste_btn = QPushButton("Add path")
        self.paste_btn.clicked.connect(self._add_pasted)
        prow.addWidget(self.paste_btn)
        lay.addLayout(prow)
        self.refused_label = QLabel("")
        self.refused_label.setWordWrap(True)
        self.refused_label.setStyleSheet(css(RED))
        lay.addWidget(self.refused_label)
        return w

    def _samples_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.samples_note = QLabel("")
        self.samples_note.setWordWrap(True)
        lay.addWidget(self.samples_note)
        self.samples_table = QTableWidget(0, 0)
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.itemChanged.connect(self._on_sample_item)
        lay.addWidget(self.samples_table)
        row = QHBoxLayout()
        self.column_edit = QLineEdit()
        self.column_edit.setPlaceholderText("a new column, for example strain or lane")
        row.addWidget(self.column_edit, 1)
        self.add_column_btn = QPushButton("Add column")
        self.add_column_btn.clicked.connect(lambda: self._add_column_from_edit())
        row.addWidget(self.add_column_btn)
        self.save_sheet_btn = QPushButton("Save sheet")
        self.save_sheet_btn.setToolTip("write samples.csv (ASCII) with the engine's writer and record it in the "
                                       "project; images whose row changed turn stale")
        self.save_sheet_btn.clicked.connect(lambda: self.save(sheet=True))
        row.addWidget(self.save_sheet_btn)
        lay.addLayout(row)
        self._filling_samples = False
        return w

    def _run_tab(self) -> QWidget:
        from germquant import api

        w = QWidget()
        lay = QVBoxLayout(w)
        self.stages_box = QGroupBox("Stages (untick = --no-<stage>, recorded in the project with Save)")
        slay = QVBoxLayout(self.stages_box)
        self.stage_checks: dict[str, QCheckBox] = {}
        for st in api.cli_switches():
            cb = QCheckBox(st.name)
            cb.setToolTip(st.help)
            cb.setChecked(True)
            cb.toggled.connect(lambda _on: self._settings_touched())
            slay.addWidget(cb)
            self.stage_checks[st.name] = cb
        lay.addWidget(self.stages_box)
        form = QFormLayout()
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 16)
        self.stride_spin.setToolTip("downsample x and y for a quick run (1 = full resolution, the validated setting); "
                                    "recorded in the project")
        self.stride_spin.valueChanged.connect(lambda _v: self._settings_touched())
        form.addRow("xy stride", self.stride_spin)
        self.device_combo = QComboBox()
        for value in DEVICES:
            self.device_combo.addItem(DEVICE_TEXT.get(value, value), value)
        self.device_combo.setToolTip("the device this computer asks for (the per-user choice the Run dock also uses); "
                                     "the project records it as a note")
        self.device_combo.currentIndexChanged.connect(self._on_device_combo)
        form.addRow("Device", self.device_combo)
        irow = QHBoxLayout()
        self.interpreter_label = QLabel("")
        self.interpreter_label.setWordWrap(True)
        irow.addWidget(self.interpreter_label, 1)
        self.interpreter_btn = QPushButton("Recipe interpreter...")
        self.interpreter_btn.setToolTip("run this project's recipe in another Python environment on this computer "
                                        "(stored for this computer; the project records it with Save)")
        self.interpreter_btn.clicked.connect(self.ask_for_recipe_interpreter)
        irow.addWidget(self.interpreter_btn)
        form.addRow("Interpreter", irow)
        lay.addLayout(form)
        rrow = QHBoxLayout()
        self.run_selected_btn = QPushButton("Run selected")
        self.run_selected_btn.setToolTip("run the images selected on the board (done images are skipped)")
        self.run_selected_btn.clicked.connect(lambda: self.run_selected())
        rrow.addWidget(self.run_selected_btn)
        self.run_all_btn = QPushButton("Run all")
        self.run_all_btn.clicked.connect(lambda: self.run_all())
        rrow.addWidget(self.run_all_btn)
        self.rerun_btn = QPushButton("Rerun selected")
        self.rerun_btn.setToolTip("run the selected images again, done or not (--no-resume)")
        self.rerun_btn.clicked.connect(lambda: self.rerun_selected())
        rrow.addWidget(self.rerun_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("stop the run: the engine process and every process it started are ended; the image "
                                   "in progress gets no results")
        self.cancel_btn.clicked.connect(self.cancel)
        rrow.addWidget(self.cancel_btn)
        lay.addLayout(rrow)
        crow = QHBoxLayout()
        self.collect_btn = QPushButton("Collect tables")
        self.collect_btn.setToolTip("germquant project collect: the batch tables of the finished images")
        self.collect_btn.clicked.connect(lambda: self.collect())
        crow.addWidget(self.collect_btn)
        self.export_btn = QPushButton("Export commands...")
        self.export_btn.setToolTip("germquant project export-commands: the commands that run this project headless")
        self.export_btn.clicked.connect(self._export_dialog)
        crow.addWidget(self.export_btn)
        crow.addStretch(1)
        lay.addLayout(crow)
        self.runs_table = QTableWidget(0, 5)
        self.runs_table.setHorizontalHeaderLabels(["Run", "Via", "Started", "Ended", "Counts"])
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.runs_table)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ engine calls, one at a time
    @property
    def idle(self) -> bool:
        """No engine call, run or header read in flight or waiting."""
        return (not self.engine_worker.busy and not self.run_worker.busy and not self.header_worker.busy
                and not self._jobs)

    def wait_idle(self, timeout_s: float = 300.0) -> bool:
        """For a script driving the dock: run the Qt event loop until ``idle`` (True) or ``timeout_s`` (False)."""
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        t0 = time.monotonic()
        while not self.idle:
            if time.monotonic() - t0 > timeout_s:
                return False
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        return True

    def _submit(self, kind: str, func, label: str, returned, errored=None, *, replace: bool = False) -> None:
        """``func(should_stop)`` in the engine worker when it is free; ``replace``: a newer request of the same kind
        takes the place of one still waiting."""
        job = {"kind": kind, "func": func, "label": label, "returned": returned, "errored": errored or self._failed}
        if replace:
            self._jobs = [j for j in self._jobs if j["kind"] != kind]
        self._jobs.append(job)
        self._pump()

    def _pump(self) -> None:
        if self.engine_worker.busy or not self._jobs:
            return
        job = self._jobs.pop(0)
        worker = self.engine_worker
        worker.start(lambda: job["func"](worker.should_stop), label=job["label"], returned=job["returned"],
                     errored=job["errored"], finished=self._pump)

    def engine_call(self, job, label: str, returned, errored=None, *, features=("projects",)) -> None:
        """For a tab: ``job(runner, should_stop)`` with the engine (``features`` required) in the engine worker;
        ``returned(result)`` and ``errored(exc)`` in the GUI thread."""
        def func(should_stop):
            return job(self.session.require(*features, should_stop=should_stop), should_stop)

        self._submit("call", func, label, returned, errored)

    def _failed(self, exc: BaseException) -> None:
        self.last_error = exc
        if isinstance(exc, RunCancelled):
            self._message(f"cancelled: {exc}", AMBER)
            return
        self._message(_error_text(exc), RED)

    # ------------------------------------------------------------------ the engine
    def _check_engine(self) -> None:
        def job(should_stop):
            return self.session.ensure(should_stop)

        def failed(exc):
            if isinstance(exc, EngineNotChosen):
                self.engine_error = None
            else:
                self.engine_error = _error_text(exc)
            self._show_engine()

        self._submit("engine", job, "checking the engine", lambda _r: self._engine_ready(), failed, replace=True)

    def _engine_ready(self) -> None:
        self.engine_error = None
        self._show_engine()
        self._show_project()

    def set_engine(self, choice) -> None:
        """Use ``choice`` (a Python path, or ``inprocess``) as this computer's engine, stored once its check passes
        (the choice the Run dock reads too). The check runs in the engine worker; ``engine_error`` says how it went
        once ``idle``."""
        choice = str(choice)

        def job(should_stop):
            return self.session.set_engine(choice, should_stop=should_stop)

        def failed(exc):
            self.engine_error = _error_text(exc)
            self._show_engine()
            self._message(f"that engine cannot be used: {self.engine_error}", RED)

        self._submit("engine", job, f"checking the engine {choice}", lambda _r: self._engine_ready(), failed,
                     replace=True)

    def ask_for_engine(self) -> InterpreterDialog:
        from ._run_widget import ENGINE_INTRO

        dlg = InterpreterDialog(detect_interpreters(), title="germquant engine for this computer", intro=ENGINE_INTRO,
                                current=_settings.engine_choice(), allow_inprocess=True, parent=self)
        dlg.accepted.connect(lambda: self.set_engine(dlg.choice()) if dlg.choice() else None)
        self.interpreter_dialog = dlg
        dlg.open()
        return dlg

    def _show_engine(self) -> None:
        if self.engine_error:
            self.engine_label.setText(f"The engine chosen for this computer cannot be used: {self.engine_error}")
            self.engine_label.setStyleSheet(css(RED))
        elif self.session.runner() is None and _settings.engine_choice() is None:
            self.engine_label.setText(NOT_CHOSEN)
            self.engine_label.setStyleSheet(css(AMBER))
        else:
            self.engine_label.setText(self.session.describe())
            self.engine_label.setStyleSheet("")
        self.engine_warn.setText(self.session.warning() if self.session.runner() is not None else "")
        self.engine_warn.setVisible(bool(self.engine_warn.text()))

    def _on_session_changed(self) -> None:
        self._show_engine()
        self._sync_device_combo()
        self._show_notes()
        self._update_buttons()

    def _this_computer(self) -> dict:
        """This computer's device, runner kind and the interpreter that runs the project's recipe here (read from
        the settings and the checked engine: nothing is started)."""
        st = self._state
        pack_id = st.pack_id if st is not None else None
        recipe_id = st.recipe_id if st is not None else None
        mapped = _settings.recipe_interpreter(pack_id, recipe_id)
        python = kind = None
        if mapped:
            python = sys.executable if mapped == INPROCESS else mapped
            kind = "inprocess" if mapped == INPROCESS else "subprocess"
        else:
            runner = self.session.runner()
            if runner is not None:
                python = runner.python
                kind = "inprocess" if isinstance(runner, InProcessRunner) else "subprocess"
            else:
                choice = _settings.engine_choice()
                if choice:
                    python = sys.executable if choice == INPROCESS else choice
                    kind = "inprocess" if choice == INPROCESS else "subprocess"
        return {"device": self.session.device(), "runner": kind, "python": python, "recipe": recipe_id}

    # ------------------------------------------------------------------ reading a project
    def _load(self, root: Path, should_stop) -> ProjectState:
        """Everything the dock shows of ``root`` (the worker's part: the manifest, the engine's status, the board, the
        recipe's Config and the sample table; the headers come after, from the header worker)."""
        manifest = P.read_manifest(root)
        status, error = None, None
        try:
            runner = self.session.runner_for(manifest["pack"].get("id"), manifest["pack"].get("recipe"),
                                             should_stop=should_stop)
            runner.require("projects", should_stop=should_stop)
            status = runner.project_status(root, should_stop=should_stop)
            if not isinstance(status, dict) or not isinstance(status.get("images"), list):
                status, error = None, RunnerError(f"the engine's project status is not a status record: {status!r}"[:400])
        except RunCancelled:
            raise
        except (RunnerError, OSError, ValueError) as e:
            error = e
        state = ProjectState(root=root, manifest=manifest, status=status, status_error=error,
                             rows=P.board_rows(root, manifest, status), kind=P.object_kind(root, manifest))
        try:
            state.config = P.recipe_config(root, manifest)
        except Exception as e:  # noqa: BLE001 - the Run tab says why the switches cannot be checked
            state.config_error = f"{type(e).__name__}: {e}"
        if status is not None:
            state.own_hash = P.own_config_hash(root)
        regex = defaults = None
        if state.config is not None:
            regex = state.config.get("metadata.filename_regex")
            defaults = state.config.get("metadata.defaults")
        try:
            state.samples = P.SampleTable.from_project(root, manifest, regex=regex, defaults=defaults)
        except Exception as e:  # noqa: BLE001 - the Samples tab says why
            state.samples_error = f"{type(e).__name__}: {e}"
        return state

    def open_project(self, folder) -> None:
        """Open the project in ``folder``: the manifest read in process, the hash computed again and the status
        asked of the engine (design 7.3), then the image headers read."""
        if self.running:
            self._message("a run is in progress: cancel it or wait for it before opening another project", AMBER)
            return
        root = Path(str(folder).strip().strip('"'))
        self._submit("open", lambda should_stop: self._load(root, should_stop), f"opening the project {root}",
                     lambda state: self._opened(state, fresh=True), self._open_failed, replace=True)

    def reload(self) -> None:
        """Read the open project again (the manifest, the engine's status, the board)."""
        if self._state is None:
            return
        root = self._state.root
        self._submit("open", lambda should_stop: self._load(root, should_stop), f"reading the project {root}",
                     lambda state: self._opened(state, fresh=False), self._open_failed, replace=True)

    def _open_failed(self, exc: BaseException) -> None:
        self.last_error = exc
        if isinstance(exc, P.ProjectFileError) and self._state is not None and \
                not P.is_project(self._state.root):
            self._close()
        self._message(_error_text(exc), RED)

    def _opened(self, state: ProjectState, *, fresh: bool) -> None:
        old = self._state
        if old is not None and old.root == state.root:
            ids = {str(e.get("id")) for e in state.manifest.get("images") or [] if isinstance(e, dict)}
            merged = {k: v for k, v in old.headers.items() if k in ids}
            merged.update(state.headers)                # the headers an add just brought win
            state.headers = merged
            if old.samples is not None and old.samples.dirty and state.samples is not None and not fresh:
                state.samples = old.samples            # unsaved sample edits survive a status refresh
        elif old is not None:
            self._close(signal=False)
        self._state = state
        self.last_message = ""
        self._apply_settings_widgets(state.manifest, force=fresh or old is None or old.root != state.root)
        self._show_project()
        if fresh or old is None or old.root != state.root:
            _settings.add_recent_project(str(state.root))
            self._fill_recent()
            self.hub.root, self.hub.state = state.root, state
            self.hub._emit(self.hub.opened, str(state.root))
        self.hub.state = state
        self.hub._emit(self.hub.changed)
        for tab in self.extra_tabs.values():
            tab.project_changed(state)
        err = state.status_error
        if err is not None:
            self.last_error = err
            if isinstance(err, EngineNotChosen):
                self._message(f"{NOT_CHOSEN} The board shows the manifest's images until then.", AMBER)
            else:
                self._message(f"the engine gave no status: {_error_text(err)}", RED if not isinstance(
                    err, FeatureMissing) else AMBER)
        elif fresh:
            self._message(f"opened {state.root}", GREEN)
        self._read_headers()

    def _read_headers(self) -> None:
        st = self._state
        if st is None or self.header_worker.busy:
            return
        entries = [dict(e) for e in st.manifest.get("images") or [] if isinstance(e, dict)
                   and str(e.get("id")) not in st.headers]
        if not entries:
            return
        root = st.root

        def job():
            return root, {str(e.get("id")): P.probe_image(root, e) for e in entries}

        self.header_worker.start(job, label="reading the image headers", returned=self._headers_read,
                                 errored=lambda exc: self._message(f"the headers could not be read: {_error_text(exc)}",
                                                                   AMBER),
                                 finished=self._read_headers, line_format=lambda _l: None)

    def _headers_read(self, res) -> None:
        root, headers = res
        if self._state is None or self._state.root != root:
            return
        self._state.headers.update(headers)
        self._show_images()
        self._show_stages()
        self._update_buttons()

    def _close(self, *, signal: bool = True) -> None:
        self._state = None
        self._loaded_settings = None
        self.board.setRowCount(0)
        self.images_table.setRowCount(0)
        self.refused_label.setText("")
        self.run_label.setText("")
        if signal:
            self.hub.root, self.hub.state = None, None
            self.hub._emit(self.hub.closed)
            for tab in self.extra_tabs.values():
                tab.project_changed(None)
        self._show_project()

    def close_project(self) -> bool:
        """Close the open project (refused while it runs)."""
        if self.running:
            self._message("a run of this project is in progress: cancel it or wait for it before closing", AMBER)
            return False
        if self._state is None:
            return False
        self._jobs = [j for j in self._jobs if j["kind"] not in ("open",)]
        self._close()
        self._message("project closed", None)
        return True

    # ------------------------------------------------------------------ new, open, save
    def new_dialog(self) -> NewProjectDialog:
        """The New dialog (non-blocking); its pack list comes from the engine."""
        dlg = NewProjectDialog(self)
        self.new_dialog_ref = dlg
        self._list_packs(dlg)
        dlg.open()
        return dlg

    def _list_packs(self, dlg: NewProjectDialog) -> None:
        def job(should_stop):
            return self.session.ensure(should_stop).list_packs(should_stop=should_stop)

        self._submit("listing", job, "listing the engine's packs", lambda listing: dlg.set_listing(listing),
                     lambda exc: dlg.set_listing(None, _error_text(exc)), replace=True)

    def _check_recipe(self, dlg: NewProjectDialog, ref: str, rid: str) -> None:
        def job(should_stop):
            runner = self.session.ensure(should_stop)
            try:
                return runner.check(ref, rid, should_stop=should_stop)
            except RunCancelled:
                raise
            except RunnerError as e:
                return {"pack": ref, "recipe": rid, "usable": False, "warnings": [], "models": [],
                        "blockers": [{"what": str(e), "fix": "choose a working engine (Choose engine...)"}]}

        self._submit("card", job, f"checking {ref} {rid}", lambda st: dlg.show_check(ref, rid, st), replace=True)

    def _fetch_example(self, dlg: NewProjectDialog, ref: str) -> bool:
        def job(should_stop):
            return self.session.ensure(should_stop).fetch_example(ref, say=_say, should_stop=should_stop)

        def done(rec):
            self.last_example = str(rec.get("image"))
            dlg.add_image(self.last_example)
            dlg.status.setText(f"example {rec.get('id')} ({rec.get('status')}): {self.last_example}")

        self._submit("fetch", job, f"getting the example of {ref}", done,
                     lambda exc: dlg.status.setText(f"could not get the example: {_error_text(exc)}"))
        return True

    def _fetch_models(self, dlg: NewProjectDialog, ref: str, rid: str) -> bool:
        def job(should_stop):
            return self.session.ensure(should_stop).fetch_models(ref, rid, say=_say, should_stop=should_stop)

        def done(res):
            lines = "\n".join(res.get("lines") or [])
            dlg.status.setText(("models in the store:\n" if res.get("ok") else "some models could not be fetched:\n")
                               + lines)
            dlg._checks.pop((ref, rid), None)
            dlg._on_selection()                   # the card is checked again

        self._submit("fetch", job, f"getting the models of {ref} {rid}", done,
                     lambda exc: dlg.status.setText(f"could not get the models: {_error_text(exc)}"))
        return True

    def new_project(self, folder, pack: str, recipe: str | None = None, *, name: str | None = None,
                    images=()) -> None:
        """``germquant project new FOLDER --pack PACK [--recipe R] [--images ...] [--name N] --plugin-version V``
        through the engine, then this computer's device, runner and interpreter recorded (``project edit``, D42),
        then the project opened."""
        if self.running:
            self._message("a run is in progress: cancel it or wait for it before making another project", AMBER)
            return
        root = Path(str(folder).strip().strip('"'))
        paths = [str(p) for p in ([images] if isinstance(images, (str, os.PathLike)) else list(images or ()))]

        def job(should_stop):
            runner = self.session.require("projects", should_stop=should_stop)
            rec = runner.project_new(root, pack, recipe=recipe, images=paths, name=name,
                                     plugin_version=plugin_version(), should_stop=should_stop)
            manifest = P.read_manifest(root)
            here = self._here(manifest, should_stop)
            changes = P.settings_changes(manifest, device=here["device"], runner=here["runner"],
                                         interpreter=(manifest["pack"].get("recipe"), here["python"]))
            if changes:
                runner.project_edit(root, **changes, plugin_version=plugin_version(), should_stop=should_stop)
            return rec, self._load(root, should_stop)

        def done(res):
            rec, state = res
            self._opened(state, fresh=True)
            refused = rec.get("refused") or []
            if refused:
                self.refused_label.setText("\n".join(f"not added: {r.get('path')}: {r.get('reason')}" for r in refused))
            warn = [str(w) for w in rec.get("warnings") or []]
            self._done(f"made the project {root} ({state.pack_ref} {state.recipe_id})"
                       + ("; " + "; ".join(warn) if warn else ""), state, AMBER if refused or warn else GREEN)

        self._submit("new", job, f"making the project {root}", done)

    def _here(self, manifest: dict, should_stop) -> dict:
        """This computer's device, runner kind and the interpreter of the manifest's recipe (the worker's part: the
        recipe's own interpreter is checked once)."""
        runner = self.session.runner_for(manifest["pack"].get("id"), manifest["pack"].get("recipe"),
                                         should_stop=should_stop)
        kind = "inprocess" if isinstance(runner, InProcessRunner) else "subprocess"
        return {"device": self.session.device(), "runner": kind, "python": runner.python}

    def open_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Project folder (it holds germquant_project.yaml)")
        if folder:
            self.open_project(folder)

    def _fill_recent(self) -> None:
        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        self.recent_combo.setPlaceholderText("recent projects ...")
        for p in _settings.recent_projects():
            self.recent_combo.addItem(p, p)
        self.recent_combo.setCurrentIndex(-1)
        self.recent_combo.blockSignals(False)

    def _on_recent(self, index: int) -> None:
        path = self.recent_combo.itemData(index)
        if path:
            self.open_project(path)

    def _pending(self) -> dict:
        """The ``project edit`` keywords of everything the Run tab holds differently from the manifest."""
        st = self._state
        if st is None:
            return {}
        here = self._this_computer()
        disabled = [name for name, cb in self.stage_checks.items() if not cb.isChecked()]
        return P.settings_changes(st.manifest, device=here["device"], runner=here["runner"],
                                  xy_stride=int(self.stride_spin.value()), disabled=disabled,
                                  interpreter=(st.recipe_id, here["python"]))

    @property
    def unsaved(self) -> bool:
        """Whether Save would write anything: Run tab settings that differ from the manifest, or sample edits."""
        st = self._state
        if st is None:
            return False
        return bool(self._pending()) or bool(st.samples is not None and st.samples.dirty)

    def save(self, *, sheet: bool = False) -> bool:
        """Save: the Run tab's settings that differ from the manifest in one ``project edit`` (with this plugin's
        version), and the sample sheet when it was edited (``sheet``: write it even without edits)."""
        st = self._state
        if st is None:
            return False
        if self._blocked("save"):
            return False
        changes = self._pending()
        samples = st.samples if st.samples is not None and (st.samples.dirty or sheet) else None
        if not changes and samples is None:
            self._message("nothing to save: the project records every setting shown", None)
            return False
        root = st.root
        records = samples.records() if samples is not None else None

        def job(should_stop):
            self._write_settings(root, changes, records, should_stop)
            return self._load(root, should_stop)

        def done(state):
            if samples is not None:
                samples.mark_saved()
            self._opened(state, fresh=False)
            self._done(f"saved {root}", state)

        self._submit("save", job, f"saving {root}", done)
        return True

    def _write_settings(self, root: Path, changes: dict, records: list | None, should_stop) -> bool:
        """The worker's part of Save (and of a run's save first): the sheet through the engine's writer, then one
        ``project edit``. True when it wrote the sheet."""
        changes = dict(changes)
        runner = self.session.require("projects", should_stop=should_stop)
        if records is not None:
            sheet = P.sheet_module()
            if sheet is None:                         # this napari environment's germquant has no sheet writer
                raise FeatureMissing(_feature_text("sample_sheet", None), "sample_sheet")
            runner.require("sample_sheet", should_stop=should_stop)
            sheet.write_sample_sheet(root / P.SAMPLES_NAME, records)
            if P.read_manifest(root).get("samples") != P.SAMPLES_NAME:
                changes["samples"] = P.SAMPLES_NAME
        if changes:
            runner.project_edit(root, **changes, plugin_version=plugin_version(), should_stop=should_stop)
        return records is not None

    def edit(self, *, on_done=None, **kwargs) -> bool:
        """``project edit`` with ``kwargs`` (the runner's project_edit keywords: edits, unset, roles, revert, ...),
        stamped with this plugin's version, then the project read again; with ``dry_run`` only ``on_done(record)``.
        Refused while a run of the project goes."""
        st = self._state
        if st is None:
            self._message("open a project first", AMBER)
            return False
        if self._blocked("edit"):
            return False
        root = st.root
        dry = bool(kwargs.get("dry_run"))

        def job(should_stop):
            runner = self.session.require("projects", should_stop=should_stop)
            rec = runner.project_edit(root, **kwargs, plugin_version=plugin_version(), should_stop=should_stop)
            return rec, None if dry else self._load(root, should_stop)

        def done(res):
            rec, state = res
            if state is not None:
                self._opened(state, fresh=False)
                self._done(f"recorded the edit: config_hash {rec.get('config_hash')}", state)
            if on_done is not None:
                on_done(rec)

        self._submit("edit", job, f"editing {root}", done)
        return True

    def _blocked(self, what: str) -> bool:
        st = self._state
        if st is not None and (self.running or P.run_in_progress(st.root)):
            self._message(f"a run of this project is in progress: the {what} waits until it has finished", AMBER)
            return True
        return False

    # ------------------------------------------------------------------ images
    def add_paths(self, paths) -> bool:
        """``germquant project add`` with files, folders or globs (a pasted path kept as it is); the refusals are
        named per path, the added images get the header the engine read."""
        st = self._state
        if st is None:
            self._message("open a project first", AMBER)
            return False
        if self._blocked("add"):
            return False
        paths = [str(p) for p in ([paths] if isinstance(paths, (str, os.PathLike)) else list(paths))]
        if not paths:
            return False
        root = st.root

        def job(should_stop):
            runner = self.session.require("projects", should_stop=should_stop)
            rec = runner.project_add(root, paths, plugin_version=plugin_version(), should_stop=should_stop)
            return rec, self._load(root, should_stop)

        def done(res):
            rec, state = res
            for iid, header in (rec.get("headers") or {}).items():
                state.headers[str(iid)] = dict(header)
            self._opened(state, fresh=False)
            refused = rec.get("refused") or []
            self.refused_label.setText("\n".join(f"not added: {r.get('path')}: {r.get('reason')}" for r in refused))
            added = rec.get("added") or []
            self._done(f"added {len(added)} image(s)" + (f"; {len(refused)} refused (named in the Images tab)"
                                                         if refused else ""), state, AMBER if refused else GREEN)
            self._show_images()

        self._submit("add", job, f"adding {len(paths)} path(s) to {root}", done)
        return True

    def _browse_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Images to add", "", _image_filter())
        if paths:
            self.add_paths(paths)

    def _browse_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "A folder of images to add")
        if folder:
            self.add_paths([folder])

    def _add_pasted(self) -> None:
        text = self.paste_edit.text().strip().strip('"')
        if text:
            self.add_paths([text])
            self.paste_edit.clear()

    # ------------------------------------------------------------------ samples
    @property
    def samples(self) -> P.SampleTable | None:
        return self._state.samples if self._state is not None else None

    def set_sample(self, image_id: str, column: str, value) -> None:
        """One cell of the sample table (ValueError for a value the sheet cannot hold, naming the cell)."""
        if self.samples is None:
            raise ValueError("no sample table: open a project first")
        if self._blocked("sample edit"):
            raise ValueError("a run of this project is in progress: sample edits wait until it has finished")
        self.samples.set(image_id, column, value)
        self._show_samples()
        self._update_buttons()

    def add_sample_column(self, name: str) -> None:
        if self.samples is None:
            raise ValueError("no sample table: open a project first")
        self.samples.add_column(name)
        self._show_samples()
        self._update_buttons()

    def _add_column_from_edit(self) -> None:
        try:
            self.add_sample_column(self.column_edit.text())
        except ValueError as e:
            self._message(str(e), RED)
            return
        self.column_edit.clear()

    def _on_sample_item(self, item: QTableWidgetItem) -> None:
        if self._filling_samples or self.samples is None:
            return
        t = self.samples
        iid, column = t.ids[item.row()], t.columns[item.column()]
        try:
            self.set_sample(iid, column, item.text())
        except (ValueError, KeyError) as e:
            self._message(str(e), RED)
            self._show_samples()

    # ------------------------------------------------------------------ the Run tab
    def set_xy_stride(self, n: int) -> None:
        self.stride_spin.setValue(int(n))

    def set_stage(self, name: str, on: bool) -> None:
        if name not in self.stage_checks:
            raise ValueError(f"not a stage with a switch: {name!r}")
        self.stage_checks[name].setChecked(bool(on))

    def set_device(self, value: str) -> None:
        """The device this computer asks for (the per-user setting the Run dock also uses)."""
        if value not in DEVICES:
            raise ValueError(f"not a device: {value!r} (one of {', '.join(DEVICES)})")
        self.session.set_device(value)
        self._sync_device_combo()
        self._settings_touched()

    def _on_device_combo(self, _index: int) -> None:
        value = str(self.device_combo.currentData() or "auto")
        if value != self.session.device():
            self.session.set_device(value)
        self._settings_touched()

    def _sync_device_combo(self) -> None:
        self.device_combo.blockSignals(True)
        i = self.device_combo.findData(self.session.device())
        self.device_combo.setCurrentIndex(i if i >= 0 else 0)
        self.device_combo.blockSignals(False)

    def ask_for_recipe_interpreter(self) -> InterpreterDialog | None:
        st = self._state
        if st is None:
            return None
        intro = (f"Recipe {st.recipe_id} of pack {st.pack_id} can run in its own Python environment on this computer, "
                 f"for example one with an optional backend the engine lacks. The project's runs and status then go "
                 f"there; Save records it in the project.")
        dlg = InterpreterDialog(detect_interpreters(), title=f"interpreter for {st.pack_id} {st.recipe_id}", intro=intro,
                                current=_settings.recipe_interpreter(st.pack_id, st.recipe_id), allow_inprocess=True,
                                allow_default=True, parent=self)
        dlg.accepted.connect(lambda: self.set_recipe_interpreter(dlg.choice() or None))
        self.interpreter_dialog = dlg
        dlg.open()
        return dlg

    def set_recipe_interpreter(self, python) -> None:
        """Store (or with None clear) the interpreter of the project's recipe on this computer, once its version
        check passes (the Run dock reads the same setting)."""
        st = self._state
        if st is None:
            return
        pack_id, recipe_id = st.pack_id, st.recipe_id
        if not python:
            _settings.set_recipe_interpreter(pack_id, recipe_id, None)
            self._settings_touched()
            return
        choice = str(python)

        def job(should_stop):
            if choice != INPROCESS:
                SubprocessRunner(choice, env=self.session.env).version(should_stop=should_stop)
            return choice

        def done(_c):
            _settings.set_recipe_interpreter(pack_id, recipe_id, choice)
            self._message(f"recipe {pack_id} {recipe_id} runs in {choice} on this computer", GREEN)
            self._settings_touched()

        self._submit("interpreter", job, f"checking {choice}", done,
                     lambda exc: self._message(f"that interpreter cannot run {pack_id} {recipe_id}: {_error_text(exc)}",
                                               RED))

    def _settings_touched(self) -> None:
        self._show_interpreter()
        self._show_notes()
        self._update_buttons()

    def _apply_settings_widgets(self, manifest: dict, *, force: bool) -> None:
        """The Run tab's widgets from the manifest: always for a project just opened or saved, else only when the
        manifest's settings changed (so the user's unsaved changes survive a status refresh)."""
        rec = P.recorded_settings(manifest)
        if not force and self._loaded_settings == rec:
            return
        self._loaded_settings = rec
        for w in (self.stride_spin, *self.stage_checks.values()):
            w.blockSignals(True)
        try:
            self.stride_spin.setValue(int(rec["xy_stride"]))
            for name, cb in self.stage_checks.items():
                cb.setChecked(name not in rec["disabled"])
        finally:
            for w in (self.stride_spin, *self.stage_checks.values()):
                w.blockSignals(False)
        self._sync_device_combo()

    def _selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.board.selectedIndexes()})
        st = self._state
        if st is None:
            return []
        return [st.rows[r].image_id for r in rows if 0 <= r < len(st.rows)]

    def run_all(self) -> bool:
        return self.run(None, resume=True)

    def run_selected(self, ids=None) -> bool:
        ids = list(ids) if ids is not None else self._selected_ids()
        if not ids:
            self._message("select the images to run on the board first", AMBER)
            return False
        return self.run(ids, resume=True)

    def rerun_selected(self, ids=None) -> bool:
        ids = list(ids) if ids is not None else self._selected_ids()
        if not ids:
            self._message("select the images to run again on the board first", AMBER)
            return False
        return self.run(ids, resume=False)

    @property
    def running(self) -> bool:
        return self.run_worker.busy

    def run(self, images=None, *, resume: bool = True) -> bool:
        """``germquant project run`` of ``images`` (None: all) through the engine runner of the project's recipe,
        with this computer's device, a run id of the dock's and ``--via``; the settings that differ from the
        manifest are recorded first. The board follows the run's log; the engine's status is asked again at its
        end."""
        st = self._state
        if st is None:
            self._message("open a project first", AMBER)
            return False
        if self.running:
            self._message("a run is already in progress", AMBER)
            return False
        if P.run_in_progress(st.root):
            self._message("another dock runs this project now", AMBER)
            return False
        root = st.root
        changes = self._pending()
        samples = st.samples if st.samples is not None and st.samples.dirty else None
        records = samples.records() if samples is not None else None
        device = self.session.device()
        run_id = new_run_id()
        pack_id, recipe_id = st.pack_id, st.recipe_id
        ids = list(images) if images else None
        bridge = self._bridge
        wrote = {"sheet": False}
        self._run_sheet = (samples, wrote)

        def job():
            stop = self.run_worker.should_stop
            if changes or records is not None:
                wrote["sheet"] = self._write_settings(root, changes, records, stop)
            runner = self.session.runner_for(pack_id, recipe_id, should_stop=stop)
            runner.require("projects", should_stop=stop)
            return runner.project_run(root, images=ids, resume=resume, device=device, run_id=run_id, say=_say,
                                      progress=lambda events: bridge.events.emit(list(events)), should_stop=stop)

        self._run_root, self._run_id = root, run_id
        self.last_result, self.last_error = None, None
        P.mark_running(root, run_id)
        what = "every image" if ids is None else ", ".join(ids)
        self.run_label.setText(f"run {run_id}: {what}{'' if resume else ' (again, without resume)'} ...")
        self.run_label.setStyleSheet("")
        self.run_worker.start(job, label=f"germquant project run {root.name}", returned=self._run_returned,
                              errored=self._run_errored, finished=self._run_finished)
        self._set_running(True)
        return True

    def _set_running(self, running: bool) -> None:
        self.hub.running = running
        self.hub._emit(self.hub.running_changed, running)
        for tab in self.extra_tabs.values():
            tab.running_changed(running)
        self._update_buttons()

    def _on_events(self, events) -> None:
        st = self._state
        if st is None or self._run_root is None or st.root != self._run_root:
            return
        st.rows = P.apply_events(st.rows, events)
        self._show_board()

    def _run_returned(self, result: dict) -> None:
        self.last_result = result
        text, level = P.run_text(result)
        self.last_message = text
        self.run_label.setText(text)
        self.run_label.setStyleSheet(css(LEVELS[level]))

    def _run_errored(self, exc: BaseException) -> None:
        self.last_error = exc
        if isinstance(exc, RunCancelled):
            outcome = run_outcome(list(getattr(exc, "events", None) or []), alive=False)
            text = (f"run {getattr(exc, 'run_id', self._run_id)} interrupted (cancelled): "
                    f"{P.counts_text(outcome['counts'])}; the image in progress has no results")
            self.run_label.setStyleSheet(css(AMBER))
        else:
            text = f"run {getattr(exc, 'run_id', self._run_id)} could not be made: {_error_text(exc)}"
            self.run_label.setStyleSheet(css(RED))
        self.last_message = text
        self.run_label.setText(text)

    def _run_finished(self) -> None:
        root = self._run_root
        if root is not None:
            P.mark_finished(root)
        self._run_root = None
        samples, wrote = getattr(self, "_run_sheet", (None, {}))
        if samples is not None and wrote.get("sheet"):
            samples.mark_saved()                      # the run's save wrote the sheet before it ran
        self._run_sheet = (None, {})
        self._set_running(False)
        if self._state is not None and root is not None and self._state.root == root:
            self.reload()

    def cancel(self) -> bool:
        """Cancel the run in progress: its engine process and every process it started are ended (in process, the
        run stops between images)."""
        if not self.running:
            return False
        self.run_worker.cancel()
        self.cancel_btn.setEnabled(False)
        self._message("cancelling the run: ending the engine process ...", AMBER)
        return True

    def collect(self) -> bool:
        """``germquant project collect``: the batch tables of the finished images."""
        st = self._state
        if st is None:
            return False
        root = st.root

        def job(should_stop):
            return self.session.require("projects", should_stop=should_stop).project_collect(root,
                                                                                            should_stop=should_stop)

        def done(rec):
            tables = rec.get("tables") if isinstance(rec, dict) else None
            text = ", ".join(str(t) for t in tables) if isinstance(tables, list) else str(rec)[:600]
            self._message(f"collected: {text}", GREEN)

        self._submit("collect", job, f"collecting the tables of {root}", done)
        return True

    def export_commands(self, out, *, shell: str | None = None) -> bool:
        """``germquant project export-commands`` into ``out`` (ps1 on Windows and sh elsewhere by default)."""
        st = self._state
        if st is None:
            return False
        root, out = st.root, Path(out)
        shell = shell or ("ps1" if os.name == "nt" else "sh")

        def job(should_stop):
            runner = self.session.require("projects", should_stop=should_stop)
            return runner.project_export_commands(root, shell=shell, out=out, should_stop=should_stop)

        self._submit("export", job, f"exporting the commands of {root}",
                     lambda rec: self._message(f"wrote the commands to {rec.get('out') or out}", GREEN))
        return True

    def _export_dialog(self) -> None:
        st = self._state
        if st is None:
            return
        suffix = "ps1" if os.name == "nt" else "sh"
        path, _ = QFileDialog.getSaveFileName(self, "Commands file", str(st.root / f"run_{st.root.name}.{suffix}"),
                                              "PowerShell (*.ps1);;sh (*.sh);;All files (*)")
        if path:
            self.export_commands(path, shell="ps1" if path.lower().endswith(".ps1") else "sh")

    def remove_stale_lock(self) -> bool:
        """``germquant project unlock`` with the holder's host and pid as the status shows them."""
        st = self._state
        lock = (st.status or {}).get("lock") if st is not None else None
        if not isinstance(lock, dict) or not lock.get("held"):
            return False
        root, host, pid = st.root, str(lock.get("host")), int(lock.get("pid"))

        def job(should_stop):
            runner = self.session.require("projects", should_stop=should_stop)
            rec = runner.project_unlock(root, host=host, pid=pid, should_stop=should_stop)
            return rec, self._load(root, should_stop)

        def done(res):
            rec, state = res
            self._opened(state, fresh=False)
            self._done(f"removed the lock of {host} (pid {pid})" if rec.get("removed") else
                       "the project was not locked any more", state)

        self._submit("unlock", job, f"removing the lock of {host} pid {pid}", done)
        return True

    # ------------------------------------------------------------------ results
    def _on_board_double_click(self, row: int, _column: int) -> None:
        st = self._state
        if st is not None and 0 <= row < len(st.rows):
            self.open_results(st.rows[row].image_id)

    def open_results(self, image_id: str) -> list:
        """The image's results as layers, with its source image on the run's grid (the Results dock's loader)."""
        from ._results import ResultsSet, load_results_layers, summary_line

        st = self._state
        if st is None:
            return []
        row = next((r for r in st.rows if r.image_id == image_id), None)
        folder = Path(row.out_dir) if row is not None and row.out_dir else st.root / "results" / image_id
        rs = ResultsSet(folder, image_id)
        if not rs.has("image_summary.csv"):
            self._message(f"{image_id} has no results yet ({folder})", AMBER)
            return []
        try:
            layers = load_results_layers(rs, include_source=True)
            added = add_layer_tuples(self.viewer, layers)
        except Exception as e:  # noqa: BLE001 - the results could not be loaded: say so, keep the dock alive
            self._message(f"the results of {image_id} could not be opened: {_error_text(e)}", RED)
            return []
        self._message(f"opened {len(added)} layer(s) of {image_id}: {summary_line(rs)}", GREEN)
        set_viewer_status(self.viewer, f"germquant: {image_id} results opened")
        return added

    # ------------------------------------------------------------------ what is shown
    @property
    def state(self) -> ProjectState | None:
        return self._state

    @property
    def root(self) -> Path | None:
        return self._state.root if self._state is not None else None

    @property
    def manifest(self) -> dict | None:
        return self._state.manifest if self._state is not None else None

    @property
    def status(self) -> dict | None:
        return self._state.status if self._state is not None else None

    @property
    def rows(self) -> list:
        return list(self._state.rows) if self._state is not None else []

    @property
    def headers(self) -> dict:
        return dict(self._state.headers) if self._state is not None else {}

    def tab(self, title: str) -> QWidget | None:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == title:
                return self.tabs.widget(i)
        return None

    def _message(self, text: str, color: QColor | None) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(css(color))

    def _done(self, text: str, state: ProjectState, color: QColor | None = GREEN) -> None:
        """A write's message, naming a status the engine could not give (the board then shows unknown)."""
        if state.status_error is not None:
            reason = _error_text(state.status_error).strip().splitlines()
            text += f"; the engine gave no status, so the board shows the manifest's images: {reason[0] if reason else ''}"
            color = AMBER
        self._message(text, color)

    def _show_project(self) -> None:
        st = self._state
        self._show_engine()
        if st is None:
            for label in (self.name_label, self.folder_label, self.pack_label, self.hash_label, self.badge_label,
                          self.warnings_label, self.lock_label):
                label.setText("")
            self.badge_label.setToolTip("")
            self._fill_recent()
            self._show_images()
            self._show_samples()
            self._show_stages()
            self._show_interpreter()
            self._show_board()
            self._show_runs()
            self._update_buttons()
            return
        m = st.manifest
        self.name_label.setText(str(m.get("name") or st.root.name))
        self.folder_label.setText(str(st.root))
        pack_text = f"{st.pack_ref}, recipe {st.recipe_id}"
        pack = (st.status or {}).get("pack")
        if isinstance(pack, dict):
            installed = ", ".join(str(v) for v in pack.get("installed") or []) or "none"
            copy = "present" if pack.get("project_copy") else "MISSING"
            pack_text += f" (the engine has installed: {installed}; the project's copy: {copy})"
        self.pack_label.setText(pack_text)
        expected = (m.get("recipe") or {}).get("expected_config_hash")
        h = st.config_hash
        text = str(h)
        if st.status is not None and expected and h != expected:
            text += f" (the project was saved with {expected})"
        self.hash_label.setText(text)
        words, level = P.badge(st.status, m, root=st.root)
        self.badge_label.setText(words)
        self.badge_label.setStyleSheet(css(LEVELS[level]))
        why = ((st.status or {}).get("validated") or m.get("validated") or {}).get("why")
        self.badge_label.setToolTip(str(why or ""))
        lock = P.lock_text((st.status or {}).get("lock"))
        self.lock_label.setText(lock or "")
        self.lock_label.setStyleSheet(css(AMBER if lock else None))
        self._show_notes()
        self._show_images()
        self._show_samples()
        self._show_stages()
        self._show_interpreter()
        self._show_board()
        self._show_runs()
        self._update_buttons()

    def _show_notes(self) -> None:
        st = self._state
        if st is None:
            self.warnings_label.setText("")
            return
        here = self._this_computer()
        notes = [str(w) for w in (st.status or {}).get("warnings") or []]
        pack = (st.status or {}).get("pack")
        if isinstance(pack, dict) and pack.get("project_copy") is False:
            notes.append(f"the project's copy of its pack {pack.get('pinned') or st.pack_ref} is missing "
                         f"(packs/{st.pack_id}/): the engine resolves the pinned version elsewhere, or not at all")
        notes += P.setting_notes(st.manifest, device=here["device"], runner=here["runner"],
                                 interpreter=(st.recipe_id, here["python"]))
        if st.own_hash and st.status is not None and st.own_hash != st.status.get("config_hash"):
            notes.append(f"this napari environment's germquant computes config_hash {st.own_hash} for the project, the "
                         f"engine {st.status.get('config_hash')}: the two are different germquant checkouts (update "
                         f"both to the same commit)")
        for model in (st.status or {}).get("models") or []:
            if isinstance(model, dict) and model.get("mismatch"):
                notes.append(f"model {model.get('ref')}: {model.get('mismatch')}")
        self.warnings_label.setText("\n".join(dict.fromkeys(notes)))

    def _show_interpreter(self) -> None:
        st = self._state
        if st is None:
            self.interpreter_label.setText("")
            return
        here = self._this_computer()
        own = _settings.recipe_interpreter(st.pack_id, st.recipe_id)
        where = here["python"] or "no engine chosen yet"
        self.interpreter_label.setText(f"{where} ({'the recipe' + chr(39) + 's own interpreter' if own else 'the engine'}"
                                       f", {here['runner'] or 'unknown'} runner)")

    def _show_images(self) -> None:
        st = self._state
        entries = [e for e in (st.manifest.get("images") or []) if isinstance(e, dict)] if st is not None else []
        self.images_table.setRowCount(len(entries))
        for i, e in enumerate(entries):
            iid = str(e.get("id"))
            header = st.headers.get(iid)
            cells = [iid, str(e.get("path") or ""), _size_text(e.get("size")), P.header_text(header),
                     f"excluded: {e.get('reason')}" if e.get("excluded") else ""]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if j == 3 and header is not None and header.get("error"):
                    item.setForeground(QBrush(AMBER if header.get("error") == P.UNREACHABLE else RED))
                self.images_table.setItem(i, j, item)
        self.images_table.resizeColumnToContents(0)

    def _show_samples(self) -> None:
        st = self._state
        t = st.samples if st is not None else None
        self._filling_samples = True
        try:
            if t is None:
                self.samples_table.setColumnCount(0)
                self.samples_table.setRowCount(0)
                note = "" if st is None else f"no sample table: {st.samples_error or 'the project lists no images'}"
                self.samples_note.setText(note)
                return
            self.samples_table.setColumnCount(len(t.columns))
            self.samples_table.setHorizontalHeaderLabels(t.columns)
            self.samples_table.setRowCount(len(t.ids))
            for i, iid in enumerate(t.ids):
                row = t.row(iid)
                for j, c in enumerate(t.columns):
                    item = QTableWidgetItem(str(row.get(c, "")))
                    if c == "image_id":
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.samples_table.setItem(i, j, item)
            notes = list(t.warnings)
            if P.sheet_module() is None:
                notes.append(_feature_text("sample_sheet", None) + " The table can be edited; Save needs it.")
            missing = t.not_in_sheet()
            if t.saved is None:
                notes.append("no sample sheet yet: the rows are prefilled from the file names where the recipe's rule "
                             "matches; Save sheet writes samples.csv")
            elif missing:
                notes.append("not in the sheet (the run falls back to the file name): " + ", ".join(missing))
            if t.dirty:
                notes.append("unsaved sample edits: saving restales " + (", ".join(t.changed_ids()) or "nothing"))
            self.samples_note.setText("\n".join(notes))
        finally:
            self._filling_samples = False

    def _show_stages(self) -> None:
        st = self._state
        rows = {}
        if st is not None and st.config is not None:
            from . import _engine

            names = next((list(h.get("channel_names") or []) for h in st.headers.values()
                          if isinstance(h, dict) and h.get("channel_names")), None)
            try:
                if names is not None:
                    role_to_idx = _engine.role_to_index(_engine.role_rows(st.config, names))
                else:
                    role_to_idx = {role: 0 for role in st.config.channel_map.roles}
                rows = {r.name: r for r in _engine.stage_rows(st.config, role_to_idx)}
            except Exception:  # noqa: BLE001 - a channel map that does not resolve: the switches stay plain
                rows = {}
        for name, cb in self.stage_checks.items():
            r = rows.get(name)
            reason = r.reason if r is not None and not r.selectable else ""
            cb.setText(f"{name}   ({reason})" if reason else name)
            cb.setProperty("selectable", r is None or r.selectable)
        if st is not None and st.config is None and st.config_error:
            self.stages_box.setToolTip("the recipe does not load in this napari environment, so the switches cannot be "
                                       f"checked against it: {st.config_error}")
        else:
            self.stages_box.setToolTip("")

    def _show_board(self) -> None:
        st = self._state
        rows = st.rows if st is not None else []
        from ._packs import object_words

        self.board.setHorizontalHeaderItem(3, QTableWidgetItem(object_words(st.kind if st else "nucleus")[1]))
        self.board.setRowCount(len(rows))
        for i, r in enumerate(rows):
            qc = "" if r.qc_pass is None else str(bool(r.qc_pass))
            cells = [r.image_id, r.status_text, qc, "" if r.objects is None else str(r.objects),
                     "" if r.spots is None else str(r.spots), r.last_run_id or "", r.device_used or ""]
            color = STATUS_COLORS.get(r.status)
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if j == 1 and color is not None:
                    item.setForeground(QBrush(color))
                self.board.setItem(i, j, item)

    def _show_runs(self) -> None:
        st = self._state
        runs = [r for r in ((st.status or {}).get("runs") or []) if isinstance(r, dict)] if st is not None else []
        runs = runs[-10:][::-1]
        self.runs_table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            counts = P.counts_text(r.get("counts")) + (" (interrupted)" if r.get("interrupted") else "")
            for j, text in enumerate([r.get("run_id"), r.get("via"), r.get("started"), r.get("ended"), counts]):
                self.runs_table.setItem(i, j, QTableWidgetItem("" if text is None else str(text)))

    def _update_buttons(self) -> None:
        st = self._state
        has = st is not None
        running = self.running
        engine_ok = has and st.status is not None
        unsaved = self.unsaved
        self.save_btn.setEnabled(has and not running)
        self.save_btn.setText("Save (unsaved changes)" if unsaved else "Save")
        self.close_btn.setEnabled(has and not running)
        self.new_btn.setEnabled(not running)
        self.open_btn.setEnabled(not running)
        self.recent_combo.setEnabled(not running)
        for b in (self.run_all_btn, self.run_selected_btn, self.rerun_btn, self.collect_btn, self.export_btn):
            b.setEnabled(engine_ok and not running)
        self.cancel_btn.setEnabled(running)
        for b in (self.add_files_btn, self.add_folder_btn, self.paste_btn, self.paste_edit):
            b.setEnabled(has and not running)
        for cb in self.stage_checks.values():
            cb.setEnabled(has and not running and cb.property("selectable") is not False)
        self.stride_spin.setEnabled(has and not running)
        self.interpreter_btn.setEnabled(has and not running)
        self.samples_table.setEnabled(has and not running)
        for b in (self.add_column_btn, self.save_sheet_btn, self.column_edit):
            b.setEnabled(has and not running and st.samples is not None)
        lock = (st.status or {}).get("lock") if has else None
        self.unlock_btn.setEnabled(bool(isinstance(lock, dict) and lock.get("held") and lock.get("stale"))
                                   and not running)


def _feature_text(feature: str, runner) -> str:
    """The one message for a feature this napari environment's germquant (or ``runner``) lacks."""
    from . import _engine

    if runner is not None and hasattr(runner, "feature_message"):
        return runner.feature_message(feature)
    from ._runner import plugin_engine_info

    return _engine.feature_missing_message(feature, info=plugin_engine_info())
