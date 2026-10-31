"""The Spot tuning dock (docs/DESIGN_system_packs.md 8.7; tranche 5 Train B plan, unit P6-TUNE, decisions D24,
D38, D39 and D42; plugin_specs.md 4.5): click truth points on a project's finished runs, tune the recipe's spot
settings on them with the engine, and apply the best setting to the project.

What it shows, top to bottom:

* Project: the engine line (the viewer's engine session, shared with the Project dock), the project folder, its
  pack recipe and config_hash, and Open, Refresh and Close. The dock works on the project the Project dock has
  open (it follows that dock's hub: opened, closed, every new reading of it, and whether a run of it goes), and
  without a Project dock a project folder can be opened here.
* Truth points: the annotator (letters, digits, dash or underscore, kept per user), the project's images (each
  with its run: finished, stale or not run yet, and its truth points), Open image, "placed on a projection" and
  Save points. Open image adds the image's finished run on its own grid: the run's labels and spots, the source
  image read at the run's stride and on its planes, and one editable Points layer "<image_id> truth points
  (<annotator>)" in add mode, holding the points of annotations/<image_id>__points_<annotator>.csv when the file
  exists (for another annotator of an image whose run's layers this dock opened, only the truth layer is added,
  while the image's run is the same; the layers of an earlier run are renamed "(old run)"). The box is kept for
  each truth layer, and a changed box counts as unsaved. Save writes that file through the engine's own writer,
  in napari's process (D39); it is off while a search runs, and it refuses a file that would come out wrong
  (points of other planes in a run of one plane, or a run made again on another grid since the layer was opened:
  the layer is then renamed "(old run)" and Open image opens the new run).
* Tune: the grid (the engine's default for the recipe's spot backend, or a grid file), Max runs, how many
  detections the search means on the annotated images, Tune and Cancel. Tune runs ``germquant tune-spots
  --project DIR --points annotations --max-runs N [--grid FILE] --json`` through the engine runner, in a thread
  worker, with the engine's lines on the status bar. Cancel ends the engine's process (an engine in napari's own
  process cannot be stopped, so Cancel is off for it).
* Result: the project's tuning runs (newest first), the top five rows of the selected run's ranked.csv (the
  columns the engine writes, the objective named), its best setting and warnings, and Apply, which runs
  ``germquant tune-spots --project DIR --apply tuning/<run>`` through the engine with the plugin version (D42).
  The project's config_hash then changes, the message says whether it reads validated (the manifest after the
  write) and shows the engine's warnings of the Apply in amber, its images turn stale, and the Project dock reads
  it again.

Tune and Apply wait while a run of the project goes, and Tune while truth points are not saved. The GUI thread
never waits: the engine's calls go to one thread worker, the file reads and the Save to another. An image opening,
a search or an Apply belongs to the project folder it was started on: when it returns while the dock has no
project open, or another one, it changes nothing in the dock; a reading of the same project (Refresh, a Save's,
the Project dock's) drops none of them. Every control has a programmatic twin (set_project, set_annotator,
select_image, open_image, save_points, set_projected, set_grid_file, set_max_runs, tune, cancel, select_run,
apply) that opens no dialog.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _settings, _tuning
from ._engine_session import NOT_CHOSEN, EngineNotChosen, EngineSession, session_for
from ._qt_common import WorkerRunner, add_layer_tuples, set_viewer_status
from ._runner import FeatureMissing, RunCancelled, RunnerError, detect_interpreters, plugin_version

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)
TUNE_LABEL = "tuning the spots (germquant tune-spots)"     # the engine worker's label while a search runs


def css(color: QColor | None) -> str:
    return "" if color is None else f"color: rgb({color.red()}, {color.green()}, {color.blue()});"


def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, (RunnerError, _tuning.TuningError)):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _viewer_hub(viewer):
    """The Project dock's hub of ``viewer`` (napari_germquant._project_widget.project_hub, unit P6-PROJECT), or None
    without a viewer."""
    if viewer is None:
        return None
    from ._project_widget import project_hub

    return project_hub(viewer)


class _Lines(QObject):
    """The engine's lines from the worker thread to the GUI thread (a signal emitted elsewhere is queued to the
    thread this object lives in)."""

    line = Signal(str)


class SpotTuningWidget(QWidget):
    """napari dock widget (module docstring). ``napari_viewer`` is injected by napari; None builds the dock without
    a viewer (nothing can be opened as layers then). ``session`` replaces the viewer's engine session and ``hub``
    the Project dock's hub (tests)."""

    def __init__(self, napari_viewer=None, parent=None, *, session: EngineSession | None = None, hub=None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session = session if session is not None else session_for(napari_viewer)
        self.hub = hub if hub is not None else _viewer_hub(napari_viewer)
        self.root: Path | None = None
        self.manifest: dict | None = None
        self.rows: list[_tuning.ImageRow] = []
        self.runs: list[str] = []
        self.views: dict[str, dict] = {}
        self.current_run: str | None = None
        self.backend: str | None = None
        self.backend_source = ""
        self.grid_file: Path | None = None
        self._grid: dict | None = None
        self._grid_error: str | None = None
        self.last_message = ""
        self.last_error: BaseException | None = None
        self.last_tune: dict | None = None
        self.last_apply: dict | None = None
        self.progress_text = ""
        self.engine_error: str | None = None
        self.interpreter_dialog = None
        self._hub_root: Path | None = None             # the project the Project dock opened, which this dock follows
        # one more at every reading of a project asked for (and at Close): only the newest reading is taken. A job of
        # the project (an image opening, a search, an Apply) is not a reading: it keeps its project's folder and is
        # dropped only when the dock no longer has that project open (``_owns``), so a refresh never drops one.
        self._reading = 0
        self._truth: dict[tuple, object] = {}           # (root, image id, annotator) -> its truth layer
        self._saved: dict[tuple, np.ndarray] = {}       # the same key -> its points as last read or saved
        # (root, image id) -> {"stamp": the run's stamp, "source": [layers], "results": [layers]}: the run's layers
        # this dock added, shown once for all the annotators of the image while its run is the same
        self._run_layers: dict[tuple, dict] = {}
        self._io_jobs: list[dict] = []
        self._lines = _Lines(self)
        self._lines.line.connect(self._on_line)
        self.engine_worker = WorkerRunner(napari_viewer, self)     # tune-spots, its apply, the engine check
        self.io_worker = WorkerRunner(None, self)                  # the project's files, the image's layers, Save
        self._build()
        try:
            self.session.changed.connect(self._show_engine)
        except (RuntimeError, AttributeError):
            pass
        if self.hub is not None:
            self.hub.opened.connect(self._on_hub_opened)
            self.hub.closed.connect(self._on_hub_closed)
            self.hub.changed.connect(self._on_hub_changed)
            self.hub.running_changed.connect(self._on_hub_running)
        self._show_project()
        self._check_engine()
        if self.hub is not None and getattr(self.hub, "root", None) is not None:
            self._on_hub_opened(str(self.hub.root))

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        head = QGroupBox("Project")
        hform = QFormLayout(head)
        erow = QHBoxLayout()
        self.engine_label = QLabel("")
        self.engine_label.setWordWrap(True)
        self.engine_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        erow.addWidget(self.engine_label, 1)
        self.engine_btn = QPushButton("Choose engine...")
        self.engine_btn.setToolTip("the Python of the germquant environment that makes the numbers (the Run and "
                                   "Project docks use the same choice)")
        self.engine_btn.clicked.connect(self.ask_for_engine)
        erow.addWidget(self.engine_btn)
        hform.addRow("Engine", erow)
        self.project_label = QLabel("")
        self.project_label.setWordWrap(True)
        self.project_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hform.addRow("Folder", self.project_label)
        prow = QHBoxLayout()
        self.open_project_btn = QPushButton("Open project...")
        self.open_project_btn.clicked.connect(self.open_dialog)
        prow.addWidget(self.open_project_btn)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("read the project's images, runs, truth points and tuning runs again")
        self.refresh_btn.clicked.connect(self.refresh)
        prow.addWidget(self.refresh_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_project)
        prow.addWidget(self.close_btn)
        prow.addStretch(1)
        hform.addRow("", prow)
        self.notice_label = QLabel("")
        self.notice_label.setWordWrap(True)
        self.notice_label.setStyleSheet(css(AMBER))
        self.notice_label.setVisible(False)
        hform.addRow("", self.notice_label)
        outer.addWidget(head)

        pts = QGroupBox("Truth points")
        pform = QFormLayout(pts)
        self.annotator_edit = QLineEdit(_settings.annotator() or "")
        self.annotator_edit.setPlaceholderText("your initials: letters, digits, - or _ (at most 32)")
        self.annotator_edit.setToolTip("names your truth-point files, annotations/<image_id>__points_<annotator>.csv; "
                                       "kept for this user")
        self.annotator_edit.editingFinished.connect(self._annotator_edited)
        pform.addRow("Annotator", self.annotator_edit)
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_list.currentRowChanged.connect(lambda _row: self._show_points(sync_box=True))
        self.image_list.itemDoubleClicked.connect(lambda _item: self.open_image())
        pform.addRow("Images", self.image_list)
        irow = QHBoxLayout()
        self.open_btn = QPushButton("Open image")
        self.open_btn.setToolTip("the image's finished run on its own grid, its source image and your truth-point "
                                 "layer")
        self.open_btn.clicked.connect(lambda: self.open_image())
        irow.addWidget(self.open_btn)
        self.projected_check = QCheckBox("placed on a projection (matched in 2D)")
        self.projected_check.setToolTip("kept for each truth layer; Save writes it for every point of the layer")
        self.projected_check.toggled.connect(self._on_box)
        irow.addWidget(self.projected_check)
        self.save_btn = QPushButton("Save points")
        self.save_btn.setToolTip("write annotations/<image_id>__points_<annotator>.csv through the engine's writer")
        self.save_btn.clicked.connect(lambda: self.save_points())
        irow.addWidget(self.save_btn)
        pform.addRow("", irow)
        self.points_label = QLabel("")
        self.points_label.setWordWrap(True)
        pform.addRow("", self.points_label)
        outer.addWidget(pts)

        tune = QGroupBox("Tune")
        tform = QFormLayout(tune)
        self.grid_label = QLabel("")
        self.grid_label.setWordWrap(True)
        tform.addRow("Grid", self.grid_label)
        grow = QHBoxLayout()
        self.grid_btn = QPushButton("Grid file...")
        self.grid_btn.setToolTip("a YAML file of dotted spots keys and the values to try (docs/TUNING.md)")
        self.grid_btn.clicked.connect(self.grid_dialog)
        grow.addWidget(self.grid_btn)
        self.default_grid_btn = QPushButton("Default grid")
        self.default_grid_btn.clicked.connect(lambda: self.set_grid_file(None))
        grow.addWidget(self.default_grid_btn)
        grow.addStretch(1)
        tform.addRow("", grow)
        self.max_runs_spin = QSpinBox()
        self.max_runs_spin.setRange(1, 1_000_000)
        self.max_runs_spin.setValue(_tuning.MAX_RUNS)
        self.max_runs_spin.setToolTip("the most detections the search may make over all images (--max-runs)")
        self.max_runs_spin.valueChanged.connect(lambda _v: self._show_count())
        tform.addRow("Max runs", self.max_runs_spin)
        self.count_label = QLabel("")
        self.count_label.setWordWrap(True)
        tform.addRow("", self.count_label)
        trow = QHBoxLayout()
        self.tune_btn = QPushButton("Tune")
        self.tune_btn.setToolTip("germquant tune-spots on the saved truth points, in the engine's environment")
        self.tune_btn.clicked.connect(lambda: self.tune())
        trow.addWidget(self.tune_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(lambda: self.cancel())
        trow.addWidget(self.cancel_btn)
        trow.addStretch(1)
        tform.addRow("", trow)
        outer.addWidget(tune)

        res = QGroupBox("Result")
        rlay = QVBoxLayout(res)
        self.runs_combo = QComboBox()
        self.runs_combo.currentIndexChanged.connect(self._on_run_chosen)
        rlay.addWidget(self.runs_combo)
        self.objective_label = QLabel("")
        self.objective_label.setWordWrap(True)
        rlay.addWidget(self.objective_label)
        self.ranked_table = QTableWidget(0, 0)
        self.ranked_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ranked_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        rlay.addWidget(self.ranked_table)
        self.best_label = QLabel("")
        self.best_label.setWordWrap(True)
        self.best_label.setStyleSheet("font-weight: bold;")
        rlay.addWidget(self.best_label)
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet(css(AMBER))
        rlay.addWidget(self.warnings_label)
        self.apply_btn = QPushButton("Apply to the project")
        self.apply_btn.setToolTip("germquant tune-spots --apply: write the run's best setting into recipe.yaml (the "
                                  "config_hash changes and the images turn stale)")
        self.apply_btn.clicked.connect(lambda: self.apply())
        rlay.addWidget(self.apply_btn)
        outer.addWidget(res)

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.message_label)
        outer.addStretch(1)

    # ------------------------------------------------------------------ state
    @property
    def idle(self) -> bool:
        """No engine call, file job or waiting file job (tests wait on it)."""
        return not self.engine_worker.busy and not self.io_worker.busy and not self._io_jobs

    @property
    def searching(self) -> bool:
        """Whether a search (tune-spots) runs in the engine worker."""
        return self.engine_worker.busy and self.engine_worker.label == TUNE_LABEL

    def _recipe(self) -> tuple[str | None, str | None]:
        """(pack id, recipe id) of the open project's pinned pack recipe, or (None, None)."""
        pin = (self.manifest or {}).get("pack") or {}
        return pin.get("id"), pin.get("recipe")

    def _cancellable(self) -> bool:
        """Whether the engine can stop a search (a subprocess engine; the in-process engine cannot): the recipe's own
        interpreter on this computer when one is stored (the search runs there), else the engine."""
        mapped = _settings.recipe_interpreter(*self._recipe()) if self.manifest else None
        if mapped:
            return mapped != _settings.INPROCESS
        try:
            return bool(getattr(self.session.runner(), "cancellable", False))
        except Exception:  # noqa: BLE001 - an engine that cannot be asked cannot be stopped either
            return False

    def _message(self, text: str, color: QColor | None = None) -> None:
        self.last_message = text
        self.message_label.setText(text)
        self.message_label.setStyleSheet(css(color))

    def _failed(self, exc: BaseException, what: str = "") -> None:
        self.last_error = exc
        if isinstance(exc, RunCancelled):
            self._message(f"cancelled: {exc}", AMBER)
        elif isinstance(exc, FeatureMissing):
            self._message(str(exc), RED)
        else:
            self._message(f"{what}{_error_text(exc)}", RED)

    def _annotator(self) -> str | None:
        """The annotator the dock works as: the name in the box when it is a valid one."""
        text = self.annotator_edit.text().strip()
        return text if text and _settings.ANNOTATOR_RE.fullmatch(text) else None

    def _running(self) -> bool:
        """Whether a run of the open project goes (the Project dock's registry, or its hub)."""
        if self.root is None:
            return False
        if _tuning.run_in_progress(self.root):
            return True
        hub = self.hub
        return bool(hub is not None and getattr(hub, "running", False) and _same(getattr(hub, "root", None),
                                                                                 self.root))

    def _row(self, image_id: str | None = None) -> _tuning.ImageRow | None:
        if image_id is None:
            item = self.image_list.currentItem()
            image_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return next((r for r in self.rows if r.image_id == image_id), None)

    def row(self, image_id: str) -> _tuning.ImageRow | None:
        return self._row(image_id)

    def annotated_images(self) -> list[str]:
        """The images a search takes: a finished run and a truth-point file in annotations/."""
        return _tuning.annotated(self.rows)

    def _key(self, image_id: str, annotator: str | None = None) -> tuple:
        return (os.path.normcase(str(self.root)), image_id, annotator or self._annotator())

    def truth_layer(self, image_id: str | None = None):
        """The open truth-point layer of ``image_id`` (the selected image when None) for the annotator, or None."""
        row = self._row(image_id)
        if row is None or self.viewer is None:
            return None
        layer = self._truth.get(self._key(row.image_id))
        try:
            return layer if layer is not None and layer in self.viewer.layers else None
        except Exception:  # noqa: BLE001 - a viewer that is closing
            return None

    @staticmethod
    def _box(layer) -> bool:
        """The "placed on a projection" box of a truth layer, as the user left it (else as its file holds it)."""
        meta = layer.metadata.get(_tuning.LAYER_KEY) or {}
        return bool(meta.get("box", meta.get("projected")))

    @staticmethod
    def _box_changed(layer) -> bool:
        """The layer holds points and its box differs from what its file holds."""
        meta = layer.metadata.get(_tuning.LAYER_KEY) or {}
        return len(layer.data) > 0 and bool(meta.get("box", meta.get("projected"))) != bool(meta.get("projected"))

    def _points_unsaved(self, key, layer) -> bool:
        saved = self._saved.get(key)
        data = np.asarray(layer.data, float)
        if saved is None:
            return len(data) > 0
        return data.shape != saved.shape or not np.allclose(data, saved)

    def _unsaved(self, key, layer) -> bool:
        """The layer's points or its box are not what its file holds."""
        return self._points_unsaved(key, layer) or self._box_changed(layer)

    def unsaved_images(self) -> list[str]:
        """The images of the open project whose truth layer holds points, or a "placed on a projection" box, that
        are not in its file."""
        out = []
        if self.viewer is None or self.root is None:
            return out
        for key, layer in list(self._truth.items()):
            if key[0] != os.path.normcase(str(self.root)):
                continue
            try:
                present = layer in self.viewer.layers
            except Exception:  # noqa: BLE001
                present = False
            if present and self._unsaved(key, layer):
                out.append(key[1])
        return sorted(set(out))

    # ------------------------------------------------------------------ workers
    def _io(self, kind: str, job, label: str, returned, errored, *, replace: bool = False) -> None:
        """A file job in the file worker, after the ones waiting (``replace``: a newer job of ``kind`` replaces a
        waiting one)."""
        if replace:
            self._io_jobs = [j for j in self._io_jobs if j["kind"] != kind]
        self._io_jobs.append({"kind": kind, "job": job, "label": label, "returned": returned, "errored": errored})
        self._pump_io()

    def _pump_io(self) -> None:
        if self.io_worker.busy or not self._io_jobs:
            return
        j = self._io_jobs.pop(0)
        worker = self.io_worker
        self.io_worker.start(lambda: j["job"](worker.should_stop), label=j["label"], returned=j["returned"],
                             errored=j["errored"], finished=self._io_finished)

    def _io_finished(self) -> None:
        self._update()
        self._pump_io()

    def _engine_job(self, job, label: str, returned, errored) -> bool:
        worker = self.engine_worker
        ok = worker.start(lambda: job(worker.should_stop), label=label, returned=returned, errored=errored,
                          finished=self._engine_finished, line_format=lambda line: line[:160])
        self._update()
        return ok

    def _engine_finished(self) -> None:
        self._update()

    def _on_line(self, line: str) -> None:
        self.progress_text = line
        set_viewer_status(self.viewer, f"spot tuning: {line[:160]}")

    # ------------------------------------------------------------------ the engine
    def _check_engine(self) -> None:
        def job(should_stop):
            return self.session.ensure(should_stop)

        def done(_runner):
            self.engine_error = None
            self._show_engine()

        def failed(exc):
            self.engine_error = None if isinstance(exc, EngineNotChosen) else _error_text(exc)
            self._show_engine()

        self._engine_job(job, "checking the engine", done, failed)

    def set_engine(self, choice) -> None:
        """Use ``choice`` (a Python path, or ``inprocess``) as this computer's engine once its check passes (the
        choice the Run and Project docks read too)."""
        choice = str(choice)

        def job(should_stop):
            return self.session.set_engine(choice, should_stop=should_stop)

        def done(_runner):
            self.engine_error = None
            self._show_engine()

        def failed(exc):
            self.engine_error = _error_text(exc)
            self._show_engine()
            self._message(f"that engine cannot be used: {self.engine_error}", RED)

        if not self._engine_job(job, f"checking the engine {choice}", done, failed):
            self._message("the engine is busy; choose it again when it has finished", AMBER)

    def ask_for_engine(self):
        from ._engine_widgets import InterpreterDialog
        from ._run_widget import ENGINE_INTRO

        dlg = InterpreterDialog(detect_interpreters(), title="germquant engine for this computer", intro=ENGINE_INTRO,
                                current=_settings.engine_choice(), allow_inprocess=True, parent=self)
        dlg.accepted.connect(lambda: self.set_engine(dlg.choice()) if dlg.choice() else None)
        self.interpreter_dialog = dlg
        dlg.open()
        return dlg

    def _show_engine(self) -> None:
        try:
            if self.engine_error:
                self.engine_label.setText(f"The engine chosen for this computer cannot be used: {self.engine_error}")
                self.engine_label.setStyleSheet(css(RED))
            elif self.session.runner() is None and _settings.engine_choice() is None:
                self.engine_label.setText(NOT_CHOSEN)
                self.engine_label.setStyleSheet(css(AMBER))
            else:
                text = self.session.describe()
                warn = self.session.warning() if self.session.runner() is not None else ""
                self.engine_label.setText(text + (f"\n{warn}" if warn else ""))
                self.engine_label.setStyleSheet("")
        except RuntimeError:                          # the widget is gone (napari closing)
            return
        self._show_notice()

    # ------------------------------------------------------------------ the project
    def open_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Project folder")
        if folder:
            self.set_project(folder)

    def set_project(self, root) -> None:
        """Open the project in ``root`` (read in the file worker): its images, runs, truth points and tuning
        runs. A folder that is no project is refused, and the project that was open stays open."""
        folder = Path(os.path.abspath(str(root).strip().strip('"')))
        self._reading += 1
        self._read(folder, self._reading)

    def refresh(self) -> None:
        """Read the open project again (an image being opened, a search and an Apply of it go on)."""
        if self.root is not None:
            self._reading += 1
            self._read(self.root, self._reading)

    def _read(self, folder: Path, reading: int) -> None:
        annotator, grid_file = self._annotator(), self.grid_file

        def job(should_stop):
            return _tuning.read_project(folder, annotator, grid_file)

        def done(state):
            if reading != self._reading:
                return                                # a newer reading (another project, or Close) came since
            self._take(state)

        def failed(exc):
            if reading != self._reading:
                return
            self._failed(exc, f"{folder}: ")

        self._io("project", job, f"reading the project {folder}", done, failed, replace=True)

    def _take(self, state: dict) -> None:
        changed = not _same(self.root, state["root"])
        if changed:
            _tuning.forget_labels()                   # the label image kept for the last project's Saves
        self.root = Path(state["root"])
        self.manifest = state["manifest"]
        self.rows = list(state["rows"])
        self.backend, self.backend_source = state["backend"], state["backend_source"]
        self._grid, self._grid_error = state["grid"], state["grid_error"]
        self.runs = list(state["runs"])
        self.views = dict(state["views"])
        if changed or self.current_run not in self.views:
            self.current_run = self.runs[0] if self.runs else None
        self._show_project()

    def close_project(self) -> None:
        """Close the project in this dock (its layers stay in the viewer); a reading of it that returns later is
        dropped, and so is an image opening, a search or an Apply of it while no project, or another one, is open
        here."""
        self._reading += 1
        self.root, self.manifest, self.rows, self.runs, self.views = None, None, [], [], {}
        self.current_run = None
        self.backend, self.backend_source = None, ""
        _tuning.forget_labels()
        self._show_project()

    def _owns(self, root) -> bool:
        """Whether the dock has the project in ``root`` open now: a job started on that project changes the dock
        only then (it was closed, or another project taken, meanwhile otherwise)."""
        return _same(self.root, root)

    def _gone(self) -> str:
        """Why a job of a project the dock no longer has open changes nothing now."""
        return ("was closed meanwhile" if self.root is None else
                f"is no longer the one open in this dock (it has {self.root} open now)")

    def _on_hub_opened(self, root: str) -> None:
        self._hub_root = Path(root)
        self.set_project(root)

    def _on_hub_closed(self) -> None:
        if self._hub_root is not None and _same(self._hub_root, self.root):
            self.close_project()
        self._hub_root = None

    def _on_hub_changed(self) -> None:
        if self.root is not None and _same(getattr(self.hub, "root", None), self.root):
            self.refresh()

    def _on_hub_running(self, _running: bool) -> None:
        self._update()

    def _show_project(self) -> None:
        if self.root is None or self.manifest is None:
            follows = " (this dock follows the Project dock: open a project there, or open its folder here)" \
                if self.hub is not None else " (open a project folder: Open project...)"
            self.project_label.setText("No project open" + follows)
        else:
            m = self.manifest
            self.project_label.setText(f"{self.root}\n{m.get('name')}: pack {_tuning.pack_text(m)}, config_hash "
                                       f"{_tuning.project_hash(m) or 'not recorded'}")
        current = self._row()
        keep = current.image_id if current is not None else None
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for r in self.rows:
            item = QListWidgetItem(r.text)
            item.setData(Qt.ItemDataRole.UserRole, r.image_id)
            if not r.usable or not r.finished:
                item.setForeground(AMBER)
            self.image_list.addItem(item)
        ids = [r.image_id for r in self.rows]
        if keep in ids:
            self.image_list.setCurrentRow(ids.index(keep))
        elif self.rows:
            first = next((i for i, r in enumerate(self.rows) if r.finished and r.usable), 0)
            self.image_list.setCurrentRow(first)
        self.image_list.blockSignals(False)
        now = self._row()
        moved = (now.image_id if now is not None else None) != keep
        self.runs_combo.blockSignals(True)
        self.runs_combo.clear()
        for run in self.runs:
            self.runs_combo.addItem(Path(run).name, run)
        if self.current_run in self.runs:
            self.runs_combo.setCurrentIndex(self.runs.index(self.current_run))
        self.runs_combo.blockSignals(False)
        self._show_grid()
        self._show_run()
        self._show_points(sync_box=moved)
        self._show_notice()

    def _show_notice(self) -> None:
        notes = []
        try:
            if self.root is not None and self.session.runner() is not None:
                missing = self.session.missing(_tuning.FEATURE)
                if missing:
                    notes.append(missing)
        except RuntimeError:
            return
        problem = _tuning.points_problem()
        if problem and self.root is not None:
            notes.append(f"Truth points cannot be saved here: {problem}")
        several = _tuning.several_annotators(self.rows)
        if several:
            notes.append(f"{', '.join(several)}: truth points of several annotators in annotations/; a search scores "
                         f"against one set of points, so the engine refuses them (keep each annotator's files in a "
                         f"folder of their own)")
        self.notice_label.setText("\n".join(notes))
        self.notice_label.setVisible(bool(notes))

    # ------------------------------------------------------------------ the annotator
    def set_annotator(self, name: str) -> None:
        """Work as ``name`` (letters, digits, dash and underscore, 1 to 32), kept for this user; ValueError for
        another name. The image list then shows that annotator's points."""
        _settings.set_annotator(name)
        self.annotator_edit.setText(name)
        self.refresh()

    def _annotator_edited(self) -> None:
        text = self.annotator_edit.text().strip()
        if not text or text == _settings.annotator():
            self._update()
            return
        try:
            self.set_annotator(text)
        except ValueError as e:
            self._message(str(e), RED)
            self._update()

    # ------------------------------------------------------------------ images and truth points
    def select_image(self, image_id: str) -> bool:
        ids = [r.image_id for r in self.rows]
        if image_id not in ids:
            return False
        self.image_list.setCurrentRow(ids.index(image_id))
        self._show_points()
        return True

    def open_image(self, image_id: str | None = None) -> bool:
        """Open the image (the selected one when None) on its finished run's grid with the annotator's truth
        layer (module docstring). True when the reading started, or the image is already open."""
        if image_id is not None and not self.select_image(image_id):
            self._message(f"{image_id} is not an image of this project", RED)
            return False
        row = self._row()
        if row is None:
            self._message("choose an image of the project first", AMBER)
            return False
        if not row.usable or not row.finished:
            self._message(row.text, AMBER)
            return False
        annotator = self._annotator()
        if annotator is None:
            self._message("give your annotator name first (letters, digits, - or _): it names your truth-point file",
                          AMBER)
            return False
        if self.viewer is None:
            self._message("this dock has no viewer to open the image in", RED)
            return False
        key = self._key(row.image_id, annotator)
        layer = self.truth_layer(row.image_id)
        if layer is not None:
            self._select_layer(layer)
            self._message(f"{row.image_id} is already open: its truth points are the layer {layer.name}", None)
            return True
        root = self.root
        rkey = (os.path.normcase(str(root)), row.image_id)
        # the run's layers this dock added for another annotator, still in the viewer: the job reads them again only
        # when the image's run is no longer the one they show (its stamp)
        entry = self._run_layers.get(rkey) or {}
        have = {part: entry.get("stamp") for part in ("source", "results") if self._in_viewer(entry.get(part))}

        def job(should_stop):
            info = _tuning.run_info(root, row)
            return _tuning.image_layers(info, annotator, have=have)

        def done(res):
            if not self._owns(root):
                self._message(f"{row.image_id} was not opened: its project {self._gone()}", AMBER)
                return
            notes = [res["note"]] if res["note"] else []
            known = self._run_layers.get(rkey)
            if known is not None and known.get("stamp") != res["stamp"]:
                # the image was run again since those layers were added: they show its earlier run
                if self._rename_old([*known.get("source", []), *known.get("results", [])]):
                    notes.append(f"the layers of an earlier run of {row.image_id} are renamed '(old run)'")
                known = None
            if known is None:
                known = self._run_layers[rkey] = {"stamp": res["stamp"], "source": [], "results": []}
            src = add_layer_tuples(self.viewer, res["source"])
            out = add_layer_tuples(self.viewer, res["results"])
            (truth,) = add_layer_tuples(self.viewer, [res["truth"]])
            if res["read_source"]:
                known["source"] = src
            if res["read_results"]:
                known["results"] = out
            truth.mode = "add"
            self._truth[key] = truth
            self._saved[key] = np.asarray(res["data"], float).reshape(-1, 3)
            truth.events.data.connect(self._on_truth_changed)
            self._select_layer(truth)
            if res["read_source"] or res["read_results"]:
                text = f"opened {row.image_id} on its run's grid ({len(src) + len(out) + 1} layers)"
            else:
                text = f"opened the truth points of {row.image_id} ({annotator}) on its run's layers in the viewer"
            note = "; ".join(notes)
            self._message(text + (f"; {note}" if note else ""), AMBER if note else None)
            self._show_points(sync_box=True)

        self._io("open", job, f"opening {row.image_id}", done, lambda e: self._failed(e, f"{row.image_id}: "))
        return True

    def _in_viewer(self, layers) -> bool:
        """Whether any of ``layers`` is still in the viewer."""
        try:
            return any(layer in self.viewer.layers for layer in layers or [])
        except Exception:  # noqa: BLE001 - a viewer that is closing
            return False

    def _select_layer(self, layer) -> None:
        try:
            self.viewer.layers.selection.active = layer
        except Exception:  # noqa: BLE001 - selection is a convenience
            pass

    def _on_truth_changed(self, *_args) -> None:
        try:
            self._show_points()
        except RuntimeError:                          # the dock is gone, its layer is not
            pass

    def set_projected(self, flag: bool) -> None:
        """Whether the points of the selected image were placed on a projection (matched in 2D when tuning); kept
        for that image's truth layer until Save writes it."""
        self.projected_check.setChecked(bool(flag))
        self._on_box(bool(flag))                      # also when the box showed it already (no toggled signal)

    def _on_box(self, on: bool) -> None:
        """The box changed by the user (or set_projected): it belongs to the selected image's truth layer."""
        row = self._row()
        layer = self.truth_layer(row.image_id) if row is not None else None
        if layer is not None:
            layer.metadata[_tuning.LAYER_KEY]["box"] = bool(on)
        self._show_points()

    def save_points(self, image_id: str | None = None) -> bool:
        """Write the truth layer of the image (the selected one when None) to its file through the engine's
        writer, in the file worker. True when the write started. Not while a search runs: it read the saved files
        when it started, and its result is what Apply offers."""
        if image_id is not None and not self.select_image(image_id):
            self._message(f"{image_id} is not an image of this project", RED)
            return False
        row = self._row()
        layer = self.truth_layer(row.image_id) if row is not None else None
        if layer is None:
            self._message("open the image first: its truth-point layer is what Save writes", AMBER)
            return False
        problem = _tuning.points_problem()
        if problem:
            self._message(problem, RED)
            return False
        if self.searching:
            self._message("a search is reading the saved truth points: save when it has finished (the points stay "
                          "in the layer)", AMBER)
            return False
        meta = dict(layer.metadata[_tuning.LAYER_KEY])
        if meta.get("other_planes"):
            self._message(_tuning.other_planes_text(meta["image_id"], int(meta["other_planes"]),
                                                    Path(meta["file"]).name), AMBER)
            return False
        key = (os.path.normcase(meta["root"]), meta["image_id"], meta["annotator"])
        data = np.array(layer.data, float).reshape(-1, 3)
        projected = self._box(layer)

        def job(should_stop):
            return _tuning.save_truth(meta, data, projected=projected)

        def done(path):
            self._saved[key] = data
            layer.metadata[_tuning.LAYER_KEY]["projected"] = projected
            n = len(data)
            self._message(f"saved {n} point{'s' if n != 1 else ''} of {meta['image_id']} to {Path(path).name}", GREEN)
            self.refresh()

        def failed(exc):
            if isinstance(exc, _tuning.RunChanged):
                self._retire(key, layer)
                self._message(f"{meta['image_id']}: {exc}. The layer stays in the viewer as {layer.name}; Open image "
                              f"opens the image on its new run with the points of its file", AMBER)
                self._show_points()
                return
            self._failed(exc, f"{meta['image_id']}: ")

        self._io("save", job, f"saving the truth points of {meta['image_id']}", done, failed)
        return True

    def _retire(self, key, layer) -> None:
        """A truth layer whose run changed its grid: no longer the image's layer here (Open image opens the new
        run), and renamed so it is not taken for it, with the run's layers this dock added for that same run."""
        self._truth.pop(key, None)
        self._saved.pop(key, None)
        olds = [layer]
        rkey = (key[0], key[1])
        known = self._run_layers.get(rkey)
        stamp = (layer.metadata.get(_tuning.LAYER_KEY) or {}).get("run_stamp")
        if known is not None and stamp is not None and known.get("stamp") == stamp:
            del self._run_layers[rkey]
            olds += [*known.get("source", []), *known.get("results", [])]
        self._rename_old(olds)

    def _rename_old(self, layers) -> int:
        """Mark ``layers`` (those still in the viewer) as layers of an earlier run of their image; how many were
        renamed."""
        n = 0
        for old in layers:
            try:
                if old in self.viewer.layers and not old.name.endswith("(old run)"):
                    old.name = f"{old.name} (old run)"
                    n += 1
            except Exception:  # noqa: BLE001 - a layer that is gone keeps nothing to rename
                pass
        return n

    def _show_points(self, *, sync_box: bool = False) -> None:
        """The selected image's truth-point line; ``sync_box`` (another image selected) also sets the projected box
        to that image's layer (as the user left it for that layer)."""
        row = self._row()
        layer = self.truth_layer(row.image_id) if row is not None else None
        if row is None:
            text = ""
        elif layer is None:
            text = (f"Open {row.image_id} to click its truth points." if row.usable and row.finished
                    else row.text)
        else:
            n = len(layer.data)
            key = self._key(row.image_id)
            points_unsaved = self._points_unsaved(key, layer)
            if n == 0 and not points_unsaved:
                text = ("Click the spots of each nucleus you annotate on the truth-points layer (it is in add mode); "
                        "every spot of an annotated nucleus counts. Then Save points.")
            elif points_unsaved:
                text = f"{n} point{'s' if n != 1 else ''}, not saved"
            elif self._box_changed(layer):
                text = f"{n} point{'s' if n != 1 else ''} saved; the 'placed on a projection' box changed, not saved"
            else:
                text = f"{n} point{'s' if n != 1 else ''} saved"
        if sync_box:
            self.projected_check.blockSignals(True)
            self.projected_check.setChecked(self._box(layer) if layer is not None else False)
            self.projected_check.blockSignals(False)
        self.points_label.setText(text)
        self._update()

    # ------------------------------------------------------------------ the grid and the count
    def grid_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Tuning grid", str(self.root or ""), "YAML (*.yaml *.yml)")
        if path:
            self.set_grid_file(path)

    def set_grid_file(self, path) -> None:
        """Search the grid of this YAML file, or the engine's default for the recipe's backend with None. The file
        is read in the file worker. A relative path is taken from the current folder (the count and the engine
        then read the same file)."""
        self.grid_file = Path(os.path.abspath(str(path))) if path else None
        self._show_grid()
        backend, grid_file = self.backend, self.grid_file

        def job(should_stop):
            return _tuning.grid_state(backend, grid_file)

        def done(res):
            if grid_file != self.grid_file:
                return
            self._grid, self._grid_error = res
            self._show_grid()

        self._io("grid", job, "reading the grid", done, lambda e: self._failed(e), replace=True)

    def set_max_runs(self, n: int) -> None:
        self.max_runs_spin.setValue(int(n))

    def _show_grid(self) -> None:
        self.grid_label.setText(_tuning.grid_text(self._grid if not self._grid_error else None, self.grid_file))
        self._show_count()

    def _show_count(self) -> None:
        if self.root is None:
            self.count_label.setText("")
        elif self._grid_error:
            self.count_label.setText(self._grid_error)
        elif self._grid is not None and self.backend is not None:
            text = _tuning.count_text(self._grid, len(self.annotated_images()), self.backend,
                                      self.max_runs_spin.value())
            self.count_label.setText(text + (f" (the backend from {self.backend_source})" if self.backend_source
                                             and self.backend_source != "the project's recipe" else ""))
        else:
            self.count_label.setText("")
        self._update()

    # ------------------------------------------------------------------ tune and apply
    def tune(self) -> bool:
        """Run ``germquant tune-spots`` on the project's saved truth points through the engine, in the engine
        worker. True when it started."""
        if self.root is None:
            self._message("open a project first", AMBER)
            return False
        if self._running():
            self._message("a run of this project is in progress: tune when it has finished", AMBER)
            return False
        if self.engine_worker.busy:
            self._message(f"the engine is busy ({self.engine_worker.label}): one search at a time", AMBER)
            return False
        unsaved = self.unsaved_images()
        if unsaved:
            self._message(f"Save the truth points of {', '.join(unsaved)} first: a search reads the saved files", AMBER)
            return False
        root = self.root
        grid, max_runs = self.grid_file, int(self.max_runs_spin.value())
        say = self._lines.line.emit
        self.progress_text = ""
        pack_id, recipe_id = self._recipe()

        def job(should_stop):
            # the search runs the recipe's spot backend on the project's runs, so it runs where they run: the
            # recipe's own interpreter on this computer when one is stored (as the Project dock's runs and status),
            # else the engine
            runner = self.session.runner_for(pack_id, recipe_id, should_stop=should_stop)
            runner.require(_tuning.FEATURE, should_stop=should_stop)
            rec = runner.tune_spots(str(root), points=_tuning.POINTS_DIR, grid=str(grid) if grid else None,
                                    max_runs=max_runs, say=say, should_stop=should_stop)
            run = rec.get("run_dir") if isinstance(rec, dict) else None
            view = _tuning.run_view(run) if run else None
            return rec, view

        def done(res):
            rec, view = res
            self.last_tune = rec
            run = str(rec.get("run_dir")) if rec.get("run_dir") else None
            if not self._owns(root):
                where = f" (its tuning run is {run})" if run else ""
                self._message(f"the search of {root} finished{where}, but the project {self._gone()}: open it "
                              f"again to see and apply the result", AMBER)
                return
            if run and view is not None:
                self.views[run] = {**view, "warnings": list(rec.get("warnings") or [])}
                if run not in self.runs:
                    self.runs.insert(0, run)
                self.current_run = run
            self._show_project()
            if rec.get("best"):
                self._message(f"tuning run {Path(run).name if run else ''} finished: {_tuning.best_text(rec['best'])}",
                              GREEN)
            else:
                self._message("the search ran, but no setting could be scored (see the warnings)", AMBER)

        return self._engine_job(job, TUNE_LABEL, done, lambda e: self._failed(e, "tune-spots: "))

    def cancel(self) -> bool:
        """Stop the search in flight (its engine process ends); False when none runs, and for an engine in this
        napari environment, which cannot stop a search (it ends on its own)."""
        if not self.searching:
            return False
        if not self._cancellable():
            self._message("a search in this napari environment cannot be stopped from the dock; it ends on its own",
                          AMBER)
            return False
        return self.engine_worker.cancel()

    def select_run(self, run_dir) -> bool:
        run = str(run_dir)
        if run not in self.runs:
            return False
        self.runs_combo.setCurrentIndex(self.runs.index(run))
        self.current_run = run
        self._show_run()
        return True

    def _on_run_chosen(self, index: int) -> None:
        data = self.runs_combo.itemData(index) if index >= 0 else None
        if data:
            self.current_run = str(data)
            self._show_run()

    def _show_run(self) -> None:
        view = self.views.get(self.current_run) if self.current_run else None
        t = self.ranked_table
        t.clear()
        if view is None:
            t.setRowCount(0)
            t.setColumnCount(0)
            self.objective_label.setText("" if self.root is None else "no tuning run yet")
            self.best_label.setText("")
            self.warnings_label.setText("")
            self._update()
            return
        head, rows = view["head"], view["rows"]
        t.setColumnCount(len(head))
        t.setRowCount(len(rows))
        t.setHorizontalHeaderLabels(head)
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                t.setItem(r, c, QTableWidgetItem(value))
        self.objective_label.setText(_tuning.objective_text(view["best"], view["record"]))
        self.best_label.setText(_tuning.best_text(view["best"]))
        warnings = view.get("warnings")
        if warnings is None:
            warnings = (view["record"] or {}).get("warnings") or []
        self.warnings_label.setText("\n".join(f"warning: {w}" for w in warnings))
        self._update()

    def apply(self) -> bool:
        """Write the selected tuning run's best setting into the project's recipe.yaml through the engine (``tune-spots
        --apply``, with the plugin version), in the engine worker. True when it started."""
        if self.root is None:
            self._message("open a project first", AMBER)
            return False
        run = self.current_run
        view = self.views.get(run) if run else None
        if view is None:
            self._message("choose a tuning run first (Tune makes one)", AMBER)
            return False
        if not view.get("best"):
            self._message(f"the tuning run {Path(run).name} has no best setting (no best.yaml): nothing to apply", AMBER)
            return False
        if self._running():
            self._message("a run of this project is in progress: apply when it has finished", AMBER)
            return False
        if self.engine_worker.busy:
            self._message(f"the engine is busy ({self.engine_worker.label}); apply when it has finished", AMBER)
            return False
        root = self.root
        old = _tuning.project_hash(self.manifest or {})
        version = plugin_version()

        def job(should_stop):
            runner = self.session.require(_tuning.FEATURE, should_stop=should_stop)
            rec = runner.tune_apply(str(root), str(run), plugin_version=version, should_stop=should_stop)
            try:                                      # what the write left: the hash and whether it reads validated
                manifest = _tuning.read_manifest(root)
            except _tuning.TuningError:
                manifest = None
            return rec, manifest

        def done(res):
            rec, manifest = res
            self.last_apply = rec
            record = rec if isinstance(rec, dict) else {}
            new = record.get("config_hash") or (_tuning.project_hash(manifest) if manifest else None)
            text = _tuning.applied_text(new, old, (manifest or {}).get("validated"))
            owned = self._owns(root)
            if not owned:                             # the write happened either way
                text = f"{root}: {text}"
            notes = _tuning.apply_warnings(record)    # the engine's TuningWarning: the write still happened
            if notes:
                text += "\n" + "\n".join(f"warning: {n}" for n in notes)
            self._message(text, AMBER if notes else GREEN)
            hub = self.hub
            if hub is not None and _same(getattr(hub, "root", None), root):
                hub.request_reload()
            if owned:
                self.refresh()

        return self._engine_job(job, "applying the tuning run (germquant tune-spots --apply)", done,
                                lambda e: self._failed(e, "tune-spots --apply: "))

    # ------------------------------------------------------------------ buttons
    def _update(self) -> None:
        try:
            has = self.root is not None
            row = self._row()
            busy = self.engine_worker.busy
            running = self._running()
            annotator = self._annotator() is not None
            self.refresh_btn.setEnabled(has)
            self.close_btn.setEnabled(has)
            self.open_btn.setEnabled(bool(has and row is not None and row.usable and row.finished and annotator
                                          and self.viewer is not None))
            layer = self.truth_layer(row.image_id) if row is not None else None
            searching = self.searching
            self.save_btn.setEnabled(bool(layer is not None and annotator and _tuning.points_problem() is None
                                          and not searching))
            self.projected_check.setEnabled(layer is not None)
            self.tune_btn.setEnabled(bool(has and not busy and not running))
            cancellable = self._cancellable() if searching else True
            self.cancel_btn.setEnabled(searching and cancellable)
            self.cancel_btn.setToolTip("end the search's engine process" if cancellable else
                                       "a search in this napari environment cannot be stopped; it ends on its own")
            view = self.views.get(self.current_run) if self.current_run else None
            self.apply_btn.setEnabled(bool(has and view is not None and view.get("best") and not busy and not running))
        except RuntimeError:                          # the widget is gone (napari closing)
            pass
