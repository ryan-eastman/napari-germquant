"""The Run dock: the engine that runs (environment and device), the image, the recipe (a system pack recipe,
or under Advanced a profile or config file), channel roles, stages, xy stride, output folder, Run, Export.

Everything the widget decides is delegated: roles, stage rows and the RunPlan to ``_engine``, the pack card
to ``_packs``, the engine calls to an engine runner (``_runner``). This module is layout and signals.

Where it runs (docs/DESIGN_system_packs.md, section 8.1): an engine interpreter chosen for this computer
(stored in QSettings) runs pack listing, checks, Get models, Get example and the run itself in its own process;
this napari environment runs them in process only when the user picks it explicitly. With no engine chosen,
Run asks for one instead of falling back. A pack recipe may have its own interpreter. The dock always says
which environment and device run, and warns when the engine's germquant differs from the plugin's own.

The GUI thread never waits for the engine. The version check, the pack listing and every card check run one
at a time in a thread worker (``check_worker``); the dock asks for them and a small scheduler (``_pump``) starts
the newest request of each kind when the worker is free, so a pick made while a check is in flight is checked
next and a stale answer is only cached, never shown. Until a recipe's check is back, its card says "checking"
and Run is off. ``idle`` says when nothing is in flight or waiting; ``wait_idle`` lets a script wait for that.

The Run button executes the plan in a napari thread worker (``_runner.execute_run``: run_start, the run,
image_done or image_failed, run_end in ``runs/<run_id>.jsonl`` next to the output folder); log lines of the
engine go to the status bar through LogBridge; when the worker returns, the results folder is loaded as layers
through ``_results`` (the loader the Results dock uses). Cancel ends a run in a separate process (the engine
process and every process it started); napari quitting does the same for a run, a download and a check.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _engine, _packs, _settings
from ._engine import Recipe, RunPlan
from ._engine_widgets import EngineBox, InterpreterDialog, PackPicker
from ._qt_common import LogBridge, WorkerRunner, add_layer_tuples, format_exception, set_viewer_status
from ._reader import header_spacing_text, image_info, open_image
from ._results import ResultsSet, load_results_layers, summary_text
from ._runner import (
    EngineRunner,
    InProcessRunner,
    RunCancelled,
    RunnerError,
    SubprocessRunner,
    detect_interpreters,
    execute_run,
    mismatch_warning,
    plugin_engine_info,
)
from germquant import api

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)

IMAGE_FILTER = "Images (" + " ".join(f"*{s}" for s in api.SUPPORTED_SUFFIXES) + ");;All files (*)"
PACK, PROFILE = "pack", "profile"
FIRST_LOOK_PACK = "celegans_germline"

engine_log = logging.getLogger("napari_germquant.engine")

ENGINE_INTRO = (
    "Choose the Python environment that runs germquant on this computer. Pack lists, checks, Get models, Get "
    "example and every run go through it, so the numbers are made where the models, Cellpose and SpotMAX are "
    "installed (on the lab workstation: the engine checkout's .venv). The choice is stored for this computer. "
    "Running in this napari environment instead is an explicit choice, never a fallback.")


def _say(line: str) -> None:
    """An engine subprocess line, to the status bar (LogBridge listens to napari_germquant)."""
    engine_log.info(line)


def _find_data(combo: QComboBox, value) -> int:
    """The index of the item whose data equals ``value`` in Python terms (QComboBox.findData compares a Python
    tuple by identity under PyQt6), or -1."""
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            return i
    return -1


def _error_text(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, RunnerError) else f"{type(exc).__name__}: {exc}"


class RunWidget(QWidget):
    """napari dock widget. ``napari_viewer`` is injected by napari; None builds the widget without a viewer
    (the headless tests of the roles table and the plan do that)."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._image: Path | None = None
        self._meta: dict = {}
        self._channel_names: list[str] = []
        self._recipe: Recipe | None = None
        self._cfg = None
        self._role_rows: list[_engine.RoleRow] = []
        self._stage_rows: list[_engine.StageRow] = []
        self.stage_checks: dict[str, QCheckBox] = {}
        self._worker = None
        self._run_cancel: threading.Event | None = None
        self._run_cancellable = False
        self._t0 = 0.0
        self._plan: RunPlan | None = None
        self.last_result: dict | None = None
        self.last_error: BaseException | None = None
        self.last_export: tuple[Path, Path] | None = None
        self.last_run_log: Path | None = None
        # system packs and the engine runner
        self._source = PROFILE
        self._listing: dict | None = None
        self._entries: list[_packs.PackEntry] = []
        self._check_cache: dict[tuple, dict] = {}
        self._pack_status: dict | None = None
        self._card_pending = False
        self._engine_runner: EngineRunner | None = None
        self._engine_info: dict | None = None
        self._engine_error: str | None = None
        self._last_engine_error: str = ""
        self.engine_ok: bool | None = None   # the outcome of the last engine choice checked (None: none yet)
        self._recipe_runners: dict[str, EngineRunner] = {}
        self._config_files: list[Path] = []
        self.runner_env: dict = {}           # extra environment for every engine subprocess this dock starts
        self._pending_run = False
        self._pending_recipe: Recipe | None = None
        self._resume_run = False
        self.interpreter_dialog: InterpreterDialog | None = None
        self.last_pack_result: dict | None = None
        self.last_pack_error: BaseException | None = None
        # the engine calls the dock waits for, newest request of each kind (started by _pump)
        self._want_engine: dict | None = None
        self._want_recipe_interp: dict | None = None
        self._want_listing: dict | None = None
        self._want_card = False
        self.pack_worker = WorkerRunner(napari_viewer, self)
        self.check_worker = WorkerRunner(None, self)       # version checks, listings and card checks
        self._bridge = LogBridge(self)
        self._bridge.message.connect(self._on_log_line)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build()
        self._populate_profiles()
        self._init_engine()
        if self.viewer is not None:
            try:
                self.viewer.layers.selection.events.active.connect(self._on_active_layer)
            except Exception:  # noqa: BLE001 - a viewer proxy without events is not worth failing over
                pass

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        root = QVBoxLayout(content)

        self.engine_box = EngineBox()
        self.engine_box.choose_clicked.connect(self.ask_for_engine)
        self.engine_box.device_changed.connect(self._on_device)
        root.addWidget(self.engine_box)

        img_box = QGroupBox("Image")
        img_form = QFormLayout(img_box)
        row = QHBoxLayout()
        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText("an .nd2 (or .tif, .czi, .lif, .zarr) file")
        self.image_edit.editingFinished.connect(lambda: self.set_image(self.image_edit.text()))
        row.addWidget(self.image_edit)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_image)
        row.addWidget(browse)
        img_form.addRow("File", row)
        row2 = QHBoxLayout()
        self.from_layer_btn = QPushButton("Use selected layer")
        self.from_layer_btn.setToolTip("take the file of the selected layer (one this plugin opened)")
        self.from_layer_btn.clicked.connect(self._use_selected_layer)
        row2.addWidget(self.from_layer_btn)
        row2.addWidget(QLabel("open with xy stride"))
        self.open_stride = QSpinBox()
        self.open_stride.setRange(1, 16)
        self.open_stride.setToolTip("1 = full resolution (a gonad is about 3 GB); 4 = every 4th pixel, for a quick look")
        row2.addWidget(self.open_stride)
        self.open_btn = QPushButton("Open in viewer")
        self.open_btn.clicked.connect(self._open_in_viewer)
        row2.addWidget(self.open_btn)
        img_form.addRow("", row2)
        self.image_info = QLabel("no image")
        self.image_info.setWordWrap(True)
        img_form.addRow("Header", self.image_info)
        root.addWidget(img_box)

        self.pack_picker = PackPicker()
        self.pack_picker.selection_changed.connect(self._on_user_pack_selection)
        self.pack_picker.refresh_clicked.connect(lambda: self.refresh_packs())
        self.pack_picker.get_models_clicked.connect(self.get_models)
        self.pack_picker.get_example_clicked.connect(self.get_example)
        self.pack_picker.interpreter_clicked.connect(self.ask_for_recipe_interpreter)
        self.pack_picker.cancel_clicked.connect(self.cancel_pack_operation)
        root.addWidget(self.pack_picker)

        self.advanced_box = QGroupBox("Advanced: profile or config file (instead of a pack)")
        self.advanced_box.setCheckable(True)
        self.advanced_box.setChecked(True)
        self.advanced_box.toggled.connect(self._on_advanced_toggled)
        rec_form = QFormLayout(self.advanced_box)
        row3 = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo)
        row3.addWidget(self.profile_combo, 1)
        cfg_btn = QPushButton("Config file...")
        cfg_btn.setToolTip("any config YAML, for example an overlay this widget exported (extends: a profile)")
        cfg_btn.clicked.connect(self._browse_config)
        row3.addWidget(cfg_btn)
        rec_form.addRow("Profile", row3)
        self.show_all_check = QCheckBox("show all (also the benchmark profiles)")
        self.show_all_check.toggled.connect(lambda _on: self._populate_profiles())
        rec_form.addRow("", self.show_all_check)
        self.recipe_label = QLabel("")
        self.recipe_label.setWordWrap(True)
        rec_form.addRow("File", self.recipe_label)
        root.addWidget(self.advanced_box)

        roles_box = QGroupBox("Channel roles (from the recipe's channel map)")
        roles_lay = QVBoxLayout(roles_box)
        self.roles_table = QTableWidget(0, 4)
        self.roles_table.setHorizontalHeaderLabels(["Role", "Channel", "Marker", "How"])
        self.roles_table.horizontalHeader().setStretchLastSection(True)
        self.roles_table.verticalHeader().setVisible(False)
        self.roles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        roles_lay.addWidget(self.roles_table)
        self.unassigned_label = QLabel("")
        self.unassigned_label.setWordWrap(True)
        roles_lay.addWidget(self.unassigned_label)
        root.addWidget(roles_box)

        self.stages_box = QGroupBox("Stages (untick = --no-<stage>; grey = cannot run here)")
        self.stages_layout = QVBoxLayout(self.stages_box)
        root.addWidget(self.stages_box)

        run_box = QGroupBox("Run")
        run_form = QFormLayout(run_box)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 16)
        self.stride_spin.setToolTip("downsample x and y for a quick test (1 = full resolution, the validated setting)")
        run_form.addRow("xy stride", self.stride_spin)
        row4 = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("results folder (created); default <image stem>_results next to the image")
        row4.addWidget(self.out_edit)
        out_btn = QPushButton("Browse...")
        out_btn.clicked.connect(self._browse_out)
        row4.addWidget(out_btn)
        run_form.addRow("Output folder", row4)
        row5 = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        row5.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("stop the run: the engine process and every process it started are ended, and "
                                   "the run log says it was cancelled (a run in this napari environment cannot be "
                                   "stopped)")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_run)
        row5.addWidget(self.cancel_btn)
        self.export_btn = QPushButton("Export profile")
        self.export_btn.setToolTip("write an extends: overlay YAML (validated: false) and the exact germquant run and "
                                   "batch commands for this plan")
        self.export_btn.clicked.connect(self._export_dialog)
        row5.addWidget(self.export_btn)
        run_form.addRow("", row5)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        run_form.addRow("Status", self.status_label)
        root.addWidget(run_box)
        root.addStretch(1)

    # ------------------------------------------------------------------ engine calls, one at a time
    @property
    def idle(self) -> bool:
        """No engine check (version, listing, card) is in flight or waiting."""
        return (not self.check_worker.busy and self._want_engine is None and self._want_recipe_interp is None
                and self._want_listing is None and not self._want_card)

    def wait_idle(self, timeout_s: float = 300.0) -> bool:
        """For a script driving the dock: run the Qt event loop until ``idle`` (True) or ``timeout_s`` (False).
        The dock itself never waits."""
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

    def _pump(self) -> None:
        """Start the next engine call the dock waits for, when the check worker is free: an engine choice first,
        then a recipe's own interpreter, the pack listing, and the card of the current selection."""
        if self.check_worker.busy:
            return
        if self._want_engine is not None:
            req, self._want_engine = self._want_engine, None
            self._start_engine_job(req)
        elif self._want_recipe_interp is not None:
            req, self._want_recipe_interp = self._want_recipe_interp, None
            self._start_recipe_interp_job(req)
        elif self._want_listing is not None:
            req, self._want_listing = self._want_listing, None
            self._start_listing_job(req)
        elif self._want_card:
            self._want_card = False
            self._start_card_job()

    def _start_check(self, func, label: str, returned, errored) -> None:
        """``func(should_stop)`` in the check worker; ``_pump`` starts the next call when it is done."""
        worker = self.check_worker
        worker.start(lambda: func(worker.should_stop), label=label, returned=returned, errored=errored,
                     finished=self._pump, line_format=lambda _line: None)

    # ------------------------------------------------------------------ the engine
    def _init_engine(self) -> None:
        self.engine_box.set_device(_settings.device())
        if not _engine.packs_supported():
            self.pack_picker.show_message(_engine.packs_unavailable_message(), RED)
            self.pack_picker.setEnabled(False)
            self.advanced_box.setChecked(True)
            self.advanced_box.setCheckable(False)        # profiles and config files only, as before packs
        choice = _settings.engine_choice()
        if choice is None:
            self._show_engine()
            if _engine.packs_supported():
                self.pack_picker.show_message("Choose the engine environment (Engine, above) to list its packs: "
                                              "the list, the checks and the models come from the environment "
                                              "that will run them.", AMBER)
            return
        self._activate_engine(choice, initial=True, store=False, origin="startup")

    def _make_runner(self, choice: str) -> EngineRunner:
        runner: EngineRunner = InProcessRunner() if choice == _settings.INPROCESS else \
            SubprocessRunner(choice, env=self.runner_env)
        runner.device = self.engine_box.device()
        return runner

    def _activate_engine(self, choice: str, *, initial: bool = False, store: bool = True, origin: str = "") -> None:
        """Ask for ``choice`` to be checked (its version, then its pack list) in the check worker; it becomes the
        dock's engine (and, with ``store``, this computer's stored choice) only when the check passes."""
        runner = self._make_runner(choice)
        self._want_engine = {"choice": choice, "runner": runner, "initial": initial, "store": store,
                             "origin": origin}
        self._want_listing = None                        # a listing of the old engine is not wanted any more
        if self._engine_runner is None:
            self.engine_box.show_engine(f"checking {runner.describe()} ...")
        else:
            self._status(f"checking the engine {runner.describe()} ...", None)
        self._pump()

    def _start_engine_job(self, req: dict) -> None:
        runner = req["runner"]
        listing_wanted = _engine.packs_supported()

        def job(should_stop):
            info = runner.version(should_stop=should_stop)
            listing, list_error = None, None
            if listing_wanted:
                try:
                    listing = runner.list_packs(should_stop=should_stop)
                except RunCancelled:
                    raise
                except RunnerError as e:
                    list_error = str(e)
            return info, listing, list_error

        self._start_check(job, f"checking {runner.describe()}", lambda res: self._engine_ready(req, *res),
                          lambda exc: self._engine_failed(req, exc))

    def _engine_ready(self, req: dict, info: dict, listing: dict | None, list_error: str | None) -> None:
        if self._want_engine is not None:               # a newer choice waits: this one is superseded
            return
        runner = req["runner"]
        self._engine_runner, self._engine_info, self._engine_error = runner, info, None
        self._recipe_runners = {}
        self._check_cache.clear()
        self.engine_ok = True
        if req["store"]:
            _settings.set_engine_choice(req["choice"])
        self._show_engine()
        self._apply_listing(listing, list_error, choose_route=req["initial"])
        if req["origin"] == "dialog":
            self._status(f"engine set for this computer: {runner.describe()}", GREEN)
            self._resume_pending_run()

    def _engine_failed(self, req: dict, exc: BaseException) -> None:
        if self._want_engine is not None:
            return
        msg = _error_text(exc)
        self._last_engine_error = msg
        self.engine_ok = False
        resume, self._resume_run = self._resume_run, False
        if req["initial"] or self._engine_runner is None:
            self._engine_runner, self._engine_info, self._engine_error = None, None, msg
            self._show_engine()
            if _engine.packs_supported():
                self.pack_picker.show_message(msg, RED)
                self._entries = []
                self.pack_picker.set_entries([])
            if req["origin"] == "dialog":
                self._status(f"that engine cannot be used: {msg}" + ("; nothing ran" if resume else ""), RED)
        else:                                    # a working engine stays; the new choice is not stored
            self._status(f"that engine cannot be used, so the dock keeps {self._engine_runner.describe()}: {msg}",
                         RED)

    def set_engine(self, choice: str | Path, env: dict | None = None) -> None:
        """Use ``choice`` (a Python path, or ``inprocess``) as this computer's engine, stored when its version
        check passes; ``env`` replaces the extra environment of the dock's engine subprocesses. The check runs in
        the check worker: ``engine`` and ``engine_ok`` say the outcome once ``idle``."""
        if env is not None:
            self.runner_env = dict(env)
        self._activate_engine(str(choice), store=True, origin="set_engine")

    @property
    def engine(self) -> EngineRunner | None:
        """The dock's engine runner (None while none is chosen or the chosen one cannot run)."""
        return self._engine_runner

    def engine_runner(self, pack_id: str | None = None, recipe_id: str | None = None) -> EngineRunner | None:
        """The runner a pack recipe (or, without one, anything else) goes to: the recipe's own interpreter when
        one is stored for this computer, else the dock's engine, else None (Run then asks)."""
        mapped = _settings.recipe_interpreter(pack_id, recipe_id)
        if mapped:
            runner = self._recipe_runners.get(mapped)
            if runner is None:
                runner = self._make_runner(mapped)
                self._recipe_runners[mapped] = runner
            runner.device = self.engine_box.device()
            return runner
        if self._engine_runner is not None:
            self._engine_runner.device = self.engine_box.device()
        return self._engine_runner

    def _current_runner(self) -> EngineRunner | None:
        if self._source == PACK:
            return self.engine_runner(self.pack_picker.current_pack_id(), self.pack_picker.current_recipe_id())
        return self.engine_runner()

    def _show_engine(self) -> None:
        r = self._engine_runner
        if r is None:
            if self._engine_error:
                self.engine_box.show_engine("The engine chosen for this computer cannot be used: "
                                            f"{self._engine_error}", RED)
            else:
                self.engine_box.show_engine("Not chosen yet. Run asks for it: the Python of the germquant "
                                            "environment that should make the numbers (on the lab workstation, the "
                                            "engine checkout's .venv). Choose engine... sets it now.", AMBER)
            self.engine_box.show_warning("")
            return
        info = self._engine_info or {}
        lines = [r.describe(), f"germquant {info.get('germquant')} (git {str(info.get('git_sha'))[:12]}), "
                               f"python {info.get('python')}"]
        warnings = [mismatch_warning(plugin_engine_info(), info)]
        if self._source == PACK:
            pid, rid = self.pack_picker.current_pack_id(), self.pack_picker.current_recipe_id()
            mapped = _settings.recipe_interpreter(pid, rid)
            if mapped:
                rr = self.engine_runner(pid, rid)
                lines.append(f"recipe {pid} {rid} runs in its own interpreter: {rr.describe() if rr else mapped}")
                version = rr.cached_version() if rr is not None else None
                if version is not None:
                    warnings.append(mismatch_warning(plugin_engine_info(), version))
                elif rr is not None and rr.version_error:
                    warnings.append(f"the recipe's interpreter cannot be used: {rr.version_error}")
        self.engine_box.show_engine("\n".join(lines))
        self.engine_box.show_warning("\n".join(w for w in warnings if w))

    def _on_device(self, value: str) -> None:
        _settings.set_device(value)
        for runner in [self._engine_runner, *self._recipe_runners.values()]:
            if runner is not None:
                runner.device = value

    def ask_for_engine(self) -> InterpreterDialog:
        """Open the dialog that asks for this computer's engine interpreter (non-blocking)."""
        dlg = InterpreterDialog(detect_interpreters(), title="germquant engine for this computer", intro=ENGINE_INTRO,
                                current=_settings.engine_choice(), allow_inprocess=True, parent=self)
        dlg.accepted.connect(lambda: self._engine_dialog_done(dlg, True))
        dlg.rejected.connect(lambda: self._engine_dialog_done(dlg, False))
        self.interpreter_dialog = dlg
        dlg.open()
        return dlg

    def _engine_dialog_done(self, dlg: InterpreterDialog, accepted: bool) -> None:
        pending, self._pending_run = self._pending_run, False
        choice = dlg.choice() if accepted else None
        if not choice:
            if pending:
                self._status("no engine chosen: nothing ran", AMBER)
            return
        # Run resumes after the check only for the recipe it was pressed with (see _resume_pending_run)
        self._resume_run = pending
        self._activate_engine(choice, store=True, origin="dialog")

    def _resume_pending_run(self) -> None:
        """After an engine chosen from Run's prompt: run, but only the recipe Run was pressed with. A pack the
        dock selected on its own once the engine listed its packs is never run unasked."""
        resume, self._resume_run = self._resume_run, False
        if not resume:
            return
        if self._recipe is not None and self._recipe == self._pending_recipe:
            self.run()
        else:
            self._status("Engine set. Pick a pack and recipe, then press Run.", AMBER)

    def ask_for_recipe_interpreter(self) -> InterpreterDialog | None:
        """Open the dialog that sets (or clears) the current pack recipe's own interpreter."""
        pid, rid = self.pack_picker.current_pack_id(), self.pack_picker.current_recipe_id()
        if not pid or not rid:
            return None
        intro = (f"Recipe {rid} of pack {pid} can run in its own Python environment, for example one with an "
                 f"optional backend the dock's engine lacks. Its pack check, models and runs then go there. The "
                 f"choice is stored for this computer and recorded in every run log.")
        dlg = InterpreterDialog(detect_interpreters(), title=f"interpreter for {pid} {rid}", intro=intro,
                                current=_settings.recipe_interpreter(pid, rid), allow_inprocess=True,
                                allow_default=True, parent=self)
        dlg.accepted.connect(lambda: self.set_recipe_interpreter(pid, rid, dlg.choice() or None))
        self.interpreter_dialog = dlg
        dlg.open()
        return dlg

    def set_recipe_interpreter(self, pack_id: str, recipe_id: str, python: str | Path | None) -> None:
        """Store (or with None clear) the interpreter of one pack recipe. A new interpreter is stored after its
        version check, which runs in the check worker."""
        if python:
            runner = self._make_runner(str(python))
            self._want_recipe_interp = {"pack": pack_id, "recipe": recipe_id, "python": str(python),
                                        "runner": runner}
            self._status(f"checking {runner.describe()} for {pack_id} {recipe_id} ...", None)
            self._pump()
            return
        _settings.set_recipe_interpreter(pack_id, recipe_id, None)
        self._recipe_interp_changed(pack_id, recipe_id)

    def _start_recipe_interp_job(self, req: dict) -> None:
        runner = req["runner"]
        self._start_check(lambda should_stop: runner.version(should_stop=should_stop),
                          f"checking {runner.describe()}", lambda _info: self._recipe_interp_ready(req),
                          lambda exc: self._status(f"that interpreter cannot run {req['pack']} {req['recipe']}: "
                                                   f"{_error_text(exc)}", RED))

    def _recipe_interp_ready(self, req: dict) -> None:
        self._recipe_runners[req["python"]] = req["runner"]
        _settings.set_recipe_interpreter(req["pack"], req["recipe"], req["python"])
        self._status(f"recipe {req['pack']} {req['recipe']} runs in {req['runner'].describe()} on this computer",
                     GREEN)
        self._recipe_interp_changed(req["pack"], req["recipe"])

    def _recipe_interp_changed(self, pack_id: str, recipe_id: str) -> None:
        self._check_cache.clear()
        if self.pack_picker.current_pack_id() == pack_id and self.pack_picker.current_recipe_id() == recipe_id:
            self.refresh_card()
        self._show_engine()

    # ------------------------------------------------------------------ system packs
    def refresh_packs(self, choose_route: bool = False) -> None:
        """List the packs again through the engine runner (``python -m germquant.packs list --json`` for a
        subprocess engine), in the check worker, and fill the pack picker when the list is back.
        ``choose_route``: switch to the pack route when packs are listed."""
        if not _engine.packs_supported():
            return
        runner = self._engine_runner
        if runner is None:
            self.pack_picker.show_message(self._engine_error or "Choose the engine environment (Engine, above) to "
                                          "list its packs.", RED if self._engine_error else AMBER)
            return
        self._want_listing = {"runner": runner, "choose_route": choose_route}
        self.pack_picker.show_message(f"listing the packs in {runner.describe()} ...", None)
        self._pump()

    def _start_listing_job(self, req: dict) -> None:
        runner = req["runner"]
        if runner is not self._engine_runner:              # the engine changed meanwhile: its own listing came
            self._pump()
            return
        self._start_check(lambda should_stop: runner.list_packs(should_stop=should_stop),
                          f"listing the packs in {runner.describe()}",
                          lambda listing: self._listing_done(req, listing, None),
                          lambda exc: self._listing_done(req, None, _error_text(exc)))

    def _listing_done(self, req: dict, listing: dict | None, error: str | None) -> None:
        if req["runner"] is self._engine_runner:
            self._apply_listing(listing, error, choose_route=req["choose_route"])

    def _first_look(self, usable_ids: list[str]) -> tuple[str, str | None] | None:
        """The pack the picker shows when nothing is picked yet: the pack and recipe the user last picked, else
        the C. elegans germline pack, else the first usable pack."""
        last_pack, last_recipe = _settings.last_pack()
        if last_pack in usable_ids:
            return last_pack, last_recipe
        if FIRST_LOOK_PACK in usable_ids:
            return FIRST_LOOK_PACK, None
        return (usable_ids[0], None) if usable_ids else None

    def _apply_listing(self, listing: dict | None, error: str | None, *, choose_route: bool) -> None:
        """Fill the picker from an engine's listing. The pack shown stays when it is still listed; otherwise the
        first look (``_first_look``). ``choose_route``: switch to the pack route (the dock's first look at a
        configured engine)."""
        if not _engine.packs_supported():
            return
        runner = self._engine_runner
        if error is not None or listing is None:
            self.pack_picker.show_message(f"could not list the packs: {error}", RED)
            self._listing = None
            self._entries = []
            self.pack_picker.set_entries([])
            self._on_pack_selection()
            return
        try:                                       # this environment's own pack discovery sees what is new too
            from germquant import packs as _gq_packs

            _gq_packs.refresh()
        except Exception:  # noqa: BLE001
            pass
        self._listing = listing
        self._check_cache.clear()
        self._entries = _packs.listing_entries(listing)
        usable = [e.id for e in self._entries if e.usable_manifest]
        note = f"{len(usable)} pack(s) found by {runner.describe() if runner else 'the engine'}"
        if listing.get("message"):
            note += f". {listing['message']}"
        self.pack_picker.show_message(note, AMBER if listing.get("message") or not usable else None)
        current = self.pack_picker.current_pack_id()
        select = None
        if choose_route or current is None or current not in [e.id for e in self._entries]:
            select = self._first_look(usable)
        self.pack_picker.blockSignals(True)
        try:
            self.pack_picker.set_entries(self._entries, select=select)
        finally:
            self.pack_picker.blockSignals(False)
        if choose_route and usable:
            self._use_route(PACK)
        else:
            self._on_pack_selection()

    @property
    def pack_entries(self) -> list[_packs.PackEntry]:
        return list(self._entries)

    @property
    def pack_status(self) -> dict | None:
        """The pack check of the selected recipe (None while it is being checked)."""
        return self._pack_status

    def _use_route(self, route: str) -> None:
        if route == PACK and not _engine.packs_supported():
            raise RuntimeError(_engine.packs_unavailable_message())
        self._source = route
        if self.advanced_box.isCheckable():
            self.advanced_box.blockSignals(True)
            self.advanced_box.setChecked(route == PROFILE)
            self.advanced_box.blockSignals(False)
        self.pack_picker.setTitle("System pack (used for the run)" if route == PACK else
                                  "System pack (not used: Advanced is ticked; pick a pack to use it)")
        if route == PACK:
            self._on_pack_selection()
        else:
            self._on_profile_combo(self.profile_combo.currentIndex())
            self._update_run_button()
        self._show_engine()

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._use_route(PROFILE if checked else PACK)

    @property
    def route(self) -> str:
        """``pack`` or ``profile``: where the recipe comes from."""
        return self._source

    def select_pack(self, pack_id: str, recipe: str | None = None, version: str | None = None) -> None:
        """Pick a pack (and recipe) from the engine's list and use it for the run (as a user's pick: it is
        remembered as the last pack)."""
        if not _engine.packs_supported():
            raise RuntimeError(_engine.packs_unavailable_message())
        if not self._entries:
            raise ValueError("no packs are listed yet: choose the engine (Choose engine...) and wait for its list")
        self.pack_picker.blockSignals(True)
        try:
            self.pack_picker.select(pack_id, recipe, version)
        finally:
            self.pack_picker.blockSignals(False)
        self._remember_pack()
        self._use_route(PACK)

    def select_recipe(self, recipe: str) -> None:
        self.pack_picker.select_recipe(recipe)

    def _remember_pack(self) -> None:
        """Store the picker's pack and recipe as the user's last pick (only ever for a pick the user made)."""
        entry, rid = self.pack_picker.current_entry(), self.pack_picker.current_recipe_id()
        if entry is not None and rid and entry.usable_manifest:
            _settings.set_last_pack(entry.id, rid)

    def _on_user_pack_selection(self) -> None:
        """A pack or recipe picked in the picker: remember it and use it for the run (the pack route)."""
        self._remember_pack()
        if _engine.packs_supported() and self._source != PACK:
            self._use_route(PACK)
        else:
            self._on_pack_selection()

    def _on_pack_selection(self) -> None:
        entry, rid = self.pack_picker.current_entry(), self.pack_picker.current_recipe_id()
        if self._source == PACK:
            if entry is not None and rid and entry.usable_manifest:
                self.set_recipe(Recipe.from_pack(entry.ref, rid, hint=entry.profile_path(rid)))
            else:
                self._recipe, self._cfg = None, None
                self.refresh()
        self.refresh_card()

    def _card_key(self, runner: EngineRunner, entry: _packs.PackEntry, rid: str | None) -> tuple:
        image = str(self._image) if self._image is not None and self._image.exists() else None
        return runner.key, entry.ref, rid, image

    def refresh_card(self, force: bool = False) -> dict | None:
        """The card of the selected pack recipe. Its check (``pack check --json``, with the image when one is
        chosen) runs in the environment that will run it, in the check worker: until it is back the card says
        "checking" and Run is off; then the card is rendered, what cannot run is greyed and Run is set. A check
        is kept until the pack list is refreshed or models or examples are fetched; ``force`` checks again.
        Returns the check when it is already known, else None."""
        if force:
            self._check_cache.clear()
        entry, rid = self.pack_picker.current_entry(), self.pack_picker.current_recipe_id()
        self._pack_status = None
        self._card_pending = False
        if entry is None:
            self._update_run_button()
            return None
        if not entry.usable_manifest:
            st = _packs.manifest_status(entry)
            self.pack_picker.show_status(entry, rid, st)
            self._pack_status = st
            self._update_run_button()
            return st
        runner = self.engine_runner(entry.id, rid)
        if runner is None:
            self.pack_picker.show_status(entry, rid, None, engine_text="no engine chosen yet")
            self._update_run_button()
            return None
        st = self._check_cache.get(self._card_key(runner, entry, rid))
        if st is None:
            self.pack_picker.show_status(entry, rid, None, engine_text=runner.describe(), pending=True)
            self._card_pending = True
            self._want_card = True
            self._update_run_button()
            self._pump()
            return None
        self._render_card(entry, rid, st, runner)
        return st

    def _start_card_job(self) -> None:
        entry, rid = self.pack_picker.current_entry(), self.pack_picker.current_recipe_id()
        runner = self.engine_runner(entry.id, rid) if entry is not None and entry.usable_manifest else None
        if runner is None:
            self._pump()
            return
        key = self._card_key(runner, entry, rid)
        if key in self._check_cache:
            self.refresh_card()
            self._pump()
            return
        own = runner is not self._engine_runner
        image = key[3]

        def job(should_stop):
            try:
                if own:
                    runner.version(should_stop=should_stop)  # a recipe's own interpreter: EngineTooOld names the fix
                return key, runner.check(entry.ref, rid, image, should_stop=should_stop)
            except RunCancelled:
                raise
            except RunnerError as e:
                return key, {"pack": entry.id, "version": entry.version, "recipe": rid, "usable": False,
                             "blockers": [{"what": str(e), "fix": "choose a working engine (Choose engine..., or "
                                                                  "Recipe interpreter... for this recipe)"}],
                             "warnings": [], "models": []}

        self._start_check(job, f"checking {entry.ref} {rid}", self._card_checked,
                          lambda exc: self._card_failed(key, exc))

    def _card_checked(self, res: tuple) -> None:
        key, st = res
        self._check_cache[key] = st
        self.refresh_card()                     # shows it when it is still the selection, else checks that
        self._show_engine()                     # a recipe interpreter's version is known now

    def _card_failed(self, key: tuple, exc: BaseException) -> None:
        if isinstance(exc, RunCancelled):       # napari is closing
            return
        self._check_cache[key] = {"pack": key[1], "recipe": key[2], "usable": False, "warnings": [], "models": [],
                                  "blockers": [{"what": _error_text(exc), "fix": "Refresh, or choose a working "
                                                                                 "engine"}]}
        self.refresh_card()

    def _render_card(self, entry: _packs.PackEntry, rid: str | None, st: dict, runner: EngineRunner) -> None:
        self.pack_picker.show_status(entry, rid, st, engine_text=runner.describe())
        info = []
        try:
            if self._source == PACK and self._recipe is not None:
                info.append(f"recipe file: {self._recipe.path()}")
        except Exception:  # noqa: BLE001 - shown in the card's blockers when it matters
            pass
        if self._source == PACK and self._recipe is not None and self._cfg is None:
            info.append("the recipe does not load in this napari environment, so no channel roles or stages can be "
                        "shown (see Status)")
        if _settings.recipe_interpreter(entry.id, rid):
            info.append(f"this recipe runs in its own interpreter: {runner.describe()}")
        self.pack_picker.info_label.setText("\n".join(info))
        self._pack_status = st
        self._update_run_button()

    def _update_run_button(self) -> None:
        busy = self._worker is not None
        checking = self._source == PACK and self._card_pending
        blocked = self._source == PACK and self._pack_status is not None and not self._pack_status.get("usable")
        self.run_btn.setEnabled(not busy and not blocked and not checking)
        if blocked:
            self.run_btn.setToolTip("this recipe cannot run here:\n" + _packs.blocker_text(self._pack_status))
        elif checking:
            self.run_btn.setToolTip("the recipe is being checked in the engine environment")
        else:
            self.run_btn.setToolTip("")

    def _expected_segmenters(self, recipe: Recipe | None) -> list[str] | None:
        if recipe is None or not recipe.is_pack:
            return None
        for e in self._entries:
            if e.ref == recipe.pack:
                r = e.recipe(recipe.recipe) or {}
                need = (r.get("needs") or {}).get("segment_backends")
                return [str(n) for n in need] if need else None
        return None

    def get_models(self) -> bool:
        """Get models: fetch the selected recipe's models into the engine's model store, in a thread worker."""
        entry, rid = self.pack_picker.current_entry(), self.pack_picker.current_recipe_id()
        runner = self.engine_runner(entry.id, rid) if entry is not None else None
        if entry is None or runner is None:
            self._status("choose the engine and a pack recipe first", AMBER)
            return False
        stop = self.pack_worker.should_stop
        return self._start_pack_operation(lambda: runner.fetch_models(entry.ref, rid, say=_say, should_stop=stop),
                                          f"getting the models of {entry.ref} {rid}", self._on_models_fetched,
                                          runner)

    def _start_pack_operation(self, func, label: str, returned, runner: EngineRunner) -> bool:
        if self.pack_worker.busy:
            self._status("a model or example download is already running", AMBER)
            return False
        self.last_pack_result, self.last_pack_error = None, None
        self.pack_picker.set_busy(True, cancellable=runner.cancellable)
        self.pack_worker.start(func, label=label, returned=returned, errored=self._on_pack_error,
                               finished=self._on_pack_finished)
        return True

    def cancel_pack_operation(self) -> bool:
        """Cancel: stop the running Get models or Get example (its engine process and children are ended)."""
        if not self.pack_worker.cancel():
            return False
        self.pack_picker.cancel_btn.setEnabled(False)
        self._status("cancelling the download: ending the engine process ...", AMBER)
        return True

    def _on_models_fetched(self, res: dict) -> None:
        self.last_pack_result = res
        self._check_cache.clear()
        lines = "\n".join(res.get("lines") or [])
        if res.get("ok"):
            self._status(f"models in the store:\n{lines}", GREEN)
        else:
            self._status(f"some models could not be fetched:\n{lines}", RED)

    def get_example(self) -> bool:
        """Get example: fetch the pack's example (a public image is checked, assembled and kept in the examples
        folder), then open it and make it the run's image."""
        entry, rid = self.pack_picker.current_entry(), self.pack_picker.current_recipe_id()
        runner = self.engine_runner(entry.id, rid) if entry is not None else None
        if entry is None or runner is None:
            self._status("choose the engine and a pack first", AMBER)
            return False
        stop = self.pack_worker.should_stop
        return self._start_pack_operation(lambda: runner.fetch_example(entry.ref, say=_say, should_stop=stop),
                                          f"getting the example of {entry.ref}", self._on_example_fetched, runner)

    def _on_example_fetched(self, rec: dict) -> None:
        self.last_pack_result = rec
        image = Path(str(rec.get("image")))
        self.set_image(image)
        self.out_edit.setText(str(self._example_out_dir(rec, image)))
        self._open_in_viewer()
        bits = [f"example {rec.get('id')} ({rec.get('status')}): {image}"]
        if rec.get("licence"):
            bits.append(f"licence {rec.get('licence')}; {rec.get('attribution', '')}")
        if rec.get("recipe"):
            bits.append(f"meant for recipe {rec.get('recipe')}")
        if rec.get("expect"):
            bits.append("expected: " + "; ".join(f"{k} in [{v[0]:g}, {v[1]:g}]" for k, v in rec["expect"].items()))
        self._status("\n".join(bits), GREEN)

    def _example_out_dir(self, rec: dict, image: Path) -> Path:
        """Where a run of an example writes: next to a fetched image; for the synthetic example inside a pack,
        the examples folder (a pack holds only its own files, so nothing may be written into it)."""
        inside_pack = any(_packs.inside(image, f) for f in _packs.pack_folders(self._listing))
        if rec.get("kind") == "local" or inside_pack:
            try:
                from germquant.packs import examples_dir

                base = examples_dir()
            except Exception:  # noqa: BLE001
                base = Path.home() / "germquant_examples"
            return base / str(rec.get("pack") or "pack") / str(rec.get("id") or "example") / \
                f"{api.image_stem(image)}_results"
        return api.default_out_dir(image)

    def _on_pack_error(self, exc: BaseException) -> None:
        self.last_pack_error = exc
        if isinstance(exc, RunCancelled):
            self._status(f"cancelled: {exc}", AMBER)
            return
        self._status(f"FAILED: {format_exception(exc, 1200)}", RED)

    def _on_pack_finished(self) -> None:
        self.pack_picker.set_busy(False)
        self.refresh_card()

    # ------------------------------------------------------------------ recipe (profiles and config files)
    def _populate_profiles(self) -> None:
        keep = self.profile_combo.currentData()
        show_all = self.show_all_check.isChecked()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for prof in _engine.profiles():
            if prof.bench and not show_all:
                continue
            self.profile_combo.addItem(prof.name, ("profile", prof.name))
        for path in self._config_files:
            self.profile_combo.addItem(f"file: {path.name}", ("config", str(path)))
        if keep is not None:
            i = _find_data(self.profile_combo, keep)
            if i >= 0:
                self.profile_combo.setCurrentIndex(i)
        self.profile_combo.blockSignals(False)
        if self.profile_combo.count():
            if self._source == PROFILE and self.profile_combo.currentData() != getattr(self, "_shown_profile", None):
                self._on_profile_combo(self.profile_combo.currentIndex())
        else:
            self._status("no profiles found: germquant's config/profiles folder is not visible from this "
                         "environment (set GERMQUANT_CONFIG_ROOT to the checkout) or pick a config file", RED)

    def _on_profile_combo(self, index: int) -> None:
        if self._source != PROFILE:
            return
        data = self.profile_combo.itemData(index)
        if not data:
            return
        self._shown_profile = data
        kind, value = data
        self.set_recipe(Recipe.from_profile(value) if kind == "profile" else Recipe.from_file(value))

    def select_profile(self, name: str) -> None:
        """Use the profile ``name`` (the Advanced route). A benchmark profile ticks "show all"."""
        if name.startswith(_engine.BENCH_PREFIX) and not self.show_all_check.isChecked():
            self.show_all_check.setChecked(True)
        for i in range(self.profile_combo.count()):
            if self.profile_combo.itemData(i) == ("profile", name):
                self.profile_combo.blockSignals(True)
                self.profile_combo.setCurrentIndex(i)
                self.profile_combo.blockSignals(False)
                self._use_route(PROFILE)
                return
        raise ValueError(f"no profile {name!r} in the list")

    def select_config_file(self, path: str | Path) -> None:
        """A config YAML as the recipe (the command line's --config): one combo entry per file."""
        path = Path(path)
        key = ("config", str(path))
        if path not in self._config_files:
            self._config_files.append(path)
            self.profile_combo.blockSignals(True)
            self.profile_combo.addItem(f"file: {path.name}", key)
            self.profile_combo.blockSignals(False)
        i = _find_data(self.profile_combo, key)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(i)
        self.profile_combo.blockSignals(False)
        self._use_route(PROFILE)

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Config YAML", "", "YAML (*.yaml *.yml);;All files (*)")
        if path:
            self.select_config_file(path)

    def set_recipe(self, recipe: Recipe) -> None:
        self._recipe = recipe
        # the Advanced box's File line belongs to the profile route; a pack recipe's file shows on its card
        label = self.pack_picker.info_label if recipe.is_pack else self.recipe_label
        try:
            self._cfg = recipe.load()
            if not recipe.is_pack:
                label.setText(str(recipe.path()))
        except Exception as e:  # noqa: BLE001 - shown to the user, the widget stays usable
            self._cfg = None
            label.setText(f"could not load: {e}")
            self._status(f"recipe error: {e}", RED)
        self.refresh()

    @property
    def recipe(self) -> Recipe | None:
        return self._recipe

    @property
    def config(self):
        return self._cfg

    # ------------------------------------------------------------------ image
    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Image", "", IMAGE_FILTER)
        if path:
            self.set_image(path)

    def _use_selected_layer(self) -> None:
        if self.viewer is None:
            return
        layer = self.viewer.layers.selection.active
        info = image_info(layer) if layer is not None else None
        if info is None:
            self._status("the selected layer was not opened by this plugin; use Browse", AMBER)
            return
        self.set_image(info["path"])

    def _on_active_layer(self, event=None) -> None:
        if self.viewer is None or self._worker is not None:
            return
        layer = self.viewer.layers.selection.active
        info = image_info(layer) if layer is not None else None
        if info is not None and (self._image is None or str(self._image) != info["path"]):
            self.set_image(info["path"])

    def set_image(self, path: str | Path) -> None:
        text = str(path).strip().strip('"')
        if not text:
            return
        p = Path(text)
        self.image_edit.setText(text)
        self._image = p
        self._meta = {}
        self._channel_names = []
        if not p.exists():
            self.image_info.setText("file not found")
            self.refresh()
            return
        try:
            api.image_format(p)
            meta = api.read_image_metadata(p)
        except Exception as e:  # noqa: BLE001 - the header could not be read; say so, keep the widget alive
            self.image_info.setText(f"cannot read the header: {e}")
            self.refresh()
            return
        self._meta = meta
        self._channel_names = [str(n) for n in meta.get("channel_names", [])]
        sizes = meta.get("sizes", {})
        size_text = " ".join(f"{k}={v}" for k, v in sizes.items())
        spacing_text = header_spacing_text(meta)      # "single plane, voxel um dy=.. dx=.." for one plane
        extra = ""
        n_t = int(sizes.get("T", 1) or 1)
        n_p = max(int(sizes.get("P", 1) or 1), int(meta.get("n_scenes", 1) or 1))
        if n_t > 1 or n_p > 1:
            extra = (f"\nREFUSED: {n_t} time points, {n_p} positions or scenes; the plugin takes fixed single-scene "
                     f"images (a series runs with germquant run --t all in a terminal; its folder opens in the "
                     f"Time-lapse dock)")
        self.image_info.setText(f"{size_text}\n{spacing_text}\nchannels: {self._channel_names}{extra}")
        if not self.out_edit.text().strip():
            self.out_edit.setText(str(api.default_out_dir(p)))
        self.refresh()
        if self._source == PACK:
            self.refresh_card()                       # the check reads the header: expected voxel size, readers

    def _open_in_viewer(self) -> None:
        if self.viewer is None or self._image is None:
            return
        try:
            layers = open_image(self._image, xy_stride=int(self.open_stride.value()))
        except Exception as e:  # noqa: BLE001 - the reader's refusal message is the thing to show
            self._status(str(e), RED)
            return
        add_layer_tuples(self.viewer, layers)
        self._status(f"opened {len(layers)} channel layer(s)", GREEN)

    # ------------------------------------------------------------------ roles and stages
    def refresh(self) -> None:
        """Rebuild the roles table and the stage checkboxes from the current recipe and image header."""
        self.roles_table.setRowCount(0)
        self._role_rows = []
        self._stage_rows = []
        if self._cfg is None:
            self._clear_stage_checks()
            self.unassigned_label.setText("")
            return
        self._role_rows = _engine.role_rows(self._cfg, self._channel_names)
        self.roles_table.setRowCount(len(self._role_rows))
        for i, r in enumerate(self._role_rows):
            channel = f"{r.index}: {r.channel_name}" if r.index is not None else "(none)"
            cells = [r.role + (" *" if r.required else ""), channel, r.marker, r.status]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if r.problem:
                    item.setBackground(QBrush(RED))
                    item.setToolTip("required role with no channel: the run would be flagged "
                                    f"channel:{r.role}:MISSING_REQUIRED and cannot start here")
                elif r.how == "index":
                    item.setBackground(QBrush(AMBER))
                    item.setToolTip("no channel name matched this role; the channel map's index fallback was used "
                                    "and the run flags it (add the channel name to a copy of the map to silence it)")
                self.roles_table.setItem(i, j, item)
        self.roles_table.resizeColumnsToContents()
        if self._channel_names:
            free = _engine.unassigned_channels(self._role_rows, self._channel_names)
            self.unassigned_label.setText(
                "channels with no role: " + ", ".join(f"{i}: {n}" for i, n in free) if free else
                "every channel has a role")
        else:
            self.unassigned_label.setText("* required role. Load an image to resolve the channels.")
        self._stage_rows = _engine.stage_rows(self._cfg, _engine.role_to_index(self._role_rows))
        self._rebuild_stage_checks()

    def _clear_stage_checks(self) -> None:
        for cb in self.stage_checks.values():
            self.stages_layout.removeWidget(cb)
            cb.hide()                     # deleteLater waits for the event loop; never paint the old row meanwhile
            cb.deleteLater()
        self.stage_checks = {}

    def _rebuild_stage_checks(self) -> None:
        previous = {name: cb.isChecked() for name, cb in self.stage_checks.items() if cb.isEnabled()}
        self._clear_stage_checks()
        for row in self._stage_rows:
            cb = QCheckBox(row.name if not row.reason else f"{row.name}   ({row.reason})")
            cb.setToolTip(row.stage.help)
            if row.selectable:
                cb.setEnabled(True)
                cb.setChecked(previous.get(row.name, True))
            else:
                cb.setEnabled(False)
                cb.setChecked(False)
            self.stages_layout.addWidget(cb)
            self.stage_checks[row.name] = cb

    @property
    def role_rows(self) -> list[_engine.RoleRow]:
        return list(self._role_rows)

    @property
    def stage_rows(self) -> list[_engine.StageRow]:
        return list(self._stage_rows)

    def disabled_stages(self) -> tuple[str, ...]:
        """Stages the user unticked among those that could run: the --no-<stage> set."""
        return tuple(r.name for r in self._stage_rows
                     if r.selectable and not self.stage_checks[r.name].isChecked())

    # ------------------------------------------------------------------ plan, export, run
    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Results folder")
        if path:
            self.out_edit.setText(path)

    def plan(self) -> RunPlan:
        """The RunPlan for the current state; ValueError with a plain message when something is missing."""
        if self._image is None or not self._image.exists():
            raise ValueError("choose an image file that exists")
        if self._source == PACK and self._recipe is None:
            raise ValueError("choose a system pack and recipe (the engine environment lists them), or tick Advanced "
                             "for a profile or config file")
        if self._recipe is None or self._cfg is None:
            raise ValueError("choose a profile or config file that loads")
        if not self._channel_names and not self._meta:
            raise ValueError("the image header could not be read")
        problems = [r.role for r in self._role_rows if r.problem]
        if problems:
            raise ValueError("required channel role(s) unbound: " + ", ".join(problems) +
                             " (pick a recipe whose channel map matches this image, or fix the channel names)")
        sizes = self._meta.get("sizes", {})
        if int(sizes.get("T", 1) or 1) > 1 or max(int(sizes.get("P", 1) or 1), int(self._meta.get("n_scenes", 1) or 1)) > 1:
            raise ValueError("time-lapse and multi-scene files are refused (see the header text)")
        out = self.out_edit.text().strip().strip('"') or str(api.default_out_dir(self._image))
        for folder in _packs.pack_folders(self._listing):
            if _packs.inside(out, folder):
                raise ValueError(f"the output folder {out} lies inside the pack folder {folder}: a pack holds only its "
                                 f"own checked files, so choose an output folder elsewhere")
        return RunPlan(image=self._image, recipe=self._recipe, out=Path(out),
                       xy_stride=int(self.stride_spin.value()), disabled=self.disabled_stages())

    def _engine_for_plan(self, plan: RunPlan) -> EngineRunner | None:
        if plan.recipe.is_pack:
            return self.engine_runner(plan.recipe.pack_id, plan.recipe.recipe)
        return self.engine_runner()

    def _engine_description(self, plan: RunPlan) -> dict | None:
        """The engine line of the commands file, from the version check already made (no engine call here)."""
        runner = self._engine_for_plan(plan)
        if runner is None:
            return None
        v = runner.cached_version()
        if v is None:
            return {"python": runner.python, "line": "version unknown (the engine has not answered its version "
                                                     "check)"}
        return {"python": runner.python, "kind": runner.kind,
                "line": f"{v.get('germquant')} git {v.get('git_sha')} python {v.get('python')}"}

    def export_to(self, folder: str | Path) -> tuple[Path, Path]:
        plan = self.plan()
        self.last_export = plan.export(folder, engine=self._engine_description(plan))
        overlay, commands = self.last_export
        self._status(f"exported {overlay.name} and {commands.name} to {overlay.parent}", GREEN)
        return self.last_export

    def _export_dialog(self) -> None:
        try:
            self.plan()
        except ValueError as e:
            self._status(str(e), RED)
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the exported profile and commands")
        if folder:
            self.export_to(folder)

    def run(self) -> None:
        """Run the plan in a napari thread worker through the engine runner; results load as layers when it
        returns. With no engine chosen, ask for one (and run once it is chosen, if the recipe is still the one
        Run was pressed with) instead of running here."""
        if self._worker is not None:
            self._status("a run is already in progress", AMBER)
            return
        if self._source == PACK and self._current_runner() is None:
            self._ask_before_run()
            return
        if self._source == PACK and self._card_pending:
            self._status("the recipe is still being checked in the engine environment: press Run when its card "
                         "is back", AMBER)
            return
        try:
            plan = self.plan()
        except ValueError as e:
            self._status(str(e), RED)
            return
        runner = self._engine_for_plan(plan)
        if runner is None:
            self._ask_before_run()
            return
        if plan.recipe.is_pack and self._pack_status is not None and not self._pack_status.get("usable"):
            self._status("this recipe cannot run here:\n" + _packs.blocker_text(self._pack_status), RED)
            return
        from napari.qt.threading import create_worker

        own = bool(plan.recipe.is_pack and _settings.recipe_interpreter(plan.recipe.pack_id, plan.recipe.recipe))
        config_hash = plan.config_hash()
        prefer = self._cfg.get("segmentation.nuclei.device") if self._cfg is not None else None
        self.last_result = None
        self.last_error = None
        self._plan = plan
        self.run_btn.setEnabled(False)
        self._t0 = time.monotonic()
        self._timer.start()
        self._bridge.attach()
        msg = f"running germquant on {plan.image.name} with {plan.recipe.label} in {runner.describe()}"
        if plan.disabled:
            msg += " (" + " ".join(f"--no-{n}" for n in plan.disabled) + ")"
        self._status(msg + " ...", None)
        set_viewer_status(self.viewer, msg)
        cancel = threading.Event()
        holder: dict = {}

        def should_stop() -> bool:              # Cancel, or napari quitting (it asks every worker to quit)
            worker = holder.get("worker")
            return cancel.is_set() or bool(worker is not None and getattr(worker, "abort_requested", False))

        def work():
            return execute_run(runner, plan, config_hash=config_hash, prefer=prefer,
                               interpreter_source="recipe" if own else "default", say=_say, should_stop=should_stop)

        self._run_cancel = cancel
        self._run_cancellable = runner.cancellable
        # an explicit errored connection: without one napari re-raises the exception in the Qt event loop
        self._worker = create_worker(
            work,
            _connect={"returned": self._on_returned, "errored": self._on_errored, "finished": self._on_finished},
        )
        holder["worker"] = self._worker
        self.cancel_btn.setEnabled(runner.cancellable)

    def _ask_before_run(self) -> None:
        self._pending_run = True
        self._pending_recipe = self._recipe
        self._status("Choose the engine environment first: Run starts once it is chosen", AMBER)
        self.ask_for_engine()

    def cancel_run(self) -> bool:
        """Cancel: stop the run in progress. Its engine process and every process it started are ended, and the
        run log records image_failed with the cancellation. A run in this napari environment cannot be stopped."""
        if self._worker is None or self._run_cancel is None:
            return False
        if not self._run_cancellable:
            self._status("a run in this napari environment cannot be stopped from the dock; it ends on its own",
                         AMBER)
            return False
        self._run_cancel.set()
        self.cancel_btn.setEnabled(False)
        self._status("cancelling: ending the engine process ...", AMBER)
        return True

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._t0)
        set_viewer_status(self.viewer, f"germquant running ... {elapsed // 60} min {elapsed % 60:02d} s")

    def _on_log_line(self, text: str) -> None:
        set_viewer_status(self.viewer, text[:200])

    def _on_returned(self, res: dict) -> None:
        self.last_result = res
        if res.get("run_log"):
            self.last_run_log = Path(res["run_log"])
        dev = res.get("device_selected")
        if dev:
            self.engine_box.device_label.setText(f"last run: {dev} (asked for {res.get('device_requested')})")
        rs = ResultsSet(Path(res["out_dir"]), res["image_id"])
        expected = self._expected_segmenters(self._plan.recipe if self._plan is not None else None)
        try:
            layers = load_results_layers(rs)
            added = add_layer_tuples(self.viewer, layers)
        except Exception as e:  # noqa: BLE001 - the run itself succeeded; say the load did not
            self._status(f"run finished ({res['n_nuclei']} nuclei) but the results could not be loaded: {e}", AMBER)
            return
        flags = "; ".join(res.get("qc_flags") or []) or "none"
        where = f"\nran in {res.get('via')} {res.get('interpreter')}; run log {res.get('run_log')}"
        self._status(f"done: {res['image_id']}: {res['n_nuclei']} objects, qc_pass={res['qc_pass']}, flags: {flags}\n"
                     f"{len(added)} layer(s) loaded from {res['out_dir']}{where}\n{summary_text(rs, expected)}",
                     GREEN if res["qc_pass"] else AMBER)
        set_viewer_status(self.viewer, f"germquant done: {res['n_nuclei']} objects ({res['image_id']})")

    def _on_errored(self, exc: BaseException) -> None:
        self.last_error = exc
        log_path = getattr(exc, "run_log", None)
        if log_path:
            self.last_run_log = Path(log_path)
        where = f"\nrun log {log_path}" if log_path else ""
        if isinstance(exc, RunCancelled):
            self._status(f"run cancelled: {exc}{where}", AMBER)
            set_viewer_status(self.viewer, "germquant run cancelled")
            return
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1500:]
        self._status(f"run FAILED: {type(exc).__name__}: {exc}{where}\n{tb}", RED)
        set_viewer_status(self.viewer, f"germquant failed: {type(exc).__name__}: {exc}")
        try:
            from napari.utils.notifications import show_error

            show_error(f"germquant run failed: {type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 - no notification manager in a bare Qt test
            pass

    def _on_finished(self) -> None:
        self._timer.stop()
        self._bridge.detach()
        self._worker = None
        self._run_cancel = None
        self.cancel_btn.setEnabled(False)
        self._update_run_button()

    # ------------------------------------------------------------------ status
    def _status(self, text: str, color: QColor | None) -> None:
        self.status_label.setText(text)
        if color is None:
            self.status_label.setStyleSheet("")
        else:
            self.status_label.setStyleSheet(f"color: rgb({color.red()}, {color.green()}, {color.blue()});")
