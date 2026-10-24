"""The Recipe tab of the Project dock: the role editor and the parameter form (docs/DESIGN_system_packs.md,
sections 7.3 and 8.5; P6 acceptance 4 and 5; tranche 5 Train B plan, decision D39).

The role editor offers one combo per role of the pack recipe's channel map, listing the channels of a reference
image (the project's first image, or the one chosen in the tab's own list or by ``set_reference_image``) after
the pack map's own binding. Below it a
preview shows, for every image of the project, how each role binds with the choices made: by name, by the index
fallback (amber: the run flags it), unbound (red for a required role) or two roles on one channel (red). The
preview is the engine's dry run, ``project edit DIR --role ROLE=CHANNEL ... --dry-run --json``, so it comes from
the code that writes the map and binds the roles in a run. Red keeps Apply off; Apply is ``project edit`` itself,
which writes channel_map.yaml (the pack map with each edited role named by its exact channel name) and recipe.yaml
in ASCII, and refuses an edit that still has a problem.

The parameter form is built only from the recipe's tunables (pack.yaml ``recipes[].tunables``): a spin box for
a float or an int (the declared range, the unit as suffix), a check box for a bool, a combo for a choice, a line
for a word; any other type is shown read only. Each row shows the project's value and the pack's, with Reset.
Keys recipe.yaml sets that the form does not edit (a tuning result, Use in project, a hand edit) are listed read
only. Apply sends the changed values (``--set``) and the reset keys (``--unset``) in one ``project edit``, which
validates them by origin; every Apply shows the new config_hash, and with every edit reset the project hashes as
the pack recipe again. Back to the pack recipe is ``project edit --revert``.

Where it runs: the tab reads the project in this napari environment (``_params.read_view``) and sends every
dry run and write to the engine runner of the viewer's engine session (``_engine_session``), in a thread worker,
one call at a time, the newest request of each kind first; the GUI thread never waits. While the host says a run
of the project is in progress (``set_running``) the tab takes no edit.

The host: the Project dock (unit P6-PROJECT) shows the tab through one line of its EXTRA_TABS,
``("Recipe", "napari_germquant._recipe_widget:RecipeProjectTab")``. RecipeProjectTab follows the dock's tab
protocol (``project_changed(state)``, ``running_changed(running)``), asks the dock to read the project again
after each write, and leaves the role preview until it is shown. The tab also works alone (``set_project``,
``set_running``, ``set_reference_image``).
"""
from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _params
from ._engine_session import session_for
from ._params import MISSING, RESET
from ._qt_common import WorkerRunner
from ._runner import EngineRefused, RunnerError

log = logging.getLogger("napari_germquant.recipe")

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)
STATE_COLOURS = {"collision": RED, "missing": RED, "index": AMBER}
STATE_TIPS = {
    "collision": "two roles bind this channel: they would read the same pixels (Apply is off)",
    "missing": "a required role binds no channel on this image (Apply is off)",
    "index": "no channel name matched: the channel map's index fallback is used, and the run flags it",
    "unbound": "an optional role with no channel on this image: its stage skips itself",
    "name": "bound by the channel's name",
}
RUNNING_TEXT = "a run of this project is in progress: recipe edits wait until it ends"
WRITING_TEXT = "a write is in progress: the next edit waits until it is done"
# the engine's dry run says changed: false (a channel the pack map's role already names, alone, is no edit)
NO_ROLE_CHANGE_TEXT = "no role change: the pack map already names these channels, so the project's map stays as it is"
INT_LIMIT = 2 ** 31 - 1
FLOAT_LIMIT = 1.0e12


def _run_elsewhere(root) -> bool:
    """Whether a Project dock of this napari runs the project in ``root`` now (unit P6-PROJECT's
    ``_project.run_in_progress``, which every dock of the viewer reads). Looked up only when that module is
    loaded: with no Project dock there is no such run."""
    check = getattr(sys.modules.get("napari_germquant._project"), "run_in_progress", None)
    if root is None or not callable(check):
        return False
    try:
        return bool(check(root))
    except Exception:  # noqa: BLE001 - the dock's bookkeeping failing must not block the tab
        return False


def _error_text(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, (RunnerError, _params.ProjectsMissing)) else f"{type(exc).__name__}: {exc}"


def _colour_css(colour: QColor | None) -> str:
    return "" if colour is None else f"color: rgb({colour.red()}, {colour.green()}, {colour.blue()});"


def _decimals(t: _params.Tunable) -> int:
    """Enough decimals for the declared bounds and the values (2 to 6)."""
    need = 2
    for v in (t.value, t.pack_value, t.lo, t.hi):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not MISSING:
            text = repr(float(v))
            if "e" in text or "E" in text:
                need = 6
            elif "." in text:
                need = max(need, len(text.split(".")[1].rstrip("0")))
    return min(6, need)


def _step_decimals(step: float) -> int:
    """The decimals a spin box needs to move by ``step`` (a step finer than its decimals does nothing)."""
    text = f"{step:.10f}".rstrip("0")
    return len(text.split(".", 1)[1]) if "." in text else 0


def _step(t: _params.Tunable) -> float:
    if t.lo is not None and t.hi is not None and t.hi > t.lo:
        span = float(t.hi - t.lo) / 100.0
        for nice in (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1,
                     0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
            if nice >= span:
                return nice
        return 100.0
    return 0.1


@dataclasses.dataclass
class FormRow:
    """The widgets of one tunable: its label, its value widget (None when shown read only), the pack value, the
    note (edited, set by tuning, read only), and Reset. ``shown`` is what the widget showed after the last load
    (its own rounding), so going back to it is no change."""

    tunable: _params.Tunable
    label: QLabel
    widget: QWidget | None
    pack_label: QLabel
    note_label: QLabel
    reset_btn: QPushButton
    shown: object = None


class RecipeTab(QWidget):
    """The Recipe tab (module docstring). ``session``: the engine session (default: the viewer's); ``reader``:
    ``(root, reference=None) -> RecipeView`` (default ``_params.read_view``), replaced by the widget tests."""

    applied = Signal(dict)       # the engine's record of a write (config_hash, validated, written, removed, ...)
    loaded = Signal()            # the project was read again (after set_project, a reload or a write)

    def __init__(self, napari_viewer=None, parent=None, *, session=None, reader=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session = session if session is not None else session_for(napari_viewer)
        self._reader = reader or _params.read_view
        self._root: Path | None = None
        self._reference: str | None = None
        self._view: _params.RecipeView | None = None
        self._running = False
        self._choices: dict[str, str | None] = {}
        self._pending: dict[str, object] = {}
        self._role_record: tuple | None = None      # (record, roles sent, digest) of the last role dry run
        self._form_record: tuple | None = None      # (record or refusal text, (edits, unset), digest)
        self._want_load: dict | None = None
        self._want_reference: str | None = None
        self._want_roles: dict | None = None
        self._want_form: tuple | None = None
        self._writes: list[dict] = []
        self._job: str | None = None
        self._runner_key: str | None = None
        self._syncing = False
        self.last_record: dict | None = None
        self.last_error: str | None = None          # the project could not be read
        self.engine_error: str | None = None        # no engine to preview and write with: the tab is read only
        self.role_combos: dict[str, QComboBox] = {}
        self.form_rows: dict[str, FormRow] = {}
        self.worker = WorkerRunner(napari_viewer, self)
        self._build()
        changed = getattr(self.session, "changed", None)
        if changed is not None:
            changed.connect(self._on_engine_changed)
        self._show_engine()
        self._update_enabled()

    # ------------------------------------------------------------------ what a host calls
    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def view(self) -> _params.RecipeView | None:
        return self._view

    @property
    def running(self) -> bool:
        return self._running

    def set_project(self, root) -> None:
        """Show the project in ``root`` (None: no project). Its recipe is read again and previewed."""
        self._root = None if root is None or str(root) == "" else Path(root)
        self._view = None
        self._choices, self._pending = {}, {}
        self._role_record = self._form_record = None
        self._want_roles = self._want_form = self._want_reference = None
        self.last_error = self.engine_error = None
        self.last_record = None
        self._rebuild()
        if self._root is None:
            self._status("", None)
            return
        self._status(f"reading the project {self._root} ...", None)
        self._request_load(keep=False)

    def reload(self) -> None:
        """Read the project again (after a change made elsewhere); unapplied choices and values are dropped."""
        if self._root is not None:
            self._choices, self._pending = {}, {}
            self._request_load(keep=False)

    def set_reference_image(self, image_id) -> None:
        """The image whose channels the role combos list."""
        self._reference = None if image_id is None or str(image_id) == "" else str(image_id)
        if self._view is not None and self._reference is not None and self._reference != self._view.reference:
            self._want_reference = self._reference
            self._pump()
        self._sync_reference_combo()

    def set_running(self, running: bool) -> None:
        """Whether a run of this project is in progress (the host says so): no edit while it is."""
        self._running = bool(running)
        if self._running:
            self._status(RUNNING_TEXT, AMBER)
        elif self.status_label.text() == RUNNING_TEXT:
            self._status("", None)
        self._update_enabled()

    # ------------------------------------------------------------------ what a test or a script reads
    @property
    def idle(self) -> bool:
        """No engine call or read is in flight or waiting."""
        return (not self.worker.busy and self._want_load is None and self._want_reference is None
                and self._want_roles is None and self._want_form is None and not self._writes)

    def wait_idle(self, timeout_s: float = 300.0) -> bool:
        """For a script driving the tab: run the Qt event loop until ``idle`` (True) or ``timeout_s`` (False)."""
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

    @property
    def config_hash(self) -> str | None:
        """The project's config_hash now (the engine's after a write, else as read on open)."""
        return None if self._view is None else self._view.config_hash

    def role_choice(self, role: str) -> str | None:
        """The channel chosen for ``role`` (None: the pack map)."""
        return self._choices.get(role)

    def role_choices(self) -> dict[str, str]:
        """What a role Apply sends (``roles=`` of project_edit): every role, "" for the pack map."""
        return _params.role_choices(self._view.roles if self._view else [], self._choices)

    @property
    def role_problems(self) -> list[str]:
        rec = self._current_role_record()
        return [] if rec is None else [str(p) for p in ((rec.get("preview") or {}).get("problems") or [])]

    @property
    def roles_hash(self) -> str | None:
        """The config_hash the role choices give (the engine's dry run), None while it is not back."""
        rec = self._current_role_record()
        return None if rec is None else rec.get("config_hash")

    def preview_states(self) -> dict[str, dict[str, str]]:
        """``{image_id: {role: state}}`` of the preview shown (states: _params.STATES)."""
        rec = self._current_role_record()
        return {} if rec is None or self._view is None else _params.preview_states(rec, self._view.roles)

    def pending(self) -> dict:
        """The form changes not applied yet: {key: value, or RESET}."""
        return dict(self._pending)

    @property
    def form_hash(self) -> str | None:
        """The config_hash the pending form changes give (the engine's dry run), None while it is not back or
        when the engine refused them."""
        cur = self._current_form_record()
        return cur.get("config_hash") if isinstance(cur, dict) else None

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        self.engine_label = QLabel("")
        self.engine_label.setWordWrap(True)
        self.engine_label.setStyleSheet("color: gray;")
        root.addWidget(self.engine_label)
        self.header_label = QLabel("No project.")
        self.header_label.setWordWrap(True)
        root.addWidget(self.header_label)
        line = QHBoxLayout()
        self.hash_label = QLabel("")
        self.badge_label = QLabel("")
        line.addWidget(self.hash_label)
        line.addWidget(self.badge_label)
        line.addStretch(1)
        root.addLayout(line)
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet(_colour_css(AMBER))
        root.addWidget(self.warnings_label)

        roles = QGroupBox("Channel roles")
        rl = QVBoxLayout(roles)
        ref = QHBoxLayout()
        ref.addWidget(QLabel("Reference image (its channels fill the lists):"))
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(self._on_reference_combo)
        ref.addWidget(self.reference_combo, 1)
        rl.addLayout(ref)
        self.roles_box = QWidget()
        self.roles_grid = QGridLayout(self.roles_box)
        self.roles_grid.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.roles_box)
        rl.addWidget(QLabel("How each role binds on every image of the project, with these choices:"))
        self.preview_table = QTableWidget(0, 1)
        self.preview_table.setHorizontalHeaderLabels(["image"])
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rl.addWidget(self.preview_table)
        self.problems_label = QLabel("")
        self.problems_label.setWordWrap(True)
        self.problems_label.setStyleSheet(_colour_css(RED))
        rl.addWidget(self.problems_label)
        self.roles_hash_label = QLabel("")
        self.roles_hash_label.setWordWrap(True)
        rl.addWidget(self.roles_hash_label)
        buttons = QHBoxLayout()
        self.apply_roles_btn = QPushButton("Apply roles")
        self.apply_roles_btn.setToolTip("write channel_map.yaml and recipe.yaml through the engine (project edit "
                                        "--role ...): a new config_hash, and validated: false")
        self.apply_roles_btn.clicked.connect(self.apply_roles)
        self.revert_roles_btn = QPushButton("Revert roles")
        self.revert_roles_btn.setToolTip("back to the pack's channel map (every role edit dropped)")
        self.revert_roles_btn.clicked.connect(self.revert_roles)
        buttons.addWidget(self.apply_roles_btn)
        buttons.addWidget(self.revert_roles_btn)
        buttons.addStretch(1)
        rl.addLayout(buttons)
        root.addWidget(roles)

        params = QGroupBox("Parameters (the tunables the pack recipe declares)")
        pl = QVBoxLayout(params)
        self.form_box = QWidget()
        self.form_grid = QGridLayout(self.form_box)
        self.form_grid.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.form_box)
        self.others_label = QLabel("")
        self.others_label.setWordWrap(True)
        pl.addWidget(self.others_label)
        self.form_hash_label = QLabel("")
        self.form_hash_label.setWordWrap(True)
        pl.addWidget(self.form_hash_label)
        fb = QHBoxLayout()
        self.apply_form_btn = QPushButton("Apply parameters")
        self.apply_form_btn.setToolTip("write the changed values into recipe.yaml through the engine (project edit "
                                       "--set and --unset): a new config_hash, and validated: false")
        self.apply_form_btn.clicked.connect(self.apply_form)
        fb.addWidget(self.apply_form_btn)
        fb.addStretch(1)
        pl.addLayout(fb)
        root.addWidget(params)

        tail = QHBoxLayout()
        self.revert_all_btn = QPushButton("Back to the pack recipe")
        self.revert_all_btn.setToolTip("drop every recipe edit, role edit and project model (project edit --revert): "
                                       "the project hashes as the pack recipe again")
        self.revert_all_btn.clicked.connect(lambda: self.revert_all(confirm=True))
        tail.addWidget(self.revert_all_btn)
        tail.addStretch(1)
        root.addLayout(tail)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addStretch(1)

    def _status(self, text: str, colour: QColor | None) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(_colour_css(colour))

    def _refused(self, why: str) -> None:
        """An edit refused for ``why`` (``_blocked``): the reason shown, and the controls set to match (a run that
        another dock started is not announced to the tab)."""
        self._status(why, AMBER)
        self._update_enabled()

    def _show_engine(self) -> None:
        describe = getattr(self.session, "describe", None)
        warning = getattr(self.session, "warning", None)
        text = describe() if callable(describe) else ""
        warn = warning() if callable(warning) else ""
        self.engine_label.setText("Engine: " + text + (f"\n{warn}" if warn else ""))

    def _on_engine_changed(self) -> None:
        try:
            self._show_engine()
        except RuntimeError:                  # the label is gone (napari closing)
            return
        if self._root is None or self._want_load is not None:
            return
        runner = getattr(self.session, "runner", None)
        current = runner() if callable(runner) else None
        key = getattr(current, "key", None)
        if key is not None and (self.engine_error is not None or (self._runner_key is not None
                                                                  and key != self._runner_key)):
            self._request_load(keep=True)     # an engine now, or another one: its previews are asked for again

    # ------------------------------------------------------------------ rebuilding from the view
    def _rebuild(self) -> None:
        v = self._view
        self._syncing = True
        try:
            if v is None:
                self.header_label.setText("No project." if self._root is None else f"Project {self._root}")
                self.hash_label.setText("")
                self.badge_label.setText("")
                self.warnings_label.setText("")
            else:
                self.header_label.setText(f"Project {v.name} ({v.root}): pack {v.pack}, recipe {v.recipe}")
                self.hash_label.setText(f"config_hash {v.config_hash} ({_params.hash_note(v)})")
                text, level = _params.validated_badge(v.validated)
                self.badge_label.setText(text)
                self.badge_label.setStyleSheet(_colour_css(GREEN if level == "ok" else AMBER))
                self.warnings_label.setText("\n".join(f"warning: {w}" for w in v.warnings))
            self._rebuild_reference_combo()
            self._rebuild_roles()
            self._rebuild_form()
        finally:
            self._syncing = False
        self._show_preview()
        self._show_form_preview()
        self._update_enabled()

    def _rebuild_reference_combo(self) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        if self._view is not None:
            for image_id in self._view.image_ids():
                self.reference_combo.addItem(image_id, image_id)
        self.reference_combo.blockSignals(False)
        self._sync_reference_combo()

    def _sync_reference_combo(self) -> None:
        want = self._view.reference if self._view is not None else None
        self.reference_combo.blockSignals(True)
        for i in range(self.reference_combo.count()):
            if self.reference_combo.itemData(i) == want:
                self.reference_combo.setCurrentIndex(i)
        self.reference_combo.blockSignals(False)

    def _clear_grid(self, grid: QGridLayout) -> None:
        while grid.count():
            item = grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _rebuild_roles(self) -> None:
        self._clear_grid(self.roles_grid)
        self.role_combos = {}
        v = self._view
        if v is None:
            return
        if not v.roles:
            self.roles_grid.addWidget(QLabel("The pack recipe's channel map has no roles to edit."), 0, 0)
            return
        if v.channels_error:
            self.roles_grid.addWidget(QLabel(v.channels_error), 0, 0, 1, 3)
        for i, r in enumerate(v.roles, start=1):
            name = QLabel(f"{r.role}{' *' if r.required else ''}")
            name.setToolTip(f"marker {r.marker}; {'required' if r.required else 'optional'}; the pack map names it "
                            f"{', '.join(r.names) or 'by index only'}"
                            + (f" (index fallback {r.index})" if r.index is not None else ""))
            combo = QComboBox()
            for text, data in _params.combo_items(r, v.channels, v.reference, self._choices.get(r.role)):
                combo.addItem(text, data)
            want = self._choices.get(r.role)
            for j in range(combo.count()):
                if combo.itemData(j) == want:
                    combo.setCurrentIndex(j)
            combo.currentIndexChanged.connect(lambda _i, role=r.role: self._on_role_combo(role))
            self.roles_grid.addWidget(name, i, 0)
            self.roles_grid.addWidget(combo, i, 1)
            self.roles_grid.addWidget(QLabel(r.marker), i, 2)
            self.role_combos[r.role] = combo

    def _rebuild_form(self) -> None:
        self._clear_grid(self.form_grid)
        self.form_rows = {}
        v = self._view
        if v is None:
            self.others_label.setText("")
            return
        if not v.tunables:
            self.form_grid.addWidget(QLabel("The pack recipe declares no tunables."), 0, 0)
        for i, t in enumerate(v.tunables):
            label = QLabel(t.title)
            label.setToolTip(f"{t.key} ({t.type or 'no type'}" + (f", {t.unit}" if t.unit else "") + ")")
            widget = self._make_widget(t)
            pack = QLabel(t.pack_text())
            note = QLabel(t.note_text())
            reset = QPushButton("Reset")
            reset.setToolTip(f"back to the pack's value ({_params.value_text(t.pack_value)})")
            reset.clicked.connect(lambda _c=False, key=t.key: self.reset(key))
            row = FormRow(t, label, widget, pack, note, reset)
            self.form_grid.addWidget(label, i, 0)
            if widget is not None:
                self.form_grid.addWidget(widget, i, 1)
            self.form_grid.addWidget(pack, i, 2)
            self.form_grid.addWidget(note, i, 3)
            self.form_grid.addWidget(reset, i, 4)
            self.form_rows[t.key] = row
            pending = self._pending.get(t.key, MISSING)
            self._restore_widget(row)
            row.shown = self._widget_value(row) if pending is MISSING else self._widget_value_for(row, t.value)
            note.setText(self._note(row))
        self.others_label.setText(
            "" if not v.others else "Set outside the form (read only): " + "; ".join(o.text() for o in v.others))

    def _note(self, row: FormRow) -> str:
        """The row's note: a pending change, else where the project's value comes from, plus the value itself
        when the widget cannot show it (outside the declared range: the spin box would show its limit)."""
        t = row.tunable
        pending = self._pending.get(t.key, MISSING)
        if pending is RESET:
            return "reset to the pack's value at Apply"
        if pending is not MISSING:
            return f"changed: {_params.value_text(pending)} at Apply"
        note = t.note_text()
        if (isinstance(row.widget, (QDoubleSpinBox, QSpinBox)) and isinstance(t.value, (int, float))
                and not isinstance(t.value, bool)
                and not _params._same(self._widget_value_for(row, t.value), self._widget_value(row))):
            note = (note + "; " if note else "") + f"now {_params.value_text(t.value)}, outside the declared range"
        return note

    def _make_widget(self, t: _params.Tunable) -> QWidget | None:
        if not t.editable:
            return None
        if t.type == "float":
            w = QDoubleSpinBox()
            step = _step(t)
            w.setDecimals(max(_decimals(t), _step_decimals(step)))
            w.setRange(float(t.lo) if t.lo is not None else -FLOAT_LIMIT, float(t.hi) if t.hi is not None else FLOAT_LIMIT)
            w.setSingleStep(step)
            if t.unit:
                w.setSuffix(f" {t.unit}")
            w.valueChanged.connect(lambda _v, key=t.key: self._on_widget(key))
            return w
        if t.type == "int":
            w = QSpinBox()
            w.setRange(int(t.lo) if t.lo is not None else -INT_LIMIT, int(t.hi) if t.hi is not None else INT_LIMIT)
            if t.unit:
                w.setSuffix(f" {t.unit}")
            w.valueChanged.connect(lambda _v, key=t.key: self._on_widget(key))
            return w
        if t.type == "bool":
            w = QCheckBox()
            w.toggled.connect(lambda _v, key=t.key: self._on_widget(key))
            return w
        if t.type == "choice":
            w = QComboBox()
            for c in t.choices:
                w.addItem(_params.value_text(c), c)
            if t.value is not MISSING and not any(_params._same(t.value, c) for c in t.choices):
                w.addItem(f"{_params.value_text(t.value)} (not a declared choice)", t.value)
            w.currentIndexChanged.connect(lambda _i, key=t.key: self._on_widget(key))
            return w
        w = QLineEdit()
        w.editingFinished.connect(lambda key=t.key: self._on_widget(key))
        return w

    def _restore_widget(self, row: FormRow) -> None:
        """Show in the row's widget what the form holds for it: the pending value, the pack's for a pending
        reset, else the project's value."""
        t = row.tunable
        pending = self._pending.get(t.key, MISSING)
        self._show_value(row, t.pack_value if pending is RESET else (t.value if pending is MISSING else pending))

    def _show_value(self, row: FormRow, value) -> None:
        """Put ``value`` into the row's widget without taking it for a change."""
        w = row.widget
        if w is None or value is MISSING:
            return
        was = self._syncing
        self._syncing = True
        try:
            if isinstance(w, QDoubleSpinBox) and isinstance(value, (int, float)) and not isinstance(value, bool):
                w.setValue(float(value))
            elif isinstance(w, QSpinBox) and isinstance(value, (int, float)) and not isinstance(value, bool):
                w.setValue(int(value))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QComboBox):
                for i in range(w.count()):
                    if _params._same(w.itemData(i), value):
                        w.setCurrentIndex(i)
            elif isinstance(w, QLineEdit):
                w.setText(str(value))
        finally:
            self._syncing = was

    def _widget_value(self, row: FormRow):
        w = row.widget
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QLineEdit):
            return w.text()
        return MISSING

    def _widget_value_for(self, row: FormRow, value):
        """What the row's widget shows for ``value`` (its rounding), without changing what it shows."""
        w = row.widget
        if isinstance(w, QDoubleSpinBox) and isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), w.decimals())
        return value

    # ------------------------------------------------------------------ the role editor
    def _on_reference_combo(self, _index: int) -> None:
        if self._syncing:
            return
        data = self.reference_combo.currentData()
        if data is not None:
            self.set_reference_image(data)

    def _on_role_combo(self, role: str) -> None:
        if self._syncing:
            return
        combo = self.role_combos.get(role)
        if combo is None:
            return
        blocked = self._blocked()
        if blocked:
            self._restore_combo(role)
            self._refused(blocked)
            return
        self._choices[role] = combo.currentData()
        self._after_role_change()

    def _restore_combo(self, role: str) -> None:
        combo = self.role_combos[role]
        was = self._syncing
        self._syncing = True
        try:
            for j in range(combo.count()):
                if combo.itemData(j) == self._choices.get(role):
                    combo.setCurrentIndex(j)
        finally:
            self._syncing = was

    def choose_channel(self, role: str, channel: str | None) -> None:
        """The programmatic twin of picking in ``role``'s combo: a channel of the reference image, the edit the
        project has, or None (the pack map)."""
        if self._view is None:
            raise ValueError("no project is shown")
        r = self._view.role(role)
        blocked = self._blocked()
        if blocked:
            self._refused(blocked)
            return
        allowed = {None, r.edit, self._choices.get(role), *self._view.channels}
        if channel not in allowed:
            raise ValueError(f"{channel!r} is not a channel of the reference image {self._view.reference} (its "
                             f"channels: {', '.join(self._view.channels) or 'none read'})")
        combo = self.role_combos[role]
        for j in range(combo.count()):
            if combo.itemData(j) == channel:
                combo.setCurrentIndex(j)                  # the combo's own signal makes the change
                return
        raise ValueError(f"{channel!r} is not in the list of the role {role}")

    def _roles_changed(self) -> bool:
        return self._view is not None and self.role_choices() != _params.role_choices(self._view.roles,
                                                                                      self._view.saved_roles())

    def _previews_now(self) -> bool:
        """Whether the role preview is asked for now: the engine's dry run reads the header of every image of the
        project. Always here; RecipeProjectTab waits while it is hidden in the dock."""
        return True

    def _after_role_change(self) -> None:
        if (self._view is not None and self._view.roles and self._current_role_record() is None
                and self._previews_now()):
            self._want_roles = {"roles": self.role_choices(), "digest": self._view.digest}
            self._pump()
        self._show_preview()
        self._update_enabled()

    def _current_role_record(self) -> dict | None:
        if self._role_record is None or self._view is None:
            return None
        record, sent, digest = self._role_record
        if digest != self._view.digest or sent != self.role_choices():
            return None
        return record

    def _show_preview(self) -> None:
        v = self._view
        rec = self._current_role_record()
        roles = v.roles if v is not None else []
        self.preview_table.clear()
        self.preview_table.setColumnCount(1 + len(roles))
        self.preview_table.setHorizontalHeaderLabels(["image", *[f"{r.role}{' *' if r.required else ''}"
                                                                 for r in roles]])
        if v is None or rec is None:
            self.preview_table.setRowCount(0)
            self.problems_label.setText("")
            if v is not None and v.roles:
                self.roles_hash_label.setText("no role preview: there is no engine to ask" if self.engine_error
                                              else "checking the role preview ...")
            else:
                self.roles_hash_label.setText("")
            return
        states = _params.preview_states(rec, roles)
        bindings = (rec.get("preview") or {}).get("bindings") or {}
        ids = [i for i in v.image_ids() if i in bindings] + [i for i in bindings if i not in v.image_ids()]
        self.preview_table.setRowCount(len(ids))
        for row, image_id in enumerate(ids):
            self.preview_table.setItem(row, 0, QTableWidgetItem(image_id))
            for col, r in enumerate(roles, start=1):
                state = states.get(image_id, {}).get(r.role, "unbound")
                item = QTableWidgetItem(_params.cell_text((bindings.get(image_id) or {}).get(r.role)))
                item.setToolTip(STATE_TIPS.get(state, ""))
                colour = STATE_COLOURS.get(state)
                if colour is not None:
                    item.setBackground(QBrush(colour))
                self.preview_table.setItem(row, col, item)
        self.preview_table.resizeColumnsToContents()
        problems = self.role_problems
        if rec.get("refused"):
            problems = [f"the engine refused the preview: {rec['refused']}"]
        self.problems_label.setText("\n".join(f"PROBLEM: {p}" for p in problems)
                                    + ("\nApply is off while a problem is left." if problems else ""))
        warnings = [str(w) for w in rec.get("warnings") or []]
        base = [f"warning: {w}" for w in v.warnings]
        self.warnings_label.setText("\n".join(base + [f"warning: {w}" for w in warnings if f"warning: {w}" not in base]))
        if not self._roles_changed():
            self.roles_hash_label.setText("no role change")
        elif rec.get("refused"):
            self.roles_hash_label.setText("with these roles: no config_hash (the engine refused the preview)")
        elif rec.get("changed") is False:
            self.roles_hash_label.setText(NO_ROLE_CHANGE_TEXT)
        else:
            self.roles_hash_label.setText(f"with these roles: config_hash {rec.get('config_hash')} (now "
                                          f"{v.config_hash})")

    def apply_roles(self) -> bool:
        """Write the role choices through the engine (``project edit --role ...``). False, with the reason in the
        status line, when there is nothing to apply, a run is in progress, the preview is not back, or the
        preview has a problem."""
        why = self._roles_refusal()
        if why:
            self._status(why, AMBER if "problem" not in why else RED)
            return False
        self._queue_write("roles", {"roles": self.role_choices()}, "applying the role choices")
        return True

    def _blocked(self) -> str:
        """Why no edit can be made now (no project read, no usable engine, a run in progress, a write queued or
        running), or "". A run is the host's (``set_running``) or one a Project dock of this napari started
        (``_run_elsewhere``). A write's answer replaces the pending values and the role choices with what was
        written, so an edit taken while it runs would be lost: none is taken then."""
        if self._view is None or self.last_error is not None:
            return "no project is read"
        if self.engine_error:
            return f"read only: {self.engine_error}"
        if self._running or _run_elsewhere(self._root):
            return RUNNING_TEXT
        if self._writes or self._job == "write":
            return WRITING_TEXT
        return ""

    def _roles_refusal(self) -> str:
        blocked = self._blocked()
        if blocked:
            return blocked
        if not self._roles_changed():
            return "no role change to apply"
        rec = self._current_role_record()
        if rec is None:
            return "the role preview is not back yet"
        if rec.get("refused"):
            return f"the engine refused the role preview, so this edit has a problem: {rec['refused']}"
        if rec.get("changed") is False:
            return NO_ROLE_CHANGE_TEXT
        if self.role_problems:
            return "the role edit has a problem left (see the preview): fix it before Apply"
        return ""

    def revert_roles(self) -> bool:
        """Back to the pack's channel map: every role edit dropped, written through the engine as ``project edit
        --unset io.channel_map`` (no role preview, so a problem the pack map itself leaves on an image does not
        refuse it; the form's edits stay), or, with none saved, the unapplied choices dropped."""
        blocked = self._blocked()
        if blocked:
            self._refused(blocked)
            return False
        saved = any(r.edit for r in self._view.roles)
        if not saved:
            if not self._roles_changed():
                return False
            self._choices = self._view.saved_roles()
            self._syncing = True
            try:
                self._rebuild_roles()
            finally:
                self._syncing = False
            self._after_role_change()
            return True
        self._queue_write("roles", {"unset": [_params.CHANNEL_MAP_KEY]}, "reverting the role edits")
        return True

    # ------------------------------------------------------------------ the parameter form
    def _on_widget(self, key: str) -> None:
        if self._syncing:
            return
        row = self.form_rows.get(key)
        if row is None:
            return
        blocked = self._blocked()
        if blocked:                           # the widgets are off then; a change made all the same is put back
            self._restore_widget(row)
            self._refused(blocked)
            return
        value = self._widget_value(row)
        if _params._same(value, row.shown):
            self._pending.pop(key, None)
            self._after_form_change()
            return
        try:
            value = row.tunable.check(value)
        except ValueError as e:
            self._status(str(e), RED)
            return
        self._set_pending(row, value)

    def _set_pending(self, row: FormRow, value) -> None:
        t = row.tunable
        if value is not RESET and t.same(value, t.value):
            self._pending.pop(t.key, None)
        else:
            self._pending[t.key] = value
        self._after_form_change()

    def set_value(self, key: str, value) -> None:
        """The programmatic twin of the row's widget: ``value`` checked by the declaration (ValueError naming the
        key and the rule), shown, and pending until Apply."""
        if self._view is None:
            raise ValueError("no project is shown")
        t = self._view.tunable(key)                      # ValueError for a key the recipe does not declare
        value = t.check(value)
        blocked = self._blocked()
        if blocked:
            self._refused(blocked)
            return
        row = self.form_rows[key]
        self._show_value(row, value)
        self._set_pending(row, value)

    def reset(self, key: str) -> None:
        """Back to the pack's value: an edit of the project is dropped at Apply, a pending change at once."""
        if self._view is None:
            raise ValueError("no project is shown")
        t = self._view.tunable(key)
        blocked = self._blocked()
        if blocked:
            self._refused(blocked)
            return
        row = self.form_rows[key]
        if t.edited:
            self._show_value(row, t.pack_value)
            self._pending[key] = RESET
        else:
            self._show_value(row, t.value)
            self._pending.pop(key, None)
        self._after_form_change()

    def _form_change(self) -> tuple[dict, list[str]]:
        if self._view is None:
            return {}, []
        return _params.form_change(self._view.tunables, self._pending)

    def _after_form_change(self) -> None:
        for key, row in self.form_rows.items():
            t = row.tunable
            pending = self._pending.get(key, MISSING)
            row.note_label.setText(self._note(row))
            row.reset_btn.setEnabled(not self._blocked() and t.editable and (t.edited or pending is not MISSING))
        edits, unset = self._form_change()
        if (edits or unset) and self._current_form_record() is None and self._view is not None:
            self._want_form = (edits, unset, self._view.digest)
            self._pump()
        self._show_form_preview()
        self._update_enabled()

    def _current_form_record(self):
        if self._form_record is None or self._view is None:
            return None
        rec, sent, digest = self._form_record
        if digest != self._view.digest or sent != self._form_change():
            return None
        return rec

    def _show_form_preview(self) -> None:
        if self._view is None:
            self.form_hash_label.setText("")
            return
        edits, unset = self._form_change()
        if not (edits or unset):
            self.form_hash_label.setText("")
            self.form_hash_label.setStyleSheet("")
            return
        cur = self._current_form_record()
        if cur is None:
            self.form_hash_label.setText("checking the new values with the engine ...")
            self.form_hash_label.setStyleSheet("")
        elif isinstance(cur, dict):
            # the dry run's warnings (PLAN 9.4 #6): a recipe file the write would refuse to replace, for one
            warnings = [str(w) for w in cur.get("warnings") or []]
            self.form_hash_label.setText(f"with these values: config_hash {cur.get('config_hash')} (now "
                                         f"{self._view.config_hash})" + "".join(f"\nwarning: {w}" for w in warnings))
            self.form_hash_label.setStyleSheet(_colour_css(AMBER) if warnings else "")
        else:
            self.form_hash_label.setText(str(cur))
            self.form_hash_label.setStyleSheet(_colour_css(RED))

    def apply_form(self) -> bool:
        """Write the pending values and resets through the engine (``project edit --set ... --unset ...``). False,
        with the reason, when there is nothing to apply, a run is in progress, or the engine refused the values in
        its dry run."""
        blocked = self._blocked()
        if blocked:
            self._refused(blocked)
            return False
        edits, unset = self._form_change()
        if not (edits or unset):
            self._status("no parameter change to apply", AMBER)
            return False
        cur = self._current_form_record()
        if cur is not None and not isinstance(cur, dict):
            self._status(f"the engine refuses these values: {cur}", RED)
            return False
        kwargs: dict = {}
        if edits:
            kwargs["edits"] = dict(edits)
        if unset:
            kwargs["unset"] = list(unset)
        self._queue_write("form", kwargs, "applying the parameters")
        return True

    def revert_all(self, confirm: bool = True) -> bool:
        """Back to the pack recipe: every recipe edit, role edit and project model dropped (``project edit
        --revert``). ``confirm`` asks first (the button does; a script passes False)."""
        blocked = self._blocked()
        if blocked:
            self._refused(blocked)
            return False
        if confirm:
            answer = QMessageBox.question(self, "Back to the pack recipe",
                                          "Drop every recipe edit, role edit and project model of this project? "
                                          "Its config_hash goes back to the pack recipe's.")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self._queue_write("revert", {"revert": True}, "going back to the pack recipe")
        return True

    # ------------------------------------------------------------------ enabling
    def _update_enabled(self) -> None:
        ok = self._view is not None and self.last_error is None
        editable = ok and not self._blocked()          # off while a run or a write goes (_blocked)
        for combo in self.role_combos.values():
            combo.setEnabled(editable)
        self.reference_combo.setEnabled(ok)
        for row in self.form_rows.values():
            if row.widget is not None:
                row.widget.setEnabled(editable)
            t = row.tunable
            row.reset_btn.setEnabled(editable and t.editable and (t.edited or t.key in self._pending))
        self.apply_roles_btn.setEnabled(editable and not self._roles_refusal())
        self.revert_roles_btn.setEnabled(editable and (any(r.edit for r in self._view.roles) or self._roles_changed()))
        edits, unset = self._form_change() if ok else ({}, [])
        cur = self._current_form_record()
        self.apply_form_btn.setEnabled(editable and bool(edits or unset) and (cur is None or isinstance(cur, dict)))
        self.revert_all_btn.setEnabled(editable and (self._view.edited or any(r.edit for r in self._view.roles)))

    # ------------------------------------------------------------------ the engine calls (one at a time)
    def _request_load(self, *, keep: bool) -> None:
        self._want_load = {"keep": keep}
        self._pump()

    def _queue_write(self, kind: str, kwargs: dict, label: str) -> None:
        """A write for the project shown now: it runs on that folder even if another project is shown meanwhile."""
        self._writes.append({"kind": kind, "kwargs": kwargs, "label": label, "root": self._root,
                             "reference": self._reference})
        self._status(f"{label} ...", None)
        self._update_enabled()
        self._pump()

    def _runner(self):
        return self.session.require("projects", should_stop=self.worker.should_stop)

    def _pump(self) -> None:
        """Start the next call when the worker is free: a write first, then a load, a reference image, the role
        preview and the form preview (each the newest request of its kind)."""
        if self.worker.busy:
            return
        if self._writes:
            job = self._writes.pop(0)
            self._start("write", lambda: self._write_job(job), job["label"])
            return
        if self._root is None:
            self._want_load = self._want_reference = self._want_roles = self._want_form = None
            return
        root, reference = self._root, self._reference
        if self._want_load is not None:
            req, self._want_load = self._want_load, None
            choices = dict(self._choices) if req.get("keep") else None
            preview = self._previews_now()            # asked here: the worker thread must not touch the widget
            self._start("load", lambda: self._load_job(root, reference, choices, req, preview), "reading the project")
        elif self._want_reference is not None:
            ref, self._want_reference = self._want_reference, None
            self._start("reference", lambda: {"root": root, "view": self._reader(root, reference=ref)},
                        "reading the reference image")
        elif self._want_roles is not None:
            req, self._want_roles = self._want_roles, None
            self._start("roles", lambda: self._preview_job(root, {"roles": req["roles"]}, req),
                        "previewing the roles")
        elif self._want_form is not None:
            edits, unset, digest = self._want_form
            self._want_form = None
            kwargs: dict = {}
            if edits:
                kwargs["edits"] = dict(edits)
            if unset:
                kwargs["unset"] = list(unset)
            self._start("form", lambda: self._preview_job(root, kwargs, {"sent": (edits, unset), "digest": digest}),
                        "checking the parameters")

    def _start(self, kind: str, func, label: str) -> None:
        self._job = kind
        started = self.worker.start(func, label=label,
                                    returned=lambda res: self._guarded(self._on_returned, kind, res),
                                    errored=lambda exc: self._guarded(self._on_errored, kind, exc),
                                    finished=lambda: self._guarded(self._on_finished))
        if not started:                       # cannot happen while _pump checks busy; never lose the request
            self._job = None

    @staticmethod
    def _guarded(fn, *args) -> None:
        """A worker's answer arriving after the tab was closed (its widgets deleted) is dropped."""
        try:
            fn(*args)
        except RuntimeError as e:
            if "deleted" not in str(e):
                raise

    # -- in the worker thread
    def _load_job(self, root, reference, choices, req, preview=True) -> dict:
        """The project read here, then (with ``preview``) the engine's role preview of its current choices. An
        engine that cannot be used (none chosen, too old, no projects feature) leaves the view shown, read only,
        with its message."""
        view = self._reader(root, reference=reference)
        out = {"root": root, "view": view, "record": None, "sent": None, "keep": bool(req.get("keep")),
               "runner_key": None, "engine_error": None}
        try:
            runner = self._runner()
        except RunnerError as e:
            out["engine_error"] = str(e)
            return out
        out["runner_key"] = getattr(runner, "key", None)
        out["sent"] = _params.role_choices(view.roles, choices if choices is not None else view.saved_roles())
        if view.roles and preview:
            try:
                out["record"] = runner.project_edit(root, roles=out["sent"], dry_run=True,
                                                    should_stop=self.worker.should_stop)
            except EngineRefused as e:
                out["record"] = {"refused": str(e), "preview": None}
        return out

    def _preview_job(self, root, kwargs, req) -> dict:
        runner = self._runner()
        try:
            record = runner.project_edit(root, dry_run=True, should_stop=self.worker.should_stop, **kwargs)
        except EngineRefused as e:
            record = str(e)
        return {"root": root, "record": record, **req}

    def _write_job(self, job) -> dict:
        root = job["root"]
        runner = self._runner()
        record = runner.project_edit(root, should_stop=self.worker.should_stop, **job["kwargs"])
        try:
            view = self._reader(root, reference=job.get("reference"))
        except Exception as e:  # noqa: BLE001 - the write is done; the read error is shown
            view = e
        return {"record": record, "view": view, **job}

    # -- back in the GUI thread
    def _on_returned(self, kind: str, res) -> None:
        if kind == "write":
            self._written(res)
            return
        if res.get("root") != self._root:     # made for a project no longer shown
            return
        if kind == "load":
            self._loaded(res)
        elif kind == "reference":
            if self._view is not None:
                self._view = res["view"]
                self._choices = {r.role: self._choices.get(r.role, r.edit) for r in self._view.roles}
                self._rebuild()
                self._after_role_change()
                self._after_form_change()
        elif kind == "roles":
            record = res["record"] if isinstance(res["record"], dict) else {"refused": res["record"], "preview": None}
            self._role_record = (record, res["roles"], res["digest"])
            self._show_preview()
            self._update_enabled()
        elif kind == "form":
            self._form_record = (res["record"], res["sent"], res["digest"])
            self._show_form_preview()
            self._update_enabled()

    def _loaded(self, res: dict) -> None:
        view = res["view"]
        self._runner_key = res.get("runner_key")
        self.engine_error = res.get("engine_error")
        keep = res.get("keep")
        self._view = view
        self.last_error = None
        names = {r.role for r in view.roles}
        if not keep:
            self._choices = view.saved_roles()
            self._pending = {}
        else:
            self._choices = {r.role: self._choices.get(r.role, r.edit) for r in view.roles}
            self._pending = {k: v for k, v in self._pending.items() if any(t.key == k for t in view.tunables)}
        self._choices = {k: v for k, v in self._choices.items() if k in names}
        record = res.get("record")
        self._role_record = None if record is None else (record, res["sent"], view.digest)
        self._form_record = None
        if self.engine_error:
            self._status(f"read only: {self.engine_error}", RED)
        elif isinstance(record, dict) and record.get("refused"):
            self._status(f"the engine refused the role preview: {record['refused']}", RED)
        elif self.status_label.text().startswith("reading the project"):
            self._status("", None)
        self._rebuild()
        if not self.engine_error:
            self._after_role_change()
            self._after_form_change()
        if self._reference is not None and self._reference != view.reference and self._reference in view.image_ids():
            self._want_reference = self._reference          # chosen while this read was under way
        self.loaded.emit()

    def _written(self, res: dict) -> None:
        record = res["record"] if isinstance(res["record"], dict) else {"result": res["record"]}
        self.last_record = record
        if res.get("root") != self._root:     # the project shown changed meanwhile: the write is still announced
            self.applied.emit(record)
            return
        view = res["view"]
        kind = res["kind"]
        if isinstance(view, BaseException):
            self.last_error = _error_text(view)
            self._view = None
            self._rebuild()
            self._status(f"{_params.written_text(record)}; the project could not be read again: {self.last_error}",
                         RED)
            self.applied.emit(record)
            return
        self._view = view
        self.last_error = None
        if kind in ("roles", "revert"):
            self._choices = view.saved_roles()
            sent = res["kwargs"].get("roles")
            preview = record.get("preview")
            self._role_record = (record, sent, view.digest) if (preview and sent is not None
                                                                 and sent == self.role_choices()) else None
        else:
            self._choices = {r.role: self._choices.get(r.role, r.edit) for r in view.roles}
            self._role_record = self._carried_preview(view, record)
        if kind in ("form", "revert"):
            self._pending = {}
        else:
            self._pending = {k: v for k, v in self._pending.items() if any(t.key == k for t in view.tunables)}
        self._form_record = None
        if record.get("config_hash") and view.config_hash != record.get("config_hash"):
            log.info("the engine wrote config_hash %s; this environment reads %s", record.get("config_hash"),
                     view.config_hash)
            view.config_hash = record.get("config_hash")      # the engine's hash is the project's
        self._rebuild()
        self._after_role_change()
        self._after_form_change()
        self._status(_params.written_text(record), GREEN)
        self.applied.emit(record)
        self.loaded.emit()

    def _carried_preview(self, view: _params.RecipeView, record: dict) -> tuple | None:
        """After a form write the role bindings are what they were (a tunable binds no channel), so the preview of
        the saved roles carries over, with the new config_hash; with role choices pending a new dry run is due."""
        prev = self._role_record
        saved = _params.role_choices(view.roles, view.saved_roles())
        if prev is None or prev[1] != saved or self.role_choices() != saved:
            return None
        rec = prev[0]
        if not isinstance(rec, dict) or rec.get("refused") or rec.get("preview") is None:
            return None
        return ({**rec, "config_hash": record.get("config_hash") or view.config_hash, "changed": False}, saved,
                view.digest)

    def _on_errored(self, kind: str, exc: BaseException) -> None:
        text = _error_text(exc)
        if kind == "load":
            self.last_error = text
            self._view = None
            self._rebuild()
            self._status(text, RED)
        elif kind == "reference":
            self._status(f"the reference image could not be read: {text}", AMBER)
        else:
            self._status(text, RED)
            if kind == "write" and self._root is not None and self._want_load is None:
                # a write that stopped part way changed files (PLAN 9.4 #6), a refused one none: read the project
                # again, keeping the choices and values not applied, so no preview is keyed to an old digest
                self._request_load(keep=True)
        self._update_enabled()

    def _on_finished(self) -> None:
        self._job = None
        self._update_enabled()
        self._pump()


class RecipeProjectTab(RecipeTab):
    """The Recipe tab as a tab of the Project dock, in the dock's own tab protocol (unit P6-PROJECT: a class named
    in its EXTRA_TABS, made with the dock): ``project_changed(state)`` at every new reading of the open project
    (None when it closes) and ``running_changed(running)`` when a run starts or ends. The tab takes the dock's
    viewer and engine session; after each write it asks the dock to read the project again (``dock.reload``), so
    the header and the status board follow the new config_hash. While the tab is hidden (another tab of the dock
    is shown) it follows the dock's readings, but the role preview, a dry run that reads every image's header,
    waits until it is shown."""

    def __init__(self, dock) -> None:
        super().__init__(getattr(dock, "viewer", None), dock, session=getattr(dock, "session", None))
        self.dock = dock
        self.applied.connect(self._reload_dock)

    def _previews_now(self) -> bool:
        return self.isVisible()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        if self._view is not None and self.last_error is None and not self.engine_error:
            self._after_role_change()          # the role preview that waited while the tab was hidden

    def project_changed(self, state) -> None:
        root = getattr(state, "root", None) if state is not None else None
        if root is None:
            if self.root is not None:
                self.set_project(None)
            return
        if self.root is None or Path(root) != self.root:
            self.set_project(root)
            return
        manifest = getattr(state, "manifest", None)
        if self._view is not None and isinstance(manifest, dict) and manifest == self._view.manifest:
            return                             # the manifest this tab read last: no write since (dicts, no file read)
        if self._want_load is None:
            self._request_load(keep=True)      # another write (the dock's, a tuning Apply): the recipe again

    def running_changed(self, running: bool) -> None:
        self.set_running(running)

    def _reload_dock(self, _record: dict) -> None:
        reload = getattr(self.dock, "reload", None)
        if callable(reload):
            reload()
