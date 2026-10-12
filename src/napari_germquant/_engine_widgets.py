"""Qt pieces of the Run dock for system packs and engine runners (docs/DESIGN_system_packs.md, sections 8.1 and
8.2): the engine box (which environment and device run), the dialog that asks for the engine interpreter, and
the pack picker with its card. Layout and rendering only: the dock (``_run_widget``) makes the engine calls and
decides; the card text comes from ``_packs``.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from . import _packs
from ._packs import PackEntry
from ._settings import INPROCESS

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)
GREY = QColor(140, 140, 140)

DEVICE_TEXT = {"auto": "auto (the engine picks CUDA, then Apple Metal, then the CPU)", "cuda": "cuda (NVIDIA GPU)",
               "mps": "mps (Apple GPU, Metal)", "cpu": "cpu"}


def _color_css(color: QColor | None) -> str:
    return "" if color is None else f"color: rgb({color.red()}, {color.green()}, {color.blue()});"


class InterpreterDialog(QDialog):
    """Asks which Python runs germquant: one of the detected interpreters, any Python picked with Browse, or
    (explicitly) this napari environment in process. For one recipe it can also go back to the dock's engine.
    The dock opens it with ``open()`` (window-modal, never blocking the event loop) and reads ``choice()``
    when it is accepted: a Python path, ``inprocess``, or an empty string (the dock's engine, for a recipe)."""

    def __init__(self, candidates: list[dict], *, title: str, intro: str, current: str | None = None,
                 allow_inprocess: bool = True, allow_default: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("germquant_interpreter_dialog")
        self._allow_inprocess = bool(allow_inprocess)
        self._allow_default = bool(allow_default)
        lay = QVBoxLayout(self)
        text = QLabel(intro)
        text.setWordWrap(True)
        lay.addWidget(text)
        self.list = QListWidget()
        for c in candidates:
            self._add(str(c["python"]), str(c.get("why", "")))
        lay.addWidget(self.list)
        row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setToolTip("the Python of the environment germquant is installed into, for example "
                                   "<engine checkout>/.venv/Scripts/python.exe or .venv/bin/python")
        self.browse_btn.clicked.connect(self._browse)
        row.addWidget(self.browse_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.inprocess_check = QCheckBox("Run in this napari environment instead (in process): only when it holds "
                                         "the engine's models, Cellpose and SpotMAX")
        self.inprocess_check.setVisible(allow_inprocess)
        self.inprocess_check.toggled.connect(self._update)
        lay.addWidget(self.inprocess_check)
        self.default_check = QCheckBox("Use the dock's engine for this recipe (forget its own interpreter)")
        self.default_check.setVisible(allow_default)
        self.default_check.toggled.connect(self._update)
        lay.addWidget(self.default_check)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.list.currentItemChanged.connect(lambda *_: self._update())
        if current == INPROCESS:
            self.inprocess_check.setChecked(True)
        elif current:
            self.select_path(current)
        elif self.list.count():
            self.list.setCurrentRow(0)
        self._update()

    def _add(self, python: str, why: str) -> QListWidgetItem:
        item = QListWidgetItem(f"{python}\n    {why}" if why else python)
        item.setData(Qt.ItemDataRole.UserRole, python)
        self.list.addItem(item)
        return item

    def candidates(self) -> list[str]:
        return [str(self.list.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.list.count())]

    def select_path(self, path: str | Path) -> None:
        path = str(path)
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == path:
                self.list.setCurrentRow(i)
                break
        else:
            self.list.setCurrentItem(self._add(path, "chosen with Browse"))
        self.inprocess_check.setChecked(False)
        self.default_check.setChecked(False)
        self._update()

    def choose_inprocess(self, on: bool = True) -> None:
        self.inprocess_check.setChecked(on)

    def choose_default(self, on: bool = True) -> None:
        self.default_check.setChecked(on)

    def choice(self) -> str | None:
        if self._allow_default and self.default_check.isChecked():
            return ""
        if self._allow_inprocess and self.inprocess_check.isChecked():
            return INPROCESS
        item = self.list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    def _update(self) -> None:
        self.list.setEnabled(not (self._allow_inprocess and self.inprocess_check.isChecked())
                             and not (self._allow_default and self.default_check.isChecked()))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self.choice() is not None)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Python of the germquant environment", "",
                                              "Python (python.exe python python3*);;All files (*)")
        if path:
            self.select_path(path)


class EngineBox(QGroupBox):
    """Which environment and device run: always visible at the top of the Run dock."""

    choose_clicked = Signal()
    device_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Engine: where germquant runs", parent)
        form = QFormLayout(self)
        self.runs_label = QLabel("")
        self.runs_label.setWordWrap(True)
        self.runs_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Runs in", self.runs_label)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet(_color_css(AMBER))
        self.warn_label.setVisible(False)
        form.addRow("", self.warn_label)
        row = QHBoxLayout()
        self.device_combo = QComboBox()
        for value, text in DEVICE_TEXT.items():
            self.device_combo.addItem(text, value)
        self.device_combo.setToolTip("the device the run asks for (GERMQUANT_DEVICE); an accelerator the computer "
                                     "lacks falls back to the next one, and the run log records the one used")
        self.device_combo.currentIndexChanged.connect(lambda _i: self.device_changed.emit(self.device()))
        row.addWidget(self.device_combo, 1)
        self.choose_btn = QPushButton("Choose engine...")
        self.choose_btn.setToolTip("the Python environment that runs germquant on this computer (stored per machine)")
        self.choose_btn.clicked.connect(self.choose_clicked.emit)
        row.addWidget(self.choose_btn)
        form.addRow("Device", row)
        self.device_label = QLabel("")
        self.device_label.setWordWrap(True)
        form.addRow("", self.device_label)

    def show_engine(self, text: str, color: QColor | None = None) -> None:
        self.runs_label.setText(text)
        self.runs_label.setStyleSheet(_color_css(color))

    def show_warning(self, text: str) -> None:
        self.warn_label.setText(text)
        self.warn_label.setVisible(bool(text))

    def device(self) -> str:
        return str(self.device_combo.currentData() or "auto")

    def set_device(self, value: str) -> None:
        self.device_combo.blockSignals(True)
        i = self.device_combo.findData(value)
        self.device_combo.setCurrentIndex(i if i >= 0 else 0)
        self.device_combo.blockSignals(False)


class PackPicker(QGroupBox):
    """The System pack box: a pack combo, its recipes, the card, Get models, Get example and the recipe's own
    interpreter. ``selection_changed`` fires when the user picks a pack or a recipe."""

    selection_changed = Signal()
    refresh_clicked = Signal()
    get_models_clicked = Signal()
    get_example_clicked = Signal()
    interpreter_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__("System pack", parent)
        self._entries: list[PackEntry] = []
        self._statuses: dict[tuple[str, str], dict] = {}
        form = QFormLayout(self)
        row = QHBoxLayout()
        self.pack_combo = QComboBox()
        self.pack_combo.currentIndexChanged.connect(self._on_pack)
        row.addWidget(self.pack_combo, 1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("list the packs again (python -m germquant.packs list in the engine environment)")
        self.refresh_btn.clicked.connect(self.refresh_clicked.emit)
        row.addWidget(self.refresh_btn)
        form.addRow("Pack", row)
        self.recipe_combo = QComboBox()
        self.recipe_combo.currentIndexChanged.connect(lambda _i: self.selection_changed.emit())
        form.addRow("Recipe", self.recipe_combo)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        form.addRow(self.message)
        self.card = QLabel("")
        self.card.setWordWrap(True)
        self.card.setTextFormat(Qt.TextFormat.RichText)
        self.card.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.card)
        btns = QHBoxLayout()
        self.get_models_btn = QPushButton("Get models")
        self.get_models_btn.setToolTip("put the recipe's models into the engine's model store (its own store, the "
                                       "Cellpose cache, then the pinned download; every file checked by sha256)")
        self.get_models_btn.clicked.connect(self.get_models_clicked.emit)
        btns.addWidget(self.get_models_btn)
        self.get_example_btn = QPushButton("Get example")
        self.get_example_btn.setToolTip("fetch the pack's example image, assemble it and open it")
        self.get_example_btn.clicked.connect(self.get_example_clicked.emit)
        btns.addWidget(self.get_example_btn)
        self.interpreter_btn = QPushButton("Recipe interpreter...")
        self.interpreter_btn.setToolTip("run this recipe in another Python environment (for example one with an "
                                        "optional backend installed); stored per machine")
        self.interpreter_btn.clicked.connect(self.interpreter_clicked.emit)
        btns.addWidget(self.interpreter_btn)
        form.addRow(btns)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        form.addRow(self.info_label)
        self._set_buttons(False, False, False)

    # ---- entries ----
    @property
    def entries(self) -> list[PackEntry]:
        return list(self._entries)

    def set_entries(self, entries: list[PackEntry], select: tuple[str | None, str | None] | None = None) -> None:
        keep = select or (self.current_pack_id(), self.current_recipe_id())
        self._entries = list(entries)
        self._statuses = {}
        self.pack_combo.blockSignals(True)
        self.pack_combo.clear()
        for i, e in enumerate(self._entries):
            self.pack_combo.addItem(e.label, i)
            if not e.usable_manifest:
                self._grey(self.pack_combo, self.pack_combo.count() - 1, "\n".join(e.problems) or e.state)
        self.pack_combo.blockSignals(False)
        index = 0
        if keep and keep[0]:
            for i, e in enumerate(self._entries):
                if e.id == keep[0]:
                    index = i
                    if e.usable_manifest:
                        break
        if self._entries:
            self.pack_combo.setCurrentIndex(index)
        self._fill_recipes(keep[1] if keep else None)
        if not self._entries:
            self.card.setText("")
            self._set_buttons(False, False, False)

    def current_entry(self) -> PackEntry | None:
        i = self.pack_combo.currentData()
        return self._entries[i] if isinstance(i, int) and 0 <= i < len(self._entries) else None

    def current_pack_id(self) -> str | None:
        e = self.current_entry()
        return e.id if e is not None else None

    def current_recipe_id(self) -> str | None:
        data = self.recipe_combo.currentData()
        return str(data) if data else None

    def select(self, pack_id: str, recipe: str | None = None, version: str | None = None) -> None:
        for i, e in enumerate(self._entries):
            if e.id == pack_id and (version is None or e.version == version):
                self.pack_combo.blockSignals(True)
                self.pack_combo.setCurrentIndex(i)
                self.pack_combo.blockSignals(False)
                self._fill_recipes(recipe)
                return
        raise ValueError(f"no pack {pack_id!r}" + (f" version {version}" if version else "") + " in the list")

    def select_recipe(self, recipe: str) -> None:
        i = self.recipe_combo.findData(recipe)
        if i < 0:
            raise ValueError(f"no recipe {recipe!r} in pack {self.current_pack_id()!r}")
        self.recipe_combo.setCurrentIndex(i)

    def _on_pack(self, _index: int) -> None:
        self._fill_recipes(None)

    def _fill_recipes(self, recipe: str | None) -> None:
        e = self.current_entry()
        self.recipe_combo.blockSignals(True)
        self.recipe_combo.clear()
        if e is not None:
            for r in e.recipes:
                rid = str(r.get("id"))
                text = f"{rid}{' (default)' if r.get('default') else ''}: {r.get('title', '')} [{r.get('dims', '3d')}]"
                self.recipe_combo.addItem(text, rid)
                st = self._statuses.get((e.ref, rid))
                if st is not None and not st.get("usable"):
                    self._grey(self.recipe_combo, self.recipe_combo.count() - 1, _packs.blocker_text(st))
            want = recipe or e.default_recipe_id()
            i = self.recipe_combo.findData(want) if want else -1
            self.recipe_combo.setCurrentIndex(i if i >= 0 else 0)
        self.recipe_combo.blockSignals(False)
        self.selection_changed.emit()

    # ---- rendering ----
    @staticmethod
    def _grey(combo: QComboBox, index: int, why: str) -> None:
        model = combo.model()
        item = model.item(index) if hasattr(model, "item") else None
        if item is not None:
            item.setForeground(QBrush(GREY))
            item.setToolTip(why)

    @staticmethod
    def _ungrey(combo: QComboBox, index: int) -> None:
        model = combo.model()
        item = model.item(index) if hasattr(model, "item") else None
        if item is not None:
            item.setData(None, Qt.ItemDataRole.ForegroundRole)
            item.setToolTip("")

    def is_greyed(self, combo: QComboBox, index: int) -> bool:
        model = combo.model()
        item = model.item(index) if hasattr(model, "item") else None
        return item is not None and item.data(Qt.ItemDataRole.ForegroundRole) is not None

    def show_message(self, text: str, color: QColor | None = None) -> None:
        self.message.setText(text)
        self.message.setStyleSheet(_color_css(color))

    def show_status(self, entry: PackEntry, recipe_id: str | None, status: dict | None, *,
                    engine_text: str = "") -> None:
        """Render the card for ``entry``/``recipe_id`` with the check ``status`` (None: not checked); grey the
        recipe (and a broken pack) with the reason when it cannot run."""
        if status is not None and recipe_id:
            self._statuses[(entry.ref, recipe_id)] = status
        self.card.setText(_packs.card_html(entry, recipe_id, status, engine_text=engine_text))
        ri = self.recipe_combo.findData(recipe_id) if recipe_id else -1
        if ri >= 0:
            if status is not None and not status.get("usable"):
                self._grey(self.recipe_combo, ri, _packs.blocker_text(status))
            else:
                self._ungrey(self.recipe_combo, ri)
        pi = self.pack_combo.currentIndex()
        if pi >= 0 and not entry.usable_manifest:
            self._grey(self.pack_combo, pi, _packs.blocker_text(status) or entry.state)
        exs = entry.examples() if entry.usable_manifest else []
        if exs:
            ex = exs[0]
            tip = (f"{ex['id']}: public data fetched on demand, licence {ex.get('licence', 'not stated')}; "
                   f"{ex.get('attribution', '')}" if ex["kind"] == "remote" else
                   f"{ex['id']}: the synthetic example inside the pack (opened from there)")
            self.get_example_btn.setToolTip(tip)
        self._set_buttons(entry.usable_manifest and _packs.models_need_fetch(status), bool(exs),
                          entry.usable_manifest)

    def _set_buttons(self, models: bool, example: bool, interpreter: bool) -> None:
        self.get_models_btn.setEnabled(models)
        self.get_example_btn.setEnabled(example)
        self.interpreter_btn.setEnabled(interpreter)

    def set_busy(self, busy: bool) -> None:
        for w in (self.pack_combo, self.recipe_combo, self.refresh_btn):
            w.setEnabled(not busy)
        if busy:
            self._set_buttons(False, False, False)
