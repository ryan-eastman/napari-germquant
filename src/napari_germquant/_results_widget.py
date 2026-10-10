"""The Results dock: an existing germquant results folder (one image, or a batch root) as layers."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._qt_common import add_layer_tuples, set_viewer_status
from ._results import ResultsSet, find_results, load_results_layers, summary_text


class ResultsWidget(QWidget):
    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._sets: list[ResultsSet] = []
        self.last_layers: list = []
        root = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("a results folder germquant wrote (or a batch output root)")
        self.folder_edit.editingFinished.connect(lambda: self.set_folder(self.folder_edit.text()))
        row.addWidget(self.folder_edit)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        form.addRow("Folder", row)
        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self._show_summary)
        form.addRow("Image", self.image_combo)
        self.source_check = QCheckBox("also open the source image (at the stride the run used)")
        form.addRow("", self.source_check)
        self.load_btn = QPushButton("Open as layers")
        self.load_btn.clicked.connect(self.load_selected)
        form.addRow("", self.load_btn)
        root.addLayout(form)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.summary)
        root.addStretch(1)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Results folder")
        if path:
            self.set_folder(path)

    def set_folder(self, folder: str | Path) -> list[ResultsSet]:
        text = str(folder).strip().strip('"')
        self.folder_edit.setText(text)
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        self._sets = []
        try:
            self._sets = find_results(text) if text else []
        except FileNotFoundError as e:
            self.summary.setText(str(e))
        for rs in self._sets:
            label = rs.image_id if rs.folder == Path(text) else f"{rs.folder.relative_to(text)} / {rs.image_id}"
            self.image_combo.addItem(label, rs)
        self.image_combo.blockSignals(False)
        if not self._sets and text:
            self.summary.setText("no <image_id>__image_summary.csv in this folder or one level below it")
        self._show_summary(self.image_combo.currentIndex())
        return list(self._sets)

    @property
    def results(self) -> list[ResultsSet]:
        return list(self._sets)

    def selected(self) -> ResultsSet | None:
        data = self.image_combo.currentData()
        return data if isinstance(data, ResultsSet) else None

    def _show_summary(self, index: int = 0) -> None:
        rs = self.selected()
        if rs is None:
            return
        try:
            self.summary.setText(summary_text(rs))
        except Exception as e:  # noqa: BLE001 - a half-written folder must not break the widget
            self.summary.setText(f"{rs.image_id}: could not read the tables: {e}")

    def load_selected(self) -> list:
        rs = self.selected()
        if rs is None:
            self.summary.setText("choose a folder with results first")
            return []
        try:
            layers = load_results_layers(rs, include_source=self.source_check.isChecked())
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.summary.setText(f"could not load {rs.image_id}: {e}")
            return []
        self.last_layers = add_layer_tuples(self.viewer, layers)
        set_viewer_status(self.viewer, f"loaded {len(layers)} layer(s) for {rs.image_id}")
        self.summary.setText(summary_text(rs) + f"\n\nloaded {len(layers)} layer(s)")
        return self.last_layers
