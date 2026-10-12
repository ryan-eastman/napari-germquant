"""The Results dock: an existing germquant results folder (one image, or a batch root) as layers.

For a system pack run (docs/DESIGN_system_packs.md, section 8.3) the dock also shows the summary line (N cells
or nuclei, M spots, mean spots per cell or nucleus), a Provenance box (pack id and version, recipe, config_hash,
model sha256, git sha and the device used) and a banner when the labels were made by another segmenter than
the one the pack's recipe declares (the classical watershed after Cellpose failed, for example).
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._qt_common import add_layer_tuples, set_viewer_status
from ._results import (
    ResultsSet,
    fallback_banner,
    find_results,
    load_results_layers,
    provenance_text,
    summary_line,
    summary_text,
)

BANNER_STYLE = ("background-color: rgb(200, 60, 60); color: white; padding: 4px; border-radius: 3px;")


class ResultsWidget(QWidget):
    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._sets: list[ResultsSet] = []
        self.last_layers: list = []
        self.expected_segmenters: list[str] | None = None
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
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(BANNER_STYLE)
        self.banner.setHidden(True)
        root.addWidget(self.banner)
        self.summary_line = QLabel("")
        self.summary_line.setWordWrap(True)
        self.summary_line.setStyleSheet("font-weight: bold;")
        root.addWidget(self.summary_line)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.summary)
        self.provenance_box = QGroupBox("Provenance")
        prov_lay = QVBoxLayout(self.provenance_box)
        self.provenance = QLabel("")
        self.provenance.setWordWrap(True)
        self.provenance.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        prov_lay.addWidget(self.provenance)
        root.addWidget(self.provenance_box)
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
            self._clear_details()
            self.summary.setText("no <image_id>__image_summary.csv in this folder or one level below it")
        self._show_summary(self.image_combo.currentIndex())
        return list(self._sets)

    @property
    def results(self) -> list[ResultsSet]:
        return list(self._sets)

    def selected(self) -> ResultsSet | None:
        data = self.image_combo.currentData()
        return data if isinstance(data, ResultsSet) else None

    def _clear_details(self) -> None:
        self.banner.setText("")
        self.banner.setHidden(True)
        self.summary_line.setText("")
        self.provenance.setText("")

    def _show_details(self, rs: ResultsSet) -> None:
        banner = fallback_banner(rs, self.expected_segmenters)
        self.banner.setText(banner or "")
        self.banner.setHidden(not banner)
        try:
            self.summary_line.setText(summary_line(rs))
        except Exception as e:  # noqa: BLE001 - a half-written folder must not break the widget
            self.summary_line.setText(f"could not read the tables: {e}")
        self.provenance.setText(provenance_text(rs))

    def _show_summary(self, index: int = 0) -> None:
        rs = self.selected()
        if rs is None:
            return
        try:
            self.summary.setText(summary_text(rs, self.expected_segmenters))
            self._show_details(rs)
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
        self.summary.setText(summary_text(rs, self.expected_segmenters) + f"\n\nloaded {len(layers)} layer(s)")
        self._show_details(rs)
        return self.last_layers
