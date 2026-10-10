"""The Training dock: export slices from finished runs, correct their labels one slice at a time, train a
nucleus model on the corrected pairs and register it.

The order of a training round (docs/TIMELAPSE.md, "Segmenting without DAPI: the training loop"):

1. Export: a results folder, a time-lapse series folder or a batch tree goes through
   ``germquant export-training`` (its Python function, in a thread worker) into a training folder of
   ``image/`` and ``labels/`` slice pairs with a ``manifest.csv``.
2. Correct: one pair at a time, an Image layer and an editable Labels layer; Previous and Next move through
   the pairs ``train-nuclei`` will read, Save writes the corrected labels back over ``labels/<name>.tif``
   (atomic, manifest untouched, the save recorded in ``corrections.json``).
3. Train: ``germquant train-nuclei`` in a thread worker on cuda, mps or cpu; the epoch lines go to the
   status bar and the viewer stays usable.
4. Register: ``germquant register-model`` adds the model to the registry, after which a profile names it.

Everything the dock decides is in ``_training`` (no Qt); this module is layout, signals and the layers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _training as T
from ._qt_common import WorkerRunner, format_exception, set_viewer_status

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)

# the metadata key on the two layers this dock adds (the pair they show)
META_KEY = "napari_germquant_training"

DEVICE_TEXT = {"cuda": "cuda (NVIDIA GPU)", "mps": "mps (Apple GPU, Metal)", "cpu": "cpu (slow: hours for 150 epochs)"}


def _path_row(edit: QLineEdit, button: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(edit)
    row.addWidget(button)
    return row


class TrainingWidget(QWidget):
    """napari dock widget. ``napari_viewer`` is injected by napari; None builds the widget without a viewer."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.runner = WorkerRunner(napari_viewer, self)
        self._busy_kind: str | None = None
        self._pairs_dir: Path | None = None
        self._pairs: list[T.Pair] = []
        self._index = -1
        self._valid: dict[str, dict] = {}
        self._loaded_labels: np.ndarray | None = None
        self._current_image: np.ndarray | None = None
        self._current_labels: np.ndarray | None = None
        self.image_layer = None
        self.labels_layer = None
        self._asked_device: str | None = None
        self.last_manifest = None
        self.last_train: dict | None = None
        self.last_register: dict | None = None
        self.last_save: dict | None = None
        self.last_error: BaseException | None = None
        self._build()
        self._update_buttons()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)

        exp_box = QGroupBox("1. Export slices to correct (germquant export-training)")
        exp = QFormLayout(exp_box)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("a results folder, a time-lapse series folder or a batch tree")
        self.source_edit.editingFinished.connect(lambda: self.set_source(self.source_edit.text()))
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse_dir(self.set_source, "Results or series folder"))
        exp.addRow("Results folder", _path_row(self.source_edit, b))
        self.pairs_edit = QLineEdit()
        self.pairs_edit.setPlaceholderText("the training folder: image/, labels/, manifest.csv (created)")
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse_dir(self.set_pairs_dir, "Training folder"))
        exp.addRow("Training folder", _path_row(self.pairs_edit, b))
        orow = QHBoxLayout()
        self.orient_checks: dict[str, QCheckBox] = {}
        for o in T.ORIENTATIONS:
            cb = QCheckBox(o)
            cb.setChecked(True)
            cb.setToolTip("xz and yz slices are resampled along z to the xy pixel size (isotropic)")
            orow.addWidget(cb)
            self.orient_checks[o] = cb
        orow.addStretch(1)
        exp.addRow("Orientations", orow)
        self.slices_spin = QSpinBox()
        self.slices_spin.setRange(1, 1000)
        self.slices_spin.setValue(3)
        self.slices_spin.setToolTip("slices per image and orientation, drawn at random (seeded per image)")
        exp.addRow("Slices per image", self.slices_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        exp.addRow("Seed", self.seed_spin)
        self.min_labels_spin = QSpinBox()
        self.min_labels_spin.setRange(1, 100_000)
        self.min_labels_spin.setValue(3)
        self.min_labels_spin.setToolTip("a slice must cut at least this many nuclei (when too few do, any slice "
                                        "that cuts one is drawn)")
        exp.addRow("Min nuclei per slice", self.min_labels_spin)
        self.max_gb_spin = QDoubleSpinBox()
        self.max_gb_spin.setRange(0.01, 1000.0)
        self.max_gb_spin.setDecimals(2)
        self.max_gb_spin.setValue(2.0)
        self.max_gb_spin.setToolTip("stop once this many GB of slices are written")
        exp.addRow("Stop at (GB)", self.max_gb_spin)
        row = QHBoxLayout()
        self.export_btn = QPushButton("Export slices")
        self.export_btn.clicked.connect(lambda: self.export())
        row.addWidget(self.export_btn)
        self.open_btn = QPushButton("Open training folder")
        self.open_btn.setToolTip("open the pairs of an existing training folder without exporting")
        self.open_btn.clicked.connect(lambda: self.open_pairs())
        row.addWidget(self.open_btn)
        exp.addRow("", row)
        root.addWidget(exp_box)

        cor_box = QGroupBox("2. Correct the labels (one slice at a time)")
        cor = QVBoxLayout(cor_box)
        self.pair_label = QLabel("no training folder open")
        self.pair_label.setWordWrap(True)
        cor.addWidget(self.pair_label)
        row = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(lambda: self.previous_pair())
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(lambda: self.next_pair())
        self.save_btn = QPushButton("Save corrections")
        self.save_btn.setToolTip("write the Labels layer over labels/<name>.tif (atomic; the manifest is not changed)")
        self.save_btn.clicked.connect(lambda: self.save_current())
        self.revert_btn = QPushButton("Revert")
        self.revert_btn.setToolTip("reload the slice's labels from disk, dropping unsaved edits")
        self.revert_btn.clicked.connect(lambda: self.revert_current())
        for w in (self.prev_btn, self.next_btn, self.save_btn, self.revert_btn):
            row.addWidget(w)
        cor.addLayout(row)
        self.corrected_label = QLabel("")
        self.corrected_label.setWordWrap(True)
        cor.addWidget(self.corrected_label)
        hint = QLabel("Fix merges and splits with the Labels layer's tools (paint a new id, fill, erase). A pair "
                      "not worth correcting can be deleted (both files); train-nuclei skips it.")
        hint.setWordWrap(True)
        cor.addWidget(hint)
        root.addWidget(cor_box)

        tr_box = QGroupBox("3. Train a nucleus model (germquant train-nuclei)")
        tr = QFormLayout(tr_box)
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("output folder: <name> (the model), training_log.json, <name>.card.json")
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse_dir(self.model_dir_edit.setText, "Model folder"))
        tr.addRow("Model folder", _path_row(self.model_dir_edit, b))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("default: the model folder's name")
        tr.addRow("Model name", self.name_edit)
        self.base_combo = QComboBox()
        self.base_combo.setEditable(True)
        try:
            bases = T.base_choices()
        except Exception:  # noqa: BLE001 - an unreadable registry still leaves the two built-in choices
            bases = ["cpsam", "none"]
        self.base_combo.addItems(bases)
        self.base_combo.setToolTip("cpsam: Cellpose's stock model; none: random start (smoke tests only); a registered "
                                   "model name; or type the path of a model file")
        tr.addRow("Starting weights", self.base_combo)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100_000)
        self.epochs_spin.setValue(150)
        tr.addRow("Epochs", self.epochs_spin)
        self.split_combo = QComboBox()
        self.split_combo.addItems(["series", "image"])
        self.split_combo.setToolTip("hold out whole series (neighbouring timepoints never straddle the split) or whole "
                                    "images; never single slices")
        tr.addRow("Hold out by", self.split_combo)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 0.95)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)
        tr.addRow("Hold-out fraction", self.val_spin)
        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setRange(0, 2_000_000_000)
        tr.addRow("Split seed", self.split_seed_spin)
        self.device_combo = QComboBox()
        have = T.device_status()
        for dev in T.DEVICES:
            text = DEVICE_TEXT[dev] + ("" if have[dev] else ": not available here (the engine falls back)")
            self.device_combo.addItem(text, dev)
        self.device_combo.setCurrentIndex(T.DEVICES.index(T.default_device()))
        tr.addRow("Device", self.device_combo)
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(lambda: self.train())
        tr.addRow("", self.train_btn)
        root.addWidget(tr_box)

        reg_box = QGroupBox("4. Register the model (germquant register-model)")
        reg = QFormLayout(reg_box)
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("the model file train-nuclei wrote (filled in after Train)")
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse_file(self.model_edit.setText, "Model file", "All files (*)"))
        reg.addRow("Model file", _path_row(self.model_edit, b))
        self.card_edit = QLineEdit()
        self.card_edit.setPlaceholderText("its <name>.card.json")
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse_file(self.card_edit.setText, "Model card", "Model card (*.card.json)"))
        reg.addRow("Model card", _path_row(self.card_edit, b))
        self.reg_name_edit = QLineEdit()
        self.reg_name_edit.setPlaceholderText("default: the card's name, else the file's name")
        reg.addRow("Registry name", self.reg_name_edit)
        self.register_btn = QPushButton("Register model")
        self.register_btn.clicked.connect(lambda: self.register_model())
        reg.addRow("", self.register_btn)
        root.addWidget(reg_box)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status_label)
        root.addStretch(1)

    # ------------------------------------------------------------------ small helpers
    def _status(self, text: str, color: QColor | None = None) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("" if color is None else
                                        f"color: rgb({color.red()}, {color.green()}, {color.blue()});")

    def _browse_dir(self, setter, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            setter(path)

    def _browse_file(self, setter, title: str, filt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if path:
            setter(path)

    @staticmethod
    def _clean(text: str) -> str:
        return str(text).strip().strip('"')

    def _update_buttons(self) -> None:
        busy = self._busy_kind
        have = bool(self._pairs)
        for w in (self.export_btn, self.open_btn, self.train_btn, self.register_btn):
            w.setEnabled(busy is None)
        nav = have and busy != "export"
        self.prev_btn.setEnabled(nav and self._index > 0)
        self.next_btn.setEnabled(nav and self._index < len(self._pairs) - 1)
        self.revert_btn.setEnabled(nav)
        # a model trains on the files as they are on disk and its card records their sha256 at the end, so a
        # save while it trains would leave a card that describes other labels than the model saw
        self.save_btn.setEnabled(nav and busy != "train")

    # ------------------------------------------------------------------ folders
    def set_source(self, path: str | Path) -> None:
        text = self._clean(path)
        self.source_edit.setText(text)
        if text and not self._clean(self.pairs_edit.text()):
            src = Path(text)
            self.set_pairs_dir(src.parent / f"{src.name}_training")

    def set_pairs_dir(self, path: str | Path) -> None:
        text = self._clean(path)
        self.pairs_edit.setText(text)
        if text and not self._clean(self.model_dir_edit.text()):
            p = Path(text)
            self.model_dir_edit.setText(str(p.parent / f"{p.name}_model"))

    # ------------------------------------------------------------------ export
    def export_settings(self) -> T.ExportSettings:
        orients = tuple(o for o, cb in self.orient_checks.items() if cb.isChecked())
        if not orients:
            raise ValueError("tick at least one orientation")
        return T.ExportSettings(orientations=orients, slices_per_image=int(self.slices_spin.value()),
                                seed=int(self.seed_spin.value()), min_labels=int(self.min_labels_spin.value()),
                                max_gb=float(self.max_gb_spin.value()))

    def _ask_overwrite(self, conflicts: list[str]) -> bool:
        shown = "\n".join(conflicts[:12]) + (f"\n... and {len(conflicts) - 12} more" if len(conflicts) > 12 else "")
        answer = QMessageBox.question(
            self, "Corrected slices would be lost",
            f"{len(conflicts)} corrected slice(s) in this training folder belong to images this export would redo; "
            f"their labels would be overwritten or dropped from the manifest:\n\n{shown}\n\n"
            "Export into a new training folder instead (No), or go ahead (Yes)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def export(self, overwrite_corrections: bool | None = None) -> bool:
        """Export slices in a thread worker. ``overwrite_corrections``: what to do when corrected slices of the
        target folder would be undone (None asks in a dialog, False cancels, True goes ahead)."""
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        source = Path(self._clean(self.source_edit.text()))
        pairs = self._clean(self.pairs_edit.text())
        if not self._clean(self.source_edit.text()) or not source.is_dir():
            self._status("choose a results or series folder that exists", RED)
            return False
        if not pairs:
            self._status("choose a training folder (it is created)", RED)
            return False
        pairs_dir = Path(pairs)
        try:
            settings = self.export_settings()
            conflicts = T.export_conflicts(source, pairs_dir)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._status(f"cannot export: {type(e).__name__}: {e}", RED)
            return False
        if conflicts:
            if overwrite_corrections is None:
                overwrite_corrections = self._ask_overwrite(conflicts)
            if not overwrite_corrections:
                self._status(f"export cancelled: {len(conflicts)} corrected slice(s) of {pairs_dir} would be undone; "
                             f"choose a new training folder", AMBER)
                return False
        if self._pairs_dir is not None and self._pairs_dir.resolve() == pairs_dir.resolve() and self.is_dirty():
            self._status("save or revert the slice you are editing first: the export rewrites this folder", AMBER)
            return False
        self.last_manifest = None
        self.last_error = None
        self._busy_kind = "export"
        self._update_buttons()
        self._status(f"exporting slices from {source} into {pairs_dir} ...\n{settings.command(source, pairs_dir)}")
        self.runner.start(lambda: T.export_slices(source, pairs_dir, settings),
                          label=f"exporting training slices from {source.name}",
                          returned=lambda manifest: self._on_exported(manifest, pairs_dir, source, settings),
                          errored=lambda e: self._on_errored(e, "export"), finished=self._on_finished)
        return True

    def _on_exported(self, manifest, pairs_dir: Path, source: Path, settings: T.ExportSettings) -> None:
        self.last_manifest = manifest
        if self.is_dirty():
            # edits made while the export ran are never thrown away: the new pairs open on request
            self._status(f"{T.export_summary(manifest, pairs_dir)}\nThe slice shown has unsaved edits: save or revert "
                         f"them, then press Open training folder.", AMBER)
            return
        try:
            self.set_pairs_dir(pairs_dir)
            n = self.open_pairs(unsaved="discard")
        except Exception as e:  # noqa: BLE001 - the export itself succeeded
            self._status(f"{T.export_summary(manifest, pairs_dir)}\nbut the pairs could not be opened: {e}", AMBER)
            return
        self._status(f"{T.export_summary(manifest, pairs_dir)}\n{n} pair(s) in the folder; correct them below.\n"
                     f"same as: {settings.command(source, pairs_dir)}", GREEN)
        set_viewer_status(self.viewer, f"exported {len(manifest)} slice pair(s) to {pairs_dir}")

    # ------------------------------------------------------------------ the pairs
    def open_pairs(self, path: str | Path | None = None, unsaved: str = "ask") -> int:
        """Load the pair list of a training folder (default: the Training folder field) and show the first pair."""
        if self._index >= 0 and not self._resolve_unsaved(unsaved):
            return len(self._pairs)
        if path is not None:
            self.set_pairs_dir(path)
        text = self._clean(self.pairs_edit.text())
        if not text or not Path(text).is_dir():
            self._status("choose a training folder that exists", RED)
            return 0
        pairs_dir = Path(text)
        try:
            plist = T.load_pair_list(pairs_dir)
        except Exception as e:  # noqa: BLE001 - a damaged manifest: say so, keep the current pairs
            self._status(f"cannot read the pairs of {pairs_dir}: {type(e).__name__}: {e}", RED)
            return len(self._pairs)
        self._pairs_dir = pairs_dir
        self._pairs = plist
        self._valid = T.valid_corrections(pairs_dir)
        self._index = -1
        self._loaded_labels = None
        if not self._pairs:
            self.pair_label.setText(f"no slice pairs in {pairs_dir} (image/*.tif with labels/*.tif)")
            self._update_corrected_label()
            self._update_buttons()
            return 0
        self.go_to(0, unsaved="discard")
        return len(self._pairs)

    @property
    def pairs(self) -> list[T.Pair]:
        return list(self._pairs)

    @property
    def index(self) -> int:
        return self._index

    def current_pair(self) -> T.Pair | None:
        return self._pairs[self._index] if 0 <= self._index < len(self._pairs) else None

    def current_labels(self) -> np.ndarray | None:
        if self.labels_layer is not None and self.viewer is not None and self.labels_layer in self.viewer.layers:
            return np.asarray(self.labels_layer.data)
        return self._current_labels

    def is_dirty(self) -> bool:
        cur = self.current_labels()
        if cur is None or self._loaded_labels is None:
            return False
        return cur.shape != self._loaded_labels.shape or not np.array_equal(cur, self._loaded_labels)

    def _resolve_unsaved(self, unsaved: str) -> bool:
        """True when it is fine to leave the current pair: no edits, or they were saved or discarded."""
        if not self.is_dirty():
            return True
        if unsaved == "ask":
            pair = self.current_pair()
            answer = QMessageBox.question(
                self, "Unsaved corrections", f"Save the corrected labels of {pair.name if pair else 'this slice'}?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save)
            unsaved = {QMessageBox.StandardButton.Save: "save",
                       QMessageBox.StandardButton.Discard: "discard"}.get(answer, "cancel")
        if unsaved == "save":
            return self.save_current() is not None
        return unsaved == "discard"

    def go_to(self, index: int, unsaved: str = "ask") -> bool:
        """Show pair ``index``. ``unsaved`` decides what happens to edits of the current pair: ``ask`` (a
        dialog, only when there are edits), ``save``, ``discard`` or ``cancel``."""
        if not self._pairs:
            return False
        index = max(0, min(int(index), len(self._pairs) - 1))
        if self._index >= 0 and not self._resolve_unsaved(unsaved):
            return False
        pair = self._pairs[index]
        try:
            image, labels = T.read_pair(pair)
        except Exception as e:  # noqa: BLE001 - one unreadable pair must not lock the dock
            self._status(f"cannot read {pair.name}: {type(e).__name__}: {e}", RED)
            return False
        self._index = index
        self._loaded_labels = labels.copy()
        self._current_image = image
        self._current_labels = labels
        self._show_layers(pair, image, labels)
        self._update_pair_label()
        self._update_buttons()
        return True

    def next_pair(self, unsaved: str = "ask") -> bool:
        return self.go_to(self._index + 1, unsaved) if self._index < len(self._pairs) - 1 else False

    def previous_pair(self, unsaved: str = "ask") -> bool:
        return self.go_to(self._index - 1, unsaved) if self._index > 0 else False

    def revert_current(self) -> bool:
        pair = self.current_pair()
        if pair is None:
            return False
        self._loaded_labels = None                 # nothing to protect: reload from disk
        idx, self._index = self._index, -1
        ok = self.go_to(idx, unsaved="discard")
        if ok:
            self._status(f"reverted {pair.name} to the labels on disk")
        return ok

    def _show_layers(self, pair: T.Pair, image: np.ndarray, labels: np.ndarray) -> None:
        if self.viewer is None:
            return
        px = pair.pixel_um or 1.0
        meta = {META_KEY: {"pairs_dir": str(self._pairs_dir), "name": pair.name, "image": str(pair.image),
                           "labels": str(pair.labels), "pixel_um": pair.pixel_um}}
        stem = Path(pair.name).stem
        layers = self.viewer.layers
        if self.image_layer is not None and self.image_layer in layers:
            self.image_layer.data = image
            self.image_layer.scale = (px, px)
            self.image_layer.metadata = meta
            self.image_layer.reset_contrast_limits()
        else:
            self.image_layer = self.viewer.add_image(image, name=f"{stem} image", scale=(px, px), metadata=meta,
                                                     colormap="gray")
        if self.labels_layer is not None and self.labels_layer in layers:
            self.labels_layer.data = labels
            self.labels_layer.scale = (px, px)
            self.labels_layer.metadata = meta
        else:
            self.labels_layer = self.viewer.add_labels(labels, name=f"{stem} labels", scale=(px, px), metadata=meta)
            try:
                self.labels_layer.events.paint.connect(lambda event=None: self._update_pair_label())
            except Exception:  # noqa: BLE001 - only the "unsaved edits" hint depends on it
                pass
        self.image_layer.name = f"{stem} image"
        self.labels_layer.name = f"{stem} labels"
        self.labels_layer.editable = True
        try:
            self.viewer.layers.selection.active = self.labels_layer
        except Exception:  # noqa: BLE001
            pass

    def _update_pair_label(self) -> None:
        pair = self.current_pair()
        if pair is None:
            self.pair_label.setText("no pair shown")
            return
        flags = []
        if pair.name in self._valid:
            flags.append("corrected")
        if self.is_dirty():
            flags.append("unsaved edits")
        tail = f"  [{', '.join(flags)}]" if flags else ""
        self.pair_label.setText(f"pair {self._index + 1} of {len(self._pairs)}: {pair.name}\n{pair.describe()}{tail}")
        self._update_corrected_label()

    def _update_corrected_label(self) -> None:
        n = len(self._pairs)
        k = sum(1 for p in self._pairs if p.name in self._valid)
        self.corrected_label.setText(f"corrected so far: {k} of {n} pair(s) (recorded in {T.CORRECTIONS_FILE}); "
                                     "about 30 across zones and timepoints is the first training set" if n else "")

    def save_current(self) -> dict | None:
        """Write the Labels layer over the pair's labels file (atomic; the manifest is not touched)."""
        pair = self.current_pair()
        if pair is None or self._pairs_dir is None:
            self._status("no pair is shown", AMBER)
            return None
        if self._busy_kind in ("export", "train"):
            self._status("saving is paused while the dock exports or trains on this folder (the model card records "
                         "the sha256 of every pair when training ends)", AMBER)
            return None
        labels = self.current_labels()
        try:
            record = T.save_labels(self._pairs_dir, pair, labels)
        except Exception as e:  # noqa: BLE001 - shown to the user; the file on disk is unchanged
            self._status(f"not saved: {type(e).__name__}: {e}", RED)
            return None
        self.last_save = record
        self._loaded_labels = np.array(labels, copy=True)
        self._valid[pair.name] = record
        self._update_pair_label()
        before = record["n_labels_before"]
        self._status(f"saved {pair.labels} ({before if before is not None else '?'} -> {record['n_labels_after']} "
                     f"nuclei); manifest unchanged", GREEN)
        set_viewer_status(self.viewer, f"saved corrected labels {pair.name}")
        return record

    # ------------------------------------------------------------------ train
    def train_settings(self) -> T.TrainSettings:
        out = self._clean(self.model_dir_edit.text())
        if not out:
            raise ValueError("choose a model folder")
        base = self._clean(self.base_combo.currentText()) or "cpsam"
        return T.TrainSettings(out=Path(out), name=self._clean(self.name_edit.text()) or None, base=base,
                               epochs=int(self.epochs_spin.value()), split_by=self.split_combo.currentText(),
                               val_frac=float(self.val_spin.value()), seed=int(self.split_seed_spin.value()),
                               device=str(self.device_combo.currentData()))

    def select_device(self, device: str) -> None:
        self.device_combo.setCurrentIndex(T.DEVICES.index(device))

    def train(self) -> bool:
        """Train in a thread worker on the pairs of the training folder."""
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        pairs_text = self._clean(self.pairs_edit.text())
        if not pairs_text or not Path(pairs_text).is_dir():
            self._status("choose a training folder with slice pairs", RED)
            return False
        pairs_dir = Path(pairs_text)
        try:
            settings = self.train_settings()
        except ValueError as e:
            self._status(str(e), RED)
            return False
        if self.is_dirty():
            self._status("save or revert the slice you are editing first: training reads the files on disk", AMBER)
            return False
        try:
            plist = T.load_pair_list(pairs_dir)
        except Exception as e:  # noqa: BLE001 - an unreadable manifest: say so before anything starts
            self._status(f"cannot read the pairs of {pairs_dir}: {type(e).__name__}: {e}", RED)
            return False
        if not plist:
            self._status(f"no slice pairs in {pairs_dir} (image/*.tif with labels/*.tif): export some first", RED)
            return False
        notes = self._train_notes(pairs_dir, plist, settings)
        self.last_train = None
        self.last_error = None
        self._asked_device = settings.device
        self._busy_kind = "train"
        self._update_buttons()
        epochs = int(settings.epochs)
        self._status(f"training {settings.model_name()} on {settings.device} ...\n{notes}\n{settings.command(pairs_dir)}")
        self.runner.start(lambda: T.train(pairs_dir, settings),
                          label=f"training {settings.model_name()} ({settings.device})",
                          returned=lambda res: self._on_trained(res, settings, pairs_dir, notes),
                          errored=lambda e: self._on_errored(e, "training"), finished=self._on_finished,
                          line_format=lambda line: T.epoch_progress(line, epochs) or line[:160])
        return True

    @staticmethod
    def _train_notes(pairs_dir: Path, plist: list[T.Pair], settings: T.TrainSettings) -> str:
        """Two facts worth knowing before a long training: how many pairs were corrected here, and whether the
        split can hold anything out at all."""
        done = T.valid_corrections(pairs_dir)
        n_corr = sum(1 for p in plist if p.name in done)
        lines = [f"{n_corr} of {len(plist)} pair(s) corrected in this dock (the others train on the run's own labels)"]
        groups = {p.series_id if settings.split_by == "series" else p.image_id for p in plist}
        if len(groups) < 2:
            hint = (" (hold out by image to score on other timepoints; neighbouring timepoints then straddle the "
                    "split)") if settings.split_by == "series" else ""
            lines.append(f"every pair comes from one {settings.split_by}: nothing is held out and the card will carry "
                         f"no hold-out metrics{hint}")
        return "\n".join(lines)

    def _on_trained(self, res: dict, settings: T.TrainSettings, pairs_dir: Path, notes: str = "") -> None:
        self.last_train = res
        self.model_edit.setText(str(res["model_path"]))
        self.card_edit.setText(str(res["card_path"]))
        self.reg_name_edit.setText(str(res["name"]))
        self._status(f"{T.train_summary(res, settings.device)}\n{notes}\n\nsame as:\n{settings.command(pairs_dir)}",
                     GREEN)
        set_viewer_status(self.viewer, f"model {res['name']} trained on {res['device']}")

    # ------------------------------------------------------------------ register
    def register_model(self) -> bool:
        """Add the model file (and its card) to the local registry, in a thread worker (the sha256 of a
        1.2 GB model takes a few seconds)."""
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        model = self._clean(self.model_edit.text())
        card = self._clean(self.card_edit.text()) or None
        name = self._clean(self.reg_name_edit.text()) or None
        if not model or not Path(model).is_file():
            self._status("choose the model file train-nuclei wrote", RED)
            return False
        self.last_register = None
        self.last_error = None
        self._busy_kind = "register"
        self._update_buttons()
        cmd = T.register_command(model, card, name)
        self.runner.start(lambda: T.register(model, card, name), label=f"registering {name or Path(model).name}",
                          returned=lambda entry: self._on_registered(entry, cmd),
                          errored=lambda e: self._on_errored(e, "register"), finished=self._on_finished)
        return True

    def _on_registered(self, entry: dict, cmd: str) -> None:
        self.last_register = entry
        self._status(f"{T.register_summary(entry)}\nsame as: {cmd}", GREEN)
        set_viewer_status(self.viewer, f"model {entry['name']} registered")

    # ------------------------------------------------------------------ worker plumbing
    def _on_errored(self, exc: BaseException, what: str) -> None:
        self.last_error = exc
        self._status(f"{what} FAILED: {format_exception(exc)}", RED)
        set_viewer_status(self.viewer, f"germquant {what} failed: {type(exc).__name__}: {exc}")
        try:
            from napari.utils.notifications import show_error

            show_error(f"germquant {what} failed: {type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 - no notification manager in a bare Qt test
            pass

    def _on_finished(self) -> None:
        self._busy_kind = None
        self._update_buttons()
