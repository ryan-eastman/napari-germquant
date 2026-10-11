"""The Time-lapse dock: a finished series folder as layers, raw or registered, plus Register and Link.

Open reads the folder ``germquant run SERIES.nd2 --t ... --out FOLDER`` wrote (``timelapse.layout``) and
adds the per-timepoint nucleus labels as one lazy time-stacked Labels layer, the reference volumes as an
Image layer when the series kept them, and the track points as a Tracks layer (plus the nuclei without a
track as Points) once the series is linked. The coordinates toggle switches every layer between the raw
microscope frame and the registered frame (raw minus the cumulative drift of each timepoint). Register and
Link run ``germquant register`` and ``germquant track-link`` (their Python functions, in thread workers)
and reload what they wrote; Register's mode is rigid (one shift per timepoint) or piecewise (one shift per
axial segment of the germline, for a germline that slides inside the worm), and Link's registration says
which of the two tables it subtracts. Everything the dock decides is in ``_timelapse`` (no Qt).
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _timelapse as TL
from ._qt_common import WorkerRunner, add_layer_tuples, format_exception, set_viewer_status

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)

LAYER_KINDS = ("reference", "labels", "unlinked", "tracks")      # bottom to top


class TimelapseWidget(QWidget):
    """napari dock widget. ``napari_viewer`` is injected by napari; None builds the widget without a viewer."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.runner = WorkerRunner(napari_viewer, self)
        self.view: TL.SeriesView | None = None
        self.layers: dict[str, object] = {}
        self.last_register: dict | None = None
        self.last_link: dict | None = None
        self.last_error: BaseException | None = None
        self._build()
        self._update_buttons()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)

        box = QGroupBox("Series folder (the --out of germquant run SERIES --t ...)")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("a folder with <stem>__series.json and t000/, t001/, ...")
        row.addWidget(self.folder_edit)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(lambda: self.open_folder())
        row.addWidget(self.open_btn)
        form.addRow("Folder", row)
        self.info_label = QLabel("no series open")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.info_label)
        root.addWidget(box)

        lay_box = QGroupBox("Layers")
        lay = QFormLayout(lay_box)
        crow = QHBoxLayout()
        self.raw_radio = QRadioButton("raw (microscope frame)")
        self.reg_radio = QRadioButton("registered (drift removed)")
        self.raw_radio.setChecked(True)
        self._coord_group = QButtonGroup(self)
        self._coord_group.addButton(self.raw_radio)
        self._coord_group.addButton(self.reg_radio)
        self.reg_radio.toggled.connect(lambda checked: self.refresh_layers())
        crow.addWidget(self.raw_radio)
        crow.addWidget(self.reg_radio)
        lay.addRow("Coordinates", crow)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 16)
        self.stride_spin.setToolTip("show every n-th pixel of the label images in y and x (1 = full resolution); "
                                    "the reference volumes are already at their own stride")
        self.stride_spin.valueChanged.connect(lambda v: self.refresh_layers(only=("labels",)))
        lay.addRow("Labels xy stride", self.stride_spin)
        krow = QHBoxLayout()
        self.kind_checks: dict[str, QCheckBox] = {}
        for kind, text in (("labels", "nucleus labels"), ("reference", "reference volumes"), ("tracks", "tracks")):
            cb = QCheckBox(text)
            cb.setChecked(True)
            krow.addWidget(cb)
            self.kind_checks[kind] = cb
        lay.addRow("Show", krow)
        self.load_btn = QPushButton("Load layers")
        self.load_btn.setToolTip("add the chosen layers again (after closing one) or refresh them")
        self.load_btn.clicked.connect(lambda: self.load_layers())
        lay.addRow("", self.load_btn)
        root.addWidget(lay_box)

        reg_box = QGroupBox("Register the drift (germquant register)")
        reg = QFormLayout(reg_box)
        defaults = TL.RegisterSettings()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(TL.REGISTER_MODES))
        self.mode_combo.setCurrentText(defaults.mode)
        self.mode_combo.setToolTip("rigid: one translation per timepoint; piecewise: one translation per segment "
                                   "along the gonad axis, for a germline that slides inside the worm (always "
                                   "against the previous timepoint)")
        self.mode_combo.currentTextChanged.connect(lambda _text: self._mode_changed())
        reg.addRow("Mode", self.mode_combo)
        self.segments_spin = QSpinBox()
        self.segments_spin.setRange(1, 64)
        self.segments_spin.setValue(defaults.segments)
        self.segments_spin.setToolTip("piecewise: segments of equal nucleus count along the axis (at least 5 "
                                      "germline nuclei each)")
        reg.addRow("Segments", self.segments_spin)
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 100.0)
        self.margin_spin.setDecimals(1)
        self.margin_spin.setValue(defaults.margin_um)
        self.margin_spin.setToolTip("piecewise: how far a segment's crop reaches beyond its nuclei in y and x")
        reg.addRow("Margin (um)", self.margin_spin)
        self.reference_combo = QComboBox()
        self.reference_combo.addItems(["previous", "t0"])
        self.reference_combo.setCurrentText(defaults.reference)
        self.reference_combo.setToolTip("previous: each timepoint against the one before, accumulated; t0: each "
                                        "against the first timepoint")
        reg.addRow("Reference", self.reference_combo)
        self.max_step_spin = QDoubleSpinBox()
        self.max_step_spin.setRange(0.0, 1000.0)
        self.max_step_spin.setDecimals(2)
        self.max_step_spin.setValue(defaults.max_step_um)
        self.max_step_spin.setToolTip("a longer step is shortened to this and flagged capped (0 = no cap)")
        reg.addRow("Max step (um)", self.max_step_spin)
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["none", "phase"])
        self.norm_combo.setCurrentText(defaults.normalization)
        self.norm_combo.setToolTip("none: plain cross-correlation (exact on smooth nuclei); phase: whitened, for "
                                   "textured volumes")
        reg.addRow("Normalization", self.norm_combo)
        self.upsample_spin = QSpinBox()
        self.upsample_spin.setRange(1, 1000)
        self.upsample_spin.setValue(defaults.upsample)
        self.upsample_spin.setToolTip("estimates are quantised to 1/N of a reference-volume voxel")
        reg.addRow("Upsample", self.upsample_spin)
        self.register_btn = QPushButton("Register")
        self.register_btn.clicked.connect(lambda: self.register())
        reg.addRow("", self.register_btn)
        self._mode_changed()
        self.shifts_label = QLabel("")
        self.shifts_label.setWordWrap(True)
        self.shifts_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        reg.addRow(self.shifts_label)
        self.shifts_table = QTableWidget(0, len(TL.SHIFT_COLUMNS))
        self.shifts_table.setHorizontalHeaderLabels([label for _key, label in TL.SHIFT_COLUMNS])
        self.shifts_table.verticalHeader().setVisible(False)
        self.shifts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        reg.addRow(self.shifts_table)
        root.addWidget(reg_box)

        link_box = QGroupBox("Link the nuclei across timepoints (germquant track-link)")
        link = QFormLayout(link_box)
        self.backend_combo = QComboBox()
        for i, (backend, usable, what) in enumerate(TL.backend_choices()):
            self.backend_combo.addItem(backend if usable else f"{backend} (not installed)", backend)
            self.backend_combo.setItemData(i, what, Qt.ItemDataRole.ToolTipRole)
            if not usable:
                item = self.backend_combo.model().item(i)
                if item is not None:
                    item.setEnabled(False)
        link.addRow("Backend", self.backend_combo)
        self.registration_combo = QComboBox()
        self.registration_combo.addItems(list(TL.REGISTRATIONS))
        self.registration_combo.setToolTip("rigid: subtract each timepoint's shift (register_shifts); piecewise: "
                                           "subtract each nucleus's displacement along the gonad axis "
                                           "(register_piecewise, from Register in piecewise mode)")
        link.addRow("Registration", self.registration_combo)
        self.all_nuclei_check = QCheckBox("link every nucleus, not only the germline ones (--all-nuclei)")
        link.addRow("", self.all_nuclei_check)
        self.link_btn = QPushButton("Link")
        self.link_btn.clicked.connect(lambda: self.link())
        link.addRow("", self.link_btn)
        self.link_label = QLabel("")
        self.link_label.setWordWrap(True)
        self.link_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        link.addRow(self.link_label)
        root.addWidget(link_box)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status_label)
        root.addStretch(1)

    # ------------------------------------------------------------------ helpers
    def _status(self, text: str, color: QColor | None = None) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("" if color is None else
                                        f"color: rgb({color.red()}, {color.green()}, {color.blue()});")

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Series folder")
        if path:
            self.open_folder(path)

    def _update_buttons(self) -> None:
        idle = not self.runner.busy
        self.open_btn.setEnabled(idle)
        self.register_btn.setEnabled(idle and self.view is not None)
        self.link_btn.setEnabled(idle and self.view is not None)
        self.load_btn.setEnabled(self.view is not None)

    @property
    def registered(self) -> bool:
        return self.reg_radio.isChecked()

    def set_coordinates(self, registered: bool) -> None:
        """Raw or registered coordinates for every layer of the dock (the radio buttons)."""
        (self.reg_radio if registered else self.raw_radio).setChecked(True)

    def _mode_changed(self) -> None:
        """Piecewise: the segment options apply and the reference is the previous timepoint."""
        piecewise = self.mode_combo.currentText() == "piecewise"
        self.segments_spin.setEnabled(piecewise)
        self.margin_spin.setEnabled(piecewise)
        if piecewise:
            self.reference_combo.setCurrentText("previous")
        self.reference_combo.setEnabled(not piecewise)

    def _set_piecewise_available(self, available: bool) -> None:
        """The piecewise link registration needs the register_piecewise table."""
        i = self.registration_combo.findText("piecewise")
        item = self.registration_combo.model().item(i) if i >= 0 else None
        if item is not None:
            item.setEnabled(bool(available))
            item.setToolTip("" if available else "no register_piecewise table yet: press Register in piecewise mode")
        if not available and self.registration_combo.currentText() == "piecewise":
            self.registration_combo.setCurrentText("rigid")

    def select_backend(self, backend: str) -> None:
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == backend:
                self.backend_combo.setCurrentIndex(i)
                return
        raise ValueError(f"no backend {backend!r}")

    # ------------------------------------------------------------------ open and show
    def open_folder(self, path: str | Path | None = None) -> TL.SeriesView | None:
        """Read the series folder (default: the Folder field) and load its layers."""
        if path is not None:
            self.folder_edit.setText(str(path).strip().strip('"'))
        text = self.folder_edit.text().strip().strip('"')
        if not text:
            self._status("choose a series folder", RED)
            return None
        try:
            view = TL.open_series(text)
        except Exception as e:  # noqa: BLE001 - not a series folder: say why, keep the dock usable
            self.view = None
            self.info_label.setText(f"cannot open {text}: {e}")
            self._status(f"{type(e).__name__}: {e}", RED)
            self._update_buttons()
            return None
        self._set_view(view)
        # the starting values the commands would use without options: the series' own config, else code defaults
        start = TL.series_defaults(view.folder)
        self.segments_spin.setValue(start["segments"])
        self.margin_spin.setValue(start["margin_um"])
        reg = start["registration"] if (start["registration"] != "piecewise" or view.has_piecewise) else "rigid"
        if view.linked_piecewise and view.has_piecewise:
            reg = "piecewise"
        self.registration_combo.setCurrentText(reg)
        self.load_layers()
        self._status(f"opened {view.stem}", GREEN)
        return view

    def _set_view(self, view: TL.SeriesView) -> None:
        self.view = view
        self.info_label.setText(TL.info_text(view))
        pw = TL.piecewise_summary(view)
        self.shifts_label.setText(TL.shifts_summary(view) + (f"\n{pw}" if pw else ""))
        self._set_piecewise_available(view.has_piecewise)
        rows = TL.shift_rows(view)
        self.shifts_table.setRowCount(len(rows))
        for i, cells in enumerate(rows):
            for j, text in enumerate(cells):
                self.shifts_table.setItem(i, j, QTableWidgetItem(text))
        self.shifts_table.resizeColumnsToContents()
        rec = view.link_record
        try:
            text = TL.link_summary({"record": rec}) if rec else ""
        except (KeyError, TypeError, ValueError):   # a record written by another version or by hand
            text = f"link record: {rec.get('link_source') or rec.get('backend') or 'unreadable'}"
        if not text and view.points is not None:
            text = "track_points present (no link record)"
        self.link_label.setText(text)
        self._update_buttons()

    def _tuples(self, kinds: list[str] | None = None) -> dict[str, tuple | None]:
        """The layer tuples of the ticked kinds, or of exactly ``kinds`` when given (a refresh of the layers
        already shown, whatever the boxes say now)."""
        view, reg = self.view, self.registered

        def want(kind: str, box: str) -> bool:
            return kind in kinds if kinds is not None else self.kind_checks[box].isChecked()

        out: dict[str, tuple | None] = {}
        if want("labels", "labels"):
            out["labels"] = TL.labels_layer(view, registered=reg, stride=int(self.stride_spin.value()))
        if want("reference", "reference"):
            out["reference"] = TL.reference_layer(view, registered=reg)
        if want("tracks", "tracks"):
            out["tracks"] = TL.tracks_layer(view, registered=reg)
        if want("unlinked", "tracks"):
            out["unlinked"] = TL.unlinked_layer(view, registered=reg)
        return out

    def load_layers(self) -> list:
        """Add (or update in place) every chosen layer of the open series; returns the dock's layers."""
        if self.view is None:
            self._status("open a series folder first", AMBER)
            return []
        try:
            tuples = self._tuples()
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._status(f"could not build the layers: {format_exception(e)}", RED)
            return []
        self._apply(tuples, [k for k in LAYER_KINDS if k in tuples])
        return [self.layers[k] for k in LAYER_KINDS if k in self.layers]

    def _apply(self, tuples: dict, kinds: list[str]) -> None:
        failed = []
        for kind in kinds:
            try:
                if tuples.get(kind) is None:
                    self._remove_layer(kind)
                else:
                    self._put_layer(kind, tuples[kind])
            except Exception as e:  # noqa: BLE001 - one layer that napari refuses must not take the others down
                failed.append(f"{kind}: {type(e).__name__}: {e}")
        if failed:
            self._status("could not show " + "; ".join(failed), RED)

    def refresh_layers(self, only: tuple[str, ...] | None = None) -> None:
        """Rebuild the layers the dock already shows (after the coordinates or the stride changed)."""
        if self.view is None or self.viewer is None:
            return
        shown = [k for k in LAYER_KINDS if self._live(k) and (only is None or k in only)]
        if not shown:
            return
        try:
            tuples = self._tuples(shown)
        except Exception as e:  # noqa: BLE001
            self._status(f"could not rebuild the layers: {format_exception(e)}", RED)
            return
        self._apply(tuples, [k for k in shown if k in tuples])
        set_viewer_status(self.viewer, f"{self.view.stem}: {'registered' if self.registered else 'raw'} coordinates")

    def _live(self, kind: str) -> bool:
        layer = self.layers.get(kind)
        return layer is not None and self.viewer is not None and layer in self.viewer.layers

    def _remove_layer(self, kind: str) -> None:
        layer = self.layers.pop(kind, None)
        if layer is not None and self.viewer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def _put_layer(self, kind: str, tup: tuple) -> None:
        if self.viewer is None:
            return
        data, kwargs, _layer_type = tup
        if self._live(kind):
            layer = self.layers[kind]
            layer.data = data
            if "features" in kwargs:
                # napari drops a Tracks or Points layer's features when its data is replaced
                layer.features = kwargs["features"]
            for key in ("scale", "translate"):
                if key in kwargs:
                    setattr(layer, key, kwargs[key])
            layer.metadata = kwargs.get("metadata", {})
            layer.name = kwargs["name"]
            return
        (layer,) = add_layer_tuples(self.viewer, [tup])
        self.layers[kind] = layer

    # ------------------------------------------------------------------ Register and Link
    def register_settings(self) -> TL.RegisterSettings:
        return TL.RegisterSettings(reference=self.reference_combo.currentText(),
                                   upsample=int(self.upsample_spin.value()),
                                   max_step_um=float(self.max_step_spin.value()),
                                   normalization=self.norm_combo.currentText(),
                                   mode=self.mode_combo.currentText(),
                                   segments=int(self.segments_spin.value()),
                                   margin_um=float(self.margin_spin.value()))

    def link_settings(self) -> TL.LinkSettings:
        return TL.LinkSettings(backend=str(self.backend_combo.currentData()),
                               all_nuclei=self.all_nuclei_check.isChecked(),
                               registration=self.registration_combo.currentText())

    def register(self) -> bool:
        """``germquant register`` on the open series, in a thread worker."""
        if self.view is None or self.runner.busy:
            self._status("open a series folder first" if self.view is None else "busy: wait for the current step",
                         AMBER)
            return False
        folder = self.view.folder
        try:
            settings = self.register_settings()
        except ValueError as e:
            self._status(str(e), RED)
            return False
        self.last_register = None
        self.last_error = None
        cmd = settings.command(folder)
        self._status(f"registering {self.view.stem} ({settings.mode}) ...\n{cmd}")
        self.runner.start(lambda: TL.register(folder, settings), label=f"registering {self.view.stem}",
                          returned=lambda res: self._on_registered(res, cmd, settings.mode),
                          errored=lambda e: self._on_errored(e, "register"), finished=self._update_buttons)
        self._update_buttons()
        return True

    def _on_registered(self, res: dict, cmd: str, mode: str = "rigid") -> None:
        self.last_register = res
        if not self._reload():
            return
        if mode in TL.REGISTRATIONS:
            self.registration_combo.setCurrentText(mode)      # Link next subtracts what was just registered
        self._status(f"registered {res['series_id']}: {res['table_path']}\nsame as: {cmd}", GREEN)
        set_viewer_status(self.viewer, f"{res['series_id']} registered")

    def link(self) -> bool:
        """``germquant track-link`` on the open series with the chosen backend, in a thread worker."""
        if self.view is None or self.runner.busy:
            self._status("open a series folder first" if self.view is None else "busy: wait for the current step",
                         AMBER)
            return False
        folder = self.view.folder
        try:
            settings = self.link_settings()
        except ValueError as e:
            self._status(str(e), RED)
            return False
        self.last_link = None
        self.last_error = None
        cmd = settings.command(folder)
        self._status(f"linking {self.view.stem} with {settings.backend} ...\n{cmd}")
        self.runner.start(lambda: TL.link(folder, settings), label=f"linking {self.view.stem} ({settings.backend})",
                          returned=lambda res: self._on_linked(res, cmd),
                          errored=lambda e: self._on_errored(e, "link"), finished=self._update_buttons)
        self._update_buttons()
        return True

    def _on_linked(self, res: dict, cmd: str) -> None:
        self.last_link = res
        if not self._reload():
            return
        self._status(f"{TL.link_summary(res)}\nsame as: {cmd}", GREEN)
        set_viewer_status(self.viewer, f"{res['record']['series_id']}: {res['record']['n_tracks']} track(s)")

    def _reload(self) -> bool:
        """Read the series folder again after a step wrote to it, then rebuild the shown layers."""
        try:
            view = TL.open_series(self.view.folder, self.view.stem)
        except Exception as e:  # noqa: BLE001
            self._status(f"the step finished but the folder could not be read back: {e}", AMBER)
            return False
        self._set_view(view)
        self.load_layers()
        return True

    def _on_errored(self, exc: BaseException, what: str) -> None:
        self.last_error = exc
        self._status(f"{what} FAILED: {format_exception(exc)}", RED)
        set_viewer_status(self.viewer, f"germquant {what} failed: {type(exc).__name__}: {exc}")
        try:
            from napari.utils.notifications import show_error

            show_error(f"germquant {what} failed: {type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 - no notification manager in a bare Qt test
            pass
