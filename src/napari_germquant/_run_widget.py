"""The Run dock: image, recipe (profile or config file), channel roles, stages, xy stride, output folder,
Run, Export profile.

Everything the widget decides is delegated to ``_engine`` (roles, stage rows, the RunPlan); this module is
only layout and signals. The Run button executes ``RunPlan.run`` (``germquant.api.run``) in a napari
thread_worker; log lines of the engine, cellpose and SpotMAX go to the status bar through LogBridge; when the
worker returns, the results folder is loaded as layers through ``_results`` (the same loader the Results
dock uses).
"""
from __future__ import annotations

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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _engine
from ._engine import Recipe, RunPlan
from ._qt_common import LogBridge, add_layer_tuples, set_viewer_status
from ._reader import image_info, open_image
from ._results import ResultsSet, load_results_layers, summary_text
from germquant import api

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)

IMAGE_FILTER = "Images (" + " ".join(f"*{s}" for s in api.SUPPORTED_SUFFIXES) + ");;All files (*)"


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
        self._t0 = 0.0
        self.last_result: dict | None = None
        self.last_error: BaseException | None = None
        self.last_export: tuple[Path, Path] | None = None
        self._bridge = LogBridge(self)
        self._bridge.message.connect(self._on_log_line)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build()
        self._populate_profiles()
        if self.viewer is not None:
            try:
                self.viewer.layers.selection.events.active.connect(self._on_active_layer)
            except Exception:  # noqa: BLE001 - a viewer proxy without events is not worth failing over
                pass

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)

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

        rec_box = QGroupBox("Recipe (profile or config file)")
        rec_form = QFormLayout(rec_box)
        row3 = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo)
        row3.addWidget(self.profile_combo, 1)
        cfg_btn = QPushButton("Config file...")
        cfg_btn.setToolTip("any config YAML, for example an overlay this widget exported (extends: a profile)")
        cfg_btn.clicked.connect(self._browse_config)
        row3.addWidget(cfg_btn)
        rec_form.addRow("Profile", row3)
        self.recipe_label = QLabel("")
        self.recipe_label.setWordWrap(True)
        rec_form.addRow("File", self.recipe_label)
        root.addWidget(rec_box)

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

    # ------------------------------------------------------------------ recipe
    def _populate_profiles(self) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for prof in _engine.profiles():
            self.profile_combo.addItem(prof.name, ("profile", prof.name))
        self.profile_combo.blockSignals(False)
        if self.profile_combo.count():
            self._on_profile_combo(self.profile_combo.currentIndex())
        else:
            self._status("no profiles found: germquant's config/profiles folder is not visible from this "
                         "environment (set GERMQUANT_CONFIG_ROOT to the checkout) or pick a config file", RED)

    def _on_profile_combo(self, index: int) -> None:
        data = self.profile_combo.itemData(index)
        if not data:
            return
        kind, value = data
        self.set_recipe(Recipe.from_profile(value) if kind == "profile" else Recipe.from_file(value))

    def select_profile(self, name: str) -> None:
        for i in range(self.profile_combo.count()):
            if self.profile_combo.itemData(i) == ("profile", name):
                self.profile_combo.setCurrentIndex(i)
                self._on_profile_combo(i)
                return
        raise ValueError(f"no profile {name!r} in the list")

    def select_config_file(self, path: str | Path) -> None:
        """A config YAML as the recipe (the command line's --config): one combo entry per file."""
        path = Path(path)
        key = ("config", str(path))
        for i in range(self.profile_combo.count()):
            if self.profile_combo.itemData(i) == key:
                self.profile_combo.setCurrentIndex(i)
                self._on_profile_combo(i)
                return
        self.profile_combo.addItem(f"file: {path.name}", key)
        self.profile_combo.setCurrentIndex(self.profile_combo.count() - 1)
        self._on_profile_combo(self.profile_combo.count() - 1)

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Config YAML", "", "YAML (*.yaml *.yml);;All files (*)")
        if path:
            self.select_config_file(path)

    def set_recipe(self, recipe: Recipe) -> None:
        self._recipe = recipe
        try:
            self._cfg = recipe.load()
            self.recipe_label.setText(str(recipe.path()))
        except Exception as e:  # noqa: BLE001 - shown to the user, the widget stays usable
            self._cfg = None
            self.recipe_label.setText(f"could not load: {e}")
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
        dz, dy, dx = meta.get("spacing", (1, 1, 1))
        size_text = " ".join(f"{k}={v}" for k, v in sizes.items())
        spacing_text = f"voxel um dz={dz:.3f} dy={dy:.3f} dx={dx:.3f}" if meta.get("spacing_ok", True) \
            else "voxel size UNREADABLE (1 um assumed)"
        extra = ""
        n_t = int(sizes.get("T", 1) or 1)
        n_p = max(int(sizes.get("P", 1) or 1), int(meta.get("n_scenes", 1) or 1))
        if n_t > 1 or n_p > 1:
            extra = f"\nREFUSED: {n_t} time points, {n_p} positions or scenes; the plugin takes fixed single-scene images"
        self.image_info.setText(f"{size_text}\n{spacing_text}\nchannels: {self._channel_names}{extra}")
        if not self.out_edit.text().strip():
            self.out_edit.setText(str(api.default_out_dir(p)))
        self.refresh()

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
        return RunPlan(image=self._image, recipe=self._recipe, out=Path(out),
                       xy_stride=int(self.stride_spin.value()), disabled=self.disabled_stages())

    def export_to(self, folder: str | Path) -> tuple[Path, Path]:
        plan = self.plan()
        self.last_export = plan.export(folder)
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
        """Run the plan in a napari thread worker; results load as layers when it returns."""
        if self._worker is not None:
            self._status("a run is already in progress", AMBER)
            return
        try:
            plan = self.plan()
        except ValueError as e:
            self._status(str(e), RED)
            return
        from napari.qt.threading import create_worker

        self.last_result = None
        self.last_error = None
        self._plan = plan
        self.run_btn.setEnabled(False)
        self._t0 = time.monotonic()
        self._timer.start()
        self._bridge.attach()
        msg = f"running germquant on {plan.image.name} with {plan.recipe.label}"
        if plan.disabled:
            msg += " (" + " ".join(f"--no-{n}" for n in plan.disabled) + ")"
        self._status(msg + " ...", None)
        set_viewer_status(self.viewer, msg)
        # an explicit errored connection: without one napari re-raises the exception in the Qt event loop
        self._worker = create_worker(
            plan.run,
            _connect={"returned": self._on_returned, "errored": self._on_errored, "finished": self._on_finished},
        )

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._t0)
        set_viewer_status(self.viewer, f"germquant running ... {elapsed // 60} min {elapsed % 60:02d} s")

    def _on_log_line(self, text: str) -> None:
        set_viewer_status(self.viewer, text[:200])

    def _on_returned(self, res: dict) -> None:
        self.last_result = res
        rs = ResultsSet(Path(res["out_dir"]), res["image_id"])
        try:
            layers = load_results_layers(rs)
            added = add_layer_tuples(self.viewer, layers)
        except Exception as e:  # noqa: BLE001 - the run itself succeeded; say the load did not
            self._status(f"run finished ({res['n_nuclei']} nuclei) but the results could not be loaded: {e}", AMBER)
            return
        flags = "; ".join(res.get("qc_flags") or []) or "none"
        self._status(f"done: {res['image_id']}: {res['n_nuclei']} nuclei, qc_pass={res['qc_pass']}, flags: {flags}\n"
                     f"{len(added)} layer(s) loaded from {res['out_dir']}\n{summary_text(rs)}",
                     GREEN if res["qc_pass"] else AMBER)
        set_viewer_status(self.viewer, f"germquant done: {res['n_nuclei']} nuclei ({res['image_id']})")

    def _on_errored(self, exc: BaseException) -> None:
        self.last_error = exc
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1500:]
        self._status(f"run FAILED: {type(exc).__name__}: {exc}\n{tb}", RED)
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
        self.run_btn.setEnabled(True)

    # ------------------------------------------------------------------ status
    def _status(self, text: str, color: QColor | None) -> None:
        self.status_label.setText(text)
        if color is None:
            self.status_label.setStyleSheet("")
        else:
            self.status_label.setStyleSheet(f"color: rgb({color.red()}, {color.green()}, {color.blue()});")
