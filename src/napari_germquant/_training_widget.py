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

Where it computes (tranche 5 Train B, decision D36; plugin_specs.md 4.4, "Route"). The dock shows the engine line
of the viewer's engine session (shared with the Project and Spot tuning docks). When an engine is chosen for this
computer, export, import and training go through that engine's runner (``python -m germquant export-training``,
``import-labels``, ``train-nuclei`` in its own environment, with the devices it offers, the epoch lines on the
status bar and Cancel). With none chosen the dock keeps computing in this napari environment and says so in amber;
it never switches silently. Correcting pairs and registering stay in napari's process either way.

The project mode (design 8.6 and 10). With a project open (the one the Project dock has open, followed through
its hub, or one opened here) the folders default to the project: the source is ``<project>/results/``, the
training folder the round ``<project>/training/<round>/`` (its pairs, corrections.json and ``models/<name>/``),
picked in the round list; the rounds come from ``germquant project rounds`` and New round asks the engine for the
next one (r01, r02 and so on). Then also:

* Import labels made elsewhere (``germquant import-labels SRC --format F --image IMG --pairs DIR``): the format
  from the file name, the project image it belongs to, its frame, and "Allow pickle", which only a Cellpose
  _seg.npy can tick, next to the engine's warning. The engine's refusals and warnings are shown.
* Orientations follow each run: the dock reads each run's geometry and shows what the export writes (xy, xz and
  yz slices of a 3D run, chosen with the boxes; one proj pair of a projected run; the plane of a single-plane
  run), and the export passes ``--orientations auto`` or the chosen subset of a 3D run's slices.
* Starting weights: the models the project's pack recipe needs, the models trained in the project, then
  Cellpose's own, a random start and the registered names; the card's object kind from the project's pack.
* Use in project (``germquant project use-model DIR --model training/<round>/models/<name>/<file>``): the project's
  recipe then names the model, its config_hash changes and validated turns false, and the Project dock reads the
  project again. It waits while a run of the project goes.
* Derive pack (``germquant pack derive --from ID@VER --recipe R --model PATH --card CARD --out
  <project>/packs``): the pack listing with the project then shows the derived pack.

Every project write goes through the engine (decision D39): the chosen engine, or napari's own germquant command
when none is chosen. The GUI thread never waits on the engine or a slow file system: the steps run in one thread
worker, the readings (the runs' geometry, the project, an image's channels) in another, the engine check and the
round listing in a third. A reading or a listing that returns after its project was closed, or after a newer one
was asked for, changes nothing.

Everything the dock decides is in ``_training`` (no Qt); this module is layout, signals and the layers.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
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

from . import _settings
from . import _training as T
from ._engine_session import EngineNotChosen, session_for
from ._qt_common import WorkerRunner, format_exception, set_viewer_status
from ._runner import (
    EngineRefused,
    FeatureMissing,
    InProcessRunner,
    RunCancelled,
    RunnerError,
    detect_interpreters,
    plugin_version,
)

RED = QColor(200, 60, 60)
AMBER = QColor(225, 160, 40)
GREEN = QColor(70, 150, 90)

# the metadata key on the two layers this dock adds (the pair they show)
META_KEY = "napari_germquant_training"

DEVICE_TEXT = {"cuda": "cuda (NVIDIA GPU)", "mps": "mps (Apple GPU, Metal)", "cpu": "cpu (slow: hours for 150 epochs)",
               "auto": "auto (the engine's choice: its GPU when it sees one)"}
KIND_TEXT = {"auto": "as the runs' packs say (else nucleus)", "nucleus": "nucleus", "cell": "cell (whole cells)"}
# the steps an engine process can be stopped in (Cancel ends it and every process it started)
CANCELLABLE = ("export", "import", "train")
NAPARI_TEXT = ("No engine chosen for this computer: export, import and training run in this napari environment "
               "({python}), on the devices it offers, and the project steps (rounds, Use in project, Derive pack) "
               "run its own germquant commands. Choose an engine to make them where the numbers are made (the Run "
               "and Project docks use the same choice).")
CHANNEL_DEFAULT = "the engine's default: the image's only channel, else channel 0 (with a warning)"
CHANNEL_NOT_PASSED = (f"{CHANNEL_DEFAULT}. This plugin's engine runner does not pass --channel yet, so the labels "
                      f"of a multi-channel image are paired with channel 0; the engine's warning says so.")

engine_log = logging.getLogger("napari_germquant.engine")


def _say(line: str) -> None:
    """An engine line, to the status bar (the worker's LogBridge listens to napari_germquant)."""
    engine_log.info(line)


def _css(color: QColor | None) -> str:
    return "" if color is None else f"color: rgb({color.red()}, {color.green()}, {color.blue()});"


def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _viewer_hub(viewer):
    """The Project dock's hub of ``viewer`` (napari_germquant._project_widget.project_hub, unit P6-PROJECT), or
    None without a viewer or without that dock in this plugin."""
    if viewer is None:
        return None
    try:
        from ._project_widget import project_hub
    except ImportError:
        return None
    return project_hub(viewer)


def _run_in_progress(root) -> bool:
    """Whether a dock of this napari runs the project now (the Project dock's registry, when it is in the plugin)."""
    try:
        from ._project import run_in_progress
    except ImportError:
        return False
    return bool(run_in_progress(root))


def _path_row(edit: QLineEdit, button: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(edit)
    row.addWidget(button)
    return row


class _Jobs:
    """Calls of one kind of reading, one at a time on a WorkerRunner: a newer request of a kind replaces the
    one of that kind still waiting, so a burst of edits reads once."""

    def __init__(self, worker: WorkerRunner, on_finished=None) -> None:
        self.worker = worker
        self.pending: list[tuple] = []
        self._on_finished = on_finished

    @property
    def idle(self) -> bool:
        return not self.worker.busy and not self.pending

    def submit(self, kind: str, func, *, label: str, returned, errored) -> None:
        job = (kind, func, label, returned, errored)
        if not self.worker.busy and not self.pending:
            self._start(job)
            return
        self.pending = [j for j in self.pending if j[0] != kind] + [job]

    def clear(self) -> None:
        self.pending = []

    def _start(self, job) -> None:
        _kind, func, label, returned, errored = job
        self.worker.start(func, label=label, returned=returned, errored=errored, finished=self._next,
                          line_format=lambda line: line[:160])

    def _next(self) -> None:
        if self.pending:
            self._start(self.pending.pop(0))
        if self._on_finished is not None:
            self._on_finished()


class TrainingWidget(QWidget):
    """napari dock widget. ``napari_viewer`` is injected by napari; None builds the widget without a viewer.
    ``session`` replaces the viewer's engine session and ``hub`` the Project dock's hub (tests)."""

    def __init__(self, napari_viewer=None, parent=None, *, session=None, hub=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.runner = WorkerRunner(napari_viewer, self)          # the steps: export, import, train, ...
        self.files_worker = WorkerRunner(None, self)              # readings: geometry, the project, headers
        self.engine_worker = WorkerRunner(None, self)             # the engine check and the round listing
        self._files = _Jobs(self.files_worker, self._update_buttons)
        self._engine_jobs = _Jobs(self.engine_worker, self._update_buttons)
        self.session = session if session is not None else session_for(napari_viewer)
        self.hub = hub if hub is not None else _viewer_hub(napari_viewer)
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
        # the engine route and the project mode (tranche 5 Train B)
        self.project: T.TrainingProject | None = None
        self.rounds: list[str] = []
        self.geometry: list[T.RunGeometry] | None = None
        self.last_import: dict | None = None
        self.last_use: dict | None = None
        self.last_derive: dict | None = None
        self.engine_error: str | None = None
        self.interpreter_dialog = None
        self._engine_devices: list[str] | None = None
        self._projects_problem: str | None = None     # why the project's rounds cannot come from the engine
        self._following_hub = False
        self._reading = 0                              # one more at every project reading asked for, and at Close
        self._hub_reading: int | None = None           # the reading of the Project dock's project still in flight
        self._device_list = "napari"                   # whose devices the device list shows: napari or engine
        self._geometry_req = 0
        self._channels_req = 0
        self._channels: list[str] = []
        self._job_cancellable = False
        self._known_choice = _settings.engine_choice()  # the engine choice the engine line shows
        self._build()
        self._update_buttons()
        try:
            self.session.changed.connect(self._on_session_changed)
        except (RuntimeError, AttributeError):
            pass
        if self.hub is not None:
            self.hub.opened.connect(self._on_hub_opened)
            self.hub.closed.connect(self._on_hub_closed)
            self.hub.changed.connect(self._on_hub_changed)
            self.hub.running_changed.connect(self._on_hub_running)
        self._show_engine()
        self._show_project()
        if self._engine_chosen():
            self._check_engine()
        if self.hub is not None and getattr(self.hub, "root", None) is not None:
            self._on_hub_opened(str(self.hub.root))

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)

        eng_box = QGroupBox("Engine and project")
        eng = QFormLayout(eng_box)
        erow = QHBoxLayout()
        self.engine_label = QLabel("")
        self.engine_label.setWordWrap(True)
        self.engine_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        erow.addWidget(self.engine_label, 1)
        self.engine_btn = QPushButton("Choose engine...")
        self.engine_btn.setToolTip("the Python of the germquant environment that exports, imports and trains (the Run "
                                   "and Project docks use the same choice)")
        self.engine_btn.clicked.connect(lambda: self.ask_for_engine())
        erow.addWidget(self.engine_btn)
        eng.addRow("Engine", erow)
        self.project_label = QLabel("")
        self.project_label.setWordWrap(True)
        self.project_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        eng.addRow("Project", self.project_label)
        prow = QHBoxLayout()
        self.open_project_btn = QPushButton("Open project...")
        self.open_project_btn.setToolTip("work in a project folder (the Project dock's project is followed by itself)")
        self.open_project_btn.clicked.connect(lambda: self.open_project_dialog())
        prow.addWidget(self.open_project_btn)
        self.close_project_btn = QPushButton("Close project")
        self.close_project_btn.setToolTip("back to folders of your own choosing")
        self.close_project_btn.clicked.connect(lambda: self.close_project())
        prow.addWidget(self.close_project_btn)
        prow.addStretch(1)
        eng.addRow("", prow)
        rrow = QHBoxLayout()
        self.round_combo = QComboBox()
        self.round_combo.setToolTip("training/<round>/: the round's pairs, corrections.json and models/<name>/")
        self.round_combo.currentTextChanged.connect(self._on_round_chosen)
        rrow.addWidget(self.round_combo, 1)
        self.new_round_btn = QPushButton("New round")
        self.new_round_btn.setToolTip("germquant project rounds --new: the engine makes the next of r01, r02 and so on")
        self.new_round_btn.clicked.connect(lambda: self.new_round())
        rrow.addWidget(self.new_round_btn)
        eng.addRow("Round", rrow)
        self.notice_label = QLabel("")
        self.notice_label.setWordWrap(True)
        self.notice_label.setStyleSheet(_css(AMBER))
        self.notice_label.setVisible(False)
        eng.addRow("", self.notice_label)
        root.addWidget(eng_box)

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
            cb.toggled.connect(lambda _on: self._show_geometry())
            orow.addWidget(cb)
            self.orient_checks[o] = cb
        orow.addStretch(1)
        exp.addRow("Orientations", orow)
        self.geometry_label = QLabel("")
        self.geometry_label.setWordWrap(True)
        self.geometry_label.setToolTip("what the export writes for each run, from its record (run_manifest.json): a "
                                       "projected run gives its proj pair and a single-plane run its plane, whatever "
                                       "the boxes say")
        exp.addRow("", self.geometry_label)
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

        imp_box = QGroupBox("Or import labels made elsewhere (germquant import-labels)")
        imp = QFormLayout(imp_box)
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText("a label image: .tif, .ome.tif or .png, a Cell-ACDC segm.npz, a Cellpose "
                                            "_seg.npy")
        self.labels_edit.editingFinished.connect(lambda: self.set_labels_file(self.labels_edit.text()))
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse_file(self.set_labels_file, "Labels file",
                                                    "Labels (*.tif *.tiff *.png *.npz *_seg.npy);;All files (*)"))
        imp.addRow("Labels file", _path_row(self.labels_edit, b))
        self.format_combo = QComboBox()
        for fmt in T.IMPORT_FORMATS:
            self.format_combo.addItem(fmt, fmt)
        self.format_combo.setToolTip("guessed from the file name, which must carry its format (the engine refuses "
                                     "another)")
        self.format_combo.currentIndexChanged.connect(lambda _i: self._on_format_changed())
        imp.addRow("Format", self.format_combo)
        self.image_combo = QComboBox()
        self.image_combo.setEditable(True)
        self.image_combo.setToolTip("the image the labels were drawn on, at full resolution (a project's images are "
                                    "listed; any image file can be typed or browsed)")
        self.image_combo.currentTextChanged.connect(lambda _t: self._read_channels())
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse_file(self.set_import_image, "Image", "All files (*)"))
        irow = QHBoxLayout()
        irow.addWidget(self.image_combo, 1)
        irow.addWidget(b)
        imp.addRow("Image", irow)
        self.channel_combo = QComboBox()
        imp.addRow("Channel", self.channel_combo)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(-1, 1_000_000)
        self.frame_spin.setValue(-1)
        self.frame_spin.setSpecialValueText("default (0)")
        self.frame_spin.setToolTip("--frame: the frame of a labels file with a T axis, and the timepoint of a time "
                                   "series image")
        imp.addRow("Frame", self.frame_spin)
        self.pickle_check = QCheckBox("Allow pickle")
        self.pickle_check.setToolTip("--allow-pickle: only for a Cellpose _seg.npy you made")
        imp.addRow("", self.pickle_check)
        self.pickle_label = QLabel(f"{T.PICKLE_WARNING}.")
        self.pickle_label.setWordWrap(True)
        self.pickle_label.setStyleSheet(_css(AMBER))
        imp.addRow("", self.pickle_label)
        self.import_btn = QPushButton("Import labels")
        self.import_btn.setToolTip("write the labels and the image into the training folder as pairs (the source "
                                   "import:<tool> in manifest.csv)")
        self.import_btn.clicked.connect(lambda: self.import_labels())
        imp.addRow("", self.import_btn)
        root.addWidget(imp_box)

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
                                   "model name; or type the path of a model file. In a project, the models its pack "
                                   "recipe needs and the models trained in it come first")
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
        self._fill_napari_devices()
        tr.addRow("Device", self.device_combo)
        self.kind_combo = QComboBox()
        for kind in ("auto", *T.OBJECT_KINDS):
            self.kind_combo.addItem(KIND_TEXT[kind], kind)
        self.kind_combo.setToolTip("the model card's object_kind (--kind): what one label is. Imported labels name "
                                   "none, so a project sets its pack's")
        tr.addRow("Object kind", self.kind_combo)
        self.system_edit = QLineEdit()
        self.system_edit.setPlaceholderText("optional: the biological system the pairs show, for the card (--system)")
        tr.addRow("System", self.system_edit)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("optional: a public dataset's id under benchmarks/ext/ (--dataset)")
        tr.addRow("Dataset", self.dataset_edit)
        row = QHBoxLayout()
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(lambda: self.train())
        row.addWidget(self.train_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("end the engine's export, import or training (an engine in napari's own process "
                                   "cannot be stopped)")
        self.cancel_btn.clicked.connect(lambda: self.cancel())
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        tr.addRow("", row)
        root.addWidget(tr_box)

        reg_box = QGroupBox("4. Register the model (germquant register-model)")
        reg = QFormLayout(reg_box)
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("the model file train-nuclei wrote (filled in after Train)")
        self.model_edit.textChanged.connect(lambda _t: self._update_buttons())
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

        use_box = QGroupBox("5. Use the model in the project")
        use = QHBoxLayout(use_box)
        self.use_btn = QPushButton("Use in project")
        self.use_btn.setToolTip("germquant project use-model: the project's recipe names this model (a new "
                                "config_hash; validated turns false)")
        self.use_btn.clicked.connect(lambda: self.use_in_project())
        use.addWidget(self.use_btn)
        self.derive_btn = QPushButton("Derive pack")
        self.derive_btn.setToolTip("germquant pack derive into <project>/packs: a pack of the project's recipe with "
                                   "this model, listed with the project's packs")
        self.derive_btn.clicked.connect(lambda: self.derive_pack())
        use.addWidget(self.derive_btn)
        use.addStretch(1)
        root.addWidget(use_box)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status_label)
        root.addStretch(1)
        self._on_format_changed()
        self._show_channels()

    # ------------------------------------------------------------------ small helpers
    def _status(self, text: str, color: QColor | None = None) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(_css(color))

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

    @property
    def idle(self) -> bool:
        """No step, reading or engine call in flight or waiting (tests wait on it)."""
        return not self.runner.busy and self._files.idle and self._engine_jobs.idle

    @property
    def root(self) -> Path | None:
        return self.project.root if self.project is not None else None

    def _running(self) -> bool:
        """Whether a run of the open project goes (the Project dock's registry, or its hub)."""
        if self.project is None:
            return False
        if _run_in_progress(self.project.root):
            return True
        hub = self.hub
        return bool(hub is not None and getattr(hub, "running", False)
                    and _same(getattr(hub, "root", None), self.project.root))

    def _update_buttons(self) -> None:
        busy = self._busy_kind
        have = bool(self._pairs)
        for w in (self.export_btn, self.open_btn, self.train_btn, self.register_btn, self.import_btn):
            w.setEnabled(busy is None)
        nav = have and busy not in ("export", "import")
        self.prev_btn.setEnabled(nav and self._index > 0)
        self.next_btn.setEnabled(nav and self._index < len(self._pairs) - 1)
        self.revert_btn.setEnabled(nav)
        # a model trains on the files as they are on disk and its card records their sha256 at the end, so a
        # save while it trains would leave a card that describes other labels than the model saw
        self.save_btn.setEnabled(nav and busy != "train")
        project = self.project is not None
        engine_rounds = project and self._projects_problem is None
        self.round_combo.setEnabled(engine_rounds and bool(self.rounds) and busy is None)
        self.new_round_btn.setEnabled(engine_rounds and busy is None)
        model = bool(self._clean(self.model_edit.text()))
        self.use_btn.setEnabled(engine_rounds and busy is None and model and not self._running())
        self.derive_btn.setEnabled(project and busy is None and model)
        self.close_project_btn.setEnabled(project)
        self.cancel_btn.setEnabled(busy in CANCELLABLE and self._job_cancellable)

    # ------------------------------------------------------------------ the engine
    def _engine_chosen(self) -> bool:
        """Whether the dock goes through an engine: one is stored for this computer (or the session holds one)."""
        try:
            if self.session.runner() is not None:
                return True
        except Exception:  # noqa: BLE001 - a session that cannot answer has no engine
            pass
        return _settings.engine_choice() is not None

    def engine_route(self) -> str:
        """``engine`` (the chosen engine's runner) or ``napari`` (this napari environment)."""
        return "engine" if self._engine_chosen() else "napari"

    def _ui_runner(self):
        """The runner the next step would go to, for what the dock may offer (never starts a process): the checked
        engine, else the in-process runner's class."""
        if self._engine_chosen():
            try:
                runner = self.session.runner()
            except Exception:  # noqa: BLE001
                runner = None
            return runner
        return InProcessRunner

    def _job_runner(self, *features: str, engine: bool, recipe: tuple | None = None, should_stop=None):
        """In a worker: the runner a step goes to, with ``features`` required (FeatureMissing otherwise). Blocks
        (the engine's check). ``recipe`` (pack id, recipe id): the interpreter that recipe has on this computer."""
        if not engine:
            runner = InProcessRunner()
            runner.device = "auto"
            runner.require(*features)
            return runner
        if recipe is not None:
            runner = self.session.runner_for(recipe[0], recipe[1], should_stop=should_stop)
        else:
            runner = self.session.ensure(should_stop)
        if features:
            runner.require(*features, should_stop=should_stop)
        return runner

    def _auto_supported(self) -> bool:
        """Whether the export that would run follows each run's geometry (``--orientations auto``)."""
        if self._engine_chosen():
            runner = self._ui_runner()
            if runner is None:
                return False
            try:
                return "train_curve" in runner.features()
            except Exception:  # noqa: BLE001 - an engine that cannot say gets the plain list
                return False
        return T.export_auto_supported()

    def _check_engine(self) -> None:
        session, worker = self.session, self.engine_worker

        def job():
            runner = session.ensure(worker.should_stop)
            try:
                devices = session.devices(worker.should_stop)
            except RunnerError as e:
                return runner, None, str(e)
            return runner, devices, None

        self._engine_jobs.submit("engine", job, label="checking the engine", returned=self._on_engine_checked,
                                 errored=self._on_engine_failed)

    def _on_engine_checked(self, result) -> None:
        _runner, devices, problem = result
        self.engine_error = None
        self._known_choice = _settings.engine_choice()
        self._engine_devices = list(devices) if devices else ["auto", "cpu"]
        self._fill_engine_devices()
        self._show_engine(problem)
        self._show_geometry()
        self._show_channels()
        if self.project is not None and self._projects_problem is not None:
            self._list_rounds()                       # the engine is known now: ask it again

    def _on_engine_failed(self, exc) -> None:
        self.engine_error = None if isinstance(exc, EngineNotChosen) else str(exc)
        self._engine_devices = None
        self._fill_napari_devices()
        self._show_engine()

    def _sync_engine(self) -> None:
        """Another dock may have stored another engine for this computer since the engine line was shown: show it
        (and check it) before a step goes there. A step's status says where it runs either way."""
        choice = _settings.engine_choice()
        if choice != self._known_choice:
            self._known_choice = choice
            self._on_session_changed()

    def _on_session_changed(self) -> None:
        self._known_choice = _settings.engine_choice()
        if self._engine_chosen():
            self._check_engine()
        else:
            self.engine_error = None
            self._engine_devices = None
            self._fill_napari_devices()
            self._show_engine()

    def set_engine(self, choice) -> None:
        """Use ``choice`` (a Python path, or ``inprocess``) as this computer's engine once its check passes (the
        choice the Run and Project docks read too)."""
        choice = str(choice)
        session, worker = self.session, self.engine_worker

        def job():
            session.set_engine(choice, should_stop=worker.should_stop)
            runner = session.ensure(worker.should_stop)
            return runner, session.devices(worker.should_stop), None

        def failed(exc):
            self._on_engine_failed(exc)
            self._status(f"that engine cannot be used: {exc}", RED)

        self._engine_jobs.submit("engine", job, label=f"checking the engine {choice}",
                                 returned=self._on_engine_checked, errored=failed)

    def ask_for_engine(self):
        from ._engine_widgets import InterpreterDialog
        from ._run_widget import ENGINE_INTRO

        dlg = InterpreterDialog(detect_interpreters(), title="germquant engine for this computer", intro=ENGINE_INTRO,
                                current=_settings.engine_choice(), allow_inprocess=True, parent=self)
        dlg.accepted.connect(lambda: self.set_engine(dlg.choice()) if dlg.choice() else None)
        self.interpreter_dialog = dlg
        dlg.open()
        return dlg

    def _show_engine(self, problem: str | None = None) -> None:
        try:
            if not self._engine_chosen():
                self.engine_label.setText(NAPARI_TEXT.format(python=sys.executable))
                self.engine_label.setStyleSheet(_css(AMBER))
            elif self.engine_error:
                self.engine_label.setText(f"The engine chosen for this computer cannot be used: {self.engine_error}")
                self.engine_label.setStyleSheet(_css(RED))
            else:
                text = self.session.describe()
                warn = self.session.warning() if self.session.runner() is not None else ""
                extra = [w for w in (warn, f"its device list could not be read: {problem}" if problem else "") if w]
                self.engine_label.setText("\n".join([text, *extra]))
                self.engine_label.setStyleSheet(_css(AMBER) if extra else "")
        except RuntimeError:                          # the widget is gone (napari closing)
            return

    def _fill_napari_devices(self) -> None:
        """The devices of this napari environment (the dock's own training): cuda, mps and cpu, each said to be
        there or not."""
        have = T.device_status()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for dev in T.DEVICES:
            text = DEVICE_TEXT[dev] + ("" if have[dev] else ": not available here (the engine falls back)")
            self.device_combo.addItem(text, dev)
        self.device_combo.setCurrentIndex(T.DEVICES.index(T.default_device()))
        self.device_combo.blockSignals(False)
        self._device_list = "napari"

    def _fill_engine_devices(self) -> None:
        """The devices the engine environment offers (its runtime probe: auto, its accelerators, cpu): auto first
        time, the choice kept when the engine's list is read again."""
        current = self.device_combo.currentData() if self._device_list == "engine" else None
        devices = self._engine_devices or ["auto", "cpu"]
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for dev in devices:
            self.device_combo.addItem(DEVICE_TEXT.get(dev, dev), dev)
        keep = self.device_combo.findData(current) if current in devices else -1
        self.device_combo.setCurrentIndex(keep if keep >= 0 else max(0, self.device_combo.findData("auto")))
        self.device_combo.blockSignals(False)
        self._device_list = "engine"

    # ------------------------------------------------------------------ the project
    def open_project_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Project folder")
        if folder:
            self.set_project(folder)

    def set_project(self, root, *, follow_hub: bool = False) -> None:
        """Work in the project at ``root`` (read in a thread worker). A folder that is no project is refused and
        the dock stays as it was. ``follow_hub``: the Project dock's project (closed when that dock closes it)."""
        root = Path(os.path.abspath(self._clean(str(root))))
        self._reading += 1
        req = self._reading
        self._hub_reading = req if follow_hub else None
        self._files.submit("project", lambda: T.read_project(root), label=f"reading the project {root.name}",
                           returned=lambda p: self._on_project_read(p, req, follow_hub),
                           errored=lambda e: self._on_project_failed(e, req, root))

    def refresh_project(self) -> None:
        """Read the open project again (its models, images and rounds)."""
        if self.project is not None:
            self.set_project(self.project.root, follow_hub=self._following_hub)

    def _on_project_read(self, project: T.TrainingProject, req: int, follow_hub: bool) -> None:
        if req != self._reading:
            return                                    # a newer reading, or a Close, came since
        same = self.project is not None and _same(self.project.root, project.root)
        self.project = project
        self._following_hub = follow_hub or (same and self._following_hub)
        if not same:
            # another project: its folders, never the last project's (the round sets the training folder)
            self.rounds = []
            self._projects_problem = None
            self._fill_rounds()
            self.pairs_edit.setText("")
            self.model_dir_edit.setText("")
            self.source_edit.setText(str(project.results_dir))
            self._read_geometry()
            if project.object_kind in T.OBJECT_KINDS:
                self.kind_combo.setCurrentIndex(self.kind_combo.findData(project.object_kind))
        self._fill_images()
        self._fill_bases(select_first=not same)
        self._show_project()
        self._update_buttons()
        # the rounds come from the engine (`project rounds`): asked when the project opens, and again only when
        # the round folders on disk are not the ones it listed (every reading of the Project dock comes here)
        if not same or self._projects_problem is not None or project.round_folders != self.rounds:
            self._list_rounds()

    def _on_project_failed(self, exc: BaseException, req: int, root: Path) -> None:
        if req != self._reading:
            return
        self._status(f"cannot work in {root}: {exc}", RED)

    def close_project(self) -> None:
        """Leave the project: the folders stay as they are, for a folder of your own choosing."""
        self._reading += 1
        was = self.project
        self.project = None
        self.rounds = []
        self._following_hub = False
        self._projects_problem = None
        self._fill_rounds()
        if was is not None:
            self._fill_images()
            self._fill_bases(select_first=True)
            self.kind_combo.setCurrentIndex(self.kind_combo.findData("auto"))
        self._show_project()
        self._update_buttons()

    def _show_project(self) -> None:
        p = self.project
        self.project_label.setText(p.describe() if p is not None else
                                   "none: the folders below are your own (open a project, or open one in the "
                                   "Project dock)")
        note = ""
        if p is not None:
            if self._projects_problem:
                note = self._projects_problem
            elif not self.rounds:
                note = ("no training round yet: New round makes training/r01/ (the engine names it), where the "
                        "pairs, their corrections and the models of the round go")
        self.notice_label.setText(note)
        self.notice_label.setVisible(bool(note))

    def _list_rounds(self, select: str | None = None) -> None:
        if self.project is None:
            return
        root, req, engine = self.project.root, self._reading, self._engine_chosen()
        worker = self.engine_worker

        def job():
            runner = self._job_runner("projects", engine=engine, should_stop=worker.should_stop)
            return runner.project_rounds(root, should_stop=worker.should_stop)

        self._engine_jobs.submit("rounds", job, label="listing the training rounds",
                                 returned=lambda rec: self._on_rounds(rec, req, select),
                                 errored=lambda e: self._on_rounds_failed(e, req))

    def _on_rounds(self, rec: dict, req: int, select: str | None = None) -> None:
        if req != self._reading or self.project is None:
            return
        self._projects_problem = None
        self.rounds = [str(r) for r in (rec or {}).get("rounds") or []]
        self._fill_rounds(select)
        self._show_project()
        self._update_buttons()

    def _on_rounds_failed(self, exc: BaseException, req: int) -> None:
        if req != self._reading or self.project is None:
            return
        self.rounds = []
        self._projects_problem = str(exc) if isinstance(exc, (FeatureMissing, EngineNotChosen)) else \
            f"the engine could not list the training rounds: {exc}"
        self._fill_rounds()
        self._show_project()
        self._update_buttons()

    def _fill_rounds(self, select: str | None = None) -> None:
        current = select or self.round_combo.currentText()
        self.round_combo.blockSignals(True)
        self.round_combo.clear()
        self.round_combo.addItems(self.rounds)
        idx = self.round_combo.findText(current) if current else -1
        if idx < 0 and self.rounds:
            idx = len(self.rounds) - 1                # the newest round
        if idx >= 0:
            self.round_combo.setCurrentIndex(idx)
        self.round_combo.blockSignals(False)
        if self.project is not None and idx >= 0:
            self._apply_round(self.rounds[idx])

    def select_round(self, name: str) -> bool:
        idx = self.round_combo.findText(str(name))
        if idx < 0:
            return False
        self.round_combo.setCurrentIndex(idx)
        if self.project is not None:
            self._apply_round(str(name))
        return True

    def _on_round_chosen(self, name: str) -> None:
        if self.project is not None and name:
            self._apply_round(name)

    def _apply_round(self, name: str) -> None:
        """The round's folders: its pairs (the training folder) and models/<round>_nuclei (the model folder); its
        pairs are shown when it has any and the slice shown has no unsaved edits."""
        project = self.project
        if project is None:
            return
        pairs = project.round_dir(name)
        same = self._clean(self.pairs_edit.text()) and _same(self._clean(self.pairs_edit.text()), pairs)
        self.pairs_edit.setText(str(pairs))
        self.model_dir_edit.setText(str(project.model_dir(name)))
        if same and self._pairs_dir is not None and _same(self._pairs_dir, pairs):
            return
        if self.is_dirty():
            self._status("the slice shown has unsaved edits: save or revert them, then press Open training folder to "
                         f"see the pairs of round {name}", AMBER)
            return
        if (pairs / "manifest.csv").is_file() or (pairs / "image").is_dir():
            self.open_pairs(unsaved="discard")
        else:
            self._pairs, self._index, self._pairs_dir, self._loaded_labels = [], -1, pairs, None
            self._valid = {}
            self.pair_label.setText(f"no slice pairs in round {name} yet: export or import some")
            self._update_corrected_label()
            self._update_buttons()

    def new_round(self, name: str | None = None) -> bool:
        """``germquant project rounds --new [NAME]`` through the engine: the engine names it (r01, r02 and so on)
        unless ``name`` is given."""
        if self.project is None:
            self._status("open a project first: rounds are folders of a project's training/", AMBER)
            return False
        if name is not None and not T.ROUND_RE.fullmatch(str(name)):
            self._status(f"round name {name!r} is not one the engine takes: letters, digits, '_' and '-', at most 40",
                         RED)
            return False
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        self._sync_engine()
        root, engine, worker = self.project.root, self._engine_chosen(), self.runner
        self.last_error = None
        self._start_step("round", lambda: self._job_runner("projects", engine=engine, should_stop=worker.should_stop)
                         .project_rounds(root, new=name or True, plugin_version=plugin_version(),
                                         should_stop=worker.should_stop),
                         label="making a training round", returned=lambda rec: self._on_round_made(rec, root),
                         what="new round", engine=False)
        return True

    def _on_round_made(self, rec: dict, root: Path) -> None:
        created = (rec or {}).get("created")
        if self.project is not None and _same(self.project.root, root):
            self._projects_problem = None
            self.rounds = [str(r) for r in rec.get("rounds") or []]
            self._fill_rounds(select=created)
            self._show_project()
        self._status(f"made the training round {created}: {rec.get('path')}", GREEN)

    def _fill_images(self) -> None:
        """The project's images in the import's image list (the id, with the path as its data). A reading of the
        same project keeps the image chosen; without a project the path of the image chosen stays typed."""
        text = self.image_combo.currentText()
        idx = self.image_combo.findText(text)
        chosen = self.image_combo.itemData(idx) if idx >= 0 else None
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for img in (self.project.images if self.project is not None else []):
            self.image_combo.addItem(img.image_id, str(img.path))
        if self.project is not None and self.project.images:
            keep = self.image_combo.findText(text)
            self.image_combo.setCurrentIndex(keep if keep >= 0 else 0)
        else:
            self.image_combo.setEditText(str(chosen) if chosen else (text if self.project is None else ""))
        self.image_combo.blockSignals(False)
        self._read_channels()

    def _fill_bases(self, select_first: bool) -> None:
        current = self.base_combo.currentText()
        items = T.project_base_choices(self.project)
        self.base_combo.blockSignals(True)
        self.base_combo.clear()
        self.base_combo.addItems(items)
        if select_first or current not in items:
            self.base_combo.setCurrentIndex(0)
        else:
            self.base_combo.setCurrentText(current)
        self.base_combo.blockSignals(False)

    def _on_hub_opened(self, root: str) -> None:
        self.set_project(root, follow_hub=True)

    def _on_hub_closed(self) -> None:
        # the Project dock's project, open here or still being read: closed with it; one opened here stays
        if self._following_hub or (self._hub_reading is not None and self._hub_reading == self._reading):
            self.close_project()

    def _on_hub_running(self, _running: bool) -> None:
        self._update_buttons()

    def _on_hub_changed(self) -> None:
        hub = self.hub
        if self.project is not None and hub is not None and _same(getattr(hub, "root", None), self.project.root):
            self.refresh_project()

    # ------------------------------------------------------------------ folders
    def set_source(self, path: str | Path) -> None:
        text = self._clean(path)
        self.source_edit.setText(text)
        if text and not self._clean(self.pairs_edit.text()):
            src = Path(text)
            self.set_pairs_dir(src.parent / f"{src.name}_training")
        self._read_geometry()

    def set_pairs_dir(self, path: str | Path) -> None:
        text = self._clean(path)
        self.pairs_edit.setText(text)
        if text and not self._clean(self.model_dir_edit.text()):
            p = Path(text)
            self.model_dir_edit.setText(str(p.parent / f"{p.name}_model"))

    # ------------------------------------------------------------------ the geometry of each run
    def _read_geometry(self) -> None:
        """Read what the export would write for each run of the source folder (in the reading worker)."""
        self._geometry_req += 1
        req = self._geometry_req
        self.geometry = None
        text = self._clean(self.source_edit.text())
        if not text:
            self._show_geometry()
            return
        source = Path(text)

        def job():
            return T.read_geometry(source) if source.is_dir() else None

        self.geometry_label.setText("reading the runs ...")
        self.geometry_label.setStyleSheet("")
        self._files.submit("geometry", job, label="reading the runs' geometry",
                           returned=lambda g: self._on_geometry(g, req),
                           errored=lambda e: self._on_geometry(None, req, problem=f"{type(e).__name__}: {e}"))

    def _on_geometry(self, geoms, req: int, problem: str | None = None) -> None:
        if req != self._geometry_req:
            return
        self.geometry = geoms
        self._show_geometry()
        if problem:
            self.geometry_label.setText(f"the runs could not be read: {problem}")
            self.geometry_label.setStyleSheet(_css(AMBER))

    def _show_geometry(self) -> None:
        ticked = tuple(o for o, cb in self.orient_checks.items() if cb.isChecked())
        auto = self._auto_supported()
        text, warn = T.geometry_text(self.geometry, ticked, auto_supported=auto)
        self.geometry_label.setText(text)
        self.geometry_label.setStyleSheet(_css(AMBER) if warn else "")
        choose = self.geometry is None or not auto or any(g.kind == "3d" for g in self.geometry)
        for cb in self.orient_checks.values():
            cb.setEnabled(choose)

    # ------------------------------------------------------------------ export
    def export_settings(self) -> T.ExportSettings:
        ticked = tuple(o for o, cb in self.orient_checks.items() if cb.isChecked())
        orients = T.export_orientations(ticked, self.geometry, auto_supported=self._auto_supported())
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
        self._sync_engine()
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
        engine = self._engine_chosen()
        where = "through the engine" if engine else "in this napari environment"
        self._status(f"exporting slices from {source} into {pairs_dir} {where} ...\n{settings.command(source, pairs_dir)}")
        if engine:
            recipe, worker = self._recipe(), self.runner

            def job():
                runner = self._job_runner(engine=True, recipe=recipe, should_stop=worker.should_stop)
                runner.export_training(str(source), str(pairs_dir), orientations=settings.orientations,
                                       slices_per_image=settings.slices_per_image, seed=settings.seed,
                                       zone_column=settings.zone_column, min_labels=settings.min_labels,
                                       max_gb=settings.max_gb, say=_say, should_stop=worker.should_stop)
                return T.read_manifest_csv(pairs_dir)
        else:
            def job():
                return T.export_slices(source, pairs_dir, settings)
        self._start_step("export", job, label=f"exporting training slices from {source.name}",
                         returned=lambda manifest: self._on_exported(manifest, pairs_dir, source, settings),
                         what="export", engine=engine)
        return True

    def _recipe(self) -> tuple | None:
        """The open project's (pack id, recipe id): its recipe's own interpreter on this computer runs its steps."""
        return (self.project.pack_id, self.project.recipe) if self.project is not None else None

    def _in_view(self, folder) -> bool:
        """Whether a step's folder belongs to what the dock works on now: any folder without a project, else a
        folder inside the open project (another project may have been opened while the step ran)."""
        return self.project is None or T.project_relative(self.project.root, folder) is not None

    def _on_exported(self, manifest, pairs_dir: Path, source: Path, settings: T.ExportSettings) -> None:
        self.last_manifest = manifest
        if not self._in_view(pairs_dir):
            self._status(f"{T.export_summary(manifest, pairs_dir)}\nAnother project was opened meanwhile: open "
                         f"{pairs_dir} to see these pairs.", AMBER)
            return
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

    # ------------------------------------------------------------------ import labels made elsewhere
    def set_labels_file(self, path: str | Path) -> None:
        """The labels file to import; its format follows from its name, and in a project the image whose id the
        name starts with is chosen."""
        text = self._clean(path)
        self.labels_edit.setText(text)
        fmt = T.guess_format(text) if text else None
        if fmt is not None:
            self.format_combo.setCurrentIndex(self.format_combo.findData(fmt))
        self._on_format_changed()
        if self.project is not None and text:
            stem = Path(text).name
            ids = sorted((img.image_id for img in self.project.images if stem.startswith(img.image_id)), key=len)
            if ids:
                self.image_combo.setCurrentIndex(self.image_combo.findText(ids[-1]))

    def set_import_format(self, fmt: str) -> None:
        self.format_combo.setCurrentIndex(self.format_combo.findData(fmt))

    def set_import_image(self, path: str | Path) -> None:
        """The image the labels were drawn on: a path (or a project image's id)."""
        text = self._clean(path)
        idx = self.image_combo.findText(text)
        if idx < 0:
            idx = next((i for i in range(self.image_combo.count()) if _same(self.image_combo.itemData(i), text)), -1)
        if idx >= 0:
            self.image_combo.setCurrentIndex(idx)
        else:
            self.image_combo.setEditText(text)
        self._read_channels()

    def _import_image(self) -> str:
        text = self.image_combo.currentText().strip()
        idx = self.image_combo.findText(text)
        data = self.image_combo.itemData(idx) if idx >= 0 else None
        return str(data) if data else self._clean(text)

    def _on_format_changed(self) -> None:
        pickle = self.format_combo.currentData() == "cellpose-npy"
        if not pickle:
            self.pickle_check.setChecked(False)
        self.pickle_check.setEnabled(pickle)
        self.pickle_label.setVisible(pickle)

    def _runner_takes_channel(self) -> bool:
        runner = self._ui_runner()
        fn = getattr(runner, "import_labels", None)
        try:
            return fn is not None and "channel" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    def _read_channels(self) -> None:
        """The channel names of the import's image (its header, in the reading worker)."""
        self._channels_req += 1
        req = self._channels_req
        self._channels = []
        image = self._import_image()
        if not image:
            self._show_channels()
            return

        def job():
            return T.image_channels(image) if os.path.isfile(image) else []

        self._files.submit("channels", job, label="reading the image's channels",
                           returned=lambda names: self._on_channels(names, req),
                           errored=lambda e: self._on_channels([], req))
        self._show_channels()

    def _on_channels(self, names: list[str], req: int) -> None:
        if req != self._channels_req:
            return
        self._channels = [str(n) for n in names or []]
        self._show_channels()

    def _show_channels(self) -> None:
        current = self.channel_combo.currentData()
        takes = self._runner_takes_channel()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItem(CHANNEL_DEFAULT, "")
        if takes:
            for i, name in enumerate(self._channels):
                unique = self._channels.count(name) == 1 and name != ""
                self.channel_combo.addItem(f"{i}: {name}", name if unique else str(i))
            idx = self.channel_combo.findData(current) if current else 0
            self.channel_combo.setCurrentIndex(max(0, idx))
        self.channel_combo.blockSignals(False)
        self.channel_combo.setEnabled(takes and bool(self._channels))
        self.channel_combo.setToolTip("--channel: the image channel the labels were drawn on" if takes
                                      else CHANNEL_NOT_PASSED)

    def select_channel(self, value) -> bool:
        """Choose the import's channel by name or index (None: the engine's default)."""
        if value is None:
            self.channel_combo.setCurrentIndex(0)
            return True
        for i in range(1, self.channel_combo.count()):
            index, _, name = self.channel_combo.itemText(i).partition(": ")
            if str(value) in (name, index):
                self.channel_combo.setCurrentIndex(i)
                return True
        return False

    def _chosen_channel(self):
        if not self.channel_combo.isEnabled() or not self._runner_takes_channel():
            return None
        data = self.channel_combo.currentData()
        return data if data not in (None, "") else None

    def import_labels(self) -> bool:
        """``germquant import-labels`` through the engine runner, in a thread worker, into the training folder (in a
        project: the round's). The engine's refusals (a pickle without the box, a test image of a benchmark split,
        labels on another grid) and warnings are shown."""
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        self._sync_engine()
        src = self._clean(self.labels_edit.text())
        fmt = self.format_combo.currentData()
        image = self._import_image()
        pairs = self._clean(self.pairs_edit.text())
        if not src:
            self._status("choose the labels file to import", RED)
            return False
        if not image:
            self._status("choose the image the labels were drawn on", RED)
            return False
        if not pairs:
            self._status("choose a training folder (in a project: a round)", RED)
            return False
        if self._pairs_dir is not None and _same(self._pairs_dir, pairs) and self.is_dirty():
            self._status("save or revert the slice you are editing first: the import rewrites this folder", AMBER)
            return False
        allow = bool(self.pickle_check.isChecked() and fmt == "cellpose-npy")
        frame = None if self.frame_spin.value() < 0 else int(self.frame_spin.value())
        channel = self._chosen_channel()
        kw: dict = {"allow_pickle": allow, "frame": frame}
        if channel is not None:
            kw["channel"] = channel
        cmd = T.import_command(src, fmt, image, pairs, allow_pickle=allow, frame=frame, channel=channel)
        engine, worker = self._engine_chosen(), self.runner
        self.last_import = None
        self.last_error = None
        self._status(f"importing {Path(src).name} for {Path(image).name} into {pairs} ...\n{cmd}")

        def job():
            runner = self._job_runner("import_labels", engine=engine, should_stop=worker.should_stop)
            return runner.import_labels(src, fmt, image, pairs, should_stop=worker.should_stop, **kw)

        self._start_step("import", job, label=f"importing labels for {Path(image).name}",
                         returned=lambda rec: self._on_imported(rec, Path(pairs), cmd), what="import", engine=engine)
        return True

    def _on_imported(self, rec: dict, pairs_dir: Path, cmd: str) -> None:
        self.last_import = rec
        summary = T.import_summary(rec)
        color = AMBER if rec.get("warnings") else GREEN
        if not self._in_view(pairs_dir):
            self._status(f"{summary}\nAnother project was opened meanwhile: open {pairs_dir} to see these pairs.\n"
                         f"same as: {cmd}", AMBER)
            return
        if self.is_dirty():
            self._status(f"{summary}\nThe slice shown has unsaved edits: save or revert them, then press Open training "
                         f"folder.\nsame as: {cmd}", AMBER)
            return
        self.set_pairs_dir(pairs_dir)
        n = self.open_pairs(unsaved="discard")
        self._status(f"{summary}\n{n} pair(s) in the folder.\nsame as: {cmd}", color)
        set_viewer_status(self.viewer, f"imported {rec.get('n_pairs')} pair(s) of {rec.get('image_id')}")

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
        # napari keeps a Labels layer's undo history when its data is replaced, so one Ctrl+Z after Next would
        # paste the previous pair's stroke into this one (and Save would accept it as a correction). Switching
        # editable off resets the history (napari's _on_editable_changed); the paint mode is restored after.
        mode = self.labels_layer.mode
        self.labels_layer.editable = False
        self.labels_layer.editable = True
        try:
            self.labels_layer.mode = mode
        except Exception:  # noqa: BLE001 - a mode the restored layer refuses falls back to napari's default
            pass
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
        if self._busy_kind in ("export", "import", "train"):
            self._status("saving is paused while the dock exports, imports or trains on this folder (the model card "
                         "records the sha256 of every pair when training ends)", AMBER)
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
        kind = self.kind_combo.currentData()
        return T.TrainSettings(out=Path(out), name=self._clean(self.name_edit.text()) or None, base=base,
                               epochs=int(self.epochs_spin.value()), split_by=self.split_combo.currentText(),
                               val_frac=float(self.val_spin.value()), seed=int(self.split_seed_spin.value()),
                               device=str(self.device_combo.currentData()),
                               system=self._clean(self.system_edit.text()) or None,
                               kind=kind if kind in T.OBJECT_KINDS else None,
                               dataset=self._clean(self.dataset_edit.text()) or None)

    def select_device(self, device: str) -> None:
        idx = self.device_combo.findData(device)
        if idx < 0:
            raise ValueError(f"{device!r} is not offered here: {[self.device_combo.itemData(i) for i in range(self.device_combo.count())]}")
        self.device_combo.setCurrentIndex(idx)

    def train(self) -> bool:
        """Train in a thread worker on the pairs of the training folder."""
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        self._sync_engine()
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
        epochs = int(settings.epochs)
        engine = self._engine_chosen()
        where = "through the engine" if engine else "in this napari environment"
        self._status(f"training {settings.model_name()} on {settings.device} {where} ...\n{notes}\n"
                     f"{settings.command(pairs_dir)}")
        if engine:
            recipe, worker = self._recipe(), self.runner

            def job():
                runner = self._job_runner(engine=True, recipe=recipe, should_stop=worker.should_stop)
                return runner.train_nuclei(str(pairs_dir), str(settings.out), base=settings.base, name=settings.name,
                                           epochs=epochs, split_by=settings.split_by, val_frac=float(settings.val_frac),
                                           seed=int(settings.seed), device=settings.device, say=_say,
                                           should_stop=worker.should_stop, **settings.card_fields())

            returned = lambda res: self._on_trained_engine(res, settings, pairs_dir, notes)  # noqa: E731
        else:
            def job():
                return T.train(pairs_dir, settings)

            returned = lambda res: self._on_trained(res, settings, pairs_dir, notes)  # noqa: E731
        self._start_step("train", job, label=f"training {settings.model_name()} ({settings.device})", returned=returned,
                         what="training", engine=engine,
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
        self._after_model_written()

    def _on_trained_engine(self, res: dict, settings: T.TrainSettings, pairs_dir: Path, notes: str = "") -> None:
        """A training through the engine runner: its record (exit code, lines, model and card paths)."""
        self.last_train = res
        model, card = res.get("model_path"), res.get("card_path")
        if model:
            self.model_edit.setText(str(model))
            self.reg_name_edit.setText(Path(str(model)).name)
        if card:
            self.card_edit.setText(str(card))
        self._status(f"{T.engine_train_summary(res, settings.device)}\n{notes}\n\nsame as:\n"
                     f"{settings.command(pairs_dir)}", GREEN)
        set_viewer_status(self.viewer, f"model {settings.model_name()} trained through the engine")
        self._after_model_written()

    def _after_model_written(self) -> None:
        """A new model of the project is one of its starting weights now: read the project again."""
        if self.project is not None:
            self.refresh_project()

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

    # ------------------------------------------------------------------ use in project, derive pack
    def _project_model(self) -> tuple[str, str | None] | None:
        """The model file and its card (the card field, else <file>.card.json next to it when there), or None after
        saying what is missing."""
        model = self._clean(self.model_edit.text())
        if not model or not os.path.isfile(model):
            self._status("choose the model file train-nuclei wrote (training/<round>/models/<name>/<file> in a "
                         "project)", RED)
            return None
        card = self._clean(self.card_edit.text()) or None
        if card is None:
            guess = Path(model).with_name(Path(model).name + ".card.json")
            card = str(guess) if guess.is_file() else None
        return model, card

    def use_in_project(self) -> bool:
        """``germquant project use-model DIR --model training/<round>/models/<name>/<file>`` through the engine:
        the project's recipe names the model by its path in the project, its config_hash changes and validated
        turns false; the Project dock then reads the project again. Waits while a run of the project goes."""
        if self.project is None:
            self._status("open a project first: Use in project names a model of the project in its recipe", AMBER)
            return False
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        self._sync_engine()
        if self._running():
            self._status("a run of this project is going: Use in project changes its recipe, so it waits until the "
                         "run has finished", AMBER)
            return False
        got = self._project_model()
        if got is None:
            return False
        model, card = got
        root = self.project.root
        rel = T.project_relative(root, model)
        if rel is None:
            self._status(f"{model} is not inside the project ({root}): Use in project names a model file inside it "
                         f"(training/<round>/models/<name>/<file>); train it into a round, or derive a pack from it",
                         RED)
            return False
        card_rel = None
        if self._clean(self.card_edit.text()):
            card_rel = T.project_relative(root, card)
            if card_rel is None:
                self._status(f"the card {card} is not inside the project ({root}): keep it next to the model", RED)
                return False
        engine, worker = self._engine_chosen(), self.runner
        self.last_use = None
        self.last_error = None
        self._status(f"using {rel} in the project ...")
        self._start_step("use", lambda: self._job_runner("projects", engine=engine, should_stop=worker.should_stop)
                         .project_use_model(root, rel, card=card_rel, plugin_version=plugin_version(),
                                            should_stop=worker.should_stop),
                         label="using the model in the project", returned=lambda rec: self._on_used(rec, root, rel),
                         what="use in project", engine=False)
        return True

    def _on_used(self, rec: dict, root: Path, rel: str) -> None:
        self.last_use = rec
        validated = (rec.get("validated") or {}).get("value") if isinstance(rec.get("validated"), dict) else None
        state = "validated: false" if not validated else "validated: true"
        self._status(f"the project's recipe now names {rel} (segmentation.nuclei.cellpose_model): config_hash "
                     f"{rec.get('config_hash')} ({state}). The project's finished images are stale until they are run "
                     f"again.\nsame as: germquant project use-model {root} --model {rel}", GREEN)
        hub = self.hub
        if hub is not None and _same(getattr(hub, "root", None), root):
            hub.request_reload()
        if self.project is not None and _same(self.project.root, root):
            self.refresh_project()

    def derive_pack(self) -> bool:
        """``germquant pack derive --from ID@VER --recipe R --model PATH --card CARD --out <project>/packs`` through
        the engine, then the engine's pack listing with the project (where the derived pack shows)."""
        if self.project is None:
            self._status("open a project first: a derived pack goes into the project's packs/", AMBER)
            return False
        if self.runner.busy:
            self._status("the dock is busy; wait for the current step to finish", AMBER)
            return False
        self._sync_engine()
        got = self._project_model()
        if got is None:
            return False
        model, card = got
        if card is None:
            self._status("pack derive needs the model's card (<file>.card.json, which train-nuclei writes next to the "
                         "model): choose it", RED)
            return False
        project = self.project
        from_ref, recipe, out, root = project.pinned, project.recipe, str(project.packs_dir), project.root
        engine, worker = self._engine_chosen(), self.runner
        self.last_derive = None
        self.last_error = None
        self._status(f"deriving a pack from {from_ref} recipe {recipe} with {Path(model).name} ...")

        def job():
            runner = self._job_runner("pack_derive", engine=engine, should_stop=worker.should_stop)
            rec = runner.pack_derive(from_ref, recipe, model, card, out, should_stop=worker.should_stop)
            try:
                return rec, runner.list_packs(project=root, should_stop=worker.should_stop), None
            except RunnerError as e:                  # derived; the listing with the project is not there
                return rec, None, str(e)

        self._start_step("derive", job, label="deriving a pack", what="pack derive", engine=False,
                         returned=lambda got: self._on_derived(*got, from_ref=from_ref, recipe=recipe, model=model,
                                                                card=card, out=out))
        return True

    def _on_derived(self, rec: dict, listing: dict | None, problem: str | None, *, from_ref: str, recipe: str,
                    model: str, card: str, out: str) -> None:
        self.last_derive = rec
        entry = T.derived_entry(listing, rec) if listing is not None else None
        lines = [f"derived pack {rec.get('id')}@{rec.get('version')} ({rec.get('status')}): recipe {rec.get('recipe')}, "
                 f"config_hash {rec.get('config_hash')}, in {rec.get('folder')}"]
        if entry is not None:
            lines.append(f"listed in this project's packs (source {entry.get('source')}): the Run and Project docks "
                         f"offer it with the project")
        elif problem:
            lines.append(f"not listed with the project: {problem}")
        else:
            lines.append("the engine's pack listing with this project does not show it")
        lines.append(f"same as: germquant pack derive --from {from_ref} --recipe {recipe} --model {model} --card "
                     f"{card} --out {out}")
        self._status("\n".join(lines), GREEN if entry is not None else AMBER)

    # ------------------------------------------------------------------ worker plumbing
    def _start_step(self, kind: str, job, *, label: str, returned, what: str, engine: bool, line_format=None) -> None:
        """One step in the dock's worker. ``engine``: an engine process runs it, so Cancel can end it."""
        runner = self._ui_runner() if engine else None
        self._job_cancellable = bool(engine and getattr(runner, "cancellable", False))
        self._busy_kind = kind
        self._update_buttons()
        self.runner.start(job, label=label, returned=returned, errored=lambda e: self._on_errored(e, what),
                          finished=self._on_finished, line_format=line_format)

    def cancel(self) -> bool:
        """End the engine's export, import or training (Cancel). False when none runs, or it cannot be stopped."""
        if not self.runner.busy or self._busy_kind not in CANCELLABLE:
            return False
        if not self._job_cancellable:
            self._status("this step runs in napari's own process and cannot be stopped; wait for it to finish", AMBER)
            return False
        self.runner.cancel()
        self._status("cancelling: the engine and every process it started are being ended ...", AMBER)
        return True

    def _on_errored(self, exc: BaseException, what: str) -> None:
        self.last_error = exc
        if isinstance(exc, RunCancelled):
            self._status(f"{what} cancelled: {exc}", AMBER)
            set_viewer_status(self.viewer, f"germquant {what} cancelled")
            return
        if isinstance(exc, EngineRefused):
            self._status(f"{what} refused by the engine:\n{exc}", RED)
        elif isinstance(exc, RunnerError):
            self._status(f"{what} FAILED: {exc}", RED)
        else:
            self._status(f"{what} FAILED: {format_exception(exc)}", RED)
        set_viewer_status(self.viewer, f"germquant {what} failed: {type(exc).__name__}: {exc}")
        try:
            from napari.utils.notifications import show_error

            show_error(f"germquant {what} failed: {type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 - no notification manager in a bare Qt test
            pass

    def _on_finished(self) -> None:
        self._busy_kind = None
        self._job_cancellable = False
        self._update_buttons()
