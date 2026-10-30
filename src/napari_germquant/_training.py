"""The nucleus training round trip without Qt: export slices, read and correct pairs, train, register.

Every step is the engine's own function, called the way its command calls it, so a step done in the
Training dock and the same step typed in a terminal do the same thing:

* ``export_slices``: ``germquant.segment.training.export_training_pairs`` (``germquant export-training``);
* ``load_pair_list``: ``training.load_pairs``, the very list ``train-nuclei`` trains on (the manifest rows
  whose two files exist, or every ``image/*.tif`` with a ``labels/`` file of the same name);
* ``save_labels``: the one thing the engine leaves to a person, writing a corrected ``labels/<name>.tif``
  back under the same name. The write is atomic (a temporary file in the same folder, flushed, then
  renamed over the old file), so a crash never leaves a half-written label image, and the manifest is not
  touched. Each save is recorded in ``corrections.json`` next to the manifest (name, time, sha256 of the
  saved file, nucleus counts before and after), which the engine ignores; a record counts only while the
  file still has that sha256, so a re-export that overwrote a correction is seen for what it is;
* ``train``: ``training.train_nuclei`` (``germquant train-nuclei``); the device choice is passed the way the
  engine reads it (``--cpu`` for the CPU, the ``GERMQUANT_DEVICE`` variable for cuda or mps);
* ``register``: ``models_registry.register_local_model`` (``germquant register-model``).

Those run in napari's own process, where the dock keeps computing when no engine is chosen for this computer.
With an engine chosen the dock sends the same steps through the engine runner instead (tranche 5 Train B,
decision D36; ``_runner``), and the project mode adds what this module also prepares without Qt (the design's
section 8.6 and 10; plugin_specs.md 4.4):

* the geometry of each finished run, read from its record as export-training reads it (``read_geometry``):
  a 3D run gives the xy, xz and yz slices the dock lets one choose, a projected run one ``proj`` pair (the 2D
  image its segmenter saw), a single-plane run its plane as ``xy``; the export then passes ``--orientations
  auto``, or the chosen subset of a 3D run's slices (``export_orientations``);
* the project the dock works in, read in process (``read_project``: the manifest, the pinned pack's copy, the
  models trained in it), its round folders (``training/<round>/``: the pairs, corrections.json and
  ``models/<name>/``) and the starting weights it offers (``project_base_choices``);
* labels made elsewhere: the import format from the file name (``guess_format``), the channels of the image
  (``image_channels``), and what an import's record says (``import_summary``);
* Use in project and Derive pack: the model named by its path in the project (``project_relative``) and the
  derived pack found in the engine's pack listing (``derived_entry``).
"""
from __future__ import annotations

# germquant.api first (through _engine): the torch-before-CuPy import order of the workstation
from . import _engine  # noqa: F401,I001

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml

from germquant import device as gq_device
from germquant import models_registry
from germquant.fsutil import long_path
from germquant.segment import training as gq_training

from ._engine import _quote

CORRECTIONS_FILE = "corrections.json"
DEVICES = ("cuda", "mps", "cpu")
# what TrainSettings takes: the devices above, and auto (the engine's own choice) for an engine's device list
TRAIN_DEVICES = ("auto", *DEVICES)
DEVICE_ENV = "GERMQUANT_DEVICE"
ORIENTATIONS = gq_training.ORIENTATIONS
AUTO = "auto"                         # --orientations auto: what export writes follows each run's geometry
PROJ = "proj"                         # the one 2D pair of a projected run
OBJECT_KINDS = ("nucleus", "cell")    # the card's object_kind (train-nuclei --kind)
IMPORT_FORMATS = ("tiff", "png", "cellacdc-npz", "cellpose-npy")
# the warning the dock shows next to "Allow pickle" (the engine's own words, germquant.import_labels)
PICKLE_WARNING = "a _seg.npy file is a Python pickle: loading it can run code; allow only files you made"
MANIFEST = "germquant_project.yaml"
# training/<round>/ (the engine's round names, germquant.project.rounds; decision D29)
ROUND_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
TRAINING_DIR = "training"
# the gate docs/TIMELAPSE.md sets for a trained model, read off its card before the model makes numbers
DESIGN_GATE = ("design gate (docs/TIMELAPSE.md): F1 at IoU 0.5 of 0.85 or better on pachytene slices and 0.75 on "
               "transition-zone slices, miss rate under 10 percent")

# one epoch line of cellpose.train.train_seg: "<epoch>, train_loss=0.1234, test_loss=0.2345, LR=..., time 1.00s"
_EPOCH_RE = re.compile(r"^\s*(\d+), train_loss=([^,]+), test_loss=([^,]+),")


# ---------------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExportSettings:
    """The ``germquant export-training`` options (the same defaults). ``orientations`` names the slices of the 3D
    runs, or is ``"auto"``: what each run's geometry gives (an engine with the proj export, export_auto_supported)."""
    orientations: tuple[str, ...] | str = ORIENTATIONS
    slices_per_image: int = 3
    seed: int = 0
    min_labels: int = 3
    max_gb: float = 2.0
    zone_column: str = "zone"

    def orientations_text(self) -> str:
        return self.orientations if isinstance(self.orientations, str) else ",".join(self.orientations)

    def argv(self, source: str | Path, pairs_dir: str | Path) -> list[str]:
        return ["export-training", str(source), "--out", str(pairs_dir), "--orientations", self.orientations_text(),
                "--slices-per-image", str(int(self.slices_per_image)), "--seed", str(int(self.seed)),
                "--zone-column", str(self.zone_column), "--min-labels", str(int(self.min_labels)),
                "--max-gb", f"{float(self.max_gb):g}"]

    def command(self, source: str | Path, pairs_dir: str | Path) -> str:
        return "germquant " + " ".join(_quote(a) for a in self.argv(source, pairs_dir))


def export_slices(source: str | Path, pairs_dir: str | Path, settings: ExportSettings = ExportSettings()) -> pd.DataFrame:
    """``export_training_pairs`` with the arguments ``germquant export-training`` hands it; returns the manifest."""
    o = settings.orientations
    return gq_training.export_training_pairs(
        source, pairs_dir, orientations=o if isinstance(o, str) else tuple(o),
        slices_per_image=int(settings.slices_per_image), seed=int(settings.seed),
        zone_column=settings.zone_column or None, min_labels=int(settings.min_labels),
        max_bytes=float(settings.max_gb) * 1e9)


def export_auto_supported() -> bool:
    """Whether this environment's export-training follows each run's geometry (``--orientations auto`` and the
    proj pair of a projected run: the tranche-5 Train B engine). An older engine slices every run in xy, xz and
    yz, a projected run too (its 2D labels copied through z)."""
    return getattr(gq_training, "PROJ", None) == PROJ


def read_manifest_csv(pairs_dir: str | Path) -> pd.DataFrame:
    """The manifest.csv of a pairs folder (an empty table without one): what an export through the engine left."""
    path = Path(pairs_dir) / "manifest.csv"
    if not os.path.isfile(long_path(path)):
        return pd.DataFrame(columns=["image_id", "orientation"])
    return pd.read_csv(long_path(path), low_memory=False)


def export_summary(manifest: pd.DataFrame, pairs_dir: str | Path) -> str:
    """The line ``germquant export-training`` prints."""
    n_img = manifest["image_id"].nunique() if len(manifest) else 0
    per = manifest.groupby("orientation").size().to_dict() if len(manifest) else {}
    return (f"{len(manifest)} slice pair(s) from {n_img} run(s) -> {pairs_dir}  "
            f"({', '.join(f'{k}: {v}' for k, v in sorted(per.items()))})")


def export_conflicts(source: str | Path, pairs_dir: str | Path) -> list[str]:
    """The corrected pairs an export of ``source`` into ``pairs_dir`` would undo: exporting again replaces the
    manifest rows of every image it exports (and overwrites the files of the slices it draws again), so a
    corrected slice of such an image would be overwritten or dropped from the manifest. Empty when nothing
    is at risk (no corrections, a new folder, or images the export does not touch)."""
    pairs_dir = Path(pairs_dir)
    if not (pairs_dir / "manifest.csv").is_file():
        return []
    done = valid_corrections(pairs_dir)
    if not done:
        return []
    ids = {r.image_id for r in gq_training.find_runs(source)}
    return sorted(p.name for p in load_pair_list(pairs_dir) if p.name in done and p.image_id in ids)


# ---------------------------------------------------------------------------------------------------
# the geometry of each run: what export writes for it
# ---------------------------------------------------------------------------------------------------

GEOMETRIES = ("3d", "projected", "planar")


@dataclasses.dataclass(frozen=True)
class RunGeometry:
    """One finished run as the export sees it: ``kind`` 3d, projected or planar, with a projected run's record
    (segment_projection: the mode, and the plane it chose)."""
    image_id: str
    folder: str
    kind: str
    projection: dict | None = None

    def writes(self, orientations=ORIENTATIONS) -> list[str]:
        """What the export writes for this run: the chosen slices of a 3D run (every one for auto), the proj pair
        of a projected run, the plane of a single-plane run as xy."""
        if self.kind == "projected":
            return [PROJ]
        if self.kind == "planar":
            return ["xy"]
        return list(ORIENTATIONS if isinstance(orientations, str) else orientations)


def run_record(folder: str | Path, image_id: str) -> dict:
    """The record a finished run's geometry is read from, as export-training reads it: the folder's
    run_manifest.json, else the run's ``<id>__stages.json`` (a batch writes its manifest at the batch root only),
    with the keys a stage record nests under run_geometry or geometry lifted to the top."""
    folder = Path(folder)
    man = folder / "run_manifest.json"
    path = man if os.path.isfile(long_path(man)) else folder / f"{image_id}__stages.json"
    with open(long_path(path), encoding="utf-8") as fh:
        rec = json.load(fh)
    if not isinstance(rec, dict):
        raise ValueError(f"{path}: not a run record")
    for nested in ("run_geometry", "geometry"):
        if isinstance(rec.get(nested), dict):
            for k, v in rec[nested].items():
                rec.setdefault(k, v)
    return rec


def geometry_of(record: dict) -> tuple[str, dict | None]:
    """The engine's rule (docs/TRAINING.md): planar for a single-plane run (``"planar": true``, checked first),
    projected for a run whose record holds segment_projection, else 3d."""
    if record.get("planar") is True:
        return "planar", None
    proj = record.get("segment_projection")
    if isinstance(proj, dict):
        return "projected", dict(proj)
    return "3d", None


def read_geometry(source: str | Path) -> list[RunGeometry]:
    """The geometry of every finished run under ``source`` (a results folder, a series folder or a batch tree), in
    the order the export takes them (find_runs). A run whose record cannot be read counts as 3D: the export reads
    it again and says what is wrong."""
    out = []
    for r in gq_training.find_runs(source):
        try:
            kind, proj = geometry_of(run_record(r.folder, r.image_id))
        except (OSError, ValueError):
            kind, proj = "3d", None
        out.append(RunGeometry(r.image_id, str(r.folder), kind, proj))
    return out


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'' if n == 1 else 's'}"


def _listed(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def geometry_text(geoms: list[RunGeometry] | None, orientations=ORIENTATIONS, *,
                  auto_supported: bool = True) -> tuple[str, bool]:
    """What the export writes for these runs, one line per geometry, and whether that is a warning: an engine
    from before the proj export slices a projected or single-plane run as if it were 3D."""
    if geoms is None:
        return "", False
    if not geoms:
        return ("no finished run here yet (a folder with run_manifest.json or <id>__stages.json, "
                "<id>__nuclei_labels.tif and <id>__image_summary.csv)"), False
    counts = {k: sum(1 for g in geoms if g.kind == k) for k in GEOMETRIES}
    chosen = ORIENTATIONS if isinstance(orientations, str) else tuple(orientations)
    lines = []
    if counts["3d"]:
        lines.append(f"{_count(counts['3d'], '3D run')}: {_listed(chosen) if chosen else 'no'} slices "
                     f"(chosen with the boxes)")
    if counts["projected"]:
        lines.append(f"{_count(counts['projected'], 'projected run')}: one proj pair each, the 2D image the "
                     f"segmenter saw")
    if counts["planar"]:
        lines.append(f"{_count(counts['planar'], 'single-plane run')}: the plane as xy (tiles above 1024 pixels a "
                     f"side)")
    if (counts["projected"] or counts["planar"]) and not auto_supported:
        lines.append("this engine's export predates proj pairs: it exports every run as xy, xz and yz slices of its "
                     "copied labels; the proj and plane pairs need the tranche-5 Train B engine")
        return "\n".join(lines), True
    return "\n".join(lines), False


def export_orientations(ticked, geoms: list[RunGeometry] | None, *, auto_supported: bool):
    """The ``--orientations`` of an export: ``"auto"`` when the engine follows each run's geometry and every slice
    box is ticked, or no run is 3D (there is nothing to choose); else the ticked slices of the 3D runs, in the
    order xy, xz, yz. ValueError when none is ticked and a 3D run (or runs not read yet) needs one."""
    chosen = set(ticked)
    ticked = tuple(o for o in ORIENTATIONS if o in chosen)
    has_3d = geoms is None or any(g.kind == "3d" for g in geoms)
    if auto_supported and (not has_3d or ticked == ORIENTATIONS):
        return AUTO
    if not ticked:
        raise ValueError("tick at least one orientation")
    return ticked


# the engines whose export-training takes --orientations auto and writes proj pairs (they came with train-curve)
PROJ_EXPORT_FEATURE = "train_curve"


def engine_export_settings(settings: ExportSettings, ticked, *, auto_supported: bool) -> ExportSettings:
    """The export settings the engine that runs the export gets: ``settings`` as they are, except that an engine
    without the proj export (``auto_supported`` False, which would refuse ``"auto"``) gets the ticked slices, every
    one when none is ticked (the boxes are off when no run is 3D): it slices every run in those orientations."""
    if settings.orientations != AUTO or auto_supported:
        return settings
    chosen = set(ticked)
    return dataclasses.replace(settings, orientations=tuple(o for o in ORIENTATIONS if o in chosen) or ORIENTATIONS)


# ---------------------------------------------------------------------------------------------------
# the project the dock works in
# ---------------------------------------------------------------------------------------------------


class TrainingError(ValueError):
    """A project folder the Training dock cannot read; the message says why."""


@dataclasses.dataclass(frozen=True)
class ProjectImage:
    image_id: str
    path: Path
    excluded: bool = False


@dataclasses.dataclass
class TrainingProject:
    """What the Training dock reads of a project, in process (decision D39: files are read here, every write goes
    through the engine): the manifest's name, pinned pack and recipe and images, the pinned pack's copy (its
    object kind and the recipe's models), and the model files trained in the project."""
    root: Path
    name: str
    pack_id: str
    pack_version: str
    recipe: str
    images: list[ProjectImage]
    models: list[dict]
    object_kind: str | None
    pack_models: list[str]
    model_files: list[Path]
    round_folders: list[str] = dataclasses.field(default_factory=list)   # training/<round>/ on disk when read

    @property
    def pinned(self) -> str:
        return f"{self.pack_id}@{self.pack_version}"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def packs_dir(self) -> Path:
        return self.root / "packs"

    def round_dir(self, name: str) -> Path:
        return self.root / TRAINING_DIR / name

    def model_dir(self, round_name: str, model: str | None = None) -> Path:
        """``training/<round>/models/<name>/`` (name default ``<round>_nuclei``), where train-nuclei writes."""
        return self.round_dir(round_name) / "models" / (model or f"{round_name}_nuclei")

    def describe(self) -> str:
        return f"{self.name}: {self.pinned}, recipe {self.recipe}\n{self.root}"


def _image_path(root: Path, path: str) -> Path:
    """An image of the manifest: a relative POSIX path lies inside the project; anything else (absolute, UNC) is
    kept as given."""
    text = str(path)
    if text.startswith(("\\\\", "//")) or Path(text).is_absolute():
        return Path(text)
    return root.joinpath(*PurePosixPath(text).parts)


def _pack_copy(root: Path, pack_id: str, version: str, recipe: str) -> tuple[str | None, list[str]]:
    """The object kind of the pinned pack's copy (packs/<id>/<version>/pack.yaml) and the models its recipe needs
    (needs.models); (None, []) when the copy cannot be read."""
    path = root / "packs" / pack_id / version / "pack.yaml"
    try:
        with open(long_path(path), encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None, []
    if not isinstance(data, dict):
        return None, []
    kind = str(data.get("object_kind") or "nucleus")
    rec = next((r for r in data.get("recipes") or [] if isinstance(r, dict) and r.get("id") == recipe), {})
    needs = rec.get("needs") if isinstance(rec.get("needs"), dict) else {}
    return kind, [str(m) for m in needs.get("models") or []]


def round_names(root: str | Path) -> list[str]:
    """The round folders under training/ (names the engine's rule allows), sorted: where a project's models are."""
    base = Path(root) / TRAINING_DIR
    try:
        return sorted(p.name for p in base.iterdir() if p.is_dir() and ROUND_RE.fullmatch(p.name))
    except OSError:
        return []


def project_model_files(root: str | Path) -> list[Path]:
    """The models trained in the project: ``training/<round>/models/<name>/<file>`` as train-nuclei writes them
    (the file next to its card), by round and name."""
    out = []
    for rnd in round_names(root):
        models = Path(root) / TRAINING_DIR / rnd / "models"
        try:
            folders = sorted(p for p in models.iterdir() if p.is_dir())
        except OSError:
            continue
        for folder in folders:
            model, _card = model_files_in(folder)
            if model is not None:
                out.append(model)
    return out


def read_project(root: str | Path) -> TrainingProject:
    """The project at ``root`` (module docstring). The manifest is read and closed at once: an open handle would
    make the engine's atomic write of it fail on Windows. TrainingError for a folder that is no project, a manifest
    that cannot be read, or another format."""
    root = Path(os.path.abspath(str(root)))
    path = root / MANIFEST
    try:
        with open(long_path(path), encoding="utf-8") as fh:
            m = yaml.safe_load(fh)
    except FileNotFoundError:
        raise TrainingError(f"{root} is no project folder (it has no {MANIFEST})") from None
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as e:
        raise TrainingError(f"{path} cannot be read: {type(e).__name__}: {e}") from None
    if not isinstance(m, dict) or m.get("format") != 1:
        raise TrainingError(f"{path} is not a project manifest of format 1")
    pack = m.get("pack") if isinstance(m.get("pack"), dict) else {}
    pack_id, version, recipe = str(pack.get("id") or ""), str(pack.get("version") or ""), str(pack.get("recipe") or "")
    if not (pack_id and version and recipe):
        raise TrainingError(f"{path} names no pack version and recipe")
    images = [ProjectImage(str(e["id"]), _image_path(root, str(e["path"])), bool(e.get("excluded")))
              for e in m.get("images") or [] if isinstance(e, dict) and e.get("id") and e.get("path")]
    kind, needs = _pack_copy(root, pack_id, version, recipe)
    models = [dict(x) for x in m.get("models") or [] if isinstance(x, dict)]
    return TrainingProject(root=root, name=str(m.get("name") or root.name), pack_id=pack_id, pack_version=version,
                           recipe=recipe, images=images, models=models, object_kind=kind, pack_models=needs,
                           model_files=project_model_files(root), round_folders=round_names(root))


def project_base_choices(project: TrainingProject | None) -> list[str]:
    """Starting weights to offer: in a project the models its pack recipe needs (cpsam_v2@0f1cc3f7, as the model
    store pins them), then the models trained in the project (their files) and those it uses; then Cellpose's
    stock model, a random start and the registered names (base_choices)."""
    out: list[str] = []
    if project is not None:
        out += list(project.pack_models)
        out += [str(p) for p in project.model_files]
        for entry in project.models:
            ref = entry.get("ref")
            if isinstance(ref, str) and ref:
                p = _image_path(project.root, ref)
                if os.path.isfile(long_path(p)):
                    out.append(str(p))
    try:
        out += base_choices()
    except Exception:  # noqa: BLE001 - an unreadable registry still leaves the two built-in choices
        out += ["cpsam", "none"]
    return list(dict.fromkeys(out))


def project_relative(root: str | Path, path: str | Path) -> str | None:
    """``path`` as a POSIX path relative to the project ``root`` when it lies inside it, else None: how the engine
    names a project model (``project use-model DIR --model training/<round>/models/<name>/<file>``)."""
    try:
        rel = Path(os.path.abspath(str(path))).relative_to(Path(os.path.abspath(str(root))))
    except ValueError:
        return None
    return rel.as_posix() if rel.parts else None


def derived_entry(listing: dict | None, record: dict) -> dict | None:
    """The derived pack of a ``pack derive`` record in a pack listing (``list --project DIR``), or None."""
    for p in (listing or {}).get("packs") or []:
        if isinstance(p, dict) and p.get("id") == record.get("id") and \
                (record.get("version") is None or p.get("version") == record.get("version")):
            return p
    return None


# ---------------------------------------------------------------------------------------------------
# labels made elsewhere (germquant import-labels)
# ---------------------------------------------------------------------------------------------------


def guess_format(path: str | Path) -> str | None:
    """The import format a labels file's name carries (the engine refuses a name that carries another): .npz
    cellacdc-npz, _seg.npy cellpose-npy, .tif and .tiff (also .ome.tif) tiff, .png png; None otherwise."""
    name = Path(str(path)).name.lower()
    if name.endswith(".npz"):
        return "cellacdc-npz"
    if name.endswith("_seg.npy"):
        return "cellpose-npy"
    if name.endswith((".tif", ".tiff")):
        return "tiff"
    if name.endswith(".png"):
        return "png"
    return None


def image_channels(path: str | Path) -> list[str]:
    """The channel names an image's header gives, read the light way in process: germquant.project.headers when this
    germquant has it (no torch, no bioio), else api.read_image_metadata."""
    try:
        from germquant.project.headers import read_header
    except ImportError:
        from germquant import api

        meta = api.read_image_metadata(str(path)) or {}
        return [str(n) for n in meta.get("channel_names") or []]
    rec = read_header(str(path)) or {}
    return [str(n) for n in rec.get("channel_names") or []]


def import_argv(src, fmt: str, image, pairs_dir, *, allow_pickle: bool = False, frame: int | None = None,
                channel=None) -> list[str]:
    """``germquant import-labels`` with the options the dock sets (each only when set)."""
    argv = ["import-labels", str(src), "--format", str(fmt), "--image", str(image), "--pairs", str(pairs_dir)]
    if allow_pickle:
        argv.append("--allow-pickle")
    if frame is not None:
        argv += ["--frame", str(int(frame))]
    if channel is not None:
        argv += ["--channel", str(channel)]
    return argv


def import_command(src, fmt: str, image, pairs_dir, **options) -> str:
    return "germquant " + " ".join(_quote(a) for a in import_argv(src, fmt, image, pairs_dir, **options))


def import_summary(record: dict) -> str:
    """What an import did, from the engine's record (import_labels / ``import-labels --json``)."""
    channel = record.get("channel_index")
    name = record.get("channel_name")
    ch = f"channel {channel}" + (f" ({name})" if name else "") if channel is not None else "the only channel"
    replaced = int(record.get("replaced") or 0)
    lines = [f"imported {record.get('n_objects')} object(s) of {record.get('image_id')} ({record.get('dims')}, {ch}) "
             f"as {record.get('n_pairs')} pair(s) into {record.get('pairs_dir')}"
             + (f"; {replaced} earlier row(s) of {record.get('image_id')} replaced" if replaced else "")]
    lines += [f"warning: {w}" for w in record.get("warnings") or []]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# the pairs
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pair:
    name: str                 # the file name image/ and labels/ share: <image_id>_<orientation>_<index>.tif
    image: Path
    labels: Path
    image_id: str
    series_id: str
    orientation: str
    slice_index: int
    zone: str
    pixel_um: float | None    # the isotropic pixel size from the manifest (None without one)

    def describe(self) -> str:
        zone = f", zone {self.zone}" if self.zone else ""
        return f"{self.image_id}: {self.orientation} slice {self.slice_index}{zone}"


def load_pair_list(pairs_dir: str | Path) -> list[Pair]:
    """The pairs of a training folder, in the order ``train-nuclei`` reads them (``training.load_pairs``)."""
    pairs_dir = Path(pairs_dir)
    rows = gq_training.load_pairs(pairs_dir)
    pixel: dict[str, float | None] = {}
    manifest = pairs_dir / "manifest.csv"
    if manifest.is_file():
        try:
            df = pd.read_csv(long_path(manifest), low_memory=False)
        except (OSError, ValueError):
            df = pd.DataFrame()
        if {"image", "pixel_um"} <= set(df.columns):
            for rel, v in zip(df["image"], pd.to_numeric(df["pixel_um"], errors="coerce")):
                pixel[Path(str(rel)).name] = float(v) if np.isfinite(v) and v > 0 else None
    out = []
    for r in rows:
        image = Path(r["image"])
        out.append(Pair(name=image.name, image=image, labels=Path(r["labels"]), image_id=str(r["image_id"]),
                        series_id=str(r["series_id"]), orientation=str(r["orientation"]),
                        slice_index=int(r["slice_index"]), zone=str(r.get("zone") or ""),
                        pixel_um=pixel.get(image.name)))
    return out


def read_pair(pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    """(image, labels): the image as written (uint16) and the labels as int32, both 2D."""
    import tifffile

    image = np.asarray(tifffile.imread(long_path(pair.image)))
    labels = np.asarray(tifffile.imread(long_path(pair.labels))).astype(np.int32, copy=False)
    return image, labels


def n_labels(labels: np.ndarray) -> int:
    ids = np.unique(np.asarray(labels))
    return int(np.count_nonzero(ids))


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(long_path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _tiff_shape(path: Path) -> tuple[int, ...]:
    import tifffile

    with tifffile.TiffFile(long_path(path)) as tf:
        return tuple(int(s) for s in tf.series[0].shape)


def _atomic_replace(path: Path, write) -> None:
    """``write(tmp)`` into ``<name>.part`` next to ``path``, flush it to disk, then rename it over ``path``
    (``os.replace``: the file is either the old one or the new one, never half of each). The temporary name
    does not end in ``.tif``, so ``load_pairs`` can never mistake a leftover for a pair."""
    tmp = path.with_name(path.name + ".part")
    try:
        write(tmp)
        with open(long_path(tmp), "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(long_path(tmp), long_path(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(long_path(tmp))
        raise


def save_labels(pairs_dir: str | Path, pair: Pair, labels) -> dict:
    """Write corrected labels over ``labels/<name>.tif`` atomically (int32, zlib, as the export writes them)
    and record the save in ``corrections.json``; the manifest is left exactly as it was. Refused (ValueError)
    for anything that is not a 2D integer image of the pair's shape with ids from 0 to 2**31 - 1. Returns the
    correction record."""
    import tifffile

    pairs_dir = Path(pairs_dir)
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"{pair.name}: corrected labels must be 2D, got shape {arr.shape}")
    if arr.dtype.kind not in "iub":
        raise ValueError(f"{pair.name}: corrected labels must be integers, got {arr.dtype}")
    shape = _tiff_shape(pair.image)
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{pair.name}: corrected labels have shape {arr.shape}, the image {shape}")
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > np.iinfo(np.int32).max):
        raise ValueError(f"{pair.name}: label ids must be between 0 and {np.iinfo(np.int32).max}")
    out = np.ascontiguousarray(arr.astype(np.int32, copy=False))
    try:
        before = n_labels(tifffile.imread(long_path(pair.labels)))
    except Exception:  # noqa: BLE001 - a missing or damaged labels file is simply replaced
        before = None
    _atomic_replace(pair.labels, lambda tmp: tifffile.imwrite(long_path(tmp), out, compression="zlib"))
    record = {"saved": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
              "sha256": sha256_of(pair.labels), "n_labels_before": before, "n_labels_after": n_labels(out),
              "image_id": pair.image_id, "orientation": pair.orientation, "slice_index": pair.slice_index}
    records = corrections(pairs_dir)
    records[pair.name] = record
    path = pairs_dir / CORRECTIONS_FILE
    text = json.dumps({"format": 1, "note": "labels saved by the napari-germquant Training dock; germquant ignores "
                                            "this file", "corrections": records}, indent=1)
    _atomic_replace(path, lambda tmp: Path(long_path(tmp)).write_text(text, encoding="utf-8"))
    return record


def corrections(pairs_dir: str | Path) -> dict[str, dict]:
    """Every record in ``corrections.json`` (empty without the file or when it is unreadable)."""
    try:
        with open(long_path(Path(pairs_dir) / CORRECTIONS_FILE), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    recs = data.get("corrections") if isinstance(data, dict) else None
    return {str(k): dict(v) for k, v in (recs or {}).items() if isinstance(v, dict)}


def valid_corrections(pairs_dir: str | Path) -> dict[str, dict]:
    """The records whose labels file still has the recorded sha256 (a re-export that overwrote the file, or a
    later edit outside this dock, voids the record)."""
    pairs_dir = Path(pairs_dir)
    out = {}
    for name, rec in corrections(pairs_dir).items():
        p = pairs_dir / "labels" / name
        try:
            if p.is_file() and sha256_of(p) == rec.get("sha256"):
                out[name] = rec
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------


def device_status() -> dict[str, bool]:
    """Which of cuda, mps and cpu this environment can train on (germquant.device.available)."""
    have = gq_device.available()
    return {"cuda": bool(have.get("cuda")), "mps": bool(have.get("mps")), "cpu": True}


def default_device() -> str:
    have = device_status()
    return next(d for d in DEVICES if have[d])


@dataclasses.dataclass(frozen=True)
class TrainSettings:
    """The ``germquant train-nuclei`` options the dock sets; the rest keep the engine's defaults (learning
    rate 1e-5, weight decay 0.1, batch 8, min_train_masks 5). ``device`` auto leaves the choice to the engine (an
    engine's device list offers it). ``system``, ``kind`` (the card's object_kind) and ``dataset`` go on the
    model card, each only when set (the tranche-5 Train B engine's --system, --kind and --dataset)."""
    out: Path
    name: str | None = None
    base: str = "cpsam"
    epochs: int = 150
    split_by: str = "series"
    val_frac: float = 0.2
    seed: int = 0
    device: str = "cuda"
    system: str | None = None
    kind: str | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        if self.device not in TRAIN_DEVICES:
            raise ValueError(f"device must be one of {TRAIN_DEVICES}, got {self.device!r}")
        if self.split_by not in ("series", "image"):
            raise ValueError(f"split_by must be 'series' or 'image', got {self.split_by!r}")
        if int(self.epochs) < 1:
            raise ValueError("epochs must be 1 or more")
        if self.kind is not None and self.kind not in OBJECT_KINDS:
            raise ValueError(f"kind must be one of {OBJECT_KINDS}, got {self.kind!r}")

    def model_name(self) -> str:
        return str(self.name or Path(self.out).name).strip()

    def card_fields(self) -> dict:
        """The card options that are set, under the names train_nuclei takes."""
        return {k: v for k, v in (("system", self.system), ("kind", self.kind), ("dataset", self.dataset)) if v}

    def argv(self, pairs_dir: str | Path) -> list[str]:
        argv = ["train-nuclei", "--pairs", str(pairs_dir), "--out", str(self.out), "--base", str(self.base),
                "--epochs", str(int(self.epochs)), "--split-by", self.split_by, "--val-frac", f"{float(self.val_frac):g}",
                "--seed", str(int(self.seed))]
        if self.name:
            argv += ["--name", str(self.name)]
        if self.device == "cpu":
            argv.append("--cpu")
        for key, value in self.card_fields().items():
            argv += [f"--{key}", str(value)]
        return argv

    def command(self, pairs_dir: str | Path) -> str:
        cmd = "germquant " + " ".join(_quote(a) for a in self.argv(pairs_dir))
        if self.device not in ("cpu", "auto"):
            # the command line has --cpu and nothing else; an accelerator is chosen with the variable the
            # engine reads (germquant.device.select_device)
            cmd = (f"# first set the device: PowerShell $env:{DEVICE_ENV}=\"{self.device}\"; "
                   f"Mac / Linux export {DEVICE_ENV}={self.device}\n{cmd}")
        return cmd


@contextlib.contextmanager
def device_env(device: str):
    """``GERMQUANT_DEVICE`` set to ``device`` (cuda or mps) for the duration, then put back as it was. The
    variable is process-wide: a run started from the Run dock while a model trains would see it too. cpu (the
    ``--cpu`` of the command) and auto (the engine's choice) leave it alone."""
    if device in ("cpu", "auto"):
        yield
        return
    old = os.environ.get(DEVICE_ENV)
    os.environ[DEVICE_ENV] = device
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(DEVICE_ENV, None)
        else:
            os.environ[DEVICE_ENV] = old


def train(pairs_dir: str | Path, settings: TrainSettings) -> dict:
    """``train_nuclei`` with the arguments ``germquant train-nuclei`` hands it. cpu is ``--cpu``; cuda or mps
    set ``GERMQUANT_DEVICE`` for the call, and the engine falls back to the next device with a warning when the
    one asked for is not available (the result's ``device`` says which one trained). The card fields go only when
    set, as the command passes them; a germquant without them refuses them (ValueError)."""
    fields = settings.card_fields()
    if fields:
        import inspect

        params = inspect.signature(gq_training.train_nuclei).parameters
        lacking = [k for k in fields if k not in params]
        if lacking:
            raise ValueError(f"this germquant's train_nuclei takes no {', '.join(lacking)} (the model card fields of "
                             f"the tranche-5 Train B engine): clear them, or update that engine checkout")
    with device_env(settings.device):
        return gq_training.train_nuclei(pairs_dir, settings.out, base=settings.base, name=settings.name,
                                        epochs=int(settings.epochs), split_by=settings.split_by,
                                        val_frac=float(settings.val_frac), seed=int(settings.seed),
                                        cpu=settings.device == "cpu", **fields)


def epoch_progress(line: str, epochs: int | None = None) -> str | None:
    """A Cellpose epoch log line as ``epoch 40/150, train loss 0.1234, test loss 0.2345``; None otherwise."""
    m = _EPOCH_RE.match(str(line))
    if not m:
        return None
    epoch = int(m.group(1))
    total = f"/{int(epochs)}" if epochs else ""
    return f"epoch {epoch}{total}, train loss {m.group(2).strip()}, test loss {m.group(3).strip()}"


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def train_summary(res: dict, asked_device: str | None = None) -> str:
    """What ``germquant train-nuclei`` prints, plus the device actually used when it differs from the one asked."""
    sp = res["split"]
    lines = [f"{res['name']}: trained on {sp['n_train']} slice(s) from {len(sp['train_groups'])} {sp['by']} group(s), "
             f"{sp['n_val']} held out from {len(sp['val_groups'])}, device {res['device']}"]
    if asked_device and asked_device != res["device"]:
        lines.append(f"asked for {asked_device}, which this environment does not offer: trained on {res['device']}")
    metrics = res.get("metrics") or {}
    overall = metrics.get("overall")
    if overall:
        lines.append(f"hold-out at IoU 0.5: F1 {_fmt(overall['f1'])}, precision {_fmt(overall['precision'])}, "
                     f"recall {_fmt(overall['recall'])}, miss rate {_fmt(overall['miss_rate'])} "
                     f"(TP {overall['tp']} FP {overall['fp']} FN {overall['fn']})")
        for o, m in metrics.get("per_orientation", {}).items():
            lines.append(f"  {o}: F1 {_fmt(m['f1'])} on {m['n_pairs']} slice(s)")
        for z, m in metrics.get("per_zone", {}).items():
            lines.append(f"  zone {z}: F1 {_fmt(m['f1'])}, miss rate {_fmt(m['miss_rate'])} on {m['n_pairs']} slice(s)")
    elif metrics.get("error"):
        lines.append(f"hold-out metrics missing: {metrics['error']}")
    else:
        lines.append("nothing held out (a single group): the card carries no hold-out metrics")
    lines.append(DESIGN_GATE)
    lines.append(f"model -> {res['model_path']} (sha256 {str(res['model_sha256'])[:12]}...)")
    lines.append(f"card  -> {res['card_path']}")
    return "\n".join(lines)


_SUMMARY_START = re.compile(r"^\S.*: trained on \d+ slice\(s\) from ")

# the engines whose train-nuclei takes the model card fields: --system, --kind and --dataset came with train-curve
CARD_FIELDS_FEATURE = PROJ_EXPORT_FEATURE


def card_fields_missing(fields: dict, *, interpreter: str | None = None, info: dict | None = None) -> str:
    """The message for an engine whose ``train-nuclei`` takes no model card fields (one from before the tranche-5
    Train B engine, which would stop at the unknown option). ``interpreter`` and ``info`` (its version document) name
    the engine."""
    opts = ", ".join(f"--{k}" for k in fields) or "--system, --kind, --dataset"
    info = info or {}
    who = (f"The engine at {interpreter} is germquant {info.get('germquant')} (git {str(info.get('git_sha'))[:12]}), "
           f"whose train-nuclei does not take them." if interpreter else
           "This napari environment's germquant does not take them.")
    return (f"The model card fields ({opts}) need the tranche-5 Train B engine (germquant with the "
            f"{CARD_FIELDS_FEATURE!r} feature, which came with them). {who} Set Object kind to auto and leave System "
            f"and Dataset empty to train with it, or update that engine checkout (git pull) and reinstall it, or "
            f"choose another engine.")


def engine_train_summary(res: dict, asked_device: str | None = None) -> str:
    """What a training through an engine runner said: the in-process runner's result as train_summary words it,
    else what ``germquant train-nuclei`` printed at its end (from its "<name>: trained on" line on)."""
    result = res.get("result")
    if isinstance(result, dict) and isinstance(result.get("split"), dict):
        return train_summary(result, asked_device if asked_device in DEVICES else None)
    lines = [str(x).rstrip() for x in res.get("lines") or []]
    start = next((i for i, ln in enumerate(lines) if _SUMMARY_START.match(ln.strip())), None)
    kept = lines[start:] if start is not None else lines[-12:]
    return "\n".join(ln for ln in kept if ln.strip())


def base_choices() -> list[str]:
    """Starting weights to offer: Cellpose's stock model, a random start (smoke tests), then every registered
    name (a release model is usable once ``get-model`` has fetched it)."""
    names = ["cpsam", "none"]
    for n in models_registry.known_models():
        if n not in names:
            names.append(n)
    return names


# ---------------------------------------------------------------------------------------------------
# registering
# ---------------------------------------------------------------------------------------------------


def register(model: str | Path, card: str | Path | None = None, name: str | None = None) -> dict:
    """``register_local_model``: the entry added to ``local_models.json`` (with ``name``)."""
    return models_registry.register_local_model(model, card or None, name or None)


def register_argv(model: str | Path, card: str | Path | None = None, name: str | None = None) -> list[str]:
    argv = ["register-model", "--model", str(model)]
    if card:
        argv += ["--card", str(card)]
    if name:
        argv += ["--name", str(name)]
    return argv


def register_command(model: str | Path, card: str | Path | None = None, name: str | None = None) -> str:
    return "germquant " + " ".join(_quote(a) for a in register_argv(model, card, name))


def register_summary(entry: dict) -> str:
    lines = [f"registered {entry['name']}: {entry['path']} (sha256 {str(entry['sha256'])[:12]}..., "
             f"{int(entry['size']) // 1_000_000} MB) in {models_registry.local_registry_path()}"]
    if entry.get("card"):
        lines.append(f"card: {entry['card']}")
    lines.append(f"use it with  segmentation.nuclei.cellpose_model: {entry['path']}  in a profile (the absolute "
                 "path works in every environment on this computer)")
    lines.append(f"the bare name {entry['name']!r} resolves only in a germquant whose registry is "
                 f"{models_registry.local_registry_path()}; the engine refuses an unregistered name rather than "
                 "segmenting with Cellpose's stock weights")
    return "\n".join(lines)


def model_files_in(folder: str | Path, name: str | None = None) -> tuple[Path | None, Path | None]:
    """The model file and its card in a ``train-nuclei`` output folder: ``<folder>/<name>`` and
    ``<folder>/<name>.card.json`` (name defaults to the folder's name, else the only card there)."""
    folder = Path(folder)
    name = name or folder.name
    model, card = folder / name, folder / f"{name}.card.json"
    if not model.is_file():
        cards = sorted(folder.glob("*.card.json"))
        if len(cards) == 1:
            stem = cards[0].name[: -len(".card.json")]
            model, card = folder / stem, cards[0]
    return (model if model.is_file() else None), (card if card.is_file() else None)
