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
from pathlib import Path

import numpy as np
import pandas as pd

from germquant import device as gq_device
from germquant import models_registry
from germquant.fsutil import long_path
from germquant.segment import training as gq_training

from ._engine import _quote

CORRECTIONS_FILE = "corrections.json"
DEVICES = ("cuda", "mps", "cpu")
DEVICE_ENV = "GERMQUANT_DEVICE"
ORIENTATIONS = gq_training.ORIENTATIONS
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
    """The ``germquant export-training`` options (the same defaults)."""
    orientations: tuple[str, ...] = ORIENTATIONS
    slices_per_image: int = 3
    seed: int = 0
    min_labels: int = 3
    max_gb: float = 2.0
    zone_column: str = "zone"

    def argv(self, source: str | Path, pairs_dir: str | Path) -> list[str]:
        return ["export-training", str(source), "--out", str(pairs_dir), "--orientations", ",".join(self.orientations),
                "--slices-per-image", str(int(self.slices_per_image)), "--seed", str(int(self.seed)),
                "--zone-column", str(self.zone_column), "--min-labels", str(int(self.min_labels)),
                "--max-gb", f"{float(self.max_gb):g}"]

    def command(self, source: str | Path, pairs_dir: str | Path) -> str:
        return "germquant " + " ".join(_quote(a) for a in self.argv(source, pairs_dir))


def export_slices(source: str | Path, pairs_dir: str | Path, settings: ExportSettings = ExportSettings()) -> pd.DataFrame:
    """``export_training_pairs`` with the arguments ``germquant export-training`` hands it; returns the manifest."""
    return gq_training.export_training_pairs(
        source, pairs_dir, orientations=tuple(settings.orientations), slices_per_image=int(settings.slices_per_image),
        seed=int(settings.seed), zone_column=settings.zone_column or None, min_labels=int(settings.min_labels),
        max_bytes=float(settings.max_gb) * 1e9)


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
    rate 1e-5, weight decay 0.1, batch 8, min_train_masks 5)."""
    out: Path
    name: str | None = None
    base: str = "cpsam"
    epochs: int = 150
    split_by: str = "series"
    val_frac: float = 0.2
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.split_by not in ("series", "image"):
            raise ValueError(f"split_by must be 'series' or 'image', got {self.split_by!r}")
        if int(self.epochs) < 1:
            raise ValueError("epochs must be 1 or more")

    def model_name(self) -> str:
        return str(self.name or Path(self.out).name).strip()

    def argv(self, pairs_dir: str | Path) -> list[str]:
        argv = ["train-nuclei", "--pairs", str(pairs_dir), "--out", str(self.out), "--base", str(self.base),
                "--epochs", str(int(self.epochs)), "--split-by", self.split_by, "--val-frac", f"{float(self.val_frac):g}",
                "--seed", str(int(self.seed))]
        if self.name:
            argv += ["--name", str(self.name)]
        if self.device == "cpu":
            argv.append("--cpu")
        return argv

    def command(self, pairs_dir: str | Path) -> str:
        cmd = "germquant " + " ".join(_quote(a) for a in self.argv(pairs_dir))
        if self.device != "cpu":
            # the command line has --cpu and nothing else; an accelerator is chosen with the variable the
            # engine reads (germquant.device.select_device)
            cmd = (f"# first set the device: PowerShell $env:{DEVICE_ENV}=\"{self.device}\"; "
                   f"Mac / Linux export {DEVICE_ENV}={self.device}\n{cmd}")
        return cmd


@contextlib.contextmanager
def device_env(device: str):
    """``GERMQUANT_DEVICE`` set to ``device`` (cuda or mps) for the duration, then put back as it was. The
    variable is process-wide: a run started from the Run dock while a model trains would see it too."""
    if device == "cpu":
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
    one asked for is not available (the result's ``device`` says which one trained)."""
    with device_env(settings.device):
        return gq_training.train_nuclei(pairs_dir, settings.out, base=settings.base, name=settings.name,
                                        epochs=int(settings.epochs), split_by=settings.split_by,
                                        val_frac=float(settings.val_frac), seed=int(settings.seed),
                                        cpu=settings.device == "cpu")


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
