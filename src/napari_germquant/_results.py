"""An existing germquant results folder as napari layers. Pure python (no Qt).

The files ``process_image`` writes for one image id (``<image_id>__<table>.csv``, ``<image_id>__nuclei_labels.tif``,
``<image_id>__spots.csv``, ``<image_id>__granules_labels.tif``, ``<image_id>__envelope_labels.tif``,
``<image_id>__spots.tif``, ``<image_id>__done.json``) become: a Labels layer per label image, with the
matching per-object table attached as the layer's features (row i describes label i, so napari's label
status line and feature widgets read the right row); a Points layer for the spots in microns with the spots
table as features (and one per extra spot instance, ``spots_<name>``); the spots blob image as a hidden Image
layer; and, on request, the source image through the plugin's reader at the stride the run used. Nothing here
writes anything.

For a system pack run (docs/DESIGN_system_packs.md, section 8.3) the pack record that ``__stages.json`` and
``run_manifest.json`` carry names what one row of the nuclei table is (object_kind nucleus or cell), so the
Labels layer is "<id> cells" or "<id> nuclei" and the summary line counts cells or nuclei. ``provenance``
collects the pack id and version, recipe, config_hash, model sha256, git sha and the device used, and
``fallback_banner`` warns when the segmenter that made the labels (image_summary.segmentation_method) is not
the one the pack's recipe declares.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _engine  # import order: germquant.api before anything else
from germquant import api

LayerData = tuple

SUMMARY_SUFFIX = "__image_summary.csv"


@dataclasses.dataclass(frozen=True)
class ResultsSet:
    folder: Path
    image_id: str

    def file(self, suffix: str) -> Path:
        return self.folder / f"{self.image_id}__{suffix}"

    def has(self, suffix: str) -> bool:
        return self.file(suffix).is_file()

    def table(self, name: str) -> pd.DataFrame | None:
        p = self.file(f"{name}.csv")
        if not p.is_file():
            return None
        return pd.read_csv(p, low_memory=False)

    def done_marker(self) -> dict | None:
        return api.read_done_marker(self.folder, self.image_id)

    def stage_record(self) -> dict | None:
        """The whole ``<image_id>__stages.json`` document (image_id, config_hash, the stage records under
        "stages" and, for a pack run, the pack, runtime and post_run records), or None."""
        p = self.file("stages.json")
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def stages(self) -> dict | None:
        data = self.stage_record()
        # the engine nests the per-stage records under "stages" next to image_id, config_hash and
        # complete; an older flat document is returned as is
        if isinstance(data, dict) and isinstance(data.get("stages"), dict):
            return data["stages"]
        return data

    def manifest(self) -> dict | None:
        """``run_manifest.json`` of the folder (a single-image run writes one per output folder), or None."""
        p = self.folder / "run_manifest.json"
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _record(self, key: str) -> dict | None:
        for doc in (self.stage_record(), self.manifest()):
            if isinstance(doc, dict) and isinstance(doc.get(key), dict):
                return doc[key]
        return None

    def pack_record(self) -> dict | None:
        """The pack record of a pack run (id, version, recipe, source, pack_sha256, object_kind, dims,
        status), or None for a profile or config run."""
        return self._record("pack")

    def runtime_record(self) -> dict | None:
        return self._record("runtime")

    def object_kind(self) -> str:
        return str((self.pack_record() or {}).get("object_kind") or "nucleus")

    def spacing(self) -> tuple[float, float, float]:
        """(dz, dy, dx) from the shared metadata block of image_summary (one row, always written), else
        nuclei, else 1 um isotropic."""
        for name in ("image_summary", "nuclei"):
            df = self.table(name)
            if df is not None and len(df) and {"voxel_dz_um", "voxel_dy_um", "voxel_dx_um"} <= set(df.columns):
                row = df.iloc[0]
                sp = tuple(float(row[c]) for c in ("voxel_dz_um", "voxel_dy_um", "voxel_dx_um"))
                if all(np.isfinite(s) and s > 0 for s in sp):
                    return sp   # type: ignore[return-value]
        return (1.0, 1.0, 1.0)

    def source_path(self) -> Path | None:
        df = self.table("image_summary")
        if df is None or not len(df) or "file_path" not in df.columns:
            return None
        p = Path(str(df["file_path"].iloc[0]))
        return p if p.is_file() or p.is_dir() else None


def find_results(folder: str | Path, depth: int = 1) -> list[ResultsSet]:
    """Every image id that has an image_summary table in ``folder`` or, for a batch root, up to ``depth``
    folder levels below it (``germquant batch`` writes one folder per image, mirroring the input tree)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    pattern = f"*{SUMMARY_SUFFIX}"
    hits = sorted(folder.glob(pattern))
    for level in range(1, depth + 1):
        hits += sorted(folder.glob("/".join(["*"] * level) + "/" + pattern))
    out = []
    for p in hits:
        image_id = p.name[: -len(SUMMARY_SUFFIX)]
        out.append(ResultsSet(p.parent, image_id))
    return out


def labels_features(table: pd.DataFrame, id_column: str, n_labels: int) -> pd.DataFrame:
    """A features table for a napari Labels layer: row i describes label i (row 0 is the background, all
    NA), so the status line shows the right object whatever ids the table skips. ``n_labels`` is the
    largest label value in the image."""
    if id_column not in table.columns:
        return pd.DataFrame(index=range(n_labels + 1))
    ids = pd.to_numeric(table[id_column], errors="coerce")
    body = table.loc[ids.notna()].copy()
    body.index = ids[ids.notna()].astype(int).to_numpy()
    body = body[~body.index.duplicated(keep="first")]
    full = body.reindex(range(max(n_labels, int(body.index.max()) if len(body) else 0) + 1))
    full.index.name = None
    return full.reset_index(drop=True)


def _read_labels(path: Path) -> np.ndarray:
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 2:
        arr = arr[None]
    return arr.astype(np.int32, copy=False)


def labels_layer(rs: ResultsSet, suffix: str, table_name: str | None, id_column: str, name: str,
                 spacing: tuple[float, float, float]) -> LayerData | None:
    path = rs.file(suffix)
    if not path.is_file():
        return None
    labels = _read_labels(path)
    kwargs: dict = {
        "name": f"{rs.image_id} {name}",
        "scale": spacing,
        "metadata": {"napari_germquant_results": {"folder": str(rs.folder), "image_id": rs.image_id,
                                                  "file": str(path), "table": table_name}},
    }
    if table_name is not None:
        table = rs.table(table_name)
        if table is not None and len(table):
            kwargs["features"] = labels_features(table, id_column, int(labels.max()))
    return (labels, kwargs, "labels")


def points_layer(rs: ResultsSet, spacing: tuple[float, float, float], table: str = "spots",
                 color: str = "yellow") -> LayerData | None:
    """A spots table as a Points layer in microns (the table's z_um / y_um / x_um are world coordinates;
    the layer gets no scale so it lands on the scaled label and image layers). None without spots.
    ``table`` is ``spots`` or an extra spot instance's ``spots_<name>``."""
    spots = rs.table(table)
    if spots is None or not len(spots) or not {"z_um", "y_um", "x_um"} <= set(spots.columns):
        return None
    coords = spots[["z_um", "y_um", "x_um"]].to_numpy(float)
    size_um = 2.0 * 0.3                       # the RAD-51 spot_radius_um default: a 0.6 um marker
    kwargs: dict = {
        "name": f"{rs.image_id} {table}",
        "size": size_um,
        "face_color": color,
        "border_color": "black",
        # napari 0.9: a spot is a 0.6 um sphere; thicken the z slice (right-click the slider) to see
        # neighbouring planes' spots drawn with the disc a sphere would show
        "projection_mode": "rescale_spherical",
        "features": spots.reset_index(drop=True),
        "metadata": {"napari_germquant_results": {"folder": str(rs.folder), "image_id": rs.image_id,
                                                  "file": str(rs.file(f"{table}.csv")), "table": table,
                                                  "spacing_um": tuple(spacing)}},
    }
    return (coords, kwargs, "points")


INSTANCE_COLORS = ("cyan", "magenta", "orange", "lime")


def instance_tables(rs: ResultsSet) -> list[str]:
    """The extra spot instance tables of the image (``spots_<name>``), in name order."""
    prefix = f"{rs.image_id}__spots_"
    return sorted(p.name[len(rs.image_id) + 2: -len(".csv")] for p in rs.folder.glob(f"{prefix}*.csv"))


def spots_image_layer(rs: ResultsSet, spacing: tuple[float, float, float]) -> LayerData | None:
    path = rs.file("spots.tif")
    if not path.is_file():
        return None
    import tifffile

    img = np.asarray(tifffile.imread(str(path)))
    kwargs = {"name": f"{rs.image_id} spots image", "scale": spacing, "visible": False, "blending": "additive",
              "colormap": "yellow",
              "metadata": {"napari_germquant_results": {"folder": str(rs.folder), "image_id": rs.image_id,
                                                        "file": str(path), "table": None}}}
    return (img, kwargs, "image")


def load_results_layers(rs: ResultsSet, *, include_source: bool = False) -> list[LayerData]:
    """Every layer a results folder yields, source image first (bottom) when asked and present."""
    spacing = rs.spacing()
    layers: list[LayerData] = []
    if include_source:
        src = rs.source_path()
        if src is not None:
            from ._reader import open_image

            done = rs.done_marker() or {}
            layers += open_image(src, xy_stride=int(done.get("xy_stride", 1) or 1))
    _singular, plural = object_words(rs.object_kind())
    for suffix, table, id_col, name in (
        ("nuclei_labels.tif", "nuclei", "nucleus_id", plural),      # "<id> cells" when one row is a cell
        ("granules_labels.tif", "granules", "granule_id", "granules"),
        ("envelope_labels.tif", None, "nucleus_id", "envelope"),
    ):
        layer = labels_layer(rs, suffix, table, id_col, name, spacing)
        if layer is not None:
            layers.append(layer)
    blob = spots_image_layer(rs, spacing)
    if blob is not None:
        layers.append(blob)
    pts = points_layer(rs, spacing)
    if pts is not None:
        layers.append(pts)
    for i, table in enumerate(instance_tables(rs)):
        extra = points_layer(rs, spacing, table=table, color=INSTANCE_COLORS[i % len(INSTANCE_COLORS)])
        if extra is not None:
            layers.append(extra)
    return layers


def object_words(kind: str | None) -> tuple[str, str]:
    """``(singular, plural)`` for an object_kind: what one row of the nuclei table is."""
    from ._packs import object_words as words

    return words(kind)


def _num(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def summary_line(rs: ResultsSet) -> str:
    """'N cells, M spots, mean spots per cell X' (nuclei for a nucleus pack or any other run), plus one part
    per extra spot instance."""
    singular, plural = object_words(rs.object_kind())
    summary = rs.table("image_summary")
    row = summary.iloc[0] if summary is not None and len(summary) else None
    n = int(row.get("n_nuclei", 0) or 0) if row is not None and _num(row.get("n_nuclei")) is not None else None
    spots = rs.table("spots")
    m = len(spots) if spots is not None else None
    mean = _num(row.get("mean_spots")) if row is not None else None
    if mean is None and n and m is not None:
        mean = m / n
    head = f"{n if n is not None else '?'} {plural if n != 1 else singular}"
    parts = [head, f"{m if m is not None else 'no'} spots"]
    if mean is not None:
        parts.append(f"mean spots per {singular} {mean:.2f}")
    text = ", ".join(parts)
    for table in instance_tables(rs):
        extra = rs.table(table)
        k = len(extra) if extra is not None else 0
        marker = table[len("spots_"):]
        if extra is not None and len(extra) and "marker" in extra.columns:
            marker = str(extra["marker"].iloc[0])
        imean = _num(row.get(f"mean_n_{table}")) if row is not None else None
        if imean is None and n:
            imean = k / n
        text += f"; {table} ({marker}): {k} spots" + (f", mean per {singular} {imean:.2f}" if imean is not None else "")
    return text


def provenance(rs: ResultsSet) -> dict:
    """Where the numbers come from: pack id and version, recipe, config_hash, model sha256, git sha and the
    device used (the device Cellpose finally ran on, else the device the run selected)."""
    stage_doc = rs.stage_record() or {}
    man = rs.manifest() or {}
    done = rs.done_marker() or {}
    pack = rs.pack_record() or {}
    runtime = rs.runtime_record() or {}
    post = stage_doc.get("post_run") if isinstance(stage_doc.get("post_run"), dict) else {}
    device = post.get("segment_device_used") or runtime.get("device_selected")
    return {
        "pack": f"{pack['id']}@{pack['version']}" if pack.get("id") else None,
        "recipe": pack.get("recipe"),
        "object_kind": pack.get("object_kind"),
        "config_hash": done.get("config_hash") or stage_doc.get("config_hash") or man.get("config_hash"),
        "model_sha256": stage_doc.get("model_sha256") or man.get("model_sha256"),
        "git_sha": man.get("git_sha"),
        "device": device,
        "device_requested": runtime.get("device_requested"),
    }


def provenance_text(rs: ResultsSet) -> str:
    p = provenance(rs)
    lines = []
    if p["pack"]:
        lines.append(f"pack: {p['pack']}   recipe: {p['recipe']}")
    else:
        lines.append("pack: none (a profile or config file run)")
    lines.append(f"config_hash: {p['config_hash'] or 'not recorded'}")
    lines.append(f"model sha256: {p['model_sha256'] or 'not recorded'}")
    lines.append(f"git sha: {p['git_sha'] or 'not recorded in this folder (a batch keeps it in its root)'}")
    if p["device"]:
        asked = f" (asked for {p['device_requested']})" if p.get("device_requested") else ""
        lines.append(f"device used: {p['device']}{asked}")
    else:
        lines.append("device used: not recorded (pack runs record it; see the run log for other runs)")
    return "\n".join(lines)


def expected_segmenters(rs: ResultsSet, hint: list[str] | None = None) -> list[str] | None:
    """The segmenters the run's recipe expects: ``hint`` when given (the dock passes the pack recipe's
    needs.segment_backends from the engine's listing), else the pack recipe's needs as this environment's
    germquant finds the pack, else what the run's own config asked for (run_manifest.json). None when
    nothing says."""
    if hint:
        return [str(h) for h in hint]
    pack = rs.pack_record() or {}
    if pack.get("id") and _engine.packs_supported():
        try:
            from germquant import packs

            p = packs.find_pack(f"{pack['id']}@{pack['version']}")
            need = p.recipe(pack.get("recipe")).need("segment_backends")
            if need:
                return need
        except Exception:  # noqa: BLE001 - the pack is not installed here; fall back to the run's config
            pass
    cfg = (rs.manifest() or {}).get("config")
    nuc = ((cfg or {}).get("segmentation") or {}).get("nuclei") if isinstance(cfg, dict) else None
    if isinstance(nuc, dict) and nuc.get("method"):
        method = str(nuc.get("method"))
        if method == "classical":
            return ["classical"]
        return [str(nuc.get("backend") or "cellpose")]
    return None


def fallback_banner(rs: ResultsSet, expected: list[str] | None = None) -> str | None:
    """A warning when image_summary.segmentation_method is not a segmenter the recipe expects (for example the
    classical watershed after Cellpose failed, where the pack expects cellpose), else None."""
    summary = rs.table("image_summary")
    if summary is None or not len(summary) or "segmentation_method" not in summary.columns:
        return None
    used = summary["segmentation_method"].iloc[0]
    if used is None or (isinstance(used, float) and np.isnan(used)) or not str(used).strip():
        return None
    want = expected_segmenters(rs, expected)
    if not want or str(used) in want:
        return None
    pack = rs.pack_record() or {}
    who = (f"pack {pack.get('id')}@{pack.get('version')} recipe {pack.get('recipe')}" if pack.get("id")
           else "the recipe")
    return (f"These labels were made by {used}, but {who} expects {', '.join(want)}. The numbers are not the "
            f"recipe's: check that the engine environment has a working {want[0]} (for Cellpose: torch and the "
            f"model) and run again.")


def summary_text(rs: ResultsSet, expected: list[str] | None = None) -> str:
    """A few lines for the widgets: the summary line, nuclei (or cells), germline, spots, QC, stages that ran,
    failed or skipped, the completion marker, the provenance and, when it applies, the fallback banner."""
    singular, plural = object_words(rs.object_kind())
    parts = [f"{rs.image_id}  ({rs.folder})"]
    banner = fallback_banner(rs, expected)
    if banner:
        parts.append(f"WARNING: {banner}")
    try:
        parts.append(summary_line(rs))
    except Exception:  # noqa: BLE001 - a half-written folder: the other lines still help
        pass
    summary = rs.table("image_summary")
    if summary is not None and len(summary):
        row = summary.iloc[0]
        n = row.get("n_nuclei", "?")
        g = row.get("n_germline_nuclei", "?")
        parts.append(f"{plural}: {n}   germline {plural}: {g}   mean spots per {singular}: "
                     f"{row.get('mean_spots', 'NA')}")
        flags = row.get("qc_flags", "")
        if flags is None or (isinstance(flags, float) and np.isnan(flags)) or not str(flags).strip():
            flags = "none"
        parts.append(f"qc_pass: {row.get('qc_pass', '?')}   flags: {flags}")
    spots = rs.table("spots")
    if spots is not None:
        parts.append(f"spots table rows: {len(spots)}")
    stages = rs.stages()
    if stages:
        ran = [k for k, v in stages.items() if isinstance(v, dict) and v.get("status") == "ran"]
        failed = [k for k, v in stages.items() if isinstance(v, dict) and v.get("status") == "failed"]
        skipped = [k for k, v in stages.items() if isinstance(v, dict) and v.get("status") == "skipped"]
        parts.append(f"stages ran: {', '.join(ran) or 'none'}")
        if failed:
            parts.append(f"stages FAILED: {', '.join(failed)}")
        if skipped:
            parts.append(f"stages skipped: {', '.join(skipped)}")
    done = rs.done_marker()
    parts.append("completion marker: " + ("present" if done else "missing (run incomplete or interrupted)"))
    if done:
        parts.append(f"config_hash: {done.get('config_hash')}   xy_stride: {done.get('xy_stride')}")
    if rs.pack_record():
        parts.append(provenance_text(rs))
    return "\n".join(parts)
