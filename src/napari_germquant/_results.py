"""An existing germquant results folder as napari layers. Pure python (no Qt).

The files ``process_image`` writes for one image id (``<image_id>__<table>.csv``, ``<image_id>__nuclei_labels.tif``,
``<image_id>__spots.csv``, ``<image_id>__granules_labels.tif``, ``<image_id>__envelope_labels.tif``,
``<image_id>__spots.tif``, ``<image_id>__done.json``) become: a Labels layer per label image, with the
matching per-object table attached as the layer's features (row i describes label i, so napari's label
status line and feature widgets read the right row); a Points layer for the spots in microns with the spots
table as features; the spots blob image as a hidden Image layer; and, on request, the source image through
the plugin's reader at the stride the run used. Nothing here writes anything.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _engine  # noqa: F401  (import order: germquant.api before anything else)
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

    def stages(self) -> dict | None:
        p = self.file("stages.json")
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return None
        # the engine nests the per-stage records under "stages" next to image_id, config_hash and
        # complete; an older flat document is returned as is
        if isinstance(data, dict) and isinstance(data.get("stages"), dict):
            return data["stages"]
        return data

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


def points_layer(rs: ResultsSet, spacing: tuple[float, float, float]) -> LayerData | None:
    """The spots table as a Points layer in microns (the table's z_um / y_um / x_um are world coordinates;
    the layer gets no scale so it lands on the scaled label and image layers). None without spots."""
    spots = rs.table("spots")
    if spots is None or not len(spots) or not {"z_um", "y_um", "x_um"} <= set(spots.columns):
        return None
    coords = spots[["z_um", "y_um", "x_um"]].to_numpy(float)
    size_um = 2.0 * 0.3                       # the RAD-51 spot_radius_um default: a 0.6 um marker
    kwargs: dict = {
        "name": f"{rs.image_id} spots",
        "size": size_um,
        "face_color": "yellow",
        "border_color": "black",
        # napari 0.9: a spot is a 0.6 um sphere; thicken the z slice (right-click the slider) to see
        # neighbouring planes' spots drawn with the disc a sphere would show
        "projection_mode": "rescale_spherical",
        "features": spots.reset_index(drop=True),
        "metadata": {"napari_germquant_results": {"folder": str(rs.folder), "image_id": rs.image_id,
                                                  "file": str(rs.file("spots.csv")), "table": "spots",
                                                  "spacing_um": tuple(spacing)}},
    }
    return (coords, kwargs, "points")


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
    for suffix, table, id_col, name in (
        ("nuclei_labels.tif", "nuclei", "nucleus_id", "nuclei"),
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
    return layers


def summary_text(rs: ResultsSet) -> str:
    """A few lines for the widget: nuclei, germline nuclei, spots, QC, stages that ran, failed or skipped."""
    parts = [f"{rs.image_id}  ({rs.folder})"]
    summary = rs.table("image_summary")
    if summary is not None and len(summary):
        row = summary.iloc[0]
        n = row.get("n_nuclei", "?")
        g = row.get("n_germline_nuclei", "?")
        parts.append(f"nuclei: {n}   germline nuclei: {g}   mean spots per nucleus: {row.get('mean_spots', 'NA')}")
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
    return "\n".join(parts)
