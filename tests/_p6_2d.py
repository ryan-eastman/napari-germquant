"""Test helpers of the P6-2D unit (tests/test_display_2d.py), kept out of conftest.py (tranche 5 Train B
plan, section 5.2): results folders and image stacks shaped like the engine's single-plane mode
(docs/DESIGN_system_packs.md 4.3, 4.6, 4.8 and 4.9, engine package P2b).

``write_planar_results`` is the hand-made planar_results folder of the P6 spec (plugin_specs.md 6.1). Every
table carries voxel_dz_um NaN with the plane's dy and dx (0.1 um); the nuclei table has volume_um3 NA,
centroid_z_um 0.0 and the planar columns area_um2 and equivalent_diameter_um; every spot has z_um 0.0. The
label image is an int32 OME-TIFF with axes YX and PhysicalSizeX and Y only, the spots image a YX uint16 TIFF
with one disk per spot (tifffile.imread gives (Y, X) for both); the done marker has the planar extra;
run_manifest.json holds the run's config (the spotmax backend with spots.spot_radius_um 0.5) and the planar
run geometry. The files are written by hand, so the tests give the same answers against every engine the
suite runs on; after P2b has merged (merge #1 of the train) the final check makes them with the engine's own
YX writers.

``plane_stack`` builds a germquant Stack as the engine's readers return one from P2b on, with the plane's
own pixel size in ``spacing_yx`` (on an engine without that field, a subclass of api.Stack that adds it).

Every file here is ASCII: the micro sign of the OME unit is written with chr()."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from napari_germquant._results import ResultsSet
from germquant import api

UM = chr(0xB5) + "m"                 # the OME unit the engine's writers use
PLANAR_ID = "plane_cells_001"
STACK_ID = "stack_cells_001"
DY = DX = 0.1
PLANE = (40, 48)                     # (Y, X) pixels of the single plane
RADIUS_UM = 0.5                      # spots.spot_radius_um of the planar run's config: a 1.0 um marker
SP3 = (0.4, 0.2, 0.2)                # (dz, dy, dx) of the 3D folder
# (nucleus_id, centre y, centre x, radius) in pixels; label 3 is left out on purpose
NUCLEI = ((1, 12, 12, 6), (2, 12, 32, 6), (4, 29, 22, 7))
# (spot_id, nucleus_id, y, x) in pixels, every spot inside its nucleus
SPOTS = ((1, 1, 11, 13), (2, 2, 10, 31), (3, 2, 14, 34), (4, 4, 29, 22), (5, 4, 26, 18))


def disk_labels(shape=PLANE, nuclei=NUCLEI) -> np.ndarray:
    yy, xx = np.indices(shape)
    out = np.zeros(shape, np.int32)
    for nid, cy, cx, r in nuclei:
        out[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = nid
    return out


def spots_image(shape=PLANE, spots=SPOTS, radius_px: int = 2) -> np.ndarray:
    yy, xx = np.indices(shape)
    out = np.zeros(shape, np.uint16)
    for _sid, _nid, y, x in spots:
        out[(yy - y) ** 2 + (xx - x) ** 2 <= radius_px * radius_px] = 65535
    return out


def write_yx_labels(path: Path, labels: np.ndarray, dy: float = DY, dx: float = DX) -> Path:
    """An int32 OME-TIFF, axes YX, with PhysicalSizeX and Y only: the format of the engine's
    write_labels_yx."""
    import tifffile

    tifffile.imwrite(str(path), labels.astype(np.int32), ome=True, photometric="minisblack",
                     metadata={"axes": "YX", "PhysicalSizeX": dx, "PhysicalSizeXUnit": UM,
                               "PhysicalSizeY": dy, "PhysicalSizeYUnit": UM})
    return path


def write_cyx_image(path: Path, data: np.ndarray, names, dy: float = DY, dx: float = DX) -> Path:
    """A single-plane OME-TIFF, axes CYX, calibrated in x and y only (no z step), as a 2D microscope
    writes it."""
    import tifffile

    tifffile.imwrite(str(path), data, ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "PhysicalSizeX": dx, "PhysicalSizeXUnit": UM,
                               "PhysicalSizeY": dy, "PhysicalSizeYUnit": UM,
                               "Channel": {"Name": list(names)}})
    return path


def write_manifest(folder: Path, *, config_hash: str, config: dict | None, **geometry) -> Path:
    """A run_manifest.json with the keys the engine's provenance.write_manifest writes, plus run geometry."""
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"pipeline_version": "0.2.0.dev0", "git_sha": "0" * 40, "pixi_lock_sha256": "absent",
           "config_hash": config_hash, "run_timestamp": "2026-10-23T00:00:00+00:00", "tool_versions": {},
           "config": config, **geometry}
    p = folder / "run_manifest.json"
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return p


def _meta(image_id: str, spacing, config_hash: str) -> dict:
    dz, dy, dx = spacing
    return {"image_id": image_id, "file_path": f"C:/nowhere/{image_id}.ome.tif", "voxel_dz_um": dz,
            "voxel_dy_um": dy, "voxel_dx_um": dx, "config_hash": config_hash}


def write_planar_results(folder: Path, image_id: str = PLANAR_ID, *, dy: float = DY, dx: float = DX,
                         config: dict | None = None, manifest: bool = True,
                         config_hash: str = "planar0hash1") -> ResultsSet:
    """The planar_results folder of the module docstring; ``config`` replaces the run's config in
    run_manifest.json, and ``manifest=False`` leaves the manifest out (a batch image folder)."""
    import tifffile

    folder.mkdir(parents=True, exist_ok=True)
    meta = _meta(image_id, (float("nan"), dy, dx), config_hash)
    labels = disk_labels()
    write_yx_labels(folder / f"{image_id}__nuclei_labels.tif", labels, dy, dx)
    tifffile.imwrite(str(folder / f"{image_id}__spots.tif"), spots_image(), photometric="minisblack")
    n_spots = {nid: sum(1 for s in SPOTS if s[1] == nid) for nid, *_ in NUCLEI}
    rows = []
    for nid, cy, cx, _r in NUCLEI:
        area = int((labels == nid).sum()) * dy * dx
        rows.append({"nucleus_id": nid, "volume_um3": float("nan"), "centroid_z_um": 0.0,
                     "centroid_y_um": cy * dy, "centroid_x_um": cx * dx, "n_spots": n_spots[nid],
                     "area_um2": area, "equivalent_diameter_um": 2.0 * float(np.sqrt(area / np.pi)), **meta})
    pd.DataFrame(rows).to_csv(folder / f"{image_id}__nuclei.csv", index=False)
    spots = [{"spot_id": sid, "nucleus_id": nid, "z_um": 0.0, "y_um": y * dy, "x_um": x * dx,
              "effect_size": 4.0 + sid, **meta} for sid, nid, y, x in SPOTS]
    pd.DataFrame(spots).to_csv(folder / f"{image_id}__spots.csv", index=False)
    summary = {"n_nuclei": len(NUCLEI), "mean_spots": len(SPOTS) / len(NUCLEI), "qc_pass": True,
               "qc_flags": "read:planar", "segmentation_method": "classical", **meta}
    pd.DataFrame([summary]).to_csv(folder / f"{image_id}__image_summary.csv", index=False)
    stages = {"read": {"status": "ran"}, "segment": {"status": "ran"}, "spots": {"status": "ran"}}
    (folder / f"{image_id}__stages.json").write_text(json.dumps({
        "image_id": image_id, "config_hash": config_hash, "planar": True, "spacing_source": "file_yx",
        "stages": stages}), encoding="utf-8")
    (folder / f"{image_id}__done.json").write_text(json.dumps({
        "image_id": image_id, "config_hash": config_hash, "enabled_stages": ["read", "segment", "spots"],
        "stage_hashes": {}, "xy_stride": 1, "z_range": None, "planar": True, "failed_stages": [],
        "finished": "2026-10-23T00:00:00+00:00"}), encoding="utf-8")
    if manifest:
        cfg = config if config is not None else {
            "io": {"planar": "auto"}, "spots": {"backend": "spotmax", "spot_radius_um": RADIUS_UM}}
        write_manifest(folder, config_hash=config_hash, config=cfg, xy_stride=1, z_range=None, planar=True,
                       spacing_source="file_yx")
    return ResultsSet(folder, image_id)


def write_stack_results(folder: Path, image_id: str = STACK_ID, *, spacing=SP3,
                        config_hash: str = "stack0hash01", instances: tuple[str, ...] = ()) -> ResultsSet:
    """A 3D results folder without run_manifest.json (a batch image folder, or a run of an older engine):
    ZYX labels and spots image, tables with (dz, dy, dx), a done marker, and ``instances`` extra spot
    tables spots_<name>."""
    import tifffile

    folder.mkdir(parents=True, exist_ok=True)
    meta = _meta(image_id, spacing, config_hash)
    planes = 3
    tifffile.imwrite(str(folder / f"{image_id}__nuclei_labels.tif"),
                     np.repeat(disk_labels()[None], planes, axis=0), photometric="minisblack")
    tifffile.imwrite(str(folder / f"{image_id}__spots.tif"), np.repeat(spots_image()[None], planes, axis=0),
                     photometric="minisblack")
    pd.DataFrame([{"nucleus_id": nid, "volume_um3": 10.0, **meta} for nid, *_ in NUCLEI]).to_csv(
        folder / f"{image_id}__nuclei.csv", index=False)
    spots = pd.DataFrame([{"spot_id": sid, "nucleus_id": nid, "z_um": spacing[0], "y_um": y * spacing[1],
                           "x_um": x * spacing[2], **meta} for sid, nid, y, x in SPOTS])
    spots.to_csv(folder / f"{image_id}__spots.csv", index=False)
    for name in instances:
        spots.to_csv(folder / f"{image_id}__spots_{name}.csv", index=False)
    pd.DataFrame([{"n_nuclei": len(NUCLEI), "qc_pass": True, **meta}]).to_csv(
        folder / f"{image_id}__image_summary.csv", index=False)
    (folder / f"{image_id}__done.json").write_text(json.dumps({
        "image_id": image_id, "config_hash": config_hash, "xy_stride": 1}), encoding="utf-8")
    return ResultsSet(folder, image_id)


def write_summary_only(folder: Path, spacing, image_id: str = "only_summary") -> ResultsSet:
    """A folder with nothing but an image_summary table carrying ``spacing`` as (dz, dy, dx)."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"n_nuclei": 0, **_meta(image_id, spacing, "x")}]).to_csv(
        folder / f"{image_id}__image_summary.csv", index=False)
    return ResultsSet(folder, image_id)


def plane_stack(data: np.ndarray, spacing, *, spacing_ok: bool, spacing_yx=None, xy_stride: int = 1,
                names=None, path: str | Path = "plane.ome.tif"):
    """A germquant Stack (C, Z, Y, X) carrying ``spacing_yx``: api.Stack itself when the engine has the
    field (tranche 5 P2b on), else a subclass of it that adds the field with the same default."""
    cls = api.Stack
    if "spacing_yx" not in {f.name for f in dataclasses.fields(cls)}:
        field = ("spacing_yx", object, dataclasses.field(default=None))
        cls = dataclasses.make_dataclass("StackWithSpacingYX", [field], bases=(api.Stack,))
    n = data.shape[0]
    return cls(data=data, spacing=tuple(float(s) for s in spacing),
               channel_names=list(names or [f"ch{i}" for i in range(n)]), path=Path(path),
               downsample_xy=int(xy_stride), spacing_ok=spacing_ok, spacing_yx=spacing_yx)
