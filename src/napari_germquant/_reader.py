"""The reader contribution: a 3D stack as one Image layer per channel, with the physical voxel size.

Every format goes through ``germquant.api.read_image`` (the nd2 reader for ``.nd2``, bioio for TIFF,
OME-TIFF, CZI, LIF and OME-Zarr), so the pixels napari shows are the pixels the pipeline measures. The
header is read first (``api.read_image_metadata``, no pixels) and a file with more than one time point,
position or scene is refused with a message that says what to do: the engine's reader would silently take
index 0, which is the wrong thing to show someone who dropped a 56 GB time-lapse on the viewer.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from . import _engine  # noqa: F401  (import order: germquant.api before anything else)
from germquant import api

# the kwargs tuple napari expects: (data, {layer kwargs}, "image")
LayerData = tuple

# the metadata key every layer this plugin adds carries; the Run widget reads the image path from it
META_KEY = "napari_germquant"

# one colormap per channel position (DAPI grey, then the common two-colour panel); cycles past six
COLORMAPS = ("gray", "green", "magenta", "cyan", "yellow", "red")


def napari_get_reader(path):
    """npe2 reader hook: a callable for every path whose suffix germquant reads, else None."""
    paths = list(path) if isinstance(path, (list, tuple)) else [path]
    for p in paths:
        try:
            api.image_format(p)
        except ValueError:
            return None
    return read_layers


def read_layers(path) -> list[LayerData]:
    """The reader napari calls: one or several paths to a flat list of layer tuples."""
    paths = list(path) if isinstance(path, (list, tuple)) else [path]
    out: list[LayerData] = []
    for p in paths:
        out.extend(open_image(p))
    return out


def check_single_timepoint(path: str | Path, meta: dict) -> None:
    """Refuse a time-lapse or a multi-position / multi-scene file before any pixel is read."""
    name = Path(path).name
    sizes = dict(meta.get("sizes") or {})
    n_t = int(sizes.get("T", 1) or 1)
    if n_t > 1:
        raise api.UnsupportedImageError(
            f"{name} is a time-lapse ({n_t} time points). napari-germquant opens fixed images only; the engine "
            f"would measure time point 0 and nothing else. Run the series in a terminal with "
            f"`germquant run {name} --profile <profile> --t all --out <folder>` (docs/TIMELAPSE.md and "
            f"docs/DESIGN_timelapse.md in the germquant checkout), then open <folder> in the Time-lapse dock "
            f"(Plugins > germquant > Time-lapse).")
    n_p = max(int(sizes.get("P", 1) or 1), int(meta.get("n_scenes", 1) or 1))
    if n_p > 1:
        raise api.UnsupportedImageError(
            f"{name} holds {n_p} positions or scenes. germquant reads one scene per file; export or split the "
            f"file into one scene per image and open those.")


def open_image(path: str | Path, xy_stride: int = 1) -> list[LayerData]:
    """Header check, then the pixels through ``api.read_image``; ``xy_stride`` keeps every n-th pixel in
    y and x (and scales the layer accordingly), for a quick look at a 3 GB gonad on a laptop."""
    path = Path(path)
    meta = api.read_image_metadata(path)
    check_single_timepoint(path, meta)
    stack = api.read_image(path, xy_stride=int(xy_stride))
    return layers_from_stack(stack, path, meta)


def layers_from_stack(stack, path: str | Path, meta: dict | None = None) -> list[LayerData]:
    """One Image layer per channel of a germquant Stack (C, Z, Y, X), scaled by (dz, dy, dx) microns."""
    path = Path(path)
    stem = api.image_stem(path)
    names = [str(n) for n in (stack.channel_names or [])]
    spacing = tuple(float(s) for s in stack.spacing)
    if not stack.spacing_ok:
        warnings.warn(f"{path.name}: voxel size unreadable; layers use 1 um per voxel (every length the "
                      f"pipeline writes for this file carries the voxel:UNREADABLE flag)", stacklevel=2)
    layers: list[LayerData] = []
    for i in range(stack.n_channels):
        ch_name = names[i] if i < len(names) and names[i] else f"C{i}"
        info = {
            "path": str(path), "image_id": stem, "channel_index": i, "channel_name": ch_name,
            "channel_names": names, "spacing_um": spacing, "spacing_ok": bool(stack.spacing_ok),
            "xy_stride": int(getattr(stack, "downsample_xy", 1) or 1),
            "sizes": dict((meta or {}).get("sizes") or {}),
        }
        kwargs = {
            "name": f"{stem} [{ch_name}]",
            "scale": spacing,
            "metadata": {META_KEY: info},
            "blending": "additive",
            "colormap": COLORMAPS[i % len(COLORMAPS)],
        }
        layers.append((stack.data[i], kwargs, "image"))
    return layers


def image_info(layer) -> dict | None:
    """The ``napari_germquant`` record of a layer this plugin added, or None for any other layer."""
    try:
        info = layer.metadata.get(META_KEY)
    except AttributeError:
        return None
    return dict(info) if isinstance(info, dict) and "path" in info else None
