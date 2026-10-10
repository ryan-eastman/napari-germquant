"""Shared fixtures. Headless: QT_QPA_PLATFORM=offscreen is set here before Qt loads (the command line may set
it too), and the GPU is hidden so no test can touch it. The synthetic gonad is the engine's own test stack
(tests/test_end_to_end.py in the germquant checkout: a chain of SYP-positive nuclei with one RAD-51 focus each
plus a separate SYP-negative cluster), written as an OME-TIFF so the plugin reads a real file through the same
reader the pipeline uses; nothing is monkeypatched."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import contextlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from napari_germquant import _engine  # noqa: E402  (imports germquant.api first)


@pytest.fixture(autouse=True, scope="session")
def _offscreen_without_opengl():
    """Qt's ``offscreen`` platform plugin cannot create an OpenGL context on Windows (and macOS), and napari
    queries two OpenGL limits (texture sizes, extensions) while it builds a viewer, through PyOpenGL. A hidden
    viewer (``show=False``, what make_napari_viewer gives) never draws, so those two queries are the only GL
    calls these tests would make: under the offscreen platform they get constants instead. Nothing is rendered
    in this suite and the GPU is never touched. With a real platform plugin (QT_QPA_PLATFORM unset, a desktop
    session) the shim is not applied and napari queries the real driver."""
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        yield
        return
    import napari._vispy.utils.gl as napari_gl

    class _StubGL:
        GL_MAX_TEXTURE_SIZE = 0x0D33
        GL_EXTENSIONS = 0x1F03
        _MAX_3D = 32883

        def glGetParameter(self, param):
            if param == self.GL_EXTENSIONS:
                return "GL_ARB_texture_float GL_OES_texture_float"
            return 16384 if param == self.GL_MAX_TEXTURE_SIZE else 2048

    saved = (napari_gl._opengl_context, napari_gl.gl)
    napari_gl._opengl_context = contextlib.nullcontext
    napari_gl.gl = _StubGL()
    try:
        yield
    finally:
        napari_gl._opengl_context, napari_gl.gl = saved

SPACING = (0.4, 0.2, 0.2)
CHANNELS = ["405", "477", "545"]           # DAPI / SYP / RAD-51 by laser line, as the n2 channel map expects
IMAGE_ID = "20251105_n2_nohs_HERM_001"     # matches the engine's filename regex (genotype n2, no HS, herm, 001)
IMAGE_NAME = IMAGE_ID + ".ome.tif"
LOCAL_ND2 = Path("C:/Users/ryane/n2_rad51_local/20251105_N2_noHS/20251105_N2_nohs_HERM _001.nd2")
NAS_ND2 = Path(r"\\129.82.125.105\SpermBank\20261005_DLW93_nohs_gfpsyp2_5min_2hr_herm.nd2")


def synthetic_gonad() -> np.ndarray:
    """(C, Z, Y, X) float32: the engine's synthetic 3-channel gonad, copied from its test_end_to_end."""
    Z, Y, X = 20, 80, 320
    dapi = np.zeros((Z, Y, X), np.float32)
    syp = np.zeros((Z, Y, X), np.float32)
    rad = np.zeros((Z, Y, X), np.float32)
    zz, yy, xx = np.indices((Z, Y, X)).astype(np.float32)
    for cx in range(20, 210, 17):                              # about 12 nuclei along x at y=40
        r2 = (((zz - 10) * SPACING[0]) ** 2 + ((yy - 40) * SPACING[1]) ** 2 + ((xx - cx) * SPACING[2]) ** 2)
        dapi += np.exp(-r2 / (2 * 1.6 ** 2)) * 1500.0
        syp[10, 40, max(0, cx - 6):min(X, cx + 6)] = 2000.0    # a short SC filament
        rad[10, 42, min(cx + 1, X - 1)] = 6000.0               # one RAD-51 focus
    for jx in (280, 295, 310):                                 # off-germline junk: DAPI only, far away
        r2 = (((zz - 10) * SPACING[0]) ** 2 + ((yy - 40) * SPACING[1]) ** 2 + ((xx - jx) * SPACING[2]) ** 2)
        dapi += np.exp(-r2 / (2 * 1.6 ** 2)) * 1500.0
    return np.stack([dapi, syp, rad], axis=0)


def write_ome(path: Path, data: np.ndarray, names, spacing=SPACING, axes: str = "CZYX") -> Path:
    import tifffile

    dz, dy, dx = spacing
    tifffile.imwrite(str(path), data, ome=True, photometric="minisblack", metadata={
        "axes": axes,
        "PhysicalSizeX": dx, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": dy, "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": dz, "PhysicalSizeZUnit": "µm",
        "Channel": {"Name": list(names)},
    })
    return path


@pytest.fixture(scope="session")
def engine_root() -> Path:
    root = _engine.checkout_root()
    assert root is not None and (root / "config" / "profiles" / "rad51_foci.yaml").is_file(), \
        "germquant's config/profiles must be visible (an editable install from a checkout)"
    return root


@pytest.fixture(scope="session")
def synthetic_image(tmp_path_factory) -> Path:
    return write_ome(tmp_path_factory.mktemp("image") / IMAGE_NAME, synthetic_gonad(), CHANNELS)


@pytest.fixture(scope="session")
def classical_overlay(tmp_path_factory, engine_root) -> Path:
    """A config file outside the checkout that forces the classical (CPU) nucleus segmenter on top of the
    rad51_foci profile: no GPU, no Cellpose model, no SpotMAX in the plugin environment."""
    base = (engine_root / "config" / "profiles" / "rad51_foci.yaml").resolve().as_posix()
    p = tmp_path_factory.mktemp("recipe") / "rad51_classical.yaml"
    p.write_text(f"extends: {base}\nsegmentation:\n  nuclei:\n    method: classical\n", encoding="utf-8")
    return p
