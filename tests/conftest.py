"""Shared fixtures. Headless: QT_QPA_PLATFORM=offscreen is set here before Qt loads (the command line may set
it too), and the GPU is hidden so no test can touch it. The synthetic gonad is the engine's own test stack
(tests/test_end_to_end.py in the germquant checkout: a chain of SYP-positive nuclei with one RAD-51 focus each
plus a separate SYP-negative cluster), written as an OME-TIFF so the plugin reads a real file through the same
reader the pipeline uses; nothing is monkeypatched.

Every test gets its own settings file ($NAPARI_GERMQUANT_SETTINGS), so no test reads or writes the engine
interpreter stored on this computer, and no engine is chosen unless the test chooses one. ``pack_env`` isolates
every pack root and store a pack test could touch (as the engine's tests/packs_helpers.py does): the test pack
copy under tests/fixtures/packs as $GERMQUANT_PACKS, an empty model store ($GERMQUANT_MODELS), an empty
Cellpose cache, a private per-user folder and a private examples folder. A subprocess engine inherits all of
it, so both runners see the same packs and stores.

The Run dock checks engines, lists packs and checks recipes in a thread worker, never in the GUI thread:
``settle(w, qtbot)`` waits until those calls are done before a test reads what they set."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import http.server  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from napari_germquant import _engine  # noqa: E402  (imports germquant.api first)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PACKS = ROOT / "tests" / "fixtures" / "packs"
TEST_PACK = FIXTURE_PACKS / "test_pack"
TEST_IMAGE = TEST_PACK / "example" / "test_pack_synthetic.ome.tif"
TINYNET = "tinynet@f5055f8a"
# the test pack's 64-byte stand-in model (its card's stand_in_content plus one line feed)
TINYNET_BYTES = b"germquant test pack: tinynet stand-in weights, not a real model\n"
TINYNET_SHA = hashlib.sha256(TINYNET_BYTES).hexdigest()
HEAVY = ("torch", "cupy", "spotmax", "cellpose")


def settle(w, qtbot, timeout: int = 120_000) -> None:
    """Wait until the Run dock ``w`` has no engine call (version check, pack list, card check) in flight or
    waiting."""
    qtbot.waitUntil(lambda: w.idle, timeout=timeout)


@pytest.fixture(autouse=True)
def _private_settings(tmp_path, monkeypatch):
    """A settings file of the test's own: the dock starts with no engine chosen, and the interpreter stored
    on this computer is never read or changed."""
    monkeypatch.setenv("NAPARI_GERMQUANT_SETTINGS", str(tmp_path / "napari_germquant_settings.ini"))
    for var in ("GERMQUANT_DEVICE",):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def pack_env(tmp_path, monkeypatch):
    """Isolated pack roots and stores (module docstring); the test pack is found under $GERMQUANT_PACKS.

    The model store sits in a short folder of the system temporary folder ($RUNNER_TEMP on a CI runner),
    removed afterwards, not under tmp_path: a blob's record is <store>/models/blobs/<64 hex>/<64 hex>.json.part,
    about 150 characters below the store, and under a long pytest base folder (a long user name, a test with a
    long name) that passes the 260-character path limit of a Windows without long paths enabled."""
    pytest.importorskip("germquant.packs", reason="system packs need germquant 0.2.0.dev0 or later")
    from germquant import packs

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("GERMQUANT_CONFIG_ROOT", raising=False)
    monkeypatch.setenv("GERMQUANT_PACKS", str(FIXTURE_PACKS))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "localappdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "xdg"))
    models = Path(tempfile.mkdtemp(prefix="gqm", dir=os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()))
    monkeypatch.setenv("GERMQUANT_MODELS", str(models))
    cache = tmp_path / "cellpose_cache"
    cache.mkdir()
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(cache))
    examples = tmp_path / "examples"
    monkeypatch.setenv("GERMQUANT_EXAMPLES", str(examples))
    monkeypatch.delenv("GERMQUANT_ALLOW_LOOPBACK_EXAMPLES", raising=False)
    packs.refresh()
    try:
        yield SimpleNamespace(tmp=tmp_path, models=models, cache=cache, examples=examples,
                              blobs=models / "models" / "blobs")
    finally:
        packs.refresh()
        shutil.rmtree(models, ignore_errors=True)


def copy_pack(src: Path, dest: Path, *, edit=None) -> Path:
    """Copy a pack folder, apply ``edit(manifest_dict)`` to its pack.yaml, and rebuild SHA256SUMS (pack.yaml is
    not hashed, so no recipe's config_hash moves)."""
    import yaml
    from germquant import packs

    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    if edit is not None:
        data = yaml.safe_load((dest / "pack.yaml").read_text(encoding="ascii"))
        edit(data)
        (dest / "pack.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False, allow_unicode=False).encode("ascii"))
    packs.write_sums(dest)
    return dest


class _Files(http.server.BaseHTTPRequestHandler):
    """A stand-in file server for model and example downloads (Range requests answered)."""

    files: dict = {}
    log: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802 - the http.server API
        cls = type(self)
        path = self.path.lstrip("/")
        cls.log.append((path, self.headers.get("Range")))
        data = cls.files.get(path)
        if data is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        rng = self.headers.get("Range")
        start, end = 0, len(data) - 1
        if rng:
            a, b = re.fullmatch(r"bytes=(\d*)-(\d*)", rng).groups()
            start, end = (len(data) - int(b), len(data) - 1) if a == "" else (int(a), int(b) if b else len(data) - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:end + 1]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def serve(files: dict):
    """``(base_url, handler)`` of a 127.0.0.1 server holding ``files`` ({path: bytes}) for the test's duration."""
    handler = type("H", (_Files,), {"files": files, "log": []})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/", handler
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def served_pack(pack_env, tmp_path, monkeypatch):
    """A copy of the test pack, ``served_pack``, whose model and real example come from a 127.0.0.1 server the
    test runs: tinynet gains a url source, and a remote example ``standin`` assembles the test pack's synthetic
    field from two per-channel TIFFs (plain http to this machine is accepted with
    $GERMQUANT_ALLOW_LOOPBACK_EXAMPLES=1, the engine's switch for test servers). Only this copy is listed."""
    import io

    import tifffile
    from germquant import packs
    from germquant.assemble import build, pixels_sha256, read_ome_tiff

    img = read_ome_tiff(TEST_IMAGE)
    data, names, spacing = img["data"], [str(n) for n in img["channel_names"]], list(img["spacing_zyx_um"])
    files: dict = {"models/tinynet": TINYNET_BYTES}
    inputs = tmp_path / "served_inputs"
    inputs.mkdir()
    for i, name in enumerate(names):
        buf = io.BytesIO()
        tifffile.imwrite(buf, np.ascontiguousarray(data[i]))
        files[f"data/standin_{name}.tif"] = buf.getvalue()
        (inputs / f"standin_{name}.tif").write_bytes(buf.getvalue())
    assembled, _ = build([inputs / f"standin_{n}.tif" for n in names], names, axes="ZYX")
    with serve(files) as (base, handler):
        def edit(d):
            d["id"] = "served_pack"
            d["models"][0]["sources"] = [*d["models"][0]["sources"], {"type": "url", "url": base + "models/tinynet"}]
            d["examples"] = [*(d.get("examples") or []), {
                "id": "standin", "recipe": "basic", "base_url": base + "data/",
                "sources": [{"path": f"standin_{n}.tif", "size": len(files[f"data/standin_{n}.tif"]),
                             "sha256": hashlib.sha256(files[f"data/standin_{n}.tif"]).hexdigest()} for n in names],
                "assemble": {"axes": "ZYX", "channels": names, "spacing_zyx_um": spacing},
                "pixels_sha256": pixels_sha256(assembled), "licence": "CC0", "licence_spdx": "CC0-1.0",
                "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "attribution": "the test pack's synthetic field, served by the test",
                "expect": {"n_objects": [1, 20], "spots_per_object": [0.0, 10.0]}}]
        root = tmp_path / "served_packs"
        copy_pack(TEST_PACK, root / "served_pack", edit=edit)
        monkeypatch.setenv("GERMQUANT_PACKS", str(root))
        monkeypatch.setenv("GERMQUANT_ALLOW_LOOPBACK_EXAMPLES", "1")
        packs.refresh()
        yield SimpleNamespace(ref="served_pack@0.1.0", base=base, log=handler.log, files=files, data=data,
                              names=names, spacing=spacing, env=pack_env, root=root)
    packs.refresh()


@pytest.fixture(scope="session")
def second_python(tmp_path_factory) -> Path:
    """A second engine interpreter: a fresh venv (no pip) whose site folder adds this interpreter's packages,
    so it runs the same germquant from another executable path. The per-recipe interpreter tests map a
    recipe to it."""
    import site

    folder = tmp_path_factory.mktemp("second_python") / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(folder)], check=True,
                   capture_output=True, timeout=300)
    py = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    out = subprocess.run([str(py), "-c", "import site, sys; print(site.getsitepackages()[-1])"], check=True,
                         capture_output=True, text=True, timeout=120).stdout.strip()
    dirs = [d for d in site.getsitepackages() if Path(d).is_dir()]
    user = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    if user and Path(user).is_dir():
        dirs.append(user)
    lines = "".join(f"import site; site.addsitedir({Path(d).as_posix()!r})\n" for d in dirs)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "napari_germquant_test_parent.pth").write_text(lines, encoding="utf-8")
    return py


def engine_device(prefer: str | None = "auto") -> str:
    """The device the engine picks for ``prefer`` in this test session: germquant.device.select_device, the
    function the engine itself calls. CUDA is hidden here (CUDA_VISIBLE_DEVICES=-1), but an Apple GPU is not:
    on the macos-14 runner torch offers MPS, so "auto" (and a "cuda" request) give mps there and cpu on the
    Linux and Windows runners. Tests compare with this instead of assuming the CPU."""
    from germquant.device import select_device

    return select_device(prefer, quiet=True)


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
