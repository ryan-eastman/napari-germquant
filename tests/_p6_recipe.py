"""Helpers of the Recipe tab tests (tranche 5 Train B, unit P6-RECIPE: the role editor and the parameter form).

* ``renamed_example(folder, names)``: the test pack's synthetic example written again with other channel names
  (for example H2B-mCherry and Rad52-GFP, which no token of the test pack's channel map matches, or a Greek-gamma
  name), the pixels and the voxel size unchanged;
* ``nofallback_pack(root)``: a copy of the test pack, ``nofallback_pack``, whose required role dna has no index
  fallback, so an image whose channel names it cannot match leaves dna unbound (the test pack's own map would
  fall back to channel 0); its recipes' expected_config_hash follow the edited map and SHA256SUMS is rebuilt;
* ``recipe_files(root)``: the bytes of a project's recipe.yaml and channel_map.yaml (None when absent);
* ``engine_edit(root, *args)``: ``python -m germquant.project edit ROOT ARGS --json`` in a child process, the
  engine's own answer to compare the tab with;
* ``FakeSession`` and ``FakeRunner``: an engine session and runner that answer ``project_edit`` with canned
  records, and ``fake_view(...)``: a RecipeView built by hand, for the widget tests that need no engine;
* ``FakePostRunner`` and ``post_rows()``: a fake runner that also lists post-stages (two listing rows shaped like
  `python -m germquant.post_stages list --json`, one like the engine's fixture post-stage) and writes
  post_stages.yaml, for the post-stage section's widget tests.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from conftest import TEST_IMAGE, TEST_PACK, write_ome

GAMMA = chr(0x03B3)          # a Greek gamma (gammaH2AX), kept out of this file's bytes
RENAMED = ("H2B-mCherry", "Rad52-GFP")


def renamed_example(folder: Path, names, *, name: str = "renamed.ome.tif") -> Path:
    """The test pack's example with the channel names ``names`` (two of them), as an OME-TIFF in ``folder``."""
    from germquant.assemble import read_ome_tiff

    img = read_ome_tiff(TEST_IMAGE)
    names = [str(n) for n in names]
    if len(names) != len(img["channel_names"]):
        raise ValueError(f"the example has {len(img['channel_names'])} channels, not {len(names)}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return write_ome(folder / name, img["data"], names, spacing=tuple(img["spacing_zyx_um"]))


def nofallback_pack(root: Path) -> Path:
    """A test pack copy (id nofallback_pack) whose dna role has no index fallback; returns the folder holding it
    (put it on $GERMQUANT_PACKS)."""
    import yaml
    from germquant import packs
    from germquant.config import load_config

    dest = Path(root) / "nofallback_pack"
    shutil.copytree(TEST_PACK, dest, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    cm = dest / "config" / "channel_maps" / "test_dna_foci.yaml"
    data = yaml.safe_load(cm.read_text(encoding="ascii"))
    data["roles"]["dna"].pop("index", None)
    cm.write_bytes(yaml.safe_dump(data, sort_keys=False, allow_unicode=False).encode("ascii"))
    manifest = yaml.safe_load((dest / "pack.yaml").read_text(encoding="ascii"))
    manifest["id"] = "nofallback_pack"
    for r in manifest["recipes"]:
        r["expected_config_hash"] = load_config(dest / r["profile"]).hash
    (dest / "pack.yaml").write_bytes(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False).encode("ascii"))
    packs.write_sums(dest)
    return Path(root)


def recipe_files(root: Path) -> dict[str, bytes | None]:
    out = {}
    for n in ("recipe.yaml", "channel_map.yaml"):
        p = Path(root) / n
        out[n] = p.read_bytes() if p.is_file() else None
    return out


def engine_edit(root: Path, *args: str) -> dict:
    """``python -m germquant.project edit ROOT ARGS --json``: the engine's record (exit 0 asserted)."""
    proc = subprocess.run([sys.executable, "-m", "germquant.project", "edit", str(root), *args, "--json"],
                          capture_output=True, text=True, encoding="utf-8", timeout=300,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def no_machine_path(text: str, root: Path) -> bool:
    """Whether ``text`` names no folder of this computer: not the project root, no drive letter, no backslash."""
    import re

    low = text.lower()
    return (str(root).lower() not in low and Path(root).as_posix().lower() not in low and "\\" not in text
            and re.search(r"(?<![A-Za-z])[A-Za-z]:/", text) is None)


# ---------------------------------------------------------------------------------------------------
# the fakes of the widget tests
# ---------------------------------------------------------------------------------------------------


class FakeRunner:
    """Answers ``project_edit`` from ``answer(kwargs) -> record`` (a dict, or an exception to raise) and keeps every
    call in ``calls``; ``gate`` (a threading.Event), when set up, holds each call until it is set."""

    kind = "fake"
    python = sys.executable

    def __init__(self, answer):
        self.answer = answer
        self.calls: list[dict] = []
        self.gate: threading.Event | None = None
        self.key = "fake:engine"

    def project_edit(self, root, *, should_stop=None, **kwargs):
        if should_stop is not None and should_stop():
            raise RuntimeError("the fake engine was asked to stop")
        self.calls.append({"root": str(root), **kwargs})
        gate = self.gate
        if gate is not None:
            gate.wait(30)
        rec = self.answer(dict(kwargs))
        if isinstance(rec, BaseException):
            raise rec
        return rec


class FakeSession:
    """The engine session of a widget test: ``require`` returns the fake runner (or raises ``error``)."""

    def __init__(self, runner=None, error: BaseException | None = None):
        self.runner_obj = runner
        self.error = error
        self.required: list[tuple] = []
        self.changed = None

    def require(self, *features, should_stop=None):
        self.required.append(features)
        if self.error is not None:
            raise self.error
        return self.runner_obj

    def runner(self):
        return self.runner_obj

    def describe(self) -> str:
        return "a fake engine for the widget tests"

    def warning(self) -> str:
        return ""


def preview_record(bindings: dict, problems=(), *, config_hash="aaaaaaaaaaaa", changed=True, warnings=(),
                   validated=None, yaml_text="") -> dict:
    """A `project edit --dry-run --json` record with a role preview."""
    return {"dry_run": True, "changed": changed, "config_hash": config_hash, "previous_config_hash": "02bb83cf74f9",
            "edited": changed, "validated": validated or {"value": False, "why": "edited: recipe.yaml changes the "
                                                                                 "pack recipe", "record": None},
            "recipe_yaml": None, "channel_map_yaml": None,
            "preview": {"yaml": yaml_text, "bindings": bindings, "problems": list(problems)},
            "written": [], "removed": [], "warnings": list(warnings)}


def fake_view(root: Path, *, tunables=(), others=(), roles=None, channels=("DAPI", "foci"), images=None,
              reference=None, config_hash="02bb83cf74f9", validated=None, warnings=(), digest="d0", saved=None,
              manifest=None, post=None):
    """A RecipeView of a project on the test pack's basic recipe, built by hand (``manifest``: the manifest as
    read, by default just the saved stamp; ``post``: the post-stage list in force, by default none)."""
    from napari_germquant._params import PostList, RecipeView, Role, Tunable

    if roles is None:
        roles = [Role("dna", "DNA", True, ("DAPI", "Hoechst", "DNA"), 0, None),
                 Role("foci", "foci", False, ("foci", "puncta", "spots"), 1, None)]
    images = images if images is not None else [{"id": "img1", "path": "images/img1.ome.tif", "excluded": False}]
    tun = [t if isinstance(t, Tunable) else Tunable.from_declaration(t[0], pack_value=t[1], value=t[2], origin=t[3])
           for t in tunables]
    return RecipeView(root=Path(root), name="p", pack="test_pack@0.1.0", recipe="basic", pack_hash="02bb83cf74f9",
                      config_hash=config_hash, edited=config_hash != "02bb83cf74f9",
                      validated=validated or {"value": False, "why": "not validated: the pack recipe "
                                                                     "test_pack@0.1.0/basic is provisional",
                                              "record": None},
                      tunables=tun, others=list(others), roles=list(roles), match_by="name", images=images,
                      reference=reference or (images[0]["id"] if images else None), channels=list(channels),
                      channels_error=None, warnings=list(warnings), digest=digest, saved=saved,
                      manifest=dict(manifest) if manifest is not None else ({"saved": saved} if saved else {}),
                      post=post if post is not None else PostList())


# ---------------------------------------------------------------------------------------------------
# the post-stage section's fakes
# ---------------------------------------------------------------------------------------------------

FIXTURE = "fixture_nuclei"            # the name of the engine's fixture post-stage (germquant.post_stages.testing)
RING = "ring_intensity"               # the worked example of the engine's docs/POST_STAGES.md


def post_rows() -> list[dict]:
    """Two rows of a post-stage listing (`python -m germquant.post_stages list --json`, PLAN 9.1 and 9.4 #9), sorted
    by name as the engine sorts them: one shaped like the engine's fixture post-stage (a float, an int, a choice and
    a bool parameter) and the worked example of docs/POST_STAGES.md (a float with a unit, a word)."""
    return [
        {"name": FIXTURE, "version": "1.0.0", "dist": "germquant-fixture-post-stage",
         "title": "Fixture: per-nucleus voxels, volume, foci statistic and spot count",
         "reads": {"roles": ["foci"], "tables": ["nuclei", "spots"], "labels": True, "image": True},
         "writes": {"per_nucleus": ["nucleus_id", "n_voxels", "volume_um3", "foci_value", "foci_value_scaled",
                                    "n_spots"]},
         "params": [{"key": "factor", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0,
                     "label": "Scale factor of foci_value"},
                    {"key": "min_voxels", "type": "int", "default": 0, "min": 0, "max": 1000000000,
                     "label": "Smallest nucleus (voxels)"},
                    {"key": "statistic", "type": "choice", "default": "mean", "choices": ["mean", "max"],
                     "label": "Foci statistic"},
                    {"key": "fail", "type": "bool", "default": False, "label": "Fail on purpose (tests)"}],
         "source": "entry point germquant_fixture_post_stage:measure"},
        {"name": RING, "version": "0.1.0", "dist": "germquant-ring-intensity",
         "title": "Foci intensity in a shell around each nucleus",
         "reads": {"roles": ["foci"], "tables": [], "labels": True, "image": True},
         "writes": {"rings": ["nucleus_id", "ring_voxels", "ring_mean", "inside_mean"]},
         "params": [{"key": "radius_um", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "unit": "um",
                     "label": "Shell width"},
                    {"key": "note", "type": "str", "default": "none"}],
         "source": "entry point germquant_ring_intensity:measure"},
    ]


class FakePostRunner(FakeRunner):
    """A FakeRunner that also has the runner's two post-stage methods: ``list_post_stages`` answers ``rows`` (or
    raises ``list_error``); ``write_post_stages(root, entries)`` writes the entries into <root>/post_stages.yaml as
    YAML (or raises ``write_error``) and answers {"path", "entries"}, the two keys both runners' records hold.
    ``post_calls`` keeps every call ("list", or ("write", entries)); ``calls`` stays project_edit's only.
    ``post_gate`` (a threading.Event), when set up, holds each write until it is set."""

    def __init__(self, answer, rows=None):
        super().__init__(answer)
        self.rows = post_rows() if rows is None else list(rows)
        self.list_error: BaseException | None = None
        self.write_error: BaseException | None = None
        self.post_calls: list = []
        self.post_gate: threading.Event | None = None

    def list_post_stages(self, *, should_stop=None) -> list:
        self.post_calls.append("list")
        if self.list_error is not None:
            raise self.list_error
        return copy.deepcopy(self.rows)

    def write_post_stages(self, root, entries, *, should_stop=None) -> dict:
        import yaml

        self.post_calls.append(("write", copy.deepcopy(list(entries))))
        gate = self.post_gate
        if gate is not None:
            gate.wait(30)
        if self.write_error is not None:
            raise self.write_error
        path = Path(root) / "post_stages.yaml"
        path.write_bytes(yaml.safe_dump(list(entries), sort_keys=True, allow_unicode=False).encode("ascii"))
        return {"path": str(path), "entries": copy.deepcopy(list(entries))}
