"""The npe2 manifest: valid, named like the distribution, discoverable, and its reader patterns equal the
suffixes the engine reads."""
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import napari_germquant
from germquant import api

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(napari_germquant.__file__).parent / "napari.yaml"


def test_manifest_is_valid_and_matches_pyproject():
    from npe2 import PluginManifest

    mf = PluginManifest.from_file(MANIFEST)
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert mf.name == pyproject["project"]["name"]
    assert pyproject["project"]["entry-points"]["napari.manifest"] == {mf.name: "napari_germquant:napari.yaml"}
    for cmd in mf.contributions.commands:
        assert cmd.id.startswith(mf.name + "."), cmd.id
        module, attr = cmd.python_name.split(":")
        assert callable(getattr(importlib.import_module(module), attr)), cmd.python_name
    assert {w.display_name for w in mf.contributions.widgets} == {"Run", "Results", "Training", "Time-lapse"}


def test_reader_patterns_equal_the_engine_suffixes():
    from npe2 import PluginManifest

    mf = PluginManifest.from_file(MANIFEST)
    (reader,) = mf.contributions.readers
    assert set(reader.filename_patterns) == {f"*{s}" for s in api.SUPPORTED_SUFFIXES}
    assert reader.accepts_directories                      # .zarr stores are folders


def test_plugin_is_discoverable_by_npe2():
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()
    mf = pm.get_manifest("napari-germquant")
    assert mf.display_name == "germquant"


def test_every_dock_opens_through_napari(make_napari_viewer):
    """The Plugins menu route: napari builds each widget from the manifest with the viewer injected (as a
    public-only proxy of the viewer, which the widgets must work through)."""
    viewer = make_napari_viewer()
    for name in ("Run", "Results", "Training", "Time-lapse"):
        _dock, widget = viewer.window.add_plugin_dock_widget("napari-germquant", name)
        assert getattr(widget.viewer, "__wrapped__", widget.viewer) is viewer, name
