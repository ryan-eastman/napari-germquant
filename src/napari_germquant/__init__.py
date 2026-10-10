"""napari-germquant: a napari front end for the germquant engine.

The plugin contributes a reader (one Image layer per channel, physical scale from the file), a Run dock
widget (channel roles, profile, stages, xy stride, output folder, Run, Export profile) and a Results dock
widget (an existing results folder as Labels and Points layers with the tables as features). Every run goes
through ``germquant.api.run``, the function the ``germquant run`` command itself calls, so a run started
here and one started in a terminal with the same settings write the same files.

Import order matters on the lab workstation: ``germquant.api`` (and with it ``germquant.pipeline``) must be
imported before anything that could import CuPy, so torch's cuBLAS loads first. The engine glue module does
that at import, and every other module of this package imports the glue first.
"""
from __future__ import annotations

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("napari-germquant")
except _metadata.PackageNotFoundError:  # pragma: no cover - a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
