"""The Time-lapse dock's Link backend rows for the engine's fifth backend, laptrack_flow (the flow prior of
germquant.timelapse.flowlink, engine pull request 16). tests/test_timelapse.py's
test_backend_choices_say_what_is_installed pins that the dock lists the engine's backends in the engine's order;
these tests pin the rest of the row: it is there only when the engine has the backend, it is usable exactly where
laptrack is installed (the backend links with laptrack), and its text is one line, the laptrack row's install hint
when laptrack is missing. Neither the workstation's plugin environment nor the CI installs laptrack, and the CI's
engine pin may predate laptrack_flow, so the first test stands in for both through monkeypatch."""
from __future__ import annotations

import importlib.util

import pytest

from napari_germquant import _timelapse as TL
from germquant.timelapse import track as gq_track

FOUR = ("overlap", "overlap_inhouse", "laptrack", "none")
FLOW = "laptrack_flow"


@pytest.mark.parametrize("installed", [True, False], ids=["laptrack", "no_laptrack"])
def test_the_flow_row_follows_the_engine_and_laptrack(monkeypatch, installed):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "laptrack":
            return object() if installed else None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(TL, "BACKENDS", FOUR)                 # an engine before laptrack_flow: four rows, as before
    assert [b for b, _usable, _what in TL.backend_choices()] == list(FOUR)
    monkeypatch.setattr(TL, "BACKENDS", (*FOUR, FLOW))
    rows = {b: (usable, what) for b, usable, what in TL.backend_choices()}
    assert list(rows) == [*FOUR, FLOW]
    assert rows[FLOW][0] is installed and rows["laptrack"][0] is installed
    text = rows[FLOW][1]
    assert text and "\n" not in text
    if installed:
        assert "motion of its neighbours" in text and "timelapse.track.flow_*" in text
    else:
        assert text == rows["laptrack"][1] and "uv pip install laptrack" in text


def test_the_flow_row_against_this_engine():
    rows = {b: (usable, what) for b, usable, what in TL.backend_choices()}
    assert (FLOW in rows) == (FLOW in gq_track.BACKENDS)
    if FLOW in rows:
        assert rows[FLOW][0] == (importlib.util.find_spec("laptrack") is not None)
