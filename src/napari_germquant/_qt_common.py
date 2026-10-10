"""Small Qt helpers shared by the two dock widgets."""
from __future__ import annotations

import logging

from qtpy.QtCore import QObject, Signal

from . import _engine  # noqa: F401  (import order: germquant.api before anything else)


def add_layer_tuples(viewer, layer_tuples) -> list:
    """Add ``(data, kwargs, layer_type)`` tuples to a viewer; returns the layers. None viewer: nothing."""
    if viewer is None:
        return []
    from napari.layers import Layer

    added = []
    for data, kwargs, layer_type in layer_tuples:
        layer = Layer.create(data, dict(kwargs), layer_type)
        viewer.add_layer(layer)
        added.append(layer)
    return added


class LogBridge(QObject):
    """Forwards log records from the worker thread to the GUI thread as a Qt signal (a signal emitted from
    another thread is queued to the receiver's thread, which is the one this object was created in)."""

    message = Signal(str)

    # the loggers whose INFO lines are worth a status-bar glance during a run
    LOGGERS = ("germquant", "cellpose", "spotmax")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._handler: logging.Handler | None = None
        self._saved_levels: dict[str, int] = {}

    def attach(self) -> None:
        if self._handler is not None:
            return
        bridge = self

        class _Handler(logging.Handler):
            def emit(self, record):
                try:
                    bridge.message.emit(record.getMessage())
                except RuntimeError:  # the bridge was deleted while the worker still logs
                    pass

        self._handler = _Handler(level=logging.INFO)
        for name in self.LOGGERS:
            log = logging.getLogger(name)
            self._saved_levels[name] = log.level
            if log.getEffectiveLevel() > logging.INFO:
                log.setLevel(logging.INFO)
            log.addHandler(self._handler)

    def detach(self) -> None:
        if self._handler is None:
            return
        for name in self.LOGGERS:
            log = logging.getLogger(name)
            log.removeHandler(self._handler)
            log.setLevel(self._saved_levels.get(name, logging.NOTSET))
        self._handler = None
        self._saved_levels.clear()


def set_viewer_status(viewer, text: str) -> None:
    if viewer is None:
        return
    try:
        viewer.status = text
    except Exception:  # noqa: BLE001 - the status bar is a convenience, never a reason to fail
        pass
