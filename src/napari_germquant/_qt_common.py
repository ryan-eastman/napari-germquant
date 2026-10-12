"""Small Qt helpers shared by the dock widgets."""
from __future__ import annotations

import logging
import threading
import time
import traceback

from qtpy.QtCore import QObject, QTimer, Signal

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


# the loggers whose INFO lines are worth a status-bar glance while the engine works ("napari_germquant.engine"
# carries the lines an engine subprocess prints, _runner.py)
_LOGGERS = ("germquant", "cellpose", "spotmax", "napari_germquant")
# Several docks can run an engine call at the same time (a Run and a training, say). The logger levels are
# raised to INFO when the first bridge attaches and put back when the last one detaches, so one dock
# finishing never silences another that is still running.
_LEVEL_LOCK = threading.Lock()
_LEVEL_USERS = 0
_SAVED_LEVELS: dict[str, int] = {}


def _raise_levels() -> None:
    global _LEVEL_USERS
    with _LEVEL_LOCK:
        if _LEVEL_USERS == 0:
            for name in _LOGGERS:
                log = logging.getLogger(name)
                _SAVED_LEVELS[name] = log.level
                if log.getEffectiveLevel() > logging.INFO:
                    log.setLevel(logging.INFO)
        _LEVEL_USERS += 1


def _restore_levels() -> None:
    global _LEVEL_USERS
    with _LEVEL_LOCK:
        _LEVEL_USERS = max(0, _LEVEL_USERS - 1)
        if _LEVEL_USERS == 0:
            for name in _LOGGERS:
                logging.getLogger(name).setLevel(_SAVED_LEVELS.get(name, logging.NOTSET))
            _SAVED_LEVELS.clear()


class LogBridge(QObject):
    """Forwards log records from the worker thread to the GUI thread as a Qt signal (a signal emitted from
    another thread is queued to the receiver's thread, which is the one this object was created in)."""

    message = Signal(str)

    LOGGERS = _LOGGERS

    def __init__(self, parent=None):
        super().__init__(parent)
        self._handler: logging.Handler | None = None

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
        _raise_levels()
        for name in self.LOGGERS:
            logging.getLogger(name).addHandler(self._handler)

    def detach(self) -> None:
        if self._handler is None:
            return
        for name in self.LOGGERS:
            logging.getLogger(name).removeHandler(self._handler)
        self._handler = None
        _restore_levels()


def set_viewer_status(viewer, text: str) -> None:
    if viewer is None:
        return
    try:
        viewer.status = text
    except Exception:  # noqa: BLE001 - the status bar is a convenience, never a reason to fail
        pass


def format_exception(exc: BaseException, limit: int = 1500) -> str:
    """``Type: message`` and the tail of the traceback, for a status label."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-limit:]
    return f"{type(exc).__name__}: {exc}\n{tb}"


class WorkerRunner(QObject):
    """One engine call at a time in a napari thread worker, so the viewer never freezes: the elapsed time
    and the engine's log lines go to the status bar while it runs. ``start`` refuses a second call while
    one is in flight. The callbacks run in the GUI thread (napari queues the worker's signals to it).

    ``line_format`` turns one log line into the progress text shown after the elapsed time (None: the line
    itself, shortened); returning None keeps the previous progress text.

    ``should_stop`` is for the call itself (any thread): True after ``cancel()``, and when napari asks the
    worker to quit (napari does that for every thread worker when it closes). A call that polls it can end its
    engine process instead of keeping napari waiting."""

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.worker = None
        self.label = ""
        self.progress = ""
        self._t0 = 0.0
        self._line_format = None
        self._finished_cb = None
        self._cancel = threading.Event()
        self._bridge = LogBridge(self)
        self._bridge.message.connect(self._on_line)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    @property
    def busy(self) -> bool:
        return self.worker is not None

    def should_stop(self) -> bool:
        """Whether the call in flight should stop (Cancel, or napari quitting)."""
        if self._cancel.is_set():
            return True
        worker = self.worker
        return bool(worker is not None and getattr(worker, "abort_requested", False))

    def cancel(self) -> bool:
        """Ask the call in flight to stop (it does so when it polls should_stop); False when none runs."""
        if self.worker is None:
            return False
        self._cancel.set()
        return True

    def elapsed_text(self) -> str:
        s = int(time.monotonic() - self._t0)
        return f"{s // 60} min {s % 60:02d} s"

    def start(self, func, *, label: str, returned, errored, finished=None, line_format=None) -> bool:
        if self.worker is not None:
            return False
        from napari.qt.threading import create_worker

        self.label = label
        self.progress = ""
        self._line_format = line_format
        self._finished_cb = finished
        self._cancel.clear()
        self._t0 = time.monotonic()
        self._bridge.attach()
        self._timer.start()
        set_viewer_status(self.viewer, f"{label} ...")
        try:
            # an explicit errored connection: without one napari re-raises the exception in the Qt event loop
            self.worker = create_worker(func, _connect={"returned": returned, "errored": errored,
                                                        "finished": self._on_finished})
        except BaseException:
            self._timer.stop()
            self._bridge.detach()
            self.worker = None
            self._finished_cb = None
            raise
        return True

    def _status_text(self) -> str:
        text = f"{self.label} ... {self.elapsed_text()}"
        return text + (f": {self.progress}" if self.progress else "")

    def _tick(self) -> None:
        set_viewer_status(self.viewer, self._status_text())

    def _on_line(self, text: str) -> None:
        if self.worker is None:
            return
        shown = self._line_format(text) if self._line_format is not None else text[:160]
        if shown is not None:
            self.progress = shown
        set_viewer_status(self.viewer, self._status_text())

    def _on_finished(self) -> None:
        self._timer.stop()
        self._bridge.detach()
        self.worker = None
        cb, self._finished_cb = self._finished_cb, None
        if cb is not None:
            cb()
