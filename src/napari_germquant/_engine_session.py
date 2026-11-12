"""One engine session per napari viewer, shared by the docks that work on projects (docs/DESIGN_system_packs.md,
sections 8.1 and 8.4 to 8.7): the Project dock, the Training dock in project mode and the Spot tuning dock ask
it which engine runs, have it checked once, and show the one message an engine without a feature gets.

The engine is the interpreter stored for this computer (``_settings.engine_choice``, the key the Run dock reads
and writes), or this napari environment when the stored choice is ``inprocess``. Nothing here guesses: with no
engine stored, ``ensure`` raises EngineNotChosen and the dock asks (the Run dock's InterpreterDialog, then
``set_engine``). A pack recipe can have its own interpreter on this computer (``_settings.recipe_interpreter``),
as in the Run dock. The device is the per-user choice the Run dock also uses.

Every call that may start an engine process (``ensure``, ``set_engine``, ``runner_for``, ``require``,
``devices``) blocks, so the docks make it in a thread worker, never in the GUI thread. Everything else reads
only what a check left behind. ``changed`` fires whenever the engine, its check or the device changes; it is
emitted from whichever thread made the change and queued to the GUI thread, where ``session_for`` makes the
session (call it from the GUI thread, in a dock's constructor).
"""
from __future__ import annotations

import threading
import weakref

from qtpy.QtCore import QObject, Signal

from . import _settings
from ._runner import (
    DEVICES,
    EngineRunner,
    FeatureMissing,
    InProcessRunner,
    RunCancelled,
    RunnerError,
    SubprocessRunner,
    device_choices,
    mismatch_warning,
    plugin_engine_info,
)

NOT_CHOSEN = ("Not chosen yet: choose the Python of the germquant environment that should make the numbers (on the "
              "lab workstation, the engine checkout's .venv). Choose the engine environment first.")


class EngineNotChosen(RunnerError):
    """No engine is stored for this computer: the dock asks for one (it never picks one itself)."""


class EngineSession(QObject):
    """The engine of one viewer's project docks (module docstring). ``env`` adds to (or, with None values,
    removes from) the environment of every engine process the session starts, as the Run dock's runner_env."""

    changed = Signal()

    def __init__(self, env: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self._lock = threading.RLock()
        self._env: dict = dict(env or {})
        self._choice: str | None = None
        self._runner: EngineRunner | None = None
        self._info: dict | None = None
        self._error: str | None = None
        self._recipe_runners: dict[str, EngineRunner] = {}
        self._runtime: dict[str, dict] = {}

    # ---- the engine ----
    @property
    def env(self) -> dict:
        return dict(self._env)

    def set_env(self, env: dict | None) -> None:
        """Replace the extra environment; the next check makes new runners with it."""
        with self._lock:
            self._env = dict(env or {})
            self._forget()
        self._emit()

    def _forget(self) -> None:
        self._choice, self._runner, self._info, self._error = None, None, None, None
        self._recipe_runners = {}
        self._runtime = {}

    def _make(self, choice: str) -> EngineRunner:
        runner: EngineRunner = InProcessRunner() if choice == _settings.INPROCESS else \
            SubprocessRunner(choice, env=self._env)
        runner.device = _settings.device()
        return runner

    def runner(self) -> EngineRunner | None:
        """The engine checked for the choice stored now, or None: nothing stored, not checked yet, its check
        failed, or the stored choice changed since (``ensure`` checks it then). Never starts a process."""
        with self._lock:
            if self._runner is not None and self._error is None and self._choice == _settings.engine_choice():
                self._runner.device = _settings.device()
                return self._runner
            return None

    def ensure(self, should_stop=None) -> EngineRunner:
        """The engine for the stored choice, checked once (its version: ``python -m germquant.packs --version
        --json``). EngineNotChosen with nothing stored; the check's RunnerError (EngineTooOld, not a Python...)
        when that engine cannot be used. Blocks."""
        choice = _settings.engine_choice()
        if choice is None:
            raise EngineNotChosen(NOT_CHOSEN)
        current = self.runner()
        if current is not None:
            return current
        runner = self._make(choice)
        try:
            info = runner.version(should_stop=should_stop)
        except RunCancelled:
            raise
        except RunnerError as e:
            with self._lock:
                self._forget()
                self._choice, self._error = choice, str(e)
            self._emit()
            raise
        with self._lock:
            self._forget()
            self._choice, self._runner, self._info = choice, runner, dict(info)
        self._emit()
        return runner

    def set_engine(self, choice, should_stop=None) -> EngineRunner:
        """Check ``choice`` (a Python path, or ``inprocess``); only when the check passes is it stored for this
        computer and used. Blocks."""
        choice = str(choice)
        runner = self._make(choice)
        info = runner.version(should_stop=should_stop)        # raises: nothing is stored or changed
        _settings.set_engine_choice(choice)
        with self._lock:
            self._forget()
            self._choice, self._runner, self._info = choice, runner, dict(info)
        self._emit()
        return runner

    def runner_for(self, pack_id: str | None = None, recipe_id: str | None = None,
                   should_stop=None) -> EngineRunner:
        """The runner a pack recipe goes to: the interpreter stored for it on this computer (checked once), else
        the engine (``ensure``). Blocks."""
        mapped = _settings.recipe_interpreter(pack_id, recipe_id)
        if not mapped:
            return self.ensure(should_stop)
        with self._lock:
            runner = self._recipe_runners.get(mapped)
        if runner is None:
            runner = self._make(mapped)
            runner.version(should_stop=should_stop)
            with self._lock:
                self._recipe_runners[mapped] = runner
        runner.device = _settings.device()
        return runner

    def info(self) -> dict | None:
        """The checked engine's version document (germquant, git_sha, python, executable, features)."""
        with self._lock:
            return dict(self._info) if self._info is not None and self.runner() is not None else None

    def error(self) -> str | None:
        """Why the stored engine cannot be used, after a failed check (None otherwise)."""
        with self._lock:
            return self._error if self._choice == _settings.engine_choice() else None

    # ---- features ----
    def features(self) -> frozenset[str]:
        """The checked engine's features (empty while none is checked). Never starts a process."""
        runner = self.runner()
        return runner.features() if runner is not None else frozenset()

    def missing(self, *features: str) -> str | None:
        """The one message for the first of ``features`` the checked engine lacks, None when it has them all;
        without a checked engine, the line ``describe`` shows. Never starts a process."""
        runner = self.runner()
        if runner is None:
            return self.describe()
        try:
            runner.require(*features)
        except FeatureMissing as e:
            return str(e)
        return None

    def require(self, *features: str, should_stop=None) -> EngineRunner:
        """The engine (``ensure``), or FeatureMissing for the first of ``features`` it lacks. Blocks."""
        runner = self.ensure(should_stop)
        runner.require(*features, should_stop=should_stop)
        return runner

    # ---- the device ----
    def device(self) -> str:
        return _settings.device()

    def set_device(self, value: str) -> None:
        """The device asked for (auto, cuda, mps or cpu), stored per user and given to every runner."""
        if value not in DEVICES:
            raise ValueError(f"not a device: {value!r} (one of {', '.join(DEVICES)})")
        _settings.set_device(value)
        with self._lock:
            for runner in [self._runner, *self._recipe_runners.values()]:
                if runner is not None:
                    runner.device = value
        self._emit()

    def devices(self, should_stop=None) -> list[str]:
        """The devices the engine environment offers (auto, its accelerators, cpu), from its runtime probe,
        asked once per engine. Blocks."""
        runner = self.ensure(should_stop)
        with self._lock:
            rt = self._runtime.get(runner.key)
        if rt is None:
            rt = runner.runtime(should_stop=should_stop)
            with self._lock:
                self._runtime[runner.key] = dict(rt)
        return device_choices(rt)

    # ---- what the docks show ----
    def describe(self) -> str:
        """The engine line: where it runs, the engine's version and git sha, python, the device asked for."""
        runner = self.runner()
        if runner is None:
            err = self.error()
            if err:
                return f"The engine chosen for this computer cannot be used: {err}"
            if _settings.engine_choice() is None:
                return NOT_CHOSEN
            return f"checking the engine {_settings.engine_choice()} ..."
        info = self._info or {}
        return (f"{runner.describe()}; germquant {info.get('germquant')} (git {str(info.get('git_sha'))[:12]}), "
                f"python {info.get('python')}; device {self.device()}")

    def warning(self) -> str:
        """The P4 warning when the engine's germquant differs from the plugin's own (version or git sha)."""
        info = self.info()
        return mismatch_warning(plugin_engine_info(), info) if info else ""

    def _emit(self) -> None:
        try:
            self.changed.emit()
        except RuntimeError:                  # the session's QObject is gone (napari closing)
            pass


_SESSIONS: dict[int, EngineSession] = {}
_SESSIONS_LOCK = threading.Lock()


def session_for(viewer) -> EngineSession:
    """The one session of ``viewer`` (napari's public proxy of a viewer, what a dock opened from the Plugins
    menu gets, counts as that viewer); a new, unshared session for None. A session goes with its viewer."""
    if viewer is None:
        return EngineSession()
    target = getattr(viewer, "__wrapped__", viewer)
    key = id(target)
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            session = EngineSession()
            _SESSIONS[key] = session
            try:
                weakref.finalize(target, _drop, key, session)
            except TypeError:                 # not weak-referenceable: kept while napari runs
                pass
    return session


def _drop(key: int, session: EngineSession) -> None:
    with _SESSIONS_LOCK:
        if _SESSIONS.get(key) is session:
            del _SESSIONS[key]
