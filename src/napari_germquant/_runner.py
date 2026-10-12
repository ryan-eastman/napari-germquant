"""Where germquant runs: the engine runners of the Run dock (docs/DESIGN_system_packs.md, section 8.1). No Qt.

The plugin's environment and the environment that makes the numbers can differ. On the lab workstation the
plugin's venv has the CPU torch wheel and no SpotMAX, while the engine's .venv has CUDA torch and SpotMAX, and
the two install germquant from different checkouts. An EngineRunner therefore says where everything that
touches packs and runs goes:

* SubprocessRunner: an engine interpreter chosen for this computer (the dock stores the choice in QSettings).
  Pack listing, checks and fetching go through the engine's torch-free entry, ``<python> -m germquant.packs``
  (``list --json``, ``check --json``, ``fetch-models``, ``fetch-example`` and ``--version --json``), so a pack
  card refresh never loads torch in the engine process. A run is ``<python> -m germquant run IMAGE ... --out
  DIR``: the command line itself, with the arguments the dock exports. Every command is a list of arguments; no
  shell is ever involved.
* InProcessRunner: this napari environment, through germquant.packs and germquant.api (in the dock's thread
  worker). The dock uses it only when the user picks it explicitly; with no engine chosen it asks instead.

Every engine call of the subprocess runner can be stopped: the dock passes ``should_stop`` (its Cancel button,
or napari quitting, which asks every thread worker to quit) and the call then ends the engine process and every
process it started (a job object or ``taskkill /T`` on Windows, the process group elsewhere) and raises
RunCancelled. On Windows each engine process also sits in a job that ends with napari, so killing napari never
leaves an engine writing into an output folder. Version checks, listings and checks have their own short
timeout; runs, model and example downloads have none (they are cancelled instead).

Each GUI run also writes a run log, ``runs/<run_id>.jsonl`` in the folder that holds the output folder (section
7.6): run_start (via, host, os, interpreter, engine version and git sha, python, device requested and selected,
config_hash, images), then image_done or image_failed, then run_end, one fsynced JSON line each. The log never
goes into the output folder itself, so a GUI run and ``germquant run`` write the same file list there. When that
folder cannot be written, or would be the output folder itself, the log goes to the per-user folder
(``user_runs_dir``) and run_start says so.
"""
from __future__ import annotations

# germquant.api first (through _engine): the torch-before-CuPy import order of the workstation
from . import _engine  # noqa: F401,I001

import contextlib
import datetime as _dt
import functools
import hashlib
import importlib.util
import json
import logging
import os
import platform
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from germquant import api

log = logging.getLogger("napari_germquant.engine")

VIA_SUBPROCESS = "gui-subprocess"
VIA_INPROCESS = "gui-inprocess"
MIN_ENGINE = _engine.MIN_ENGINE
DEVICES = ("auto", "cuda", "mps", "cpu")
DEVICE_ENV = "GERMQUANT_DEVICE"
PACKS_MODULE = "germquant.packs"
# no console window flashes up for each engine call when napari runs without a console (Windows)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# how long a version check, a pack listing or a pack check may take (runs and downloads have no limit)
CHECK_TIMEOUT_S = 120.0
# how often a running engine call asks should_stop
_POLL_S = 0.1

# The device probe a subprocess run starts with (run_start records the device requested and selected). It
# imports torch in the engine environment, never CuPy, so the cuBLAS load order stays as the engine wants it.
_RUNTIME_PROBE = (
    "import json, sys\n"
    "from germquant.device import runtime_record\n"
    "prefer = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None\n"
    "print(json.dumps(runtime_record(prefer)))\n"
)

# A chosen executable whose name does not start with "python" is asked to print this first (python -I -c).
_PYTHON_PROBE_MARK = "germquant-python-probe"
# commands that sit next to python.exe in an environment and are easily picked by mistake: refused unrun
_NOT_PYTHON = ("germquant", "napari", "pip", "pip3")
_CHOOSE_PYTHON = ("Choose python.exe (Windows) or bin/python (Mac, Linux) of the engine environment, not "
                  "germquant.exe or napari.exe.")


class RunnerError(RuntimeError):
    """An engine call that could not be made or did not succeed; the message says what to do."""


class EngineTooOld(RunnerError):
    """The engine has no system packs (germquant older than 0.2.0.dev0)."""


class RunCancelled(RunnerError):
    """The call was stopped (Cancel, or napari quitting): the engine process and its children were ended."""


def _not_stopped(should_stop) -> None:
    if should_stop is not None and should_stop():
        raise RunCancelled("cancelled by the user before it started")


# ---------------------------------------------------------------------------------------------------
# versions: the plugin's own germquant against the engine's
# ---------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def plugin_engine_info() -> dict:
    """The germquant this napari environment imports: version, git sha, python and interpreter (what
    ``python -m germquant.packs --version --json`` prints for an engine)."""
    import germquant

    try:
        from germquant.packs import version_info     # germquant 0.2.0.dev0 and later
    except ImportError:
        from germquant.provenance import git_sha

        return {"germquant": str(germquant.__version__), "git_sha": git_sha(), "python": platform.python_version(),
                "executable": sys.executable}
    return dict(version_info())


def engine_too_old(version: str) -> bool:
    """Whether an engine version lies below MIN_ENGINE (PEP 440 order: 0.2.0.dev0 sorts before 0.2)."""
    try:
        from packaging.version import Version

        return Version(str(version)) < Version(MIN_ENGINE)
    except Exception:  # noqa: BLE001 - an unparsable version is not ours to judge; the call itself will tell
        return False


def version_mismatch(own: dict, engine: dict) -> list[str]:
    """The differences between the plugin's germquant and an engine's (version, git sha), in words."""
    out = []
    if str(own.get("germquant")) != str(engine.get("germquant")):
        out.append(f"the engine is germquant {engine.get('germquant')}, the plugin's own germquant is "
                   f"{own.get('germquant')}")
    if str(own.get("git_sha")) != str(engine.get("git_sha")):
        out.append(f"the engine's git sha is {engine.get('git_sha')}, the plugin's is {own.get('git_sha')}")
    return out


def mismatch_warning(own: dict, engine: dict) -> str:
    """One warning for the dock when the engine and the plugin's germquant differ, else an empty string."""
    diffs = version_mismatch(own, engine)
    if not diffs:
        return ""
    return ("Warning: " + "; ".join(diffs) + ". The channel roles and stage list the dock shows come from the "
            "plugin's copy, the numbers from the engine's. Point both environments at the same engine checkout, "
            "or choose an engine interpreter that matches.")


# ---------------------------------------------------------------------------------------------------
# interpreters
# ---------------------------------------------------------------------------------------------------


def engine_checkout() -> Path | None:
    """The engine checkout the plugin's own germquant is imported from (the folder holding src/germquant)."""
    import germquant

    pkg = Path(germquant.__file__).resolve().parent
    if pkg.parent.name == "src" and (pkg.parent.parent / "pyproject.toml").is_file():
        return pkg.parent.parent
    return None


def venv_python(folder: str | Path) -> Path | None:
    """``<folder>/.venv/Scripts/python.exe`` (Windows) or ``<folder>/.venv/bin/python``, when present."""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        p = Path(folder) / rel
        if p.is_file():
            return p
    return None


def own_environment_gaps() -> tuple[list[str], list[str]]:
    """``(what this napari environment lacks, what therefore cannot run here)``, from find_spec and the torch
    already imported (germquant.api imports it); nothing is imported to find out."""
    gaps, effects = [], []
    if importlib.util.find_spec("spotmax") is None:
        gaps.append("no SpotMAX")
        effects.append("outside packs and the spots stage cannot run here")
    torch = sys.modules.get("torch")
    if torch is None and importlib.util.find_spec("torch") is None:
        gaps.append("no torch")
        effects.append("Cellpose recipes cannot run here")
    elif torch is not None and sys.platform != "darwin" and getattr(getattr(torch, "version", None), "cuda", None) is None:
        gaps.append("CPU torch")
        effects.append("Cellpose runs on the CPU only")
    if importlib.util.find_spec("cellpose") is None and "no torch" not in gaps:
        gaps.append("no Cellpose")
        effects.append("Cellpose recipes cannot run here")
    return gaps, effects


def detect_interpreters() -> list[dict]:
    """Suggestions for the engine interpreter, never a decision: the .venv of the engine checkout the plugin's
    germquant comes from (when it has one, ``detected``: the dialog preselects only that one), then this napari
    environment's own Python as a separate process, labelled with what it lacks. Each is ``{"python", "why",
    "detected"}``."""
    out: list[dict] = []
    checkout = engine_checkout()
    if checkout is not None:
        p = venv_python(checkout)
        if p is not None:
            out.append({"python": str(p), "why": f"the .venv of the engine checkout {checkout}", "detected": True})
    why = "this napari environment's own Python, as a separate process"
    gaps, effects = own_environment_gaps()
    if gaps:
        why += f" ({', '.join(gaps)}): " + "; ".join(dict.fromkeys(effects))
    out.append({"python": sys.executable, "why": why, "detected": False})
    seen, unique = set(), []
    for c in out:
        key = os.path.normcase(str(Path(c["python"]).resolve()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------


def _jsonable(obj):
    """The value as the subprocess route would deliver it (JSON round trip: paths become strings)."""
    return json.loads(json.dumps(obj, default=str))


def _parse_json(text: str):
    """The JSON document a ``--json`` command printed (tolerating stray lines before it)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("{", "[")):
            try:
                return json.loads("\n".join(lines[i:]))
            except ValueError:
                continue
    raise RunnerError(f"the engine printed no JSON: {_tail(text, 600)}")


def _tail(text: str, n: int = 1500) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


@contextlib.contextmanager
def device_env(device: str | None):
    """``GERMQUANT_DEVICE`` set to ``device`` for the duration (nothing for auto or None), then restored. The
    variable is process-wide: an in-process run shares it with every other engine call in napari."""
    if not device or device == "auto":
        yield
        return
    old = os.environ.get(DEVICE_ENV)
    os.environ[DEVICE_ENV] = device
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(DEVICE_ENV, None)
        else:
            os.environ[DEVICE_ENV] = old


def read_result(plan) -> dict:
    """The result record of a finished run read back from its output folder (what api.run returns, without
    the tables): image_id, n_nuclei, qc_pass, qc_flags and out_dir."""
    import pandas as pd

    image_id = api.image_stem(plan.image)
    out = Path(plan.out)
    summary = out / f"{image_id}__image_summary.csv"
    if not summary.is_file():
        raise RunnerError(f"the run finished but wrote no {summary.name} in {out}")
    row = pd.read_csv(summary, low_memory=False).iloc[0]
    flags = row.get("qc_flags", "")
    flags = [] if flags is None or (isinstance(flags, float) and flags != flags) else \
        [f for f in str(flags).split(";") if f.strip()]
    qc = row.get("qc_pass", False)
    qc_pass = qc if isinstance(qc, bool) else str(qc).strip().lower() == "true"
    return {"image_id": image_id, "n_nuclei": int(row.get("n_nuclei", 0) or 0), "qc_pass": bool(qc_pass),
            "qc_flags": flags, "out_dir": str(out)}


# ---------------------------------------------------------------------------------------------------
# engine processes: started, read line by line, and ended with everything they started
# ---------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _win32():
    """kernel32 with the job-object calls typed, and the JOBOBJECT_EXTENDED_LIMIT_INFORMATION structure."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                   "OtherOperationCount", "ReadTransferCount",
                                                   "WriteTransferCount", "OtherTransferCount")]

    class _Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", _Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.TerminateJobObject.restype = wintypes.BOOL
    k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return ctypes, k32, _Extended


def _windows_job(proc: subprocess.Popen):
    """A job object holding ``proc`` and every process it starts, set to end them all when its last handle
    closes: when napari ends, even killed from Task Manager, the engine ends with it. None outside Windows, or
    when no job can be made (the process then runs without one; Cancel still ends its tree)."""
    if os.name != "nt":
        return None
    try:
        ctypes, k32, extended = _win32()
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = extended()
        info.BasicLimitInformation.LimitFlags = 0x2000           # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # extended limits
        if not ok or not k32.AssignProcessToJobObject(job, int(proc._handle)):
            k32.CloseHandle(job)
            return None
        return job
    except Exception:  # noqa: BLE001 - a job is a safety net, never a reason not to run
        return None


def _close_job(job) -> None:
    """Close the job handle: whatever is still in it (an orphan the engine left behind) is ended."""
    if job is not None:
        with contextlib.suppress(Exception):
            _win32()[1].CloseHandle(job)


def _end_tree(proc: subprocess.Popen, job) -> None:
    """End ``proc`` and every process it started: the job and ``taskkill /T`` on Windows (a venv's python.exe
    is a launcher whose child is the real interpreter), the process group elsewhere (the child was started in a
    session of its own), then wait for it."""
    if os.name == "nt":
        if job is not None:
            with contextlib.suppress(Exception):
                _win32()[1].TerminateJobObject(job, 1)
        if proc.poll() is None:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                               stdin=subprocess.DEVNULL, timeout=60, creationflags=_NO_WINDOW)
    else:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=30)


def _read_lines(stream, name: str, sink: queue.Queue) -> None:
    try:
        for raw in stream:
            sink.put((name, raw.rstrip("\r\n")))
    except (OSError, ValueError):                     # the stream was closed under us: nothing more to read
        pass


# ---------------------------------------------------------------------------------------------------
# the runners
# ---------------------------------------------------------------------------------------------------


class EngineRunner:
    """Where the dock's pack and run calls go. ``device`` is the device asked for (auto, cuda, mps or cpu).
    ``should_stop`` (every call takes it) is asked while the call runs; a runner that can stop answers by
    raising RunCancelled (``cancellable``)."""

    kind = "engine"
    via = ""
    python = ""
    cancellable = False

    def __init__(self) -> None:
        self.device: str = "auto"
        self.version_error: str | None = None

    @property
    def key(self) -> str:
        """Identifies the environment (the check cache of the dock is keyed on it)."""
        return f"{self.kind}:{self.python}"

    def describe(self) -> str:
        raise NotImplementedError

    def version(self, refresh: bool = False, should_stop=None) -> dict:
        raise NotImplementedError

    def cached_version(self) -> dict | None:
        """The version check's answer when it has run (never starts one: the GUI thread reads this)."""
        return None

    def list_packs(self, should_stop=None) -> dict:
        raise NotImplementedError

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None,
              should_stop=None) -> dict:
        raise NotImplementedError

    def fetch_models(self, pack: str, recipe: str | None = None, say=None, should_stop=None) -> dict:
        raise NotImplementedError

    def fetch_example(self, pack: str, example: str | None = None, out: str | Path | None = None,
                      say=None, should_stop=None) -> dict:
        raise NotImplementedError

    def runtime(self, prefer: str | None = None, should_stop=None) -> dict:
        raise NotImplementedError

    def run(self, plan, say=None, should_stop=None) -> dict:
        raise NotImplementedError


class InProcessRunner(EngineRunner):
    """This napari environment: germquant.packs for listing, checks and fetching, ``RunPlan.run``
    (germquant.api.run) for runs. Chosen only explicitly. A call that has started cannot be stopped."""

    kind = "inprocess"
    via = VIA_INPROCESS

    def __init__(self) -> None:
        super().__init__()
        self.python = sys.executable

    def describe(self) -> str:
        return f"this napari environment, in process ({self.python})"

    def version(self, refresh: bool = False, should_stop=None) -> dict:
        return dict(plugin_engine_info())

    def cached_version(self) -> dict | None:
        return dict(plugin_engine_info())

    def _packs(self):
        if not _engine.packs_supported():
            raise EngineTooOld(_engine.packs_unavailable_message())
        from germquant import packs

        return packs

    def list_packs(self, should_stop=None) -> dict:
        _not_stopped(should_stop)
        P = self._packs()
        P.refresh()                                   # a pack installed or fetched since is seen now
        disc = P.discover()
        msg = P.no_builtin_root_message()
        return _jsonable({"version": P.version_info(), "builtin_root_found": msg is None, "message": msg,
                          "roots": [{"kind": k, "path": str(r)} for k, r in disc.roots],
                          "packs": P.list_packs(all=True)})

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None,
              should_stop=None) -> dict:
        _not_stopped(should_stop)
        P = self._packs()
        try:
            st = P.pack_status(pack, recipe=recipe, image=str(image) if image else None)
        except FileNotFoundError as e:                # an unknown pack is unusable, as `pack check` says
            st = {"pack": pack, "recipe": recipe, "usable": False,
                  "blockers": [{"what": str(e), "fix": "germquant pack list"}], "warnings": []}
        return _jsonable(st)

    def fetch_models(self, pack: str, recipe: str | None = None, say=None, should_stop=None) -> dict:
        _not_stopped(should_stop)
        P = self._packs()
        try:
            res = P.fetch_pack_models(pack, recipe, download=True, say=say)
        except (P.PackFormatError, FileNotFoundError, ValueError) as e:
            raise RunnerError(str(e)) from e
        return {"ok": not any(r["status"] == "missing" for r in res), "models": _jsonable(res),
                "lines": [f"{r['name']}: {r['status']} -> {r['path']}" for r in res]}

    def fetch_example(self, pack: str, example: str | None = None, out: str | Path | None = None,
                      say=None, should_stop=None) -> dict:
        _not_stopped(should_stop)
        P = self._packs()
        try:
            return _jsonable(P.fetch_example(P.find_pack(pack), example, out, say=say))
        except FileNotFoundError as e:
            raise RunnerError(str(e)) from e
        except Exception as e:  # noqa: BLE001 - ExampleError, a download or assembly error: one message
            raise RunnerError(f"{type(e).__name__}: {e}") from e

    def runtime(self, prefer: str | None = None, should_stop=None) -> dict:
        try:
            from germquant.device import runtime_record
        except ImportError:                           # an engine older than the runtime record
            from germquant.device import select_device

            req = (os.environ.get(DEVICE_ENV) or prefer or "auto").strip().lower()
            return {"device_requested": req, "device_selected": select_device(prefer)}
        with device_env(self.device):
            return _jsonable(runtime_record(prefer))

    def run(self, plan, say=None, should_stop=None) -> dict:
        _not_stopped(should_stop)
        if plan.recipe.is_pack:                       # as `germquant run --pack` does before it runs
            st = self.check(plan.recipe.pack, plan.recipe.recipe, plan.image)
            if not st.get("usable"):
                raise RunnerError(refusal_text(plan, st))
            for w in st.get("warnings") or []:
                log.warning("warning: %s (%s)", w.get("what"), w.get("fix"))
        with device_env(self.device):
            return plan.run()


class SubprocessRunner(EngineRunner):
    """An engine interpreter in its own process. ``env`` overrides variables of the inherited environment
    (None removes one); ``interpreter_options`` go between the interpreter and ``-m`` (for example
    ``("-X", "importtime")`` in a test); ``timeout`` bounds a version check, a listing and a check (runs and
    downloads are unbounded and cancelled instead). ``history`` keeps every argument list run, newest last."""

    kind = "subprocess"
    via = VIA_SUBPROCESS
    cancellable = True

    def __init__(self, python: str | Path, *, env: dict | None = None, interpreter_options=(),
                 timeout: float = CHECK_TIMEOUT_S) -> None:
        super().__init__()
        self.python = str(python)
        self.env = dict(env or {})
        self.interpreter_options = tuple(str(o) for o in interpreter_options)
        self.timeout = float(timeout)
        self.history: list[list[str]] = []
        self.last_stderr = ""
        self._version: dict | None = None

    def describe(self) -> str:
        return f"{self.python} (a separate process)"

    # ---- plumbing ----
    def child_env(self, extra: dict | None = None) -> dict:
        env = dict(os.environ)
        for k, v in {**self.env, **(extra or {})}.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)
        env["PYTHONIOENCODING"] = "utf-8"     # the engine prints check marks; read them as UTF-8
        env["PYTHONUNBUFFERED"] = "1"         # run lines reach the status bar as they come
        env["PYTHONSAFEPATH"] = "1"           # Python 3.11+: never import from the working directory
        return env

    def _device_env(self) -> dict:
        return {} if self.device in (None, "", "auto") else {DEVICE_ENV: self.device}

    def packs_argv(self, *args) -> list[str]:
        return [self.python, *self.interpreter_options, "-m", PACKS_MODULE, *[str(a) for a in args]]

    def _run_proc(self, argv: list[str], *, extra_env: dict | None = None, say=None, should_stop=None,
                  timeout: float | None = None, merge: bool = False) -> tuple[int, list[str], list[str]]:
        """Run ``argv``: ``(returncode, stdout lines, stderr lines)``. With ``merge`` stderr joins stdout. Each
        stdout line goes to ``say`` as it arrives. ``should_stop`` is asked every 0.1 s: when it answers True,
        the process and every process it started are ended and RunCancelled is raised; past ``timeout``
        seconds they are ended and RunnerError says so."""
        _not_stopped(should_stop)
        self.history.append(list(argv))
        options: dict = {"creationflags": _NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT if merge else subprocess.PIPE,
                                    stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                    env=self.child_env(extra_env), bufsize=1, **options)
        except OSError as e:
            raise RunnerError(f"the engine interpreter {self.python} cannot be started: {e}") from e
        job = _windows_job(proc)
        lines: queue.Queue = queue.Queue()
        readers = [threading.Thread(target=_read_lines, args=(stream, name, lines), daemon=True)
                   for name, stream in (("out", proc.stdout), ("err", proc.stderr)) if stream is not None]
        for t in readers:
            t.start()
        out: list[str] = []
        err: list[str] = []

        def take(wait: float) -> bool:
            try:
                name, line = lines.get(timeout=wait) if wait > 0 else lines.get_nowait()
            except queue.Empty:
                return False
            (out if name == "out" else err).append(line)
            if say is not None and name == "out" and line.strip():
                say(line)
            return True

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        ended = ""
        try:
            while proc.poll() is None:
                if should_stop is not None and should_stop():
                    ended = "cancelled"
                    break
                if deadline is not None and time.monotonic() > deadline:
                    ended = "timeout"
                    break
                take(_POLL_S)
            if ended:
                _end_tree(proc, job)
            _close_job(job)                           # anything the engine left running ends here
            job = None
            until = time.monotonic() + 10             # the readers get the rest of the output (EOF), 10 s at most
            for t in readers:
                t.join(timeout=max(0.0, until - time.monotonic()))
            while take(0):
                pass
        finally:
            if proc.poll() is None:                   # an error in say: never leave the engine behind
                _end_tree(proc, job)
            _close_job(job)
            # A stream whose reader still waits (a process that escaped the tree kill keeps the pipe open) is left
            # to that daemon thread: closing it would wait for the blocked read, and so would napari's quit.
            for t, stream in zip(readers, [s for s in (proc.stdout, proc.stderr) if s is not None]):
                if not t.is_alive():
                    with contextlib.suppress(OSError, ValueError):
                        stream.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
        self.last_stderr = "\n".join(out if merge else err)
        if ended == "cancelled":
            raise RunCancelled(f"cancelled by the user: {self.python} and every process it started were ended")
        if ended == "timeout":
            raise RunnerError(f"the engine interpreter {self.python} did not answer within {int(timeout)} s "
                              f"(it was ended)")
        return int(proc.returncode), out, err

    def _call(self, argv: list[str], *, timeout: float | None = None, extra_env: dict | None = None,
              should_stop=None) -> subprocess.CompletedProcess:
        rc, out, err = self._run_proc(argv, extra_env=extra_env, should_stop=should_stop,
                                      timeout=timeout or self.timeout)
        return subprocess.CompletedProcess(argv, rc, "\n".join(out), "\n".join(err))

    def _stream(self, argv: list[str], *, say=None, extra_env: dict | None = None,
                should_stop=None) -> tuple[int, list[str]]:
        """Run ``argv`` with stdout and stderr merged, handing each line to ``say`` as it arrives."""
        rc, out, _err = self._run_proc(argv, extra_env=extra_env, say=say, should_stop=should_stop, merge=True)
        return rc, out

    def _module_missing(self, text: str) -> None:
        if "No module named germquant.packs" in text or "No module named 'germquant.packs'" in text:
            raise EngineTooOld(
                f"The engine at {self.python} has a germquant without system packs (older than {MIN_ENGINE}). "
                f"Update that engine checkout (git pull) and reinstall it into its environment, or choose "
                f"another engine interpreter.")
        if "No module named germquant" in text or "No module named 'germquant'" in text:
            raise RunnerError(f"{self.python} has no germquant installed: choose the Python of the environment "
                              f"germquant was installed into (for example the engine checkout's .venv)")

    def _json(self, argv: list[str], *, ok=(0,), timeout: float | None = None, should_stop=None):
        proc = self._call(argv, timeout=timeout, should_stop=should_stop)
        if proc.returncode not in ok:
            text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            self._module_missing(text)
            raise RunnerError(f"`{' '.join(argv[1:])}` failed (exit {proc.returncode}): {_tail(text)}")
        return _parse_json(proc.stdout)

    def _check_interpreter(self, should_stop=None) -> None:
        """Refuse what is not a Python interpreter before asking it anything: a folder, an environment's
        germquant or napari command, or (for any other name not starting with python) an executable that does
        not answer ``-I -c print(...)`` as Python does."""
        p = Path(self.python)
        if p.is_dir():
            raise RunnerError(f"{self.python} is a folder, not a Python interpreter. {_CHOOSE_PYTHON}")
        stem = p.name.lower().removesuffix(".exe")
        if stem in _NOT_PYTHON:
            raise RunnerError(f"{self.python} is not a Python interpreter but the {stem} command of an "
                              f"environment. {_CHOOSE_PYTHON}")
        if stem.startswith("python"):
            return
        rc, out, err = self._run_proc([self.python, "-I", "-c", f"print({_PYTHON_PROBE_MARK!r})"], timeout=30,
                                      should_stop=should_stop)
        if _PYTHON_PROBE_MARK not in out:
            answer = _tail("\n".join(out + err), 300) or f"nothing, exit code {rc}"
            raise RunnerError(f"{self.python} is not a Python interpreter (it answered: {answer}). {_CHOOSE_PYTHON}")

    # ---- the engine calls ----
    def version(self, refresh: bool = False, should_stop=None) -> dict:
        """``<python> -m germquant.packs --version --json``; EngineTooOld for an engine without packs."""
        if self._version is not None and not refresh:
            return dict(self._version)
        try:
            self._check_interpreter(should_stop)
            info = self._json(self.packs_argv("--version", "--json"), should_stop=should_stop)
            if not isinstance(info, dict) or "germquant" not in info:
                raise RunnerError(f"{self.python} answered --version with {info!r}")
            if engine_too_old(info["germquant"]):
                raise EngineTooOld(f"The engine at {self.python} is germquant {info['germquant']}, older than "
                                   f"{MIN_ENGINE}, which system packs and this plugin's runner need. Update that "
                                   f"engine checkout and reinstall it, or choose another engine interpreter.")
        except RunCancelled:
            raise
        except RunnerError as e:
            self.version_error = str(e)
            raise
        self.version_error = None
        self._version = dict(info)
        return dict(info)

    def cached_version(self) -> dict | None:
        return dict(self._version) if self._version is not None else None

    def list_packs(self, should_stop=None) -> dict:
        return self._json(self.packs_argv("list", "--json", "--all"), should_stop=should_stop)

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None,
              should_stop=None) -> dict:
        args = ["check", pack]
        if recipe:
            args += ["--recipe", recipe]
        if image:
            args += ["--image", str(image)]
        # exit 2: unusable, still a JSON report
        return self._json(self.packs_argv(*args, "--json"), ok=(0, 2), should_stop=should_stop)

    def fetch_models(self, pack: str, recipe: str | None = None, say=None, should_stop=None) -> dict:
        args = ["fetch-models", pack] + (["--recipe", recipe] if recipe else [])
        rc, lines = self._stream(self.packs_argv(*args), say=say, should_stop=should_stop)
        if rc not in (0, 1):
            text = "\n".join(lines)
            self._module_missing(text)
            raise RunnerError(f"fetch-models failed (exit {rc}): {_tail(text)}")
        models = []
        for line in lines:                            # "  <name>: <status> -> <path>"
            s = line.strip()
            if ": " in s and " -> " in s and not s.startswith(("tried:", "ERROR")):
                name, rest = s.split(": ", 1)
                status, path = rest.split(" -> ", 1)
                models.append({"name": name, "status": status.strip(), "path": path.strip()})
        return {"ok": rc == 0, "models": models, "lines": lines}

    def fetch_example(self, pack: str, example: str | None = None, out: str | Path | None = None,
                      say=None, should_stop=None) -> dict:
        """``fetch-example`` with its progress lines streamed to ``say`` (its --json form prints nothing until
        the end), then the record from ``fetch-example --no-download --json``, which only reads what the
        first call left in place; the record's status is the first call's (fetched, present or local)."""
        args = ["fetch-example", pack]
        if example:
            args += ["--example", example]
        if out:
            args += ["--out", str(out)]
        rc, lines = self._stream(self.packs_argv(*args), say=say, should_stop=should_stop)
        if rc != 0:
            text = "\n".join(lines)
            self._module_missing(text)
            raise RunnerError(f"fetch-example failed (exit {rc}): {_tail(text)}")
        status = None
        for line in lines:                            # "<id>@<version> example <example> (<status>): <image>"
            m = re.match(r"^\S+ example \S+ \((\w+)\): ", line.strip())
            if m:
                status = m.group(1)
        proc = self._call(self.packs_argv(*args, "--no-download", "--json"), should_stop=should_stop)
        if proc.returncode != 0:
            raise RunnerError(f"fetch-example --no-download --json failed (exit {proc.returncode}): "
                              f"{_tail((proc.stderr or '') + chr(10) + (proc.stdout or ''))}")
        rec = _parse_json(proc.stdout)
        if status and isinstance(rec, dict):
            rec["status"] = status
        return rec

    def runtime(self, prefer: str | None = None, should_stop=None) -> dict:
        argv = [self.python, *self.interpreter_options, "-c", _RUNTIME_PROBE, prefer or ""]
        proc = self._call(argv, timeout=300, extra_env=self._device_env(), should_stop=should_stop)
        if proc.returncode != 0:
            raise RunnerError(f"the device probe failed in {self.python} (exit {proc.returncode}): "
                              f"{_tail(proc.stderr)}")
        return _parse_json(proc.stdout)

    def run_argv(self, plan) -> list[str]:
        """``<python> -m germquant`` and the plan's ``germquant run`` arguments."""
        return [self.python, *self.interpreter_options, "-m", "germquant", *plan.run_argv()]

    def run(self, plan, say=None, should_stop=None) -> dict:
        rc, lines = self._stream(self.run_argv(plan), say=say, extra_env=self._device_env(),
                                 should_stop=should_stop)
        if rc != 0:
            text = "\n".join(lines)
            self._module_missing(text)
            raise RunnerError(f"germquant run exited with code {rc} in {self.python}:\n{_tail(text, 2500)}")
        return read_result(plan)


def refusal_text(plan, status: dict) -> str:
    """What the engine's command line prints when it refuses an unusable pack, as one message."""
    lines = [f"REFUSED: pack {plan.recipe.pack} recipe {plan.recipe.recipe} is not usable here:"]
    for b in status.get("blockers") or []:
        lines.append(f"  {b.get('what')}\n      fix: {b.get('fix')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# run logs (section 7.6)
# ---------------------------------------------------------------------------------------------------


def new_run_id() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def user_runs_dir() -> Path:
    """The per-user folder for run logs that cannot sit next to their output folder."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "napari-germquant" / "runs"


def runs_folder_for(out_dir: str | Path) -> tuple[Path, str]:
    """``(folder, where)``: ``<parent of the output folder>/runs`` (the place a project keeps its runs/, next to
    its results), else the per-user folder: when that cannot be written, and when it would be the output folder
    itself (an output folder named runs, or a drive or share root, whose parent is itself), because nothing but
    what ``germquant run`` writes may go into the output folder."""
    out = Path(out_dir).absolute()
    beside = out.parent / "runs"
    if out.parent == out or os.path.normcase(str(beside)) == os.path.normcase(str(out)):
        fallback = user_runs_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, ("the per-user folder (the output folder is itself named runs, or is a drive or share "
                          "root, so a runs folder beside it would be the output folder)")
    try:
        beside.mkdir(parents=True, exist_ok=True)
        probe = beside / f".write_test_{uuid.uuid4().hex[:8]}"
        probe.write_bytes(b"")
        probe.unlink()
        return beside, "beside the output folder"
    except OSError:
        fallback = user_runs_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, f"the per-user folder ({beside} could not be written)"


class RunLog:
    """One run's log, ``<folder>/<run_id>.jsonl``: one JSON object per event, flushed and fsynced at once, so a
    crash never leaves half a line and no other process ever appends to this file."""

    def __init__(self, folder: str | Path, run_id: str) -> None:
        self.run_id = run_id
        self.path = Path(folder) / f"{run_id}.jsonl"

    def event(self, kind: str, **fields) -> dict:
        rec = {"event": kind, "run_id": self.run_id,
               "time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"), **fields}
        line = json.dumps(_jsonable(rec), sort_keys=False) + "\n"
        with open(self.path, "ab") as fh:
            fh.write(line.encode("ascii"))
            fh.flush()
            os.fsync(fh.fileno())
        return rec

    def events(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(ln) for ln in self.path.read_text(encoding="ascii").splitlines() if ln.strip()]


def read_run_log(path: str | Path) -> list[dict]:
    return RunLog(Path(path).parent, Path(path).stem).events()


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def execute_run(runner: EngineRunner, plan, *, config_hash: str | None = None, prefer: str | None = None,
                interpreter_source: str = "default", say=None, log_dir: str | Path | None = None,
                should_stop=None) -> dict:
    """Run ``plan`` through ``runner`` and write its run log: run_start before the run, image_done (or
    image_failed, then the error is raised again; a cancelled run says so in its error) and run_end after it.
    ``config_hash`` is the hash napari's own germquant computed; run_start records it only when that germquant
    is the engine's (same version and git sha), and says where it came from (config_hash_source); image_done
    always carries the run's own, read from its completion marker. Returns the run's result record with
    ``run_id`` and ``run_log`` added."""
    import germquant

    run_id = new_run_id()
    if log_dir is not None:
        folder, where = Path(log_dir), "given"
        folder.mkdir(parents=True, exist_ok=True)
    else:
        folder, where = runs_folder_for(plan.out)
    rlog = RunLog(folder, run_id)
    version = runner.version(should_stop=should_stop)
    try:
        rt = runner.runtime(prefer, should_stop=should_stop)
    except RunCancelled:
        rt = {"device_requested": runner.device, "device_selected": None, "probe_error": "cancelled"}
    except RunnerError as e:                          # the run itself will say what is wrong; record the gap
        rt = {"device_requested": runner.device, "device_selected": None, "probe_error": str(e)}
    diffs = version_mismatch(plugin_engine_info(), version)
    if config_hash is None:
        hash_source = "none: the recipe does not load in napari's environment (image_done carries the run's own)"
    elif diffs:
        config_hash = None
        hash_source = ("none: napari's germquant is not the engine's (" + "; ".join(diffs) + "), so its hash "
                       "could differ (image_done carries the run's own)")
    else:
        hash_source = "napari's germquant, the same version and git sha as the engine"
    image_id = api.image_stem(plan.image)
    rlog.event("run_start", via=runner.via, host=socket.gethostname(), os=platform.platform(),
               interpreter=runner.python, interpreter_source=interpreter_source,
               engine={"germquant": version.get("germquant"), "git_sha": version.get("git_sha")},
               python=version.get("python"), constraints=None,
               device_requested=rt.get("device_requested"), device_selected=rt.get("device_selected"),
               config_hash=config_hash, config_hash_source=hash_source, recipe=plan.recipe.describe(),
               xy_stride=int(plan.xy_stride), disabled_stages=list(plan.disabled), argv=list(plan.run_argv()),
               images=[{"path": str(plan.image), "image_id": image_id, "out_dir": str(plan.out)}],
               plugin={"napari_germquant": _plugin_version(), "germquant": str(germquant.__version__)},
               log_location=where)
    t0 = time.monotonic()
    try:
        res = runner.run(plan, say=say, should_stop=should_stop)
    except BaseException as e:
        rlog.event("image_failed", image_id=image_id, out_dir=str(plan.out), status="failed",
                   elapsed_s=round(time.monotonic() - t0, 3), error=f"{type(e).__name__}: {e}"[:4000])
        rlog.event("run_end", counts={"done": 0, "failed": 1, "skipped": 0})
        try:
            e.run_log = str(rlog.path)        # the dock names the log of a failed run too
        except Exception:  # noqa: BLE001 - an exception type that takes no attributes
            pass
        raise
    out = Path(res.get("out_dir") or plan.out)
    marker_path = out / f"{res.get('image_id', image_id)}__done.json"
    marker = api.read_done_marker(out, res.get("image_id", image_id)) or {}
    stages_doc: dict = {}
    try:
        stages_doc = json.loads((out / f"{res.get('image_id', image_id)}__stages.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    post = stages_doc.get("post_run") if isinstance(stages_doc, dict) else None
    rlog.event("image_done", image_id=res.get("image_id", image_id), out_dir=str(out), status="done",
               done_sha256=_sha256_file(marker_path), config_hash=marker.get("config_hash"),
               stage_hashes=marker.get("stage_hashes"), failed_stages=marker.get("failed_stages"),
               qc_pass=res.get("qc_pass"),
               segment_device_used=(post or {}).get("segment_device_used") if isinstance(post, dict) else None,
               elapsed_s=round(time.monotonic() - t0, 3), error=None)
    rlog.event("run_end", counts={"done": 1, "failed": 0, "skipped": 0})
    res = dict(res)
    res.update({"run_id": run_id, "run_log": str(rlog.path), "device_selected": rt.get("device_selected"),
                "device_requested": rt.get("device_requested"), "interpreter": runner.python, "via": runner.via})
    return res


def _plugin_version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"
