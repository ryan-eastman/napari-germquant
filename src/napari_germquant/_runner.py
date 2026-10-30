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

Projects, training and tuning (tranche 5 Train B; the commands of section 9 of the train's plan). The runners
have one method per engine command, each taking ``should_stop`` and, where the command prints as it works, ``say``
for the streamed lines. Through the subprocess runner:

* ``project new|add|edit|use-model|rounds|unlock`` go through the torch-free ``<python> -m germquant.project``,
  post-stage listing and writing through ``-m germquant.post_stages``, ``pack derive`` and the pack calls with
  ``project=`` (``--project DIR``) through ``-m germquant.packs``; every other command is ``engine_argv``,
  ``<python> -m germquant ...``, which imports torch first. Every project write carries ``--plugin-version``.
* ``project run`` passes the dock's own ``--run-id`` and ``--via gui-subprocess`` and tails the engine's log,
  ``<project>/runs/<run_id>.jsonl``, from the polling loop that watches the engine (``LogTail``: complete lines
  only, a partial last line kept for the next read), handing new events to ``progress``. The plugin writes no
  log of its own for project runs. Exit codes: 0 every image done or skipped, 1 some images failed (a result,
  not an error of the call), 2 refused before any image, with the reason and fix the engine printed
  (``state_for_exit``, ``run_outcome``).
* A command exits 2 when the engine refuses it: EngineRefused, with what it printed (the reason is the engine's
  REFUSED: or ERROR: lines, or the error or problems of its JSON record). ``pack derive`` refuses with exit 1
  and a REFUSED line, which is EngineRefused too.

The in-process runner calls the same commands through germquant.api (and germquant.segment.training for the
training commands), passing the project explicitly (napari's thread workers start with no pack context). The
six project writes (new, add, edit, use-model, rounds, unlock) and ``pack derive`` are the exception: they run
the torch-free entry of this napari environment's own Python as a child process, the same germquant, so each
write is the command's own (an edit is one planned write under the project lock, with the command's refusals
and record), not a second implementation of it here.

The engine version string cannot tell a Train B engine (it stays 0.2.0.dev0), so the runners gate on the
``features`` list of ``python -m germquant.packs --version --json`` (an engine from before the list has packs and
nothing else); in process the project API names also decide. A call that needs a feature the engine lacks
raises FeatureMissing with one message (``_engine.feature_missing_message``) before any command starts.
"""
from __future__ import annotations

# germquant.api first (through _engine): the torch-before-CuPy import order of the workstation
from . import _engine  # noqa: F401,I001

import contextlib
import datetime as _dt
import functools
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import numbers
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
# the torch-free project entry (new, add, edit, use-model, rounds, unlock)
PROJECT_MODULE = "germquant.project"
POST_STAGES_MODULE = "germquant.post_stages"   # the torch-free post-stage entry (list, write)
FEATURES = _engine.FEATURES
# in process, a feature whose API function this germquant has is there, whatever its version document says
_API_FEATURES = {"projects": "open_project", "import_labels": "import_labels", "tune_spots": "tune_spots",
                 "post_stages": "list_post_stages"}
# run ids: the plugin's own generator (new_run_id), which `project run --run-id` checks the same way
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{8}$")
# training round names (training/<round>/)
ROUND_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
IMPORT_FORMATS = ("tiff", "png", "cellacdc-npz", "cellpose-npy")
# a project run's exit code -> the dock's state (anything else is an error)
EXIT_STATES = {0: "done", 1: "failed", 2: "refused"}
# how often a project run's log is read while the engine works
TAIL_S = 0.5
# no console window flashes up for each engine call when napari runs without a console (Windows)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# how long a version check, a pack listing or a pack check may take (runs and downloads have no limit)
CHECK_TIMEOUT_S = 120.0
# how often a running engine call asks should_stop
_POLL_S = 0.1

# The device probe a subprocess run starts with (run_start records the device requested and selected). It
# imports torch in the engine environment, never CuPy, so the cuBLAS load order stays as the engine wants it.
# ``available`` is germquant.device.available(), the accelerators that environment's torch sees (the device
# list of the Training dock).
_RUNTIME_PROBE = (
    "import json, sys\n"
    "from germquant.device import runtime_record\n"
    "prefer = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None\n"
    "rec = runtime_record(prefer)\n"
    "try:\n"
    "    from germquant.device import available\n"
    "    rec['available'] = available()\n"
    "except ImportError:\n"
    "    pass\n"
    "print(json.dumps(rec))\n"
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
    """The call was stopped (Cancel, or napari quitting): the engine process and its children were ended. A
    cancelled project run carries ``run_id``, ``run_log`` and the ``events`` read from its log."""


class FeatureMissing(RunnerError):
    """The engine lacks a tranche-5 Train B feature the call needs (``feature``); the message is the one the
    docks show (``_engine.feature_missing_message``). Nothing was started."""

    def __init__(self, message: str, feature: str) -> None:
        super().__init__(message)
        self.feature = feature


class EngineRefused(RunnerError):
    """The engine refused a command before doing anything (exit code 2): the message holds what it printed,
    the reason and the fix (``output`` is that text, ``exit_code`` the code)."""

    def __init__(self, message: str, *, output: str = "", exit_code: int = 2) -> None:
        super().__init__(message)
        self.output = output
        self.exit_code = exit_code


def _not_stopped(should_stop) -> None:
    if should_stop is not None and should_stop():
        raise RunCancelled("cancelled by the user before it started")


def plugin_version() -> str:
    """This plugin's version: what every project write records (``--plugin-version``, design 7.2)."""
    return _plugin_version()


def features_of(info: dict | None) -> frozenset[str]:
    """The features an engine's version document names (``features``). An engine from before the list answers
    ``--version`` from its packs entry all the same, so it has packs and nothing else."""
    if not info:
        return frozenset()
    feats = info.get("features")
    if isinstance(feats, (list, tuple)):
        return frozenset(str(f) for f in feats)
    return frozenset({"packs"})


def device_choices(runtime: dict | None) -> list[str]:
    """The devices a dock offers for an engine: auto, then the accelerators its torch sees (the runtime probe's
    ``available``), then cpu."""
    have = (runtime or {}).get("available") or {}
    return ["auto", *[d for d in ("cuda", "mps") if have.get(d)], "cpu"]


def check_run_id(run_id: str) -> str:
    """``run_id`` when it has the run-id form (``<UTC yyyymmddThhmmssZ>_<8 hex>``), else ValueError."""
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"not a run id: {run_id!r} (the form is 20261023T120000Z_0123abcd; new_run_id makes one)")
    return run_id


def set_value_text(value) -> str:
    """``VALUE`` of ``project edit --set KEY=VALUE``, written so that YAML and JSON read it back with its type: a
    number, true or false, null, a plain string, and any string another reading could take for something else
    (1.0, yes, a colon, spaces at an end, non-ASCII text) in double quotes with ASCII escapes."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError(f"a --set value must be a finite scalar, not {value!r}")
        import yaml

        return yaml.safe_dump(f).splitlines()[0]          # always with a point or an exponent YAML reads as float
    if isinstance(value, str):
        import yaml

        try:
            plain = value != "" and value.isascii() and value == value.strip() and yaml.safe_load(value) == value
        except yaml.YAMLError:
            plain = False
        return value if plain else json.dumps(value, ensure_ascii=True)
    raise ValueError(f"a --set value must be a scalar (number, bool, string or null), not {value!r}")


def project_log_path(root: str | Path, run_id: str) -> Path:
    """Where the engine writes a project run's log: ``<project>/runs/<run_id>.jsonl``."""
    return Path(root) / "runs" / f"{run_id}.jsonl"


class LogTail:
    """Reads a run log while the engine writes it: each ``read`` returns the events of the lines completed since
    the last one. A partial last line (the engine writes one fsynced line per event, but a reader can arrive
    between its bytes) is kept for the next read. A missing file reads as no events yet; a complete line that is
    not a JSON object is set aside in ``bad``. The file is opened and closed on every read, never held."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.bad: list[str] = []
        self._partial = b""

    def read(self) -> list[dict]:
        from germquant.fsutil import long_path

        try:
            with open(long_path(self.path), "rb") as fh:
                fh.seek(self.offset)
                data = fh.read()
        except OSError:                                # not there yet, or a share that is gone for a moment
            return []
        if not data:
            return []
        self.offset += len(data)
        lines = (self._partial + data).split(b"\n")
        self._partial = lines.pop()                    # b"" after a complete last line
        out = []
        for raw in lines:
            text = raw.decode("ascii", errors="replace").strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except ValueError:
                self.bad.append(text[:200])
                continue
            if isinstance(rec, dict):
                out.append(rec)
            else:
                self.bad.append(text[:200])
        return out


def state_for_exit(code) -> str | None:
    """A project run's exit code as the dock's state: 0 done (every image done or skipped), 1 failed (some
    images failed), 2 refused (before any image); None for any other code (an error)."""
    return EXIT_STATES.get(code) if isinstance(code, int) and not isinstance(code, bool) else None


def run_outcome(events: list[dict], *, alive: bool = True) -> dict:
    """What a run log says, for the status board: ``state`` (running while there is no run_end and the engine
    is ``alive``, interrupted once it is gone without one or when run_end says so, failed when an image failed,
    else done), ``counts`` (run_end's, or tallied from the image events), ``images`` ({image_id: status} from
    the image events, the last one winning), ``started`` and ``ended`` (their times), ``finished`` (whether
    there is a run_end) and ``interrupted``."""
    start = next((e for e in events if e.get("event") == "run_start"), None)
    end = next((e for e in reversed(events) if e.get("event") == "run_end"), None)
    images: dict[str, str] = {}
    for e in events:
        kind = str(e.get("event") or "")
        if kind in ("image_done", "image_failed", "image_skipped"):
            images[str(e.get("image_id"))] = str(e.get("status") or kind.removeprefix("image_"))
    tally = {"done": 0, "failed": 0, "skipped": 0}
    for status in images.values():
        tally[status] = tally.get(status, 0) + 1
    counts = dict(end.get("counts") or tally) if end is not None else tally
    interrupted = bool(end.get("interrupted")) if end is not None else not alive
    if end is None:
        state = "running" if alive else "interrupted"
    elif interrupted:
        state = "interrupted"
    elif counts.get("failed"):
        state = "failed"
    else:
        state = "done"
    return {"run_id": (start or end or {}).get("run_id"), "state": state, "counts": counts, "images": images,
            "started": (start or {}).get("time"), "ended": (end or {}).get("time"), "finished": end is not None,
            "interrupted": interrupted}


def project_run_result(run_id: str, log_path: str | Path, events: list[dict], exit_code: int,
                       output: list[str]) -> dict:
    """The record a project run returns: ``run_id``, ``log``, ``exit_code``, ``state`` (state_for_exit),
    ``counts``, ``images``, ``interrupted``, ``message`` (the end of what the engine printed, for a refusal or a
    failure) and every ``events`` of its log. RunnerError, with the run and the events attached, for a crash
    (exit 1 before run_end) and for any exit code other than 0, 1 and 2."""
    outcome = run_outcome(events, alive=False)
    text = "\n".join(output)
    state = state_for_exit(exit_code)
    ended = outcome["finished"]
    if state is None or (exit_code == 1 and not ended):
        why = "before the end of the run" if exit_code == 1 else "(not 0, 1 or 2)"
        err = RunnerError(f"germquant project run exited with code {exit_code} {why}:\n{_tail(text, 2500)}")
        err.run_id, err.run_log, err.events = run_id, str(log_path), list(events)
        raise err
    if state in ("done", "failed") and ended and outcome["interrupted"]:
        state = "interrupted"                         # the engine stopped between images and said so
    return {"run_id": run_id, "log": str(log_path), "exit_code": int(exit_code), "state": state,
            "counts": outcome["counts"], "images": outcome["images"], "interrupted": outcome["interrupted"],
            "message": _tail(text, 2500) if state in ("refused", "failed") else "", "events": list(events)}


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

    def list_packs(self, should_stop=None, project=None) -> dict:
        """The pack listing; with ``project`` (a project folder) its packs/ is searched first (I1)."""
        raise NotImplementedError

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None,
              should_stop=None, project=None) -> dict:
        raise NotImplementedError

    def fetch_models(self, pack: str, recipe: str | None = None, say=None, should_stop=None,
                     project=None) -> dict:
        raise NotImplementedError

    def fetch_example(self, pack: str, example: str | None = None, out: str | Path | None = None,
                      say=None, should_stop=None) -> dict:
        raise NotImplementedError

    def runtime(self, prefer: str | None = None, should_stop=None) -> dict:
        raise NotImplementedError

    def run(self, plan, say=None, should_stop=None) -> dict:
        raise NotImplementedError

    # ---- the features gate (tranche 5 Train B) ----
    def features(self, should_stop=None) -> frozenset[str]:
        """The tranche-5 Train B features this engine has (its version document's ``features``)."""
        return features_of(self.version(should_stop=should_stop))

    def feature_message(self, feature: str) -> str:
        """The one message this engine gets for a feature it lacks."""
        raise NotImplementedError

    def require(self, *features: str, should_stop=None) -> None:
        """FeatureMissing for the first of ``features`` this engine lacks (nothing is started then)."""
        have = self.features(should_stop=should_stop)
        for f in features:
            if f not in have:
                raise FeatureMissing(self.feature_message(f), f)

    # ---- the commands of tranche 5 Train B (the subclasses implement them) ----
    def project_new(self, root, pack: str, *, recipe: str | None = None, images=(), name: str | None = None,
                    plugin_version: str | None = None, should_stop=None) -> dict:
        """``project new``: the command's record, the project's summary plus ``added`` and ``refused`` (``root``
        always present). ``images``: one path, folder or glob, or a list of them."""
        raise NotImplementedError

    def project_add(self, root, paths, *, plugin_version: str | None = None, should_stop=None) -> dict:
        """``project add``: ``{"added": [image_id], "refused": [{"path", "reason"}], "images", "headers"}``.
        ``paths``: one path, folder or glob, or a list of them."""
        raise NotImplementedError

    def project_edit(self, root, *, edits: dict | None = None, unset=(), roles: dict | None = None,
                     revert: bool = False, device: str | None = None, xy_stride: int | None = None,
                     z_range=None, runner: str | None = None, interpreters: dict | None = None,
                     samples=None, disable=(), dry_run: bool = False, plugin_version: str | None = None,
                     should_stop=None) -> dict:
        """``project edit``: ``config_hash`` (the new one; with ``dry_run`` the one an edit would give) and, for a
        dry run, the preview. ``edits`` (``--set``) maps dotted keys to scalars, ``roles`` roles to channel names,
        ``disable`` names stages switched off (``--no-<stage>``), ``interpreters`` maps a recipe to a Python."""
        raise NotImplementedError

    def project_use_model(self, root, model, *, card=None, plugin_version: str | None = None,
                          should_stop=None) -> dict:
        """``project use-model``: ``config_hash``."""
        raise NotImplementedError

    def project_rounds(self, root, *, new=None, plugin_version: str | None = None, should_stop=None) -> dict:
        """``project rounds``: ``rounds`` (names), and with ``new`` (True for the next of r01, r02 and so on, or a
        name) ``created`` (the new round's name) and ``path`` (its folder)."""
        raise NotImplementedError

    def project_unlock(self, root, *, host: str, pid: int, should_stop=None) -> dict:
        """``project unlock``: ``removed`` (whether a lock of that host and pid was there and is gone; False when
        the project was not locked). A lock someone else holds is refused: EngineRefused with the holder."""
        raise NotImplementedError

    def project_run(self, root, *, images=None, resume: bool = True, device: str | None = None,
                    shard: str | None = None, run_id: str | None = None, say=None, progress=None,
                    should_stop=None) -> dict:
        """``project run``: project_run_result's record. ``progress(events)`` gets the run log's new events as
        the engine writes them, in the calling thread (the dock's worker: it hands them on to the GUI thread)."""
        raise NotImplementedError

    def project_status(self, root, *, should_stop=None) -> dict:
        """``project status``: the status record (equal to api.project_status)."""
        raise NotImplementedError

    def project_collect(self, root, *, should_stop=None) -> dict:
        raise NotImplementedError

    def project_export_commands(self, root, *, shell: str = "sh", shards: int | None = None, out=None,
                                should_stop=None) -> dict:
        """``project export-commands``: ``text`` (the commands) and ``out`` (the file written, or None)."""
        raise NotImplementedError

    def import_labels(self, src, fmt: str, image, pairs, *, allow_pickle: bool = False, frame: int | None = None,
                      channel=None, should_stop=None) -> dict:
        """``import-labels``: the engine's record (plan 9.4 #11). ``channel``: the image channel the labels were
        drawn on, an index or a name (``--channel``), passed only when set."""
        raise NotImplementedError

    def export_training(self, results, out, *, orientations=None, slices_per_image: int | None = None,
                        seed: int | None = None, zone_column: str | None = None, min_labels: int | None = None,
                        max_gb: float | None = None, say=None, should_stop=None) -> dict:
        """``export-training``: ``exit_code`` and ``lines``. ``orientations``: "auto", a list, or None (the
        engine's default)."""
        raise NotImplementedError

    def train_nuclei(self, pairs, out, *, base: str = "cpsam", name: str | None = None, epochs: int | None = None,
                     split_by: str | None = None, val_frac: float | None = None, seed: int | None = None,
                     device: str | None = None, system: str | None = None, kind: str | None = None,
                     dataset: str | None = None, say=None, should_stop=None) -> dict:
        """``train-nuclei``: ``exit_code``, ``lines``, ``model_path`` and ``card_path``. ``device`` (None: the
        runner's device): cpu is ``--cpu``; cuda or mps go through GERMQUANT_DEVICE (the engine falls back when
        it is missing); auto leaves the choice to the engine."""
        raise NotImplementedError

    def train_curve(self, pairs, out, *, sizes, repeats: int, seed: int, base: str, keep_models: str | None = None,
                    device: str | None = None, say=None, should_stop=None) -> dict:
        """``train-curve``: its JSON record. ``device`` (None: the runner's device) goes through
        GERMQUANT_DEVICE for every model of the curve (cpu included: the command has no --cpu)."""
        raise NotImplementedError

    def pack_derive(self, from_ref: str, recipe: str, model, card, out, *, new_id: str | None = None,
                    should_stop=None) -> dict:
        raise NotImplementedError

    def tune_spots(self, project, *, points=None, grid=None, max_runs: int | None = None, out=None, say=None,
                   should_stop=None) -> dict:
        """``tune-spots``: ``{"run_dir", "best", "ranked", "warnings"}``, also when the search ran but nothing
        could be scored (the command's exit 1: ``best`` is None and the warnings say why)."""
        raise NotImplementedError

    def tune_apply(self, project, run_dir, *, plugin_version: str | None = None, should_stop=None) -> dict:
        """``tune-spots --apply``: writes recipe.yaml; ``config_hash`` when the engine says it."""
        raise NotImplementedError

    def list_post_stages(self, *, should_stop=None) -> list:
        raise NotImplementedError

    def write_post_stages(self, root, entries: list, *, should_stop=None) -> dict:
        raise NotImplementedError


def _with_root(doc, root) -> dict:
    """A project record with its ``root`` (absolute) always present."""
    out = dict(doc) if isinstance(doc, dict) else {"result": doc}
    out.setdefault("root", str(Path(root)))
    return out


def _disabled_flags(disable) -> list[str]:
    """``--no-<stage>`` for each stage name, refusing a stage that has no such switch."""
    from germquant.stages import BY_NAME

    names = [str(n) for n in disable or ()]
    unknown = [n for n in names if n not in BY_NAME or not BY_NAME[n].cli_switchable]
    if unknown:
        raise ValueError(f"not switchable stages: {unknown}")
    return [f"--no-{n}" for n in names]


def _check_round(new) -> None:
    if isinstance(new, str) and new and not ROUND_RE.fullmatch(new):
        raise ValueError(f"not a round name: {new!r} (letters, digits, dash and underscore, at most 40)")


# the lines in which an engine command says why it refused (any case): REFUSED: (the project commands, pack
# derive, train-curve), ERROR: (tune-spots), refused: (post_stages write)
_REASON_PREFIXES = ("refused:", "error:")


def _refusal_reason(lines: list[str], stdout: str | None = None) -> str:
    """The reason a refused command gave: its REFUSED: or ERROR: lines (any case), else the ``error`` (or the
    ``problems``) of the JSON record it printed: the whole of ``stdout`` (an indented record over many lines)
    or a one-line record among ``lines``. An empty string when it gave none."""
    said = [ln.strip() for ln in lines if ln.strip().lower().startswith(_REASON_PREFIXES)]
    if said:
        return "\n".join(dict.fromkeys(said))
    docs = []
    if stdout and stdout.strip():
        with contextlib.suppress(RunnerError):
            docs.append(_parse_json(stdout))
    for ln in lines:
        s = ln.strip()
        if s.startswith("{") and ('"error"' in s or '"problems"' in s):
            with contextlib.suppress(ValueError):
                docs.append(json.loads(s))
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("error"):
            return f"REFUSED: {doc['error']}"
        problems = doc.get("problems")
        if isinstance(problems, list) and problems:
            return "\n".join(f"REFUSED: {p}" for p in problems)
    return ""


def _path_list(paths) -> list:
    """``paths`` as a list: one path (a str or os.PathLike) is a list of one, never a string split into letters."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths)


def _write_atomic(path, data: bytes) -> Path:
    """Write ``data`` to ``path`` through a temporary file next to it and os.replace, both through
    germquant.fsutil.long_path (past the 260-character limit on Windows): a reader never sees half a file, and
    no handle is open during the replace."""
    from germquant.fsutil import long_path

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        with open(long_path(tmp), "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(long_path(tmp), long_path(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(long_path(tmp))
        raise
    return path


def _command_name(argv: list[str]) -> str:
    """The command an argument list runs, for a message: the module and its command words ("germquant project
    status", "germquant.project new"), or the first two arguments after a wrapper's code."""
    if "-m" in argv:
        i = argv.index("-m")
        words = [str(a) for a in argv[i + 1:i + 4]]
        keep = 3 if words[:1] == ["germquant"] and words[1:2] in (["project"], ["pack"]) else 2
        return " ".join(words[:keep])
    return " ".join(str(a) for a in argv[3:5])


def _refusal_types() -> tuple:
    """The errors the project command line turns into a refusal (exit 2)."""
    types_: list = [ValueError, OSError]
    try:
        from germquant.packs import PackInstallError

        types_.append(PackInstallError)
    except ImportError:
        pass
    return tuple(types_)


def _check_edit(revert, edits, unset, roles) -> None:
    if revert and (edits or unset or roles):
        raise ValueError("a revert cannot be combined with edits, unset keys or roles: it drops every recipe edit")


def _join_list(value) -> str:
    return value if isinstance(value, str) else ",".join(str(v) for v in value)


class InProcessRunner(EngineRunner):
    """This napari environment: germquant.packs for listing, checks and fetching, ``RunPlan.run``
    (germquant.api.run) for runs. Chosen only explicitly. A call that has started cannot be stopped, except
    the project writes and pack derive, which run this environment's torch-free entries as a child process
    (``_child``) and end it on Cancel."""

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

    def _project_packs(self, project):
        """The context in which pack lookups search ``project``'s packs/ first (packs.use_project, set in this
        thread: napari's thread workers start without one), or nothing for no project."""
        if project is None:
            return contextlib.nullcontext()
        self.require("projects")
        return self._packs().use_project(Path(project))

    def list_packs(self, should_stop=None, project=None) -> dict:
        _not_stopped(should_stop)
        P = self._packs()
        with self._project_packs(project):
            P.refresh()                               # a pack installed or fetched since is seen now
            disc = P.discover(project) if project is not None else P.discover()
            msg = P.no_builtin_root_message()
            return _jsonable({"version": P.version_info(), "builtin_root_found": msg is None, "message": msg,
                              "roots": [{"kind": k, "path": str(r)} for k, r in disc.roots],
                              "packs": P.list_packs(all=True, project=project) if project is not None
                              else P.list_packs(all=True)})

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None,
              should_stop=None, project=None) -> dict:
        _not_stopped(should_stop)
        P = self._packs()
        with self._project_packs(project):
            try:
                st = P.pack_status(pack, recipe=recipe, image=str(image) if image else None)
            except FileNotFoundError as e:            # an unknown pack is unusable, as `pack check` says
                st = {"pack": pack, "recipe": recipe, "usable": False,
                      "blockers": [{"what": str(e), "fix": "germquant pack list"}], "warnings": []}
        return _jsonable(st)

    def fetch_models(self, pack: str, recipe: str | None = None, say=None, should_stop=None, project=None) -> dict:
        _not_stopped(should_stop)
        P = self._packs()
        with self._project_packs(project):
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
            rec = _jsonable(runtime_record(prefer))
        try:
            from germquant.device import available
        except ImportError:
            return rec
        rec["available"] = _jsonable(available())
        return rec

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

    # ---- tranche 5 Train B: the features and the commands, through germquant.api ----
    def features(self, should_stop=None) -> frozenset[str]:
        """The version document's list, with the API names deciding the features they serve (projects:
        ``api.open_project``, and import_labels, tune_spots and post_stages likewise)."""
        if not _engine.packs_supported():
            return frozenset()
        have = set(features_of(plugin_engine_info()))
        for feature, name in _API_FEATURES.items():
            if hasattr(api, name):
                have.add(feature)
            else:
                have.discard(feature)
        return frozenset(have)

    def feature_message(self, feature: str) -> str:
        return _engine.feature_missing_message(feature, info=plugin_engine_info())

    def _api(self, name: str):
        fn = getattr(api, name, None)
        if fn is None:
            raise RunnerError(f"this napari environment's germquant has no api.{name}: update that engine "
                              f"checkout, or choose an engine interpreter")
        return fn

    def _open(self, root):
        """api.open_project(root); a folder that is no project (or cannot be read) is EngineRefused, as the
        commands refuse it with exit 2."""
        return self._api_call(self._api("open_project"), root)

    def _child(self) -> SubprocessRunner:
        """This napari environment's own Python as an engine process: the same germquant, through its torch-free
        entries (``-m germquant.project``, ``-m germquant.packs``). The project writes and pack derive go there,
        so each is the command's own code: an edit is planned whole and written once under the project lock,
        with the command's refusals and record (section 9 names no single call for an edit, and a second
        implementation of the command's merge here could drift from it). Its version document is this
        germquant's, with the features this runner has (the API names decide), so no version check runs."""
        child = SubprocessRunner(sys.executable)
        child._version = {**plugin_engine_info(), "features": sorted(self.features())}
        return child

    def _api_call(self, func, *args, refusals=None, **kwargs):
        """``func(*args, **kwargs)``; a refusal becomes EngineRefused, any other error a RunnerError naming the
        call. ``refusals``: what the engine's command line turns into exit 2 (by default as the project commands
        do: ValueError, which ProjectError is, OSError and PackInstallError); ``()`` for none."""
        if refusals is None:
            refusals = _refusal_types()
        try:
            return func(*args, **kwargs)
        except RunnerError:
            raise
        except refusals as e:
            raise EngineRefused(f"REFUSED: {e}", output=str(e)) from e
        except Exception as e:  # noqa: BLE001 - one message for the dock, the cause chained
            raise RunnerError(f"{getattr(func, '__name__', 'the call')} failed: {type(e).__name__}: {e}") from e

    def _start(self, should_stop, *features: str) -> None:
        _not_stopped(should_stop)
        self.require(*features)

    # -- the six project writes: the torch-free command of this environment, in a child process (``_child``)
    def project_new(self, root, pack, *, recipe=None, images=(), name=None, plugin_version=None,
                    should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return self._child().project_new(root, pack, recipe=recipe, images=images, name=name,
                                         plugin_version=plugin_version, should_stop=should_stop)

    def project_add(self, root, paths, *, plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return self._child().project_add(root, paths, plugin_version=plugin_version, should_stop=should_stop)

    def project_edit(self, root, *, edits=None, unset=(), roles=None, revert=False, device=None, xy_stride=None,
                     z_range=None, runner=None, interpreters=None, samples=None, disable=(), dry_run=False,
                     plugin_version=None, should_stop=None) -> dict:
        """``project edit`` itself (``_child``): the command plans the whole change (recipe keys, roles, the run
        settings, the stage switches), checks all of it, then writes it once under the project lock, so a
        refusal leaves the project as it was; a dry run gives the preview and the config_hash an edit would
        give."""
        self._start(should_stop, "projects", *(("sample_sheet",) if samples else ()))
        return self._child().project_edit(root, edits=edits, unset=unset, roles=roles, revert=revert,
                                          device=device, xy_stride=xy_stride, z_range=z_range, runner=runner,
                                          interpreters=interpreters, samples=samples, disable=disable,
                                          dry_run=dry_run, plugin_version=plugin_version, should_stop=should_stop)

    def project_use_model(self, root, model, *, card=None, plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return self._child().project_use_model(root, model, card=card, plugin_version=plugin_version,
                                               should_stop=should_stop)

    def project_rounds(self, root, *, new=None, plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return self._child().project_rounds(root, new=new, plugin_version=plugin_version, should_stop=should_stop)

    def project_unlock(self, root, *, host, pid, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return self._child().project_unlock(root, host=host, pid=pid, should_stop=should_stop)

    def project_run(self, root, *, images=None, resume=True, device=None, shard=None, run_id=None, say=None,
                    progress=None, should_stop=None) -> dict:
        """api.run_project with via gui-inprocess and the run id; its progress callback (whatever it passes)
        reads the engine's log, so ``progress`` gets the same events as through the subprocess runner. As the
        command answers: a refusal before the run started (no run_start in its log: a folder that is no project,
        a model digest mismatch, a series it cannot run) is state refused, as exit 2; any other error is a
        RunnerError with the run, its log and its events attached. The run stops between images when
        ``should_stop`` answers True, then RunCancelled, unless the run had finished every image."""
        self._start(should_stop, "projects")
        run_id = check_run_id(run_id or new_run_id())
        tail = LogTail(project_log_path(root, run_id))
        events: list[dict] = []

        def tick(*_a, **_k) -> None:
            new = tail.read()
            if new:
                events.extend(new)
                if progress is not None:
                    progress(new)

        def with_run(err: RunnerError) -> RunnerError:
            err.run_id, err.run_log, err.events = run_id, str(tail.path), events
            return err

        run_project, open_project = self._api("run_project"), self._api("open_project")
        try:
            res = run_project(open_project(root), images=list(images) if images else None, resume=bool(resume),
                              device=device or self.device, shard=shard, progress=tick, via=self.via,
                              run_id=run_id, should_stop=should_stop)
        except RunnerError:
            raise
        except _refusal_types() as e:
            tick()
            if not any(ev.get("event") == "run_start" for ev in events):
                return project_run_result(run_id, tail.path, events, 2, [f"REFUSED: {e}"])
            raise with_run(RunnerError(f"run_project failed: {type(e).__name__}: {e}")) from e
        except Exception as e:  # noqa: BLE001 - one message for the dock, the cause chained
            tick()
            raise with_run(RunnerError(f"run_project failed: {type(e).__name__}: {e}")) from e
        tick()
        res = res if isinstance(res, dict) else {}
        if should_stop is not None and should_stop():
            board = run_outcome(events, alive=False)
            if not board["finished"] or board["interrupted"]:      # a run that finished keeps its result
                raise with_run(RunCancelled("cancelled by the user: the in-process run stopped between images"))
        message = [str(res["message"])] if res.get("message") else []
        return project_run_result(run_id, res.get("log") or tail.path, events, int(res.get("exit_code", 0)),
                                  message)

    def project_status(self, root, *, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return _jsonable(self._api_call(self._api("project_status"), self._open(root)))

    def project_collect(self, root, *, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return _jsonable(self._api_call(self._api("collect_project"), self._open(root)))

    def project_export_commands(self, root, *, shell="sh", shards=None, out=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        text = self._api_call(self._api("export_commands"), self._open(root), shell=shell, shards=shards)
        if out is not None:
            _write_atomic(out, str(text).encode("utf-8"))
        return {"text": str(text), "out": str(out) if out is not None else None}

    def import_labels(self, src, fmt, image, pairs, *, allow_pickle=False, frame=None, channel=None,
                      should_stop=None) -> dict:
        self._start(should_stop, "import_labels")
        if fmt not in IMPORT_FORMATS:
            raise ValueError(f"not an import format: {fmt!r} (one of {', '.join(IMPORT_FORMATS)})")
        kw = {"allow_pickle": bool(allow_pickle)}
        if frame is not None:
            kw["frame"] = int(frame)
        if channel is not None:
            kw["channel"] = channel
        return _jsonable(self._api_call(self._api("import_labels"), src, fmt, image, pairs, **kw))

    def export_training(self, results, out, *, orientations=None, slices_per_image=None, seed=None,
                        zone_column=None, min_labels=None, max_gb=None, say=None, should_stop=None) -> dict:
        """segment.training.export_training_pairs with the arguments ``export-training`` hands it."""
        _not_stopped(should_stop)
        from germquant.segment import training as T

        kw: dict = {}
        if orientations is not None:
            kw["orientations"] = orientations if isinstance(orientations, str) else tuple(orientations)
        for key, val in (("slices_per_image", slices_per_image), ("seed", seed), ("min_labels", min_labels)):
            if val is not None:
                kw[key] = int(val)
        if zone_column is not None:
            kw["zone_column"] = zone_column or None
        if max_gb is not None:
            kw["max_bytes"] = float(max_gb) * 1e9
        manifest = self._api_call(T.export_training_pairs, results, out, refusals=(), **kw)
        line = f"{len(manifest) if manifest is not None else 0} slice pair(s) -> {out}"
        if say is not None:
            say(line)
        return {"exit_code": 0, "lines": [line]}

    def train_nuclei(self, pairs, out, *, base="cpsam", name=None, epochs=None, split_by=None, val_frac=None,
                     seed=None, device=None, system=None, kind=None, dataset=None, say=None,
                     should_stop=None) -> dict:
        """segment.training.train_nuclei as ``train-nuclei`` calls it (``device``, else the runner's: cuda or mps
        through GERMQUANT_DEVICE for the call, cpu as cpu=True). The card fields (system, kind, dataset) go only
        when set, under the names the function takes (kind is the card's object_kind); an engine without them
        says so."""
        _not_stopped(should_stop)
        import inspect

        from germquant.segment import training as T

        device = device or self.device
        kw: dict = {"base": base, "name": name, "cpu": device == "cpu"}
        for key, val in (("epochs", epochs), ("seed", seed)):
            if val is not None:
                kw[key] = int(val)
        if split_by is not None:
            kw["split_by"] = split_by
        if val_frac is not None:
            kw["val_frac"] = float(val_frac)
        params = inspect.signature(T.train_nuclei).parameters
        for names, val in ((("system",), system), (("object_kind", "kind"), kind), (("dataset",), dataset)):
            if val is None:
                continue
            key = next((n for n in names if n in params), None)
            if key is None:
                raise RunnerError(f"this germquant's train_nuclei takes no {names[-1]} (the card fields of the "
                                  f"tranche-5 Train B engine): update that engine checkout")
            kw[key] = val
        with device_env(None if device in (None, "cpu") else device):
            res = _jsonable(self._api_call(T.train_nuclei, pairs, out, refusals=(), **kw))
        return {"exit_code": 0, "lines": [], "model_path": res.get("model_path"),
                "card_path": res.get("card_path"), "result": res}

    def train_curve(self, pairs, out, *, sizes, repeats, seed, base, keep_models=None, device=None, say=None,
                    should_stop=None) -> dict:
        """segment.training.train_curve, every model of the curve on ``device`` (else the runner's) through
        GERMQUANT_DEVICE for the call."""
        _not_stopped(should_stop)
        self.require("train_curve")
        from germquant.segment import training as T

        fn = getattr(T, "train_curve", None)
        if fn is None:
            raise FeatureMissing(self.feature_message("train_curve"), "train_curve")
        kw = {"sizes": list(sizes) if not isinstance(sizes, str) else sizes, "repeats": int(repeats),
              "seed": int(seed), "base": base}
        if keep_models is not None:
            kw["keep_models"] = keep_models
        with device_env(device or self.device):
            return _jsonable(self._api_call(fn, pairs, out, **kw))

    def pack_derive(self, from_ref, recipe, model, card, out, *, new_id=None, should_stop=None) -> dict:
        """The API names no derive function, so this runs the torch-free ``-m germquant.packs derive`` of this
        environment (``_child``): the same germquant and packs, in a process of its own."""
        self._start(should_stop, "pack_derive")
        return self._child().pack_derive(from_ref, recipe, model, card, out, new_id=new_id, should_stop=should_stop)

    def tune_spots(self, project, *, points=None, grid=None, max_runs=None, out=None, say=None,
                   should_stop=None) -> dict:
        self._start(should_stop, "tune_spots")
        res = self._api_call(self._api("tune_spots"), self._open(project), points=str(points or "annotations"),
                             grid=str(grid) if grid is not None else None,
                             out=str(out) if out is not None else None,
                             max_runs=int(max_runs) if max_runs is not None else 50)
        return _jsonable(res)

    def tune_apply(self, project, run_dir, *, plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "tune_spots")
        h = self._api_call(self._api("apply_tuning"), self._open(project), str(run_dir),
                           plugin_version=plugin_version or _plugin_version())
        return {"config_hash": h}

    def list_post_stages(self, *, should_stop=None) -> list:
        self._start(should_stop, "post_stages")
        return _jsonable(self._api_call(self._api("list_post_stages")))

    def write_post_stages(self, root, entries, *, should_stop=None) -> dict:
        """post_stages.write_post_stages_file into <project>/post_stages.yaml, validated against the post-stages
        installed in this environment."""
        self._start(should_stop, "post_stages")
        try:
            post = importlib.import_module(POST_STAGES_MODULE)
        except ImportError as e:
            raise FeatureMissing(self.feature_message("post_stages"), "post_stages") from e
        path = Path(root) / "post_stages.yaml"
        written = self._api_call(post.write_post_stages_file, path, list(entries))
        return {"path": str(written or path), "entries": _jsonable(list(entries))}

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

    def module_argv(self, module: str, *args) -> list[str]:
        """``<python> [interpreter options] -m MODULE ARGS``."""
        return [self.python, *self.interpreter_options, "-m", module, *[str(a) for a in args]]

    def packs_argv(self, *args) -> list[str]:
        """The torch-free pack entry, ``-m germquant.packs``."""
        return self.module_argv(PACKS_MODULE, *args)

    def project_argv(self, *args) -> list[str]:
        """The torch-free project entry, ``-m germquant.project`` (new, add, edit, use-model, rounds, unlock)."""
        return self.module_argv(PROJECT_MODULE, *args)

    def post_stages_argv(self, *args) -> list[str]:
        """The torch-free post-stage entry, ``-m germquant.post_stages`` (list, write)."""
        return self.module_argv(POST_STAGES_MODULE, *args)

    def engine_argv(self, *args) -> list[str]:
        """The full command line, ``-m germquant ARGS`` (it imports torch first). A test runner may start it
        through a wrapper instead (the seams of a child process)."""
        return self.module_argv("germquant", *args)

    def _run_proc(self, argv: list[str], *, extra_env: dict | None = None, say=None, should_stop=None,
                  timeout: float | None = None, merge: bool = False, tick=None,
                  say_streams: tuple[str, ...] = ("out",)) -> tuple[int, list[str], list[str]]:
        """Run ``argv``: ``(returncode, stdout lines, stderr lines)``. With ``merge`` stderr joins stdout. Each
        line of ``say_streams`` ("out", "err") goes to ``say`` as it arrives. ``should_stop`` is asked every
        0.1 s: when it answers True, the process and every process it started are ended and RunCancelled is
        raised; past ``timeout`` seconds they are ended and RunnerError says so. ``tick()`` is called every
        TAIL_S seconds while the process runs and once after it ended (a project run reads its log there)."""
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
            if say is not None and name in say_streams and line.strip():
                say(line)
            return True

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        next_tick = time.monotonic() + TAIL_S
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
                if tick is not None and time.monotonic() >= next_tick:
                    tick()
                    next_tick = time.monotonic() + TAIL_S
            if ended:
                _end_tree(proc, job)
            _close_job(job)                           # anything the engine left running ends here
            job = None
            until = time.monotonic() + 10             # the readers get the rest of the output (EOF), 10 s at most
            for t in readers:
                t.join(timeout=max(0.0, until - time.monotonic()))
            while take(0):
                pass
            if tick is not None:                      # what the engine wrote last, also before a Cancel
                tick()
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

    def _project_option(self, project, should_stop=None) -> list[str]:
        """``--project DIR`` (I1: discovery with the project's packs/ first), for an engine with projects."""
        if project is None:
            return []
        self.require("projects", should_stop=should_stop)
        return ["--project", str(project)]

    def list_packs(self, should_stop=None, project=None) -> dict:
        return self._json(self.packs_argv("list", "--json", "--all", *self._project_option(project, should_stop)),
                          should_stop=should_stop)

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None,
              should_stop=None, project=None) -> dict:
        args = ["check", pack]
        if recipe:
            args += ["--recipe", recipe]
        if image:
            args += ["--image", str(image)]
        args += self._project_option(project, should_stop)
        # exit 2: unusable, still a JSON report
        return self._json(self.packs_argv(*args, "--json"), ok=(0, 2), should_stop=should_stop)

    def fetch_models(self, pack: str, recipe: str | None = None, say=None, should_stop=None, project=None) -> dict:
        args = ["fetch-models", pack] + (["--recipe", recipe] if recipe else [])
        args += self._project_option(project, should_stop)
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
        return self.engine_argv(*plan.run_argv())

    def run(self, plan, say=None, should_stop=None) -> dict:
        rc, lines = self._stream(self.run_argv(plan), say=say, extra_env=self._device_env(),
                                 should_stop=should_stop)
        if rc != 0:
            text = "\n".join(lines)
            self._module_missing(text)
            raise RunnerError(f"germquant run exited with code {rc} in {self.python}:\n{_tail(text, 2500)}")
        return read_result(plan)

    # ---- tranche 5 Train B: the features gate and one method per command ----
    def feature_message(self, feature: str) -> str:
        return _engine.feature_missing_message(feature, info=self.cached_version(), interpreter=self.python)

    def _start(self, should_stop, *features: str) -> None:
        _not_stopped(should_stop)
        if features:
            self.require(*features, should_stop=should_stop)

    def _command_json(self, argv: list[str], *, say=None, should_stop=None, extra_env: dict | None = None,
                      ok=(0,), refusal_codes=()):
        """Run a ``--json`` command without a time limit (it is cancelled instead): its JSON document. Its
        stderr lines (the engine's log) go to ``say``. Exit 2 (or a code of ``refusal_codes`` with a reason):
        EngineRefused with what it printed; any other code not in ``ok``: RunnerError."""
        rc, out, err = self._run_proc(argv, say=say, say_streams=("err",), should_stop=should_stop,
                                      extra_env=extra_env)
        if rc in ok:
            try:
                return _parse_json("\n".join(out))
            except RunnerError:
                if rc == 0:
                    raise
        # a code in ok without a record is an error after all
        self._check_exit(argv, rc, out + err, stdout="\n".join(out), refusal_codes=refusal_codes)
        raise RunnerError(f"`{_command_name(argv)}` exited with code {rc} and no record")

    def _entry_missing(self, text: str) -> None:
        """A clear message when a torch-free Train B entry is not there (an engine part way through the train):
        Python's "No module named germquant.project" would otherwise read as no germquant at all."""
        for module in (PROJECT_MODULE, POST_STAGES_MODULE):
            if f"No module named {module}" in text or f"No module named '{module}'" in text:
                raise RunnerError(f"The engine at {self.python} has no {module} entry: its germquant predates that "
                                  f"part of {_engine.TRAIN_B_ENGINE}. Update that engine checkout (git pull) and "
                                  f"reinstall it into its environment, or choose another engine.")

    def _command_lines(self, argv: list[str], *, say=None, should_stop=None, extra_env: dict | None = None,
                       tick=None) -> tuple[int, list[str]]:
        """Run a command with stdout and stderr merged, each line to ``say``: ``(exit code, lines)``."""
        rc, lines, _err = self._run_proc(argv, say=say, should_stop=should_stop, extra_env=extra_env, merge=True,
                                         tick=tick)
        return rc, lines

    def _check_exit(self, argv: list[str], rc: int, lines: list[str], ok=(0,), *, stdout: str | None = None,
                    refusal_codes=()) -> None:
        """Nothing for a code in ``ok``. Exit 2 is the engine's refusal, EngineRefused with its reason (the
        REFUSED: or ERROR: lines, or the error or problems of its JSON record, ``stdout``); so is a code of
        ``refusal_codes`` when the engine gave a reason (pack derive refuses with exit 1). Anything else is a
        RunnerError with the end of what it printed."""
        if rc in ok:
            return
        text = "\n".join(lines)
        self._entry_missing(text)
        self._module_missing(text)
        what = _command_name(argv)
        reason = _refusal_reason(lines, stdout)
        if rc == 2 or (rc in refusal_codes and reason):
            raise EngineRefused(f"`{what}` was refused by the engine (exit {rc}):\n{reason or _tail(text, 2500)}",
                                output=text, exit_code=rc)
        raise RunnerError(f"`{what}` failed (exit {rc}) in {self.python}:\n{_tail(text, 2500)}")

    def _lines_result(self, argv, rc, lines) -> dict:
        self._check_exit(argv, rc, lines)
        return {"exit_code": rc, "lines": lines}

    # -- the torch-free project entry (every write stamps the plugin version, D42)
    def project_new(self, root, pack, *, recipe=None, images=(), name=None, plugin_version=None,
                    should_stop=None) -> dict:
        self._start(should_stop, "projects")
        args = ["new", root, "--pack", pack]
        if recipe:
            args += ["--recipe", recipe]
        images = _path_list(images)
        if images:
            args += ["--images", *images]
        if name:
            args += ["--name", name]
        args += ["--plugin-version", plugin_version or _plugin_version(), "--json"]
        # exit 1 with its record: some --images were refused, each named (the others were added)
        return _with_root(self._command_json(self.project_argv(*args), should_stop=should_stop, ok=(0, 1)), root)

    def project_add(self, root, paths, *, plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        args = ["add", root, *_path_list(paths), "--plugin-version", plugin_version or _plugin_version(), "--json"]
        # exit 1 with its record: some paths were refused, and the record says which (the others were added)
        return self._command_json(self.project_argv(*args), should_stop=should_stop, ok=(0, 1))

    def project_edit(self, root, *, edits=None, unset=(), roles=None, revert=False, device=None, xy_stride=None,
                     z_range=None, runner=None, interpreters=None, samples=None, disable=(), dry_run=False,
                     plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "projects", *(("sample_sheet",) if samples else ()))
        _check_edit(revert, edits, unset, roles)
        args: list = ["edit", root]
        for key, value in (edits or {}).items():
            args += ["--set", f"{key}={set_value_text(value)}"]
        for key in unset or ():
            args += ["--unset", key]
        for role, channel in (roles or {}).items():
            args += ["--role", f"{role}={channel}"]
        if revert:
            args.append("--revert")
        if device:
            args += ["--device", device]
        if xy_stride is not None:
            args += ["--xy-stride", int(xy_stride)]
        if isinstance(z_range, str):
            args += ["--z-range", z_range]             # the word none clears the range
        elif z_range is not None:
            z0, z1 = z_range
            args += ["--z-range", int(z0), int(z1)]
        if runner:
            args += ["--runner", runner]
        for recipe, python in (interpreters or {}).items():
            args += ["--interpreter", f"{recipe}={python}"]
        if samples is not None:
            args += ["--samples", samples]
        args += _disabled_flags(disable)
        args += ["--plugin-version", plugin_version or _plugin_version()]
        if dry_run:
            args.append("--dry-run")
        args.append("--json")
        return self._command_json(self.project_argv(*args), should_stop=should_stop)

    def project_use_model(self, root, model, *, card=None, plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        args = ["use-model", root, "--model", model] + (["--card", card] if card else [])
        args += ["--plugin-version", plugin_version or _plugin_version(), "--json"]
        return self._command_json(self.project_argv(*args), should_stop=should_stop)

    def project_rounds(self, root, *, new=None, plugin_version=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        _check_round(new)
        args: list = ["rounds", root]
        if new:
            args += ["--new", *([new] if isinstance(new, str) else []), "--plugin-version",
                     plugin_version or _plugin_version()]
        args.append("--json")
        return self._command_json(self.project_argv(*args), should_stop=should_stop)

    def project_unlock(self, root, *, host, pid, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        argv = self.project_argv("unlock", root, "--host", host, "--pid", int(pid))
        rc, lines = self._command_lines(argv, should_stop=should_stop)
        # exit 2 (a lock someone else holds, a folder that is no project): EngineRefused with the engine's reason
        self._check_exit(argv, rc, lines)
        removed = not any("not locked" in line for line in lines)
        return {"removed": removed, "exit_code": rc, "lines": lines}

    # -- the full command line
    def project_run(self, root, *, images=None, resume=True, device=None, shard=None, run_id=None, say=None,
                    progress=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        run_id = check_run_id(run_id or new_run_id())
        args: list = ["project", "run", root]
        if images:
            args += ["--images", *images]
        if not resume:
            args.append("--no-resume")
        dev = device or self.device
        if dev:
            args += ["--device", dev]                 # the choice on this computer wins over the project's record
        if shard:
            args += ["--shard", shard]
        args += ["--run-id", run_id, "--via", self.via]
        tail = LogTail(project_log_path(root, run_id))
        events: list[dict] = []

        def tick() -> None:
            new = tail.read()
            if new:
                events.extend(new)
                if progress is not None:
                    progress(new)

        argv = self.engine_argv(*args)
        try:
            rc, lines = self._command_lines(argv, say=say, should_stop=should_stop, tick=tick)
        except RunCancelled as e:
            e.run_id, e.run_log, e.events = run_id, str(tail.path), events
            raise
        if rc not in (0, 1, 2):
            self._module_missing("\n".join(lines))
        return project_run_result(run_id, tail.path, events, rc, lines)

    def project_status(self, root, *, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return self._command_json(self.engine_argv("project", "status", root, "--json"), should_stop=should_stop)

    def project_collect(self, root, *, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        return self._command_json(self.engine_argv("project", "collect", root, "--json"), should_stop=should_stop)

    def project_export_commands(self, root, *, shell="sh", shards=None, out=None, should_stop=None) -> dict:
        self._start(should_stop, "projects")
        args: list = ["project", "export-commands", root, "--shell", shell]
        if shards is not None:
            args += ["--shards", int(shards)]
        if out is not None:
            args += ["--out", out]
        argv = self.engine_argv(*args)
        rc, out_lines, err = self._run_proc(argv, should_stop=should_stop)
        self._check_exit(argv, rc, out_lines + err, stdout="\n".join(out_lines))
        text = "\n".join(out_lines) + ("\n" if out_lines else "")
        if out is not None:                           # the commands are what the engine wrote, not what it said
            from germquant.fsutil import long_path

            try:
                with open(long_path(out), "rb") as fh:
                    text = fh.read().decode("utf-8-sig", errors="replace")
            except OSError:
                pass                                  # no file after all: what the engine printed
        return {"text": text, "out": str(out) if out is not None else None}

    def import_labels(self, src, fmt, image, pairs, *, allow_pickle=False, frame=None, channel=None,
                      should_stop=None) -> dict:
        self._start(should_stop, "import_labels")
        if fmt not in IMPORT_FORMATS:
            raise ValueError(f"not an import format: {fmt!r} (one of {', '.join(IMPORT_FORMATS)})")
        args: list = ["import-labels", src, "--format", fmt, "--image", image, "--pairs", pairs]
        if allow_pickle:
            args.append("--allow-pickle")
        if frame is not None:
            args += ["--frame", int(frame)]
        if channel is not None:
            args += ["--channel", channel]
        return self._command_json(self.engine_argv(*args, "--json"), should_stop=should_stop)

    def export_training(self, results, out, *, orientations=None, slices_per_image=None, seed=None,
                        zone_column=None, min_labels=None, max_gb=None, say=None, should_stop=None) -> dict:
        self._start(should_stop)
        args: list = ["export-training", results, "--out", out]
        if orientations is not None:
            args += ["--orientations", _join_list(orientations)]
        if slices_per_image is not None:
            args += ["--slices-per-image", int(slices_per_image)]
        if seed is not None:
            args += ["--seed", int(seed)]
        if zone_column is not None:
            args += ["--zone-column", zone_column]
        if min_labels is not None:
            args += ["--min-labels", int(min_labels)]
        if max_gb is not None:
            args += ["--max-gb", f"{float(max_gb):g}"]
        argv = self.engine_argv(*args)
        rc, lines = self._command_lines(argv, say=say, should_stop=should_stop)
        return self._lines_result(argv, rc, lines)

    def train_nuclei(self, pairs, out, *, base="cpsam", name=None, epochs=None, split_by=None, val_frac=None,
                     seed=None, device=None, system=None, kind=None, dataset=None, say=None,
                     should_stop=None) -> dict:
        self._start(should_stop)
        device = device or self.device
        args: list = ["train-nuclei", "--pairs", pairs, "--out", out, "--base", base]
        if name:
            args += ["--name", name]
        if epochs is not None:
            args += ["--epochs", int(epochs)]
        if split_by is not None:
            args += ["--split-by", split_by]
        if val_frac is not None:
            args += ["--val-frac", f"{float(val_frac):g}"]
        if seed is not None:
            args += ["--seed", int(seed)]
        if device == "cpu":
            args.append("--cpu")
        for option, value in (("--system", system), ("--kind", kind), ("--dataset", dataset)):
            if value is not None:
                args += [option, value]
        env = {DEVICE_ENV: device} if device in ("cuda", "mps") else {}
        argv = self.engine_argv(*args)
        rc, lines = self._command_lines(argv, say=say, should_stop=should_stop, extra_env=env)
        res = self._lines_result(argv, rc, lines)
        for line in lines:                            # "  model -> PATH (sha256 ...)", "  card  -> PATH"
            m = re.match(r"^\s*(model|card)\s+->\s+(.+?)(?:\s+\(sha256 [0-9a-f]+\.\.\.\))?\s*$", line)
            if m:
                res[f"{m.group(1)}_path"] = m.group(2)
        res.setdefault("model_path", None)
        res.setdefault("card_path", None)
        return res

    def train_curve(self, pairs, out, *, sizes, repeats, seed, base, keep_models=None, device=None, say=None,
                    should_stop=None) -> dict:
        self._start(should_stop, "train_curve")
        args: list = ["train-curve", "--pairs", pairs, "--out", out, "--sizes", _join_list(sizes), "--repeats",
                      int(repeats), "--seed", int(seed), "--base", base]
        if keep_models is not None:
            args += ["--keep-models", keep_models]
        device = device or self.device
        env = {} if device in (None, "", "auto") else {DEVICE_ENV: device}
        return self._command_json(self.engine_argv(*args, "--json"), say=say, should_stop=should_stop,
                                  extra_env=env)

    def tune_spots(self, project, *, points=None, grid=None, max_runs=None, out=None, say=None,
                   should_stop=None) -> dict:
        self._start(should_stop, "tune_spots")
        args: list = ["tune-spots", "--project", project]
        if points is not None:
            args += ["--points", points]
        if grid is not None:
            args += ["--grid", grid]
        if max_runs is not None:
            args += ["--max-runs", int(max_runs)]
        if out is not None:
            args += ["--out", out]
        # exit 1 with its record: the search ran but nothing could be scored (best None, the warnings say why)
        return self._command_json(self.engine_argv(*args, "--json"), say=say, should_stop=should_stop, ok=(0, 1))

    def tune_apply(self, project, run_dir, *, plugin_version=None, should_stop=None) -> dict:
        """``tune-spots --project DIR --apply tuning/<run>`` (the engine stamps its own write)."""
        self._start(should_stop, "tune_spots")
        argv = self.engine_argv("tune-spots", "--project", project, "--apply", run_dir)
        rc, lines = self._command_lines(argv, should_stop=should_stop)
        res = self._lines_result(argv, rc, lines)
        res["config_hash"] = next((m.group(1) for m in (re.search(r"config_hash\W+([0-9a-f]{12})\b", ln)
                                                       for ln in lines) if m), None)
        return res

    # -- the torch-free packs entry: pack derive
    def pack_derive(self, from_ref, recipe, model, card, out, *, new_id=None, should_stop=None) -> dict:
        self._start(should_stop, "pack_derive")
        args: list = ["derive", "--from", from_ref, "--recipe", recipe, "--model", model, "--card", card, "--out",
                      out]
        if new_id:
            args += ["--id", new_id]
        # pack derive refuses with exit 1 and a REFUSED line ({"ok": false, "error"} with --json): EngineRefused
        return self._command_json(self.packs_argv(*args, "--json"), should_stop=should_stop, refusal_codes=(1,))

    # -- the torch-free post-stage entry
    def list_post_stages(self, *, should_stop=None) -> list:
        self._start(should_stop, "post_stages")
        return self._command_json(self.post_stages_argv("list", "--json"), should_stop=should_stop)

    def write_post_stages(self, root, entries, *, should_stop=None) -> dict:
        """``post_stages write DIR --entries JSON``: the engine environment, where the post-stages are installed,
        validates the entries and writes <project>/post_stages.yaml (D39)."""
        self._start(should_stop, "post_stages")
        text = json.dumps(list(entries), separators=(",", ":"), ensure_ascii=True)
        return self._command_json(self.post_stages_argv("write", root, "--entries", text, "--json"),
                                  should_stop=should_stop)

    # -- other commands of the full command line
    def info(self, image, *, should_stop=None) -> dict:
        """``germquant info IMAGE``: ``exit_code`` and the printed ``lines`` (Train B adds "single plane")."""
        self._start(should_stop)
        argv = self.engine_argv("info", image)
        rc, lines = self._command_lines(argv, should_stop=should_stop)
        return self._lines_result(argv, rc, lines)

    def collect(self, root, *, samples=None, say=None, should_stop=None) -> dict:
        """``germquant collect ROOT [--samples FILE]``."""
        self._start(should_stop, *(("sample_sheet",) if samples else ()))
        argv = self.engine_argv("collect", root, *(["--samples", samples] if samples else []))
        rc, lines = self._command_lines(argv, say=say, should_stop=should_stop)
        return self._lines_result(argv, rc, lines)

    def batch(self, folder, *, options=(), samples=None, post_stages=None, say=None, should_stop=None) -> dict:
        """``germquant batch FOLDER OPTIONS [--samples FILE] [--post-stages FILE]`` (the device as for runs)."""
        self._start(should_stop, *(["sample_sheet"] if samples else []), *(["post_stages"] if post_stages else []))
        args: list = ["batch", folder, *options]
        if samples:
            args += ["--samples", samples]
        if post_stages:
            args += ["--post-stages", post_stages]
        argv = self.engine_argv(*args)
        rc, lines = self._command_lines(argv, say=say, should_stop=should_stop, extra_env=self._device_env())
        return self._lines_result(argv, rc, lines)


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
