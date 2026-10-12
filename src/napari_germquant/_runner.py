"""Where germquant runs: the engine runners of the Run dock (docs/DESIGN_system_packs.md, section 8.1). No Qt.

The plugin's environment and the environment that makes the numbers can differ. On the lab workstation the
plugin's venv has the CPU torch wheel and no SpotMAX, while the engine's .venv has CUDA torch and SpotMAX, and
the two install germquant from different checkouts. An EngineRunner therefore says where everything that
touches packs and runs goes:

* SubprocessRunner: an engine interpreter chosen for this computer (the dock stores the choice in QSettings).
  Pack listing, checks and fetching go through the engine's torch-free entry, ``<python> -m germquant.packs``
  (``list --json``, ``check --json``, ``fetch-models``, ``fetch-example --json`` and ``--version --json``), so a
  pack card refresh never loads torch in the engine process. A run is ``<python> -m germquant run IMAGE ...
  --out DIR``: the command line itself, with the arguments the dock exports. Every command is a list of
  arguments; no shell is ever involved.
* InProcessRunner: this napari environment, through germquant.packs and germquant.api (in the dock's thread
  worker). The dock uses it only when the user picks it explicitly; with no engine chosen it asks instead.

Each GUI run also writes a run log, ``runs/<run_id>.jsonl`` in the folder that holds the output folder (section
7.6): run_start (via, host, os, interpreter, engine version and git sha, python, device requested and selected,
config_hash, images), then image_done or image_failed, then run_end, one fsynced JSON line each. The log never
goes into the output folder itself, so a GUI run and ``germquant run`` write the same file list there. When that
folder cannot be written, the log goes to the per-user folder (``user_runs_dir``) and run_start says so.
"""
from __future__ import annotations

# germquant.api first (through _engine): the torch-before-CuPy import order of the workstation
from . import _engine  # noqa: F401,I001

import contextlib
import datetime as _dt
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
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

# The device probe a subprocess run starts with (run_start records the device requested and selected). It
# imports torch in the engine environment, never CuPy, so the cuBLAS load order stays as the engine wants it.
_RUNTIME_PROBE = (
    "import json, sys\n"
    "from germquant.device import runtime_record\n"
    "prefer = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None\n"
    "print(json.dumps(runtime_record(prefer)))\n"
)


class RunnerError(RuntimeError):
    """An engine call that could not be made or did not succeed; the message says what to do."""


class EngineTooOld(RunnerError):
    """The engine has no system packs (germquant older than 0.2.0.dev0)."""


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


def detect_interpreters() -> list[dict]:
    """Suggestions for the engine interpreter, never a decision: the .venv of the engine checkout the plugin's
    germquant comes from (when it has one), then this napari environment's own Python as a separate process.
    Each is ``{"python", "why"}``."""
    out: list[dict] = []
    checkout = engine_checkout()
    if checkout is not None:
        p = venv_python(checkout)
        if p is not None:
            out.append({"python": str(p), "why": f"the .venv of the engine checkout {checkout}"})
    out.append({"python": sys.executable, "why": "this napari environment's own Python, as a separate process"})
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
# the runners
# ---------------------------------------------------------------------------------------------------


class EngineRunner:
    """Where the dock's pack and run calls go. ``device`` is the device asked for (auto, cuda, mps or cpu)."""

    kind = "engine"
    via = ""
    python = ""

    def __init__(self) -> None:
        self.device: str = "auto"

    @property
    def key(self) -> str:
        """Identifies the environment (the check cache of the dock is keyed on it)."""
        return f"{self.kind}:{self.python}"

    def describe(self) -> str:
        raise NotImplementedError

    def version(self) -> dict:
        raise NotImplementedError

    def list_packs(self) -> dict:
        raise NotImplementedError

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None) -> dict:
        raise NotImplementedError

    def fetch_models(self, pack: str, recipe: str | None = None, say=None) -> dict:
        raise NotImplementedError

    def fetch_example(self, pack: str, example: str | None = None, out: str | Path | None = None,
                      say=None) -> dict:
        raise NotImplementedError

    def runtime(self, prefer: str | None = None) -> dict:
        raise NotImplementedError

    def run(self, plan, say=None) -> dict:
        raise NotImplementedError


class InProcessRunner(EngineRunner):
    """This napari environment: germquant.packs for listing, checks and fetching, ``RunPlan.run``
    (germquant.api.run) for runs. Chosen only explicitly."""

    kind = "inprocess"
    via = VIA_INPROCESS

    def __init__(self) -> None:
        super().__init__()
        self.python = sys.executable

    def describe(self) -> str:
        return f"this napari environment, in process ({self.python})"

    def version(self) -> dict:
        return dict(plugin_engine_info())

    def _packs(self):
        if not _engine.packs_supported():
            raise EngineTooOld(_engine.packs_unavailable_message())
        from germquant import packs

        return packs

    def list_packs(self) -> dict:
        P = self._packs()
        P.refresh()                                   # a pack installed or fetched since is seen now
        disc = P.discover()
        msg = P.no_builtin_root_message()
        return _jsonable({"version": P.version_info(), "builtin_root_found": msg is None, "message": msg,
                          "roots": [{"kind": k, "path": str(r)} for k, r in disc.roots],
                          "packs": P.list_packs(all=True)})

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None) -> dict:
        P = self._packs()
        try:
            st = P.pack_status(pack, recipe=recipe, image=str(image) if image else None)
        except FileNotFoundError as e:                # an unknown pack is unusable, as `pack check` says
            st = {"pack": pack, "recipe": recipe, "usable": False,
                  "blockers": [{"what": str(e), "fix": "germquant pack list"}], "warnings": []}
        return _jsonable(st)

    def fetch_models(self, pack: str, recipe: str | None = None, say=None) -> dict:
        P = self._packs()
        try:
            res = P.fetch_pack_models(pack, recipe, download=True, say=say)
        except (P.PackFormatError, FileNotFoundError, ValueError) as e:
            raise RunnerError(str(e)) from e
        return {"ok": not any(r["status"] == "missing" for r in res), "models": _jsonable(res),
                "lines": [f"{r['name']}: {r['status']} -> {r['path']}" for r in res]}

    def fetch_example(self, pack: str, example: str | None = None, out: str | Path | None = None,
                      say=None) -> dict:
        P = self._packs()
        try:
            return _jsonable(P.fetch_example(P.find_pack(pack), example, out, say=say))
        except FileNotFoundError as e:
            raise RunnerError(str(e)) from e
        except Exception as e:  # noqa: BLE001 - ExampleError, a download or assembly error: one message
            raise RunnerError(f"{type(e).__name__}: {e}") from e

    def runtime(self, prefer: str | None = None) -> dict:
        try:
            from germquant.device import runtime_record
        except ImportError:                           # an engine older than the runtime record
            from germquant.device import select_device

            req = (os.environ.get(DEVICE_ENV) or prefer or "auto").strip().lower()
            return {"device_requested": req, "device_selected": select_device(prefer)}
        with device_env(self.device):
            return _jsonable(runtime_record(prefer))

    def run(self, plan, say=None) -> dict:
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
    ``("-X", "importtime")`` in a test). ``history`` keeps every argument list run, newest last."""

    kind = "subprocess"
    via = VIA_SUBPROCESS

    def __init__(self, python: str | Path, *, env: dict | None = None, interpreter_options=(),
                 timeout: float = 600.0) -> None:
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

    def _call(self, argv: list[str], *, timeout: float | None = None,
              extra_env: dict | None = None) -> subprocess.CompletedProcess:
        self.history.append(list(argv))
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  env=self.child_env(extra_env), stdin=subprocess.DEVNULL,
                                  timeout=timeout or self.timeout, creationflags=_NO_WINDOW)
        except subprocess.TimeoutExpired as e:
            raise RunnerError(f"the engine interpreter {self.python} did not answer within "
                              f"{int(timeout or self.timeout)} s") from e
        except OSError as e:
            raise RunnerError(f"the engine interpreter {self.python} cannot be started: {e}") from e
        self.last_stderr = proc.stderr or ""
        return proc

    def _stream(self, argv: list[str], *, say=None, extra_env: dict | None = None) -> tuple[int, list[str]]:
        """Run ``argv`` with stdout and stderr merged, handing each line to ``say`` as it arrives."""
        self.history.append(list(argv))
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                    env=self.child_env(extra_env), creationflags=_NO_WINDOW, bufsize=1)
        except OSError as e:
            raise RunnerError(f"the engine interpreter {self.python} cannot be started: {e}") from e
        lines: list[str] = []
        with proc:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                lines.append(line)
                if say is not None and line.strip():
                    say(line)
            rc = proc.wait()
        self.last_stderr = "\n".join(lines)
        return rc, lines

    def _module_missing(self, text: str) -> None:
        if "No module named germquant.packs" in text or "No module named 'germquant.packs'" in text:
            raise EngineTooOld(
                f"The engine at {self.python} has a germquant without system packs (older than {MIN_ENGINE}). "
                f"Update that engine checkout (git pull) and reinstall it into its environment, or choose "
                f"another engine interpreter.")
        if "No module named germquant" in text or "No module named 'germquant'" in text:
            raise RunnerError(f"{self.python} has no germquant installed: choose the Python of the environment "
                              f"germquant was installed into (for example the engine checkout's .venv)")

    def _json(self, argv: list[str], *, ok=(0,), timeout: float | None = None):
        proc = self._call(argv, timeout=timeout)
        if proc.returncode not in ok:
            text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            self._module_missing(text)
            raise RunnerError(f"`{' '.join(argv[1:])}` failed (exit {proc.returncode}): {_tail(text)}")
        return _parse_json(proc.stdout)

    # ---- the engine calls ----
    def version(self, refresh: bool = False) -> dict:
        """``<python> -m germquant.packs --version --json``; EngineTooOld for an engine without packs."""
        if self._version is not None and not refresh:
            return dict(self._version)
        info = self._json(self.packs_argv("--version", "--json"), timeout=120)
        if not isinstance(info, dict) or "germquant" not in info:
            raise RunnerError(f"{self.python} answered --version with {info!r}")
        if engine_too_old(info["germquant"]):
            raise EngineTooOld(f"The engine at {self.python} is germquant {info['germquant']}, older than "
                               f"{MIN_ENGINE}, which system packs and this plugin's runner need. Update that "
                               f"engine checkout and reinstall it, or choose another engine interpreter.")
        self._version = dict(info)
        return dict(info)

    def list_packs(self) -> dict:
        return self._json(self.packs_argv("list", "--json", "--all"))

    def check(self, pack: str, recipe: str | None = None, image: str | Path | None = None) -> dict:
        args = ["check", pack]
        if recipe:
            args += ["--recipe", recipe]
        if image:
            args += ["--image", str(image)]
        return self._json(self.packs_argv(*args, "--json"), ok=(0, 2))    # 2: unusable, still a JSON report

    def fetch_models(self, pack: str, recipe: str | None = None, say=None) -> dict:
        args = ["fetch-models", pack] + (["--recipe", recipe] if recipe else [])
        rc, lines = self._stream(self.packs_argv(*args), say=say)
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
                      say=None) -> dict:
        args = ["fetch-example", pack]
        if example:
            args += ["--example", example]
        if out:
            args += ["--out", str(out)]
        proc = self._call(self.packs_argv(*args, "--json"), timeout=24 * 3600)
        if proc.returncode != 0:
            text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            self._module_missing(text)
            raise RunnerError(f"fetch-example failed (exit {proc.returncode}): {_tail(text)}")
        return _parse_json(proc.stdout)

    def runtime(self, prefer: str | None = None) -> dict:
        argv = [self.python, *self.interpreter_options, "-c", _RUNTIME_PROBE, prefer or ""]
        proc = self._call(argv, timeout=300, extra_env=self._device_env())
        if proc.returncode != 0:
            raise RunnerError(f"the device probe failed in {self.python} (exit {proc.returncode}): "
                              f"{_tail(proc.stderr)}")
        return _parse_json(proc.stdout)

    def run_argv(self, plan) -> list[str]:
        """``<python> -m germquant`` and the plan's ``germquant run`` arguments."""
        return [self.python, *self.interpreter_options, "-m", "germquant", *plan.run_argv()]

    def run(self, plan, say=None) -> dict:
        rc, lines = self._stream(self.run_argv(plan), say=say, extra_env=self._device_env())
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
    its results), else the per-user folder when that cannot be written."""
    out = Path(out_dir).absolute()
    beside = out.parent / "runs"
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
                interpreter_source: str = "default", say=None, log_dir: str | Path | None = None) -> dict:
    """Run ``plan`` through ``runner`` and write its run log: run_start before the run, image_done (or
    image_failed, then the error is raised again) and run_end after it. Returns the run's result record with
    ``run_id`` and ``run_log`` added."""
    import germquant

    run_id = new_run_id()
    if log_dir is not None:
        folder, where = Path(log_dir), "given"
        folder.mkdir(parents=True, exist_ok=True)
    else:
        folder, where = runs_folder_for(plan.out)
    rlog = RunLog(folder, run_id)
    version = runner.version()
    try:
        rt = runner.runtime(prefer)
    except RunnerError as e:                          # the run itself will say what is wrong; record the gap
        rt = {"device_requested": runner.device, "device_selected": None, "probe_error": str(e)}
    image_id = api.image_stem(plan.image)
    rlog.event("run_start", via=runner.via, host=socket.gethostname(), os=platform.platform(),
               interpreter=runner.python, interpreter_source=interpreter_source,
               engine={"germquant": version.get("germquant"), "git_sha": version.get("git_sha")},
               python=version.get("python"), constraints=None,
               device_requested=rt.get("device_requested"), device_selected=rt.get("device_selected"),
               config_hash=config_hash, recipe=plan.recipe.describe(), xy_stride=int(plan.xy_stride),
               disabled_stages=list(plan.disabled), argv=list(plan.run_argv()),
               images=[{"path": str(plan.image), "image_id": image_id, "out_dir": str(plan.out)}],
               plugin={"napari_germquant": _plugin_version(), "germquant": str(germquant.__version__)},
               log_location=where)
    t0 = time.monotonic()
    try:
        res = runner.run(plan, say=say)
    except BaseException as e:
        rlog.event("image_failed", image_id=image_id, out_dir=str(plan.out), status="failed",
                   elapsed_s=round(time.monotonic() - t0, 3), error=f"{type(e).__name__}: {e}"[:4000])
        rlog.event("run_end", counts={"done": 0, "failed": 1, "skipped": 0})
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
