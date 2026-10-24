"""Stand-in engines for the runner tests of the tranche-5 Train B plugin work (unit P6-RUNNER): a fake germquant
package whose four command entries answer as the Train B engine's will (``python -m germquant``, the torch-free
``python -m germquant.packs``, ``python -m germquant.project`` and ``python -m germquant.post_stages``), so the
runner's argument lists, run-log tailing, exit codes, Cancel and the features gate are tested before the engine
has them, and without loading torch.

Every entry but the two below prints one JSON line, ``{"module", "argv", "device_env"}``, and exits with
$FAKE_ENGINE_EXIT (default 0). ``python -m germquant.packs --version --json`` prints the version document, with
the ``features`` list only when one was given (an engine from before the list has none).
``python -m germquant project run DIR --run-id ID --via V ...`` plays the scenario in $FAKE_PROJECT_RUN (JSON):
it writes DIR/runs/ID.jsonl as the engine does (one fsynced ASCII line per event) and then exits, hangs or
crashes:

* ``refuse``: print "REFUSED: <text>" on stderr and exit 2 before any event;
* ``ids``: the image ids run_start lists;
* ``steps``: one event each, ``{"event", "image_id", "status", "sleep"}`` (seconds before it); with
  ``"partial": S`` the line is written in two halves S seconds apart;
* ``hang``: a file to write "<pid> <child pid>" into, then sleep (the child sleeps too);
* ``crash``: a traceback on stderr and exit 1 without run_end;
* ``counts`` and ``exit``: run_end's counts (default: tallied from the steps) and the exit code.

Only the standard library is used, in the parent and in the stand-in alike.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# the seams wrapper: replaces the engine's seams named in $NAPARI_GERMQUANT_SEAMS ({"module.attr":
# "module:attr"}) and then runs the full command line, as `python -m germquant ARGS` would
SEAMS_ENV = "NAPARI_GERMQUANT_SEAMS"
SEAMS_WRAPPER = (
    "import importlib, json, os, sys\n"
    "from germquant import cli\n"
    f"for target, source in json.loads(os.environ.get({SEAMS_ENV!r}) or '{{}}').items():\n"
    "    mod, _, attr = target.rpartition('.')\n"
    "    smod, _, sattr = source.partition(':')\n"
    "    setattr(importlib.import_module(mod), attr, getattr(importlib.import_module(smod), sattr))\n"
    "sys.exit(cli.main(sys.argv[1:]))\n"
)
# the engine's three Cellpose training seams, replaced as the plugin's fake_cellpose fixture does in process
TRAINING_SEAMS = {
    "germquant.segment.training._build_net": "_p6_runner_seams:build_net",
    "germquant.segment.training._train_seg": "_p6_runner_seams:train_seg",
    "germquant.segment.training._predict": "_p6_runner_seams:predict",
}
TESTS_DIR = Path(__file__).resolve().parent

STANDIN = '''\
import json, os, platform, subprocess, sys, time
from pathlib import Path

VERSION = {version!r}
GIT_SHA = {git_sha!r}
FEATURES = {features!r}


def _now():
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03d+00:00" % int((t % 1) * 1000)


def _opt(args, name):
    if name in args and args.index(name) + 1 < len(args):
        return args[args.index(name) + 1]
    return None


def _write(log, text):
    with open(log, "ab") as fh:
        fh.write(text.encode("ascii"))
        fh.flush()
        os.fsync(fh.fileno())


def _line(kind, run_id, **fields):
    return json.dumps({{"event": kind, "run_id": run_id, "time": _now(), **fields}}) + "\\n"


def project_run(args):
    spec = json.loads(os.environ.get("FAKE_PROJECT_RUN") or "{{}}")
    root, run_id, via = Path(args[0]), _opt(args, "--run-id"), _opt(args, "--via") or "cli"
    if spec.get("refuse"):
        print("REFUSED: " + spec["refuse"], file=sys.stderr, flush=True)
        sys.exit(2)
    log = root / "runs" / (run_id + ".jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    print("project run " + run_id + " via " + via, flush=True)
    _write(log, _line("run_start", run_id, via=via, project=str(root), shard=_opt(args, "--shard"), argv=args,
                      images=[{{"image_id": i}} for i in spec.get("ids", [])]))
    tally = {{"done": 0, "failed": 0, "skipped": 0}}
    for step in spec.get("steps", []):
        time.sleep(float(step.get("sleep", 0)))
        status = step.get("status") or step["event"].replace("image_", "")
        tally[status] = tally.get(status, 0) + 1
        text = _line(step["event"], run_id, image_id=step.get("image_id"), status=status,
                     error=step.get("error"))
        if step.get("partial"):
            cut = len(text) // 2
            _write(log, text[:cut])
            time.sleep(float(step["partial"]))
            _write(log, text[cut:])
        else:
            _write(log, text)
        print(step["event"] + " " + str(step.get("image_id")), flush=True)
    if spec.get("hang"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        Path(spec["hang"]).write_text(str(os.getpid()) + " " + str(child.pid), encoding="ascii")
        time.sleep(600)
    if spec.get("crash"):
        print("Traceback (most recent call last):", file=sys.stderr)
        print("RuntimeError: the stand-in engine crashed", file=sys.stderr, flush=True)
        sys.exit(1)
    _write(log, _line("run_end", run_id, counts=spec.get("counts") or tally, interrupted=False))
    sys.exit(int(spec.get("exit", 0)))


def main(module):
    args = sys.argv[1:]
    if module == "germquant.packs" and "--version" in args:
        info = {{"germquant": VERSION, "git_sha": GIT_SHA, "python": platform.python_version(),
                "executable": sys.executable}}
        if FEATURES is not None:
            info["features"] = list(FEATURES)
        print(json.dumps(info))
        sys.exit(0)
    if module == "germquant" and args[:2] == ["project", "run"]:
        project_run(args[2:])
    print(json.dumps({{"module": module, "argv": args, "device_env": os.environ.get("GERMQUANT_DEVICE")}}),
          flush=True)
    sys.exit(int(os.environ.get("FAKE_ENGINE_EXIT", "0")))
'''

ENTRY = "from germquant._standin import main\nmain({module!r})\n"


def standin_engine(root: Path, *, version: str, git_sha: str = "feedfacecafebeef",
                   features: list[str] | tuple[str, ...] | None = None) -> Path:
    """Write the stand-in germquant package under ``root`` and return ``root`` (put it first on PYTHONPATH).
    ``features`` None: a version document without the list, as every engine before Train B prints it."""
    pkg = root / "germquant"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"__version__ = {version!r}\n", encoding="ascii")
    feats = None if features is None else list(features)
    (pkg / "_standin.py").write_text(STANDIN.format(version=version, git_sha=git_sha, features=feats),
                                     encoding="ascii")
    (pkg / "__main__.py").write_text(ENTRY.format(module="germquant"), encoding="ascii")
    for sub in ("packs", "project", "post_stages"):
        (pkg / sub).mkdir()
        (pkg / sub / "__init__.py").write_text("", encoding="ascii")
        (pkg / sub / "__main__.py").write_text(ENTRY.format(module=f"germquant.{sub}"), encoding="ascii")
    return root


def run_spec(**spec) -> str:
    """$FAKE_PROJECT_RUN for one scenario (keys in the module docstring)."""
    return json.dumps(spec)


def echo(line_or_lines) -> dict:
    """The JSON record a stand-in entry printed (from a list of lines, the first that is one)."""
    lines = [line_or_lines] if isinstance(line_or_lines, str) else list(line_or_lines)
    for line in lines:
        s = str(line).strip()
        if s.startswith("{") and '"argv"' in s:
            return json.loads(s)
    raise AssertionError(f"no stand-in record in {lines!r}")


def seams_env(seams: dict | None = None, base: dict | None = None) -> dict:
    """The environment of a seams runner: the seams to replace and this tests folder first on PYTHONPATH, so
    the child imports the stand-ins of tests/_p6_runner_seams.py."""
    env = dict(base or {})
    path = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join([str(TESTS_DIR), *([path] if path else [])])
    env[SEAMS_ENV] = json.dumps(dict(TRAINING_SEAMS if seams is None else seams))
    return env


def make_seams_runner(python: str | Path | None = None, *, seams: dict | None = None, env: dict | None = None):
    """A SubprocessRunner whose ``python -m germquant`` commands start through SEAMS_WRAPPER: the seams replaced
    (by default the three Cellpose training seams), then germquant.cli.main. The torch-free entries are left as
    they are. ``seams`` maps "module.attr" to "module:attr"; a module of this tests folder can be named."""
    from napari_germquant._runner import SubprocessRunner

    class SeamsRunner(SubprocessRunner):
        kind = "seams"

        def engine_argv(self, *args) -> list[str]:
            return [self.python, *self.interpreter_options, "-c", SEAMS_WRAPPER, *[str(a) for a in args]]

    return SeamsRunner(python or sys.executable, env=seams_env(seams, env))
