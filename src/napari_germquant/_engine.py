"""Glue between the widgets and ``germquant.api``: pure python, no Qt, so every rule here is tested headless.

What lives here: the profile list (the same lookup ``germquant --profile`` uses), the channel-role table for
an image under a recipe, the stage rows (on or off in the profile, greyed when a channel role is missing),
and the RunPlan: the one description of a run that the Run button executes (through an engine runner,
``_runner.py``) and that the Export button writes out as the matching ``germquant run`` and ``germquant batch``
commands plus an ``extends:`` overlay profile marked ``validated: false``.

The contract that keeps a GUI run and a terminal run identical: the GUI can only do what the command line
can do. A recipe is a profile name (``--profile``), a config file (``--config``) or a system pack recipe
(``--pack ID@VERSION --recipe R``); a stage can only be switched OFF (``--no-<stage>``:
``germquant.stages.apply_cli_switches`` sets ``<stage>.config_key`` to False, and RunPlan.overrides sets exactly
the same keys to False through ``api.run(overrides=...)``, so the config hash, the manifest and every table
agree); ``--xy-stride`` is the only geometry control. Turning a stage ON that the recipe has off, or changing
any other parameter, is done by choosing or writing a profile, never by a hidden widget setting.

System packs (docs/DESIGN_system_packs.md, section 8) need germquant 0.2.0.dev0 or later: every pack feature
is gated on ``hasattr(api, "list_packs")`` (``packs_supported``), and an older engine gets one message
(``packs_unavailable_message``) while profiles and config files keep working.
"""
from __future__ import annotations

# germquant.api first: it imports germquant.pipeline, whose torch import must come before any CuPy import
# (the cuBLAS load-order guard the engine documents for the RTX 5090 workstation).
from germquant import api  # noqa: I001

import dataclasses
import datetime as _dt
import sys
from pathlib import Path

import yaml

from germquant.config import Config, config_roots
from germquant.stages import BY_NAME, Stage, missing_roles

# the oldest engine with system packs, the torch-free entry `python -m germquant.packs` and `python -m germquant`
MIN_ENGINE = "0.2.0.dev0"

# ---------------------------------------------------------------------------------------------------
# system packs: the feature gate
# ---------------------------------------------------------------------------------------------------


def packs_supported() -> bool:
    """Whether this environment's germquant has system packs (germquant 0.2.0.dev0 and later)."""
    return hasattr(api, "list_packs")


def packs_unavailable_message() -> str:
    """The one message an engine without system packs gets."""
    return (f"System packs need germquant {MIN_ENGINE} or later. The germquant this napari environment uses is "
            f"{api.__version__} ({Path(api.__file__).parent}). Update that engine checkout (git pull), reinstall it "
            f"into this environment and restart napari. Profiles and config files (Advanced, below) still work.")


# ---------------------------------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------------------------------

# the benchmark profiles (bench_log_tuned, bench_spotmax_tuned): tuning records, hidden in the dock unless
# "show all" is ticked
BENCH_PREFIX = "bench_"


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str      # the file stem: what ``--profile`` takes
    path: Path

    @property
    def bench(self) -> bool:
        return self.name.startswith(BENCH_PREFIX)


def profiles() -> list[Profile]:
    """``config/profiles/*.yaml`` under the config roots germquant searches (the checkout the installed
    package lives in, the working directory, then ``$GERMQUANT_CONFIG_ROOT``). The first root that has a
    name wins, exactly as ``germquant.config.find_profile`` resolves ``--profile``."""
    found: dict[str, Profile] = {}
    for root in config_roots():
        for p in sorted((root / "config" / "profiles").glob("*.yaml")):
            found.setdefault(p.stem, Profile(p.stem, p.resolve()))
    return list(found.values())


def checkout_root() -> Path | None:
    """The folder whose ``config/`` the profiles come from (the germquant checkout), or None."""
    for root in config_roots():
        if (root / "config" / "profiles").is_dir():
            return root
    return None


# ---------------------------------------------------------------------------------------------------
# recipe: what --profile / --config takes
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Recipe:
    """What the command line takes: a profile name (``--profile``), a config file (``--config``) or a system
    pack recipe (``--pack ID@VERSION --recipe R``), exactly one of the three. ``pack`` always carries the
    version, so a run, an export and its log name the same pack version. ``hint`` is the recipe's profile
    file as the engine environment reported it: the dock loads that file when this environment's germquant
    does not find the same pack version (it is not part of the recipe's identity)."""
    profile: str | None = None
    config: Path | None = None
    pack: str | None = None
    recipe: str | None = None
    hint: Path | None = dataclasses.field(default=None, compare=False)

    def __post_init__(self) -> None:
        given = [v for v in (self.profile, self.config, self.pack) if v is not None]
        if len(given) != 1:
            raise ValueError("a Recipe is a profile name, a config path or a pack recipe, exactly one of them")
        if self.pack is not None:
            if not self.recipe:
                raise ValueError("a pack recipe names its recipe")
            pid, _, version = str(self.pack).partition("@")
            if not pid or not version or "/" in self.pack:
                raise ValueError(f"a pack is named with its version, <id>@<version>, not {self.pack!r}")
        elif self.recipe is not None:
            raise ValueError("recipe goes with pack")

    @classmethod
    def from_profile(cls, name: str) -> "Recipe":
        return cls(profile=str(name))

    @classmethod
    def from_file(cls, path: str | Path) -> "Recipe":
        return cls(config=Path(path))

    @classmethod
    def from_pack(cls, pack: str, recipe: str, hint: str | Path | None = None) -> "Recipe":
        """``pack`` is ``<id>@<version>``; ``recipe`` one of its recipe ids."""
        return cls(pack=str(pack), recipe=str(recipe), hint=Path(hint) if hint else None)

    @property
    def is_pack(self) -> bool:
        return self.pack is not None

    @property
    def pack_id(self) -> str | None:
        return str(self.pack).partition("@")[0] if self.pack else None

    @property
    def pack_version(self) -> str | None:
        return str(self.pack).partition("@")[2] if self.pack else None

    @property
    def label(self) -> str:
        if self.pack is not None:
            return f"{self.pack}/{self.recipe}"
        return self.profile if self.profile is not None else Path(self.config).name   # type: ignore[arg-type]

    @property
    def stem(self) -> str:
        """A file-name-safe name for exports."""
        if self.pack is not None:
            return f"{self.pack_id}_{self.recipe}"
        return self.label.replace(".yaml", "").replace(".yml", "")

    def extends_ref(self) -> str:
        """``pack:<id>@<version>/<recipe>``: the portable extends line of a pack recipe (no machine path)."""
        if self.pack is None:
            raise ValueError("only a pack recipe has a pack: reference")
        return f"pack:{self.pack}/{self.recipe}"

    def path(self) -> Path:
        """The YAML file the recipe resolves to (the profile file for a name, the recipe's profile for a pack)."""
        if self.config is not None:
            return Path(self.config).resolve()
        if self.pack is not None:
            try:
                from germquant import packs

                p = packs.find_pack(self.pack)
                return packs.recipe_profile_path(p, p.recipe(self.recipe)).resolve()
            except (ImportError, FileNotFoundError):
                if self.hint is not None and Path(self.hint).is_file():
                    return Path(self.hint).resolve()
                raise
        return api.find_profile(self.profile).resolve()   # type: ignore[arg-type]

    def load(self) -> Config:
        """A fresh Config, loaded as ``germquant run`` loads it (api.resolve_config; for a pack recipe with its
        pack record). When this environment's germquant lacks the pack version the engine listed, the profile
        file the engine reported (``hint``) is loaded instead: the same text, so the same config_hash."""
        if self.pack is not None:
            if not packs_supported():
                raise RuntimeError(packs_unavailable_message())
            try:
                return api.resolve_config(pack=self.pack, recipe=self.recipe)
            except FileNotFoundError:
                if self.hint is not None and Path(self.hint).is_file():
                    return api.load_config(Path(self.hint))
                raise
        return api.resolve_config(config=self.config, profile=self.profile)

    def cli_args(self) -> list[str]:
        if self.pack is not None:
            return ["--pack", str(self.pack), "--recipe", str(self.recipe)]
        if self.profile is not None:
            return ["--profile", self.profile]
        return ["--config", str(self.config)]

    def run_kwargs(self) -> dict:
        """The api.run keywords for this recipe."""
        if self.pack is not None:
            return {"pack": self.pack, "recipe": self.recipe}
        return {"config": self.config, "profile": self.profile}

    def describe(self) -> dict:
        """The recipe as a run log records it."""
        if self.pack is not None:
            return {"kind": "pack", "pack": self.pack, "recipe": self.recipe}
        if self.profile is not None:
            return {"kind": "profile", "profile": self.profile}
        return {"kind": "config", "config": str(self.config)}


# ---------------------------------------------------------------------------------------------------
# channel roles for an image under a recipe
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RoleRow:
    role: str
    marker: str
    required: bool                  # ``required: true`` in the channel map
    index: int | None               # the channel the role resolved to, None when unbound
    channel_name: str
    how: str                        # "name" | "index" | "unbound"

    @property
    def problem(self) -> bool:
        """Red: a required role with no channel. The run would flag channel:<role>:MISSING_REQUIRED."""
        return self.required and self.index is None

    @property
    def status(self) -> str:
        if self.index is None:
            return "UNBOUND (required)" if self.required else "unbound (optional)"
        if self.how == "index":
            return "by position: no name matched, index fallback (flagged in the run)"
        return "by name"


def role_rows(cfg: Config, channel_names: list[str]) -> list[RoleRow]:
    """One row per role of the recipe's channel map, resolved against the image's channel names with the
    engine's own resolver (``ChannelMap.resolve``), so the table shows exactly what the run will use."""
    mapping, flags = cfg.channel_map.resolve(list(channel_names))
    rows = []
    for role, spec in cfg.channel_map.roles.items():
        idx = mapping.get(role)
        if idx is None:
            how = "unbound"
        elif f"channel:{role}:name_unmatched_used_index_" in " ".join(flags):
            how = "index"
        else:
            how = "name"
        name = channel_names[idx] if idx is not None and idx < len(channel_names) else ""
        rows.append(RoleRow(role=role, marker=spec.marker, required=bool(spec.required), index=idx,
                            channel_name=str(name), how=how))
    return rows


def role_to_index(rows: list[RoleRow]) -> dict[str, int | None]:
    return {r.role: r.index for r in rows}


def unassigned_channels(rows: list[RoleRow], channel_names: list[str]) -> list[tuple[int, str]]:
    """Channels no role claims (they are read, measured for intensity, and otherwise ignored)."""
    used = {r.index for r in rows if r.index is not None}
    return [(i, str(n)) for i, n in enumerate(channel_names) if i not in used]


# ---------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageRow:
    stage: Stage
    on_in_profile: bool
    missing: tuple[str, ...]        # channel roles the stage needs that the image does not bind

    @property
    def name(self) -> str:
        return self.stage.name

    @property
    def selectable(self) -> bool:
        """The checkbox is live only for a stage the profile has on and whose roles are all bound; the
        command line can switch a stage off (``--no-<stage>``) but not on, and a stage with a missing
        role skips itself whatever the switch says."""
        return self.on_in_profile and not self.missing

    @property
    def reason(self) -> str:
        if not self.on_in_profile:
            return "off in this profile (choose or export a profile that enables it)"
        if self.missing:
            return "needs channel role(s) the image does not bind: " + ", ".join(self.missing)
        return ""


def stage_rows(cfg: Config, role_to_idx: dict[str, int | None]) -> list[StageRow]:
    """One row per stage that has a ``--no-<stage>`` switch (``germquant.api.cli_switches``)."""
    return [StageRow(st, api.stage_enabled(cfg, st.name), tuple(missing_roles(role_to_idx, st.name)))
            for st in api.cli_switches()]


# ---------------------------------------------------------------------------------------------------
# the run plan: one description, two executions (api.run here, the command line elsewhere)
# ---------------------------------------------------------------------------------------------------


def _quote(arg: str) -> str:
    """Double quotes around anything with a space or a shell character: PowerShell, cmd, bash and zsh
    all read them the same way for a path (the local gonad files have a space in their names)."""
    s = str(arg)
    if s and not any(c in s for c in ' \t"\'()&|;<>$`'):
        return s
    return '"' + s.replace('"', '\\"') + '"'


def _nest(dotted: dict[str, object]) -> dict:
    out: dict = {}
    for key, value in dotted.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


@dataclasses.dataclass(frozen=True)
class RunPlan:
    image: Path
    recipe: Recipe
    out: Path
    xy_stride: int = 1
    disabled: tuple[str, ...] = ()      # stage names switched off: the --no-<stage> set

    def __post_init__(self) -> None:
        unknown = [n for n in self.disabled if n not in BY_NAME or not BY_NAME[n].cli_switchable]
        if unknown:
            raise ValueError(f"not switchable stages: {unknown}")
        if int(self.xy_stride) < 1:
            raise ValueError("xy_stride must be 1 or more")

    # ---- the GUI execution ----
    def overrides(self) -> dict[str, bool]:
        """Exactly what ``apply_cli_switches`` does for the matching ``--no-<stage>`` flags."""
        return {BY_NAME[n].config_key: False for n in self.disabled}   # type: ignore[misc]

    def run(self) -> dict:
        """``germquant.api.run`` with this plan: the function ``germquant run`` itself calls (in this process;
        the dock's subprocess runner runs ``run_argv`` with the engine interpreter instead)."""
        return api.run(str(self.image), out=self.out, xy_stride=int(self.xy_stride), overrides=self.overrides(),
                       **self.recipe.run_kwargs())

    def config_hash(self) -> str | None:
        """The config_hash the run will carry: the recipe loaded as the engine loads it, with the switches set.
        None when the recipe does not load here."""
        try:
            cfg = self.recipe.load()
            for key, value in self.overrides().items():
                cfg.set(key, value)
            return cfg.hash
        except Exception:  # noqa: BLE001 - only recorded in the run log; the run reports its own hash
            return None

    # ---- the command-line execution ----
    def run_argv(self) -> list[str]:
        argv = ["run", str(self.image), *self.recipe.cli_args(), "--out", str(self.out)]
        if int(self.xy_stride) > 1:
            argv += ["--xy-stride", str(int(self.xy_stride))]
        argv += [f"--no-{n}" for n in self.disabled]
        return argv

    def batch_argv(self, folder: Path | None = None, out: Path | None = None) -> list[str]:
        folder = Path(folder) if folder is not None else Path(self.image).parent
        out = Path(out) if out is not None else Path(self.out).parent / f"{folder.name}_results"
        argv = ["batch", str(folder), *self.recipe.cli_args(), "--out", str(out)]
        if int(self.xy_stride) > 1:
            argv += ["--xy-stride", str(int(self.xy_stride))]
        argv += [f"--no-{n}" for n in self.disabled]
        return argv

    def run_command(self) -> str:
        return "germquant " + " ".join(_quote(a) for a in self.run_argv())

    def batch_command(self, folder: Path | None = None, out: Path | None = None) -> str:
        return "germquant " + " ".join(_quote(a) for a in self.batch_argv(folder, out))

    # ---- the export: an overlay profile plus the commands ----
    def overlay_mapping(self) -> dict:
        """The YAML mapping: ``validated: false``, ``extends:`` the base recipe (the recipe file, or for a pack
        recipe the portable ``pack:<id>@<version>/<recipe>``), then the switched-off stages as config keys."""
        base = self.recipe.extends_ref() if self.recipe.is_pack else self.recipe.path().as_posix()
        body: dict = {"validated": False, "extends": base}
        body.update(_nest(self.overrides()))
        return body

    def overlay_yaml(self, stamp: str | None = None) -> str:
        stamp = stamp or _dt.datetime.now().isoformat(timespec="seconds")
        if self.recipe.is_pack:
            # ASCII and no machine path anywhere (comments included): the file is hashed as text, and the same
            # file must give the same config_hash on every computer that has the pack version
            where = (f"# Base recipe: pack {self.recipe.pack}, recipe {self.recipe.recipe}. The extends line names it\n"
                     "# by version, so this file resolves the same way wherever that pack version is installed.\n")
        else:
            where = f"# Base recipe: {self.recipe.label} ({self.recipe.path().as_posix()}).\n"
        head = (
            f"# Profile overlay exported by napari-germquant on {stamp}.\n"
            + where +
            "# validated: false means no golden gate has run with this exact recipe. A profile hashes as text, so\n"
            "# runs made with this file carry their own config_hash (the GUI run applied the same switches as\n"
            "# --no-<stage> overrides and has a different hash); the numbers are the same.\n"
            "# To run without the GPU nucleus model (a laptop smoke test), add:\n"
            "#   segmentation:\n"
            "#     nuclei:\n"
            "#       method: classical\n"
        )
        text = head + yaml.safe_dump(self.overlay_mapping(), sort_keys=False, default_flow_style=False,
                                     allow_unicode=False)
        if self.recipe.is_pack:
            text.encode("ascii")                      # a pack overlay is ASCII: fail here rather than write it
        return text

    def commands_text(self, overlay_path: Path | None = None, stamp: str | None = None,
                      engine: dict | None = None) -> str:
        """The commands file. ``engine`` describes the environment the dock runs in (``python``, ``line``: the
        engine version line, ``kind``: subprocess or inprocess); by default this napari environment."""
        stamp = stamp or _dt.datetime.now().isoformat(timespec="seconds")
        root = checkout_root()
        engine = engine or {}
        python = engine.get("python") or sys.executable
        version_line = engine.get("line") or f"{api.__version__} at {Path(api.__file__).parent}"
        lines = [
            f"# germquant commands exported by napari-germquant on {stamp}.",
            "# The first command reproduces the run made from the GUI file for file: the GUI ran germquant with",
            "# these settings (the command line itself through the engine interpreter, or germquant.api.run, the",
            "# function `germquant run` calls; only the run_timestamp fields differ). Run it in the environment",
            "# that holds germquant:",
            f"#   interpreter : {python}",
            f"#   germquant   : {version_line}",
            f"#   checkout    : {root if root is not None else 'not found (set GERMQUANT_CONFIG_ROOT)'}",
            "# `germquant` is .venv\\Scripts\\germquant.exe (Windows) or .venv/bin/germquant (Mac, Linux) of that",
            "# environment; `<interpreter> -m germquant` is the same command.",
            "",
            "# One image, as run from the GUI:",
            self.run_command(),
            "",
            "# The same, as the dock's engine runner starts it:",
            " ".join(_quote(a) for a in [python, "-m", "germquant", *self.run_argv()]),
            "",
            "# Every image under the image's folder, mirroring the tree (edit the folder and the output root;",
            "# batch reads the folder's .nd2 files):",
            self.batch_command(),
        ]
        if overlay_path is not None:
            no_flags = dataclasses.replace(self, recipe=Recipe.from_file(overlay_path), disabled=())
            lines += [
                "",
                "# The same recipe as a profile file (the YAML next to this file): same numbers, its own config_hash.",
                no_flags.run_command(),
            ]
        return "\n".join(lines) + "\n"

    def export(self, folder: str | Path, stem: str | None = None, engine: dict | None = None) -> tuple[Path, Path]:
        """Write ``<stem>.yaml`` (the overlay) and ``<stem>_commands.txt`` into ``folder``; returns both
        paths. ``stem`` defaults to ``<recipe>_<image stem>`` (``<pack id>_<recipe>_<image stem>`` for a pack)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{self.recipe.stem}_{api.image_stem(self.image)}"
        stamp = _dt.datetime.now().isoformat(timespec="seconds")
        overlay = folder / f"{stem}.yaml"
        commands = folder / f"{stem}_commands.txt"
        overlay.write_text(self.overlay_yaml(stamp), encoding="ascii" if self.recipe.is_pack else "utf-8")
        commands.write_text(self.commands_text(overlay, stamp, engine=engine), encoding="utf-8")
        return overlay, commands
