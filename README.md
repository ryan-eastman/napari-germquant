# napari-germquant

A napari front end for **germquant**, the lab's engine for 3D quantification of the C. elegans germline
(nuclei, germline isolation, RAD-51 foci, SYP / PGL-1 colocalization, and the rest of its stages). The plugin
does no measuring of its own: every run goes through `germquant.api.run`, the function the `germquant run`
command itself calls, so a run started from the viewer and one started in a terminal with the same settings
write the same files. `tests/test_run_widget.py::test_gui_run_writes_the_same_files_as_the_cli` proves that on
a synthetic gonad: same file list, every table cell-identical, same completion marker.

`napari-germquant` is a working name (see `NAMING.md` for the one-step rename). This is the scaffold of the
plugin described in the engine's `docs/PROPOSAL_general_modular_tool.md` (sections 5, 11 and 12); the engine
README (`README.md` at the top of the `C_elegans_germline_masking_and_spot_counting` checkout) explains what
the stages measure, which profile to use and what the output tables mean. Read that first if you have not.

## What it does today

- **Reader.** File > Open on an `.nd2` (or a TIFF / OME-TIFF, CZI, LIF or OME-Zarr file, through the engine's
  readers extra) adds one Image layer per channel, scaled in microns from the file's voxel size, with the file
  path and channel index kept in the layer's metadata. A time-lapse, multi-position or multi-scene file is
  refused before any pixel is read, with a message that says why (the engine would otherwise silently measure
  time point 0; time-lapse support is planned, see the engine's `docs/DESIGN_timelapse.md`).
- **Run dock** (Plugins > germquant > Run). Image file (Browse, or the selected layer), a header read (sizes,
  voxel size, channel names), the recipe (a profile from the engine checkout's `config/profiles/*.yaml`, or any
  config YAML through a file picker), a channel-roles table resolved with the engine's own channel-map rules
  (red: a required role with no channel; amber: a role bound by position because no name matched), one
  checkbox per stage that has a `--no-<stage>` switch (greyed when the profile has it off or the image lacks a
  channel role it needs), the xy stride, the output folder, **Run** (in a napari thread worker, progress in the
  status bar) and **Export profile**.
- **Results as layers.** When a run finishes, or from the **Results dock** (Plugins > germquant > Results) on
  any folder germquant wrote: the nucleus labels as a Labels layer with the nuclei table as its features (hover
  a nucleus: its row shows in the status bar), granule and envelope labels when present, the spots as a Points
  layer in microns with the spots table as features, the spots blob image hidden, and on request the source
  image at the stride the run used. A batch root lists the images one folder level down.
- **Export profile.** Writes `<recipe>_<image>.yaml`, an `extends:` overlay on the chosen profile marked
  `validated: false` that carries the switched-off stages, and `<recipe>_<image>_commands.txt` with the exact
  `germquant run` and `germquant batch` commands that reproduce the GUI run (plus the overlay form, which
  gives the same numbers under its own `config_hash`).

## What it does not do yet

- A stage can only be switched **off**. The command line has `--no-<stage>` and nothing else, and the plugin
  keeps that parity on purpose. To turn a stage on that a profile has off, or to change any parameter, choose
  or write a profile (Export profile gives you a starting file; pick it with "Config file...").
- No time-lapse, no tracking, no label correction round trip, no training, no system packs, no project files.
  These are the later phases of the proposal.
- The reader loads the whole stack into memory (a full-resolution gonad is about 3 GB). Use the stride in the
  Run dock for a quick look; lazy loading is a to-do.
- The heavy stages need the engine's own extras in this environment: Cellpose and a torch wheel for nucleus
  segmentation with the shipped model, SpotMAX and Cell-ACDC for the spots stage. Without them the segment
  stage falls back to the classical watershed (a baseline, not a substitute) and the spots stage records a
  failure in `<image_id>__stages.json`.
- Tested on the lab workstation (Windows 11) on 2026-10-09 with the CPU torch wheel. Not yet run on a Mac.

## Install

The plugin lives in **its own environment**, never in the engine's `.venv`: Qt and napari must not enter the
pinned environment the validated numbers depend on. You need Python 3.11, the `uv` helper (the engine README's
setup sections say how to get both) and a checkout of the engine (`git clone`, or the unzipped download) with
its `config/` folder: the profiles and channel maps come from there. Below, `<germquant>` is the path of that
checkout, for example `C:\Users\you\C_elegans_germline_masking_and_spot_counting`.

### Windows

Open PowerShell, go into this project's folder (`cd`, a space, drag the folder in, Enter) and run, one at a
time:

```
uv venv --python 3.11 .venv
```

Then torch. On a computer with an NVIDIA graphics card (the workstation), the CUDA wheel the engine README
uses:

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -c <germquant>\constraints\desktop-2026-09.txt --extra-index-url https://download.pytorch.org/whl/cu128 -e "<germquant>[gpu,sc,readers]"
uv pip install -c <germquant>\constraints\desktop-2026-09.txt --extra-index-url https://download.pytorch.org/whl/cu128 spotmax cellacdc
```

On a laptop without NVIDIA (this is the variant that was actually run, on 2026-10-09, with the GPU left to a
running job; the `constraints\desktop-2026-09-cpu.txt` file in this folder is the engine's constraints with
the torch, torchvision and cupy lines removed so the CPU wheel is kept):

```
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -c <germquant>\constraints\desktop-2026-09.txt -e "<germquant>[readers]"
uv pip install -c constraints\desktop-2026-09-cpu.txt cellpose==4.2.1.1 bioio bioio-ome-tiff bioio-tifffile
```

In both cases, finish with the plugin itself (napari 0.9 with PyQt6 and the test tools):

```
uv pip install -c <germquant>\constraints\desktop-2026-09.txt -e ".[dev]"
```

Then the nucleus model, once, into the engine checkout (the plugin's germquant reads it from there):

```
.venv\Scripts\germquant.exe get-model
```

Start napari with `.venv\Scripts\napari.exe`.

### Mac (Apple silicon)

Not yet run on a real Mac: these are the engine README's Mac lines pointed at this environment. Tell Ryan what
you see. In Terminal, in this project's folder:

```
uv venv --python 3.11 .venv
uv pip install -c <germquant>/constraints/mac-arm64-2026-09.txt -e "<germquant>[gpu,sc,readers]"
uv pip install -c <germquant>/constraints/mac-arm64-2026-09.txt spotmax cellacdc
uv pip install -c <germquant>/constraints/mac-arm64-2026-09.txt -e ".[dev]"
.venv/bin/germquant get-model
```

Do not use the `download.pytorch.org/whl/cu128` lines: that is the NVIDIA build. The plain install brings the
Mac torch, which drives the graphics chip through Metal (`docs/MAC_METAL.md` in the engine). Start napari with
`.venv/bin/napari`.

### Checking the install

```
.venv\Scripts\python.exe -c "import germquant, napari; print(germquant.__file__, napari.__version__)"
```

should print a path inside your engine checkout and `0.9.x`. `constraints/plugin-venv-2026-10-09.txt` records
every package version the workstation install resolved to (napari 0.9.2, PyQt6 6.11.0 with Qt 6.11.2, npe2 0.9.0,
magicgui 0.10.2, numpy 2.4.6, pandas 3.0.3, torch 2.11.0+cpu, cellpose 4.2.1.1, bioio 3.5.1).

## The first screen

1. Start napari. Open the two docks from the Plugins menu: germquant > Run, and germquant > Results.
2. Open an image: File > Open File(s)..., pick a per-gonad `.nd2`. If napari asks which reader to use, choose
   germquant (it can remember the choice in Preferences > Plugins). One layer per channel appears, named
   `<image> [<channel>]`; the channel names come from the microscope file. For a quick look at a 3 GB gonad
   on a laptop, use the Run dock instead: Browse..., set "open with xy stride" to 4, "Open in viewer".
3. In the Run dock the **Header** line shows the sizes, the voxel size in microns and the channel names.
   A time-lapse or multi-scene file says REFUSED here and cannot be run.
4. **Profile**: pick the recipe, for example `rad51_foci` for a DAPI / SYP / RAD-51 panel or `generic` for a
   dish of cells. The file it resolves to is shown. "Config file..." takes any YAML instead, for example one
   this dock exported.
5. **Channel roles**: one row per role of the recipe's channel map, with the channel it resolved to. A `*`
   marks a required role. Red means a required role has no channel: the run cannot start; pick a recipe whose
   map fits the panel, or fix the channel names. Amber means the name did not match and the map's position
   fallback was used; the run will carry a `channel:<role>:name_unmatched_used_index_N` flag.
6. **Stages**: one checkbox per switchable stage, generated from the engine's registry. Ticked = runs.
   Untick one to skip it (the same as `--no-<stage>` in a terminal). Grey with "(off in this profile)" means
   the recipe has it off; grey with "(needs channel role(s) ...)" means the image lacks a channel it needs and
   the engine will skip it on its own.
7. **xy stride** 1 is the validated setting; 2 or 4 is a quick test. **Output folder** defaults to
   `<image stem>_results` next to the image.
8. **Run**. The status bar shows the elapsed time and the engine's log lines; the Status field under the
   buttons shows the result: nuclei, QC flags, which stages ran, failed or skipped. The nucleus labels and
   the spots appear as layers. Hover a nucleus to see its measurements; the full tables are the CSV files in
   the output folder.
9. **Export profile** asks for a folder and writes the overlay YAML and the commands file. Paste the `run`
   line into a terminal of the engine environment to repeat the run headless, or the `batch` line to process
   every image under a folder on the workstation or the cluster, exactly as the GUI would.
10. **Results dock**: Browse to any results folder (or a batch root), pick the image, "Open as layers".
    Tick "also open the source image" to see the labels on the original stack.

## Headless, and why the numbers agree

The GUI runs `germquant.api.run(image, profile=..., out=..., xy_stride=..., overrides={...})` where the
overrides are exactly the `<stage>.config_key: False` entries that `--no-<stage>` sets on the command line.
Both routes load the same config file, apply the same overrides, so the `config_hash`, the per-stage hashes
and every output file agree (only the `run_timestamp` fields differ). The exported overlay YAML is a
different object: a profile hashes as text, so a run made with it has its own `config_hash` and the same
numbers; the commands file says so.

## Tests

Headless, with napari's `make_napari_viewer` fixture (pytest-qt). From this folder:

PowerShell:
```
$env:QT_QPA_PLATFORM="offscreen"; $env:CUDA_VISIBLE_DEVICES="-1"; .venv\Scripts\python.exe -m pytest tests -q
```
Mac / Linux:
```
QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest tests -q
```

What runs: the npe2 manifest (valid, named like the distribution, reader patterns equal the engine's suffix
list), the engine glue (profiles, recipes, role rows, stage rows, the plan and its commands, the exported
overlay loading through the engine), the reader (one layer per channel, the stride, the pixels equal what the
pipeline reads, the time-lapse and multi-scene refusals from the header), the Run dock (roles table and stage
checkboxes on the synthetic gonad and on a one-channel file, the plan, the export, and the GUI-versus-CLI
identity run with the classical segmenter and spots off), and the results loader and dock on a hand-made
results folder. Two more are data dependent: `realdata` reads the header of a local `.nd2` copy and skips
when it is absent; `nas` reads the header of the live GFP::SYP-2 series on the lab NAS (56 GB, header only,
T=25 refused) and runs only with `NAPARI_GERMQUANT_NAS_TEST=1`.

Qt's `offscreen` platform plugin cannot create an OpenGL context on Windows (or macOS), and napari queries two
OpenGL limits while it builds a viewer. A hidden viewer never draws, so `tests/conftest.py` gives those two
queries constants whenever the platform is `offscreen`; nothing is rendered in the suite and the GPU is never
touched. With `QT_QPA_PLATFORM` unset on a desktop session, napari queries the real driver as usual.

## Layout

- `src/napari_germquant/napari.yaml`: the npe2 manifest (reader, two widgets).
- `_engine.py`: the glue to `germquant.api`, no Qt: profiles, recipes, role rows, stage rows, `RunPlan`
  (the one description a run has, executed by `api.run` here and spelled out as commands for the terminal).
- `_reader.py`: the reader and the single-time-point check.
- `_results.py`: a results folder as layer tuples, no Qt.
- `_run_widget.py`, `_results_widget.py`, `_qt_common.py`: the two docks and the log-to-status-bar bridge.
- `constraints/`: the CPU-filtered constraints copy and the record of the workstation install.

## License

MIT, like germquant.
