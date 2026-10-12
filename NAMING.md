# The plugin name

`napari-germquant` is a working name; Ryan has not chosen one. Two spellings exist: the distribution and
plugin name with a hyphen (`napari-germquant`) and the Python import name with an underscore
(`napari_germquant`). Both must change together.

## Where the hyphen name appears

| Place | What |
|---|---|
| `pyproject.toml` | `name = "napari-germquant"` under `[project]` (THE name) |
| `pyproject.toml` | the key of the `[project.entry-points."napari.manifest"]` entry |
| `src/napari_germquant/napari.yaml` | `name:` and the prefix of every command `id:` (npe2 requires the prefix to equal `name`) |
| `src/napari_germquant/__init__.py` | `metadata.version("napari-germquant")` |
| `tests/test_manifest.py`, `tests/test_reader.py`, `tests/test_training.py`, `tests/test_timelapse.py` | the plugin name passed to `viewer.open(..., plugin=...)`, `get_manifest(...)` and `viewer.window.add_plugin_dock_widget(...)` |
| `src/napari_germquant/_engine.py`, `_reader.py`, `_training.py` | text in the exported YAML and commands headers, in user-facing messages and in the note of the `corrections.json` a training folder gets |
| `src/napari_germquant/_settings.py` | `ORG` and `APP`: the QSettings names the engine interpreter is stored under (renaming forgets the stored choice; the dock then asks again) |
| `src/napari_germquant/_runner.py` | `user_runs_dir()`: the per-user folder for run logs that cannot sit next to their output folder |
| `README.md`, `NAMING.md`, `LICENSE` | prose |
| the folder `C:/Users/ryane/napari-germquant` and its `.venv` | the checkout itself |

The manifest's `display_name: germquant` is what the napari menus show; it can stay or change independently.

## Where the underscore name appears

| Place | What |
|---|---|
| `src/napari_germquant/` | the package folder (rename it) |
| `pyproject.toml` | `napari_germquant:napari.yaml` (entry-point value) and `[tool.setuptools.package-data] napari_germquant` |
| `src/napari_germquant/napari.yaml` | every `python_name: napari_germquant.<module>:<object>` |
| `src/napari_germquant/_reader.py` | `META_KEY = "napari_germquant"`: the metadata key on every layer the reader adds |
| `src/napari_germquant/_results.py` | the `napari_germquant_results` metadata key on every layer the results loader adds |
| `src/napari_germquant/_training_widget.py` | `META_KEY = "napari_germquant_training"`: the metadata key on the two layers the Training dock shows |
| `src/napari_germquant/_timelapse.py` | `META_KEY = "napari_germquant_timelapse"`: the metadata key on every layer the Time-lapse dock adds |
| `src/napari_germquant/_runner.py`, `_qt_common.py` | the logger `napari_germquant.engine` that carries an engine subprocess's lines to the status bar |
| `src/napari_germquant/_settings.py`, `tests/conftest.py` | the environment variable `NAPARI_GERMQUANT_SETTINGS` (a private settings file) |
| `tests/*.py` | `from napari_germquant import ...` |
| `tests/conftest.py`, `tests/test_reader.py` | the opt-in environment variable `NAPARI_GERMQUANT_NAS_TEST` |

## Renaming

1. Replace the text `napari-germquant` with the new hyphen name and `napari_germquant` with the new import
   name in every file listed above (a plain find-and-replace over `*.py`, `*.toml`, `*.yaml`, `*.md`).
2. Rename the folder `src/napari_germquant` to the new import name.
3. Reinstall into the environment: `uv pip install -e ".[dev]"` from the project folder (the editable install
   registers the entry point under the new name; the old registration disappears with the uninstall uv does).
4. Run the tests: `tests/test_manifest.py` checks that the manifest name, the pyproject name and the
   entry-point key agree and that every command id carries the prefix.
5. Check nothing was missed: `grep -rn "napari.germquant" --include=*.py --include=*.toml --include=*.yaml --include=*.md .`
   should print nothing.
