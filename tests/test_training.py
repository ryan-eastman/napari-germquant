"""The Training dock and its glue: slices exported from finished runs through the engine's export-training
function (in a thread worker), the pairs paged one at a time as an Image and an editable Labels layer, a
correction saved atomically with the manifest left byte-identical, training with the engine's Cellpose seams
replaced (the model is never loaded, trained or written for real; the UI stays live while it "trains"), and
registration into a private model registry (GERMQUANT_MODELS points into the test's folder)."""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from conftest import write_ome

from napari_germquant import _training as T
from napari_germquant._training_widget import TrainingWidget
from germquant import cli
from germquant import models_registry as R
from germquant.segment import training as TR

SP = (0.4, 0.2, 0.2)
SHAPE = (8, 48, 48)
IDS = ("20250101_n2_nohs_HERM_001", "20250101_n2_nohs_HERM_002")


def _grid(offset: int) -> np.ndarray:
    """Nine box nuclei in a 3 x 3 grid (planes 2 to 5); every xy plane there cuts nine, an xz row cuts three."""
    lab = np.zeros(SHAPE, np.int32)
    k = 1
    for cy in (10, 24, 38):
        for cx in (10, 24, 38):
            lab[2:6, cy - 4 + offset:cy + 4 + offset, cx - 4:cx + 4] = k
            k += 1
    return lab


def make_results(root: Path) -> Path:
    """A batch-like tree of two finished runs, each with the files export-training needs: run_manifest.json,
    the labels, an image_summary naming the source OME-TIFF and its channel map, and a nuclei table with zones."""
    for i, iid in enumerate(IDS):
        lab = _grid(offset=i)
        rng = np.random.default_rng(i)
        dapi = (lab > 0) * 1000 + rng.integers(0, 50, SHAPE)
        syp = rng.integers(0, 100, SHAPE)
        src = root / "images" / f"{iid}.ome.tif"
        src.parent.mkdir(parents=True, exist_ok=True)
        write_ome(src, np.stack([dapi, syp]).astype(np.uint16), ["405", "477"], spacing=SP)
        out = root / "results" / iid
        out.mkdir(parents=True)
        tifffile.imwrite(out / f"{iid}__nuclei_labels.tif", lab, compression="zlib")
        pd.DataFrame([{"image_id": iid, "file_path": str(src), "n_nuclei": 9, "voxel_dz_um": SP[0], "voxel_dy_um": SP[1],
                       "voxel_dx_um": SP[2], "channel_map_json": json.dumps({"dna": 0, "central_element": 1})}]).to_csv(
            out / f"{iid}__image_summary.csv", index=False)
        pd.DataFrame({"nucleus_id": range(1, 10), "zone": ["transition"] * 3 + ["pachytene"] * 6}).to_csv(
            out / f"{iid}__nuclei.csv", index=False)
        (out / "run_manifest.json").write_text(json.dumps({"xy_stride": 1, "z_range": None, "t_index": None,
                                                           "config": {"segmentation": {"nuclei": {"channel": "dna"}}}}))
    return root / "results"


@pytest.fixture
def results(tmp_path) -> Path:
    return make_results(tmp_path)


@pytest.fixture
def pairs(tmp_path, results) -> Path:
    out = tmp_path / "pairs"
    T.export_slices(results, out, T.ExportSettings(orientations=("xy", "xz"), slices_per_image=2, seed=0))
    return out


@pytest.fixture
def models_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "models_root"
    monkeypatch.setenv("GERMQUANT_MODELS", str(root))
    return root


def _key(img) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(img, dtype=np.float32)).tobytes()).hexdigest()


@pytest.fixture
def fake_cellpose(monkeypatch):
    """The engine's three Cellpose seams replaced (as its own tests do): the net is a token, training logs a
    Cellpose epoch line, optionally waits on a gate, writes a small 'model' file and returns linear losses;
    prediction answers with the labels on disk for each image (perfect recall)."""
    calls: dict = {"build": [], "train": [], "predict": [], "env": [], "gate": None, "lookup": {}}

    def build(base, device):
        calls["build"].append((dict(base), device))
        return "net-token"

    def train_seg(net, **kw):
        calls["train"].append(kw)
        calls["env"].append(os.environ.get("GERMQUANT_DEVICE"))
        logging.getLogger("cellpose.train").info("0, train_loss=0.5000, test_loss=0.6000, LR=0.000010, time 0.01s")
        if calls["gate"] is not None:
            assert calls["gate"].wait(60), "the test never opened the gate"
        p = Path(kw["save_path"]) / "models" / kw["model_name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake cellpose weights " * 64)
        n = int(kw["n_epochs"])
        return p, np.linspace(1.0, 0.5, n), np.linspace(0.9, 0.6, n)

    def predict(model_path, images, device):
        calls["predict"].append((str(model_path), len(images), device))
        return [calls["lookup"][_key(img)] for img in images]

    monkeypatch.setattr(TR, "_build_net", build)
    monkeypatch.setattr(TR, "_train_seg", train_seg)
    monkeypatch.setattr(TR, "_predict", predict)
    return calls


def _index(pairs_dir: Path, lookup: dict) -> None:
    for p in T.load_pair_list(pairs_dir):
        image, labels = T.read_pair(p)
        lookup[_key(image)] = labels


# ---- the glue ------------------------------------------------------------------------------------------------

def test_pairs_load_and_save_atomically_without_touching_the_manifest(pairs):
    plist = T.load_pair_list(pairs)
    assert len(plist) == 2 * 2 * 2 and [p.name for p in plist] == [Path(r["image"]).name for r in TR.load_pairs(pairs)]
    p = plist[0]
    assert p.image_id == IDS[0] and p.orientation == "xy" and p.pixel_um == pytest.approx(0.2) and p.zone
    image, labels = T.read_pair(p)
    assert image.dtype == np.uint16 and labels.dtype == np.int32 and image.shape == labels.shape == (48, 48)
    manifest_bytes = (pairs / "manifest.csv").read_bytes()
    before = T.n_labels(labels)
    fixed = labels.copy()
    fixed[0:3, 0:3] = 77                                     # a new nucleus painted in a corner
    rec = T.save_labels(pairs, p, fixed)
    assert np.array_equal(tifffile.imread(p.labels), fixed) and tifffile.imread(p.labels).dtype == np.int32
    assert (pairs / "manifest.csv").read_bytes() == manifest_bytes
    assert rec["n_labels_before"] == before and rec["n_labels_after"] == before + 1
    assert rec["sha256"] == hashlib.sha256(p.labels.read_bytes()).hexdigest()
    assert not list(pairs.rglob("*.part")) and set(T.valid_corrections(pairs)) == {p.name}
    assert json.loads((pairs / T.CORRECTIONS_FILE).read_text(encoding="utf-8"))["corrections"][p.name] == rec
    # refused, and the file stays as it was: wrong shape, 3D, floats, negative ids
    for bad in (fixed[:10], fixed[None], fixed.astype(float), fixed - 100):
        with pytest.raises(ValueError):
            T.save_labels(pairs, p, bad)
    assert np.array_equal(tifffile.imread(p.labels), fixed)
    # a file changed behind the dock's back voids its record
    tifffile.imwrite(p.labels, labels, compression="zlib")
    assert T.valid_corrections(pairs) == {} and p.name in T.corrections(pairs)


def test_export_conflicts_name_corrected_slices_an_export_would_undo(results, pairs):
    assert T.export_conflicts(results, pairs) == []                  # nothing corrected yet
    p = T.load_pair_list(pairs)[0]
    _, labels = T.read_pair(p)
    T.save_labels(pairs, p, labels + (labels > 0))                   # ids shifted by one: a correction
    assert T.export_conflicts(results, pairs) == [p.name]
    assert T.export_conflicts(results / IDS[1], pairs) == []         # an export of the other image leaves it alone
    assert T.export_conflicts(results, pairs.parent / "fresh") == []


def test_settings_reproduce_the_command_line_arguments(tmp_path, results, pairs, monkeypatch):
    """export_slices, train and register call the engine with exactly what the commands pass for the same choices."""
    class Stop(BaseException):
        pass

    def recorder(fn, sink):
        sig = inspect.signature(fn)

        def rec(*a, **k):
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            sink.append({key: (str(v) if isinstance(v, Path) else v) for key, v in bound.arguments.items()})
            raise Stop
        return rec

    seen: list = []
    monkeypatch.setattr(TR, "export_training_pairs", recorder(TR.export_training_pairs, seen))
    es = T.ExportSettings(orientations=("xy", "yz"), slices_per_image=4, seed=3, min_labels=2, max_gb=0.5)
    for call in (lambda: T.export_slices(results, tmp_path / "p", es), lambda: cli.main(es.argv(results, tmp_path / "p"))):
        with pytest.raises(Stop):
            call()
    assert seen[0] == seen[1] and seen[0]["orientations"] == ("xy", "yz") and seen[0]["max_bytes"] == 0.5e9

    seen.clear()
    monkeypatch.setattr(TR, "train_nuclei", recorder(TR.train_nuclei, seen))
    for device in ("cpu", "cuda"):
        ts = T.TrainSettings(out=tmp_path / "m" / "model_a", name="model_a", base="none", epochs=7, split_by="image",
                             val_frac=0.25, seed=2, device=device)
        for call in (lambda: T.train(pairs, ts), lambda: cli.main(ts.argv(pairs))):
            with pytest.raises(Stop):
                call()
    assert seen[0] == seen[1] and seen[0]["cpu"] is True and seen[0]["epochs"] == 7
    assert seen[2] == seen[3] and seen[2]["cpu"] is False
    assert "--cpu" in T.TrainSettings(out=tmp_path, device="cpu").argv(pairs)
    cuda_cmd = T.TrainSettings(out=tmp_path, device="cuda").command(pairs)
    assert "--cpu" not in cuda_cmd and 'GERMQUANT_DEVICE="cuda"' in cuda_cmd

    seen.clear()
    monkeypatch.setattr(R, "register_local_model", recorder(R.register_local_model, seen))
    for call in (lambda: T.register("m.bin", "m.card.json", "my_model"),
                 lambda: cli.main(T.register_argv("m.bin", "m.card.json", "my_model"))):
        with pytest.raises(Stop):
            call()
    assert seen[0] == seen[1] == {"model": "m.bin", "card": "m.card.json", "name": "my_model", "base_dir": None}


def test_device_choice_reaches_the_engine_and_the_variable_is_restored(tmp_path, pairs, fake_cellpose, monkeypatch):
    monkeypatch.delenv("GERMQUANT_DEVICE", raising=False)
    _index(pairs, fake_cellpose["lookup"])
    res = T.train(pairs, T.TrainSettings(out=tmp_path / "m" / "a", base="none", epochs=2, split_by="image",
                                         val_frac=0.5, device="cuda"))
    # CUDA is hidden in the test session and there is no Apple GPU: the engine falls back to the CPU and says so
    assert fake_cellpose["env"] == ["cuda"] and res["device"] == "cpu" and fake_cellpose["build"][-1][1] == "cpu"
    assert "GERMQUANT_DEVICE" not in os.environ
    assert "asked for cuda" in T.train_summary(res, "cuda") and "asked for" not in T.train_summary(res, "cpu")
    monkeypatch.setenv("GERMQUANT_DEVICE", "auto")
    T.train(pairs, T.TrainSettings(out=tmp_path / "m" / "b", base="none", epochs=1, split_by="image", device="mps"))
    assert fake_cellpose["env"][-1] == "mps" and os.environ["GERMQUANT_DEVICE"] == "auto"
    T.train(pairs, T.TrainSettings(out=tmp_path / "m" / "c", base="none", epochs=1, split_by="image", device="cpu"))
    assert fake_cellpose["env"][-1] == "auto"                       # cpu is --cpu; the variable is left alone
    assert T.epoch_progress("40, train_loss=0.1234, test_loss=0.2345, LR=0.000010, time 9.00s", 150) == \
        "epoch 40/150, train loss 0.1234, test loss 0.2345"
    assert T.epoch_progress("saving network parameters to x") is None
    with pytest.raises(ValueError):
        T.TrainSettings(out=tmp_path, device="tpu")


# ---- the dock -----------------------------------------------------------------------------------------------

def _wait(qtbot, w, attr):
    qtbot.waitUntil(lambda: getattr(w, attr) is not None or w.last_error is not None, timeout=120_000)
    qtbot.waitUntil(lambda: not w.runner.busy, timeout=30_000)
    assert w.last_error is None, w.last_error


def test_dock_exports_pages_and_saves_corrections(make_napari_viewer, qtbot, results, tmp_path):
    viewer = make_napari_viewer()
    w = TrainingWidget(viewer)
    w.set_source(results)
    assert w.pairs_edit.text() == str(results.parent / "results_training")
    assert w.model_dir_edit.text() == str(results.parent / "results_training_model")
    w.orient_checks["yz"].setChecked(False)
    w.slices_spin.setValue(2)
    assert not w.next_btn.isEnabled() and not w.save_btn.isEnabled()
    assert w.export() and not w.export_btn.isEnabled() and not w.train_btn.isEnabled()
    _wait(qtbot, w, "last_manifest")
    pairs_dir = results.parent / "results_training"
    assert len(w.last_manifest) == 8 and len(w.pairs) == 8 and w.index == 0
    assert "8 slice pair(s) from 2 run(s)" in w.status_label.text() and "germquant export-training" in w.status_label.text()
    # the same export through the command line writes the same manifest and the same slices
    cli_dir = tmp_path / "cli_pairs"
    assert cli.main(w.export_settings().argv(results, cli_dir)) == 0
    a, b = pd.read_csv(pairs_dir / "manifest.csv"), pd.read_csv(cli_dir / "manifest.csv")
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    for rel in list(a["image"]) + list(a["labels"]):
        assert np.array_equal(tifffile.imread(pairs_dir / rel), tifffile.imread(cli_dir / rel)), rel

    # one Image and one editable Labels layer, showing pair 0 at its pixel size
    first = w.pairs[0]
    assert [layer.name for layer in viewer.layers] == [f"{Path(first.name).stem} image", f"{Path(first.name).stem} labels"]
    img_layer, lab_layer = w.image_layer, w.labels_layer
    image, labels = T.read_pair(first)
    assert np.array_equal(img_layer.data, image) and np.array_equal(lab_layer.data, labels) and lab_layer.editable
    assert tuple(np.round(lab_layer.scale, 6)) == (0.2, 0.2) and viewer.layers.selection.active is lab_layer
    assert "pair 1 of 8" in w.pair_label.text() and not w.prev_btn.isEnabled() and w.next_btn.isEnabled()

    # paint, save: the file changes, the manifest does not
    manifest_bytes = (pairs_dir / "manifest.csv").read_bytes()
    lab_layer.data[0:3, 0:3] = 77
    lab_layer.refresh()
    assert w.is_dirty()
    rec = w.save_current()
    assert rec is not None and not w.is_dirty() and "corrected" in w.pair_label.text()
    saved = tifffile.imread(first.labels)
    assert saved[0, 0] == 77 and saved.dtype == np.int32 and np.array_equal(saved, lab_layer.data)
    assert (pairs_dir / "manifest.csv").read_bytes() == manifest_bytes and not list(pairs_dir.rglob("*.part"))
    assert "corrected so far: 1 of 8" in w.corrected_label.text()

    # next and previous swap the two layers' data in place
    assert w.next_pair() and w.index == 1 and len(viewer.layers) == 2 and w.labels_layer is lab_layer
    assert np.array_equal(lab_layer.data, T.read_pair(w.pairs[1])[1])
    assert w.previous_pair() and w.index == 0 and lab_layer.data[0, 0] == 77

    # edits left behind: discarded, saved, or the move cancelled, as asked
    lab_layer.data[5:7, 5:7] = 88
    assert w.next_pair(unsaved="discard") and tifffile.imread(first.labels)[5, 5] != 88
    assert w.previous_pair()
    lab_layer.data[5:7, 5:7] = 88
    assert not w.next_pair(unsaved="cancel") and w.index == 0 and w.is_dirty()
    assert w.next_pair(unsaved="save") and tifffile.imread(first.labels)[5, 5] == 88
    lab_layer.data[0:2, 0:2] = 99
    assert w.revert_current() and not w.is_dirty() and lab_layer.data[0, 0] != 99

    # exporting again over corrected slices is refused unless the caller says so
    before = tifffile.imread(first.labels)
    assert not w.export(overwrite_corrections=False) and "export cancelled" in w.status_label.text()
    assert np.array_equal(tifffile.imread(first.labels), before)


def test_dock_trains_without_blocking_and_registers(make_napari_viewer, qtbot, pairs, tmp_path, fake_cellpose,
                                                    models_root):
    viewer = make_napari_viewer()
    statuses: list[str] = []
    viewer.events.status.connect(lambda e: statuses.append(str(e.value)))
    w = TrainingWidget(viewer)
    assert w.open_pairs(pairs) == 8
    _index(pairs, fake_cellpose["lookup"])
    w.model_dir_edit.setText(str(tmp_path / "models" / "live_test"))
    w.base_combo.setCurrentText("none")
    w.epochs_spin.setValue(2)
    w.split_combo.setCurrentText("image")
    w.val_spin.setValue(0.5)
    w.select_device("cpu")
    gate = threading.Event()
    fake_cellpose["gate"] = gate
    assert w.train()
    # the call returned at once: the training runs in a worker, the buttons wait, saving is paused
    assert w.runner.busy and not w.train_btn.isEnabled() and not w.save_btn.isEnabled() and w.next_btn.isEnabled()
    assert w.save_current() is None and "paused" in w.status_label.text()
    qtbot.waitUntil(lambda: any("epoch 0/2" in s for s in statuses), timeout=60_000)
    assert w.next_pair() and w.index == 1                           # the viewer stays usable meanwhile
    gate.set()
    _wait(qtbot, w, "last_train")
    res = w.last_train
    model = Path(res["model_path"])
    assert model == tmp_path / "models" / "live_test" / "live_test" and model.is_file() and res["device"] == "cpu"
    assert fake_cellpose["build"][0] == ({"name": "none", "kind": "random_init", "path": None, "sha256": None}, "cpu")
    assert fake_cellpose["train"][0]["n_epochs"] == 2 and res["split"]["by"] == "image"
    text = w.status_label.text()
    assert "hold-out at IoU 0.5: F1 1.000" in text and "design gate" in text and "train-nuclei --pairs" in text
    assert w.model_edit.text() == str(model) and w.card_edit.text() == res["card_path"] and w.reg_name_edit.text() == "live_test"
    assert w.save_btn.isEnabled() and w.train_btn.isEnabled()

    assert w.register_model()
    _wait(qtbot, w, "last_register")
    assert R.resolve_registered("live_test") == model.resolve()
    assert R.local_registry_path() == models_root / "models" / "local_models.json"
    status = w.status_label.text()
    assert "registered live_test" in status and "cellpose_model: " in status and "live_test" in status
    assert "absolute" in status and "resolves only in a germquant whose registry is" in status


def test_dock_reports_a_failed_training(make_napari_viewer, qtbot, pairs, tmp_path, fake_cellpose):
    viewer = make_napari_viewer()
    w = TrainingWidget(viewer)
    w.open_pairs(pairs)
    w.model_dir_edit.setText(str(tmp_path / "models" / "bad"))
    w.base_combo.setCurrentText("not_a_model_anyone_knows")
    w.select_device("cpu")
    assert w.train()
    qtbot.waitUntil(lambda: w.last_error is not None, timeout=60_000)
    qtbot.waitUntil(lambda: not w.runner.busy, timeout=30_000)
    assert isinstance(w.last_error, ValueError) and "training FAILED" in w.status_label.text()
    assert w.train_btn.isEnabled() and w.save_btn.isEnabled()


def test_dock_without_a_viewer_and_without_input():
    w = TrainingWidget(None)
    assert not w.export() and "results or series folder" in w.status_label.text()
    assert w.open_pairs() == 0 and not w.train() and not w.register_model()
    assert w.device_combo.count() == 3 and [w.device_combo.itemData(i) for i in range(3)] == ["cuda", "mps", "cpu"]
    assert w.device_combo.currentData() == T.default_device() == "cpu"     # CUDA hidden, no Apple GPU here


def test_dock_through_the_plugin_menu_keeps_two_layers(make_napari_viewer, pairs):
    """Through napari's viewer proxy, paging swaps the data of the same two layers."""
    viewer = make_napari_viewer()
    _dock, w = viewer.window.add_plugin_dock_widget("napari-germquant", "Training")
    assert w.open_pairs(pairs) == 8
    assert w.next_pair() and w.next_pair() and w.previous_pair()
    assert len(viewer.layers) == 2 and w.index == 1
    assert np.array_equal(viewer.layers[1].data, T.read_pair(w.pairs[1])[1])


def test_undo_after_paging_does_not_carry_a_stroke_into_the_next_pair(make_napari_viewer, pairs):
    """napari keeps a Labels layer's undo history when its data is replaced; the dock resets it on every page,
    so Ctrl+Z after Next cannot paste the previous pair's stroke into this one (and Save cannot record it)."""
    viewer = make_napari_viewer()
    w = TrainingWidget(viewer)
    assert w.open_pairs(pairs) == 8
    lab = w.labels_layer
    lab.brush_size = 3
    lab.paint((2, 2), 55)                                 # recorded in the layer's undo history
    assert w.is_dirty()
    assert w.save_current() is not None
    assert w.next_pair()
    expected = T.read_pair(w.pairs[1])[1]
    lab.undo()
    assert np.array_equal(lab.data, expected) and not w.is_dirty()
    lab.redo()
    assert np.array_equal(lab.data, expected) and not w.is_dirty()
    assert lab.editable
