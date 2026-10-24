"""Stand-ins for the engine's Cellpose training seams in an engine child process (the seams runner of
tests/_p6_runner.py imports this module there): the net is a token, training logs one Cellpose epoch line and
writes a small "model" file, and prediction answers with the labels of the training folder named by --pairs
(perfect recall), as the plugin's in-process fake_cellpose fixture does. Nothing is trained or loaded for real.
Standard library and numpy only (tifffile on use): this module never imports napari, Qt or the plugin."""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

CALLS: dict = {"build": [], "train": [], "predict": []}
_LOOKUP: dict = {}


def _key(img) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(np.asarray(img, dtype=np.float32)).tobytes()).hexdigest()


def _pairs_dir() -> Path | None:
    args = sys.argv
    return Path(args[args.index("--pairs") + 1]) if "--pairs" in args and args.index("--pairs") + 1 < len(args) \
        else None


def build_net(base, device):
    CALLS["build"].append((dict(base), device))
    return "net-token"


def train_seg(net, **kw):
    import numpy as np

    CALLS["train"].append(sorted(kw))
    logging.getLogger("cellpose.train").info("0, train_loss=0.5000, test_loss=0.6000, LR=0.000010, time 0.01s")
    p = Path(kw["save_path"]) / "models" / kw["model_name"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake cellpose weights " * 64)
    n = int(kw["n_epochs"])
    return p, np.linspace(1.0, 0.5, n), np.linspace(0.9, 0.6, n)


def predict(model_path, images, device):
    import numpy as np
    import tifffile

    CALLS["predict"].append((str(model_path), len(images), device))
    if not _LOOKUP and _pairs_dir() is not None:
        from germquant.segment.training import load_pairs

        for p in load_pairs(_pairs_dir()):
            _LOOKUP[_key(tifffile.imread(p["image"]))] = np.asarray(tifffile.imread(p["labels"])).astype(np.int32)
    return [_LOOKUP.get(_key(img), np.zeros(np.shape(img), np.int32)) for img in images]
