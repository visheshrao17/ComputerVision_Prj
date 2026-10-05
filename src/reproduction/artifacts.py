"""Validate compact evidence before displaying results or resuming training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES, fingerprint
from .evaluation import classification_metrics


def paired_experiment_identity(left: dict, right: dict) -> bool:
    ignored = {"models", "model", "data_dir", "output_dir"}

    def settings(metadata):
        return {k: v for k, v in metadata["config"].items() if k not in ignored}

    return (
        left.get("data_fingerprint") == right.get("data_fingerprint")
        and left.get("data_fingerprint") is not None
        and left.get("protocol") == right.get("protocol")
        and settings(left) == settings(right)
    )


def validate_saved_run(run_dir: Path, require_checkpoint: bool = False) -> bool:
    """Check histories, manifests, ordered predictions and recomputed metrics.

    Checkpoints can be omitted from a small presentation bundle; resuming requires one.
    """
    run_dir = Path(run_dir)
    try:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if metadata["status"] != "complete":
            return False
        if require_checkpoint and not (run_dir / "model.keras").is_file():
            return False
        history = json.loads((run_dir / "history.json").read_text())
        epochs = metadata["config"]["epochs"]
        if not all(key in history for key in ("accuracy", "loss")):
            return False
        if any(
            len(values) != epochs or not np.isfinite(values).all() for values in history.values()
        ):
            return False
        manifests = {}
        tables = {}
        for split, count in metadata["counts"].items():
            table = pd.read_csv(run_dir / f"{split}_manifest.csv")
            if len(table) != count or table.path.duplicated().any():
                return False
            manifests[split] = table[["path", "label", "pixel_hash"]].to_dict("records")
            tables[split] = table
        if fingerprint(manifests) != metadata["data_fingerprint"]:
            return False
        views = ["clean", "transformed"] if metadata["protocol"] == "paper_based" else ["clean"]
        for view in views:
            predictions = pd.read_csv(run_dir / f"predictions_{view}.csv")
            if not predictions.path.equals(tables["test"].path):
                return False
            if not predictions.label.equals(tables["test"].label):
                return False
            measured = classification_metrics(
                predictions.label.to_numpy(),
                predictions[[f"p_{name}" for name in CLASSES]].to_numpy(),
            )
            saved = json.loads((run_dir / f"metrics_{view}.json").read_text())
            for key in (
                "n_images",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
                "macro_roc_auc",
            ):
                if measured[key] is None:
                    if saved[key] is not None:
                        return False
                elif not np.isclose(measured[key], saved[key], atol=1e-8, rtol=1e-7):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
