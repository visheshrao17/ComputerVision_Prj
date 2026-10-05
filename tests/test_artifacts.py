import json

import numpy as np
import pandas as pd

from scripts.reproduce_disci2025 import refresh_results
from src.reproduction.artifacts import paired_experiment_identity, validate_saved_run
from src.reproduction.config import CLASSES, fingerprint, load_config


def compact_run(folder):
    folder.mkdir(parents=True)
    manifests = {}
    for split in ["train", "test"]:
        table = pd.DataFrame(
            {
                "path": [f"{split}/{i}.png" for i in range(4)],
                "label": range(4),
                "pixel_hash": [f"{split}-{i}" for i in range(4)],
            }
        )
        table.to_csv(folder / f"{split}_manifest.csv", index=False)
        manifests[split] = table.to_dict("records")
    predictions = pd.DataFrame({"path": [f"test/{i}.png" for i in range(4)], "label": range(4)})
    for index, name in enumerate(CLASSES):
        predictions[f"p_{name}"] = np.eye(4)[:, index]
    predictions.to_csv(folder / "predictions_clean.csv", index=False)
    (folder / "history.json").write_text(json.dumps({"accuracy": [0.25] * 5, "loss": [1.0] * 5}))
    (folder / "metrics_clean.json").write_text(
        json.dumps(
            {
                "n_images": 4,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "balanced_accuracy": 1.0,
                "macro_roc_auc": 1.0,
            }
        )
    )
    config = load_config("configs/disci2025.yaml")
    config["protocol"] = "leakage_audited"
    config = {k: v for k, v in config.items() if k not in ["models", "data_dir", "output_dir"]}
    metadata = {
        "status": "complete",
        "smoke": False,
        "model": "MobileNetV2",
        "protocol": "leakage_audited",
        "config": config,
        "data_fingerprint": fingerprint(manifests),
        "counts": {"train": 4, "test": 4},
        "total_parameters": 10,
        "training_seconds": 1.0,
        "environment": {"devices": ["CPU"]},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_compact_evidence_is_analyzable_without_heavy_checkpoint(tmp_path):
    folder = tmp_path / "leakage_audited/MobileNetV2"
    compact_run(folder)
    assert validate_saved_run(folder)
    assert not validate_saved_run(folder, require_checkpoint=True)
    assert len(refresh_results(tmp_path, tmp_path / "results")) == 1
    exported = tmp_path / "results/runs/leakage_audited/MobileNetV2"
    assert validate_saved_run(exported)
    assert not (exported / "model.keras").exists()


def test_incomplete_history_cannot_be_published(tmp_path):
    folder = tmp_path / "leakage_audited/MobileNetV2"
    compact_run(folder)
    (folder / "history.json").write_text(json.dumps({"accuracy": [0.2], "loss": [1.0]}))
    assert not validate_saved_run(folder)
    assert refresh_results(tmp_path, tmp_path / "results").empty


def test_missing_or_changed_predictions_cannot_be_published(tmp_path):
    folder = tmp_path / "leakage_audited/MobileNetV2"
    compact_run(folder)
    path = folder / "predictions_clean.csv"
    saved = pd.read_csv(path)
    saved.loc[0, "label"] = 1
    saved.to_csv(path, index=False)
    assert not validate_saved_run(folder)
    path.unlink()
    assert refresh_results(tmp_path, tmp_path / "results").empty


def test_pair_rejects_equal_counts_with_changed_data_or_normalization(tmp_path):
    metadata = compact_run(tmp_path / "one")
    other = json.loads(json.dumps(metadata))
    other["model"] = "Xception"
    assert paired_experiment_identity(metadata, other)
    other["data_fingerprint"] = "different"
    assert not paired_experiment_identity(metadata, other)
    other["data_fingerprint"] = metadata["data_fingerprint"]
    other["config"]["input_scaling"] = "imagenet"
    assert not paired_experiment_identity(metadata, other)
