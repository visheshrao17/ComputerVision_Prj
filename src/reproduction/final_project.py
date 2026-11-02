"""One normalization experiment, compared with preserved baseline evidence."""

import json
from pathlib import Path

import numpy as np

from .artifacts import paired_experiment_identity, validate_saved_run
from .config import CLASSES, ROOT, save_json


def compare_normalization(run_dir: Path, baseline_dir: Path = ROOT / "results/disci2025/runs"):
    """Require the same data/settings, allowing only paper -> ImageNet scaling."""
    run_dir, baseline_dir = Path(run_dir), Path(baseline_dir)
    folders = {
        "new": run_dir,
        "mobile": baseline_dir / "leakage_audited/MobileNetV2",
        "xception": baseline_dir / "leakage_audited/Xception",
    }
    for name, folder in folders.items():
        if not validate_saved_run(folder):
            raise ValueError(f"Incomplete or inconsistent {name} evidence: {folder}")
    metadata = {
        name: json.loads((folder / "metadata.json").read_text()) for name, folder in folders.items()
    }
    new, mobile, xception = (metadata[name] for name in ("new", "mobile", "xception"))
    if new["smoke"] or new["model"] != "MobileNetV2":
        raise ValueError("A full MobileNetV2 run is required")
    if new["config"]["input_scaling"] != "imagenet":
        raise ValueError("The extension must use ImageNet input scaling")
    adjusted = {**new, "config": {**new["config"], "input_scaling": "paper"}}
    if not paired_experiment_identity(adjusted, mobile):
        raise ValueError("Only input_scaling may change; data and all other settings must match")
    if not paired_experiment_identity(mobile, xception):
        raise ValueError("The original comparison must have matched data and settings")
    metrics = {
        name: json.loads((folder / "metrics_clean.json").read_text())
        for name, folder in folders.items()
    }
    score = metrics["new"]["macro_f1"]
    margin = 0.02
    gap = metrics["xception"]["macro_f1"] - score
    return {
        "experiment": "MobileNetV2 ImageNet normalization only",
        "accuracy": metrics["new"]["accuracy"],
        "macro_f1": score,
        "original_mobile_macro_f1": metrics["mobile"]["macro_f1"],
        "audited_xception_macro_f1": metrics["xception"]["macro_f1"],
        "macro_f1_improvement": score - metrics["mobile"]["macro_f1"],
        "xception_minus_new_macro_f1": gap,
        "allowed_margin": margin,
        "total_parameters": new["total_parameters"],
        "meets_exploratory_target": bool(
            gap <= margin and new["total_parameters"] < xception["total_parameters"]
        ),
        "same_data_and_other_settings": True,
        "same_recorded_environment_as_original": new["environment"] == mobile["environment"],
        "interpretation": (
            "Exploratory follow-up on a previously evaluated test set, one seed. "
            "The original unsupported hypothesis result remains unchanged. "
            "Hardware/library differences may affect comparisons with the earlier run."
        ),
    }


def predict_image(run_dir: Path, image_path: Path) -> dict:
    """Apply the saved training crop once; normalization lives inside the model."""
    import tensorflow as tf

    from . import models  # noqa: F401 -- register ImageNetScaling before loading
    from .preprocessing import preprocess_image

    run_dir = Path(run_dir)
    if not validate_saved_run(run_dir, require_checkpoint=True):
        raise ValueError("Prediction needs a complete run with its model.keras checkpoint")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata["config"]["class_names"] != CLASSES:
        raise ValueError("Unexpected class order")
    image, crop_info = preprocess_image(image_path, metadata["config"])
    model = tf.keras.models.load_model(run_dir / "model.keras", compile=False)
    if model.count_params() != metadata["total_parameters"]:
        raise ValueError("Checkpoint parameter count differs from its run metadata")
    probabilities = model(image[None], training=False).numpy()
    if (
        probabilities.shape != (1, 4)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("Model did not produce four valid class probabilities")
    return {
        "predicted_class": CLASSES[int(probabilities.argmax())],
        "scores": dict(zip(CLASSES, map(float, probabilities[0]))),
        "crop": crop_info,
        "note": "Model softmax scores are not calibrated diagnostic probabilities.",
    }


def save_comparison(run_dir: Path, results_dir: Path) -> dict:
    result = compare_normalization(run_dir)
    save_json(Path(results_dir) / "comparison.json", result)
    return result
