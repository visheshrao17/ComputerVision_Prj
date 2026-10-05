"""Small configuration and provenance helpers shared by CLI and notebook."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
MODELS = ["MobileNetV2", "Xception", "InceptionV3", "ResNet50", "VGG16", "DenseNet121"]
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | str) -> dict:
    with open(path) as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config["class_names"] != CLASSES:
        raise ValueError(f"Class order must be {CLASSES}")
    if not config["models"] or set(config["models"]) - set(MODELS):
        raise ValueError(f"Models must be selected from {MODELS}")
    if config["protocol"] not in ("paper_based", "leakage_audited"):
        raise ValueError("Unknown protocol")
    for key in ("image_size", "batch_size", "epochs", "learning_rate"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["image_size"] < 75:
        raise ValueError("Six-model input size must be at least 75")
    if config["blur_kernel"] < 1 or config["blur_kernel"] % 2 != 1:
        raise ValueError("blur_kernel must be a positive odd integer")
    if not 0 <= config["threshold"] <= 255:
        raise ValueError("threshold must be in [0, 255]")
    if config["input_scaling"] not in ("paper", "imagenet"):
        raise ValueError("input_scaling must be paper or imagenet")
    if not 0 < config["validation_fraction"] < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")


def environment_metadata() -> dict:
    packages = [
        "tensorflow",
        "tensorflow-metal",
        "keras",
        "numpy",
        "pandas",
        "scikit-learn",
        "opencv-python-headless",
    ]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def fingerprint(value: dict | list) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path: Path | str, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
