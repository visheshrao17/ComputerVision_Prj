"""Held-out metrics and clearly labeled published comparison targets."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .config import CLASSES, save_json

PAPER = pd.DataFrame(
    [
        ["Xception", 0.9952, 0.9527, 0.9491, 0.9529, 0.9873],
        ["MobileNetV2", 0.9898, 0.9451, 0.9411, 0.9457, 0.9815],
        ["InceptionV3", 0.9810, 0.9451, 0.9409, 0.9452, 0.9743],
        ["ResNet50", 0.9897, 0.9062, 0.9010, 0.9045, 0.9741],
        ["VGG16", 0.9721, 0.9504, 0.9470, 0.9478, 0.9680],
        ["DenseNet121", 0.9652, 0.9285, 0.9220, 0.9265, 0.9583],
    ],
    columns=[
        "model",
        "paper_train_accuracy",
        "paper_test_accuracy",
        "paper_macro_f1",
        "paper_weighted_f1",
        "paper_combined_accuracy",
    ],
)


def benchmark_model(model, repeats: int = 20) -> dict:
    """Synchronize completed batch-1 calls; exclude I/O and preprocessing."""
    import tensorflow as tf

    if repeats < 1:
        raise ValueError("repeats must be positive")
    sample = tf.zeros((1, *model.input_shape[1:]), dtype=tf.float32)

    @tf.function(jit_compile=False)
    def infer(image):
        return model(image, training=False)

    for _ in range(5):
        infer(sample).numpy()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        infer(sample).numpy()  # Force GPU completion, not just queue submission.
        elapsed.append((time.perf_counter() - start) * 1000)
    return {
        "batch_size": 1,
        "n_timed_inferences": repeats,
        "median_inference_ms": float(np.median(elapsed)),
        "p95_inference_ms": float(np.percentile(elapsed, 95)),
    }


def combined_accuracy(train: float, test: float, n_train: int, n_test: int) -> float:
    return (n_train * train + n_test * test) / (n_train + n_test)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities)
    if (
        not len(y_true)
        or probabilities.shape != (len(y_true), 4)
        or not np.isfinite(probabilities).all()
        or probabilities.min() < 0
        or probabilities.max() > 1
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        or set(y_true) - set(range(4))
    ):
        raise ValueError("Expected nonempty labels 0..3 and finite (N,4) probability distributions")
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, predictions, labels=range(4))
    details = {}
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        target = y_true == index
        auc = (
            float(roc_auc_score(target, probabilities[:, index]))
            if 0 < target.sum() < len(target)
            else None
        )
        details[name] = {
            "support": tp + fn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "f1": 2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else 0.0,
            "roc_auc": auc,
        }
    recalls = [item["recall"] for item in details.values() if item["recall"] is not None]
    aucs = [item["roc_auc"] for item in details.values()]
    return {
        "n_images": len(y_true),
        "accuracy": float(np.mean(y_true == predictions)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(y_true, predictions, labels=range(4), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predictions, labels=range(4), average="weighted", zero_division=0)
        ),
        "macro_roc_auc": float(np.mean(aucs)) if all(a is not None for a in aucs) else None,
        "confusion_matrix": matrix.tolist(),
        "per_class": details,
    }


def evaluate_model(
    checkpoint: Path, manifest: pd.DataFrame, config: dict, evaluation_view: str = "clean"
) -> tuple[pd.DataFrame, dict]:
    import tensorflow as tf

    from . import models  # noqa: F401 -- register the serializable scaling layer
    from .preprocessing import make_dataset

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    dataset = make_dataset(manifest, config, "test", evaluation_view)
    # Calling each batch directly avoids graph prediction/thread-pool overhead.
    probabilities = np.concatenate([model(x, training=False).numpy() for x, _ in dataset])
    predictions = manifest[["path", "label", "class_name"]].copy()
    predictions["predicted_label"] = probabilities.argmax(axis=1)
    for index, name in enumerate(CLASSES):
        predictions[f"p_{name}"] = probabilities[:, index]
    return predictions, classification_metrics(manifest.label.to_numpy(), probabilities)


def compare_with_paper(measured: pd.DataFrame) -> pd.DataFrame:
    comparison = measured.merge(PAPER, on="model", how="left", validate="many_to_one")
    comparison["test_accuracy_difference_pp"] = 100 * (
        comparison.accuracy - comparison.paper_test_accuracy
    )
    comparison["macro_f1_difference_pp"] = 100 * (comparison.macro_f1 - comparison.paper_macro_f1)
    return comparison


def plot_results(run_dir: Path) -> None:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    history = json.loads((run_dir / "history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, title, axis in [("accuracy", "Accuracy", axes[0]), ("loss", "Loss", axes[1])]:
        axis.plot(range(1, len(history[key]) + 1), history[key], "o-", label="training")
        if f"val_{key}" in history:
            axis.plot(
                range(1, len(history[key]) + 1), history[f"val_{key}"], "o-", label="validation"
            )
        axis.set(xlabel="Epoch", title=title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    metrics = json.loads((run_dir / "metrics_clean.json").read_text())
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.imshow(matrix, cmap="Blues")
    for i in range(4):
        for j in range(4):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    axis.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title="Clean held-out test confusion matrix",
    )
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def export_evaluation(run_dir: Path, manifest: pd.DataFrame, config: dict, view: str) -> dict:
    predictions, metrics = evaluate_model(run_dir / "model.keras", manifest, config, view)
    predictions.to_csv(run_dir / f"predictions_{view}.csv", index=False)
    save_json(run_dir / f"metrics_{view}.json", metrics)
    return metrics
