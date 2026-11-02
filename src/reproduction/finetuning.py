"""Validation-selected training; the paper's final-epoch path stays unchanged."""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import save_json
from .evaluation import benchmark_model


def set_backbone_trainable(model, trainable: bool, freeze_batchnorm: bool) -> None:
    """The only nested model is the ImageNet backbone. Recompile after calling."""
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    backbone.trainable = trainable
    if trainable and freeze_batchnorm:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


class ValidationCheckpoint(tf.keras.callbacks.Callback):
    """Keep the global best checkpoint/history across warm-up and fine-tuning."""

    def __init__(self, run_dir: Path, patience: int):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.patience = patience
        self.history = {}
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.wait = 0
        self.stage_best = float("inf")
        self.early_stopping = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "val_loss" not in logs or not all(np.isfinite(value) for value in logs.values()):
            raise ValueError("Finite validation loss is required for checkpoint selection")
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        save_json(self.run_dir / "history.json", self.history)
        loss = float(logs["val_loss"])
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = len(self.history["loss"])
            self.model.save(self.run_dir / "model.keras")
        if loss < self.stage_best:
            self.stage_best, self.wait = loss, 0
        else:
            self.wait += 1
        if self.early_stopping and self.wait >= self.patience:
            self.model.stop_training = True


def fit_with_validation(model, training, validation, config: dict, run_dir: Path) -> dict:
    """Fresh ImageNet run: optional fixed warm-up, then validation-stopped training.

    Epochs is the total budget, including warm-up. Selection uses the minimum
    validation loss across both stages; the test set is never passed here.
    """
    if validation is None:
        raise ValueError("Validation-selected training requires a validation dataset")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = ValidationCheckpoint(run_dir, config["patience"])
    warmup = config["warmup_epochs"]
    phases = []
    if warmup:
        phases.append(("warmup", warmup, config["learning_rate"], False))
    phases.append(("finetune", config["epochs"] - warmup, config["finetune_learning_rate"], True))
    stages = []
    started = time.perf_counter()
    for name, budget, rate, trainable in phases:
        set_backbone_trainable(model, trainable, config["freeze_batchnorm"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=False,
        )
        recorder.wait, recorder.stage_best = 0, float("inf")
        recorder.early_stopping = name != "warmup"
        previous = len(recorder.history.get("loss", []))
        print(f"Stage: {name}; maximum {budget} epochs; learning rate={rate:g}", flush=True)
        model.fit(
            training,
            validation_data=validation,
            epochs=budget,
            callbacks=[recorder],
            verbose=2,
            shuffle=False,
        )
        stages.append(
            {
                "name": name,
                "epochs": len(recorder.history["loss"]) - previous,
                "learning_rate": rate,
                "backbone_trainable": trainable,
            }
        )
    elapsed = time.perf_counter() - started
    selected = tf.keras.models.load_model(run_dir / "model.keras", compile=False)
    return {
        "training_seconds": elapsed,
        "total_parameters": selected.count_params(),
        "trainable_parameters": int(sum(np.prod(w.shape) for w in selected.trainable_weights)),
        "epochs_completed": len(recorder.history["loss"]),
        "selected_epoch": recorder.best_epoch,
        "best_val_loss": recorder.best_loss,
        "checkpoint_selection": "minimum_val_loss",
        "stages": stages,
        **benchmark_model(selected),
    }
