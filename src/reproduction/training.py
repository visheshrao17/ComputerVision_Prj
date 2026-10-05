"""One model at a time; final-epoch checkpoints, with auditable run metadata."""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import environment_metadata, fingerprint, save_json


def configure_runtime(seed: int, cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir).resolve()
    os.environ.setdefault("KERAS_HOME", str(cache_dir / "keras"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    # Metal's reproducibility can differ from CUDA/CPU despite fixed random seeds.
    devices = tf.config.list_physical_devices()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # A notebook may already have initialized its runtime.
    return {
        **environment_metadata(),
        "devices": [str(device) for device in devices],
        "determinism": "Fixed seeds and deterministic data order; GPU bitwise equivalence not guaranteed.",
    }


def fit_and_save(model, train_dataset, val_dataset, config: dict, run_dir: Path) -> dict:
    import tensorflow as tf

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )

    class RecordHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.records = {}

        def on_epoch_end(self, epoch, logs=None):
            for key, value in logs.items():
                self.records.setdefault(key, []).append(float(value))
            save_json(run_dir / "history.json", self.records)

    started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["epochs"],
        verbose=2,
        shuffle=False,
        callbacks=[RecordHistory()],
    )
    elapsed = time.perf_counter() - started
    for key in ("accuracy", "loss"):
        values = history.history.get(key, [])
        if len(values) != config["epochs"] or not np.isfinite(values).all():
            raise ValueError(f"Incomplete/nonfinite {key} history; training run rejected")
    model.save(run_dir / "model.keras")
    save_json(
        run_dir / "history.json",
        {key: list(map(float, values)) for key, values in history.history.items()},
    )
    from .evaluation import benchmark_model

    return {
        "training_seconds": elapsed,
        "total_parameters": model.count_params(),
        "trainable_parameters": int(
            sum(np.prod(weight.shape) for weight in model.trainable_weights)
        ),
        **benchmark_model(model),
    }


def verify_resume(run_dir: Path, signature: str) -> bool:
    metadata_path = Path(run_dir) / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata["signature"] != signature:
        raise ValueError(
            f"Resume mismatch at {run_dir}; use a new output directory for changed settings/data"
        )
    from .artifacts import validate_saved_run

    return validate_saved_run(run_dir, require_checkpoint=True)


def train_model(
    name: str,
    manifests: dict[str, pd.DataFrame],
    config: dict,
    run_dir: Path,
    resume: bool = False,
    smoke: bool = False,
) -> Path:
    import tensorflow as tf

    from .evaluation import combined_accuracy, export_evaluation, plot_results
    from .models import build_model
    from .preprocessing import make_dataset

    run_dir = Path(run_dir)
    portable_config = {
        key: value
        for key, value in config.items()
        if key not in ("data_dir", "output_dir", "models")
    }
    manifest_identity = {
        split: table[["path", "label", "pixel_hash"]].to_dict("records")
        for split, table in manifests.items()
    }
    signature = fingerprint(
        {"config": portable_config, "data": manifest_identity, "model": name, "smoke": smoke}
    )
    if resume and verify_resume(run_dir, signature):
        print(f"{name}: verified complete run, skipping", flush=True)
        return run_dir / "model.keras"
    if run_dir.exists() and (run_dir / "metadata.json").exists() and not resume:
        raise ValueError(
            f"Run already exists: {run_dir}; use --resume or a different output directory"
        )
    tf.keras.backend.clear_session()
    runtime = configure_runtime(config["seed"], Path(".cache"))
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": name,
        "protocol": config["protocol"],
        "signature": signature,
        "status": "training",
        "smoke": smoke,
        "config": portable_config,
        "environment": runtime,
        "counts": {key: len(table) for key, table in manifests.items()},
        "data_fingerprint": fingerprint(manifest_identity),
        "patient_independent": "patient_id" in manifests["train"].columns
        and config["protocol"] == "leakage_audited",
    }
    for split, table in manifests.items():
        table.to_csv(run_dir / f"{split}_manifest.csv", index=False)
    save_json(run_dir / "metadata.json", metadata)
    model = build_model(name, config)
    training = make_dataset(manifests["train"], config, "train")
    validation = make_dataset(manifests["val"], config, "val") if "val" in manifests else None
    print(
        f"{name}: {len(manifests['train'])} training images; {model.count_params():,} parameters; "
        f"{config['epochs']} epochs; devices={runtime['devices']}",
        flush=True,
    )
    metadata.update(fit_and_save(model, training, validation, config, run_dir))
    del model, training, validation
    tf.keras.backend.clear_session()
    gc.collect()
    metadata["status"] = "evaluating"
    save_json(run_dir / "metadata.json", metadata)
    clean = export_evaluation(run_dir, manifests["test"], config, "clean")
    print(
        f"{name}: clean test accuracy={clean['accuracy']:.4f}, macro-F1={clean['macro_f1']:.4f}",
        flush=True,
    )
    if config["protocol"] == "paper_based":
        transformed = export_evaluation(run_dir, manifests["test"], config, "transformed")
        # Evaluation dataset always uses split='test', preserving order and fixed transformations.
        from .evaluation import evaluate_model

        _, train_metrics = evaluate_model(
            run_dir / "model.keras", manifests["train"], config, "transformed"
        )
        save_json(run_dir / "metrics_train_transformed.json", train_metrics)
        metadata["combined_train_test_accuracy"] = combined_accuracy(
            train_metrics["accuracy"],
            transformed["accuracy"],
            len(manifests["train"]),
            len(manifests["test"]),
        )
    tf.keras.backend.clear_session()
    gc.collect()
    plot_results(run_dir)
    metadata["status"] = "complete"
    save_json(run_dir / "metadata.json", metadata)
    return run_dir / "model.keras"
