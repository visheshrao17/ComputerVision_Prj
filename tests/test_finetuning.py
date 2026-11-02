"""Small real models test selection and BatchNorm without claiming MRI results."""

import json

import numpy as np
import pytest
import tensorflow as tf

from src.reproduction.artifacts import validate_saved_run
from src.reproduction.config import load_config
from src.reproduction.finetuning import (
    ValidationCheckpoint,
    fit_with_validation,
    set_backbone_trainable,
)
from src.reproduction.models import build_model
from src.reproduction.training_comparison import (
    check_extension_identity,
    comparison_table,
    extension_config,
)
from tests.test_artifacts import compact_run


def tiny_model(size=4):
    inputs = tf.keras.Input((size, size, 3))
    x = tf.keras.layers.Conv2D(2, 1)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    backbone = tf.keras.Model(inputs, x, name="tiny_backbone")
    outer = tf.keras.Input((size, size, 3))
    x = backbone(outer, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(outer, outputs), backbone


def test_proposed_head_is_smaller_and_saved_batchnorm_stays_in_inference_mode(tmp_path):
    control = build_model("MobileNetV2", extension_config("control"), weights=None)
    proposed = build_model("MobileNetV2", extension_config("proposed"), weights=None)
    assert control.count_params() == 4_880_068
    assert proposed.count_params() < control.count_params()
    backbone = next(layer for layer in proposed.layers if isinstance(layer, tf.keras.Model))
    set_backbone_trainable(proposed, True, freeze_batchnorm=True)
    bn = next(
        layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
    )
    before = bn.moving_mean.numpy().copy()
    image = np.ones((2, 128, 128, 3), dtype=np.float32)
    proposed(image, training=True)
    np.testing.assert_array_equal(before, bn.moving_mean.numpy())
    proposed.save(tmp_path / "proposed.keras")
    loaded = tf.keras.models.load_model(tmp_path / "proposed.keras", compile=False)
    np.testing.assert_allclose(
        proposed(image, training=False), loaded(image, training=False), atol=1e-6
    )
    tf.keras.backend.clear_session()


def test_checkpoint_keeps_best_epoch_even_when_later_epoch_is_worse(tmp_path):
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(4)])
    callback = ValidationCheckpoint(tmp_path, patience=2)
    callback.set_model(model)
    best_weights = None
    for epoch, loss in enumerate([0.8, 0.3, 0.5, 0.6]):
        model.set_weights([np.full_like(w, epoch + 1) for w in model.get_weights()])
        if epoch == 1:
            best_weights = model.get_weights()
        callback.on_epoch_end(
            epoch, {"loss": loss, "accuracy": 0.5, "val_loss": loss, "val_accuracy": 0.5}
        )
    assert callback.best_epoch == 2
    assert model.stop_training
    restored = tf.keras.models.load_model(tmp_path / "model.keras", compile=False)
    for saved, expected in zip(restored.get_weights(), best_weights):
        np.testing.assert_array_equal(saved, expected)


def test_two_stage_fit_updates_features_but_preserves_batchnorm_statistics(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model, backbone = tiny_model()
    conv, bn = backbone.layers[1:]
    before_kernel = conv.kernel.numpy().copy()
    before_mean = bn.moving_mean.numpy().copy()
    x = np.random.default_rng(42).normal(size=(8, 4, 4, 3)).astype(np.float32)
    y = np.arange(8, dtype=np.int32) % 4
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    config = extension_config("proposed")
    config.update(epochs=3, warmup_epochs=1, finetune_learning_rate=0.001)
    result = fit_with_validation(model, dataset, dataset, config, tmp_path)
    history = json.loads((tmp_path / "history.json").read_text())
    assert result["epochs_completed"] == 3
    assert [stage["epochs"] for stage in result["stages"]] == [1, 2]
    assert result["selected_epoch"] == int(np.argmin(history["val_loss"])) + 1
    assert not np.array_equal(before_kernel, conv.kernel.numpy())
    np.testing.assert_array_equal(before_mean, bn.moving_mean.numpy())
    assert not bn.trainable


def test_evidence_accepts_early_stopping_but_rejects_wrong_selected_epoch(tmp_path):
    metadata = compact_run(tmp_path / "run")
    config = extension_config("control")
    metadata["config"] = {
        k: v for k, v in config.items() if k not in ("models", "data_dir", "output_dir")
    }
    metadata.update(
        epochs_completed=3,
        selected_epoch=2,
        best_val_loss=0.2,
        checkpoint_selection="minimum_val_loss",
    )
    history = {
        "accuracy": [0.5] * 3,
        "loss": [0.8] * 3,
        "val_accuracy": [0.5] * 3,
        "val_loss": [0.8, 0.2, 0.3],
    }
    (tmp_path / "run/history.json").write_text(json.dumps(history))
    path = tmp_path / "run/metadata.json"
    path.write_text(json.dumps(metadata))
    assert validate_saved_run(path.parent)
    metadata["selected_epoch"] = 3
    path.write_text(json.dumps(metadata))
    assert not validate_saved_run(path.parent)


def test_approved_comparison_rejects_changed_data_or_extra_hyperparameters():
    metadata = {
        "model": "MobileNetV2",
        "smoke": False,
        "data_fingerprint": "same",
        "config": extension_config("proposed"),
    }
    reference = {"data_fingerprint": "same"}
    check_extension_identity(metadata, "proposed", reference)
    metadata["config"]["image_size"] = 224
    with pytest.raises(ValueError, match="settings"):
        check_extension_identity(metadata, "proposed", reference)
    metadata["config"] = extension_config("proposed")
    metadata["data_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="split"):
        check_extension_identity(metadata, "proposed", reference)


def test_unrun_experiments_are_pending_not_failed_or_measured(tmp_path, monkeypatch):
    import src.reproduction.training_comparison as comparison
    from src.reproduction.config import ROOT

    # Keep real historical controls but ensure there are no local extension results.
    source = ROOT
    fake = tmp_path / "project"
    for name in ("results/disci2025", "results/final_project"):
        target = fake / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source / name, target_is_directory=True)
    monkeypatch.setattr(comparison, "ROOT", fake)
    table = comparison_table(tmp_path / "outputs")
    pending = table[table.experiment.isin(["control", "proposed"])]
    assert (pending.status == "pending").all()
    assert pending.macro_f1.isna().all()
    assert pending.improves_normalized_with_fewer_parameters.isna().all()


def test_validation_policy_cannot_be_used_without_a_validation_split():
    from src.reproduction.config import validate_config

    config = load_config("configs/training_control.yaml")
    config["protocol"] = "paper_based"
    with pytest.raises(ValueError, match="audited"):
        validate_config(config)


def test_new_training_path_exports_valid_evidence_and_resumes_only_complete_runs(
    tmp_path, monkeypatch
):
    import cv2
    import pandas as pd

    import src.reproduction.models as models
    from src.reproduction.config import CLASSES
    from src.reproduction.training import train_model

    monkeypatch.setattr(models, "build_model", lambda *args: tiny_model(128)[0])
    data = tmp_path / "images"
    data.mkdir()
    manifests = {}
    for split in ["train", "val", "test"]:
        rows = []
        for label, name in enumerate(CLASSES):
            path = f"{split}-{label}.png"
            image = np.full((16, 16), 60 + label * 40, dtype=np.uint8)
            assert cv2.imwrite(str(data / path), image)
            rows.append(
                {
                    "path": path,
                    "label": label,
                    "class_name": name,
                    "pixel_hash": f"synthetic-{split}-{label}",
                }
            )
        manifests[split] = pd.DataFrame(rows)
    config = extension_config("proposed")
    config.update(data_dir=str(data), epochs=2, warmup_epochs=1)
    run = tmp_path / "run"
    checkpoint = train_model("MobileNetV2", manifests, config, run, smoke=True)
    assert checkpoint.exists()
    assert validate_saved_run(run, require_checkpoint=True)
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["smoke"] is True  # Never presented as a research result.
    assert metadata["epochs_completed"] == 2
    monkeypatch.setattr(models, "build_model", lambda *args: pytest.fail("Complete run retrained"))
    assert train_model("MobileNetV2", manifests, config, run, resume=True, smoke=True) == checkpoint
