"""Protect the controlled comparison and fresh-image inference contract."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.reproduction.final_project import compare_normalization, predict_image
from src.reproduction.models import ImageNetScaling

BASE = Path("results/disci2025/runs/leakage_audited/MobileNetV2")


@pytest.fixture
def proposed_run(tmp_path):
    folder = tmp_path / "run"
    shutil.copytree(BASE, folder)
    meta = json.loads((folder / "metadata.json").read_text())
    meta["config"]["input_scaling"] = "imagenet"
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


def test_comparison_does_not_turn_unchanged_predictions_into_improvement(proposed_run):
    outcome = compare_normalization(proposed_run)
    assert outcome["macro_f1_improvement"] == 0
    assert outcome["meets_exploratory_target"] is False


def test_comparison_rejects_other_training_changes(proposed_run):
    path = proposed_run / "metadata.json"
    meta = json.loads(path.read_text())
    meta["config"]["learning_rate"] *= 10
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Only input_scaling"):
        compare_normalization(proposed_run)


def test_scaling_survives_save_load_without_double_normalization(tmp_path):
    inputs = tf.keras.Input((1, 1, 3))
    model = tf.keras.Model(inputs, ImageNetScaling("MobileNetV2")(inputs))
    values = np.array([[[[0, 0.5, 1]]]], dtype=np.float32)
    np.testing.assert_allclose(model(values), [[[[-1, 0, 1]]]], atol=1e-6)
    path = tmp_path / "scaling.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(restored(values), model(values), atol=1e-6)


def test_prediction_requires_exported_weights(proposed_run, tmp_path):
    with pytest.raises(ValueError, match="model.keras"):
        predict_image(proposed_run, tmp_path / "not_needed.png")


def test_prediction_roundtrip_uses_training_preprocessing(proposed_run, tmp_path):
    import cv2

    from src.reproduction.preprocessing import preprocess_image

    inputs = tf.keras.Input((128, 128, 3))
    features = tf.keras.layers.GlobalAveragePooling2D()(ImageNetScaling("MobileNetV2")(inputs))
    outputs = tf.keras.layers.Dense(4, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.save(proposed_run / "model.keras")
    path = proposed_run / "metadata.json"
    meta = json.loads(path.read_text())
    meta["total_parameters"] = model.count_params()
    path.write_text(json.dumps(meta))
    image_path = tmp_path / "slice.png"
    cv2.imwrite(str(image_path), np.full((180, 180), 180, dtype=np.uint8))
    expected_image, _ = preprocess_image(image_path, meta["config"])
    expected = model(expected_image[None], training=False).numpy()[0]
    result = predict_image(proposed_run, image_path)
    np.testing.assert_allclose(list(result["scores"].values()), expected, atol=1e-6)
    assert result["predicted_class"] == meta["config"]["class_names"][expected.argmax()]
