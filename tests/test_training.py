"""Real tiny-model checks keep integration tests inexpensive."""

import numpy as np
import pytest
import tensorflow as tf

from src.reproduction.config import load_config
from src.reproduction.evaluation import benchmark_model
from src.reproduction.models import build_model
from src.reproduction.preprocessing import augment_image
from src.reproduction.training import fit_and_save, verify_resume


@pytest.mark.parametrize(
    "name", ["MobileNetV2", "Xception", "InceptionV3", "ResNet50", "VGG16", "DenseNet121"]
)
def test_backbone_accepts_paper_input(name):
    config = load_config("configs/disci2025.yaml")
    model = build_model(name, config, weights=None)
    output = model(np.zeros((1, 128, 128, 3), np.float32), training=False).numpy()
    assert output.shape == (1, 4)
    np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6)
    tf.keras.backend.clear_session()


def test_seeded_augmentation_is_stable_and_bounded():
    image = tf.ones((128, 128, 3)) * 0.5
    one = augment_image(image, (42, 7)).numpy()
    two = augment_image(image, (42, 7)).numpy()
    np.testing.assert_array_equal(one, two)
    assert 0.4 <= one.min() <= one.max() <= 0.6


def test_training_checkpoint_roundtrip(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [tf.keras.Input((2,)), tf.keras.layers.Dense(4, activation="softmax")]
    )
    x = np.eye(2, dtype=np.float32).repeat(4, axis=0)
    y = np.tile(np.arange(4), 2)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    before = [w.copy() for w in model.get_weights()]
    config = load_config("configs/disci2025.yaml")
    config["epochs"] = 2
    fit_and_save(model, dataset, None, config, tmp_path)
    import json

    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history["accuracy"]) == len(history["loss"]) == 2
    assert np.isfinite(history["loss"]).all()
    assert any(not np.array_equal(a, b) for a, b in zip(before, model.get_weights()))
    loaded = tf.keras.models.load_model(tmp_path / "model.keras", compile=False)
    np.testing.assert_allclose(model(x).numpy(), loaded(x).numpy(), atol=1e-6)


def test_resume_rejects_changed_data_or_settings(tmp_path):
    import json

    metadata = {"signature": "old", "status": "complete"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mismatch"):
        verify_resume(tmp_path, "changed")


def test_resume_does_not_skip_missing_checkpoint(tmp_path):
    import json

    (tmp_path / "metadata.json").write_text(json.dumps({"signature": "same", "status": "complete"}))
    assert not verify_resume(tmp_path, "same")


def test_benchmark_reports_completed_batch_one_inferences():
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(4)])
    measurements = benchmark_model(model, repeats=3)
    assert measurements["batch_size"] == 1
    assert measurements["n_timed_inferences"] == 3
    assert 0 < measurements["median_inference_ms"] <= measurements["p95_inference_ms"]
