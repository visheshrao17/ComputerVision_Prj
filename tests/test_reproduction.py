"""Small fixtures exercise data leakage and metrics, not model accuracy."""

from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import pytest

from src.reproduction.config import load_config, validate_config
from src.reproduction.data import audit_duplicates, build_manifests, inventory_dataset
from src.reproduction.evaluation import classification_metrics, combined_accuracy
from src.reproduction.preprocessing import preprocess_image


def test_invalid_experiments_are_rejected():
    config = load_config("configs/disci2025.yaml")
    for key, value in [
        ("models", ["unknown"]),
        ("protocol", "unknown"),
        ("class_names", ["glioma", "pituitary", "no_tumor", "meningioma"]),
    ]:
        bad = deepcopy(config)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_inventory_alias_and_corrupt_file(tmp_path):
    folder = tmp_path / "Training" / "notumor"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "image.png"), np.full((16, 16), 120, np.uint8))
    table = inventory_dataset(tmp_path, require_counts=False)
    assert table.iloc[0]["class_name"] == "no_tumor"
    assert table.iloc[0]["label"] == 2
    (folder / "broken.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="broken.jpg"):
        inventory_dataset(tmp_path, require_counts=False)


def test_wrong_dataset_counts_stop_paper_comparison(tmp_path):
    with pytest.raises(ValueError, match="counts"):
        inventory_dataset(tmp_path)


def test_pixel_duplicates_survive_encoding_changes(tmp_path):
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for partition, extension in [("Training", "png"), ("Testing", "bmp")]:
        folder = tmp_path / partition / "glioma"
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / f"image.{extension}"), image)
    records = inventory_dataset(tmp_path, require_counts=False)
    audit = audit_duplicates(records)
    assert audit["cross_split_exact_count"] == 1
    assert records.pixel_hash.nunique() == 1


def fixture_records():
    rows = []
    for label, name in enumerate(["glioma", "meningioma", "no_tumor", "pituitary"]):
        for i in range(12):
            rows.append(
                dict(
                    path=f"{name}/{i}.png",
                    class_name=name,
                    label=label,
                    original_partition="train",
                    pixel_hash=f"{name}-{i}",
                    phash="0",
                    patient_id=f"{name}-{i // 2}",
                )
            )
        rows.append(
            dict(
                path=f"{name}/test.png",
                class_name=name,
                label=label,
                original_partition="test",
                pixel_hash=f"{name}-0",
                phash="0",
                patient_id=f"{name}-0",
            )
        )
    return pd.DataFrame(rows)


def test_audited_split_excludes_test_duplicates_and_is_reproducible():
    records = fixture_records().drop(columns="patient_id")
    one = build_manifests(records, "leakage_audited", 42)
    two = build_manifests(records, "leakage_audited", 42)
    assert len(one["test"]) == 4
    assert len(one["train"]) + len(one["val"]) == 44
    assert set(one["train"].pixel_hash).isdisjoint(one["test"].pixel_hash)
    assert set(one["val"].pixel_hash).isdisjoint(one["test"].pixel_hash)
    assert one["train"].path.tolist() == two["train"].path.tolist()


def test_patient_groups_do_not_cross_partitions():
    splits = build_manifests(fixture_records(), "leakage_audited", 42)
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert set(splits[left].patient_id).isdisjoint(splits[right].patient_id)


def test_duplicate_conflicting_labels_are_rejected():
    records = fixture_records().drop(columns="patient_id")
    records.loc[1, "pixel_hash"] = records.loc[13, "pixel_hash"]
    with pytest.raises(ValueError, match="Conflicting"):
        build_manifests(records, "leakage_audited", 42)


def test_empty_contour_falls_back_to_full_slice(tmp_path):
    path = tmp_path / "black.png"
    cv2.imwrite(str(path), np.zeros((24, 32), np.uint8))
    image, info = preprocess_image(path, load_config("configs/disci2025.yaml"))
    assert image.shape == (128, 128, 3)
    assert info["crop_fallback"]
    assert not image.any()


def test_preprocessing_preserves_grayscale_and_range(tmp_path):
    path = tmp_path / "brain.png"
    image = np.zeros((40, 40), np.uint8)
    image[10:30, 10:30] = 180
    cv2.imwrite(str(path), image)
    output, info = preprocess_image(path, load_config("configs/disci2025.yaml"))
    assert output.dtype == np.float32
    assert output.min() >= 0 and output.max() <= 1
    np.testing.assert_array_equal(output[..., 0], output[..., 2])
    assert not info["crop_fallback"]
    with pytest.raises(ValueError):
        preprocess_image(tmp_path / "missing.jpg", load_config("configs/disci2025.yaml"))


def test_four_class_metrics_hand_calculated():
    # CM: [[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    truth = np.array([0, 0, 1, 2, 3])
    probs = np.eye(4)[[0, 1, 1, 2, 3]]
    metrics = classification_metrics(truth, probs)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["balanced_accuracy"] == pytest.approx(0.875)
    assert metrics["macro_f1"] == pytest.approx(5 / 6)
    assert metrics["weighted_f1"] == pytest.approx(0.8)
    assert metrics["per_class"]["meningioma"]["specificity"] == pytest.approx(0.75)
    assert combined_accuracy(0.9952, 0.9527, 5712, 1311) == pytest.approx(0.9872664246)


def test_missing_class_auc_is_explicitly_undefined():
    metrics = classification_metrics(np.array([0, 1]), np.eye(4)[[0, 1]])
    assert np.asarray(metrics["confusion_matrix"]).shape == (4, 4)
    assert metrics["macro_roc_auc"] is None
    assert metrics["per_class"]["pituitary"]["roc_auc"] is None
    assert metrics["per_class"]["pituitary"]["recall"] is None


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        classification_metrics(np.array([0]), np.array([[np.nan, 0, 0, 1]]))


def test_validation_fraction_controls_holdout():
    records = pd.DataFrame(
        [
            {
                "path": f"{split}/{label}/{i}.png",
                "label": label,
                "original_partition": split,
                "pixel_hash": f"{split}-{label}-{i}",
            }
            for split, size in [("train", 40), ("test", 4)]
            for label in range(4)
            for i in range(size)
        ]
    )
    small = build_manifests(records, "leakage_audited", 42, validation_fraction=0.15)
    large = build_manifests(records, "leakage_audited", 42, validation_fraction=0.25)
    assert len(small["val"]) == 24
    assert len(large["val"]) == 40
    assert set(large["train"].path).isdisjoint(large["val"].path)


def test_crop_fallback_audit_records_paths(tmp_path):
    from src.reproduction.preprocessing import audit_preprocessing

    cv2.imwrite(str(tmp_path / "black.png"), np.zeros((20, 20), np.uint8))
    records = pd.DataFrame({"path": ["black.png"]})
    audit = audit_preprocessing(records, tmp_path, load_config("configs/disci2025.yaml"))
    assert audit["crop_fallback_count"] == 1
    assert audit["crop_fallback_paths"] == ["black.png"]
