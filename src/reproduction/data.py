"""Inventory the pinned dataset and preserve an auditable split manifest."""

import hashlib
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import CLASSES

EXPECTED = {"train": [1321, 1339, 1595, 1457], "test": [300, 306, 405, 300]}
ALIASES = {"notumor": "no_tumor", **{name: name for name in CLASSES}}


def safe_extract(archive: Path, destination: Path) -> None:
    destination = Path(destination).resolve()
    with ZipFile(archive) as stream:
        for entry in stream.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive symlink")
        stream.extractall(destination)


def perceptual_hash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (32, 32)).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    return format(int("".join("1" if bit else "0" for bit in bits), 2), "016x")


def inventory_dataset(root: Path | str, require_counts: bool = True) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for folder, partition in [("Training", "train"), ("Testing", "test")]:
        for path in sorted((root / folder).rglob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            name = ALIASES.get(path.parent.name.lower())
            if name is None:
                raise ValueError(f"Unknown class folder: {path.parent}")
            pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f"Unreadable image: {path}")
            digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "class_name": name,
                    "label": CLASSES.index(name),
                    "original_partition": partition,
                    "pixel_hash": digest,
                    "phash": perceptual_hash(gray),
                }
            )
    table = pd.DataFrame(
        rows, columns=["path", "class_name", "label", "original_partition", "pixel_hash", "phash"]
    )
    if require_counts:
        for partition, expected in EXPECTED.items():
            actual = table[table.original_partition == partition].label.value_counts()
            counts = [int(actual.get(i, 0)) for i in range(4)]
            if counts != expected:
                raise ValueError(
                    f"Dataset counts mismatch for {partition}: {counts}; expected {expected}. "
                    "Download Kaggle dataset version 1, not the current version 2."
                )
    return table


def audit_duplicates(records: pd.DataFrame) -> dict:
    train = records[records.original_partition == "train"]
    test = records[records.original_partition == "test"]
    test_hashes = set(test.pixel_hash)
    exact = train[train.pixel_hash.isin(test_hashes)]
    conflicts = records.groupby("pixel_hash").label.nunique()
    candidates = []
    # 7.5 million inexpensive XOR comparisons; no large image-distance matrix.
    test_items = [(row.path, row.pixel_hash, int(row.phash, 16)) for row in test.itertuples()]
    for row in train.itertuples():
        value = int(row.phash, 16)
        for test_path, pixel_hash, phash in test_items:
            distance = (value ^ phash).bit_count()
            if distance <= 4 and row.pixel_hash != pixel_hash:
                candidates.append(
                    {"train_path": row.path, "test_path": test_path, "hamming_distance": distance}
                )
    return {
        "cross_split_exact_count": len(exact),
        "cross_split_exact_train_paths": exact.path.tolist(),
        "within_train_exact_redundant_count": int(train.pixel_hash.duplicated().sum()),
        "within_test_exact_redundant_count": int(test.pixel_hash.duplicated().sum()),
        "conflicting_hashes": conflicts[conflicts > 1].index.tolist(),
        "near_duplicate_candidates": candidates,
        "patient_ids_available": "patient_id" in records.columns,
        "limitation": "Perceptual similarity is not proof of a duplicate or a shared patient.",
    }


def build_manifests(
    records: pd.DataFrame,
    protocol: str,
    seed: int,
    patient_metadata: Path | None = None,
    duplicate_decisions: Path | None = None,
    validation_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    records = records.copy()
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    if patient_metadata is not None:
        metadata = pd.read_csv(patient_metadata, dtype={"patient_id": str})
        if metadata.path.duplicated().any():
            raise ValueError("Patient metadata has duplicate paths")
        records = records.merge(
            metadata[["path", "patient_id"]], on="path", how="left", validate="one_to_one"
        )
    train = records[records.original_partition == "train"].copy()
    test = records[records.original_partition == "test"].copy()
    if protocol == "paper_based":
        return {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True)}
    if protocol != "leakage_audited":
        raise ValueError("Unknown protocol")
    if (records.groupby("pixel_hash").label.nunique() > 1).any():
        raise ValueError("Conflicting labels for identical decoded pixels")
    train = train[~train.pixel_hash.isin(test.pixel_hash)].drop_duplicates("pixel_hash")
    if duplicate_decisions is not None:
        decisions = pd.read_csv(duplicate_decisions)
        if set(decisions.decision) - {"confirmed_duplicate", "distinct", "unresolved"}:
            raise ValueError("Unknown duplicate decision")
        # Decisions remove only training copies; official test is never altered.
        confirmed = decisions.loc[decisions.decision == "confirmed_duplicate", "train_path"]
        if set(confirmed) - set(records.loc[records.original_partition == "train", "path"]):
            raise ValueError("Unknown training path in duplicate decisions")
        train = train[~train.path.isin(confirmed)]
    if "patient_id" in records:
        if (
            records.patient_id.isna().any()
            or (records.patient_id.astype(str).str.strip() == "").any()
        ):
            raise ValueError("Incomplete patient metadata")
        train = train[~train.patient_id.isin(test.patient_id)]
        groups_per_class = train.groupby("label").patient_id.nunique().min()
        if groups_per_class < 2:
            raise ValueError("Too few patient groups for four-class validation")
        splitter = StratifiedGroupKFold(
            n_splits=min(round(1 / validation_fraction), int(groups_per_class)),
            shuffle=True,
            random_state=seed,
        )
        left, right = next(splitter.split(train, train.label, train.patient_id))
        train, val = train.iloc[left], train.iloc[right]
    else:
        train, val = train_test_split(
            train, test_size=validation_fraction, stratify=train.label, random_state=seed
        )
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        if set(subset.label) != set(range(4)):
            raise ValueError(
                f"{name} does not contain all four classes; adjust data/group allocation"
            )
    return {
        name: subset.sort_values("path").reset_index(drop=True)
        for name, subset in [("train", train), ("val", val), ("test", test)]
    }
