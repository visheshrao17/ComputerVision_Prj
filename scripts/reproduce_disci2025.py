#!/usr/bin/env python3
"""Download/prepare/train/analyze the same experiments used by the notebook."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd

from src.reproduction.artifacts import validate_saved_run
from src.reproduction.config import load_config, save_json
from src.reproduction.data import audit_duplicates, build_manifests, inventory_dataset, safe_extract
from src.reproduction.evaluation import compare_with_paper

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=1"
)


def download_dataset(destination: Path) -> dict:
    destination = Path(destination)
    # Never overlay a different dataset version or delete user-provided images.
    if (destination / "Training").exists() or (destination / "Testing").exists():
        raise ValueError(
            "Dataset folder already exists. Use prepare to verify it or choose an empty --data-dir."
        )
    archive = ROOT / "data/.download/disci-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".part")
    print("Downloading the pinned Kaggle version 1 (7,023 images)...", flush=True)
    with (
        urllib.request.urlopen(DATASET_URL, timeout=120) as response,
        open(temporary, "wb") as target,
    ):
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    safe_extract(archive, destination)
    records = inventory_dataset(destination)
    source = {
        "dataset": "masoudnickparvar/brain-tumor-mri-dataset",
        "version": 1,
        "download_url": DATASET_URL,
        "archive_sha256": digest.hexdigest(),
        "n_images": len(records),
        "note": "Version 2 has 7200 images and does not match the paper.",
    }
    save_json(ROOT / "results/disci2025/dataset_source.json", source)
    archive.unlink()  # Avoid keeping a second 150 MB copy after extraction.
    return source


def prepare(
    config: dict, patient_metadata: Path | None = None, duplicate_decisions: Path | None = None
) -> dict[str, pd.DataFrame]:
    records = inventory_dataset(Path(config["data_dir"]))
    audit = audit_duplicates(records)
    from src.reproduction.preprocessing import audit_preprocessing

    audit["preprocessing"] = audit_preprocessing(records, Path(config["data_dir"]), config)
    manifests = build_manifests(
        records,
        config["protocol"],
        config["seed"],
        patient_metadata,
        duplicate_decisions,
        validation_fraction=config["validation_fraction"],
    )
    directory = Path(config["output_dir"]) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    records.to_csv(directory / "inventory.csv", index=False)
    save_json(directory / "audit.json", audit)
    for split, table in manifests.items():
        table.to_csv(directory / f"{config['protocol']}_{split}.csv", index=False)
    print(
        f"Counts: { {split: len(table) for split, table in manifests.items()} }; "
        f"exact train/test overlap: {audit['cross_split_exact_count']}; "
        f"unreviewed near-match pairs: {len(audit['near_duplicate_candidates'])}",
        flush=True,
    )
    return manifests


def refresh_results(
    output_dir: Path, results_dir: Path = ROOT / "results/disci2025"
) -> pd.DataFrame:
    rows = []
    provenance = []
    for path in sorted(Path(output_dir).glob("*/*/metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata["smoke"] or not validate_saved_run(path.parent):
            continue
        compact = Path(results_dir) / "runs" / metadata["protocol"] / metadata["model"]
        compact.mkdir(parents=True, exist_ok=True)
        for artifact in path.parent.iterdir():
            if artifact.suffix in {".csv", ".json", ".png"} and artifact.is_file():
                target = compact / artifact.name
                if artifact.resolve() != target.resolve():
                    shutil.copy2(artifact, target)
        provenance.append(metadata)
        for view in ("clean", "transformed"):
            metrics_path = path.parent / f"metrics_{view}.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "model": metadata["model"],
                    "protocol": metadata["protocol"],
                    "evaluation_view": view,
                    "backbone_trainable": metadata["config"]["trainable_backbone"],
                    "input_scaling": metadata["config"]["input_scaling"],
                    "epochs": metadata["config"]["epochs"],
                    "seed": metadata["config"]["seed"],
                    "n_train": metadata["counts"]["train"],
                    "n_test": metrics["n_images"],
                    **{
                        key: metrics[key]
                        for key in [
                            "accuracy",
                            "macro_f1",
                            "weighted_f1",
                            "balanced_accuracy",
                            "macro_roc_auc",
                        ]
                    },
                    "total_parameters": metadata["total_parameters"],
                    "training_seconds": metadata["training_seconds"],
                    "median_inference_ms": metadata.get("median_inference_ms"),
                    "p95_inference_ms": metadata.get("p95_inference_ms"),
                    "combined_train_test_accuracy": metadata.get("combined_train_test_accuracy"),
                    "run_dir": path.parent.relative_to(Path(output_dir)).as_posix(),
                }
            )
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    table = compare_with_paper(pd.DataFrame(rows)) if rows else pd.DataFrame()
    table.to_csv(results_dir / "summary.csv", index=False)
    save_json(results_dir / "provenance.json", {"runs": provenance})
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare", "train", "analyze"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/disci2025.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", choices=["paper_based", "leakage_audited"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--patient-metadata", type=Path)
    parser.add_argument("--duplicate-decisions", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="32/class training, 8/class test, one epoch; excluded from results",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    for option in ("data_dir", "output_dir", "protocol", "models"):
        value = getattr(args, option)
        if value is not None:
            config[option] = str(value) if isinstance(value, Path) else value
    from src.reproduction.config import validate_config

    validate_config(config)
    for key in ("data_dir", "output_dir"):
        config[key] = str((ROOT / config[key]).resolve())
    if args.command == "download":
        download_dataset(Path(config["data_dir"]))
    elif args.command == "analyze":
        print(refresh_results(Path(config["output_dir"])).to_string(index=False))
    else:
        manifests = prepare(config, args.patient_metadata, args.duplicate_decisions)
        if args.command == "train":
            from src.reproduction.training import train_model

            if args.smoke:
                config["epochs"] = 1
                manifests = {
                    split: table.groupby("label")
                    .head(32 if split == "train" else 8)
                    .reset_index(drop=True)
                    for split, table in manifests.items()
                }
            for name in config["models"]:
                mode = "smoke" if args.smoke else config["protocol"]
                run_dir = Path(config["output_dir"]) / mode / name
                train_model(name, manifests, config, run_dir, resume=args.resume, smoke=args.smoke)
                refresh_results(Path(config["output_dir"]))


if __name__ == "__main__":
    main()
