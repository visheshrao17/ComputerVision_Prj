"""Compare accepted evidence; pending experiments never become measured results."""

import json
from pathlib import Path

import pandas as pd

from .artifacts import validate_saved_run
from .config import ROOT, load_config

EXPERIMENTS = ("control", "proposed")


def extension_config(experiment: str) -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Choose one of {EXPERIMENTS}")
    return load_config(ROOT / f"configs/training_{experiment}.yaml")


def check_extension_identity(metadata: dict, experiment: str, reference: dict) -> None:
    """Reject changed splits, smoke runs or settings outside the approved experiment."""
    ignored = {"models", "data_dir", "output_dir"}
    expected = {k: v for k, v in extension_config(experiment).items() if k not in ignored}
    actual = {k: v for k, v in metadata["config"].items() if k not in ignored}
    if actual != expected or metadata["model"] != "MobileNetV2" or metadata["smoke"]:
        raise ValueError(
            f"Unexpected {experiment} model/settings; do not publish as this experiment"
        )
    if metadata["data_fingerprint"] != reference["data_fingerprint"]:
        raise ValueError("The audited data split must match the preserved normalization run")


def comparison_table(output_root: Path = ROOT / "outputs/training_extension") -> pd.DataFrame:
    rows = []
    references = {
        "Xception reference": ROOT / "results/disci2025/runs/leakage_audited/Xception",
        "Normalized MobileNetV2 (5 epochs)": ROOT
        / "results/final_project/runs/leakage_audited/MobileNetV2",
    }
    reference = json.loads(
        (references["Normalized MobileNetV2 (5 epochs)"] / "metadata.json").read_text()
    )
    candidates = dict(references)
    for experiment in EXPERIMENTS:
        run = Path(output_root) / experiment / "leakage_audited/MobileNetV2"
        if not (run / "metadata.json").exists():
            run = ROOT / f"results/training_extension/{experiment}/runs/leakage_audited/MobileNetV2"
        candidates[experiment] = run
    for name, run in candidates.items():
        if not (run / "metadata.json").exists():
            rows.append({"experiment": name, "status": "pending"})
            continue
        if not validate_saved_run(run):
            raise ValueError(f"Incomplete or inconsistent evidence for {name}: {run}")
        metadata = json.loads((run / "metadata.json").read_text())
        if metadata["data_fingerprint"] != reference["data_fingerprint"]:
            raise ValueError("Reference and experiment data splits must match")
        if name in EXPERIMENTS:
            check_extension_identity(metadata, name, reference)
        metrics = json.loads((run / "metrics_clean.json").read_text())
        rows.append(
            {
                "experiment": name,
                "status": "measured",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "pituitary_recall": metrics["per_class"]["pituitary"]["recall"],
                "total_parameters": metadata["total_parameters"],
                "epochs_completed": metadata.get("epochs_completed", metadata["config"]["epochs"]),
                "selected_epoch": metadata.get("selected_epoch", metadata["config"]["epochs"]),
                "best_val_loss": metadata.get("best_val_loss"),
                "training_seconds": metadata["training_seconds"],
                "median_inference_ms": metadata.get("median_inference_ms"),
                "environment_matches_normalized_run": metadata["environment"]
                == reference["environment"],
                "run_dir": str(run),
            }
        )
    table = pd.DataFrame(rows)
    xception, normalized = table.iloc[0], table.iloc[1]
    table["macro_f1_gain_over_normalized"] = table.macro_f1 - normalized.macro_f1
    table["xception_macro_f1_gap"] = xception.macro_f1 - table.macro_f1
    table["parameter_reduction_vs_xception_pct"] = 100 * (
        1 - table.total_parameters / xception.total_parameters
    )
    # Only completed NEW experiments have a hypothesis outcome; no NaN -> false conversion.
    table["improves_normalized_with_fewer_parameters"] = pd.Series(
        [None] * len(table), dtype=object
    )
    table["within_xception_margin_with_fewer_parameters"] = pd.Series(
        [None] * len(table), dtype=object
    )
    for i, row in table.iterrows():
        if row.experiment in EXPERIMENTS and row.status == "measured":
            table.at[i, "improves_normalized_with_fewer_parameters"] = bool(
                row.macro_f1 > normalized.macro_f1
                and row.total_parameters < normalized.total_parameters
            )
            table.at[i, "within_xception_margin_with_fewer_parameters"] = bool(
                row.xception_macro_f1_gap <= 0.02
                and row.total_parameters < xception.total_parameters
            )
    return table
