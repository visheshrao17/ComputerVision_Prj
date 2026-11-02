#!/usr/bin/env python3
"""Export this extension's source or verified runs, excluding raw MRI images."""

import argparse
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproduction.artifacts import validate_saved_run
from src.reproduction.training_comparison import EXPERIMENTS


def package_extension(destination: Path, include_models: bool = False) -> Path:
    files = [
        ROOT / name
        for name in [
            "README.md",
            "requirements.txt",
            "requirements-mac.txt",
            "docs/training_extension_plan.md",
            "docs/final_evaluation_plan.md",
            "docs/report/Final_Evaluation_Report.md",
            "notebooks/training_extension_colab.ipynb",
        ]
    ]
    for folder in [
        "src/reproduction",
        "scripts",
        "configs",
        "docs/latex",
        "results/disci2025",
        "results/final_project",
        "results/training_extension",
        "deliverables/final_evaluation",
    ]:
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and "local_mac" not in p.parts
        )
    accepted = []
    if include_models:
        for experiment in EXPERIMENTS:
            run = ROOT / f"outputs/training_extension/{experiment}/leakage_audited/MobileNetV2"
            if validate_saved_run(run, require_checkpoint=True):
                accepted.append(experiment)
                files.extend(p for p in run.iterdir() if p.is_file())
        if not accepted:
            raise ValueError("No verified completed checkpoints to export")
        print("Exporting completed models:", ", ".join(accepted))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT))
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deliverables/brain_mri_training_source.zip"
    )
    args = parser.parse_args()
    package_extension(args.output, args.include_models)
