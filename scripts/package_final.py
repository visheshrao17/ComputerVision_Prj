#!/usr/bin/env python3
"""Bundle the final experiment for Colab, optionally including a trained run."""

import argparse
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package_final(destination: Path, include_model: bool = False) -> Path:
    files = [ROOT / "README.md", ROOT / "requirements.txt", ROOT / "requirements-mac.txt"]
    files.append(ROOT / "docs/final_evaluation_plan.md")
    folders = [
        "src/reproduction",
        "scripts",
        "configs",
        "results/disci2025",
        "results/final_project",
        "deliverables/final_evaluation",
    ]
    for folder in folders:
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and "local_mac" not in p.parts
        )
    files.append(ROOT / "notebooks/final_project_colab.ipynb")
    report_source = ROOT / "docs/report/Final_Evaluation_Report.md"
    if report_source.exists():
        files.append(report_source)
    if include_model:
        from src.reproduction.artifacts import validate_saved_run

        run = ROOT / "outputs/final_project/leakage_audited/MobileNetV2"
        if not validate_saved_run(run, require_checkpoint=True):
            raise ValueError("A complete verified run with model.keras is required for export")
        files.extend(p for p in run.iterdir() if p.is_file())
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for p in sorted(set(files)):
            archive.write(p, p.relative_to(ROOT))
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deliverables/brain_mri_final_source.zip"
    )
    args = parser.parse_args()
    package_final(args.output, args.include_model)
