#!/usr/bin/env python3
"""Run the longer-training control and compact two-stage MobileNetV2 experiment."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reproduce_disci2025 import download_dataset, prepare, refresh_results
from src.reproduction.config import fingerprint
from src.reproduction.final_project import predict_image
from src.reproduction.training_comparison import EXPERIMENTS, comparison_table, extension_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "analyze", "predict"])
    parser.add_argument("--experiment", choices=["both", *EXPERIMENTS], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/training_extension")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/training_extension")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "predict":
        if args.experiment == "both" or args.image is None:
            parser.error("predict requires --experiment control|proposed and --image PATH")
        run = args.output_dir / args.experiment / "leakage_audited/MobileNetV2"
        print(json.dumps(predict_image(run, args.image), indent=2))
        return
    if args.command == "train":
        from src.reproduction.training import train_model

        if not (args.data_dir / "Training").exists():
            download_dataset(args.data_dir, ROOT / "results/training_extension/dataset_source.json")
        manifests = None
        for experiment in EXPERIMENTS if args.experiment == "both" else [args.experiment]:
            config = extension_config(experiment)
            config.update(
                data_dir=str(args.data_dir.resolve()),
                output_dir=str(args.output_dir.resolve() / experiment),
            )
            if manifests is None:
                manifests = prepare(config)
                identity = {
                    split: table[["path", "label", "pixel_hash"]].to_dict("records")
                    for split, table in manifests.items()
                }
                reference = json.loads(
                    (
                        ROOT
                        / "results/final_project/runs/leakage_audited/MobileNetV2/metadata.json"
                    ).read_text()
                )
                if fingerprint(identity) != reference["data_fingerprint"]:
                    raise ValueError(
                        "Audited split differs from preserved evidence; training stopped"
                    )
            run = Path(config["output_dir"]) / "leakage_audited/MobileNetV2"
            train_model("MobileNetV2", manifests, config, run, resume=args.resume)
            refresh_results(
                Path(config["output_dir"]), ROOT / "results/training_extension" / experiment
            )
    table = comparison_table(args.output_dir)
    results = ROOT / "results/training_extension"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "comparison.csv", index=False)
    print(table.drop(columns="run_dir").to_string(index=False))
    print(
        "Exploratory reused test set. A vs B tests a combined method, not individual component effects."
    )


if __name__ == "__main__":
    main()
