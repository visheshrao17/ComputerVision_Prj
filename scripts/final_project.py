#!/usr/bin/env python3
"""Train the final normalization experiment, compare results, or predict one image."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse its cache setup, pinned downloader, data preparation and evidence exporter.
from scripts.reproduce_disci2025 import download_dataset, prepare, refresh_results
from src.reproduction.config import load_config, save_json
from src.reproduction.final_project import predict_image, save_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "analyze", "predict"])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/final_project")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/mobilenet_imagenet.yaml")
    config["data_dir"] = str((args.data_dir or ROOT / config["data_dir"]).resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    run_dir = Path(config["output_dir"]) / config["protocol"] / "MobileNetV2"
    results_dir = ROOT / "results/final_project"

    if args.command == "predict":
        if args.image is None:
            parser.error("predict requires --image /path/to/mri.jpg")
        print(json.dumps(predict_image(run_dir, args.image), indent=2))
        return
    if args.command == "train":
        from src.reproduction.training import train_model

        if not (Path(config["data_dir"]) / "Training").exists():
            download_dataset(Path(config["data_dir"]))
        manifests = prepare(config)
        train_model("MobileNetV2", manifests, config, run_dir, resume=args.resume)
    # The old baseline results directory is never a destination for this experiment.
    result = save_comparison(run_dir, results_dir)
    refresh_results(Path(config["output_dir"]), results_dir)
    save_json(results_dir / "config.json", config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
