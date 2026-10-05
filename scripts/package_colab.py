"""Package portable code and compact evidence, excluding images and checkpoints."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def package_colab():
    target = ROOT / "deliverables/brain_mri_colab.zip"
    target.parent.mkdir(exist_ok=True)
    directories = ["configs", "src/reproduction", "scripts", "notebooks", "results", "docs/report"]
    files = [ROOT / name for name in ["README.md", "requirements.txt"]]
    for name in directories:
        files.extend(
            path
            for path in (ROOT / name).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT))
    print(f"{target}: {target.stat().st_size:,} bytes; no raw images or model checkpoints")
    return target


if __name__ == "__main__":
    package_colab()
