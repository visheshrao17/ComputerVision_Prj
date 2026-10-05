from zipfile import ZipFile

import pytest

from src.reproduction.data import safe_extract


def test_archive_cannot_escape_dataset_folder(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "images")
    assert not (tmp_path / "escaped.txt").exists()


def test_archive_extracts_expected_layout(tmp_path):
    archive = tmp_path / "safe.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("Training/glioma/a.jpg", b"content")
    safe_extract(archive, tmp_path / "images")
    assert (tmp_path / "images/Training/glioma/a.jpg").read_bytes() == b"content"
