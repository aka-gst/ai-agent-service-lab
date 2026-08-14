import zipfile
from pathlib import Path

import pytest

from agent_lab.backup import create_backup, restore_backup


def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "rag.sqlite3").write_bytes(b"rag-data")
    (source / "memory.sqlite3").write_bytes(b"memory-data")

    archive = create_backup(source, tmp_path / "backups")
    restored = restore_backup(archive, tmp_path / "restored")

    assert {path.name for path in restored} == {"rag.sqlite3", "memory.sqlite3"}
    assert (tmp_path / "restored" / "rag.sqlite3").read_bytes() == b"rag-data"


def test_backup_ignores_unexpected_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "rag.sqlite3").write_bytes(b"rag")
    (source / "password.txt").write_text("do not copy")

    archive = create_backup(source, tmp_path / "backups")

    with zipfile.ZipFile(archive) as bundle:
        assert "data/password.txt" not in bundle.namelist()


def test_restore_refuses_non_empty_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "rag.sqlite3").write_bytes(b"rag")
    archive = create_backup(source, tmp_path / "backups")
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing.txt").write_text("keep")

    with pytest.raises(ValueError, match="должна быть пустой"):
        restore_backup(archive, target)


def test_restore_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.sqlite3", b"bad")
        bundle.writestr("manifest.json", '{"files": []}')

    with pytest.raises(ValueError, match="Небезопасный путь"):
        restore_backup(archive, tmp_path / "restored")
