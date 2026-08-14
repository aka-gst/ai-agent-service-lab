"""Безопасное резервное копирование и восстановление локальных данных."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


DEFAULT_SOURCE = Path("data/private")
DEFAULT_BACKUP_DIR = Path("artifacts/private/backups")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_backup(source: Path, backup_dir: Path) -> Path:
    """Создать ZIP с SQLite-файлами и манифестом контрольных сумм."""

    files = sorted(path for path in source.glob("*.sqlite3") if path.is_file())
    if not files:
        raise ValueError(f"В {source} нет SQLite-файлов для резервного копирования")
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = backup_dir / f"private-data-{timestamp}.zip"
    if archive.exists():
        raise FileExistsError(f"Архив уже существует: {archive}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
    }
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=f"data/{path.name}")
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
    return archive


def _safe_members(bundle: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = bundle.infolist()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Небезопасный путь внутри архива: {member.filename}")
    return members


def restore_backup(archive: Path, target: Path) -> list[Path]:
    """Восстановить данные только в новую или пустую папку."""

    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Папка восстановления должна быть пустой: {target}")
    target.mkdir(parents=True, exist_ok=True)

    restored: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        members = _safe_members(bundle)
        manifest = json.loads(bundle.read("manifest.json"))
        expected = {item["name"]: item for item in manifest["files"]}
        for member in members:
            member_path = PurePosixPath(member.filename)
            if len(member_path.parts) != 2 or member_path.parts[0] != "data":
                continue
            name = member_path.name
            if name not in expected or not name.endswith(".sqlite3"):
                raise ValueError(f"Неожиданный файл в архиве: {member.filename}")
            destination = target / name
            destination.write_bytes(bundle.read(member))
            if sha256(destination) != expected[name]["sha256"]:
                raise ValueError(f"Контрольная сумма не совпала: {name}")
            restored.append(destination)
    if set(path.name for path in restored) != set(expected):
        raise ValueError("Архив не содержит все файлы из манифеста")
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup локальных данных агента")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup_parser = subparsers.add_parser("create")
    backup_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    backup_parser.add_argument("--output", type=Path, default=DEFAULT_BACKUP_DIR)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("archive", type=Path)
    restore_parser.add_argument("--target", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "create":
            archive = create_backup(args.source, args.output)
            print(f"BACKUP: {archive}")
        else:
            restored = restore_backup(args.archive, args.target)
            for path in restored:
                print(f"RESTORED: {path}")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Ошибка: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
