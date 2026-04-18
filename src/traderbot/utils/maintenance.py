from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


RESULTS_CLEANUP_STATE_FILENAME = "results_cleanup_state.json"


@dataclass(frozen=True)
class ResultsCleanupReport:
    checked_files: int
    deleted_files: int
    reclaimed_bytes: int
    cutoff_iso: str
    failed_files: tuple[str, ...] = ()


def _resolve_directory(path: str | Path) -> Path:
    directory = Path(path)
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cleanup_state_path(state_dir: str | Path) -> Path:
    return _resolve_directory(state_dir) / RESULTS_CLEANUP_STATE_FILENAME


def _read_cleanup_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_cleanup_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def prune_results_json_files(
    results_dir: str | Path,
    *,
    retention_days: int = 7,
    now: datetime | None = None,
) -> ResultsCleanupReport:
    reference_now = now or datetime.now()
    cutoff = reference_now - timedelta(days=max(1, int(retention_days)))
    results_path = _resolve_directory(results_dir)

    checked_files = 0
    deleted_files = 0
    reclaimed_bytes = 0
    failed_files: list[str] = []

    for file_path in sorted(results_path.rglob("*.json")):
        if not file_path.is_file():
            continue

        checked_files += 1
        try:
            stat = file_path.stat()
        except OSError:
            failed_files.append(str(file_path))
            continue

        modified_at = datetime.fromtimestamp(stat.st_mtime)
        if modified_at >= cutoff:
            continue

        try:
            file_size = int(stat.st_size)
            file_path.unlink()
            deleted_files += 1
            reclaimed_bytes += file_size
        except OSError:
            failed_files.append(str(file_path))

    return ResultsCleanupReport(
        checked_files=checked_files,
        deleted_files=deleted_files,
        reclaimed_bytes=reclaimed_bytes,
        cutoff_iso=cutoff.isoformat(),
        failed_files=tuple(failed_files),
    )


def cleanup_results_json_once_per_day(
    *,
    results_dir: str | Path,
    state_dir: str | Path,
    logger: logging.Logger | None = None,
    retention_days: int = 7,
    now: datetime | None = None,
) -> ResultsCleanupReport | None:
    reference_now = now or datetime.now()
    today_key = reference_now.date().isoformat()
    state_path = _cleanup_state_path(state_dir)
    state = _read_cleanup_state(state_path)

    if str(state.get("last_results_cleanup_date") or "") == today_key:
        return None

    report = prune_results_json_files(
        results_dir,
        retention_days=retention_days,
        now=reference_now,
    )
    _write_cleanup_state(
        state_path,
        {
            "last_results_cleanup_date": today_key,
            "last_results_cleanup_at": reference_now.isoformat(),
            "retention_days": int(retention_days),
            "checked_files": int(report.checked_files),
            "deleted_files": int(report.deleted_files),
            "reclaimed_bytes": int(report.reclaimed_bytes),
            "failed_files": list(report.failed_files),
        },
    )

    if logger is not None:
        logger.info(
            "Manutencao diaria results concluida | checked=%s | deleted=%s | reclaimed_bytes=%s | cutoff=%s | failures=%s",
            report.checked_files,
            report.deleted_files,
            report.reclaimed_bytes,
            report.cutoff_iso,
            len(report.failed_files),
        )

    return report
