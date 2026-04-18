from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from traderbot.config import AppConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_GUARD_LOCK_PORT = 54323


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name_fragment(value: Any) -> str:
    raw = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    return cleaned or "unknown"


def resolve_logs_dir(logs_dir: str | Path) -> Path:
    path = Path(logs_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_state_path_for(*, network: str, symbol: str, logs_dir: str | Path) -> Path:
    filename = (
        "runtime_state_"
        f"{_safe_name_fragment(network)}_"
        f"{_safe_name_fragment(symbol)}.json"
    )
    return resolve_logs_dir(logs_dir) / filename


def runtime_state_path(cfg: AppConfig) -> Path:
    return runtime_state_path_for(
        network=str(cfg.hyperliquid.network),
        symbol=str(cfg.hyperliquid.symbol),
        logs_dir=cfg.paths.logs_dir,
    )


def write_runtime_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def read_runtime_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_python_executable() -> str:
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        candidate = executable.with_name("python.exe")
        if candidate.exists():
            return str(candidate)
    return str(executable)


def build_guard_subprocess_args(
    *,
    config_path: str | Path,
    state_path: Path,
    network_override: str | None,
    execution_mode_override: str | None,
    allow_live_trading: bool,
    order_slippage_override: float | None,
) -> list[str]:
    args = ["-u", "-m", "traderbot.main", "--config", str(config_path)]
    if network_override:
        args.extend(["--network-override", str(network_override)])
    if execution_mode_override:
        args.extend(["--execution-mode-override", str(execution_mode_override)])
    if allow_live_trading:
        args.append("--allow-live-trading")
    if order_slippage_override is not None:
        args.extend(["--order-slippage-override", str(order_slippage_override)])
    args.extend(["runtime-guard", "--state-path", str(state_path)])
    return args


def spawn_runtime_guard_process(
    *,
    logger,
    config_path: str | Path,
    state_path: Path,
    network_override: str | None,
    execution_mode_override: str | None,
    allow_live_trading: bool,
    order_slippage_override: float | None,
) -> subprocess.Popen[bytes] | None:
    python_executable = _resolve_python_executable()
    args = build_guard_subprocess_args(
        config_path=config_path,
        state_path=state_path,
        network_override=network_override,
        execution_mode_override=execution_mode_override,
        allow_live_trading=allow_live_trading,
        order_slippage_override=order_slippage_override,
    )
    popen_kwargs: dict[str, Any] = {
        "args": [python_executable, *args],
        "cwd": str(REPO_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(**popen_kwargs)
    except Exception:
        logger.exception("Falha ao iniciar processo separado do runtime guard.")
        return None

    logger.info(
        "Runtime guard destacado iniciado | pid=%s | state_path=%s",
        int(process.pid or 0),
        state_path,
    )
    return process


def acquire_runtime_guard_lock() -> socket.socket | None:
    guard_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        guard_lock.bind(("127.0.0.1", RUNTIME_GUARD_LOCK_PORT))
    except OSError:
        try:
            guard_lock.close()
        except Exception:
            pass
        return None
    return guard_lock


def is_process_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False

    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return int(exit_code.value) == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False

    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _timestamp_age_seconds(value: Any, *, now: float) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return max(0.0, now - moment.timestamp())


@dataclass
class RuntimeHeartbeat:
    """Shared engine snapshot consumed by the runtime guard and GUI clients."""

    cfg: AppConfig
    logger: Any
    state_path: Path = field(init=False)
    heartbeat_interval_seconds: float = field(init=False)
    session_id: str = field(init=False)
    _payload: dict[str, Any] = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.state_path = runtime_state_path(self.cfg)
        self.heartbeat_interval_seconds = max(
            5.0,
            float(self.cfg.execution.runtime_guard_heartbeat_interval_seconds),
        )
        self.session_id = f"{int(os.getpid())}-{int(time.time())}"
        self._payload = {
            "session_id": self.session_id,
            "status": "starting",
            "runtime_pid": int(os.getpid()),
            "started_at": _utc_now_iso(),
            "last_heartbeat_at": _utc_now_iso(),
            "network": str(self.cfg.hyperliquid.network).lower().strip(),
            "symbol": str(self.cfg.hyperliquid.symbol).upper().strip(),
            "timeframe": str(self.cfg.hyperliquid.timeframe).strip(),
            "execution_mode": str(self.cfg.execution.execution_mode).lower().strip(),
            "last_cycle_payload": None,
        }

    def start(self) -> None:
        self._write_state_locked()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="traderbot-runtime-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._payload.update(fields)
            self._payload["last_heartbeat_at"] = _utc_now_iso()
            self._write_state_locked()

    def mark_status(self, status: str, **fields: Any) -> None:
        self.update(status=str(status), **fields)

    def stop(self, *, final_status: str, **fields: Any) -> None:
        with self._lock:
            self._payload.update(fields)
            self._payload["status"] = str(final_status)
            self._payload["last_heartbeat_at"] = _utc_now_iso()
            self._payload["stopped_at"] = _utc_now_iso()
            self._write_state_locked()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_interval_seconds))

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_seconds):
            with self._lock:
                if self._payload.get("status") == "starting":
                    self._payload["status"] = "running"
                self._payload["last_heartbeat_at"] = _utc_now_iso()
                self._write_state_locked()

    def _write_state_locked(self) -> None:
        try:
            write_runtime_state(self.state_path, dict(self._payload))
        except Exception:
            self.logger.exception("Falha ao persistir heartbeat do runtime em %s.", self.state_path)


def run_runtime_guard(
    *,
    cfg: AppConfig,
    logger,
    notifier,
    state_path: str | Path,
    close_position_callback: Callable[..., dict[str, Any]],
) -> int:
    if bool(getattr(cfg.execution, "exchange_native_tp_sl_enabled", True)):
        logger.info(
            "Runtime guard desabilitado porque exchange_native_tp_sl_enabled=true; a protecao passa a ficar no TP/SL nativo da Hyperliquid."
        )
        return 0

    guard_lock = acquire_runtime_guard_lock()
    if guard_lock is None:
        logger.info("Outro runtime guard ja esta rodando; esta instancia sera encerrada.")
        return 0

    state_file = Path(state_path)
    poll_interval = max(5.0, float(cfg.execution.runtime_guard_poll_interval_seconds))
    stale_after = max(
        poll_interval * 2.0,
        float(cfg.execution.runtime_guard_stale_after_seconds),
    )
    idle_exit_after = max(
        poll_interval * 3.0,
        float(cfg.execution.runtime_guard_idle_exit_after_seconds),
    )
    confirmations_needed = max(1, int(cfg.execution.runtime_guard_stale_confirmations))
    retry_backoff_seconds = max(20.0, poll_interval * 2.0)

    logger.info(
        "Runtime guard monitorando %s | stale_after=%.1fs | poll=%.1fs | confirmations=%s",
        state_file,
        stale_after,
        poll_interval,
        confirmations_needed,
    )

    last_session_id = ""
    stale_confirmations = 0
    idle_since: float | None = None
    retry_not_before = 0.0

    try:
        while True:
            now = time.time()
            state = read_runtime_state(state_file)
            if state is None:
                if idle_since is None:
                    idle_since = now
                if (now - idle_since) >= idle_exit_after:
                    logger.info("Heartbeat do runtime ausente ha muito tempo; encerrando runtime guard.")
                    return 0
                time.sleep(poll_interval)
                continue

            session_id = str(state.get("session_id") or "")
            if session_id != last_session_id:
                last_session_id = session_id
                stale_confirmations = 0
                retry_not_before = 0.0
                logger.info(
                    "Runtime guard assumiu sessao %s | runtime_pid=%s | network=%s | symbol=%s",
                    session_id or "sem-id",
                    int(state.get("runtime_pid", 0) or 0),
                    state.get("network"),
                    state.get("symbol"),
                )

            status = str(state.get("status") or "").strip().lower()
            runtime_pid = int(state.get("runtime_pid", 0) or 0)

            if status in {
                "stopping",
                "stopped",
                "stopped_after_error",
                "closed_by_guard",
                "guard_checked_no_position",
            }:
                stale_confirmations = 0
                if idle_since is None:
                    idle_since = now
                if (now - idle_since) >= idle_exit_after and not is_process_running(runtime_pid):
                    logger.info("Runtime encerrado de forma conclusiva; encerrando runtime guard.")
                    return 0
                time.sleep(poll_interval)
                continue

            idle_since = None

            heartbeat_age = _timestamp_age_seconds(state.get("last_heartbeat_at"), now=now)
            if heartbeat_age is None or heartbeat_age < stale_after:
                stale_confirmations = 0
                time.sleep(poll_interval)
                continue

            if now < retry_not_before:
                time.sleep(poll_interval)
                continue

            stale_confirmations += 1
            pid_alive = is_process_running(runtime_pid)
            logger.warning(
                "Heartbeat do runtime stale | session_id=%s | runtime_pid=%s | pid_alive=%s | stale_for=%.1fs | tentativa=%s/%s",
                session_id or "sem-id",
                runtime_pid,
                pid_alive,
                heartbeat_age,
                stale_confirmations,
                confirmations_needed,
            )
            if stale_confirmations < confirmations_needed:
                time.sleep(poll_interval)
                continue

            state["status"] = "guard_closing_position"
            state["guard_triggered_at"] = _utc_now_iso()
            state["guard_reason"] = "runtime_heartbeat_stale"
            write_runtime_state(state_file, state)

            try:
                payload = close_position_callback(
                    cfg,
                    logger,
                    notifier=notifier,
                    trigger="runtime_orphan_guard",
                )
            except Exception as exc:
                logger.exception("Runtime guard falhou ao tentar fechar a posicao orphan.")
                state["status"] = "guard_close_retry_pending"
                state["guard_error"] = f"{type(exc).__name__}: {exc}"
                state["guard_completed_at"] = _utc_now_iso()
                write_runtime_state(state_file, state)
                retry_not_before = time.time() + retry_backoff_seconds
                stale_confirmations = confirmations_needed - 1
                time.sleep(poll_interval)
                continue

            snapshot = payload.get("position_snapshot") or {}
            position_still_open = int(snapshot.get("side", 0) or 0) != 0
            state["guard_completed_at"] = _utc_now_iso()
            state["guard_result"] = {
                "ok": bool(payload.get("ok", False)),
                "closed_position": bool(payload.get("closed_position", False)),
                "trigger": payload.get("trigger"),
                "message": payload.get("message"),
                "position_side": int(snapshot.get("side", 0) or 0),
            }
            if bool(payload.get("closed_position", False)):
                state["status"] = "closed_by_guard"
                write_runtime_state(state_file, state)
                logger.warning("Runtime guard fechou a posicao aberta apos perda do heartbeat do runtime.")
                return 0

            if not position_still_open:
                state["status"] = "guard_checked_no_position"
                write_runtime_state(state_file, state)
                logger.warning("Runtime guard acionado, mas nao encontrou posicao aberta na exchange.")
                return 0

            state["status"] = "guard_close_retry_pending"
            write_runtime_state(state_file, state)
            retry_not_before = time.time() + retry_backoff_seconds
            stale_confirmations = confirmations_needed - 1
            logger.warning("Runtime guard nao conseguiu fechar a posicao; nova tentativa sera feita em breve.")
            time.sleep(poll_interval)
    finally:
        try:
            guard_lock.close()
        except Exception:
            pass
