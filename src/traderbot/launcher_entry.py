from __future__ import annotations

import os
import sys
from pathlib import Path

from traderbot.utils.fault_logging import install_fatal_error_hooks, uninstall_fatal_error_hooks
from traderbot.utils.logger import setup_logger


def _missing_gui_dependencies_message(exc: BaseException) -> str:
    return (
        "As dependencias da interface grafica nao estao disponiveis neste ambiente.\n\n"
        "Instale o launcher desktop com um destes comandos:\n"
        "  pip install -r requirements-ui.txt\n"
        '  pip install ".[gui]"\n\n'
        f"Detalhe tecnico: {exc}"
    )


def _notify_missing_gui_dependencies(exc: BaseException) -> None:
    message = _missing_gui_dependencies_message(exc)
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "Traderbot Launcher", 0x10)
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


def _launcher_logs_dir() -> Path:
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def main() -> int:
    logs_dir = _launcher_logs_dir()
    logger = setup_logger("traderbot-launcher", str(logs_dir))
    fault_handle = install_fatal_error_hooks("traderbot-launcher", logs_dir, logger=logger)
    try:
        try:
            from traderbot.launcher import main as launcher_main
        except ModuleNotFoundError as exc:
            missing_name = getattr(exc, "name", "") or ""
            if missing_name == "PySide6" or missing_name.startswith("PySide6."):
                logger.error("Dependencias graficas ausentes para iniciar o launcher: %s", exc)
                _notify_missing_gui_dependencies(exc)
                return 1
            raise
        except ImportError as exc:
            if "PySide6" in str(exc):
                logger.error("ImportError ao inicializar dependencias graficas do launcher: %s", exc)
                _notify_missing_gui_dependencies(exc)
                return 1
            raise
        except OSError as exc:
            logger.exception("Falha do sistema ao iniciar dependencias graficas do launcher.")
            _notify_missing_gui_dependencies(exc)
            return 1

        logger.info("Inicializando launcher desktop.")
        exit_code = int(launcher_main())
        logger.info("Launcher desktop encerrado com exit code=%s.", exit_code)
        return exit_code
    finally:
        uninstall_fatal_error_hooks(fault_handle, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
