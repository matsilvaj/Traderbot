from __future__ import annotations

import os
import sys


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


def main() -> int:
    try:
        from traderbot.launcher import main as launcher_main
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or ""
        if missing_name == "PySide6" or missing_name.startswith("PySide6."):
            _notify_missing_gui_dependencies(exc)
            return 1
        raise
    except ImportError as exc:
        if "PySide6" in str(exc):
            _notify_missing_gui_dependencies(exc)
            return 1
        raise
    except OSError as exc:
        _notify_missing_gui_dependencies(exc)
        return 1
    return int(launcher_main())


if __name__ == "__main__":
    raise SystemExit(main())
