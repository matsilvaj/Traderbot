from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass
class FaultLoggingHandle:
    name: str
    stream: TextIO
    previous_sys_excepthook: object
    previous_thread_excepthook: object | None


def _write_crash_banner(stream: TextIO, name: str, *, title: str, rendered_traceback: str) -> None:
    try:
        stream.write(f"\n[{title}] {name}\n")
        stream.write(rendered_traceback)
        if not rendered_traceback.endswith("\n"):
            stream.write("\n")
        stream.flush()
    except Exception:
        pass


def install_fatal_error_hooks(
    name: str,
    logs_dir: str | Path,
    logger: logging.Logger | None = None,
) -> FaultLoggingHandle:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    stream = (logs_path / f"{name}-fatal.log").open("a", encoding="utf-8", buffering=1)
    previous_sys_excepthook = sys.excepthook
    previous_thread_excepthook = getattr(threading, "excepthook", None)

    def _sys_excepthook(exc_type, exc_value, exc_traceback) -> None:
        rendered = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _write_crash_banner(stream, name, title="UNHANDLED EXCEPTION", rendered_traceback=rendered)
        if logger is not None:
            logger.critical("Excecao nao tratada em %s.", name, exc_info=(exc_type, exc_value, exc_traceback))
        try:
            previous_sys_excepthook(exc_type, exc_value, exc_traceback)
        except Exception:
            pass

    def _thread_excepthook(args) -> None:
        rendered = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        thread_name = getattr(args.thread, "name", "thread-desconhecida")
        _write_crash_banner(
            stream,
            name,
            title=f"UNHANDLED THREAD EXCEPTION ({thread_name})",
            rendered_traceback=rendered,
        )
        if logger is not None:
            logger.critical(
                "Excecao nao tratada na thread %s (%s).",
                thread_name,
                name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        if previous_thread_excepthook is None:
            return
        try:
            previous_thread_excepthook(args)
        except Exception:
            pass

    try:
        faulthandler.enable(file=stream, all_threads=True)
    except Exception:
        if logger is not None:
            logger.exception("Falha ao habilitar faulthandler para %s.", name)

    sys.excepthook = _sys_excepthook
    if previous_thread_excepthook is not None:
        threading.excepthook = _thread_excepthook

    return FaultLoggingHandle(
        name=name,
        stream=stream,
        previous_sys_excepthook=previous_sys_excepthook,
        previous_thread_excepthook=previous_thread_excepthook,
    )


def uninstall_fatal_error_hooks(handle: FaultLoggingHandle, logger: logging.Logger | None = None) -> None:
    sys.excepthook = handle.previous_sys_excepthook
    if handle.previous_thread_excepthook is not None:
        threading.excepthook = handle.previous_thread_excepthook

    try:
        faulthandler.disable()
    except Exception:
        if logger is not None:
            logger.exception("Falha ao desabilitar faulthandler para %s.", handle.name)

    try:
        handle.stream.flush()
    except Exception:
        pass

    try:
        handle.stream.close()
    except Exception:
        if logger is not None:
            logger.exception("Falha ao fechar arquivo fatal de %s.", handle.name)
