from __future__ import annotations

import atexit
import logging
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Queue
from threading import Lock

_LOGGER_SETUP_LOCK = Lock()
_ASYNC_CONFIGURED_ATTR = "_traderbot_async_configured"
_QUEUE_LISTENER_ATTR = "_traderbot_queue_listener"


def _shutdown_async_logger(logger: logging.Logger) -> None:
    listener = getattr(logger, _QUEUE_LISTENER_ATTR, None)
    if listener is None:
        return

    try:
        listener.stop()
    except Exception:
        pass

    for handler in listener.handlers:
        try:
            handler.close()
        except Exception:
            pass

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    setattr(logger, _QUEUE_LISTENER_ATTR, None)
    setattr(logger, _ASYNC_CONFIGURED_ATTR, False)


def setup_logger(name: str, logs_dir: str, level: int = logging.INFO) -> logging.Logger:
    """Configura logger com escrita assincrona em arquivo rotativo e console."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    with _LOGGER_SETUP_LOCK:
        if getattr(logger, _ASYNC_CONFIGURED_ATTR, False):
            return logger

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            Path(logs_dir) / f"{name}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        log_queue: Queue[logging.LogRecord] = Queue()
        queue_handler = QueueHandler(log_queue)

        listener = QueueListener(
            log_queue,
            file_handler,
            console_handler,
            respect_handler_level=True,
        )
        listener.start()

        logger.addHandler(queue_handler)

        setattr(logger, _QUEUE_LISTENER_ATTR, listener)
        setattr(logger, _ASYNC_CONFIGURED_ATTR, True)

        atexit.register(_shutdown_async_logger, logger)

    return logger
