"""Loguru-based logging configuration for uav_nav.

Provides a single ``setup_logger`` entry point that configures loguru with:
- Console sink with coloured, human-readable output.
- Rotating file sink for persistent logs.
- Optional Rerun SDK sink for real-time visualisation.
- Structured JSON sink for post-hoc analysis.

Usage::

    from uav_nav.runtime.logger import setup_logger, get_logger

    setup_logger(log_dir="logs/", level="DEBUG")
    logger = get_logger(__name__)
    logger.info("Pipeline started", frame=0)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger


def setup_logger(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "50 MB",
    retention: str = "7 days",
    json_logs: bool = False,
    rerun_sink: bool = False,
    colorize: bool = True,
) -> None:
    """Configure loguru for the uav_nav package.

    Removes the default loguru handler and installs:
    - A stderr sink with the given log level and optional colourisation.
    - A rotating file sink if ``log_dir`` is provided.
    - An optional JSON structured sink for machine-readable logs.
    - An optional Rerun SDK text-log sink for live visualisation.

    Args:
        log_dir: Directory for log files. If None, file logging is disabled.
        level: Minimum log level string (e.g. "DEBUG", "INFO", "WARNING").
        rotation: Loguru rotation spec (e.g. "50 MB", "1 day").
        retention: Loguru retention spec (e.g. "7 days", "10 files").
        json_logs: If True, write a parallel JSON-structured log file.
        rerun_sink: If True, forward log messages to Rerun SDK (requires
            ``rerun-sdk`` installed and a Rerun viewer running).
        colorize: If True, colour the console output.
    """
    _loguru_logger.remove()

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=console_format,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        _loguru_logger.add(
            log_dir / "uav_nav_{time:YYYY-MM-DD}.log",
            level=level,
            rotation=rotation,
            retention=retention,
            format=console_format,
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
        )

        if json_logs:
            _loguru_logger.add(
                log_dir / "uav_nav_{time:YYYY-MM-DD}.jsonl",
                level=level,
                rotation=rotation,
                retention=retention,
                serialize=True,
                encoding="utf-8",
            )

    if rerun_sink:
        _setup_rerun_sink(level)

    _loguru_logger.debug("Logger configured: level={}, log_dir={}", level, log_dir)


def _setup_rerun_sink(level: str) -> None:
    """Attach a Rerun SDK text-log sink to loguru.

    Args:
        level: Minimum log level to forward to Rerun.

    Raises:
        NotImplementedError: Rerun sink integration not yet implemented.
    """
    raise NotImplementedError("_setup_rerun_sink is not yet implemented.")


def get_logger(name: str) -> "_LoggerAdapter":
    """Return a module-scoped loguru logger adapter.

    Args:
        name: Module name, typically ``__name__``.

    Returns:
        Loguru logger bound with the module context.

    Example::

        logger = get_logger(__name__)
        logger.info("Starting segmentation")
    """
    return _loguru_logger.bind(module=name)


class _LoggerAdapter:
    """Type stub for the loguru bound logger returned by get_logger.

    In practice loguru returns its own internal type; this stub is here
    for type-checker purposes only.
    """

    def debug(self, msg: str, **kwargs: object) -> None: ...
    def info(self, msg: str, **kwargs: object) -> None: ...
    def warning(self, msg: str, **kwargs: object) -> None: ...
    def error(self, msg: str, **kwargs: object) -> None: ...
    def critical(self, msg: str, **kwargs: object) -> None: ...
    def exception(self, msg: str, **kwargs: object) -> None: ...
