"""Structured logging via structlog, with CLEF shipping to Seq."""
from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import structlog

_SEQ_LEVEL_MAP = {
    "DEBUG": "Debug",
    "INFO": "Information",
    "WARNING": "Warning",
    "ERROR": "Error",
    "CRITICAL": "Fatal",
}

_BATCH_SIZE = 50
_FLUSH_INTERVAL_S = 2.0


class SeqHandler(logging.Handler):
    """Ships CLEF-formatted log events to Seq's ingestion API in batches."""

    def __init__(self, seq_url: str, api_key: str = "") -> None:
        super().__init__()
        self._url = f"{seq_url.rstrip('/')}/api/events/raw?clef"
        self._api_key = api_key
        self._queue: queue.Queue[str] = queue.Queue()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def emit(self, record: logging.LogRecord) -> None:
        event: dict[str, Any] = {
            "@t": datetime.now(timezone.utc).isoformat(),
            "@mt": record.getMessage(),
            "@l": _SEQ_LEVEL_MAP.get(record.levelname, "Information"),
            "SourceContext": record.name,
        }
        if record.exc_info and record.exc_info[1]:
            event["@x"] = self.format(record).split("\n", 1)[-1]
        extra = getattr(record, "_structlog_extra", None)
        if isinstance(extra, dict):
            event.update(extra)
        self._queue.put(json.dumps(event, default=str))

    def _worker(self) -> None:
        while not self._shutdown.is_set():
            batch: list[str] = []
            try:
                batch.append(self._queue.get(timeout=_FLUSH_INTERVAL_S))
                while len(batch) < _BATCH_SIZE:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue
            self._ship(batch)

    def _ship(self, batch: list[str]) -> None:
        payload = "\n".join(batch)
        headers: dict[str, str] = {"Content-Type": "application/vnd.serilog.clef"}
        if self._api_key:
            headers["X-Seq-ApiKey"] = self._api_key
        try:
            req = Request(self._url, data=payload.encode(), headers=headers, method="POST")
            urlopen(req, timeout=5)  # noqa: S310
        except (URLError, OSError):
            pass

    def close(self) -> None:
        self._shutdown.set()
        self._thread.join(timeout=5)
        remaining: list[str] = []
        while not self._queue.empty():
            try:
                remaining.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if remaining:
            self._ship(remaining)
        super().close()


def _add_structlog_extra(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Stash structlog context into the LogRecord so SeqHandler can pick it up."""
    record: logging.LogRecord | None = event_dict.get("_record")
    if record is not None:
        extra = {k: v for k, v in event_dict.items() if k not in (
            "event", "_record", "_from_structlog",
        )}
        record._structlog_extra = extra  # type: ignore[attr-defined]
    return event_dict


def setup_logging(seq_url: str = "", seq_api_key: str = "", level: int = logging.INFO) -> None:
    """Configure structlog wrapping stdlib logging, with optional Seq shipping."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            _add_structlog_extra,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if seq_url:
        root.addHandler(SeqHandler(seq_url, api_key=seq_api_key))
