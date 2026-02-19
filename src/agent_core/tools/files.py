"""
agent_core/tools/files.py  —  File read/write tool
====================================================
Reads and writes text files safely.
Access is restricted to files within the current working directory.
"""

import logging
import os
from pathlib import Path

from agent_core.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer("tools.files")

_BASE_DIR = Path.cwd()


def _safe_path(path: str) -> Path:
    """Resolve path relative to _BASE_DIR; raise if outside."""
    p = (_BASE_DIR / path).resolve()
    if not str(p).startswith(str(_BASE_DIR)):
        raise ValueError(f"Path not allowed: {path}")
    return p


def read_file(path: str) -> str:
    """
    Read a text file and return its contents.

    Parameters
    ----------
    path : str
        File path (relative to current working directory)

    Returns
    -------
    str
        File content string
    """
    with tracer.start_as_current_span("tool-call") as span:
        from opentelemetry.trace import StatusCode

        span.set_attribute("tool.name", "read_file")
        span.set_attribute("tool.input.path", path)

        logger.info(f"[FileTool] read: {path}")
        try:
            safe = _safe_path(path)
            content = safe.read_text(encoding="utf-8")
            span.set_attribute("tool.output.length", len(content))
            span.set_attribute("tool.success", True)
            span.set_status(StatusCode.OK)
            logger.info(f"[FileTool] read complete: {len(content)} chars")
            return content
        except FileNotFoundError:
            msg = f"File not found: {path}"
            span.set_attribute("tool.success", False)
            span.set_attribute("tool.error", msg)
            span.set_status(StatusCode.ERROR, msg)
            logger.warning(f"[FileTool] {msg}")
            return f"[Error] {msg}"
        except Exception as e:
            span.set_attribute("tool.success", False)
            span.set_attribute("tool.error", str(e))
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"[FileTool] read failed: {e}")
            return f"[Error] {e}"


def list_files(path: str = ".") -> list[str]:
    """
    Return a list of files in a directory.

    Parameters
    ----------
    path : str
        Directory path (relative to cwd, default ".")

    Returns
    -------
    list[str]
        Relative path strings
    """
    with tracer.start_as_current_span("tool-call") as span:
        from opentelemetry.trace import StatusCode

        span.set_attribute("tool.name", "list_files")
        span.set_attribute("tool.input.path", path)

        logger.info(f"[FileTool] list: {path}")
        try:
            safe = _safe_path(path)
            if not safe.is_dir():
                raise NotADirectoryError(f"Not a directory: {path}")

            entries = sorted(
                str(p.relative_to(_BASE_DIR))
                for p in safe.rglob("*")
                if p.is_file()
            )
            span.set_attribute("tool.output.count", len(entries))
            span.set_attribute("tool.success", True)
            span.set_status(StatusCode.OK)
            logger.info(f"[FileTool] found {len(entries)} files")
            return entries
        except Exception as e:
            span.set_attribute("tool.success", False)
            span.set_attribute("tool.error", str(e))
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"[FileTool] list failed: {e}")
            return []


def write_file(path: str, content: str) -> str:
    """
    Write content to a text file (creates file if it does not exist).

    Parameters
    ----------
    path : str
        File path (relative to cwd)
    content : str
        Content to write

    Returns
    -------
    str
        Success/failure message
    """
    with tracer.start_as_current_span("tool-call") as span:
        from opentelemetry.trace import StatusCode

        span.set_attribute("tool.name", "write_file")
        span.set_attribute("tool.input.path", path)
        span.set_attribute("tool.input.content_length", len(content))

        logger.info(f"[FileTool] write: {path} ({len(content)} chars)")
        try:
            safe = _safe_path(path)
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8")
            msg = f"File saved: {path} ({len(content)} chars written)"
            span.set_attribute("tool.success", True)
            span.set_attribute("tool.output.message", msg)
            span.set_status(StatusCode.OK)
            logger.info(f"[FileTool] {msg}")
            return msg
        except Exception as e:
            msg = f"[Error] write failed ({path}): {e}"
            span.set_attribute("tool.success", False)
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("tool.error", str(e))
            logger.error(f"[FileTool] {msg}")
            return msg
