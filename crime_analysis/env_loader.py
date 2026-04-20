"""
Minimal `.env` loader — stdlib only, no dependency on python-dotenv.

Loads `KEY=VALUE` lines from `.env` into os.environ at import time.
Shell-exported vars take precedence (setdefault), so `export KEY=…` still
overrides the file. Silently no-ops if no `.env` exists.

搜尋順序：
  1. 呼叫時指定的 path
  2. 目前工作目錄的 .env
  3. 此模組所在目錄的 .env（crime_analysis/）
  4. crime_analysis/.env（repo root）

Use:
    from env_loader import load_dotenv
    load_dotenv()  # call once at entry point
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent


def _candidate_paths(explicit: Optional[Path]) -> List[Path]:
    if explicit is not None:
        return [Path(explicit)]
    return [
        Path.cwd() / ".env",
        _HERE / ".env",
        _HERE.parent / ".env",
    ]


def _parse_line(raw: str) -> Optional[tuple]:
    """Return (key, value) for a valid line, else None."""
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if not key or not key.replace("_", "").isalnum():
        return None
    # Trim matching quotes but keep interior whitespace
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
        value = value[1:-1]
    return key, value


def load_dotenv(
    path: Optional[Path] = None,
    *,
    override: bool = False,
    verbose: bool = False,
) -> List[str]:
    """
    Load `.env` into os.environ.

    Parameters
    ----------
    path : Path, optional
        Explicit file path. If None, first existing file among
        cwd/.env, crime_analysis/.env, repo-root/.env is used.
    override : bool
        If True, values from the file override existing env vars.
        Default False — shell exports win.
    verbose : bool
        If True, log which file was loaded and key names set (never values).

    Returns
    -------
    list[str]
        Names of env vars that were set from the file (in original order).
    """
    for candidate in _candidate_paths(path):
        if candidate.exists():
            return _load_from_file(candidate, override=override, verbose=verbose)
    if verbose:
        logger.debug("No .env file found in candidate paths.")
    return []


def _load_from_file(path: Path, *, override: bool, verbose: bool) -> List[str]:
    set_keys: List[str] = []
    try:
        raw_lines: Iterable[str] = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []

    for line in raw_lines:
        parsed = _parse_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
            set_keys.append(key)

    if verbose:
        if set_keys:
            logger.info("Loaded %d env var(s) from %s: %s",
                        len(set_keys), path, ", ".join(set_keys))
        else:
            logger.info("No new env vars set from %s", path)
    return set_keys
