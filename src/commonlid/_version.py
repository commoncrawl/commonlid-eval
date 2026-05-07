"""Single source of truth for the package version.

Read from the wheel metadata so it can never drift from ``pyproject.toml``.
The fallback only fires for source-tree imports that were never installed
(e.g. ``PYTHONPATH=src python …``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("commonlid")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
