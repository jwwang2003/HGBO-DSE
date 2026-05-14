from __future__ import annotations

import os
from pathlib import Path


HGBO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HGBO_ROOT.parents[1]
RAPIDWRIGHT_PATH = REPO_ROOT / "3rdParty" / "RapidWright"

_IMPORT_ERROR: Exception | None = None


def configure_rapidwright_env() -> None:
    if not RAPIDWRIGHT_PATH.exists():
        return

    os.environ.setdefault("RAPIDWRIGHT_PATH", str(RAPIDWRIGHT_PATH))
    classpath_parts = [
        str(RAPIDWRIGHT_PATH / "bin"),
        str(RAPIDWRIGHT_PATH / "jars" / "*"),
        str(RAPIDWRIGHT_PATH / "build" / "libs" / "rapidwright.jar"),
    ]
    existing = os.environ.get("CLASSPATH", "").strip()
    if existing:
        classpath_parts.append(existing)
    os.environ["CLASSPATH"] = os.pathsep.join(classpath_parts)


configure_rapidwright_env()

try:
    import rapidwright  # type: ignore[import-untyped]  # noqa: F401
    from com.xilinx.rapidwright.device import Device  # type: ignore[import-untyped]  # noqa: F401
except Exception as exc:  # pragma: no cover
    Device = None
    _IMPORT_ERROR = exc


def require_rapidwright_device() -> object:
    if Device is None:
        detail = f" Original import error: {_IMPORT_ERROR!r}" if _IMPORT_ERROR else ""
        raise RuntimeError(
            "RapidWright Python API is not available in HGBO-DSE. "
            "Run `uv sync` in 3rdParty/HGBO-DSE so the local "
            "`../RapidWright/python` package is installed."
            f"{detail}"
        )
    return Device
