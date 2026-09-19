"""Thin entry point for ``poe env-check``; the implementation lives in ``tools/check_env.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_IMPL = Path(__file__).resolve().parents[1] / "tools" / "check_env.py"


def main(argv: list[str] | None = None) -> int:
    """Load tools/check_env.py by path (tools/ is not a package) and run its ``main``."""
    spec = importlib.util.spec_from_file_location("aicls_tools_check_env", _IMPL)
    if spec is None or spec.loader is None:
        print(f"check_env: implementation not found at {_IMPL}")
        return 1
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    result: int = module.main(argv)
    return result


if __name__ == "__main__":
    sys.exit(main())
