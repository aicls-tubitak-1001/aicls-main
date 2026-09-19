"""``python -m aicls.eval`` — run the evaluation harness from the repository root."""

from __future__ import annotations

import sys

from aicls.eval.harness import main

if __name__ == "__main__":
    sys.exit(main())
