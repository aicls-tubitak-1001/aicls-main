"""Experiment runner template (docs/deneme-alani.md).

Copy via ``poe new-exp -- <slug>``. Keep it small: load ``config.yaml``, seed everything, use
``aicls.eval`` metrics and shared ``datasets/`` sets, write a compact ``results/summary.json``.
Large artefacts go to ``AICLS_DATA_ROOT/runs/<experiment>/``. Approval gates apply here too:
synthetic data, stub models, CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 1234


def load_config(path: Path) -> dict[str, object]:
    """Read the experiment config (YAML)."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data)


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch (torch optional)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run(config: dict[str, object]) -> dict[str, object]:
    """Replace with the actual experiment; return a JSON-serialisable summary."""
    return {"status": "template", "config": config}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "summary.json")
    args = parser.parse_args(argv)
    seed_all(args.seed)
    config = load_config(args.config)
    summary = run(config)
    summary["seed"] = args.seed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
