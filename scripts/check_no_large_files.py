"""Repository size guard (docs/gelistirme/is-akisi.md §5).

Blocks files larger than 1 MB; ``datasets/`` and ``docs/assets/`` may hold up to 5 MB (fixtures,
figures). Model weights, audio corpora and checkpoints never belong in git — they live under
``AICLS_DATA_ROOT`` (ADR-0008). Used as a pre-commit hook (file names passed as arguments) or
standalone, where it walks the tree (skipping .git, site/, node_modules/, venvs).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

DEFAULT_LIMIT = 1 * 1024 * 1024
RELAXED_LIMIT = 5 * 1024 * 1024
RELAXED_PREFIXES = ("datasets/", "docs/assets/", "tests/fixtures/")
SKIP_DIRS = {
    ".git",
    "site",
    "node_modules",
    ".venv",
    "venv",
    "htmlcov",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}
BINARY_WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}


def limit_for(rel_posix: str) -> int:
    """Byte limit that applies to *rel_posix* (repo-relative, forward slashes)."""
    return RELAXED_LIMIT if rel_posix.startswith(RELAXED_PREFIXES) else DEFAULT_LIMIT


def check_files(paths: list[Path], root: Path) -> list[str]:
    """Return problem strings for oversized files or weight files."""
    problems: list[str] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            rel = p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = p.as_posix()
        size = p.stat().st_size
        if p.suffix.lower() in BINARY_WEIGHT_SUFFIXES:
            problems.append(f"{rel}: model weight/checkpoint files are never committed (ADR-0008)")
            continue
        lim = limit_for(rel)
        if size > lim:
            problems.append(
                f"{rel}: {size / 1024 / 1024:.2f} MB > {lim / 1024 / 1024:.0f} MB limit"
            )
    return problems


def walk(root: Path) -> list[Path]:
    """All files below *root* except skipped directories."""
    out: list[Path] = []
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit 1 on any problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root: Path = args.root
    files = [f if f.is_absolute() else root / f for f in args.files] or walk(root)
    problems = check_files(files, root)
    if problems:
        print("check_no_large_files: FAIL")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print(f"check_no_large_files: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
