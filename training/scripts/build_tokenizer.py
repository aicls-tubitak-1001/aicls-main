"""Build ``vocab.json`` from prepared shards (**ungated**).

The LaTeX vocabulary is derived from the *training* labels only — never from valid or test —
so a token that only ever appears in the test set stays out of the model's output space and
cannot leak an answer into it. The work itself is
:func:`aicls.data.shards.build_vocab_from_shards`; this script is the command line around it.

Ungated: it reads shards that ``training/scripts/prepare_mathwriting.py`` already produced.

Usage (from the repository root, project venv)::

    python training/scripts/build_tokenizer.py --dry-run
    python training/scripts/build_tokenizer.py --split train --min-freq 2

The written file is what a config points ``model.vocab_path`` at, and the same document is
copied next to every checkpoint (``vocab.json``) so a checkpoint can always be decoded with the
vocabulary it was trained with.

Exit codes: ``0`` success or dry run, ``1`` a typed AI-CLS error (missing shard manifest).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

from aicls.core.errors import AiclsError  # noqa: E402
from aicls.core.logging import get_logger  # noqa: E402
from aicls.core.settings import get_settings  # noqa: E402
from aicls.data import build_vocab_from_shards, load_shard_manifest  # noqa: E402
from aicls.ocr import LatexTokenizer  # noqa: E402
from aicls.ocr.curriculum import DEFAULT_EXTRA_TOKENS  # noqa: E402

_log = get_logger(__name__)

#: Default shards root (the output of ``prepare_mathwriting.py``).
DEFAULT_SHARDS_SUBDIR = "mathwriting/shards"

#: Default vocabulary path, next to the shards root.
DEFAULT_VOCAB_NAME = "vocab.json"

#: Splits the vocabulary is built from. Training labels only, on purpose.
DEFAULT_SPLITS: tuple[str, ...] = ("train",)


def describe_sources(root: Path, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read each split's shard manifest so the plan can state what the vocabulary covers.

    Args:
        root: Shards root.
        splits: Splits to describe.

    Returns:
        One row per split with its sample count and fingerprint; a split whose manifest is
        missing is reported with ``available=False`` rather than raising, so ``--dry-run``
        can show the whole picture at once.
    """
    rows: list[dict[str, Any]] = []
    for split in splits:
        try:
            manifest = load_shard_manifest(root, split)
        except AiclsError as exc:
            rows.append({"split": split, "available": False, "error": str(exc.details)})
            continue
        rows.append(
            {
                "split": split,
                "available": True,
                "samples": manifest.n_samples,
                "shards": manifest.n_shards,
                "fingerprint": manifest.fingerprint,
            }
        )
    return rows


def build(
    root: Path, out: Path, *, splits: tuple[str, ...] = DEFAULT_SPLITS, min_freq: int = 1
) -> tuple[Path, int]:
    """Build the vocabulary and return ``(path, vocab_size)``.

    Args:
        root: Shards root.
        out: Destination ``vocab.json``.
        splits: Splits whose labels feed the vocabulary.
        min_freq: Drop tokens rarer than this.

    Returns:
        The written path and the resulting vocabulary size (specials included).

    Raises:
        NotFoundError: If a split has no shard manifest.
    """
    path = build_vocab_from_shards(
        root, out, splits=tuple(splits), min_freq=min_freq, extra_tokens=DEFAULT_EXTRA_TOKENS
    )
    tokenizer = LatexTokenizer.from_file(path)
    _log.info(
        "training.build_tokenizer.done",
        path=str(path),
        vocab_size=tokenizer.vocab_size,
        splits=list(splits),
        min_freq=min_freq,
    )
    return path, int(tokenizer.vocab_size)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    settings = get_settings()
    shards_default = settings.data_root / DEFAULT_SHARDS_SUBDIR
    parser = argparse.ArgumentParser(
        prog="build_tokenizer.py",
        description="Build vocab.json from prepared shards (ungated; local data only).",
    )
    parser.add_argument(
        "--shards",
        type=Path,
        default=shards_default,
        help="shards root (default AICLS_DATA_ROOT/mathwriting/shards)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=shards_default.parent / DEFAULT_VOCAB_NAME,
        help="destination vocab.json (default next to the shards root)",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="splits whose labels feed the vocabulary (training labels only by default)",
    )
    parser.add_argument("--min-freq", type=int, default=1, help="drop tokens rarer than this")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit 0, writing nothing"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line (``sys.argv[1:]`` when omitted).

    Returns:
        Process exit code: 0 ok/dry run, 1 typed error.
    """
    args = build_parser().parse_args(argv)
    root, out = Path(args.shards), Path(args.out)
    splits = tuple(str(name) for name in args.splits)

    sources = describe_sources(root, splits)
    print("plan: build LaTeX vocabulary (ungated)")
    print(f"  shards root : {root}{'' if root.is_dir() else '   (MISSING)'}")
    print(f"  out         : {out}")
    print(f"  splits      : {', '.join(splits)}   min_freq: {args.min_freq}")
    for row in sources:
        if row["available"]:
            print(
                f"  {row['split']:10s} {row['samples']:>8d} labels  "
                f"{row['shards']:>4d} shards  {row['fingerprint']}"
            )
        else:
            print(f"  {row['split']:10s} shard manifest MISSING — {row['error']}")
    if args.dry_run:
        print("dry run: nothing was written.")
        return 0

    try:
        path, size = build(root, out, splits=splits, min_freq=int(args.min_freq))
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)

    summary = {
        "vocab": str(path),
        "vocab_size": size,
        "splits": list(splits),
        "min_freq": int(args.min_freq),
        "sources": sources,
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"vocab written: {path} ({size} tokens, specials included)")
        print(f"next: set model.vocab_path: {path} in the OCR config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
