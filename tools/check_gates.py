"""Static approval-gate scan (ADR-0012): find implicit downloads in src/ and training/.

Flags, outside the explicitly gated modules:

* ``from_pretrained("org/name")`` / ``load_dataset("org/name")`` with a Hugging Face hub id
* ``hf_hub_download`` / ``snapshot_download`` / ``torch.hub``
* torchvision ``weights=`` other than ``None`` and ``pretrained=True``
* ``urllib.request`` / ``urlopen`` / ``requests.get|post`` and any literal HF / PyTorch URL

Allowed exceptions: ``training/scripts/download_*.py`` and ``scripts/install_ml.ps1`` (they call
``require_gate``), plus a line ending in ``# gate-ok: <reason>`` after review. Exit code 1 on hits.
Run as ``python tools/check_gates.py`` (CI job ``gates``) or on explicit paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import contextlib
from dataclasses import dataclass
from pathlib import Path
import re
import sys

SCAN_DIRS = ("src", "training", "experiments", "scripts", "tools")
SCAN_SUFFIXES = {".py", ".ps1", ".sh", ".yaml", ".yml", ".toml"}
ALLOWED_GATED_FILES = (
    re.compile(r"^training/scripts/download_[a-z0-9_]+\.py$"),
    re.compile(r"^scripts/install_ml\.ps1$"),
    re.compile(r"^scripts/install_ml\.sh$"),
    re.compile(r"^tools/check_gates\.py$"),  # this file lists the patterns
)
ALLOW_MARK = "# gate-ok:"

HUB_ID = r"""["'][A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9_.\-]+["']"""

# (rule id, compiled regex, message)
RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "hub-id",
        re.compile(r"\bfrom_pretrained\(\s*" + HUB_ID),
        "from_pretrained() with a hub id downloads weights — pass a local Path under "
        "AICLS_MODELS_ROOT (gate WEIGHTS_DOWNLOAD)",
    ),
    (
        "hub-id",
        re.compile(r"\bload_dataset\(\s*" + HUB_ID),
        "load_dataset() with a hub id downloads data (gate DATASET_DOWNLOAD)",
    ),
    (
        "hf-download",
        re.compile(r"\b(hf_hub_download|snapshot_download)\s*\("),
        "explicit Hugging Face download call (gate WEIGHTS_DOWNLOAD / DATASET_DOWNLOAD)",
    ),
    (
        "torch-hub",
        re.compile(r"\btorch\.hub\."),
        "torch.hub fetches code/weights from the network (gate WEIGHTS_DOWNLOAD)",
    ),
    (
        "torchvision-weights",
        re.compile(r"[(,]\s*weights\s*=\s*(?!None\b)\S"),
        "torchvision weights= must be None; pre-trained encoders are loaded from a local "
        "checkpoint (ADR-0012, test_torchvision_encoders_built_with_weights_none)",
    ),
    (
        "torchvision-weights",
        re.compile(r"\bpretrained\s*=\s*True\b"),
        "pretrained=True triggers a download; use pretrained=False and a local checkpoint",
    ),
    (
        "url-fetch",
        re.compile(r"\b(urllib\.request|urlopen\s*\(|requests\.(get|post|head)\s*\()"),
        "raw network fetch; only gated download scripts may fetch",
    ),
    (
        "url-literal",
        re.compile(
            r"https?://(huggingface\.co|hf\.co|download\.pytorch\.org|storage\.googleapis\.com/mathwriting|commonvoice\.mozilla\.org)",
        ),
        "hub / dataset URL literal outside a gated script",
    ),
    (
        "hub-id",
        re.compile(
            r"\b(AutoModel\w*|AutoTokenizer|AutoProcessor|AutoConfig|Whisper\w+|Wav2Vec2\w+)\.from_pretrained\(\s*\w*model_id\b"
        ),
        "from_pretrained(model_id) — ensure model_id resolves to a local path (review needed)",
    ),
)


@dataclass(frozen=True, slots=True)
class Hit:
    """One flagged line."""

    path: Path
    line_no: int
    rule: str
    line: str
    message: str

    def render(self, root: Path | None = None) -> str:
        """Human-readable one-liner."""
        p = self.path
        if root is not None:
            with contextlib.suppress(ValueError):
                p = p.relative_to(root)
        return (
            f"{p.as_posix()}:{self.line_no}: [{self.rule}] {self.line.strip()}\n"
            f"    -> {self.message}"
        )


def _is_allowed_file(rel_posix: str) -> bool:
    return any(pat.match(rel_posix) for pat in ALLOWED_GATED_FILES)


def _is_comment(line: str, suffix: str) -> bool:
    s = line.lstrip()
    if suffix in {".py", ".ps1", ".sh", ".yaml", ".yml", ".toml"}:
        return s.startswith("#")
    return False


def scan_text(text: str, path: Path, *, suffix: str | None = None) -> list[Hit]:
    """Scan *text* (already read) and return hits; comments and ``# gate-ok:`` lines are skipped."""
    suffix = suffix or path.suffix
    hits: list[Hit] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if _is_comment(line, suffix) or ALLOW_MARK in line:
            continue
        for rule, pattern, message in RULES:
            if pattern.search(line):
                hits.append(Hit(path, i, rule, line, message))
                break
    return hits


def scan_file(path: Path, *, root: Path | None = None) -> list[Hit]:
    """Scan one file, honouring the gated-file allow list relative to *root*."""
    if root is not None:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        if _is_allowed_file(rel):
            return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return scan_text(text, path)


def iter_files(root: Path, dirs: Iterable[str] = SCAN_DIRS) -> Iterable[Path]:
    """Yield scannable files below *root*/<dirs>."""
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in SCAN_SUFFIXES and "__pycache__" not in p.parts:
                yield p


def scan_tree(root: Path, dirs: Iterable[str] = SCAN_DIRS) -> list[Hit]:
    """Scan the default directories under *root*."""
    hits: list[Hit] = []
    for p in iter_files(root, dirs):
        hits.extend(scan_file(p, root=root))
    return hits


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit 1 when any hit is found."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="*", type=Path, help="files/dirs to scan (default: src, training, ...)"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()

    hits: list[Hit] = []
    if args.paths:
        for arg in args.paths:
            p = arg if arg.is_absolute() else root / arg
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f.suffix in SCAN_SUFFIXES:
                        hits.extend(scan_file(f, root=root))
            elif p.is_file():
                hits.extend(scan_file(p, root=root))
    else:
        hits = scan_tree(root)

    scanned = ", ".join(d for d in SCAN_DIRS if (root / d).is_dir()) or "(nothing)"
    if hits:
        print(f"check_gates: {len(hits)} implicit download pattern(s) found:")
        for h in hits:
            print(h.render(root))
        print(
            "\nUse the gated scripts (training/scripts/download_*.py, scripts/install_ml.ps1) "
            "or add `# gate-ok: <reason>` after review (ADR-0012)."
        )
        return 1
    print(f"check_gates: OK — no implicit download patterns in {scanned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
