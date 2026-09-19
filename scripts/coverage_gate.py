"""Enforce per-package coverage tiers (ADR-0010) on a Cobertura ``coverage.xml``.

Thresholds come from ``[tool.aicls.coverage]`` in ``pyproject.toml`` (``overall`` plus dotted
package names; a more specific key such as ``aicls.eval.metrics`` applies to that module only).
Line coverage is computed per package from the ``<class filename=...>`` entries so that the
result does not depend on how coverage.py groups packages.

Packages that have no measured lines yet (not implemented) are reported as ``n/a`` and skipped
unless ``--require-all`` is given (used at ``report-YYYY-MM`` tags).

Usage::

    python scripts/coverage_gate.py coverage.xml [--require-all] [--pyproject pyproject.toml]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
import xml.etree.ElementTree as ET


@dataclass(frozen=True, slots=True)
class LineStats:
    """Covered / valid line counts."""

    covered: int
    valid: int

    @property
    def rate(self) -> float:
        """Percentage in ``[0, 100]``; ``0`` when nothing is measured."""
        return 100.0 * self.covered / self.valid if self.valid else 0.0


def load_thresholds(pyproject: Path) -> dict[str, float]:
    """Read ``[tool.aicls.coverage]``."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    table = data.get("tool", {}).get("aicls", {}).get("coverage", {})
    return {str(k): float(v) for k, v in table.items()}


def module_name_from_filename(filename: str) -> str:
    """``src/aicls/eval/metrics.py`` or ``aicls/eval/metrics.py`` -> ``aicls.eval.metrics``."""
    parts = Path(filename.replace("\\", "/")).with_suffix("").parts
    if "aicls" in parts:
        parts = parts[parts.index("aicls") :]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def collect_module_stats(xml_path: Path) -> dict[str, LineStats]:
    """Per-module line stats from a Cobertura XML."""
    tree = ET.parse(xml_path)
    stats: dict[str, LineStats] = {}
    for cls in tree.getroot().iter("class"):
        filename = cls.get("filename", "")
        if not filename:
            continue
        lines = cls.find("lines")
        if lines is None:
            continue
        valid = 0
        covered = 0
        for line in lines.iter("line"):
            valid += 1
            if int(line.get("hits", "0")) > 0:
                covered += 1
        name = module_name_from_filename(filename)
        prev = stats.get(name)
        if prev is not None:
            covered += prev.covered
            valid += prev.valid
        stats[name] = LineStats(covered, valid)
    return stats


def aggregate(stats: dict[str, LineStats], prefix: str) -> LineStats:
    """Sum stats over modules equal to *prefix* or below it."""
    covered = valid = 0
    for name, st in stats.items():
        if name == prefix or name.startswith(prefix + "."):
            covered += st.covered
            valid += st.valid
    return LineStats(covered, valid)


def evaluate(
    stats: dict[str, LineStats], thresholds: dict[str, float], *, require_all: bool
) -> tuple[list[str], list[str]]:
    """Return ``(report_lines, failures)``."""
    report: list[str] = []
    failures: list[str] = []
    by_pkg: dict[str, LineStats] = defaultdict(lambda: LineStats(0, 0))
    for key, minimum in sorted(thresholds.items(), key=lambda kv: (kv[0] != "overall", kv[0])):
        st = aggregate(stats, "aicls") if key == "overall" else aggregate(stats, key)
        by_pkg[key] = st
        if st.valid == 0:
            report.append(f"  {key:<24} n/a      (min {minimum:>5.1f})  no measured lines")
            if require_all:
                failures.append(f"{key}: no measured lines but --require-all given")
            continue
        status = "OK " if st.rate >= minimum else "LOW"
        report.append(f"  {key:<24} {st.rate:>6.1f}%  (min {minimum:>5.1f})  {status}")
        if st.rate < minimum:
            failures.append(f"{key}: {st.rate:.1f}% < {minimum:.1f}%")
    return report, failures


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xml", type=Path, nargs="?", default=Path("coverage.xml"))
    parser.add_argument(
        "--pyproject", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    parser.add_argument(
        "--require-all", action="store_true", help="fail on packages with no measured lines"
    )
    args = parser.parse_args(argv)
    if not args.xml.is_file():
        print(f"coverage_gate: {args.xml} not found (run `poe test` first)")
        return 1
    thresholds = load_thresholds(args.pyproject)
    if not thresholds:
        print("coverage_gate: no [tool.aicls.coverage] thresholds in pyproject.toml")
        return 1
    stats = collect_module_stats(args.xml)
    report, failures = evaluate(stats, thresholds, require_all=args.require_all)
    print("coverage_gate: per-package tiers (ADR-0010)")
    print("\n".join(report))
    if failures:
        print("\ncoverage_gate: FAIL")
        print("\n".join(f"  {f}" for f in failures))
        return 1
    print("\ncoverage_gate: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
