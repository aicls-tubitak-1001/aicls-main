"""Where a number came from: commit, working-tree state, timestamp (ADR-0015 §4).

A reported metric is only defensible if the code that produced it can be checked out again, so
every :class:`aicls.eval.report.EvalReport` carries a commit SHA and a ``dirty`` flag. This
module produces both **without running ``git``**: it reads ``.git/HEAD``, the ref file and
``.git/packed-refs`` directly, which works offline, needs no subprocess and cannot alter the
repository.

The working-tree state cannot be derived from those files alone, so the rule is conservative:

* ``dirty`` is ``True`` and ``dirty_known`` is ``False`` unless someone *proves* the tree is
  clean by handing in the output of ``git status --porcelain``;
* a run whose cleanliness is unknown is therefore never citable.

Erring towards "dirty" can only cost a report a citation; erring the other way would publish a
number nobody can reproduce. :func:`git_status_output` is the escape hatch for CI, which may
call ``git`` — the test suite and the evaluation harness never do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re

from aicls.core.logging import get_logger
from aicls.core.paths import find_repo_root

_logger = get_logger(__name__)

#: A full commit hash as git writes it.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Length of the short SHA used in file names and docs footnotes.
SHORT_SHA_LEN = 7

#: Placeholder short SHA for a run made outside a git checkout.
NO_GIT = "nogit"

#: git's "no such commit" placeholder, written as the parent of the first commit.
_NULL_SHA = "0" * 40


@dataclass(frozen=True, slots=True)
class GitInfo:
    """Commit identity and working-tree state of an evaluation run.

    Attributes:
        sha: Full commit hash, or ``""`` when the checkout could not be read.
        branch: Branch name, or ``""`` for a detached HEAD.
        dirty: Whether the tree carried uncommitted changes. ``True`` also when unknown.
        dirty_known: Whether ``dirty`` was actually measured (see the module docstring).
        source: ``"filesystem"`` (refs only), ``"status"`` (a porcelain output was supplied) or
            ``"provided"`` (the caller supplied everything).
    """

    sha: str = ""
    branch: str = ""
    dirty: bool = True
    dirty_known: bool = False
    source: str = "filesystem"

    @property
    def short_sha(self) -> str:
        """First seven characters of :attr:`sha`, or ``"nogit"``."""
        return self.sha[:SHORT_SHA_LEN] if self.sha else NO_GIT

    @property
    def citable(self) -> bool:
        """Whether a report produced under this state may be cited: committed and proven clean."""
        return bool(self.sha) and self.dirty_known and not self.dirty


def git_dir(root: Path | None = None) -> Path | None:
    """Return the ``.git`` directory of ``root`` (following a ``gitdir:`` file), or None.

    Args:
        root: Checkout root; the repository root of this source tree when omitted.

    Returns:
        The resolved git directory, or ``None`` when ``root`` is not a checkout.
    """
    base = Path(root) if root is not None else find_repo_root()
    candidate = base / ".git"
    if candidate.is_dir():
        return candidate
    if candidate.is_file():
        # a worktree or submodule: ".git" holds "gitdir: <path>"
        text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("gitdir:"):
            target = Path(text.split(":", 1)[1].strip())
            resolved = target if target.is_absolute() else (base / target)
            return resolved if resolved.is_dir() else None
    return None


def read_head(root: Path | None = None) -> tuple[str, str]:
    """Read ``(sha, branch)`` of HEAD straight from the git files.

    Args:
        root: Checkout root.

    Returns:
        The full SHA (``""`` when unreadable) and the branch name (``""`` when detached).
    """
    directory = git_dir(root)
    if directory is None:
        return "", ""
    head_file = directory / "HEAD"
    if not head_file.is_file():
        return "", ""
    head = head_file.read_text(encoding="utf-8", errors="replace").strip()
    if not head.startswith("ref:"):
        return (head if _SHA_RE.match(head) else ""), ""
    ref = head.split(":", 1)[1].strip()
    branch = ref.rsplit("/", 1)[-1]
    ref_file = directory / Path(ref)
    if ref_file.is_file():
        sha = ref_file.read_text(encoding="utf-8", errors="replace").strip()
        return (sha if _SHA_RE.match(sha) else ""), branch
    return _packed_ref(directory, ref), branch


def _packed_ref(directory: Path, ref: str) -> str:
    """Look ``ref`` up in ``.git/packed-refs`` (a freshly cloned repository has no loose ref)."""
    packed = directory / "packed-refs"
    if not packed.is_file():
        return ""
    for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or line.startswith("^") or " " not in line:
            continue
        sha, name = line.split(" ", 1)
        if name.strip() == ref and _SHA_RE.match(sha):
            return sha
    return ""


def known_shas(root: Path | None = None, *, limit: int = 5000) -> set[str]:
    """Commits HEAD is known to have pointed at, read from ``.git/logs/HEAD``.

    This is an **approximation** of ``git rev-list HEAD`` for offline use: it lists what this
    checkout recorded, so it can miss commits fetched into the branch by someone else. CI must
    pass the real ancestry to :func:`aicls.eval.report.check_report`; this helper only keeps a
    local ``report-check`` useful.

    Args:
        root: Checkout root.
        limit: Maximum number of reflog lines to read.

    Returns:
        The set of full SHAs (possibly empty).
    """
    directory = git_dir(root)
    if directory is None:
        return set()
    log = directory / "logs" / "HEAD"
    if not log.is_file():
        return set()
    shas: set[str] = set()
    with log.open(encoding="utf-8", errors="replace") as handle:
        for count, line in enumerate(handle):
            if count >= limit:
                break
            parts = line.split(" ", 2)
            for candidate in parts[:2]:
                if _SHA_RE.match(candidate) and candidate != _NULL_SHA:
                    shas.add(candidate)
    return shas


def parse_status_porcelain(output: str) -> bool:
    """Whether ``git status --porcelain`` output describes a dirty tree.

    Any non-empty line means something is modified, staged or untracked. An untracked file
    counts as dirty on purpose: a metric produced by a source file that is not committed yet is
    exactly the case this flag exists for.
    """
    return any(line.strip() for line in output.splitlines())


def git_status_output(root: Path | None = None, *, timeout_s: float = 10.0) -> str:
    """Run ``git status --porcelain`` and return its output (CI only).

    The evaluation harness and the test suite never call this: they either accept the
    conservative "unknown means dirty" default or receive a status string from their caller.
    It exists so that ``poe report-check`` can prove a tree clean.

    Args:
        root: Checkout root.
        timeout_s: Subprocess timeout.

    Returns:
        The porcelain output, or ``""`` when git is unavailable.
    """
    import subprocess  # local, read-only status call; CI entry point only

    base = Path(root) if root is not None else find_repo_root()
    try:  # pragma: no cover - not exercised by the test suite (it must not run git)
        completed = subprocess.run(  # fixed argv, no shell, no user input
            ["git", "status", "--porcelain"],
            cwd=base,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git missing
        return ""
    return completed.stdout if completed.returncode == 0 else ""  # pragma: no cover


def git_info(root: Path | None = None, *, status_output: str | None = None) -> GitInfo:
    """Collect the provenance of a run.

    Args:
        root: Checkout root.
        status_output: Output of ``git status --porcelain``; when omitted the tree state is
            unknown and the run is treated as dirty (and therefore not citable).

    Returns:
        The :class:`GitInfo`.
    """
    sha, branch = read_head(root)
    if status_output is None:
        info = GitInfo(sha=sha, branch=branch, dirty=True, dirty_known=False, source="filesystem")
        _logger.debug("eval.provenance.tree_state_unknown", sha=info.short_sha)
        return info
    return GitInfo(
        sha=sha,
        branch=branch,
        dirty=parse_status_porcelain(status_output),
        dirty_known=True,
        source="status",
    )


def utc_timestamp() -> str:
    """Current UTC time as ``2027-03-14T10:22:31Z`` (the report's ``timestamp`` format)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "NO_GIT",
    "SHORT_SHA_LEN",
    "GitInfo",
    "git_dir",
    "git_info",
    "git_status_output",
    "known_shas",
    "parse_status_porcelain",
    "read_head",
    "utc_timestamp",
]
