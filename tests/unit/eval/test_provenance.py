"""Commit identity and working-tree state, read from ``.git`` without running git.

The fixtures build small fake checkouts, so these tests never touch the real repository and
never spawn a subprocess.
"""

from __future__ import annotations

from pathlib import Path

from aicls.eval.provenance import (
    NO_GIT,
    GitInfo,
    git_dir,
    git_info,
    known_shas,
    parse_status_porcelain,
    read_head,
    utc_timestamp,
)

SHA = "1" * 40
OLD_SHA = "2" * 40


def _checkout(tmp_path: Path, *, head: str = "ref: refs/heads/main", sha: str = SHA) -> Path:
    """A minimal fake checkout with HEAD, a branch ref and a reflog."""
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "logs").mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")
    (git / "logs" / "HEAD").write_text(
        f"{'0' * 40} {OLD_SHA} Author <a@b> 1 +0000\tcommit (initial)\n"
        f"{OLD_SHA} {sha} Author <a@b> 2 +0000\tcommit\n",
        encoding="utf-8",
    )
    return tmp_path


def test_read_head_follows_the_branch_ref(tmp_path: Path) -> None:
    """``HEAD -> refs/heads/main -> sha`` is read straight from the files."""
    root = _checkout(tmp_path)
    assert read_head(root) == (SHA, "main")


def test_read_head_of_a_detached_head(tmp_path: Path) -> None:
    """A detached HEAD holds the SHA itself and has no branch name."""
    root = _checkout(tmp_path, head=SHA)
    assert read_head(root) == (SHA, "")


def test_read_head_falls_back_to_packed_refs(tmp_path: Path) -> None:
    """A freshly cloned repository has no loose ref file."""
    root = _checkout(tmp_path)
    (root / ".git" / "refs" / "heads" / "main").unlink()
    (root / ".git" / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{SHA} refs/heads/main\n"
        f"^{OLD_SHA}\ngarbage-line\n",
        encoding="utf-8",
    )
    assert read_head(root) == (SHA, "main")


def test_read_head_of_a_directory_that_is_not_a_checkout(tmp_path: Path) -> None:
    """Outside a checkout there is no SHA — and no exception either."""
    assert read_head(tmp_path) == ("", "")
    assert git_dir(tmp_path) is None


def test_read_head_without_a_head_file_or_with_junk(tmp_path: Path) -> None:
    """A malformed checkout degrades to "no commit", never to a wrong commit."""
    root = _checkout(tmp_path, head="not-a-sha")
    assert read_head(root) == ("", "")
    (root / ".git" / "HEAD").unlink()
    assert read_head(root) == ("", "")


def test_git_dir_follows_a_gitdir_pointer_file(tmp_path: Path) -> None:
    """A worktree's ``.git`` is a file pointing at the real directory."""
    real = _checkout(tmp_path / "real")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {real / '.git'}\n", encoding="utf-8")
    assert git_dir(worktree) == real / ".git"
    assert read_head(worktree) == (SHA, "main")


def test_git_dir_pointer_to_a_missing_directory(tmp_path: Path) -> None:
    """A dangling pointer file is "not a checkout", not a crash."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: ../gone\n", encoding="utf-8")
    assert git_dir(worktree) is None


def test_known_shas_reads_the_reflog(tmp_path: Path) -> None:
    """The offline ancestry approximation lists what HEAD has pointed at."""
    root = _checkout(tmp_path)
    assert known_shas(root) == {SHA, OLD_SHA}
    assert known_shas(root, limit=1) == {OLD_SHA}
    assert known_shas(tmp_path / "elsewhere") == set()


def test_known_shas_without_a_reflog(tmp_path: Path) -> None:
    """A checkout whose reflog was pruned yields an empty set, not an error."""
    root = _checkout(tmp_path)
    (root / ".git" / "logs" / "HEAD").unlink()
    assert known_shas(root) == set()


def test_parse_status_porcelain_treats_untracked_files_as_dirty() -> None:
    """An uncommitted new source file is exactly what the dirty flag must catch."""
    assert parse_status_porcelain("") is False
    assert parse_status_porcelain("\n \n") is False
    assert parse_status_porcelain(" M src/aicls/eval/metrics.py\n") is True
    assert parse_status_porcelain("?? src/aicls/eval/new_file.py\n") is True


def test_git_info_without_a_status_is_dirty_and_not_citable(tmp_path: Path) -> None:
    """Unknown tree state is treated as dirty: a report can only lose a citation, never gain one."""
    info = git_info(_checkout(tmp_path))
    assert (info.sha, info.branch) == (SHA, "main")
    assert info.dirty is True
    assert info.dirty_known is False
    assert info.citable is False
    assert info.short_sha == "1" * 7


def test_git_info_with_a_clean_status_is_citable(tmp_path: Path) -> None:
    """A proven-clean tree at a real commit is the only citable combination."""
    info = git_info(_checkout(tmp_path), status_output="")
    assert (info.dirty, info.dirty_known, info.citable) == (False, True, True)
    assert info.source == "status"


def test_git_info_with_a_dirty_status(tmp_path: Path) -> None:
    """A modified tree stays uncitable even though the SHA is known."""
    info = git_info(_checkout(tmp_path), status_output=" M pyproject.toml\n")
    assert (info.dirty, info.dirty_known, info.citable) == (True, True, False)


def test_git_info_outside_a_checkout_has_no_short_sha(tmp_path: Path) -> None:
    """``nogit`` is the file-name placeholder for a run made outside version control."""
    info = git_info(tmp_path, status_output="")
    assert info.sha == ""
    assert info.short_sha == NO_GIT
    assert info.citable is False


def test_default_git_info_is_conservative() -> None:
    """The dataclass defaults describe "we know nothing", which is the safe direction."""
    assert GitInfo() == GitInfo(sha="", branch="", dirty=True, dirty_known=False)
    assert GitInfo().citable is False


def test_utc_timestamp_is_second_precision_zulu() -> None:
    """The report timestamp format is fixed so files sort and diff predictably."""
    stamp = utc_timestamp()
    assert stamp.endswith("Z")
    assert len(stamp) == len("2027-03-14T10:22:31Z")
    assert "." not in stamp
