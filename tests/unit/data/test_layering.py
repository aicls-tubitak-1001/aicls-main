"""Layering guards for :mod:`aicls.data` (ADR-0002, ADR-0006).

Importing the dataset layer must stay cheap and torch-free: the API process never imports it, the
CLI imports it for ``aicls data ...`` and a training run pays for torch only when it actually
builds a loader. matplotlib is likewise paid for only by :func:`aicls.data.synth_image`.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

PROBE = (
    "import sys; import aicls.data; "
    "print('ok', int('torch' in sys.modules), int('transformers' in sys.modules), "
    "int('matplotlib' in sys.modules))"
)

#: Upper layers (ADR-0002) that importing :mod:`aicls.data` must not drag in.
UPPER_LAYERS = ("aicls.ml", "aicls.eval", "aicls.api", "aicls.cli", "aicls.nlu")

UPPER_LAYER_PROBE = (
    "import sys; import aicls.data; "
    f"print('ok', *[m for m in {UPPER_LAYERS!r} if m in sys.modules])"
)


def _probe(repo_root: Path, source: str) -> str:
    """Run ``source`` in a clean interpreter rooted at the repo and return its stdout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_importing_aicls_data_does_not_import_torch_or_matplotlib(repo_root: Path) -> None:
    assert _probe(repo_root, PROBE).split() == ["ok", "0", "0", "0"]


def test_data_package_does_not_import_upper_layers(repo_root: Path) -> None:
    """Importing :mod:`aicls.data` must not pull in any layer above it.

    Probed in a clean subprocess on purpose. Reading this process' ``sys.modules`` would only
    describe whichever tests happened to run first: once a sibling test legitimately imports
    :mod:`aicls.nlu`, an in-process check fails even though ``aicls.data`` never imports it.
    ``test_no_static_import_path_to_forbidden_modules`` covers the same ADR-0002 boundary
    statically; this one covers what an actual import *executes*.
    """
    assert _probe(repo_root, UPPER_LAYER_PROBE).split() == ["ok"]


def test_public_api_is_exported() -> None:
    from aicls import data

    expected = {
        "DataManifest",
        "VerifyReport",
        "make_splits",
        "mathwriting_official",
        "prepare_shards",
        "MathWritingDataset",
        "AudioCommandDataset",
        "load_curriculum",
        "Problem",
        "synth_ink",
        "synth_image",
        "synth_audio",
        "synth_inkml_dataset",
        "collate_ink",
        "collate_image",
        "collate_audio",
    }
    assert expected <= set(data.__all__)
    assert all(hasattr(data, name) for name in data.__all__)
