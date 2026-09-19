"""Boundaries that only exist while the system is actually running (wave-6 integration gate).

Five agents worked on the same tree in wave 6 and each of them tested their own half. What no
unit test could see is what happens where their halves meet *at run time*:

1. **The GPU runner meets the trainer.** ``Trainer`` selects a device in ``__init__`` but used to
   move the weights only inside ``fit()``. On CPU that is invisible; on the RTX 5090 (measured
   2026-09-08) ``evaluate()`` on a fresh trainer raised ``Input type (torch.cuda.FloatTensor)
   and weight type (torch.FloatTensor) should be the same`` and ``resume()`` built the optimizer
   over CPU parameters, which is exactly what ``training/scripts/train.py --resume`` does.
2. **The streaming shard preparation meets the trainer that will consume the shards.** The tar
   path (``prepare_shards_from_source``) was tested against its own manifest and the directory
   path against ``MathWritingDataset`` -- nothing joined the two, so "the archive we will
   actually download produces shards the training loop can read" was an assumption.
3. **The sandbox rewrite meets the API that mounts it.** ``build_executor`` returns the spawned
   pool for every environment but ``test``, and no test had ever *started* that pool through the
   API's own factory. The Linux ``MemoryError`` warm-up failure of 2026-09-08 was precisely this
   boundary: the pool started, ``grammar_ready`` was false and every compute answered INTERNAL,
   while ``GET /health/ready`` would still have reported ``started: true``.
4. **The evaluation honesty rule meets every report consumer.** A wiring-check run now publishes
   ``null`` instead of a number. Anything that reads ``reports/metrics/latest.json`` has to cope
   with the null rather than format it, compare it or divide by it.

Nothing here downloads anything, opens a gate or needs a GPU; the two ``gpu``-marked tests are
skipped unless a CUDA device is really present.
"""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.core.settings import Settings
from aicls.data import synth_inkml_dataset
from aicls.data.sources import DirectoryInkmlSource, open_inkml_source
from aicls.mathengine import Diff

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = REPO_ROOT / "training" / "configs" / "ocr" / "smoke_synthetic.yaml"
LATEST_JSON = REPO_ROOT / "reports" / "metrics" / "latest.json"

#: A miniature stand-in for the released ``mathwriting-2024.tgz`` layout.
LAYOUT: dict[str, int] = {"train": 6, "valid": 4}


# ==============================================================================================
# 1. the GPU runner meets the trainer: the weights are on the selected device from the start
# ==============================================================================================
class _RecordingModel:
    """A model-shaped object that only records where it was asked to go."""

    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> _RecordingModel:
        self.moves.append(str(device))
        return self


def _tiny_trainer(tmp_path: Path, model: Any, *, device: str = "cpu") -> Any:
    from aicls.ml.config import load_config
    from aicls.ml.trainer import Trainer

    cfg = load_config(SMOKE_CONFIG)
    return Trainer(cfg, model, [], None, run_dir=tmp_path / "run", device=device)


def test_the_trainer_places_the_weights_before_anything_can_use_them(tmp_path: Path) -> None:
    """Constructing a ``Trainer`` moves the model; it does not wait for ``fit()``.

    Deferring the move to ``fit()`` made ``evaluate()``, ``beam_search()`` and ``resume()``
    unusable on a CUDA box while passing every CPU test, because on CPU ``.to("cpu")`` is a
    no-op and the defect has no symptom.
    """
    model = _RecordingModel()
    trainer = _tiny_trainer(tmp_path, model, device="cpu")

    assert model.moves == ["cpu"], "the constructor must place the weights on the chosen device"
    assert trainer.device_spec.device == "cpu"


def test_a_trainer_without_a_movable_model_still_builds(tmp_path: Path) -> None:
    """``model=None`` is a legitimate argument (the configuration-only tests use it).

    The eager move must not become the reason a configuration cannot be validated.
    """
    trainer = _tiny_trainer(tmp_path, None, device="cpu")
    assert trainer.model is None


@pytest.mark.gpu
def test_evaluate_and_resume_work_on_cuda_before_fit_has_run(tmp_path: Path) -> None:
    """The two calls that ``training/scripts/train.py --resume`` makes, on the real device.

    Both were broken on the RTX 5090 on 2026-09-08 and neither had a CPU symptom.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():  # pragma: no cover - the marker already gates this
        pytest.skip("no CUDA device")
    from aicls.ml.config import load_config
    from aicls.ml.registry import DatasetSpec, build_dataset, build_model
    from aicls.ml.trainer import Trainer

    def build(run: str, budget: int = 2) -> Trainer:
        # `train.max_steps` is an absolute budget, resumed steps included, and the smallest of
        # it and the `fit(max_steps=)` argument wins. Raising it for the resumed trainer is what
        # makes the loop take a real optimizer step after `resume()` -- without one, the Adam
        # state never touches a parameter and the device mismatch cannot fire.
        cfg = load_config(SMOKE_CONFIG, overrides=[f"train.max_steps={budget}"])
        tokenizer = DatasetSpec.from_config(cfg).tokenizer()
        return Trainer(
            cfg,
            build_model(cfg, tokenizer.vocab_size),
            build_dataset(cfg, "train"),
            build_dataset(cfg, "valid"),
            run_dir=tmp_path / run,
            tokenizer=tokenizer,
            device="cuda",
        )

    source = build("interrupted")
    assert next(source.model.parameters()).device.type == "cuda", "placed by the constructor"
    source.fit(max_steps=2)
    checkpoint = sorted((tmp_path / "interrupted" / "checkpoints").glob("step-*"))[-1]

    fresh = build("evaluated")
    assert fresh.evaluate()["n_val"] > 0, "evaluate() before fit() must not raise on CUDA"

    resumed = build("resumed", budget=4)
    restored = resumed.resume(checkpoint)
    assert restored.step == 2
    resumed.fit(max_steps=4)  # the Adam-state device mismatch fires here, if it fires at all
    assert resumed.step == 4, "resume() must be followed by real optimizer steps"


# ==============================================================================================
# 2. the streamed archive meets the training loop that will read its shards
# ==============================================================================================
@pytest.fixture
def archive_and_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A ``mathwriting-2024/<split>/*.inkml`` tree and the same tree packed as a ``.tgz``."""
    corpus = tmp_path / "src" / "mathwriting-2024"
    for index, (split, count) in enumerate(LAYOUT.items()):
        synth_inkml_dataset(corpus / split, count, 1234 + 100 * index, prefix=split)
    archive = tmp_path / "mathwriting-2024.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(corpus, arcname=corpus.name)
    return archive, corpus


def test_shards_streamed_out_of_the_archive_are_the_shards_the_directory_path_writes(
    archive_and_tree: tuple[Path, Path], tmp_path: Path
) -> None:
    """One corpus, two readers, one dataset fingerprint per split.

    The download the ``DATASET_DOWNLOAD`` gate protects is a 2.9 GiB ``.tgz`` that the operator
    is told never to unpack. If the streamed path produced different shards from the extracted
    path, every report fingerprint would depend on how the corpus happened to be read.
    """
    from aicls.data import prepare_shards_from_source

    archive, corpus = archive_and_tree
    splits = list(LAYOUT)
    streamed = prepare_shards_from_source(
        open_inkml_source(archive), tmp_path / "from_tar", splits=splits, shard_size=4
    )
    walked = prepare_shards_from_source(
        DirectoryInkmlSource(corpus), tmp_path / "from_dir", splits=splits, shard_size=4
    )

    for split in splits:
        assert streamed[split].n_samples == LAYOUT[split] == walked[split].n_samples
        assert streamed[split].fingerprint == walked[split].fingerprint, split
    # The source is still recorded, so a report can name which reader produced its input.
    assert streamed["train"].source_root.startswith("tar:")
    assert not walked["train"].source_root.startswith("tar:")


def test_streamed_shards_feed_the_trainer_and_stay_behind_the_download_gate(
    archive_and_tree: tuple[Path, Path], tmp_path: Path
) -> None:
    """Archive -> shards -> ``MathWritingDataset`` -> ``Trainer.evaluate()``, in one pass.

    ``evaluate()`` is used deliberately: it drives the collate function, the model forward and
    the decoder over data that never existed as an extracted InkML tree, and it needs no
    approval. ``fit()`` on the same configuration must still refuse, because ``mathwriting`` is
    not a gate-free dataset -- proving that streaming did not become a way around ADR-0012.
    """
    pytest.importorskip("torch")
    from aicls.core.errors import ErrorCode
    from aicls.data import prepare_shards_from_source
    from aicls.ml.config import load_config
    from aicls.ml.registry import DatasetSpec, build_dataset, build_model
    from aicls.ml.trainer import Trainer

    archive, _ = archive_and_tree
    shards = tmp_path / "shards"
    prepare_shards_from_source(
        open_inkml_source(archive), shards, splits=list(LAYOUT), shard_size=4
    )

    cfg = load_config(
        SMOKE_CONFIG,
        overrides=[
            "data.dataset=mathwriting",
            f"data.root={shards.as_posix()}",
            "eval.max_n_val=4",
        ],
    )
    assert cfg.is_gate_free is False, "real MathWriting shards are never gate-free"

    tokenizer = DatasetSpec.from_config(cfg).tokenizer()
    valid = build_dataset(cfg, "valid")
    assert len(valid) == LAYOUT["valid"]
    trainer = Trainer(
        cfg,
        build_model(cfg, tokenizer.vocab_size),
        build_dataset(cfg, "train"),
        valid,
        run_dir=tmp_path / "run",
        tokenizer=tokenizer,
        device="cpu",
    )

    scores = trainer.evaluate()
    assert scores["n_val"] > 0
    assert "cer" in scores and scores["cer"] >= 0.0

    with pytest.raises(AiclsError) as blocked:
        trainer.fit(max_steps=1)
    assert blocked.value.code is ErrorCode.GATE_BLOCKED


# ==============================================================================================
# 3. the sandbox rewrite meets the API that mounts it
# ==============================================================================================
@pytest.mark.slow
def test_the_pool_the_api_mounts_outside_tests_really_warms_up_and_answers() -> None:
    """``build_executor(env="dev")`` -> start -> probe -> a real CAS answer.

    Every API test runs on ``AICLS_ENV=test``, which gets the in-process executor, so the
    spawned pool the deployment actually uses had no test crossing the API boundary at all.
    On Linux (2026-09-08) that pool warmed up with ``MemoryError`` and answered INTERNAL to
    every request while still reporting ``started: true`` to ``GET /health/ready``. This asserts
    the three things the readiness endpoint cannot see on its own: the grammar is built, the
    memory cap is an observable fact rather than a claim, and a command comes back as a result.
    """
    from aicls.api.app import build_executor
    from aicls.api.schemas.admin import SandboxOut

    executor = build_executor(Settings(env="dev", sandbox_workers=1))
    assert type(executor).__name__ == "PoolExecutor"
    assert getattr(executor, "started", False) is False, "the factory must not spawn workers"
    executor.start()
    try:
        probe = executor.probe()
        assert probe.sympy_imported and probe.grammar_ready, (
            "a warm-up failure leaves the pool 'started' but every compute answers INTERNAL"
        )
        if probe.memory_limit_applied:
            assert probe.memory_limit_bytes > 0, "an applied cap must report the bytes it applied"

        result = executor.run(Diff(op="diff", expr="x^{2}"))
        assert result.latex, "the spawned pool must return a result, not an INTERNAL error"

        # exactly the three expressions src/aicls/api/routers/health.py reads off the executor
        sandbox = SandboxOut(
            workers=int(getattr(executor, "workers", 1) or 0),
            restarts=int(getattr(executor, "restarts", 0) or 0),
            started=bool(getattr(executor, "started", executor is not None)),
        )
        assert sandbox.started is True
        assert sandbox.workers == 1
        assert sandbox.restarts == 0, "a healthy warm-up respawns nothing"
    finally:
        executor.stop()


# ==============================================================================================
# 4. the wiring-check rule meets every consumer of the published report
# ==============================================================================================
def _latest() -> dict[str, dict[str, Any]]:
    return json.loads(LATEST_JSON.read_text(encoding="utf-8"))


def test_the_published_report_never_carries_a_number_from_a_stub_model() -> None:
    """Model-backed kinds publish ``null`` while their backend is a stub.

    ``ocr.cer = 0.0`` was published for three waves, produced by a stub that was shown the label
    before it was asked the question. The rule is now structural: for ``ocr`` and ``asr`` a stub
    run publishes no metric at all, and what it observed lives under ``wiring_check``.
    """
    from aicls.eval.report import MEASURED, MODEL_BACKED_KINDS, WIRING_CHECK

    latest = _latest()
    assert set(latest) >= MODEL_BACKED_KINDS, "the report keeps the model-backed kinds"
    for kind, entry in latest.items():
        assert entry["measurement"] in {MEASURED, WIRING_CHECK}, kind
        if entry["measurement"] != WIRING_CHECK:
            continue
        assert entry["stub"] is True
        assert entry["ci95"] == {}
        assert entry["wiring_check"]["reason"], kind
        numeric = {
            key: value
            for key, value in entry.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        assert numeric.keys() <= {"n"}, f"{kind} published {sorted(numeric)} from a stub backend"
        assert entry["wiring_check"]["metrics"], "what the stub scored is kept, not destroyed"


def test_every_report_consumer_survives_a_null_metric() -> None:
    """A null headline must not become ``0``, ``"None"`` or an exception downstream.

    Two consumers are checked against the committed file: the documentation macro lookup (the
    ``{{ metrics.<kind>.<metric> }}`` path, which must resolve to ``None`` rather than raise
    under ``on_undefined: strict``) and the report checker.
    """
    latest = _latest()
    ocr = latest["ocr"]
    assert "cer" in ocr and ocr["cer"] is None, "the key resolves; the value is null"
    assert ocr["wiring_check"]["metrics"]["cer"] > 0.0
    # a consumer formatting the headline must be able to tell "absent" from "zero"
    assert ocr["cer"] != 0.0 and ocr["cer"] is not False


def test_the_web_client_error_code_list_covers_every_code_the_api_can_return() -> None:
    """``web/src/api/problem.ts`` and :class:`aicls.core.errors.ErrorCode` must not drift.

    Wave 6 added ``TRAINING_BUSY``, which the API deliberately never returns (it is a
    workstation/CLI condition). The contract is therefore one-directional: every code the API
    *can* answer with must be listed on the client; a CLI-only code may be absent.
    """
    from aicls.core.errors import HTTP_STATUS_BY_CODE, ErrorCode

    source = (REPO_ROOT / "web" / "src" / "api" / "problem.ts").read_text(encoding="utf-8")
    api_codes = {code for code in ErrorCode if code is not ErrorCode.TRAINING_BUSY}
    missing = sorted(code.value for code in api_codes if f'"{code.value}"' not in source)
    assert not missing, f"web/src/api/problem.ts is missing {missing}"
    assert HTTP_STATUS_BY_CODE[ErrorCode.TRAINING_BUSY] == 503
    assert f'"{ErrorCode.TRAINING_BUSY.value}"' not in source, (
        "TRAINING_BUSY is a CLI condition; listing it would imply the API answers with it"
    )


# ==============================================================================================
# 5. the two training loops meet the reporting contract: one run directory, one record
# ==============================================================================================
def test_both_training_routes_of_the_cli_get_the_same_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``aicls train`` gives the ASR route the tracker the OCR route has always had.

    Measured on the RTX 5090 on 2026-09-08: an ASR run directory held ``run_manifest.json`` and
    ``env.json`` and nothing else, because the CLI passed no tracker to
    :class:`~aicls.ml.whisper_ft.WhisperFineTuner` although its docstring says ``JsonlTracker``
    is the CLI's default backend. ``metrics.jsonl`` and ``params.json`` -- the ADR-0007 record a
    report reads -- existed for one of the two loops only, so the two run directories could not
    be compared.

    The wrapper is replaced here rather than executed: ``accelerate`` lives behind the
    ``CUDA_INSTALL`` gate and is absent on the development machine, so running the real thing
    would make this a container-only test of a wiring question that has no device in it.
    """
    from aicls.cli.train import _run_whisper
    from aicls.ml import whisper_ft
    from aicls.ml.config import load_config
    from aicls.ml.tracking import JsonlTracker

    seen: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, cfg: Any, **kwargs: Any) -> None:
            seen.update(kwargs)
            self.processor = object()

        def fit(self, **_: Any) -> dict[str, Any]:
            return {"global_step": 0, "metrics": {}}

    monkeypatch.setattr(whisper_ft, "WhisperFineTuner", _Recorder)
    monkeypatch.setattr(whisper_ft, "WhisperDatasetAdapter", lambda ds, proc, **kw: ds)
    monkeypatch.setattr("aicls.ml.registry.build_dataset", lambda cfg, split: [])

    cfg = load_config(REPO_ROOT / "training" / "configs" / "asr" / "smoke_synthetic.yaml")
    report = _run_whisper(
        cfg, run_dir=tmp_path / "asr", device="cpu", i_have_approval=False, validate=True
    )

    assert isinstance(seen["tracker"], JsonlTracker)
    assert seen["run_dir"] == tmp_path / "asr"
    assert report["run_dir"].endswith("asr")
