"""Whisper fine-tuning as a thin wrapper around Hugging Face ``Seq2SeqTrainer`` (§4.2, ADR-0006).

The decision on record is that we do **not** rewrite Whisper fine-tuning. Its sharp edges — the
forced decoder prompt (``language="tr"``, ``task="transcribe"``), label shifting, ``-100`` label
padding, generate-based evaluation and gradient checkpointing — are handled by
``transformers.Seq2SeqTrainer``, which is also what the literature we compare against uses. A
second hand-written loop would produce a WER nobody could compare with anything.

What this wrapper adds is the *reporting contract* the OCR trainer already satisfies, so the
reviewers' "two training loops" objection is answered by three guarantees and nothing else:

1. :class:`AiclsManifestCallback` writes the **same** ``run_manifest.json`` (git SHA, dirty flag,
   config hash, data fingerprints, environment) before the first optimizer step — ADR-0007;
2. :meth:`WhisperFineTuner.compute_metrics` reports ``wer`` through
   :func:`aicls.eval.metrics.corpus_wer` with ``norm="tr_v1"`` **and** ``wer_raw`` with
   ``norm="none"``, so an ASR number is defined exactly once in this project (ADR-0015);
3. every checkpoint directory Hugging Face writes is completed with our layout —
   ``config.yaml``, ``run_manifest.json`` and ``metrics.json`` beside the HF files
   (:func:`complete_checkpoint_dir`, docs/ml/egitim-altyapisi.md §5).

Optional PEFT LoRA (``model.lora``) and encoder freezing (``model.freeze_encoder``) are supported;
both import their dependency lazily and fail with a typed error when it is absent.

**Weights are never downloaded.** The base checkpoint comes from a local directory under
``AICLS_MODELS_ROOT`` that the ``WEIGHTS_DOWNLOAD`` gate is responsible for populating
(``${AICLS_MODELS_ROOT}/pretrained/<name>``). :func:`resolve_weights_dir` refuses anything else —
including a bare ``org/name`` hub identifier — before ``transformers`` ever sees it, and every
``from_pretrained`` call passes ``local_files_only=True`` (ADR-0012, ``tools/check_gates.py``).

**Running is gated.** :meth:`WhisperFineTuner.fit` calls
:func:`aicls.core.gates.require_gate` for ``TRAINING`` unless the run is on a ``synthetic_*``
dataset. Nothing here can open a gate; it can only refuse.

**Honesty.** A tiny randomly initialised Whisper produces a number, not a result. Any WER measured
before the ``WEIGHTS_DOWNLOAD``, ``CUDA_INSTALL`` and ``TRAINING`` gates open is a WIRING CHECK of
the metric path and must be labelled as such wherever it is quoted.

``torch``, ``transformers``, ``accelerate`` and ``peft`` are imported inside functions, so
importing this module stays cheap and torch-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError, InternalError, NotFoundError
from aicls.core.gates import GateName, require_gate
from aicls.core.logging import get_logger
from aicls.ml.checkpoint import CONFIG_FILENAME, METRICS_FILENAME
from aicls.ml.config import DEFAULT_MAX_LEN, LoraCfg, TrainConfig, resolved_dict
from aicls.ml.manifest import MANIFEST_FILENAME, RunManifest, capture_env, seed_everything
from aicls.ml.tracking import Tracker

_log = get_logger(__name__)

#: Decoder language token forced for every run of this project (proposal A.2.3).
DEFAULT_LANGUAGE = "tr"
#: Decoder task token; never ``"translate"``.
DEFAULT_TASK = "transcribe"
#: Label padding Hugging Face ignores in the cross-entropy loss.
LABEL_PAD_ID = -100
#: Prefix Hugging Face gives its checkpoint directories (``transformers.trainer_utils``).
HF_CHECKPOINT_PREFIX = "checkpoint"
#: Sub-directory of the run directory that holds the Hugging Face checkpoints.
CHECKPOINTS_DIRNAME = "checkpoints"
#: Keys :meth:`WhisperFineTuner.compute_metrics` returns; ``wer`` is the reported figure.
WER_METRIC_KEYS: tuple[str, ...] = ("wer", "wer_raw", "n_samples")
#: Generation cap used when neither ``model.max_len`` nor ``data.max_label_len`` is set.
DEFAULT_GENERATION_MAX_LENGTH = DEFAULT_MAX_LEN
#: Sample rate every audio path of this project works at.
TARGET_SAMPLE_RATE = 16000
#: Analysis hop and window of the smoke feature extractor — the values
#: :func:`aicls.asr.features.log_mel` uses, so the two describe the same frames.
SMOKE_HOP_LENGTH = 160
SMOKE_N_FFT = 400
#: Gate that is responsible for putting a base checkpoint on disk.
WEIGHTS_GATE: GateName = "WEIGHTS_DOWNLOAD"
#: Gate every non-synthetic training run needs.
TRAINING_GATE: GateName = "TRAINING"

#: ``org/name`` — a Hugging Face hub identifier, which is a download, not a path.
_HUB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9_.\-]+$")
#: HF prefixes evaluation metrics with this.
_EVAL_PREFIX = "eval_"
#: Hugging Face's own wall-clock measurement, used for ``RunManifest.steps_per_s``.
_RUNTIME_LOG_KEY = "train_runtime"
#: Log keys that are bookkeeping rather than measurements.
_NON_METRIC_LOG_KEYS = frozenset({"epoch", "step", "total_flos", _RUNTIME_LOG_KEY})


# ==============================================================================================
# errors
# ==============================================================================================
class DependencyUnavailableError(InternalError):
    """An optional ML dependency is not installed in this environment.

    ``transformers``/``accelerate`` are needed for the Hugging Face trainer and ``peft`` for LoRA.
    None of them is ever installed by this code: ``accelerate`` and ``peft`` live in the gated
    ``ml`` extra (``CUDA_INSTALL``, ADR-0012) and ``transformers`` in the ``dev`` extra.
    """

    def __init__(self, package: str, error: str, hint: str) -> None:
        super().__init__(
            messages_tr.MSG_MODEL_NOT_READY,
            details={
                "reason": f"{package}_not_installed",
                "package": package,
                "error": error,
                "hint": hint,
            },
        )


def _require(package: str, hint: str) -> Any:
    """Import ``package`` lazily, as a typed error rather than a bare ``ImportError``."""
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as exc:  # pragma: no cover - depends on what is installed
        raise DependencyUnavailableError(package, str(exc), hint) from exc


def require_transformers() -> Any:
    """The ``transformers`` module, or :class:`DependencyUnavailableError`."""
    return _require("transformers", "pip install -e .[dev] (requirements/dev.txt)")


def require_accelerate() -> Any:
    """The ``accelerate`` module, or :class:`DependencyUnavailableError`.

    ``Seq2SeqTrainingArguments`` cannot even be constructed without it, so this is checked before
    the arguments are built rather than letting a Hugging Face ``ImportError` escape.
    """
    return _require("accelerate", "gated ml extra (requirements/ml.txt, gate CUDA_INSTALL)")


def require_peft() -> Any:
    """The ``peft`` module, or :class:`DependencyUnavailableError`."""
    return _require("peft", "gated ml extra (requirements/ml.txt, gate CUDA_INSTALL)")


# ==============================================================================================
# metric naming across the two training loops
# ==============================================================================================
def hf_metric_name(metric: str) -> str:
    """Translate a ``checkpoint.metric`` value into Hugging Face's spelling.

    The two loops name the same measurement differently and the mismatch is not cosmetic.
    :meth:`aicls.ml.trainer.Trainer.evaluate` publishes ``val_cer`` / ``val_wer`` / ``val_loss``,
    which is what every shipped configuration asks for. Hugging Face prefixes its evaluation
    metrics with ``eval_`` and, in ``Trainer._determine_best_metric`` (transformers 4.57),
    prepends that prefix to ``metric_for_best_model`` when it is missing — so passing
    ``val_wer`` through unchanged makes it look for ``eval_val_wer``, which
    :meth:`WhisperFineTuner.compute_metrics` never produces, and the **first evaluation of the
    run raises KeyError**. Stripping the ``val_`` prefix leaves ``wer`` → ``eval_wer``, which
    exists.

    Args:
        metric: The configured metric name (``val_wer``, ``wer``, ``val_loss``, ``eval_wer``…).

    Returns:
        The name to hand Hugging Face as ``metric_for_best_model``.
    """
    name = metric.strip()
    if name.startswith("eval_"):
        return name
    return name.removeprefix("val_") or metric


# ==============================================================================================
# local weights only
# ==============================================================================================
def resolve_weights_dir(weights_dir: Path | str | None, *, key: str = "model.base_dir") -> Path:
    """Resolve the base checkpoint to an existing **local** directory.

    Args:
        weights_dir: A filesystem path, normally ``${AICLS_MODELS_ROOT}/pretrained/<name>``.
        key: Config key quoted in the error, for a caller that resolved it from somewhere else.

    Returns:
        The resolved directory.

    Raises:
        InputValidationError: If nothing was given, or the value looks like a Hugging Face hub
            identifier (``org/name``) — that is a download and sits behind ``WEIGHTS_DOWNLOAD``.
        NotFoundError: If the directory does not exist. It is never created and never fetched.
        aicls.asr.errors.AsrModelError: If the directory exists but holds no ``config.json`` or
            no weight file (an interrupted ``aicls models import``).
    """
    if weights_dir is None:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "missing_config_key", "key": key, "gate": WEIGHTS_GATE},
        )
    text = str(weights_dir).strip()
    path = Path(text).expanduser()
    if not path.is_dir():
        if _HUB_ID_RE.match(text):
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "hub_id_rejected",
                    "key": key,
                    "value": text,
                    "gate": WEIGHTS_GATE,
                    "hint": "pass a local directory under AICLS_MODELS_ROOT, never a hub id",
                },
            )
        raise NotFoundError(
            messages_tr.msg("FILE_NOT_FOUND", path=str(path)),
            details={
                "reason": "weights_dir_missing",
                "key": key,
                "path": str(path),
                "gate": WEIGHTS_GATE,
                "hint": "training/scripts/download_weights.py populates it once the gate opens",
            },
        )
    from aicls.asr._backend import require_model_dir

    return require_model_dir(path).resolve()


# ==============================================================================================
# model surgery: encoder freezing and LoRA
# ==============================================================================================
def freeze_encoder(model: Any) -> int:
    """Freeze every encoder parameter of a Whisper model.

    Args:
        model: A ``WhisperForConditionalGeneration`` (anything exposing ``get_encoder()`` or
            ``model.encoder``).

    Returns:
        The number of frozen parameters, so the caller can log what actually happened.

    Raises:
        InputValidationError: If no encoder can be found — freezing nothing silently would be
            reported as "encoder frozen" in a table.
    """
    encoder = None
    getter = getattr(model, "get_encoder", None)
    if callable(getter):
        encoder = getter()
    if encoder is None:
        encoder = getattr(getattr(model, "model", None), "encoder", None)
    if encoder is None or not hasattr(encoder, "parameters"):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "encoder_not_found", "type": type(model).__name__},
        )
    frozen = 0
    for parameter in encoder.parameters():
        parameter.requires_grad = False
        frozen += int(parameter.numel())
    _log.info("ml.whisper_ft.encoder_frozen", parameters=frozen)
    return frozen


def apply_lora(model: Any, lora: LoraCfg) -> Any:
    """Wrap ``model`` in a PEFT LoRA adapter when ``lora.enabled``.

    The large ASR candidate (``large-v3-turbo``) is fine-tuned with LoRA only; the small model is
    fine-tuned fully (design document §8).

    Args:
        model: The base model.
        lora: ``model.lora`` from the training config.

    Returns:
        The wrapped model, or ``model`` unchanged when LoRA is disabled.

    Raises:
        DependencyUnavailableError: If ``peft`` is not installed.
    """
    if not lora.enabled:
        return model
    peft = require_peft()
    config = peft.LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.targets),
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    wrapped = peft.get_peft_model(model, config)
    _log.info("ml.whisper_ft.lora_applied", r=lora.r, alpha=lora.alpha, targets=list(lora.targets))
    return wrapped


def configure_generation(
    model: Any, *, language: str = DEFAULT_LANGUAGE, task: str = DEFAULT_TASK
) -> None:
    """Force Turkish transcription on the model's generation config, when it has one to force.

    Modern ``transformers`` resolves the decoder prompt from ``generation_config.language`` and
    ``generation_config.task``; the older ``forced_decoder_ids`` field is deprecated and is
    cleared here so the two cannot disagree. The suppression list is cleared for the same reason
    the reference fine-tuning recipe clears it: it belongs to the pre-trained decoding setup, not
    to a fine-tune.

    ``language``/``task`` are only meaningful for a **multilingual** checkpoint: Whisper's
    ``generate`` translates them through ``generation_config.lang_to_id``. A model without that
    map — the randomly initialised smoke model, and the English-only ``.en`` checkpoints — raises
    ``AttributeError: 'GenerationConfig' object has no attribute 'lang_to_id'`` inside
    ``_retrieve_init_tokens`` the first time ``predict_with_generate`` evaluates (measured with
    transformers 4.57.6, 2026-09-08). Such a model is therefore left monolingual, and the fact is
    logged rather than hidden.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if getattr(generation_config, "lang_to_id", None):
            generation_config.language = language
            generation_config.task = task
        else:
            _log.info(
                "ml.whisper_ft.generation_not_multilingual",
                reason="generation_config carries no lang_to_id map",
                language=language,
            )
        generation_config.forced_decoder_ids = None
    config = getattr(model, "config", None)
    if config is not None:
        config.forced_decoder_ids = None
        config.suppress_tokens = []


# ==============================================================================================
# guarantee 3: our three files beside the Hugging Face checkpoint files
# ==============================================================================================
def complete_checkpoint_dir(
    ckpt_dir: Path | str,
    *,
    cfg: TrainConfig | None = None,
    manifest: RunManifest | Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> list[Path]:
    """Add ``config.yaml``, ``run_manifest.json`` and ``metrics.json`` to a HF checkpoint.

    Hugging Face writes the weights, the optimizer and the trainer state; those three files are
    what makes the directory explainable in a report (docs/ml/egitim-altyapisi.md §5). Existing
    Hugging Face files are never touched.

    Args:
        ckpt_dir: The ``checkpoint-<step>`` directory ``Seq2SeqTrainer`` just wrote.
        cfg: Resolved training config, written as ``config.yaml``.
        manifest: The run manifest (or its document), written as ``run_manifest.json``.
        metrics: The metrics this checkpoint was selected by, written as ``metrics.json``.

    Returns:
        The paths written, in the order they were written (empty when nothing was given).

    Raises:
        NotFoundError: If ``ckpt_dir`` does not exist — this completes a checkpoint, it never
            invents one.
    """
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        raise NotFoundError(
            messages_tr.msg("FILE_NOT_FOUND", path=str(directory)),
            details={"reason": "checkpoint_dir_missing", "path": str(directory)},
        )
    written: list[Path] = []
    if cfg is not None:
        written.append(_write_text(directory / CONFIG_FILENAME, cfg.to_yaml()))
    document = _manifest_document(manifest)
    if document is not None:
        written.append(_write_json(directory / MANIFEST_FILENAME, document))
    if metrics is not None:
        payload = {str(name): float(value) for name, value in metrics.items()}
        written.append(_write_json(directory / METRICS_FILENAME, payload))
    _log.info(
        "ml.whisper_ft.checkpoint_completed",
        path=str(directory),
        files=[p.name for p in written],
    )
    return written


def _manifest_document(
    manifest: RunManifest | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """The manifest as a plain document, whatever shape it arrived in."""
    if manifest is None:
        return None
    if isinstance(manifest, RunManifest):
        return manifest.to_dict()
    return dict(manifest)


def _write_text(path: Path, text: str) -> Path:
    """Write UTF-8 text with LF endings, replacing the file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON document (sorted keys, trailing newline)."""
    text = json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    return _write_text(path, text)


# ==============================================================================================
# guarantee 1: the same run manifest the OCR trainer writes
# ==============================================================================================
class AiclsManifestCallback:
    """Hugging Face trainer callback that writes ``run_manifest.json`` and feeds our tracker.

    It is deliberately a **plain class**, not a ``transformers.TrainerCallback`` subclass, so
    ``aicls.ml.whisper_ft`` imports without ``transformers``. Hugging Face's ``CallbackHandler``
    dispatches with ``getattr(callback, event)(...)``, so duck typing is enough; every event this
    class does not implement is absorbed by :meth:`__getattr__`.

    Args:
        cfg: The resolved training config the run will use.
        run_dir: Run directory; ``run_manifest.json`` and ``env.json`` are written here.
        tracker: Optional :class:`aicls.ml.tracking.Tracker`; started on ``on_train_begin`` and
            fed from ``on_log``/``on_evaluate``.
        allow_dirty: Permit a dirty working tree (the run is then recorded ``citable=false``).
        repo_root: Directory the ``git`` commands run in (tests).
        capture_environment: Also write ``env.json`` (full package list, GPUs). Off in fast tests.
        checkpoints_dir: Where Hugging Face writes ``checkpoint-<step>`` directories; defaults to
            ``args.output_dir`` at ``on_save`` time.

    Attributes:
        manifest: The captured manifest, or ``None`` before ``on_train_begin``.
        manifest_path: Where it was written.
        metrics: The most recent metrics seen, un-prefixed (``eval_wer`` becomes ``wer``).
        train_runtime_s: Hugging Face's measured wall-clock time, the divisor of ``steps_per_s``.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        run_dir: Path | str,
        *,
        tracker: Tracker | None = None,
        allow_dirty: bool = False,
        repo_root: Path | None = None,
        capture_environment: bool = True,
        checkpoints_dir: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.tracker = tracker
        self.allow_dirty = allow_dirty
        self.repo_root = repo_root
        self.capture_environment = capture_environment
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else None
        self.manifest: RunManifest | None = None
        self.manifest_path: Path | None = None
        self.metrics: dict[str, float] = {}
        self.train_runtime_s: float | None = None

    # ------------------------------------------------------------------------------- events
    def on_train_begin(
        self, args: Any = None, state: Any = None, control: Any = None, **kw: Any
    ) -> None:
        """Capture and write the run manifest **before** the first optimizer step (ADR-0007).

        Raises:
            aicls.ml.manifest.DirtyWorkingTreeError: If the working tree is dirty and neither
                ``allow_dirty`` nor ``cfg.train.allow_dirty`` is set.
        """
        del args, state, control, kw
        if self.manifest is not None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = RunManifest.capture(
            self.cfg, allow_dirty=self.allow_dirty, repo_root=self.repo_root
        )
        # One run directory, one run id: the tracker minted a ULID of its own in its constructor,
        # and letting the two records disagree would make metrics.jsonl unjoinable to
        # run_manifest.json. The OCR loop does the same (aicls.ml.trainer._capture_manifest).
        tracker_run_id = getattr(self.tracker, "run_id", None)
        if isinstance(tracker_run_id, str) and tracker_run_id:
            manifest.run_id = tracker_run_id
        self.manifest = manifest
        self.manifest_path = manifest.save(self.run_dir)
        if self.capture_environment:
            capture_env(self.run_dir)
        if self.tracker is not None:
            self.tracker.start_run(manifest.run_name, resolved_dict(self.cfg))
        _log.info(
            "ml.whisper_ft.manifest_written",
            run_id=manifest.run_id,
            path=str(self.manifest_path),
            citable=manifest.citable,
        )

    def on_log(self, args: Any = None, state: Any = None, control: Any = None, **kw: Any) -> None:
        """Forward Hugging Face's training logs to our tracker (split ``train``)."""
        del args, control
        self._record(kw.get("logs"), state, split="train")

    def on_evaluate(
        self, args: Any = None, state: Any = None, control: Any = None, **kw: Any
    ) -> None:
        """Forward the evaluation metrics to our tracker (split ``valid``) and remember them."""
        del args, control
        self._record(kw.get("metrics"), state, split="valid")

    def on_save(self, args: Any = None, state: Any = None, control: Any = None, **kw: Any) -> None:
        """Complete the checkpoint directory Hugging Face has just written (guarantee 3)."""
        del control, kw
        directory = self._checkpoint_dir(args, state)
        if directory is None or not directory.is_dir():
            _log.debug("ml.whisper_ft.checkpoint_not_found", path=str(directory))
            return
        complete_checkpoint_dir(
            directory, cfg=self.cfg, manifest=self.manifest, metrics=self.metrics or None
        )

    def on_train_end(
        self, args: Any = None, state: Any = None, control: Any = None, **kw: Any
    ) -> None:
        """Record the measured throughput on the manifest, once the run has a step count."""
        del args, control, kw
        steps = int(getattr(state, "global_step", 0) or 0)
        runtime = self.train_runtime_s or 0.0
        if self.manifest is not None and steps > 0 and runtime > 0:
            self.manifest.set_steps_per_s(steps / runtime, run_dir=self.run_dir)

    def __getattr__(self, name: str) -> Any:
        """Absorb every other ``on_*`` trainer event as a no-op.

        Raises:
            AttributeError: For any other name, so a typo in our own code still fails.
        """
        if name.startswith("on_"):
            return _noop
        raise AttributeError(name)

    # ------------------------------------------------------------------------------ helpers
    def _record(self, values: Any, state: Any, *, split: str) -> None:
        """Store numeric entries of a HF log/metric mapping and hand them to the tracker."""
        if not isinstance(values, Mapping):
            return
        numeric: dict[str, float] = {}
        for raw_key, value in values.items():
            key = str(raw_key)
            key = key[len(_EVAL_PREFIX) :] if key.startswith(_EVAL_PREFIX) else key
            if isinstance(value, bool):
                continue
            if key == _RUNTIME_LOG_KEY and isinstance(value, int | float):
                # bookkeeping, not a metric, but it is what steps_per_s is measured from
                self.train_runtime_s = float(value)
            if key in _NON_METRIC_LOG_KEYS:
                continue
            if isinstance(value, int | float):
                numeric[key] = float(value)
        if not numeric:
            return
        self.metrics.update(numeric)
        if self.tracker is not None:
            step = int(getattr(state, "global_step", 0) or 0)
            self.tracker.log_metrics(numeric, step, split=split)

    def _checkpoint_dir(self, args: Any, state: Any) -> Path | None:
        """``<output_dir>/checkpoint-<step>`` for the save that just happened."""
        step = int(getattr(state, "global_step", 0) or 0)
        root = self.checkpoints_dir
        if root is None:
            output_dir = getattr(args, "output_dir", None)
            root = Path(str(output_dir)) if output_dir else None
        if root is None:
            return None
        return root / f"{HF_CHECKPOINT_PREFIX}-{step}"


def _noop(*args: Any, **kwargs: Any) -> None:
    """Ignore a Hugging Face trainer event we do not act on."""
    del args, kwargs


# ==============================================================================================
# the weight-free smoke path
# ==============================================================================================
#: Ids the smoke vocabulary reserves, in the order :func:`aicls.ml.registry.build_whisper` hard
#: codes them (``pad_token_id=0``, ``bos_token_id=1``, ``eos_token_id=2``, start token = bos).
SMOKE_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<sot>", "<eot>", "<unk>")
#: Characters the smoke tokenizer covers: the Turkish alphabet in both cases, digits and the
#: punctuation the command inventory uses. Anything else decodes through ``<unk>``.
SMOKE_ALPHABET = (
    "abcçdefgğhıijklmnoöprsştuüvyzqwx"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX"
    "0123456789"
    " .,;:!?'\"()[]{}+-*/=^_<>%\\"
)


@dataclass(frozen=True, slots=True)
class SmokeEncoding:
    """What :meth:`SmokeTokenizer.__call__` returns (the one field callers read)."""

    input_ids: list[int]


class SmokeTokenizer:
    """Character tokenizer for the weight-free ASR smoke run.

    A Whisper tokenizer is a *downloaded artefact* — it needs ``vocab.json`` and ``merges.txt``
    from the checkpoint directory, which sits behind ``WEIGHTS_DOWNLOAD``. The gate-free smoke
    configuration (``training/configs/asr/smoke_synthetic.yaml``, ``model.base_dir: null``) has
    no checkpoint at all, so it needs a tokenizer that can be built out of nothing. This is that
    tokenizer, and it implements exactly the four members the rest of this module uses:
    ``__call__``, :meth:`pad`, :meth:`batch_decode` and ``pad_token_id``.

    **It is not a Whisper tokenizer and produces no comparable number.** A WER measured through
    it is a wiring check of the metric path — a different tokenisation gives a different word
    segmentation — and the run it belongs to is on randomly initialised weights anyway.

    Args:
        alphabet: Characters that get an id of their own.
    """

    def __init__(self, alphabet: str = SMOKE_ALPHABET) -> None:
        tokens = [*SMOKE_SPECIAL_TOKENS, *dict.fromkeys(alphabet)]
        self._ids: dict[str, int] = {token: index for index, token in enumerate(tokens)}
        self._tokens: list[str] = tokens
        self.pad_token_id = self._ids["<pad>"]
        self.bos_token_id = self._ids["<sot>"]
        self.eos_token_id = self._ids["<eot>"]
        self.unk_token_id = self._ids["<unk>"]

    def __len__(self) -> int:
        """Number of ids, specials included."""
        return len(self._tokens)

    @property
    def vocab_size(self) -> int:
        """Size of the output vocabulary the model is built with."""
        return len(self._tokens)

    def __call__(self, text: str, **kwargs: Any) -> SmokeEncoding:
        """Encode ``text`` as ``<sot> … <eot>``; unknown characters become ``<unk>``.

        Args:
            text: The transcript.
            **kwargs: Accepted and ignored, so the call site is the same as a real tokenizer's
                (``add_special_tokens=False`` changes nothing here: the two markers are what the
                trainer's label shifting expects).
        """
        del kwargs
        body = [self._ids.get(char, self.unk_token_id) for char in str(text)]
        return SmokeEncoding(input_ids=[self.bos_token_id, *body, self.eos_token_id])

    def pad(self, features: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Pad ``input_ids`` rows to the longest one, with an attention mask beside them.

        Args:
            features: ``[{"input_ids": [...]}, ...]``.
            **kwargs: ``return_tensors`` is accepted for signature compatibility; this tokenizer
                always returns torch tensors, which is what the collator needs.

        Returns:
            ``{"input_ids": (B, T) int64, "attention_mask": (B, T) int64}``.
        """
        del kwargs
        from aicls.ocr.torch_utils import require_torch

        torch = require_torch()
        rows = [[int(value) for value in row["input_ids"]] for row in features]
        width = max((len(row) for row in rows), default=0)
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def batch_decode(self, sequences: Any, *, skip_special_tokens: bool = True) -> list[str]:
        """Ids back to text, dropping the special tokens by default."""
        texts: list[str] = []
        for row in sequences:
            characters = []
            for value in row:
                index = int(value)
                if not 0 <= index < len(self._tokens):
                    continue
                if skip_special_tokens and index < len(SMOKE_SPECIAL_TOKENS):
                    continue
                characters.append(self._tokens[index])
            texts.append("".join(characters))
        return texts


@dataclass(frozen=True, slots=True)
class SmokeProcessor:
    """``WhisperProcessor``-shaped pair for the weight-free run.

    ``transformers.WhisperProcessor`` refuses a tokenizer that is not a ``WhisperTokenizer``, so
    the smoke path carries this two-field holder instead. The feature extractor **is** a real
    ``transformers.WhisperFeatureExtractor`` (it is buildable without any downloaded file), which
    is what ``Seq2SeqTrainer`` saves into every checkpoint directory as ``processing_class``.
    """

    feature_extractor: Any
    tokenizer: SmokeTokenizer


def build_smoke_processor(cfg: TrainConfig) -> SmokeProcessor:
    """Build the processor of a weight-free run from the configuration alone.

    Args:
        cfg: The resolved training configuration; ``model.num_mel_bins`` has to agree with the
            feature function ``data.feature_extractor`` names.

    Returns:
        A :class:`SmokeProcessor`. Nothing is read from disk and nothing is downloaded.

    Raises:
        DependencyUnavailableError: If ``transformers`` is not installed.
    """
    transformers = require_transformers()
    mel_bins = int(cfg.model.num_mel_bins)
    extractor = transformers.WhisperFeatureExtractor(
        feature_size=mel_bins,
        num_mel_bins=mel_bins,
        sampling_rate=TARGET_SAMPLE_RATE,
        hop_length=SMOKE_HOP_LENGTH,
        n_fft=SMOKE_N_FFT,
    )
    tokenizer = SmokeTokenizer()
    _log.warning(
        "ml.whisper_ft.smoke_processor_built",
        vocab_size=tokenizer.vocab_size,
        num_mel_bins=mel_bins,
        evidence="character tokenizer: a WER measured through it is a wiring check",
    )
    return SmokeProcessor(feature_extractor=extractor, tokenizer=tokenizer)


def silence_hf_printer(trainer: Any) -> bool:
    """Stop Hugging Face printing its log dicts onto stdout.

    With ``disable_tqdm=True`` the trainer installs ``PrinterCallback``, which ``print``s
    ``{'loss': 4.65, ...}`` between our own records. On ``aicls train --json`` that lands in the
    middle of the command's JSON document and makes the output unparsable (reproduced 2026-09-08).
    Nothing is lost by removing it: :class:`AiclsManifestCallback` receives the same ``on_log``
    events and forwards them to the tracker and the structured log.

    Args:
        trainer: A ``transformers.Trainer``.

    Returns:
        Whether a printer callback was found and removed.
    """
    transformers = require_transformers()
    printer = getattr(transformers, "PrinterCallback", None)
    if printer is None:  # pragma: no cover - the class has been public since transformers 4.0
        return False
    trainer.remove_callback(printer)
    return True


def encoder_input_frames(config: Any) -> int | None:
    """Mel frames Whisper's encoder demands, or ``None`` when the model does not say.

    ``WhisperEncoder`` checks the input width against
    ``max_source_positions * conv1.stride * conv2.stride`` (1 and 2), so a real checkpoint wants
    exactly 3000 frames — 30 seconds — and refuses anything else. ``data.feature_extractor:
    whisper`` already produces that; ``log_mel`` produces one frame per hop of the actual
    utterance, which is why :class:`WhisperDataCollator` has to align the batch.
    """
    positions = getattr(config, "max_source_positions", None)
    if isinstance(positions, int) and positions > 0:
        return positions * 2
    return None


# ==============================================================================================
# data plumbing
# ==============================================================================================
@dataclass(slots=True)
class WhisperDataCollator:
    """Pad a batch the way Whisper fine-tuning needs it.

    This is the reference collator, kept here rather than reimplemented per run: input features
    are padded by the feature extractor, labels by the tokenizer, and the label padding is then
    replaced with ``-100`` so it is ignored by the loss. A leading decoder-start token is dropped
    because ``Seq2SeqTrainer`` re-adds it when it shifts the labels.

    Attributes:
        processor: A ``WhisperProcessor`` (feature extractor + tokenizer).
        label_pad_id: Ignore index of the loss (``-100``).
        decoder_start_token_id: Token stripped when every row starts with it.
        input_frames: Mel frames every row is zero-padded (or truncated) to before padding, from
            :func:`encoder_input_frames`. ``None`` passes the features through unchanged, which
            is only correct when they already share a width.
    """

    processor: Any
    label_pad_id: int = LABEL_PAD_ID
    decoder_start_token_id: int | None = None
    input_frames: int | None = None

    def align_features(self, features: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Bring every ``input_features`` row to :attr:`input_frames` mel frames.

        Whisper's encoder accepts one width and one only (:func:`encoder_input_frames`), while
        ``data.feature_extractor: log_mel`` produces a width that follows the utterance length.
        Padding is on the right with zeros, which is what ``WhisperFeatureExtractor`` does to a
        clip shorter than 30 seconds; a longer clip is truncated, as ``data.max_seconds`` says.

        Args:
            features: The rows handed to :meth:`__call__`.

        Returns:
            ``[{"input_features": (n_mels, input_frames) array}, ...]``.
        """
        import numpy as np

        rows: list[dict[str, Any]] = []
        for row in features:
            array = np.asarray(row["input_features"], dtype=np.float32)
            if self.input_frames is not None and array.ndim == 2:
                width = array.shape[-1]
                if width > self.input_frames:
                    array = array[:, : self.input_frames]
                elif width < self.input_frames:
                    pad = ((0, 0), (0, self.input_frames - width))
                    array = np.pad(array, pad, mode="constant", constant_values=0.0)
            rows.append({"input_features": array})
        return rows

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Collate one batch into ``{"input_features": ..., "labels": ...}``."""
        batch = self.processor.feature_extractor.pad(
            self.align_features(features), return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": list(row["labels"])} for row in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), self.label_pad_id
        )
        start = self.decoder_start_token_id
        if start is not None and labels.shape[1] > 1 and bool((labels[:, 0] == start).all()):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return dict(batch)


@dataclass(slots=True)
class WhisperDatasetAdapter:
    """Map-style view that turns :mod:`aicls.data` audio samples into Whisper trainer rows.

    ``aicls.data`` datasets yield ``AudioSample(features, text, speaker, ...)`` (or a
    ``(features, text)`` tuple) where ``features`` is already what ``data.feature_extractor``
    produced. This adapter only tokenises the transcript, so no feature pipeline is duplicated.

    Attributes:
        dataset: Any sized, indexable dataset.
        processor: A ``WhisperProcessor``; its tokenizer produces the label ids.
        max_label_len: Truncate label ids to this length (``data.max_label_len``).
        sampling_rate: When set, the first element is treated as a **waveform** and run through
            the feature extractor instead of being passed through.
    """

    dataset: Any
    processor: Any
    max_label_len: int | None = None
    sampling_rate: int | None = None

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Row ``index`` as ``{"input_features": ..., "labels": [...]}``."""
        item = self.dataset[index]
        if isinstance(item, Mapping) and "input_features" in item and "labels" in item:
            return dict(item)
        features, text = _split_audio_item(item)
        if self.sampling_rate is not None:
            extracted = self.processor.feature_extractor(
                features, sampling_rate=self.sampling_rate, return_tensors="np"
            )
            features = extracted["input_features"][0]
        labels = list(self.processor.tokenizer(text).input_ids)
        if self.max_label_len is not None:
            labels = labels[: self.max_label_len]
        return {"input_features": features, "labels": labels}


def _split_audio_item(item: Any) -> tuple[Any, str]:
    """``(features, text)`` from an ``AudioSample``, a mapping or a tuple.

    Raises:
        InputValidationError: If the item carries no transcript — training Whisper on unlabelled
            audio would silently produce a model that learns nothing.
    """
    if hasattr(item, "features") and hasattr(item, "text"):
        return item.features, str(item.text)
    if isinstance(item, Mapping):
        if "text" in item:
            return item.get("features", item.get("input_features")), str(item["text"])
    elif isinstance(item, Sequence) and not isinstance(item, str | bytes) and len(item) >= 2:
        return item[0], str(item[1])
    raise InputValidationError(
        messages_tr.MSG_VALIDATION_ERROR,
        details={"reason": "audio_item_without_text", "type": type(item).__name__},
    )


# ==============================================================================================
# the report
# ==============================================================================================
@dataclass(frozen=True, slots=True)
class FitReport:
    """What one fine-tuning run produced (the value :meth:`WhisperFineTuner.fit` returns).

    Attributes:
        run_dir: Run directory holding the manifest, the tracker files and the checkpoints.
        run_id: ULID shared by the manifest and the tracker.
        manifest_path: The written ``run_manifest.json`` — the primary citation key (ADR-0007).
        checkpoint_dir: Best (or last) checkpoint directory, when one was written.
        global_step: Optimizer steps completed.
        train_loss: Mean training loss reported by Hugging Face, when it reported one.
        metrics: The last evaluation metrics, un-prefixed (``wer``, ``wer_raw``, ``n_samples``).
        citable: Manifest's ``citable`` flag; ``False`` means a dirty tree or unknown git state,
            and ``poe report-check`` must not take metrics from such a run.
        wall_time_s: Wall-clock duration of :meth:`WhisperFineTuner.fit`.
    """

    run_dir: Path
    run_id: str
    manifest_path: Path
    checkpoint_dir: Path | None
    global_step: int
    train_loss: float | None
    metrics: dict[str, float]
    citable: bool
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible document (paths become strings)."""
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "manifest_path": str(self.manifest_path),
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "global_step": self.global_step,
            "train_loss": self.train_loss,
            "metrics": dict(self.metrics),
            "citable": self.citable,
            "wall_time_s": self.wall_time_s,
        }


# ==============================================================================================
# the wrapper
# ==============================================================================================
class WhisperFineTuner:
    """Fine-tune Whisper with ``Seq2SeqTrainer`` under the AI-CLS reporting contract.

    A **gate-free** run (``data.dataset: synthetic_*``) may leave ``model.base_dir`` unset: the
    model is then randomly initialised through :func:`aicls.ml.registry.build_whisper` and the
    processor through :func:`build_smoke_processor`, so ``aicls train --config
    training/configs/asr/smoke_synthetic.yaml`` runs end to end with no weights on disk and no
    approval. That path produces a loss and a WER; both are **wiring checks**, never results. Any
    other run must name a local checkpoint — random weights on real data would be a fabrication.

    Args:
        cfg: A resolved :class:`~aicls.ml.config.TrainConfig` with ``model.family: whisper``.
        weights_dir: Local base checkpoint directory; ``None`` falls back to ``model.base_dir``,
            and both unset means a randomly initialised smoke model (gate-free configs only).
            Never a hub identifier (ADR-0012).
        run_dir: Run directory: manifest, tracker files and ``checkpoints/`` live here.
        tracker: Optional experiment tracker (ADR-0007); ``JsonlTracker`` is the default backend
            of the CLI, but this class works without one.
        model: Pre-built model, for tests and for a caller that already loaded one.
        processor: Pre-built ``WhisperProcessor``; loaded from ``weights_dir`` when omitted.
        allow_dirty: Permit a dirty working tree (the run becomes non-citable).
        repo_root: Directory the ``git`` commands run in (tests).
        gradient_checkpointing: Trade compute for memory (design document §8). The config schema
            has no key for it yet, so it is a constructor argument.

    Raises:
        InputValidationError: If ``cfg.model.family`` is not ``whisper``, or no weights directory
            can be resolved.
        NotFoundError: If the weights directory does not exist.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        weights_dir: Path | str | None = None,
        run_dir: Path | str,
        tracker: Tracker | None = None,
        model: Any = None,
        processor: Any = None,
        allow_dirty: bool = False,
        repo_root: Path | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        if cfg.model.family != "whisper":
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "wrong_model_family",
                    "family": cfg.model.family,
                    "expected": "whisper",
                },
            )
        self.cfg = cfg
        given = weights_dir if weights_dir is not None else cfg.model.base_dir
        # No checkpoint at all is legitimate exactly once: a gate-free synthetic run, which is
        # what the CI smoke configuration is. On anything else `resolve_weights_dir` refuses and
        # names the WEIGHTS_DOWNLOAD gate, because a WER from random weights on real data would
        # be a fabricated number rather than a missing one.
        self.random_init = given is None and cfg.is_gate_free
        self.weights_dir = None if self.random_init else resolve_weights_dir(given)
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = self.run_dir / CHECKPOINTS_DIRNAME
        self.tracker = tracker
        self.allow_dirty = allow_dirty
        self.repo_root = repo_root
        self.gradient_checkpointing = gradient_checkpointing
        self._model = model
        self._processor = processor
        self._callback: AiclsManifestCallback | None = None

    # ------------------------------------------------------------------------ lazy artefacts
    @property
    def model(self) -> Any:
        """The base model, loaded from the local directory on first use.

        Encoder freezing and LoRA are applied here, and the generation config is forced to
        Turkish transcription.
        """
        if self._model is None:
            self._model = self._build_model()
        return self._model

    @property
    def processor(self) -> Any:
        """The ``WhisperProcessor``, loaded from the local directory on first use.

        Raises:
            NotFoundError: If the directory holds no processor files.
        """
        if self._processor is None:
            self._processor = self._load_processor()
        return self._processor

    def _build_model(self) -> Any:
        """Load the local checkpoint — or build a random one — and apply our switches."""
        if self.random_init:
            model = self._build_random_model()
            local_dir = ""
        else:
            transformers = require_transformers()
            local_dir = str(self.weights_dir)
            model = transformers.WhisperForConditionalGeneration.from_pretrained(
                local_dir, local_files_only=True
            )
        configure_generation(model)
        if self.cfg.model.freeze_encoder:
            freeze_encoder(model)
        model = apply_lora(model, self.cfg.model.lora)
        _log.info(
            "ml.whisper_ft.model_built",
            path=local_dir,
            random_init=self.random_init,
            lora=self.cfg.model.lora.enabled,
            freeze_encoder=self.cfg.model.freeze_encoder,
        )
        return model

    def _build_random_model(self) -> Any:
        """The gate-free smoke model: a tiny ``WhisperConfig`` with random weights.

        Built through :func:`aicls.ml.registry.build_whisper`, which is where every model of this
        project is built, at the vocabulary size of :attr:`processor`'s tokenizer. Nothing is
        read from disk and nothing is downloaded.
        """
        from aicls.ml.registry import build_model

        model = build_model(self.cfg, int(self.processor.tokenizer.vocab_size))
        _log.warning(
            "ml.whisper_ft.random_model_built",
            dataset=self.cfg.data.dataset,
            evidence="randomly initialised weights: every number of this run is a wiring check",
        )
        return model

    def _load_processor(self) -> Any:
        """Load the processor from the same local directory.

        Raises:
            NotFoundError: If ``transformers`` cannot read a processor there (a weights-only
                directory, or an interrupted ``aicls models import``).
        """
        if self.random_init:
            return build_smoke_processor(self.cfg)
        transformers = require_transformers()
        local_dir = str(self.weights_dir)
        try:
            return transformers.WhisperProcessor.from_pretrained(local_dir, local_files_only=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise NotFoundError(
                messages_tr.msg("FILE_NOT_FOUND", path=local_dir),
                details={
                    "reason": "processor_load_failed",
                    "path": local_dir,
                    "gate": WEIGHTS_GATE,
                    "error": str(exc),
                },
            ) from exc

    # -------------------------------------------------------------------------------- gates
    def require_training_gate(self, *, i_have_approval: bool = False) -> bool:
        """Enforce the ``TRAINING`` gate unless the run is on synthetic data.

        Args:
            i_have_approval: Whether the caller passed ``--i-have-approval``. This method can
                only *check*; it never sets the environment variable or the ledger row.

        Returns:
            ``True`` when the gate was required and every condition held, ``False`` when the run
            is gate-free (``data.dataset`` starts with ``synthetic_``, ADR-0012).

        Raises:
            aicls.core.gates.GateError: If the gate is required and any condition is missing
                (exit code 3).
        """
        if self.cfg.is_gate_free:
            _log.info("ml.whisper_ft.gate_free", dataset=self.cfg.data.dataset)
            return False
        require_gate(TRAINING_GATE, cli_flag=i_have_approval)
        return True

    # --------------------------------------------------------------------- training arguments
    def training_args_dict(self, *, with_eval: bool = True) -> dict[str, Any]:
        """The Hugging Face training arguments as a plain dict.

        This is the whole mapping from :class:`~aicls.ml.config.TrainConfig` onto Hugging Face
        names, kept separate from :meth:`training_args` so it can be asserted without
        ``accelerate`` installed (``Seq2SeqTrainingArguments`` cannot be constructed without it).

        ``amp`` degrades cleanly: ``bf16``/``fp16`` are requested only when CUDA is actually
        available, so a CPU host runs fp32 rather than failing (the CUDA wheel is behind the
        ``CUDA_INSTALL`` gate).

        ``checkpoint.metric`` is translated with :func:`hf_metric_name`, because our loop and
        Hugging Face's spell the same measurement differently (see that function).

        Args:
            with_eval: Whether an evaluation dataset will be passed. Without one, evaluation and
                best-checkpoint selection are switched off rather than silently misconfigured.

        Returns:
            Keyword arguments for ``transformers.Seq2SeqTrainingArguments``.
        """
        cfg = self.cfg
        cuda = _cuda_available()
        bf16 = cfg.train.amp == "bf16" and cuda
        fp16 = cfg.train.amp == "fp16" and cuda
        if cfg.train.amp != "off" and not cuda:
            _log.warning(
                "ml.whisper_ft.amp_downgraded",
                requested=cfg.train.amp,
                effective="fp32",
                reason="torch.cuda.is_available() is False (gate CUDA_INSTALL)",
            )
        keep_best = with_eval and cfg.checkpoint.keep_best > 0
        return {
            "output_dir": str(self.checkpoints_dir),
            "overwrite_output_dir": False,
            "seed": cfg.run.seed,
            "data_seed": cfg.run.seed,
            "per_device_train_batch_size": cfg.data.batch_size,
            "per_device_eval_batch_size": cfg.data.batch_size,
            "gradient_accumulation_steps": cfg.train.grad_accum,
            "learning_rate": cfg.optim.lr,
            "weight_decay": cfg.optim.weight_decay,
            "adam_beta1": cfg.optim.betas[0],
            "adam_beta2": cfg.optim.betas[1],
            "max_grad_norm": cfg.train.grad_clip if cfg.train.grad_clip is not None else 0.0,
            "lr_scheduler_type": cfg.optim.scheduler,
            "warmup_steps": cfg.optim.warmup_steps,
            "num_train_epochs": float(cfg.train.epochs or 1),
            "max_steps": cfg.train.max_steps if cfg.train.max_steps is not None else -1,
            "eval_strategy": "steps" if with_eval else "no",
            "eval_steps": cfg.train.eval_every,
            "save_strategy": "steps",
            "save_steps": cfg.train.ckpt_every,
            "save_total_limit": cfg.checkpoint.keep_last or None,
            "save_safetensors": True,
            "logging_strategy": "steps",
            "logging_steps": cfg.train.log_every,
            "load_best_model_at_end": keep_best,
            "metric_for_best_model": hf_metric_name(cfg.checkpoint.metric),
            "greater_is_better": cfg.checkpoint.mode == "max",
            "predict_with_generate": True,
            "generation_max_length": self.generation_max_length(),
            "generation_num_beams": cfg.eval.beam,
            "bf16": bf16,
            "fp16": fp16,
            "use_cpu": not cuda,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_num_workers": cfg.data.num_workers,
            "dataloader_pin_memory": cuda,
            "remove_unused_columns": False,
            "label_names": ["labels"],
            "report_to": [],
            "disable_tqdm": True,
        }

    def training_args(self, *, with_eval: bool = True) -> Any:
        """``transformers.Seq2SeqTrainingArguments`` built from :meth:`training_args_dict`.

        Raises:
            DependencyUnavailableError: If ``transformers`` or ``accelerate`` is missing.
        """
        transformers = require_transformers()
        require_accelerate()
        return transformers.Seq2SeqTrainingArguments(**self.training_args_dict(with_eval=with_eval))

    def generation_max_length(self) -> int:
        """Decoding cap, clamped to what the loaded checkpoint's decoder can actually emit."""
        length = self.cfg.model.max_len or self.cfg.data.max_label_len
        length = length or DEFAULT_GENERATION_MAX_LENGTH
        positions = getattr(getattr(self._model, "config", None), "max_target_positions", None)
        if isinstance(positions, int) and positions > 0:
            length = min(length, positions)
        return int(length)

    # ------------------------------------------------------------ guarantee 2: one WER, twice
    def compute_metrics(self, eval_prediction: Any) -> dict[str, float]:
        """WER of one evaluation batch, normalised and raw.

        Both numbers come from :func:`aicls.eval.metrics.corpus_wer`, the single definition used
        by every other report in this project (ADR-0015): ``wer`` under ``tr_v1`` (the ``< 10 %``
        target) and ``wer_raw`` under ``none`` (the column every report also prints).

        Args:
            eval_prediction: Hugging Face ``EvalPrediction``, or a ``(predictions, label_ids)``
                pair. ``predictions`` may be a tuple whose first element holds the token ids.

        Returns:
            ``{"wer": ..., "wer_raw": ..., "n_samples": ...}``.

        Note:
            Measured on a randomly initialised model this is a WIRING CHECK of the metric path,
            not an ASR result. Only a run whose manifest says ``citable`` may be quoted.
        """
        from aicls.eval.metrics import corpus_wer

        predictions, labels = _unpack_eval_prediction(eval_prediction)
        hypotheses = self.decode(predictions)
        references = self.decode(labels)
        pairs = list(zip(references, hypotheses, strict=True))
        metrics = {
            "wer": corpus_wer(pairs, norm="tr_v1"),
            "wer_raw": corpus_wer(pairs, norm="none"),
            "n_samples": float(len(pairs)),
        }
        _log.debug("ml.whisper_ft.metrics", **metrics)
        return metrics

    def decode(self, sequences: Any) -> list[str]:
        """Token ids to text, with the ``-100`` label padding put back to the pad token.

        Args:
            sequences: Rows of token ids (lists, numpy arrays or tensors).

        Returns:
            One stripped string per row.
        """
        tokenizer = self.processor.tokenizer
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        # -100 is the loss ignore index the collator wrote; it is not a token id.
        cleaned = [[pad_id if int(value) < 0 else int(value) for value in row] for row in sequences]
        if not cleaned:
            return []
        decoded = tokenizer.batch_decode(cleaned, skip_special_tokens=True)
        return [str(text).strip() for text in decoded]

    # --------------------------------------------------------------------------- the trainer
    def data_collator(self) -> WhisperDataCollator:
        """The padding collator, wired to this run's processor, start token and feature width.

        The model is built here if it has not been already: only it knows how many mel frames its
        encoder accepts (:func:`encoder_input_frames`), and a batch of the wrong width is a hard
        error in ``WhisperEncoder.forward``, not a warning.
        """
        config = getattr(self.model, "config", None)
        return WhisperDataCollator(
            self.processor,
            decoder_start_token_id=getattr(config, "decoder_start_token_id", None),
            input_frames=encoder_input_frames(config),
        )

    def manifest_callback(self) -> AiclsManifestCallback:
        """The callback that carries guarantees 1 and 3 (created once per fine-tuner)."""
        if self._callback is None:
            self._callback = AiclsManifestCallback(
                self.cfg,
                self.run_dir,
                tracker=self.tracker,
                allow_dirty=self.allow_dirty,
                repo_root=self.repo_root,
                checkpoints_dir=self.checkpoints_dir,
            )
        return self._callback

    def build_trainer(self, *, train_dataset: Any, eval_dataset: Any = None) -> Any:
        """Assemble ``transformers.Seq2SeqTrainer`` with our arguments, collator and callback.

        Args:
            train_dataset: Anything ``Seq2SeqTrainer`` accepts; wrap an :mod:`aicls.data` dataset
                in :class:`WhisperDatasetAdapter` first.
            eval_dataset: Optional validation dataset; without one, evaluation is switched off.

        Returns:
            The trainer, ready for ``.train()``.

        Raises:
            DependencyUnavailableError: If ``transformers`` or ``accelerate`` is missing.
        """
        transformers = require_transformers()
        require_accelerate()
        args = self.training_args(with_eval=eval_dataset is not None)
        trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator(),
            compute_metrics=self.compute_metrics if eval_dataset is not None else None,
            processing_class=self.processor.feature_extractor,
            callbacks=[self.manifest_callback()],
        )
        silence_hf_printer(trainer)
        return trainer

    def fit(
        self,
        *,
        train_dataset: Any,
        eval_dataset: Any = None,
        i_have_approval: bool = False,
    ) -> FitReport:
        """Run the fine-tuning and return what it produced.

        Order matters: the ``TRAINING`` gate is checked **first**, so a refused run never loads a
        model, never seeds and never writes into the run directory.

        Args:
            train_dataset: Training data.
            eval_dataset: Optional validation data (enables generate-based evaluation).
            i_have_approval: Whether ``--i-have-approval`` was given. Only the CLI passes this.

        Returns:
            The :class:`FitReport`.

        Raises:
            aicls.core.gates.GateError: If the run needs the ``TRAINING`` gate and it is closed.
            DependencyUnavailableError: If ``transformers`` or ``accelerate`` is missing.
            aicls.ml.manifest.DirtyWorkingTreeError: If the tree is dirty and dirty runs are not
                allowed.
        """
        self.require_training_gate(i_have_approval=i_have_approval)
        started = perf_counter()
        seed_everything(self.cfg.run.seed, deterministic=self.cfg.run.deterministic)
        trainer = self.build_trainer(train_dataset=train_dataset, eval_dataset=eval_dataset)
        callback = self.manifest_callback()
        status = "failed"
        try:
            result = trainer.train()
            status = "finished"
        finally:
            if self.tracker is not None:
                self.tracker.end_run(status)  # type: ignore[arg-type]  # RunStatus literal
        best = getattr(trainer.state, "best_model_checkpoint", None)
        checkpoint_dir = Path(str(best)) if best else self.last_checkpoint_dir()
        if checkpoint_dir is not None and checkpoint_dir.is_dir():
            complete_checkpoint_dir(
                checkpoint_dir,
                cfg=self.cfg,
                manifest=callback.manifest,
                metrics=callback.metrics or None,
            )
        manifest = callback.manifest
        report = FitReport(
            run_dir=self.run_dir,
            run_id=manifest.run_id if manifest else "",
            manifest_path=callback.manifest_path or self.run_dir / MANIFEST_FILENAME,
            checkpoint_dir=checkpoint_dir,
            global_step=int(getattr(trainer.state, "global_step", 0) or 0),
            train_loss=float(result.training_loss) if result.training_loss is not None else None,
            metrics=dict(callback.metrics),
            citable=bool(manifest.citable) if manifest else False,
            wall_time_s=perf_counter() - started,
        )
        _log.info(
            "ml.whisper_ft.fit_finished",
            run_id=report.run_id,
            steps=report.global_step,
            citable=report.citable,
            evidence="wiring check unless the base weights and the data are the real ones",
        )
        return report

    def last_checkpoint_dir(self) -> Path | None:
        """Highest-numbered ``checkpoint-<step>`` directory under ``run_dir/checkpoints``.

        Returns:
            The directory, or ``None`` when the run wrote none (a run shorter than
            ``train.ckpt_every``, or one that was refused before it started).
        """
        if not self.checkpoints_dir.is_dir():
            return None
        candidates = [
            path
            for path in self.checkpoints_dir.iterdir()
            if path.is_dir() and path.name.startswith(f"{HF_CHECKPOINT_PREFIX}-")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: _trailing_int(p.name))


# ==============================================================================================
# small helpers
# ==============================================================================================
def _trailing_int(name: str) -> int:
    """Trailing integer of ``checkpoint-1200``; ``-1`` when there is none."""
    tail = name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _cuda_available() -> bool:
    """Whether a CUDA device is usable right now (never assumed; the wheel here is ``+cpu``)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dev dependency and is installed
        return False
    return bool(torch.cuda.is_available())


def _unpack_eval_prediction(eval_prediction: Any) -> tuple[Any, Any]:
    """``(predictions, label_ids)`` from an ``EvalPrediction`` or a plain pair.

    Raises:
        InputValidationError: If neither shape is recognised.
    """
    predictions = getattr(eval_prediction, "predictions", None)
    labels = getattr(eval_prediction, "label_ids", None)
    if predictions is None or labels is None:
        if not isinstance(eval_prediction, Sequence) or len(eval_prediction) < 2:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "eval_prediction_unrecognised",
                    "type": type(eval_prediction).__name__,
                },
            )
        predictions, labels = eval_prediction[0], eval_prediction[1]
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return predictions, labels


__all__ = [
    "CHECKPOINTS_DIRNAME",
    "DEFAULT_GENERATION_MAX_LENGTH",
    "DEFAULT_LANGUAGE",
    "DEFAULT_TASK",
    "HF_CHECKPOINT_PREFIX",
    "LABEL_PAD_ID",
    "SMOKE_ALPHABET",
    "SMOKE_HOP_LENGTH",
    "SMOKE_N_FFT",
    "SMOKE_SPECIAL_TOKENS",
    "TARGET_SAMPLE_RATE",
    "TRAINING_GATE",
    "WEIGHTS_GATE",
    "WER_METRIC_KEYS",
    "AiclsManifestCallback",
    "DependencyUnavailableError",
    "FitReport",
    "SmokeEncoding",
    "SmokeProcessor",
    "SmokeTokenizer",
    "WhisperDataCollator",
    "WhisperDatasetAdapter",
    "WhisperFineTuner",
    "apply_lora",
    "build_smoke_processor",
    "complete_checkpoint_dir",
    "configure_generation",
    "encoder_input_frames",
    "freeze_encoder",
    "hf_metric_name",
    "require_accelerate",
    "require_peft",
    "require_transformers",
    "resolve_weights_dir",
    "silence_hf_printer",
]
