"""The report document, its writer, the generated table and ``report-check`` (ADR-0015 §4).

One :class:`EvalReport` is one measurement: what was measured, on which corpus, from which
commit, with which normaliser versions, and how uncertain the number is. Reports are written to
``reports/metrics/<name>_<gitsha7>.json``; the ones a reviewer accepts are merged into
``reports/metrics/latest.json``, which is the single file the MkDocs macros read — so a number
in the documentation is never typed by a human.

Three invariants are enforced here rather than left to discipline:

#. **A report can only claim to be citable if it is.** ``citable`` is recomputed from
   ``git_sha`` and ``dirty``; a caller that passes ``citable=True`` for a dirty run gets
   ``False`` and can see why in the file.
#. **A promoted report is re-checked.** :func:`check_report` refuses a commit that is not an
   ancestor of HEAD, a dirty run, a stale normaliser version, an unknown dataset fingerprint or
   a missing definitions reference — the ``poe report-check`` gate of ADR-0015.
#. **A wiring check publishes no number.** :func:`promote` writes ``null`` for every metric of a
   wiring check and says why in ``measurement`` and ``wiring_check``; the observed values stay
   in the per-run report file and under ``wiring_check.metrics``, where nobody can mistake them
   for evidence. This exists because ``latest.json`` published ``ocr.cer = 0.0`` and
   ``ocr.exprate = 1.0`` for a stub that was being handed the answer key — the single most
   quotable-by-accident pair of numbers in the repository, against a contractual target of
   ``CER < 10 %``.

**What "wiring check" means.** It used to mean one thing — "the backend was a stub" on the
evaluation side, "the dataset was synthetic" on the training side. That definition let the first
real GPU run (20 optimizer steps over 100 samples, a randomly initialised model, 16 validation
samples) write ``wiring_check: false``. Over-claiming is the one direction this flag must never
fail in, so it is now a **structured verdict**: :func:`assess_wiring_check` collects every reason
a run's numbers may not be quoted and returns them as a
:class:`WiringCheckVerdict`; the boolean is derived from the list, and the list travels with the
number so a reader learns *why*. The reasons are deliberately independent — synthetic data,
randomly initialised weights, a step budget too small to have trained, a learning-rate schedule
that never left its warmup, a validation subset too small to estimate the metric, a non-citable
tree — and each carries its own threshold and rationale next to the constant that sets it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.logging import get_logger
from aicls.eval.messages import tr
from aicls.eval.normalizers import NORMALIZER_VERSIONS
from aicls.eval.provenance import NO_GIT, SHORT_SHA_LEN, utc_timestamp

_logger = get_logger(__name__)

#: Evaluation kinds (mirrors :class:`aicls.db.enums.EvalKind`).
EvalKindName = Literal["parser", "cas", "nlu", "ocr", "asr", "book", "latency"]

#: Section of the documentation every report points at.
DEFINITIONS_REF = "docs/ml/degerlendirme.md#2-metrikler"

#: Default output directory of :func:`write_report`.
DEFAULT_OUT_DIR = Path("reports/metrics")

#: Pointer file the MkDocs macros read.
LATEST_FILENAME = "latest.json"

#: Evaluation kinds whose subject is a **learned model**. For every other kind the component
#: under test (the parser, the CAS, the rule-based NLU layer, the İP-8 resolver, the local
#: pipeline) exists and really ran, so a number from it is a measurement even when it carries
#: caveats. For these two, a stub backend means the thing the metric names does not exist yet.
MODEL_BACKED_KINDS: frozenset[str] = frozenset({"ocr", "asr"})

#: ``measurement`` value of a promoted entry that measured a real component.
MEASURED = "measured"

#: ``measurement`` value of a promoted entry produced by a stub model backend.
WIRING_CHECK = "wiring_check"

#: Why a wiring-check entry publishes no number. English, like every other machine-read note in
#: a report file; the Turkish wording readers see is in :mod:`aicls.eval.messages`.
WIRING_CHECK_REASON = (
    "WIRING CHECK, NOT A MEASUREMENT: this run used a stub backend, so the metric describes the "
    "evaluation plumbing and not a model. The headline value is null on purpose — a stub may "
    "never publish a number that could be quoted against a contractual target. What the stub "
    "actually scored is kept under 'wiring_check' and in the per-run report file."
)

#: Prefix every wiring-check explanation starts with, whatever the reasons are.
WIRING_CHECK_PREFIX = "WIRING CHECK, NOT A MEASUREMENT:"

#: Closing sentence of a generated wiring-check explanation.
WIRING_CHECK_SUFFIX = (
    "The headline value is null on purpose — such a run may never publish a number that could be "
    "quoted against a contractual target. What it actually scored is kept under 'wiring_check' "
    "and in the per-run report file."
)

#: Key under which a producing run records its verdict in ``EvalReport.breakdown`` and under
#: which :func:`promote` republishes it in ``latest.json``.
WIRING_CHECK_KEY = "wiring_check"

# ---------------------------------------------------------------------------- wiring-check codes
#: The corpus was generated in-process instead of being the corpus the metric names.
WIRING_SYNTHETIC_DATA = "synthetic_data"
#: The component under test was a stub, so the metric describes the plumbing around it.
WIRING_STUB_BACKEND = "stub_backend"
#: The weights were randomly initialised where the design expects a pretrained checkpoint.
WIRING_RANDOM_INIT = "random_init"
#: The optimizer never left its warmup ramp, so the weights are still essentially at init.
WIRING_WARMUP_NOT_COMPLETED = "warmup_not_completed"
#: Fewer optimizer steps than the floor below which "training" is not a defensible word.
WIRING_STEP_BUDGET_TOO_SMALL = "step_budget_too_small"
#: The validation subset is too small for the metric's interval to mean anything.
WIRING_VAL_SUBSET_TOO_SMALL = "val_subset_too_small"
#: The tree was dirty or git did not answer, so the number cannot be tied to a commit.
WIRING_NOT_CITABLE = "not_citable"
#: A producing run said "wiring check" without naming a reason (an older record, or a bare bool).
WIRING_UNSPECIFIED = "unspecified"

#: Every reason code, in the order :func:`assess_wiring_check` emits them.
WIRING_REASON_CODES: tuple[str, ...] = (
    WIRING_SYNTHETIC_DATA,
    WIRING_STUB_BACKEND,
    WIRING_RANDOM_INIT,
    WIRING_WARMUP_NOT_COMPLETED,
    WIRING_STEP_BUDGET_TOO_SMALL,
    WIRING_VAL_SUBSET_TOO_SMALL,
    WIRING_NOT_CITABLE,
    WIRING_UNSPECIFIED,
)

#: Optimizer steps below which a run has not trained, whatever its configuration says.
#:
#: This is a **conservative floor, not a sufficiency threshold**: clearing it is not evidence
#: that a model converged, it only stops a smoke run from calling itself a measurement. 100 is
#: chosen because every encoder–decoder OCR schedule in this project is specified in tens of
#: thousands of steps (``training/configs/ocr/offline_resnet34_tfm.yaml``: 30 epochs with a
#: 2000-step warmup), so two orders of magnitude below the warmup alone cannot be a result. The
#: run that prompted this rule took 20 steps.
MIN_REPORTABLE_STEPS = 100

#: Validation samples below which a metric has no interval worth publishing.
#:
#: Derived rather than picked: the OCR target is ``CER < 0.10``, so the measurement must at least
#: separate "target met" from "target missed by a factor of two". With a worst-case per-sample
#: standard deviation of 0.5, the 95 % half-width ``1.96 · 0.5 / sqrt(n)`` falls below 0.10 only
#: for ``n > 96``. The run that prompted this rule validated on 16 samples, where that half-width
#: is ±0.245 — wider than the target itself.
MIN_VAL_SAMPLES = 100

_SHA_RE = re.compile(r"^(?:[0-9a-f]{7,40})?$")


@dataclass(frozen=True, slots=True)
class WiringCheckReason:
    """One reason a run's numbers may not be quoted.

    Attributes:
        code: A stable machine token from :data:`WIRING_REASON_CODES`.
        detail: English sentence naming the observed values behind the code.
    """

    code: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        """JSON-ready ``{"code": ..., "detail": ...}``."""
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class WiringCheckVerdict:
    """Why (or whether) a run is a wiring check rather than a measurement.

    The boolean everything downstream used to carry is :attr:`is_wiring_check`, derived from the
    reasons: an empty verdict is a real measurement, a non-empty one never is.

    Attributes:
        reasons: The reasons, in :data:`WIRING_REASON_CODES` order, without duplicates.
    """

    reasons: tuple[WiringCheckReason, ...] = ()

    @property
    def is_wiring_check(self) -> bool:
        """True when at least one reason applies."""
        return bool(self.reasons)

    @property
    def codes(self) -> tuple[str, ...]:
        """The reason codes alone, in order."""
        return tuple(reason.code for reason in self.reasons)

    def merge(self, other: WiringCheckVerdict) -> WiringCheckVerdict:
        """This verdict plus ``other``'s reasons, first occurrence of each code winning.

        Two normalisations keep the merged verdict readable: the result is ordered by
        :data:`WIRING_REASON_CODES` (unknown codes last, in the order they arrived), and the
        :data:`WIRING_UNSPECIFIED` placeholder is dropped as soon as a named reason is present —
        it exists only to say "a wiring check was recorded but not explained".
        """
        seen: dict[str, WiringCheckReason] = {}
        for reason in (*self.reasons, *other.reasons):
            seen.setdefault(reason.code, reason)
        if len(seen) > 1:
            seen.pop(WIRING_UNSPECIFIED, None)
        order = {code: index for index, code in enumerate(WIRING_REASON_CODES)}
        merged = sorted(
            enumerate(seen.values()),
            key=lambda pair: (order.get(pair[1].code, len(order)), pair[0]),
        )
        return WiringCheckVerdict(tuple(reason for _, reason in merged))

    def summary(self) -> str:
        """One English line: the codes with their details, or ``"measurement"``."""
        if not self.reasons:
            return "measurement"
        return "; ".join(
            f"{reason.code} ({reason.detail})" if reason.detail else reason.code
            for reason in self.reasons
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready verdict: the boolean, the codes and the reasons with their details."""
        return {
            "is_wiring_check": self.is_wiring_check,
            "codes": list(self.codes),
            "reasons": [reason.to_dict() for reason in self.reasons],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> WiringCheckVerdict:
        """Read a verdict back out of free-form JSON, never raising.

        Accepts everything the record has ever held: this class's own ``to_dict``, the
        ``latest.json`` ``wiring_check`` block, a bare list of reason dicts, a list of codes, or
        the plain ``True``/``False`` boolean older runs wrote. Anything unrecognised is treated
        as "no verdict recorded", because a malformed provenance note must not crash the report
        gate — it simply fails to excuse the run.

        Args:
            payload: Whatever was stored under :data:`WIRING_CHECK_KEY`.

        Returns:
            The verdict; an empty one when nothing usable was found.
        """
        if payload is None or payload is False:
            return cls()
        if payload is True:
            return cls((WiringCheckReason(WIRING_UNSPECIFIED, _UNSPECIFIED_DETAIL),))
        if isinstance(payload, Mapping):
            if payload.get("is_wiring_check") is False and not payload.get("reasons"):
                return cls()
            items: Any = payload.get("reasons") or payload.get("codes") or ()
            verdict = cls._from_items(items)
            if verdict.reasons or not payload:
                return verdict
            return cls((WiringCheckReason(WIRING_UNSPECIFIED, _UNSPECIFIED_DETAIL),))
        if isinstance(payload, str):
            return cls((WiringCheckReason(payload, ""),))
        if isinstance(payload, Sequence):
            return cls._from_items(payload)
        return cls()

    @classmethod
    def _from_items(cls, items: Any) -> WiringCheckVerdict:
        """Build a verdict from a sequence of reason dicts or bare code strings."""
        if isinstance(items, str) or not isinstance(items, Sequence):
            return cls()
        reasons: list[WiringCheckReason] = []
        seen: set[str] = set()
        for item in items:
            if isinstance(item, Mapping):
                code = str(item.get("code", "")).strip()
                detail = str(item.get("detail", ""))
            elif isinstance(item, str):
                code, detail = item.strip(), ""
            else:
                continue
            if not code or code in seen:
                continue
            seen.add(code)
            reasons.append(WiringCheckReason(code, detail))
        return cls(tuple(reasons))


_UNSPECIFIED_DETAIL = (
    "the producing run recorded a wiring check without naming a reason (an older record, or a "
    "bare boolean flag)"
)


def assess_wiring_check(
    *,
    synthetic_data: bool = False,
    stub_backend: bool = False,
    random_init: bool = False,
    citable: bool | None = None,
    steps: int | None = None,
    warmup_steps: int | None = None,
    n_val: int | None = None,
    min_steps: int = MIN_REPORTABLE_STEPS,
    min_val_samples: int = MIN_VAL_SAMPLES,
) -> WiringCheckVerdict:
    """Decide whether a run measured a model or only proved that a path works.

    Every argument that can be unknown defaults to "do not judge this" (``None`` / ``False``), so
    a caller that cannot observe a fact never causes a false positive. This matters as much as
    catching the wiring checks: the flag exists to stop a throwaway number being quoted, and it
    stops being trusted the moment it also fires on the final run.

    Args:
        synthetic_data: The corpus was generated in-process (``TrainConfig.is_gate_free``).
        stub_backend: The component under test was a stub rather than a model.
        random_init: The weights were randomly initialised where the design expects a pretrained
            checkpoint. **Not** set for the project's OCR models, which are trained from scratch
            by design (ADR-0012): there, random weights are the legitimate starting point and the
            step budget is what says whether they ever became a model. It is set for the Whisper
            path, where a randomly initialised Whisper is not Whisper.
        citable: Whether the run can be tied to a clean commit (``RunManifest.citable``);
            ``None`` when the caller has no manifest to ask.
        steps: Optimizer steps actually performed; ``None`` for a run with no training loop
            (a pure evaluation), which is then judged on its other facts alone.
        warmup_steps: ``optim.warmup_steps`` of the schedule; ``None`` or ``0`` disables the
            warmup rule.
        n_val: Number of validation samples the reported metric was computed over; ``None`` when
            no validation ran (nothing to judge — the missing metric speaks for itself).
        min_steps: Override for :data:`MIN_REPORTABLE_STEPS`.
        min_val_samples: Override for :data:`MIN_VAL_SAMPLES`.

    Returns:
        The :class:`WiringCheckVerdict`; empty (``is_wiring_check`` false) for a run that may be
        reported.

    Raises:
        InputValidationError: If a threshold is negative.
    """
    if min_steps < 0 or min_val_samples < 0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "negative_wiring_check_threshold",
                "min_steps": min_steps,
                "min_val_samples": min_val_samples,
            },
        )
    reasons: list[WiringCheckReason] = []
    if synthetic_data:
        reasons.append(
            WiringCheckReason(
                WIRING_SYNTHETIC_DATA,
                "the corpus was generated in-process, not the corpus the metric names",
            )
        )
    if stub_backend:
        reasons.append(
            WiringCheckReason(
                WIRING_STUB_BACKEND,
                "the component under test was a stub, so the metric describes the plumbing "
                "around it and not a model",
            )
        )
    if random_init:
        reasons.append(
            WiringCheckReason(
                WIRING_RANDOM_INIT,
                "the weights were randomly initialised where the design expects a pretrained "
                "checkpoint",
            )
        )
    if steps is not None and warmup_steps and steps <= int(warmup_steps):
        reasons.append(
            WiringCheckReason(
                WIRING_WARMUP_NOT_COMPLETED,
                f"{steps} optimizer steps <= optim.warmup_steps {int(warmup_steps)}: the "
                "learning-rate schedule never left its warmup ramp, so the weights are still "
                "essentially at initialisation",
            )
        )
    if steps is not None and steps < min_steps:
        reasons.append(
            WiringCheckReason(
                WIRING_STEP_BUDGET_TOO_SMALL,
                f"{steps} optimizer steps < {min_steps} (MIN_REPORTABLE_STEPS)",
            )
        )
    if n_val is not None and n_val < min_val_samples:
        reasons.append(
            WiringCheckReason(
                WIRING_VAL_SUBSET_TOO_SMALL,
                f"the metric was computed over {n_val} validation samples < {min_val_samples} "
                "(MIN_VAL_SAMPLES); its 95 % interval is wider than the contractual threshold",
            )
        )
    if citable is False:
        reasons.append(
            WiringCheckReason(
                WIRING_NOT_CITABLE,
                "the working tree was dirty or git did not answer, so the number cannot be tied "
                "to a commit",
            )
        )
    verdict = WiringCheckVerdict(tuple(reasons))
    _logger.debug("eval.report.wiring_check_assessed", verdict=verdict.summary())
    return verdict


def wiring_check_reason_text(verdict: WiringCheckVerdict) -> str:
    """The English explanation published next to a nulled metric.

    A verdict whose only reason is :data:`WIRING_STUB_BACKEND` reproduces
    :data:`WIRING_CHECK_REASON` verbatim, so the text the repository already documents does not
    move. Any other verdict gets the same prefix followed by its own reasons.

    Args:
        verdict: The verdict to explain.

    Returns:
        The explanation; ``""`` for a verdict that is not a wiring check.
    """
    if not verdict.is_wiring_check:
        return ""
    if verdict.codes == (WIRING_STUB_BACKEND,):
        return WIRING_CHECK_REASON
    return f"{WIRING_CHECK_PREFIX} {verdict.summary()}. {WIRING_CHECK_SUFFIX}"


@dataclass(frozen=True, slots=True)
class Target:
    """A contractual target from the proposal.

    Attributes:
        comparison: ``"<"``, ``">"``, ``">="`` or ``"=="``.
        threshold: The number the metric is compared against.
        note: Short English note naming the work package.
    """

    comparison: str
    threshold: float
    note: str = ""

    def holds(self, value: float) -> bool:
        """Whether ``value`` meets this target."""
        if self.comparison == "<":
            return value < self.threshold
        if self.comparison == ">":
            return value > self.threshold
        if self.comparison == ">=":
            return value >= self.threshold
        return value == self.threshold

    def render(self) -> str:
        """The target as it appears in the generated table (``"< 0.10"``, ``"< 3000"``)."""
        number = f"{self.threshold:.2f}" if abs(self.threshold) <= 1.0 else f"{self.threshold:g}"
        return f"{self.comparison} {number}"


#: The contractual numbers of the proposal, frozen next to the metric definitions.
#: ``sus`` is not a runner: the questionnaire is collected by the İP-9 study and scored with
#: :func:`aicls.eval.metrics.sus_score`, but its target belongs in the same table.
TARGETS: dict[tuple[str, str], Target] = {
    ("ocr", "cer"): Target("<", 0.10, "IP-3 handwriting CER"),
    ("asr", "wer_tr_v1"): Target("<", 0.10, "IP-3 speaker-disjoint WER"),
    ("asr", "command_intent_accuracy"): Target(">", 0.95, "IP-2 command delivery"),
    ("parser", "success_rate"): Target(">", 0.99, "IP-4 parse success"),
    ("cas", "correct_rate"): Target(">", 0.99, "IP-4 CAS accuracy"),
    ("nlu", "intent_accuracy"): Target(">=", 0.95, "IP-4 text-to-command"),
    ("book", "fetch_top1"): Target(">", 0.95, "IP-8 fetch accuracy"),
    ("book", "tagging_coverage"): Target("==", 1.0, "IP-8 tagging coverage"),
    ("latency", "mean_ms"): Target("<", 3000.0, "IP-2/IP-4/IP-9 mean response time"),
    ("sus", "sus_score"): Target(">", 75.0, "IP-2/IP-9 usability"),
}


class EvalReport(BaseModel):
    """One measurement, with everything needed to defend it.

    Attributes:
        name: Report name, usually ``"<kind>_<split>"`` (``"ocr_test"``).
        kind: Which runner produced it.
        git_sha: Commit the code was at; ``""`` outside a checkout.
        dirty: Whether the working tree carried uncommitted changes (or was unknown).
        citable: Whether the number may be quoted; recomputed, never taken on trust.
        dataset_fingerprint: ``"sha256:<hex>"`` of the corpus
            (:func:`aicls.core.hashing.fingerprint_jsonl`).
        dataset_version: Human-readable corpus version (``"curriculum-2026.09-v1"``).
        model_version: Model registry version, when a model was involved.
        model_sha256: Checkpoint digest, when a model was involved.
        config_hash: Hash of the configuration the run used.
        normalizer_versions: The frozen normaliser versions the numbers were computed with.
        n: Corpus size the metrics were computed over.
        metrics: The numbers themselves.
        ci95: 95 % bootstrap intervals, per metric name.
        breakdown: Per-topic/per-source/per-length detail and provenance notes.
        failures_path: Repository-relative path of the worst-cases file, when written.
        timestamp: UTC ISO-8601 instant of the run.
        definitions_ref: Documentation section that defines the metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=120)
    kind: EvalKindName
    git_sha: str = ""
    dirty: bool = True
    citable: bool = False
    dataset_fingerprint: str = ""
    dataset_version: str = ""
    model_version: str | None = None
    model_sha256: str | None = None
    config_hash: str | None = None
    normalizer_versions: dict[str, str] = Field(default_factory=lambda: dict(NORMALIZER_VERSIONS))
    n: int = Field(default=0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    ci95: dict[str, tuple[float, float]] = Field(default_factory=dict)
    breakdown: dict[str, Any] = Field(default_factory=dict)
    failures_path: str | None = None
    timestamp: str = Field(default_factory=utc_timestamp)
    definitions_ref: str = DEFINITIONS_REF

    @field_validator("git_sha")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        """A commit hash is lowercase hex of 7..40 characters, or empty."""
        if not _SHA_RE.match(value):
            raise ValueError(f"git_sha must be 7..40 hex characters or empty, got {value!r}")
        return value

    @model_validator(mode="after")
    def _derive_citable(self) -> EvalReport:  # noqa: N804 - mode="after" runs on the instance
        """Downgrade ``citable`` when the provenance does not support it.

        A run without a commit, or from a dirty tree, is not citable — whatever the caller
        passed. Downgrading (rather than raising) keeps a long evaluation run from dying at the
        last step, and the file still says plainly that the number may not be quoted.
        """
        allowed = bool(self.git_sha) and not self.dirty
        if self.citable and not allowed:
            _logger.warning(
                "eval.report.citable_downgraded",
                name=self.name,
                dirty=self.dirty,
                has_sha=bool(self.git_sha),
            )
            object.__setattr__(self, "citable", False)
        return self

    @property
    def short_sha(self) -> str:
        """First seven characters of :attr:`git_sha`, or ``"nogit"``."""
        return self.git_sha[:SHORT_SHA_LEN] if self.git_sha else NO_GIT

    @property
    def is_stub(self) -> bool:
        """Whether the run's backend was a stub (or its corpus synthetic)."""
        return bool(self.breakdown.get("stub", False))

    @property
    def wiring_check_verdict(self) -> WiringCheckVerdict:
        """Every reason this report's numbers may not be quoted, as a structured verdict.

        Two sources are merged: what the report can see about itself, and what the producing run
        chose to record.

        The report judges three of its own fields. A stub backend counts only for a
        :data:`MODEL_BACKED_KINDS` kind, because ``latency`` is also marked ``stub`` and there it
        means "server side, not the browser figure the contract names" — the pipeline it timed is
        real code that really ran. :attr:`citable` and :attr:`n` are judged for every kind: a
        number nobody can tie to a commit and an interval over a handful of samples are not
        quotable whatever produced them.

        Judging those two here rather than trusting each runner to is what closes a leak found on
        2026-09-09. ``assess_wiring_check`` has always weighed them, but its only caller passed
        ``stub_backend`` alone, so a run that was not a stub could not be a wiring check for any
        other reason and :func:`promote` published its numbers. The speech study found it from the
        ASR side — the first real Whisper run would have filed a dev-set WER next to the
        contractual "WER < 10 %" — but nothing about it was specific to ASR.

        On top of that comes whatever the producing run recorded under
        ``breakdown[:data:`WIRING_CHECK_KEY`]``: the training-side verdict (too few steps, a
        warmup never completed, randomly initialised weights where a checkpoint was expected),
        carried across the boundary as data, since only the run that trained the model knows it.
        """
        verdict = assess_wiring_check(
            stub_backend=self.is_stub and self.kind in MODEL_BACKED_KINDS,
            citable=self.citable,
            n_val=self.n,
        )
        return verdict.merge(WiringCheckVerdict.from_payload(self.breakdown.get(WIRING_CHECK_KEY)))

    @property
    def is_wiring_check(self) -> bool:
        """Whether this run proved a path works rather than measuring a model."""
        return self.wiring_check_verdict.is_wiring_check

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping (the shape ``reports/metrics/*.json`` and ``eval_runs`` store)."""
        return self.model_dump(mode="json")


def write_report(r: EvalReport, out_dir: Path | str = DEFAULT_OUT_DIR) -> Path:
    """Write ``r`` as ``<name>_<gitsha7>.json`` and return the path.

    Args:
        r: The report.
        out_dir: Target directory; created when missing.

    Returns:
        The written path (UTF-8, LF, two-space indent, trailing newline).
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{r.name}_{r.short_sha}.json"
    text = json.dumps(r.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    _logger.info("eval.report.written", path=str(path), kind=r.kind, n=r.n)
    return path


def write_failures(rows: Sequence[Mapping[str, Any]], path: Path | str) -> Path:
    """Write the worst cases of a run as JSONL (the ``failures_path`` of a report)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def promote(reports: Iterable[EvalReport], out_dir: Path | str = DEFAULT_OUT_DIR) -> Path:
    """Merge ``reports`` into ``latest.json`` — the file the docs macros read.

    A measured run contributes its numbers. A **wiring check** (:attr:`EvalReport
    .is_wiring_check`) contributes ``null`` for every metric plus the reason: promotion is
    publication, and a stub has nothing to publish. The metric keys stay present so
    ``{{ metrics.<kind>.<metric> }}`` still resolves under ``on_undefined: strict`` and a report
    page renders a visible ``None`` rather than breaking the build; what the stub scored is
    preserved verbatim under ``wiring_check``.

    Args:
        reports: One report per evaluation kind.
        out_dir: Directory holding ``latest.json``.

    Returns:
        The path of the written pointer file.

    Raises:
        InputValidationError: If two reports share a ``kind``; keeping the last one silently
            would publish a number nobody chose.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for report in reports:
        if report.kind in payload:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "duplicate_eval_kind", "kind": report.kind},
            )
        verdict = report.wiring_check_verdict
        wiring = verdict.is_wiring_check
        intervals = {key: list(value) for key, value in report.ci95.items()}
        entry: dict[str, Any] = {
            key: (None if wiring else value) for key, value in report.metrics.items()
        }
        entry.update(
            {
                "n": report.n,
                "ci95": {} if wiring else intervals,
                "git_sha": report.git_sha,
                "dataset_fingerprint": report.dataset_fingerprint,
                "dataset_version": report.dataset_version,
                "model_version": report.model_version,
                "normalizer_versions": dict(report.normalizer_versions),
                "dirty": report.dirty,
                "citable": report.citable,
                "generated_at": report.timestamp,
                "source_file": f"{report.name}_{report.short_sha}.json",
                "definitions_ref": report.definitions_ref,
                # the caveat must travel with the value: the MkDocs macros read this file and
                # nothing else, so a stub number has to be recognisable as one from here alone.
                "stub": report.is_stub,
                "measurement": WIRING_CHECK if wiring else MEASURED,
                "notes": report.breakdown.get("notes", ""),
            }
        )
        if wiring:
            entry[WIRING_CHECK_KEY] = {
                "reason": wiring_check_reason_text(verdict),
                "codes": list(verdict.codes),
                "reasons": [reason.to_dict() for reason in verdict.reasons],
                "model_version": report.model_version,
                "metrics": dict(report.metrics),
                "ci95": intervals,
            }
        payload[report.kind] = entry
    path = directory / LATEST_FILENAME
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _logger.info("eval.report.promoted", path=str(path), kinds=sorted(payload))
    return path


def load_latest(path: Path | str = DEFAULT_OUT_DIR / LATEST_FILENAME) -> dict[str, Any]:
    """Read ``latest.json``; an absent or empty file is ``{}`` (CI must stay green before use).

    Raises:
        InputValidationError: If the file exists but is not a JSON object.
    """
    target = Path(path)
    if not target.is_file():
        return {}
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "latest_not_json", "path": str(target), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "latest_not_an_object", "path": str(target)},
        )
    return payload


# --------------------------------------------------------------------------------------------
# report-check
# --------------------------------------------------------------------------------------------
def check_report(
    report: EvalReport,
    *,
    head_sha: str,
    ancestors: Collection[str],
    fingerprints: Collection[str] | None = None,
    reject_wiring_check: bool = False,
) -> list[str]:
    """Verify one promoted report; returns the problems (empty means "may be cited").

    Args:
        report: The report to check.
        head_sha: Current ``HEAD``.
        ancestors: Commits reachable from ``HEAD`` (CI passes ``git rev-list HEAD``; a local run
            may pass :func:`aicls.eval.provenance.known_shas`).
        fingerprints: Fingerprints of the committed corpora; ``None`` skips that check.
        reject_wiring_check: Also refuse a run whose :attr:`EvalReport.wiring_check_verdict` is
            not empty. Off by default because this function answers "is this measurement's
            provenance sound?", and a wiring check has sound provenance — it is simply not a
            result. :func:`check_promoted` turns it on, because *promotion is publication*: the
            entries in ``latest.json`` are the numbers the documentation quotes.

    Returns:
        Human-readable English problem strings, one per violated rule.
    """
    problems: list[str] = []
    if not report.git_sha:
        problems.append("git_sha is empty: the run cannot be tied to a commit")
    elif report.git_sha != head_sha and report.git_sha not in set(ancestors):
        problems.append(
            f"git_sha {report.short_sha} is not an ancestor of HEAD {head_sha[:SHORT_SHA_LEN]}"
        )
    if report.dirty:
        problems.append("dirty working tree: the measured code is not what is committed")
    if not report.citable:
        problems.append("citable is false: this report may not be quoted in a progress report")
    if report.normalizer_versions != NORMALIZER_VERSIONS:
        problems.append(
            f"normalizer versions {report.normalizer_versions} != current {NORMALIZER_VERSIONS}"
        )
    if not report.definitions_ref:
        problems.append("definitions_ref is empty: the metric definition is not referenced")
    if fingerprints is not None and report.dataset_fingerprint not in set(fingerprints):
        problems.append(
            f"dataset fingerprint {report.dataset_fingerprint or '(empty)'} matches no known corpus"
        )
    if reject_wiring_check:
        verdict = report.wiring_check_verdict
        if verdict.is_wiring_check:
            problems.append(
                f"wiring check [{', '.join(verdict.codes)}]: this run measured the plumbing, not "
                f"the model, and may not be promoted into a report — {verdict.summary()}"
            )
    return problems


def check_promoted(
    latest: Mapping[str, Any],
    *,
    head_sha: str,
    ancestors: Collection[str],
    fingerprints: Collection[str] | None = None,
) -> list[str]:
    """Run :func:`check_report` over every entry of ``latest.json``.

    Promotion is publication, so this is stricter than :func:`check_report` alone: an entry a
    producing run marked as a wiring check — through ``measurement`` or through the reasons it
    recorded under :data:`WIRING_CHECK_KEY` — is refused here even though the metric values it
    publishes are already ``null``. Otherwise a wiring check could sit in the file the
    documentation reads with nothing objecting to it.

    Args:
        latest: The parsed pointer file.
        head_sha: Current ``HEAD``.
        ancestors: Commits reachable from ``HEAD``.
        fingerprints: Fingerprints of the committed corpora.

    Returns:
        Problems, each prefixed with the evaluation kind.
    """
    problems: list[str] = []
    for kind, entry in sorted(latest.items()):
        if not isinstance(entry, Mapping):
            problems.append(f"{kind}: entry is not an object")
            continue
        report = EvalReport(
            name=str(entry.get("source_file", kind)).removesuffix(".json") or kind,
            kind=kind,  # type: ignore[arg-type]  # validated by the model, reported as a problem
            git_sha=str(entry.get("git_sha", "")),
            dirty=bool(entry.get("dirty", True)),
            citable=bool(entry.get("citable", False)),
            dataset_fingerprint=str(entry.get("dataset_fingerprint", "")),
            normalizer_versions=dict(entry.get("normalizer_versions", {})),
            n=int(entry.get("n", 0)),
            definitions_ref=str(entry.get("definitions_ref", "")),
            breakdown=_promoted_breakdown(entry),
        )
        problems.extend(
            f"{kind}: {problem}"
            for problem in check_report(
                report,
                head_sha=head_sha,
                ancestors=ancestors,
                fingerprints=fingerprints,
                reject_wiring_check=True,
            )
        )
    return problems


def _promoted_breakdown(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the ``breakdown`` an entry of ``latest.json`` implies.

    ``promote`` flattens a report into an entry, so the verdict has to be read back out of the
    published fields: the ``wiring_check`` block when it is there, and the bare
    ``measurement == "wiring_check"`` label for an entry written before the block carried its
    reasons.
    """
    breakdown: dict[str, Any] = {"stub": bool(entry.get("stub", False))}
    payload = entry.get(WIRING_CHECK_KEY)
    if payload is not None:
        breakdown[WIRING_CHECK_KEY] = payload
    elif str(entry.get("measurement", "")) == WIRING_CHECK:
        breakdown[WIRING_CHECK_KEY] = True
    return breakdown


# --------------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------------
def _format_value(value: float) -> str:
    """Four decimals for a rate, no decimals for a millisecond figure."""
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.4f}"


def _status(target: Target | None, value: float, *, stub: bool) -> str:
    """The status cell: does the contractual target hold?

    A stub run gets no verdict at all. The OCR wiring check scores ``cer = 0`` because the stub
    echoes the label it was handed and the ASR one scores ``wer > 1`` because it draws a random
    sentence; printing "tuttu"/"tutmadı" next to the İP-3 targets for those rows would tell a
    reader that a contractual metric was met (or missed) by a model that does not exist yet.
    The target itself stays in its column, so the table still shows what will have to be met.
    """
    if target is None:
        return tr("REPORT_STATUS_NONE")
    if stub:
        return tr("REPORT_STATUS_STUB")
    return tr("REPORT_STATUS_PASS") if target.holds(value) else tr("REPORT_STATUS_FAIL")


def render_markdown(reports: Sequence[EvalReport]) -> str:
    """Render the results table quoted in the progress report (Turkish, generated).

    Args:
        reports: The reports to tabulate, in the order they should appear.

    Returns:
        A Markdown table with one row per metric — value, 95 % interval, n, contractual target
        and whether it holds — followed by a provenance line per report. This is the source of
        ``docs/raporlama``: no number in the documentation is typed by hand (ADR-0015 §4).
    """
    header = (
        f"| {tr('REPORT_COL_KIND')} | {tr('REPORT_COL_METRIC')} | {tr('REPORT_COL_VALUE')} "
        f"| {tr('REPORT_COL_CI')} | {tr('REPORT_COL_N')} | {tr('REPORT_COL_TARGET')} "
        f"| {tr('REPORT_COL_STATUS')} |"
    )
    lines = [header, "|---|---|---:|---:|---:|---|---|"]
    for report in reports:
        # A wiring check gets no verdict cell either: the reason it may not be quoted has
        # widened past "the backend was a stub", and the table must widen with it.
        stub = bool(report.breakdown.get("stub")) or report.is_wiring_check
        for metric, value in sorted(report.metrics.items()):
            target = TARGETS.get((report.kind, metric))
            interval = report.ci95.get(metric)
            ci_text = (
                f"[{_format_value(interval[0])}, {_format_value(interval[1])}]"
                if interval
                else tr("REPORT_STATUS_NONE")
            )
            status = _status(target, value, stub=stub)
            target_text = target.render() if target else tr("REPORT_STATUS_NONE")
            lines.append(
                f"| {report.kind} | {metric} | {_format_value(value)} | {ci_text} "
                f"| {report.n} | {target_text} | {status} |"
            )
    lines.append("")
    for report in reports:
        lines.append(
            tr(
                "REPORT_PROVENANCE",
                name=report.name,
                sha=report.short_sha,
                fingerprint=report.dataset_fingerprint or "-",
                timestamp=report.timestamp,
            )
        )
        if not report.citable:
            lines.append(tr("REPORT_NOT_CITABLE"))
        verdict = report.wiring_check_verdict
        if report.breakdown.get("stub"):
            lines.append(tr("REPORT_STUB_WARNING"))
        elif verdict.is_wiring_check:
            # Until 2026-09-09 a wiring check of any kind printed the *stub* sentence, because a
            # verdict could only ever mean "stub" for a model-backed kind. It can now mean a
            # dirty tree, a generated corpus or too few samples, and telling a reader that a real
            # model was a fake one is a worse error than the one the warning exists to prevent.
            lines.append(tr("REPORT_WIRING_WARNING"))
        if verdict.is_wiring_check:
            # The codes are machine tokens, not prose, so they belong in the generated table as
            # they are; the Turkish sentence above already says what they mean.
            lines.append(f"`wiring_check`: {', '.join(verdict.codes)}")
        notes = report.breakdown.get("notes")
        if notes:
            lines.append(str(notes))
        lines.append("")
    lines.append(tr("REPORT_DEFINITIONS", ref=DEFINITIONS_REF))
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_OUT_DIR",
    "DEFINITIONS_REF",
    "LATEST_FILENAME",
    "MEASURED",
    "MIN_REPORTABLE_STEPS",
    "MIN_VAL_SAMPLES",
    "MODEL_BACKED_KINDS",
    "TARGETS",
    "WIRING_CHECK",
    "WIRING_CHECK_KEY",
    "WIRING_CHECK_PREFIX",
    "WIRING_CHECK_REASON",
    "WIRING_CHECK_SUFFIX",
    "WIRING_NOT_CITABLE",
    "WIRING_RANDOM_INIT",
    "WIRING_REASON_CODES",
    "WIRING_STEP_BUDGET_TOO_SMALL",
    "WIRING_STUB_BACKEND",
    "WIRING_SYNTHETIC_DATA",
    "WIRING_UNSPECIFIED",
    "WIRING_VAL_SUBSET_TOO_SMALL",
    "WIRING_WARMUP_NOT_COMPLETED",
    "EvalKindName",
    "EvalReport",
    "Target",
    "WiringCheckReason",
    "WiringCheckVerdict",
    "assess_wiring_check",
    "check_promoted",
    "check_report",
    "load_latest",
    "promote",
    "render_markdown",
    "wiring_check_reason_text",
    "write_failures",
    "write_report",
]
