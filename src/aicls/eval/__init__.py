"""Frozen metric definitions, normalizers, bootstrap CIs and evaluation harnesses.

This package is where every number in a TÜBİTAK progress report comes from. Its promise is
narrow and strict: **one definition per metric, one implementation, and provenance attached to
every value** (ADR-0015, docs/ml/degerlendirme.md).

Public entry points::

    latex_v1(s) / tr_v1(s) / NORMALIZER_VERSIONS  # frozen, versioned normalisation
    cer / corpus_cer / ter / corpus_ter  # character and token error rates
    wer / corpus_wer  # word error rate, raw and tr_v1
    exact_match / symbolic_equivalence  # expression level, tri-state
    parse_success_rate / latency_stats / sus_score  # IP-4, IP-2/IP-9
    bootstrap_ci(values, n_resamples=1000, seed=1234)  # 95 % intervals
    EvalReport / write_report / render_markdown  # the report document and its writers
    promote / load_latest / check_report  # reports/metrics/latest.json + CI gate
    run_parser_eval / run_cas_eval / run_nlu_eval / run_ocr_eval / run_asr_eval
    run_book_eval / run_latency_eval  # the per-component runs
    run_all  # the whole harness, one command

It depends on ``core``, ``latex2sympy``, ``mathengine``, ``nlu``, ``ocr``, ``asr``, ``book``,
``db`` and ``data``, and never on ``aicls.ml``: a metric must not be able to import the trainer
whose model it is judging (ADR-0002, ``eval-no-ml``). It never imports ``torch`` either — the
stub backends keep the whole harness runnable on a CPU-only machine, and a stub number is
always marked as such in the report it appears in.
"""

from __future__ import annotations

from aicls.eval.checkers import compare, compare_latex, rebuild_srepr
from aicls.eval.harness import HarnessOutput, run_all
from aicls.eval.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    LatencyStats,
    RateResult,
    bootstrap_ci,
    cer,
    corpus_cer,
    corpus_ter,
    corpus_wer,
    edit_distance,
    exact_match,
    exact_match_rate,
    latency_stats,
    parse_success_rate,
    percentile,
    sus_score,
    symbolic_equivalence,
    ter,
    wer,
)
from aicls.eval.normalizers import (
    NORMALIZER_VERSIONS,
    NORMALIZERS,
    apply_norm,
    latex_tokens,
    latex_v1,
    tr_v1,
    tr_words,
    turkish_number_words,
)
from aicls.eval.pipeline import LocalPipelineClient
from aicls.eval.provenance import GitInfo, git_info, known_shas, utc_timestamp
from aicls.eval.report import (
    DEFINITIONS_REF,
    LATEST_FILENAME,
    MIN_REPORTABLE_STEPS,
    MIN_VAL_SAMPLES,
    TARGETS,
    WIRING_CHECK_KEY,
    WIRING_REASON_CODES,
    EvalReport,
    Target,
    WiringCheckReason,
    WiringCheckVerdict,
    assess_wiring_check,
    check_promoted,
    check_report,
    load_latest,
    promote,
    render_markdown,
    wiring_check_reason_text,
    write_failures,
    write_report,
)
from aicls.eval.runners import (
    run_asr_eval,
    run_book_eval,
    run_cas_eval,
    run_latency_eval,
    run_nlu_eval,
    run_ocr_eval,
    run_parser_eval,
)

__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "DEFINITIONS_REF",
    "LATEST_FILENAME",
    "MIN_REPORTABLE_STEPS",
    "MIN_VAL_SAMPLES",
    "NORMALIZERS",
    "NORMALIZER_VERSIONS",
    "TARGETS",
    "WIRING_CHECK_KEY",
    "WIRING_REASON_CODES",
    "EvalReport",
    "GitInfo",
    "HarnessOutput",
    "LatencyStats",
    "LocalPipelineClient",
    "RateResult",
    "Target",
    "WiringCheckReason",
    "WiringCheckVerdict",
    "apply_norm",
    "assess_wiring_check",
    "bootstrap_ci",
    "cer",
    "check_promoted",
    "check_report",
    "compare",
    "compare_latex",
    "corpus_cer",
    "corpus_ter",
    "corpus_wer",
    "edit_distance",
    "exact_match",
    "exact_match_rate",
    "git_info",
    "known_shas",
    "latency_stats",
    "latex_tokens",
    "latex_v1",
    "load_latest",
    "parse_success_rate",
    "percentile",
    "promote",
    "rebuild_srepr",
    "render_markdown",
    "run_all",
    "run_asr_eval",
    "run_book_eval",
    "run_cas_eval",
    "run_latency_eval",
    "run_nlu_eval",
    "run_ocr_eval",
    "run_parser_eval",
    "sus_score",
    "symbolic_equivalence",
    "ter",
    "tr_v1",
    "tr_words",
    "turkish_number_words",
    "utc_timestamp",
    "wer",
    "wiring_check_reason_text",
    "write_failures",
    "write_report",
]
