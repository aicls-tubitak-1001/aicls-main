/**
 * The measured numbers rendered by the status page, transcribed from `reports/metrics/latest.json`.
 *
 * Why a transcription and not a fetch: the numbers belong to a specific evaluation run identified
 * by a git sha and a dataset fingerprint, and the guide's whole point is that a reader can see
 * *which* run they are looking at. A live endpoint would show whatever the server last computed
 * (and `GET /admin/eval-runs` is admin-only anyway, so a student could never see it). This module
 * is therefore a dated snapshot: it carries the run's provenance with it and is refreshed
 * deliberately, the same way a report table is.
 *
 * Nothing here is rounded, filtered or improved. Every value below appears in
 * `reports/metrics/latest.json` exactly as written, including the ones that miss their target, and
 * every provenance warning printed by `reports/metrics/summary.md` is attached to the run it
 * belongs to. Metric definitions are frozen in `docs/ml/degerlendirme.md#2-metrikler` (ADR-0015).
 */
import type { MessageKey } from "@/i18n";

/**
 * How much a number is worth as evidence.
 *
 * - `evidence` — measured on a set the project did not author; a performance claim may rest on it.
 * - `regression` — measured on a corpus written inside the project; it catches breakage, it does
 *   not demonstrate independent performance.
 * - `stub` — produced by an untrained placeholder model; it proves the wiring, nothing else.
 */
export type EvidenceLevel = "evidence" | "regression" | "stub";

/** How a value is formatted for a Turkish reader. */
export type MetricUnit = "ratio" | "ms" | "count";

/** A contractual threshold and whether the harness judged it met. */
export interface MetricTarget {
  /** The threshold as it is printed, e.g. `"< 0,10"`. */
  readonly text: string;
  /** What the evaluation harness reported, independently of the evidence level. */
  readonly met: boolean;
}

/** One measured number. */
export interface GuideMetric {
  readonly id: string;
  readonly labelKey: MessageKey;
  readonly value: number;
  readonly unit: MetricUnit;
  /** Bootstrap 95 % interval when the harness computed one. */
  readonly ci95: readonly [number, number] | null;
  readonly target: MetricTarget | null;
}

/** One evaluation run, with everything needed to say where its numbers came from. */
export interface GuideRun {
  readonly id: string;
  readonly titleKey: MessageKey;
  readonly evidence: EvidenceLevel;
  /** Sample size of the run. */
  readonly n: number;
  readonly datasetVersion: string;
  readonly datasetFingerprint: string;
  readonly gitSha: string;
  readonly generatedAt: string;
  readonly modelVersion: string | null;
  /** False when the run was produced on a dirty tree; such a run may not enter a report. */
  readonly citable: boolean;
  /** Provenance warnings printed with this run, in the order they must be read. */
  readonly noticeKeys: readonly MessageKey[];
  readonly metrics: readonly GuideMetric[];
}

/** Everything the status page renders. */
export interface MetricsSnapshot {
  readonly gitSha: string;
  readonly generatedAt: string;
  /** Where the frozen metric definitions live. */
  readonly definitionsRef: string;
  readonly runs: readonly GuideRun[];
}

/** Short form of the run's commit, as printed in `summary.md`. */
export function shortSha(sha: string): string {
  return sha.slice(0, 7);
}

/**
 * The committed snapshot.
 *
 * Source: `reports/metrics/latest.json`, run `1581885`, generated 2026-09-07. Every run in that
 * file is marked `citable: false` because the tree was dirty; the status page says so out loud.
 */
export const METRICS_SNAPSHOT: MetricsSnapshot = {
  gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
  generatedAt: "2026-09-07T19:09:31Z",
  definitionsRef: "docs/ml/degerlendirme.md#2-metrikler",
  runs: [
    {
      id: "parser",
      titleKey: "guide.status.run.parser",
      evidence: "regression",
      n: 3709,
      datasetVersion: "latex-valid-inputs-2026.09",
      datasetFingerprint: "sha256:9ff283b2c0d163194867f05716d73078c899b95cbda41e5574f3b6707c152ed0",
      gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
      generatedAt: "2026-09-07T19:09:16Z",
      modelVersion: null,
      citable: false,
      noticeKeys: [],
      metrics: [
        {
          id: "parser.success_rate",
          labelKey: "guide.status.metric.successRate",
          value: 1,
          unit: "ratio",
          ci95: [1, 1],
          target: { text: "> 0,99", met: true },
        },
        {
          id: "parser.failures",
          labelKey: "guide.status.metric.failures",
          value: 0,
          unit: "count",
          ci95: null,
          target: null,
        },
      ],
    },
    {
      id: "cas",
      titleKey: "guide.status.run.cas",
      evidence: "regression",
      n: 1204,
      datasetVersion: "curriculum-2026.09-v1",
      datasetFingerprint: "sha256:a84a2e1ee92baf12a534f5a69e293164e7955f0c283e5fba0405c740872e20e5",
      gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
      generatedAt: "2026-09-07T19:09:26Z",
      modelVersion: null,
      citable: false,
      noticeKeys: ["guide.status.notice.curriculum"],
      metrics: [
        {
          id: "cas.correct_rate",
          labelKey: "guide.status.metric.correctRate",
          value: 1,
          unit: "ratio",
          ci95: [1, 1],
          target: { text: "> 0,99", met: true },
        },
        {
          id: "cas.error_rate",
          labelKey: "guide.status.metric.errorRate",
          value: 0,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "cas.unsupported_rate",
          labelKey: "guide.status.metric.unsupportedRate",
          value: 0,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "cas.parse_success_rate",
          labelKey: "guide.status.metric.parseSuccessRate",
          value: 1,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "cas.mean_ms",
          labelKey: "guide.status.metric.meanMs",
          value: 5.262149501661123,
          unit: "ms",
          ci95: null,
          target: null,
        },
        {
          id: "cas.p95_ms",
          labelKey: "guide.status.metric.p95Ms",
          value: 17.3113,
          unit: "ms",
          ci95: null,
          target: null,
        },
      ],
    },
    {
      id: "nlu",
      titleKey: "guide.status.run.nlu",
      evidence: "regression",
      n: 351,
      datasetVersion: "nlu-commands-2026.09",
      datasetFingerprint: "sha256:bc4a60ac1665c0044f751a03cf56406ca3b61c0fac54cf034896c39b31d81a74",
      gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
      generatedAt: "2026-09-07T19:09:27Z",
      modelVersion: null,
      citable: false,
      noticeKeys: ["guide.status.notice.nluCorpus"],
      metrics: [
        {
          id: "nlu.intent_accuracy",
          labelKey: "guide.status.metric.intentAccuracy",
          value: 0.9971509971509972,
          unit: "ratio",
          ci95: [0.9914529914529915, 1],
          target: { text: ">= 0,95", met: true },
        },
        {
          id: "nlu.command_exact",
          labelKey: "guide.status.metric.commandExact",
          value: 0.9971509971509972,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "nlu.slot_f1",
          labelKey: "guide.status.metric.slotF1",
          value: 0.7064305684995341,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "nlu.slot_f1_declared_only",
          labelKey: "guide.status.metric.slotF1Declared",
          value: 1,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "nlu.no_match_rate",
          labelKey: "guide.status.metric.noMatchRate",
          value: 0.06837606837606838,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "nlu.clarification_rate",
          labelKey: "guide.status.metric.clarificationRate",
          value: 0,
          unit: "ratio",
          ci95: null,
          target: null,
        },
      ],
    },
    {
      id: "ocr",
      titleKey: "guide.status.run.ocr",
      evidence: "stub",
      n: 128,
      datasetVersion: "synthetic-curriculum-ink-v1",
      datasetFingerprint: "sha256:6317985ce7728e72f609ff63eb436dea1bd68af73c87f71a51e44630295133bb",
      gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
      generatedAt: "2026-09-07T19:09:27Z",
      modelVersion: "stub-0.1.0",
      citable: false,
      noticeKeys: ["guide.status.notice.ocrStub"],
      metrics: [
        {
          id: "ocr.cer",
          labelKey: "guide.status.metric.cer",
          value: 0,
          unit: "ratio",
          ci95: [0, 0],
          target: { text: "< 0,10", met: true },
        },
        {
          id: "ocr.ter",
          labelKey: "guide.status.metric.ter",
          value: 0,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "ocr.exprate",
          labelKey: "guide.status.metric.exprate",
          value: 1,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "ocr.p95_ms",
          labelKey: "guide.status.metric.p95Ms",
          value: 0.0957900017965585,
          unit: "ms",
          ci95: null,
          target: null,
        },
      ],
    },
    {
      id: "asr",
      titleKey: "guide.status.run.asr",
      evidence: "stub",
      n: 25,
      datasetVersion: "synthetic-command-audio-v1",
      datasetFingerprint: "sha256:83edf842c3dbde2b1d96a7704c5ec91fea90cf66469e0d7d2061466b2f0c0579",
      gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
      generatedAt: "2026-09-07T19:09:30Z",
      modelVersion: "asr-stub-1",
      citable: false,
      noticeKeys: ["guide.status.notice.asrStub"],
      metrics: [
        {
          id: "asr.wer_tr_v1",
          labelKey: "guide.status.metric.werTr",
          value: 1.111764705882353,
          unit: "ratio",
          ci95: [1, 1.2618084851390219],
          target: { text: "< 0,10", met: false },
        },
        {
          id: "asr.wer_raw",
          labelKey: "guide.status.metric.werRaw",
          value: 1.111764705882353,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "asr.cer",
          labelKey: "guide.status.metric.cer",
          value: 0.8922363847045192,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "asr.command_intent_accuracy",
          labelKey: "guide.status.metric.commandIntentAccuracy",
          value: 0,
          unit: "ratio",
          ci95: null,
          target: { text: "> 0,95", met: false },
        },
        {
          id: "asr.command_exact",
          labelKey: "guide.status.metric.commandExact",
          value: 0,
          unit: "ratio",
          ci95: null,
          target: null,
        },
      ],
    },
    {
      id: "book",
      titleKey: "guide.status.run.book",
      evidence: "regression",
      n: 72,
      datasetVersion: "book-fetch-2026.09",
      datasetFingerprint: "sha256:168e60b283d99255e758a03fa59d61d40457a44636e8b7daca34e87b098076d9",
      gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
      generatedAt: "2026-09-07T19:09:30Z",
      modelVersion: null,
      citable: false,
      noticeKeys: ["guide.status.notice.bookSample"],
      metrics: [
        {
          id: "book.fetch_top1",
          labelKey: "guide.status.metric.fetchTop1",
          value: 1,
          unit: "ratio",
          ci95: [1, 1],
          target: { text: "> 0,95", met: true },
        },
        {
          id: "book.fetch_top3",
          labelKey: "guide.status.metric.fetchTop3",
          value: 1,
          unit: "ratio",
          ci95: null,
          target: null,
        },
        {
          id: "book.tagging_coverage",
          labelKey: "guide.status.metric.taggingCoverage",
          value: 1,
          unit: "ratio",
          ci95: null,
          target: { text: "= 1,00", met: true },
        },
      ],
    },
    {
      id: "latency",
      titleKey: "guide.status.run.latency",
      evidence: "stub",
      n: 200,
      datasetVersion: "nlu-commands-2026.09",
      datasetFingerprint: "sha256:bc4a60ac1665c0044f751a03cf56406ca3b61c0fac54cf034896c39b31d81a74",
      gitSha: "15818854eaf7fd7fcf5b82e961dec0d4e06f7aad",
      generatedAt: "2026-09-07T19:09:31Z",
      modelVersion: null,
      citable: false,
      noticeKeys: ["guide.status.notice.latencyStub"],
      metrics: [
        {
          id: "latency.mean_ms",
          labelKey: "guide.status.metric.meanMs",
          value: 4.889779499499127,
          unit: "ms",
          ci95: [3.246441037616023, 6.64017079914629],
          target: { text: "< 3000", met: true },
        },
        {
          id: "latency.p50_ms",
          labelKey: "guide.status.metric.p50Ms",
          value: 1.7185999968205579,
          unit: "ms",
          ci95: null,
          target: null,
        },
        {
          id: "latency.p95_ms",
          labelKey: "guide.status.metric.p95Ms",
          value: 39.35216000172655,
          unit: "ms",
          ci95: null,
          target: null,
        },
      ],
    },
  ],
};

/** How much evidence a contractual target has today. */
export type ContractEvidence = "none" | "stub" | "regression" | "partial";

/** One row of the "what is proven today" table. */
export interface ContractTarget {
  readonly id: string;
  readonly labelKey: MessageKey;
  readonly todayKey: MessageKey;
  readonly evidence: ContractEvidence;
}

/**
 * The seven binding success metrics of the proposal, against today's evidence.
 *
 * `evidence` is deliberately the harshest defensible reading: a number produced by a stub is
 * `"stub"` even when the harness printed "tuttu", and a number measured on a corpus the project
 * wrote itself is `"regression"` even when it is 1,0000.
 */
export const CONTRACT_TARGETS: readonly ContractTarget[] = [
  {
    id: "ocr",
    labelKey: "guide.status.contract.ocr",
    todayKey: "guide.status.contract.ocrToday",
    evidence: "stub",
  },
  {
    id: "asr",
    labelKey: "guide.status.contract.asr",
    todayKey: "guide.status.contract.asrToday",
    evidence: "stub",
  },
  {
    id: "parse",
    labelKey: "guide.status.contract.parse",
    todayKey: "guide.status.contract.parseToday",
    evidence: "regression",
  },
  {
    id: "cas",
    labelKey: "guide.status.contract.cas",
    todayKey: "guide.status.contract.casToday",
    evidence: "regression",
  },
  {
    id: "latency",
    labelKey: "guide.status.contract.latency",
    todayKey: "guide.status.contract.latencyToday",
    evidence: "stub",
  },
  {
    id: "sus",
    labelKey: "guide.status.contract.sus",
    todayKey: "guide.status.contract.susToday",
    evidence: "none",
  },
  {
    id: "book",
    labelKey: "guide.status.contract.book",
    todayKey: "guide.status.contract.bookToday",
    evidence: "regression",
  },
];

/** The Turkish label of an evidence level (status table badge). */
export const EVIDENCE_LABEL_KEY: Readonly<Record<EvidenceLevel, MessageKey>> = {
  evidence: "guide.status.evidence.evidenceLabel",
  regression: "guide.status.evidence.regressionLabel",
  stub: "guide.status.evidence.stubLabel",
};

/** The one-sentence explanation of an evidence level. */
export const EVIDENCE_HINT_KEY: Readonly<Record<EvidenceLevel, MessageKey>> = {
  evidence: "guide.status.evidence.evidenceHint",
  regression: "guide.status.evidence.regressionHint",
  stub: "guide.status.evidence.stubHint",
};

/** The Turkish label of a contract-evidence level. */
export const CONTRACT_EVIDENCE_KEY: Readonly<Record<ContractEvidence, MessageKey>> = {
  none: "guide.status.contract.evidence.none",
  stub: "guide.status.contract.evidence.stub",
  regression: "guide.status.contract.evidence.regression",
  partial: "guide.status.contract.evidence.partial",
};

const RATIO_FORMAT = new Intl.NumberFormat("tr-TR", {
  minimumFractionDigits: 4,
  maximumFractionDigits: 4,
});
const MS_FORMAT = new Intl.NumberFormat("tr-TR", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const COUNT_FORMAT = new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 0 });

/**
 * Format a measured value for a Turkish reader.
 *
 * Ratios keep four decimals because that is the precision `summary.md` prints and the precision a
 * reviewer will compare against; milliseconds keep two. The decimal separator is a comma.
 */
export function formatMetricValue(value: number, unit: MetricUnit): string {
  if (unit === "ms") {
    return `${MS_FORMAT.format(value)} ms`;
  }
  if (unit === "count") {
    return COUNT_FORMAT.format(value);
  }
  return RATIO_FORMAT.format(value);
}

/** Format a bootstrap interval as `[alt, üst]`, in the same precision as the value. */
export function formatInterval(interval: readonly [number, number], unit: MetricUnit): string {
  return `[${formatMetricValue(interval[0], unit)}; ${formatMetricValue(interval[1], unit)}]`;
}
