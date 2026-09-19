/**
 * `ModelBadge` — what produced this answer.
 *
 * Every response carries `model_versions`; when any of them is a deterministic stand-in the badge
 * says so in plain Turkish ("taklit model") next to the result, not in a tooltip and not in grey
 * six-point type. A demonstration must never be mistaken for a trained system: this is the same
 * requirement that makes the CAS deterministic, expressed in the interface.
 *
 * The colour is never the only signal — the word "taklit" is in the label, and the detail line
 * spells out the component and its version string.
 */
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import { classifyModelVersions, type ModelKind, type ModelStatus } from "@/lib/modelVersions";
import { Chip } from "./Chip";
import styles from "./ModelBadge.module.css";

/** i18n key of the short label for each kind. */
const KIND_LABEL: Record<ModelKind, MessageKey> = {
  stub: "model.stub",
  override: "model.override",
  trained: "model.trained",
  unknown: "model.unknown",
};

/** i18n key of the sentence explaining each kind. */
const KIND_EXPLANATION: Record<ModelKind, MessageKey> = {
  stub: "model.stubExplanation",
  override: "model.overrideExplanation",
  trained: "model.trainedExplanation",
  unknown: "model.unknownExplanation",
};

const KIND_TONE = {
  stub: "warn",
  override: "warn",
  trained: "ok",
  unknown: "neutral",
} as const;

/** i18n key of a component's Turkish name; unknown components fall back to their raw name. */
const COMPONENT_LABEL: Record<string, MessageKey> = {
  ocr: "model.componentOcr",
  asr: "model.componentAsr",
};

function componentLabel(component: string): string {
  const key = COMPONENT_LABEL[component];
  return key === undefined ? component : t(key);
}

export interface ModelBadgeProps {
  /** `ComputeOut.model_versions` (or `VersionOut.models`). */
  versions: Readonly<Record<string, string | null>> | undefined;
  /** Hide components that reported no version at all (the default). */
  hideUnknown?: boolean;
  /**
   * Render only the stand-in components, and render nothing at all when every model is trained.
   *
   * This is how the warning gets to sit directly above the answer without adding a line to a
   * normal result: the badge disappears completely when there is nothing to warn about.
   */
  warningOnly?: boolean;
  /**
   * Show the warning paragraph (default `true`).
   *
   * Set to `false` on the second, detailed badge of the same answer — the warning is stated once,
   * where it cannot be scrolled past, and the versions are listed once, in the footer.
   */
  showWarning?: boolean;
  className?: string;
}

/**
 * Render one chip per model component, with a prominent warning when a stand-in answered.
 *
 * @example
 * ```tsx
 * <ModelBadge versions={outcome.model_versions} />
 * ```
 */
export function ModelBadge({
  versions,
  hideUnknown = true,
  warningOnly = false,
  showWarning = true,
  className,
}: ModelBadgeProps): JSX.Element | null {
  const all = classifyModelVersions(versions);
  const shown = hideUnknown ? all.filter((status) => status.kind !== "unknown") : all;
  const stubs = shown.filter((status) => status.kind === "stub" || status.kind === "override");

  if (shown.length === 0) {
    return null;
  }
  if (warningOnly && stubs.length === 0) {
    return null;
  }

  const listed: ModelStatus[] = warningOnly ? stubs : shown;
  return (
    <div className={cx(styles.badge, className)}>
      {showWarning && stubs.length > 0 ? (
        <p className={styles.warning} data-testid="model-stub-warning">
          <strong>{t("model.stubWarningTitle")}</strong> {t("model.stubWarningBody")}
        </p>
      ) : null}
      <ul className={styles.list}>
        {listed.map((status) => (
          <li key={status.component} className={styles.item}>
            <Chip tone={KIND_TONE[status.kind]} title={t(KIND_EXPLANATION[status.kind])}>
              {componentLabel(status.component)}: {t(KIND_LABEL[status.kind])}
              {/* `title` is a pointer convenience and nothing more: on a touch board and to a
                  screen reader the same sentence arrives through the text below. */}
              <span className="visually-hidden"> — {t(KIND_EXPLANATION[status.kind])}</span>
            </Chip>
            <span className={styles.version}>{status.version ?? t("model.noVersion")}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
