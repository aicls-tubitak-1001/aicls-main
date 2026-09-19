/**
 * Telling a trained model from a stand-in — a scientific-integrity requirement, not decoration.
 *
 * The platform ships deterministic stubs so the whole pipeline can be demonstrated before the OCR
 * and ASR models are trained (`aicls.api.model_registry.ModelRegistry` answers "stub" whenever no
 * checkpoint is promoted). A demonstration driven by those stubs must never be mistaken for a
 * trained system — in a lecture, in a screenshot, or in a progress report — so every response's
 * `model_versions` is classified here and the UI says, in Turkish, what produced the answer.
 *
 * The version strings are the real ones:
 * `aicls.ocr.stub.STUB_MODEL_VERSION == "stub-0.1.0"`,
 * `aicls.asr.stub.STUB_MODEL_VERSION == "asr-stub-1"`,
 * and `ModelRegistry.set_recognizer` / `set_transcriber` record `"override"` for a model injected
 * at runtime (a warm swap or a test double), which is equally not a promoted checkpoint.
 */

/** What produced an answer. */
export type ModelKind =
  /** A deterministic stand-in: the pipeline runs, the recognition is imitated. */
  | "stub"
  /** A model injected at runtime rather than promoted through the registry. */
  | "override"
  /** A promoted checkpoint. */
  | "trained"
  /** The component reported no version (it was not involved, or the registry is silent). */
  | "unknown";

/** Substring that marks a stand-in version string. */
const STUB_MARKER = "stub";

/** Exact version string recorded for a runtime-injected model. */
const OVERRIDE_VERSION = "override";

/**
 * Classify one version string.
 *
 * @param version - A value of `ComputeOut.model_versions`, `RecognizeOut.model_version`, …
 * @returns Which kind of model produced the answer.
 *
 * @example
 * ```ts
 * classifyModelVersion("stub-0.1.0"); // "stub"
 * classifyModelVersion("asr-stub-1"); // "stub"
 * classifyModelVersion("ocr-2027.03"); // "trained"
 * classifyModelVersion(null); // "unknown"
 * ```
 */
export function classifyModelVersion(version: string | null | undefined): ModelKind {
  if (version === null || version === undefined || version.trim() === "") {
    return "unknown";
  }
  const normalised = version.trim().toLowerCase();
  if (normalised.includes(STUB_MARKER)) {
    return "stub";
  }
  if (normalised === OVERRIDE_VERSION) {
    return "override";
  }
  return "trained";
}

/** One classified component of a response. */
export interface ModelStatus {
  /** Component name as the API spelled it (`ocr`, `asr`, …). */
  readonly component: string;
  /** The raw version string, or `null`. */
  readonly version: string | null;
  readonly kind: ModelKind;
}

/**
 * Classify a whole `model_versions` map.
 *
 * Components are returned in a stable order (`ocr` then `asr` then anything else alphabetically)
 * so the badge does not reshuffle between two answers.
 */
export function classifyModelVersions(
  versions: Readonly<Record<string, string | null>> | undefined,
): ModelStatus[] {
  if (versions === undefined) {
    return [];
  }
  const preferred = ["ocr", "asr"];
  const rank = (name: string): number => {
    const index = preferred.indexOf(name);
    return index === -1 ? preferred.length : index;
  };
  return Object.entries(versions)
    .map(([component, version]) => ({
      component,
      version: version ?? null,
      kind: classifyModelVersion(version),
    }))
    .sort(
      (a, b) =>
        rank(a.component) - rank(b.component) || a.component.localeCompare(b.component, "tr"),
    );
}

/** Whether any component that actually answered is a stand-in. */
export function hasStubModel(
  versions: Readonly<Record<string, string | null>> | undefined,
): boolean {
  return classifyModelVersions(versions).some((status) => status.kind === "stub");
}
