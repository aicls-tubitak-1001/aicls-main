/**
 * The model registry (`GET /admin/models`, `POST /admin/models/activate`).
 *
 * The panel answers one question first, in words: **is the system currently answering with a
 * trained model or with a deterministic stand-in?** `lib/modelVersions` classifies each version
 * string exactly as the rest of the UI does (`stub-0.1.0`, `asr-stub-1`, `override`), so the badge
 * next to a checkpoint here and the badge next to a result in the workspace can never disagree.
 * A component with no active row at all is called out explicitly, because that is precisely the
 * state in which a demonstration could be mistaken for a trained system.
 *
 * Activation is audited server-side and reloads the registry, so the list is refetched afterwards
 * rather than patched locally.
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import type { ModelVersionOut } from "@/api/types";
import { Button, Card, Chip, EmptyState, ErrorState, Spinner } from "@/components/ui";
import { t } from "@/i18n";
import { classifyModelVersion, type ModelKind } from "@/lib/modelVersions";
import { activateModel, fetchModels, type ModelComponent } from "./api";
import { metricEntries } from "./format";
import styles from "./Admin.module.css";

/** The two components the registry knows (`ActivateIn.component`). */
const COMPONENTS: ModelComponent[] = ["ocr", "asr"];

const KIND_TONE = {
  stub: "warn",
  override: "warn",
  trained: "ok",
  unknown: "neutral",
} as const;

const KIND_LABEL = {
  stub: "model.stub",
  override: "model.override",
  trained: "model.trained",
  unknown: "model.unknown",
} as const satisfies Record<ModelKind, string>;

function componentLabel(component: string): string {
  if (component === "ocr") {
    return t("model.componentOcr");
  }
  if (component === "asr") {
    return t("model.componentAsr");
  }
  return component;
}

/** The registry. */
export function ModelRegistryPanel(): JSX.Element {
  const [models, setModels] = useState<ModelVersionOut[]>([]);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setLoading(true);
    const { data, problem: failure } = await fetchModels();
    setLoading(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    setProblem(null);
    setModels(data);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const activate = async (row: ModelVersionOut): Promise<void> => {
    setBusy(`${row.component}:${row.version}`);
    setProblem(null);
    const { problem: failure } = await activateModel(row.component as ModelComponent, row.version);
    setBusy(null);
    if (failure !== null) {
      setProblem(failure);
      return;
    }
    setNotice(t("admin.modelsActivated", { version: row.version }));
    await load();
  };

  const activeOf = (component: string): ModelVersionOut | undefined =>
    models.find((row) => row.component === component && row.is_active);

  return (
    <Card
      title={t("admin.modelsTitle")}
      actions={
        <Button
          variant="ghost"
          onClick={() => {
            void load();
          }}
        >
          {t("admin.refresh")}
        </Button>
      }
    >
      <p className={styles.lead}>{t("admin.modelsIntro")}</p>

      <ul className={styles.metrics}>
        {COMPONENTS.map((component) => {
          const active = activeOf(component);
          const kind = classifyModelVersion(active?.version);
          return (
            <li className={styles.metric} key={component}>
              <span className={styles.metricLabel}>{componentLabel(component)}</span>
              <span className={styles.metricValue}>{active?.version ?? t("model.noVersion")}</span>
              <Chip tone={KIND_TONE[kind]}>{t(KIND_LABEL[kind])}</Chip>
              {active === undefined ? (
                <span className={styles.metricHint}>
                  {t("admin.modelsNoActive", { component: componentLabel(component) })}
                </span>
              ) : null}
              {kind === "stub" || kind === "override" ? (
                <span className={styles.warningText}>{t("admin.modelsStubWarning")}</span>
              ) : null}
            </li>
          );
        })}
      </ul>

      {problem === null ? null : (
        <ErrorState
          problem={problem}
          onRetry={() => {
            void load();
          }}
        />
      )}
      {notice === null ? null : (
        <p className={styles.notice} role="status">
          {notice}
        </p>
      )}
      {loading && models.length === 0 ? <Spinner label={t("common.loading")} /> : null}
      {!loading && models.length === 0 ? <EmptyState title={t("admin.modelsEmpty")} /> : null}

      {models.length === 0 ? null : (
        <div className="scroll-x">
          <table className={styles.table}>
            <caption className="visually-hidden">
              {t("admin.tableCaption", { title: t("admin.modelsTitle") })}
            </caption>
            <thead>
              <tr>
                <th scope="col">{t("admin.modelsComponent")}</th>
                <th scope="col">{t("admin.modelsVersion")}</th>
                <th scope="col">{t("admin.modelsActive")}</th>
                <th scope="col">{t("admin.modelsPromoted")}</th>
                <th scope="col">{t("admin.modelsMetrics")}</th>
                <th scope="col">{t("admin.actions")}</th>
              </tr>
            </thead>
            <tbody>
              {models.map((row) => {
                const kind = classifyModelVersion(row.version);
                const key = `${row.component}:${row.version}`;
                return (
                  <tr key={key}>
                    <td>{componentLabel(row.component)}</td>
                    <th scope="row" className={styles.mono}>
                      {row.version} <Chip tone={KIND_TONE[kind]}>{t(KIND_LABEL[kind])}</Chip>
                    </th>
                    <td>
                      {row.is_active ? (
                        <Chip tone="ok">{t("admin.modelsActiveBadge")}</Chip>
                      ) : (
                        t("common.no")
                      )}
                    </td>
                    <td>{row.promoted_at ?? t("common.none")}</td>
                    <td className={styles.mono}>
                      {metricEntries(row.metrics ?? {})
                        .map((entry) => `${entry.name}=${entry.value}`)
                        .join(" ") || t("common.none")}
                    </td>
                    <td>
                      {row.is_active ? null : (
                        <Button
                          variant="secondary"
                          loading={busy === key}
                          onClick={() => {
                            void activate(row);
                          }}
                        >
                          {t("admin.modelsActivate")}
                        </Button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
