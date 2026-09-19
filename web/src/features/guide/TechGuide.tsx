/**
 * Audience B: the technical reader.
 *
 * A reviewer, a new developer or a maintainer who has to continue the project should be able to
 * read this page and know what the system is made of, in what order the pieces run, and which
 * decisions are load-bearing. Every statement here is drawn from the repository — the architecture
 * documents, the ADRs, the generated OpenAPI document and the code itself — and nothing is
 * invented: where a number appears (operation counts, timeouts, key lengths) it is the number the
 * repository actually holds.
 *
 * The page deliberately spends its longest passage on why the answer is computed by a computer
 * algebra system rather than by a language model, because that is the project's central claim and
 * the reason its metrics mean anything.
 */
import { t, type MessageKey } from "@/i18n";
import { Callout } from "./Callout";
import { CodeBlock } from "./CodeBlock";
import { PipelineDiagram } from "./PipelineDiagram";
import { Section } from "./Section";
import { TECH } from "./sections";
import {
  API_TAGS,
  API_TOTALS,
  AUDIO_PROTOCOL,
  ERROR_ROWS,
  INK_PROTOCOL,
  PROBLEM_EXAMPLE,
  SERVER_TIMING_EXAMPLE,
} from "./apiSurface";
import styles from "./GuideArticle.module.css";
import tables from "./tables.module.css";

interface TermRow {
  readonly term: string;
  readonly descKey: MessageKey;
}

const FRONTEND: readonly TermRow[] = [
  { term: "React 18 + React Router 6", descKey: "guide.tech.frontend.react" },
  { term: "TypeScript (strict)", descKey: "guide.tech.frontend.ts" },
  { term: "Vite 6", descKey: "guide.tech.frontend.vite" },
  { term: "KaTeX", descKey: "guide.tech.frontend.katex" },
  { term: "openapi-fetch", descKey: "guide.tech.frontend.openapi" },
  { term: "WebSocket", descKey: "guide.tech.frontend.ws" },
  { term: "CSS Modules", descKey: "guide.tech.frontend.css" },
  { term: "Vitest + Playwright", descKey: "guide.tech.frontend.test" },
];

const BACKEND: readonly TermRow[] = [
  { term: "FastAPI", descKey: "guide.tech.backend.fastapi" },
  { term: "SQLAlchemy 2 + Alembic", descKey: "guide.tech.backend.sqlalchemy" },
  { term: "SQLite / MySQL 8", descKey: "guide.tech.backend.db" },
  { term: "Lark", descKey: "guide.tech.backend.lark" },
  { term: "SymPy", descKey: "guide.tech.backend.sympy" },
  { term: "Matplotlib", descKey: "guide.tech.backend.matplotlib" },
  { term: "PyTorch / Hugging Face", descKey: "guide.tech.backend.torch" },
  { term: "structlog", descKey: "guide.tech.backend.structlog" },
];

const DATA_GROUPS: readonly TermRow[] = [
  { term: "users, refresh_tokens, workspaces", descKey: "guide.tech.data.identity" },
  { term: "operations", descKey: "guide.tech.data.operations" },
  { term: "assets", descKey: "guide.tech.data.assets" },
  { term: "recognitions, transcripts", descKey: "guide.tech.data.models" },
  {
    term: "consent_versions, speakers, prompt_sentences, recordings",
    descKey: "guide.tech.data.collect",
  },
  { term: "book_units, book_aliases, book_triples, book_assets", descKey: "guide.tech.data.book" },
  {
    term: "model_versions, eval_runs, latency_samples, feedback, sus_responses, audit_log",
    descKey: "guide.tech.data.eval",
  },
];

const SECURITY: readonly MessageKey[] = [
  "guide.tech.security.roles",
  "guide.tech.security.tokens",
  "guide.tech.security.passwords",
  "guide.tech.security.latex",
  "guide.tech.security.input",
  "guide.tech.security.kvkk",
  "guide.tech.security.transport",
];

const GATES: readonly TermRow[] = [
  { term: "DATASET_DOWNLOAD", descKey: "guide.tech.gates.dataset" },
  { term: "WEIGHTS_DOWNLOAD", descKey: "guide.tech.gates.weights" },
  { term: "CUDA_INSTALL", descKey: "guide.tech.gates.cuda" },
  { term: "TRAINING", descKey: "guide.tech.gates.training" },
];

const GUARANTEES: readonly MessageKey[] = [
  "guide.tech.why.g1",
  "guide.tech.why.g2",
  "guide.tech.why.g3",
  "guide.tech.why.g4",
];

const EVAL_METRICS: readonly TermRow[] = [
  { term: "CER", descKey: "guide.tech.eval.cer" },
  { term: "WER", descKey: "guide.tech.eval.wer" },
  { term: "parse_success_rate", descKey: "guide.tech.eval.parse" },
  { term: "correct_rate", descKey: "guide.tech.eval.cas" },
  { term: "command_intent_accuracy", descKey: "guide.tech.eval.intent" },
  { term: "latency", descKey: "guide.tech.eval.latency" },
  { term: "tagging_coverage / fetch_top1", descKey: "guide.tech.eval.book" },
  { term: "SUS", descKey: "guide.tech.eval.sus" },
];

/** A definition list of "name — what it is". */
function TermList({ rows }: { rows: readonly TermRow[] }): JSX.Element {
  return (
    <dl className={styles.terms}>
      {rows.map((row) => (
        <div key={row.term} className={styles.termRow}>
          <dt className={styles.term}>{row.term}</dt>
          <dd className={styles.definition}>{t(row.descKey)}</dd>
        </div>
      ))}
    </dl>
  );
}

/** The technical guide's article body. */
export function TechGuide(): JSX.Element {
  return (
    <>
      <p className={styles.lead}>{t("guide.tech.lead")}</p>

      <Section {...TECH["boru-hatti"]}>
        <p>{t("guide.tech.pipeline.p1")}</p>
        <PipelineDiagram />
        <p>{t("guide.tech.pipeline.p2")}</p>
      </Section>

      <Section {...TECH["neden-cebir-motoru"]}>
        <p>{t("guide.tech.why.p1")}</p>
        <p>{t("guide.tech.why.p2")}</p>
        <ul className={styles.list}>
          {GUARANTEES.map((key) => (
            <li key={key}>{t(key)}</li>
          ))}
        </ul>
        <p>{t("guide.tech.why.p3")}</p>
        <Callout tone="info" label={t("guide.tech.why.llmLabel")}>
          <p>{t("guide.tech.why.llm")}</p>
        </Callout>
      </Section>

      <Section {...TECH["on-yuz-yigini"]}>
        <p>{t("guide.tech.frontend.p1")}</p>
        <TermList rows={FRONTEND} />
      </Section>

      <Section {...TECH["arka-uc-yigini"]}>
        <p>{t("guide.tech.backend.p1")}</p>
        <TermList rows={BACKEND} />
      </Section>

      <Section {...TECH["api-yuzeyi"]}>
        <p>{t("guide.tech.api.p1")}</p>
        <p data-testid="guide-api-counts">
          {t("guide.tech.api.counts", {
            paths: API_TOTALS.paths,
            ops: API_TOTALS.operations,
            schemas: API_TOTALS.schemas,
          })}
        </p>

        <div className={`${tables.wrap} scroll-x`}>
          <table className={tables.table}>
            <caption className={tables.caption}>{t("guide.tech.api.tagsTitle")}</caption>
            <thead>
              <tr>
                <th scope="col">{t("guide.tech.api.colTag")}</th>
                <th scope="col">{t("guide.tech.api.colCount")}</th>
                <th scope="col">{t("guide.tech.api.colPurpose")}</th>
              </tr>
            </thead>
            <tbody>
              {API_TAGS.map((row) => (
                <tr key={row.tag}>
                  <th scope="row" className={tables.mono}>
                    {row.tag}
                  </th>
                  <td className={tables.numeric}>{row.count}</td>
                  <td>{t(row.purposeKey)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 className={styles.panelTitle}>{t("guide.tech.api.errorTitle")}</h3>
        <p>{t("guide.tech.api.errorP1")}</p>
        <CodeBlock code={PROBLEM_EXAMPLE} label={t("guide.tech.api.errorTitle")} />
        <p>{t("guide.tech.api.errorP2")}</p>

        <div className={`${tables.wrap} scroll-x`}>
          <table className={tables.table} data-testid="guide-error-table">
            <caption className={tables.caption}>{t("guide.tech.api.errorTitle")}</caption>
            <thead>
              <tr>
                <th scope="col">{t("guide.tech.api.colCode")}</th>
                <th scope="col">{t("guide.tech.api.colStatus")}</th>
                <th scope="col">{t("guide.tech.api.colWhen")}</th>
              </tr>
            </thead>
            <tbody>
              {ERROR_ROWS.map((row) => (
                <tr key={row.code}>
                  <th scope="row" className={tables.mono}>
                    {row.code}
                  </th>
                  <td className={tables.numeric}>{row.status}</td>
                  <td>{t(row.whenKey)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <CodeBlock code={SERVER_TIMING_EXAMPLE} label={t("timings.title")} />
      </Section>

      <Section {...TECH["websocket-akislari"]}>
        <p>{t("guide.tech.ws.p1")}</p>
        <p>{t("guide.tech.ws.p2")}</p>

        <h3 className={styles.panelTitle}>{t("guide.tech.ws.inkTitle")}</h3>
        <p>{t("guide.tech.ws.inkP")}</p>
        <CodeBlock code={INK_PROTOCOL} label="/ws/v1/ink" />

        <h3 className={styles.panelTitle}>{t("guide.tech.ws.audioTitle")}</h3>
        <p>{t("guide.tech.ws.audioP")}</p>
        <CodeBlock code={AUDIO_PROTOCOL} label="/ws/v1/audio" />

        <p>{t("guide.tech.ws.p3")}</p>
      </Section>

      <Section {...TECH["veri-modeli"]}>
        <p>{t("guide.tech.data.p1")}</p>
        <TermList rows={DATA_GROUPS} />
      </Section>

      <Section {...TECH["kitap-bilgi-grafigi"]}>
        <p>{t("guide.tech.book.p1")}</p>
        <p>{t("guide.tech.book.p2")}</p>
        <p>{t("guide.tech.book.p3")}</p>
      </Section>

      <Section {...TECH["guvenlik-ve-gizlilik"]}>
        <ul className={styles.list}>
          {SECURITY.map((key) => (
            <li key={key}>{t(key)}</li>
          ))}
        </ul>
      </Section>

      <Section {...TECH["onay-kapilari"]}>
        <p>{t("guide.tech.gates.p1")}</p>
        <div className={`${tables.wrap} scroll-x`}>
          <table className={tables.table} data-testid="guide-gate-table">
            <caption className={tables.caption}>{t("guide.tech.gates.heading")}</caption>
            <thead>
              <tr>
                <th scope="col">{t("guide.tech.gates.colGate")}</th>
                <th scope="col">{t("guide.tech.gates.colMeaning")}</th>
              </tr>
            </thead>
            <tbody>
              {GATES.map((row) => (
                <tr key={row.term}>
                  <th scope="row" className={tables.mono}>
                    {row.term}
                  </th>
                  <td>{t(row.descKey)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>{t("guide.tech.gates.p2")}</p>
        <p>{t("guide.tech.gates.p3")}</p>
      </Section>

      <Section {...TECH["degerlendirme"]}>
        <p>{t("guide.tech.eval.p1")}</p>
        <p>{t("guide.tech.eval.p2")}</p>
        <TermList rows={EVAL_METRICS} />
        <p>{t("guide.tech.eval.p3")}</p>
      </Section>
    </>
  );
}
