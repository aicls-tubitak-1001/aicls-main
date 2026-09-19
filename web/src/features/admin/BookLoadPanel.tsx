/**
 * Book ingest (`POST /admin/book/load`, İP-8).
 *
 * The interesting case is the refusal: the loader accepts a chapter set only when **every**
 * unit-looking heading was tagged, and lists the ones that were not
 * (`aicls.book.ingest.validators.UntaggedHeadingError`). That list is the whole value of the
 * error, so it is rendered as a list — one heading per line, selectable — rather than folded into
 * a sentence. A successful load reports the four figures the criterion is measured with.
 */
import { useRef, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import { Button, Card, Chip, ErrorState } from "@/components/ui";
import { t } from "@/i18n";
import { taggedRatioOf, untaggedHeadingsOf, uploadBookArchive, type LoadReport } from "./api";
import styles from "./Admin.module.css";

/** The book loader. */
export function BookLoadPanel(): JSX.Element {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [report, setReport] = useState<LoadReport | null>(null);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [missing, setMissing] = useState(false);
  const [busy, setBusy] = useState(false);

  const upload = async (): Promise<void> => {
    if (file === null) {
      setMissing(true);
      return;
    }
    setBusy(true);
    setProblem(null);
    setReport(null);
    const { data, problem: failure } = await uploadBookArchive(file);
    setBusy(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    setReport(data);
  };

  const untagged = untaggedHeadingsOf(problem);
  const ratio = taggedRatioOf(problem);

  return (
    <Card title={t("admin.bookTitle")}>
      <p className={styles.lead}>{t("admin.bookIntro")}</p>

      <div className={styles.filters}>
        <label className={styles.fileLabel} htmlFor="book-archive">
          {t("admin.bookFile")}
        </label>
        <input
          id="book-archive"
          ref={inputRef}
          className={styles.file}
          type="file"
          accept=".zip,application/zip"
          onChange={(event) => {
            setMissing(false);
            setFile(event.target.files?.[0] ?? null);
          }}
        />
        <Button
          variant="primary"
          loading={busy}
          onClick={() => {
            void upload();
          }}
        >
          {t("admin.bookUpload")}
        </Button>
      </div>

      {missing ? (
        <p className={styles.errorText} role="alert">
          {t("admin.bookNoFile")}
        </p>
      ) : null}

      {problem === null ? null : <ErrorState problem={problem} />}

      {untagged.length === 0 ? null : (
        <div className={styles.untagged} data-testid="untagged-headings">
          <h3 className={styles.subheading}>{t("admin.bookUntaggedTitle")}</h3>
          <p className={styles.lead}>{t("admin.bookUntaggedHint")}</p>
          {ratio === null ? null : (
            <Chip tone="warn">
              {t("admin.bookRatio")}:{" "}
              {t("common.percent", { value: Math.round(ratio * 1000) / 10 })}
            </Chip>
          )}
          <ul className={styles.headings}>
            {untagged.map((heading) => (
              <li key={heading}>{heading}</li>
            ))}
          </ul>
        </div>
      )}

      {report === null ? null : (
        <div className={styles.notice} role="status">
          <p>{t("admin.bookLoaded")}</p>
          <dl className={styles.meta}>
            <div className={styles.metaRow}>
              <dt>{t("admin.bookUnits")}</dt>
              <dd>{report.units}</dd>
            </div>
            <div className={styles.metaRow}>
              <dt>{t("admin.bookTriples")}</dt>
              <dd>{report.triples}</dd>
            </div>
            <div className={styles.metaRow}>
              <dt>{t("admin.bookAliases")}</dt>
              <dd>{report.aliases}</dd>
            </div>
            <div className={styles.metaRow}>
              <dt>{t("admin.bookRatio")}</dt>
              <dd>{t("common.percent", { value: Math.round(report.tagged_ratio * 1000) / 10 })}</dd>
            </div>
          </dl>
        </div>
      )}
    </Card>
  );
}
