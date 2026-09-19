/**
 * The textbook panel (İP-8).
 *
 * Three ways in, all of them the backend's:
 *
 * 1. `GET /book/search` — free search over unit titles and aliases (alias → full text → fuzzy).
 * 2. `POST /book/resolve` — the same box, second try: when a search finds nothing the text is
 *    treated as a Turkish *fetch command* ("türev tanımını getir"), which is what the NLU would
 *    have done with it. The lecturer types one thing and the platform tries both readings.
 * 3. Units handed in by the workspace, when a command already resolved to `kind: "book"`.
 *
 * A unit can be pinned beside the result so a definition stays on the board while the class works
 * through an example, and `GET /book/units/{id}/related` walks one hop of the graph without leaving
 * the panel.
 */
import { useEffect, useState } from "react";

import { api, call } from "@/api/client";
import type { ContentUnitOut, HitOut } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import { BookUnitView } from "./BookUnitView";
import styles from "./BookPanel.module.css";

/** Hits requested per search (the endpoint allows up to 20). */
const SEARCH_LIMIT = 5;

export interface BookPanelProps {
  /** Units the workspace already resolved (e.g. from a `kind: "book"` outcome). */
  hits?: readonly HitOut[];
  /** Hide the search box when the panel is driven entirely by commands. */
  searchable?: boolean;
  /** Pin a unit beside the result. */
  onPin?: (unit: ContentUnitOut) => void;
  /** Id of the currently pinned unit, if any. */
  pinnedId?: string | null;
  /** Close the panel (rendered only when given). */
  onClose?: () => void;
}

/** Wrap plain units in the hit envelope the list renders. */
function asHits(units: readonly ContentUnitOut[], stage: string): HitOut[] {
  return units.map((unit) => ({ unit, score: 1, stage, matched_alias: null }));
}

/** Search box, results and the related-units walk. */
export function BookPanel({
  hits = [],
  searchable = true,
  onPin,
  pinnedId = null,
  onClose,
}: BookPanelProps): JSX.Element {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<readonly HitOut[]>(hits);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);

  // Units handed in by a command replace whatever the box had found: the lecture just moved on.
  useEffect(() => {
    if (hits.length > 0) {
      setResults(hits);
      setSearched(true);
    }
  }, [hits]);

  const search = async (): Promise<void> => {
    const trimmed = query.trim();
    if (trimmed === "") {
      return;
    }
    setLoading(true);
    setErrorText(null);
    const found = await call(
      api.GET("/book/search", { params: { query: { q: trimmed, limit: SEARCH_LIMIT } } }),
      { fallbackDetailTr: t("errors.network") },
    );
    if (found.data !== null && found.data.length > 0) {
      setLoading(false);
      setSearched(true);
      setResults(found.data);
      return;
    }
    // Nothing matched as a search phrase: try it as the fetch command it may have been.
    const resolved = await call(api.POST("/book/resolve", { body: { text: trimmed } }), {
      fallbackDetailTr: t("errors.network"),
    });
    setLoading(false);
    setSearched(true);
    setResults(resolved.data ?? []);
    if (found.data === null && resolved.data === null) {
      setErrorText(found.problem?.detail_tr ?? t("errors.network"));
    }
  };

  const showRelated = async (unit: ContentUnitOut): Promise<void> => {
    setLoading(true);
    const { data } = await call(
      api.GET("/book/units/{unit_id}/related", { params: { path: { unit_id: unit.id } } }),
      { fallbackDetailTr: t("errors.network") },
    );
    setLoading(false);
    setSearched(true);
    if (data !== null && data.length > 0) {
      setResults(asHits(data, "related"));
    }
  };

  const shown = results.length > 0 ? results : hits;

  return (
    <Card
      title={t("book.title")}
      className={cx(styles.card)}
      actions={
        onClose === undefined ? undefined : (
          <Button variant="ghost" onClick={onClose} aria-label={t("book.close")}>
            <span aria-hidden="true">×</span>
          </Button>
        )
      }
    >
      <div className={styles.panel} data-testid="book-panel">
        {searchable ? (
          <>
            <div className={styles.searchRow}>
              <label className="visually-hidden" htmlFor="book-search">
                {t("book.searchPlaceholder")}
              </label>
              <input
                id="book-search"
                className={styles.input}
                type="search"
                value={query}
                placeholder={t("book.searchPlaceholder")}
                onChange={(event) => {
                  setQuery(event.target.value);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void search();
                  }
                }}
              />
              <Button variant="primary" loading={loading} onClick={() => void search()}>
                {t("book.search")}
              </Button>
            </div>
            <p className={cx(styles.hint, "chrome-secondary")}>{t("book.searchHint")}</p>
          </>
        ) : null}

        {errorText === null ? null : (
          <p className={styles.error} role="status">
            {errorText}
          </p>
        )}

        {shown.length === 0 ? (
          <p className={styles.empty}>
            {loading ? t("book.searching") : searched ? t("book.empty") : ""}
          </p>
        ) : (
          <ul className={styles.results}>
            {shown.map((hit) => (
              <li key={hit.unit.id}>
                <BookUnitView
                  unit={hit.unit}
                  compact
                  onShowRelated={(unit) => {
                    void showRelated(unit);
                  }}
                  {...(onPin === undefined || pinnedId === hit.unit.id ? {} : { onPin })}
                />
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}
