/**
 * The lecture's own trail: every command of this workspace, newest first.
 *
 * Rows come from `GET /workspaces/{ws_id}/operations` with cursor pagination (api.md §3.3); the
 * endpoint is owner-scoped in the repository layer, so the client never sends a user id and never
 * needs to filter.
 *
 * Two behaviours are what make it usable during a lecture rather than after one:
 *
 * - **One tap re-runs a command.** The stored Turkish sentence (or the stored LaTeX) goes back
 *   through the same pipeline, so the answer is recomputed by SymPy instead of being replayed from
 *   a cache the lecturer cannot see.
 * - **Delete is never final on the first tap.** A removed row stays in place for
 *   {@link UNDO_WINDOW_MS} as a quiet "silindi — geri al" strip, and only then does the
 *   `DELETE /workspaces/{ws_id}/operations/{op_id}` request leave. Nothing destructive happens in
 *   front of a class without a way back.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { api, call } from "@/api/client";
import type { OperationOut } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import { MathView } from "@/features/results/MathView";
import styles from "./HistoryPanel.module.css";

/** How long a deleted row can still be brought back. */
export const UNDO_WINDOW_MS = 8000;

/** Rows fetched per page (the endpoint allows up to 200). */
const PAGE_SIZE = 50;

const KIND_LABELS: Record<OperationOut["input_kind"], MessageKey> = {
  text: "history.inputKind.text",
  latex: "history.inputKind.latex",
  ink: "history.inputKind.ink",
  audio: "history.inputKind.audio",
};

export interface HistoryPanelProps {
  /** Workspace whose operations to list; `null` shows the empty state. */
  workspaceId: string | null;
  /** Called when the user wants to run a stored operation again. */
  onReuse?: (operation: OperationOut) => void;
  /** Change this to force a reload (the workspace bumps it after every new operation). */
  refreshToken?: number;
}

/** Local time of an ISO timestamp, or the raw string when it cannot be parsed. */
function formatTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleTimeString("tr-TR");
}

/** The history list with paging, one-tap re-run and undoable delete. */
export function HistoryPanel({
  workspaceId,
  onReuse,
  refreshToken = 0,
}: HistoryPanelProps): JSX.Element {
  const [items, setItems] = useState<OperationOut[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState<readonly string[]>([]);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());

  const load = useCallback(
    async (next: string | null): Promise<void> => {
      if (workspaceId === null) {
        setItems([]);
        setCursor(null);
        return;
      }
      setLoading(true);
      const { data } = await call(
        api.GET("/workspaces/{ws_id}/operations", {
          params: {
            path: { ws_id: workspaceId },
            query: next === null ? { limit: PAGE_SIZE } : { limit: PAGE_SIZE, cursor: next },
          },
        }),
        { fallbackDetailTr: t("errors.network") },
      );
      setLoading(false);
      if (data === null) {
        return;
      }
      setItems((previous) => (next === null ? data.items : [...previous, ...data.items]));
      setCursor(data.next_cursor ?? null);
    },
    [workspaceId],
  );

  useEffect(() => {
    setItems([]);
    setCursor(null);
    void load(null);
  }, [load, refreshToken]);

  /** Send the delete the undo window has run out on. */
  const commitDelete = useCallback(
    async (operationId: string): Promise<void> => {
      timers.current.delete(operationId);
      if (workspaceId === null) {
        return;
      }
      const { problem } = await call(
        api.DELETE("/workspaces/{ws_id}/operations/{op_id}", {
          params: { path: { ws_id: workspaceId, op_id: operationId } },
        }),
        { fallbackDetailTr: t("history.deleteFailed") },
      );
      setPending((current) => current.filter((id) => id !== operationId));
      if (problem !== null) {
        // The row survived on the server: put it back rather than lying about what happened.
        return;
      }
      setItems((current) => current.filter((item) => item.id !== operationId));
    },
    [workspaceId],
  );

  const scheduleDelete = useCallback(
    (operationId: string): void => {
      setPending((current) =>
        current.includes(operationId) ? current : [...current, operationId],
      );
      timers.current.set(
        operationId,
        setTimeout(() => {
          void commitDelete(operationId);
        }, UNDO_WINDOW_MS),
      );
    },
    [commitDelete],
  );

  const undoDelete = useCallback((operationId: string): void => {
    const timer = timers.current.get(operationId);
    if (timer !== undefined) {
      clearTimeout(timer);
      timers.current.delete(operationId);
    }
    setPending((current) => current.filter((id) => id !== operationId));
  }, []);

  useEffect(() => {
    const scheduled = timers.current;
    return () => {
      // Leaving the screen is not "undo": what the lecturer asked for still happens, immediately.
      for (const [operationId, timer] of scheduled) {
        clearTimeout(timer);
        void commitDelete(operationId);
      }
      scheduled.clear();
    };
  }, [commitDelete]);

  const empty = items.length === 0;

  return (
    <Card title={t("history.title")} muted className={cx(styles.card)}>
      {empty ? (
        <p className={styles.empty}>{loading ? t("history.loading") : t("history.empty")}</p>
      ) : (
        <ul className={styles.list}>
          {items.map((operation) => {
            const removing = pending.includes(operation.id);
            return (
              <li
                key={operation.id}
                className={cx(styles.item, removing && styles.removing)}
                data-testid="history-item"
              >
                {removing ? (
                  <div className={styles.undoRow} role="status">
                    <span>{t("history.deleted")}</span>
                    <Button
                      variant="secondary"
                      onClick={() => {
                        undoDelete(operation.id);
                      }}
                    >
                      {t("toast.undo")}
                    </Button>
                  </div>
                ) : (
                  <>
                    <div className={styles.head}>
                      <Chip tone="neutral">{t(KIND_LABELS[operation.input_kind])}</Chip>
                      <span className={styles.time}>{formatTime(operation.created_at)}</span>
                      {operation.error_code === null ||
                      operation.error_code === undefined ? null : (
                        <Chip tone="warn">{operation.error_code}</Chip>
                      )}
                    </div>
                    <span className={styles.summary}>{operation.input_summary}</span>
                    {operation.result_latex === null ||
                    operation.result_latex === undefined ? null : (
                      <span className={styles.result}>
                        <MathView latex={operation.result_latex} />
                      </span>
                    )}
                    <div className={styles.actions}>
                      <Button
                        variant="secondary"
                        onClick={() => {
                          onReuse?.(operation);
                        }}
                      >
                        {t("history.reuse")}
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => {
                          scheduleDelete(operation.id);
                        }}
                      >
                        {t("history.delete")}
                      </Button>
                    </div>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {cursor === null ? null : (
        <Button
          variant="secondary"
          disabled={loading}
          onClick={() => {
            void load(cursor);
          }}
        >
          {t("history.loadMore")}
        </Button>
      )}
    </Card>
  );
}
