/**
 * The two states a panel spends most of its life in: {@link EmptyState} and {@link ErrorState},
 * plus {@link SkipLink}.
 *
 * `ErrorState` is the visible half of the RFC 9457 contract. It shows the Turkish `detail_tr` in
 * full size, the stable error code and the request id in small print (so a lecturer can quote one
 * line to support), the parse position when the backend located one, and a single retry action.
 * It announces itself through `role="alert"` — but it never navigates, never opens a modal and
 * never removes what is already on screen.
 */
import type { ReactNode } from "react";

import type { AiclsProblem } from "@/api/problem";
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import { Button } from "./Button";
import styles from "./States.module.css";

export interface EmptyStateProps {
  /** Short Turkish sentence describing what would normally be here. */
  title: string;
  /** One more sentence telling the user how to fill it. */
  description?: string;
  /** A single call to action. */
  action?: ReactNode;
  /** Decorative glyph; hidden from assistive technology. */
  icon?: ReactNode;
  className?: string;
}

/** What a panel shows before it has anything to show. */
export function EmptyState({
  title,
  description,
  action,
  icon,
  className,
}: EmptyStateProps): JSX.Element {
  return (
    <div className={cx(styles.state, className)}>
      {icon === undefined ? null : (
        <span className={styles.icon} aria-hidden="true">
          {icon}
        </span>
      )}
      <p className={styles.title}>{title}</p>
      {description === undefined ? null : <p className={styles.description}>{description}</p>}
      {action === undefined ? null : <div className={styles.actions}>{action}</div>}
    </div>
  );
}

export interface ErrorStateProps {
  /** The parsed problem; its `detail_tr` is the message shown. */
  problem: AiclsProblem | null;
  /** Shown when `problem` is `null` (a failure the caller detected itself). */
  message?: string;
  /** Retry handler; omitted when retrying makes no sense. */
  onRetry?: () => void;
  /** Label of the retry button; defaults to the Turkish "Yeniden dene". */
  retryLabel?: string;
  /** Quieter presentation for an inline error under a field. */
  inline?: boolean;
  className?: string;
}

/** A calm, quotable failure. */
export function ErrorState({
  problem,
  message,
  onRetry,
  retryLabel,
  inline = false,
  className,
}: ErrorStateProps): JSX.Element {
  const detail = problem?.detail_tr ?? message ?? t("errors.unknown");
  const code = problem?.code;
  const requestId = problem?.instance;
  const position = problem?.position;
  return (
    <div
      className={cx(styles.state, styles.error, inline && styles.inline, className)}
      role="alert"
    >
      <p className={styles.title}>{detail}</p>
      {position === undefined ? null : (
        <p className={styles.description}>{t("errors.position", { position })}</p>
      )}
      {problem?.suggestions_tr === undefined || problem.suggestions_tr.length === 0 ? null : (
        <p className={styles.description}>
          {t("errors.suggestions", { list: problem.suggestions_tr.join(", ") })}
        </p>
      )}
      {onRetry === undefined ? null : (
        <div className={styles.actions}>
          <Button variant="secondary" onClick={onRetry}>
            {retryLabel ?? t("common.retry")}
          </Button>
        </div>
      )}
      {code === undefined && (requestId === undefined || requestId === null) ? null : (
        <p className={styles.meta}>
          {code === undefined ? null : <span className={styles.code}>{code}</span>}
          {requestId === undefined || requestId === null ? null : (
            <span className={styles.requestId}>{t("errors.requestId", { id: requestId })}</span>
          )}
        </p>
      )}
    </div>
  );
}

export interface SkipLinkProps {
  /** Id of the landmark to jump to; `main` by default. */
  targetId?: string;
  children?: ReactNode;
}

/** The first tab stop of every page: jump straight to the main landmark. */
export function SkipLink({ targetId = "main", children }: SkipLinkProps): JSX.Element {
  return (
    <a className="skip-link" href={`#${targetId}`}>
      {children ?? t("app.skipToContent")}
    </a>
  );
}
