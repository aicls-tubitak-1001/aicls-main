/**
 * A copyable sample: a Turkish command, an endpoint, a JSON fragment.
 *
 * Everything a reader is expected to *type* is rendered through this component, because reading a
 * command off a projector and retyping it is exactly the moment a demonstration goes wrong. The
 * copy button is a real button with a 44 px target, and the outcome is announced through a local
 * polite live region rather than a toast: the guide can be read while a lecture is running, and a
 * toast would compete with the workspace's own messages.
 *
 * The clipboard API is absent in a few browsers and in jsdom, and it rejects when the document is
 * not focused; both cases fall back to a visible "select it yourself" message instead of throwing.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { t } from "@/i18n";
import styles from "./CodeBlock.module.css";

export interface CodeBlockProps {
  /** The exact text a reader should get on the clipboard. */
  readonly code: string;
  /** Optional caption above the block (already-translated Turkish). */
  readonly label?: string;
  /** Render inside a quieter surface (used inside worked examples). */
  readonly compact?: boolean;
}

type CopyState = "idle" | "copied" | "failed";

/** A code sample with a copy button. */
export function CodeBlock({ code, label, compact = false }: CodeBlockProps): JSX.Element {
  const [state, setState] = useState<CopyState>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current !== null) {
        clearTimeout(timer.current);
      }
    },
    [],
  );

  const copy = useCallback(() => {
    const reset = (next: CopyState): void => {
      setState(next);
      if (timer.current !== null) {
        clearTimeout(timer.current);
      }
      timer.current = setTimeout(() => {
        setState("idle");
      }, 4000);
    };
    const clipboard = globalThis.navigator?.clipboard;
    if (clipboard === undefined) {
      reset("failed");
      return;
    }
    clipboard.writeText(code).then(
      () => {
        reset("copied");
      },
      () => {
        reset("failed");
      },
    );
  }, [code]);

  return (
    <div className={compact ? `${styles.block} ${styles.compact}` : styles.block}>
      {label === undefined ? null : <span className={styles.label}>{label}</span>}
      <div className={styles.row}>
        <pre className={`${styles.pre} scroll-x`}>
          <code>{code}</code>
        </pre>
        <Button
          className={styles.copy}
          variant="ghost"
          onClick={copy}
          aria-label={`${t("guide.copy")}: ${code}`}
        >
          {t("guide.copy")}
        </Button>
      </div>
      <p className={styles.status} role="status">
        {state === "copied" ? t("guide.copied") : null}
        {state === "failed" ? t("guide.copyFailed") : null}
      </p>
    </div>
  );
}
