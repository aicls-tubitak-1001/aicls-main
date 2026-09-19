/**
 * The "İlk beş dakika" card on the workspace.
 *
 * A guide nobody opens is not documentation, so the workspace carries a compact five-step card
 * that links into it. Three rules keep it out of the way of a lecture:
 *
 * - it never interrupts — no modal, no focus steal, no overlay over the result area;
 * - dismissing it is undoable, through the toast the design system already uses for undo;
 * - the dismissal is remembered, so the card appears once per browser and not once per reload.
 *
 * It is rendered by the workspace but lives in the guide feature, because everything it says and
 * every link it holds belongs to the guide.
 */
import { useCallback, useState } from "react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/toastContext";
import { t, type MessageKey } from "@/i18n";
import { isFirstRunVisible, setFirstRunVisible } from "./firstRun";
import { GUIDE_ROUTES } from "./routes";
import styles from "./FirstRunCard.module.css";

const STEP_KEYS: readonly MessageKey[] = [
  "guide.firstRun.step1",
  "guide.firstRun.step2",
  "guide.firstRun.step3",
  "guide.firstRun.step4",
  "guide.firstRun.step5",
];

/** The dismissible first-run card. */
export function FirstRunCard(): JSX.Element | null {
  const [visible, setVisible] = useState<boolean>(isFirstRunVisible);
  const toast = useToast();

  const restore = useCallback(() => {
    setFirstRunVisible(true);
    setVisible(true);
  }, []);

  const dismiss = useCallback(() => {
    setFirstRunVisible(false);
    setVisible(false);
    toast.show(t("guide.firstRun.dismissed"), {
      tone: "info",
      action: { label: t("guide.firstRun.restore"), onAction: restore },
    });
  }, [restore, toast]);

  if (!visible) {
    return null;
  }

  return (
    <section
      className={`${styles.card} chrome-secondary`}
      aria-labelledby="ilk-bes-dakika"
      data-testid="first-run-card"
    >
      <div className={styles.header}>
        <h2 id="ilk-bes-dakika" className={styles.title}>
          {t("guide.firstRun.title")}
        </h2>
        <Button variant="ghost" onClick={dismiss}>
          {t("guide.firstRun.dismiss")}
        </Button>
      </div>

      <p className={styles.intro}>{t("guide.firstRun.intro")}</p>

      <ol className={styles.steps}>
        {STEP_KEYS.map((key) => (
          <li key={key}>{t(key)}</li>
        ))}
      </ol>

      <Link className={styles.link} to={GUIDE_ROUTES.usage}>
        {t("guide.firstRun.open")}
      </Link>
    </section>
  );
}
