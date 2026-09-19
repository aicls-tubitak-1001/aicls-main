/**
 * Withdrawal (KVKK "geri çekme"), explained before it is offered.
 *
 * The dialog exists because withdrawal is the one place in this tab where a volunteer needs to
 * know exactly what happens *before* they press anything: the audio is deleted from the server and
 * removed from the training set, while the pseudonymous record that consent was once given stays,
 * because it is the legal proof of that consent and contains no identity
 * (`aicls.api.routers.contribute.withdraw_consent`). Both sentences are on screen, in that order,
 * with the reassurance that consent can be given again later.
 *
 * It is a modal only because it is opened by a deliberate press; it never appears on its own.
 */
import { useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import { Button, Dialog, ErrorState } from "@/components/ui";
import { t } from "@/i18n";
import { withdrawConsent } from "./api";
import { clearParticipant } from "./participant";
import styles from "./WithdrawDialog.module.css";

export interface WithdrawDialogProps {
  open: boolean;
  /** Close without doing anything. */
  onClose: () => void;
  /** Participant token of this tab, or `null` for a signed-in contributor. */
  token: string | null;
  /** Called after the server confirmed the withdrawal (204). */
  onWithdrawn: () => void;
}

/** The withdrawal confirmation. */
export function WithdrawDialog({
  open,
  onClose,
  token,
  onWithdrawn,
}: WithdrawDialogProps): JSX.Element | null {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);

  const confirm = async (): Promise<void> => {
    setBusy(true);
    setProblem(null);
    const { problem: failure } = await withdrawConsent(token);
    setBusy(false);
    if (failure !== null) {
      setProblem(failure);
      return;
    }
    clearParticipant();
    onWithdrawn();
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={t("contribute.withdrawTitle")}
      description={t("contribute.withdrawExplain")}
      footer={
        <div className={styles.footer}>
          <Button variant="secondary" onClick={onClose} disabled={busy}>
            {t("common.cancel")}
          </Button>
          <Button
            variant="danger"
            loading={busy}
            onClick={() => {
              void confirm();
            }}
          >
            {t("contribute.withdrawConfirm")}
          </Button>
        </div>
      }
    >
      <ul className={styles.points}>
        <li>{t("contribute.withdrawKeeps")}</li>
        <li>{t("contribute.withdrawReversible")}</li>
      </ul>
      {problem === null ? null : <ErrorState problem={problem} inline />}
    </Dialog>
  );
}
