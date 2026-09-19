/**
 * The feedback route: the SUS questionnaire and the free-text form (İP-9).
 *
 * Both live on one page because they are filled in at the same moment of a pilot session — the
 * questionnaire produces the contractual number, the text box captures what the number cannot say.
 * Neither is gated behind a sign-in *screen*: the forms render for everyone, and an unauthenticated
 * submission comes back as a calm Turkish problem with a link to the sign-in page, so nothing that
 * was typed is thrown away by a redirect.
 */
import { useLocation } from "react-router-dom";

import { t } from "@/i18n";
import { FeedbackForm } from "./FeedbackForm";
import { SusForm } from "./SusForm";
import styles from "./FeedbackPage.module.css";

/** Default cohort label until the pilot study assigns one. */
const DEFAULT_COHORT = "pilot";

/** The feedback route. */
export function FeedbackPage(): JSX.Element {
  const location = useLocation();

  return (
    <section className={styles.page}>
      <header className={styles.intro}>
        <h1 className={styles.title}>{t("feedback.title")}</h1>
        <p className={styles.lead}>{t("sus.subtitle")}</p>
      </header>

      <SusForm cohort={DEFAULT_COHORT} />
      <FeedbackForm page={location.pathname} />
    </section>
  );
}
