/**
 * The two feedback endpoints of İP-9 (api.md §3.9).
 *
 * Both require a signed-in user (`CurrentUser` in `aicls.api.routers.feedback`), so both can fail
 * with an authentication problem; the screens render that as a calm sentence with a link to the
 * sign-in page and **keep the form filled in**, because a questionnaire that loses ten answers to
 * an expired token will not be filled in a second time.
 */
import { api, call, type ApiResult } from "@/api/client";
import type { FeedbackIn, FeedbackOut, SusOut } from "@/api/types";
import { t } from "@/i18n";

/** The ten answers of one SUS questionnaire, in question order. */
export type SusAnswers = number[];

/** Submit the questionnaire; the score is computed server-side (ADR-0015). */
export async function submitSus(
  answers: SusAnswers,
  cohort: string,
  collectedWeek?: number,
): Promise<ApiResult<SusOut>> {
  return await call(
    api.POST("/feedback/sus", {
      body: {
        answers,
        cohort,
        ...(collectedWeek === undefined ? {} : { collected_week: collectedWeek }),
      },
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/** Record one free-text feedback entry (state `open`). */
export async function submitFeedback(body: FeedbackIn): Promise<ApiResult<FeedbackOut>> {
  return await call(api.POST("/feedback", { body }), {
    fallbackDetailTr: t("errors.network"),
  });
}
