/**
 * Informed-consent step of the voice-collection tab (KVKK, docs/guvenlik-gizlilik.md §2).
 *
 * Nothing may be recorded before this form has produced a speaker, which is the client half of the
 * consent gate the backend enforces on every collection endpoint. The screen therefore does three
 * things and no more: it shows the versioned consent text in full (never a summary, never a
 * scroll-jacked box the reader cannot leave), it collects **band-only** demographics — the exact
 * enumerations `DemographicsIn` allows, because anything narrower would identify a volunteer — and
 * it keeps the accept button disabled until the checkbox is ticked.
 *
 * The text's `version` and `sha256` are displayed and the accepted `version` is sent back verbatim,
 * so an acceptance can later be tied to the exact wording that was on screen.
 */
import { useState, type FormEvent } from "react";

import type { AiclsProblem } from "@/api/problem";
import type { ConsentTextOut, DemographicsIn } from "@/api/types";
import { Button, Card, ErrorState, Field, Input, Select } from "@/components/ui";
import { t, type MessageKey } from "@/i18n";
import { submitConsent } from "./api";
import { setParticipant, type Participant } from "./participant";
import styles from "./ConsentForm.module.css";

/** The four age bands of `DemographicsIn.age_band`. */
const AGE_BANDS: DemographicsIn["age_band"][] = ["18-24", "25-34", "35-44", "45+"];

/** The three gender values, with their Turkish labels. */
const GENDERS: { value: NonNullable<DemographicsIn["gender"]>; label: MessageKey }[] = [
  { value: "f", label: "contribute.gender.f" },
  { value: "m", label: "contribute.gender.m" },
  { value: "na", label: "contribute.gender.na" },
];

/** The four universities of the pilot (`DemographicsIn.university`). */
const UNIVERSITIES: { value: NonNullable<DemographicsIn["university"]>; label: MessageKey }[] = [
  { value: "mus", label: "contribute.university.mus" },
  { value: "mersin", label: "contribute.university.mersin" },
  { value: "amasya", label: "contribute.university.amasya" },
  { value: "other", label: "contribute.university.other" },
];

export interface ConsentFormProps {
  /** The consent text to display, as `GET /contribute/consent/current` returned it. */
  consent: ConsentTextOut;
  /** Called with the participant this acceptance created. */
  onConsented: (participant: Participant) => void;
}

/** The consent screen. */
export function ConsentForm({ consent, onConsented }: ConsentFormProps): JSX.Element {
  const [accepted, setAccepted] = useState(false);
  const [ageBand, setAgeBand] = useState<DemographicsIn["age_band"]>("25-34");
  const [gender, setGender] = useState<DemographicsIn["gender"]>("na");
  const [university, setUniversity] = useState<DemographicsIn["university"]>("other");
  const [accentRegion, setAccentRegion] = useState("");
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!accepted || busy) {
      return;
    }
    setBusy(true);
    setProblem(null);
    const trimmed = accentRegion.trim();
    const { data, problem: failure } = await submitConsent(consent.version, {
      age_band: ageBand,
      gender,
      university,
      ...(trimmed === "" ? {} : { accent_region: trimmed }),
    });
    setBusy(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    const participant: Participant = {
      pseudonym: data.speaker_pseudonym,
      token: data.token,
      consentVersion: consent.version,
    };
    setParticipant(participant);
    onConsented(participant);
  };

  return (
    <form
      className={styles.form}
      onSubmit={(event) => {
        void submit(event);
      }}
    >
      <Card title={t("contribute.consentTitle")}>
        <p className={styles.intro}>{t("contribute.consentIntro")}</p>

        {/* A scrollable region needs to be reachable by keyboard, so it is focusable and named. */}
        <div
          className={styles.consentText}
          data-testid="consent-text"
          role="region"
          aria-label={t("contribute.consentReadArea")}
          tabIndex={0}
        >
          {consent.text_tr}
        </div>

        <dl className={styles.meta}>
          <div className={styles.metaRow}>
            <dt>{t("contribute.consentVersionLabel")}</dt>
            <dd>{consent.version}</dd>
          </div>
          {consent.effective_from === "" ? null : (
            <div className={styles.metaRow}>
              <dt>{t("contribute.consentEffectiveLabel")}</dt>
              <dd>{consent.effective_from}</dd>
            </div>
          )}
          <div className={styles.metaRow}>
            <dt>{t("contribute.consentHashLabel")}</dt>
            <dd className={styles.hash}>{consent.sha256}</dd>
          </div>
        </dl>
      </Card>

      <Card title={t("contribute.demographics")}>
        <p className={styles.intro}>{t("contribute.demographicsHint")}</p>
        <div className={styles.grid}>
          <Field label={t("contribute.ageBand")}>
            {(control) => (
              <Select
                {...control}
                value={ageBand}
                onChange={(event) => {
                  setAgeBand(event.target.value as DemographicsIn["age_band"]);
                }}
              >
                {AGE_BANDS.map((band) => (
                  <option key={band} value={band}>
                    {band}
                  </option>
                ))}
              </Select>
            )}
          </Field>

          <Field label={t("contribute.gender")}>
            {(control) => (
              <Select
                {...control}
                value={gender}
                onChange={(event) => {
                  setGender(event.target.value as DemographicsIn["gender"]);
                }}
              >
                {GENDERS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {t(option.label)}
                  </option>
                ))}
              </Select>
            )}
          </Field>

          <Field label={t("contribute.university")}>
            {(control) => (
              <Select
                {...control}
                value={university}
                onChange={(event) => {
                  setUniversity(event.target.value as DemographicsIn["university"]);
                }}
              >
                {UNIVERSITIES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {t(option.label)}
                  </option>
                ))}
              </Select>
            )}
          </Field>

          <Field label={t("contribute.accentRegion")} hint={t("contribute.accentRegionHint")}>
            {(control) => (
              <Input
                {...control}
                type="text"
                maxLength={40}
                value={accentRegion}
                onChange={(event) => {
                  setAccentRegion(event.target.value);
                }}
              />
            )}
          </Field>
        </div>
      </Card>

      <Card>
        <label className={styles.accept}>
          <input
            type="checkbox"
            checked={accepted}
            onChange={(event) => {
              setAccepted(event.target.checked);
            }}
          />
          <span>{t("contribute.consentAccept")}</span>
        </label>

        {problem === null ? null : <ErrorState problem={problem} inline />}

        <div className={styles.actions}>
          <Button type="submit" variant="primary" size="lg" disabled={!accepted} loading={busy}>
            {t("contribute.consentSubmit")}
          </Button>
          {accepted ? null : <p className={styles.hint}>{t("contribute.consentRequired")}</p>}
        </div>
      </Card>
    </form>
  );
}
