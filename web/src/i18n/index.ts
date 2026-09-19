/**
 * Minimal Turkish-only i18n layer.
 *
 * The platform ships a single locale (Turkish); an i18n library would only add weight. All UI
 * strings live in `tr.json` with **English keys and Turkish values**, mirroring the backend rule
 * that Turkish user-facing text lives in exactly one place (`aicls.core.messages_tr`). Keys are
 * type-checked: `t("nav.workspace")` compiles, `t("nav.workspce")` does not.
 */
import tr from "./tr.json";

/** Every translation key present in `tr.json`. */
export type MessageKey = keyof typeof tr;

/** Values substituted into `{{placeholder}}` slots. */
export type MessageParams = Readonly<Record<string, string | number>>;

const PLACEHOLDER_RE = /\{\{(\w+)\}\}/g;

/**
 * Look up a Turkish UI string and interpolate `{{name}}` placeholders.
 *
 * @param key - Key from `tr.json`.
 * @param params - Values for the placeholders in the template.
 * @returns The Turkish string; unknown placeholders are left untouched.
 */
export function t(key: MessageKey, params?: MessageParams): string {
  const template: string = tr[key];
  if (params === undefined) {
    return template;
  }
  return template.replace(PLACEHOLDER_RE, (match, name: string) => {
    const value = params[name];
    return value === undefined ? match : String(value);
  });
}

/** The raw message catalogue (used by tests and by the i18n completeness check). */
export const messages: Readonly<Record<string, string>> = tr;
