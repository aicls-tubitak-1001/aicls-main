/**
 * `cx` — join class names, dropping anything falsy.
 *
 * CSS-module lookups are typed `string | undefined` under `noUncheckedIndexedAccess`, and a
 * conditional modifier is naturally `false`; without this helper every component would repeat the
 * same `.filter(Boolean).join(" ")` dance.
 */
export type ClassValue = string | false | null | undefined;

/** Join the truthy class names with a single space. */
export function cx(...values: ClassValue[]): string {
  return values
    .filter((value): value is string => typeof value === "string" && value !== "")
    .join(" ");
}
