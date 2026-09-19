/**
 * Human-readable rendering of a typed CAS command.
 *
 * The confirmation step must show the user *exactly* what the engine will run, in a form a
 * first-year student can check at a glance: `diff(expr=x^{2}, var=x, order=1)`. Keeping it here
 * (rather than in the component file) lets the workspace, the history panel and the tests share
 * one formatting rule.
 *
 * The command travels as a free-form object (`ComputeLatexIn.command`; the backend validates it
 * against `mathengine.CasCommand`), so every value is formatted defensively — a command shape the
 * client has never seen must still render, not throw, in front of a class.
 */
import type { CommandDto } from "@/api/types";

/** Format one slot value: strings and numbers as they are, anything else as JSON. */
function formatValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value) ?? "";
}

/** Format a command object as a compact call: `op(slot=value, ...)`. */
export function formatCommand(command: CommandDto): string {
  const { op, ...rest } = command;
  const name = typeof op === "string" ? op : "?";
  const args = Object.entries(rest)
    .filter(([, value]) => value !== undefined && value !== null)
    .map(([key, value]) => `${key}=${formatValue(value)}`);
  return `${name}(${args.join(", ")})`;
}
