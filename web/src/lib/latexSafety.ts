/**
 * LaTeX sanitiser — browser port of `aicls.core.latex_safety` (ADR-0019, guvenlik-gizlilik.md §6).
 *
 * Every LaTeX string that reaches KaTeX (user input, OCR/ASR output, CAS results, book content)
 * goes through {@link sanitizeLatex} first. The policy is an allow-list of KaTeX math commands and
 * environments and it is deliberately identical to the server-side implementation so that the two
 * never disagree:
 *
 * - allowed commands are kept verbatim;
 * - forbidden commands (`\href`, `\url`, `\includegraphics`, `\def`, `\csname`, KaTeX HTML
 *   extensions, ...) are removed **together with their arguments**;
 * - any other command is removed but its brace groups stay (`\foo{x}` → `{x}`);
 * - unknown environments are unwrapped; stray `$`/`#`, control characters and a trailing backslash
 *   are dropped; braces are balanced.
 *
 * The function is idempotent and never returns a forbidden command. Client-side sanitising is a
 * defence-in-depth layer: the server sanitises too, but the client must never hand a hostile string
 * to `katex.renderToString` even when the payload came from a compromised or stubbed backend.
 */

/** Contract limit for LaTeX inputs (docs/guvenlik-gizlilik.md §6). */
export const MAX_LATEX_LENGTH = 2048;

const GREEK =
  "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu " +
  "xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega " +
  "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega";

const FUNCTIONS =
  "sin cos tan cot sec csc arcsin arccos arctan arccot sinh cosh tanh coth sech csch " +
  "log ln lg exp lim limsup liminf sup inf max min det dim gcd deg arg ker hom Pr " +
  "operatorname";

const STRUCTURE =
  "frac dfrac tfrac cfrac sqrt sum prod coprod int iint iiint oint binom dbinom tbinom " +
  "left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr " +
  "begin end substack overset underset stackrel overbrace underbrace overline underline " +
  "hat widehat bar vec tilde widetilde dot ddot dddot mathring overrightarrow overleftarrow " +
  "displaystyle textstyle scriptstyle scriptscriptstyle limits nolimits " +
  "mathrm mathbf mathit mathsf mathtt mathbb mathcal mathfrak mathscr boldsymbol bm " +
  "text textbf textit textrm textsf texttt textnormal " +
  "phantom hphantom vphantom smash not quad qquad space thinspace medspace thickspace " +
  "enspace negthinspace mathbin mathrel mathop mathord mathpunct mathopen mathclose " +
  "color textcolor";

const SYMBOLS =
  "cdot cdots ldots vdots ddots dots dotsb dotsc times div pm mp ast star circ bullet " +
  "infty partial nabla prime to rightarrow leftarrow Rightarrow Leftarrow leftrightarrow " +
  "Leftrightarrow longrightarrow longleftarrow Longrightarrow Longleftarrow " +
  "longleftrightarrow Longleftrightarrow mapsto longmapsto uparrow downarrow implies iff " +
  "impliedby leq le geq ge neq ne approx equiv sim simeq cong propto ll gg prec succ " +
  "subset subseteq supset supseteq subsetneq supsetneq in notin ni cup cap setminus " +
  "emptyset varnothing forall exists nexists neg lnot land lor wedge vee oplus otimes " +
  "ominus odot angle measuredangle perp parallel nparallel mid nmid langle rangle lfloor " +
  "rfloor lceil rceil vert Vert lvert rvert lVert rVert therefore because aleph hbar ell " +
  "Re Im wp top bot triangle square diamond degree checkmark dagger ddagger S P " +
  "lbrace rbrace lbrack rbrack backslash colon ldotp cdotp";

/** Commands kept verbatim (KaTeX math subset). */
export const ALLOWED_COMMANDS: ReadonlySet<string> = new Set(
  `${GREEK} ${FUNCTIONS} ${STRUCTURE} ${SYMBOLS}`.split(" ").filter(Boolean),
);

/** One-character control symbols kept verbatim (`\,` `\{` `\%` ...). */
export const ALLOWED_SYMBOL_COMMANDS: ReadonlySet<string> = new Set([
  "\\",
  ",",
  ";",
  ":",
  "!",
  " ",
  "{",
  "}",
  "%",
  "&",
  "#",
  "$",
  "_",
  "|",
]);

/** Environments kept; others are unwrapped. */
export const ALLOWED_ENVIRONMENTS: ReadonlySet<string> = new Set([
  "matrix",
  "pmatrix",
  "bmatrix",
  "Bmatrix",
  "vmatrix",
  "Vmatrix",
  "smallmatrix",
  "cases",
  "rcases",
  "dcases",
  "aligned",
  "gathered",
  "alignedat",
  "array",
  "darray",
  "split",
  "subarray",
]);

/** Commands removed together with their arguments (number of brace groups to drop). */
export const FORBIDDEN_COMMAND_ARGS: ReadonlyMap<string, number> = new Map<string, number>([
  // linking / resources
  ["href", 2],
  ["url", 1],
  ["includegraphics", 1],
  ["input", 1],
  ["include", 1],
  ["usepackage", 1],
  ["documentclass", 1],
  ["openin", 0],
  ["openout", 0],
  ["closeout", 0],
  ["closein", 0],
  ["read", 0],
  ["write", 1],
  ["immediate", 0],
  ["special", 1],
  ["pdfliteral", 1],
  ["shipout", 0],
  ["message", 1],
  ["errmessage", 1],
  ["scantokens", 1],
  ["jobname", 0],
  // macro machinery
  ["def", 1],
  ["gdef", 1],
  ["edef", 1],
  ["xdef", 1],
  ["let", 0],
  ["futurelet", 0],
  ["newcommand", 2],
  ["renewcommand", 2],
  ["providecommand", 2],
  ["DeclareMathOperator", 2],
  ["newenvironment", 3],
  ["renewenvironment", 3],
  ["csname", 0],
  ["endcsname", 0],
  ["expandafter", 0],
  ["noexpand", 0],
  ["relax", 0],
  ["catcode", 0],
  ["uccode", 0],
  ["lccode", 0],
  ["char", 0],
  ["chardef", 0],
  ["mathchardef", 0],
  ["countdef", 0],
  ["global", 0],
  ["long", 0],
  ["outer", 0],
  ["afterassignment", 0],
  ["aftergroup", 0],
  ["makeatletter", 0],
  ["makeatother", 0],
  ["meaning", 0],
  ["string", 0],
  // KaTeX HTML extensions (trust: false already blocks them; belt and braces)
  ["html", 0],
  ["htmlClass", 2],
  ["htmlId", 2],
  ["htmlStyle", 2],
  ["htmlData", 2],
  ["includegraphicsx", 1],
  ["rule", 2],
]);

/** Names of forbidden commands (never present in sanitised output). */
export const FORBIDDEN_COMMANDS: ReadonlySet<string> = new Set(FORBIDDEN_COMMAND_ARGS.keys());

/** Forbidden commands that may be followed by control sequences to consume (`\def\x`). */
const CONTROL_SEQ_BUDGET: ReadonlyMap<string, number> = new Map<string, number>([
  ["def", 1],
  ["gdef", 1],
  ["edef", 1],
  ["xdef", 1],
  ["let", 2],
  ["futurelet", 2],
  ["newcommand", 1],
  ["renewcommand", 1],
  ["providecommand", 1],
  ["DeclareMathOperator", 1],
  ["chardef", 1],
  ["mathchardef", 1],
  ["countdef", 1],
]);

const FILE_STREAM_COMMANDS = new Set(["write", "openout", "openin", "closeout", "closein", "read"]);
const ASSIGNMENT_COMMANDS = new Set(["openout", "openin", "catcode", "uccode", "lccode"]);

const CMD_RE = /\\([A-Za-z]+)/y;
const ENV_RE = /\s*\{([A-Za-z*]+)\}/y;
const CS_RE = /\\([A-Za-z]+|[\s\S])/y;
const DIGITS_RE = /\d+/y;
const OTHER_CATEGORY_RE = /^\p{C}$/u;
const LETTER_RE = /^\p{L}$/u;
const SPACE_RE = /^\s$/;
const ASCII_DIGIT_RE = /^[0-9]$/;

const DELIMITERS: ReadonlyArray<readonly [string, string]> = [
  ["$$", "$$"],
  ["\\[", "\\]"],
  ["\\(", "\\)"],
  ["$", "$"],
];

/** Thrown when `sanitizeLatex` is called with `maxLen` and the input is longer. */
export class LatexTooLongError extends Error {
  readonly maxLen: number;
  readonly length: number;

  constructor(maxLen: number, length: number) {
    super(`LaTeX input exceeds the ${String(maxLen)}-character limit (was ${String(length)}).`);
    this.name = "LatexTooLongError";
    this.maxLen = maxLen;
    this.length = length;
  }
}

function matchAt(re: RegExp, s: string, index: number): RegExpExecArray | null {
  re.lastIndex = index;
  return re.exec(s);
}

/** NFC-normalise, unify newlines and drop control/format characters (keep `\n`, `\t`). */
function cleanChars(s: string): string {
  const text = s.normalize("NFC").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  let out = "";
  for (const ch of text) {
    if (ch === "\n" || ch === "\t" || !OTHER_CATEGORY_RE.test(ch)) {
      out += ch;
    }
  }
  return out;
}

/** Strip surrounding whitespace but keep a trailing escaped space (`\ `). */
function trim(s: string): string {
  let out = s.replace(/^\s+/, "");
  while (out.length > 0 && SPACE_RE.test(out.slice(-1))) {
    let j = out.length - 2;
    let backslashes = 0;
    while (j >= 0 && out[j] === "\\") {
      backslashes += 1;
      j -= 1;
    }
    if (backslashes % 2 === 1) {
      break; // the space is the control symbol "\ "
    }
    out = out.slice(0, -1);
  }
  return out;
}

function stripDelimiters(s: string): string {
  for (const [start, end] of DELIMITERS) {
    if (
      s.length >= start.length + end.length &&
      s.startsWith(start) &&
      s.endsWith(end) &&
      !(start === end && s === start)
    ) {
      return trim(s.slice(start.length, s.length - end.length));
    }
  }
  return s;
}

function prepare(s: string): string {
  return stripDelimiters(trim(cleanChars(s)));
}

/**
 * Index just past the brace group opening at `start` (`s[start] === "{"`).
 *
 * Escaped braces do not count. An unterminated group extends to the end of the string.
 */
function findGroupEnd(s: string, start: number): number {
  let depth = 0;
  let i = start;
  const n = s.length;
  while (i < n) {
    const ch = s[i];
    if (ch === "\\") {
      i += 2;
      continue;
    }
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        return i + 1;
      }
    }
    i += 1;
  }
  return n;
}

/** Skip the arguments of forbidden command `name` starting at `i`; return the new index. */
function skipArguments(s: string, i: number, name: string): number {
  let groupsLeft = FORBIDDEN_COMMAND_ARGS.get(name) ?? 1;
  let csLeft = CONTROL_SEQ_BUDGET.get(name) ?? 0;
  const n = s.length;
  let pos = i;
  while (pos < n) {
    let j = pos;
    while (j < n && (s[j] === " " || s[j] === "\t")) {
      j += 1;
    }
    if (j >= n) {
      break;
    }
    const ch = s[j] as string;
    if (ch === "[") {
      const close = s.indexOf("]", j + 1);
      if (close === -1) {
        break;
      }
      pos = close + 1;
      continue;
    }
    if (ch === "{" && groupsLeft > 0) {
      pos = findGroupEnd(s, j);
      groupsLeft -= 1;
      continue;
    }
    if (ch === "\\" && csLeft > 0) {
      const m = matchAt(CS_RE, s, j);
      if (m === null) {
        break;
      }
      pos = j + m[0].length;
      csLeft -= 1;
      continue;
    }
    if (ch === "#" && CONTROL_SEQ_BUDGET.has(name)) {
      // macro parameter text (\def\cmd#1#2{...})
      pos = j + 1 < n && ASCII_DIGIT_RE.test(s[j + 1] as string) ? j + 2 : j + 1;
      continue;
    }
    if (ASCII_DIGIT_RE.test(ch) && FILE_STREAM_COMMANDS.has(name)) {
      const digits = matchAt(DIGITS_RE, s, j);
      pos = digits === null ? j + 1 : j + digits[0].length;
      continue;
    }
    if (ch === "=" && ASSIGNMENT_COMMANDS.has(name)) {
      pos = j + 1;
      continue;
    }
    break;
  }
  return pos;
}

/** Single pass over `s`: keep allowed tokens, drop or unwrap the rest, balance braces. */
function scan(s: string, collect?: string[]): string {
  const out: string[] = [];
  let depth = 0;
  let i = 0;
  const n = s.length;
  // True right after an alphabetic command was emitted: a following letter would otherwise fuse
  // with it into a new command name (\frac + a -> \fraca) once removed tokens vanish.
  let pendingCmd = false;
  while (i < n) {
    const ch = s[i] as string;
    if (ch === "\\") {
      const m = matchAt(CMD_RE, s, i);
      if (m !== null) {
        const name = m[1] as string;
        i += m[0].length;
        if (name === "begin" || name === "end") {
          const envMatch = matchAt(ENV_RE, s, i);
          if (envMatch !== null) {
            const env = envMatch[1] as string;
            i += envMatch[0].length;
            if (ALLOWED_ENVIRONMENTS.has(env)) {
              out.push(`\\${name}{${env}}`);
              pendingCmd = false;
            } else {
              collect?.push(`${name}{${env}}`);
            }
          } else {
            collect?.push(name);
          }
        } else if (ALLOWED_COMMANDS.has(name)) {
          out.push(`\\${name}`);
          pendingCmd = true;
          if (name === "operatorname" && i < n && s[i] === "*") {
            out.push("*");
            pendingCmd = false;
            i += 1;
          }
        } else if (FORBIDDEN_COMMANDS.has(name)) {
          collect?.push(name);
          i = skipArguments(s, i, name);
        } else {
          collect?.push(name);
        }
        continue;
      }
      if (i + 1 < n) {
        const sym = s[i + 1] as string;
        if (ALLOWED_SYMBOL_COMMANDS.has(sym)) {
          out.push(`\\${sym}`);
          pendingCmd = false;
        } else {
          collect?.push(sym);
        }
        i += 2;
        continue;
      }
      i += 1; // trailing lone backslash
      continue;
    }
    if (ch === "$" || ch === "#") {
      i += 1;
      continue;
    }
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      if (depth === 0) {
        i += 1;
        continue;
      }
      depth -= 1;
    }
    if (pendingCmd && LETTER_RE.test(ch)) {
      out.push(" ");
    }
    pendingCmd = false;
    out.push(ch);
    i += 1;
  }
  // Trim whitespace at the token level so an escaped space ("\ ") at either end survives.
  while (out.length > 0 && SPACE_RE.test(out[0] as string)) {
    out.shift();
  }
  while (out.length > 0 && SPACE_RE.test(out[out.length - 1] as string)) {
    out.pop();
  }
  return out.join("") + "}".repeat(depth);
}

/**
 * Return `s` reduced to the KaTeX allow-list (see the module docstring).
 *
 * @param s - Raw LaTeX (math mode; `$...$`, `$$...$$`, `\(...\)`, `\[...\]` wrappers are stripped).
 * @param options.maxLen - When given, reject inputs longer than this many characters.
 * @returns Sanitised LaTeX; empty string for empty/whitespace input.
 * @throws {LatexTooLongError} When `maxLen` is set and exceeded.
 */
export function sanitizeLatex(s: string, options: { maxLen?: number } = {}): string {
  const { maxLen } = options;
  if (maxLen !== undefined && s.length > maxLen) {
    throw new LatexTooLongError(maxLen, s.length);
  }
  return scan(prepare(s));
}

/**
 * List (in order, without duplicates) the command names that sanitising would remove.
 *
 * Used for diagnostics and for the "unsupported construct" hint under the command bar.
 */
export function findDisallowedCommands(s: string): string[] {
  const found: string[] = [];
  scan(prepare(s), found);
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const name of found) {
    if (!seen.has(name)) {
      seen.add(name);
      unique.push(name);
    }
  }
  return unique;
}
