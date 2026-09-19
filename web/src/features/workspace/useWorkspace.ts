/**
 * The workspace controller: one place where a lecture's commands are actually run.
 *
 * Every modality ends here. Typed Turkish goes straight to `POST /compute/text`; a voice command
 * arrives as a finished `ComputeOut` from `POST /asr/command` and is *applied* rather than re-run;
 * handwriting contributes `context.last_latex` so "türevini al" means the expression on the board.
 * Keeping the three paths in one hook is what makes the rules below true everywhere at once:
 *
 * - **The last answer stays on screen.** A new command sets `busy`; it never clears `outcome`.
 *   A failure sets `problem` and still leaves the previous result visible, because a lecturer who
 *   is pointing at an answer must not lose it to a network hiccup.
 * - **Latency is measured to the render.** `markSubmit` runs here, before the request leaves, under
 *   a provisional key; `ResultCard` marks the render and posts the pair with the server's real
 *   `operation_id` (ADR-0014, api.md §3.10). The contractual metric is command → *displayed*.
 * - **Nothing is invented.** Only fields of `ComputeOut` are read, and the four `kind` values of
 *   api.md §3.4 (`result`, `clarification`, `book`, `ui`) are all handled.
 *
 * Shared state lives in `app/store.ts` so the shell and the panels observe the same values; the
 * transient parts of one answer (the parsed problem, the `Server-Timing` breakdown, the latency
 * key) stay local to the hook.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { api, call, type AiclsProblem } from "@/api/client";
import type { ComputeOut, Modality } from "@/api/types";
import { useSession } from "@/app/auth";
import { appStore, useAppState } from "@/app/store";
import { submitTextCommand } from "@/features/command/submitCommand";
import { t } from "@/i18n";
import type { StageTiming } from "@/lib/serverTiming";

/** A client-side action the backend asked the interface to perform (`nlu.UiActionName`). */
export type UiActionName = "clear" | "undo" | "repeat_last" | "export";

/** Options accepted by {@link WorkspaceApi.run}. */
export interface RunCommandOptions {
  /** Intent picked from a clarification, echoed back as `confirm_intent` (api.md §3.4). */
  confirmIntent?: string;
  /** How the command was entered; drives the telemetry modality and the confirmation policy. */
  modality?: Modality;
  /** Override the board expression sent as `context.last_latex` for this one call. */
  lastLatex?: string;
}

/** What {@link useWorkspace} hands the screen. */
export interface WorkspaceApi {
  /** Active workspace, or `null` while anonymous or still being created. */
  workspaceId: string | null;
  /** True once the workspace bootstrap finished (successfully or not). */
  ready: boolean;
  /** The last outcome that was committed to the board. */
  outcome: ComputeOut | null;
  /** The last failure, already Turkish; the previous outcome stays on screen next to it. */
  problem: AiclsProblem | null;
  /** `Server-Timing` breakdown of the request that produced {@link WorkspaceApi.outcome}. */
  stages: StageTiming[];
  /** True while a command is in flight. */
  busy: boolean;
  /** Provisional latency key of the outcome on screen (see the module docstring). */
  latencyKey: string | null;
  /** Text currently in the command bar. */
  commandText: string;
  /** Replace the command bar text. */
  setCommandText: (value: string) => void;
  /** The expression on the board (`context.last_latex`). */
  inputLatex: string;
  /** Replace the board expression. */
  setInputLatex: (value: string) => void;
  /** The last command that was actually sent, for "yinele". */
  lastCommandText: string | null;
  /** Run a Turkish command through `POST /compute/text`. */
  run: (text: string, options?: RunCommandOptions) => Promise<void>;
  /** Commit an outcome produced elsewhere (the voice path) without recomputing it. */
  applyOutcome: (outcome: ComputeOut, options?: { latencyKey?: string }) => void;
  /** Re-run the last command; a no-op when nothing has been run yet. */
  repeatLast: () => void;
  /** Show a failure the screen detected itself (a microphone refusal, an upload that failed). */
  setProblem: (problem: AiclsProblem | null) => void;
  /** Forget the current answer and the current failure (Escape). */
  clearBoard: () => void;
  /** Whether the book panel is open. */
  bookOpen: boolean;
  /** Open or close the book panel. */
  setBookOpen: (open: boolean) => void;
  /** Increments whenever a new operation was persisted, so the history reloads. */
  historyToken: number;
  /**
   * The client-side action the backend last asked for (`kind: "ui"`), if any.
   *
   * A `ui` outcome is an instruction, not an answer: it never replaces the result on the board.
   */
  uiAction: UiActionName | null;
  /** Increments with every `ui` outcome, so the same action twice is acted on twice. */
  uiActionId: number;
}

/** Recognised UI action names; anything else is ignored rather than guessed at. */
const UI_ACTIONS: ReadonlySet<string> = new Set<string>(["clear", "undo", "repeat_last", "export"]);

/** Read `command.action` of a `kind: "ui"` outcome. */
export function uiActionOf(outcome: ComputeOut | null): UiActionName | null {
  if (outcome === null || outcome.kind !== "ui") {
    return null;
  }
  const action = outcome.command?.action;
  return typeof action === "string" && UI_ACTIONS.has(action) ? (action as UiActionName) : null;
}

/** Default title of the workspace created for a lecture that has none yet. */
function defaultWorkspaceTitle(): string {
  const today = new Date();
  const stamp = Number.isNaN(today.getTime()) ? "" : today.toLocaleDateString("tr-TR");
  return stamp === "" ? t("workspace.defaultTitle") : `${t("workspace.defaultTitle")} — ${stamp}`;
}

/**
 * Own the workspace: its id, the command in flight and the answer on the board.
 *
 * @returns The {@link WorkspaceApi} the screen and its panels are built from.
 */
export function useWorkspace(): WorkspaceApi {
  const workspaceId = useAppState((state) => state.workspaceId);
  const outcome = useAppState((state) => state.outcome);
  const commandText = useAppState((state) => state.commandText);
  const inputLatex = useAppState((state) => state.inputLatex);
  const phase = useAppState((state) => state.phase);
  const bookOpen = useAppState((state) => state.bookOpen);
  const authenticated = useSession((session) => session.accessToken !== null);

  const [ready, setReady] = useState(false);
  const [problem, setProblemState] = useState<AiclsProblem | null>(null);
  const [stages, setStages] = useState<StageTiming[]>([]);
  const [latencyKey, setLatencyKey] = useState<string | null>(null);
  const [historyToken, setHistoryToken] = useState(0);
  const [uiAction, setUiAction] = useState<{ action: UiActionName; id: number } | null>(null);
  const lastCommandRef = useRef<string | null>(null);
  const runRef = useRef<((text: string, options?: RunCommandOptions) => Promise<void>) | null>(
    null,
  );

  /* -- bootstrap ------------------------------------------------------------------------- */

  useEffect(() => {
    if (!authenticated || workspaceId !== null) {
      return undefined;
    }
    let cancelled = false;
    void (async () => {
      const listed = await call(api.GET("/workspaces", {}), {
        fallbackDetailTr: t("errors.network"),
      });
      if (cancelled) {
        return;
      }
      const existing = listed.data?.items?.[0]?.id ?? null;
      if (existing !== null) {
        appStore.setState({ workspaceId: existing });
        setReady(true);
        return;
      }
      const created = await call(
        api.POST("/workspaces", { body: { title: defaultWorkspaceTitle() } }),
        { fallbackDetailTr: t("errors.network") },
      );
      if (cancelled) {
        return;
      }
      if (created.data !== null) {
        appStore.setState({ workspaceId: created.data.id });
      }
      setReady(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [authenticated, workspaceId]);

  useEffect(() => {
    if (workspaceId !== null || !authenticated) {
      setReady(true);
    }
  }, [workspaceId, authenticated]);

  /* -- commands -------------------------------------------------------------------------- */

  const commit = useCallback((next: ComputeOut, key: string | null): void => {
    setProblemState(null);
    const action = uiActionOf(next);
    if (action !== null) {
      // An instruction to the interface: run it, keep the answer that is already on the board.
      appStore.setState({ phase: "idle", errorText: null });
      setUiAction((current) => ({ action, id: (current?.id ?? 0) + 1 }));
      return;
    }
    setLatencyKey(key);
    appStore.setState({
      outcome: next,
      phase: next.kind === "clarification" ? "awaiting-confirmation" : "idle",
      errorText: null,
      ...(next.kind === "book" ? { bookOpen: true } : {}),
      ...(typeof next.input_latex === "string" && next.input_latex !== ""
        ? { inputLatex: next.input_latex }
        : {}),
    });
    if (typeof next.operation_id === "string" && next.operation_id !== "") {
      setHistoryToken((token) => token + 1);
    }
  }, []);

  const run = useCallback(
    async (text: string, options: RunCommandOptions = {}): Promise<void> => {
      const trimmed = text.trim();
      if (trimmed === "") {
        return;
      }
      const modality: Modality = options.modality ?? "text";
      const previousCommand = lastCommandRef.current;
      lastCommandRef.current = trimmed;
      appStore.setState({ phase: "busy", commandText: trimmed, errorText: null });

      const state = appStore.getState();
      const submission = await submitTextCommand(trimmed, {
        workspaceId: state.workspaceId,
        lastLatex: options.lastLatex ?? state.inputLatex,
        modality,
        ...(options.confirmIntent === undefined ? {} : { confirmIntent: options.confirmIntent }),
      });
      setStages(submission.stages);
      if (submission.outcome === null) {
        setProblemState(submission.problem);
        appStore.setState({
          phase: "idle",
          errorText: submission.problem?.detail_tr ?? t("errors.unknown"),
        });
        return;
      }
      if (submission.outcome.kind === "ui") {
        // "Son komutu yinele" must not become the command that gets repeated.
        lastCommandRef.current = previousCommand;
      }
      commit(submission.outcome, submission.latencyKey);
    },
    [commit],
  );

  runRef.current = run;

  const applyOutcome = useCallback(
    (next: ComputeOut, options: { latencyKey?: string } = {}): void => {
      commit(next, options.latencyKey ?? null);
    },
    [commit],
  );

  const repeatLast = useCallback((): void => {
    const previous = lastCommandRef.current;
    if (previous === null) {
      return;
    }
    void runRef.current?.(previous);
  }, []);

  const setCommandText = useCallback((value: string): void => {
    appStore.setState({ commandText: value });
  }, []);

  const setInputLatex = useCallback((value: string): void => {
    appStore.setState({ inputLatex: value });
  }, []);

  const setProblem = useCallback((next: AiclsProblem | null): void => {
    setProblemState(next);
    appStore.setState({ errorText: next?.detail_tr ?? null, phase: "idle" });
  }, []);

  const clearBoard = useCallback((): void => {
    setProblemState(null);
    appStore.setState({ outcome: null, errorText: null, phase: "idle" });
  }, []);

  const setBookOpen = useCallback((open: boolean): void => {
    appStore.setState({ bookOpen: open });
  }, []);

  return {
    workspaceId,
    ready,
    outcome,
    problem,
    stages,
    busy: phase === "busy",
    latencyKey,
    commandText,
    setCommandText,
    inputLatex,
    setInputLatex,
    lastCommandText: lastCommandRef.current,
    run,
    applyOutcome,
    repeatLast,
    setProblem,
    clearBoard,
    bookOpen,
    setBookOpen,
    historyToken,
    uiAction: uiAction?.action ?? null,
    uiActionId: uiAction?.id ?? 0,
  };
}
