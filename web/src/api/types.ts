/**
 * Domain type aliases — every one of them *derived* from the generated OpenAPI schema.
 *
 * Nothing here describes the wire format itself; each alias is a lookup into
 * `components["schemas"]` (see {@link file://./schema.ts}). That is deliberate: the hand-written
 * DTOs this module used to contain could drift away from the backend silently, whereas a lookup
 * into a schema name the backend no longer emits is a compile error the moment `npm run gen:api`
 * is re-run. Field names stay `snake_case` because that is what travels on the wire (api.md §1).
 *
 * The UI-facing names (`ResultDto`, `StepDto`, …) are kept because they read better in components
 * than `ResultOut`/`StepOut`, which are FastAPI response-model names; both spellings are exported.
 */
import type { Schema } from "./schema";

export type { AiclsProblem, AiclsProblem as ProblemDetail, ErrorCode } from "./problem";

/* -------------------------------------------------------------------------- enumerations --- */

/** The four roles of api.md §1 ("Roller"). */
export type Role = Schema<"UserOut">["role"];

/** How a command was entered (`operations.input_kind`, veri-modeli.md §3.1). */
export type InputKind = Schema<"OperationOut">["input_kind"];

/** Latency telemetry modality — the same values as {@link InputKind} (api.md §3.10). */
export type Modality = NonNullable<Schema<"LatencySampleIn">["modality"]>;

/** Outcome discriminator of {@link ComputeOut} (api.md §3.4). */
export type ComputeKind = Schema<"ComputeOut">["kind"];

/** Shape of what the CAS produced (`mathengine.Result.kind`). */
export type ResultKind = Schema<"ResultOut">["kind"];

/** Where the understood intent came from. */
export type NluSource = NonNullable<Schema<"NluOut">["source"]>;

/* ------------------------------------------------------------------------------ compute --- */

/**
 * A typed CAS command.
 *
 * The backend validates it against `mathengine.CasCommand` and therefore types it on the wire as a
 * free-form object; the client treats it the same way and formats it for the confirmation step
 * (see `lib/commandFormat.ts`) instead of pretending to know every operator's slots.
 */
export type CommandDto = Schema<"ComputeLatexIn">["command"];

/** Per-stage durations in milliseconds (`StageTimer`, ADR-0014). */
export type TimingsDto = Schema<"TimingsOut">;
export type TimingsOut = Schema<"TimingsOut">;

/** One narrated derivation step. */
export type StepDto = Schema<"StepOut">;
export type StepOut = Schema<"StepOut">;

/** What SymPy computed. */
export type ResultDto = Schema<"ResultOut">;
export type ResultOut = Schema<"ResultOut">;

/** An ambiguous input: a Turkish question plus tappable candidates (a 200, never an error). */
export type ClarificationDto = Schema<"ClarificationOut">;
export type ClarificationOut = Schema<"ClarificationOut">;

/** One option of a clarification question. */
export type ClarificationCandidateDto = Schema<"ClarificationCandidateOut">;
export type ClarificationCandidateOut = Schema<"ClarificationCandidateOut">;

/** The "how I understood you" panel. */
export type NluDto = Schema<"NluOut">;
export type NluOut = Schema<"NluOut">;

/** The one response shape of the command path (api.md §3.4). */
export type ComputeOut = Schema<"ComputeOut">;

/** Request body of `POST /compute/text`. */
export type ComputeTextIn = Schema<"ComputeTextIn">;

/** Request body of `POST /compute/latex`. */
export type ComputeLatexIn = Schema<"ComputeLatexIn">;

/** What the session already has on the board (`ComputeTextIn.context`). */
export type ContextIn = Schema<"ContextIn">;

/** `POST /compute/parse` — a diagnostic, never a computation. */
export type ParseOut = Schema<"ParseOut">;

/** `POST /ocr/recognize` response (api.md §3.5). */
export type RecognizeOut = Schema<"RecognizeOut">;

/** `POST /asr/transcribe` response (api.md §3.6). */
export type TranscribeOut = Schema<"TranscribeOut">;

/* ------------------------------------------------------------------- identity & sessions --- */

/** A user as the API exposes it (never the password hash). */
export type UserOut = Schema<"UserOut">;

/** `POST /auth/login` body. */
export type LoginIn = Schema<"LoginIn">;

/** `POST /auth/register` body. */
export type RegisterIn = Schema<"RegisterIn">;

/** `PATCH /users/me` body. */
export type UserPatch = Schema<"UserPatch">;

/** The bearer access token (the refresh token travels as a cookie). */
export type TokenOut = Schema<"TokenOut">;

/** `POST /auth/ws-ticket` response: a single-use 60 s WebSocket ticket. */
export type WsTicketOut = Schema<"WsTicketOut">;

/* ----------------------------------------------------------------- workspace and history --- */

/** A calculus workspace of one user. */
export type WorkspaceOut = Schema<"WorkspaceOut">;

/** `POST /workspaces` body. */
export type WorkspaceIn = Schema<"WorkspaceIn">;

/** One row of the history panel (api.md §3.3). */
export type OperationOut = Schema<"OperationOut">;

/**
 * Cursor page envelope (api.md §1).
 *
 * FastAPI emits one concrete `Page[…]` model per item type; the generic is rebuilt from one of
 * them so the envelope's own fields (`next_cursor`) still come from the schema.
 */
export type Page<T> = Omit<Schema<"Page_OperationOut_">, "items"> & { items: T[] };

/* ---------------------------------------------------------------------------- book (İP-8) --- */

/** A tagged textbook unit plus its rendered form (api.md §3.7). */
export type ContentUnitDto = Schema<"ContentUnitOut">;
export type ContentUnitOut = Schema<"ContentUnitOut">;

/** Ready-to-render form of a textbook unit. */
export type RenderedOut = Schema<"RenderedOut">;

/** A figure attached to a textbook unit. */
export type AssetOut = Schema<"AssetOut">;

/** One chapter of the book tree. */
export type ChapterOut = Schema<"ChapterOut">;

/** One textbook search hit (api.md §3.7). */
export type HitOut = Schema<"HitOut">;

/* ------------------------------------------------------------- voice data collection (İP-3) --- */

/** Consent text served by `GET /contribute/consent/current`. */
export type ConsentTextOut = Schema<"ConsentTextOut">;

/** Band-only demographics (KVKK data minimisation). */
export type ConsentDemographics = Schema<"DemographicsIn">;
export type DemographicsIn = Schema<"DemographicsIn">;

/** `POST /contribute/consent` body. */
export type ConsentIn = Schema<"ConsentIn">;

/** `POST /contribute/consent` response: pseudonym plus participant token. */
export type SpeakerOut = Schema<"SpeakerOut">;

/** One sentence to read aloud. */
export type PromptOut = Schema<"PromptOut">;

/** The automatic quality-control verdict of one recording. */
export type QcReport = Schema<"QcOut">;
export type QcOut = Schema<"QcOut">;

/** One automatic quality measurement and its verdict. */
export type QcCheckOut = Schema<"QcCheckOut">;

/** `POST /contribute/recordings` response (always HTTP 200; see api.md §2). */
export type RecordingOut = Schema<"RecordingOut">;

/** `GET /contribute/me/progress` response. */
export type ContributeProgressOut = Schema<"ProgressOut">;
export type ProgressOut = Schema<"ProgressOut">;

/* --------------------------------------------------------- feedback, telemetry and health --- */

/** `POST /feedback` body (İP-9). */
export type FeedbackIn = Schema<"FeedbackIn">;

/** One feedback record. */
export type FeedbackOut = Schema<"FeedbackOut">;

/** `POST /feedback/sus` body — ten answers in 1..5. */
export type SusIn = Schema<"SusIn">;

/** `POST /feedback/sus` response. */
export type SusOut = Schema<"SusOut">;

/** Client-measured latency sample (api.md §3.10, ADR-0014). */
export type LatencySampleIn = Schema<"LatencySampleIn">;

/** `GET /health/version` response. */
export type VersionOut = Schema<"VersionOut">;

/** `GET /health/ready` response; 503 when anything is not ready. */
export type ReadyOut = Schema<"ReadyOut">;

/** One row of `model_versions` (`GET /admin/models`). */
export type ModelVersionOut = Schema<"ModelVersionOut">;
