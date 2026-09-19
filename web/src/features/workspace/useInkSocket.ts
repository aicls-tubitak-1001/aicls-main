/**
 * React binding for the `/ws/v1/ink` stream (api.md §4.1).
 *
 * The implementation lives in `@/api/sockets`, next to the typed envelopes it speaks; this module
 * stays as the workspace-facing import path so the feature code reads `./useInkSocket`.
 */
export { useInkSocket, type InkSocketState, type UseInkSocketOptions } from "@/api/sockets";
export type { InkClientMessage, InkServerMessage } from "@/api/ws";
