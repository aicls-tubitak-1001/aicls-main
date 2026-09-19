/**
 * React binding for the `/ws/v1/audio` stream (api.md §4.2).
 *
 * The implementation lives in `@/api/sockets`, next to the typed envelopes it speaks; this module
 * stays as the voice-facing import path so the feature code reads `./useAudioSocket`.
 */
export { useAudioSocket, type AudioSocketState, type UseAudioSocketOptions } from "@/api/sockets";
export type { AudioClientMessage, AudioServerMessage } from "@/api/ws";
