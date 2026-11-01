/**
 * Cross-Tab & Electron Window Synchronization Bus (`WorkspaceSyncBus.ts`).
 * Synchronizes foreground view transitions (`Shift+G` -> Control Panel, Generation Done -> Piano Roll),
 * initial pop-out workspace state handshakes (`REQUEST_WORKSPACE_STATE` -> `SYNC_WORKSPACE_STATE`),
 * and lossless multi-track / 0-note edits across browser tabs (`BroadcastChannel`) and native
 * Chromium/Electron windows (`window.compassDesktop`), with FIFO message-ID deduplication.
 */

import {
  ActiveCandidateRange,
  SerializedNoteState,
  SerializedTrackState,
  SerializedWorkspaceState,
} from '../controllers/IDawController';
import {
  DawTrack,
  isGenuineDawTrack,
  ReadonlyCanonicalIdSet,
  TrackInstrument,
} from '../domain/DawTrack';
import {
  canonicalBeatsPerMeasure,
  deriveTimeSignatureFromBeats,
} from '../domain/MeasureRange';
import { truncateWellFormedUtf16 } from '../domain/MidiNote';

import {
  cloneCleanUint8Array,
  extractValidatedCandidateMidiBytes,
  GeneratedMidiCandidate,
  GenerationReason,
  inspectGenuineUint8ArrayPayload,
  isRuntimeProxy,
  isValidCandidateMidiBytesPayload,
  MAX_CANDIDATE_MIDI_BYTES,
  MAX_PLAIN_ARRAY_MIDI_BYTES,
  measureValidatedCandidateMidiBytes,
  sanitizeGenerationReason,
} from './CompassApiClient';

export {
  cloneCleanUint8Array,
  extractValidatedCandidateMidiBytes,
  inspectGenuineUint8ArrayPayload,
  isRuntimeProxy,
  isValidCandidateMidiBytesPayload,
  MAX_CANDIDATE_MIDI_BYTES,
  MAX_PLAIN_ARRAY_MIDI_BYTES,
  measureValidatedCandidateMidiBytes,
  sanitizeGenerationReason,
};

export type WorkspaceTabId = 'compass-ai' | 'pianoroll';

export type SyncControllerId = 'demo-pianoroll' | 'cubase-midi-remote' | 'ableton-osc-bridge';

export type WorkspaceSyncMessage =
  | {
      readonly messageId: string;
      readonly type: 'FOCUS_TAB';
      readonly tab: WorkspaceTabId;
      readonly reason: string;
      readonly senderId: string;
    }
  | {
      readonly messageId: string;
      readonly type: 'GENERATION_STATUS';
      readonly isGenerating: boolean;
      readonly statusDetail: string;
      readonly senderId: string;
    }
  | {
      readonly messageId: string;
      readonly type: 'REQUEST_WORKSPACE_STATE';
      readonly senderId: string;
    }
  | {
      readonly messageId: string;
      readonly type: 'SYNC_WORKSPACE_STATE';
      readonly workspaceState: SerializedWorkspaceState;
      readonly senderId: string;
    }
  | {
      readonly messageId: string;
      readonly type: 'SYNC_ACTIVE_CONTROLLER';
      readonly controllerId: SyncControllerId;
      readonly senderId: string;
    };

export interface CompassDesktopApiConfig {
  readonly apiBaseUrl: string;
  readonly handshakeToken: string;
  readonly backendReady?: boolean;
  readonly hasPeerWindows?: boolean;
}

export interface CompassDesktopBridge {
  readonly apiBaseUrl?: string;
  readonly handshakeToken?: string;
  getApiConfig?: () => CompassDesktopApiConfig;
  requestApiConfig?: (force?: boolean) => Promise<CompassDesktopApiConfig>;
  onApiConfigChanged?: (callback: (cfg: CompassDesktopApiConfig) => void) => () => void;
  hasPeerWindows?: () => boolean;
  bringWindowToFront?: (targetTab: WorkspaceTabId) => Promise<void> | void;
  openPianoRollWindow?: () => Promise<void> | void;
  sendWorkspaceSync?: (msg: WorkspaceSyncMessage) => void;
  onWorkspaceSync?: (callback: (msg: WorkspaceSyncMessage) => void) => () => void;
  onBootLog?: (callback: (line: string) => void) => () => void;
  onFocusTabRequest?: (callback: (tab: WorkspaceTabId) => void) => () => void;
}

declare global {
  interface Window {
    compassDesktop?: CompassDesktopBridge;
  }
}

const MAX_SEEN_IDS = 256;
const VALID_SYNC_INSTRUMENTS: ReadonlySet<string> = new Set([
  'PIANO',
  'SAX',
  'BASS',
  'VIOLIN',
  'SYNTH',
  'GUITAR',
  'DRUMS',
]);
let _msgSeq = 1;

function nextMessageId(instanceId: string): string {
  _msgSeq += 1;
  return `${instanceId}_m${_msgSeq}`;
}

const ALLOWED_TRACK_KEYS: ReadonlySet<string> = new Set([
  'id',
  'name',
  'instrument',
  'colorHex',
  'isMuted',
  'isSoloed',
  'notes',
]);
const ALLOWED_SLOT_KEYS: ReadonlySet<string> = new Set(['trackId', 'startTick', 'endTick']);
const ALLOWED_REASON_KEYS: ReadonlySet<string> = new Set([
  'source',
  'task',
  'thinking',
  'instruments',
  'genre',
  'gen_measure_count',
  'key',
  'note_density',
]);
const ALLOWED_CANDIDATE_KEYS: ReadonlySet<string> = new Set([
  'index',
  'filename',
  'rawMidiBytes',
  'reason',
]);
const ALL_SYNC_ENVELOPE_KEYS: ReadonlySet<string> = new Set([
  'messageId',
  'type',
  'senderId',
  'tab',
  'reason',
  'isGenerating',
  'statusDetail',
  'workspaceState',
  'controllerId',
]);
const ALLOWED_FOCUS_MSG_KEYS: ReadonlySet<string> = new Set([
  'messageId',
  'type',
  'tab',
  'reason',
  'senderId',
]);
const ALLOWED_GEN_STATUS_MSG_KEYS: ReadonlySet<string> = new Set([
  'messageId',
  'type',
  'isGenerating',
  'statusDetail',
  'senderId',
]);
const ALLOWED_REQ_STATE_MSG_KEYS: ReadonlySet<string> = new Set([
  'messageId',
  'type',
  'senderId',
]);
const ALLOWED_SYNC_STATE_MSG_KEYS: ReadonlySet<string> = new Set([
  'messageId',
  'type',
  'workspaceState',
  'senderId',
]);
const ALLOWED_SYNC_CONTROLLER_MSG_KEYS: ReadonlySet<string> = new Set([
  'messageId',
  'type',
  'controllerId',
  'senderId',
]);
const VALID_SYNC_CONTROLLER_IDS: ReadonlySet<string> = new Set([
  'demo-pianoroll',
  'cubase-midi-remote',
  'ableton-osc-bridge',
]);
const ALLOWED_WORKSPACE_KEYS: ReadonlySet<string> = new Set([
  'ppqn',
  'bpm',
  'beatsPerMeasure',
  'timeSigNumerator',
  'timeSigDenominator',
  'activeTrackId',
  'targetStartMeasure',
  'targetEndMeasure',
  'quantizeDivision',
  'isTriplet',
  'tracks',
  'candidates',
  'activeCandidateIndex',
  'activeCandidateNoteIds',
  'activeCandidateRange',
  'candidateOriginSlot',
  'latestReason',
]);

function isFastCleanAsciiSyncId(raw: string): boolean {
  const len = raw.length;
  if (len === 0 || len > 128) return false;
  const c0 = raw.charCodeAt(0);
  const cLast = raw.charCodeAt(len - 1);
  if (c0 <= 0x20 || c0 >= 0x7f || cLast <= 0x20 || cLast >= 0x7f) {
    return false;
  }
  for (let i = 0; i < len; i += 1) {
    const c = raw.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdfff) {
      return false;
    }
  }
  return true;
}

function canonSyncId(raw: string): string {
  if (isFastCleanAsciiSyncId(raw)) {
    return raw;
  }
  return truncateWellFormedUtf16(raw.trim(), 128).trim();
}

/**
 * Single-pass plain-object snapshot (`[R14-MED-01]`).
 * Reads `Object.getOwnPropertyNames` and each own property descriptor strictly ONCE into a fresh `Map`
 * so a stateful browser `Proxy` (where Node's `util.types.isProxy` is unavailable) can never mutate
 * between validation and extraction.
 */
function snapshotAllowedPlainObject(
  raw: unknown,
  allowed: ReadonlySet<string>,
): Map<string, unknown> | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw) || isRuntimeProxy(raw)) return null;
  const proto = Object.getPrototypeOf(raw);
  if (proto !== null && proto !== Object.prototype) return null;
  if (Object.getOwnPropertySymbols(raw).length > 0) return null;
  const keys = Object.getOwnPropertyNames(raw);
  if (keys.length > allowed.size) return null;
  const snap = new Map<string, unknown>();
  for (let i = 0; i < keys.length; i += 1) {
    const k = keys[i];
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') return null;
    if (!allowed.has(k) || snap.has(k)) return null;
    const desc = Object.getOwnPropertyDescriptor(raw, k);
    if (!desc || !('value' in desc)) return null;
    snap.set(k, desc.value);
  }
  return snap;
}

/**
 * Single-pass dense-array snapshot (`[R14-MED-01]`).
 * Reads `length` and each element `0..len-1` descriptor strictly ONCE into a fresh `Array`.
 */
function snapshotPlainDenseArray(raw: unknown, maxItems: number): unknown[] | null {
  if (
    !raw ||
    typeof raw !== 'object' ||
    isRuntimeProxy(raw) ||
    !Array.isArray(raw) ||
    Object.getPrototypeOf(raw) !== Array.prototype
  ) {
    return null;
  }
  const lenDesc = Object.getOwnPropertyDescriptor(raw, 'length');
  if (!lenDesc || !('value' in lenDesc)) return null;
  const len = lenDesc.value;
  if (typeof len !== 'number' || !Number.isInteger(len) || len < 0 || len > maxItems) {
    return null;
  }
  if (Object.getOwnPropertySymbols(raw).length > 0) return null;
  if (Object.getOwnPropertyNames(raw).length !== len + 1) {
    return null;
  }
  const out: unknown[] = new Array(len);
  for (let i = 0; i < len; i += 1) {
    const desc = Object.getOwnPropertyDescriptor(raw, i);
    if (!desc || !('value' in desc)) return null;
    out[i] = desc.value;
  }
  return out;
}

function readValidatedPlainNote(
  rawNote: unknown,
  maxProjectTicks: number,
  rawTrimmedByCanonNoteId: Map<string, string>,
  clonePayload: boolean,
): SerializedNoteState | true | null {
  if (!rawNote || typeof rawNote !== 'object' || Array.isArray(rawNote) || isRuntimeProxy(rawNote)) {
    return null;
  }
  const proto = Object.getPrototypeOf(rawNote);
  if (proto !== null && proto !== Object.prototype) return null;
  if (Object.getOwnPropertySymbols(rawNote).length > 0) return null;
  const keys = Object.getOwnPropertyNames(rawNote);
  if (keys.length < 5 || keys.length > 6) return null;

  let seenMask = 0;
  let rawNoteId: unknown;
  let pitch: unknown;
  let startTick: unknown;
  let durationTicks: unknown;
  let velocity: unknown;
  let channel: unknown;

  for (let i = 0; i < keys.length; i += 1) {
    const k = keys[i];
    switch (k) {
      case 'id': {
        if ((seenMask & 1) !== 0) return null;
        seenMask |= 1;
        const desc = Object.getOwnPropertyDescriptor(rawNote, 'id');
        if (!desc || !('value' in desc)) return null;
        rawNoteId = desc.value;
        break;
      }
      case 'pitch': {
        if ((seenMask & 2) !== 0) return null;
        seenMask |= 2;
        const desc = Object.getOwnPropertyDescriptor(rawNote, 'pitch');
        if (!desc || !('value' in desc)) return null;
        pitch = desc.value;
        break;
      }
      case 'startTick': {
        if ((seenMask & 4) !== 0) return null;
        seenMask |= 4;
        const desc = Object.getOwnPropertyDescriptor(rawNote, 'startTick');
        if (!desc || !('value' in desc)) return null;
        startTick = desc.value;
        break;
      }
      case 'durationTicks': {
        if ((seenMask & 8) !== 0) return null;
        seenMask |= 8;
        const desc = Object.getOwnPropertyDescriptor(rawNote, 'durationTicks');
        if (!desc || !('value' in desc)) return null;
        durationTicks = desc.value;
        break;
      }
      case 'velocity': {
        if ((seenMask & 16) !== 0) return null;
        seenMask |= 16;
        const desc = Object.getOwnPropertyDescriptor(rawNote, 'velocity');
        if (!desc || !('value' in desc)) return null;
        velocity = desc.value;
        break;
      }
      case 'channel': {
        if ((seenMask & 32) !== 0) return null;
        seenMask |= 32;
        const desc = Object.getOwnPropertyDescriptor(rawNote, 'channel');
        if (!desc || !('value' in desc)) return null;
        channel = desc.value;
        break;
      }
      default:
        return null;
    }
  }
  if ((seenMask & 31) !== 31) return null;

  if (typeof rawNoteId !== 'string' || rawNoteId.length > 128) {
    return null;
  }
  let rawTrimmedNoteId: string;
  let canonNoteId: string;
  if (isFastCleanAsciiSyncId(rawNoteId)) {
    rawTrimmedNoteId = rawNoteId;
    canonNoteId = rawNoteId;
  } else {
    rawTrimmedNoteId = rawNoteId.trim();
    canonNoteId = truncateWellFormedUtf16(rawTrimmedNoteId, 128).trim();
  }
  if (!canonNoteId || rawTrimmedByCanonNoteId.has(canonNoteId)) {
    return null;
  }
  if (
    typeof pitch !== 'number' ||
    !Number.isFinite(pitch) ||
    pitch < 0 ||
    pitch > 127 ||
    typeof startTick !== 'number' ||
    !Number.isFinite(startTick) ||
    startTick < 0 ||
    typeof durationTicks !== 'number' ||
    !Number.isFinite(durationTicks) ||
    durationTicks < 1 ||
    typeof velocity !== 'number' ||
    !Number.isFinite(velocity) ||
    velocity < 1 ||
    velocity > 127 ||
    (channel !== undefined &&
      (typeof channel !== 'number' ||
        !Number.isFinite(channel) ||
        channel < 0 ||
        channel > 15))
  ) {
    return null;
  }
  const roundedStart = Math.round(startTick);
  const roundedDur = Math.round(durationTicks);
  if (
    roundedStart < 0 ||
    roundedDur < 1 ||
    roundedStart >= maxProjectTicks ||
    roundedStart + roundedDur > maxProjectTicks
  ) {
    return null;
  }
  rawTrimmedByCanonNoteId.set(canonNoteId, rawTrimmedNoteId);
  if (!clonePayload) {
    return true;
  }
  return (
    channel !== undefined
      ? { id: rawNoteId, pitch, startTick, durationTicks, velocity, channel }
      : { id: rawNoteId, pitch, startTick, durationTicks, velocity }
  ) as SerializedNoteState;
}

function parseBoundedStringArray(
  arr: unknown,
  maxItems: number,
  maxItemLen: number,
  requireNonEmptyTrimmed: boolean,
  cloneOut: boolean,
): { valid: true; value: string[] } | { valid: false } {
  const snapArr = snapshotPlainDenseArray(arr, maxItems);
  if (!snapArr) return { valid: false };
  const out: string[] = cloneOut ? new Array(snapArr.length) : [];
  for (let i = 0; i < snapArr.length; i += 1) {
    const x = snapArr[i];
    if (typeof x !== 'string' || x.length > maxItemLen) return { valid: false };
    if (requireNonEmptyTrimmed && x.trim().length === 0) return { valid: false };
    if (cloneOut) {
      out[i] = x;
    }
  }
  return { valid: true, value: out };
}

/**
 * Single-pass validator + deep plain-POJO cloner for `GenerationReason` (`[R13-MED-02]`, `[R14-MED-01]`).
 * Reads every property and element strictly ONCE so a stateful `Proxy` can never mutate
 * between validation and cloning, while maintaining 1-to-1 acceptance parity with `ipc_security.cjs`.
 */
export function parseValidatedGenerationReason(
  reason: unknown,
  cloneOut = true,
): { valid: true; value: GenerationReason | null } | { valid: false } {
  try {
    if (reason === null || reason === undefined) {
      return { valid: true, value: null };
    }
    const rSnap = snapshotAllowedPlainObject(reason, ALLOWED_REASON_KEYS);
    if (!rSnap) {
      return { valid: false };
    }
    const source = rSnap.get('source');
    if (source !== 'model_cot' && source !== 'request' && source !== 'foundation') {
      return { valid: false };
    }
    const task = rSnap.get('task');
    if (typeof task !== 'string' || task.trim().length === 0 || task.length > 128) {
      return { valid: false };
    }
    const thinking = rSnap.get('thinking');
    if (typeof thinking !== 'boolean') {
      return { valid: false };
    }

    const rawInstruments = rSnap.get('instruments');
    let cleanInstruments: string[] | null = null;
    if (rawInstruments !== null && rawInstruments !== undefined) {
      const parsedInst = parseBoundedStringArray(rawInstruments, 64, 64, false, cloneOut);
      if (!parsedInst.valid) return { valid: false };
      cleanInstruments = parsedInst.value;
    }

    const rawGenre = rSnap.get('genre');
    let cleanGenre: string[] | null = null;
    if (rawGenre !== null && rawGenre !== undefined) {
      const parsedGenre = parseBoundedStringArray(rawGenre, 64, 64, false, cloneOut);
      if (!parsedGenre.valid) return { valid: false };
      cleanGenre = parsedGenre.value;
    }

    const genMeasureCount = rSnap.get('gen_measure_count');
    let cleanMeasureCount: number | null = null;
    if (genMeasureCount !== null && genMeasureCount !== undefined) {
      if (
        typeof genMeasureCount !== 'number' ||
        !Number.isFinite(genMeasureCount) ||
        genMeasureCount < 0 ||
        genMeasureCount > 512
      ) {
        return { valid: false };
      }
      cleanMeasureCount = genMeasureCount;
    }

    const key = rSnap.get('key');
    let cleanKey: string | null = null;
    if (key !== null && key !== undefined) {
      if (typeof key !== 'string' || key.length > 32) {
        return { valid: false };
      }
      cleanKey = key;
    }

    const noteDensity = rSnap.get('note_density');
    let cleanDensity: Record<string, number | null> | null = null;
    if (noteDensity !== null && noteDensity !== undefined) {
      if (
        typeof noteDensity !== 'object' ||
        Array.isArray(noteDensity) ||
        isRuntimeProxy(noteDensity)
      ) {
        return { valid: false };
      }
      const nd = noteDensity as Record<string, unknown>;
      const ndProto = Object.getPrototypeOf(nd);
      if (ndProto !== null && ndProto !== Object.prototype) return { valid: false };
      if (Object.getOwnPropertySymbols(nd).length > 0) return { valid: false };
      const keys = Object.getOwnPropertyNames(nd);
      if (keys.length > 32) return { valid: false };
      const seenNdKeys = new Set<string>();
      if (cloneOut) {
        cleanDensity = {};
      }
      for (let ki = 0; ki < keys.length; ki += 1) {
        const k = keys[ki];
        if (
          k.length === 0 ||
          k.length > 64 ||
          k === '__proto__' ||
          k === 'constructor' ||
          k === 'prototype' ||
          seenNdKeys.has(k)
        ) {
          return { valid: false };
        }
        seenNdKeys.add(k);
        const kDesc = Object.getOwnPropertyDescriptor(nd, k);
        if (!kDesc || !('value' in kDesc)) return { valid: false };
        const v = kDesc.value;
        if (v !== null && (typeof v !== 'number' || !Number.isFinite(v))) {
          return { valid: false };
        }
        if (cloneOut && cleanDensity) {
          Object.defineProperty(cleanDensity, k, {
            value: v,
            writable: true,
            enumerable: true,
            configurable: true,
          });
        }
      }
    }

    return {
      valid: true,
      value: cloneOut
        ? {
            source,
            task,
            thinking,
            instruments: cleanInstruments,
            note_density: cleanDensity,
            gen_measure_count: cleanMeasureCount,
            genre: cleanGenre,
            key: cleanKey,
          }
        : null,
    };
  } catch {
    return { valid: false };
  }
}

export function isValidGenerationReasonPayload(reason: unknown): boolean {
  return parseValidatedGenerationReason(reason, false).valid;
}

function parseAndCloneSyncMessage(
  raw: unknown,
  clonePayload: boolean,
  allowCompaction = false,
): WorkspaceSyncMessage | null {
  try {
    const envSnap = snapshotAllowedPlainObject(raw, ALL_SYNC_ENVELOPE_KEYS);
    if (!envSnap) return null;
    const msgType = envSnap.get('type');

    let allowedEnvKeys: ReadonlySet<string>;
    if (msgType === 'FOCUS_TAB') {
      allowedEnvKeys = ALLOWED_FOCUS_MSG_KEYS;
    } else if (msgType === 'GENERATION_STATUS') {
      allowedEnvKeys = ALLOWED_GEN_STATUS_MSG_KEYS;
    } else if (msgType === 'REQUEST_WORKSPACE_STATE') {
      allowedEnvKeys = ALLOWED_REQ_STATE_MSG_KEYS;
    } else if (msgType === 'SYNC_WORKSPACE_STATE') {
      allowedEnvKeys = ALLOWED_SYNC_STATE_MSG_KEYS;
    } else if (msgType === 'SYNC_ACTIVE_CONTROLLER') {
      allowedEnvKeys = ALLOWED_SYNC_CONTROLLER_MSG_KEYS;
    } else {
      return null;
    }

    for (const k of envSnap.keys()) {
      if (!allowedEnvKeys.has(k)) return null;
    }

    const senderId = envSnap.get('senderId');
    const messageId = envSnap.get('messageId');
    if (
      typeof senderId !== 'string' ||
      senderId.trim().length === 0 ||
      senderId.length > 128 ||
      typeof messageId !== 'string' ||
      messageId.trim().length === 0 ||
      messageId.length > 128
    ) {
      return null;
    }

    if (msgType === 'FOCUS_TAB') {
      const reason = envSnap.get('reason');
      if (reason !== undefined && (typeof reason !== 'string' || reason.length > 256)) {
        return null;
      }
      const tab = envSnap.get('tab');
      if (tab !== 'compass-ai' && tab !== 'pianoroll') {
        return null;
      }
      return {
        messageId,
        type: 'FOCUS_TAB',
        tab,
        reason: reason ?? '',
        senderId,
      };
    }

    if (msgType === 'GENERATION_STATUS') {
      const isGenerating = envSnap.get('isGenerating');
      const statusDetail = envSnap.get('statusDetail');
      if (
        typeof isGenerating !== 'boolean' ||
        typeof statusDetail !== 'string' ||
        statusDetail.length > 512
      ) {
        return null;
      }
      return {
        messageId,
        type: 'GENERATION_STATUS',
        isGenerating,
        statusDetail,
        senderId,
      };
    }

    if (msgType === 'REQUEST_WORKSPACE_STATE') {
      return {
        messageId,
        type: 'REQUEST_WORKSPACE_STATE',
        senderId,
      };
    }

    if (msgType === 'SYNC_ACTIVE_CONTROLLER') {
      const controllerId = envSnap.get('controllerId');
      if (typeof controllerId !== 'string' || !VALID_SYNC_CONTROLLER_IDS.has(controllerId)) {
        return null;
      }
      return {
        messageId,
        type: 'SYNC_ACTIVE_CONTROLLER',
        controllerId: controllerId as SyncControllerId,
        senderId,
      };
    }

    if (msgType === 'SYNC_WORKSPACE_STATE') {
      const wsSnap = snapshotAllowedPlainObject(envSnap.get('workspaceState'), ALLOWED_WORKSPACE_KEYS);
      if (!wsSnap) {
        return null;
      }
      const bpm = wsSnap.get('bpm');
      if (typeof bpm !== 'number' || !Number.isFinite(bpm) || bpm < 20 || bpm > 300) {
        return null;
      }
      const ppqn = wsSnap.get('ppqn');
      if (
        ppqn !== undefined &&
        (typeof ppqn !== 'number' || !Number.isFinite(ppqn) || ppqn < 24 || ppqn > 32767)
      ) {
        return null;
      }
      const effectivePpqn = ppqn !== undefined ? Math.round(ppqn) : 480;
      const maxProjectTicks = 4128 * effectivePpqn;
      const maxSlotTick = 16_512 * effectivePpqn;
      const timeSigNum = wsSnap.get('timeSigNumerator');
      const timeSigDen = wsSnap.get('timeSigDenominator');
      const beatsPerMeasure = wsSnap.get('beatsPerMeasure');
      if ((timeSigNum === undefined) !== (timeSigDen === undefined)) {
        return null;
      }
      if (timeSigNum !== undefined && timeSigDen !== undefined) {
        if (
          typeof timeSigNum !== 'number' ||
          !Number.isInteger(timeSigNum) ||
          timeSigNum < 1 ||
          timeSigNum > 32 ||
          typeof timeSigDen !== 'number' ||
          ![1, 2, 4, 8, 16, 32].includes(timeSigDen)
        ) {
          return null;
        }
        const expectedBeats = canonicalBeatsPerMeasure(timeSigNum, timeSigDen);
        if (beatsPerMeasure !== undefined) {
          if (
            typeof beatsPerMeasure !== 'number' ||
            !Number.isFinite(beatsPerMeasure) ||
            beatsPerMeasure < 0.5 ||
            beatsPerMeasure > 16 ||
            Math.abs(beatsPerMeasure - expectedBeats) > 1e-6
          ) {
            return null;
          }
        }
      } else if (beatsPerMeasure !== undefined) {
        if (
          typeof beatsPerMeasure !== 'number' ||
          !Number.isFinite(beatsPerMeasure) ||
          beatsPerMeasure < 0.5 ||
          beatsPerMeasure > 16
        ) {
          return null;
        }
        const derivedTs = deriveTimeSignatureFromBeats(beatsPerMeasure);
        const expectedBeats = canonicalBeatsPerMeasure(derivedTs.numerator, derivedTs.denominator);
        if (Math.abs(beatsPerMeasure - expectedBeats) > 1e-6) {
          return null;
        }
      }

      const tracks = snapshotPlainDenseArray(wsSnap.get('tracks'), 64);
      if (!tracks || tracks.length === 0) {
        return null;
      }
      const rawTrimmedByCanonTrackId = new Map<string, string>();
      const rawTrimmedByCanonNoteId = new Map<string, string>();
      const cleanTracks: SerializedTrackState[] = clonePayload ? new Array(tracks.length) : [];

      for (let ti = 0; ti < tracks.length; ti += 1) {
        const trkSnap = snapshotAllowedPlainObject(tracks[ti], ALLOWED_TRACK_KEYS);
        if (!trkSnap) return null;
        const rawTrackId = trkSnap.get('id');
        if (typeof rawTrackId !== 'string' || rawTrackId.length > 128) {
          return null;
        }
        const rawTrimmedTrackId = rawTrackId.trim();
        const canonTrackId = canonSyncId(rawTrackId);
        if (!canonTrackId || rawTrimmedByCanonTrackId.has(canonTrackId)) {
          return null;
        }
        const tName = trkSnap.get('name');
        if (typeof tName !== 'string' || tName.trim().length === 0 || tName.length > 256) {
          return null;
        }
        const tInst = trkSnap.get('instrument');
        if (typeof tInst !== 'string' || !VALID_SYNC_INSTRUMENTS.has(tInst)) {
          return null;
        }
        const tColor = trkSnap.get('colorHex');
        if (
          typeof tColor !== 'string' ||
          tColor.trim().length === 0 ||
          tColor.length > 32
        ) {
          return null;
        }
        const tMuted = trkSnap.get('isMuted');
        const tSoloed = trkSnap.get('isSoloed');
        if (typeof tMuted !== 'boolean' || typeof tSoloed !== 'boolean') {
          return null;
        }
        const notes = snapshotPlainDenseArray(
          trkSnap.get('notes'),
          32768 - rawTrimmedByCanonNoteId.size,
        );
        if (!notes) return null;
        if (rawTrimmedByCanonNoteId.size + notes.length > 32768) return null;
        rawTrimmedByCanonTrackId.set(canonTrackId, rawTrimmedTrackId);

        const cleanNotes: SerializedNoteState[] = clonePayload ? new Array(notes.length) : [];
        for (let ni = 0; ni < notes.length; ni += 1) {
          const parsedNote = readValidatedPlainNote(
            notes[ni],
            maxProjectTicks,
            rawTrimmedByCanonNoteId,
            clonePayload,
          );
          if (parsedNote === null) {
            return null;
          }
          if (clonePayload) {
            cleanNotes[ni] = parsedNote as SerializedNoteState;
          }
        }

        if (clonePayload) {
          cleanTracks[ti] = {
            id: rawTrackId,
            name: tName,
            instrument: tInst as TrackInstrument,
            colorHex: tColor,
            isMuted: tMuted,
            isSoloed: tSoloed,
            notes: cleanNotes,
          };
        }
      }

      const activeTrackId = wsSnap.get('activeTrackId');
      if (typeof activeTrackId !== 'string' || activeTrackId.length > 128) {
        return null;
      }
      const canonActiveTrackId = canonSyncId(activeTrackId);
      if (
        !canonActiveTrackId ||
        rawTrimmedByCanonTrackId.get(canonActiveTrackId) !== activeTrackId.trim()
      ) {
        return null;
      }

      const parseSlot = (
        slot: unknown,
      ):
        | {
            valid: true;
            canonTrackId: string | null;
            present: boolean;
            value: { trackId: string; startTick: number; endTick: number } | null;
          }
        | { valid: false } => {
        if (slot === undefined) {
          return { valid: true, canonTrackId: null, present: false, value: null };
        }
        if (slot === null) {
          return { valid: true, canonTrackId: null, present: true, value: null };
        }
        const slotSnap = snapshotAllowedPlainObject(slot, ALLOWED_SLOT_KEYS);
        if (!slotSnap) return { valid: false };
        const slotTrackId = slotSnap.get('trackId');
        const slotStart = slotSnap.get('startTick');
        const slotEnd = slotSnap.get('endTick');
        if (typeof slotTrackId !== 'string' || slotTrackId.length > 128) return { valid: false };
        const canonSlotTrackId = canonSyncId(slotTrackId);
        if (
          !canonSlotTrackId ||
          rawTrimmedByCanonTrackId.get(canonSlotTrackId) !== slotTrackId.trim() ||
          typeof slotStart !== 'number' ||
          !Number.isFinite(slotStart) ||
          slotStart < 0 ||
          typeof slotEnd !== 'number' ||
          !Number.isFinite(slotEnd) ||
          slotEnd < slotStart ||
          slotEnd > maxSlotTick
        ) {
          return { valid: false };
        }
        return {
          valid: true,
          canonTrackId: canonSlotTrackId,
          present: true,
          value: clonePayload
            ? { trackId: slotTrackId, startTick: slotStart, endTick: slotEnd }
            : null,
        };
      };

      const parsedActiveSlot = parseSlot(wsSnap.get('activeCandidateRange'));
      const parsedOriginSlot = parseSlot(wsSnap.get('candidateOriginSlot'));
      if (!parsedActiveSlot.valid || !parsedOriginSlot.valid) {
        return null;
      }
      if (
        parsedActiveSlot.canonTrackId !== null &&
        parsedOriginSlot.canonTrackId !== null &&
        parsedActiveSlot.canonTrackId !== parsedOriginSlot.canonTrackId
      ) {
        return null;
      }

      const targetStartMeasure = wsSnap.get('targetStartMeasure');
      const targetEndMeasure = wsSnap.get('targetEndMeasure');
      if (
        typeof targetStartMeasure !== 'number' ||
        !Number.isFinite(targetStartMeasure) ||
        targetStartMeasure < 1 ||
        targetStartMeasure > 1024 ||
        typeof targetEndMeasure !== 'number' ||
        !Number.isFinite(targetEndMeasure) ||
        targetEndMeasure < targetStartMeasure ||
        targetEndMeasure > 1032
      ) {
        return null;
      }
      const activeCandidateIndex = wsSnap.get('activeCandidateIndex');
      if (
        typeof activeCandidateIndex !== 'number' ||
        !Number.isInteger(activeCandidateIndex) ||
        activeCandidateIndex < -1 ||
        activeCandidateIndex > 64
      ) {
        return null;
      }

      const rawActiveCandNoteIds = snapshotPlainDenseArray(wsSnap.get('activeCandidateNoteIds'), 32768);
      if (!rawActiveCandNoteIds) {
        return null;
      }
      const cleanCandNoteIds: string[] = clonePayload
        ? new Array(rawActiveCandNoteIds.length)
        : [];
      const seenRawTrimmedByCanonCandId = new Map<string, string>();
      for (let ci = 0; ci < rawActiveCandNoteIds.length; ci += 1) {
        const cid = rawActiveCandNoteIds[ci];
        if (typeof cid !== 'string' || cid.length > 128) {
          return null;
        }
        let rawTrimmedCid: string;
        let canonCid: string;
        if (isFastCleanAsciiSyncId(cid)) {
          rawTrimmedCid = cid;
          canonCid = cid;
        } else {
          rawTrimmedCid = cid.trim();
          if (rawTrimmedCid.length === 0) return null;
          canonCid = truncateWellFormedUtf16(rawTrimmedCid, 128).trim();
          if (!canonCid) return null;
        }
        if (
          rawTrimmedByCanonNoteId.has(canonCid) &&
          rawTrimmedByCanonNoteId.get(canonCid) !== rawTrimmedCid
        ) {
          return null;
        }
        if (
          seenRawTrimmedByCanonCandId.has(canonCid) &&
          seenRawTrimmedByCanonCandId.get(canonCid) !== rawTrimmedCid
        ) {
          return null;
        }
        seenRawTrimmedByCanonCandId.set(canonCid, rawTrimmedCid);
        if (clonePayload) {
          cleanCandNoteIds[ci] = cid;
        }
      }

      const quantizeDivision = wsSnap.get('quantizeDivision');
      if (
        quantizeDivision !== undefined &&
        quantizeDivision !== 4 &&
        quantizeDivision !== 8 &&
        quantizeDivision !== 16 &&
        quantizeDivision !== 32
      ) {
        return null;
      }
      const isTriplet = wsSnap.get('isTriplet');
      if (isTriplet !== undefined && typeof isTriplet !== 'boolean') {
        return null;
      }

      const candidates = snapshotPlainDenseArray(wsSnap.get('candidates'), 32);
      if (!candidates) return null;
      const cleanCandidates: Array<{
        index: number;
        filename: string;
        rawMidiBytes: Uint8Array;
        reason: GenerationReason | null;
      }> = clonePayload ? new Array(candidates.length) : [];
      let totalCandBytes = 0;

      for (let ci = 0; ci < candidates.length; ci += 1) {
        const candSnap = snapshotAllowedPlainObject(candidates[ci], ALLOWED_CANDIDATE_KEYS);
        if (!candSnap) return null;
        const cIndex = candSnap.get('index');
        if (
          typeof cIndex !== 'number' ||
          !Number.isInteger(cIndex) ||
          cIndex < 0 ||
          cIndex > 64
        ) {
          return null;
        }
        const cFilename = candSnap.get('filename');
        if (
          typeof cFilename !== 'string' ||
          cFilename.trim().length === 0 ||
          cFilename.length > 256
        ) {
          return null;
        }
        const rawMidiBytes = candSnap.get('rawMidiBytes');
        const remainingCandBytes = MAX_CANDIDATE_MIDI_BYTES - totalCandBytes;
        let cleanBytes: Uint8Array | null = null;
        let accountedBytes = 0;
        if (clonePayload) {
          const extracted = extractValidatedCandidateMidiBytes(
            rawMidiBytes,
            allowCompaction,
            remainingCandBytes,
          );
          if (!extracted) return null;
          cleanBytes = extracted.bytes;
          accountedBytes = extracted.accountedBytes;
        } else {
          const measured = measureValidatedCandidateMidiBytes(rawMidiBytes, remainingCandBytes);
          if (measured === null) return null;
          accountedBytes = measured;
        }
        totalCandBytes += accountedBytes;
        if (totalCandBytes > MAX_CANDIDATE_MIDI_BYTES) return null;

        const parsedCandReason = parseValidatedGenerationReason(
          candSnap.get('reason'),
          clonePayload,
        );
        if (!parsedCandReason.valid) return null;

        if (clonePayload && cleanBytes) {
          cleanCandidates[ci] = {
            index: cIndex,
            filename: cFilename,
            rawMidiBytes: cleanBytes,
            reason: parsedCandReason.value,
          };
        }
      }

      const hasLatestReasonKey = wsSnap.has('latestReason');
      const parsedLatestReason = parseValidatedGenerationReason(
        wsSnap.get('latestReason'),
        clonePayload,
      );
      if (!parsedLatestReason.valid) return null;

      if (!clonePayload) {
        return {
          messageId,
          type: 'REQUEST_WORKSPACE_STATE',
          senderId,
        };
      }

      const cleanWs = {
        ...(ppqn !== undefined ? { ppqn } : {}),
        bpm,
        ...(beatsPerMeasure !== undefined ? { beatsPerMeasure } : {}),
        ...(timeSigNum !== undefined ? { timeSigNumerator: timeSigNum } : {}),
        ...(timeSigDen !== undefined ? { timeSigDenominator: timeSigDen } : {}),
        activeTrackId,
        targetStartMeasure,
        targetEndMeasure,
        ...(quantizeDivision !== undefined ? { quantizeDivision } : {}),
        ...(isTriplet !== undefined ? { isTriplet } : {}),
        tracks: cleanTracks,
        candidates: cleanCandidates,
        activeCandidateIndex,
        activeCandidateNoteIds: cleanCandNoteIds,
        ...(parsedActiveSlot.present ? { activeCandidateRange: parsedActiveSlot.value } : {}),
        ...(parsedOriginSlot.present ? { candidateOriginSlot: parsedOriginSlot.value } : {}),
        ...(hasLatestReasonKey ? { latestReason: parsedLatestReason.value } : {}),
      } as SerializedWorkspaceState;

      return {
        messageId,
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: cleanWs,
        senderId,
      };
    }

    return null;
  } catch {
    return null;
  }
}

export function isValidIncomingSyncMessage(raw: unknown): raw is WorkspaceSyncMessage {
  return parseAndCloneSyncMessage(raw, false, false) !== null;
}

const TRUSTED_WORKSPACE_STATES = new WeakSet<SerializedWorkspaceState>();

const ALLOWED_TRUSTED_SNAPSHOT_PARAM_KEYS: ReadonlySet<string> = new Set([
  'ppqn',
  'bpm',
  'beatsPerMeasure',
  'timeSigNumerator',
  'timeSigDenominator',
  'activeTrackId',
  'targetStartMeasure',
  'targetEndMeasure',
  'quantizeDivision',
  'isTriplet',
  'domainTracks',
  'candidates',
  'activeCandidateIndex',
  'activeCandidateNoteIds',
  'activeCandidateRange',
  'candidateOriginSlot',
  'latestReason',
]);

const ALLOWED_TRUSTED_CANDIDATE_INPUT_KEYS: ReadonlySet<string> = new Set([
  'index',
  'filename',
  'rawMidiBytes',
  'parsedMidi',
  'reason',
]);

function deepFreezeGenerationReason(reason: GenerationReason | null): GenerationReason | null {
  if (reason === null) return null;
  if (Array.isArray(reason.instruments)) {
    Object.freeze(reason.instruments);
  }
  if (Array.isArray(reason.genre)) {
    Object.freeze(reason.genre);
  }
  if (reason.note_density && typeof reason.note_density === 'object') {
    Object.freeze(reason.note_density);
  }
  return Object.freeze(reason);
}

const VERIFIED_INBOUND_SYNC_STATES = new WeakSet<SerializedWorkspaceState>();

/**
 * One-shot consumer token for freshly cloned `SYNC_WORKSPACE_STATE` payloads emitted by
 * `WorkspaceSyncBus._dispatchIncoming` (`[R25-HIGH-02]`), avoiding duplicate 32k-element
 * descriptor walks when `DemoPianoRollController.applyWorkspaceState` consumes the payload immediately.
 */
export function consumeVerifiedInboundSyncState(ws: unknown): ws is SerializedWorkspaceState {
  if (!ws || typeof ws !== 'object') return false;
  return VERIFIED_INBOUND_SYNC_STATES.delete(ws as SerializedWorkspaceState);
}

/**
 * Constructs a deeply `Object.freeze`-d `SerializedWorkspaceState` directly from genuine `DawTrack`
 * domain objects (`[R24-MED-03]`, `[R25-LOW-01]`, `oop-extensible-craft` §3.2, §4.1), returning `null`
 * whenever any 5-way sync wire invariant is violated (e.g. `totalNotes > 32768`, `endTick > maxProjectTicks`,
 * stateful getters, or `Proxy`-wrapped `DawTrack`s).
 * Only snapshots verified and frozen here are branded in the unexported `TRUSTED_WORKSPACE_STATES` `WeakSet`.
 */
export function buildTrustedWorkspaceSnapshot(params: {
  readonly ppqn: number;
  readonly bpm: number;
  readonly beatsPerMeasure: number;
  readonly timeSigNumerator: number;
  readonly timeSigDenominator: number;
  readonly activeTrackId: string;
  readonly targetStartMeasure: number;
  readonly targetEndMeasure: number;
  readonly quantizeDivision: 4 | 8 | 16 | 32;
  readonly isTriplet: boolean;
  readonly domainTracks: readonly DawTrack[];
  readonly candidates: readonly GeneratedMidiCandidate[];
  readonly activeCandidateIndex: number;
  readonly activeCandidateNoteIds: ReadonlySet<string>;
  readonly activeCandidateRange: ActiveCandidateRange | null;
  readonly candidateOriginSlot: ActiveCandidateRange | null;
  readonly latestReason: GenerationReason | null;
}): SerializedWorkspaceState | null {
  try {
    const paramSnap = snapshotAllowedPlainObject(params, ALLOWED_TRUSTED_SNAPSHOT_PARAM_KEYS);
    if (!paramSnap) return null;
    const ppqn = paramSnap.get('ppqn');
    const bpm = paramSnap.get('bpm');
    const beatsPerMeasure = paramSnap.get('beatsPerMeasure');
    const timeSigNumerator = paramSnap.get('timeSigNumerator');
    const timeSigDenominator = paramSnap.get('timeSigDenominator');
    const activeTrackId = paramSnap.get('activeTrackId');
    const targetStartMeasure = paramSnap.get('targetStartMeasure');
    const targetEndMeasure = paramSnap.get('targetEndMeasure');
    const quantizeDivision = paramSnap.get('quantizeDivision');
    const isTriplet = paramSnap.get('isTriplet');
    const rawDomainTracks = paramSnap.get('domainTracks');
    const rawCandidates = paramSnap.get('candidates');
    const activeCandidateIndex = paramSnap.get('activeCandidateIndex');
    const activeCandidateNoteIds = paramSnap.get('activeCandidateNoteIds');
    const activeCandidateRange = paramSnap.get('activeCandidateRange');
    const candidateOriginSlot = paramSnap.get('candidateOriginSlot');
    const latestReason = paramSnap.get('latestReason');

    if (
      typeof ppqn !== 'number' ||
      !Number.isFinite(ppqn) ||
      ppqn < 24 ||
      ppqn > 32767 ||
      typeof bpm !== 'number' ||
      !Number.isFinite(bpm) ||
      bpm < 20 ||
      bpm > 300 ||
      typeof timeSigNumerator !== 'number' ||
      !Number.isInteger(timeSigNumerator) ||
      timeSigNumerator < 1 ||
      timeSigNumerator > 32 ||
      typeof timeSigDenominator !== 'number' ||
      ![1, 2, 4, 8, 16, 32].includes(timeSigDenominator) ||
      typeof beatsPerMeasure !== 'number' ||
      !Number.isFinite(beatsPerMeasure) ||
      beatsPerMeasure < 0.5 ||
      beatsPerMeasure > 16 ||
      Math.abs(beatsPerMeasure - canonicalBeatsPerMeasure(timeSigNumerator, timeSigDenominator)) >
        1e-6 ||
      typeof targetStartMeasure !== 'number' ||
      !Number.isFinite(targetStartMeasure) ||
      targetStartMeasure < 1 ||
      targetStartMeasure > 1024 ||
      typeof targetEndMeasure !== 'number' ||
      !Number.isFinite(targetEndMeasure) ||
      targetEndMeasure < targetStartMeasure ||
      targetEndMeasure > 1032 ||
      (quantizeDivision !== 4 &&
        quantizeDivision !== 8 &&
        quantizeDivision !== 16 &&
        quantizeDivision !== 32) ||
      typeof isTriplet !== 'boolean' ||
      typeof activeCandidateIndex !== 'number' ||
      !Number.isInteger(activeCandidateIndex) ||
      activeCandidateIndex < -1 ||
      activeCandidateIndex > 64 ||
      typeof activeTrackId !== 'string' ||
      activeTrackId.length > 128
    ) {
      return null;
    }

    const domainTracks = snapshotPlainDenseArray(rawDomainTracks, 64);
    if (!domainTracks || domainTracks.length === 0) {
      return null;
    }

    const effectivePpqn = Math.round(ppqn);
    const maxProjectTicks = 4128 * effectivePpqn;
    const maxSlotTick = 16_512 * effectivePpqn;

    const seenTrackIds = new Set<string>();
    const frozenTracks = new Array<SerializedTrackState>(domainTracks.length);
    let totalNotes = 0;
    const multiTrackNoteIds = domainTracks.length > 1 ? new Set<string>() : null;

    for (let ti = 0; ti < domainTracks.length; ti += 1) {
      const trk = domainTracks[ti];
      if (!isGenuineDawTrack(trk)) {
        return null;
      }
      const trkId = trk.id;
      if (typeof trkId !== 'string' || !trkId || seenTrackIds.has(trkId)) {
        return null;
      }
      const cTrkId = canonSyncId(trkId);
      if (!cTrkId || cTrkId !== trkId.trim() || seenTrackIds.has(cTrkId)) {
        return null;
      }
      seenTrackIds.add(cTrkId);

      const { track: frozenTrk, maxEndTick } = trk.toFrozenPlainSyncTrack();
      const noteLen = frozenTrk.notes.length;
      totalNotes += noteLen;
      if (totalNotes > 32_768 || (noteLen > 0 && maxEndTick > maxProjectTicks)) {
        return null;
      }
      if (multiTrackNoteIds !== null) {
        for (let ni = 0; ni < noteLen; ni += 1) {
          const nid = frozenTrk.notes[ni].id;
          if (multiTrackNoteIds.has(nid)) {
            return null;
          }
          multiTrackNoteIds.add(nid);
        }
      }
      frozenTracks[ti] = frozenTrk;
    }

    const canonActiveTrackId = canonSyncId(activeTrackId);
    if (
      !canonActiveTrackId ||
      canonActiveTrackId !== activeTrackId.trim() ||
      !seenTrackIds.has(canonActiveTrackId)
    ) {
      return null;
    }

    const freezeSlot = (
      slot: unknown,
    ): { valid: true; canonTrackId: string | null; value: ActiveCandidateRange | null } | { valid: false } => {
      if (slot === null) {
        return { valid: true, canonTrackId: null, value: null };
      }
      const slotSnap = snapshotAllowedPlainObject(slot, ALLOWED_SLOT_KEYS);
      if (!slotSnap) {
        return { valid: false };
      }
      const tid = slotSnap.get('trackId');
      const st = slotSnap.get('startTick');
      const et = slotSnap.get('endTick');
      if (typeof tid !== 'string' || tid.length > 128) return { valid: false };
      const cTid = canonSyncId(tid);
      if (
        !cTid ||
        cTid !== tid.trim() ||
        !seenTrackIds.has(cTid) ||
        typeof st !== 'number' ||
        !Number.isFinite(st) ||
        st < 0 ||
        typeof et !== 'number' ||
        !Number.isFinite(et) ||
        et < st ||
        et > maxSlotTick
      ) {
        return { valid: false };
      }
      return {
        valid: true,
        canonTrackId: cTid,
        value: Object.freeze({ trackId: tid, startTick: st, endTick: et }),
      };
    };

    const frozenActiveSlot = freezeSlot(activeCandidateRange);
    const frozenOriginSlot = freezeSlot(candidateOriginSlot);
    if (!frozenActiveSlot.valid || !frozenOriginSlot.valid) return null;
    if (
      frozenActiveSlot.canonTrackId !== null &&
      frozenOriginSlot.canonTrackId !== null &&
      frozenActiveSlot.canonTrackId !== frozenOriginSlot.canonTrackId
    ) {
      return null;
    }

    if (
      !activeCandidateNoteIds ||
      typeof activeCandidateNoteIds !== 'object' ||
      isRuntimeProxy(activeCandidateNoteIds)
    ) {
      return null;
    }
    let candIdArr: string[];
    if (
      activeCandidateNoteIds instanceof ReadonlyCanonicalIdSet &&
      Object.getPrototypeOf(activeCandidateNoteIds) === ReadonlyCanonicalIdSet.prototype
    ) {
      const sz = activeCandidateNoteIds.size;
      if (typeof sz !== 'number' || sz > 32_768) return null;
      candIdArr = new Array<string>(sz);
      let idx = 0;
      for (const cid of activeCandidateNoteIds) {
        if (idx >= 32_768) return null;
        candIdArr[idx++] = cid;
      }
      if (idx !== sz) {
        candIdArr.length = idx;
      }
    } else {
      const iterSrc = activeCandidateNoteIds as Iterable<unknown>;
      if (typeof iterSrc[Symbol.iterator] !== 'function') return null;
      candIdArr = [];
      const seenCandIds = new Set<string>();
      for (const cid of iterSrc) {
        if (typeof cid !== 'string' || cid.length > 128) return null;
        const cCanon = canonSyncId(cid);
        if (!cCanon || cCanon !== cid.trim() || seenCandIds.has(cCanon)) {
          return null;
        }
        seenCandIds.add(cCanon);
        candIdArr.push(cid);
        if (candIdArr.length > 32_768) return null;
      }
    }

    const candidates = snapshotPlainDenseArray(rawCandidates, 32);
    if (!candidates) {
      return null;
    }
    const frozenCandidates = new Array<{
      readonly index: number;
      readonly filename: string;
      readonly rawMidiBytes: Uint8Array;
      readonly reason: GenerationReason | null;
    }>(candidates.length);
    let totalCandBytes = 0;
    for (let ci = 0; ci < candidates.length; ci += 1) {
      const cSnap = snapshotAllowedPlainObject(candidates[ci], ALLOWED_TRUSTED_CANDIDATE_INPUT_KEYS);
      if (!cSnap) return null;
      const cIdx = cSnap.get('index');
      const cFilename = cSnap.get('filename');
      const cRawMidiBytes = cSnap.get('rawMidiBytes');
      const cReason = cSnap.get('reason');
      if (
        typeof cIdx !== 'number' ||
        !Number.isInteger(cIdx) ||
        cIdx < 0 ||
        cIdx > 64 ||
        typeof cFilename !== 'string' ||
        cFilename.trim().length === 0 ||
        cFilename.length > 256
      ) {
        return null;
      }
      const extracted = extractValidatedCandidateMidiBytes(
        cRawMidiBytes,
        true,
        MAX_CANDIDATE_MIDI_BYTES - totalCandBytes,
      );
      if (!extracted) return null;
      totalCandBytes += extracted.accountedBytes;
      if (totalCandBytes > MAX_CANDIDATE_MIDI_BYTES) return null;
      const parsedReason = parseValidatedGenerationReason(cReason, true);
      if (!parsedReason.valid) return null;
      frozenCandidates[ci] = Object.freeze({
        index: cIdx,
        filename: cFilename,
        rawMidiBytes: extracted.bytes,
        reason: deepFreezeGenerationReason(parsedReason.value),
      });
    }

    const parsedLatest = parseValidatedGenerationReason(latestReason, true);
    if (!parsedLatest.valid) return null;

    const frozenWs = Object.freeze({
      ppqn,
      bpm,
      beatsPerMeasure,
      timeSigNumerator,
      timeSigDenominator,
      activeTrackId,
      targetStartMeasure,
      targetEndMeasure,
      quantizeDivision,
      isTriplet,
      tracks: Object.freeze(frozenTracks),
      candidates: Object.freeze(frozenCandidates),
      activeCandidateIndex,
      activeCandidateNoteIds: Object.freeze(candIdArr),
      activeCandidateRange: frozenActiveSlot.value,
      candidateOriginSlot: frozenOriginSlot.value,
      latestReason: deepFreezeGenerationReason(parsedLatest.value),
    }) as SerializedWorkspaceState;

    TRUSTED_WORKSPACE_STATES.add(frozenWs);
    return frozenWs;
  } catch {
    return null;
  }
}

export class WorkspaceSyncBus {
  public readonly instanceId: string;
  private readonly _channel: BroadcastChannel | null;
  private readonly _listeners = new Set<(msg: WorkspaceSyncMessage) => void>();
  private readonly _seenMessageIds = new Set<string>();
  private _unsubscribeDesktopFocus: (() => void) | null = null;
  private _unsubscribeDesktopSync: (() => void) | null = null;
  private _lastFocusDispatch: { tab: WorkspaceTabId; at: number } | null = null;

  constructor(channelName = 'compass_ai_workspace_v1') {
    this.instanceId = `ws_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;

    if (typeof BroadcastChannel !== 'undefined') {
      try {
        this._channel = new BroadcastChannel(
          typeof channelName === 'string' && channelName ? channelName : 'compass_ai_workspace_v1',
        );
        this._channel.onmessage = (ev: MessageEvent<WorkspaceSyncMessage>) => {
          this._dispatchIncoming(ev?.data);
        };
      } catch {
        this._channel = null;
      }
    } else {
      this._channel = null;
    }

    if (typeof window !== 'undefined' && window.compassDesktop?.onFocusTabRequest) {
      try {
        this._unsubscribeDesktopFocus = window.compassDesktop.onFocusTabRequest((tab) => {
          if (tab !== 'compass-ai' && tab !== 'pianoroll') return;
          const msg: WorkspaceSyncMessage = {
            messageId: nextMessageId('electron-main'),
            type: 'FOCUS_TAB',
            tab,
            reason: 'desktop-ipc',
            senderId: 'electron-main',
          };
          this._dispatchIncoming(msg);
        });
      } catch {
        this._unsubscribeDesktopFocus = null;
      }
    }

    if (typeof window !== 'undefined' && window.compassDesktop?.onWorkspaceSync) {
      try {
        this._unsubscribeDesktopSync = window.compassDesktop.onWorkspaceSync((msg) => {
          this._dispatchIncoming(msg);
        });
      } catch {
        this._unsubscribeDesktopSync = null;
      }
    }
  }

  public publishFocusTab(tab: WorkspaceTabId, reason: string): void {
    try {
      if (tab !== 'compass-ai' && tab !== 'pianoroll') return;
      const cleanReason =
        typeof reason === 'string' ? truncateWellFormedUtf16(reason, 256) : '';
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      this._lastFocusDispatch = { tab, at: now };

      const msg: WorkspaceSyncMessage = {
        messageId: nextMessageId(this.instanceId),
        type: 'FOCUS_TAB',
        tab,
        reason: cleanReason,
        senderId: this.instanceId,
      };
      this._recordSeen(msg.messageId);
      try {
        this._channel?.postMessage(msg);
      } catch {
        // Ignore closed channel or structured-clone restriction
      }
      if (typeof window !== 'undefined') {
        if (!window.compassDesktop?.bringWindowToFront) {
          try {
            window.compassDesktop?.sendWorkspaceSync?.(msg);
          } catch {
            // Ignore desktop IPC bridge error
          }
        }
        try {
          void Promise.resolve(window.compassDesktop?.bringWindowToFront?.(tab)).catch(() => {});
        } catch {
          // Ignore focus IPC error
        }
      }
    } catch {
      // Never propagate outbound sync error to caller
    }
  }

  public publishGenerationStatus(isGenerating: boolean, statusDetail: string): void {
    try {
      if (typeof isGenerating !== 'boolean' || typeof statusDetail !== 'string') return;
      const msg: WorkspaceSyncMessage = {
        messageId: nextMessageId(this.instanceId),
        type: 'GENERATION_STATUS',
        isGenerating,
        statusDetail: truncateWellFormedUtf16(statusDetail, 512),
        senderId: this.instanceId,
      };
      this._postToPeers(msg);
    } catch {
      // Never propagate outbound sync error to caller
    }
  }

  public requestWorkspaceState(): void {
    try {
      const msg: WorkspaceSyncMessage = {
        messageId: nextMessageId(this.instanceId),
        type: 'REQUEST_WORKSPACE_STATE',
        senderId: this.instanceId,
      };
      this._postToPeers(msg);
    } catch {
      // Never propagate outbound sync error to caller
    }
  }

  public publishActiveController(
    controllerId: string,
    options?: { readonly forceDesktopIpc?: boolean },
  ): void {
    try {
      if (typeof controllerId !== 'string' || !VALID_SYNC_CONTROLLER_IDS.has(controllerId)) {
        return;
      }
      const forceDesktopIpc = Boolean(
        options &&
          typeof options === 'object' &&
          !isRuntimeProxy(options) &&
          Object.getOwnPropertyDescriptor(options, 'forceDesktopIpc')?.value === true,
      );
      const msg: WorkspaceSyncMessage = {
        messageId: nextMessageId(this.instanceId),
        type: 'SYNC_ACTIVE_CONTROLLER',
        controllerId: controllerId as SyncControllerId,
        senderId: this.instanceId,
      };
      this._postToPeers(msg, forceDesktopIpc);
    } catch {
      // Never propagate outbound sync error to caller
    }
  }

  public publishWorkspaceState(
    workspaceState: SerializedWorkspaceState,
    options?: { readonly forceDesktopIpc?: boolean },
  ): void {
    try {
      const forceDesktopIpc = Boolean(
        options &&
          typeof options === 'object' &&
          !isRuntimeProxy(options) &&
          Object.getOwnPropertyDescriptor(options, 'forceDesktopIpc')?.value === true,
      );
      if (
        !workspaceState ||
        typeof workspaceState !== 'object' ||
        Array.isArray(workspaceState) ||
        isRuntimeProxy(workspaceState)
      ) {
        return;
      }
      if (TRUSTED_WORKSPACE_STATES.has(workspaceState)) {
        const rawCands = workspaceState.candidates;
        const freshCands = new Array<{
          index: number;
          filename: string;
          rawMidiBytes: Uint8Array;
          reason: GenerationReason | null;
        }>(rawCands.length);
        let totalCandBytes = 0;
        for (let ci = 0; ci < rawCands.length; ci += 1) {
          const c = rawCands[ci];
          const extracted = extractValidatedCandidateMidiBytes(
            c.rawMidiBytes,
            true,
            MAX_CANDIDATE_MIDI_BYTES - totalCandBytes,
          );
          if (!extracted) {
            return;
          }
          totalCandBytes += extracted.accountedBytes;
          if (totalCandBytes > MAX_CANDIDATE_MIDI_BYTES) {
            return;
          }
          freshCands[ci] = {
            index: c.index,
            filename: c.filename,
            rawMidiBytes: extracted.bytes,
            reason: c.reason,
          };
        }
        const trustedMsg: WorkspaceSyncMessage = {
          messageId: nextMessageId(this.instanceId),
          type: 'SYNC_WORKSPACE_STATE',
          workspaceState:
            freshCands.length === 0
              ? workspaceState
              : {
                  ...workspaceState,
                  candidates: freshCands,
                },
          senderId: this.instanceId,
        };
        this._postToPeers(trustedMsg, forceDesktopIpc);
        return;
      }
      const candidateEnvelope = {
        messageId: nextMessageId(this.instanceId),
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState,
        senderId: this.instanceId,
      };
      const cleanMsg = parseAndCloneSyncMessage(candidateEnvelope, true, true);
      if (!cleanMsg) {
        return;
      }
      this._postToPeers(cleanMsg, forceDesktopIpc);
    } catch {
      // Never propagate outbound sync or DataCloneError to caller
    }
  }

  public subscribe(listener: (msg: WorkspaceSyncMessage) => void): () => void {
    if (typeof listener !== 'function') {
      return () => {};
    }
    this._listeners.add(listener);
    return () => {
      this._listeners.delete(listener);
    };
  }

  public dispose(): void {
    try {
      this._unsubscribeDesktopFocus?.();
    } catch {
      // Ignore
    }
    try {
      this._unsubscribeDesktopSync?.();
    } catch {
      // Ignore
    }
    try {
      this._channel?.close();
    } catch {
      // Ignore
    }
    this._listeners.clear();
    this._seenMessageIds.clear();
  }

  private _postToPeers(msg: WorkspaceSyncMessage, forceDesktopIpc = false): void {
    this._recordSeen(msg.messageId);
    try {
      this._channel?.postMessage(msg);
    } catch {
      // Ignore closed channel or structured-clone restriction
    }
    if (typeof window !== 'undefined') {
      try {
        if (
          !forceDesktopIpc &&
          msg.type === 'SYNC_WORKSPACE_STATE' &&
          window.compassDesktop?.hasPeerWindows?.() === false
        ) {
          return;
        }
        window.compassDesktop?.sendWorkspaceSync?.(msg);
      } catch {
        // Ignore desktop IPC bridge error
      }
    }
  }

  private _dispatchIncoming(msg: unknown): void {
    try {
      if (!msg || typeof msg !== 'object' || Array.isArray(msg) || isRuntimeProxy(msg)) {
        return;
      }
      const senderDesc = Object.getOwnPropertyDescriptor(msg, 'senderId');
      if (!senderDesc || !('value' in senderDesc) || typeof senderDesc.value !== 'string') {
        return;
      }
      if (senderDesc.value === this.instanceId) {
        return;
      }
      const idDesc = Object.getOwnPropertyDescriptor(msg, 'messageId');
      if (!idDesc || !('value' in idDesc) || typeof idDesc.value !== 'string') {
        return;
      }
      const rawMessageId = idDesc.value;
      if (rawMessageId.length === 0 || rawMessageId.length > 128) {
        return;
      }
      const trimmedMessageId = rawMessageId.trim();
      if (
        !trimmedMessageId ||
        this._seenMessageIds.has(trimmedMessageId) ||
        this._seenMessageIds.has(rawMessageId)
      ) {
        return;
      }
      const cleanMsg = parseAndCloneSyncMessage(msg, true, false);
      if (!cleanMsg || cleanMsg.senderId === this.instanceId) {
        return;
      }
      if (cleanMsg.messageId && this._seenMessageIds.has(cleanMsg.messageId)) {
        return;
      }
      if (cleanMsg.messageId) {
        this._recordSeen(cleanMsg.messageId);
      }
      if (cleanMsg.type === 'FOCUS_TAB') {
        const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
        if (
          this._lastFocusDispatch &&
          this._lastFocusDispatch.tab === cleanMsg.tab &&
          now - this._lastFocusDispatch.at < 80
        ) {
          return;
        }
        this._lastFocusDispatch = { tab: cleanMsg.tab, at: now };
      }
      const listeners = Array.from(this._listeners);
      const singleVerifiedSync =
        cleanMsg.type === 'SYNC_WORKSPACE_STATE' && listeners.length === 1;
      if (singleVerifiedSync) {
        VERIFIED_INBOUND_SYNC_STATES.add(cleanMsg.workspaceState);
      }
      try {
        for (const listener of listeners) {
          try {
            listener(cleanMsg);
          } catch {
            // Isolate listener errors so sibling subscribers stay healthy
          }
        }
      } finally {
        if (singleVerifiedSync) {
          VERIFIED_INBOUND_SYNC_STATES.delete(cleanMsg.workspaceState);
        }
      }
    } catch {
      // Never propagate malformed incoming sync payload errors
    }
  }

  private _recordSeen(id: string): void {
    if (this._seenMessageIds.size >= MAX_SEEN_IDS) {
      const oldest = this._seenMessageIds.values().next().value;
      if (oldest !== undefined) {
        this._seenMessageIds.delete(oldest);
      }
    }
    this._seenMessageIds.add(id);
  }
}
