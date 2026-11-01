/**
 * COMPASS AI Local Model Engine HTTP Client (`oop-extensible-craft`).
 * Communicates with FastAPI (`127.0.0.1:8000`), polls live token-generation telemetry
 * (`GET /generation_progress`) during `POST /generate`, unpacks multi-candidate ZIP archives
 * via JSZip, and decodes `X-Generation-Reason` URL-encoded JSON metadata.
 */

import JSZip from 'jszip';
import { TrackInstrument } from '../domain/DawTrack';
import {
  MAX_SAFE_MIDI_TICK,
  MidiNote,
  isGenuineMidiNote,
  truncateWellFormedUtf16,
} from '../domain/MidiNote';
import {
  MAX_PARSED_SMF_NOTES,
  MAX_PARSED_SMF_TRACKS,
  ParsedMidiFile,
  ParsedMidiTrack,
  StandardMidiCodec,
} from '../domain/midi/StandardMidiCodec';

const NODE_IS_PROXY: ((v: unknown) => boolean) | undefined = (() => {
  try {
    const g = globalThis as {
      process?: {
        getBuiltinModule?: (id: string) => { types?: { isProxy?: (v: unknown) => boolean } } | undefined;
      };
    };
    const fn = g.process?.getBuiltinModule?.('node:util')?.types?.isProxy;
    return typeof fn === 'function' ? fn : undefined;
  } catch {
    return undefined;
  }
})();

export function isRuntimeProxy(val: unknown): boolean {
  try {
    return NODE_IS_PROXY ? Boolean(NODE_IS_PROXY(val)) : false;
  } catch {
    return true;
  }
}

export interface ModelTagInfo {
  readonly instruments?: readonly string[];
  readonly model?: string;
  readonly task?: readonly string[];
  readonly type?: string;
  readonly version?: string;
  readonly genres?: readonly string[];
}

export interface ModelRuleInfo {
  readonly input_midi?: boolean;
  readonly multi_instrument?: boolean;
  readonly temperature?: boolean;
  readonly top_p?: boolean;
  readonly number_of_generation?: boolean;
  readonly send_context_past?: boolean;
  readonly send_context_condition?: boolean;
  readonly send_context_future?: boolean;
  readonly gen_measure_count?: boolean;
  readonly gen_note_dense?: boolean;
  readonly send_genre?: boolean;
}

export interface ModelMetadata {
  readonly model_name: string;
  readonly description?: string;
  readonly source?: string;
  readonly has_weights?: boolean;
  readonly tag?: ModelTagInfo;
  readonly rule?: ModelRuleInfo;
}

export interface HardwareStatusInfo {
  readonly device: string;
  readonly dtype: string;
  readonly backend_class: string;
  readonly user_models_dir: string;
  readonly bundled_models_dir: string;
  readonly available_models_count: number;
  readonly models_with_weights: readonly string[];
}

export interface GenerationProgressSnapshot {
  readonly active: boolean;
  readonly stage: string;
  readonly generated_tokens: number;
  readonly max_tokens: number;
  readonly message: string;
  readonly updated_at: number;
  readonly job_id?: string | null;
}

export interface GenerationReason {
  readonly source: 'model_cot' | 'request' | 'foundation';
  readonly task: string;
  readonly thinking: boolean;
  readonly instruments: readonly string[] | null;
  readonly note_density: Readonly<Record<string, number | null>> | null;
  readonly gen_measure_count: number | null;
  readonly genre: readonly string[] | null;
  readonly key: string | null;
}

/**
 * Sanitizes an untrusted `GenerationReason` object (from HTTP `X-Generation-Reason` headers,
 * external adapters, or direct controller calls) into a prototype-safe, schema-compliant
 * `GenerationReason` guaranteed to satisfy `ipc_security.cjs`, `WorkspaceSyncBus.ts`, and
 * `PianoRollView` (`[R8-MED-01]`, `[R8-LOW-01]`).
 */
export function sanitizeGenerationReason(raw: unknown): GenerationReason | null {
  try {
    if (
      raw === null ||
      raw === undefined ||
      typeof raw !== 'object' ||
      Array.isArray(raw) ||
      isRuntimeProxy(raw)
    ) {
      return null;
    }
    const proto = Object.getPrototypeOf(raw);
    if (proto !== null && proto !== Object.prototype) {
      return null;
    }
    const r = raw as Record<string, unknown>;
    const reasonKeys = [
      'task',
      'source',
      'thinking',
      'note_density',
      'gen_measure_count',
      'key',
      'instruments',
      'genre',
    ] as const;
    const snap = new Map<string, unknown>();
    for (let ki = 0; ki < reasonKeys.length; ki += 1) {
      const k = reasonKeys[ki];
      const desc = Object.getOwnPropertyDescriptor(r, k);
      if (desc) {
        if (!('value' in desc)) return null;
        snap.set(k, desc.value);
      }
    }

    const rawTask = snap.get('task');
    if (typeof rawTask !== 'string') {
      return null;
    }
    const cleanTask = truncateWellFormedUtf16(rawTask.trim(), 128).trim();
    if (cleanTask.length === 0) {
      return null;
    }
    const rawSource = snap.get('source');
    const cleanSource: GenerationReason['source'] =
      rawSource === 'model_cot' || rawSource === 'request' || rawSource === 'foundation'
        ? rawSource
        : 'request';
    const cleanThinking = Boolean(snap.get('thinking'));

    const sanitizeStrList = (val: unknown): string[] | null => {
      if (typeof val === 'string') {
        const s = truncateWellFormedUtf16(val.trim(), 64).trim();
        return s.length > 0 ? [s] : null;
      }
      if (
        !Array.isArray(val) ||
        isRuntimeProxy(val) ||
        Object.getPrototypeOf(val) !== Array.prototype
      ) {
        return null;
      }
      const lenDesc = Object.getOwnPropertyDescriptor(val, 'length');
      if (!lenDesc || !('value' in lenDesc) || typeof lenDesc.value !== 'number') {
        return null;
      }
      const out: string[] = [];
      const limit = Math.min(Math.max(0, Math.trunc(lenDesc.value)), 256);
      for (let i = 0; i < limit && out.length < 64; i += 1) {
        const elemDesc = Object.getOwnPropertyDescriptor(val, i);
        if (!elemDesc || !('value' in elemDesc)) continue;
        if (typeof elemDesc.value === 'string') {
          out.push(truncateWellFormedUtf16(elemDesc.value, 64));
        }
      }
      return out;
    };

    const rawNd = snap.get('note_density');
    let cleanDensity: Record<string, number | null> | null = null;
    if (
      rawNd &&
      typeof rawNd === 'object' &&
      !Array.isArray(rawNd) &&
      !isRuntimeProxy(rawNd)
    ) {
      const ndProto = Object.getPrototypeOf(rawNd);
      if (ndProto === null || ndProto === Object.prototype) {
        const nd = rawNd as Record<string, unknown>;
        const rawKeys = Object.getOwnPropertyNames(nd);
        cleanDensity = {};
        let count = 0;
        for (let i = 0; i < rawKeys.length && count < 32; i += 1) {
          const rawKey = rawKeys[i];
          if (rawKey === '__proto__' || rawKey === 'constructor' || rawKey === 'prototype') {
            continue;
          }
          const kDesc = Object.getOwnPropertyDescriptor(nd, rawKey);
          if (!kDesc || !('value' in kDesc)) {
            continue;
          }
          const k = truncateWellFormedUtf16(rawKey.trim(), 64).trim();
          if (
            k.length === 0 ||
            k === '__proto__' ||
            k === 'constructor' ||
            k === 'prototype'
          ) {
            continue;
          }
          const v = kDesc.value;
          Object.defineProperty(cleanDensity, k, {
            value: typeof v === 'number' && Number.isFinite(v) ? v : null,
            writable: true,
            enumerable: true,
            configurable: true,
          });
          count += 1;
        }
      }
    }

    const rawMc = snap.get('gen_measure_count');
    const cleanMeasureCount =
      typeof rawMc === 'number' && Number.isFinite(rawMc)
        ? Math.max(0, Math.min(512, rawMc))
        : null;

    const rawKey = snap.get('key');
    const cleanKey =
      typeof rawKey === 'string' ? truncateWellFormedUtf16(rawKey, 32) : null;

    const rawInst = snap.get('instruments');
    const rawGenre = snap.get('genre');

    return {
      source: cleanSource,
      task: cleanTask,
      thinking: cleanThinking,
      instruments: rawInst === null || rawInst === undefined ? null : sanitizeStrList(rawInst),
      note_density: cleanDensity,
      gen_measure_count: cleanMeasureCount,
      genre: rawGenre === null || rawGenre === undefined ? null : sanitizeStrList(rawGenre),
      key: cleanKey,
    };
  } catch {
    return null;
  }
}

export interface GenerateMetaPayload {
  readonly model_type: string;
  readonly task: string;
  readonly program: readonly string[];
  readonly tempo: number;
  readonly num_gems: number;
  readonly genfield_measure: number | null;
  readonly gen_note_dense: Readonly<Record<string, number>> | number | null;
  readonly p: number;
  readonly temperature: number;
  readonly cot_temperature: number;
  readonly thinking: boolean;
  readonly genre?: readonly string[] | null;
  readonly key?: string | null;
  readonly time_signature_numerator?: number;
  readonly time_signature_denominator?: number;
}

export interface GenerateMusicRequest {
  readonly meta: GenerateMetaPayload;
  readonly pastMidiBytes?: Uint8Array | null;
  readonly conditionsMidiBytes?: Uint8Array | null;
  readonly futureMidiBytes?: Uint8Array | null;
}

export interface GeneratedMidiCandidate {
  readonly index: number;
  readonly filename: string;
  readonly rawMidiBytes: Uint8Array;
  readonly parsedMidi: ParsedMidiFile;
  readonly reason: GenerationReason | null;
}

export const MAX_CANDIDATE_MIDI_BYTES = 2_000_000;
export const MAX_PLAIN_ARRAY_MIDI_BYTES = 4096;
export const MAX_FAILED_CANDIDATE_ATTEMPTS = 8;

const TYPED_ARRAY_PROTO = Object.getPrototypeOf(Uint8Array.prototype);
const TYPED_ARRAY_TAG_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  Symbol.toStringTag,
)?.get;
const TYPED_ARRAY_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'byteLength',
)?.get;
const TYPED_ARRAY_BYTEOFFSET_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'byteOffset',
)?.get;
const TYPED_ARRAY_BUFFER_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'buffer',
)?.get;
const ARRAYBUFFER_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength',
)?.get;
const ARRAYBUFFER_DETACHED_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'detached',
)?.get;
const NATIVE_UINT8ARRAY_CTOR_RE = /^function Uint8Array\(\)\s*\{\s*\[native code\]\s*\}$/;

function isGenuineUint8ArrayPrototype(proto: unknown): boolean {
  if (proto === Uint8Array.prototype) {
    return true;
  }
  if (
    typeof Buffer === 'function' &&
    Boolean((Buffer as unknown as { prototype?: unknown }).prototype) &&
    proto === (Buffer as unknown as { prototype?: unknown }).prototype
  ) {
    return true;
  }
  if (!proto || typeof proto !== 'object' || isRuntimeProxy(proto)) {
    return false;
  }
  const typedArrProto = Object.getPrototypeOf(proto);
  if (!typedArrProto || typeof typedArrProto !== 'object' || isRuntimeProxy(typedArrProto)) {
    return false;
  }
  const objProto = Object.getPrototypeOf(typedArrProto);
  if (
    !objProto ||
    typeof objProto !== 'object' ||
    isRuntimeProxy(objProto) ||
    Object.getPrototypeOf(objProto) !== null
  ) {
    return false;
  }
  const bpeDesc = Object.getOwnPropertyDescriptor(proto, 'BYTES_PER_ELEMENT');
  if (
    !bpeDesc ||
    bpeDesc.value !== 1 ||
    bpeDesc.writable !== false ||
    bpeDesc.enumerable !== false ||
    bpeDesc.configurable !== false
  ) {
    return false;
  }
  const ctorDesc = Object.getOwnPropertyDescriptor(proto, 'constructor');
  if (
    !ctorDesc ||
    !('value' in ctorDesc) ||
    typeof ctorDesc.value !== 'function' ||
    isRuntimeProxy(ctorDesc.value)
  ) {
    return false;
  }
  const ctor = ctorDesc.value as (...args: unknown[]) => unknown;
  if (!NATIVE_UINT8ARRAY_CTOR_RE.test(Function.prototype.toString.call(ctor))) {
    return false;
  }
  const protoDesc = Object.getOwnPropertyDescriptor(ctor, 'prototype');
  return Boolean(
    protoDesc &&
      'value' in protoDesc &&
      protoDesc.value === proto &&
      protoDesc.writable === false &&
      protoDesc.configurable === false,
  );
}

export function inspectGenuineUint8ArrayPayload(raw: unknown): {
  viewLen: number;
  byteOffset: number;
  backingLen: number;
  backingBuf: ArrayBuffer;
  u8: Uint8Array;
} | null {
  try {
    if (!raw || typeof raw !== 'object' || isRuntimeProxy(raw) || !ArrayBuffer.isView(raw)) {
      return null;
    }
    if (!isGenuineUint8ArrayPrototype(Object.getPrototypeOf(raw))) {
      return null;
    }
    if (Object.getOwnPropertySymbols(raw).length !== 0) {
      return null;
    }
    if (TYPED_ARRAY_TAG_GETTER?.call(raw) !== 'Uint8Array') {
      return null;
    }
    const viewLen = TYPED_ARRAY_BYTELENGTH_GETTER!.call(raw) as number;
    const byteOffset = TYPED_ARRAY_BYTEOFFSET_GETTER!.call(raw) as number;
    const backingBuf = TYPED_ARRAY_BUFFER_GETTER!.call(raw) as ArrayBuffer;
    if (ARRAYBUFFER_DETACHED_GETTER && ARRAYBUFFER_DETACHED_GETTER.call(backingBuf) === true) {
      return null;
    }
    const backingLen = ARRAYBUFFER_BYTELENGTH_GETTER!.call(backingBuf) as number;
    if (
      typeof viewLen !== 'number' ||
      typeof byteOffset !== 'number' ||
      typeof backingLen !== 'number' ||
      byteOffset < 0 ||
      viewLen < 0 ||
      byteOffset + viewLen > backingLen
    ) {
      return null;
    }
    if (viewLen <= 4096) {
      const ownNames = Object.getOwnPropertyNames(raw);
      if (ownNames.length !== viewLen) {
        return null;
      }
      for (let i = 0; i < viewLen; i += 1) {
        const desc = Object.getOwnPropertyDescriptor(raw, i);
        if (!desc || !('value' in desc)) {
          return null;
        }
      }
    }
    return { viewLen, byteOffset, backingLen, backingBuf, u8: raw as Uint8Array };
  } catch {
    return null;
  }
}

/**
 * Copies `u8Info` bytes into a brand-new `Uint8Array(viewLen)` (`byteOffset === 0`,
 * `buffer.byteLength === viewLen`, `Object.getPrototypeOf === Uint8Array.prototype`, zero expando/symbol keys)
 * WITHOUT invoking `backingBuf.slice(...)` (`[R13-MED-02]`, `[R14-MED-03]` detached `ArrayBuffer` safe).
 */
export function cloneCleanUint8Array(u8Info: {
  backingBuf: ArrayBuffer;
  byteOffset: number;
  viewLen: number;
}): Uint8Array {
  try {
    if (!u8Info || typeof u8Info !== 'object' || Array.isArray(u8Info) || isRuntimeProxy(u8Info)) {
      return new Uint8Array(0);
    }
    const infoProto = Object.getPrototypeOf(u8Info);
    if (infoProto !== null && infoProto !== Object.prototype) {
      return new Uint8Array(0);
    }
    const bufDesc = Object.getOwnPropertyDescriptor(u8Info, 'backingBuf');
    const offDesc = Object.getOwnPropertyDescriptor(u8Info, 'byteOffset');
    const lenDesc = Object.getOwnPropertyDescriptor(u8Info, 'viewLen');
    if (
      !bufDesc ||
      !('value' in bufDesc) ||
      !offDesc ||
      !('value' in offDesc) ||
      !lenDesc ||
      !('value' in lenDesc)
    ) {
      return new Uint8Array(0);
    }
    const backingBuf = bufDesc.value as ArrayBuffer | null | undefined;
    const byteOffset = offDesc.value;
    const viewLen = lenDesc.value;
    if (!backingBuf || typeof backingBuf !== 'object' || isRuntimeProxy(backingBuf)) {
      return new Uint8Array(0);
    }
    if (ARRAYBUFFER_DETACHED_GETTER && ARRAYBUFFER_DETACHED_GETTER.call(backingBuf) === true) {
      return new Uint8Array(0);
    }
    const backingLen = ARRAYBUFFER_BYTELENGTH_GETTER
      ? (ARRAYBUFFER_BYTELENGTH_GETTER.call(backingBuf) as number)
      : -1;
    if (
      typeof byteOffset !== 'number' ||
      !Number.isInteger(byteOffset) ||
      byteOffset < 0 ||
      typeof viewLen !== 'number' ||
      !Number.isInteger(viewLen) ||
      viewLen < 0 ||
      viewLen > MAX_CANDIDATE_MIDI_BYTES ||
      byteOffset + viewLen > backingLen
    ) {
      return new Uint8Array(0);
    }
    const out = new Uint8Array(viewLen);
    out.set(new Uint8Array(backingBuf, byteOffset, viewLen));
    return out;
  } catch {
    return new Uint8Array(0);
  }
}

type CandidateMidiExtractOutcome =
  | { readonly status: 'ok'; readonly bytes: Uint8Array; readonly accountedBytes: number }
  | { readonly status: 'over_budget' }
  | { readonly status: 'invalid' };

function extractCandidateMidiBytesWithBudget(
  raw: unknown,
  allowOversizedBackingForCompaction: boolean,
  maxAllowedBytes: number,
  distinguishBudgetOverflow: boolean,
): CandidateMidiExtractOutcome {
  try {
    const effectiveCap =
      typeof maxAllowedBytes === 'number' && Number.isFinite(maxAllowedBytes)
        ? Math.min(MAX_CANDIDATE_MIDI_BYTES, Math.floor(maxAllowedBytes))
        : MAX_CANDIDATE_MIDI_BYTES;
    if (!distinguishBudgetOverflow && effectiveCap < 22) {
      return { status: 'invalid' };
    }
    if (!raw || typeof raw !== 'object' || isRuntimeProxy(raw)) {
      return { status: 'invalid' };
    }
    const u8Info = inspectGenuineUint8ArrayPayload(raw);
    if (u8Info !== null) {
      if (
        u8Info.viewLen < 22 ||
        u8Info.viewLen > MAX_CANDIDATE_MIDI_BYTES ||
        (!allowOversizedBackingForCompaction && u8Info.backingLen > MAX_CANDIDATE_MIDI_BYTES)
      ) {
        return { status: 'invalid' };
      }
      // Zero-copy 4-byte `MThd` header check BEFORE copying up to 2 MB (`[R20-MED-01]`)
      const hdr = new Uint8Array(u8Info.backingBuf, u8Info.byteOffset, 4);
      if (hdr[0] !== 0x4d || hdr[1] !== 0x54 || hdr[2] !== 0x68 || hdr[3] !== 0x64) {
        return { status: 'invalid' };
      }
      const accountedBytes = allowOversizedBackingForCompaction
        ? u8Info.viewLen
        : Math.max(u8Info.viewLen, u8Info.backingLen);
      if (
        u8Info.viewLen > effectiveCap ||
        (!allowOversizedBackingForCompaction && u8Info.backingLen > effectiveCap) ||
        accountedBytes > effectiveCap
      ) {
        return distinguishBudgetOverflow ? { status: 'over_budget' } : { status: 'invalid' };
      }
      const copied = cloneCleanUint8Array(u8Info);
      if (
        copied.byteLength !== u8Info.viewLen ||
        copied[0] !== 0x4d ||
        copied[1] !== 0x54 ||
        copied[2] !== 0x68 ||
        copied[3] !== 0x64
      ) {
        return { status: 'invalid' };
      }
      return { status: 'ok', bytes: copied, accountedBytes };
    }
    const plainCap = distinguishBudgetOverflow
      ? MAX_PLAIN_ARRAY_MIDI_BYTES
      : Math.min(MAX_PLAIN_ARRAY_MIDI_BYTES, effectiveCap);
    if (plainCap < 22) {
      return { status: 'invalid' };
    }
    const rawByteArr = readStrictDenseArray(raw, plainCap);
    if (rawByteArr !== null) {
      const len = rawByteArr.length;
      if (
        len < 22 ||
        rawByteArr[0] !== 0x4d ||
        rawByteArr[1] !== 0x54 ||
        rawByteArr[2] !== 0x68 ||
        rawByteArr[3] !== 0x64
      ) {
        return { status: 'invalid' };
      }
      for (let i = 0; i < len; i += 1) {
        const b = rawByteArr[i];
        if (typeof b !== 'number' || !Number.isInteger(b) || b < 0 || b > 255) {
          return { status: 'invalid' };
        }
      }
      if (len > effectiveCap) {
        return distinguishBudgetOverflow ? { status: 'over_budget' } : { status: 'invalid' };
      }
      const out = new Uint8Array(len);
      for (let i = 0; i < len; i += 1) {
        out[i] = rawByteArr[i] as number;
      }
      return { status: 'ok', bytes: out, accountedBytes: len };
    }
    return { status: 'invalid' };
  } catch {
    return { status: 'invalid' };
  }
}

export function extractValidatedCandidateMidiBytes(
  raw: unknown,
  allowOversizedBackingForCompaction = false,
  maxAllowedBytes = MAX_CANDIDATE_MIDI_BYTES,
): { bytes: Uint8Array; accountedBytes: number } | null {
  const res = extractCandidateMidiBytesWithBudget(
    raw,
    allowOversizedBackingForCompaction,
    maxAllowedBytes,
    false,
  );
  return res.status === 'ok' ? { bytes: res.bytes, accountedBytes: res.accountedBytes } : null;
}

/**
 * Single-pass validation + byte-size accounting without cloning (`[R14-MED-01]` zero `.length` re-read, `[R19-MED-01]`).
 */
export function measureValidatedCandidateMidiBytes(
  raw: unknown,
  maxAllowedBytes = MAX_CANDIDATE_MIDI_BYTES,
): number | null {
  try {
    const effectiveCap =
      typeof maxAllowedBytes === 'number' && Number.isFinite(maxAllowedBytes)
        ? Math.min(MAX_CANDIDATE_MIDI_BYTES, Math.floor(maxAllowedBytes))
        : MAX_CANDIDATE_MIDI_BYTES;
    if (effectiveCap < 22) return null;
    if (!raw || typeof raw !== 'object' || isRuntimeProxy(raw)) return null;
    const u8Info = inspectGenuineUint8ArrayPayload(raw);
    if (u8Info !== null) {
      if (
        u8Info.viewLen < 22 ||
        u8Info.viewLen > effectiveCap ||
        u8Info.backingLen > effectiveCap
      ) {
        return null;
      }
      const hdr = new Uint8Array(u8Info.backingBuf, u8Info.byteOffset, 4);
      if (hdr[0] !== 0x4d || hdr[1] !== 0x54 || hdr[2] !== 0x68 || hdr[3] !== 0x64) {
        return null;
      }
      return Math.max(u8Info.viewLen, u8Info.backingLen);
    }
    const plainCap = Math.min(MAX_PLAIN_ARRAY_MIDI_BYTES, effectiveCap);
    const rawByteArr = readStrictDenseArray(raw, plainCap);
    if (rawByteArr !== null) {
      const len = rawByteArr.length;
      if (len < 22) return null;
      for (let i = 0; i < len; i += 1) {
        const b = rawByteArr[i];
        if (typeof b !== 'number' || !Number.isInteger(b) || b < 0 || b > 255) return null;
        if (i === 0 && b !== 0x4d) return null;
        if (i === 1 && b !== 0x54) return null;
        if (i === 2 && b !== 0x68) return null;
        if (i === 3 && b !== 0x64) return null;
      }
      return len;
    }
    return null;
  } catch {
    return null;
  }
}

export function isValidCandidateMidiBytesPayload(
  raw: unknown,
): raw is Uint8Array | readonly number[] {
  return measureValidatedCandidateMidiBytes(raw) !== null;
}

const ALLOWED_PARSED_MIDI_KEYS: ReadonlySet<string> = new Set([
  'format',
  'ppqn',
  'bpm',
  'beatsPerMeasure',
  'timeSigNumerator',
  'timeSigDenominator',
  'tracks',
]);
const ALLOWED_PARSED_TRACK_KEYS: ReadonlySet<string> = new Set([
  'index',
  'name',
  'program',
  'isDrum',
  'instrument',
  'notes',
]);
const ALLOWED_PARSED_NOTE_KEYS: ReadonlySet<string> = new Set([
  'id',
  'pitch',
  'startTick',
  'durationTicks',
  'velocity',
  'channel',
]);
const VALID_PARSED_INSTRUMENTS: ReadonlySet<string> = new Set([
  'PIANO',
  'SAX',
  'BASS',
  'VIOLIN',
  'SYNTH',
  'GUITAR',
  'DRUMS',
]);

function readStrictPlainObjectFields(
  obj: unknown,
  allowedKeys: ReadonlySet<string>,
  allowMidiNoteProto = false,
): Record<string, unknown> | null {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj) || isRuntimeProxy(obj)) {
    return null;
  }
  const proto = Object.getPrototypeOf(obj);
  const protoOk =
    proto === Object.prototype ||
    proto === null ||
    (allowMidiNoteProto && obj instanceof MidiNote && proto === MidiNote.prototype);
  if (!protoOk) {
    return null;
  }
  if (Object.getOwnPropertySymbols(obj).length > 0) {
    return null;
  }
  const keys = Object.getOwnPropertyNames(obj);
  if (keys.length > allowedKeys.size) {
    return null;
  }
  const out = Object.create(null) as Record<string, unknown>;
  for (let i = 0; i < keys.length; i += 1) {
    const k = keys[i];
    if (k === '__proto__' || k === 'constructor' || k === 'prototype' || !allowedKeys.has(k)) {
      return null;
    }
    const desc = Object.getOwnPropertyDescriptor(obj, k);
    if (!desc || !('value' in desc)) {
      return null;
    }
    out[k] = desc.value;
  }
  return out;
}

function parseCanonicalArrayIndex(key: string, maxExclusive: number): number {
  const kLen = key.length;
  if (kLen === 0 || kLen > 10) return -1;
  const c0 = key.charCodeAt(0);
  if (kLen === 1) {
    const d = c0 - 48;
    return d >= 0 && d < maxExclusive ? d : -1;
  }
  if (c0 < 49 || c0 > 57) return -1;
  let n = c0 - 48;
  for (let i = 1; i < kLen; i += 1) {
    const d = key.charCodeAt(i) - 48;
    if (d < 0 || d > 9) return -1;
    n = n * 10 + d;
  }
  return n < maxExclusive ? n : -1;
}

function readStrictDenseArray(arr: unknown, maxLen: number): unknown[] | null {
  if (
    !arr ||
    typeof arr !== 'object' ||
    isRuntimeProxy(arr) ||
    !Array.isArray(arr) ||
    Object.getPrototypeOf(arr) !== Array.prototype
  ) {
    return null;
  }
  const lenDesc = Object.getOwnPropertyDescriptor(arr, 'length');
  if (!lenDesc || !('value' in lenDesc)) return null;
  const len = lenDesc.value;
  if (typeof len !== 'number' || !Number.isInteger(len) || len < 0 || len > maxLen) {
    return null;
  }
  if (Object.getOwnPropertySymbols(arr).length > 0) return null;
  const ownNames = Object.getOwnPropertyNames(arr);
  if (ownNames.length !== len + 1) {
    return null;
  }
  let ordered = ownNames[len] === 'length';
  if (ordered) {
    for (let i = 0; i < len; i += 1) {
      if (parseCanonicalArrayIndex(ownNames[i], len) !== i) {
        ordered = false;
        break;
      }
    }
  }
  if (!ordered) {
    const seen = new Uint8Array(len);
    let sawLength = false;
    for (let i = 0; i < ownNames.length; i += 1) {
      const k = ownNames[i];
      if (k === 'length') {
        if (sawLength) return null;
        sawLength = true;
        continue;
      }
      const idx = parseCanonicalArrayIndex(k, len);
      if (idx < 0 || seen[idx] !== 0) return null;
      seen[idx] = 1;
    }
    if (!sawLength) return null;
  }
  const out: unknown[] = new Array(len);
  for (let i = 0; i < len; i += 1) {
    const desc = Object.getOwnPropertyDescriptor(arr, i);
    if (!desc || !('value' in desc)) return null;
    out[i] = desc.value;
  }
  return out;
}

/**
 * Single-pass deep sanitizer/cloner for `ParsedMidiFile` (`[R13-MED-01]`).
 * Rejects `Proxy`, getters/accessors, sparse arrays, symbol/expando keys, and invalid note fields,
 * and reconstructs fresh `ParsedMidiFile`, fresh `ParsedMidiTrack[]`, and genuine `MidiNote[]`
 * instances (including plain JSON-deserialized note POJOs).
 */
function sanitizeParsedMidiFile(val: unknown): ParsedMidiFile | null {
  try {
    const root = readStrictPlainObjectFields(val, ALLOWED_PARSED_MIDI_KEYS, false);
    if (!root) return null;

    const rawPpqn = root.ppqn;
    if (
      typeof rawPpqn !== 'number' ||
      !Number.isInteger(rawPpqn) ||
      rawPpqn < 1 ||
      rawPpqn > 32767
    ) {
      return null;
    }
    const rawFormat = root.format;
    if (
      rawFormat !== undefined &&
      (typeof rawFormat !== 'number' ||
        !Number.isInteger(rawFormat) ||
        (rawFormat !== 0 && rawFormat !== 1 && rawFormat !== 2))
    ) {
      return null;
    }
    const rawBpm = root.bpm;
    if (
      rawBpm !== undefined &&
      (typeof rawBpm !== 'number' || !Number.isFinite(rawBpm) || rawBpm <= 0 || rawBpm > 999)
    ) {
      return null;
    }
    const rawBeats = root.beatsPerMeasure;
    if (
      rawBeats !== undefined &&
      (typeof rawBeats !== 'number' || !Number.isFinite(rawBeats) || rawBeats <= 0 || rawBeats > 64)
    ) {
      return null;
    }
    const rawTsNum = root.timeSigNumerator;
    if (
      rawTsNum !== undefined &&
      (typeof rawTsNum !== 'number' || !Number.isInteger(rawTsNum) || rawTsNum < 1 || rawTsNum > 64)
    ) {
      return null;
    }
    const rawTsDen = root.timeSigDenominator;
    if (
      rawTsDen !== undefined &&
      (typeof rawTsDen !== 'number' || ![1, 2, 4, 8, 16, 32].includes(rawTsDen))
    ) {
      return null;
    }

    const rawTracks = readStrictDenseArray(root.tracks, MAX_PARSED_SMF_TRACKS);
    if (!rawTracks) return null;

    interface ValidatedPojoNoteSpec {
      readonly kind: 'pojo';
      readonly id: string | undefined;
      readonly pitch: number;
      readonly startTick: number;
      readonly durationTicks: number;
      readonly velocity: number;
      readonly channel: number;
    }
    type ValidatedNoteSpec =
      | { readonly kind: 'genuine'; readonly note: MidiNote }
      | ValidatedPojoNoteSpec;
    interface ValidatedTrackSpec {
      readonly index: number;
      readonly name: string;
      readonly program: number;
      readonly isDrum: boolean;
      readonly instrument: TrackInstrument;
      readonly directGenuineNotes: MidiNote[] | null;
      readonly noteSpecs: readonly ValidatedNoteSpec[];
    }

    // Pass 1 (`[R20-MED-01]`): Validate 100% of track and note scalar descriptors BEFORE
    // instantiating any `MidiNote` / UUIDs so malformed `parsedMidi` POJOs allocate 0 `MidiNote`s.
    const validatedTracks: ValidatedTrackSpec[] = [];
    let totalNotes = 0;
    for (let ti = 0; ti < rawTracks.length; ti += 1) {
      const trk = readStrictPlainObjectFields(rawTracks[ti], ALLOWED_PARSED_TRACK_KEYS, false);
      if (!trk) return null;

      const tIndex = trk.index;
      if (
        tIndex !== undefined &&
        (typeof tIndex !== 'number' || !Number.isInteger(tIndex) || tIndex < 0 || tIndex > 65_535)
      ) {
        return null;
      }
      const tName = trk.name;
      if (tName !== undefined && (typeof tName !== 'string' || tName.length > 512)) {
        return null;
      }
      const tProg = trk.program;
      if (
        tProg !== undefined &&
        (typeof tProg !== 'number' || !Number.isInteger(tProg) || tProg < 0 || tProg > 127)
      ) {
        return null;
      }
      const tDrum = trk.isDrum;
      if (tDrum !== undefined && typeof tDrum !== 'boolean') {
        return null;
      }
      const tInst = trk.instrument;
      if (tInst !== undefined && (typeof tInst !== 'string' || !VALID_PARSED_INSTRUMENTS.has(tInst))) {
        return null;
      }

      const rawNotes = readStrictDenseArray(trk.notes, MAX_PARSED_SMF_NOTES - totalNotes);
      if (!rawNotes || totalNotes + rawNotes.length > MAX_PARSED_SMF_NOTES) {
        return null;
      }
      totalNotes += rawNotes.length;

      let allGenuine = true;
      for (let ni = 0; ni < rawNotes.length; ni += 1) {
        if (!isGenuineMidiNote(rawNotes[ni])) {
          allGenuine = false;
          break;
        }
      }

      let noteSpecs: ValidatedNoteSpec[] = [];
      if (!allGenuine) {
        noteSpecs = new Array(rawNotes.length);
        for (let ni = 0; ni < rawNotes.length; ni += 1) {
          const rawNoteItem = rawNotes[ni];
          if (isGenuineMidiNote(rawNoteItem)) {
            noteSpecs[ni] = { kind: 'genuine', note: rawNoteItem };
            continue;
          }
          const nf = readStrictPlainObjectFields(rawNoteItem, ALLOWED_PARSED_NOTE_KEYS, true);
          if (!nf) return null;

          const pitch = nf.pitch;
          const startTick = nf.startTick;
          const durationTicks = nf.durationTicks;
          const velocity = nf.velocity;
          const channel = nf.channel;
          const nId = nf.id;

          let cleanNId: string | undefined;
          if (nId !== undefined) {
            if (typeof nId !== 'string' || nId.length > 128) return null;
            const trimmedNId = nId.trim();
            if (trimmedNId.length === 0) return null;
            const canonNId = truncateWellFormedUtf16(trimmedNId, 128).trim();
            if (!canonNId) return null;
            cleanNId = canonNId;
          }

          if (
            typeof pitch !== 'number' ||
            !Number.isFinite(pitch) ||
            pitch < 0 ||
            pitch > 127 ||
            typeof startTick !== 'number' ||
            !Number.isFinite(startTick) ||
            startTick < 0 ||
            startTick > MAX_SAFE_MIDI_TICK ||
            typeof durationTicks !== 'number' ||
            !Number.isFinite(durationTicks) ||
            durationTicks < 1 ||
            durationTicks > MAX_SAFE_MIDI_TICK ||
            (velocity !== undefined &&
              (typeof velocity !== 'number' ||
                !Number.isFinite(velocity) ||
                velocity < 1 ||
                velocity > 127)) ||
            (channel !== undefined &&
              (typeof channel !== 'number' ||
                !Number.isFinite(channel) ||
                channel < 0 ||
                channel > 15))
          ) {
            return null;
          }

          noteSpecs[ni] = {
            kind: 'pojo',
            id: cleanNId,
            pitch,
            startTick,
            durationTicks,
            velocity: typeof velocity === 'number' ? velocity : 100,
            channel: typeof channel === 'number' ? channel : 0,
          };
        }
      }

      validatedTracks.push({
        index: typeof tIndex === 'number' ? tIndex : ti,
        name:
          typeof tName === 'string' && tName.trim()
            ? truncateWellFormedUtf16(tName.trim(), 256).trim() || `Track ${ti + 1}`
            : `Track ${ti + 1}`,
        program: typeof tProg === 'number' ? tProg : 0,
        isDrum: typeof tDrum === 'boolean' ? tDrum : false,
        instrument:
          typeof tInst === 'string' && VALID_PARSED_INSTRUMENTS.has(tInst)
            ? (tInst as TrackInstrument)
            : 'PIANO',
        directGenuineNotes: allGenuine ? (rawNotes as MidiNote[]) : null,
        noteSpecs,
      });
    }

    // Pass 2: Materialize clean `MidiNote` instances only after full structural validation passed.
    const cleanTracks: ParsedMidiTrack[] = new Array(validatedTracks.length);
    for (let ti = 0; ti < validatedTracks.length; ti += 1) {
      const vt = validatedTracks[ti];
      let cleanNotes: MidiNote[];
      if (vt.directGenuineNotes !== null) {
        cleanNotes = vt.directGenuineNotes;
      } else {
        cleanNotes = new Array(vt.noteSpecs.length);
        for (let ni = 0; ni < vt.noteSpecs.length; ni += 1) {
          const spec = vt.noteSpecs[ni];
          cleanNotes[ni] =
            spec.kind === 'genuine'
              ? spec.note
              : new MidiNote({
                  id: spec.id,
                  pitch: spec.pitch,
                  startTick: spec.startTick,
                  durationTicks: spec.durationTicks,
                  velocity: spec.velocity,
                  channel: spec.channel,
                });
        }
      }
      cleanTracks[ti] = {
        index: vt.index,
        name: vt.name,
        program: vt.program,
        isDrum: vt.isDrum,
        instrument: vt.instrument,
        notes: cleanNotes,
      };
    }

    return {
      format: typeof rawFormat === 'number' ? rawFormat : 1,
      ppqn: rawPpqn,
      bpm: typeof rawBpm === 'number' ? rawBpm : 120,
      beatsPerMeasure: typeof rawBeats === 'number' ? rawBeats : 4,
      ...(typeof rawTsNum === 'number' ? { timeSigNumerator: rawTsNum } : {}),
      ...(typeof rawTsDen === 'number' ? { timeSigDenominator: rawTsDen } : {}),
      tracks: cleanTracks,
    };
  } catch {
    return null;
  }
}

export function normalizeGeneratedCandidatesBatch(
  candidates: readonly GeneratedMidiCandidate[],
  reasons?: readonly GenerationReason[],
  preserveExistingReasonsOnly = false,
): GeneratedMidiCandidate[] {
  try {
    if (!Array.isArray(candidates) || isRuntimeProxy(candidates)) {
      return [];
    }
    const candLenDesc = Object.getOwnPropertyDescriptor(candidates, 'length');
    const rawCandLen =
      candLenDesc &&
      'value' in candLenDesc &&
      typeof candLenDesc.value === 'number' &&
      Number.isFinite(candLenDesc.value)
        ? Math.trunc(candLenDesc.value)
        : 0;
    if (rawCandLen <= 0) {
      return [];
    }
    let reasonsLen = 0;
    let fallbackFirstReason: GenerationReason | null = null;
    if (!preserveExistingReasonsOnly && Array.isArray(reasons) && !isRuntimeProxy(reasons)) {
      try {
        const rLenDesc = Object.getOwnPropertyDescriptor(reasons, 'length');
        if (
          rLenDesc &&
          'value' in rLenDesc &&
          typeof rLenDesc.value === 'number' &&
          Number.isFinite(rLenDesc.value) &&
          rLenDesc.value > 0
        ) {
          reasonsLen = Math.min(256, Math.trunc(rLenDesc.value));
          const r0Desc = Object.getOwnPropertyDescriptor(reasons, 0);
          if (r0Desc && 'value' in r0Desc) {
            fallbackFirstReason = sanitizeGenerationReason(r0Desc.value);
          }
        }
      } catch {
        reasonsLen = 0;
        fallbackFirstReason = null;
      }
    }
    const out: GeneratedMidiCandidate[] = [];
    let totalBytes = 0;
    let failedAttempts = 0;
    const scanLimit = Math.min(rawCandLen, 256);
    for (let i = 0; i < scanLimit && out.length < 32; i += 1) {
      if (failedAttempts >= MAX_FAILED_CANDIDATE_ATTEMPTS) {
        break;
      }
      if (MAX_CANDIDATE_MIDI_BYTES - totalBytes < 22) {
        break;
      }
      try {
        const candDesc = Object.getOwnPropertyDescriptor(candidates, i);
        if (!candDesc || !('value' in candDesc)) {
          failedAttempts += 1;
          continue;
        }
        const c = candDesc.value as Record<string, unknown> | null | undefined;
        if (!c || typeof c !== 'object' || Array.isArray(c) || isRuntimeProxy(c)) {
          failedAttempts += 1;
          continue;
        }
        const cProto = Object.getPrototypeOf(c);
        if (cProto !== null && cProto !== Object.prototype) {
          failedAttempts += 1;
          continue;
        }

        const rawBytesDesc = Object.getOwnPropertyDescriptor(c, 'rawMidiBytes');
        if (!rawBytesDesc || !('value' in rawBytesDesc)) {
          failedAttempts += 1;
          continue;
        }
        const extractedOutcome = extractCandidateMidiBytesWithBudget(
          rawBytesDesc.value,
          false,
          MAX_CANDIDATE_MIDI_BYTES - totalBytes,
          true,
        );
        if (extractedOutcome.status === 'over_budget') {
          break;
        }
        if (extractedOutcome.status !== 'ok') {
          failedAttempts += 1;
          continue;
        }

        const compactBytes = extractedOutcome.bytes;
        if (totalBytes + extractedOutcome.accountedBytes > MAX_CANDIDATE_MIDI_BYTES) {
          break;
        }

        const parsedDesc = Object.getOwnPropertyDescriptor(c, 'parsedMidi');
        const candidateParsed =
          parsedDesc && 'value' in parsedDesc ? parsedDesc.value : undefined;
        const sanitizedParsed = sanitizeParsedMidiFile(candidateParsed);
        const parsedMidi: ParsedMidiFile =
          sanitizedParsed !== null
            ? sanitizedParsed
            : StandardMidiCodec.parseMidi(compactBytes.buffer as ArrayBuffer);

        const fnameDesc = Object.getOwnPropertyDescriptor(c, 'filename');
        const rawFname = fnameDesc && 'value' in fnameDesc ? fnameDesc.value : undefined;
        const idx = out.length;
        const cleanFilename =
          typeof rawFname === 'string' && rawFname.trim()
            ? truncateWellFormedUtf16(rawFname.trim(), 256).trim() || `output_${idx}.mid`
            : `output_${idx}.mid`;

        const reasonDesc = Object.getOwnPropertyDescriptor(c, 'reason');
        const rawCandReason = reasonDesc && 'value' in reasonDesc ? reasonDesc.value : null;
        let fallbackIndexedReason: GenerationReason | null = null;
        if (!preserveExistingReasonsOnly && reasonsLen > 0 && i < reasonsLen) {
          try {
            const riDesc = Object.getOwnPropertyDescriptor(reasons!, i);
            if (riDesc && 'value' in riDesc) {
              fallbackIndexedReason = sanitizeGenerationReason(riDesc.value);
            }
          } catch {
            fallbackIndexedReason = null;
          }
        }
        const cleanReason = preserveExistingReasonsOnly
          ? sanitizeGenerationReason(rawCandReason)
          : (sanitizeGenerationReason(rawCandReason) ??
              fallbackIndexedReason ??
              fallbackFirstReason ??
              null);

        totalBytes += extractedOutcome.accountedBytes;
        out.push({
          index: idx,
          filename: cleanFilename,
          rawMidiBytes: compactBytes,
          parsedMidi,
          reason: cleanReason,
        });
      } catch {
        failedAttempts += 1;
        continue;
      }
    }
    return out;
  } catch {
    return [];
  }
}

export interface GenerateMusicResult {
  readonly candidates: readonly GeneratedMidiCandidate[];
  readonly reasons: readonly GenerationReason[];
}

export const ALL_SFT_GENRES: readonly string[] = [
  '80s', '90s', 'alternative', 'ambient', 'blues', 'celtic', 'chillout',
  'classical', 'country', 'dance', 'drumnbass', 'easylistening', 'electronic',
  'electropop', 'experimental', 'folk', 'funk', 'hiphop', 'house', 'indie',
  'instrumentalpop', 'instrumentalrock', 'jazz', 'jazzfusion', 'latin', 'lounge',
  'metal', 'newage', 'orchestral', 'pop', 'popfolk', 'poprock', 'punkrock',
  'reggae', 'rock', 'soundtrack', 'swing', 'symphonic', 'synthpop', 'techno',
  'trance', 'world',
];

export const ALL_MUSICAL_KEYS: readonly string[] = [
  'CM', 'Cm', 'C#M', 'C#m', 'DM', 'Dm', 'EbM', 'Ebm',
  'EM', 'Em', 'FM', 'Fm', 'F#M', 'F#m', 'GM', 'Gm',
  'AbM', 'Abm', 'AM', 'Am', 'BbM', 'Bbm', 'BM', 'Bm',
];

export class CompassApiClient {
  private _baseUrl: string;
  private _handshakeToken: string;
  private _hasAuthoritativeToken = false;
  private _desktopBackendReady = false;
  private _refreshConfigPromise: Promise<boolean> | null = null;
  private _refreshConfigIsForce = false;

  constructor(baseUrl?: string, handshakeToken?: string) {
    const inferredOrigin =
      typeof window !== 'undefined' &&
      /^https?:$/i.test(window.location.protocol) &&
      window.location.port !== '5173' &&
      window.location.port !== '3000' &&
      window.location.origin !== 'null'
        ? window.location.origin
        : null;

    const desktopCfg =
      typeof window !== 'undefined' ? window.compassDesktop?.getApiConfig?.() : undefined;

    const envUrl =
      (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_COMPASS_API_URL ||
      desktopCfg?.apiBaseUrl ||
      (typeof window !== 'undefined' &&
        (window.compassDesktop?.apiBaseUrl ||
          (window as unknown as { COMPASS_API_URL?: string }).COMPASS_API_URL)) ||
      inferredOrigin ||
      'http://127.0.0.1:8000';
    const safeEnvUrl =
      typeof envUrl === 'string' && envUrl.trim() ? envUrl.trim() : 'http://127.0.0.1:8000';
    const safeExplicitUrl =
      typeof baseUrl === 'string' && baseUrl.trim() ? baseUrl.trim() : null;
    this._baseUrl = (safeExplicitUrl ?? safeEnvUrl).replace(/\/+$/, '');

    const envHandshake =
      (import.meta as unknown as { env?: Record<string, string> }).env
        ?.VITE_COMPASS_HANDSHAKE_TOKEN ||
      desktopCfg?.handshakeToken ||
      (typeof window !== 'undefined' ? (window.compassDesktop?.handshakeToken ?? '') : '');
    const safeEnvHandshake = typeof envHandshake === 'string' ? envHandshake : '';
    if (typeof handshakeToken === 'string') {
      this._handshakeToken = handshakeToken.trim();
      this._hasAuthoritativeToken = true;
    } else {
      this._handshakeToken = safeEnvHandshake.trim();
      this._hasAuthoritativeToken = Boolean(this._handshakeToken);
    }
    this._desktopBackendReady = desktopCfg?.backendReady === true;
  }

  public get baseUrl(): string {
    return this._baseUrl;
  }

  public setBaseUrl(url: string): void {
    if (typeof url !== 'string') return;
    const next = url.trim().replace(/\/+$/, '');
    if (!next) return;
    if (next !== this._baseUrl) {
      this._baseUrl = next;
      this._desktopBackendReady = false;
    }
  }

  public get handshakeToken(): string {
    if (this._hasAuthoritativeToken) {
      return this._handshakeToken;
    }
    if (typeof window !== 'undefined') {
      const dynamicTok =
        window.compassDesktop?.getApiConfig?.()?.handshakeToken ??
        window.compassDesktop?.handshakeToken ??
        '';
      return typeof dynamicTok === 'string' ? dynamicTok.trim() : '';
    }
    return this._handshakeToken;
  }

  public setHandshakeToken(token: string): void {
    if (typeof token !== 'string') return;
    this._handshakeToken = token.trim();
    this._hasAuthoritativeToken = true;
  }

  public get desktopBackendReady(): boolean {
    if (this._desktopBackendReady) return true;
    if (typeof window !== 'undefined') {
      return window.compassDesktop?.getApiConfig?.()?.backendReady === true;
    }
    return false;
  }

  public setDesktopBackendReady(ready: boolean): void {
    this._desktopBackendReady = Boolean(ready);
  }

  private _authHeaders(extra?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = { ...(extra ?? {}) };
    const token = this.handshakeToken;
    if (token) {
      headers['X-Compass-Handshake'] = token;
    }
    return headers;
  }

  /**
   * Lazily probes Electron `compass:get-api-config` (`ensureBackendAuthenticatedLazy`) whenever
   * neither `handshakeToken` nor `backendReady` is populated yet, or after an `HTTP 403`
   * (`[R6B-MED-01]`, `[R6E-MED-01]`), preventing permanent `403` lockout on cold boots while
   * avoiding redundant `/health` probes against `auth_required: false` explicit backends.
   */
  private async _refreshDesktopHandshake(force = false): Promise<boolean> {
    if (typeof window === 'undefined' || !window.compassDesktop?.requestApiConfig) {
      return false;
    }
    if (!force) {
      if (this.handshakeToken || this._desktopBackendReady) {
        return false;
      }
      const cachedCfg = window.compassDesktop?.getApiConfig?.();
      if (cachedCfg?.backendReady === true) {
        if (cachedCfg.apiBaseUrl) {
          const nextBase = cachedCfg.apiBaseUrl.trim().replace(/\/+$/, '');
          if (nextBase) {
            this._baseUrl = nextBase;
          }
        }
        if (typeof cachedCfg.handshakeToken === 'string') {
          this._handshakeToken = cachedCfg.handshakeToken.trim();
          this._hasAuthoritativeToken = true;
        }
        this._desktopBackendReady = true;
        return false;
      }
    } else {
      this._desktopBackendReady = false;
      if (this._refreshConfigPromise && !this._refreshConfigIsForce) {
        await this._refreshConfigPromise;
      }
    }
    if (this._refreshConfigPromise) {
      return this._refreshConfigPromise;
    }
    this._refreshConfigIsForce = force;
    this._refreshConfigPromise = (async () => {
      try {
        const prevBaseUrl = this._baseUrl;
        const prevToken = this.handshakeToken;
        const cfg = force
          ? await window.compassDesktop?.requestApiConfig?.(true)
          : await window.compassDesktop?.requestApiConfig?.();
        if (cfg?.apiBaseUrl) {
          const nextBase = cfg.apiBaseUrl.trim().replace(/\/+$/, '');
          if (nextBase) {
            this._baseUrl = nextBase;
          }
        }
        if (typeof cfg?.handshakeToken === 'string') {
          this._handshakeToken = cfg.handshakeToken.trim();
          this._hasAuthoritativeToken = true;
        }
        this._desktopBackendReady = cfg?.backendReady === true;
        return this._baseUrl !== prevBaseUrl || this.handshakeToken !== prevToken;
      } catch {
        return false;
      } finally {
        this._refreshConfigPromise = null;
        this._refreshConfigIsForce = false;
      }
    })();
    return this._refreshConfigPromise;
  }

  private _buildGenerateFormData(request: GenerateMusicRequest, jobId: string): FormData {
    const formData = new FormData();
    formData.append('meta', JSON.stringify(request.meta));
    formData.append('job_id', jobId);

    if (request.pastMidiBytes && request.pastMidiBytes.byteLength > 0) {
      formData.append(
        'past_midi',
        new Blob([request.pastMidiBytes as Uint8Array<ArrayBuffer>], { type: 'audio/midi' }),
        'past_context.mid',
      );
    }
    if (request.conditionsMidiBytes && request.conditionsMidiBytes.byteLength > 0) {
      formData.append(
        'conditions_midi',
        new Blob([request.conditionsMidiBytes as Uint8Array<ArrayBuffer>], { type: 'audio/midi' }),
        'conditions_context.mid',
      );
    }
    if (request.futureMidiBytes && request.futureMidiBytes.byteLength > 0) {
      formData.append(
        'future_midi',
        new Blob([request.futureMidiBytes as Uint8Array<ArrayBuffer>], { type: 'audio/midi' }),
        'future_context.mid',
      );
    }
    return formData;
  }

  public async checkHealth(signal?: AbortSignal): Promise<{
    status: string;
    engine: string;
    device: string;
    dtype: string;
    models_dir_ready: boolean;
    available_models_count: number;
    auth_required?: boolean;
  }> {
    const res = await fetch(`${this._baseUrl}/health`, {
      method: 'GET',
      headers: this._authHeaders(),
      signal,
    });
    if (!res.ok) {
      throw new Error(`推論エンジンの応答エラー (HTTP ${res.status})`);
    }
    const rawJson = (await res.json()) as unknown;
    if (
      !rawJson ||
      typeof rawJson !== 'object' ||
      Array.isArray(rawJson) ||
      isRuntimeProxy(rawJson) ||
      (Object.getPrototypeOf(rawJson) !== null &&
        Object.getPrototypeOf(rawJson) !== Object.prototype)
    ) {
      throw new Error('推論エンジンのヘルスチェック応答が不正です。');
    }
    const readOwnHealthField = (key: string): unknown => {
      try {
        const d = Object.getOwnPropertyDescriptor(rawJson, key);
        return d && 'value' in d ? d.value : undefined;
      } catch {
        return undefined;
      }
    };
    const rawStatus = readOwnHealthField('status');
    if (typeof rawStatus !== 'string' || !rawStatus.trim()) {
      throw new Error('推論エンジンのヘルスチェック応答が不正です。');
    }
    const rawEngine = readOwnHealthField('engine');
    const rawDevice = readOwnHealthField('device');
    const rawDtype = readOwnHealthField('dtype');
    const rawDirReady = readOwnHealthField('models_dir_ready');
    const rawCount = readOwnHealthField('available_models_count');
    const rawAuthReq = readOwnHealthField('auth_required');
    return {
      status: CompassApiClient._sanitizeSafeLabel(rawStatus, 'ok', 64),
      engine: CompassApiClient._sanitizeSafeLabel(rawEngine, 'MORTM', 64),
      device: CompassApiClient._sanitizeSafeLabel(rawDevice, 'cpu', 64),
      dtype: CompassApiClient._sanitizeSafeLabel(rawDtype, 'float32', 64),
      models_dir_ready: typeof rawDirReady === 'boolean' ? rawDirReady : false,
      available_models_count:
        typeof rawCount === 'number' && Number.isFinite(rawCount)
          ? Math.max(0, Math.min(100_000, Math.trunc(rawCount)))
          : 0,
      ...(typeof rawAuthReq === 'boolean' ? { auth_required: rawAuthReq } : {}),
    };
  }

  private static _sanitizeSafeLabel(raw: unknown, fallback: string, maxLen = 128): string {
    if (typeof raw !== 'string') return fallback;
    const trimmed = truncateWellFormedUtf16(raw.trim(), maxLen).trim();
    if (!trimmed || /^(nan|undefined|null|[+-]?infinity)$/i.test(trimmed)) return fallback;
    return trimmed;
  }

  private static _sanitizeModelList(rawPayload: unknown): ModelMetadata[] {
    let rawArr: unknown[] | null = null;
    if (Array.isArray(rawPayload) && !isRuntimeProxy(rawPayload)) {
      rawArr = rawPayload;
    } else if (
      rawPayload &&
      typeof rawPayload === 'object' &&
      !Array.isArray(rawPayload) &&
      !isRuntimeProxy(rawPayload)
    ) {
      const mDesc = Object.getOwnPropertyDescriptor(rawPayload, 'models');
      if (mDesc && 'value' in mDesc && Array.isArray(mDesc.value) && !isRuntimeProxy(mDesc.value)) {
        rawArr = mDesc.value;
      }
    }
    if (!rawArr) return [];

    const lenDesc = Object.getOwnPropertyDescriptor(rawArr, 'length');
    const rawLen =
      lenDesc && 'value' in lenDesc && typeof lenDesc.value === 'number' && Number.isFinite(lenDesc.value)
        ? Math.min(Math.max(0, Math.trunc(lenDesc.value)), 256)
        : 0;
    const out: ModelMetadata[] = [];

    const readStrArr = (v: unknown): string[] | undefined => {
      if (!Array.isArray(v) || isRuntimeProxy(v)) return undefined;
      const ld = Object.getOwnPropertyDescriptor(v, 'length');
      const l =
        ld && 'value' in ld && typeof ld.value === 'number' && Number.isFinite(ld.value)
          ? Math.min(Math.max(0, Math.trunc(ld.value)), 128)
          : 0;
      const clean: string[] = [];
      for (let i = 0; i < l; i += 1) {
        const ed = Object.getOwnPropertyDescriptor(v, i);
        if (ed && 'value' in ed && typeof ed.value === 'string') {
          const s = CompassApiClient._sanitizeSafeLabel(ed.value, '', 64);
          if (s) clean.push(s);
        }
      }
      return clean;
    };

    for (let i = 0; i < rawLen; i += 1) {
      const itemDesc = Object.getOwnPropertyDescriptor(rawArr, i);
      if (!itemDesc || !('value' in itemDesc)) continue;
      const item = itemDesc.value;
      if (!item || typeof item !== 'object' || Array.isArray(item) || isRuntimeProxy(item)) {
        continue;
      }
      const nameDesc = Object.getOwnPropertyDescriptor(item, 'model_name');
      if (!nameDesc || !('value' in nameDesc) || typeof nameDesc.value !== 'string') {
        continue;
      }
      const modelName = CompassApiClient._sanitizeSafeLabel(nameDesc.value, '', 128);
      if (!modelName) continue;

      const descProp = Object.getOwnPropertyDescriptor(item, 'description');
      const description =
        descProp && 'value' in descProp && typeof descProp.value === 'string'
          ? truncateWellFormedUtf16(descProp.value.trim(), 512)
          : undefined;
      const srcProp = Object.getOwnPropertyDescriptor(item, 'source');
      const source =
        srcProp && 'value' in srcProp && typeof srcProp.value === 'string'
          ? CompassApiClient._sanitizeSafeLabel(srcProp.value, '', 64) || undefined
          : undefined;
      const hwProp = Object.getOwnPropertyDescriptor(item, 'has_weights');
      const hasWeights =
        hwProp && 'value' in hwProp && typeof hwProp.value === 'boolean'
          ? hwProp.value
          : undefined;

      let cleanTag: ModelTagInfo | undefined;
      const tagProp = Object.getOwnPropertyDescriptor(item, 'tag');
      if (
        tagProp &&
        'value' in tagProp &&
        tagProp.value &&
        typeof tagProp.value === 'object' &&
        !Array.isArray(tagProp.value) &&
        !isRuntimeProxy(tagProp.value)
      ) {
        const t = tagProp.value as Record<string, unknown>;
        const readTagVal = (k: string) => {
          const d = Object.getOwnPropertyDescriptor(t, k);
          return d && 'value' in d ? d.value : undefined;
        };
        const tModel = readTagVal('model');
        const tType = readTagVal('type');
        const tVer = readTagVal('version');
        cleanTag = {
          instruments: readStrArr(readTagVal('instruments')),
          model: typeof tModel === 'string' ? CompassApiClient._sanitizeSafeLabel(tModel, '', 64) || undefined : undefined,
          task: readStrArr(readTagVal('task')),
          type: typeof tType === 'string' ? CompassApiClient._sanitizeSafeLabel(tType, '', 64) || undefined : undefined,
          version: typeof tVer === 'string' ? CompassApiClient._sanitizeSafeLabel(tVer, '', 32) || undefined : undefined,
          genres: readStrArr(readTagVal('genres')),
        };
      }

      let cleanRule: ModelRuleInfo | undefined;
      const ruleProp = Object.getOwnPropertyDescriptor(item, 'rule');
      if (
        ruleProp &&
        'value' in ruleProp &&
        ruleProp.value &&
        typeof ruleProp.value === 'object' &&
        !Array.isArray(ruleProp.value) &&
        !isRuntimeProxy(ruleProp.value)
      ) {
        const rl = ruleProp.value as Record<string, unknown>;
        const readBool = (k: string): boolean | undefined => {
          const d = Object.getOwnPropertyDescriptor(rl, k);
          return d && 'value' in d && typeof d.value === 'boolean' ? d.value : undefined;
        };
        cleanRule = {
          input_midi: readBool('input_midi'),
          multi_instrument: readBool('multi_instrument'),
          temperature: readBool('temperature'),
          top_p: readBool('top_p'),
          number_of_generation: readBool('number_of_generation'),
          send_context_past: readBool('send_context_past'),
          send_context_condition: readBool('send_context_condition'),
          send_context_future: readBool('send_context_future'),
          gen_measure_count: readBool('gen_measure_count'),
          gen_note_dense: readBool('gen_note_dense'),
          send_genre: readBool('send_genre'),
        };
      }

      out.push({
        model_name: modelName,
        ...(description !== undefined ? { description } : {}),
        ...(source !== undefined ? { source } : {}),
        ...(hasWeights !== undefined ? { has_weights: hasWeights } : {}),
        ...(cleanTag !== undefined ? { tag: cleanTag } : {}),
        ...(cleanRule !== undefined ? { rule: cleanRule } : {}),
      });
    }
    return out;
  }

  public async getHardwareInfo(signal?: AbortSignal): Promise<HardwareStatusInfo> {
    await this._refreshDesktopHandshake(false);
    let res = await fetch(`${this._baseUrl}/hardware_info`, {
      method: 'GET',
      headers: this._authHeaders(),
      signal,
    });
    if (res.status === 403 && (await this._refreshDesktopHandshake(true))) {
      res = await fetch(`${this._baseUrl}/hardware_info`, {
        method: 'GET',
        headers: this._authHeaders(),
        signal,
      });
    }
    if (!res.ok) {
      throw new Error(`ハードウェア情報の取得に失敗しました (HTTP ${res.status})`);
    }
    const rawJson = (await res.json()) as unknown;
    const isPlainObj = Boolean(
      rawJson &&
        typeof rawJson === 'object' &&
        !Array.isArray(rawJson) &&
        !isRuntimeProxy(rawJson) &&
        (Object.getPrototypeOf(rawJson) === null ||
          Object.getPrototypeOf(rawJson) === Object.prototype),
    );
    const readOwnHwField = (key: string): unknown => {
      if (!isPlainObj) return undefined;
      try {
        const d = Object.getOwnPropertyDescriptor(rawJson as object, key);
        return d && 'value' in d ? d.value : undefined;
      } catch {
        return undefined;
      }
    };
    const rawDevice = readOwnHwField('device');
    const rawDtype = readOwnHwField('dtype');
    const rawBackend = readOwnHwField('backend_class');
    const rawUserDir = readOwnHwField('user_models_dir');
    const rawBundledDir = readOwnHwField('bundled_models_dir');
    const rawModelsCount = readOwnHwField('available_models_count');
    const rawWeights = readOwnHwField('models_with_weights');

    const cleanWeights: string[] = [];
    if (Array.isArray(rawWeights) && !isRuntimeProxy(rawWeights)) {
      const ld = Object.getOwnPropertyDescriptor(rawWeights, 'length');
      const l =
        ld && 'value' in ld && typeof ld.value === 'number' && Number.isFinite(ld.value)
          ? Math.min(Math.max(0, Math.trunc(ld.value)), 256)
          : 0;
      for (let i = 0; i < l; i += 1) {
        const ed = Object.getOwnPropertyDescriptor(rawWeights, i);
        if (ed && 'value' in ed && typeof ed.value === 'string') {
          const s = CompassApiClient._sanitizeSafeLabel(ed.value, '', 128);
          if (s) cleanWeights.push(s);
        }
      }
    }
    return {
      device: CompassApiClient._sanitizeSafeLabel(rawDevice, 'cpu', 64),
      dtype: CompassApiClient._sanitizeSafeLabel(rawDtype, 'float32', 64),
      backend_class: CompassApiClient._sanitizeSafeLabel(rawBackend, 'CpuBackend', 64),
      user_models_dir: CompassApiClient._sanitizeSafeLabel(rawUserDir, '', 512),
      bundled_models_dir: CompassApiClient._sanitizeSafeLabel(rawBundledDir, '', 512),
      available_models_count:
        typeof rawModelsCount === 'number' &&
        Number.isFinite(rawModelsCount) &&
        rawModelsCount >= 0
          ? Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(rawModelsCount))
          : 0,
      models_with_weights: cleanWeights,
    };
  }

  public async getGenerationProgress(
    signal?: AbortSignal,
    jobId?: string,
  ): Promise<GenerationProgressSnapshot | null> {
    try {
      await this._refreshDesktopHandshake(false);
      let normalizedJobId = '';
      if (jobId !== undefined) {
        const trimmedJobId = jobId.trim();
        if (!trimmedJobId || trimmedJobId.length > 4096 || !/^[A-Za-z0-9_.:-]+$/.test(trimmedJobId)) {
          return null;
        }
        normalizedJobId = trimmedJobId.slice(0, 128);
      }
      const query = normalizedJobId ? `?job_id=${encodeURIComponent(normalizedJobId)}` : '';
      const extraHeaders = normalizedJobId ? { 'X-Compass-Job-Id': normalizedJobId } : undefined;
      let res = await fetch(`${this._baseUrl}/generation_progress${query}`, {
        method: 'GET',
        headers: this._authHeaders(extraHeaders),
        signal,
      });
      if (res && res.status === 403 && (await this._refreshDesktopHandshake(true))) {
        res = await fetch(`${this._baseUrl}/generation_progress${query}`, {
          method: 'GET',
          headers: this._authHeaders(extraHeaders),
          signal,
        });
      }
      if (!res || !res.ok) {
        return null;
      }
      const rawJson = (await res.json()) as unknown;
      if (
        !rawJson ||
        typeof rawJson !== 'object' ||
        Array.isArray(rawJson) ||
        isRuntimeProxy(rawJson)
      ) {
        return null;
      }
      const proto = Object.getPrototypeOf(rawJson);
      if (proto !== null && proto !== Object.prototype) {
        return null;
      }
      const readOwnProgField = (
        key: string,
      ): { ok: true; val: unknown } | { ok: false } => {
        try {
          const desc = Object.getOwnPropertyDescriptor(rawJson, key);
          if (!desc) return { ok: true, val: undefined };
          if (!('value' in desc)) return { ok: false };
          return { ok: true, val: desc.value };
        } catch {
          return { ok: false };
        }
      };
      const activeRes = readOwnProgField('active');
      const stageRes = readOwnProgField('stage');
      const genTokRes = readOwnProgField('generated_tokens');
      const maxTokRes = readOwnProgField('max_tokens');
      const msgRes = readOwnProgField('message');
      const updRes = readOwnProgField('updated_at');
      const jobRes = readOwnProgField('job_id');
      if (
        !activeRes.ok ||
        !stageRes.ok ||
        !genTokRes.ok ||
        !maxTokRes.ok ||
        !msgRes.ok ||
        !updRes.ok ||
        !jobRes.ok ||
        typeof activeRes.val !== 'boolean'
      ) {
        return null;
      }

      let cleanJobId: string | undefined;
      if (jobRes.val !== undefined && jobRes.val !== null) {
        if (typeof jobRes.val !== 'string') {
          return null;
        }
        const trimmedJob = jobRes.val.trim();
        if (
          !trimmedJob ||
          trimmedJob.length > 128 ||
          !/^[A-Za-z0-9_.:-]+$/.test(trimmedJob)
        ) {
          return null;
        }
        if (normalizedJobId && trimmedJob !== normalizedJobId) {
          return null;
        }
        cleanJobId = trimmedJob;
      } else if (normalizedJobId && activeRes.val) {
        return null;
      }

      const generatedTokens =
        typeof genTokRes.val === 'number' &&
        Number.isFinite(genTokRes.val) &&
        genTokRes.val >= 0
          ? Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(genTokRes.val))
          : 0;
      const maxTokens =
        typeof maxTokRes.val === 'number' &&
        Number.isFinite(maxTokRes.val) &&
        maxTokRes.val >= 0
          ? Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(maxTokRes.val))
          : 0;
      const updatedAt =
        typeof updRes.val === 'number' && Number.isFinite(updRes.val) && updRes.val >= 0
          ? updRes.val
          : 0;

      return {
        active: activeRes.val,
        stage: CompassApiClient._sanitizeSafeLabel(stageRes.val, '', 64),
        generated_tokens: generatedTokens,
        max_tokens: maxTokens,
        message: CompassApiClient._sanitizeSafeLabel(msgRes.val, '', 512),
        updated_at: updatedAt,
        ...(cleanJobId !== undefined ? { job_id: cleanJobId } : {}),
      };
    } catch {
      return null;
    }
  }

  public async getModelInfo(signal?: AbortSignal): Promise<ModelMetadata[]> {
    await this._refreshDesktopHandshake(false);
    let res = await fetch(`${this._baseUrl}/model_info`, {
      method: 'POST',
      headers: this._authHeaders({ 'Content-Type': 'application/json' }),
      body: '{}',
      signal,
    });
    if (res.status === 403 && (await this._refreshDesktopHandshake(true))) {
      res = await fetch(`${this._baseUrl}/model_info`, {
        method: 'POST',
        headers: this._authHeaders({ 'Content-Type': 'application/json' }),
        body: '{}',
        signal,
      });
    }
    if (!res.ok) {
      throw new Error(`モデル一覧の取得に失敗しました (HTTP ${res.status})`);
    }
    const rawPayload = (await res.json()) as unknown;
    return CompassApiClient._sanitizeModelList(rawPayload);
  }

  public async rescanModels(signal?: AbortSignal): Promise<ModelMetadata[]> {
    await this._refreshDesktopHandshake(false);
    let res = await fetch(`${this._baseUrl}/rescan_models`, {
      method: 'POST',
      headers: this._authHeaders(),
      signal,
    });
    if (res.status === 403 && (await this._refreshDesktopHandshake(true))) {
      res = await fetch(`${this._baseUrl}/rescan_models`, {
        method: 'POST',
        headers: this._authHeaders(),
        signal,
      });
    }
    if (!res.ok) {
      throw new Error(`モデルフォルダの再スキャンに失敗しました (HTTP ${res.status})`);
    }
    const payload = (await res.json()) as unknown;
    return CompassApiClient._sanitizeModelList(payload);
  }

  public async generateMusic(
    request: GenerateMusicRequest,
    onProgress?: (statusText: string) => void,
    signal?: AbortSignal,
  ): Promise<GenerateMusicResult> {
    const createAbortError = () => {
      const err = new Error('The operation was aborted.');
      err.name = 'AbortError';
      return err;
    };
    const emitProgress = (statusText: string) => {
      if (!signal?.aborted) {
        onProgress?.(statusText);
      }
    };
    if (signal?.aborted) {
      throw createAbortError();
    }
    emitProgress('コンテキストMIDIと生成パラメータをパッケージング中...');
    await this._refreshDesktopHandshake(false);
    if (signal?.aborted) {
      throw createAbortError();
    }

    const jobId =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;

    const formData = this._buildGenerateFormData(request, jobId);

    const useCotLog = request.meta.thinking && request.meta.task !== 'meta';
    emitProgress(
      useCotLog
        ? `${request.meta.model_type} で Chain-of-Thought 構造推論と音符生成を開始...`
        : `${request.meta.model_type} でローカル自己回帰推論を開始...`,
    );

    let lastReportedProgressMsg = '';
    let pollInFlight = false;
    let settled = false;
    const pollAbortCtrl = new AbortController();

    const onCallerAbort = () => {
      pollAbortCtrl.abort();
      void fetch(`${this._baseUrl}/cancel_generate`, {
        method: 'POST',
        headers: this._authHeaders({
          'Content-Type': 'application/json',
          'X-Compass-Job-Id': jobId,
        }),
        body: JSON.stringify({ job_id: jobId }),
      }).catch(() => {});
    };
    if (signal) {
      if (signal.aborted) {
        onCallerAbort();
      } else {
        signal.addEventListener('abort', onCallerAbort, { once: true });
      }
    }

    const pollInterval =
      onProgress && (!signal || !signal.aborted)
        ? setInterval(() => {
            if (settled || pollInFlight || signal?.aborted || pollAbortCtrl.signal.aborted) {
              return;
            }
            pollInFlight = true;
            void this.getGenerationProgress(pollAbortCtrl.signal, jobId)
              .then((snap) => {
                if (settled || pollAbortCtrl.signal.aborted) return;
                if (
                  !snap ||
                  !snap.active ||
                  typeof snap.message !== 'string' ||
                  snap.message.length === 0
                ) {
                  return;
                }
                if (snap.message !== lastReportedProgressMsg) {
                  lastReportedProgressMsg = snap.message;
                  emitProgress(snap.message);
                }
              })
              .catch(() => {})
              .finally(() => {
                pollInFlight = false;
              });
          }, 180)
        : null;

    let res: Response;
    try {
      res = await fetch(`${this._baseUrl}/generate`, {
        method: 'POST',
        headers: this._authHeaders({ 'X-Compass-Job-Id': jobId }),
        body: formData,
        signal,
      });
      if (res.status === 403 && (await this._refreshDesktopHandshake(true))) {
        res = await fetch(`${this._baseUrl}/generate`, {
          method: 'POST',
          headers: this._authHeaders({ 'X-Compass-Job-Id': jobId }),
          body: this._buildGenerateFormData(request, jobId),
          signal,
        });
      }
    } finally {
      settled = true;
      pollAbortCtrl.abort();
      signal?.removeEventListener('abort', onCallerAbort);
      if (pollInterval !== null) {
        clearInterval(pollInterval);
      }
    }

    if (!res.ok) {
      let detail = `生成リクエストに失敗しました (HTTP ${res.status})`;
      try {
        const errJson = (await res.json()) as { detail?: unknown };
        if (typeof errJson.detail === 'string') {
          detail = errJson.detail;
        } else if (Array.isArray(errJson.detail) && errJson.detail.length > 0) {
          detail = errJson.detail
            .map((item: { msg?: string }) => item.msg ?? JSON.stringify(item))
            .join('; ');
        }
      } catch {
        // Fallback to status message
      }
      throw new Error(detail);
    }

    if (signal?.aborted) {
      throw createAbortError();
    }
    emitProgress('推論完了 — 生成されたMIDIストリームとCoTメタデータを展開中...');

    const alignedReasons = this._decodeReasonHeader(res.headers.get('X-Generation-Reason'));
    const reasons = alignedReasons.filter((r): r is GenerationReason => r !== null);
    const rawBuffer = await res.arrayBuffer();
    if (signal?.aborted) {
      throw createAbortError();
    }
    if (rawBuffer.byteLength > 16 * 1024 * 1024) {
      throw new Error('生成レスポンスのサイズが上限 (16MB) を超えています。');
    }
    const bytes = new Uint8Array(rawBuffer);

    const candidates: GeneratedMidiCandidate[] = [];

    // Check ZIP magic header ('PK')
    if (bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b) {
      const zip = await JSZip.loadAsync(rawBuffer);
      if (signal?.aborted) {
        throw createAbortError();
      }
      const allZipNames = Object.keys(zip.files);
      if (allZipNames.length > 512) {
        throw new Error('生成アーカイブ内のエントリ数が上限を超えています。');
      }
      const midNames = allZipNames
        .filter((name) => !zip.files[name].dir && /\.(mid|midi)$/i.test(name))
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
        .slice(0, 32);

      let totalCandidateBytes = 0;
      let failedZipAttempts = 0;
      for (let i = 0; i < midNames.length && candidates.length < 32; i += 1) {
        if (signal?.aborted) {
          throw createAbortError();
        }
        if (failedZipAttempts >= MAX_FAILED_CANDIDATE_ATTEMPTS) {
          break;
        }
        if (MAX_CANDIDATE_MIDI_BYTES - totalCandidateBytes < 22) {
          break;
        }
        const fname = midNames[i];
        const rawBytes = await CompassApiClient._decompressZipEntryBounded(
          zip.files[fname],
          MAX_CANDIDATE_MIDI_BYTES,
        );
        if (signal?.aborted) {
          throw createAbortError();
        }
        if (!rawBytes || !isValidCandidateMidiBytesPayload(rawBytes)) {
          failedZipAttempts += 1;
          continue;
        }
        if (totalCandidateBytes + rawBytes.byteLength > MAX_CANDIDATE_MIDI_BYTES) {
          break;
        }
        const midBuf = rawBytes.buffer.slice(
          rawBytes.byteOffset,
          rawBytes.byteOffset + rawBytes.byteLength,
        ) as ArrayBuffer;
        let parsed: ParsedMidiFile;
        try {
          parsed = StandardMidiCodec.parseMidi(midBuf);
        } catch {
          failedZipAttempts += 1;
          continue;
        }
        totalCandidateBytes += rawBytes.byteLength;
        const candIdx = candidates.length;
        const safeFname =
          truncateWellFormedUtf16(fname.trim(), 256).trim() || `output_${candIdx}.mid`;
        const candReason =
          i < alignedReasons.length ? alignedReasons[i] : (alignedReasons[0] ?? null);
        candidates.push({
          index: candIdx,
          filename: safeFname,
          rawMidiBytes: rawBytes,
          parsedMidi: parsed,
          reason: candReason,
        });
      }
    } else if (
      bytes.byteLength >= 22 &&
      bytes.byteLength <= MAX_CANDIDATE_MIDI_BYTES &&
      bytes[0] === 0x4d &&
      bytes[1] === 0x54 &&
      bytes[2] === 0x68 &&
      bytes[3] === 0x64
    ) {
      const compactBytes =
        bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes
          : new Uint8Array(bytes);
      if (isValidCandidateMidiBytesPayload(compactBytes)) {
        try {
          const parsed = StandardMidiCodec.parseMidi(compactBytes.buffer as ArrayBuffer);
          candidates.push({
            index: 0,
            filename: 'output_0.mid',
            rawMidiBytes: compactBytes,
            parsedMidi: parsed,
            reason: alignedReasons[0] ?? null,
          });
        } catch {
          // Handled below when candidates.length === 0
        }
      }
    }

    if (signal?.aborted) {
      throw createAbortError();
    }
    if (candidates.length === 0) {
      throw new Error('生成アーカイブ内に有効なMIDIファイルが含まれていませんでした。');
    }

    emitProgress('ピアノロールへ生成ノートを配置しています...');
    return { candidates, reasons };
  }

  private static _decompressZipEntryBounded(
    entry: JSZip.JSZipObject,
    maxBytes: number,
  ): Promise<Uint8Array | null> {
    try {
      const rawData = (entry as unknown as { _data?: { uncompressedSize?: unknown } })._data;
      if (
        rawData &&
        typeof rawData === 'object' &&
        typeof rawData.uncompressedSize === 'number' &&
        Number.isFinite(rawData.uncompressedSize) &&
        (rawData.uncompressedSize > maxBytes || rawData.uncompressedSize < 0)
      ) {
        return Promise.resolve(null);
      }
      interface JsZipInternalStream {
        on(event: 'data', cb: (chunk: Uint8Array) => void): JsZipInternalStream;
        on(event: 'error', cb: (err: unknown) => void): JsZipInternalStream;
        on(event: 'end', cb: () => void): JsZipInternalStream;
        resume(): JsZipInternalStream;
        pause(): JsZipInternalStream;
      }
      const internalStreamFn = (
        entry as unknown as { internalStream?: (type: 'uint8array') => JsZipInternalStream }
      ).internalStream;
      if (typeof internalStreamFn === 'function') {
        return new Promise<Uint8Array | null>((resolve) => {
          const chunks: Uint8Array[] = [];
          let total = 0;
          let settled = false;
          let stream: JsZipInternalStream;
          try {
            stream = internalStreamFn.call(entry, 'uint8array');
          } catch {
            resolve(null);
            return;
          }
          stream
            .on('data', (chunk: Uint8Array) => {
              if (settled) return;
              if (!(chunk instanceof Uint8Array)) {
                settled = true;
                try {
                  stream.pause();
                } catch {
                  // Ignore stream pause errors
                }
                chunks.length = 0;
                resolve(null);
                return;
              }
              total += chunk.byteLength;
              if (total > maxBytes) {
                settled = true;
                try {
                  stream.pause();
                } catch {
                  // Ignore stream pause errors
                }
                chunks.length = 0;
                resolve(null);
                return;
              }
              chunks.push(chunk);
            })
            .on('error', () => {
              if (settled) return;
              settled = true;
              chunks.length = 0;
              resolve(null);
            })
            .on('end', () => {
              if (settled) return;
              settled = true;
              const out = new Uint8Array(total);
              let offset = 0;
              for (let i = 0; i < chunks.length; i += 1) {
                out.set(chunks[i], offset);
                offset += chunks[i].byteLength;
              }
              chunks.length = 0;
              resolve(out);
            })
            .resume();
        });
      }
      return entry
        .async('uint8array')
        .then((u8) => (u8 instanceof Uint8Array && u8.byteLength <= maxBytes ? u8 : null))
        .catch(() => null);
    } catch {
      return Promise.resolve(null);
    }
  }

  private _decodeReasonHeader(encodedHeader: string | null): Array<GenerationReason | null> {
    if (!encodedHeader) return [];
    try {
      const jsonStr = decodeURIComponent(encodedHeader);
      const parsed = JSON.parse(jsonStr) as unknown;
      if (Array.isArray(parsed) && !isRuntimeProxy(parsed)) {
        const lenDesc = Object.getOwnPropertyDescriptor(parsed, 'length');
        const rawLen =
          lenDesc &&
          'value' in lenDesc &&
          typeof lenDesc.value === 'number' &&
          Number.isFinite(lenDesc.value)
            ? Math.max(0, Math.trunc(lenDesc.value))
            : 0;
        const out: Array<GenerationReason | null> = [];
        const limit = Math.min(rawLen, 64);
        for (let i = 0; i < limit; i += 1) {
          const elemDesc = Object.getOwnPropertyDescriptor(parsed, i);
          out.push(
            elemDesc && 'value' in elemDesc ? sanitizeGenerationReason(elemDesc.value) : null,
          );
        }
        return out;
      }
      if (parsed && typeof parsed === 'object' && !isRuntimeProxy(parsed)) {
        const sanitized = sanitizeGenerationReason(parsed);
        return sanitized ? [sanitized] : [];
      }
      return [];
    } catch {
      return [];
    }
  }
}
