/**
 * COMPASS AI Electron IPC & Origin Security Validators (`electron/ipc_security.cjs`).
 * Pure Node.js CommonJS module shared by `main.cjs` and `verify_e2e_audit.cjs` (`oop-extensible-craft` SRP).
 * Handles WHATWG custom-scheme canonical origin extraction (`compass-app://studio`) and
 * workspace-wide sync payload validation (`secure-coding-guard`).
 */

const util = require('util');

const COMPASS_APP_ORIGIN = 'compass-app://studio';
const MAX_SYNC_PAYLOAD_CHARS = 2_000_000;

const TYPED_ARRAY_PROTO = Object.getPrototypeOf(Uint8Array.prototype);
const TYPED_ARRAY_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'byteLength',
).get;
const TYPED_ARRAY_BYTEOFFSET_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'byteOffset',
).get;
const TYPED_ARRAY_BUFFER_GETTER = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTO, 'buffer').get;
const ARRAYBUFFER_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength',
).get;
const ARRAYBUFFER_DETACHED_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'detached',
)?.get;

const VALID_TRACK_INSTRUMENTS = new Set([
  'PIANO',
  'SAX',
  'BASS',
  'VIOLIN',
  'SYNTH',
  'GUITAR',
  'DRUMS',
]);

/**
 * Truncates `raw` to at most `maxUnits` UTF-16 code units without splitting surrogate pairs
 * and replaces lone surrogates 1-for-1 with `\uFFFD` (`[R9-LOW-03]`, `[R12-LOW-01]`).
 * @param {string} raw
 * @param {number} maxUnits
 * @returns {string}
 */
function truncateWellFormedUtf16(raw, maxUnits) {
  if (
    typeof raw !== 'string' ||
    raw.length === 0 ||
    typeof maxUnits !== 'number' ||
    Number.isNaN(maxUnits) ||
    maxUnits <= 0
  ) {
    return '';
  }
  let limit = Math.min(raw.length, Math.floor(maxUnits));
  if (limit <= 0) return '';
  if (limit < raw.length) {
    const last = raw.charCodeAt(limit - 1);
    if (last >= 0xd800 && last <= 0xdbff) {
      const next = raw.charCodeAt(limit);
      if (next >= 0xdc00 && next <= 0xdfff) {
        limit -= 1;
        if (limit <= 0) return '';
      }
    }
  }
  let needsFix = false;
  for (let i = 0; i < limit; i += 1) {
    const c = raw.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      if (i + 1 < limit) {
        const c2 = raw.charCodeAt(i + 1);
        if (c2 >= 0xdc00 && c2 <= 0xdfff) {
          i += 1;
          continue;
        }
      }
      needsFix = true;
      break;
    }
    if (c >= 0xdc00 && c <= 0xdfff) {
      needsFix = true;
      break;
    }
  }
  if (!needsFix) {
    return limit === raw.length ? raw : raw.slice(0, limit);
  }
  let out = '';
  for (let i = 0; i < limit; i += 1) {
    const c = raw.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      if (i + 1 < limit) {
        const c2 = raw.charCodeAt(i + 1);
        if (c2 >= 0xdc00 && c2 <= 0xdfff) {
          out += raw.slice(i, i + 2);
          i += 1;
          continue;
        }
      }
      out += '\ufffd';
    } else if (c >= 0xdc00 && c <= 0xdfff) {
      out += '\ufffd';
    } else {
      out += raw[i];
    }
  }
  return out;
}

/**
 * @param {string} raw
 * @returns {boolean}
 */
function isFastCleanAsciiSyncId(raw) {
  const len = raw.length;
  if (len === 0 || len > 128) return false;
  const c0 = raw.charCodeAt(0);
  const cLast = raw.charCodeAt(len - 1);
  if (c0 <= 0x20 || c0 >= 0x7f || cLast <= 0x20 || cLast >= 0x7f) return false;
  for (let i = 0; i < len; i += 1) {
    const c = raw.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdfff) return false;
  }
  return true;
}

/**
 * @param {string} raw
 * @returns {string}
 */
function canonSyncId(raw) {
  if (isFastCleanAsciiSyncId(raw)) return raw;
  return truncateWellFormedUtf16(raw.trim(), 128).trim();
}

/**
 * @param {object} obj
 * @param {string | number} key
 * @returns {any}
 */
function getOwnVal(obj, key) {
  const desc = Object.getOwnPropertyDescriptor(obj, key);
  return desc && 'value' in desc ? desc.value : undefined;
}

/**
 * Extracts a canonical origin string across both standard WHATWG special schemes (`http:`, `https:`)
 * and Electron privileged custom schemes (`compass-app://studio`), where Node.js `new URL(...).origin`
 * returns literal `'null'` (`[R4-HIGH-01]`).
 * @param {unknown} rawUrl
 * @returns {string}
 */
function extractCanonicalOrigin(rawUrl) {
  try {
    if (typeof rawUrl !== 'string' || !rawUrl) return '';
    const u = new URL(rawUrl);
    if (u.origin && u.origin !== 'null') {
      return u.origin.toLowerCase();
    }
    if (u.protocol === 'compass-app:' && u.host.toLowerCase() === 'studio') {
      return COMPASS_APP_ORIGIN;
    }
    return '';
  } catch {
    return '';
  }
}

/**
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {string}
 */
function getAllowedAppOrigin(env = process.env) {
  try {
    if (env && typeof env === 'object' && !util.types.isProxy(env)) {
      const devUrl = env.COMPASS_VITE_DEV_URL;
      if (typeof devUrl === 'string' && devUrl) {
        const parsedOrigin = extractCanonicalOrigin(devUrl);
        if (parsedOrigin) {
          return parsedOrigin;
        }
      }
    }
    return COMPASS_APP_ORIGIN;
  } catch {
    return COMPASS_APP_ORIGIN;
  }
}

/**
 * @param {any} event
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {boolean}
 */
function isTrustedIpcSender(event, env = process.env) {
  try {
    if (!event || typeof event !== 'object' || util.types.isProxy(event)) return false;
    const rawUrl = event?.senderFrame?.url || event?.sender?.getURL?.() || '';
    const senderOrigin = extractCanonicalOrigin(rawUrl);
    return Boolean(senderOrigin && senderOrigin === getAllowedAppOrigin(env));
  } catch {
    return false;
  }
}

const ALLOWED_NOTE_KEYS = new Set([
  'id',
  'pitch',
  'startTick',
  'durationTicks',
  'velocity',
  'channel',
]);
const ALLOWED_TRACK_KEYS = new Set([
  'id',
  'name',
  'instrument',
  'colorHex',
  'isMuted',
  'isSoloed',
  'notes',
]);
const ALLOWED_SLOT_KEYS = new Set(['trackId', 'startTick', 'endTick']);
const ALLOWED_REASON_KEYS = new Set([
  'source',
  'task',
  'thinking',
  'instruments',
  'genre',
  'gen_measure_count',
  'key',
  'note_density',
]);
const ALLOWED_CANDIDATE_KEYS = new Set(['index', 'filename', 'rawMidiBytes', 'reason']);
const ALLOWED_WORKSPACE_KEYS = new Set([
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

/**
 * Verifies `obj` is a plain object whose own property names (and zero symbols) strictly belong
 * to `allowedSet`, blocking `__proto__`/`constructor`/`prototype` and non-enumerable smuggling (`[R6B-MED-02]`).
 * @param {Record<string, unknown>} obj
 * @param {Set<string>} allowedSet
 * @returns {boolean}
 */
function hasOnlyAllowedKeys(obj, allowedSet) {
  if (!obj || typeof obj !== 'object' || util.types.isProxy(obj) || Array.isArray(obj)) {
    return false;
  }
  const proto = Object.getPrototypeOf(obj);
  if (proto !== null && proto !== Object.prototype) return false;
  if (Object.getOwnPropertySymbols(obj).length > 0) return false;
  const keys = Object.getOwnPropertyNames(obj);
  if (keys.length > allowedSet.size) return false;
  for (let i = 0; i < keys.length; i += 1) {
    const k = keys[i];
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') return false;
    if (!allowedSet.has(k)) return false;
    const desc = Object.getOwnPropertyDescriptor(obj, k);
    if (!desc || !('value' in desc)) return false;
  }
  return true;
}

/**
 * @param {unknown} arr
 * @param {number} maxItems
 * @returns {arr is any[]}
 */
function isPlainDenseArray(arr, maxItems) {
  if (!arr || typeof arr !== 'object' || util.types.isProxy(arr)) return false;
  if (!Array.isArray(arr) || Object.getPrototypeOf(arr) !== Array.prototype) return false;
  const lenDesc = Object.getOwnPropertyDescriptor(arr, 'length');
  if (!lenDesc || !('value' in lenDesc)) return false;
  const len = lenDesc.value;
  if (typeof len !== 'number' || !Number.isInteger(len) || len < 0 || len > maxItems) {
    return false;
  }
  if (Object.getOwnPropertySymbols(arr).length > 0) return false;
  if (Object.getOwnPropertyNames(arr).length !== len + 1) return false;
  for (let i = 0; i < len; i += 1) {
    const desc = Object.getOwnPropertyDescriptor(arr, i);
    if (!desc || !('value' in desc)) return false;
  }
  return true;
}

/**
 * Single-pass dense array snapshot (`[R19-MED-01]`, `[R19-MED-02]`).
 * Checks `length <= maxItems` BEFORE `Object.getOwnPropertySymbols` / `Object.getOwnPropertyNames`
 * and reads each element descriptor `0..len-1` strictly once.
 * @param {unknown} raw
 * @param {number} maxItems
 * @returns {any[] | null}
 */
function snapshotPlainDenseArray(raw, maxItems) {
  if (!raw || typeof raw !== 'object' || util.types.isProxy(raw)) return null;
  if (!Array.isArray(raw) || Object.getPrototypeOf(raw) !== Array.prototype) return null;
  const lenDesc = Object.getOwnPropertyDescriptor(raw, 'length');
  if (!lenDesc || !('value' in lenDesc)) return null;
  const len = lenDesc.value;
  if (typeof len !== 'number' || !Number.isInteger(len) || len < 0 || len > maxItems) {
    return null;
  }
  if (Object.getOwnPropertySymbols(raw).length > 0) return null;
  if (Object.getOwnPropertyNames(raw).length !== len + 1) return null;
  const out = new Array(len);
  for (let i = 0; i < len; i += 1) {
    const desc = Object.getOwnPropertyDescriptor(raw, i);
    if (!desc || !('value' in desc)) return null;
    out[i] = desc.value;
  }
  return out;
}

const MAX_SMF_QUARTER_NOTES = 4128;

/**
 * @param {any} n
 * @param {number} maxProjectTicks
 * @param {Set<string> | null} [noteIdSet]
 * @param {Map<string, string> | null} [canonNoteMap]
 * @returns {boolean}
 */
function isValidSerializedNote(n, maxProjectTicks, noteIdSet = null, canonNoteMap = null) {
  try {
    if (!n || typeof n !== 'object' || util.types.isProxy(n) || Array.isArray(n)) {
      return false;
    }
    const proto = Object.getPrototypeOf(n);
    if (proto !== null && proto !== Object.prototype) return false;
    if (Object.getOwnPropertySymbols(n).length > 0) return false;
    const keys = Object.getOwnPropertyNames(n);
    if (keys.length < 5 || keys.length > 6) return false;

    let seenMask = 0;
    let id;
    let pitch;
    let startTick;
    let durationTicks;
    let velocity;
    let channel;

    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      switch (k) {
        case 'id': {
          if ((seenMask & 1) !== 0) return false;
          seenMask |= 1;
          const desc = Object.getOwnPropertyDescriptor(n, 'id');
          if (!desc || !('value' in desc)) return false;
          id = desc.value;
          break;
        }
        case 'pitch': {
          if ((seenMask & 2) !== 0) return false;
          seenMask |= 2;
          const desc = Object.getOwnPropertyDescriptor(n, 'pitch');
          if (!desc || !('value' in desc)) return false;
          pitch = desc.value;
          break;
        }
        case 'startTick': {
          if ((seenMask & 4) !== 0) return false;
          seenMask |= 4;
          const desc = Object.getOwnPropertyDescriptor(n, 'startTick');
          if (!desc || !('value' in desc)) return false;
          startTick = desc.value;
          break;
        }
        case 'durationTicks': {
          if ((seenMask & 8) !== 0) return false;
          seenMask |= 8;
          const desc = Object.getOwnPropertyDescriptor(n, 'durationTicks');
          if (!desc || !('value' in desc)) return false;
          durationTicks = desc.value;
          break;
        }
        case 'velocity': {
          if ((seenMask & 16) !== 0) return false;
          seenMask |= 16;
          const desc = Object.getOwnPropertyDescriptor(n, 'velocity');
          if (!desc || !('value' in desc)) return false;
          velocity = desc.value;
          break;
        }
        case 'channel': {
          if ((seenMask & 32) !== 0) return false;
          seenMask |= 32;
          const desc = Object.getOwnPropertyDescriptor(n, 'channel');
          if (!desc || !('value' in desc)) return false;
          channel = desc.value;
          break;
        }
        default:
          return false;
      }
    }
    if ((seenMask & 31) !== 31) return false;

    if (
      typeof id !== 'string' ||
      id.length > 128 ||
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
      return false;
    }

    let rawTrimmedNoteId;
    let canonNoteId;
    if (isFastCleanAsciiSyncId(id)) {
      rawTrimmedNoteId = id;
      canonNoteId = id;
    } else {
      rawTrimmedNoteId = id.trim();
      if (rawTrimmedNoteId.length === 0) return false;
      canonNoteId = truncateWellFormedUtf16(rawTrimmedNoteId, 128).trim();
    }
    if (!canonNoteId) return false;
    if (noteIdSet !== null && noteIdSet.has(canonNoteId)) return false;
    if (canonNoteMap !== null && canonNoteMap.has(canonNoteId)) return false;

    const roundedStart = Math.round(startTick);
    const roundedDur = Math.round(durationTicks);
    if (
      roundedStart < 0 ||
      roundedDur < 1 ||
      roundedStart >= maxProjectTicks ||
      roundedStart + roundedDur > maxProjectTicks
    ) {
      return false;
    }
    if (noteIdSet !== null) noteIdSet.add(canonNoteId);
    if (canonNoteMap !== null) canonNoteMap.set(canonNoteId, rawTrimmedNoteId);
    return true;
  } catch {
    return false;
  }
}

/**
 * @param {any} t
 * @param {Set<string>} [seenNoteIds]
 * @param {number} [maxProjectTicks]
 * @param {Map<string, string> | null} [rawTrimmedByCanonNoteId]
 * @returns {boolean}
 */
function isValidSerializedTrack(
  t,
  seenNoteIds,
  maxProjectTicks = MAX_SMF_QUARTER_NOTES * 480,
  rawTrimmedByCanonNoteId = null,
) {
  try {
    if (!hasOnlyAllowedKeys(t, ALLOWED_TRACK_KEYS)) {
      return false;
    }
    const id = getOwnVal(t, 'id');
    const name = getOwnVal(t, 'name');
    const instrument = getOwnVal(t, 'instrument');
    const colorHex = getOwnVal(t, 'colorHex');
    const isMuted = getOwnVal(t, 'isMuted');
    const isSoloed = getOwnVal(t, 'isSoloed');
    const notes = getOwnVal(t, 'notes');
    if (typeof id !== 'string' || id.length > 128 || canonSyncId(id).length === 0) return false;
    if (typeof name !== 'string' || name.trim().length === 0 || name.length > 256) return false;
    if (typeof instrument !== 'string' || !VALID_TRACK_INSTRUMENTS.has(instrument)) return false;
    if (typeof colorHex !== 'string' || colorHex.trim().length === 0 || colorHex.length > 32) {
      return false;
    }
    if (typeof isMuted !== 'boolean' || typeof isSoloed !== 'boolean') return false;
    const hasNoteSet = seenNoteIds instanceof Set;
    const hasCanonMap = rawTrimmedByCanonNoteId instanceof Map;
    const noteIdSet = hasNoteSet ? seenNoteIds : hasCanonMap ? null : new Set();
    const canonNoteMap = hasCanonMap ? rawTrimmedByCanonNoteId : new Map();
    const existingCount = Math.max(noteIdSet ? noteIdSet.size : 0, canonNoteMap.size);
    const noteArr = snapshotPlainDenseArray(notes, 32768 - existingCount);
    if (!noteArr || existingCount + noteArr.length > 32768) return false;
    for (let i = 0; i < noteArr.length; i += 1) {
      if (!isValidSerializedNote(noteArr[i], maxProjectTicks, noteIdSet, canonNoteMap)) {
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * @param {any} r
 * @param {Set<string>} seenTrackIds
 * @param {number} maxSlotTick
 * @param {Map<string, string> | null} [rawTrimmedByCanonTrackId]
 * @returns {boolean}
 */
function isValidCandidateSlotRange(
  r,
  seenTrackIds,
  maxSlotTick,
  rawTrimmedByCanonTrackId = null,
) {
  try {
    if (r === null || r === undefined) return true;
    if (!hasOnlyAllowedKeys(r, ALLOWED_SLOT_KEYS)) {
      return false;
    }
    const trackId = getOwnVal(r, 'trackId');
    const startTick = getOwnVal(r, 'startTick');
    const endTick = getOwnVal(r, 'endTick');
    if (typeof trackId !== 'string' || trackId.length > 128) return false;
    const rawTrimmedTrackId = trackId.trim();
    const canonTrackId = canonSyncId(trackId);
    if (!canonTrackId) return false;
    if (rawTrimmedByCanonTrackId instanceof Map) {
      if (rawTrimmedByCanonTrackId.get(canonTrackId) !== rawTrimmedTrackId) return false;
    } else if (!(seenTrackIds instanceof Set)) {
      return false;
    } else if (!seenTrackIds.has(canonTrackId) && !seenTrackIds.has(rawTrimmedTrackId)) {
      return false;
    }
    return (
      typeof startTick === 'number' &&
      Number.isFinite(startTick) &&
      startTick >= 0 &&
      typeof endTick === 'number' &&
      Number.isFinite(endTick) &&
      endTick >= startTick &&
      endTick <= maxSlotTick
    );
  } catch {
    return false;
  }
}

/**
 * Validates that `arr` is a dense `string[]` with zero sparse holes (`!(i in arr)`) (`[R8-LOW-01]`).
 * @param {unknown} arr
 * @param {number} maxItems
 * @param {number} maxItemLen
 * @param {boolean} [requireNonEmptyTrimmed]
 * @returns {boolean}
 */
function isDenseBoundedStringArray(arr, maxItems, maxItemLen, requireNonEmptyTrimmed = false) {
  try {
    if (!isPlainDenseArray(arr, maxItems)) return false;
    for (let i = 0; i < arr.length; i += 1) {
      const x = getOwnVal(arr, i);
      if (typeof x !== 'string' || x.length > maxItemLen) return false;
      if (requireNonEmptyTrimmed && x.trim().length === 0) return false;
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * Computes canonical `beatsPerMeasure` (`0.5..16`, `1/8`-beat resolution) (`[R8-MED-02]`).
 * @param {number} numerator
 * @param {number} denominator
 * @returns {number}
 */
function canonicalBeatsPerMeasure(numerator, denominator) {
  const safeNum =
    typeof numerator === 'number' && Number.isFinite(numerator)
      ? Math.max(1, Math.min(32, Math.round(numerator)))
      : 4;
  const safeDen =
    typeof denominator === 'number' &&
    Number.isFinite(denominator) &&
    [1, 2, 4, 8, 16, 32].includes(Math.round(denominator))
      ? Math.round(denominator)
      : 4;
  const rawBeats = (safeNum * 4) / safeDen;
  return Math.max(0.5, Math.min(16, Math.round(rawBeats * 8) / 8));
}

/**
 * Derives `(numerator, denominator)` from a standalone `beatsPerMeasure` (`[R8-MED-02]`).
 * @param {number} beatsPerMeasure
 * @returns {{ numerator: number; denominator: number }}
 */
function deriveTimeSignatureFromBeats(beatsPerMeasure) {
  const raw =
    typeof beatsPerMeasure === 'number' && Number.isFinite(beatsPerMeasure)
      ? beatsPerMeasure
      : 4;
  const q = Math.max(0.5, Math.min(16, Math.round(raw * 8) / 8));
  const eighths = Math.round(q * 8);
  if (eighths % 8 === 0) {
    return { numerator: Math.max(1, Math.min(32, eighths / 8)), denominator: 4 };
  }
  if (eighths % 4 === 0) {
    return { numerator: Math.max(1, Math.min(32, eighths / 4)), denominator: 8 };
  }
  if (eighths % 2 === 0) {
    const sixteenths = eighths / 2;
    if (sixteenths <= 32) {
      return { numerator: sixteenths, denominator: 16 };
    }
    return { numerator: Math.max(1, Math.min(32, Math.round(sixteenths / 2))), denominator: 8 };
  }
  if (eighths <= 32) {
    return { numerator: eighths, denominator: 32 };
  }
  const roundedSixteenths = Math.round(eighths / 2);
  if (roundedSixteenths <= 32) {
    return { numerator: roundedSixteenths, denominator: 16 };
  }
  return { numerator: Math.max(1, Math.min(32, Math.round(eighths / 4))), denominator: 8 };
}

/**
 * Validates `GenerationReason` schema so malformed objects (`genre: "jazz"`, `genre: 123`)
 * can never crash `PianoRollView` (`genre.join`) across IPC (`[R3-MED-01]`, `[R8-LOW-01]`).
 * @param {any} reason
 * @returns {boolean}
 */
function isValidGenerationReason(reason) {
  try {
    if (reason === null || reason === undefined) return true;
    if (!hasOnlyAllowedKeys(reason, ALLOWED_REASON_KEYS)) {
      return false;
    }
    const source = getOwnVal(reason, 'source');
    if (source !== 'model_cot' && source !== 'request' && source !== 'foundation') {
      return false;
    }
    const task = getOwnVal(reason, 'task');
    if (typeof task !== 'string' || task.trim().length === 0 || task.length > 128) {
      return false;
    }
    if (typeof getOwnVal(reason, 'thinking') !== 'boolean') return false;
    const instruments = getOwnVal(reason, 'instruments');
    if (
      instruments !== null &&
      instruments !== undefined &&
      !isDenseBoundedStringArray(instruments, 64, 64, false)
    ) {
      return false;
    }
    const genre = getOwnVal(reason, 'genre');
    if (
      genre !== null &&
      genre !== undefined &&
      !isDenseBoundedStringArray(genre, 64, 64, false)
    ) {
      return false;
    }
    const genMeasureCount = getOwnVal(reason, 'gen_measure_count');
    if (
      genMeasureCount !== null &&
      genMeasureCount !== undefined &&
      (typeof genMeasureCount !== 'number' ||
        !Number.isFinite(genMeasureCount) ||
        genMeasureCount < 0 ||
        genMeasureCount > 512)
    ) {
      return false;
    }
    const key = getOwnVal(reason, 'key');
    if (key !== null && key !== undefined && (typeof key !== 'string' || key.length > 32)) {
      return false;
    }
    const noteDensity = getOwnVal(reason, 'note_density');
    if (noteDensity !== null && noteDensity !== undefined) {
      if (
        !noteDensity ||
        typeof noteDensity !== 'object' ||
        util.types.isProxy(noteDensity) ||
        Array.isArray(noteDensity)
      ) {
        return false;
      }
      const ndProto = Object.getPrototypeOf(noteDensity);
      if (ndProto !== null && ndProto !== Object.prototype) return false;
      if (Object.getOwnPropertySymbols(noteDensity).length > 0) return false;
      const keys = Object.getOwnPropertyNames(noteDensity);
      if (keys.length > 32) return false;
      for (let i = 0; i < keys.length; i += 1) {
        const k = keys[i];
        if (
          k.length === 0 ||
          k.length > 64 ||
          k === '__proto__' ||
          k === 'constructor' ||
          k === 'prototype'
        ) {
          return false;
        }
        const kDesc = Object.getOwnPropertyDescriptor(noteDensity, k);
        if (!kDesc || !('value' in kDesc)) return false;
        const v = kDesc.value;
        if (v !== null && (typeof v !== 'number' || !Number.isFinite(v))) {
          return false;
        }
      }
    }
    return true;
  } catch {
    return false;
  }
}

const MAX_CANDIDATE_MIDI_BYTES = 2_000_000;
const MAX_PLAIN_ARRAY_MIDI_BYTES = 4096;
const NATIVE_UINT8ARRAY_CTOR_RE = /^function Uint8Array\(\)\s*\{\s*\[native code\]\s*\}$/;

/**
 * Accepts same-realm `Uint8Array.prototype`, Node `Buffer.prototype`, or a genuine cross-realm
 * `%Uint8Array.prototype%` (e.g. `structuredClone` / `BroadcastChannel` / `vm` realm), while
 * strictly rejecting `class SubU8 extends Uint8Array {}` and forged prototypes (`[R14-LOW-01]`).
 * @param {unknown} proto
 * @returns {boolean}
 */
function isGenuineUint8ArrayPrototype(proto) {
  if (proto === Uint8Array.prototype) return true;
  if (typeof Buffer === 'function' && Boolean(Buffer.prototype) && proto === Buffer.prototype) {
    return true;
  }
  if (!proto || typeof proto !== 'object' || util.types.isProxy(proto)) return false;
  const typedArrProto = Object.getPrototypeOf(proto);
  if (!typedArrProto || typeof typedArrProto !== 'object' || util.types.isProxy(typedArrProto)) {
    return false;
  }
  const objProto = Object.getPrototypeOf(typedArrProto);
  if (
    !objProto ||
    typeof objProto !== 'object' ||
    util.types.isProxy(objProto) ||
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
    util.types.isProxy(ctorDesc.value)
  ) {
    return false;
  }
  const ctor = ctorDesc.value;
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

/**
 * Extracts genuine `Uint8Array` view + underlying `ArrayBuffer` byte lengths via V8 intrinsic
 * internal-slot getters (`[R4-MED-05]`), defeating `[Symbol.toStringTag]` spoofs, `Proxy`,
 * property shadowing, `SharedArrayBuffer`, and subarray backing-buffer memory amplification.
 * @param {unknown} raw
 * @returns {{ viewLen: number; byteOffset: number; backingLen: number; backingBuf: ArrayBuffer } | null}
 */
function inspectGenuineUint8Array(raw) {
  try {
    if (!raw || typeof raw !== 'object' || util.types.isProxy(raw) || !util.types.isUint8Array(raw)) {
      return null;
    }
    if (!isGenuineUint8ArrayPrototype(Object.getPrototypeOf(raw))) {
      return null;
    }
    if (Object.getOwnPropertySymbols(raw).length !== 0) {
      return null;
    }
    const viewLen = TYPED_ARRAY_BYTELENGTH_GETTER.call(raw);
    const byteOffset = TYPED_ARRAY_BYTEOFFSET_GETTER.call(raw);
    const backingBuf = TYPED_ARRAY_BUFFER_GETTER.call(raw);
    if (ARRAYBUFFER_DETACHED_GETTER && ARRAYBUFFER_DETACHED_GETTER.call(backingBuf) === true) {
      return null;
    }
    const backingLen = ARRAYBUFFER_BYTELENGTH_GETTER.call(backingBuf);
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
    return { viewLen, byteOffset, backingLen, backingBuf };
  } catch {
    return null;
  }
}

/**
 * Single-pass validation + byte accounting for candidate SMF buffers (`[R3-MED-01]`, `[R4-MED-05]`, `[R19-MED-01]`).
 * Pre-caps against `maxAllowedBytes` in `O(1)` before header/element inspection and bounds legacy plain
 * `number[]` inputs to `MAX_PLAIN_ARRAY_MIDI_BYTES` (`4096`) while allowing genuine `Uint8Array`s up to `2 MB`.
 * @param {any} raw
 * @param {number} [maxAllowedBytes]
 * @returns {number | null}
 */
function measureValidCandidateMidiBytes(raw, maxAllowedBytes = MAX_CANDIDATE_MIDI_BYTES) {
  try {
    const effectiveCap =
      typeof maxAllowedBytes === 'number' && Number.isFinite(maxAllowedBytes)
        ? Math.min(MAX_CANDIDATE_MIDI_BYTES, Math.floor(maxAllowedBytes))
        : MAX_CANDIDATE_MIDI_BYTES;
    if (effectiveCap < 22) return null;
    if (!raw || typeof raw !== 'object' || util.types.isProxy(raw)) return null;
    const u8Info = inspectGenuineUint8Array(raw);
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
    const rawByteArr = snapshotPlainDenseArray(raw, plainCap);
    if (rawByteArr !== null) {
      const len = rawByteArr.length;
      if (len < 22) return null;
      for (let i = 0; i < len; i += 1) {
        const b = rawByteArr[i];
        if (typeof b !== 'number' || !Number.isInteger(b) || b < 0 || b > 255) {
          return null;
        }
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

/**
 * @param {any} raw
 * @param {number} [maxAllowedBytes]
 * @returns {boolean}
 */
function isValidCandidateMidiBytes(raw, maxAllowedBytes = MAX_CANDIDATE_MIDI_BYTES) {
  return measureValidCandidateMidiBytes(raw, maxAllowedBytes) !== null;
}

const ALLOWED_FOCUS_MSG_KEYS = new Set(['messageId', 'type', 'tab', 'reason', 'senderId']);
const ALLOWED_GEN_STATUS_MSG_KEYS = new Set([
  'messageId',
  'type',
  'isGenerating',
  'statusDetail',
  'senderId',
]);
const ALLOWED_REQ_STATE_MSG_KEYS = new Set(['messageId', 'type', 'senderId']);
const ALLOWED_SYNC_STATE_MSG_KEYS = new Set(['messageId', 'type', 'workspaceState', 'senderId']);
const ALLOWED_SYNC_CONTROLLER_MSG_KEYS = new Set([
  'messageId',
  'type',
  'controllerId',
  'senderId',
]);
const VALID_SYNC_CONTROLLER_IDS = new Set([
  'demo-pianoroll',
  'cubase-midi-remote',
  'ableton-osc-bridge',
]);

/**
 * @param {any} c
 * @param {number} [maxAllowedBytes]
 * @returns {number | null}
 */
function measureValidSerializedCandidate(c, maxAllowedBytes = MAX_CANDIDATE_MIDI_BYTES) {
  try {
    if (!hasOnlyAllowedKeys(c, ALLOWED_CANDIDATE_KEYS)) {
      return null;
    }
    const index = getOwnVal(c, 'index');
    const filename = getOwnVal(c, 'filename');
    const rawMidiBytes = getOwnVal(c, 'rawMidiBytes');
    const reason = getOwnVal(c, 'reason');
    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0 || index > 64) {
      return null;
    }
    if (typeof filename !== 'string' || filename.trim().length === 0 || filename.length > 256) {
      return null;
    }
    const accountedBytes = measureValidCandidateMidiBytes(rawMidiBytes, maxAllowedBytes);
    if (accountedBytes === null) return null;
    if (!isValidGenerationReason(reason)) return null;
    return accountedBytes;
  } catch {
    return null;
  }
}

/**
 * @param {any} c
 * @param {number} [maxAllowedBytes]
 * @returns {boolean}
 */
function isValidSerializedCandidate(c, maxAllowedBytes = MAX_CANDIDATE_MIDI_BYTES) {
  return measureValidSerializedCandidate(c, maxAllowedBytes) !== null;
}

/**
 * @param {any} msg
 * @returns {boolean}
 */
function isValidWorkspaceSyncPayload(msg) {
  try {
    if (!msg || typeof msg !== 'object' || util.types.isProxy(msg) || Array.isArray(msg)) {
      return false;
    }
    const msgTypeDesc = Object.getOwnPropertyDescriptor(msg, 'type');
    if (!msgTypeDesc || !('value' in msgTypeDesc)) return false;
    const msgType = msgTypeDesc.value;

    if (msgType === 'FOCUS_TAB') {
      if (!hasOnlyAllowedKeys(msg, ALLOWED_FOCUS_MSG_KEYS)) return false;
    } else if (msgType === 'GENERATION_STATUS') {
      if (!hasOnlyAllowedKeys(msg, ALLOWED_GEN_STATUS_MSG_KEYS)) return false;
    } else if (msgType === 'REQUEST_WORKSPACE_STATE') {
      if (!hasOnlyAllowedKeys(msg, ALLOWED_REQ_STATE_MSG_KEYS)) return false;
    } else if (msgType === 'SYNC_WORKSPACE_STATE') {
      if (!hasOnlyAllowedKeys(msg, ALLOWED_SYNC_STATE_MSG_KEYS)) return false;
    } else if (msgType === 'SYNC_ACTIVE_CONTROLLER') {
      if (!hasOnlyAllowedKeys(msg, ALLOWED_SYNC_CONTROLLER_MSG_KEYS)) return false;
    } else {
      return false;
    }

    const senderId = getOwnVal(msg, 'senderId');
    const messageId = getOwnVal(msg, 'messageId');
    if (typeof senderId !== 'string' || typeof messageId !== 'string') return false;
    if (
      senderId.trim().length === 0 ||
      senderId.length > 128 ||
      messageId.trim().length === 0 ||
      messageId.length > 128
    ) {
      return false;
    }

    if (msgType === 'FOCUS_TAB') {
      const reason = getOwnVal(msg, 'reason');
      if (reason !== undefined && (typeof reason !== 'string' || reason.length > 256)) {
        return false;
      }
      const tab = getOwnVal(msg, 'tab');
      return tab === 'compass-ai' || tab === 'pianoroll';
    }
    if (msgType === 'GENERATION_STATUS') {
      const isGenerating = getOwnVal(msg, 'isGenerating');
      const statusDetail = getOwnVal(msg, 'statusDetail');
      return (
        typeof isGenerating === 'boolean' &&
        typeof statusDetail === 'string' &&
        statusDetail.length <= 512
      );
    }
    if (msgType === 'REQUEST_WORKSPACE_STATE') {
      return true;
    }
    if (msgType === 'SYNC_ACTIVE_CONTROLLER') {
      const controllerId = getOwnVal(msg, 'controllerId');
      return typeof controllerId === 'string' && VALID_SYNC_CONTROLLER_IDS.has(controllerId);
    }
    if (msgType === 'SYNC_WORKSPACE_STATE') {
      const ws = getOwnVal(msg, 'workspaceState');
      if (!hasOnlyAllowedKeys(ws, ALLOWED_WORKSPACE_KEYS)) {
        return false;
      }
      const bpm = getOwnVal(ws, 'bpm');
      if (typeof bpm !== 'number' || !Number.isFinite(bpm) || bpm < 20 || bpm > 300) {
        return false;
      }
      const ppqn = getOwnVal(ws, 'ppqn');
      if (
        ppqn !== undefined &&
        (typeof ppqn !== 'number' || !Number.isFinite(ppqn) || ppqn < 24 || ppqn > 32767)
      ) {
        return false;
      }
      const effectivePpqn = ppqn !== undefined ? Math.round(ppqn) : 480;
      const maxProjectTicks = MAX_SMF_QUARTER_NOTES * effectivePpqn;
      const maxSlotTick = 16512 * effectivePpqn;
      const timeSigNum = getOwnVal(ws, 'timeSigNumerator');
      const timeSigDen = getOwnVal(ws, 'timeSigDenominator');
      const beatsPerMeasure = getOwnVal(ws, 'beatsPerMeasure');
      if ((timeSigNum === undefined) !== (timeSigDen === undefined)) {
        return false;
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
          return false;
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
            return false;
          }
        }
      } else if (beatsPerMeasure !== undefined) {
        if (
          typeof beatsPerMeasure !== 'number' ||
          !Number.isFinite(beatsPerMeasure) ||
          beatsPerMeasure < 0.5 ||
          beatsPerMeasure > 16
        ) {
          return false;
        }
        const derivedTs = deriveTimeSignatureFromBeats(beatsPerMeasure);
        const expectedBeats = canonicalBeatsPerMeasure(derivedTs.numerator, derivedTs.denominator);
        if (Math.abs(beatsPerMeasure - expectedBeats) > 1e-6) {
          return false;
        }
      }
      const tracks = snapshotPlainDenseArray(getOwnVal(ws, 'tracks'), 64);
      if (!tracks || tracks.length === 0) return false;
      const seenTrackIds = new Set();
      const rawTrimmedByCanonTrackId = new Map();
      const rawTrimmedByCanonNoteId = new Map();
      for (let i = 0; i < tracks.length; i += 1) {
        const trk = tracks[i];
        if (!isValidSerializedTrack(trk, null, maxProjectTicks, rawTrimmedByCanonNoteId)) {
          return false;
        }
        const rawTrackId = getOwnVal(trk, 'id');
        const rawTrimmedTrackId = rawTrackId.trim();
        const canonTrackId = canonSyncId(rawTrackId);
        if (!canonTrackId || rawTrimmedByCanonTrackId.has(canonTrackId)) return false;
        seenTrackIds.add(canonTrackId);
        rawTrimmedByCanonTrackId.set(canonTrackId, rawTrimmedTrackId);
      }
      const activeTrackId = getOwnVal(ws, 'activeTrackId');
      if (typeof activeTrackId !== 'string' || activeTrackId.length > 128) {
        return false;
      }
      const canonActiveTrackId = canonSyncId(activeTrackId);
      if (
        !canonActiveTrackId ||
        rawTrimmedByCanonTrackId.get(canonActiveTrackId) !== activeTrackId.trim()
      ) {
        return false;
      }
      const activeCandidateRange = getOwnVal(ws, 'activeCandidateRange');
      const candidateOriginSlot = getOwnVal(ws, 'candidateOriginSlot');
      if (
        !isValidCandidateSlotRange(
          activeCandidateRange,
          seenTrackIds,
          maxSlotTick,
          rawTrimmedByCanonTrackId,
        )
      ) {
        return false;
      }
      if (
        !isValidCandidateSlotRange(
          candidateOriginSlot,
          seenTrackIds,
          maxSlotTick,
          rawTrimmedByCanonTrackId,
        )
      ) {
        return false;
      }
      if (
        activeCandidateRange &&
        candidateOriginSlot &&
        canonSyncId(getOwnVal(activeCandidateRange, 'trackId')) !==
          canonSyncId(getOwnVal(candidateOriginSlot, 'trackId'))
      ) {
        return false;
      }
      const targetStartMeasure = getOwnVal(ws, 'targetStartMeasure');
      const targetEndMeasure = getOwnVal(ws, 'targetEndMeasure');
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
        return false;
      }
      const activeCandidateIndex = getOwnVal(ws, 'activeCandidateIndex');
      if (
        typeof activeCandidateIndex !== 'number' ||
        !Number.isInteger(activeCandidateIndex) ||
        activeCandidateIndex < -1 ||
        activeCandidateIndex > 64
      ) {
        return false;
      }
      const candIdArr = snapshotPlainDenseArray(getOwnVal(ws, 'activeCandidateNoteIds'), 32768);
      if (!candIdArr) {
        return false;
      }
      const seenRawTrimmedByCanonCandId = new Map();
      for (let i = 0; i < candIdArr.length; i += 1) {
        const cid = candIdArr[i];
        if (typeof cid !== 'string' || cid.length > 128) return false;
        let rawTrimmedCid;
        let canonCid;
        if (isFastCleanAsciiSyncId(cid)) {
          rawTrimmedCid = cid;
          canonCid = cid;
        } else {
          rawTrimmedCid = cid.trim();
          if (rawTrimmedCid.length === 0) return false;
          canonCid = truncateWellFormedUtf16(rawTrimmedCid, 128).trim();
          if (!canonCid) return false;
        }
        if (
          rawTrimmedByCanonNoteId.has(canonCid) &&
          rawTrimmedByCanonNoteId.get(canonCid) !== rawTrimmedCid
        ) {
          return false;
        }
        if (
          seenRawTrimmedByCanonCandId.has(canonCid) &&
          seenRawTrimmedByCanonCandId.get(canonCid) !== rawTrimmedCid
        ) {
          return false;
        }
        seenRawTrimmedByCanonCandId.set(canonCid, rawTrimmedCid);
      }
      const quantizeDivision = getOwnVal(ws, 'quantizeDivision');
      if (
        quantizeDivision !== undefined &&
        quantizeDivision !== 4 &&
        quantizeDivision !== 8 &&
        quantizeDivision !== 16 &&
        quantizeDivision !== 32
      ) {
        return false;
      }
      const isTriplet = getOwnVal(ws, 'isTriplet');
      if (isTriplet !== undefined && typeof isTriplet !== 'boolean') {
        return false;
      }
      const candidates = getOwnVal(ws, 'candidates');
      if (!isPlainDenseArray(candidates, 32)) {
        return false;
      }
      let totalCandidateBytes = 0;
      for (let i = 0; i < candidates.length; i += 1) {
        const cand = getOwnVal(candidates, i);
        const accountedBytes = measureValidSerializedCandidate(
          cand,
          MAX_SYNC_PAYLOAD_CHARS - totalCandidateBytes,
        );
        if (accountedBytes === null) return false;
        totalCandidateBytes += accountedBytes;
        if (totalCandidateBytes > MAX_SYNC_PAYLOAD_CHARS) return false;
      }
      if (!isValidGenerationReason(getOwnVal(ws, 'latestReason'))) {
        return false;
      }
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

module.exports = {
  COMPASS_APP_ORIGIN,
  MAX_SYNC_PAYLOAD_CHARS,
  MAX_PLAIN_ARRAY_MIDI_BYTES,
  extractCanonicalOrigin,
  getAllowedAppOrigin,
  isTrustedIpcSender,
  isValidSerializedNote,
  isValidSerializedTrack,
  isValidCandidateSlotRange,
  isValidGenerationReason,
  isValidCandidateMidiBytes,
  isValidSerializedCandidate,
  isValidWorkspaceSyncPayload,
};
