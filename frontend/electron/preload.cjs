/**
 * COMPASS AI Hardened Electron Preload Bridge (`preload.cjs`).
 * Enforces `contextIsolation: true` and `nodeIntegration: false` (`secure-coding-guard`).
 */

const { contextBridge, ipcRenderer } = require('electron');

const MAX_CANDIDATE_MIDI_BYTES = 2_000_000;
const MAX_PLAIN_ARRAY_MIDI_BYTES = 4096;
const MAX_SMF_QUARTER_NOTES = 4128;
const TYPED_ARRAY_PROTO = Object.getPrototypeOf(Uint8Array.prototype);
const TYPED_ARRAY_TAG_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  Symbol.toStringTag,
)?.get;
const TYPED_ARRAY_BUFFER_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'buffer',
)?.get;
const TYPED_ARRAY_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'byteLength',
)?.get;
const TYPED_ARRAY_BYTEOFFSET_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTO,
  'byteOffset',
)?.get;
const ARRAYBUFFER_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength',
)?.get;
const ARRAYBUFFER_DETACHED_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'detached',
)?.get;

const ALL_SYNC_ENVELOPE_KEYS = new Set([
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
const ALLOWED_TRACK_KEYS = new Set([
  'id',
  'name',
  'instrument',
  'colorHex',
  'isMuted',
  'isSoloed',
  'notes',
]);
const ALLOWED_NOTE_KEYS = new Set([
  'id',
  'pitch',
  'startTick',
  'durationTicks',
  'velocity',
  'channel',
]);
const ALLOWED_SLOT_KEYS = new Set(['trackId', 'startTick', 'endTick']);
const ALLOWED_CANDIDATE_KEYS = new Set(['index', 'filename', 'rawMidiBytes', 'reason']);
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
 * Single-pass plain-object snapshot (`sandbox: true` safe — zero Node built-in imports).
 * Reads `Object.getOwnPropertyNames` and each `Object.getOwnPropertyDescriptor` strictly once
 * so stateful `Proxy` wrappers, getters, symbols, or uncloneable extra properties are rejected.
 * @param {unknown} raw
 * @param {Set<string>} allowedSet
 * @returns {Map<string, any> | null}
 */
function snapshotPlainObject(raw, allowedSet) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const proto = Object.getPrototypeOf(raw);
  if (proto !== null && proto !== Object.prototype) return null;
  if (Object.getOwnPropertySymbols(raw).length !== 0) return null;
  const keys = Object.getOwnPropertyNames(raw);
  if (keys.length > allowedSet.size) return null;
  const snap = new Map();
  for (let i = 0; i < keys.length; i += 1) {
    const k = keys[i];
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') return null;
    if (!allowedSet.has(k) || snap.has(k)) return null;
    const desc = Object.getOwnPropertyDescriptor(raw, k);
    if (!desc || !('value' in desc)) return null;
    snap.set(k, desc.value);
  }
  return snap;
}

/**
 * @param {string} key
 * @param {number} maxExclusive
 * @returns {number}
 */
function parseCanonicalArrayIndex(key, maxExclusive) {
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

/**
 * Single-pass dense array snapshot.
 * @param {unknown} raw
 * @param {number} maxItems
 * @returns {any[] | null}
 */
function snapshotPlainDenseArray(raw, maxItems) {
  if (!raw || typeof raw !== 'object' || !Array.isArray(raw)) return null;
  if (Object.getPrototypeOf(raw) !== Array.prototype) return null;
  const lenDesc = Object.getOwnPropertyDescriptor(raw, 'length');
  if (!lenDesc || !('value' in lenDesc)) return null;
  const len = lenDesc.value;
  if (typeof len !== 'number' || !Number.isInteger(len) || len < 0 || len > maxItems) {
    return null;
  }
  if (Object.getOwnPropertySymbols(raw).length !== 0) return null;
  const ownNames = Object.getOwnPropertyNames(raw);
  if (ownNames.length !== len + 1) return null;
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
  const out = new Array(len);
  for (let i = 0; i < len; i += 1) {
    const desc = Object.getOwnPropertyDescriptor(raw, i);
    if (!desc || !('value' in desc)) return null;
    out[i] = desc.value;
  }
  return out;
}

/**
 * Pure ECMAScript UTF-16 truncation (`sandbox: true` safe — zero Node built-in modules).
 * Truncates `raw` to at most `maxUnits` UTF-16 code units without splitting surrogate pairs
 * and replaces lone surrogates 1-for-1 with `\uFFFD` (`[R15-MED-01]`).
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
 * Zero-Map single-pass plain note validator + cloner (`[R24-MED-03]`).
 * @param {unknown} rawNote
 * @param {number} maxProjectTicks
 * @param {Map<string, string>} rawTrimmedByCanonNoteId
 * @returns {any | null}
 */
function readValidatedPlainNote(rawNote, maxProjectTicks, rawTrimmedByCanonNoteId) {
  if (!rawNote || typeof rawNote !== 'object' || Array.isArray(rawNote)) return null;
  const proto = Object.getPrototypeOf(rawNote);
  if (proto !== null && proto !== Object.prototype) return null;
  if (Object.getOwnPropertySymbols(rawNote).length !== 0) return null;
  const keys = Object.getOwnPropertyNames(rawNote);
  if (keys.length < 5 || keys.length > 6) return null;

  let seenMask = 0;
  let nId;
  let pitch;
  let startTick;
  let durationTicks;
  let velocity;
  let channel;

  for (let i = 0; i < keys.length; i += 1) {
    const k = keys[i];
    switch (k) {
      case 'id': {
        if ((seenMask & 1) !== 0) return null;
        seenMask |= 1;
        const desc = Object.getOwnPropertyDescriptor(rawNote, 'id');
        if (!desc || !('value' in desc)) return null;
        nId = desc.value;
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

  if (
    typeof nId !== 'string' ||
    nId.length > 128 ||
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
      (typeof channel !== 'number' || !Number.isFinite(channel) || channel < 0 || channel > 15))
  ) {
    return null;
  }

  let rawTrimmedNoteId;
  let canonNoteId;
  if (isFastCleanAsciiSyncId(nId)) {
    rawTrimmedNoteId = nId;
    canonNoteId = nId;
  } else {
    rawTrimmedNoteId = nId.trim();
    if (rawTrimmedNoteId.length === 0) return null;
    canonNoteId = truncateWellFormedUtf16(rawTrimmedNoteId, 128).trim();
  }
  if (!canonNoteId || rawTrimmedByCanonNoteId.has(canonNoteId)) return null;

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
  return channel !== undefined
    ? { id: nId, pitch, startTick, durationTicks, velocity, channel }
    : { id: nId, pitch, startTick, durationTicks, velocity };
}

/**
 * Computes canonical `beatsPerMeasure` (`0.5..16`, `1/8`-beat resolution) (`[R15-MED-01]`).
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
 * Derives `(numerator, denominator)` from a standalone `beatsPerMeasure` (`[R15-MED-01]`).
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
 * @param {unknown} raw
 * @returns {{ valid: true; value: any } | { valid: false }}
 */
function cloneSafeGenerationReason(raw) {
  if (raw === null || raw === undefined) return { valid: true, value: null };
  const snap = snapshotPlainObject(raw, ALLOWED_REASON_KEYS);
  if (!snap) return { valid: false };
  const source = snap.get('source');
  if (source !== 'model_cot' && source !== 'request' && source !== 'foundation') {
    return { valid: false };
  }
  const task = snap.get('task');
  if (typeof task !== 'string' || task.trim().length === 0 || task.length > 128) {
    return { valid: false };
  }
  const thinking = snap.get('thinking');
  if (typeof thinking !== 'boolean') return { valid: false };

  /** @param {unknown} listVal */
  const cloneStrList = (listVal) => {
    if (listVal === null || listVal === undefined) return { ok: true, list: null };
    const arr = snapshotPlainDenseArray(listVal, 64);
    if (!arr) return { ok: false, list: null };
    const clean = new Array(arr.length);
    for (let i = 0; i < arr.length; i += 1) {
      if (typeof arr[i] !== 'string' || arr[i].length > 64) return { ok: false, list: null };
      clean[i] = arr[i];
    }
    return { ok: true, list: clean };
  };

  const instRes = cloneStrList(snap.get('instruments'));
  if (!instRes.ok) return { valid: false };
  const genreRes = cloneStrList(snap.get('genre'));
  if (!genreRes.ok) return { valid: false };

  const mc = snap.get('gen_measure_count');
  let cleanMc = null;
  if (mc !== null && mc !== undefined) {
    if (typeof mc !== 'number' || !Number.isFinite(mc) || mc < 0 || mc > 512) {
      return { valid: false };
    }
    cleanMc = mc;
  }

  const key = snap.get('key');
  let cleanKey = null;
  if (key !== null && key !== undefined) {
    if (typeof key !== 'string' || key.length > 32) return { valid: false };
    cleanKey = key;
  }

  const nd = snap.get('note_density');
  /** @type {Record<string, number | null> | null} */
  let cleanDensity = null;
  if (nd !== null && nd !== undefined) {
    if (typeof nd !== 'object' || Array.isArray(nd)) return { valid: false };
    const ndProto = Object.getPrototypeOf(nd);
    if (ndProto !== null && ndProto !== Object.prototype) return { valid: false };
    if (Object.getOwnPropertySymbols(nd).length !== 0) return { valid: false };
    const ndKeys = Object.getOwnPropertyNames(nd);
    if (ndKeys.length > 32) return { valid: false };
    const seenNdKeys = new Set();
    cleanDensity = {};
    for (let i = 0; i < ndKeys.length; i += 1) {
      const k = ndKeys[i];
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
      const d = Object.getOwnPropertyDescriptor(nd, k);
      if (!d || !('value' in d)) return { valid: false };
      const v = d.value;
      if (v !== null && (typeof v !== 'number' || !Number.isFinite(v))) {
        return { valid: false };
      }
      Object.defineProperty(cleanDensity, k, {
        value: v,
        writable: true,
        enumerable: true,
        configurable: true,
      });
    }
  }

  return {
    valid: true,
    value: {
      source,
      task,
      thinking,
      instruments: instRes.list,
      note_density: cleanDensity,
      gen_measure_count: cleanMc,
      genre: genreRes.list,
      key: cleanKey,
    },
  };
}

const NATIVE_UINT8ARRAY_CTOR_RE = /^function Uint8Array\(\)\s*\{\s*\[native code\]\s*\}$/;

/**
 * @param {unknown} proto
 * @returns {boolean}
 */
function isGenuineUint8ArrayPrototype(proto) {
  if (proto === Uint8Array.prototype) return true;
  if (typeof Buffer === 'function' && Boolean(Buffer.prototype) && proto === Buffer.prototype) {
    return true;
  }
  if (!proto || typeof proto !== 'object') return false;
  const typedArrProto = Object.getPrototypeOf(proto);
  if (!typedArrProto || typeof typedArrProto !== 'object') return false;
  const objProto = Object.getPrototypeOf(typedArrProto);
  if (!objProto || typeof objProto !== 'object' || Object.getPrototypeOf(objProto) !== null) {
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
  if (!ctorDesc || !('value' in ctorDesc) || typeof ctorDesc.value !== 'function') return false;
  const ctor = ctorDesc.value;
  if (!NATIVE_UINT8ARRAY_CTOR_RE.test(Function.prototype.toString.call(ctor))) return false;
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
 * @param {unknown} raw
 * @param {number} [maxAllowedBytes]
 * @returns {{ bytes: Uint8Array; accountedBytes: number } | null}
 */
function cloneSafeCandidateMidiBytes(raw, maxAllowedBytes = MAX_CANDIDATE_MIDI_BYTES) {
  const effectiveCap =
    typeof maxAllowedBytes === 'number' && Number.isFinite(maxAllowedBytes)
      ? Math.min(MAX_CANDIDATE_MIDI_BYTES, Math.floor(maxAllowedBytes))
      : MAX_CANDIDATE_MIDI_BYTES;
  if (effectiveCap < 22) return null;
  if (!raw || typeof raw !== 'object') return null;
  if (
    TYPED_ARRAY_TAG_GETTER &&
    TYPED_ARRAY_BYTELENGTH_GETTER &&
    TYPED_ARRAY_BYTEOFFSET_GETTER &&
    TYPED_ARRAY_BUFFER_GETTER &&
    ARRAYBUFFER_BYTELENGTH_GETTER
  ) {
    let tag = '';
    try {
      tag = TYPED_ARRAY_TAG_GETTER.call(raw);
    } catch {
      tag = '';
    }
    if (tag === 'Uint8Array') {
      if (!isGenuineUint8ArrayPrototype(Object.getPrototypeOf(raw))) return null;
      if (Object.getOwnPropertySymbols(raw).length !== 0) return null;
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
        viewLen < 22 ||
        viewLen > effectiveCap ||
        backingLen > effectiveCap ||
        byteOffset + viewLen > backingLen
      ) {
        return null;
      }
      if (viewLen <= 4096) {
        const ownNames = Object.getOwnPropertyNames(raw);
        if (ownNames.length !== viewLen) return null;
        for (let i = 0; i < viewLen; i += 1) {
          const desc = Object.getOwnPropertyDescriptor(raw, i);
          if (!desc || !('value' in desc)) return null;
        }
      }
      const hdr = new Uint8Array(backingBuf, byteOffset, 4);
      if (hdr[0] !== 0x4d || hdr[1] !== 0x54 || hdr[2] !== 0x68 || hdr[3] !== 0x64) {
        return null;
      }
      const copy = new Uint8Array(viewLen);
      copy.set(new Uint8Array(backingBuf, byteOffset, viewLen));
      return { bytes: copy, accountedBytes: Math.max(viewLen, backingLen) };
    }
  }

  const plainCap = Math.min(MAX_PLAIN_ARRAY_MIDI_BYTES, effectiveCap);
  const rawByteArr = snapshotPlainDenseArray(raw, plainCap);
  if (rawByteArr !== null) {
    const len = rawByteArr.length;
    if (
      len < 22 ||
      rawByteArr[0] !== 0x4d ||
      rawByteArr[1] !== 0x54 ||
      rawByteArr[2] !== 0x68 ||
      rawByteArr[3] !== 0x64
    ) {
      return null;
    }
    const copy = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      const b = rawByteArr[i];
      if (typeof b !== 'number' || !Number.isInteger(b) || b < 0 || b > 255) return null;
      copy[i] = b;
    }
    return { bytes: copy, accountedBytes: len };
  }
  return null;
}

/**
 * Validates and deep-clones all 4 `WorkspaceSyncMessage` variants into fresh plain POJOs /
 * clean `Uint8Array`s so `ipcRenderer.send` can never encounter `Proxy`, functions, `Symbol`s,
 * getters, or detached `ArrayBuffer`s (`[R14-MED-02]`, `[R14-MED-03]`).
 * @param {unknown} msg
 * @returns {Record<string, any> | null}
 */
function cloneSafeWorkspaceSyncMessage(msg) {
  try {
    const envSnap = snapshotPlainObject(msg, ALL_SYNC_ENVELOPE_KEYS);
    if (!envSnap) return null;
    const msgType = envSnap.get('type');
    let allowedEnvKeys;
    if (msgType === 'FOCUS_TAB') allowedEnvKeys = ALLOWED_FOCUS_MSG_KEYS;
    else if (msgType === 'GENERATION_STATUS') allowedEnvKeys = ALLOWED_GEN_STATUS_MSG_KEYS;
    else if (msgType === 'REQUEST_WORKSPACE_STATE') allowedEnvKeys = ALLOWED_REQ_STATE_MSG_KEYS;
    else if (msgType === 'SYNC_WORKSPACE_STATE') allowedEnvKeys = ALLOWED_SYNC_STATE_MSG_KEYS;
    else if (msgType === 'SYNC_ACTIVE_CONTROLLER') allowedEnvKeys = ALLOWED_SYNC_CONTROLLER_MSG_KEYS;
    else return null;

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
      const tab = envSnap.get('tab');
      if (tab !== 'compass-ai' && tab !== 'pianoroll') return null;
      const reason = envSnap.get('reason');
      if (reason !== undefined && (typeof reason !== 'string' || reason.length > 256)) {
        return null;
      }
      return reason !== undefined
        ? { messageId, type: 'FOCUS_TAB', tab, reason, senderId }
        : { messageId, type: 'FOCUS_TAB', tab, senderId };
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
      return { messageId, type: 'GENERATION_STATUS', isGenerating, statusDetail, senderId };
    }

    if (msgType === 'REQUEST_WORKSPACE_STATE') {
      return { messageId, type: 'REQUEST_WORKSPACE_STATE', senderId };
    }

    if (msgType === 'SYNC_ACTIVE_CONTROLLER') {
      const controllerId = envSnap.get('controllerId');
      if (typeof controllerId !== 'string' || !VALID_SYNC_CONTROLLER_IDS.has(controllerId)) {
        return null;
      }
      return { messageId, type: 'SYNC_ACTIVE_CONTROLLER', controllerId, senderId };
    }

    // `SYNC_WORKSPACE_STATE`
    const wsSnap = snapshotPlainObject(envSnap.get('workspaceState'), ALLOWED_WORKSPACE_KEYS);
    if (!wsSnap) return null;

    const bpm = wsSnap.get('bpm');
    if (typeof bpm !== 'number' || !Number.isFinite(bpm) || bpm < 20 || bpm > 300) return null;
    const ppqn = wsSnap.get('ppqn');
    if (
      ppqn !== undefined &&
      (typeof ppqn !== 'number' || !Number.isFinite(ppqn) || ppqn < 24 || ppqn > 32767)
    ) {
      return null;
    }
    const effectivePpqn = ppqn !== undefined ? Math.round(ppqn) : 480;
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * effectivePpqn;
    const maxSlotTick = 16512 * effectivePpqn;

    const timeSigNum = wsSnap.get('timeSigNumerator');
    const timeSigDen = wsSnap.get('timeSigDenominator');
    const beatsPerMeasure = wsSnap.get('beatsPerMeasure');
    if ((timeSigNum === undefined) !== (timeSigDen === undefined)) return null;
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

    const activeTrackId = wsSnap.get('activeTrackId');
    if (
      typeof activeTrackId !== 'string' ||
      activeTrackId.trim().length === 0 ||
      activeTrackId.length > 128
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
    if (isTriplet !== undefined && typeof isTriplet !== 'boolean') return null;

    const activeCandidateIndex = wsSnap.get('activeCandidateIndex');
    if (
      typeof activeCandidateIndex !== 'number' ||
      !Number.isInteger(activeCandidateIndex) ||
      activeCandidateIndex < -1 ||
      activeCandidateIndex > 64
    ) {
      return null;
    }

    const rawTracks = snapshotPlainDenseArray(wsSnap.get('tracks'), 64);
    if (!rawTracks || rawTracks.length === 0) return null;
    const rawTrimmedByCanonTrackId = new Map();
    const rawTrimmedByCanonNoteId = new Map();
    const cleanTracks = new Array(rawTracks.length);
    for (let ti = 0; ti < rawTracks.length; ti += 1) {
      const tSnap = snapshotPlainObject(rawTracks[ti], ALLOWED_TRACK_KEYS);
      if (!tSnap) return null;
      const tId = tSnap.get('id');
      const tName = tSnap.get('name');
      const tInst = tSnap.get('instrument');
      const tColor = tSnap.get('colorHex');
      const tMuted = tSnap.get('isMuted');
      const tSoloed = tSnap.get('isSoloed');
      if (
        typeof tId !== 'string' ||
        tId.trim().length === 0 ||
        tId.length > 128 ||
        typeof tName !== 'string' ||
        tName.trim().length === 0 ||
        tName.length > 256 ||
        typeof tInst !== 'string' ||
        !VALID_TRACK_INSTRUMENTS.has(tInst) ||
        typeof tColor !== 'string' ||
        tColor.trim().length === 0 ||
        tColor.length > 32 ||
        typeof tMuted !== 'boolean' ||
        typeof tSoloed !== 'boolean'
      ) {
        return null;
      }
      const rawTrimmedTrackId = tId.trim();
      const canonTrackId = canonSyncId(tId);
      if (!canonTrackId || rawTrimmedByCanonTrackId.has(canonTrackId)) return null;
      const rawNotes = snapshotPlainDenseArray(
        tSnap.get('notes'),
        32768 - rawTrimmedByCanonNoteId.size,
      );
      if (!rawNotes || rawTrimmedByCanonNoteId.size + rawNotes.length > 32768) return null;
      rawTrimmedByCanonTrackId.set(canonTrackId, rawTrimmedTrackId);
      const cleanNotes = new Array(rawNotes.length);
      for (let ni = 0; ni < rawNotes.length; ni += 1) {
        const cleanNote = readValidatedPlainNote(
          rawNotes[ni],
          maxProjectTicks,
          rawTrimmedByCanonNoteId,
        );
        if (!cleanNote) return null;
        cleanNotes[ni] = cleanNote;
      }
      cleanTracks[ti] = {
        id: tId,
        name: tName,
        instrument: tInst,
        colorHex: tColor,
        isMuted: tMuted,
        isSoloed: tSoloed,
        notes: cleanNotes,
      };
    }

    const canonActiveTrackId = canonSyncId(activeTrackId);
    if (
      !canonActiveTrackId ||
      rawTrimmedByCanonTrackId.get(canonActiveTrackId) !== activeTrackId.trim()
    ) {
      return null;
    }

    /** @param {unknown} rawSlot */
    const cloneSlot = (rawSlot) => {
      if (rawSlot === undefined) {
        return { ok: true, canonTrackId: null, present: false, val: null };
      }
      if (rawSlot === null) {
        return { ok: true, canonTrackId: null, present: true, val: null };
      }
      const sSnap = snapshotPlainObject(rawSlot, ALLOWED_SLOT_KEYS);
      if (!sSnap) return { ok: false, canonTrackId: null, present: false, val: null };
      const sTrackId = sSnap.get('trackId');
      const sStart = sSnap.get('startTick');
      const sEnd = sSnap.get('endTick');
      if (typeof sTrackId !== 'string' || sTrackId.trim().length === 0 || sTrackId.length > 128) {
        return { ok: false, canonTrackId: null, present: false, val: null };
      }
      const canonSlotTrackId = canonSyncId(sTrackId);
      if (
        !canonSlotTrackId ||
        rawTrimmedByCanonTrackId.get(canonSlotTrackId) !== sTrackId.trim() ||
        typeof sStart !== 'number' ||
        !Number.isFinite(sStart) ||
        sStart < 0 ||
        typeof sEnd !== 'number' ||
        !Number.isFinite(sEnd) ||
        sEnd < sStart ||
        sEnd > maxSlotTick
      ) {
        return { ok: false, canonTrackId: null, present: false, val: null };
      }
      return {
        ok: true,
        canonTrackId: canonSlotTrackId,
        present: true,
        val: { trackId: sTrackId, startTick: sStart, endTick: sEnd },
      };
    };

    const actSlot = cloneSlot(
      wsSnap.has('activeCandidateRange') ? wsSnap.get('activeCandidateRange') : undefined,
    );
    const origSlot = cloneSlot(
      wsSnap.has('candidateOriginSlot') ? wsSnap.get('candidateOriginSlot') : undefined,
    );
    if (!actSlot.ok || !origSlot.ok) return null;
    if (
      actSlot.canonTrackId !== null &&
      origSlot.canonTrackId !== null &&
      actSlot.canonTrackId !== origSlot.canonTrackId
    ) {
      return null;
    }

    const rawCandIds = snapshotPlainDenseArray(wsSnap.get('activeCandidateNoteIds'), 32768);
    if (!rawCandIds) return null;
    const cleanCandIds = new Array(rawCandIds.length);
    const seenRawTrimmedByCanonCandId = new Map();
    for (let i = 0; i < rawCandIds.length; i += 1) {
      const cid = rawCandIds[i];
      if (typeof cid !== 'string' || cid.length > 128) return null;
      let rawTrimmedCid;
      let canonCid;
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
      cleanCandIds[i] = cid;
    }

    const rawCands = snapshotPlainDenseArray(wsSnap.get('candidates'), 32);
    if (!rawCands) return null;
    const cleanCands = new Array(rawCands.length);
    let totalBytes = 0;
    for (let ci = 0; ci < rawCands.length; ci += 1) {
      const cSnap = snapshotPlainObject(rawCands[ci], ALLOWED_CANDIDATE_KEYS);
      if (!cSnap) return null;
      const cIdx = cSnap.get('index');
      const cName = cSnap.get('filename');
      if (
        typeof cIdx !== 'number' ||
        !Number.isInteger(cIdx) ||
        cIdx < 0 ||
        cIdx > 64 ||
        typeof cName !== 'string' ||
        cName.trim().length === 0 ||
        cName.length > 256
      ) {
        return null;
      }
      const bytesRes = cloneSafeCandidateMidiBytes(
        cSnap.get('rawMidiBytes'),
        MAX_CANDIDATE_MIDI_BYTES - totalBytes,
      );
      if (!bytesRes) return null;
      totalBytes += bytesRes.accountedBytes;
      if (totalBytes > MAX_CANDIDATE_MIDI_BYTES) return null;
      const reasonRes = cloneSafeGenerationReason(cSnap.get('reason'));
      if (!reasonRes.valid) return null;
      cleanCands[ci] = {
        index: cIdx,
        filename: cName,
        rawMidiBytes: bytesRes.bytes,
        reason: reasonRes.value,
      };
    }

    const hasLatestReason = wsSnap.has('latestReason');
    const latestReasonRes = cloneSafeGenerationReason(wsSnap.get('latestReason'));
    if (!latestReasonRes.valid) return null;

    /** @type {Record<string, any>} */
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
      candidates: cleanCands,
      activeCandidateIndex,
      activeCandidateNoteIds: cleanCandIds,
      ...(actSlot.present ? { activeCandidateRange: actSlot.val } : {}),
      ...(origSlot.present ? { candidateOriginSlot: origSlot.val } : {}),
      ...(hasLatestReason ? { latestReason: latestReasonRes.value } : {}),
    };

    return {
      messageId,
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: cleanWs,
      senderId,
    };
  } catch {
    return null;
  }
}

function resolveArgValue(prefix, fallback) {
  try {
    const rawArgv = typeof process === 'object' && process !== null ? process.argv : null;
    if (!Array.isArray(rawArgv)) return fallback;
    const lenDesc = Object.getOwnPropertyDescriptor(rawArgv, 'length');
    const len =
      lenDesc && 'value' in lenDesc && typeof lenDesc.value === 'number' && Number.isFinite(lenDesc.value)
        ? Math.min(256, Math.max(0, Math.trunc(lenDesc.value)))
        : 0;
    for (let i = 0; i < len; i += 1) {
      const desc = Object.getOwnPropertyDescriptor(rawArgv, i);
      const arg = desc && 'value' in desc ? desc.value : undefined;
      if (typeof arg === 'string' && arg.startsWith(prefix)) {
        return arg.slice(prefix.length);
      }
    }
  } catch {
    // Ignore inaccessible or malformed argv in restricted preload contexts
  }
  return fallback;
}

let currentApiBaseUrl = resolveArgValue('--compass-api-url=', 'http://127.0.0.1:8000');
let currentHandshakeToken = '';
let currentBackendReady = false;
let currentHasPeerWindows = true;
const configListeners = new Set();

const ALLOWED_API_CONFIG_KEYS = new Set([
  'apiBaseUrl',
  'handshakeToken',
  'backendReady',
  'hasPeerWindows',
]);
const VALID_API_BASE_URL_RE = /^https?:\/\/[^\s\x00-\x1f\x7f]+$/i;

function applyIncomingApiConfig(cfg) {
  let nextUrl = currentApiBaseUrl;
  let nextToken = currentHandshakeToken;
  let nextReady = currentBackendReady;
  let nextHasPeers = currentHasPeerWindows;
  let apiConfigChanged = false;
  let peerChanged = false;
  try {
    const snap = snapshotPlainObject(cfg, ALLOWED_API_CONFIG_KEYS);
    if (!snap) return;
    if (Object.prototype.toString.call(cfg) !== '[object Object]') return;

    const urlVal = snap.get('apiBaseUrl');
    const tokVal = snap.get('handshakeToken');
    const rdyVal = snap.get('backendReady');
    const peerVal = snap.get('hasPeerWindows');

    if (urlVal !== undefined) {
      if (typeof urlVal !== 'string') return;
      const trimmedUrl = urlVal.trim();
      if (
        trimmedUrl.length === 0 ||
        trimmedUrl.length > 2048 ||
        !VALID_API_BASE_URL_RE.test(trimmedUrl)
      ) {
        return;
      }
      if (trimmedUrl !== nextUrl) {
        nextUrl = trimmedUrl;
        apiConfigChanged = true;
      }
    }
    if (tokVal !== undefined) {
      if (typeof tokVal !== 'string' || tokVal.length > 512) return;
      if (tokVal !== nextToken) {
        nextToken = tokVal;
        apiConfigChanged = true;
      }
    }
    if (rdyVal !== undefined) {
      if (typeof rdyVal !== 'boolean') return;
      if (rdyVal !== nextReady) {
        nextReady = rdyVal;
        apiConfigChanged = true;
      }
    }
    if (peerVal !== undefined) {
      if (typeof peerVal !== 'boolean') return;
      if (peerVal !== nextHasPeers) {
        nextHasPeers = peerVal;
        peerChanged = true;
      }
    }
  } catch {
    return;
  }
  if (!apiConfigChanged && !peerChanged) return;
  currentApiBaseUrl = nextUrl;
  currentHandshakeToken = nextToken;
  currentBackendReady = nextReady;
  currentHasPeerWindows = nextHasPeers;
  if (!apiConfigChanged) return;
  const snapshot = Object.freeze({
    apiBaseUrl: currentApiBaseUrl,
    handshakeToken: currentHandshakeToken,
    backendReady: currentBackendReady,
  });
  for (const listener of Array.from(configListeners)) {
    try {
      listener(snapshot);
    } catch {
      // Ignore listener error
    }
  }
}

ipcRenderer.on('compass:api-config', (_event, cfg) => {
  applyIncomingApiConfig(cfg);
});

void ipcRenderer
  .invoke('compass:get-api-config')
  .then((cfg) => {
    applyIncomingApiConfig(cfg);
  })
  .catch(() => {
    // Handler not registered in standalone browser environments
  });

contextBridge.exposeInMainWorld('compassDesktop', Object.freeze({
  apiBaseUrl: currentApiBaseUrl,

  hasPeerWindows: () => currentHasPeerWindows,

  getApiConfig: () =>
    Object.freeze({
      apiBaseUrl: currentApiBaseUrl,
      handshakeToken: currentHandshakeToken,
      backendReady: currentBackendReady,
    }),

  requestApiConfig: (forceRefresh) =>
    (forceRefresh === true
      ? ipcRenderer.invoke('compass:get-api-config', true)
      : ipcRenderer.invoke('compass:get-api-config')
    )
      .then((cfg) => {
        applyIncomingApiConfig(cfg);
        return Object.freeze({
          apiBaseUrl: currentApiBaseUrl,
          handshakeToken: currentHandshakeToken,
          backendReady: currentBackendReady,
        });
      })
      .catch(() =>
        Object.freeze({
          apiBaseUrl: currentApiBaseUrl,
          handshakeToken: currentHandshakeToken,
          backendReady: currentBackendReady,
        }),
      ),

  onApiConfigChanged: (callback) => {
    if (typeof callback !== 'function') return () => {};
    configListeners.add(callback);
    return () => {
      configListeners.delete(callback);
    };
  },

  bringWindowToFront: (targetTab) => {
    const safeTab = targetTab === 'pianoroll' ? 'pianoroll' : 'compass-ai';
    return ipcRenderer.invoke('compass:bring-to-front', safeTab);
  },

  openPianoRollWindow: () => {
    return ipcRenderer.invoke('compass:open-pianoroll-window');
  },

  sendWorkspaceSync: (msg) => {
    try {
      const cleanMsg = cloneSafeWorkspaceSyncMessage(msg);
      if (!cleanMsg) return;
      ipcRenderer.send('compass:workspace-sync', cleanMsg);
    } catch {
      // Never throw synchronously into renderer callers on hostile/uncloneable inputs
    }
  },

  onWorkspaceSync: (callback) => {
    if (typeof callback !== 'function') return () => {};
    const handler = (_event, payload) => {
      try {
        callback(payload);
      } catch {
        // Isolate subscriber exception so sibling IPC listeners keep firing
      }
    };
    ipcRenderer.on('compass:workspace-sync', handler);
    return () => {
      ipcRenderer.removeListener('compass:workspace-sync', handler);
    };
  },

  onBootLog: (callback) => {
    if (typeof callback !== 'function') return () => {};
    const handler = (_event, line) => {
      if (typeof line === 'string') {
        try {
          callback(line);
        } catch {
          // Isolate subscriber exception
        }
      }
    };
    ipcRenderer.on('compass:boot-log', handler);
    return () => {
      ipcRenderer.removeListener('compass:boot-log', handler);
    };
  },

  onFocusTabRequest: (callback) => {
    if (typeof callback !== 'function') return () => {};
    const handler = (_event, tab) => {
      if (tab === 'compass-ai' || tab === 'pianoroll') {
        try {
          callback(tab);
        } catch {
          // Isolate subscriber exception
        }
      }
    };
    ipcRenderer.on('compass:focus-tab', handler);
    return () => {
      ipcRenderer.removeListener('compass:focus-tab', handler);
    };
  },
}));
