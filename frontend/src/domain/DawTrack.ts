/**
 * Encapsulated DAW Track Entity (`oop-extensible-craft`).
 * Maintains `MidiNote[]` sorted by `startTick` together with a flat implicit balanced
 * interval-tree `subtreeMaxEndTick[]` index so both start-window queries and true
 * interval-overlap queries (`note.startTick < endTick && note.endTick > startTick`,
 * including long sustaining notes starting arbitrarily earlier) execute in O(log N + K) time.
 * Also caches immutable `DawTrackSnapshot` instances until track state mutates.
 */

import {
  createNoteId,
  isDomainProxy,
  isGenuineMidiNote,
  MidiNote,
  safeIsArray,
  truncateWellFormedUtf16,
} from './MidiNote';

export type TrackInstrument = 'PIANO' | 'SAX' | 'BASS' | 'VIOLIN' | 'SYNTH' | 'GUITAR' | 'DRUMS';

export const TRACK_SWATCHES = [
  '#38BDF8', // Primary Sky Accent
  '#93C5FD', // Pale Steel Blue
  '#D4D4D8', // Studio Zinc 300
  '#7DD3FC', // Light Sky
  '#A1A1AA', // Studio Zinc 400
  '#60A5FA', // Muted Blue
] as const;

function mintTrackRandomEntropy(byteCount = 4): string {
  try {
    const g = globalThis.crypto;
    if (g && typeof g.getRandomValues === 'function') {
      const buf = new Uint8Array(byteCount);
      g.getRandomValues(buf);
      let out = '';
      for (let i = 0; i < buf.length; i += 1) {
        out += buf[i].toString(16).padStart(2, '0');
      }
      return out;
    }
  } catch {
    // Fallback below when WebCrypto is unavailable
  }
  return Math.floor(Math.random() * 0xffffffff)
    .toString(16)
    .padStart(8, '0');
}

const _trackInstanceEntropy = mintTrackRandomEntropy(4);
let _trackSeq = 1;
export function createTrackId(): string {
  _trackSeq += 1;
  return `trk_${_trackInstanceEntropy}_${Date.now().toString(36)}_${_trackSeq.toString(36)}`;
}

export interface DawTrackSnapshot {
  readonly id: string;
  readonly name: string;
  readonly instrument: TrackInstrument;
  readonly colorHex: string;
  readonly isMuted: boolean;
  readonly isSoloed: boolean;
  readonly isMonophonic: boolean;
  readonly notes: readonly MidiNote[];
  readonly subtreeMaxEndTick: readonly number[];
}

function readOwnDataField(obj: object, key: string): { ok: true; val: unknown } | { ok: false } {
  try {
    const desc = Object.getOwnPropertyDescriptor(obj, key);
    if (!desc) return { ok: true, val: undefined };
    if (!('value' in desc)) return { ok: false };
    return { ok: true, val: desc.value };
  } catch {
    return { ok: false };
  }
}

/**
 * Normalizes a note entry for storage inside `DawTrack` (`[R15-LOW-01]`).
 * - `O(1)` zero-allocation fast path for genuine `MidiNote` instances (`WeakSet`-backed,
 *   preserving Undo/Redo reference identity while rejecting Browser-Mode `Proxy(midiNote)`).
 * - Safe single-pass own-descriptor normalization for plain duck-typed note objects into frozen
 *   `MidiNote` instances so `ReplaceRangeNotesCommand` (`note.withTiming`) and binary-search
 *   invariants can never be broken by missing methods, negative ticks, or post-insert mutation.
 */
export function normalizeTrackInputNote(raw: unknown): MidiNote | null {
  if (isGenuineMidiNote(raw)) {
    return raw;
  }
  try {
    if (!raw || typeof raw !== 'object' || safeIsArray(raw) || isDomainProxy(raw)) {
      return null;
    }
    const proto = Object.getPrototypeOf(raw);
    if (proto !== null && proto !== Object.prototype) {
      return null;
    }
    const idRes = readOwnDataField(raw, 'id');
    const pitchRes = readOwnDataField(raw, 'pitch');
    const startRes = readOwnDataField(raw, 'startTick');
    const durRes = readOwnDataField(raw, 'durationTicks');
    const endRes = readOwnDataField(raw, 'endTick');
    const velRes = readOwnDataField(raw, 'velocity');
    const chRes = readOwnDataField(raw, 'channel');
    if (
      !idRes.ok ||
      !pitchRes.ok ||
      !startRes.ok ||
      !durRes.ok ||
      !endRes.ok ||
      !velRes.ok ||
      !chRes.ok
    ) {
      return null;
    }
    const pitch = pitchRes.val;
    const startTick = startRes.val;
    if (
      typeof pitch !== 'number' ||
      !Number.isFinite(pitch) ||
      pitch < 0 ||
      pitch > 127 ||
      typeof startTick !== 'number' ||
      !Number.isFinite(startTick) ||
      startTick < 0
    ) {
      return null;
    }
    const dur =
      typeof durRes.val === 'number' && Number.isFinite(durRes.val)
        ? durRes.val
        : typeof endRes.val === 'number' && Number.isFinite(endRes.val)
          ? endRes.val - startTick
          : Number.NaN;
    if (!Number.isFinite(dur) || dur <= 0) {
      return null;
    }
    let velocity = 100;
    if (velRes.val !== undefined) {
      if (
        typeof velRes.val !== 'number' ||
        !Number.isFinite(velRes.val) ||
        velRes.val < 1 ||
        velRes.val > 127
      ) {
        return null;
      }
      velocity = velRes.val;
    }
    let channel = 0;
    if (chRes.val !== undefined) {
      if (
        typeof chRes.val !== 'number' ||
        !Number.isFinite(chRes.val) ||
        chRes.val < 0 ||
        chRes.val > 15
      ) {
        return null;
      }
      channel = chRes.val;
    }
    if (idRes.val !== undefined && typeof idRes.val !== 'string') {
      return null;
    }
    return new MidiNote({
      id: typeof idRes.val === 'string' ? idRes.val : undefined,
      pitch,
      startTick,
      durationTicks: dur,
      velocity,
      channel,
    });
  } catch {
    return null;
  }
}

export const MAX_NOTES_PER_TRACK = 32_768;
export const MAX_WORKSPACE_NOTES = 32_768;

export function normalizeTrackNoteArray(
  rawNotes: unknown,
  maxNotes = MAX_NOTES_PER_TRACK,
): MidiNote[] {
  if (!safeIsArray(rawNotes)) return [];
  const cap =
    typeof maxNotes === 'number' && Number.isFinite(maxNotes)
      ? Math.max(0, Math.min(MAX_NOTES_PER_TRACK, Math.trunc(maxNotes)))
      : MAX_NOTES_PER_TRACK;
  if (cap === 0) return [];
  let len = 0;
  try {
    const rawLen = rawNotes.length;
    len =
      typeof rawLen === 'number' && Number.isFinite(rawLen) ? Math.max(0, Math.trunc(rawLen)) : 0;
  } catch {
    return [];
  }
  const out: MidiNote[] = [];
  const seenIds = new Set<string>();
  for (let i = 0; i < len && out.length < cap; i += 1) {
    let elem: unknown;
    try {
      elem = rawNotes[i];
    } catch {
      continue;
    }
    const normalized = isGenuineMidiNote(elem) ? elem : normalizeTrackInputNote(elem);
    if (normalized !== null) {
      const prevSeenSize = seenIds.size;
      seenIds.add(normalized.id);
      if (seenIds.size !== prevSeenSize) {
        out.push(normalized);
      }
    }
  }
  return out;
}

function fastCanonicalizeNoteOrTrackId(elem: string): string {
  const len = elem.length;
  if (len === 0) return '';
  const c0 = elem.charCodeAt(0);
  const cLast = elem.charCodeAt(len - 1);
  if (len <= 128 && c0 > 0x20 && c0 < 0x7f && cLast > 0x20 && cLast < 0x7f) {
    let hasSurrogate = false;
    for (let i = 0; i < len; i += 1) {
      const c = elem.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdfff) {
        hasSurrogate = true;
        break;
      }
    }
    if (!hasSurrogate) {
      return elem;
    }
  }
  const trimmed = elem.trim();
  if (!trimmed) return '';
  return truncateWellFormedUtf16(trimmed, 128).trim();
}

const PRIVATE_OWNED_SET_SENTINEL = Symbol('ReadonlyCanonicalIdSet.owned');
const PRIVATE_BATCH_SENTINEL = Symbol('PreparedCandidateBatch.owned');

let createOwnedCanonicalIdSet!: (owned: Set<string>) => ReadonlyCanonicalIdSet;
let unwrapCanonicalIdSetInner!: (val: unknown) => Set<string> | null;

/**
 * Unforgeable, lexically sealed `ReadonlySet<string>` Value Object (`oop-extensible-craft` §3.2, §3.3).
 * Backed by an ECMAScript `#inner` private `Set<string>` that no external code can ever mutate,
 * allowing O(1) zero-copy reuse across candidate switches and Undo/Redo commands while providing
 * O(N) C++ Torque fast-cloning (`new Set(this.#inner)`) when a mutable selection copy is needed.
 */
export class ReadonlyCanonicalIdSet implements ReadonlySet<string> {
  readonly #inner: Set<string>;

  static {
    createOwnedCanonicalIdSet = (owned: Set<string>): ReadonlyCanonicalIdSet =>
      new ReadonlyCanonicalIdSet(PRIVATE_OWNED_SET_SENTINEL, owned as unknown as number);
    unwrapCanonicalIdSetInner = (val: unknown): Set<string> | null => {
      if (
        val &&
        typeof val === 'object' &&
        #inner in val &&
        !isDomainProxy(val) &&
        Object.getPrototypeOf(val) === ReadonlyCanonicalIdSet.prototype
      ) {
        return (val as ReadonlyCanonicalIdSet).#inner;
      }
      return null;
    };
  }

  constructor(rawIds?: unknown, maxCount = 32_768) {
    if (rawIds === PRIVATE_OWNED_SET_SENTINEL) {
      this.#inner = maxCount as unknown as Set<string>;
      Object.freeze(this);
      return;
    }
    this.#inner = normalizeTrackIdIterable(rawIds, maxCount);
    Object.freeze(this);
  }

  public get size(): number {
    return this.#inner.size;
  }

  public has(value: string): boolean {
    return this.#inner.has(value);
  }

  public forEach(
    callbackfn: (value: string, value2: string, set: ReadonlySet<string>) => void,
    thisArg?: unknown,
  ): void {
    for (const v of this.#inner) {
      callbackfn.call(thisArg, v, v, this);
    }
  }

  public entries(): SetIterator<[string, string]> {
    return this.#inner.entries();
  }

  public keys(): SetIterator<string> {
    return this.#inner.keys();
  }

  public values(): SetIterator<string> {
    return this.#inner.values();
  }

  public [Symbol.iterator](): SetIterator<string> {
    return this.#inner[Symbol.iterator]();
  }

  public cloneMutableSet(): Set<string> {
    return new Set(this.#inner);
  }
}

export const EMPTY_CANONICAL_ID_SET: ReadonlyCanonicalIdSet =
  createOwnedCanonicalIdSet(new Set<string>());

export function normalizeOrReuseCanonicalIdSet(
  rawIds: unknown,
  maxCount = 32_768,
): ReadonlyCanonicalIdSet {
  const inner = unwrapCanonicalIdSetInner(rawIds);
  if (inner !== null && inner.size <= maxCount) {
    return rawIds as ReadonlyCanonicalIdSet;
  }
  return createOwnedCanonicalIdSet(normalizeTrackIdIterable(rawIds, maxCount));
}

export function cloneCanonicalIdSetMutable(ids: ReadonlySet<string>): Set<string> {
  const inner = unwrapCanonicalIdSetInner(ids);
  if (inner !== null) {
    return new Set(inner);
  }
  return new Set(ids);
}

const ALWAYS_FALSE_ID_PREDICATE = (): boolean => false;

/**
 * Returns a direct O(1) membership predicate over `ids` without exposing any internal mutable `Set`
 * (`oop-extensible-craft` §3.3 Information Hiding, `[R25-MED-01]`).
 */
export function createFastCanonicalIdHas(
  ids: ReadonlySet<string>,
): (canonicalId: string) => boolean {
  const inner = unwrapCanonicalIdSetInner(ids);
  if (inner !== null) {
    if (inner.size === 0) {
      return ALWAYS_FALSE_ID_PREDICATE;
    }
    return (canonicalId: string) => inner.has(canonicalId);
  }
  if (!ids || typeof ids !== 'object' || isDomainProxy(ids) || typeof ids.has !== 'function') {
    return ALWAYS_FALSE_ID_PREDICATE;
  }
  return (canonicalId: string) => ids.has(canonicalId);
}

let isPreparedCandidateBatchInternal!: (val: unknown) => val is PreparedCandidateBatch;
let getPreparedBatchNotes!: (batch: PreparedCandidateBatch) => readonly MidiNote[];
let takePreparedBatchIdsOnce!: (batch: PreparedCandidateBatch) => Set<string>;

/**
 * Lexically sealed batch of genuine, workspace-unique `MidiNote`s prepared for atomic track placement.
 */
export class PreparedCandidateBatch {
  readonly #notes: readonly MidiNote[];
  #transferableIds: Set<string> | null;

  static {
    isPreparedCandidateBatchInternal = (val: unknown): val is PreparedCandidateBatch =>
      Boolean(
        val &&
          typeof val === 'object' &&
          #notes in val &&
          !isDomainProxy(val) &&
          Object.getPrototypeOf(val) === PreparedCandidateBatch.prototype,
      );
    getPreparedBatchNotes = (batch: PreparedCandidateBatch): readonly MidiNote[] => batch.#notes;
    takePreparedBatchIdsOnce = (batch: PreparedCandidateBatch): Set<string> => {
      const owned = batch.#transferableIds;
      if (owned !== null) {
        batch.#transferableIds = null;
        return owned;
      }
      const rebuilt = new Set<string>();
      for (let i = 0; i < batch.#notes.length; i += 1) {
        rebuilt.add(batch.#notes[i].id);
      }
      return rebuilt;
    };
  }

  constructor(sentinel: unknown, notes: readonly MidiNote[], ids: Set<string>) {
    if (sentinel !== PRIVATE_BATCH_SENTINEL) {
      throw new Error('PreparedCandidateBatch must be constructed via prepareScaledCandidateBatch.');
    }
    this.#notes = notes;
    this.#transferableIds = ids;
    Object.freeze(this);
  }
}

export function isPreparedCandidateBatch(val: unknown): val is PreparedCandidateBatch {
  return isPreparedCandidateBatchInternal(val);
}

export function prepareScaledCandidateBatch(params: {
  readonly sourceTracks: readonly { readonly notes: readonly MidiNote[] }[];
  readonly scale: number;
  readonly placementStartTick: number;
  readonly effectiveMaxEndTick: number;
  readonly maxInsertableNotes: number;
  readonly existingIds: ReadonlySet<string>;
  readonly mintFallbackId: (
    preferredId: string | undefined,
    existingIds: ReadonlySet<string>,
    batchIds: ReadonlySet<string>,
  ) => string;
}): PreparedCandidateBatch {
  const placedBatchCanon = new Set<string>();
  const placedNotes: MidiNote[] = [];
  if (!params || typeof params !== 'object' || isDomainProxy(params)) {
    return new PreparedCandidateBatch(PRIVATE_BATCH_SENTINEL, placedNotes, placedBatchCanon);
  }
  const rawCap = params.maxInsertableNotes;
  const cap =
    typeof rawCap === 'number' && Number.isFinite(rawCap)
      ? Math.max(0, Math.min(MAX_NOTES_PER_TRACK, Math.trunc(rawCap)))
      : 0;
  if (cap <= 0 || !safeIsArray(params.sourceTracks)) {
    return new PreparedCandidateBatch(PRIVATE_BATCH_SENTINEL, placedNotes, placedBatchCanon);
  }
  const safeExistingIds: ReadonlySet<string> =
    params.existingIds &&
    typeof params.existingIds === 'object' &&
    !isDomainProxy(params.existingIds) &&
    typeof params.existingIds.has === 'function'
      ? (unwrapCanonicalIdSetInner(params.existingIds) ?? params.existingIds)
      : EMPTY_CANONICAL_ID_SET;
  const readonlyBatchIds: ReadonlySet<string> = createOwnedCanonicalIdSet(placedBatchCanon);

  outerTracks: for (let ti = 0; ti < params.sourceTracks.length; ti += 1) {
    const srcTrk = params.sourceTracks[ti];
    if (!srcTrk || typeof srcTrk !== 'object' || isDomainProxy(srcTrk)) continue;
    const trkNotes = srcTrk.notes;
    if (!safeIsArray(trkNotes)) continue;
    for (let ni = 0; ni < trkNotes.length; ni += 1) {
      if (placedNotes.length >= cap) {
        break outerTracks;
      }
      let rawItem: unknown;
      try {
        rawItem = trkNotes[ni];
      } catch {
        continue;
      }
      const n = isGenuineMidiNote(rawItem) ? rawItem : normalizeTrackInputNote(rawItem);
      if (n === null) {
        continue;
      }
      const placed = n.placeInScaledWindow(
        params.scale,
        params.placementStartTick,
        params.effectiveMaxEndTick,
      );
      if (placed === null) {
        continue;
      }
      if (n.id && !safeExistingIds.has(n.id) && !placedBatchCanon.has(n.id)) {
        placedBatchCanon.add(n.id);
        placedNotes.push(placed);
        continue;
      }
      let candidateFallback = '';
      try {
        const rawFallback = params.mintFallbackId(n.id, safeExistingIds, readonlyBatchIds);
        if (typeof rawFallback === 'string') {
          candidateFallback = fastCanonicalizeNoteOrTrackId(rawFallback);
        }
      } catch {
        candidateFallback = '';
      }
      while (
        !candidateFallback ||
        safeExistingIds.has(candidateFallback) ||
        placedBatchCanon.has(candidateFallback)
      ) {
        candidateFallback = fastCanonicalizeNoteOrTrackId(createNoteId());
      }
      let rebound = placed.placeInScaledWindow(
        1,
        0,
        Number.POSITIVE_INFINITY,
        candidateFallback,
      )!;
      while (
        !rebound.id ||
        safeExistingIds.has(rebound.id) ||
        placedBatchCanon.has(rebound.id)
      ) {
        const freshId = fastCanonicalizeNoteOrTrackId(createNoteId());
        if (!freshId) continue;
        rebound = placed.placeInScaledWindow(1, 0, Number.POSITIVE_INFINITY, freshId)!;
      }
      placedBatchCanon.add(rebound.id);
      placedNotes.push(rebound);
    }
  }
  if (placedBatchCanon.size !== placedNotes.length) {
    placedBatchCanon.clear();
    for (let i = 0; i < placedNotes.length; i += 1) {
      placedBatchCanon.add(placedNotes[i].id);
    }
  }
  return new PreparedCandidateBatch(PRIVATE_BATCH_SENTINEL, placedNotes, placedBatchCanon);
}

export function normalizeTrackIdIterable(rawIds: unknown, maxCount = 32_768): Set<string> {
  const unwrapped = unwrapCanonicalIdSetInner(rawIds);
  if (unwrapped !== null && unwrapped.size <= maxCount) {
    return new Set(unwrapped);
  }
  const out = new Set<string>();
  if (!rawIds || typeof rawIds !== 'object' || isDomainProxy(rawIds)) {
    return out;
  }
  if (safeIsArray(rawIds)) {
    let len = 0;
    try {
      const rawLen = rawIds.length;
      len =
        typeof rawLen === 'number' && Number.isFinite(rawLen)
          ? Math.min(maxCount, Math.max(0, Math.trunc(rawLen)))
          : 0;
    } catch {
      return out;
    }
    for (let i = 0; i < len; i += 1) {
      let elem: unknown;
      try {
        elem = rawIds[i];
      } catch {
        continue;
      }
      if (typeof elem === 'string') {
        const clean = fastCanonicalizeNoteOrTrackId(elem);
        if (clean) out.add(clean);
      }
    }
    return out;
  }
  try {
    let count = 0;
    for (const elem of rawIds as Iterable<unknown>) {
      if (count >= maxCount) break;
      count += 1;
      if (typeof elem === 'string') {
        const clean = fastCanonicalizeNoteOrTrackId(elem);
        if (clean) out.add(clean);
      }
    }
  } catch {
    // Ignore revoked/throwing custom iterables
  }
  return out;
}

function readVisitedNoteForQuery(
  raw: unknown,
): { readonly note: MidiNote; readonly startTick: number; readonly endTick: number } | null {
  if (isGenuineMidiNote(raw)) {
    return { note: raw, startTick: raw.startTick, endTick: raw.endTick };
  }
  const normalized = normalizeTrackInputNote(raw);
  if (normalized !== null) {
    return { note: normalized, startTick: normalized.startTick, endTick: normalized.endTick };
  }
  return null;
}

/**
 * Builds an implicit balanced binary interval tree's `subtreeMaxEndTick` array in O(N) time
 * with zero node object allocations. Every index `mid = (L + R) >>> 1` stores the maximum
 * `endTick` across `notes[L .. R-1]`.
 */
export function buildImplicitIntervalTreeMaxEnd(notes: readonly MidiNote[]): number[] {
  if (!safeIsArray(notes)) return [];
  let len = 0;
  try {
    const rawLen = notes.length;
    len = typeof rawLen === 'number' && Number.isFinite(rawLen) ? Math.max(0, Math.trunc(rawLen)) : 0;
  } catch {
    return [];
  }
  if (len === 0) return [];
  const maxEnds = new Array<number>(len);

  const build = (left: number, right: number): number => {
    if (left >= right) return 0;
    const mid = (left + right) >>> 1;
    const leftMax = build(left, mid);
    const rightMax = build(mid + 1, right);
    let rawMid: unknown;
    try {
      rawMid = notes[mid];
    } catch {
      rawMid = undefined;
    }
    let maxVal = 0;
    if (isGenuineMidiNote(rawMid)) {
      maxVal = rawMid.endTick;
    } else {
      const visited = readVisitedNoteForQuery(rawMid);
      if (visited !== null) maxVal = visited.endTick;
    }
    if (leftMax > maxVal) maxVal = leftMax;
    if (rightMax > maxVal) maxVal = rightMax;
    maxEnds[mid] = maxVal;
    return maxVal;
  };

  build(0, len);
  return maxEnds;
}

/**
 * Queries all notes overlapping `[startTick, endTick)` (`note.startTick < endTick && note.endTick > startTick`)
 * in provable O(log N + K log N) worst-case time (even in the presence of full-song drone notes)
 * and returns them in sorted `(startTick, pitch)` order.
 */
export function queryOverlappingNotesIndexed(
  notes: readonly MidiNote[],
  subtreeMaxEndTick: readonly number[],
  startTick: number,
  endTick: number,
): MidiNote[] {
  if (
    !safeIsArray(notes) ||
    !Number.isFinite(startTick) ||
    !Number.isFinite(endTick) ||
    !(endTick > startTick)
  ) {
    return [];
  }
  let len = 0;
  try {
    const rawLen = notes.length;
    len = typeof rawLen === 'number' && Number.isFinite(rawLen) ? Math.max(0, Math.trunc(rawLen)) : 0;
  } catch {
    return [];
  }
  if (len === 0) {
    return [];
  }
  let indexMatchesLen = false;
  if (safeIsArray(subtreeMaxEndTick)) {
    try {
      indexMatchesLen = subtreeMaxEndTick.length === len;
    } catch {
      indexMatchesLen = false;
    }
  }
  if (!indexMatchesLen) {
    // Defensive fallback if an external caller constructed a snapshot without index
    const fallbackOut: MidiNote[] = [];
    for (let i = 0; i < len; i += 1) {
      let elem: unknown;
      try {
        elem = notes[i];
      } catch {
        continue;
      }
      if (isGenuineMidiNote(elem)) {
        if (elem.startTick < endTick && elem.endTick > startTick) {
          fallbackOut.push(elem);
        }
        continue;
      }
      const v = readVisitedNoteForQuery(elem);
      if (v !== null && v.startTick < endTick && v.endTick > startTick) {
        fallbackOut.push(v.note);
      }
    }
    fallbackOut.sort((a, b) => a.startTick - b.startTick || a.pitch - b.pitch);
    return fallbackOut;
  }

  const result: MidiNote[] = [];

  const visit = (left: number, right: number): void => {
    if (left >= right) return;
    const mid = (left + right) >>> 1;
    let subMax: unknown;
    try {
      subMax = subtreeMaxEndTick[mid];
    } catch {
      subMax = undefined;
    }
    // Prune entire subtree [left, right) in O(1) if no note inside extends past startTick
    if (typeof subMax === 'number' && Number.isFinite(subMax) && subMax <= startTick) {
      return;
    }

    visit(left, mid);

    let rawNote: unknown;
    try {
      rawNote = notes[mid];
    } catch {
      rawNote = undefined;
    }
    if (isGenuineMidiNote(rawNote)) {
      if (rawNote.startTick < endTick && rawNote.endTick > startTick) {
        result.push(rawNote);
      }
      if (rawNote.startTick < endTick) {
        visit(mid + 1, right);
      }
      return;
    }
    const visited = readVisitedNoteForQuery(rawNote);
    if (visited === null) {
      visit(mid + 1, right);
      return;
    }
    if (visited.startTick < endTick && visited.endTick > startTick) {
      result.push(visited.note);
    }

    // Because `notes` is sorted ascending by `startTick`, right subtree [mid + 1, right)
    // can only contain notes with `startTick >= note.startTick`.
    if (visited.startTick < endTick) {
      visit(mid + 1, right);
    }
  };

  visit(0, len);
  return result;
}

export function queryOverlappingNotesInSnapshot(
  snapshot: Pick<DawTrackSnapshot, 'notes' | 'subtreeMaxEndTick'>,
  startTick: number,
  endTick: number,
): MidiNote[] {
  if (!snapshot || typeof snapshot !== 'object' || isDomainProxy(snapshot)) {
    return [];
  }
  const notesRes = readOwnDataField(snapshot, 'notes');
  const treeRes = readOwnDataField(snapshot, 'subtreeMaxEndTick');
  if (!notesRes.ok || !treeRes.ok) {
    return [];
  }
  return queryOverlappingNotesIndexed(
    notesRes.val as readonly MidiNote[],
    treeRes.val as readonly number[],
    startTick,
    endTick,
  );
}

const VALID_TRACK_INSTRUMENT_SET: ReadonlySet<string> = new Set<TrackInstrument>([
  'PIANO',
  'SAX',
  'BASS',
  'VIOLIN',
  'SYNTH',
  'GUITAR',
  'DRUMS',
]);

export function isValidTrackInstrument(val: unknown): val is TrackInstrument {
  return typeof val === 'string' && VALID_TRACK_INSTRUMENT_SET.has(val);
}

function truncateSafeUtf16(str: string, maxUnits: number): string {
  return truncateWellFormedUtf16(str, maxUnits);
}

function sanitizeTrackName(raw: unknown, fallback = 'Track'): string {
  if (typeof raw !== 'string') return fallback;
  const trimmed = truncateSafeUtf16(raw.trim(), 256).trim();
  return trimmed || fallback;
}

export interface FrozenPlainSyncNote {
  readonly id: string;
  readonly pitch: number;
  readonly startTick: number;
  readonly durationTicks: number;
  readonly velocity: number;
  readonly channel: number;
}

export interface FrozenPlainSyncTrack {
  readonly id: string;
  readonly name: string;
  readonly instrument: TrackInstrument;
  readonly colorHex: string;
  readonly isMuted: boolean;
  readonly isSoloed: boolean;
  readonly notes: readonly FrozenPlainSyncNote[];
}

let isGenuineDawTrackInternal!: (val: unknown) => val is DawTrack;
let applyValidatedUniqueSyncNotesInternal!: (track: DawTrack, notes: readonly MidiNote[]) => void;

export function isGenuineDawTrack(val: unknown): val is DawTrack {
  return isGenuineDawTrackInternal(val);
}

/**
 * Applies pre-validated, canonically-unique genuine `MidiNote` instances to a genuine `DawTrack`
 * without constructing a redundant 32k-entry deduplication `Set` (`[R25-HIGH-02]`, `oop-extensible-craft` §4.1).
 */
export function applyValidatedUniqueSyncNotes(
  track: DawTrack,
  notes: readonly MidiNote[],
): void {
  applyValidatedUniqueSyncNotesInternal(track, notes);
}

export class DawTrack {
  public readonly id: string;
  readonly #dawTrackBrand: boolean;
  private _name: string;
  private _instrument: TrackInstrument;
  private _colorHex: string;
  private _isMuted: boolean;
  private _isSoloed: boolean;
  private _notes: MidiNote[];
  private _subtreeMaxEndTick: number[] = [];
  private _cachedSnapshot: DawTrackSnapshot | null = null;
  private _cachedFrozenPlainNotes: readonly FrozenPlainSyncNote[] | null = null;
  private _cachedFrozenPlainSyncTrack: {
    readonly track: FrozenPlainSyncTrack;
    readonly maxEndTick: number;
  } | null = null;

  static {
    isGenuineDawTrackInternal = (val: unknown): val is DawTrack => {
      try {
        return Boolean(
          val &&
            typeof val === 'object' &&
            #dawTrackBrand in val &&
            val.#dawTrackBrand === true &&
            !isDomainProxy(val) &&
            Object.getPrototypeOf(val) === DawTrack.prototype,
        );
      } catch {
        return false;
      }
    };

    applyValidatedUniqueSyncNotesInternal = (
      track: DawTrack,
      notes: readonly MidiNote[],
    ): void => {
      if (!isGenuineDawTrackInternal(track)) return;
      if (!safeIsArray(notes) || isDomainProxy(notes)) {
        track.setAllNotes(notes);
        return;
      }
      const cap = Math.min(notes.length, MAX_NOTES_PER_TRACK);
      const copied = new Array<MidiNote>(cap);
      for (let i = 0; i < cap; i += 1) {
        const elem = notes[i];
        if (!isGenuineMidiNote(elem)) {
          track.setAllNotes(notes);
          return;
        }
        copied[i] = elem;
      }
      track._notes = copied;
      track._rebuildIndex(true);
    };
  }

  constructor(params: {
    id?: string;
    name: string;
    instrument?: TrackInstrument;
    colorHex?: string;
    isMuted?: boolean;
    isSoloed?: boolean;
    notes?: readonly MidiNote[];
  }) {
    this.#dawTrackBrand = new.target === DawTrack;
    const hasValidParams = Boolean(
      params && typeof params === 'object' && !safeIsArray(params) && !isDomainProxy(params),
    );
    const idField = hasValidParams ? readOwnDataField(params, 'id') : { ok: true as const, val: undefined };
    const nameField = hasValidParams ? readOwnDataField(params, 'name') : { ok: true as const, val: 'Track' };
    const instField = hasValidParams ? readOwnDataField(params, 'instrument') : { ok: true as const, val: undefined };
    const colorField = hasValidParams ? readOwnDataField(params, 'colorHex') : { ok: true as const, val: undefined };
    const mutedField = hasValidParams ? readOwnDataField(params, 'isMuted') : { ok: true as const, val: false };
    const soloedField = hasValidParams ? readOwnDataField(params, 'isSoloed') : { ok: true as const, val: false };
    const notesField = hasValidParams ? readOwnDataField(params, 'notes') : { ok: true as const, val: undefined };

    const rawIdVal = idField.ok ? idField.val : undefined;
    const rawId = typeof rawIdVal === 'string' && rawIdVal.trim() ? rawIdVal.trim() : '';
    const cleanId = rawId ? truncateSafeUtf16(rawId, 128).trim() : '';
    this.id = cleanId || createTrackId();
    this._name = sanitizeTrackName(nameField.ok ? nameField.val : 'Track', 'Track');
    const rawInstVal = instField.ok ? instField.val : undefined;
    this._instrument =
      typeof rawInstVal === 'string' && VALID_TRACK_INSTRUMENT_SET.has(rawInstVal)
        ? (rawInstVal as TrackInstrument)
        : 'PIANO';
    const rawColorVal = colorField.ok ? colorField.val : undefined;
    const cleanColor =
      typeof rawColorVal === 'string' && rawColorVal.trim()
        ? truncateSafeUtf16(rawColorVal.trim(), 32).trim()
        : '';
    this._colorHex = cleanColor || TRACK_SWATCHES[0];
    this._isMuted = Boolean(mutedField.ok ? mutedField.val : false);
    this._isSoloed = Boolean(soloedField.ok ? soloedField.val : false);
    this._notes = normalizeTrackNoteArray(notesField.ok ? notesField.val : undefined);
    this._rebuildIndex();
  }

  public get name(): string {
    return this._name;
  }

  public setName(name: string): void {
    if (typeof name !== 'string') return;
    const next = sanitizeTrackName(name, '');
    if (next && next !== this._name) {
      this._name = next;
      this._cachedSnapshot = null;
      this._cachedFrozenPlainSyncTrack = null;
    }
  }

  public get instrument(): TrackInstrument {
    return this._instrument;
  }

  public setInstrument(instrument: TrackInstrument): void {
    if (!isValidTrackInstrument(instrument)) return;
    if (instrument !== this._instrument) {
      this._instrument = instrument;
      this._cachedSnapshot = null;
      this._cachedFrozenPlainSyncTrack = null;
    }
  }

  public get colorHex(): string {
    return this._colorHex;
  }

  public setColorHex(colorHex: string): void {
    if (typeof colorHex !== 'string') return;
    const next = truncateSafeUtf16(colorHex.trim(), 32).trim();
    if (next && next !== this._colorHex) {
      this._colorHex = next;
      this._cachedSnapshot = null;
      this._cachedFrozenPlainSyncTrack = null;
    }
  }

  public get isMuted(): boolean {
    return this._isMuted;
  }

  public setMuted(muted: boolean): void {
    if (typeof muted !== 'boolean') return;
    if (muted !== this._isMuted) {
      this._isMuted = muted;
      this._cachedSnapshot = null;
      this._cachedFrozenPlainSyncTrack = null;
    }
  }

  public get isSoloed(): boolean {
    return this._isSoloed;
  }

  public setSoloed(soloed: boolean): void {
    if (typeof soloed !== 'boolean') return;
    if (soloed !== this._isSoloed) {
      this._isSoloed = soloed;
      this._cachedSnapshot = null;
      this._cachedFrozenPlainSyncTrack = null;
    }
  }

  /** Matches Java COMPASS `Track.isMonophonic()`. */
  public isMonophonic(): boolean {
    return (
      this._instrument === 'SAX' ||
      this._instrument === 'VIOLIN' ||
      this._instrument === 'BASS'
    );
  }

  /** MORTM 4.5 SFT-Gen maps monophonic tracks to SAX and polyphonic tracks to PIANO. */
  public toMortmInstrument(): 'PIANO' | 'SAX' {
    return this.isMonophonic() ? 'SAX' : 'PIANO';
  }

  /** Returns a defensive shallow copy (`oop-extensible-craft` encapsulation invariant). */
  public getNotes(): readonly MidiNote[] {
    return [...this._notes];
  }

  /**
   * Visits each `MidiNote` in sorted order without allocating or freezing a 32k-element snapshot array
   * (`oop-extensible-craft` §3.3 Information Hiding & §4.2 Space Complexity).
   */
  public forEachNote(visitor: (note: MidiNote, index: number) => void): void {
    const arr = this._notes;
    const len = arr.length;
    for (let i = 0; i < len; i += 1) {
      visitor(arr[i], i);
    }
  }

  public get noteCount(): number {
    return this._notes.length;
  }

  /**
   * Returns the immutable `MidiNote` at `index` in O(1) without allocating a snapshot array
   * (`oop-extensible-craft` §3.2 Value Object Immutability & §3.3 Information Hiding).
   */
  public getNoteAt(index: number): MidiNote | undefined {
    return this._notes[index];
  }

  public getMaxEndTick(): number {
    if (this._notes.length === 0) return 0;
    const rootMid = this._notes.length >>> 1;
    return this._subtreeMaxEndTick[rootMid] ?? 0;
  }

  /**
   * Binary-search range query returning notes with `startTick` in `[startTick, endTick)` in O(log N + K).
   */
  public queryNotesStartingIn(startTick: number, endTick: number): MidiNote[] {
    if (
      !Number.isFinite(startTick) ||
      !Number.isFinite(endTick) ||
      !(endTick > startTick) ||
      this._notes.length === 0
    ) {
      return [];
    }
    let lo = 0;
    let hi = this._notes.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this._notes[mid].startTick < startTick) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const result: MidiNote[] = [];
    for (let i = lo; i < this._notes.length; i += 1) {
      const n = this._notes[i];
      if (n.startTick >= endTick) {
        break;
      }
      result.push(n);
    }
    return result;
  }

  /**
   * Implicit balanced interval-tree query returning all notes overlapping `[startTick, endTick)`
   * (`note.startTick < endTick && note.endTick > startTick`), including sustaining notes that
   * started prior to `startTick`.
   */
  public queryNotesOverlapping(startTick: number, endTick: number): MidiNote[] {
    return queryOverlappingNotesIndexed(
      this._notes,
      this._subtreeMaxEndTick,
      startTick,
      endTick,
    );
  }

  public addNotes(notesToAdd: readonly MidiNote[]): MidiNote[] {
    const room = Math.max(0, MAX_NOTES_PER_TRACK - this._notes.length);
    if (room === 0) return [];
    const normalizedBatch = normalizeTrackNoteArray(notesToAdd, MAX_NOTES_PER_TRACK);
    if (normalizedBatch.length === 0) return [];
    const existingIds = new Set<string>();
    for (let i = 0; i < this._notes.length; i += 1) {
      existingIds.add(this._notes[i].id);
    }
    const inserted: MidiNote[] = [];
    for (let i = 0; i < normalizedBatch.length && inserted.length < room; i += 1) {
      const cand = normalizedBatch[i];
      if (existingIds.has(cand.id)) {
        continue;
      }
      existingIds.add(cand.id);
      this._notes.push(cand);
      inserted.push(cand);
    }
    if (inserted.length > 0) {
      this._rebuildIndex();
    }
    return inserted;
  }

  public removeNotesByIds(ids: ReadonlySet<string>): MidiNote[] {
    if (!ids || typeof ids !== 'object' || isDomainProxy(ids) || this._notes.length === 0) {
      return [];
    }
    try {
      const fastIds = unwrapCanonicalIdSetInner(ids) ?? ids;
      if (typeof fastIds.has !== 'function' || fastIds.size === 0) return [];
      const removed: MidiNote[] = [];
      const kept: MidiNote[] = [];
      for (const note of this._notes) {
        if (fastIds.has(note.id)) {
          removed.push(note);
        } else {
          kept.push(note);
        }
      }
      if (removed.length > 0) {
        this._notes = kept;
        this._rebuildIndex(false);
      }
      return removed;
    } catch {
      return [];
    }
  }

  public updateNotes(updatedById: ReadonlyMap<string, MidiNote>): void {
    if (
      !updatedById ||
      typeof updatedById !== 'object' ||
      isDomainProxy(updatedById) ||
      this._notes.length === 0
    ) {
      return;
    }
    try {
      if (typeof updatedById.has !== 'function' || typeof updatedById.get !== 'function' || updatedById.size === 0) {
        return;
      }
      let mutated = false;
      this._notes = this._notes.map((n) => {
        if (!updatedById.has(n.id)) return n;
        const rawReplacement = normalizeTrackInputNote(updatedById.get(n.id));
        if (rawReplacement === null) return n;
        const replacement =
          rawReplacement.id === n.id
            ? rawReplacement
            : new MidiNote({
                id: n.id,
                pitch: rawReplacement.pitch,
                startTick: rawReplacement.startTick,
                durationTicks: rawReplacement.durationTicks,
                velocity: rawReplacement.velocity,
                channel: rawReplacement.channel,
              });
        if (replacement !== n) {
          mutated = true;
          return replacement;
        }
        return n;
      });
      if (mutated) {
        this._rebuildIndex();
      }
    } catch {
      // Ignore hostile/revoked Map input without corrupting track state
    }
  }

  public setAllNotes(notes: readonly MidiNote[]): void {
    this._notes = normalizeTrackNoteArray(notes);
    this._rebuildIndex();
  }

  /**
   * Single-pass atomic range replacement (`[R23-MED-03]`, `oop-extensible-craft` §4.1):
   * Removes range/extra notes, clips sustaining overhang notes starting prior to `rangeStartTick`,
   * appends pre-normalized `insertedNotes` up to `MAX_NOTES_PER_TRACK`, and rebuilds the interval
   * tree index exactly once instead of 2–3 separate passes.
   */
  public replaceRangeNotesAtomic(
    rangeStartTick: number,
    rangeEndTick: number,
    insertedNotes: readonly MidiNote[] | PreparedCandidateBatch,
    extraIdsToRemove: ReadonlySet<string>,
  ): {
    readonly removedNotes: MidiNote[];
    readonly truncatedBeforeById: Map<string, MidiNote>;
    readonly actuallyInsertedIds: ReadonlyCanonicalIdSet;
  } {
    try {
      const safeStart =
        typeof rangeStartTick === 'number' && Number.isFinite(rangeStartTick)
          ? Math.max(0, Math.round(rangeStartTick))
          : 0;
      const safeEnd =
        typeof rangeEndTick === 'number' && Number.isFinite(rangeEndTick)
          ? Math.max(safeStart, Math.round(rangeEndTick))
          : safeStart;
      const hasValidWindow =
        typeof rangeStartTick === 'number' &&
        typeof rangeEndTick === 'number' &&
        Number.isFinite(rangeStartTick) &&
        Number.isFinite(rangeEndTick) &&
        safeEnd > safeStart;

      const isBatch = isPreparedCandidateBatchInternal(insertedNotes);
      if (!isBatch && (!safeIsArray(insertedNotes) || isDomainProxy(insertedNotes))) {
        return {
          removedNotes: [],
          truncatedBeforeById: new Map(),
          actuallyInsertedIds: EMPTY_CANONICAL_ID_SET,
        };
      }
      if (
        extraIdsToRemove !== undefined &&
        (!extraIdsToRemove ||
          typeof extraIdsToRemove !== 'object' ||
          isDomainProxy(extraIdsToRemove) ||
          typeof (extraIdsToRemove as ReadonlySet<string>).has !== 'function')
      ) {
        return {
          removedNotes: [],
          truncatedBeforeById: new Map(),
          actuallyInsertedIds: EMPTY_CANONICAL_ID_SET,
        };
      }
      const unwrappedExtra = unwrapCanonicalIdSetInner(extraIdsToRemove);
      const fastRemoveSet: ReadonlySet<string> =
        unwrappedExtra !== null
          ? unwrappedExtra
          : extraIdsToRemove !== undefined
            ? normalizeTrackIdIterable(extraIdsToRemove, MAX_NOTES_PER_TRACK)
            : EMPTY_CANONICAL_ID_SET;

      const normalizedArrayNotes: readonly MidiNote[] = isBatch
        ? []
        : normalizeTrackNoteArray(insertedNotes);

      let removedNotes: MidiNote[] = [];
      const truncatedBeforeById = new Map<string, MidiNote>();
      let kept: MidiNote[] = [];

      if (
        isBatch &&
        this._notes.length > 0 &&
        hasValidWindow &&
        safeStart === 0 &&
        safeEnd >= this.getMaxEndTick()
      ) {
        removedNotes = this._notes.slice();
      } else {
        const hasExtraRemove = fastRemoveSet.size > 0;
        for (let i = 0; i < this._notes.length; i += 1) {
          const note = this._notes[i];
          if (hasExtraRemove && fastRemoveSet.has(note.id)) {
            removedNotes.push(note);
            continue;
          }
          if (hasValidWindow && note.startTick < safeEnd && note.endTick > safeStart) {
            if (note.startTick >= safeStart) {
              removedNotes.push(note);
              continue;
            }
            const clippedDur = Math.max(1, safeStart - note.startTick);
            const clippedNote = note.withTiming(note.startTick, clippedDur);
            truncatedBeforeById.set(note.id, note);
            kept.push(clippedNote);
            continue;
          }
          kept.push(note);
        }
      }

      const room = Math.max(0, MAX_NOTES_PER_TRACK - kept.length);
      let rawInsertedIds: Set<string>;
      if (isBatch) {
        const batchNotes = getPreparedBatchNotes(insertedNotes);
        if (room <= 0 || batchNotes.length === 0) {
          rawInsertedIds = new Set<string>();
        } else if (kept.length === 0 && batchNotes.length <= room) {
          kept = batchNotes.slice();
          rawInsertedIds = takePreparedBatchIdsOnce(insertedNotes);
        } else {
          rawInsertedIds = new Set<string>();
          const existingIds = kept.length > 0 ? new Set<string>() : null;
          if (existingIds !== null) {
            for (let i = 0; i < kept.length; i += 1) {
              existingIds.add(kept[i].id);
            }
          }
          for (let i = 0; i < batchNotes.length && rawInsertedIds.size < room; i += 1) {
            const cand = batchNotes[i];
            if (existingIds !== null && existingIds.has(cand.id)) {
              continue;
            }
            rawInsertedIds.add(cand.id);
            kept.push(cand);
          }
        }
      } else {
        rawInsertedIds = new Set<string>();
        if (room > 0 && normalizedArrayNotes.length > 0) {
          const existingIds = kept.length > 0 ? new Set<string>() : null;
          if (existingIds !== null) {
            for (let i = 0; i < kept.length; i += 1) {
              existingIds.add(kept[i].id);
            }
          }
          for (let i = 0; i < normalizedArrayNotes.length && rawInsertedIds.size < room; i += 1) {
            const cand = normalizedArrayNotes[i];
            if (existingIds !== null && existingIds.has(cand.id)) {
              continue;
            }
            const prevInsertedSize = rawInsertedIds.size;
            rawInsertedIds.add(cand.id);
            if (rawInsertedIds.size !== prevInsertedSize) {
              kept.push(cand);
            }
          }
        }
      }

      if (
        removedNotes.length > 0 ||
        truncatedBeforeById.size > 0 ||
        rawInsertedIds.size > 0
      ) {
        this._notes = kept;
        this._rebuildIndex(rawInsertedIds.size > 0 || truncatedBeforeById.size > 0);
      }

      return {
        removedNotes,
        truncatedBeforeById,
        actuallyInsertedIds:
          rawInsertedIds.size === 0
            ? EMPTY_CANONICAL_ID_SET
            : createOwnedCanonicalIdSet(rawInsertedIds),
      };
    } catch {
      return {
        removedNotes: [],
        truncatedBeforeById: new Map<string, MidiNote>(),
        actuallyInsertedIds: EMPTY_CANONICAL_ID_SET,
      };
    }
  }

  /**
   * Returns a cached, deeply frozen snapshot when track state has not changed since the last call,
   * eliminating O(N) array cloning during 40Hz transport playhead updates.
   */
  public toSnapshot(): DawTrackSnapshot {
    if (this._cachedSnapshot) {
      return this._cachedSnapshot;
    }
    const frozenNotes = Object.freeze([...this._notes]);
    const frozenMaxEnds = Object.freeze([...this._subtreeMaxEndTick]);
    this._cachedSnapshot = Object.freeze({
      id: this.id,
      name: this._name,
      instrument: this._instrument,
      colorHex: this._colorHex,
      isMuted: this._isMuted,
      isSoloed: this._isSoloed,
      isMonophonic: this.isMonophonic(),
      notes: frozenNotes,
      subtreeMaxEndTick: frozenMaxEnds,
    });
    return this._cachedSnapshot;
  }

  public toFrozenPlainSyncTrack(): {
    readonly track: FrozenPlainSyncTrack;
    readonly maxEndTick: number;
  } {
    if (this._cachedFrozenPlainSyncTrack) {
      return this._cachedFrozenPlainSyncTrack;
    }
    let plainNotes = this._cachedFrozenPlainNotes;
    if (!plainNotes) {
      const len = this._notes.length;
      const arr = new Array<FrozenPlainSyncNote>(len);
      for (let i = 0; i < len; i += 1) {
        const n = this._notes[i];
        arr[i] = Object.freeze({
          id: n.id,
          pitch: n.pitch,
          startTick: n.startTick,
          durationTicks: n.durationTicks,
          velocity: n.velocity,
          channel: n.channel,
        });
      }
      plainNotes = Object.freeze(arr);
      this._cachedFrozenPlainNotes = plainNotes;
    }
    const maxEndTick =
      this._notes.length > 0 ? this._subtreeMaxEndTick[this._notes.length >>> 1] : 0;
    const frozenTrack: FrozenPlainSyncTrack = Object.freeze({
      id: this.id,
      name: this._name,
      instrument: this._instrument,
      colorHex: this._colorHex,
      isMuted: this._isMuted,
      isSoloed: this._isSoloed,
      notes: plainNotes,
    });
    this._cachedFrozenPlainSyncTrack = Object.freeze({
      track: frozenTrack,
      maxEndTick,
    });
    return this._cachedFrozenPlainSyncTrack;
  }

  private _rebuildIndex(needsSort = true): void {
    if (needsSort && this._notes.length > 1) {
      let alreadySorted = true;
      for (let i = 1; i < this._notes.length; i += 1) {
        const prev = this._notes[i - 1];
        const curr = this._notes[i];
        if (
          prev.startTick > curr.startTick ||
          (prev.startTick === curr.startTick && prev.pitch > curr.pitch)
        ) {
          alreadySorted = false;
          break;
        }
      }
      if (!alreadySorted) {
        this._notes.sort((a, b) => a.startTick - b.startTick || a.pitch - b.pitch);
      }
    }
    this._subtreeMaxEndTick = buildInternalIntervalTreeMaxEnd(this._notes);
    this._cachedSnapshot = null;
    this._cachedFrozenPlainNotes = null;
    this._cachedFrozenPlainSyncTrack = null;
  }
}

function buildInternalIntervalTreeMaxEnd(notes: readonly MidiNote[]): number[] {
  const len = notes.length;
  if (len === 0) return [];
  const maxEnds = new Array<number>(len);
  const build = (left: number, right: number): number => {
    if (left >= right) return 0;
    const mid = (left + right) >>> 1;
    const leftMax = build(left, mid);
    const rightMax = build(mid + 1, right);
    const midNote = notes[mid];
    let maxVal = midNote.startTick + midNote.durationTicks;
    if (leftMax > maxVal) maxVal = leftMax;
    if (rightMax > maxVal) maxVal = rightMax;
    maxEnds[mid] = maxVal;
    return maxVal;
  };
  build(0, len);
  return maxEnds;
}
