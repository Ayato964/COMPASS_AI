/**
 * Immutable Value Object representing a single MIDI note event (`oop-extensible-craft`).
 */

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;
const BLACK_KEY_MODS = new Set<number>([1, 3, 6, 8, 10]);

const NODE_DOMAIN_IS_PROXY: ((v: unknown) => boolean) | undefined = (() => {
  try {
    const g = globalThis as {
      process?: {
        getBuiltinModule?: (
          id: string,
        ) => { types?: { isProxy?: (v: unknown) => boolean } } | undefined;
      };
    };
    const fn = g.process?.getBuiltinModule?.('node:util')?.types?.isProxy;
    return typeof fn === 'function' ? fn : undefined;
  } catch {
    return undefined;
  }
})();

export function isDomainProxy(val: unknown): boolean {
  try {
    return NODE_DOMAIN_IS_PROXY ? Boolean(NODE_DOMAIN_IS_PROXY(val)) : false;
  } catch {
    return true;
  }
}

export function safeIsArray(val: unknown): val is readonly unknown[] {
  try {
    return !isDomainProxy(val) && Array.isArray(val);
  } catch {
    return false;
  }
}

let isGenuineMidiNoteInternal!: (val: unknown) => val is MidiNote;

export function isGenuineMidiNote(val: unknown): val is MidiNote {
  return isGenuineMidiNoteInternal(val);
}

export const MAX_SAFE_MIDI_TICK = Math.floor(Number.MAX_SAFE_INTEGER / 2);
const MAX_SAFE_MIDI_TICK_BIG = BigInt(MAX_SAFE_MIDI_TICK);

const F64_SCRATCH_BUF = new ArrayBuffer(8);
const F64_SCRATCH_VIEW = new Float64Array(F64_SCRATCH_BUF);
const U64_SCRATCH_VIEW = new BigUint64Array(F64_SCRATCH_BUF);

function decodeFiniteFloat64Exact(v: number): { m: bigint; e: number } {
  if (v === 0) return { m: 0n, e: 0 };
  F64_SCRATCH_VIEW[0] = v;
  const bits = U64_SCRATCH_VIEW[0];
  const sign = (bits >> 63n) !== 0n ? -1n : 1n;
  const rawExp = Number((bits >> 52n) & 0x7ffn);
  const frac = bits & 0xfffffffffffffn;
  if (rawExp === 0) {
    return { m: sign * frac, e: -1074 };
  }
  return { m: sign * ((1n << 52n) | frac), e: rawExp - 1075 };
}

/**
 * Computes `Math.round(baseInt * effScale + effOffset)` clamped to `[minTick, MAX_SAFE_MIDI_TICK]`
 * with exact infinite-precision intermediate arithmetic over IEEE-754 binary64 inputs (`[R12-LOW-03]`, `[R13-LOW-02]`).
 */
function roundAffineExact(
  baseInt: number,
  effScale: number,
  effOffset: number,
  minTick: number,
): number {
  if (effOffset === Number.NEGATIVE_INFINITY) {
    return minTick;
  }
  if (effOffset === Number.POSITIVE_INFINITY) {
    return MAX_SAFE_MIDI_TICK;
  }
  if (effScale === 0 || baseInt === 0) {
    return clampSafeTick(effOffset, baseInt, minTick);
  }
  if (effScale === Number.POSITIVE_INFINITY) {
    return MAX_SAFE_MIDI_TICK;
  }
  if (Number.isSafeInteger(effScale) && Number.isSafeInteger(effOffset)) {
    if (
      baseInt >= 0 &&
      baseInt <= 0x7fffffff &&
      effScale >= 0 &&
      effScale <= 0x1fffff &&
      effOffset >= -MAX_SAFE_MIDI_TICK &&
      effOffset <= MAX_SAFE_MIDI_TICK
    ) {
      const exactNum = baseInt * effScale + effOffset;
      if (exactNum <= minTick) return minTick;
      if (exactNum >= MAX_SAFE_MIDI_TICK) return MAX_SAFE_MIDI_TICK;
      return exactNum;
    }
    const minBig = BigInt(minTick);
    const exact = BigInt(baseInt) * BigInt(effScale) + BigInt(effOffset);
    if (exact <= minBig) return minTick;
    if (exact >= MAX_SAFE_MIDI_TICK_BIG) return MAX_SAFE_MIDI_TICK;
    return Number(exact);
  }
  const minBig = BigInt(minTick);
  const { m: ms, e: es } = decodeFiniteFloat64Exact(effScale);
  const { m: mo, e: eo } = decodeFiniteFloat64Exact(effOffset);
  const mp = BigInt(baseInt) * ms;
  const eMin = Math.min(es, eo, 0);
  const num = (mp << BigInt(es - eMin)) + (mo << BigInt(eo - eMin));
  if (num <= 0n) {
    return minTick;
  }
  const k = -eMin;
  const rounded = k === 0 ? num : (num + (1n << BigInt(k - 1))) >> BigInt(k);
  if (rounded <= minBig) return minTick;
  if (rounded >= MAX_SAFE_MIDI_TICK_BIG) return MAX_SAFE_MIDI_TICK;
  return Number(rounded);
}

function finiteOr(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function clampSafeTick(value: number | undefined, fallback: number, minTick: number): number {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return Math.max(minTick, Math.min(MAX_SAFE_MIDI_TICK, Math.round(fallback)));
  }
  if (value === Number.POSITIVE_INFINITY) {
    return MAX_SAFE_MIDI_TICK;
  }
  if (value === Number.NEGATIVE_INFINITY) {
    return minTick;
  }
  return Math.max(minTick, Math.min(MAX_SAFE_MIDI_TICK, Math.round(value)));
}

/**
 * Truncates `raw` to at most `maxUnits` UTF-16 code units without splitting surrogate pairs
 * (`0xD800..0xDBFF` + `0xDC00..0xDFFF`) and replaces any lone surrogates 1-for-1 with `\uFFFD`
 * so `out.length <= maxUnits` and `out.isWellFormed() === true` (`[R9-LOW-03]`).
 */
export function truncateWellFormedUtf16(raw: string, maxUnits: number): string {
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

export function midiPitchToName(pitch: number): string {
  const safe = finiteOr(pitch, 60);
  const clamped = Math.max(0, Math.min(127, Math.round(safe)));
  const octave = Math.floor(clamped / 12) - 1;
  return `${NOTE_NAMES[clamped % 12]}${octave}`;
}

function mintRandomIdEntropy(byteCount = 4): string {
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

const _noteInstanceEntropy = mintRandomIdEntropy(4);
let _noteSeqCounter = 1;
export function createNoteId(): string {
  _noteSeqCounter += 1;
  return `n_${_noteInstanceEntropy}_${Date.now().toString(36)}_${_noteSeqCounter.toString(36)}`;
}

const TRUSTED_NOTE_SENTINEL = Symbol('MidiNote.trusted');

function canonicalizeNoteIdInput(rawId: unknown): string {
  if (typeof rawId !== 'string' || rawId.length === 0) {
    return '';
  }
  const len = rawId.length;
  const c0 = rawId.charCodeAt(0);
  const cLast = rawId.charCodeAt(len - 1);
  if (len <= 128 && c0 > 0x20 && c0 < 0x7f && cLast > 0x20 && cLast < 0x7f) {
    let hasSurrogate = false;
    for (let i = 0; i < len; i += 1) {
      const c = rawId.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdfff) {
        hasSurrogate = true;
        break;
      }
    }
    if (!hasSurrogate) {
      return rawId;
    }
  }
  const trimmed = rawId.trim();
  return trimmed ? truncateWellFormedUtf16(trimmed, 128).trim() : '';
}

function createTrustedMidiNote(
  id: string,
  pitch: number,
  startTick: number,
  durationTicks: number,
  velocity: number,
  channel: number,
): MidiNote {
  return new (MidiNote as unknown as new (
    sentinel: typeof TRUSTED_NOTE_SENTINEL,
    id: string,
    pitch: number,
    startTick: number,
    durationTicks: number,
    velocity: number,
    channel: number,
  ) => MidiNote)(TRUSTED_NOTE_SENTINEL, id, pitch, startTick, durationTicks, velocity, channel);
}

export function createValidatedSyncMidiNote(
  id: string,
  pitch: number,
  startTick: number,
  durationTicks: number,
  velocity: number,
  channel: number,
): MidiNote {
  if (
    typeof id === 'string' &&
    id.length > 0 &&
    id.length <= 128 &&
    Number.isInteger(pitch) &&
    pitch >= 0 &&
    pitch <= 127 &&
    Number.isSafeInteger(startTick) &&
    startTick >= 0 &&
    startTick <= MAX_SAFE_MIDI_TICK &&
    Number.isSafeInteger(durationTicks) &&
    durationTicks >= 1 &&
    durationTicks <= MAX_SAFE_MIDI_TICK &&
    Number.isInteger(velocity) &&
    velocity >= 1 &&
    velocity <= 127 &&
    Number.isInteger(channel) &&
    channel >= 0 &&
    channel <= 15
  ) {
    const cleanId = canonicalizeNoteIdInput(id);
    if (cleanId) {
      return createTrustedMidiNote(cleanId, pitch, startTick, durationTicks, velocity, channel);
    }
  }
  return new MidiNote({ id, pitch, startTick, durationTicks, velocity, channel });
}

export interface MidiNoteProps {
  readonly id?: string;
  readonly pitch: number;
  readonly startTick: number;
  readonly durationTicks: number;
  readonly velocity?: number;
  readonly channel?: number;
}

interface MutableMidiNoteFields {
  id: string;
  pitch: number;
  startTick: number;
  durationTicks: number;
  velocity: number;
  channel: number;
}

export class MidiNote {
  declare public readonly id: string;
  declare public readonly pitch: number;
  declare public readonly startTick: number;
  declare public readonly durationTicks: number;
  declare public readonly velocity: number;
  declare public readonly channel: number;
  readonly #genuineBrand: boolean;

  static {
    isGenuineMidiNoteInternal = (val: unknown): val is MidiNote => {
      try {
        return Boolean(
          val &&
            typeof val === 'object' &&
            #genuineBrand in val &&
            val.#genuineBrand === true &&
            !isDomainProxy(val) &&
            Object.getPrototypeOf(val) === MidiNote.prototype,
        );
      } catch {
        return false;
      }
    };
  }

  constructor(props: MidiNoteProps);
  constructor(
    props: MidiNoteProps | typeof TRUSTED_NOTE_SENTINEL,
    trustedId?: string,
    trustedPitch?: number,
    trustedStartTick?: number,
    trustedDurationTicks?: number,
    trustedVelocity?: number,
    trustedChannel?: number,
  ) {
    const self = this as unknown as MutableMidiNoteFields;
    const isDirectTarget = new.target === MidiNote;
    if (props === TRUSTED_NOTE_SENTINEL) {
      self.id = trustedId!;
      self.pitch = trustedPitch!;
      self.startTick = trustedStartTick!;
      self.durationTicks = trustedDurationTicks!;
      self.velocity = trustedVelocity!;
      self.channel = trustedChannel!;
      this.#genuineBrand = isDirectTarget;
      Object.freeze(this);
      return;
    }
    let rawId: unknown;
    let rawPitch: unknown;
    let rawStart: unknown;
    let rawDur: unknown;
    let rawVel: unknown;
    let rawCh: unknown;
    if (props && typeof props === 'object') {
      try {
        rawId = props.id;
        rawPitch = props.pitch;
        rawStart = props.startTick;
        rawDur = props.durationTicks;
        rawVel = props.velocity;
        rawCh = props.channel;
      } catch {
        // Leave partially read fields undefined so safe defaults apply below
      }
    }
    const cleanId = canonicalizeNoteIdInput(rawId);
    self.id = cleanId || createNoteId();
    self.pitch = Math.max(0, Math.min(127, Math.round(finiteOr(rawPitch as number | undefined, 60))));
    self.startTick = clampSafeTick(rawStart as number | undefined, 0, 0);
    self.durationTicks = clampSafeTick(rawDur as number | undefined, 120, 1);
    self.velocity = Math.max(
      1,
      Math.min(127, Math.round(finiteOr(rawVel as number | undefined, 100))),
    );
    self.channel = Math.max(
      0,
      Math.min(15, Math.round(finiteOr(rawCh as number | undefined, 0))),
    );
    this.#genuineBrand = isDirectTarget;
    Object.freeze(this);
  }

  public get endTick(): number {
    return this.startTick + this.durationTicks;
  }

  public isBlackKey(): boolean {
    return BLACK_KEY_MODS.has(this.pitch % 12);
  }

  public pitchLabel(): string {
    const octave = Math.floor(this.pitch / 12) - 1;
    return `${NOTE_NAMES[this.pitch % 12]}${octave}`;
  }

  public shiftTicks(deltaTicks: number): MidiNote {
    const nextStart =
      typeof deltaTicks === 'number' && !Number.isNaN(deltaTicks)
        ? roundAffineExact(this.startTick, 1, deltaTicks, 0)
        : this.startTick;
    return new MidiNote({
      id: this.id,
      pitch: this.pitch,
      startTick: nextStart,
      durationTicks: this.durationTicks,
      velocity: this.velocity,
      channel: this.channel,
    });
  }

  public scaleAndShift(scale: number, tickOffset: number, regenerateId = false): MidiNote {
    const effScale =
      typeof scale === 'number' && !Number.isNaN(scale) && scale >= 0 ? scale : 1;
    const effOffset =
      typeof tickOffset === 'number' && !Number.isNaN(tickOffset) ? tickOffset : 0;

    const nextStart = roundAffineExact(this.startTick, effScale, effOffset, 0);
    const nextDuration =
      effScale === 0 ? 1 : roundAffineExact(this.durationTicks, effScale, 0, 1);

    return new MidiNote({
      id: regenerateId ? createNoteId() : this.id,
      pitch: this.pitch,
      startTick: nextStart,
      durationTicks: nextDuration,
      velocity: this.velocity,
      channel: this.channel,
    });
  }

  public computeScaledWindowPlacement(
    scale: number,
    tickOffset: number,
    maxEndTick: number,
  ): { readonly startTick: number; readonly durationTicks: number } | null {
    const effMaxEnd =
      typeof maxEndTick === 'number' && !Number.isNaN(maxEndTick)
        ? maxEndTick === Number.POSITIVE_INFINITY
          ? Number.POSITIVE_INFINITY
          : Math.floor(maxEndTick)
        : Number.POSITIVE_INFINITY;
    if (effMaxEnd <= 0) {
      return null;
    }
    const effScale =
      typeof scale === 'number' && !Number.isNaN(scale) && scale >= 0 ? scale : 1;
    const effOffset =
      typeof tickOffset === 'number' && !Number.isNaN(tickOffset) ? tickOffset : 0;

    if (effScale === 1 && effOffset === 0) {
      if (this.startTick >= effMaxEnd) {
        return null;
      }
      const nextDur =
        this.endTick > effMaxEnd
          ? Math.max(1, Math.min(MAX_SAFE_MIDI_TICK, effMaxEnd - this.startTick))
          : this.durationTicks;
      return { startTick: this.startTick, durationTicks: nextDur };
    }

    const nextStart = roundAffineExact(this.startTick, effScale, effOffset, 0);
    if (nextStart >= effMaxEnd) {
      return null;
    }
    const rawDuration =
      effScale === 1
        ? this.durationTicks
        : effScale === 0
          ? 1
          : roundAffineExact(this.durationTicks, effScale, 0, 1);
    const nextDur =
      nextStart + rawDuration > effMaxEnd
        ? Math.max(1, Math.min(MAX_SAFE_MIDI_TICK, effMaxEnd - nextStart))
        : rawDuration;
    return { startTick: nextStart, durationTicks: nextDur };
  }

  public placeInScaledWindow(
    scale: number,
    tickOffset: number,
    maxEndTick: number,
    overrideId?: string,
  ): MidiNote | null {
    const effMaxEnd =
      typeof maxEndTick === 'number' && !Number.isNaN(maxEndTick)
        ? maxEndTick === Number.POSITIVE_INFINITY
          ? Number.POSITIVE_INFINITY
          : Math.floor(maxEndTick)
        : Number.POSITIVE_INFINITY;
    if (effMaxEnd <= 0) {
      return null;
    }
    const effScale =
      typeof scale === 'number' && !Number.isNaN(scale) && scale >= 0 ? scale : 1;
    const effOffset =
      typeof tickOffset === 'number' && !Number.isNaN(tickOffset) ? tickOffset : 0;

    let nextStart: number;
    let nextDur: number;
    if (effScale === 1 && effOffset === 0) {
      if (this.startTick >= effMaxEnd) {
        return null;
      }
      nextStart = this.startTick;
      nextDur =
        this.endTick > effMaxEnd
          ? Math.max(1, Math.min(MAX_SAFE_MIDI_TICK, effMaxEnd - this.startTick))
          : this.durationTicks;
    } else {
      nextStart = roundAffineExact(this.startTick, effScale, effOffset, 0);
      if (nextStart >= effMaxEnd) {
        return null;
      }
      const rawDuration =
        effScale === 1
          ? this.durationTicks
          : effScale === 0
            ? 1
            : roundAffineExact(this.durationTicks, effScale, 0, 1);
      nextDur =
        nextStart + rawDuration > effMaxEnd
          ? Math.max(1, Math.min(MAX_SAFE_MIDI_TICK, effMaxEnd - nextStart))
          : rawDuration;
    }

    if (!isGenuineMidiNoteInternal(this as unknown)) {
      return new MidiNote({
        id: overrideId !== undefined ? overrideId : this.id,
        pitch: this.pitch,
        startTick: nextStart,
        durationTicks: nextDur,
        velocity: this.velocity,
        channel: this.channel,
      });
    }

    const resolvedId =
      overrideId === undefined || overrideId === this.id
        ? this.id
        : canonicalizeNoteIdInput(overrideId) || createNoteId();

    if (
      resolvedId === this.id &&
      nextStart === this.startTick &&
      nextDur === this.durationTicks
    ) {
      return this;
    }
    return createTrustedMidiNote(
      resolvedId,
      this.pitch,
      nextStart,
      nextDur,
      this.velocity,
      this.channel,
    );
  }

  public withPitchAndStart(pitch: number, startTick: number): MidiNote {
    return new MidiNote({
      id: this.id,
      pitch: finiteOr(pitch, this.pitch),
      startTick: clampSafeTick(startTick, this.startTick, 0),
      durationTicks: this.durationTicks,
      velocity: this.velocity,
      channel: this.channel,
    });
  }

  public withDuration(durationTicks: number): MidiNote {
    return new MidiNote({
      id: this.id,
      pitch: this.pitch,
      startTick: this.startTick,
      durationTicks: clampSafeTick(durationTicks, this.durationTicks, 1),
      velocity: this.velocity,
      channel: this.channel,
    });
  }

  public withTiming(startTick: number, durationTicks: number): MidiNote {
    return new MidiNote({
      id: this.id,
      pitch: this.pitch,
      startTick: clampSafeTick(startTick, this.startTick, 0),
      durationTicks: clampSafeTick(durationTicks, this.durationTicks, 1),
      velocity: this.velocity,
      channel: this.channel,
    });
  }

  public withVelocity(velocity: number): MidiNote {
    return new MidiNote({
      id: this.id,
      pitch: this.pitch,
      startTick: this.startTick,
      durationTicks: this.durationTicks,
      velocity: finiteOr(velocity, this.velocity),
      channel: this.channel,
    });
  }

  public withChannel(channel: number): MidiNote {
    return new MidiNote({
      id: this.id,
      pitch: this.pitch,
      startTick: this.startTick,
      durationTicks: this.durationTicks,
      velocity: this.velocity,
      channel: finiteOr(channel, this.channel),
    });
  }

  public cloneWithNewId(tickOffset = 0): MidiNote {
    const nextStart =
      typeof tickOffset === 'number' && !Number.isNaN(tickOffset)
        ? roundAffineExact(this.startTick, 1, tickOffset, 0)
        : this.startTick;
    return new MidiNote({
      id: createNoteId(),
      pitch: this.pitch,
      startTick: nextStart,
      durationTicks: this.durationTicks,
      velocity: this.velocity,
      channel: this.channel,
    });
  }
}
