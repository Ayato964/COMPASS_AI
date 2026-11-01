/**
 * Immutable Value Object representing a 1-indexed musical measure range (`[startMeasure, endMeasure]`).
 * Encapsulates COMPASS's 8-measure Past/Future context window math and 1..8 measure SFT target clamping.
 */

import { MAX_SAFE_MIDI_TICK } from './MidiNote';

function finiteOr(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export interface GenerationMeterEvaluation {
  readonly supported: boolean;
  readonly numerator: number;
  readonly denominator: number;
  readonly label: string;
  readonly stepsPerBar: number;
  readonly reason: string | null;
}

/**
 * Validates whether a musical time signature is supported by MORTM 4.5's 24-steps/quarter-note
 * bar-local step vocabulary (`s_0..s_95`, requiring `12 <= stepsPerBar <= 96` and power-of-2
 * denominator `<= 32`) (`[R4-MED-02]`).
 */
export function evaluateGenerationMeter(
  numerator: number,
  denominator: number,
): GenerationMeterEvaluation {
  const numValid =
    typeof numerator === 'number' &&
    Number.isInteger(numerator) &&
    numerator > 0 &&
    numerator <= 64;
  const denValid =
    typeof denominator === 'number' &&
    Number.isInteger(denominator) &&
    denominator > 0 &&
    denominator <= 32 &&
    (denominator & (denominator - 1)) === 0;
  if (!numValid || !denValid) {
    const numLabel =
      typeof numerator === 'number' && Number.isFinite(numerator) && Math.abs(numerator) <= 9999
        ? String(Math.trunc(numerator) || 0)
        : '?';
    const denLabel =
      typeof denominator === 'number' &&
      Number.isFinite(denominator) &&
      Math.abs(denominator) <= 9999
        ? String(Math.trunc(denominator) || 0)
        : '?';
    const label = `${numLabel}/${denLabel}`;
    const safeNumOut =
      typeof numerator === 'number' && Number.isFinite(numerator)
        ? Math.trunc(numerator) || 0
        : 0;
    const safeDenOut =
      typeof denominator === 'number' && Number.isFinite(denominator)
        ? Math.trunc(denominator) || 0
        : 0;
    return {
      supported: false,
      numerator: safeNumOut,
      denominator: safeDenOut,
      label,
      stepsPerBar: 0,
      reason: `現在の拍子 (${label}) は不正またはサポート外の拍子記号です。2/4, 3/4, 4/4, 6/8, 7/8, 12/16 (1小節あたり 12〜96 ステップ) の MIDI を使用してください。`,
    };
  }
  const label = `${numerator}/${denominator}`;
  const stepsPerBar = Math.round(numerator * (4 / denominator) * 24);
  if (stepsPerBar < 12 || stepsPerBar > 96) {
    return {
      supported: false,
      numerator,
      denominator,
      label,
      stepsPerBar,
      reason: `現在の拍子 (${label}, 1小節=${stepsPerBar}ステップ) は MORTM 4.5 トークナイザの許容範囲 (12〜96ステップ / 最大 4/4・8/8 相当) 外です。4/4, 3/4, 6/8, 7/8, 12/16 等の MIDI で生成してください。`,
    };
  }
  return {
    supported: true,
    numerator,
    denominator,
    label,
    stepsPerBar,
    reason: null,
  };
}

/**
 * Computes the canonical `beatsPerMeasure` (`0.5..16`, `1/8`-beat resolution) corresponding
 * to a normalized `(numerator, denominator)` time signature (`[R8-MED-02]`).
 */
export function canonicalBeatsPerMeasure(numerator: number, denominator: number): number {
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
 * Derives a consistent `(numerator, denominator)` (`1..32` / `{1,2,4,8,16,32}`) from a
 * standalone `beatsPerMeasure` value when explicit time signature fields are omitted (`[R8-MED-02]`).
 */
export function deriveTimeSignatureFromBeats(
  beatsPerMeasure: number,
): { numerator: number; denominator: number } {
  const q = Math.max(0.5, Math.min(16, Math.round(finiteOr(beatsPerMeasure, 4) * 8) / 8));
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

export class MeasureRange {
  public readonly startMeasure: number;
  public readonly endMeasure: number;
  public readonly ppqn: number;
  public readonly beatsPerMeasure: number;

  constructor(
    startMeasure: number,
    endMeasure: number,
    ppqn = 480,
    beatsPerMeasure = 4,
  ) {
    const cleanStart = Math.max(1, Math.min(1024, Math.floor(finiteOr(startMeasure, 1))));
    const cleanEnd = Math.max(cleanStart, Math.min(1032, Math.floor(finiteOr(endMeasure, cleanStart))));
    this.startMeasure = cleanStart;
    this.endMeasure = cleanEnd;
    this.ppqn = Math.max(24, Math.min(32767, Math.round(finiteOr(ppqn, 480))));
    const rawBeats = finiteOr(beatsPerMeasure, 4);
    this.beatsPerMeasure = Math.max(0.5, Math.min(16, Math.round(rawBeats * 8) / 8));
    Object.freeze(this);
  }

  public get ticksPerMeasure(): number {
    return Math.max(1, Math.round(this.ppqn * this.beatsPerMeasure));
  }

  public get startTick(): number {
    return (this.startMeasure - 1) * this.ticksPerMeasure;
  }

  public get endTick(): number {
    return this.endMeasure * this.ticksPerMeasure;
  }

  public get measureCount(): number {
    return this.endMeasure - this.startMeasure + 1;
  }

  /** MORTM 4.5 SFT-Gen generates 1..8 measures per invocation. */
  public get clampedSftMeasureCount(): number {
    return Math.max(1, Math.min(8, this.measureCount));
  }

  /** Up to 8 measures immediately preceding `startTick` (matches Java COMPASS). */
  public pastWindowTicks(maxContextMeasures = 8): { startTick: number; endTick: number } {
    const bars = Math.max(0, Math.min(1032, Math.floor(finiteOr(maxContextMeasures, 8))));
    const pastStart = Math.max(0, this.startTick - bars * this.ticksPerMeasure);
    return {
      startTick: Math.min(MAX_SAFE_MIDI_TICK, pastStart),
      endTick: Math.min(MAX_SAFE_MIDI_TICK, this.startTick),
    };
  }

  /** Up to 8 measures immediately following `endTick` (matches Java COMPASS). */
  public futureWindowTicks(maxContextMeasures = 8): { startTick: number; endTick: number } {
    const bars = Math.max(0, Math.min(1032, Math.floor(finiteOr(maxContextMeasures, 8))));
    const start = Math.min(MAX_SAFE_MIDI_TICK, this.endTick);
    const rawEnd = start + bars * this.ticksPerMeasure;
    const clampedEnd = Number.isFinite(rawEnd)
      ? Math.max(start, Math.min(MAX_SAFE_MIDI_TICK, Math.round(rawEnd)))
      : MAX_SAFE_MIDI_TICK;
    return {
      startTick: start,
      endTick: clampedEnd,
    };
  }

  public withStartMeasure(nextStart: number): MeasureRange {
    const s = Math.max(1, Math.min(1024, Math.floor(finiteOr(nextStart, this.startMeasure))));
    const e = Math.max(s, Math.min(s + 7, this.endMeasure));
    return new MeasureRange(s, e, this.ppqn, this.beatsPerMeasure);
  }

  public withEndMeasure(nextEnd: number): MeasureRange {
    const e = Math.max(this.startMeasure, Math.min(1032, Math.floor(finiteOr(nextEnd, this.endMeasure))));
    return new MeasureRange(this.startMeasure, e, this.ppqn, this.beatsPerMeasure);
  }
}
