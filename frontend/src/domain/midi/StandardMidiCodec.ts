/**
 * Pure TypeScript Standard MIDI File (`.mid` / `.midi`, Format 0, 1, & 2) Codec.
 * Implements strict SMF 1.0 binary validation, VLQ encoding/decoding (including >=128-byte UTF-8
 * track names), Running Status cancellation on Meta/SysEx events, per-channel GM program & Drum Ch 10
 * handling, Time Signature (`FF 58 04`) round-tripping, single-track export (`past_midi` / `future_midi`),
 * dual-channel `inst_comp` conditioning export (`conditions_midi`: Ch 0 Piano Prog 0, Ch 1 Sax Prog 64),
 * and Format 1 multi-track export.
 */

import { DawTrack, DawTrackSnapshot, TrackInstrument } from '../DawTrack';
import { canonicalBeatsPerMeasure } from '../MeasureRange';
import {
  isDomainProxy,
  isGenuineMidiNote,
  MidiNote,
  safeIsArray,
  truncateWellFormedUtf16,
} from '../MidiNote';

export interface ParsedMidiTrack {
  readonly index: number;
  readonly name: string;
  readonly program: number;
  readonly isDrum: boolean;
  readonly instrument: TrackInstrument;
  readonly notes: readonly MidiNote[];
}

export interface ParsedMidiFile {
  readonly format: number;
  readonly ppqn: number;
  readonly bpm: number;
  readonly beatsPerMeasure: number;
  readonly timeSigNumerator?: number;
  readonly timeSigDenominator?: number;
  readonly tracks: readonly ParsedMidiTrack[];
}

export function mapGmProgramToInstrument(program: number, isDrum = false): TrackInstrument {
  if (isDrum) return 'DRUMS';
  if (program >= 32 && program <= 39) return 'BASS';
  if (program >= 40 && program <= 47) return 'VIOLIN';
  if (program >= 56 && program <= 79) return 'SAX';
  if (program >= 80 && program <= 95) return 'SYNTH';
  if (program >= 24 && program <= 31) return 'GUITAR';
  return 'PIANO';
}

export function mapInstrumentToGmProgram(instrument: TrackInstrument): number {
  switch (instrument) {
    case 'SAX':
      return 65; // Alto Sax
    case 'BASS':
      return 32; // Acoustic Bass
    case 'VIOLIN':
      return 40; // Violin
    case 'SYNTH':
      return 81; // Saw Lead
    case 'GUITAR':
      return 24; // Nylon Guitar
    case 'DRUMS':
    case 'PIANO':
    default:
      return 0; // Acoustic Grand Piano
  }
}

function readVarLen(data: Uint8Array, offset: number, limit: number): { value: number; nextOffset: number } {
  let value = 0;
  let cursor = offset;
  for (let i = 0; i < 4; i += 1) {
    if (cursor >= limit) {
      throw new Error('SMF 可変長数値 (VLQ) が途中で切断されています。');
    }
    const byte = data[cursor];
    cursor += 1;
    value = (value << 7) | (byte & 0x7f);
    if ((byte & 0x80) === 0) {
      return { value, nextOffset: cursor };
    }
  }
  throw new Error('不正な SMF 可変長数値 (VLQ) です (4バイトを超過しています)。');
}

const MAX_SMF_VLQ_VALUE = 0x0fffffff; // 28-bit SMF 1.0 VLQ ceiling (268,435,455)
export const MAX_SMF_QUARTER_NOTES = 4128; // 1,032 bars of 4/4 (1,981,440 ticks at 480 PPQN) (`[R4-MED-06]`)

function clampSmfTick(rawTick: number, minTick = 0, maxTick = MAX_SMF_VLQ_VALUE): number {
  if (!Number.isFinite(rawTick)) return minTick;
  return Math.max(minTick, Math.min(maxTick, Math.round(rawTick)));
}

function writeVarLen(value: number, out: number[]): void {
  let v = Number.isFinite(value)
    ? Math.max(0, Math.min(MAX_SMF_VLQ_VALUE, Math.floor(value)))
    : 0;
  const bytes: number[] = [v & 0x7f];
  v = Math.floor(v / 128);
  while (v > 0 && bytes.length < 4) {
    bytes.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  for (let i = bytes.length - 1; i >= 0; i -= 1) {
    out.push(bytes[i]);
  }
}

const AB_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength',
)?.get;
const AB_DETACHED_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'detached',
)?.get;

export const MAX_SMF_PARSE_BYTES = 32 * 1024 * 1024; // 32 MB SMF parse ceiling (`[R18-LOW-01]`)
export const MAX_PARSED_SMF_NOTES = 65_536; // Per-SMF parsed note ceiling (`[R20-HIGH-01]`)
export const MAX_PARSED_SMF_TRACKS = 512; // Per-SMF emitted note-track ceiling (`[R21-HIGH-02]`)

const SMF_UTF8_DECODER = new TextDecoder('utf-8', { fatal: false });
const MAX_TOTAL_SMF_TRACK_NAME_DECODE_ATTEMPTS = 2048;

export interface CanonicalParsedMidiTrackEntry {
  readonly canonicalIndex: number;
  readonly name: unknown;
  readonly program: unknown;
  readonly isDrum: unknown;
  readonly instrument: unknown;
  readonly notes: unknown;
}

/**
 * Shared 2-pass collision-free canonical track extraction (`[R22-LOW-01]`, `[R23-MED-02]`).
 * Guarantees identical own-property reading and canonical index assignment across
 * `MidiTrackImportModal`, `PianoRollView`, and `DemoPianoRollController.importParsedMidiTracks`,
 * with `extractCanonicalParsedMidiTracks(parsed, 512)` strictly prefix-identical to `4096`.
 */
export function extractCanonicalParsedMidiTracks(
  parsedMidi: unknown,
  maxTracks: number = MAX_PARSED_SMF_TRACKS,
): CanonicalParsedMidiTrackEntry[] {
  try {
    if (!parsedMidi || typeof parsedMidi !== 'object' || isDomainProxy(parsedMidi) || safeIsArray(parsedMidi)) {
      return [];
    }
    const rootProto = Object.getPrototypeOf(parsedMidi);
    if (rootProto !== null && rootProto !== Object.prototype) {
      return [];
    }
    const tracksDesc = Object.getOwnPropertyDescriptor(parsedMidi, 'tracks');
    if (!tracksDesc || !('value' in tracksDesc)) {
      return [];
    }
    const rawTracks = tracksDesc.value;
    if (!rawTracks || typeof rawTracks !== 'object' || !safeIsArray(rawTracks)) {
      return [];
    }
    const tLenDesc = Object.getOwnPropertyDescriptor(rawTracks, 'length');
    const scanLimit =
      tLenDesc && 'value' in tLenDesc && typeof tLenDesc.value === 'number' && Number.isFinite(tLenDesc.value)
        ? Math.min(4096, Math.max(0, Math.trunc(tLenDesc.value)))
        : 0;
    const returnCap =
      typeof maxTracks === 'number' && Number.isFinite(maxTracks)
        ? Math.max(0, Math.min(4096, Math.trunc(maxTracks)))
        : MAX_PARSED_SMF_TRACKS;
    if (returnCap <= 0 || scanLimit <= 0) {
      return [];
    }

    const rawEntries: Array<{
      readonly rawIndex: unknown;
      readonly name: unknown;
      readonly program: unknown;
      readonly isDrum: unknown;
      readonly instrument: unknown;
      readonly notes: unknown;
    }> = [];

    for (let ti = 0; ti < scanLimit && rawEntries.length < 4096; ti += 1) {
      const td = Object.getOwnPropertyDescriptor(rawTracks, ti);
      if (!td || !('value' in td)) continue;
      const pt = td.value;
      if (!pt || typeof pt !== 'object' || isDomainProxy(pt) || safeIsArray(pt)) {
        continue;
      }
      const ptProto = Object.getPrototypeOf(pt);
      if (ptProto !== null && ptProto !== Object.prototype) {
        continue;
      }
      const idxDesc = Object.getOwnPropertyDescriptor(pt, 'index');
      const nameDesc = Object.getOwnPropertyDescriptor(pt, 'name');
      const progDesc = Object.getOwnPropertyDescriptor(pt, 'program');
      const drumDesc = Object.getOwnPropertyDescriptor(pt, 'isDrum');
      const instDesc = Object.getOwnPropertyDescriptor(pt, 'instrument');
      const notesDesc = Object.getOwnPropertyDescriptor(pt, 'notes');
      if (
        (idxDesc && !('value' in idxDesc)) ||
        (nameDesc && !('value' in nameDesc)) ||
        (progDesc && !('value' in progDesc)) ||
        (drumDesc && !('value' in drumDesc)) ||
        (instDesc && !('value' in instDesc)) ||
        (notesDesc && !('value' in notesDesc))
      ) {
        continue;
      }
      rawEntries.push({
        rawIndex: idxDesc ? idxDesc.value : undefined,
        name: nameDesc ? nameDesc.value : undefined,
        program: progDesc ? progDesc.value : undefined,
        isDrum: drumDesc ? drumDesc.value : undefined,
        instrument: instDesc ? instDesc.value : undefined,
        notes: notesDesc ? notesDesc.value : undefined,
      });
    }

    const usedIndices = new Set<number>();
    const resolvedIndices = new Array<number>(rawEntries.length).fill(-1);
    for (let i = 0; i < rawEntries.length; i += 1) {
      const rIdx = rawEntries[i].rawIndex;
      if (
        typeof rIdx === 'number' &&
        Number.isSafeInteger(rIdx) &&
        rIdx >= 0 &&
        !usedIndices.has(rIdx)
      ) {
        usedIndices.add(rIdx);
        resolvedIndices[i] = rIdx;
      }
    }
    const outLen = Math.min(rawEntries.length, returnCap);
    let nextFree = 0;
    for (let i = 0; i < outLen; i += 1) {
      if (resolvedIndices[i] < 0) {
        while (usedIndices.has(nextFree)) {
          nextFree += 1;
        }
        usedIndices.add(nextFree);
        resolvedIndices[i] = nextFree;
        nextFree += 1;
      }
    }

    const out: CanonicalParsedMidiTrackEntry[] = new Array(outLen);
    for (let i = 0; i < outLen; i += 1) {
      const entry = rawEntries[i];
      out[i] = {
        canonicalIndex: resolvedIndices[i],
        name: entry.name,
        program: entry.program,
        isDrum: entry.isDrum,
        instrument: entry.instrument,
        notes: entry.notes,
      };
    }
    return out;
  } catch {
    return [];
  }
}

export class StandardMidiCodec {
  /**
   * Parses a Standard MIDI File binary buffer (`MThd` + `MTrk` chunks) in O(N) time.
   */
  public static parseMidi(buffer: ArrayBuffer): ParsedMidiFile {
    if (!buffer || typeof buffer !== 'object' || isDomainProxy(buffer)) {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    let rawByteLen = -1;
    try {
      if (AB_DETACHED_GETTER && AB_DETACHED_GETTER.call(buffer) === true) {
        throw new Error('detached');
      }
      rawByteLen = AB_BYTELENGTH_GETTER
        ? (AB_BYTELENGTH_GETTER.call(buffer) as number)
        : (buffer as ArrayBuffer).byteLength;
    } catch {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    if (typeof rawByteLen !== 'number' || !Number.isFinite(rawByteLen) || rawByteLen < 22) {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    if (rawByteLen > MAX_SMF_PARSE_BYTES) {
      throw new Error('MIDIファイルのサイズが解析上限 (32MB) を超えています。');
    }

    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);

    if (
      bytes.length < 22 ||
      bytes[0] !== 0x4d || // 'M'
      bytes[1] !== 0x54 || // 'T'
      bytes[2] !== 0x68 || // 'h'
      bytes[3] !== 0x64    // 'd'
    ) {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }

    const headerLen = view.getUint32(4, false);
    if (headerLen < 6 || 8 + headerLen > bytes.length) {
      throw new Error('MIDI ヘッダーチャンク (MThd) の長さが不正です。');
    }

    const format = view.getUint16(8, false);
    const numTracks = view.getUint16(10, false);
    const rawDivision = view.getUint16(12, false);

    if ((format !== 0 && format !== 1 && format !== 2) || numTracks < 1 || rawDivision === 0) {
      throw new Error('サポートされていない、または破損した SMF ヘッダー形式です。');
    }

    const isSmpte = (rawDivision & 0x8000) !== 0;
    let ppqn = 480;
    let smpteTicksPerQuarter = 480;
    let maxTrackTicks = MAX_SMF_QUARTER_NOTES * 480;
    if (!isSmpte) {
      ppqn = rawDivision;
      maxTrackTicks = MAX_SMF_QUARTER_NOTES * ppqn;
    } else {
      // SMPTE timecode: upper byte is negative frames/sec, lower byte is ticks/frame (`[R5-MED-02]`).
      // Linear SMPTE ticks are independent of `0xFF 0x51` Set Tempo; normalize onto metrical 480 PPQN @ 120 BPM (0.5s/beat).
      const fps = 256 - ((rawDivision >>> 8) & 0xff);
      const ticksPerFrame = rawDivision & 0xff;
      if (fps <= 0 || fps > 120 || ticksPerFrame === 0) {
        throw new Error('不正な SMPTE タイムコード分割値です。');
      }
      const actualFps = fps === 29 ? 29.97 : fps;
      smpteTicksPerQuarter = (actualFps * ticksPerFrame) / 2;
      ppqn = 480;
      maxTrackTicks = Math.floor(MAX_SMF_QUARTER_NOTES * smpteTicksPerQuarter);
    }
    const maxOutTicks = MAX_SMF_QUARTER_NOTES * ppqn;
    const toOutTick = (rawTick: number): number =>
      isSmpte
        ? Math.min(maxOutTicks, Math.max(0, Math.round((rawTick * 480) / smpteTicksPerQuarter)))
        : rawTick;

    let cursor = 8 + headerLen;
    let detectedBpm: number | null = null;
    let detectedBeatsPerMeasure: number | null = null;
    let detectedTimeSigNumerator: number | null = null;
    let detectedTimeSigDenominator: number | null = null;
    let mtrkCount = 0;
    let totalNoteOns = 0;
    let totalTrackNameDecodeAttempts = 0;
    const parsedTracks: ParsedMidiTrack[] = [];

    for (let tIdx = 0; tIdx < numTracks && cursor + 8 <= bytes.length; tIdx += 1) {
      const isMTrk =
        bytes[cursor] === 0x4d &&
        bytes[cursor + 1] === 0x54 &&
        bytes[cursor + 2] === 0x72 &&
        bytes[cursor + 3] === 0x6b;

      const chunkLen = view.getUint32(cursor + 4, false);
      const chunkStart = cursor + 8;
      if (chunkStart + chunkLen > bytes.length) {
        throw new Error('MIDI トラックチャンク (MTrk) が途中で切断されています。');
      }
      const chunkEnd = chunkStart + chunkLen;
      cursor = chunkEnd;

      if (!isMTrk) {
        continue;
      }
      mtrkCount += 1;

      let pos = chunkStart;
      let absTick = 0;
      let runningStatus = 0;
      let trackName = '';
      let trackNameDecodeAttempts = 0;
      // Lazily allocated per-track structures (`[R23-LOW-02]`: 0 allocations on note-less MTrk chunks)
      let channelPrograms: Uint8Array | null = null;
      let activeNotes: Map<
        number,
        { items: Array<{ startTick: number; velocity: number }>; head: number }
      > | null = null;
      let completedNotes: MidiNote[] | null = null;

      while (pos < chunkEnd) {
        const delta = readVarLen(bytes, pos, chunkEnd);
        absTick += delta.value;
        if (absTick > maxTrackTicks) {
          throw new Error('MIDI トラックの累積ティック数が上限を超えています。');
        }
        pos = delta.nextOffset;
        if (pos >= chunkEnd) {
          throw new Error('デルタタイム直後で SMF トラックイベントが途中で切断されています。');
        }

        let statusByte = bytes[pos];
        if (statusByte < 0x80) {
          if (runningStatus === 0) {
            throw new Error(
              '不正な MIDI データ: 有効なランニングステータスがない状態でデータバイトが検出されました。',
            );
          }
          statusByte = runningStatus;
        } else {
          pos += 1;
          if (statusByte < 0xf0) {
            runningStatus = statusByte;
          } else if (statusByte === 0xf0 || statusByte === 0xf7 || statusByte === 0xff) {
            // SMF 1.0 RP-001: Meta (0xFF) and SysEx (0xF0/0xF7) cancel Running Status
            runningStatus = 0;
          } else if (statusByte < 0xf8) {
            runningStatus = 0;
          }
        }

        if (statusByte === 0xff) {
          // Meta Event
          if (pos >= chunkEnd) {
            throw new Error('SMF メタイベントが途中で切断されています。');
          }
          const metaType = bytes[pos];
          pos += 1;
          const metaLenInfo = readVarLen(bytes, pos, chunkEnd);
          pos = metaLenInfo.nextOffset;
          if (pos + metaLenInfo.value > chunkEnd) {
            throw new Error('SMF メタイベントのデータ長が途中で切断されています。');
          }
          const metaEnd = pos + metaLenInfo.value;

          if (!isSmpte && metaType === 0x51 && metaLenInfo.value === 3 && detectedBpm === null && pos + 3 <= chunkEnd) {
            const mpqn = (bytes[pos] << 16) | (bytes[pos + 1] << 8) | bytes[pos + 2];
            if (mpqn > 0) {
              const rawDetectedBpm = Math.round((60_000_000 / mpqn) * 10) / 10;
              detectedBpm = Math.max(1, Math.min(999, rawDetectedBpm));
            }
          } else if (
            metaType === 0x58 &&
            metaLenInfo.value >= 2 &&
            detectedBeatsPerMeasure === null &&
            pos + 2 <= chunkEnd
          ) {
            const rawNumerator = bytes[pos];
            const denomPow = bytes[pos + 1];
            if (rawNumerator > 0) {
              let scaledNum: number;
              let clampedDenom: number;
              if (denomPow <= 5) {
                scaledNum = rawNumerator;
                clampedDenom = 1 << denomPow;
              } else {
                const shift = Math.min(20, denomPow - 5);
                scaledNum = Math.max(1, Math.round(rawNumerator / (1 << shift)));
                clampedDenom = 32;
              }
              while (scaledNum > 32 && clampedDenom > 1) {
                scaledNum = Math.max(1, Math.round(scaledNum / 2));
                clampedDenom = Math.floor(clampedDenom / 2);
              }
              const clampedNum = Math.max(1, Math.min(32, scaledNum));
              detectedTimeSigNumerator = clampedNum;
              detectedTimeSigDenominator = clampedDenom;
              detectedBeatsPerMeasure = canonicalBeatsPerMeasure(clampedNum, clampedDenom);
            }
          } else if (
            metaType === 0x03 &&
            !trackName &&
            metaLenInfo.value > 0 &&
            trackNameDecodeAttempts < 8 &&
            totalTrackNameDecodeAttempts < MAX_TOTAL_SMF_TRACK_NAME_DECODE_ATTEMPTS
          ) {
            let startNonWs = pos;
            while (startNonWs < metaEnd && bytes[startNonWs] <= 0x20) {
              startNonWs += 1;
            }
            if (startNonWs < metaEnd) {
              trackNameDecodeAttempts += 1;
              totalTrackNameDecodeAttempts += 1;
              try {
                const sliceEnd = Math.min(metaEnd, startNonWs + 2048);
                const decoded = SMF_UTF8_DECODER.decode(bytes.subarray(startNonWs, sliceEnd))
                  .replace(/\0/g, '')
                  .trim();
                if (decoded) {
                  trackName = truncateWellFormedUtf16(decoded, 256).trim();
                }
              } catch {
                trackName = '';
              }
            }
          }
          pos = metaEnd;
          if (metaType === 0x2f) {
            break;
          }
        } else if (statusByte === 0xf0 || statusByte === 0xf7) {
          // SysEx Event
          const sysexLen = readVarLen(bytes, pos, chunkEnd);
          if (sysexLen.nextOffset + sysexLen.value > chunkEnd) {
            throw new Error('SMF システムエクスクルーシブ (SysEx) イベントが途中で切断されています。');
          }
          pos = sysexLen.nextOffset + sysexLen.value;
        } else if (statusByte >= 0xf8) {
          // System Real-Time 1-byte message (0 bytes payload)
          continue;
        } else if (statusByte >= 0xf1 && statusByte <= 0xf6) {
          if (statusByte === 0xf2) {
            if (pos + 2 > chunkEnd) {
              throw new Error('SMF システムコモンイベントが途中で切断されています。');
            }
            pos += 2;
          } else if (statusByte === 0xf1 || statusByte === 0xf3) {
            if (pos + 1 > chunkEnd) {
              throw new Error('SMF システムコモンイベントが途中で切断されています。');
            }
            pos += 1;
          }
        } else {
          const msgType = statusByte & 0xf0;
          const channel = statusByte & 0x0f;

          if (msgType === 0xc0 || msgType === 0xd0) {
            if (pos + 1 > chunkEnd) {
              throw new Error('SMF チャンネルイベントが途中で切断されています。');
            }
            const d1 = bytes[pos];
            pos += 1;
            if (msgType === 0xc0 && channel !== 9) {
              if (!channelPrograms) {
                channelPrograms = new Uint8Array(16);
              }
              channelPrograms[channel] = d1 & 0x7f;
            }
          } else {
            if (pos + 2 > chunkEnd) {
              throw new Error('SMF チャンネルイベントが途中で切断されています。');
            }
            const d1 = bytes[pos];
            const d2 = bytes[pos + 1];
            pos += 2;

            if (msgType === 0x90 && d2 > 0) {
              totalNoteOns += 1;
              if (totalNoteOns > MAX_PARSED_SMF_NOTES) {
                throw new Error(
                  `MIDIファイル内のノート数が解析上限 (${MAX_PARSED_SMF_NOTES}音) を超えています。`,
                );
              }
              if (!activeNotes) {
                activeNotes = new Map();
              }
              const key = (channel << 7) | (d1 & 0x7f);
              let queue = activeNotes.get(key);
              if (!queue) {
                queue = { items: [], head: 0 };
                activeNotes.set(key, queue);
              }
              queue.items.push({ startTick: absTick, velocity: d2 & 0x7f });
            } else if (msgType === 0x80 || (msgType === 0x90 && d2 === 0)) {
              if (activeNotes) {
                const key = (channel << 7) | (d1 & 0x7f);
                const queue = activeNotes.get(key);
                if (queue && queue.head < queue.items.length) {
                  const started = queue.items[queue.head];
                  queue.head += 1;
                  if (queue.head >= queue.items.length) {
                    queue.items.length = 0;
                    queue.head = 0;
                  } else if (queue.head >= 1024 && queue.head * 2 >= queue.items.length) {
                    queue.items.splice(0, queue.head);
                    queue.head = 0;
                  }
                  const outStart = toOutTick(started.startTick);
                  const outEnd = toOutTick(absTick);
                  if (outStart < maxOutTicks) {
                    const dur = Math.max(1, Math.min(maxOutTicks - outStart, outEnd - outStart));
                    if (!completedNotes) {
                      completedNotes = [];
                    }
                    completedNotes.push(
                      new MidiNote({
                        pitch: d1 & 0x7f,
                        startTick: outStart,
                        durationTicks: dur,
                        velocity: started.velocity,
                        channel,
                      }),
                    );
                  }
                }
              }
            }
          }
        }
      }

      // Close any dangling unclosed notes gracefully without exceeding maxOutTicks (`[R6-LOW-01]`, `[R20-HIGH-01]`)
      if (activeNotes) {
        for (const [key, queue] of activeNotes.entries()) {
          if (queue.head >= queue.items.length) {
            continue;
          }
          const channel = (key >>> 7) & 0x0f;
          const pitch = key & 0x7f;
          for (let qi = queue.head; qi < queue.items.length; qi += 1) {
            const started = queue.items[qi];
            const outStart = toOutTick(started.startTick);
            const outEnd = toOutTick(absTick);
            if (outStart >= maxOutTicks) {
              continue;
            }
            const dur = Math.max(
              1,
              Math.min(
                maxOutTicks - outStart,
                Math.max(Math.round(ppqn / 4), outEnd - outStart),
              ),
            );
            if (!completedNotes) {
              completedNotes = [];
            }
            completedNotes.push(
              new MidiNote({
                pitch,
                startTick: outStart,
                durationTicks: dur,
                velocity: started.velocity,
                channel,
              }),
            );
          }
        }
      }

      if (completedNotes && completedNotes.length > 0) {
        // Group notes by MIDI channel when multiple channels carry notes in the same chunk (e.g. Format 0)
        const byChannel = new Map<number, MidiNote[]>();
        for (const note of completedNotes) {
          let list = byChannel.get(note.channel);
          if (!list) {
            list = [];
            byChannel.set(note.channel, list);
          }
          list.push(note);
        }

        const distinctChannels = Array.from(byChannel.keys()).sort((a, b) => a - b);
        if (parsedTracks.length + distinctChannels.length > MAX_PARSED_SMF_TRACKS) {
          throw new Error(
            `MIDIファイルのトラック数が上限 (${MAX_PARSED_SMF_TRACKS}トラック) を超えています。`,
          );
        }
        if (distinctChannels.length === 1) {
          const ch = distinctChannels[0];
          const isDrum = ch === 9;
          const prog = isDrum ? 0 : (channelPrograms ? channelPrograms[ch] : 0);
          const instrument = mapGmProgramToInstrument(prog, isDrum);
          completedNotes.sort((a, b) => a.startTick - b.startTick || a.pitch - b.pitch);
          parsedTracks.push({
            index: parsedTracks.length,
            name: trackName || `${instrument} ${parsedTracks.length + 1}`,
            program: prog,
            isDrum,
            instrument,
            notes: completedNotes,
          });
        } else {
          for (const ch of distinctChannels) {
            const chNotes = byChannel.get(ch)!;
            chNotes.sort((a, b) => a.startTick - b.startTick || a.pitch - b.pitch);
            const isDrum = ch === 9;
            const prog = isDrum ? 0 : (channelPrograms ? channelPrograms[ch] : 0);
            const instrument = mapGmProgramToInstrument(prog, isDrum);
            const suffix = `Ch ${ch + 1}`;
            parsedTracks.push({
              index: parsedTracks.length,
              name: trackName ? `${trackName} (${suffix})` : `${instrument} (${suffix})`,
              program: prog,
              isDrum,
              instrument,
              notes: chNotes,
            });
          }
        }
      }
    }

    if (mtrkCount === 0) {
      throw new Error('有効な MTrk トラックチャンクが含まれていません。');
    }

    return {
      format,
      ppqn,
      bpm: detectedBpm ?? 120,
      beatsPerMeasure: detectedBeatsPerMeasure ?? 4,
      timeSigNumerator: detectedTimeSigNumerator ?? 4,
      timeSigDenominator: detectedTimeSigDenominator ?? 4,
      tracks: parsedTracks,
    };
  }

  private static _resolveSmfTimeSignatureBytes(
    beatsPerMeasure: number,
    timeSignature?: { readonly numerator: number; readonly denominator: number },
  ): { readonly tsNumerator: number; readonly tsDenomPow: number } {
    try {
      if (timeSignature && typeof timeSignature === 'object' && !isDomainProxy(timeSignature)) {
        const numDesc = Object.getOwnPropertyDescriptor(timeSignature, 'numerator');
        const denDesc = Object.getOwnPropertyDescriptor(timeSignature, 'denominator');
        const num = numDesc && 'value' in numDesc ? numDesc.value : undefined;
        const den = denDesc && 'value' in denDesc ? denDesc.value : undefined;
        if (
          typeof num === 'number' &&
          Number.isFinite(num) &&
          num > 0 &&
          typeof den === 'number' &&
          Number.isFinite(den) &&
          den > 0
        ) {
          return {
            tsNumerator: Math.max(1, Math.min(32, Math.round(num))),
            tsDenomPow: Math.max(0, Math.min(5, Math.round(Math.log2(den)))),
          };
        }
      }
    } catch {
      // Fall through to sanitized beatsPerMeasure fallback
    }
    const safeBeats =
      typeof beatsPerMeasure === 'number' &&
      Number.isFinite(beatsPerMeasure) &&
      beatsPerMeasure > 0
        ? Math.max(0.125, Math.min(32, beatsPerMeasure))
        : 4;
    if (!Number.isInteger(safeBeats) && Number.isInteger(safeBeats * 2)) {
      return {
        tsNumerator: Math.max(1, Math.min(32, Math.round(safeBeats * 2))),
        tsDenomPow: 3, // 2^3 = 8 (eighth-note meter such as 7/8)
      };
    }
    if (!Number.isInteger(safeBeats * 2) && Number.isInteger(safeBeats * 4)) {
      return {
        tsNumerator: Math.max(1, Math.min(32, Math.round(safeBeats * 4))),
        tsDenomPow: 4, // 2^4 = 16 (sixteenth-note meter such as 3/16)
      };
    }
    if (!Number.isInteger(safeBeats * 4) && Number.isInteger(safeBeats * 8)) {
      return {
        tsNumerator: Math.max(1, Math.min(32, Math.round(safeBeats * 8))),
        tsDenomPow: 5, // 2^5 = 32 (thirty-second-note meter such as 5/32, 7/32)
      };
    }
    return {
      tsNumerator: Math.max(1, Math.min(32, Math.round(safeBeats))),
      tsDenomPow: 2, // 2^2 = 4 (quarter-note meter)
    };
  }

  private static _extractEncodableNote(
    raw: unknown,
    maxTickCeiling: number,
  ): { safeStart: number; safeEnd: number; safePitch: number; safeVel: number } | null {
    const startMax = Math.max(0, maxTickCeiling - 1);
    const endMax = Math.max(1, maxTickCeiling);
    if (isGenuineMidiNote(raw)) {
      const safeStart = clampSmfTick(raw.startTick, 0, startMax);
      const safeEnd = clampSmfTick(raw.endTick, Math.min(endMax, safeStart + 1), endMax);
      return {
        safeStart,
        safeEnd,
        safePitch: raw.pitch,
        safeVel: raw.velocity,
      };
    }
    try {
      if (!raw || typeof raw !== 'object' || safeIsArray(raw) || isDomainProxy(raw)) {
        return null;
      }
      const startDesc = Object.getOwnPropertyDescriptor(raw, 'startTick');
      const pitchDesc = Object.getOwnPropertyDescriptor(raw, 'pitch');
      if (!startDesc || !('value' in startDesc) || !pitchDesc || !('value' in pitchDesc)) {
        return null;
      }
      const startTick = startDesc.value;
      const pitch = pitchDesc.value;
      if (
        typeof startTick !== 'number' ||
        !Number.isFinite(startTick) ||
        typeof pitch !== 'number' ||
        !Number.isFinite(pitch)
      ) {
        return null;
      }
      const endDesc = Object.getOwnPropertyDescriptor(raw, 'endTick');
      const durDesc = Object.getOwnPropertyDescriptor(raw, 'durationTicks');
      const velDesc = Object.getOwnPropertyDescriptor(raw, 'velocity');
      const endVal = endDesc && 'value' in endDesc ? endDesc.value : undefined;
      const durVal = durDesc && 'value' in durDesc ? durDesc.value : undefined;
      const velVal = velDesc && 'value' in velDesc ? velDesc.value : undefined;
      const rawEnd =
        typeof endVal === 'number' && Number.isFinite(endVal)
          ? endVal
          : typeof durVal === 'number' && Number.isFinite(durVal)
            ? startTick + durVal
            : startTick + 1;
      const safeStart = clampSmfTick(startTick, 0, startMax);
      const safeEnd = clampSmfTick(rawEnd, Math.min(endMax, safeStart + 1), endMax);
      const safePitch = Math.max(0, Math.min(127, Math.round(pitch)));
      const safeVel =
        typeof velVal === 'number' && Number.isFinite(velVal)
          ? Math.max(1, Math.min(127, Math.round(velVal)))
          : 100;
      return { safeStart, safeEnd, safePitch, safeVel };
    } catch {
      return null;
    }
  }

  private static _readSafeArrayElements(rawArr: unknown, maxLen = 65_536): unknown[] {
    if (!safeIsArray(rawArr)) return [];
    let len = 0;
    try {
      const rawLen = rawArr.length;
      len =
        typeof rawLen === 'number' && Number.isFinite(rawLen)
          ? Math.min(maxLen, Math.max(0, Math.trunc(rawLen)))
          : 0;
    } catch {
      return [];
    }
    const out: unknown[] = [];
    for (let i = 0; i < len; i += 1) {
      try {
        const elem = rawArr[i];
        if (elem !== undefined) {
          out.push(elem);
        }
      } catch {
        // Skip throwing/revoked element
      }
    }
    return out;
  }

  private static _sanitizeEncodePpqn(ppqn: number): number {
    return typeof ppqn === 'number' && Number.isFinite(ppqn) && ppqn > 0
      ? Math.max(1, Math.min(32767, Math.round(ppqn)))
      : 480;
  }

  private static _computeSafeMpqn(bpm: number): number {
    const safeBpm = typeof bpm === 'number' && Number.isFinite(bpm) && bpm > 0 ? bpm : 120;
    return Math.max(1, Math.min(0xffffff, Math.round(60_000_000 / safeBpm)));
  }

  /**
   * Encodes a single track's notes into SMF Format 0 (matches Java COMPASS `saveMidiFile`).
   * Uses MIDI Channel 10 (index 9) automatically when `instrument === 'DRUMS'`.
   */
  public static encodeSingleTrackMidi(
    notes: readonly MidiNote[],
    ppqn = 480,
    bpm = 120,
    instrument: TrackInstrument = 'PIANO',
    beatsPerMeasure = 4,
    timeSignature?: { readonly numerator: number; readonly denominator: number },
  ): Uint8Array {
    const safePpqn = StandardMidiCodec._sanitizeEncodePpqn(ppqn);
    const mpqn = StandardMidiCodec._computeSafeMpqn(bpm);
    const isDrum = instrument === 'DRUMS';
    const ch = isDrum ? 9 : 0;
    const program = mapInstrumentToGmProgram(instrument);
    const { tsNumerator, tsDenomPow } = StandardMidiCodec._resolveSmfTimeSignatureBytes(
      beatsPerMeasure,
      timeSignature,
    );

    type RawEvent = { tick: number; order: number; bytes: number[] };
    const events: RawEvent[] = [
      {
        tick: 0,
        order: 0,
        bytes: [0xff, 0x58, 0x04, tsNumerator, tsDenomPow, 0x18, 0x08],
      },
      {
        tick: 0,
        order: 0,
        bytes: [0xff, 0x51, 0x03, (mpqn >>> 16) & 0xff, (mpqn >>> 8) & 0xff, mpqn & 0xff],
      },
    ];

    if (!isDrum) {
      events.push({
        tick: 0,
        order: 1,
        bytes: [0xc0 | ch, program & 0x7f],
      });
    }

    const safeNotes = StandardMidiCodec._readSafeArrayElements(notes);
    for (const rawNote of safeNotes) {
      const parsed = StandardMidiCodec._extractEncodableNote(rawNote, MAX_SMF_VLQ_VALUE);
      if (parsed === null) continue;
      events.push({
        tick: parsed.safeStart,
        order: 3,
        bytes: [0x90 | ch, parsed.safePitch & 0x7f, parsed.safeVel],
      });
      events.push({
        tick: parsed.safeEnd,
        order: 2, // Note-Off before Note-On at identical tick
        bytes: [0x80 | ch, parsed.safePitch & 0x7f, 0x00],
      });
    }

    return StandardMidiCodec._packFormat0Track(safePpqn, events);
  }

  /**
   * Encodes `conditions_midi` for `inst_comp` (Ch 0 = Piano Prog 0, Ch 1 = Sax Prog 64).
   */
  public static encodeConditionsMidi(
    notes: readonly MidiNote[],
    ppqn = 480,
    bpm = 120,
    isMonophonic = false,
    beatsPerMeasure = 4,
    timeSignature?: { readonly numerator: number; readonly denominator: number },
  ): Uint8Array {
    const safePpqn = StandardMidiCodec._sanitizeEncodePpqn(ppqn);
    const mpqn = StandardMidiCodec._computeSafeMpqn(bpm);
    const targetCh = isMonophonic ? 1 : 0;
    const { tsNumerator, tsDenomPow } = StandardMidiCodec._resolveSmfTimeSignatureBytes(
      beatsPerMeasure,
      timeSignature,
    );

    type RawEvent = { tick: number; order: number; bytes: number[] };
    const events: RawEvent[] = [
      {
        tick: 0,
        order: 0,
        bytes: [0xff, 0x58, 0x04, tsNumerator, tsDenomPow, 0x18, 0x08],
      },
      {
        tick: 0,
        order: 0,
        bytes: [0xff, 0x51, 0x03, (mpqn >>> 16) & 0xff, (mpqn >>> 8) & 0xff, mpqn & 0xff],
      },
      { tick: 0, order: 1, bytes: [0xc0, 0] },  // Ch 0: Acoustic Grand Piano
      { tick: 0, order: 1, bytes: [0xc1, 64] }, // Ch 1: Soprano/Alto Sax
    ];

    const safeCondNotes = StandardMidiCodec._readSafeArrayElements(notes);
    for (const rawNote of safeCondNotes) {
      const parsed = StandardMidiCodec._extractEncodableNote(rawNote, MAX_SMF_VLQ_VALUE);
      if (parsed === null) continue;
      events.push({
        tick: parsed.safeStart,
        order: 3,
        bytes: [0x90 | targetCh, parsed.safePitch & 0x7f, parsed.safeVel],
      });
      events.push({
        tick: parsed.safeEnd,
        order: 2,
        bytes: [0x80 | targetCh, parsed.safePitch & 0x7f, 0x00],
      });
    }

    return StandardMidiCodec._packFormat0Track(safePpqn, events);
  }

  /**
   * Encodes multiple tracks into SMF Format 1 for full DAW MIDI export.
   * Properly encodes UTF-8 `FF 03` Track Name lengths via Variable-Length Quantity (`writeVarLen`).
   */
  public static encodeMultiTrackMidi(
    tracks: readonly DawTrackSnapshot[],
    ppqn = 480,
    bpm = 120,
    beatsPerMeasure = 4,
    timeSignature?: { readonly numerator: number; readonly denominator: number },
  ): Uint8Array {
    const safePpqn = StandardMidiCodec._sanitizeEncodePpqn(ppqn);
    const maxTrackTicks = MAX_SMF_QUARTER_NOTES * safePpqn;
    const mpqn = StandardMidiCodec._computeSafeMpqn(bpm);
    const { tsNumerator, tsDenomPow } = StandardMidiCodec._resolveSmfTimeSignatureBytes(
      beatsPerMeasure,
      timeSignature,
    );
    const trackChunks: number[][] = [];

    // Track 0: Conductor / Time Signature + Tempo Track
    const conductorBody: number[] = [];
    writeVarLen(0, conductorBody);
    conductorBody.push(0xff, 0x58, 0x04, tsNumerator, tsDenomPow, 0x18, 0x08);
    writeVarLen(0, conductorBody);
    conductorBody.push(0xff, 0x51, 0x03, (mpqn >>> 16) & 0xff, (mpqn >>> 8) & 0xff, mpqn & 0xff);
    writeVarLen(0, conductorBody);
    conductorBody.push(0xff, 0x2f, 0x00);
    trackChunks.push(conductorBody);

    let channelCursor = 0;
    const safeTracks = StandardMidiCodec._readSafeArrayElements(tracks, 64);
    for (const rawTrk of safeTracks) {
      if (!rawTrk || typeof rawTrk !== 'object' || isDomainProxy(rawTrk)) continue;
      let trkInst: unknown;
      let trkName: unknown;
      let trkNotes: unknown;
      try {
        const resolvedTrk =
          rawTrk instanceof DawTrack && Object.getPrototypeOf(rawTrk) === DawTrack.prototype
            ? rawTrk.toSnapshot()
            : rawTrk;
        const instDesc = Object.getOwnPropertyDescriptor(resolvedTrk, 'instrument');
        const nameDesc = Object.getOwnPropertyDescriptor(resolvedTrk, 'name');
        const notesDesc = Object.getOwnPropertyDescriptor(resolvedTrk, 'notes');
        if (
          (instDesc && !('value' in instDesc)) ||
          (nameDesc && !('value' in nameDesc)) ||
          (notesDesc && !('value' in notesDesc))
        ) {
          continue;
        }
        trkInst = instDesc ? instDesc.value : undefined;
        trkName = nameDesc ? nameDesc.value : undefined;
        trkNotes = notesDesc ? notesDesc.value : undefined;
      } catch {
        continue;
      }
      const ch = trkInst === 'DRUMS' ? 9 : channelCursor;
      if (trkInst !== 'DRUMS') {
        channelCursor += 1;
        if (channelCursor === 9) channelCursor = 10;
        if (channelCursor > 15) channelCursor = 0;
      }

      const safeTrackName =
        typeof trkName === 'string' && trkName.trim()
          ? truncateWellFormedUtf16(trkName.trim(), 256).trim() || 'Track'
          : 'Track';
      const nameBytes = new TextEncoder().encode(safeTrackName);
      const nameEventBytes: number[] = [0xff, 0x03];
      writeVarLen(nameBytes.length, nameEventBytes);
      for (let bi = 0; bi < nameBytes.length; bi += 1) {
        nameEventBytes.push(nameBytes[bi]);
      }

      const program = mapInstrumentToGmProgram(trkInst as TrackInstrument);

      type RawEvent = { tick: number; order: number; bytes: number[] };
      const events: RawEvent[] = [
        {
          tick: 0,
          order: 0,
          bytes: nameEventBytes,
        },
      ];
      if (ch !== 9) {
        events.push({ tick: 0, order: 1, bytes: [0xc0 | ch, program & 0x7f] });
      }

      const safeNotes = StandardMidiCodec._readSafeArrayElements(trkNotes);
      for (const rawNote of safeNotes) {
        const parsed = StandardMidiCodec._extractEncodableNote(rawNote, maxTrackTicks);
        if (parsed === null) continue;
        events.push({
          tick: parsed.safeStart,
          order: 3,
          bytes: [0x90 | ch, parsed.safePitch & 0x7f, parsed.safeVel],
        });
        events.push({
          tick: parsed.safeEnd,
          order: 2,
          bytes: [0x80 | ch, parsed.safePitch & 0x7f, 0x00],
        });
      }

      events.sort((a, b) => a.tick - b.tick || a.order - b.order);
      const body: number[] = [];
      let prevTick = 0;
      for (const ev of events) {
        writeVarLen(Math.max(0, ev.tick - prevTick), body);
        prevTick = ev.tick;
        for (let bi = 0; bi < ev.bytes.length; bi += 1) {
          body.push(ev.bytes[bi]);
        }
      }
      writeVarLen(0, body);
      body.push(0xff, 0x2f, 0x00);
      trackChunks.push(body);
    }

    let totalBytes = 14;
    for (const chunk of trackChunks) {
      totalBytes += 8 + chunk.length;
    }
    const result = new Uint8Array(totalBytes);
    const dv = new DataView(result.buffer);
    result[0] = 0x4d;
    result[1] = 0x54;
    result[2] = 0x68;
    result[3] = 0x64; // MThd
    dv.setUint32(4, 6, false);
    dv.setUint16(8, 1, false); // Format 1
    dv.setUint16(10, trackChunks.length, false);
    dv.setUint16(12, safePpqn, false);

    let offset = 14;
    for (const chunk of trackChunks) {
      result[offset] = 0x4d;
      result[offset + 1] = 0x54;
      result[offset + 2] = 0x72;
      result[offset + 3] = 0x6b; // MTrk
      dv.setUint32(offset + 4, chunk.length, false);
      offset += 8;
      for (let i = 0; i < chunk.length; i += 1) {
        result[offset + i] = chunk[i];
      }
      offset += chunk.length;
    }

    return result;
  }

  /**
   * Built-in 4-bar Piano & Saxophone Demo Arrangement for instant audition & AI continuation testing.
   */
  public static createDemoMidiPreset(): ParsedMidiFile {
    const ppqn = 480;
    const bar = ppqn * 4;

    // 4-bar Jazz-Voiced Piano Progression: Cmaj9 -> Am9 -> Dm9 -> G13
    const chordBars: Array<{ barIdx: number; pitches: number[] }> = [
      { barIdx: 0, pitches: [48, 55, 59, 64, 74] },
      { barIdx: 1, pitches: [45, 52, 55, 60, 71] },
      { barIdx: 2, pitches: [50, 57, 60, 65, 76] },
      { barIdx: 3, pitches: [43, 53, 59, 64, 72] },
    ];

    const pianoNotes: MidiNote[] = [];
    for (const cb of chordBars) {
      const start = cb.barIdx * bar;
      for (let i = 0; i < cb.pitches.length; i += 1) {
        pianoNotes.push(
          new MidiNote({
            pitch: cb.pitches[i],
            startTick: start + (i === cb.pitches.length - 1 ? ppqn * 2 : 0),
            durationTicks: i === cb.pitches.length - 1 ? ppqn * 2 - 40 : bar - 60,
            velocity: i === 0 ? 92 : 78,
            channel: 0,
          }),
        );
      }
    }

    // 4-bar Lyrical Saxophone Lead
    const saxPhrase: Array<[number, number, number, number]> = [
      [76, 0, ppqn * 1.5, 96],
      [79, ppqn * 1.5, ppqn * 0.5, 88],
      [81, ppqn * 2, ppqn * 2 - 60, 102],
      [79, bar + ppqn * 0.5, ppqn * 1.5, 94],
      [76, bar + ppqn * 2, ppqn, 90],
      [72, bar + ppqn * 3, ppqn - 60, 84],
      [74, bar * 2, ppqn * 1.5, 98],
      [77, bar * 2 + ppqn * 1.5, ppqn * 1.5, 95],
      [76, bar * 2 + ppqn * 3, ppqn - 60, 88],
      [79, bar * 3, ppqn * 2, 104],
      [74, bar * 3 + ppqn * 2, ppqn * 2 - 80, 92],
    ];

    const saxNotes = saxPhrase.map(
      ([pitch, startTick, durationTicks, velocity]) =>
        new MidiNote({
          pitch,
          startTick: Math.round(startTick),
          durationTicks: Math.round(durationTicks),
          velocity,
          channel: 1,
        }),
    );

    return {
      format: 1,
      ppqn,
      bpm: 116,
      beatsPerMeasure: 4,
      tracks: [
        {
          index: 0,
          name: 'Grand Piano (Chords)',
          program: 0,
          isDrum: false,
          instrument: 'PIANO',
          notes: pianoNotes,
        },
        {
          index: 1,
          name: 'Alto Sax (Lead)',
          program: 65,
          isDrum: false,
          instrument: 'SAX',
          notes: saxNotes,
        },
      ],
    };
  }

  private static _packFormat0Track(
    ppqn: number,
    events: Array<{ tick: number; order: number; bytes: number[] }>,
  ): Uint8Array {
    events.sort((a, b) => a.tick - b.tick || a.order - b.order);
    const body: number[] = [];
    let prevTick = 0;
    for (const ev of events) {
      writeVarLen(Math.max(0, ev.tick - prevTick), body);
      prevTick = ev.tick;
      for (let bi = 0; bi < ev.bytes.length; bi += 1) {
        body.push(ev.bytes[bi]);
      }
    }
    writeVarLen(0, body);
    body.push(0xff, 0x2f, 0x00);

    const safePpqn = StandardMidiCodec._sanitizeEncodePpqn(ppqn);
    const result = new Uint8Array(22 + body.length);
    const dv = new DataView(result.buffer);
    result[0] = 0x4d;
    result[1] = 0x54;
    result[2] = 0x68;
    result[3] = 0x64; // MThd
    dv.setUint32(4, 6, false);
    dv.setUint16(8, 0, false); // Format 0
    dv.setUint16(10, 1, false); // 1 track
    dv.setUint16(12, safePpqn, false);
    result[14] = 0x4d;
    result[15] = 0x54;
    result[16] = 0x72;
    result[17] = 0x6b; // MTrk
    dv.setUint32(18, body.length, false);
    for (let i = 0; i < body.length; i += 1) {
      result[22 + i] = body[i];
    }

    return result;
  }
}
