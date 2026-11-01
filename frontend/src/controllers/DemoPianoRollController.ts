/**
 * Standalone Demo Piano Roll DAW Controller (`DemoPianoRollController`).
 * Implements `IDawInteractivePianoRoll` (`IDawController`), modeled faithfully after
 * the user's Java `COMPASS` DTM (`ArrangementFrame` + `PianoRoll` + `MidiHandler` + `PlaybackManager`)
 * while cleanly decoupled from the `COMPASS AI` Control Panel View.
 */

import { IAudioEngine } from '../audio/WebAudioEngine';
import { CommandManager } from '../domain/commands/CommandManager';
import {
  AddNotesCommand,
  DeleteNotesCommand,
  ReplaceRangeNotesCommand,
  TransformNotesCommand,
} from '../domain/commands/NoteCommands';
import {
  applyValidatedUniqueSyncNotes,
  cloneCanonicalIdSetMutable,
  createFastCanonicalIdHas,
  createTrackId,
  DawTrack,
  EMPTY_CANONICAL_ID_SET,
  isValidTrackInstrument,
  MAX_NOTES_PER_TRACK,
  normalizeTrackIdIterable,
  normalizeTrackInputNote,
  prepareScaledCandidateBatch,
  TRACK_SWATCHES,
  TrackInstrument,
} from '../domain/DawTrack';
import {
  canonicalBeatsPerMeasure,
  deriveTimeSignatureFromBeats,
  MeasureRange,
} from '../domain/MeasureRange';
import {
  createNoteId,
  createValidatedSyncMidiNote,
  MidiNote,
  truncateWellFormedUtf16,
} from '../domain/MidiNote';
import {
  extractCanonicalParsedMidiTracks,
  MAX_SMF_QUARTER_NOTES,
  ParsedMidiFile,
  StandardMidiCodec,
} from '../domain/midi/StandardMidiCodec';
import {
  extractValidatedCandidateMidiBytes,
  GeneratedMidiCandidate,
  GenerationReason,
  isRuntimeProxy,
  MAX_CANDIDATE_MIDI_BYTES,
  normalizeGeneratedCandidatesBatch,
  sanitizeGenerationReason,
} from '../services/CompassApiClient';
import {
  buildTrustedWorkspaceSnapshot,
  consumeVerifiedInboundSyncState,
  parseValidatedGenerationReason,
} from '../services/WorkspaceSyncBus';
import {
  ActiveCandidateRange,
  ContextAvailability,
  DawCapabilities,
  DawSessionSnapshot,
  ExtractedMidiContext,
  IDawInteractivePianoRoll,
  resolveEffectiveSftTask,
  MidiTrackImportResult,
  SerializedWorkspaceState,
  SftTaskMode,
} from './IDawController';

const EMPTY_MIDI_IMPORT_RESULT: MidiTrackImportResult = Object.freeze({
  importedTracks: 0,
  importedNotes: 0,
});
const EMPTY_STRING_SET: ReadonlySet<string> = EMPTY_CANONICAL_ID_SET;

const MAX_UPLOAD_MIDI_BYTES = 10 * 1024 * 1024; // 10 MB (`secure-coding-guard` §4)
const UPLOAD_AB_BYTELENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength',
)?.get;
const UPLOAD_AB_DETACHED_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'detached',
)?.get;
const MAX_WORKSPACE_TRACKS = 64; // Matches `ipc_security.cjs` & `WorkspaceSyncBus.ts` (`[R4-MED-07]`)
const MAX_WORKSPACE_NOTES = 32_768; // Matches `ipc_security.cjs` & `WorkspaceSyncBus.ts` (`[R4-MED-07]`)

const ALLOWED_SYNC_TRACK_KEYS: ReadonlySet<string> = new Set([
  'id',
  'name',
  'instrument',
  'colorHex',
  'isMuted',
  'isSoloed',
  'notes',
]);
const ALLOWED_SYNC_SLOT_KEYS: ReadonlySet<string> = new Set(['trackId', 'startTick', 'endTick']);
const ALLOWED_SYNC_CANDIDATE_KEYS: ReadonlySet<string> = new Set([
  'index',
  'filename',
  'rawMidiBytes',
  'reason',
]);
const ALLOWED_SYNC_WORKSPACE_KEYS: ReadonlySet<string> = new Set([
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

function normalizeWorkspaceTimeSignature(
  rawNum: number | undefined,
  rawDen: number | undefined,
): { numerator: number; denominator: number } {
  const cleanDenRaw =
    typeof rawDen === 'number' && Number.isFinite(rawDen) && rawDen >= 1
      ? Math.round(rawDen)
      : 4;
  const pow = Math.max(0, Math.min(5, Math.round(Math.log2(cleanDenRaw))));
  let denominator = 1 << pow; // Strictly in {1, 2, 4, 8, 16, 32}
  const scaleFactor = cleanDenRaw / denominator;
  let scaledNum =
    typeof rawNum === 'number' && Number.isFinite(rawNum) && rawNum >= 1
      ? Math.max(1, Math.round(rawNum / scaleFactor))
      : 4;
  while (scaledNum > 32 && denominator > 1) {
    scaledNum = Math.max(1, Math.round(scaledNum / 2));
    denominator = Math.floor(denominator / 2);
  }
  const numerator = Math.max(1, Math.min(32, scaledNum));
  return { numerator, denominator };
}
const VALID_TRACK_INSTRUMENTS: ReadonlySet<string> = new Set([
  'PIANO',
  'SAX',
  'BASS',
  'VIOLIN',
  'SYNTH',
  'GUITAR',
  'DRUMS',
]);

function areUint8ArraysByteEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a === b) return true;
  const len = a.byteLength;
  if (len !== b.byteLength) return false;
  if ((a.byteOffset & 7) === 0 && (b.byteOffset & 7) === 0 && len >= 8) {
    const wordCount = len >>> 3;
    const va = new BigUint64Array(a.buffer, a.byteOffset, wordCount);
    const vb = new BigUint64Array(b.buffer, b.byteOffset, wordCount);
    for (let i = 0; i < wordCount; i += 1) {
      if (va[i] !== vb[i]) return false;
    }
    for (let i = wordCount << 3; i < len; i += 1) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }
  for (let i = 0; i < len; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export class DemoPianoRollController implements IDawInteractivePianoRoll {
  public readonly id = 'demo-pianoroll';
  public readonly displayName = 'COMPASS Demo Studio (内蔵ピアノロール)';
  public readonly capabilities: DawCapabilities = Object.freeze({
    hasEmbeddedPianoRoll: true,
    supportsMultiTrack: true,
    supportsTransportControl: true,
    supportsDirectNoteEditing: true,
    connectionProtocol: 'Internal In-Memory Sequencer + WebAudio',
  });

  private readonly _audioEngine: IAudioEngine;
  private readonly _commands = new CommandManager(120);
  private readonly _listeners = new Set<(snapshot: DawSessionSnapshot) => void>();

  private _ppqn = 480;
  private _bpm = 120;
  private _beatsPerMeasure = 4;
  private _timeSigNumerator = 4;
  private _timeSigDenominator = 4;
  private _tracks: DawTrack[] = [];
  private _activeTrackId = '';
  private _targetRange: MeasureRange;
  private _quantizeDivision: 4 | 8 | 16 | 32 = 16;
  private _isTriplet = false;
  private _selectedNoteIds: ReadonlySet<string> = EMPTY_STRING_SET;
  private _clipboard: MidiNote[] = [];

  private _isPlaying = false;
  private _playRequestId = 0;
  private _playheadTick = 0;
  private _schedulerTimer: ReturnType<typeof setInterval> | null = null;
  private _lastAudioTime = 0;
  private _scheduledUpToTick = 0;

  private _candidates: GeneratedMidiCandidate[] = [];
  private _activeCandidateIndex = 0;
  private _activeCandidateNoteIds: ReadonlySet<string> = EMPTY_STRING_SET;
  private _activeCandidateRange: ActiveCandidateRange | null = null;
  private _candidateOriginSlot: ActiveCandidateRange | null = null;
  private _latestReason: GenerationReason | null = null;
  private _cachedSnapshot: Omit<DawSessionSnapshot, 'selectedNoteIds'> | null = null;

  constructor(audioEngine: IAudioEngine) {
    this._audioEngine = audioEngine;
    const initialTrack = new DawTrack({
      name: 'Piano 1',
      instrument: 'PIANO',
      colorHex: TRACK_SWATCHES[0],
    });
    const secondTrack = new DawTrack({
      name: 'Saxophone 2',
      instrument: 'SAX',
      colorHex: TRACK_SWATCHES[1],
    });
    this._tracks = [initialTrack, secondTrack];
    this._activeTrackId = initialTrack.id;
    this._targetRange = new MeasureRange(1, 4, this._ppqn, this._beatsPerMeasure);
  }

  public async connect(): Promise<void> {
    this._invalidateAndNotify();
  }

  public async disconnect(): Promise<void> {
    this.stop();
  }

  public subscribe(listener: (snapshot: DawSessionSnapshot) => void): () => void {
    if (typeof listener !== 'function' || isRuntimeProxy(listener)) {
      return () => {};
    }
    this._listeners.add(listener);
    return () => {
      this._listeners.delete(listener);
    };
  }

  public getSnapshot(): DawSessionSnapshot {
    if (!this._cachedSnapshot) {
      const contextAvailability = Object.freeze({
        ...this._computeContextAvailability(),
      });
      this._cachedSnapshot = {
        controllerId: this.id,
        controllerName: this.displayName,
        isConnected: true,
        connectionStatusText: '内蔵デモシーケンサー稼働中',
        capabilities: this.capabilities,
        ppqn: this._ppqn,
        bpm: this._bpm,
        beatsPerMeasure: this._beatsPerMeasure,
        timeSigNumerator: this._timeSigNumerator,
        timeSigDenominator: this._timeSigDenominator,
        isPlaying: this._isPlaying,
        playheadTick: Math.round(this._playheadTick),
        tracks: Object.freeze(this._tracks.map((t) => t.toSnapshot())),
        activeTrackId: this._activeTrackId,
        targetRange: this._targetRange,
        quantizeDivision: this._quantizeDivision,
        isTriplet: this._isTriplet,
        canUndo: this._commands.canUndo(),
        canRedo: this._commands.canRedo(),
        undoDescription: this._commands.peekUndoDescription(),
        candidates: Object.freeze([...this._candidates]),
        activeCandidateIndex: this._activeCandidateIndex,
        activeCandidateRange: this._activeCandidateRange
          ? Object.freeze({ ...this._activeCandidateRange })
          : null,
        latestReason: this._latestReason,
        contextAvailability,
      };
    }
    const capturedSelection = this._selectedNoteIds;
    let lazySelectedSet: Set<string> | null = null;
    const baseSnapshot = this._cachedSnapshot;
    return Object.freeze({
      ...baseSnapshot,
      get selectedNoteIds(): ReadonlySet<string> {
        if (lazySelectedSet === null) {
          lazySelectedSet = cloneCanonicalIdSetMutable(capturedSelection);
        }
        return lazySelectedSet;
      },
    });
  }

  // =========================================================================
  // Transport & Playback Scheduler
  // =========================================================================

  public async play(): Promise<void> {
    if (this._isPlaying) return;
    const reqId = ++this._playRequestId;
    await this._audioEngine.ensureResumed();
    if (reqId !== this._playRequestId || this._isPlaying) {
      return;
    }
    if (this._schedulerTimer !== null) {
      clearInterval(this._schedulerTimer);
      this._schedulerTimer = null;
    }
    this._isPlaying = true;
    this._lastAudioTime = this._audioEngine.getCurrentTime();
    this._scheduledUpToTick = this._playheadTick;

    this._advanceTransport();
    if (!this._isPlaying) {
      return;
    }
    this._schedulerTimer = setInterval(() => {
      this._advanceTransport();
    }, 25);
    this._invalidateAndNotify();
  }

  public pause(): void {
    this._playRequestId += 1;
    const wasPlaying = this._isPlaying;
    this._isPlaying = false;
    if (this._schedulerTimer !== null) {
      clearInterval(this._schedulerTimer);
      this._schedulerTimer = null;
    }
    this._audioEngine.stopAllVoices();
    if (!wasPlaying) return;
    this._invalidateAndNotify();
  }

  public stop(): void {
    this.pause();
    this._playheadTick = this._targetRange.startTick;
    this._scheduledUpToTick = this._playheadTick;
    this._invalidateAndNotify();
  }

  private _silenceActivePlaybackVoices(): void {
    if (this._isPlaying) {
      this._audioEngine.stopAllVoices();
      this._scheduledUpToTick = this._playheadTick;
      this._lastAudioTime = this._audioEngine.getCurrentTime();
    }
  }

  public setPlayheadTick(tick: number): void {
    if (typeof tick !== 'number' || !Number.isFinite(tick)) return;
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * this._ppqn;
    this._playheadTick = Math.max(0, Math.min(maxProjectTicks, Math.round(tick)));
    this._scheduledUpToTick = this._playheadTick;
    if (this._isPlaying) {
      this._audioEngine.stopAllVoices();
      this._lastAudioTime = this._audioEngine.getCurrentTime();
    }
    this._invalidateAndNotify();
  }

  public setBpm(bpm: number): void {
    if (!Number.isFinite(bpm)) return;
    this._bpm = Math.max(20, Math.min(300, Math.round(bpm * 10) / 10));
    this._invalidateAndNotify();
  }

  public setTimeSignature(numerator: number, denominator: number): void {
    if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) return;
    const num = Math.max(1, Math.min(32, Math.round(numerator)));
    const den = Math.round(denominator);
    if (![1, 2, 4, 8, 16, 32].includes(den)) return;
    const nextBeats = canonicalBeatsPerMeasure(num, den);
    this._timeSigNumerator = num;
    this._timeSigDenominator = den;
    this._beatsPerMeasure = nextBeats;
    this._targetRange = new MeasureRange(
      this._targetRange.startMeasure,
      this._targetRange.endMeasure,
      this._ppqn,
      this._beatsPerMeasure,
    );
    this._invalidateAndNotify();
  }

  public setTargetMeasureRange(startMeasure: number, endMeasure: number): void {
    if (!Number.isFinite(startMeasure) || !Number.isFinite(endMeasure)) return;
    this._targetRange = new MeasureRange(
      startMeasure,
      endMeasure,
      this._ppqn,
      this._beatsPerMeasure,
    );
    this._invalidateAndNotify();
  }

  public setActiveTrack(trackId: string): void {
    if (typeof trackId !== 'string') return;
    if (this._tracks.some((t) => t.id === trackId)) {
      this._activeTrackId = trackId;
      this._selectedNoteIds = EMPTY_STRING_SET;
      this._invalidateAndNotify();
    }
  }

  // =========================================================================
  // MIDI Upload / Export / Multi-Window Workspace Sync
  // =========================================================================

  public async parseUploadedMidiFile(file: File): Promise<ParsedMidiFile> {
    if (!file || typeof file !== 'object' || isRuntimeProxy(file)) {
      throw new Error('MIDIファイル (.mid / .midi) のみアップロード可能です。');
    }
    let rawName: unknown;
    let rawSize: unknown;
    let arrayBufferFn: unknown;
    try {
      rawName = file.name;
      rawSize = file.size;
      arrayBufferFn = file.arrayBuffer;
    } catch {
      throw new Error('MIDIファイルの読み込みに失敗しました。');
    }
    const fileName = typeof rawName === 'string' ? rawName.trim() : '';
    if (!/\.(mid|midi)$/i.test(fileName)) {
      throw new Error('MIDIファイル (.mid / .midi) のみアップロード可能です。');
    }
    if (typeof rawSize !== 'number' || !Number.isFinite(rawSize)) {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    if (rawSize > MAX_UPLOAD_MIDI_BYTES) {
      throw new Error('MIDIファイルのサイズが上限 (10MB) を超えています。');
    }
    if (rawSize < 22) {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    if (typeof arrayBufferFn !== 'function' || isRuntimeProxy(arrayBufferFn)) {
      throw new Error('MIDIファイルの読み込みに失敗しました。');
    }
    let buffer: unknown;
    try {
      buffer = await Reflect.apply(arrayBufferFn as () => Promise<ArrayBuffer>, file, []);
    } catch {
      throw new Error('MIDIファイルの読み込みに失敗しました。');
    }
    if (!buffer || typeof buffer !== 'object' || isRuntimeProxy(buffer)) {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    let actualByteLen = -1;
    try {
      if (UPLOAD_AB_DETACHED_GETTER && UPLOAD_AB_DETACHED_GETTER.call(buffer) === true) {
        throw new Error('detached');
      }
      if (!UPLOAD_AB_BYTELENGTH_GETTER) {
        throw new Error('missing-getter');
      }
      actualByteLen = UPLOAD_AB_BYTELENGTH_GETTER.call(buffer) as number;
    } catch {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    if (typeof actualByteLen !== 'number' || !Number.isFinite(actualByteLen) || actualByteLen < 22) {
      throw new Error('有効な Standard MIDI ファイル (MThd) ではありません。');
    }
    if (actualByteLen > MAX_UPLOAD_MIDI_BYTES) {
      throw new Error('MIDIファイルのサイズが上限 (10MB) を超えています。');
    }
    return StandardMidiCodec.parseMidi(buffer as ArrayBuffer);
  }

  public importParsedMidiTracks(
    parsedFile: ParsedMidiFile,
    selectedTrackIndices: readonly number[],
    replaceExisting: boolean,
  ): MidiTrackImportResult {
    const safeReplaceExisting = Boolean(replaceExisting);
    let rawPpqn: unknown;
    let rawBpm: unknown;
    let rawBeats: unknown;
    let rawTsNum: unknown;
    let rawTsDen: unknown;
    const chosenTracks: Array<{
      readonly name: unknown;
      readonly instrument: unknown;
      readonly notes: readonly unknown[];
    }> = [];

    try {
      if (
        !parsedFile ||
        typeof parsedFile !== 'object' ||
        isRuntimeProxy(parsedFile) ||
        Array.isArray(parsedFile)
      ) {
        return EMPTY_MIDI_IMPORT_RESULT;
      }
      const parsedProto = Object.getPrototypeOf(parsedFile);
      if (parsedProto !== null && parsedProto !== Object.prototype) {
        return EMPTY_MIDI_IMPORT_RESULT;
      }
      if (
        !selectedTrackIndices ||
        typeof selectedTrackIndices !== 'object' ||
        isRuntimeProxy(selectedTrackIndices)
      ) {
        return EMPTY_MIDI_IMPORT_RESULT;
      }

      const tracksDesc = Object.getOwnPropertyDescriptor(parsedFile, 'tracks');
      const ppqnDesc = Object.getOwnPropertyDescriptor(parsedFile, 'ppqn');
      const bpmDesc = Object.getOwnPropertyDescriptor(parsedFile, 'bpm');
      const beatsDesc = Object.getOwnPropertyDescriptor(parsedFile, 'beatsPerMeasure');
      const tsNumDesc = Object.getOwnPropertyDescriptor(parsedFile, 'timeSigNumerator');
      const tsDenDesc = Object.getOwnPropertyDescriptor(parsedFile, 'timeSigDenominator');
      if (
        !tracksDesc ||
        !('value' in tracksDesc) ||
        (ppqnDesc && !('value' in ppqnDesc)) ||
        (bpmDesc && !('value' in bpmDesc)) ||
        (beatsDesc && !('value' in beatsDesc)) ||
        (tsNumDesc && !('value' in tsNumDesc)) ||
        (tsDenDesc && !('value' in tsDenDesc))
      ) {
        return EMPTY_MIDI_IMPORT_RESULT;
      }
      const rawTracks = tracksDesc.value;
      rawPpqn = ppqnDesc ? ppqnDesc.value : undefined;
      rawBpm = bpmDesc ? bpmDesc.value : undefined;
      rawBeats = beatsDesc ? beatsDesc.value : undefined;
      rawTsNum = tsNumDesc ? tsNumDesc.value : undefined;
      rawTsDen = tsDenDesc ? tsDenDesc.value : undefined;

      if (!rawTracks || typeof rawTracks !== 'object' || isRuntimeProxy(rawTracks) || !Array.isArray(rawTracks)) {
        return EMPTY_MIDI_IMPORT_RESULT;
      }

      const chosenSet = new Set<number>();
      if (Array.isArray(selectedTrackIndices)) {
        const ld = Object.getOwnPropertyDescriptor(selectedTrackIndices, 'length');
        const selLen =
          ld && 'value' in ld && typeof ld.value === 'number' && Number.isFinite(ld.value)
            ? Math.min(4096, Math.max(0, Math.trunc(ld.value)))
            : 0;
        for (let i = 0; i < selLen; i += 1) {
          const ed = Object.getOwnPropertyDescriptor(selectedTrackIndices, i);
          if (
            ed &&
            'value' in ed &&
            typeof ed.value === 'number' &&
            Number.isInteger(ed.value) &&
            ed.value >= 0
          ) {
            chosenSet.add(ed.value);
          }
        }
      } else {
        let count = 0;
        for (const item of selectedTrackIndices as Iterable<unknown>) {
          if (count >= 4096) break;
          count += 1;
          if (typeof item === 'number' && Number.isInteger(item) && item >= 0) {
            chosenSet.add(item);
          }
        }
      }
      if (chosenSet.size === 0) {
        return EMPTY_MIDI_IMPORT_RESULT;
      }

      const maxTracksToImport = safeReplaceExisting
        ? MAX_WORKSPACE_TRACKS
        : Math.max(0, MAX_WORKSPACE_TRACKS - this._tracks.length);
      if (maxTracksToImport <= 0) {
        return EMPTY_MIDI_IMPORT_RESULT;
      }

      const canonicalEntries = extractCanonicalParsedMidiTracks(parsedFile, 4096);
      for (
        let ti = 0;
        ti < canonicalEntries.length && chosenTracks.length < maxTracksToImport;
        ti += 1
      ) {
        const entry = canonicalEntries[ti];
        if (
          !chosenSet.has(entry.canonicalIndex) ||
          !entry.notes ||
          typeof entry.notes !== 'object' ||
          isRuntimeProxy(entry.notes) ||
          !Array.isArray(entry.notes)
        ) {
          continue;
        }
        chosenTracks.push({
          name: entry.name,
          instrument: entry.instrument,
          notes: entry.notes,
        });
      }
    } catch {
      return EMPTY_MIDI_IMPORT_RESULT;
    }

    if (chosenTracks.length === 0) {
      return EMPTY_MIDI_IMPORT_RESULT;
    }

    this.pause();

    const safeSourcePpqn =
      typeof rawPpqn === 'number' && Number.isFinite(rawPpqn) && rawPpqn > 0
        ? Math.max(1, Math.min(32767, Math.round(rawPpqn)))
        : 480;
    const targetPpqn = this._ppqn;
    const scale = targetPpqn / safeSourcePpqn;
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * targetPpqn;
    let remainingNoteBudget = safeReplaceExisting
      ? MAX_WORKSPACE_NOTES
      : Math.max(0, MAX_WORKSPACE_NOTES - this._totalNoteCount());
    const seenWorkspaceNoteIds = new Set<string>();
    const seenWorkspaceTrackIds = new Set<string>();
    if (!safeReplaceExisting) {
      for (const existingTrk of this._tracks) {
        seenWorkspaceTrackIds.add(truncateWellFormedUtf16(existingTrk.id.trim(), 128).trim());
        for (const en of existingTrk.toSnapshot().notes) {
          seenWorkspaceNoteIds.add(truncateWellFormedUtf16(en.id.trim(), 128).trim());
        }
      }
    }

    const importedDawTracks: DawTrack[] = chosenTracks.map((pt, idx) => {
      const scaledNotes: MidiNote[] = [];
      let rawNotesLen = 0;
      try {
        const nld = Object.getOwnPropertyDescriptor(pt.notes, 'length');
        rawNotesLen =
          nld && 'value' in nld && typeof nld.value === 'number' && Number.isFinite(nld.value)
            ? Math.min(MAX_WORKSPACE_NOTES, Math.max(0, Math.trunc(nld.value)))
            : 0;
      } catch {
        rawNotesLen = 0;
      }
      for (let ni = 0; ni < rawNotesLen && scaledNotes.length < MAX_NOTES_PER_TRACK; ni += 1) {
        if (remainingNoteBudget <= 0) break;
        let rawElem: unknown;
        try {
          const ned = Object.getOwnPropertyDescriptor(pt.notes, ni);
          if (!ned || !('value' in ned)) continue;
          rawElem = ned.value;
        } catch {
          continue;
        }
        const normalizedNote = normalizeTrackInputNote(rawElem);
        if (normalizedNote === null) continue;
        const scaled = normalizedNote.scaleAndShift(scale, 0, true);
        if (scaled.startTick >= maxProjectTicks) continue;
        const clampedDur = Math.max(
          1,
          Math.min(scaled.durationTicks, maxProjectTicks - scaled.startTick),
        );
        let uniqueNoteId = truncateWellFormedUtf16(scaled.id.trim(), 128).trim();
        while (!uniqueNoteId || seenWorkspaceNoteIds.has(uniqueNoteId)) {
          uniqueNoteId = truncateWellFormedUtf16(createNoteId().trim(), 128).trim();
        }
        const finalNote =
          clampedDur === scaled.durationTicks && uniqueNoteId === scaled.id
            ? scaled
            : new MidiNote({
                id: uniqueNoteId,
                pitch: scaled.pitch,
                startTick: scaled.startTick,
                durationTicks: clampedDur,
                velocity: scaled.velocity,
                channel: scaled.channel,
              });
        seenWorkspaceNoteIds.add(truncateWellFormedUtf16(finalNote.id.trim(), 128).trim());
        scaledNotes.push(finalNote);
        remainingNoteBudget -= 1;
      }
      let uniqueTrackId = truncateWellFormedUtf16(createTrackId().trim(), 128).trim();
      while (!uniqueTrackId || seenWorkspaceTrackIds.has(uniqueTrackId)) {
        uniqueTrackId = truncateWellFormedUtf16(createTrackId().trim(), 128).trim();
      }
      seenWorkspaceTrackIds.add(uniqueTrackId);
      const color =
        TRACK_SWATCHES[(idx + (safeReplaceExisting ? 0 : this._tracks.length)) % TRACK_SWATCHES.length];
      return new DawTrack({
        id: uniqueTrackId,
        name: typeof pt.name === 'string' ? pt.name : `Track ${idx + 1}`,
        instrument: pt.instrument as TrackInstrument,
        colorHex: color,
        notes: scaledNotes,
      });
    });

    if (safeReplaceExisting) {
      this._tracks = importedDawTracks;
      this._bpm =
        typeof rawBpm === 'number' && Number.isFinite(rawBpm) && rawBpm > 0
          ? Math.max(20, Math.min(300, Math.round(rawBpm * 10) / 10))
          : this._bpm;
      if (typeof rawTsNum === 'number' && typeof rawTsDen === 'number') {
        const normTs = normalizeWorkspaceTimeSignature(rawTsNum, rawTsDen);
        this._timeSigNumerator = normTs.numerator;
        this._timeSigDenominator = normTs.denominator;
        this._beatsPerMeasure = canonicalBeatsPerMeasure(normTs.numerator, normTs.denominator);
      } else {
        const safeRawBeats =
          typeof rawBeats === 'number' && Number.isFinite(rawBeats) ? rawBeats : 4;
        const derivedTs = deriveTimeSignatureFromBeats(safeRawBeats);
        this._timeSigNumerator = derivedTs.numerator;
        this._timeSigDenominator = derivedTs.denominator;
        this._beatsPerMeasure = canonicalBeatsPerMeasure(
          derivedTs.numerator,
          derivedTs.denominator,
        );
      }
    } else {
      for (const dt of importedDawTracks) {
        this._tracks.push(dt);
      }
    }

    this._commands.clear();
    this._activeTrackId = importedDawTracks[0].id;
    this._selectedNoteIds = EMPTY_STRING_SET;
    this._activeCandidateNoteIds = EMPTY_STRING_SET;
    this._activeCandidateRange = null;
    this._candidateOriginSlot = null;

    if (safeReplaceExisting) {
      const maxEndTick = importedDawTracks[0].getMaxEndTick();
      if (maxEndTick > 0) {
        const totalBars = Math.max(
          1,
          Math.ceil(maxEndTick / Math.max(1, Math.round(this._ppqn * this._beatsPerMeasure))),
        );
        const nextStartBar = totalBars + 1;
        this._targetRange = new MeasureRange(
          nextStartBar,
          nextStartBar + 3,
          this._ppqn,
          this._beatsPerMeasure,
        );
        this._playheadTick = 0;
      } else {
        this._targetRange = new MeasureRange(1, 4, this._ppqn, this._beatsPerMeasure);
        this._playheadTick = 0;
      }
    }

    this._candidates = [];
    this._activeCandidateIndex = 0;
    this._latestReason = null;
    this._silenceActivePlaybackVoices();
    this._invalidateAndNotify();
    const importedNotes = importedDawTracks.reduce(
      (sum, dt) => sum + dt.toSnapshot().notes.length,
      0,
    );
    return {
      importedTracks: importedDawTracks.length,
      importedNotes,
    };
  }

  public loadBuiltInDemoArrangement(): void {
    const demo = StandardMidiCodec.createDemoMidiPreset();
    this.importParsedMidiTracks(
      demo,
      demo.tracks.map((t) => t.index),
      true,
    );
    // Position target range at Bars 5..8 immediately after the 4-bar demo arrangement
    this._targetRange = new MeasureRange(5, 8, this._ppqn, this._beatsPerMeasure);
    this._invalidateAndNotify();
  }

  public exportProjectMidiBytes(): Uint8Array {
    return StandardMidiCodec.encodeMultiTrackMidi(
      this._tracks.map((t) => t.toSnapshot()),
      this._ppqn,
      this._bpm,
      this._beatsPerMeasure,
      { numerator: this._timeSigNumerator, denominator: this._timeSigDenominator },
    );
  }

  public exportWorkspaceState(): SerializedWorkspaceState {
    return {
      ppqn: this._ppqn,
      bpm: this._bpm,
      beatsPerMeasure: this._beatsPerMeasure,
      timeSigNumerator: this._timeSigNumerator,
      timeSigDenominator: this._timeSigDenominator,
      activeTrackId: this._activeTrackId,
      targetStartMeasure: this._targetRange.startMeasure,
      targetEndMeasure: this._targetRange.endMeasure,
      quantizeDivision: this._quantizeDivision,
      isTriplet: this._isTriplet,
      tracks: this._tracks.map((trk) => {
        const snap = trk.toSnapshot();
        return {
          id: snap.id,
          name: snap.name,
          instrument: snap.instrument,
          colorHex: snap.colorHex,
          isMuted: snap.isMuted,
          isSoloed: snap.isSoloed,
          notes: snap.notes.map((n) => ({
            id: n.id,
            pitch: n.pitch,
            startTick: n.startTick,
            durationTicks: n.durationTicks,
            velocity: n.velocity,
            channel: n.channel,
          })),
        };
      }),
      candidates: this._candidates.map((c) => ({
        index: c.index,
        filename: c.filename,
        rawMidiBytes: new Uint8Array(c.rawMidiBytes),
        reason: c.reason,
      })),
      activeCandidateIndex: this._activeCandidateIndex,
      activeCandidateNoteIds: Array.from(this._activeCandidateNoteIds),
      activeCandidateRange: this._activeCandidateRange ? { ...this._activeCandidateRange } : null,
      candidateOriginSlot: this._candidateOriginSlot ? { ...this._candidateOriginSlot } : null,
      latestReason: this._latestReason,
    };
  }

  public exportTrustedWorkspaceState(): SerializedWorkspaceState {
    return (
      buildTrustedWorkspaceSnapshot({
        ppqn: this._ppqn,
        bpm: this._bpm,
        beatsPerMeasure: this._beatsPerMeasure,
        timeSigNumerator: this._timeSigNumerator,
        timeSigDenominator: this._timeSigDenominator,
        activeTrackId: this._activeTrackId,
        targetStartMeasure: this._targetRange.startMeasure,
        targetEndMeasure: this._targetRange.endMeasure,
        quantizeDivision: this._quantizeDivision,
        isTriplet: this._isTriplet,
        domainTracks: this._tracks,
        candidates: this._candidates,
        activeCandidateIndex: this._activeCandidateIndex,
        activeCandidateNoteIds: this._activeCandidateNoteIds,
        activeCandidateRange: this._activeCandidateRange,
        candidateOriginSlot: this._candidateOriginSlot,
        latestReason: this._latestReason,
      }) ?? this.exportWorkspaceState()
    );
  }

  public applyWorkspaceState(state: SerializedWorkspaceState): void {
    let nextPpqn: number;
    let nextBpm: number;
    let nextTimeSigNum = 4;
    let nextTimeSigDen = 4;
    let nextBeatsPerMeasure = 4;
    let nextQuantizeDivision: 4 | 8 | 16 | 32;
    let nextIsTriplet: boolean;
    let nextActiveTrackId: string;
    let nextTargetRange: MeasureRange;
    let nextActiveCandidateIndex: number;
    let nextActiveCandidateNoteIds: Set<string>;
    let nextActiveCandidateRange: ActiveCandidateRange | null;
    let nextCandidateOriginSlot: ActiveCandidateRange | null;
    let nextLatestReason: GenerationReason | null;
    const validatedTrackSpecs: Array<{
      id: string;
      name: string;
      instrument: TrackInstrument;
      colorHex: string;
      isMuted: boolean;
      isSoloed: boolean;
      notes: MidiNote[];
      allSameAsExisting: boolean;
    }> = [];
    const reconstructedCandidates: GeneratedMidiCandidate[] = [];

    const preVerifiedInbound = consumeVerifiedInboundSyncState(state);

    try {
      const snapshotAllowedPlainObject = (
        raw: unknown,
        allowed: ReadonlySet<string>,
      ): Map<string, unknown> | null => {
        if (!raw || typeof raw !== 'object' || Array.isArray(raw) || isRuntimeProxy(raw)) {
          return null;
        }
        const proto = Object.getPrototypeOf(raw);
        if (proto !== null && proto !== Object.prototype) {
          return null;
        }
        if (Object.getOwnPropertySymbols(raw).length > 0) {
          return null;
        }
        const keys = Object.getOwnPropertyNames(raw);
        if (keys.length > allowed.size) {
          return null;
        }
        const snap = new Map<string, unknown>();
        for (let i = 0; i < keys.length; i += 1) {
          const k = keys[i];
          if (k === '__proto__' || k === 'constructor' || k === 'prototype') {
            return null;
          }
          if (!allowed.has(k) || snap.has(k)) {
            return null;
          }
          const desc = Object.getOwnPropertyDescriptor(raw, k);
          if (!desc || !('value' in desc)) {
            return null;
          }
          snap.set(k, desc.value);
        }
        return snap;
      };

      const snapshotPlainDenseArray = (raw: unknown, maxLen: number): unknown[] | null => {
        if (preVerifiedInbound && Array.isArray(raw)) {
          return raw.length <= maxLen ? raw : null;
        }
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
        if (typeof len !== 'number' || !Number.isInteger(len) || len < 0 || len > maxLen) {
          return null;
        }
        if (Object.getOwnPropertySymbols(raw).length > 0) return null;
        if (Object.getOwnPropertyNames(raw).length !== len + 1) {
          return null;
        }
        const out: unknown[] = new Array(len);
        for (let i = 0; i < len; i += 1) {
          const d = Object.getOwnPropertyDescriptor(raw, i);
          if (!d || !('value' in d)) return null;
          out[i] = d.value;
        }
        return out;
      };

      const stateSnap = snapshotAllowedPlainObject(state, ALLOWED_SYNC_WORKSPACE_KEYS);
      if (!stateSnap) {
        return;
      }

      const isFastCleanAsciiSyncId = (raw: string): boolean => {
        const sLen = raw.length;
        if (sLen === 0 || sLen > 128) return false;
        const c0 = raw.charCodeAt(0);
        const cLast = raw.charCodeAt(sLen - 1);
        if (c0 <= 0x20 || c0 >= 0x7f || cLast <= 0x20 || cLast >= 0x7f) {
          return false;
        }
        for (let i = 0; i < sLen; i += 1) {
          const c = raw.charCodeAt(i);
          if (c >= 0xd800 && c <= 0xdfff) return false;
        }
        return true;
      };

      const canonSyncId = (raw: string): string =>
        isFastCleanAsciiSyncId(raw) ? raw : truncateWellFormedUtf16(raw.trim(), 128).trim();

      const rawTracks = snapshotPlainDenseArray(stateSnap.get('tracks'), 64);
      if (!rawTracks || rawTracks.length === 0) {
        return;
      }

      const ppqnVal = stateSnap.get('ppqn');
      if (
        ppqnVal !== undefined &&
        (typeof ppqnVal !== 'number' ||
          !Number.isFinite(ppqnVal) ||
          ppqnVal < 24 ||
          ppqnVal > 32767)
      ) {
        return;
      }
      nextPpqn = ppqnVal !== undefined ? Math.round(ppqnVal) : 480;
      const maxProjectTicks = MAX_SMF_QUARTER_NOTES * nextPpqn;
      const maxSlotTick = 16_512 * nextPpqn;

      const tsNumVal = stateSnap.get('timeSigNumerator');
      const tsDenVal = stateSnap.get('timeSigDenominator');
      const bpmVal = stateSnap.get('bpm');
      const beatsVal = stateSnap.get('beatsPerMeasure');
      if (
        typeof bpmVal !== 'number' ||
        !Number.isFinite(bpmVal) ||
        bpmVal < 20 ||
        bpmVal > 300
      ) {
        return;
      }

      if ((tsNumVal === undefined) !== (tsDenVal === undefined)) {
        return;
      }
      if (tsNumVal !== undefined && tsDenVal !== undefined) {
        if (
          typeof tsNumVal !== 'number' ||
          !Number.isInteger(tsNumVal) ||
          tsNumVal < 1 ||
          tsNumVal > 32 ||
          typeof tsDenVal !== 'number' ||
          ![1, 2, 4, 8, 16, 32].includes(tsDenVal)
        ) {
          return;
        }
        nextTimeSigNum = tsNumVal;
        nextTimeSigDen = tsDenVal;
        nextBeatsPerMeasure = canonicalBeatsPerMeasure(nextTimeSigNum, nextTimeSigDen);
        if (beatsVal !== undefined) {
          if (
            typeof beatsVal !== 'number' ||
            !Number.isFinite(beatsVal) ||
            beatsVal < 0.5 ||
            beatsVal > 16 ||
            Math.abs(beatsVal - nextBeatsPerMeasure) > 1e-6
          ) {
            return;
          }
        }
      } else if (beatsVal !== undefined) {
        if (
          typeof beatsVal !== 'number' ||
          !Number.isFinite(beatsVal) ||
          beatsVal < 0.5 ||
          beatsVal > 16
        ) {
          return;
        }
        const derivedTs = deriveTimeSignatureFromBeats(beatsVal);
        const expectedBeats = canonicalBeatsPerMeasure(derivedTs.numerator, derivedTs.denominator);
        if (Math.abs(beatsVal - expectedBeats) > 1e-6) {
          return;
        }
        nextTimeSigNum = derivedTs.numerator;
        nextTimeSigDen = derivedTs.denominator;
        nextBeatsPerMeasure = expectedBeats;
      }

      // Atomically pre-validate and build ALL incoming track payloads BEFORE mutating state or clearing undo history ([R2-MED-03], [R6-MED-04], [R12-LOW-01], [R14-MED-01], [R14-LOW-01], [R25-HIGH-02])
      const rawTrimmedByCanonTrackId = new Map<string, string>();
      const rawTrimmedByCanonNoteId = preVerifiedInbound ? null : new Map<string, string>();
      let totalValidatedNoteCount = 0;
      const previewExistingById = new Map(this._tracks.map((t) => [t.id, t]));

      for (let ti = 0; ti < rawTracks.length; ti += 1) {
        const stSnap = snapshotAllowedPlainObject(rawTracks[ti], ALLOWED_SYNC_TRACK_KEYS);
        if (!stSnap) {
          return;
        }
        const idVal = stSnap.get('id');
        const nameVal = stSnap.get('name');
        const instVal = stSnap.get('instrument');
        const colorVal = stSnap.get('colorHex');
        const mutedVal = stSnap.get('isMuted');
        const soloedVal = stSnap.get('isSoloed');
        if (
          typeof idVal !== 'string' ||
          idVal.length > 128 ||
          typeof nameVal !== 'string' ||
          nameVal.trim().length === 0 ||
          nameVal.length > 256 ||
          typeof instVal !== 'string' ||
          !VALID_TRACK_INSTRUMENTS.has(instVal) ||
          typeof colorVal !== 'string' ||
          colorVal.trim().length === 0 ||
          colorVal.length > 32 ||
          typeof mutedVal !== 'boolean' ||
          typeof soloedVal !== 'boolean'
        ) {
          return;
        }
        const rawTrimmedTrackId = idVal.trim();
        const safeId = canonSyncId(idVal);
        const safeName = truncateWellFormedUtf16(nameVal.trim(), 256).trim();
        const safeColor = truncateWellFormedUtf16(colorVal.trim(), 32).trim();
        if (!safeId || !safeName || !safeColor || rawTrimmedByCanonTrackId.has(safeId)) {
          return;
        }
        const rawNotes = snapshotPlainDenseArray(
          stSnap.get('notes'),
          MAX_WORKSPACE_NOTES - totalValidatedNoteCount,
        );
        if (!rawNotes || totalValidatedNoteCount + rawNotes.length > MAX_WORKSPACE_NOTES) {
          return;
        }
        rawTrimmedByCanonTrackId.set(safeId, rawTrimmedTrackId);

        const safeInst = instVal as TrackInstrument;
        const previewExisting = previewExistingById.get(safeId);
        let allSameAsExisting =
          previewExisting !== undefined && previewExisting.noteCount === rawNotes.length;
        const domainNotes = new Array<MidiNote>(rawNotes.length);
        for (let ni = 0; ni < rawNotes.length; ni += 1) {
          const rawNote = rawNotes[ni];
          let nIdVal: unknown;
          let pitchVal: unknown;
          let startVal: unknown;
          let durVal: unknown;
          let velVal: unknown;
          let chVal: unknown;
          if (preVerifiedInbound) {
            if (!rawNote || typeof rawNote !== 'object') return;
            const fastNote = rawNote as {
              id?: unknown;
              pitch?: unknown;
              startTick?: unknown;
              durationTicks?: unknown;
              velocity?: unknown;
              channel?: unknown;
            };
            nIdVal = fastNote.id;
            pitchVal = fastNote.pitch;
            startVal = fastNote.startTick;
            durVal = fastNote.durationTicks;
            velVal = fastNote.velocity;
            chVal = fastNote.channel;
          } else {
            if (
              !rawNote ||
              typeof rawNote !== 'object' ||
              Array.isArray(rawNote) ||
              isRuntimeProxy(rawNote)
            ) {
              return;
            }
            const nProto = Object.getPrototypeOf(rawNote);
            if (nProto !== null && nProto !== Object.prototype) {
              return;
            }
            if (Object.getOwnPropertySymbols(rawNote).length > 0) {
              return;
            }
            const nKeys = Object.getOwnPropertyNames(rawNote);
            if (nKeys.length < 5 || nKeys.length > 6) {
              return;
            }
            let seenMask = 0;
            for (let ki = 0; ki < nKeys.length; ki += 1) {
              const nk = nKeys[ki];
              switch (nk) {
                case 'id': {
                  if ((seenMask & 1) !== 0) return;
                  seenMask |= 1;
                  const d = Object.getOwnPropertyDescriptor(rawNote, 'id');
                  if (!d || !('value' in d)) return;
                  nIdVal = d.value;
                  break;
                }
                case 'pitch': {
                  if ((seenMask & 2) !== 0) return;
                  seenMask |= 2;
                  const d = Object.getOwnPropertyDescriptor(rawNote, 'pitch');
                  if (!d || !('value' in d)) return;
                  pitchVal = d.value;
                  break;
                }
                case 'startTick': {
                  if ((seenMask & 4) !== 0) return;
                  seenMask |= 4;
                  const d = Object.getOwnPropertyDescriptor(rawNote, 'startTick');
                  if (!d || !('value' in d)) return;
                  startVal = d.value;
                  break;
                }
                case 'durationTicks': {
                  if ((seenMask & 8) !== 0) return;
                  seenMask |= 8;
                  const d = Object.getOwnPropertyDescriptor(rawNote, 'durationTicks');
                  if (!d || !('value' in d)) return;
                  durVal = d.value;
                  break;
                }
                case 'velocity': {
                  if ((seenMask & 16) !== 0) return;
                  seenMask |= 16;
                  const d = Object.getOwnPropertyDescriptor(rawNote, 'velocity');
                  if (!d || !('value' in d)) return;
                  velVal = d.value;
                  break;
                }
                case 'channel': {
                  if ((seenMask & 32) !== 0) return;
                  seenMask |= 32;
                  const d = Object.getOwnPropertyDescriptor(rawNote, 'channel');
                  if (!d || !('value' in d)) return;
                  chVal = d.value;
                  break;
                }
                default:
                  return;
              }
            }
            if ((seenMask & 31) !== 31) {
              return;
            }
          }
          if (
            typeof nIdVal !== 'string' ||
            nIdVal.length > 128 ||
            typeof pitchVal !== 'number' ||
            !Number.isFinite(pitchVal) ||
            pitchVal < 0 ||
            pitchVal > 127 ||
            typeof startVal !== 'number' ||
            !Number.isFinite(startVal) ||
            startVal < 0 ||
            typeof durVal !== 'number' ||
            !Number.isFinite(durVal) ||
            durVal < 1 ||
            typeof velVal !== 'number' ||
            !Number.isFinite(velVal) ||
            velVal < 1 ||
            velVal > 127 ||
            (chVal !== undefined &&
              (typeof chVal !== 'number' ||
                !Number.isFinite(chVal) ||
                chVal < 0 ||
                chVal > 15))
          ) {
            return;
          }
          const roundedStart = Math.round(startVal);
          const roundedDur = Math.round(durVal);
          if (
            roundedStart < 0 ||
            roundedDur < 1 ||
            roundedStart >= maxProjectTicks ||
            roundedStart + roundedDur > maxProjectTicks
          ) {
            return;
          }
          const fastAsciiId = isFastCleanAsciiSyncId(nIdVal);
          const rawTrimmedNoteId = fastAsciiId ? nIdVal : nIdVal.trim();
          const explicitId = fastAsciiId
            ? nIdVal
            : truncateWellFormedUtf16(rawTrimmedNoteId, 128).trim();
          if (
            !explicitId ||
            (rawTrimmedByCanonNoteId !== null && rawTrimmedByCanonNoteId.has(explicitId))
          ) {
            return;
          }
          const clampedPitch = Math.max(0, Math.min(127, Math.round(pitchVal)));
          const clampedVel = Math.max(1, Math.min(127, Math.round(velVal)));
          const clampedCh =
            typeof chVal === 'number' ? Math.max(0, Math.min(15, Math.round(chVal))) : 0;

          const prevNote = previewExisting?.getNoteAt(ni);
          let noteObj: MidiNote;
          if (
            prevNote !== undefined &&
            prevNote.id === explicitId &&
            prevNote.pitch === clampedPitch &&
            prevNote.startTick === roundedStart &&
            prevNote.durationTicks === roundedDur &&
            prevNote.velocity === clampedVel &&
            prevNote.channel === clampedCh
          ) {
            noteObj = prevNote;
          } else {
            allSameAsExisting = false;
            noteObj = createValidatedSyncMidiNote(
              explicitId,
              clampedPitch,
              roundedStart,
              roundedDur,
              clampedVel,
              clampedCh,
            );
            if (
              rawTrimmedByCanonNoteId !== null &&
              noteObj.id !== explicitId &&
              rawTrimmedByCanonNoteId.has(noteObj.id)
            ) {
              return;
            }
          }
          if (rawTrimmedByCanonNoteId !== null) {
            rawTrimmedByCanonNoteId.set(noteObj.id, rawTrimmedNoteId);
          }
          totalValidatedNoteCount += 1;
          domainNotes[ni] = noteObj;
        }

        validatedTrackSpecs.push({
          id: safeId,
          name: safeName,
          instrument: safeInst,
          colorHex: safeColor,
          isMuted: mutedVal,
          isSoloed: soloedVal,
          notes: domainNotes,
          allSameAsExisting,
        });
      }

      const activeTrackVal = stateSnap.get('activeTrackId');
      if (typeof activeTrackVal !== 'string' || activeTrackVal.length > 128) {
        return;
      }
      const canonActive = canonSyncId(activeTrackVal);
      if (
        !canonActive ||
        rawTrimmedByCanonTrackId.get(canonActive) !== activeTrackVal.trim()
      ) {
        return;
      }
      nextActiveTrackId = canonActive;

      const validateRangeSlot = (
        raw: unknown,
      ): { valid: boolean; value: ActiveCandidateRange | null } => {
        if (raw === null || raw === undefined) {
          return { valid: true, value: null };
        }
        const slotSnap = snapshotAllowedPlainObject(raw, ALLOWED_SYNC_SLOT_KEYS);
        if (!slotSnap) {
          return { valid: false, value: null };
        }
        const tIdVal = slotSnap.get('trackId');
        const sTickVal = slotSnap.get('startTick');
        const eTickVal = slotSnap.get('endTick');
        if (typeof tIdVal !== 'string' || tIdVal.length > 128) {
          return { valid: false, value: null };
        }
        const safeRangeTrackId = canonSyncId(tIdVal);
        if (
          !safeRangeTrackId ||
          rawTrimmedByCanonTrackId.get(safeRangeTrackId) !== tIdVal.trim() ||
          typeof sTickVal !== 'number' ||
          !Number.isFinite(sTickVal) ||
          sTickVal < 0 ||
          typeof eTickVal !== 'number' ||
          !Number.isFinite(eTickVal) ||
          eTickVal < sTickVal ||
          eTickVal > maxSlotTick
        ) {
          return { valid: false, value: null };
        }
        return {
          valid: true,
          value: {
            trackId: safeRangeTrackId,
            startTick: sTickVal,
            endTick: eTickVal,
          },
        };
      };

      const parsedActiveRange = validateRangeSlot(stateSnap.get('activeCandidateRange'));
      const parsedOriginSlot = validateRangeSlot(stateSnap.get('candidateOriginSlot'));
      if (
        !parsedActiveRange.valid ||
        !parsedOriginSlot.valid ||
        (parsedActiveRange.value !== null &&
          parsedOriginSlot.value !== null &&
          parsedActiveRange.value.trackId !== parsedOriginSlot.value.trackId)
      ) {
        return;
      }

      // Single-pass validation + cloning of `latestReason` (`[R13-MED-02]`, `[R14-MED-01]`)
      const parsedLatestReason = parseValidatedGenerationReason(stateSnap.get('latestReason'), true);
      if (!parsedLatestReason.valid) {
        return;
      }

      const rawCandidates = snapshotPlainDenseArray(stateSnap.get('candidates'), 32);
      if (!rawCandidates) {
        return;
      }

      const rawCandNoteIds = snapshotPlainDenseArray(
        stateSnap.get('activeCandidateNoteIds'),
        32_768,
      );
      if (!rawCandNoteIds) {
        return;
      }
      const builtCandNoteIdSet = new Set<string>();
      const seenRawTrimmedByCanonCandId = preVerifiedInbound ? null : new Map<string, string>();
      for (let i = 0; i < rawCandNoteIds.length; i += 1) {
        const id = rawCandNoteIds[i];
        if (typeof id !== 'string' || id.length > 128) {
          return;
        }
        let rawTrimmedCid: string;
        let canonCid: string;
        if (isFastCleanAsciiSyncId(id)) {
          rawTrimmedCid = id;
          canonCid = id;
        } else {
          rawTrimmedCid = id.trim();
          if (rawTrimmedCid.length === 0) return;
          canonCid = truncateWellFormedUtf16(rawTrimmedCid, 128).trim();
          if (!canonCid) return;
        }
        if (
          rawTrimmedByCanonNoteId !== null &&
          rawTrimmedByCanonNoteId.has(canonCid) &&
          rawTrimmedByCanonNoteId.get(canonCid) !== rawTrimmedCid
        ) {
          return;
        }
        if (seenRawTrimmedByCanonCandId !== null) {
          if (
            seenRawTrimmedByCanonCandId.has(canonCid) &&
            seenRawTrimmedByCanonCandId.get(canonCid) !== rawTrimmedCid
          ) {
            return;
          }
          seenRawTrimmedByCanonCandId.set(canonCid, rawTrimmedCid);
        }
        builtCandNoteIdSet.add(canonCid);
      }

      const clampedBpm = bpmVal;
      const reconstructedOriginalIndices: number[] = [];
      let totalCandidateBytes = 0;
      for (let ci = 0; ci < rawCandidates.length; ci += 1) {
        const scSnap = snapshotAllowedPlainObject(rawCandidates[ci], ALLOWED_SYNC_CANDIDATE_KEYS);
        if (!scSnap) {
          return;
        }
        const idxVal = scSnap.get('index');
        const fnameVal = scSnap.get('filename');
        const bytesVal = scSnap.get('rawMidiBytes');
        const reasonVal = scSnap.get('reason');
        if (
          typeof idxVal !== 'number' ||
          !Number.isInteger(idxVal) ||
          idxVal < 0 ||
          idxVal > 64 ||
          typeof fnameVal !== 'string' ||
          fnameVal.trim().length === 0 ||
          fnameVal.length > 256
        ) {
          return;
        }
        // Single-pass byte validation + clean copy without `.slice()` (`[R13-MED-02]`, `[R14-MED-03]`, `[R19-MED-01]`)
        let rawBytes: Uint8Array;
        if (
          preVerifiedInbound &&
          bytesVal instanceof Uint8Array &&
          bytesVal.byteLength >= 18 &&
          bytesVal.byteLength <= MAX_CANDIDATE_MIDI_BYTES - totalCandidateBytes
        ) {
          totalCandidateBytes += bytesVal.byteLength;
          rawBytes = bytesVal;
        } else {
          const extractedBytes = extractValidatedCandidateMidiBytes(
            bytesVal,
            false,
            MAX_CANDIDATE_MIDI_BYTES - totalCandidateBytes,
          );
          if (!extractedBytes) {
            return;
          }
          totalCandidateBytes += extractedBytes.accountedBytes;
          if (totalCandidateBytes > MAX_CANDIDATE_MIDI_BYTES) {
            return;
          }
          rawBytes = extractedBytes.bytes;
        }

        // Single-pass reason validation + cloning (`[R13-MED-02]`, `[R14-MED-01]`)
        const parsedCandReason = parseValidatedGenerationReason(reasonVal, true);
        if (!parsedCandReason.valid) {
          return;
        }

        // Reuse `parsedMidi` when `rawMidiBytes` is byte-identical to an existing candidate (`[R25-HIGH-02]`),
        // falling back to `StandardMidiCodec.parseMidi` + empty-tracks fallback (`[R13-MED-03]`) on new bytes.
        let cachedParsedCandMidi: ParsedMidiFile | null = null;
        if (this._candidates.length > 0) {
          const sameSlot = this._candidates[ci];
          if (
            sameSlot &&
            sameSlot.parsedMidi.tracks.length > 0 &&
            areUint8ArraysByteEqual(sameSlot.rawMidiBytes, rawBytes)
          ) {
            cachedParsedCandMidi = sameSlot.parsedMidi;
          } else {
            for (let ei = 0; ei < this._candidates.length; ei += 1) {
              if (ei === ci) continue;
              const prevCand = this._candidates[ei];
              if (
                prevCand.parsedMidi.tracks.length > 0 &&
                areUint8ArraysByteEqual(prevCand.rawMidiBytes, rawBytes)
              ) {
                cachedParsedCandMidi = prevCand.parsedMidi;
                break;
              }
            }
          }
        }
        let parsedCandMidi: ParsedMidiFile;
        if (cachedParsedCandMidi !== null) {
          parsedCandMidi = cachedParsedCandMidi;
        } else {
          try {
            parsedCandMidi = StandardMidiCodec.parseMidi(rawBytes.buffer as ArrayBuffer);
          } catch {
            parsedCandMidi = {
              format: 0,
              ppqn: nextPpqn,
              bpm: clampedBpm,
              beatsPerMeasure: nextBeatsPerMeasure,
              timeSigNumerator: nextTimeSigNum,
              timeSigDenominator: nextTimeSigDen,
              tracks: [],
            };
          }
        }

        const safeFilename = truncateWellFormedUtf16(fnameVal.trim(), 256).trim();
        reconstructedOriginalIndices.push(idxVal);
        reconstructedCandidates.push({
          index: reconstructedCandidates.length,
          filename: safeFilename || `output_${reconstructedCandidates.length}.mid`,
          rawMidiBytes: rawBytes,
          parsedMidi: parsedCandMidi,
          reason:
            parsedCandReason.value !== null
              ? (sanitizeGenerationReason(parsedCandReason.value) ?? parsedCandReason.value)
              : null,
        });
      }

      const startMeasVal = stateSnap.get('targetStartMeasure');
      const endMeasVal = stateSnap.get('targetEndMeasure');
      const quantVal = stateSnap.get('quantizeDivision');
      const tripletVal = stateSnap.get('isTriplet');
      const activeCandIdxVal = stateSnap.get('activeCandidateIndex');

      if (
        typeof startMeasVal !== 'number' ||
        !Number.isFinite(startMeasVal) ||
        startMeasVal < 1 ||
        startMeasVal > 1024 ||
        typeof endMeasVal !== 'number' ||
        !Number.isFinite(endMeasVal) ||
        endMeasVal < startMeasVal ||
        endMeasVal > 1032
      ) {
        return;
      }
      if (
        quantVal !== undefined &&
        quantVal !== 4 &&
        quantVal !== 8 &&
        quantVal !== 16 &&
        quantVal !== 32
      ) {
        return;
      }
      if (tripletVal !== undefined && typeof tripletVal !== 'boolean') {
        return;
      }
      if (
        typeof activeCandIdxVal !== 'number' ||
        !Number.isInteger(activeCandIdxVal) ||
        activeCandIdxVal < -1 ||
        activeCandIdxVal > 64
      ) {
        return;
      }

      nextBpm = clampedBpm;
      nextQuantizeDivision =
        quantVal === 4 || quantVal === 8 || quantVal === 16 || quantVal === 32 ? quantVal : 16;
      nextIsTriplet = Boolean(tripletVal);
      nextTargetRange = new MeasureRange(
        startMeasVal,
        endMeasVal,
        nextPpqn,
        nextBeatsPerMeasure,
      );

      const rawCandIdx = activeCandIdxVal;
      let mappedCandIdx = rawCandIdx;
      if (reconstructedCandidates.length > 0 && rawCandIdx >= 0) {
        const matchedPos = reconstructedOriginalIndices.indexOf(rawCandIdx);
        if (matchedPos >= 0 && matchedPos < reconstructedCandidates.length) {
          mappedCandIdx = matchedPos;
        }
      }
      nextActiveCandidateIndex =
        reconstructedCandidates.length === 0
          ? 0
          : Math.max(-1, Math.min(mappedCandIdx, reconstructedCandidates.length - 1));
      nextActiveCandidateNoteIds = builtCandNoteIdSet;
      nextActiveCandidateRange = parsedActiveRange.value;
      nextCandidateOriginSlot =
        parsedOriginSlot.value ??
        (parsedActiveRange.value ? { ...parsedActiveRange.value } : null);
      nextLatestReason =
        parsedLatestReason.value !== null
          ? (sanitizeGenerationReason(parsedLatestReason.value) ?? parsedLatestReason.value)
          : null;
    } catch {
      return;
    }

    // Commit phase: all validations succeeded; apply local snapshot values atomically with zero re-reads of `state`
    const existingById = new Map(this._tracks.map((t) => [t.id, t]));
    const hasLocalUndoHistory = this._commands.canUndo() || this._commands.canRedo();
    let preserveUndoHistory = false;
    if (
      hasLocalUndoHistory &&
      nextPpqn === this._ppqn &&
      this._candidates.length === reconstructedCandidates.length &&
      this._activeCandidateIndex === nextActiveCandidateIndex
    ) {
      const sameCandidateSlot = (
        a: ActiveCandidateRange | null | undefined,
        b: ActiveCandidateRange | null | undefined,
      ): boolean => {
        if (!a && !b) return true;
        if (!a || !b) return false;
        return (
          a.trackId === b.trackId &&
          a.startTick === b.startTick &&
          a.endTick === b.endTick
        );
      };
      let sameCandNoteIds = this._activeCandidateNoteIds.size === nextActiveCandidateNoteIds.size;
      if (sameCandNoteIds) {
        for (const id of this._activeCandidateNoteIds) {
          if (!nextActiveCandidateNoteIds.has(id)) {
            sameCandNoteIds = false;
            break;
          }
        }
      }
      preserveUndoHistory =
        sameCandNoteIds &&
        sameCandidateSlot(this._activeCandidateRange, nextActiveCandidateRange) &&
        sameCandidateSlot(this._candidateOriginSlot, nextCandidateOriginSlot) &&
        this._candidates.every((c, idx) =>
          areUint8ArraysByteEqual(c.rawMidiBytes, reconstructedCandidates[idx].rawMidiBytes),
        ) &&
        validatedTrackSpecs.every((spec) => {
          const existing = existingById.get(spec.id);
          return existing
            ? spec.allSameAsExisting && existing.noteCount === spec.notes.length
            : spec.notes.length === 0;
        });
    }

    this._ppqn = nextPpqn;
    this._bpm = nextBpm;
    this._timeSigNumerator = nextTimeSigNum;
    this._timeSigDenominator = nextTimeSigDen;
    this._beatsPerMeasure = nextBeatsPerMeasure;
    this._quantizeDivision = nextQuantizeDivision;
    this._isTriplet = nextIsTriplet;
    if (preserveUndoHistory) {
      const nextTrackIdSet = new Set(validatedTrackSpecs.map((s) => s.id));
      for (const prevTrack of this._tracks) {
        if (!nextTrackIdSet.has(prevTrack.id)) {
          this._commands.pruneTrack(prevTrack.id);
        }
      }
    } else {
      this._commands.clear();
    }

    let audibleWorkspaceChanged = validatedTrackSpecs.length !== existingById.size;
    for (const spec of validatedTrackSpecs) {
      const prevTrk = existingById.get(spec.id);
      if (
        !prevTrk ||
        prevTrk.instrument !== spec.instrument ||
        prevTrk.isMuted !== spec.isMuted ||
        prevTrk.isSoloed !== spec.isSoloed ||
        !spec.allSameAsExisting ||
        prevTrk.noteCount !== spec.notes.length
      ) {
        audibleWorkspaceChanged = true;
      }
    }

    const nextTracks: DawTrack[] = [];
    for (const spec of validatedTrackSpecs) {
      const existing = existingById.get(spec.id);
      if (existing) {
        existing.setName(spec.name);
        existing.setInstrument(spec.instrument);
        existing.setColorHex(spec.colorHex);
        existing.setMuted(spec.isMuted);
        existing.setSoloed(spec.isSoloed);
        if (!spec.allSameAsExisting || existing.noteCount !== spec.notes.length) {
          applyValidatedUniqueSyncNotes(existing, spec.notes);
        }
        nextTracks.push(existing);
      } else {
        const freshTrack = new DawTrack({
          id: spec.id,
          name: spec.name,
          instrument: spec.instrument,
          colorHex: spec.colorHex,
          isMuted: spec.isMuted,
          isSoloed: spec.isSoloed,
        });
        applyValidatedUniqueSyncNotes(freshTrack, spec.notes);
        nextTracks.push(freshTrack);
      }
    }

    this._tracks = nextTracks;
    this._activeTrackId = nextTracks.some((t) => t.id === nextActiveTrackId)
      ? nextActiveTrackId
      : nextTracks[0].id;
    this._targetRange = nextTargetRange;
    this._candidates = reconstructedCandidates;
    this._activeCandidateIndex = nextActiveCandidateIndex;
    this._activeCandidateNoteIds = nextActiveCandidateNoteIds;
    this._activeCandidateRange = nextActiveCandidateRange;
    this._candidateOriginSlot = nextCandidateOriginSlot;
    this._latestReason = nextLatestReason;

    if (this._selectedNoteIds.size > 0) {
      const validIds = new Set<string>();
      this._getActiveTrack().forEachNote((n) => {
        validIds.add(n.id);
      });
      const pruned = new Set<string>();
      for (const selId of this._selectedNoteIds) {
        if (validIds.has(selId)) {
          pruned.add(selId);
        }
      }
      this._selectedNoteIds = pruned.size > 0 ? pruned : EMPTY_STRING_SET;
    }

    if (audibleWorkspaceChanged) {
      this._silenceActivePlaybackVoices();
    }
    this._invalidateAndNotify();
  }

  // =========================================================================
  // Track & Note Editing Operations
  // =========================================================================

  private _totalNoteCount(): number {
    let total = 0;
    for (const t of this._tracks) {
      total += t.noteCount;
    }
    return total;
  }

  private _collectWorkspaceTrackCanonIds(): Set<string> {
    const out = new Set<string>();
    for (const t of this._tracks) {
      out.add(truncateWellFormedUtf16(t.id.trim(), 128).trim());
    }
    return out;
  }

  private _collectWorkspaceNoteCanonIds(): Set<string> {
    const out = new Set<string>();
    for (const t of this._tracks) {
      t.forEachNote((n) => {
        out.add(truncateWellFormedUtf16(n.id.trim(), 128).trim());
      });
    }
    return out;
  }

  private _mintUniqueWorkspaceTrackId(): string {
    const existing = this._collectWorkspaceTrackCanonIds();
    let cand = truncateWellFormedUtf16(createTrackId().trim(), 128).trim();
    while (!cand || existing.has(cand)) {
      cand = truncateWellFormedUtf16(createTrackId().trim(), 128).trim();
    }
    return cand;
  }

  private _mintUniqueWorkspaceNoteId(
    preferredId?: string,
    existingCanon?: ReadonlySet<string>,
    batchCanon?: Set<string> | ReadonlySet<string>,
    recordInBatch = true,
  ): string {
    const baseSet = existingCanon ?? this._collectWorkspaceNoteCanonIds();
    let cand =
      typeof preferredId === 'string'
        ? truncateWellFormedUtf16(preferredId.trim(), 128).trim()
        : '';
    while (!cand || baseSet.has(cand) || (batchCanon !== undefined && batchCanon.has(cand))) {
      cand = truncateWellFormedUtf16(createNoteId().trim(), 128).trim();
    }
    if (
      recordInBatch &&
      batchCanon &&
      'add' in batchCanon &&
      typeof batchCanon.add === 'function'
    ) {
      batchCanon.add(cand);
    }
    return cand;
  }

  public addTrack(name?: string, instrument: TrackInstrument = 'PIANO'): string {
    if (this._tracks.length >= MAX_WORKSPACE_TRACKS) {
      return this._activeTrackId;
    }
    const idx = this._tracks.length + 1;
    const safeInst: TrackInstrument = isValidTrackInstrument(instrument) ? instrument : 'PIANO';
    const cleanCustomName =
      typeof name === 'string' && name.trim()
        ? truncateWellFormedUtf16(name.trim(), 256).trim()
        : '';
    const fallbackLabel = safeInst === 'SAX' ? 'Saxophone' : 'Piano';
    const track = new DawTrack({
      id: this._mintUniqueWorkspaceTrackId(),
      name: cleanCustomName || `${fallbackLabel} ${idx}`,
      instrument: safeInst,
      colorHex: TRACK_SWATCHES[this._tracks.length % TRACK_SWATCHES.length],
    });
    this._tracks.push(track);
    this._activeTrackId = track.id;
    this._selectedNoteIds = EMPTY_STRING_SET;
    this._invalidateAndNotify();
    return track.id;
  }

  public deleteTrack(trackId: string): void {
    if (typeof trackId !== 'string' || this._tracks.length <= 1) return;
    const idx = this._tracks.findIndex((t) => t.id === trackId);
    if (idx < 0) return;
    this._tracks.splice(idx, 1);
    this._commands.pruneTrack(trackId);
    if (this._activeTrackId === trackId) {
      this._activeTrackId = this._tracks[Math.max(0, idx - 1)].id;
      this._selectedNoteIds = EMPTY_STRING_SET;
    }
    if (
      this._activeCandidateRange?.trackId === trackId ||
      this._candidateOriginSlot?.trackId === trackId
    ) {
      this._candidates = [];
      this._activeCandidateIndex = 0;
      this._activeCandidateNoteIds = EMPTY_STRING_SET;
      this._activeCandidateRange = null;
      this._candidateOriginSlot = null;
    }
    this._silenceActivePlaybackVoices();
    this._invalidateAndNotify();
  }

  public renameTrack(trackId: string, name: string): void {
    if (typeof trackId !== 'string' || typeof name !== 'string') return;
    const trk = this._tracks.find((t) => t.id === trackId);
    if (!trk) return;
    trk.setName(name);
    this._invalidateAndNotify();
  }

  public setTrackInstrument(trackId: string, instrument: TrackInstrument): void {
    if (typeof trackId !== 'string' || !isValidTrackInstrument(instrument)) return;
    const trk = this._tracks.find((t) => t.id === trackId);
    if (!trk || trk.instrument === instrument) return;
    trk.setInstrument(instrument);
    this._silenceActivePlaybackVoices();
    this._invalidateAndNotify();
  }

  public toggleTrackMute(trackId: string): void {
    if (typeof trackId !== 'string') return;
    const trk = this._tracks.find((t) => t.id === trackId);
    if (!trk) return;
    trk.setMuted(!trk.isMuted);
    this._silenceActivePlaybackVoices();
    this._invalidateAndNotify();
  }

  public toggleTrackSolo(trackId: string): void {
    if (typeof trackId !== 'string') return;
    const trk = this._tracks.find((t) => t.id === trackId);
    if (!trk) return;
    trk.setSoloed(!trk.isSoloed);
    this._silenceActivePlaybackVoices();
    this._invalidateAndNotify();
  }

  public setQuantize(division: 4 | 8 | 16 | 32, isTriplet = this._isTriplet): void {
    if (division !== 4 && division !== 8 && division !== 16 && division !== 32) {
      return;
    }
    if (typeof isTriplet !== 'boolean') {
      return;
    }
    this._quantizeDivision = division;
    this._isTriplet = isTriplet;
    this._invalidateAndNotify();
  }

  public getSnapStepTicks(): number {
    const safeDiv =
      this._quantizeDivision === 4 ||
      this._quantizeDivision === 8 ||
      this._quantizeDivision === 16 ||
      this._quantizeDivision === 32
        ? this._quantizeDivision
        : 16;
    const base = (this._ppqn * 4) / safeDiv;
    return Math.max(1, Math.round(this._isTriplet === true ? (base * 2) / 3 : base));
  }

  public snapTick(rawTick: number): number {
    if (typeof rawTick !== 'number' || !Number.isFinite(rawTick)) return 0;
    const maxTick = MAX_SMF_QUARTER_NOTES * this._ppqn;
    const clamped = Math.max(0, Math.min(maxTick, rawTick));
    const safeDiv =
      this._quantizeDivision === 4 ||
      this._quantizeDivision === 8 ||
      this._quantizeDivision === 16 ||
      this._quantizeDivision === 32
        ? this._quantizeDivision
        : 16;
    const exactStep = Math.max(
      1,
      (this._ppqn * 4) / (safeDiv * (this._isTriplet === true ? 1.5 : 1)),
    );
    const safeBeats =
      Number.isFinite(this._beatsPerMeasure) && this._beatsPerMeasure > 0
        ? Math.max(0.0625, Math.min(256, this._beatsPerMeasure))
        : 4;
    const measureTicks = Math.max(1, Math.round(this._ppqn * safeBeats));
    const measureStart = Math.floor(clamped / measureTicks) * measureTicks;
    const measureEnd = measureStart + measureTicks;
    const relTick = clamped - measureStart;
    const kFloor = Math.max(0, Math.floor(relTick / exactStep));
    const candLow = Math.min(measureEnd, measureStart + Math.round(kFloor * exactStep));
    const rawHigh = measureStart + Math.round((kFloor + 1) * exactStep);
    const candHigh = rawHigh < measureEnd ? rawHigh : measureEnd;
    const snapped = clamped - candLow >= candHigh - clamped ? candHigh : candLow;
    return Math.max(0, Math.min(maxTick, snapped));
  }

  public setSelectedNotes(noteIds: Iterable<string>): void {
    this._selectedNoteIds = normalizeTrackIdIterable(noteIds);
    this._invalidateAndNotify();
  }

  public selectAllNotesOnActiveTrack(): void {
    const active = this._getActiveTrack();
    this._selectedNoteIds = new Set(active.toSnapshot().notes.map((n) => n.id));
    this._invalidateAndNotify();
  }

  public auditionPitch(pitch: number): void {
    if (typeof pitch !== 'number' || !Number.isFinite(pitch)) return;
    const safePitch = Math.max(0, Math.min(127, Math.round(pitch)));
    const active = this._getActiveTrack();
    this._audioEngine.auditionPitch(safePitch, active.instrument);
  }

  public createNoteAt(
    pitch: number,
    startTick: number,
    durationTicks?: number,
    velocity = 100,
    options?: { readonly silent?: boolean },
  ): void {
    if (
      typeof pitch !== 'number' ||
      !Number.isFinite(pitch) ||
      typeof startTick !== 'number' ||
      !Number.isFinite(startTick)
    ) {
      return;
    }
    if (this._totalNoteCount() >= MAX_WORKSPACE_NOTES) return;
    const active = this._getActiveTrack();
    if (active.toSnapshot().notes.length >= MAX_NOTES_PER_TRACK) return;
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * this._ppqn;
    const snappedStart = Math.min(Math.max(0, maxProjectTicks - 1), this.snapTick(startTick));
    const rawDur =
      typeof durationTicks === 'number' && Number.isFinite(durationTicks)
        ? Math.round(durationTicks)
        : this.getSnapStepTicks();
    const dur = Math.max(1, Math.min(rawDur, maxProjectTicks - snappedStart));
    const safeVel =
      typeof velocity === 'number' && Number.isFinite(velocity)
        ? Math.max(1, Math.min(127, Math.round(velocity)))
        : 100;
    const note = new MidiNote({
      id: this._mintUniqueWorkspaceNoteId(),
      pitch: Math.max(0, Math.min(127, Math.round(pitch))),
      startTick: snappedStart,
      durationTicks: dur,
      velocity: safeVel,
    });
    const addCmd = new AddNotesCommand(active, [note], 'ノート作成');
    this._commands.execute(addCmd);
    this._selectedNoteIds = addCmd.actuallyInsertedIds;
    if (!options?.silent) {
      this._audioEngine.auditionPitch(note.pitch, active.instrument);
    }
    this._invalidateAndNotify();
  }

  public deleteSelectedNotes(): void {
    if (this._selectedNoteIds.size === 0) return;
    const active = this._getActiveTrack();
    this._commands.execute(new DeleteNotesCommand(active, this._selectedNoteIds));
    this._selectedNoteIds = EMPTY_STRING_SET;
    this._silenceActivePlaybackVoices();
    this._invalidateAndNotify();
  }

  public transformNotes(
    updates: ReadonlyArray<{
      id: string;
      pitch: number;
      startTick: number;
      durationTicks: number;
      velocity?: number;
    }>,
    description = 'ノート移動・伸縮',
  ): void {
    let safeUpdates: unknown[] = [];
    try {
      if (
        !updates ||
        typeof updates !== 'object' ||
        isRuntimeProxy(updates) ||
        !Array.isArray(updates)
      ) {
        return;
      }
      const ld = Object.getOwnPropertyDescriptor(updates, 'length');
      const len =
        ld && 'value' in ld && typeof ld.value === 'number' && Number.isFinite(ld.value)
          ? Math.min(MAX_NOTES_PER_TRACK, Math.max(0, Math.trunc(ld.value)))
          : 0;
      if (len === 0) return;
      safeUpdates = [];
      for (let i = 0; i < len; i += 1) {
        const ed = Object.getOwnPropertyDescriptor(updates, i);
        if (ed && 'value' in ed && ed.value !== undefined) {
          safeUpdates.push(ed.value);
        }
      }
    } catch {
      return;
    }
    if (safeUpdates.length === 0) return;

    const active = this._getActiveTrack();
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * this._ppqn;
    const currentById = new Map(active.toSnapshot().notes.map((n) => [n.id, n]));
    const beforeById = new Map<string, MidiNote>();
    const afterById = new Map<string, MidiNote>();

    for (const rawUpd of safeUpdates) {
      try {
        if (
          !rawUpd ||
          typeof rawUpd !== 'object' ||
          isRuntimeProxy(rawUpd) ||
          Array.isArray(rawUpd)
        ) {
          continue;
        }
        const proto = Object.getPrototypeOf(rawUpd);
        if (proto !== null && proto !== Object.prototype) continue;
        const idDesc = Object.getOwnPropertyDescriptor(rawUpd, 'id');
        const pitchDesc = Object.getOwnPropertyDescriptor(rawUpd, 'pitch');
        const startDesc = Object.getOwnPropertyDescriptor(rawUpd, 'startTick');
        const durDesc = Object.getOwnPropertyDescriptor(rawUpd, 'durationTicks');
        const velDesc = Object.getOwnPropertyDescriptor(rawUpd, 'velocity');
        if (
          !idDesc ||
          !('value' in idDesc) ||
          typeof idDesc.value !== 'string' ||
          (pitchDesc && !('value' in pitchDesc)) ||
          (startDesc && !('value' in startDesc)) ||
          (durDesc && !('value' in durDesc)) ||
          (velDesc && !('value' in velDesc))
        ) {
          continue;
        }
        const cleanId = truncateWellFormedUtf16(idDesc.value.trim(), 128).trim();
        if (!cleanId) continue;
        const existing = currentById.get(cleanId);
        if (!existing) continue;

        const rawPitch = pitchDesc ? pitchDesc.value : undefined;
        const safePitch =
          typeof rawPitch === 'number' && Number.isFinite(rawPitch)
            ? Math.max(0, Math.min(127, Math.round(rawPitch)))
            : existing.pitch;

        const rawStart = startDesc ? startDesc.value : undefined;
        const clampedStart =
          typeof rawStart === 'number' && Number.isFinite(rawStart)
            ? Math.max(0, Math.min(maxProjectTicks - 1, Math.round(rawStart)))
            : existing.startTick;

        const rawDur = durDesc ? durDesc.value : undefined;
        const baseDur =
          typeof rawDur === 'number' && Number.isFinite(rawDur)
            ? Math.round(rawDur)
            : existing.durationTicks;
        const clampedDur = Math.max(1, Math.min(baseDur, maxProjectTicks - clampedStart));

        const rawVel = velDesc ? velDesc.value : undefined;
        const safeVel =
          typeof rawVel === 'number' && Number.isFinite(rawVel)
            ? Math.max(1, Math.min(127, Math.round(rawVel)))
            : existing.velocity;

        const next = new MidiNote({
          id: existing.id,
          pitch: safePitch,
          startTick: clampedStart,
          durationTicks: clampedDur,
          velocity: safeVel,
          channel: existing.channel,
        });
        if (
          next.pitch !== existing.pitch ||
          next.startTick !== existing.startTick ||
          next.durationTicks !== existing.durationTicks ||
          next.velocity !== existing.velocity
        ) {
          beforeById.set(existing.id, existing);
          afterById.set(existing.id, next);
        } else {
          beforeById.delete(existing.id);
          afterById.delete(existing.id);
        }
      } catch {
        continue;
      }
    }

    if (afterById.size === 0) return;
    const prevCandIdsSnapshot = this._activeCandidateNoteIds;
    let touchesCandIds = false;
    if (prevCandIdsSnapshot.size > 0) {
      for (const editedId of afterById.keys()) {
        if (prevCandIdsSnapshot.has(editedId)) {
          touchesCandIds = true;
          break;
        }
      }
    }
    const nextCandIdsSnapshot: ReadonlySet<string> = touchesCandIds
      ? (() => {
          const pruned = new Set<string>();
          for (const id of prevCandIdsSnapshot) {
            if (!afterById.has(id)) {
              pruned.add(id);
            }
          }
          return pruned.size > 0 ? pruned : EMPTY_STRING_SET;
        })()
      : prevCandIdsSnapshot;

    this._commands.execute(
      new TransformNotesCommand(
        active,
        Array.from(beforeById.values()),
        Array.from(afterById.values()),
        description,
        touchesCandIds
          ? {
              onExecute: () => {
                this._activeCandidateNoteIds = nextCandIdsSnapshot;
              },
              onUndo: () => {
                this._activeCandidateNoteIds = prevCandIdsSnapshot;
              },
            }
          : undefined,
      ),
    );
    this._silenceActivePlaybackVoices();
    this._invalidateAndNotify();
  }

  public copySelectedNotes(): number {
    const active = this._getActiveTrack();
    const selected = active.toSnapshot().notes.filter((n) => this._selectedNoteIds.has(n.id));
    if (selected.length === 0) return 0;
    const minTick = selected.reduce((acc, n) => Math.min(acc, n.startTick), Infinity);
    this._clipboard = selected.map((n) => n.shiftTicks(-minTick));
    return this._clipboard.length;
  }

  public pasteNotesAtPlayhead(): number {
    if (this._clipboard.length === 0) return 0;
    const active = this._getActiveTrack();
    const trackRoom = Math.max(0, MAX_NOTES_PER_TRACK - active.toSnapshot().notes.length);
    const remainingBudget = Math.min(
      trackRoom,
      Math.max(0, MAX_WORKSPACE_NOTES - this._totalNoteCount()),
    );
    if (remainingBudget === 0) return 0;
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * this._ppqn;
    const anchor = this.snapTick(this._playheadTick);
    const existingCanon = this._collectWorkspaceNoteCanonIds();
    const batchCanon = new Set<string>();
    const pasted: MidiNote[] = [];
    for (const n of this._clipboard) {
      if (pasted.length >= remainingBudget) break;
      const cloned = n.cloneWithNewId(anchor);
      if (cloned.startTick >= maxProjectTicks) continue;
      const clampedDur = Math.max(
        1,
        Math.min(cloned.durationTicks, maxProjectTicks - cloned.startTick),
      );
      const uniqueId = this._mintUniqueWorkspaceNoteId(cloned.id, existingCanon, batchCanon);
      pasted.push(
        clampedDur === cloned.durationTicks && uniqueId === cloned.id
          ? cloned
          : new MidiNote({
              id: uniqueId,
              pitch: cloned.pitch,
              startTick: cloned.startTick,
              durationTicks: clampedDur,
              velocity: cloned.velocity,
              channel: cloned.channel,
            }),
      );
    }
    if (pasted.length === 0) return 0;
    const addCmd = new AddNotesCommand(active, pasted, 'ノート貼り付け');
    this._commands.execute(addCmd);
    this._selectedNoteIds = addCmd.actuallyInsertedIds;
    this._invalidateAndNotify();
    return addCmd.actuallyInsertedIds.size;
  }

  public undo(): void {
    if (this._commands.undo()) {
      this._clampWorkspaceNoteOverflow();
      this._selectedNoteIds = EMPTY_STRING_SET;
      this._silenceActivePlaybackVoices();
      this._invalidateAndNotify();
    }
  }

  public redo(): void {
    if (this._commands.redo()) {
      this._clampWorkspaceNoteOverflow();
      this._selectedNoteIds = EMPTY_STRING_SET;
      this._silenceActivePlaybackVoices();
      this._invalidateAndNotify();
    }
  }

  private _clampWorkspaceNoteOverflow(): void {
    let excess = this._totalNoteCount() - MAX_WORKSPACE_NOTES;
    if (excess <= 0) return;
    for (let ti = this._tracks.length - 1; ti >= 0 && excess > 0; ti -= 1) {
      const trk = this._tracks[ti];
      const snap = trk.toSnapshot();
      if (snap.notes.length === 0) continue;
      const removeCount = Math.min(excess, snap.notes.length);
      trk.setAllNotes(snap.notes.slice(0, snap.notes.length - removeCount));
      excess -= removeCount;
    }
    if (this._activeCandidateNoteIds.size > 0) {
      const aliveIds = new Set<string>();
      for (const trk of this._tracks) {
        for (const n of trk.toSnapshot().notes) {
          aliveIds.add(n.id);
        }
      }
      const nextActiveIds = new Set<string>();
      let pruned = false;
      for (const id of this._activeCandidateNoteIds) {
        if (aliveIds.has(id)) {
          nextActiveIds.add(id);
        } else {
          pruned = true;
        }
      }
      if (pruned) {
        this._activeCandidateNoteIds = nextActiveIds;
      }
    }
  }

  // =========================================================================
  // AI Generation Context Slicing & Candidate Placement (Matches Java COMPASS)
  // =========================================================================

  public async extractGenerationContext(options: {
    taskOverride?: SftTaskMode;
    conditionTrackId?: string | null;
    forceMonophonic?: boolean;
  }): Promise<ExtractedMidiContext> {
    let safeTaskOverride: SftTaskMode | undefined;
    let safeConditionTrackId: string | null = null;
    let safeForceMonophonic = false;
    try {
      if (options && typeof options === 'object' && !isRuntimeProxy(options) && !Array.isArray(options)) {
        const td = Object.getOwnPropertyDescriptor(options, 'taskOverride');
        if (
          td &&
          'value' in td &&
          (td.value === 'auto' ||
            td.value === 'meta' ||
            td.value === 'meta_past' ||
            td.value === 'meta_future' ||
            td.value === 'infill' ||
            td.value === 'inst_comp')
        ) {
          safeTaskOverride = td.value;
        }
        const cd = Object.getOwnPropertyDescriptor(options, 'conditionTrackId');
        if (cd && 'value' in cd && typeof cd.value === 'string' && cd.value.trim()) {
          safeConditionTrackId = cd.value;
        }
        const fd = Object.getOwnPropertyDescriptor(options, 'forceMonophonic');
        if (fd && 'value' in fd && typeof fd.value === 'boolean') {
          safeForceMonophonic = fd.value;
        }
      }
    } catch {
      // Fallback to safe defaults if options is hostile
    }

    const activeTrack = this._getActiveTrack();
    const targetMortmInstrument: 'PIANO' | 'SAX' =
      safeForceMonophonic || activeTrack.isMonophonic() ? 'SAX' : 'PIANO';

    const pastWin = this._targetRange.pastWindowTicks(8);
    const futureWin = this._targetRange.futureWindowTicks(8);
    const excludedCandidateIds = this._isSameRangeAsActiveCandidate(activeTrack.id)
      ? this._activeCandidateNoteIds
      : undefined;

    const pastNotes = this._sliceOverlappingWindowNotes(
      activeTrack,
      pastWin.startTick,
      pastWin.endTick,
      excludedCandidateIds,
    );
    const futureNotes = this._sliceOverlappingWindowNotes(
      activeTrack,
      futureWin.startTick,
      futureWin.endTick,
      excludedCandidateIds,
    );

    const validExplicitCondTrack = safeConditionTrackId
      ? this._tracks.find((t) => t.id === safeConditionTrackId && t.id !== activeTrack.id)
      : undefined;
    let condTrack: DawTrack | undefined = validExplicitCondTrack;
    if (!condTrack && safeTaskOverride === 'inst_comp') {
      condTrack = this._tracks.find(
        (t) =>
          t.id !== activeTrack.id &&
          this._sliceOverlappingWindowNotes(
            t,
            this._targetRange.startTick,
            this._targetRange.endTick,
          ).length > 0,
      );
    }

    const condNotes = condTrack
      ? this._sliceOverlappingWindowNotes(
          condTrack,
          this._targetRange.startTick,
          this._targetRange.endTick,
        )
      : [];

    const hasExplicitCondSelection = validExplicitCondTrack !== undefined;
    const resolvedTask = resolveEffectiveSftTask({
      requestedTask: safeTaskOverride ?? 'auto',
      hasPastContext: pastNotes.length > 0,
      hasFutureContext: futureNotes.length > 0,
      hasExplicitConditionTrackSelection: hasExplicitCondSelection,
      hasSelectedConditionContext: hasExplicitCondSelection && condNotes.length > 0,
      hasAnyConditionContext: condNotes.length > 0 && condTrack !== undefined,
    });

    const sendPast =
      (resolvedTask === 'meta_past' || resolvedTask === 'infill') && pastNotes.length > 0;
    const sendFuture =
      (resolvedTask === 'meta_future' || resolvedTask === 'infill') && futureNotes.length > 0;
    const sendCond =
      resolvedTask === 'inst_comp' && condNotes.length > 0 && condTrack !== undefined;

    const timeSig = {
      numerator: this._timeSigNumerator,
      denominator: this._timeSigDenominator,
    };

    const pastMidiBytes = sendPast
      ? StandardMidiCodec.encodeSingleTrackMidi(
          pastNotes,
          this._ppqn,
          120,
          targetMortmInstrument,
          this._beatsPerMeasure,
          timeSig,
        )
      : null;

    const futureMidiBytes = sendFuture
      ? StandardMidiCodec.encodeSingleTrackMidi(
          futureNotes,
          this._ppqn,
          120,
          targetMortmInstrument,
          this._beatsPerMeasure,
          timeSig,
        )
      : null;

    const conditionsMidiBytes =
      sendCond && condTrack
        ? StandardMidiCodec.encodeConditionsMidi(
            condNotes,
            this._ppqn,
            120,
            condTrack.isMonophonic(),
            this._beatsPerMeasure,
            timeSig,
          )
        : null;

    return {
      resolvedTask,
      targetMortmInstrument,
      targetMeasureCount: this._targetRange.clampedSftMeasureCount,
      pastMidiBytes,
      futureMidiBytes,
      conditionsMidiBytes,
      targetSlot: Object.freeze({
        trackId: activeTrack.id,
        startTick: this._targetRange.startTick,
        endTick: this._targetRange.endTick,
      }),
    };
  }

  public async applyGeneratedCandidates(
    candidates: readonly GeneratedMidiCandidate[],
    reasons: readonly GenerationReason[],
    targetSlot?: ActiveCandidateRange,
  ): Promise<void> {
    try {
      const hasExplicitSlot = targetSlot !== undefined;
      const safeExplicitSlot = hasExplicitSlot
        ? this._normalizeExplicitTargetSlot(targetSlot)
        : null;
      if (hasExplicitSlot && !safeExplicitSlot) return;

      const normalizedCandidates = normalizeGeneratedCandidatesBatch(candidates, reasons, false);
      if (normalizedCandidates.length === 0) return;
      // Snapshot prior candidate tuple inside `_insertCandidateIntoActiveTrack` BEFORE mutating `_candidates`
      // so both same-range overhang cleanup (`[R2-HIGH-04]`) and cross-range Undo (`[R3-MED-07]`) preserve full fidelity.
      const inserted = this._insertCandidateIntoActiveTrack(
        normalizedCandidates[0],
        normalizedCandidates[0].reason,
        'new_generation',
        normalizedCandidates,
        safeExplicitSlot,
      );
      if (!inserted) return;
      this._playheadTick = safeExplicitSlot
        ? safeExplicitSlot.startTick
        : this._targetRange.startTick;
      this._silenceActivePlaybackVoices();
      this._invalidateAndNotify();
    } catch {
      // Never propagate malformed candidate batch exception
    }
  }

  public async selectGeneratedCandidate(index: number): Promise<void> {
    try {
      if (typeof index !== 'number' || !Number.isInteger(index) || index < 0) return;
      const chosen =
        this._candidates.find((c) => c.index === index) ??
        (index < this._candidates.length ? this._candidates[index] : undefined);
      if (!chosen) return;
      const slot = this._activeCandidateRange ?? this._candidateOriginSlot;
      if (!slot || !this._tracks.some((t) => t.id === slot.trackId)) {
        this._candidates = [];
        this._activeCandidateIndex = 0;
        this._activeCandidateNoteIds = EMPTY_STRING_SET;
        this._activeCandidateRange = null;
        this._candidateOriginSlot = null;
        this._invalidateAndNotify();
        return;
      }
      const inserted = this._insertCandidateIntoActiveTrack(
        chosen,
        chosen.reason,
        'switch_candidate',
      );
      if (!inserted) return;
      this._playheadTick = slot.startTick;
      this._silenceActivePlaybackVoices();
      this._invalidateAndNotify();
    } catch {
      // Never propagate candidate switch exception
    }
  }

  // =========================================================================
  // Internal Helpers
  // =========================================================================

  private _isSameRangeAsActiveCandidate(trackId: string): boolean {
    return (
      this._activeCandidateRange !== null &&
      this._activeCandidateRange.trackId === trackId &&
      this._activeCandidateRange.startTick === this._targetRange.startTick &&
      this._activeCandidateRange.endTick === this._targetRange.endTick
    );
  }

  private _sliceOverlappingWindowNotes(
    track: DawTrack,
    winStartTick: number,
    winEndTick: number,
    excludeNoteIds?: ReadonlySet<string>,
  ): MidiNote[] {
    if (winEndTick <= winStartTick) return [];
    const overlapping = track.queryNotesOverlapping(winStartTick, winEndTick);
    const sliced: MidiNote[] = [];
    for (const note of overlapping) {
      if (excludeNoteIds && excludeNoteIds.has(note.id)) {
        continue;
      }
      const clampedStart = Math.max(winStartTick, note.startTick);
      const clampedEnd = Math.min(winEndTick, note.endTick);
      if (clampedEnd > clampedStart) {
        sliced.push(
          new MidiNote({
            pitch: note.pitch,
            startTick: clampedStart - winStartTick,
            durationTicks: clampedEnd - clampedStart,
            velocity: note.velocity,
            channel: note.channel,
          }),
        );
      }
    }
    return sliced;
  }

  private _normalizeExplicitTargetSlot(raw: unknown): ActiveCandidateRange | null {
    try {
      if (!raw || typeof raw !== 'object' || isRuntimeProxy(raw) || Array.isArray(raw)) {
        return null;
      }
      const td = Object.getOwnPropertyDescriptor(raw, 'trackId');
      const sd = Object.getOwnPropertyDescriptor(raw, 'startTick');
      const ed = Object.getOwnPropertyDescriptor(raw, 'endTick');
      if (
        !td ||
        !('value' in td) ||
        typeof td.value !== 'string' ||
        !td.value.trim() ||
        !sd ||
        !('value' in sd) ||
        typeof sd.value !== 'number' ||
        !Number.isFinite(sd.value) ||
        !ed ||
        !('value' in ed) ||
        typeof ed.value !== 'number' ||
        !Number.isFinite(ed.value)
      ) {
        return null;
      }
      const maxProjectTicks = MAX_SMF_QUARTER_NOTES * this._ppqn;
      const startTick = Math.min(maxProjectTicks, Math.max(0, Math.round(sd.value)));
      const endTick = Math.min(maxProjectTicks, Math.max(0, Math.round(ed.value)));
      if (endTick <= startTick) {
        return null;
      }
      return {
        trackId: td.value.trim(),
        startTick,
        endTick,
      };
    } catch {
      return null;
    }
  }

  private _insertCandidateIntoActiveTrack(
    candidate: GeneratedMidiCandidate,
    nextReason: GenerationReason | null,
    mode: 'new_generation' | 'switch_candidate' = 'new_generation',
    incomingCandidates?: readonly GeneratedMidiCandidate[],
    explicitTargetSlot?: ActiveCandidateRange | null,
  ): boolean {
    const pinnedRange =
      mode === 'switch_candidate'
        ? (this._activeCandidateRange ?? this._candidateOriginSlot)
        : (explicitTargetSlot ?? null);
    if (mode === 'switch_candidate' && !pinnedRange) {
      return false;
    }
    const targetTrack = pinnedRange
      ? this._tracks.find((t) => t.id === pinnedRange.trackId)
      : this._getActiveTrack();
    if (!targetTrack) {
      return false;
    }
    const placementStartTick = pinnedRange ? pinnedRange.startTick : this._targetRange.startTick;
    const placementEndTick = pinnedRange ? pinnedRange.endTick : this._targetRange.endTick;

    // Capture complete prior candidate tuple BEFORE mutating `_candidates` or `_candidateOriginSlot` (`[R3-MED-07]`)
    const prevCandidatesSnapshot = [...this._candidates];
    const prevCandidateIndexSnapshot = this._activeCandidateIndex;
    const prevAllCandidateIdsSnapshot: ReadonlySet<string> = this._activeCandidateNoteIds;
    const prevFullCandidateRangeSnapshot = this._activeCandidateRange
      ? { ...this._activeCandidateRange }
      : null;
    const prevOriginSlotSnapshot = this._candidateOriginSlot
      ? { ...this._candidateOriginSlot }
      : null;
    const prevReason = this._latestReason;
    const sanitizedNextReason = sanitizeGenerationReason(nextReason);
    const effectiveNextCandidates: readonly GeneratedMidiCandidate[] =
      mode === 'new_generation'
        ? (incomingCandidates ?? normalizeGeneratedCandidatesBatch([candidate], undefined, true))
        : this._candidates;
    if (effectiveNextCandidates.length === 0) {
      return false;
    }
    const resolvedCandIdx = effectiveNextCandidates.findIndex(
      (ec) => ec.index === candidate.index,
    );
    const effectiveActiveCandidateIndex =
      resolvedCandIdx >= 0
        ? resolvedCandIdx
        : Math.max(0, Math.min(candidate.index, Math.max(0, effectiveNextCandidates.length - 1)));

    const parsed = candidate.parsedMidi;
    const targetInst = targetTrack.toMortmInstrument();
    const matchingTracks =
      parsed.tracks.length > 1
        ? parsed.tracks.filter((t) => (t.instrument === 'SAX' ? 'SAX' : 'PIANO') === targetInst)
        : parsed.tracks;
    const sourceTracks = matchingTracks.length > 0 ? matchingTracks : parsed.tracks;

    const sameRangeAsPrevCandidate =
      this._activeCandidateRange !== null &&
      this._activeCandidateRange.trackId === targetTrack.id &&
      this._activeCandidateRange.startTick === placementStartTick &&
      this._activeCandidateRange.endTick === placementEndTick;
    const replaceablePrevIds: ReadonlySet<string> = sameRangeAsPrevCandidate
      ? prevAllCandidateIdsSnapshot
      : EMPTY_STRING_SET;

    const isReplaceablePrev = createFastCanonicalIdHas(replaceablePrevIds);

    const hasFutureUserNotes = targetTrack
      .queryNotesOverlapping(
        placementEndTick,
        Math.max(placementEndTick + 1, targetTrack.getMaxEndTick() + 1),
      )
      .some((n) => !isReplaceablePrev(n.id));

    const existingWorkspaceCanon = new Set<string>();
    let willRemoveCount = 0;
    targetTrack.forEachNote((existing) => {
      if (
        isReplaceablePrev(existing.id) ||
        (existing.startTick >= placementStartTick && existing.startTick < placementEndTick)
      ) {
        willRemoveCount += 1;
      } else {
        existingWorkspaceCanon.add(existing.id);
      }
    });
    for (let ti = 0; ti < this._tracks.length; ti += 1) {
      const t = this._tracks[ti];
      if (t.id === targetTrack.id) continue;
      t.forEachNote((n) => {
        existingWorkspaceCanon.add(n.id);
      });
    }
    const trackRemainingAfterRemoval = Math.max(
      0,
      MAX_NOTES_PER_TRACK - Math.max(0, targetTrack.noteCount - willRemoveCount),
    );
    const workspaceRemainingAfterRemoval = Math.max(
      0,
      MAX_WORKSPACE_NOTES - Math.max(0, this._totalNoteCount() - willRemoveCount),
    );
    const maxInsertableNotes = Math.min(trackRemainingAfterRemoval, workspaceRemainingAfterRemoval);
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * this._ppqn;
    const effectiveMaxEndTick = hasFutureUserNotes
      ? Math.min(placementEndTick, maxProjectTicks)
      : maxProjectTicks;

    const scale = this._ppqn / Math.max(1, parsed.ppqn);
    const preparedBatch = prepareScaledCandidateBatch({
      sourceTracks,
      scale,
      placementStartTick,
      effectiveMaxEndTick,
      maxInsertableNotes,
      existingIds: existingWorkspaceCanon,
      mintFallbackId: (prefId, existingIds, batchIds) =>
        this._mintUniqueWorkspaceNoteId(prefId, existingIds, batchIds, false),
    });

    const nextRange: ActiveCandidateRange = {
      trackId: targetTrack.id,
      startTick: placementStartTick,
      endTick: placementEndTick,
    };

    const replaceCmd = new ReplaceRangeNotesCommand(
      targetTrack,
      placementStartTick,
      placementEndTick,
      preparedBatch,
      `AI生成フレーズ配置 (候補 ${effectiveActiveCandidateIndex + 1})`,
      replaceablePrevIds,
      {
        onExecute: (insertedIds) => {
          if (!this._tracks.some((t) => t.id === targetTrack.id)) {
            this._candidates = [];
            this._activeCandidateIndex = 0;
            this._activeCandidateNoteIds = EMPTY_STRING_SET;
            this._activeCandidateRange = null;
            this._candidateOriginSlot = null;
            return;
          }
          this._candidates = [...effectiveNextCandidates];
          this._activeCandidateIndex = effectiveActiveCandidateIndex;
          this._activeCandidateNoteIds = insertedIds;
          this._activeCandidateRange = { ...nextRange };
          if (
            mode === 'new_generation' ||
            !this._candidateOriginSlot ||
            this._candidateOriginSlot.trackId !== nextRange.trackId ||
            !this._tracks.some((t) => t.id === this._candidateOriginSlot!.trackId)
          ) {
            this._candidateOriginSlot = { ...nextRange };
          }
          this._latestReason = sanitizedNextReason;
        },
        onUndo: () => {
          if (!this._tracks.some((t) => t.id === targetTrack.id)) {
            this._candidates = [];
            this._activeCandidateIndex = 0;
            this._activeCandidateNoteIds = EMPTY_STRING_SET;
            this._activeCandidateRange = null;
            this._candidateOriginSlot = null;
            return;
          }
          if (mode === 'new_generation' && prevCandidatesSnapshot.length === 0) {
            // First generation undone when no prior run existed: keep Run 1 candidate buttons selectable
            // on their pinned `_candidateOriginSlot` while marking `activeCandidateIndex = -1` (`[R4-MED-03]`).
            this._candidates = [...effectiveNextCandidates];
            this._activeCandidateIndex = -1;
            this._activeCandidateNoteIds = EMPTY_STRING_SET;
            this._activeCandidateRange = null;
            this._candidateOriginSlot = { ...nextRange };
            this._latestReason = null;
            return;
          }
          const activeSlotAlive =
            !prevFullCandidateRangeSnapshot ||
            this._tracks.some((t) => t.id === prevFullCandidateRangeSnapshot.trackId);
          const originSlotAlive =
            !prevOriginSlotSnapshot ||
            this._tracks.some((t) => t.id === prevOriginSlotSnapshot.trackId);
          const slotsConsistent =
            !prevFullCandidateRangeSnapshot ||
            !prevOriginSlotSnapshot ||
            prevFullCandidateRangeSnapshot.trackId === prevOriginSlotSnapshot.trackId;
          if (!activeSlotAlive || !originSlotAlive || !slotsConsistent) {
            this._candidates = [];
            this._activeCandidateIndex = 0;
            this._activeCandidateNoteIds = EMPTY_STRING_SET;
            this._activeCandidateRange = null;
            this._candidateOriginSlot = null;
            this._latestReason = null;
            return;
          }
          // Restore prior candidate run or prior candidate selection across both same-range and cross-range Undo (`[R3-MED-07]`)
          this._candidates = [...prevCandidatesSnapshot];
          this._activeCandidateIndex = prevCandidateIndexSnapshot;
          this._activeCandidateNoteIds = prevAllCandidateIdsSnapshot;
          this._activeCandidateRange = prevFullCandidateRangeSnapshot
            ? { ...prevFullCandidateRangeSnapshot }
            : null;
          this._candidateOriginSlot = prevOriginSlotSnapshot
            ? { ...prevOriginSlotSnapshot }
            : null;
          this._latestReason = prevReason;
        },
      },
    );
    this._commands.execute(replaceCmd);
    this._activeTrackId = targetTrack.id;
    this._selectedNoteIds = replaceCmd.actuallyInsertedIds;
    return true;
  }

  private _computeContextAvailability(): ContextAvailability {
    const activeTrack = this._getActiveTrack();
    const pastWin = this._targetRange.pastWindowTicks(8);
    const futureWin = this._targetRange.futureWindowTicks(8);
    const sameRange = this._isSameRangeAsActiveCandidate(activeTrack.id);
    const isActiveCandNote = sameRange
      ? createFastCanonicalIdHas(this._activeCandidateNoteIds)
      : null;

    const pastCount =
      pastWin.endTick > pastWin.startTick
        ? activeTrack
            .queryNotesOverlapping(pastWin.startTick, pastWin.endTick)
            .filter(
              (n) =>
                (!isActiveCandNote || !isActiveCandNote(n.id)) &&
                n.endTick > pastWin.startTick &&
                n.startTick < pastWin.endTick,
            ).length
        : 0;
    const futureCount =
      futureWin.endTick > futureWin.startTick
        ? activeTrack
            .queryNotesOverlapping(futureWin.startTick, futureWin.endTick)
            .filter(
              (n) =>
                (!isActiveCandNote || !isActiveCandNote(n.id)) &&
                n.endTick > futureWin.startTick &&
                n.startTick < futureWin.endTick,
            ).length
        : 0;

    const targetStart = this._targetRange.startTick;
    const targetEnd = this._targetRange.endTick;
    const conditionCandidateTracks = this._tracks
      .filter((t) => t.id !== activeTrack.id)
      .map((t) => ({
        id: t.id,
        name: t.name,
        instrument: t.instrument,
        noteCount:
          targetEnd > targetStart
            ? t
                .queryNotesOverlapping(targetStart, targetEnd)
                .filter((n) => n.endTick > targetStart && n.startTick < targetEnd).length
            : 0,
      }))
      .filter((item) => item.noteCount > 0);

    let autoResolvedTask: 'meta' | 'meta_past' | 'meta_future' | 'infill' | 'inst_comp' = 'meta';
    if (pastCount > 0 && futureCount > 0) {
      autoResolvedTask = 'infill';
    } else if (pastCount > 0) {
      autoResolvedTask = 'meta_past';
    } else if (futureCount > 0) {
      autoResolvedTask = 'meta_future';
    }

    return {
      hasPastContext: pastCount > 0,
      pastNoteCount: pastCount,
      hasFutureContext: futureCount > 0,
      futureNoteCount: futureCount,
      conditionCandidateTracks,
      autoResolvedTask,
    };
  }

  private _advanceTransport(): void {
    if (!this._isPlaying) return;
    const nowAudio = this._audioEngine.getCurrentTime();
    const dtSec = Math.max(0, Math.min(0.25, nowAudio - this._lastAudioTime));
    this._lastAudioTime = nowAudio;

    const ticksPerSec = (this._bpm / 60) * this._ppqn;
    this._playheadTick += dtSec * ticksPerSec;

    const lookaheadSec = 0.1;
    const horizonTick = this._playheadTick + lookaheadSec * ticksPerSec;
    const windowStartTick = Math.max(
      this._scheduledUpToTick,
      this._playheadTick - dtSec * ticksPerSec,
    );

    if (horizonTick > windowStartTick) {
      const hasAnySolo = this._tracks.some((t) => t.isSoloed);
      for (const trk of this._tracks) {
        const audible = hasAnySolo ? trk.isSoloed && !trk.isMuted : !trk.isMuted;
        if (!audible) continue;

        const notesInLookahead = trk.queryNotesStartingIn(windowStartTick, horizonTick);
        for (const note of notesInLookahead) {
          const noteOffsetSec = Math.max(0, (note.startTick - this._playheadTick) / ticksPerSec);
          const noteDurSec = Math.max(0.05, note.durationTicks / ticksPerSec);
          this._audioEngine.triggerNote(
            note.pitch,
            note.velocity,
            noteDurSec,
            trk.instrument,
            nowAudio + noteOffsetSec,
          );
        }
      }
      this._scheduledUpToTick = horizonTick;
    }

    // O(1) per track via cached interval-tree root maxEndTick
    const maxNoteTick = this._tracks.reduce((acc, t) => Math.max(acc, t.getMaxEndTick()), 0);
    const stopLimitTick = Math.max(
      maxNoteTick + this._ppqn * 4,
      this._targetRange.endTick + this._ppqn * 8,
    );
    if (this._playheadTick >= stopLimitTick) {
      this.stop();
      return;
    }

    this._invalidateAndNotify();
  }

  private _getActiveTrack(): DawTrack {
    const found = this._tracks.find((t) => t.id === this._activeTrackId);
    if (found) return found;
    this._activeTrackId = this._tracks[0].id;
    return this._tracks[0];
  }

  private _invalidateAndNotify(): void {
    this._cachedSnapshot = null;
    if (this._listeners.size === 0) {
      return;
    }
    const snap = this.getSnapshot();
    for (const listener of Array.from(this._listeners)) {
      try {
        listener(snap);
      } catch {
        // Isolate faulty subscriber so sibling UI listeners still update
      }
    }
  }
}
