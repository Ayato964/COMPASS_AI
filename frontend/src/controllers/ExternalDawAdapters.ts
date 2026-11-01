/**
 * Pluggable External DAW Controller Adapters (`Cubase` & `Ableton Live` — `oop-extensible-craft`).
 * Strictly satisfies Liskov Substitution (`LSP`) and Open/Closed (`OCP`) interchangeability
 * with `DemoPianoRollController` inside `DawControllerRegistry`:
 * - Reflects actual standby/connected transport capabilities accurately
 * - Extracts real proxy-track context MIDI when proxy notes exist, and resolves missing-context
 *   overrides cleanly to `'meta'` so `resolvedTask` and MIDI byte payloads are always consistent
 */

import { ReplaceRangeNotesCommand } from '../domain/commands/NoteCommands';
import {
  DawTrack,
  EMPTY_CANONICAL_ID_SET,
  prepareScaledCandidateBatch,
} from '../domain/DawTrack';
import { canonicalBeatsPerMeasure, MeasureRange } from '../domain/MeasureRange';
import { createNoteId, MidiNote, truncateWellFormedUtf16 } from '../domain/MidiNote';
import { MAX_SMF_QUARTER_NOTES, StandardMidiCodec } from '../domain/midi/StandardMidiCodec';
import {
  GeneratedMidiCandidate,
  GenerationReason,
  isRuntimeProxy,
  normalizeGeneratedCandidatesBatch,
  sanitizeGenerationReason,
} from '../services/CompassApiClient';

const MAX_WORKSPACE_NOTES = 32_768;
import {
  ActiveCandidateRange,
  ContextAvailability,
  DawCapabilities,
  DawSessionSnapshot,
  ExtractedMidiContext,
  IDawController,
  resolveEffectiveSftTask,
  SftTaskMode,
} from './IDawController';

export interface ExternalDawAdapterOptions {
  readonly bridgeBaseUrl?: string;
}

function resolveBridgeUrl(
  options: string | ExternalDawAdapterOptions | undefined,
  adapterEnvKey: string,
  defaultUrl: string,
): string {
  let explicit = '';
  try {
    if (typeof options === 'string') {
      explicit = options.trim();
    } else if (
      options &&
      typeof options === 'object' &&
      !Array.isArray(options) &&
      !isRuntimeProxy(options)
    ) {
      const optProto = Object.getPrototypeOf(options);
      if (optProto === null || optProto === Object.prototype) {
        const urlDesc = Object.getOwnPropertyDescriptor(options, 'bridgeBaseUrl');
        if (urlDesc && 'value' in urlDesc && typeof urlDesc.value === 'string') {
          explicit = urlDesc.value.trim();
        }
      }
    }
  } catch {
    explicit = '';
  }
  if (explicit) {
    return explicit.replace(/\/+$/, '');
  }
  try {
    const metaEnv = (import.meta as unknown as { env?: Record<string, unknown> }).env;
    if (metaEnv && typeof metaEnv === 'object' && !isRuntimeProxy(metaEnv)) {
      const adapterDesc = Object.getOwnPropertyDescriptor(metaEnv, adapterEnvKey);
      const fromAdapterEnv =
        adapterDesc && 'value' in adapterDesc && typeof adapterDesc.value === 'string'
          ? adapterDesc.value.trim()
          : '';
      if (fromAdapterEnv) {
        return fromAdapterEnv.replace(/\/+$/, '');
      }
      const sharedDesc = Object.getOwnPropertyDescriptor(metaEnv, 'VITE_COMPASS_DAW_BRIDGE_URL');
      const fromSharedEnv =
        sharedDesc && 'value' in sharedDesc && typeof sharedDesc.value === 'string'
          ? sharedDesc.value.trim()
          : '';
      if (fromSharedEnv) {
        return fromSharedEnv.replace(/\/+$/, '');
      }
    }
  } catch {
    // Fall back to defaultUrl
  }
  return defaultUrl;
}

abstract class BaseExternalDawBridgeController implements IDawController {
  public abstract readonly id: string;
  public abstract readonly displayName: string;
  public readonly bridgeBaseUrl: string;
  protected abstract readonly bridgeEndpointLabel: string;
  protected abstract readonly protocolName: string;

  protected constructor(bridgeBaseUrl: string) {
    this.bridgeBaseUrl = bridgeBaseUrl;
  }

  protected get bridgeEndpoint(): string {
    return `${this.bridgeEndpointLabel} (${this.bridgeBaseUrl})`;
  }

  private _isConnected = false;
  private _statusText = 'DAWブリッジ待機中 (スタンドアロン補完モード)';
  private _bpm = 120;
  private _beatsPerMeasure = 4;
  private _timeSigNumerator = 4;
  private _timeSigDenominator = 4;
  private _playheadTick = 0;
  private _targetRange = new MeasureRange(1, 4, 480, 4);
  private readonly _proxyTrack = new DawTrack({
    name: 'Host Instrument Track 1',
    instrument: 'PIANO',
  });
  private _candidates: GeneratedMidiCandidate[] = [];
  private _activeCandidateIndex = 0;
  private _activeCandidateNoteIds: ReadonlySet<string> = EMPTY_CANONICAL_ID_SET;
  private _activeCandidateRange: { startTick: number; endTick: number } | null = null;
  private _latestReason: GenerationReason | null = null;
  private readonly _listeners = new Set<(snapshot: DawSessionSnapshot) => void>();

  public get capabilities(): DawCapabilities {
    return Object.freeze({
      hasEmbeddedPianoRoll: false,
      supportsMultiTrack: this._isConnected,
      supportsTransportControl: this._isConnected,
      supportsDirectNoteEditing: false,
      connectionProtocol: this.protocolName,
    });
  }

  public async connect(): Promise<void> {
    this._isConnected = false;
    this._statusText = `${this.bridgeEndpoint} ブリッジ待機中 (生成MIDI書き出し利用可能)`;
    this._notify();
  }

  public async disconnect(): Promise<void> {
    this._isConnected = false;
    this._notify();
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
    const contextAvailability = Object.freeze(this._computeProxyContextAvailability());
    return Object.freeze({
      controllerId: this.id,
      controllerName: this.displayName,
      isConnected: this._isConnected,
      connectionStatusText: this._statusText,
      capabilities: this.capabilities,
      ppqn: 480,
      bpm: this._bpm,
      beatsPerMeasure: this._beatsPerMeasure,
      timeSigNumerator: this._timeSigNumerator,
      timeSigDenominator: this._timeSigDenominator,
      isPlaying: false,
      playheadTick: this._playheadTick,
      tracks: Object.freeze([this._proxyTrack.toSnapshot()]),
      activeTrackId: this._proxyTrack.id,
      targetRange: this._targetRange,
      quantizeDivision: 16,
      isTriplet: false,
      selectedNoteIds: new Set<string>(),
      canUndo: false,
      canRedo: false,
      undoDescription: null,
      candidates: Object.freeze([...this._candidates]),
      activeCandidateIndex: this._activeCandidateIndex,
      activeCandidateRange: this._activeCandidateRange
        ? Object.freeze({
            trackId: this._proxyTrack.id,
            startTick: this._activeCandidateRange.startTick,
            endTick: this._activeCandidateRange.endTick,
          })
        : null,
      latestReason: this._latestReason,
      contextAvailability,
    });
  }

  public async play(): Promise<void> {}
  public pause(): void {}
  public stop(): void {
    this._playheadTick = this._targetRange.startTick;
    this._notify();
  }

  public setPlayheadTick(tick: number): void {
    if (!Number.isFinite(tick)) return;
    this._playheadTick = Math.max(0, Math.round(tick));
    this._notify();
  }

  public setBpm(bpm: number): void {
    if (!Number.isFinite(bpm)) return;
    this._bpm = Math.max(20, Math.min(300, Math.round(bpm * 10) / 10));
    this._notify();
  }

  public setTimeSignature(numerator: number, denominator: number): void {
    if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) return;
    const num = Math.max(1, Math.min(32, Math.round(numerator)));
    const den = Math.round(denominator);
    if (![1, 2, 4, 8, 16, 32].includes(den)) return;
    this._timeSigNumerator = num;
    this._timeSigDenominator = den;
    this._beatsPerMeasure = canonicalBeatsPerMeasure(num, den);
    this._targetRange = new MeasureRange(
      this._targetRange.startMeasure,
      this._targetRange.endMeasure,
      480,
      this._beatsPerMeasure,
    );
    this._notify();
  }

  public setTargetMeasureRange(startMeasure: number, endMeasure: number): void {
    if (!Number.isFinite(startMeasure) || !Number.isFinite(endMeasure)) return;
    this._targetRange = new MeasureRange(startMeasure, endMeasure, 480, this._beatsPerMeasure);
    this._notify();
  }

  public setActiveTrack(_trackId: string): void {}

  public async extractGenerationContext(options: {
    taskOverride?: SftTaskMode;
    conditionTrackId?: string | null;
    forceMonophonic?: boolean;
  }): Promise<ExtractedMidiContext> {
    let safeForceMonophonic = false;
    let requested: SftTaskMode = 'auto';
    try {
      if (
        options &&
        typeof options === 'object' &&
        !isRuntimeProxy(options) &&
        !Array.isArray(options)
      ) {
        const fd = Object.getOwnPropertyDescriptor(options, 'forceMonophonic');
        if (fd && 'value' in fd && typeof fd.value === 'boolean') {
          safeForceMonophonic = fd.value;
        }
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
          requested = td.value;
        }
      }
    } catch {
      // Fallback to safe defaults if options is hostile
    }

    const targetMortmInstrument: 'PIANO' | 'SAX' =
      safeForceMonophonic || this._proxyTrack.isMonophonic() ? 'SAX' : 'PIANO';

    const pastWin = this._targetRange.pastWindowTicks(8);
    const futureWin = this._targetRange.futureWindowTicks(8);
    const excludedIds = this._isSameRangeAsActiveCandidate()
      ? this._activeCandidateNoteIds
      : undefined;

    const pastNotes = this._sliceProxyWindow(pastWin.startTick, pastWin.endTick, excludedIds);
    const futureNotes = this._sliceProxyWindow(futureWin.startTick, futureWin.endTick, excludedIds);

    const resolvedTask = resolveEffectiveSftTask({
      requestedTask: requested,
      hasPastContext: pastNotes.length > 0,
      hasFutureContext: futureNotes.length > 0,
      hasExplicitConditionTrackSelection: false,
      hasSelectedConditionContext: false,
      hasAnyConditionContext: false,
    });

    const timeSig = {
      numerator: this._timeSigNumerator,
      denominator: this._timeSigDenominator,
    };

    const pastMidiBytes =
      (resolvedTask === 'meta_past' || resolvedTask === 'infill') && pastNotes.length > 0
        ? StandardMidiCodec.encodeSingleTrackMidi(
            pastNotes,
            480,
            120,
            targetMortmInstrument,
            this._beatsPerMeasure,
            timeSig,
          )
        : null;

    const futureMidiBytes =
      (resolvedTask === 'meta_future' || resolvedTask === 'infill') && futureNotes.length > 0
        ? StandardMidiCodec.encodeSingleTrackMidi(
            futureNotes,
            480,
            120,
            targetMortmInstrument,
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
      conditionsMidiBytes: null,
      targetSlot: Object.freeze({
        trackId: this._proxyTrack.id,
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
      let safeSlot: ActiveCandidateRange | null = null;
      if (hasExplicitSlot) {
        if (!targetSlot || typeof targetSlot !== 'object' || isRuntimeProxy(targetSlot) || Array.isArray(targetSlot)) {
          return;
        }
        const td = Object.getOwnPropertyDescriptor(targetSlot, 'trackId');
        const sd = Object.getOwnPropertyDescriptor(targetSlot, 'startTick');
        const ed = Object.getOwnPropertyDescriptor(targetSlot, 'endTick');
        if (
          !td ||
          !('value' in td) ||
          typeof td.value !== 'string' ||
          td.value.trim() !== this._proxyTrack.id ||
          !sd ||
          !('value' in sd) ||
          typeof sd.value !== 'number' ||
          !Number.isFinite(sd.value) ||
          !ed ||
          !('value' in ed) ||
          typeof ed.value !== 'number' ||
          !Number.isFinite(ed.value)
        ) {
          return;
        }
        const maxTicks = MAX_SMF_QUARTER_NOTES * 480;
        const startTick = Math.min(maxTicks, Math.max(0, Math.round(sd.value)));
        const endTick = Math.min(maxTicks, Math.max(0, Math.round(ed.value)));
        if (endTick <= startTick) return;
        safeSlot = { trackId: this._proxyTrack.id, startTick, endTick };
      }
      const normalized = normalizeGeneratedCandidatesBatch(candidates, reasons, false);
      if (normalized.length === 0) return;
      const placed = this._placeCandidateOnProxyTrack(
        normalized[0],
        normalized[0].reason,
        'new_generation',
        normalized,
        safeSlot,
      );
      if (!placed) return;
      this._playheadTick = safeSlot ? safeSlot.startTick : this._targetRange.startTick;
      this._notify();
    } catch {
      // Transactional no-op on malformed candidate batch (`[R13-MED-01]`)
    }
  }

  public async selectGeneratedCandidate(index: number): Promise<void> {
    try {
      if (typeof index !== 'number' || !Number.isInteger(index) || index < 0) return;
      const chosen =
        this._candidates.find((c) => c.index === index) ??
        (index < this._candidates.length ? this._candidates[index] : undefined);
      if (!chosen) return;
      const placed = this._placeCandidateOnProxyTrack(chosen, chosen.reason, 'switch_candidate');
      if (!placed) return;
      if (this._activeCandidateRange) {
        this._playheadTick = this._activeCandidateRange.startTick;
      }
      this._notify();
    } catch {
      // Transactional no-op on candidate switch error
    }
  }

  private _isSameRangeAsActiveCandidate(): boolean {
    return (
      this._activeCandidateRange !== null &&
      this._activeCandidateRange.startTick === this._targetRange.startTick &&
      this._activeCandidateRange.endTick === this._targetRange.endTick
    );
  }

  private _placeCandidateOnProxyTrack(
    candidate: GeneratedMidiCandidate,
    nextReason: GenerationReason | null,
    mode: 'new_generation' | 'switch_candidate' = 'new_generation',
    incomingCandidates?: readonly GeneratedMidiCandidate[],
    explicitTargetSlot?: ActiveCandidateRange | null,
  ): boolean {
    const pinnedRange =
      mode === 'switch_candidate' ? this._activeCandidateRange : (explicitTargetSlot ?? null);
    if (mode === 'switch_candidate' && !pinnedRange) return false;
    const placementStartTick = pinnedRange ? pinnedRange.startTick : this._targetRange.startTick;
    const placementEndTick = pinnedRange ? pinnedRange.endTick : this._targetRange.endTick;

    const effectiveNextCandidates: readonly GeneratedMidiCandidate[] =
      mode === 'new_generation'
        ? (incomingCandidates ?? normalizeGeneratedCandidatesBatch([candidate], undefined, true))
        : this._candidates;
    if (effectiveNextCandidates.length === 0) return false;

    const parsed = candidate.parsedMidi;
    const targetInst = this._proxyTrack.toMortmInstrument();
    const matchingTracks =
      parsed.tracks.length > 1
        ? parsed.tracks.filter((t) => (t.instrument === 'SAX' ? 'SAX' : 'PIANO') === targetInst)
        : parsed.tracks;
    const sourceTracks = matchingTracks.length > 0 ? matchingTracks : parsed.tracks;

    const sameRange =
      this._activeCandidateRange !== null &&
      this._activeCandidateRange.startTick === placementStartTick &&
      this._activeCandidateRange.endTick === placementEndTick;
    const replaceablePrevIds: ReadonlySet<string> = sameRange
      ? this._activeCandidateNoteIds
      : EMPTY_CANONICAL_ID_SET;

    const hasFutureProxyNotes = this._proxyTrack
      .queryNotesOverlapping(
        placementEndTick,
        Math.max(placementEndTick + 1, this._proxyTrack.getMaxEndTick() + 1),
      )
      .some((n) => !replaceablePrevIds.has(n.id));

    const proxySnapshotNotes = this._proxyTrack.toSnapshot().notes;
    const seenProxyIds = new Set<string>();
    let willRemoveCount = 0;
    for (let i = 0; i < proxySnapshotNotes.length; i += 1) {
      const existing = proxySnapshotNotes[i];
      if (
        replaceablePrevIds.has(existing.id) ||
        (existing.startTick >= placementStartTick && existing.startTick < placementEndTick)
      ) {
        willRemoveCount += 1;
      } else {
        seenProxyIds.add(existing.id);
      }
    }
    const maxInsertableNotes = Math.min(
      Math.max(0, 32_768 - (proxySnapshotNotes.length - willRemoveCount)),
      Math.max(0, MAX_WORKSPACE_NOTES - (proxySnapshotNotes.length - willRemoveCount)),
    );
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * 480;
    const effectiveMaxEndTick = hasFutureProxyNotes
      ? Math.min(placementEndTick, maxProjectTicks)
      : maxProjectTicks;

    const scale = 480 / Math.max(1, parsed.ppqn);
    const preparedBatch = prepareScaledCandidateBatch({
      sourceTracks,
      scale,
      placementStartTick,
      effectiveMaxEndTick,
      maxInsertableNotes,
      existingIds: seenProxyIds,
      mintFallbackId: (_prefId, existingIds, batchIds) => {
        let nextId = truncateWellFormedUtf16(createNoteId().trim(), 128).trim();
        while (!nextId || existingIds.has(nextId) || batchIds.has(nextId)) {
          nextId = truncateWellFormedUtf16(createNoteId().trim(), 128).trim();
        }
        return nextId;
      },
    });

    const nextRange = {
      startTick: placementStartTick,
      endTick: placementEndTick,
    };

    const resolvedCandIdx = effectiveNextCandidates.findIndex((ec) => ec.index === candidate.index);
    const effectiveActiveCandidateIndex =
      resolvedCandIdx >= 0
        ? resolvedCandIdx
        : Math.max(0, Math.min(candidate.index, Math.max(0, effectiveNextCandidates.length - 1)));
    const sanitizedReason = sanitizeGenerationReason(nextReason);
    new ReplaceRangeNotesCommand(
      this._proxyTrack,
      placementStartTick,
      placementEndTick,
      preparedBatch,
      `AI生成フレーズ配置 (候補 ${effectiveActiveCandidateIndex + 1})`,
      replaceablePrevIds,
      {
        onExecute: (insertedIds) => {
          this._candidates = [...effectiveNextCandidates];
          this._activeCandidateIndex = effectiveActiveCandidateIndex;
          this._activeCandidateNoteIds = insertedIds;
          this._activeCandidateRange = nextRange;
          this._latestReason = sanitizedReason;
        },
      },
    ).execute();
    return true;
  }

  private _sliceProxyWindow(
    winStartTick: number,
    winEndTick: number,
    excludeNoteIds?: ReadonlySet<string>,
  ): MidiNote[] {
    if (winEndTick <= winStartTick) return [];
    const overlapping = this._proxyTrack.queryNotesOverlapping(winStartTick, winEndTick);
    return overlapping
      .filter((n) => !excludeNoteIds || !excludeNoteIds.has(n.id))
      .map((n) => {
        const s = Math.max(winStartTick, n.startTick);
        const e = Math.min(winEndTick, n.endTick);
        return e > s
          ? new MidiNote({
              pitch: n.pitch,
              startTick: s - winStartTick,
              durationTicks: e - s,
              velocity: n.velocity,
              channel: n.channel,
            })
          : null;
      })
      .filter((n): n is MidiNote => n !== null);
  }

  private _computeProxyContextAvailability(): ContextAvailability {
    const pastWin = this._targetRange.pastWindowTicks(8);
    const futureWin = this._targetRange.futureWindowTicks(8);
    const sameRange = this._isSameRangeAsActiveCandidate();
    const pastCount =
      pastWin.endTick > pastWin.startTick
        ? this._proxyTrack
            .queryNotesOverlapping(pastWin.startTick, pastWin.endTick)
            .filter(
              (n) =>
                (!sameRange || !this._activeCandidateNoteIds.has(n.id)) &&
                n.endTick > pastWin.startTick &&
                n.startTick < pastWin.endTick,
            ).length
        : 0;
    const futureCount =
      futureWin.endTick > futureWin.startTick
        ? this._proxyTrack
            .queryNotesOverlapping(futureWin.startTick, futureWin.endTick)
            .filter(
              (n) =>
                (!sameRange || !this._activeCandidateNoteIds.has(n.id)) &&
                n.endTick > futureWin.startTick &&
                n.startTick < futureWin.endTick,
            ).length
        : 0;

    let autoResolvedTask: 'meta' | 'meta_past' | 'meta_future' | 'infill' = 'meta';
    if (pastCount > 0 && futureCount > 0) autoResolvedTask = 'infill';
    else if (pastCount > 0) autoResolvedTask = 'meta_past';
    else if (futureCount > 0) autoResolvedTask = 'meta_future';

    return {
      hasPastContext: pastCount > 0,
      pastNoteCount: pastCount,
      hasFutureContext: futureCount > 0,
      futureNoteCount: futureCount,
      conditionCandidateTracks: [],
      autoResolvedTask,
    };
  }

  private _notify(): void {
    if (this._listeners.size === 0) {
      return;
    }
    const snap = this.getSnapshot();
    for (const listener of Array.from(this._listeners)) {
      try {
        listener(snap);
      } catch {
        // Isolate faulty subscriber so sibling listeners still update
      }
    }
  }
}

export class CubaseDawController extends BaseExternalDawBridgeController {
  public readonly id = 'cubase-midi-remote';
  public readonly displayName = 'Steinberg Cubase Pro / Artist (MIDI Remote Adapter)';
  protected readonly bridgeEndpointLabel = 'Cubase MIDI Remote API';
  protected readonly protocolName = 'Cubase MIDI Remote JS Bridge';

  constructor(options?: string | ExternalDawAdapterOptions) {
    super(resolveBridgeUrl(options, 'VITE_COMPASS_CUBASE_BRIDGE_URL', 'ws://127.0.0.1:9120'));
  }
}

export class AbletonDawController extends BaseExternalDawBridgeController {
  public readonly id = 'ableton-osc-bridge';
  public readonly displayName = 'Ableton Live 11 / 12 (Max for Live OSC Adapter)';
  protected readonly bridgeEndpointLabel = 'AbletonOSC Bridge';
  protected readonly protocolName = 'Ableton Live Remote Script / OSC';

  constructor(options?: string | ExternalDawAdapterOptions) {
    super(resolveBridgeUrl(options, 'VITE_COMPASS_ABLETON_BRIDGE_URL', 'ws://127.0.0.1:9121'));
  }
}
