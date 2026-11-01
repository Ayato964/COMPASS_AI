/**
 * Universal Pluggable DAW Controller Interfaces (`C` in MVC — `oop-extensible-craft`).
 * Enables hot-swapping between the built-in Demo Piano Roll DAW, Cubase, Ableton Live, etc.
 */

import { DawTrackSnapshot, TrackInstrument } from '../domain/DawTrack';
import { MeasureRange } from '../domain/MeasureRange';
import { isDomainProxy } from '../domain/MidiNote';
import { ParsedMidiFile } from '../domain/midi/StandardMidiCodec';
import { GeneratedMidiCandidate, GenerationReason } from '../services/CompassApiClient';

export type SftTaskMode = 'auto' | 'meta' | 'meta_past' | 'meta_future' | 'infill' | 'inst_comp';

export interface DawCapabilities {
  readonly hasEmbeddedPianoRoll: boolean;
  readonly supportsMultiTrack: boolean;
  readonly supportsTransportControl: boolean;
  readonly supportsDirectNoteEditing: boolean;
  readonly connectionProtocol: string;
}

export interface ContextAvailability {
  readonly hasPastContext: boolean;
  readonly pastNoteCount: number;
  readonly hasFutureContext: boolean;
  readonly futureNoteCount: number;
  readonly conditionCandidateTracks: readonly {
    id: string;
    name: string;
    instrument: TrackInstrument;
    noteCount: number;
  }[];
  readonly autoResolvedTask: 'meta' | 'meta_past' | 'meta_future' | 'infill' | 'inst_comp';
}

export interface ExtractedMidiContext {
  readonly resolvedTask: 'meta' | 'meta_past' | 'meta_future' | 'infill' | 'inst_comp';
  readonly targetMortmInstrument: 'PIANO' | 'SAX';
  readonly targetMeasureCount: number;
  readonly pastMidiBytes: Uint8Array | null;
  readonly futureMidiBytes: Uint8Array | null;
  readonly conditionsMidiBytes: Uint8Array | null;
  readonly targetSlot?: ActiveCandidateRange;
}

export interface SerializedNoteState {
  readonly id: string;
  readonly pitch: number;
  readonly startTick: number;
  readonly durationTicks: number;
  readonly velocity: number;
  readonly channel: number;
}

export interface SerializedTrackState {
  readonly id: string;
  readonly name: string;
  readonly instrument: TrackInstrument;
  readonly colorHex: string;
  readonly isMuted: boolean;
  readonly isSoloed: boolean;
  readonly notes: readonly SerializedNoteState[];
}

export interface ActiveCandidateRange {
  readonly trackId: string;
  readonly startTick: number;
  readonly endTick: number;
}

export interface SerializedWorkspaceState {
  readonly ppqn: number;
  readonly bpm: number;
  readonly beatsPerMeasure: number;
  readonly timeSigNumerator?: number;
  readonly timeSigDenominator?: number;
  readonly activeTrackId: string;
  readonly targetStartMeasure: number;
  readonly targetEndMeasure: number;
  readonly quantizeDivision: 4 | 8 | 16 | 32;
  readonly isTriplet: boolean;
  readonly tracks: readonly SerializedTrackState[];
  readonly candidates: ReadonlyArray<{
    readonly index: number;
    readonly filename: string;
    readonly rawMidiBytes: Uint8Array;
    readonly reason: GenerationReason | null;
  }>;
  readonly activeCandidateIndex: number;
  readonly activeCandidateNoteIds: readonly string[];
  readonly activeCandidateRange?: ActiveCandidateRange | null;
  readonly candidateOriginSlot?: ActiveCandidateRange | null;
  readonly latestReason: GenerationReason | null;
}

export interface DawSessionSnapshot {
  readonly controllerId: string;
  readonly controllerName: string;
  readonly isConnected: boolean;
  readonly connectionStatusText: string;
  readonly capabilities: DawCapabilities;
  readonly ppqn: number;
  readonly bpm: number;
  readonly beatsPerMeasure: number;
  readonly timeSigNumerator: number;
  readonly timeSigDenominator: number;
  readonly isPlaying: boolean;
  readonly playheadTick: number;
  readonly tracks: readonly DawTrackSnapshot[];
  readonly activeTrackId: string;
  readonly targetRange: MeasureRange;
  readonly quantizeDivision: 4 | 8 | 16 | 32;
  readonly isTriplet: boolean;
  readonly selectedNoteIds: ReadonlySet<string>;
  readonly canUndo: boolean;
  readonly canRedo: boolean;
  readonly undoDescription: string | null;
  readonly candidates: readonly GeneratedMidiCandidate[];
  readonly activeCandidateIndex: number;
  readonly activeCandidateRange?: ActiveCandidateRange | null;
  readonly latestReason: GenerationReason | null;
  readonly contextAvailability: ContextAvailability;
}

export interface IDawController {
  readonly id: string;
  readonly displayName: string;
  readonly capabilities: DawCapabilities;

  connect(): Promise<void>;
  disconnect(): Promise<void>;
  getSnapshot(): DawSessionSnapshot;
  subscribe(listener: (snapshot: DawSessionSnapshot) => void): () => void;

  // Transport & Target Range
  play(): Promise<void>;
  pause(): void;
  stop(): void;
  setPlayheadTick(tick: number): void;
  setBpm(bpm: number): void;
  setTimeSignature(numerator: number, denominator: number): void;
  setTargetMeasureRange(startMeasure: number, endMeasure: number): void;
  setActiveTrack(trackId: string): void;

  // AI Context Extraction & Candidate Insertion
  extractGenerationContext(options: {
    taskOverride?: SftTaskMode;
    conditionTrackId?: string | null;
    forceMonophonic?: boolean;
  }): Promise<ExtractedMidiContext>;

  applyGeneratedCandidates(
    candidates: readonly GeneratedMidiCandidate[],
    reasons: readonly GenerationReason[],
    targetSlot?: ActiveCandidateRange,
  ): Promise<void>;

  selectGeneratedCandidate(index: number): Promise<void>;
}

export interface MidiTrackImportResult {
  readonly importedTracks: number;
  readonly importedNotes: number;
}

/**
 * Role Interface (`ISP`) for DAW controllers that provide an interactive in-app Piano Roll
 * (such as `DemoPianoRollController`).
 */
export interface IDawInteractivePianoRoll extends IDawController {
  // MIDI Import / Export / Cross-Window State Sync
  parseUploadedMidiFile(file: File): Promise<ParsedMidiFile>;
  importParsedMidiTracks(
    parsedFile: ParsedMidiFile,
    selectedTrackIndices: readonly number[],
    replaceExisting: boolean,
  ): MidiTrackImportResult;
  loadBuiltInDemoArrangement(): void;
  exportProjectMidiBytes(): Uint8Array;
  exportWorkspaceState(): SerializedWorkspaceState;
  exportTrustedWorkspaceState?(): SerializedWorkspaceState;
  applyWorkspaceState(state: SerializedWorkspaceState): void;

  // Track Management
  addTrack(name?: string, instrument?: TrackInstrument): string;
  deleteTrack(trackId: string): void;
  renameTrack(trackId: string, name: string): void;
  setTrackInstrument(trackId: string, instrument: TrackInstrument): void;
  toggleTrackMute(trackId: string): void;
  toggleTrackSolo(trackId: string): void;

  // Grid & Selection
  setQuantize(division: 4 | 8 | 16 | 32, isTriplet?: boolean): void;
  snapTick(rawTick: number): number;
  getSnapStepTicks(): number;
  setSelectedNotes(noteIds: Iterable<string>): void;
  selectAllNotesOnActiveTrack(): void;

  // Reversible Note Commands + Audition
  auditionPitch(pitch: number): void;
  createNoteAt(
    pitch: number,
    startTick: number,
    durationTicks?: number,
    velocity?: number,
    options?: { readonly silent?: boolean },
  ): void;
  deleteSelectedNotes(): void;
  transformNotes(
    updates: ReadonlyArray<{
      id: string;
      pitch: number;
      startTick: number;
      durationTicks: number;
      velocity?: number;
    }>,
    description?: string,
  ): void;
  copySelectedNotes(): number;
  pasteNotesAtPlayhead(): number;
  undo(): void;
  redo(): void;
}

export function resolveEffectiveSftTask(params: {
  readonly requestedTask: SftTaskMode;
  readonly hasPastContext: boolean;
  readonly hasFutureContext: boolean;
  readonly hasExplicitConditionTrackSelection: boolean;
  readonly hasSelectedConditionContext: boolean;
  readonly hasAnyConditionContext: boolean;
}): Exclude<SftTaskMode, 'auto'> {
  const {
    requestedTask,
    hasPastContext,
    hasFutureContext,
    hasExplicitConditionTrackSelection,
    hasSelectedConditionContext,
    hasAnyConditionContext,
  } = params;
  if (requestedTask === 'inst_comp') {
    const canUseCond = hasExplicitConditionTrackSelection
      ? hasSelectedConditionContext
      : hasAnyConditionContext;
    return canUseCond ? 'inst_comp' : 'meta';
  }
  if (requestedTask === 'infill') {
    if (hasPastContext && hasFutureContext) return 'infill';
    if (hasPastContext) return 'meta_past';
    if (hasFutureContext) return 'meta_future';
    return 'meta';
  }
  if (requestedTask === 'meta_past') {
    return hasPastContext ? 'meta_past' : 'meta';
  }
  if (requestedTask === 'meta_future') {
    return hasFutureContext ? 'meta_future' : 'meta';
  }
  if (requestedTask === 'meta') {
    return 'meta';
  }
  // 'auto'
  if (hasExplicitConditionTrackSelection && hasSelectedConditionContext) {
    return 'inst_comp';
  }
  if (hasPastContext && hasFutureContext) return 'infill';
  if (hasPastContext) return 'meta_past';
  if (hasFutureContext) return 'meta_future';
  return 'meta';
}

export function isInteractivePianoRollController(
  controller: IDawController,
): controller is IDawInteractivePianoRoll {
  try {
    if (!controller || typeof controller !== 'object' || isDomainProxy(controller)) {
      return false;
    }
    const caps = controller.capabilities;
    if (!caps || typeof caps !== 'object' || isDomainProxy(caps)) {
      return false;
    }
    return caps.hasEmbeddedPianoRoll === true;
  } catch {
    return false;
  }
}
