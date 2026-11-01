/**
 * Studio Piano Roll Tab View (`PianoRollView.tsx`).
 * Modeled after the user's Java `COMPASS` DTM (`ArrangementFrame` + `PianoRoll` + `VelocityPanel`),
 * cleanly separated from the `COMPASS AI` Control Panel (`Shift+G` / `COMPASS AI を起動`).
 * Strictly adheres to `anti-ai-web-design` & `oop-extensible-craft`:
 * - Monochrome `#09090B` / `#111113` studio palette + single sky-400 functional accent
 * - Single primary solid white CTA (`COMPASS AI を起動`) per view (§5.3 One Primary Action)
 * - O(log N + K) implicit-interval-tree viewport rendering, hit-testing, marquee & velocity queries
 * - Zero listener churn, zero 40Hz scroll-hijack, live 60fps Canvas drag previews while paused
 */

import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  BoxSelect,
  ChevronDown,
  Download,
  Music,
  PanelLeft,
  Pause,
  Pencil,
  Play,
  Plus,
  Redo2,
  Square,
  Trash2,
  Undo2,
  Upload,
  X,
} from 'lucide-react';
import {
  DawTrackSnapshot,
  MAX_NOTES_PER_TRACK,
  MAX_WORKSPACE_NOTES,
  queryOverlappingNotesInSnapshot,
  TrackInstrument,
} from '../domain/DawTrack';
import { evaluateGenerationMeter, MeasureRange } from '../domain/MeasureRange';
import { MidiNote, midiPitchToName } from '../domain/MidiNote';
import {
  extractCanonicalParsedMidiTracks,
  MAX_PARSED_SMF_TRACKS,
  MAX_SMF_QUARTER_NOTES,
  ParsedMidiFile,
  StandardMidiCodec,
} from '../domain/midi/StandardMidiCodec';
import {
  DawSessionSnapshot,
  IDawController,
  IDawInteractivePianoRoll,
  isInteractivePianoRollController,
} from '../controllers/IDawController';
import { MidiTrackImportModal } from './MidiTrackImportModal';
import { useAdaptiveTooltipAlignment } from '../utils/useAdaptiveTooltipAlignment';

interface PianoRollViewProps {
  readonly isActive?: boolean;
  readonly isSecondaryPopout?: boolean;
  readonly isGenerating?: boolean;
  readonly controller: IDawController;
  readonly session: DawSessionSnapshot;
  readonly focusRevision: number;
  readonly candidateFocusRevision?: number;
  readonly onLaunchCompassAi: () => void;
  readonly onSwitchToDemoController: () => void;
  readonly onProjectMutated?: () => void;
  readonly onModalOpenChange?: (isOpen: boolean) => void;
}

type EditToolMode = 'select' | 'draw';

type DragInteraction =
  | {
      readonly kind: 'move' | 'resize';
      readonly pointerId: number;
      readonly trackId: string;
      readonly notesSnapshotRef: readonly MidiNote[];
      readonly anchorNoteId: string;
      readonly startClientX: number;
      readonly startClientY: number;
      readonly initialNotes: ReadonlyArray<{
        id: string;
        pitch: number;
        startTick: number;
        durationTicks: number;
        velocity: number;
      }>;
      readonly initialSelection: ReadonlySet<string>;
      readonly toggleDeselectOnClick?: boolean;
      readonly isolateSelectionOnClick?: boolean;
      fineResize?: boolean;
      hasMovedPointer?: boolean;
      currentDeltaTick: number;
      currentDeltaPitch: number;
    }
  | {
      readonly kind: 'marquee';
      readonly pointerId: number;
      readonly trackId: string;
      readonly notesSnapshotRef: readonly MidiNote[];
      readonly startClientX: number;
      readonly startClientY: number;
      readonly additive: boolean;
      readonly initialSelection: ReadonlySet<string>;
      readonly startWorldX: number;
      readonly startWorldY: number;
      currentWorldX: number;
      currentWorldY: number;
      hasExceededDragThreshold: boolean;
    }
  | {
      readonly kind: 'velocity';
      readonly pointerId: number;
      readonly trackId: string;
      readonly notesSnapshotRef: readonly MidiNote[];
      readonly updates: Map<string, number>;
    }
  | {
      readonly kind: 'pan';
      readonly pointerId: number;
      readonly startClientX: number;
      readonly startClientY: number;
      readonly startScrollTickX: number;
      readonly startScrollPxY: number;
    }
  | {
      readonly kind: 'ruler-scrub';
      readonly pointerId: number;
      readonly anchorPlayheadTick: number;
    }
  | {
      readonly kind: 'ruler-range';
      readonly pointerId: number;
      readonly edge: 'start' | 'end';
      readonly anchorStartMeasure: number;
      readonly anchorEndMeasure: number;
    }
  | {
      readonly kind: 'draw-create';
      readonly pointerId: number;
      readonly trackId: string;
      readonly notesSnapshotRef: readonly MidiNote[];
      readonly startClientX: number;
      readonly startClientY: number;
      readonly pitch: number;
      readonly rawTick: number;
      readonly snappedStartTick: number;
      readonly initialDurationTicks: number;
      readonly velocity: number;
      fineResize?: boolean;
      hasMovedPointer?: boolean;
      currentDeltaTick: number;
    }
  | {
      readonly kind: 'touch-draw-tap';
      readonly pointerId: number;
      readonly trackId: string;
      readonly notesSnapshotRef: readonly MidiNote[];
      readonly startClientX: number;
      readonly startClientY: number;
      readonly pitch: number;
      readonly rawTick: number;
      readonly hitNoteId: string | null;
    };

const KEYBOARD_WIDTH = 72;
const RULER_HEIGHT = 32;
function computeVelocityLaneHeight(canvasHeightPx: number): number {
  if (canvasHeightPx < 220) return 44;
  if (canvasHeightPx < 280) return 56;
  return 72;
}
const TOTAL_PITCHES = 128;
const DRAG_DEADZONE_PX = 2.5;
const CANVAS_MONO_FONT =
  '"JetBrains Mono", Consolas, "Hiragino Sans", "BIZ UDGothic", "BIZ UDPGothic", Meiryo, monospace';

const BLACK_KEY_PCS = new Set([1, 3, 6, 8, 10]);

const TIME_SIGNATURE_PRESETS: readonly string[] = [
  '4/4',
  '3/4',
  '2/4',
  '6/8',
  '7/8',
  '5/8',
  '3/8',
];

function isTextEntryControl(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable || target.getAttribute('contenteditable') === 'true') {
    return true;
  }
  if (target.closest('textarea, [contenteditable="true"]')) {
    return true;
  }
  const input = target.closest('input');
  if (input) {
    const t = (input.type || 'text').toLowerCase();
    return (
      t !== 'checkbox' &&
      t !== 'radio' &&
      t !== 'button' &&
      t !== 'range' &&
      t !== 'file'
    );
  }
  return false;
}

function srgbByteToRelativeLinear(c8: number): number {
  const s = Math.max(0, Math.min(255, c8)) / 255;
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

export interface NoteLabelBackdropOptions {
  readonly inTargetRange?: boolean;
  readonly inContextWindow?: boolean;
  readonly ghostStackCount?: number;
  readonly backdropRgb?: readonly [number, number, number];
}

function parseHexRgb(fillHex: string): [number, number, number] {
  const cleaned = fillHex.trim().replace(/^#/, '');
  if (/^[0-9a-fA-F]{6}$/.test(cleaned)) {
    return [
      parseInt(cleaned.slice(0, 2), 16),
      parseInt(cleaned.slice(2, 4), 16),
      parseInt(cleaned.slice(4, 6), 16),
    ];
  }
  if (/^[0-9a-fA-F]{3}$/.test(cleaned)) {
    return [
      parseInt(cleaned[0] + cleaned[0], 16),
      parseInt(cleaned[1] + cleaned[1], 16),
      parseInt(cleaned[2] + cleaned[2], 16),
    ];
  }
  return [244, 244, 245];
}

function computeCompositedNoteLuminance(
  noteRgb: readonly [number, number, number],
  fillAlpha: number,
  pitch: number,
  backdrop?: NoteLabelBackdropOptions,
): number {
  const alpha = Math.max(0, Math.min(1, Number.isFinite(fillAlpha) ? fillAlpha : 1));
  const isBlackKey = BLACK_KEY_PCS.has(((pitch % 12) + 12) % 12);
  let bgR = backdrop?.backdropRgb ? backdrop.backdropRgb[0] : isBlackKey ? 13 : 18;
  let bgG = backdrop?.backdropRgb ? backdrop.backdropRgb[1] : isBlackKey ? 13 : 18;
  let bgB = backdrop?.backdropRgb ? backdrop.backdropRgb[2] : isBlackKey ? 16 : 22;

  if (backdrop?.inTargetRange) {
    bgR = 56 * 0.07 + bgR * 0.93;
    bgG = 189 * 0.07 + bgG * 0.93;
    bgB = 248 * 0.07 + bgB * 0.93;
  } else if (backdrop?.inContextWindow) {
    bgR = 161 * 0.03 + bgR * 0.97;
    bgG = 161 * 0.03 + bgG * 0.97;
    bgB = 170 * 0.03 + bgB * 0.97;
  }

  const ghostCount = Math.max(0, Math.min(16, Math.floor(backdrop?.ghostStackCount ?? 0)));
  for (let i = 0; i < ghostCount; i += 1) {
    bgR = 161 * 0.18 + bgR * 0.82;
    bgG = 161 * 0.18 + bgG * 0.82;
    bgB = 170 * 0.18 + bgB * 0.82;
  }

  const compR = noteRgb[0] * alpha + bgR * (1 - alpha);
  const compG = noteRgb[1] * alpha + bgG * (1 - alpha);
  const compB = noteRgb[2] * alpha + bgB * (1 - alpha);

  return (
    0.2126 * srgbByteToRelativeLinear(compR) +
    0.7152 * srgbByteToRelativeLinear(compG) +
    0.0722 * srgbByteToRelativeLinear(compB)
  );
}

/**
 * Computes WCAG 2.1 Porter-Duff `source-over` relative luminance of a note rectangle over the
 * studio pitch-row (`#0D0D10` / `#121216`), target/context span tint, and multi-track ghost
 * notes, returning `'#000000'` or `'#FFFFFF'` so pitch labels achieve `>= 4.58:1` contrast
 * across all track swatches, velocities (`1..127`), tints, and ghost overlaps (`[R3-LOW-04]`, `[R4-LOW-03]`).
 */
export function pickAccessibleNoteLabelColor(
  fillHex: string,
  fillAlpha: number,
  pitch = 60,
  backdrop?: NoteLabelBackdropOptions,
): '#000000' | '#FFFFFF' {
  const lum = computeCompositedNoteLuminance(parseHexRgb(fillHex), fillAlpha, pitch, backdrop);
  // Exact WCAG crossover `(L + 0.05) / 0.05 === 1.05 / (L + 0.05)` -> L = sqrt(0.0525) - 0.05 = 0.17912878
  return lum > 0.17912878 ? '#000000' : '#FFFFFF';
}

function formatUnsupportedGenerationMeterNotice(parsed: ParsedMidiFile): string {
  const num = parsed?.timeSigNumerator ?? 4;
  const den = parsed?.timeSigDenominator ?? 4;
  const evalRes = evaluateGenerationMeter(num, den);
  if (!evalRes.supported) {
    if (evalRes.stepsPerBar > 96) {
      return ` (拍子 ${evalRes.label} はモデルの1小節上限 [4/4相当] を超えるため、AI生成時はツールバーで4/4等へ変更してください)`;
    }
    return ` (拍子 ${evalRes.label} はモデルの許容範囲 [12〜96ステップ] 外のため、AI生成時はツールバーで4/4等へ変更してください)`;
  }
  return '';
}

function isSpaceActivatableControl(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tabEl = target.closest('[role="tab"]');
  if (tabEl) {
    return tabEl.getAttribute('aria-selected') !== 'true';
  }
  return Boolean(
    target.closest(
      'button, a, summary, input, select, textarea, [role="button"], [role="checkbox"], [role="menuitem"]',
    ),
  );
}

function isImeComposing(
  e: KeyboardEvent | React.KeyboardEvent<HTMLElement>,
): boolean {
  const native = 'nativeEvent' in e ? e.nativeEvent : e;
  return Boolean(native.isComposing) || e.keyCode === 229;
}

const INSTRUMENT_OPTIONS: readonly { value: TrackInstrument; label: string }[] = [
  { value: 'PIANO', label: 'PIANO' },
  { value: 'SAX', label: 'SAX' },
  { value: 'BASS', label: 'BASS' },
  { value: 'VIOLIN', label: 'VIOLIN' },
  { value: 'SYNTH', label: 'SYNTH' },
  { value: 'GUITAR', label: 'GUITAR' },
  { value: 'DRUMS', label: 'DRUMS' },
];

const MAX_RENDERED_MEASURE_INDEX = MAX_SMF_QUARTER_NOTES * 2;
const MAX_VISIBLE_MEASURES_PER_FRAME = 4096;
const MAX_WHEEL_DELTA_PX = 4096;

function clampPixelsPerBeat(val: number, fallback = 64): number {
  const safe = Number.isFinite(val) ? val : Number.isFinite(fallback) ? fallback : 64;
  return Math.max(20, Math.min(240, safe));
}

function clampRowHeight(val: number, fallback = 16): number {
  const safe = Number.isFinite(val) ? val : Number.isFinite(fallback) ? fallback : 16;
  return Math.max(10, Math.min(28, Math.round(safe)));
}

function clampScrollTickX(val: number, ppqn: number, fallback = 0): number {
  const cleanPpqn = Number.isFinite(ppqn) && ppqn >= 1 ? Math.min(32767, ppqn) : 480;
  const maxTick = MAX_SMF_QUARTER_NOTES * cleanPpqn;
  const safe = Number.isFinite(val) ? val : Number.isFinite(fallback) ? fallback : 0;
  return Math.max(0, Math.min(maxTick, safe));
}

function clampScrollPxY(val: number, maxScrollY: number, fallback = 0): number {
  const safeMax = Number.isFinite(maxScrollY) && maxScrollY >= 0 ? maxScrollY : 0;
  const safe = Number.isFinite(val) ? val : Number.isFinite(fallback) ? fallback : 0;
  return Math.max(0, Math.min(safeMax, safe));
}

/**
 * Group-clamped multi-note movement (`PianoRollView.java` parity):
 * Clamps `(deltaTick, deltaPitch)` against the collective bounding box of all moving notes
 * so multi-note chords and phrases never squash onto `tick = 0`, `maxProjectTicks`, or `pitch = 0/127`.
 */
export function computeClampedGroupMoveDeltas(
  initialNotes: ReadonlyArray<{
    readonly pitch: number;
    readonly startTick: number;
    readonly durationTicks: number;
  }>,
  rawDeltaTick: number,
  rawDeltaPitch: number,
  ppqn: number,
): { readonly deltaTick: number; readonly deltaPitch: number } {
  if (!Array.isArray(initialNotes) || initialNotes.length === 0) {
    return { deltaTick: 0, deltaPitch: 0 };
  }
  const cleanPpqn = Number.isFinite(ppqn) && ppqn >= 1 ? Math.min(32767, Math.round(ppqn)) : 480;
  const maxProjectTicks = MAX_SMF_QUARTER_NOTES * cleanPpqn;

  let minStart = Infinity;
  let maxStart = -Infinity;
  let maxEnd = -Infinity;
  let minPitch = Infinity;
  let maxPitch = -Infinity;

  for (const n of initialNotes) {
    if (
      !n ||
      !Number.isFinite(n.pitch) ||
      !Number.isFinite(n.startTick) ||
      !Number.isFinite(n.durationTicks)
    ) {
      continue;
    }
    const p = Math.max(0, Math.min(127, Math.round(n.pitch)));
    const s = Math.max(0, Math.round(n.startTick));
    const d = Math.max(1, Math.round(n.durationTicks));
    if (s < minStart) minStart = s;
    if (s > maxStart) maxStart = s;
    if (s + d > maxEnd) maxEnd = s + d;
    if (p < minPitch) minPitch = p;
    if (p > maxPitch) maxPitch = p;
  }

  if (!Number.isFinite(minStart) || !Number.isFinite(minPitch)) {
    return { deltaTick: 0, deltaPitch: 0 };
  }

  const minAllowedTickDelta = -minStart;
  const maxAllowedTickDelta = Math.max(
    minAllowedTickDelta,
    Math.min(maxProjectTicks - 1 - maxStart, maxProjectTicks - maxEnd),
  );
  const cleanDeltaTick = Number.isFinite(rawDeltaTick) ? Math.round(rawDeltaTick) : 0;
  const deltaTick = Math.max(minAllowedTickDelta, Math.min(maxAllowedTickDelta, cleanDeltaTick));

  const minAllowedPitchDelta = -minPitch;
  const maxAllowedPitchDelta = Math.max(minAllowedPitchDelta, 127 - maxPitch);
  const cleanDeltaPitch = Number.isFinite(rawDeltaPitch) ? Math.round(rawDeltaPitch) : 0;
  const deltaPitch = Math.max(
    minAllowedPitchDelta,
    Math.min(maxAllowedPitchDelta, cleanDeltaPitch),
  );

  return { deltaTick, deltaPitch };
}

function fillRoundedCanvasRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  radius: number,
): void {
  const r = Math.max(0, Math.min(radius, w * 0.5, h * 0.5));
  if (r > 0.5 && typeof ctx.roundRect === 'function') {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, w, h);
  }
}

function strokeRoundedCanvasRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  radius: number,
): void {
  const r = Math.max(0, Math.min(radius, w * 0.5, h * 0.5));
  if (r > 0.5 && typeof ctx.roundRect === 'function') {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.stroke();
  } else {
    ctx.strokeRect(x, y, w, h);
  }
}

function areNoteSnapshotsIdentical(a: readonly MidiNote[], b: readonly MidiNote[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const na = a[i];
    const nb = b[i];
    if (
      na !== nb &&
      (na.id !== nb.id ||
        na.pitch !== nb.pitch ||
        na.startTick !== nb.startTick ||
        na.durationTicks !== nb.durationTicks ||
        na.velocity !== nb.velocity)
    ) {
      return false;
    }
  }
  return true;
}

export const PianoRollView: React.FC<PianoRollViewProps> = ({
  isActive = true,
  isSecondaryPopout = false,
  isGenerating = false,
  controller,
  session,
  focusRevision,
  candidateFocusRevision = 0,
  onLaunchCompassAi,
  onSwitchToDemoController,
  onProjectMutated,
  onModalOpenChange,
}) => {
  useAdaptiveTooltipAlignment();
  const interactiveCtrl: IDawInteractivePianoRoll | null = isInteractivePianoRollController(
    controller,
  )
    ? controller
    : null;
  const hasInteractiveCanvas = Boolean(interactiveCtrl);

  const pianoRollRootRef = useRef<HTMLDivElement | null>(null);
  const prevHasInteractiveCanvasRef = useRef<boolean>(hasInteractiveCanvas);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadMidiButtonRef = useRef<HTMLButtonElement | null>(null);
  const secondaryStripRef = useRef<HTMLDivElement | null>(null);
  const undoButtonRef = useRef<HTMLButtonElement | null>(null);
  const redoButtonRef = useRef<HTMLButtonElement | null>(null);
  const launchCompassAiButtonRef = useRef<HTMLButtonElement | null>(null);
  const pendingToolbarFocusTargetRef = useRef<'redo' | 'undo' | null>(null);
  const canvasContainerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const userToggledTrackRackRef = useRef<boolean>(false);

  const [toolMode, setToolMode] = useState<EditToolMode>('select');
  const [showTrackRack, setShowTrackRack] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 768 : true,
  );
  const [pixelsPerBeat, setPixelsPerBeat] = useState<number>(64);
  const [rowHeight, setRowHeight] = useState<number>(16);
  const [scrollTickX, setScrollTickX] = useState<number>(0);
  // Default vertical scroll centered near C4..C5 (row ~53 from top)
  const [scrollPxY, setScrollPxY] = useState<number>(() => (127 - 74) * 16);
  const [canvasCursor, setCanvasCursor] = useState<string>('cursor-default');
  const canvasCursorRef = useRef<string>('cursor-default');
  const updateCanvasCursor = useCallback((nextCursor: string) => {
    if (canvasCursorRef.current !== nextCursor) {
      canvasCursorRef.current = nextCursor;
      setCanvasCursor(nextCursor);
    }
  }, []);
  const [editingTrackId, setEditingTrackId] = useState<string | null>(null);
  const [editingTrackDraft, setEditingTrackDraft] = useState<string>('');
  const cancelTrackRenameRef = useRef<boolean>(false);
  const trackRenameOpenedViaPointerRef = useRef<boolean>(false);
  const lastTrackBtnPointerTypeRef = useRef<string>('mouse');
  const restoreFocusTrackIdRef = useRef<string | null>(null);
  const restoreFocusTrackPointerModalityRef = useRef<boolean>(false);
  const trackRackRef = useRef<HTMLElement | null>(null);
  const pointerFocusedButtonRef = useRef<HTMLButtonElement | null>(null);
  const pendingPointerFocusButtonRef = useRef<HTMLButtonElement | null>(null);
  const blurredPointerButtonRef = useRef<HTMLButtonElement | null>(null);

  const focusButtonWithPointerModality = useCallback(
    (btn: HTMLButtonElement | null | undefined, options?: FocusOptions) => {
      if (!btn || btn.disabled || !btn.isConnected) return;
      if (typeof document !== 'undefined' && document.activeElement === btn) {
        pointerFocusedButtonRef.current = btn;
        return;
      }
      pendingPointerFocusButtonRef.current = btn;
      try {
        btn.focus(options);
        if (typeof document !== 'undefined' && document.activeElement === btn) {
          pointerFocusedButtonRef.current = btn;
        }
      } finally {
        if (pendingPointerFocusButtonRef.current === btn) {
          pendingPointerFocusButtonRef.current = null;
        }
      }
    },
    [],
  );

  const [canvasSize, setCanvasSize] = useState<{ width: number; height: number }>({
    width: 960,
    height: 540,
  });

  const safeCandidates = useMemo(
    () =>
      Array.isArray(session?.candidates)
        ? session.candidates.filter((c) => Boolean(c && typeof c === 'object'))
        : [],
    [session?.candidates],
  );
  const safeTracks = useMemo(
    () =>
      Array.isArray(session?.tracks)
        ? session.tracks.filter((t) => Boolean(t && typeof t === 'object' && typeof t.id === 'string'))
        : [],
    [session?.tracks],
  );
  useLayoutEffect(() => {
    if (
      restoreFocusTrackIdRef.current &&
      editingTrackId !== restoreFocusTrackIdRef.current
    ) {
      const targetId = restoreFocusTrackIdRef.current;
      const usePointerModality = restoreFocusTrackPointerModalityRef.current;
      restoreFocusTrackIdRef.current = null;
      restoreFocusTrackPointerModalityRef.current = false;
      const rackEl = trackRackRef.current;
      if (rackEl) {
        const buttons = Array.from(rackEl.querySelectorAll('button[data-track-select-id]'));
        const match = buttons.find((b) => b.getAttribute('data-track-select-id') === targetId);
        if (match instanceof HTMLButtonElement && !match.disabled) {
          const rackRect = rackEl.getBoundingClientRect();
          const stickyHdr = rackEl.querySelector('[data-tooltip-sticky-header="true"]');
          const minTop =
            stickyHdr instanceof HTMLElement ? stickyHdr.getBoundingClientRect().bottom : rackRect.top;
          const mRect = match.getBoundingClientRect();
          if (rackRect.height > 0 && mRect.height > 0) {
            if (mRect.top < minTop - 2) {
              rackEl.scrollTop = Math.max(0, rackEl.scrollTop - (minTop - mRect.top + 6));
            } else if (mRect.bottom > rackRect.bottom + 2 && mRect.top > minTop + 4) {
              rackEl.scrollTop += Math.min(
                mRect.bottom - rackRect.bottom + 6,
                mRect.top - minTop,
              );
            }
          }
          if (usePointerModality) {
            focusButtonWithPointerModality(match, { preventScroll: true });
          } else {
            match.focus({ preventScroll: true });
          }
        }
      }
    }
  }, [editingTrackId, focusButtonWithPointerModality, safeTracks]);
  const safeSelectedNoteIds: ReadonlySet<string> = useMemo(
    () => (session?.selectedNoteIds instanceof Set ? session.selectedNoteIds : new Set<string>()),
    [session?.selectedNoteIds],
  );
  const safeIsPlaying = Boolean(session?.isPlaying);
  const safeCanUndo = Boolean(session?.canUndo);
  const safeCanRedo = Boolean(session?.canRedo);
  useLayoutEffect(() => {
    const pendingTarget = pendingToolbarFocusTargetRef.current;
    if (!pendingTarget) return;
    pendingToolbarFocusTargetRef.current = null;
    if (pendingTarget === 'redo') {
      if (redoButtonRef.current && !redoButtonRef.current.disabled) {
        redoButtonRef.current.focus();
        return;
      }
    } else if (pendingTarget === 'undo') {
      if (undoButtonRef.current && !undoButtonRef.current.disabled) {
        undoButtonRef.current.focus();
        return;
      }
    }
    launchCompassAiButtonRef.current?.focus();
  }, [safeCanRedo, safeCanUndo]);
  const safeControllerName =
    typeof session?.controllerName === 'string' && session.controllerName
      ? session.controllerName
      : 'External DAW';
  const safePlayheadTick =
    typeof session?.playheadTick === 'number' && Number.isFinite(session.playheadTick)
      ? Math.max(0, session.playheadTick)
      : 0;

  const [pendingImport, setPendingImport] = useState<{
    fileName: string;
    parsed: ParsedMidiFile;
    isSamplePreset?: boolean;
  } | null>(null);
  const modalTriggerFocusRef = useRef<HTMLElement | null>(null);
  const isMountedRef = useRef<boolean>(true);
  const midiUploadSeqRef = useRef<number>(0);
  const latestInteractiveCtrlRef = useRef<IDawInteractivePianoRoll | null>(interactiveCtrl);
  const isActiveAndInteractiveRef = useRef<boolean>(Boolean(isActive && interactiveCtrl));
  latestInteractiveCtrlRef.current = interactiveCtrl;
  isActiveAndInteractiveRef.current = Boolean(isActive && interactiveCtrl);

  const modalOpenedViaPointerRef = useRef<boolean>(false);
  const uploadOpenedViaPointerRef = useRef<boolean>(false);
  const standbySwitchModalityRef = useRef<'pointer' | 'keyboard' | null>(null);
  const lastInputModalityRef = useRef<'pointer' | 'keyboard'>('pointer');
  const canvasHadRecentPointerRef = useRef<boolean>(false);

  useLayoutEffect(() => {
    isMountedRef.current = true;
    latestInteractiveCtrlRef.current = interactiveCtrl;
    const activeInteractive = Boolean(isActive && interactiveCtrl);
    isActiveAndInteractiveRef.current = activeInteractive;
    if (!activeInteractive || pendingImport !== null) {
      canvasHadRecentPointerRef.current = false;
    }
    if (!activeInteractive) {
      midiUploadSeqRef.current += 1;
      uploadOpenedViaPointerRef.current = false;
    }
    if (!isActive) {
      standbySwitchModalityRef.current = null;
    }
    return () => {
      isMountedRef.current = false;
      isActiveAndInteractiveRef.current = false;
      midiUploadSeqRef.current += 1;
      uploadOpenedViaPointerRef.current = false;
      canvasHadRecentPointerRef.current = false;
    };
  }, [isActive, interactiveCtrl, pendingImport]);

  useEffect(() => {
    const inputEl = fileInputRef.current;
    if (!inputEl) return;
    const onFilePickerCancel = () => {
      uploadOpenedViaPointerRef.current = false;
    };
    inputEl.addEventListener('cancel', onFilePickerCancel);
    return () => {
      inputEl.removeEventListener('cancel', onFilePickerCancel);
    };
  }, [hasInteractiveCanvas]);

  const openImportModal = useCallback(
    (
      nextPending: { fileName: string; parsed: ParsedMidiFile; isSamplePreset?: boolean },
      fallbackFocusEl?: HTMLElement | null,
      openedViaPointer = false,
    ) => {
      if (!isActiveAndInteractiveRef.current) {
        return;
      }
      modalOpenedViaPointerRef.current = openedViaPointer;
      const activeEl =
        document.activeElement instanceof HTMLElement &&
        document.activeElement !== document.body &&
        document.activeElement !== fileInputRef.current
          ? document.activeElement
          : null;
      modalTriggerFocusRef.current = activeEl ?? fallbackFocusEl ?? null;
      if (
        document.activeElement instanceof HTMLElement &&
        document.activeElement !== document.body
      ) {
        document.activeElement.blur();
      }
      onModalOpenChange?.(true);
      setPendingImport(nextPending);
    },
    [onModalOpenChange],
  );
  const closeImportModal = useCallback(() => {
    onModalOpenChange?.(false);
    setPendingImport(null);
  }, [onModalOpenChange]);
  const isImportModalRendered = Boolean(isActive && interactiveCtrl && pendingImport !== null);
  useLayoutEffect(() => {
    if ((!isActive || !interactiveCtrl) && pendingImport !== null) {
      modalTriggerFocusRef.current = null;
      modalOpenedViaPointerRef.current = false;
      setPendingImport(null);
      onModalOpenChange?.(false);
    }
  }, [interactiveCtrl, isActive, onModalOpenChange, pendingImport]);
  useLayoutEffect(() => {
    onModalOpenChange?.(isImportModalRendered);
    return () => {
      onModalOpenChange?.(false);
    };
  }, [isImportModalRendered, onModalOpenChange]);

  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const [bpmDraft, setBpmDraft] = useState<string | null>(null);
  const bpmFocusAnchorRef = useRef<number>(120);
  const cancelBpmDraftRef = useRef<boolean>(false);
  const [startBarDraft, setStartBarDraft] = useState<string | null>(null);
  const [endBarDraft, setEndBarDraft] = useState<string | null>(null);
  const startBarAnchorEndRef = useRef<number>(4);
  const rangeFocusAnchorRef = useRef<{ start: number; end: number }>({ start: 1, end: 4 });
  const cancelStartBarDraftRef = useRef<boolean>(false);
  const cancelEndBarDraftRef = useRef<boolean>(false);
  const prevControllerRef = useRef<IDawController>(controller);
  const lastFirstCandidateRef = useRef<unknown>(safeCandidates[0] ?? null);

  useLayoutEffect(() => {
    if (prevControllerRef.current === controller) {
      return;
    }
    prevControllerRef.current = controller;
    setBpmDraft(null);
    setStartBarDraft(null);
    setEndBarDraft(null);
    cancelBpmDraftRef.current = false;
    cancelStartBarDraftRef.current = false;
    cancelEndBarDraftRef.current = false;
    setEditingTrackId(null);
    setEditingTrackDraft('');
    cancelTrackRenameRef.current = false;
    restoreFocusTrackIdRef.current = null;
    restoreFocusTrackPointerModalityRef.current = false;
    trackRenameOpenedViaPointerRef.current = false;
  }, [controller]);

  useLayoutEffect(() => {
    if (!isActive || !showTrackRack) {
      if (editingTrackId !== null) {
        setEditingTrackId(null);
        setEditingTrackDraft('');
        cancelTrackRenameRef.current = false;
        restoreFocusTrackIdRef.current = null;
        restoreFocusTrackPointerModalityRef.current = false;
        trackRenameOpenedViaPointerRef.current = false;
      }
    }
    if (!isActive) {
      setBpmDraft(null);
      setStartBarDraft(null);
      setEndBarDraft(null);
      cancelBpmDraftRef.current = false;
      cancelStartBarDraftRef.current = false;
      cancelEndBarDraftRef.current = false;
    }
  }, [editingTrackId, isActive, showTrackRack]);

  // Clear any prior upload/error notice & close stale import modal when a new AI candidate batch arrives (`[R5-LOW-01]`, `[R3-MED-02]`)
  useEffect(() => {
    const firstCand = safeCandidates[0] ?? null;
    if (firstCand !== lastFirstCandidateRef.current) {
      lastFirstCandidateRef.current = firstCand;
      if (firstCand !== null) {
        setUploadNotice(null);
        if (pendingImport !== null) {
          closeImportModal();
        }
      }
    }
  }, [closeImportModal, pendingImport, safeCandidates]);

  const dragRef = useRef<DragInteraction | null>(null);
  const setActiveDrag = useCallback((next: DragInteraction | null) => {
    dragRef.current = next;
    const el = canvasRef.current;
    if (el) {
      if (next !== null) {
        el.setAttribute('data-pointer-dragging', 'true');
      } else {
        el.removeAttribute('data-pointer-dragging');
      }
    }
  }, []);
  useEffect(() => {
    return () => {
      canvasRef.current?.removeAttribute('data-pointer-dragging');
    };
  }, []);
  const [dragRenderTick, setDragRenderTick] = useState<number>(0);
  const lastHandledFocusRevisionRef = useRef<number>(0);
  const [candidateCenterSeq, setCandidateCenterSeq] = useState<number>(0);
  const lastHandledCandidateCenterKeyRef = useRef<string>('0:0');

  const totalWorkspaceNotes = useMemo(
    () =>
      safeTracks.reduce(
        (sum, trk) => sum + (Array.isArray(trk.notes) ? trk.notes.length : 0),
        0,
      ),
    [safeTracks],
  );

  const activeTrack: DawTrackSnapshot | undefined = useMemo(
    () => safeTracks.find((t) => t.id === session?.activeTrackId) ?? safeTracks[0],
    [safeTracks, session?.activeTrackId],
  );

  // Memoized O(1) note lookup map — rebuilt only when `activeTrack` snapshot mutates
  const activeNoteById = useMemo(() => {
    const map = new Map<string, MidiNote>();
    if (activeTrack && Array.isArray(activeTrack.notes)) {
      for (const n of activeTrack.notes) {
        if (n && typeof n.id === 'string') {
          map.set(n.id, n);
        }
      }
    }
    return map;
  }, [activeTrack]);

  const safePpqn = useMemo(
    () =>
      typeof session?.ppqn === 'number' && Number.isFinite(session.ppqn) && session.ppqn >= 1
        ? Math.min(32767, Math.round(session.ppqn))
        : 480,
    [session?.ppqn],
  );

  const safeQuantDiv: 4 | 8 | 16 | 32 =
    session?.quantizeDivision === 4 ||
    session?.quantizeDivision === 8 ||
    session?.quantizeDivision === 16 ||
    session?.quantizeDivision === 32
      ? session.quantizeDivision
      : 16;

  const safeBpm =
    typeof session?.bpm === 'number' && Number.isFinite(session.bpm)
      ? Math.max(20, Math.min(300, Math.round(session.bpm * 10) / 10))
      : 120;

  const safeToolbarTsDen =
    typeof session?.timeSigDenominator === 'number' &&
    Number.isFinite(session.timeSigDenominator) &&
    session.timeSigDenominator >= 1
      ? Math.min(64, Math.round(session.timeSigDenominator))
      : 4;
  const safeToolbarTsNum =
    typeof session?.timeSigNumerator === 'number' &&
    Number.isFinite(session.timeSigNumerator) &&
    session.timeSigNumerator >= 1
      ? Math.min(64, Math.round(session.timeSigNumerator))
      : 4;
  const safeTimeSigLabel = `${safeToolbarTsNum}/${safeToolbarTsDen}`;

  const safeStartMeasure =
    typeof session?.targetRange?.startMeasure === 'number' &&
    Number.isFinite(session.targetRange.startMeasure)
      ? Math.max(1, Math.min(1024, Math.round(session.targetRange.startMeasure)))
      : 1;
  const safeEndMeasure =
    typeof session?.targetRange?.endMeasure === 'number' &&
    Number.isFinite(session.targetRange.endMeasure)
      ? Math.max(
          safeStartMeasure,
          Math.min(safeStartMeasure + 7, Math.round(session.targetRange.endMeasure)),
        )
      : Math.min(1031, safeStartMeasure + 3);

  const safeTargetRange = useMemo(() => {
    const tr = session?.targetRange;
    if (
      tr &&
      typeof tr === 'object' &&
      typeof tr.pastWindowTicks === 'function' &&
      typeof tr.futureWindowTicks === 'function' &&
      typeof tr.startTick === 'number' &&
      Number.isFinite(tr.startTick) &&
      typeof tr.endTick === 'number' &&
      Number.isFinite(tr.endTick) &&
      typeof tr.startMeasure === 'number' &&
      Number.isFinite(tr.startMeasure) &&
      typeof tr.endMeasure === 'number' &&
      Number.isFinite(tr.endMeasure)
    ) {
      return tr;
    }
    const fallbackBeats =
      typeof session?.beatsPerMeasure === 'number' &&
      Number.isFinite(session.beatsPerMeasure) &&
      session.beatsPerMeasure > 0
        ? Math.max(0.0625, Math.min(256, session.beatsPerMeasure))
        : 4;
    return new MeasureRange(safeStartMeasure, safeEndMeasure, safePpqn, fallbackBeats);
  }, [session?.targetRange, session?.beatsPerMeasure, safeStartMeasure, safeEndMeasure, safePpqn]);

  const pixelsPerTick = useMemo(
    () => clampPixelsPerBeat(pixelsPerBeat) / safePpqn,
    [pixelsPerBeat, safePpqn],
  );

  const centerViewportOnNotes = useCallback(
    (notes: readonly MidiNote[], startTickHint?: number) => {
      if (Array.isArray(notes) && notes.length > 0) {
        const sortedPitches = notes
          .filter((n) => Boolean(n && typeof n.pitch === 'number' && Number.isFinite(n.pitch)))
          .map((n) => n.pitch)
          .sort((a, b) => a - b);
        if (sortedPitches.length > 0) {
          const medianPitch = sortedPitches[Math.floor(sortedPitches.length / 2)];
          const liveRectH = canvasContainerRef.current?.getBoundingClientRect().height ?? 0;
          const effectiveHeight =
            liveRectH > 0 ? Math.max(64, Math.floor(liveRectH)) : canvasSize.height;
          const gridHeight = Math.max(
            1,
            effectiveHeight - RULER_HEIGHT - computeVelocityLaneHeight(effectiveHeight),
          );
          const targetY = (127 - medianPitch) * rowHeight - gridHeight * 0.5;
          const maxScrollY = Math.max(0, TOTAL_PITCHES * rowHeight - gridHeight);
          setScrollPxY(clampScrollPxY(targetY, maxScrollY, 0));
        }
      }
      if (typeof startTickHint === 'number' && Number.isFinite(startTickHint)) {
        setScrollTickX(clampScrollTickX(startTickHint - safePpqn * 4, safePpqn, 0));
      }
    },
    [canvasSize.height, rowHeight, safePpqn],
  );

  // Auto-center strictly ONCE per `focusRevision` increment while visible (never on 40Hz playback ticks or edits!)
  useEffect(() => {
    if (
      !isActive ||
      focusRevision <= 0 ||
      focusRevision === lastHandledFocusRevisionRef.current ||
      !activeTrack
    ) {
      return;
    }
    lastHandledFocusRevisionRef.current = focusRevision;
    const rangeNotes = queryOverlappingNotesInSnapshot(
      activeTrack,
      safeTargetRange.startTick,
      safeTargetRange.endTick,
    );
    centerViewportOnNotes(
      rangeNotes.length > 0 ? rangeNotes : Array.isArray(activeTrack.notes) ? activeTrack.notes : [],
      safeTargetRange.startTick,
    );
  }, [
    isActive,
    focusRevision,
    activeTrack,
    safeTargetRange.startTick,
    safeTargetRange.endTick,
    centerViewportOnNotes,
  ]);

  // Auto-center strictly ONCE per `candidateFocusRevision` or `candidateCenterSeq` increment onto `activeCandidateRange` while visible (`[R6-BUG-01]`, `[R7-MED-03]`)
  useEffect(() => {
    const centerKey = `${candidateFocusRevision}:${candidateCenterSeq}`;
    if (
      !isActive ||
      (candidateFocusRevision <= 0 && candidateCenterSeq <= 0) ||
      centerKey === lastHandledCandidateCenterKeyRef.current
    ) {
      return;
    }
    const candRange = session?.activeCandidateRange;
    const candTrack = candRange
      ? (safeTracks.find((t) => t.id === candRange.trackId) ?? activeTrack)
      : activeTrack;
    if (!candTrack) {
      return;
    }
    lastHandledCandidateCenterKeyRef.current = centerKey;
    const startTick =
      candRange && typeof candRange.startTick === 'number' && Number.isFinite(candRange.startTick)
        ? candRange.startTick
        : safeTargetRange.startTick;
    const endTick =
      candRange && typeof candRange.endTick === 'number' && Number.isFinite(candRange.endTick)
        ? candRange.endTick
        : safeTargetRange.endTick;
    const rangeNotes = queryOverlappingNotesInSnapshot(candTrack, startTick, endTick);
    centerViewportOnNotes(
      rangeNotes.length > 0 ? rangeNotes : Array.isArray(candTrack.notes) ? candTrack.notes : [],
      startTick,
    );
  }, [
    isActive,
    candidateFocusRevision,
    candidateCenterSeq,
    session?.activeCandidateRange,
    safeTracks,
    activeTrack,
    safeTargetRange.startTick,
    safeTargetRange.endTick,
    centerViewportOnNotes,
  ]);

  // Sync responsive default for Track Rack across `< 768px` mobile viewports unless manually overridden
  useEffect(() => {
    const syncRackByViewport = () => {
      if (!userToggledTrackRackRef.current) {
        setShowTrackRack(window.innerWidth >= 768);
      }
    };
    window.addEventListener('resize', syncRackByViewport);
    return () => window.removeEventListener('resize', syncRackByViewport);
  }, []);

  // Observe canvas container size cleanly (ResizeObserver + window resize + initial measure)
  useLayoutEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;

    const measure = () => {
      if (!userToggledTrackRackRef.current && typeof window !== 'undefined' && window.innerWidth > 0) {
        const shouldOpen = window.innerWidth >= 768;
        setShowTrackRack((prev) => (prev === shouldOpen ? prev : shouldOpen));
      }
      const rect = el.getBoundingClientRect();
      const rawW = rect.width || el.clientWidth;
      const rawH = rect.height || el.clientHeight;
      const w = rawW > 0 ? Math.max(160, Math.floor(rawW)) : 960;
      const h = rawH > 0 ? Math.max(64, Math.floor(rawH)) : 540;
      setCanvasSize((prev) =>
        prev.width === w && prev.height === h ? prev : { width: w, height: h },
      );
    };

    measure();
    window.addEventListener('resize', measure);

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => {
        measure();
      });
      observer.observe(el);
      return () => {
        window.removeEventListener('resize', measure);
        observer.disconnect();
      };
    }

    return () => {
      window.removeEventListener('resize', measure);
    };
  }, [isActive, showTrackRack, hasInteractiveCanvas]);

  // Page-advance horizontal scroll during playback when playhead crosses 92% of visible grid
  // without hijacking manual wheel scrolling (`PianoRoll.java` page-follow parity)
  const prevPlayheadTickRef = useRef<number>(safePlayheadTick);
  useEffect(() => {
    const prevTick = prevPlayheadTickRef.current;
    prevPlayheadTickRef.current = safePlayheadTick;
    if (!isActive || !safeIsPlaying || dragRef.current) return;
    const gridWidth = Math.max(1, canvasSize.width - KEYBOARD_WIDTH);
    if (pixelsPerTick <= 0) return;
    const visibleSpanTicks = gridWidth / pixelsPerTick;
    if (visibleSpanTicks <= 0) return;

    const effScrollX = clampScrollTickX(scrollTickX, safePpqn, 0);
    const pageRightThreshold = effScrollX + visibleSpanTicks * 0.92;
    if (
      prevTick >= effScrollX &&
      prevTick <= pageRightThreshold &&
      safePlayheadTick > pageRightThreshold
    ) {
      setScrollTickX(
        clampScrollTickX(safePlayheadTick - visibleSpanTicks * 0.12, safePpqn, effScrollX),
      );
    } else if (safePlayheadTick === 0 && prevTick > 0 && effScrollX > 0) {
      setScrollTickX(0);
    }
  }, [canvasSize.width, isActive, pixelsPerTick, safeIsPlaying, safePlayheadTick, safePpqn, scrollTickX]);

  const exportMidiRef = useRef<() => void>(() => {});
  const cancelActiveDragRef = useRef<() => void>(() => {});
  const pointerDownButtonsRef = useRef<number>(0);
  const pointerPressedElementRef = useRef<HTMLButtonElement | null>(null);

  const isPointerClickEvent = (e: React.MouseEvent<HTMLButtonElement>): boolean => {
    if (e.detail > 0) return true;
    const pressed = pointerPressedElementRef.current;
    return pressed !== null && pressed === e.currentTarget;
  };

  useEffect(() => {
    let pressedFallbackTimer: number | null = null;
    const clearPressedFallbackTimer = () => {
      if (pressedFallbackTimer !== null) {
        window.clearTimeout(pressedFallbackTimer);
        pressedFallbackTimer = null;
      }
    };
    const clearPointerPressed = () => {
      clearPressedFallbackTimer();
      pointerPressedElementRef.current = null;
    };
    const onWindowBlur = () => {
      if (
        pointerFocusedButtonRef.current !== null &&
        document.activeElement === pointerFocusedButtonRef.current
      ) {
        blurredPointerButtonRef.current = pointerFocusedButtonRef.current;
      }
    };
    const onWindowFocusIn = (ev: FocusEvent) => {
      canvasHadRecentPointerRef.current = false;
      const pendingBtn = pendingPointerFocusButtonRef.current;
      pendingPointerFocusButtonRef.current = null;
      const blurredBtn = blurredPointerButtonRef.current;
      blurredPointerButtonRef.current = null;
      if (
        ev.target instanceof HTMLButtonElement &&
        ((pointerPressedElementRef.current !== null &&
          ev.target === pointerPressedElementRef.current) ||
          (pendingBtn !== null && ev.target === pendingBtn) ||
          (blurredBtn !== null && ev.target === blurredBtn) ||
          (pointerFocusedButtonRef.current !== null &&
            ev.target === pointerFocusedButtonRef.current))
      ) {
        pointerFocusedButtonRef.current = ev.target;
      } else {
        pointerFocusedButtonRef.current = null;
      }
    };
    const onWindowPointerDownCapture = (ev: PointerEvent) => {
      clearPressedFallbackTimer();
      const isHandledPointerButton =
        ev.pointerType === 'touch' || ev.button === 0 || ev.button === 1 || ev.button === 2;
      if (!isHandledPointerButton) {
        return;
      }
      lastInputModalityRef.current = 'pointer';
      if (ev.target !== canvasRef.current) {
        canvasHadRecentPointerRef.current = false;
      }
      const isPrimary = ev.button === 0;
      const targetEl = ev.target instanceof Element ? ev.target : null;
      const rawBtn = targetEl?.closest('button') ?? null;
      const enabledBtn =
        isPrimary && rawBtn instanceof HTMLButtonElement && !rawBtn.disabled ? rawBtn : null;
      pointerPressedElementRef.current = enabledBtn;
      if (enabledBtn && document.activeElement === enabledBtn) {
        pointerFocusedButtonRef.current = enabledBtn;
      } else if (isPrimary && !enabledBtn) {
        pointerFocusedButtonRef.current = null;
        blurredPointerButtonRef.current = null;
        pendingPointerFocusButtonRef.current = null;
      }
    };
    const onWindowPointerUpCapture = (ev: PointerEvent) => {
      const pressed = pointerPressedElementRef.current;
      if (!pressed) return;
      const targetEl = ev.target instanceof Element ? ev.target : null;
      if (!targetEl || !pressed.contains(targetEl) || pressed.disabled || !pressed.isConnected) {
        clearPointerPressed();
        return;
      }
      clearPressedFallbackTimer();
      pressedFallbackTimer = window.setTimeout(() => {
        pressedFallbackTimer = null;
        pointerPressedElementRef.current = null;
      }, 120);
    };
    const onWindowKeyDownCapture = (ev: KeyboardEvent) => {
      clearPointerPressed();
      if (isImeComposing(ev)) {
        return;
      }
      const isTabArrowNav =
        (ev.key === 'ArrowLeft' ||
          ev.key === 'ArrowRight' ||
          ev.key === 'Home' ||
          ev.key === 'End') &&
        ev.target instanceof HTMLElement &&
        ev.target.getAttribute('role') === 'tab';
      const isNonTextEnter = ev.key === 'Enter' && !isTextEntryControl(ev.target);
      if (ev.key === 'Tab' || isNonTextEnter || isTabArrowNav) {
        pointerFocusedButtonRef.current = null;
        blurredPointerButtonRef.current = null;
        pendingPointerFocusButtonRef.current = null;
        standbySwitchModalityRef.current = null;
        canvasHadRecentPointerRef.current = false;
        lastInputModalityRef.current = 'keyboard';
      } else if (
        ev.code === 'Space' &&
        !isTextEntryControl(ev.target) &&
        !canvasHadRecentPointerRef.current &&
        !(
          pointerFocusedButtonRef.current !== null &&
          ev.target === pointerFocusedButtonRef.current
        ) &&
        isSpaceActivatableControl(ev.target)
      ) {
        standbySwitchModalityRef.current = null;
        lastInputModalityRef.current = 'keyboard';
      }
    };
    window.addEventListener('blur', onWindowBlur);
    window.addEventListener('focusin', onWindowFocusIn);
    window.addEventListener('pointerdown', onWindowPointerDownCapture, true);
    window.addEventListener('pointerup', onWindowPointerUpCapture, true);
    window.addEventListener('pointercancel', clearPointerPressed, true);
    window.addEventListener('click', clearPointerPressed);
    window.addEventListener('keydown', onWindowKeyDownCapture, true);
    return () => {
      clearPressedFallbackTimer();
      window.removeEventListener('blur', onWindowBlur);
      window.removeEventListener('focusin', onWindowFocusIn);
      window.removeEventListener('pointerdown', onWindowPointerDownCapture, true);
      window.removeEventListener('pointerup', onWindowPointerUpCapture, true);
      window.removeEventListener('pointercancel', clearPointerPressed, true);
      window.removeEventListener('click', clearPointerPressed);
      window.removeEventListener('keydown', onWindowKeyDownCapture, true);
    };
  }, []);

  // Cancel any active pointer drag if Piano Roll becomes inactive, switches controller, opens import modal,
  // or experiences an external track/note-list mutation (`[R3-MED-03]`).
  useEffect(() => {
    const drag = dragRef.current;
    if (!drag) return;
    if (!isActive || !interactiveCtrl || isImportModalRendered) {
      cancelActiveDragRef.current();
      return;
    }
    if (
      drag.kind === 'move' ||
      drag.kind === 'resize' ||
      drag.kind === 'marquee' ||
      drag.kind === 'velocity' ||
      drag.kind === 'draw-create' ||
      drag.kind === 'touch-draw-tap'
    ) {
      const curNotes = activeTrack && Array.isArray(activeTrack.notes) ? activeTrack.notes : null;
      if (
        !activeTrack ||
        activeTrack.id !== drag.trackId ||
        !curNotes ||
        !areNoteSnapshotsIdentical(drag.notesSnapshotRef, curNotes)
      ) {
        cancelActiveDragRef.current();
      }
    }
  }, [activeTrack, interactiveCtrl, isActive, isImportModalRendered]);

  // Keyboard shortcuts strictly scoped when Piano Roll is active; focused `<button>` / `[role="tab"]`
  // never block editing hotkeys (`Delete`/`Backspace`/`Ctrl+Z/Y/A/C/V/S`), while `Space` respects button activation.
  useEffect(() => {
    if (!isActive || !interactiveCtrl || isImportModalRendered) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (isImeComposing(e)) {
        return;
      }
      if (document.querySelector('[role="dialog"]')) {
        return;
      }
      if (e.key === 'Escape' && dragRef.current !== null) {
        e.preventDefault();
        cancelActiveDragRef.current();
        return;
      }
      if (isTextEntryControl(e.target)) {
        return;
      }

      if (e.code === 'Space') {
        const isPointerFocusedBtn =
          pointerFocusedButtonRef.current !== null && e.target === pointerFocusedButtonRef.current;
        if (
          !canvasHadRecentPointerRef.current &&
          !isPointerFocusedBtn &&
          isSpaceActivatableControl(e.target)
        ) {
          return;
        }
        e.preventDefault();
        if (safeIsPlaying) interactiveCtrl.pause();
        else void interactiveCtrl.play();
        return;
      }

      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        if (dragRef.current !== null) cancelActiveDragRef.current();
        interactiveCtrl.deleteSelectedNotes();
        onProjectMutated?.();
        return;
      }

      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        if (dragRef.current !== null) cancelActiveDragRef.current();
        const undoHadKeyboardFocus =
          document.activeElement === undoButtonRef.current &&
          pointerFocusedButtonRef.current !== undoButtonRef.current;
        interactiveCtrl.undo();
        if (undoHadKeyboardFocus && !interactiveCtrl.getSnapshot().canUndo) {
          pendingToolbarFocusTargetRef.current = 'redo';
        }
        onProjectMutated?.();
      } else if (
        (mod && e.key.toLowerCase() === 'y') ||
        (mod && e.shiftKey && e.key.toLowerCase() === 'z')
      ) {
        e.preventDefault();
        if (dragRef.current !== null) cancelActiveDragRef.current();
        const redoHadKeyboardFocus =
          document.activeElement === redoButtonRef.current &&
          pointerFocusedButtonRef.current !== redoButtonRef.current;
        interactiveCtrl.redo();
        if (redoHadKeyboardFocus && !interactiveCtrl.getSnapshot().canRedo) {
          pendingToolbarFocusTargetRef.current = 'undo';
        }
        onProjectMutated?.();
      } else if (mod && e.key.toLowerCase() === 'a') {
        e.preventDefault();
        if (dragRef.current !== null) cancelActiveDragRef.current();
        interactiveCtrl.selectAllNotesOnActiveTrack();
      } else if (mod && e.key.toLowerCase() === 'c') {
        e.preventDefault();
        interactiveCtrl.copySelectedNotes();
      } else if (mod && e.key.toLowerCase() === 'v') {
        e.preventDefault();
        if (dragRef.current !== null) cancelActiveDragRef.current();
        interactiveCtrl.pasteNotesAtPlayhead();
        onProjectMutated?.();
      } else if (mod && !e.shiftKey && e.key.toLowerCase() === 's') {
        e.preventDefault();
        exportMidiRef.current();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [interactiveCtrl, isActive, isImportModalRendered, onProjectMutated, safeIsPlaying, toolMode, updateCanvasCursor]);

  // Handle MIDI File Upload (`MIDIをアップロード`)
  const handleMidiFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const viaPointer = uploadOpenedViaPointerRef.current;
    uploadOpenedViaPointerRef.current = false;
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file || !interactiveCtrl) return;

    const targetCtrl = interactiveCtrl;
    const reqSeq = ++midiUploadSeqRef.current;

    try {
      const parsed = await targetCtrl.parseUploadedMidiFile(file);
      if (
        reqSeq !== midiUploadSeqRef.current ||
        !isMountedRef.current ||
        !isActiveAndInteractiveRef.current ||
        latestInteractiveCtrlRef.current !== targetCtrl
      ) {
        return;
      }
      const meterHint = formatUnsupportedGenerationMeterNotice(parsed);
      const canonicalTracks = extractCanonicalParsedMidiTracks(parsed, MAX_PARSED_SMF_TRACKS);
      if (canonicalTracks.length === 1) {
        const singleTrack = canonicalTracks[0];
        const importRes = targetCtrl.importParsedMidiTracks(
          parsed,
          [singleTrack.canonicalIndex],
          true,
        );
        if (importRes.importedTracks === 0) {
          setUploadNotice(`${file.name} から配置可能なトラックが見つかりませんでした。`);
        } else {
          if (Array.isArray(singleTrack.notes)) {
            centerViewportOnNotes(singleTrack.notes as readonly MidiNote[], 0);
          }
          const importedBpm =
            typeof parsed.bpm === 'number' && Number.isFinite(parsed.bpm)
              ? Math.max(20, Math.min(300, Math.round(parsed.bpm * 10) / 10))
              : 120;
          setUploadNotice(
            `${file.name} を読み込みました (${importRes.importedNotes} ノート / ${importedBpm} BPM)${meterHint}`,
          );
          onProjectMutated?.();
        }
      } else if (canonicalTracks.length > 1) {
        openImportModal({ fileName: file.name, parsed }, uploadMidiButtonRef.current, viaPointer);
      } else {
        setUploadNotice('選択されたMIDIファイル内に音符イベントが含まれていません。');
      }
    } catch (err) {
      if (
        reqSeq !== midiUploadSeqRef.current ||
        !isMountedRef.current ||
        !isActiveAndInteractiveRef.current ||
        latestInteractiveCtrlRef.current !== targetCtrl
      ) {
        return;
      }
      const msg = err instanceof Error ? err.message : 'MIDIファイルの解析に失敗しました。';
      setUploadNotice(`エラー: ${msg}`);
    }
  };

  const handleConfirmModalImport = (
    selectedIndices: readonly number[],
    replaceExisting: boolean,
  ) => {
    if (!pendingImport || !interactiveCtrl) return;
    const importRes = interactiveCtrl.importParsedMidiTracks(
      pendingImport.parsed,
      selectedIndices,
      replaceExisting,
    );
    if (importRes.importedTracks === 0) {
      setUploadNotice(
        !replaceExisting
          ? `${pendingImport.fileName} を追加できませんでした（最大64トラックの上限に達しています）。`
          : `${pendingImport.fileName} から配置可能なトラックが見つかりませんでした。`,
      );
      closeImportModal();
      return;
    }
    const selectedSet = new Set(selectedIndices);
    const canonicalTracks = extractCanonicalParsedMidiTracks(
      pendingImport.parsed,
      MAX_PARSED_SMF_TRACKS,
    );
    const firstSelected = canonicalTracks.find(
      (t) => selectedSet.has(t.canonicalIndex) && Array.isArray(t.notes),
    );
    if (firstSelected && Array.isArray(firstSelected.notes)) {
      centerViewportOnNotes(firstSelected.notes as readonly MidiNote[], 0);
    }
    if (pendingImport.isSamplePreset && replaceExisting) {
      interactiveCtrl.setTargetMeasureRange(5, 8);
    }
    const meterHint = replaceExisting
      ? formatUnsupportedGenerationMeterNotice(pendingImport.parsed)
      : '';
    setUploadNotice(
      pendingImport.isSamplePreset
        ? `検証用サンプルアレンジ (${importRes.importedNotes} ノート) をピアノロールに配置しました。`
        : `${pendingImport.fileName} (${importRes.importedNotes} ノート) をピアノロールに配置しました。${meterHint}`,
    );
    closeImportModal();
    onProjectMutated?.();
  };

  const handleExportMidi = () => {
    let url = '';
    try {
      let bytes: Uint8Array | null = null;
      const timeSig = {
        numerator: safeToolbarTsNum,
        denominator: safeToolbarTsDen,
      };
      if (interactiveCtrl) {
        bytes = interactiveCtrl.exportProjectMidiBytes();
      } else if (safeTracks.some((t) => Array.isArray(t.notes) && t.notes.length > 0)) {
        bytes = StandardMidiCodec.encodeMultiTrackMidi(
          safeTracks,
          safePpqn,
          safeBpm,
          session?.beatsPerMeasure,
          timeSig,
        );
      } else if (safeCandidates.length > 0) {
        const safeActIdx =
          typeof session?.activeCandidateIndex === 'number' &&
          Number.isFinite(session.activeCandidateIndex)
            ? Math.max(0, Math.trunc(session.activeCandidateIndex))
            : 0;
        const candBytes =
          safeCandidates[safeActIdx]?.rawMidiBytes ?? safeCandidates[0]?.rawMidiBytes ?? null;
        bytes = candBytes instanceof Uint8Array ? candBytes : null;
      } else if (activeTrack) {
        bytes = StandardMidiCodec.encodeMultiTrackMidi(
          safeTracks,
          safePpqn,
          safeBpm,
          session?.beatsPerMeasure,
          timeSig,
        );
      }
      if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) return;
      const cleanCopy = new Uint8Array(bytes.byteLength);
      cleanCopy.set(bytes);

      const blob = new Blob([cleanCopy], { type: 'audio/midi' });
      url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'compass_ai_session.mid';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch {
      // Never crash PianoRollView on hostile or corrupted external controller snapshot bytes
    } finally {
      if (url) {
        try {
          URL.revokeObjectURL(url);
        } catch {
          // Ignore URL revocation errors
        }
      }
    }
  };
  exportMidiRef.current = handleExportMidi;

  const handleSelectNotesInTargetRange = useCallback(() => {
    if (!interactiveCtrl || !activeTrack) return;
    if (dragRef.current !== null) cancelActiveDragRef.current();
    const overlapping = queryOverlappingNotesInSnapshot(
      activeTrack,
      safeTargetRange.startTick,
      safeTargetRange.endTick,
    );
    const ids = overlapping
      .filter(
        (n) =>
          Boolean(n && typeof n.id === 'string') &&
          n.endTick > safeTargetRange.startTick &&
          n.startTick < safeTargetRange.endTick,
      )
      .map((n) => n.id);
    interactiveCtrl.setSelectedNotes(ids);
  }, [activeTrack, interactiveCtrl, safeTargetRange.endTick, safeTargetRange.startTick]);

  const selectedNotesTelemetry = useMemo(() => {
    const safeActiveName =
      activeTrack && typeof activeTrack.name === 'string' && activeTrack.name.trim()
        ? activeTrack.name.trim()
        : 'Track 1';
    if (safeSelectedNoteIds.size === 0) {
      return `${safeActiveName} · 対象 ${safeStartMeasure}–${safeEndMeasure}小節`;
    }
    const picked: MidiNote[] = [];
    for (const id of safeSelectedNoteIds) {
      const n = activeNoteById.get(id);
      if (n) picked.push(n);
    }
    if (picked.length === 0) {
      return `${safeActiveName} · 対象 ${safeStartMeasure}–${safeEndMeasure}小節`;
    }
    if (picked.length === 1) {
      const n = picked[0];
      const safeBeats =
        typeof session?.beatsPerMeasure === 'number' &&
        Number.isFinite(session.beatsPerMeasure) &&
        session.beatsPerMeasure > 0
          ? Math.max(0.0625, Math.min(256, session.beatsPerMeasure))
          : 4;
      const mTicks = Math.max(1, Math.round(safePpqn * safeBeats));
      const barIdx = Math.floor(Math.max(0, n.startTick) / mTicks) + 1;
      const beatInBar = ((Math.max(0, n.startTick) % mTicks) / safePpqn + 1).toFixed(2);
      return `${safeActiveName} · Pitch ${midiPitchToName(n.pitch)} (${n.pitch}) · Bar ${barIdx} Beat ${beatInBar} (${n.startTick}t) · Dur ${n.durationTicks}t · Vel ${n.velocity}`;
    }
    let minP = 127;
    let maxP = 0;
    for (const n of picked) {
      if (n.pitch < minP) minP = n.pitch;
      if (n.pitch > maxP) maxP = n.pitch;
    }
    return `${safeActiveName} · ${picked.length} notes selected (${midiPitchToName(minP)}–${midiPitchToName(maxP)})`;
  }, [
    activeNoteById,
    activeTrack,
    safeEndMeasure,
    safePpqn,
    safeSelectedNoteIds,
    safeStartMeasure,
    session?.beatsPerMeasure,
  ]);

  // =========================================================================
  // High-DPI Canvas Rendering (Piano Keys + Ruler + Grid + Notes + Velocity)
  // =========================================================================

  useLayoutEffect(() => {
    if (!isActive) return;
    const canvas = canvasRef.current;
    if (!canvas || !activeTrack) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const liveRect = canvasContainerRef.current?.getBoundingClientRect();
    const liveW = liveRect && liveRect.width > 0 ? liveRect.width : 0;
    const liveH = liveRect && liveRect.height > 0 ? liveRect.height : 0;
    const w = liveW > 0 ? Math.max(160, Math.floor(liveW)) : canvasSize.width;
    const h = liveH > 0 ? Math.max(64, Math.floor(liveH)) : canvasSize.height;
    const targetW = Math.round(w * dpr);
    const targetH = Math.round(h * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    ctx.save();
    ctx.scale(targetW / w, targetH / h);
    ctx.clearRect(0, 0, w, h);

    const velocityHeight = computeVelocityLaneHeight(h);
    const gridLeft = KEYBOARD_WIDTH;
    const gridTop = RULER_HEIGHT;
    const gridBottom = h - velocityHeight;
    const gridWidth = Math.max(1, w - KEYBOARD_WIDTH);
    const gridHeight = Math.max(1, gridBottom - gridTop);
    const maxCanvasScrollY = Math.max(0, TOTAL_PITCHES * rowHeight - gridHeight);
    const effScrollTickX = clampScrollTickX(scrollTickX, safePpqn, 0);
    const effScrollPxY = Math.round(clampScrollPxY(scrollPxY, maxCanvasScrollY, 0));

    const startVisibleTick = effScrollTickX;
    const endVisibleTick = effScrollTickX + gridWidth / pixelsPerTick;
    const safeBeatsPerMeasure =
      typeof session?.beatsPerMeasure === 'number' &&
      Number.isFinite(session.beatsPerMeasure) &&
      session.beatsPerMeasure > 0
        ? Math.max(0.0625, Math.min(256, session.beatsPerMeasure))
        : 4;
    const safeTimeSigDenom =
      typeof session?.timeSigDenominator === 'number' &&
      Number.isFinite(session.timeSigDenominator) &&
      session.timeSigDenominator >= 1
        ? Math.min(64, Math.round(session.timeSigDenominator))
        : 4;
    const measureTicks = Math.max(1, Math.round(safePpqn * safeBeatsPerMeasure));
    const rawDenomUnits =
      typeof session?.timeSigNumerator === 'number' &&
      Number.isFinite(session.timeSigNumerator) &&
      session.timeSigNumerator >= 1
        ? Math.round(session.timeSigNumerator)
        : Math.max(1, Math.round((safeBeatsPerMeasure * safeTimeSigDenom) / 4));
    const effectiveDenomUnits = Math.max(1, Math.min(64, rawDenomUnits));
    const isCompoundEighth =
      safeTimeSigDenom === 8 && effectiveDenomUnits % 3 === 0;
    const beatCount = isCompoundEighth ? effectiveDenomUnits / 3 : effectiveDenomUnits;
    const exactBeatTicks = isCompoundEighth
      ? safePpqn * 1.5
      : safeTimeSigDenom === 8
        ? safePpqn / 2
        : safeTimeSigDenom === 16
          ? safePpqn / 4
          : safeTimeSigDenom === 32
            ? safePpqn / 8
            : safeTimeSigDenom === 2
              ? safePpqn * 2
              : safeTimeSigDenom === 1
                ? safePpqn * 4
                : safePpqn;
    const exactSnapTicks = Math.max(
      1,
      (safePpqn * 4) / (safeQuantDiv * (session?.isTriplet ? 1.5 : 1)),
    );
    const snapTicks = interactiveCtrl
      ? interactiveCtrl.getSnapStepTicks()
      : Math.max(1, Math.round(exactSnapTicks));
    const minResizableTicks = Math.max(2, Math.round((safePpqn * 24) / 480));

    const tickToScreenX = (tick: number) => gridLeft + (tick - effScrollTickX) * pixelsPerTick;
    const pitchToScreenY = (pitch: number) => gridTop + (127 - pitch) * rowHeight - effScrollPxY;

    // 1. Pitch Rows Background & Horizontal Hairlines
    ctx.save();
    ctx.beginPath();
    ctx.rect(gridLeft, gridTop, gridWidth, gridHeight);
    ctx.clip();

    const rawHighPitch = 127 - Math.floor(effScrollPxY / rowHeight) + 1;
    const rawLowPitch = 127 - Math.ceil((effScrollPxY + gridHeight) / rowHeight) - 1;
    const highestPitch = Number.isFinite(rawHighPitch)
      ? Math.max(0, Math.min(127, rawHighPitch))
      : 127;
    const lowestPitch = Number.isFinite(rawLowPitch)
      ? Math.max(0, Math.min(127, rawLowPitch))
      : 0;

    for (let pitch = lowestPitch; pitch <= highestPitch; pitch += 1) {
      const y = pitchToScreenY(pitch);
      const pc = pitch % 12;
      const isBlack = BLACK_KEY_PCS.has(pc);
      ctx.fillStyle = isBlack ? 'rgba(10, 10, 14, 0.60)' : 'rgba(20, 20, 26, 0.62)';
      ctx.fillRect(gridLeft, y, gridWidth, rowHeight);

      ctx.strokeStyle = pc === 0 ? 'rgba(255, 255, 255, 0.085)' : 'rgba(255, 255, 255, 0.035)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(gridLeft, y + rowHeight - 0.5);
      ctx.lineTo(w, y + rowHeight - 0.5);
      ctx.stroke();
    }

    // 2. Past / Target / Future Measure Shading
    const pastWin = safeTargetRange.pastWindowTicks(8);
    const futureWin = safeTargetRange.futureWindowTicks(8);

    const drawSpanTint = (fromTick: number, toTick: number, fill: string) => {
      if (toTick <= startVisibleTick || fromTick >= endVisibleTick) return;
      const x1 = Math.max(gridLeft, tickToScreenX(fromTick));
      const x2 = Math.min(w, tickToScreenX(toTick));
      if (x2 > x1) {
        ctx.fillStyle = fill;
        ctx.fillRect(x1, gridTop, x2 - x1, gridHeight);
      }
    };

    drawSpanTint(pastWin.startTick, pastWin.endTick, 'rgba(161, 161, 170, 0.03)');
    drawSpanTint(
      safeTargetRange.startTick,
      safeTargetRange.endTick,
      'rgba(56, 189, 248, 0.07)',
    );
    drawSpanTint(futureWin.startTick, futureWin.endTick, 'rgba(161, 161, 170, 0.03)');

    // 3. Vertical Grid Lines — Measure-Anchored & Beat-Index Quantized so All Meters & PPQNs Never Drift
    const rawFirstGridM = Math.floor(startVisibleTick / measureTicks);
    const rawLastGridM = Math.ceil(endVisibleTick / measureTicks);
    const hasFiniteGridSpan =
      Number.isFinite(rawFirstGridM) && Number.isFinite(rawLastGridM) && measureTicks > 0;
    const firstGridMeasure = hasFiniteGridSpan
      ? Math.max(0, Math.min(MAX_RENDERED_MEASURE_INDEX, rawFirstGridM))
      : 1;
    const lastGridMeasure = hasFiniteGridSpan
      ? Math.max(
          -1,
          Math.min(
            MAX_RENDERED_MEASURE_INDEX,
            firstGridMeasure + MAX_VISIBLE_MEASURES_PER_FRAME,
            rawLastGridM,
          ),
        )
      : 0;

    const renderedBeatRels = new Set<number>();
    const primaryBeatRels: number[] = [];
    for (let b = 1; b < beatCount; b += 1) {
      const rel = Math.round((b * measureTicks) / beatCount);
      if (rel > 0 && rel < measureTicks && !renderedBeatRels.has(rel)) {
        renderedBeatRels.add(rel);
        primaryBeatRels.push(rel);
      }
    }
    const compoundSubBeatRels: number[] = [];
    const compoundSubBeatSpacing = isCompoundEighth ? measureTicks / effectiveDenomUnits : 0;
    if (isCompoundEighth && compoundSubBeatSpacing * pixelsPerTick >= 7) {
      for (let s = 1; s < effectiveDenomUnits; s += 1) {
        if (s % 3 === 0) continue;
        const rel = Math.round((s * measureTicks) / effectiveDenomUnits);
        if (rel > 0 && rel < measureTicks && !renderedBeatRels.has(rel)) {
          renderedBeatRels.add(rel);
          compoundSubBeatRels.push(rel);
        }
      }
    }

    if (exactSnapTicks * pixelsPerTick >= 7) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.035)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const maxSubStepsPerBar = Math.min(1024, Math.floor((measureTicks - 0.5) / exactSnapTicks));
      for (let m = firstGridMeasure; m <= lastGridMeasure; m += 1) {
        const mStart = m * measureTicks;
        const mEnd = mStart + measureTicks;
        for (let k = 1; k <= maxSubStepsPerBar; k += 1) {
          const roundedRel = Math.round(k * exactSnapTicks);
          if (
            roundedRel <= 1 ||
            roundedRel >= measureTicks - 1 ||
            renderedBeatRels.has(roundedRel) ||
            renderedBeatRels.has(roundedRel - 1) ||
            renderedBeatRels.has(roundedRel + 1)
          ) {
            continue;
          }
          const t = mStart + roundedRel;
          if (t < startVisibleTick || t > endVisibleTick || t >= mEnd) continue;
          const x = Math.round(tickToScreenX(t)) + 0.5;
          ctx.moveTo(x, gridTop);
          ctx.lineTo(x, gridBottom);
        }
      }
      ctx.stroke();
    }

    if (compoundSubBeatRels.length > 0) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let m = firstGridMeasure; m <= lastGridMeasure; m += 1) {
        const mStart = m * measureTicks;
        for (let i = 0; i < compoundSubBeatRels.length; i += 1) {
          const t = mStart + compoundSubBeatRels[i];
          if (t < startVisibleTick || t > endVisibleTick) continue;
          const x = Math.round(tickToScreenX(t)) + 0.5;
          ctx.moveTo(x, gridTop);
          ctx.lineTo(x, gridBottom);
        }
      }
      ctx.stroke();
    }

    if (primaryBeatRels.length > 0) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.075)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let m = firstGridMeasure; m <= lastGridMeasure; m += 1) {
        const mStart = m * measureTicks;
        for (let i = 0; i < primaryBeatRels.length; i += 1) {
          const t = mStart + primaryBeatRels[i];
          if (t < startVisibleTick || t > endVisibleTick) continue;
          const x = Math.round(tickToScreenX(t)) + 0.5;
          ctx.moveTo(x, gridTop);
          ctx.lineTo(x, gridBottom);
        }
      }
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let m = firstGridMeasure; m <= lastGridMeasure; m += 1) {
      const barTick = m * measureTicks;
      if (barTick < startVisibleTick || barTick > endVisibleTick) continue;
      const x = Math.round(tickToScreenX(barTick)) + 0.5;
      ctx.moveTo(x, gridTop);
      ctx.lineTo(x, gridBottom);
    }
    ctx.stroke();

    // Target Range Boundary Lines
    for (const boundaryTick of [safeTargetRange.startTick, safeTargetRange.endTick]) {
      const bx = tickToScreenX(boundaryTick);
      if (bx >= gridLeft && bx <= w) {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(bx, gridTop);
        ctx.lineTo(bx, gridBottom);
        ctx.stroke();
      }
    }

    // 4. Ghost Notes from Other Non-Muted Tracks — O(log N + K) Interval-Tree Overlap Query
    const ghostScreenSpansByPitch = new Map<number, Array<{ x1: number; x2: number }>>();
    for (const trk of safeTracks) {
      if (trk.id === activeTrack.id || trk.isMuted) continue;
      ctx.fillStyle = 'rgba(161, 161, 170, 0.18)';
      const ghostVisible = queryOverlappingNotesInSnapshot(
        trk,
        startVisibleTick,
        endVisibleTick,
      );
      for (const note of ghostVisible) {
        if (
          !note ||
          typeof note.pitch !== 'number' ||
          !Number.isFinite(note.pitch) ||
          typeof note.startTick !== 'number' ||
          !Number.isFinite(note.startTick) ||
          typeof note.durationTicks !== 'number' ||
          !Number.isFinite(note.durationTicks) ||
          note.durationTicks <= 0
        ) {
          continue;
        }
        if (note.pitch < lowestPitch || note.pitch > highestPitch) continue;
        const nx = tickToScreenX(note.startTick);
        const nw = Math.max(3, note.durationTicks * pixelsPerTick - 1);
        const ny = pitchToScreenY(note.pitch) + 2;
        const gh = Math.max(2, rowHeight - 4);
        if (!Number.isFinite(nx) || !Number.isFinite(nw) || !Number.isFinite(ny)) continue;
        fillRoundedCanvasRect(ctx, nx, ny, nw, gh, Math.min(2.5, gh * 0.3));
        let pitchSpans = ghostScreenSpansByPitch.get(note.pitch);
        if (!pitchSpans) {
          pitchSpans = [];
          ghostScreenSpansByPitch.set(note.pitch, pitchSpans);
        }
        pitchSpans.push({ x1: nx, x2: nx + nw });
      }
    }

    const targetScreenX1 = tickToScreenX(safeTargetRange.startTick);
    const targetScreenX2 = tickToScreenX(safeTargetRange.endTick);
    const pastScreenX1 = tickToScreenX(pastWin.startTick);
    const pastScreenX2 = tickToScreenX(pastWin.endTick);
    const futureScreenX1 = tickToScreenX(futureWin.startTick);
    const futureScreenX2 = tickToScreenX(futureWin.endTick);

    // 5. Active Track Notes — O(log N + K) Interval-Tree Overlap Query + Live 60fps Drag Preview
    const drag = dragRef.current;
    const maxProjectTicksForPreview = MAX_SMF_QUARTER_NOTES * safePpqn;
    const dragMovedMap = new Map<
      string,
      { pitch: number; startTick: number; durationTicks: number; velocity: number }
    >();
    if (drag && (drag.kind === 'move' || drag.kind === 'resize') && activeTrack.id === drag.trackId) {
      const liveInitialNotes = drag.initialNotes.filter((init) => activeNoteById.has(init.id));
      const groupMove =
        drag.kind === 'move'
          ? computeClampedGroupMoveDeltas(
              liveInitialNotes,
              drag.currentDeltaTick,
              drag.currentDeltaPitch,
              safePpqn,
            )
          : { deltaTick: 0, deltaPitch: 0 };
      for (const init of liveInitialNotes) {
        if (drag.kind === 'move') {
          dragMovedMap.set(init.id, {
            pitch: Math.max(0, Math.min(127, init.pitch + groupMove.deltaPitch)),
            startTick: Math.max(0, init.startTick + groupMove.deltaTick),
            durationTicks: init.durationTicks,
            velocity: init.velocity,
          });
        } else {
          const minResizeFloor = drag.fineResize ? minResizableTicks : snapTicks;
          const noteMinResize = Math.max(1, Math.min(init.durationTicks, minResizeFloor));
          const maxNoteDur = Math.max(1, maxProjectTicksForPreview - init.startTick);
          dragMovedMap.set(init.id, {
            pitch: init.pitch,
            startTick: init.startTick,
            durationTicks: Math.max(
              noteMinResize,
              Math.min(maxNoteDur, init.durationTicks + drag.currentDeltaTick),
            ),
            velocity: init.velocity,
          });
        }
      }
    }
    let drawCreatePreview: {
      pitch: number;
      startTick: number;
      durationTicks: number;
      velocity: number;
    } | null = null;
    if (drag && drag.kind === 'draw-create' && activeTrack.id === drag.trackId) {
      const minResizeFloor = drag.fineResize ? minResizableTicks : snapTicks;
      const noteMinResize = Math.max(1, Math.min(drag.initialDurationTicks, minResizeFloor));
      const maxNoteDur = Math.max(1, maxProjectTicksForPreview - drag.snappedStartTick);
      drawCreatePreview = {
        pitch: drag.pitch,
        startTick: drag.snappedStartTick,
        durationTicks: Math.max(
          noteMinResize,
          Math.min(maxNoteDur, drag.initialDurationTicks + drag.currentDeltaTick),
        ),
        velocity: drag.velocity,
      };
    }

    ctx.font = `11px ${CANVAS_MONO_FONT}`;
    ctx.textBaseline = 'middle';

    const drawNoteRect = (
      id: string,
      pitch: number,
      startTick: number,
      durationTicks: number,
      velocity: number,
    ) => {
      if (
        !Number.isFinite(pitch) ||
        !Number.isFinite(startTick) ||
        !Number.isFinite(durationTicks) ||
        durationTicks <= 0
      ) {
        return;
      }
      const endTick = startTick + durationTicks;
      if (endTick < startVisibleTick || startTick > endVisibleTick) return;
      if (pitch < lowestPitch - 1 || pitch > highestPitch + 1) return;

      const nx = tickToScreenX(startTick);
      const nw = Math.max(4, durationTicks * pixelsPerTick - 1);
      const ny = pitchToScreenY(pitch) + 1;
      const nh = Math.max(4, rowHeight - 2);
      if (!Number.isFinite(nx) || !Number.isFinite(nw) || !Number.isFinite(ny) || !Number.isFinite(nh)) {
        return;
      }
      const isSelected =
        id === '__draw_preview__' || (!drawCreatePreview && safeSelectedNoteIds.has(id));
      const safeVelocity =
        typeof velocity === 'number' && Number.isFinite(velocity)
          ? Math.max(1, Math.min(127, velocity))
          : 100;

      const safeActiveTrackHex =
        typeof activeTrack.colorHex === 'string' && activeTrack.colorHex
          ? activeTrack.colorHex
          : '#38BDF8';
      const fillHex = isSelected ? '#F4F4F5' : safeActiveTrackHex;
      const fillAlpha = isSelected
        ? 0.96
        : Math.max(0.05, Math.min(1, 0.58 + (safeVelocity / 127) * 0.38));
      const noteRadius = Math.min(4, nh * 0.32, nw * 0.32);
      ctx.fillStyle = fillHex;
      ctx.globalAlpha = fillAlpha;
      fillRoundedCanvasRect(ctx, nx, ny, nw, nh, noteRadius);
      ctx.globalAlpha = 1;

      ctx.strokeStyle = isSelected ? '#38BDF8' : '#09090B';
      ctx.lineWidth = isSelected ? 1.25 : 1;
      strokeRoundedCanvasRect(
        ctx,
        nx + 0.5,
        ny + 0.5,
        Math.max(1, nw - 1),
        Math.max(1, nh - 1),
        Math.max(0, noteRadius - 0.5),
      );

      const label = midiPitchToName(pitch);
      const rawMeasuredW =
        typeof ctx.measureText === 'function' ? ctx.measureText(label)?.width ?? 0 : 0;
      const measuredW =
        Number.isFinite(rawMeasuredW) && rawMeasuredW > 0
          ? Math.ceil(rawMeasuredW)
          : Math.ceil(label.length * 6.8);
      const chipW = Math.min(Math.max(4, nw - 4), Math.max(20, measuredW + 6));
      const minNoteWidthForLabel = Math.max(30, measuredW + 8);
      if (nw >= minNoteWidthForLabel && rowHeight >= 13) {
        const labelX1 = nx + 2;
        const labelX2 = Math.min(nx + nw - 2, nx + 2 + chipW);
        const overlapsTarget = labelX2 > targetScreenX1 && labelX1 < targetScreenX2;
        const fullyInTarget = labelX1 >= targetScreenX1 && labelX2 <= targetScreenX2;
        const overlapsContext =
          (labelX2 > pastScreenX1 && labelX1 < pastScreenX2) ||
          (labelX2 > futureScreenX1 && labelX1 < futureScreenX2);
        const fullyInContext =
          (labelX1 >= pastScreenX1 && labelX2 <= pastScreenX2) ||
          (labelX1 >= futureScreenX1 && labelX2 <= futureScreenX2);

        const ghostSpans = ghostScreenSpansByPitch.get(pitch);
        let minGhostLayers = 0;
        let maxGhostLayers = 0;
        if (ghostSpans && ghostSpans.length > 0) {
          for (const gs of ghostSpans) {
            if (gs.x2 > labelX1 && gs.x1 < labelX2) {
              maxGhostLayers += 1;
              if (gs.x1 <= labelX1 && gs.x2 >= labelX2) {
                minGhostLayers += 1;
              }
            }
          }
        }

        const noteRgb = parseHexRgb(fillHex);
        const isBlackNoteRow = BLACK_KEY_PCS.has(((pitch % 12) + 12) % 12);
        const darkRowRgb: readonly [number, number, number] = isBlackNoteRow
          ? [10, 10, 13]
          : [16, 16, 20];
        const auroraRowRgb: readonly [number, number, number] = isBlackNoteRow
          ? [44, 14, 28]
          : [50, 20, 35];
        const minLum = Math.min(
          computeCompositedNoteLuminance(noteRgb, fillAlpha, pitch, {
            backdropRgb: darkRowRgb,
            inTargetRange: fullyInTarget,
            inContextWindow: !fullyInTarget && fullyInContext,
            ghostStackCount: minGhostLayers,
          }),
          computeCompositedNoteLuminance(noteRgb, fillAlpha, pitch, {
            backdropRgb: auroraRowRgb,
            inTargetRange: fullyInTarget,
            inContextWindow: !fullyInTarget && fullyInContext,
            ghostStackCount: minGhostLayers,
          }),
        );
        const maxLum = Math.max(
          computeCompositedNoteLuminance(noteRgb, fillAlpha, pitch, {
            backdropRgb: darkRowRgb,
            inTargetRange: overlapsTarget,
            inContextWindow: !overlapsTarget && overlapsContext,
            ghostStackCount: maxGhostLayers,
          }),
          computeCompositedNoteLuminance(noteRgb, fillAlpha, pitch, {
            backdropRgb: auroraRowRgb,
            inTargetRange: overlapsTarget,
            inContextWindow: !overlapsTarget && overlapsContext,
            ghostStackCount: maxGhostLayers,
          }),
        );

        const blackContrastMin = (minLum + 0.05) / 0.05;
        const whiteContrastMin = 1.05 / (maxLum + 0.05);
        if (blackContrastMin >= 4.5 && blackContrastMin >= whiteContrastMin) {
          ctx.fillStyle = '#000000';
        } else if (whiteContrastMin >= 4.5) {
          ctx.fillStyle = '#FFFFFF';
        } else {
          // Split backdrop straddling luminance crossover directly beneath the glyph:
          // draw a crisp rounded monochrome label backing chip so WCAG AA contrast stays >= 18:1.
          ctx.fillStyle = '#09090B';
          fillRoundedCanvasRect(ctx, nx + 2, ny + 1, chipW, Math.max(2, nh - 2), 3);
          ctx.fillStyle = '#FFFFFF';
        }
        ctx.fillText(label, nx + 4, ny + nh / 2 + 0.5);
      }
    };

    const visibleNotes = queryOverlappingNotesInSnapshot(
      activeTrack,
      startVisibleTick,
      endVisibleTick,
    );

    for (const note of visibleNotes) {
      if (dragMovedMap.has(note.id)) {
        continue;
      }
      drawNoteRect(note.id, note.pitch, note.startTick, note.durationTicks, note.velocity);
    }

    for (const [id, preview] of dragMovedMap.entries()) {
      drawNoteRect(
        id,
        preview.pitch,
        preview.startTick,
        preview.durationTicks,
        preview.velocity,
      );
    }
    if (drawCreatePreview) {
      drawNoteRect(
        '__draw_preview__',
        drawCreatePreview.pitch,
        drawCreatePreview.startTick,
        drawCreatePreview.durationTicks,
        drawCreatePreview.velocity,
      );
    }

    // Marquee Selection Box
    if (drag && drag.kind === 'marquee' && drag.hasExceededDragThreshold) {
      const mx1 = tickToScreenX(Math.min(drag.startWorldX, drag.currentWorldX));
      const mx2 = tickToScreenX(Math.max(drag.startWorldX, drag.currentWorldX));
      const my1 = gridTop + Math.min(drag.startWorldY, drag.currentWorldY) - effScrollPxY;
      const my2 = gridTop + Math.max(drag.startWorldY, drag.currentWorldY) - effScrollPxY;
      if (
        Number.isFinite(mx1) &&
        Number.isFinite(mx2) &&
        Number.isFinite(my1) &&
        Number.isFinite(my2)
      ) {
        ctx.fillStyle = 'rgba(244, 244, 245, 0.08)';
        fillRoundedCanvasRect(ctx, mx1, my1, mx2 - mx1, my2 - my1, 4);
        ctx.strokeStyle = '#A1A1AA';
        ctx.lineWidth = 1;
        strokeRoundedCanvasRect(ctx, mx1 + 0.5, my1 + 0.5, mx2 - mx1, my2 - my1, 4);
      }
    }

    ctx.restore(); // End Grid Clip

    // 6. Bottom Velocity Lane — O(log N + K) Visible Note Stems (Translucent over Bottom Aurora)
    ctx.save();
    ctx.clearRect(gridLeft, gridBottom, gridWidth, velocityHeight);
    ctx.fillStyle = 'rgba(10, 10, 14, 0.52)';
    ctx.fillRect(gridLeft, gridBottom, gridWidth, velocityHeight);
    ctx.strokeStyle = '#27272A';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, gridBottom + 0.5);
    ctx.lineTo(w, gridBottom + 0.5);
    ctx.stroke();

    ctx.beginPath();
    ctx.rect(gridLeft, gridBottom, gridWidth, velocityHeight);
    ctx.clip();

    const velMarginTicks = Math.max(exactBeatTicks, 8 / Math.max(0.0001, pixelsPerTick));
    const velUpdates = drag && drag.kind === 'velocity' ? drag.updates : null;
    const drawVelocityStem = (id: string, startTick: number, baseVelocity: number) => {
      if (
        !Number.isFinite(startTick) ||
        startTick < startVisibleTick - velMarginTicks ||
        startTick > endVisibleTick + velMarginTicks
      ) {
        return;
      }
      const x = tickToScreenX(startTick);
      const rawVel = velUpdates?.get(id) ?? baseVelocity;
      const vel =
        typeof rawVel === 'number' && Number.isFinite(rawVel)
          ? Math.max(1, Math.min(127, rawVel))
          : 100;
      const barH = Math.max(3, (VEL_INNER_MAX(velocityHeight) * vel) / 127);
      const y = h - 6 - barH;
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(barH)) return;
      const isSelected =
        id === '__draw_preview__' || (!drawCreatePreview && safeSelectedNoteIds.has(id));

      ctx.strokeStyle = isSelected ? '#F4F4F5' : '#38BDF8';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x + 1, h - 6);
      ctx.lineTo(x + 1, y);
      ctx.stroke();
      ctx.lineCap = 'butt';

      ctx.fillStyle = isSelected ? '#F4F4F5' : '#38BDF8';
      fillRoundedCanvasRect(ctx, x - 1.5, y - 2, 5, 4, 2);
    };

    for (const note of visibleNotes) {
      if (dragMovedMap.has(note.id)) {
        continue;
      }
      drawVelocityStem(note.id, note.startTick, note.velocity);
    }
    for (const [id, preview] of dragMovedMap.entries()) {
      drawVelocityStem(id, preview.startTick, preview.velocity);
    }
    if (drawCreatePreview) {
      drawVelocityStem('__draw_preview__', drawCreatePreview.startTick, drawCreatePreview.velocity);
    }
    ctx.restore();

    // Velocity Left Label Box
    ctx.clearRect(0, gridBottom + 1, KEYBOARD_WIDTH, velocityHeight - 1);
    ctx.fillStyle = 'rgba(17, 17, 19, 0.65)';
    ctx.fillRect(0, gridBottom + 1, KEYBOARD_WIDTH, velocityHeight - 1);
    ctx.fillStyle = '#A1A1AA';
    ctx.font = `11px ${CANVAS_MONO_FONT}`;
    ctx.fillText('VEL', 12, gridBottom + Math.min(20, Math.round(velocityHeight * 0.48)));

    // 7. Left 128-Key Piano Keyboard
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, gridTop, KEYBOARD_WIDTH, gridHeight);
    ctx.clip();

    ctx.fillStyle = '#111113';
    ctx.fillRect(0, gridTop, KEYBOARD_WIDTH, gridHeight);

    for (let pitch = lowestPitch; pitch <= highestPitch; pitch += 1) {
      const y = pitchToScreenY(pitch);
      const pc = pitch % 12;
      const isBlack = BLACK_KEY_PCS.has(pc);

      if (isBlack) {
        ctx.fillStyle = '#D4D4D8';
        ctx.fillRect(0, y, KEYBOARD_WIDTH - 1, rowHeight);
        ctx.fillStyle = '#18181B';
        const blackW = KEYBOARD_WIDTH * 0.64;
        const blackH = Math.max(2, rowHeight - 1);
        if (typeof ctx.roundRect === 'function') {
          ctx.beginPath();
          ctx.roundRect(0, y + 0.5, blackW, blackH, [0, 3, 3, 0]);
          ctx.fill();
        } else {
          ctx.fillRect(0, y + 0.5, blackW, blackH);
        }
      } else {
        ctx.fillStyle = '#E4E4E7';
        ctx.fillRect(0, y, KEYBOARD_WIDTH - 1, rowHeight);
      }

      ctx.strokeStyle = '#27272A';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y + rowHeight);
      ctx.lineTo(KEYBOARD_WIDTH, y + rowHeight);
      ctx.stroke();

      if (pc === 0) {
        ctx.fillStyle = '#09090B';
        ctx.font = `11px ${CANVAS_MONO_FONT}`;
        ctx.fillText(midiPitchToName(pitch), 41, y + rowHeight * 0.72);
      }
    }
    ctx.restore();

    // Keyboard Right Border
    ctx.strokeStyle = '#27272A';
    ctx.beginPath();
    ctx.moveTo(KEYBOARD_WIDTH + 0.5, 0);
    ctx.lineTo(KEYBOARD_WIDTH + 0.5, h);
    ctx.stroke();

    // 8. Top Measure & Context Ruler
    ctx.clearRect(0, 0, w, RULER_HEIGHT);
    ctx.fillStyle = 'rgba(17, 17, 19, 0.90)';
    ctx.fillRect(0, 0, w, RULER_HEIGHT);
    ctx.strokeStyle = '#27272A';
    ctx.beginPath();
    ctx.moveTo(0, RULER_HEIGHT - 0.5);
    ctx.lineTo(w, RULER_HEIGHT - 0.5);
    ctx.stroke();

    ctx.save();
    ctx.beginPath();
    ctx.rect(gridLeft, 0, gridWidth, RULER_HEIGHT);
    ctx.clip();

    // Highlight target range in ruler + rounded COMPASS range bracket handles (`PianoRollView.java` parity)
    const rawTargetStartX = tickToScreenX(safeTargetRange.startTick);
    const rawTargetEndX = tickToScreenX(safeTargetRange.endTick);
    const rx1 = Math.max(gridLeft, rawTargetStartX);
    const rx2 = Math.min(w, rawTargetEndX);
    if (rx2 > rx1) {
      ctx.fillStyle = 'rgba(56, 189, 248, 0.18)';
      fillRoundedCanvasRect(ctx, rx1, 2, rx2 - rx1, RULER_HEIGHT - 4, 5);
      ctx.fillStyle = '#7DD3FC';
      ctx.font = `11px ${CANVAS_MONO_FONT}`;
      if (rx2 - rx1 >= 88) {
        ctx.fillText(
          `AI TARGET (${safeTargetRange.startMeasure}-${safeTargetRange.endMeasure})`,
          rx1 + 9,
          12,
        );
      }
    }
    if (rawTargetStartX >= gridLeft - 4 && rawTargetStartX <= w + 4) {
      ctx.fillStyle = '#38BDF8';
      fillRoundedCanvasRect(ctx, rawTargetStartX, 2, 6, 10, 3);
    }
    if (rawTargetEndX >= gridLeft - 4 && rawTargetEndX <= w + 4) {
      ctx.fillStyle = '#38BDF8';
      fillRoundedCanvasRect(ctx, rawTargetEndX - 6, 2, 6, 10, 3);
    }

    const firstMeasureIndex = firstGridMeasure;
    const lastMeasureIndex = lastGridMeasure;
    ctx.font = `11px ${CANVAS_MONO_FONT}`;
    for (let m = firstMeasureIndex; m <= lastMeasureIndex; m += 1) {
      const mTick = m * measureTicks;
      const mx = tickToScreenX(mTick);
      ctx.strokeStyle = '#3F3F46';
      ctx.beginPath();
      ctx.moveTo(mx + 0.5, 14);
      ctx.lineTo(mx + 0.5, RULER_HEIGHT);
      ctx.stroke();

      ctx.fillStyle = '#D4D4D8';
      ctx.fillText(String(m + 1), mx + 6, 26);
    }
    ctx.restore(); // End Ruler Clip so Step 9 Playhead spans full canvas height!

    // 9. Full-Height Playhead Line & Rounded Scrub Cap
    ctx.save();
    ctx.beginPath();
    ctx.rect(gridLeft, 0, gridWidth, h);
    ctx.clip();
    const playheadX = tickToScreenX(safePlayheadTick);
    if (playheadX >= gridLeft - 5 && playheadX <= w + 5) {
      ctx.strokeStyle = '#F4F4F5';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(playheadX, 14);
      ctx.lineTo(playheadX, h);
      ctx.stroke();

      ctx.fillStyle = '#F4F4F5';
      fillRoundedCanvasRect(ctx, playheadX - 4, 14, 8, 7, 2.5);
    }
    ctx.restore();

    // Top-left corner label
    ctx.fillStyle = '#A1A1AA';
    ctx.font = `11px ${CANVAS_MONO_FONT}`;
    ctx.fillText('BAR', 12, 21);

    ctx.restore();
  }, [
    activeTrack,
    canvasSize,
    dragRenderTick,
    interactiveCtrl,
    isActive,
    pixelsPerTick,
    rowHeight,
    scrollPxY,
    scrollTickX,
    safePlayheadTick,
    safePpqn,
    safeQuantDiv,
    safeSelectedNoteIds,
    safeTargetRange,
    safeTracks,
    session,
    showTrackRack,
  ]);

  // =========================================================================
  // Pointer & Wheel Interactions on Canvas (Ref-Stabilized Zero Listener Churn)
  // =========================================================================

  const getCanvasCoords = useCallback((clientX: number, clientY: number) => {
    if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return { x: 0, y: 0 };
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || !Number.isFinite(rect.left) || !Number.isFinite(rect.top)) return { x: 0, y: 0 };
    return { x: clientX - rect.left, y: clientY - rect.top };
  }, []);

  const findNoteAt = useCallback(
    (
      tick: number,
      pitch: number,
      isTouchPointer = false,
    ): { note: MidiNote; nearRightEdge: boolean } | null => {
      if (!activeTrack) return null;
      const edgePx = isTouchPointer ? 14 : 8;
      const minEdgeTicks = Math.max(1, Math.round((safePpqn * 8) / 480));
      const minResizableTicks = Math.max(2, Math.round((safePpqn * 24) / 480));
      const rawEdgeTicks = Math.max(minEdgeTicks, edgePx / pixelsPerTick);
      const overlapping = queryOverlappingNotesInSnapshot(activeTrack, tick - 1, tick + 1);
      for (let i = overlapping.length - 1; i >= 0; i -= 1) {
        const n = overlapping[i];
        if (n.pitch === pitch && tick >= n.startTick && tick <= n.endTick) {
          const effectiveEdgeTicks = Math.min(rawEdgeTicks, n.durationTicks * 0.4);
          const nearRightEdge =
            n.durationTicks >= minResizableTicks && n.endTick - tick <= effectiveEdgeTicks;
          return { note: n, nearRightEdge };
        }
      }
      return null;
    },
    [activeTrack, pixelsPerTick, safePpqn],
  );

  const applyVelocityAtPoint = useCallback(
    (
      trackSnap: DawTrackSnapshot,
      selectedIds: ReadonlySet<string>,
      heightPx: number,
      scrollX: number,
      pxPerTick: number,
      ppqn: number,
      canvasX: number,
      canvasY: number,
      updates: Map<string, number>,
    ) => {
      const relY = Math.max(
        0,
        Math.min(1, (heightPx - 6 - canvasY) / VEL_INNER_MAX(computeVelocityLaneHeight(heightPx))),
      );
      const newVel = Math.max(1, Math.min(127, Math.round(relY * 127)));
      const centerTick = scrollX + (canvasX - KEYBOARD_WIDTH) / pxPerTick;
      const safeVelPpqn =
        typeof ppqn === 'number' && Number.isFinite(ppqn) && ppqn >= 1 ? ppqn : 480;
      const minCatchTicks = Math.max(2, Math.round((safeVelPpqn * 30) / 480));
      const catchRadiusTicks = Math.max(minCatchTicks, 12 / pxPerTick);

      const nearby = queryOverlappingNotesInSnapshot(
        trackSnap,
        centerTick - catchRadiusTicks,
        centerTick + catchRadiusTicks + 1,
      );
      for (const n of nearby) {
        if (Math.abs(n.startTick - centerTick) <= catchRadiusTicks) {
          if (selectedIds.size === 0 || selectedIds.has(n.id)) {
            updates.set(n.id, newVel);
          }
        }
      }
    },
    [],
  );

  const activeTouchPointersRef = useRef<Map<number, { clientX: number; clientY: number }>>(
    new Map(),
  );
  const multiTouchGestureRef = useRef<{
    startMidX: number;
    startMidY: number;
    startSpanX: number;
    startScrollTickX: number;
    startScrollPxY: number;
    startPixelsPerBeat: number;
  } | null>(null);

  const handleCanvasPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!activeTrack || !Number.isFinite(e.clientX) || !Number.isFinite(e.clientY)) return;
    const isHandledCanvasPointer =
      e.pointerType === 'touch' || e.button === 0 || e.button === 1 || e.button === 2;
    if (!isHandledCanvasPointer) return;

    if (
      document.activeElement instanceof HTMLElement &&
      (document.activeElement.matches('input, select, textarea') ||
        document.activeElement.isContentEditable)
    ) {
      document.activeElement.blur();
    }
    canvasHadRecentPointerRef.current = true;
    pointerDownButtonsRef.current = typeof e.buttons === 'number' ? e.buttons : 0;

    const isTouch = e.pointerType === 'touch';
    if (isTouch) {
      activeTouchPointersRef.current.set(e.pointerId, {
        clientX: e.clientX,
        clientY: e.clientY,
      });
      if (activeTouchPointersRef.current.size >= 2) {
        // Cancel any single-finger drag cleanly (restoring ruler anchors if active) and switch to 2-finger pan/pinch
        cancelActiveDragRef.current();
        const pts = Array.from(activeTouchPointersRef.current.values());
        const p0 = pts[0];
        const p1 = pts[1];
        multiTouchGestureRef.current = {
          startMidX: (p0.clientX + p1.clientX) * 0.5,
          startMidY: (p0.clientY + p1.clientY) * 0.5,
          startSpanX: Math.max(24, Math.hypot(p1.clientX - p0.clientX, p1.clientY - p0.clientY)),
          startScrollTickX: scrollTickX,
          startScrollPxY: scrollPxY,
          startPixelsPerBeat: pixelsPerBeat,
        };
        setDragRenderTick((c) => c + 1);
        return;
      }
    } else if (e.button !== 0 && e.button !== 1) {
      return;
    }

    const { x, y } = getCanvasCoords(e.clientX, e.clientY);
    const h = canvasSize.height;
    const gridBottom = h - computeVelocityLaneHeight(h);

    // Middle-mouse button OR `Alt + Drag` on grid pans the viewport modelessly (`COMPASS DTM` parity)
    if (
      e.button === 1 ||
      (e.button === 0 &&
        e.altKey &&
        x >= KEYBOARD_WIDTH &&
        y >= RULER_HEIGHT &&
        y < gridBottom)
    ) {
      updateCanvasCursor('cursor-grabbing');
      setActiveDrag({
        kind: 'pan',
        pointerId: e.pointerId,
        startClientX: e.clientX,
        startClientY: e.clientY,
        startScrollTickX: scrollTickX,
        startScrollPxY: scrollPxY,
      });
      return;
    }

    // 1. Top Ruler Click/Drag -> Live Playhead Scrub or Direct `AI TARGET` Range Adjustment
    if (y < RULER_HEIGHT && x >= KEYBOARD_WIDTH) {
      const rawTick = Math.max(0, scrollTickX + (x - KEYBOARD_WIDTH) / pixelsPerTick);
      const safeRulerBeats =
        typeof session?.beatsPerMeasure === 'number' &&
        Number.isFinite(session.beatsPerMeasure) &&
        session.beatsPerMeasure > 0
          ? Math.max(0.0625, Math.min(256, session.beatsPerMeasure))
          : 4;
      const measureTicks = Math.max(1, Math.round(safePpqn * safeRulerBeats));
      const clickedMeasure = Math.max(1, Math.min(1024, Math.floor(rawTick / measureTicks) + 1));

      if (e.ctrlKey || e.metaKey) {
        const nextEnd = Math.max(
          clickedMeasure,
          Math.min(clickedMeasure + 7, safeTargetRange.endMeasure),
        );
        controller.setTargetMeasureRange(clickedMeasure, nextEnd);
        onProjectMutated?.();
        return;
      }
      if (e.altKey) {
        const nextStart = Math.min(safeTargetRange.startMeasure, clickedMeasure);
        const clampedEnd = Math.min(nextStart + 7, Math.max(nextStart, clickedMeasure));
        controller.setTargetMeasureRange(nextStart, clampedEnd);
        onProjectMutated?.();
        return;
      }

      const startHandleX = KEYBOARD_WIDTH + (safeTargetRange.startTick - scrollTickX) * pixelsPerTick;
      const endHandleX = KEYBOARD_WIDTH + (safeTargetRange.endTick - scrollTickX) * pixelsPerTick;
      const distStart = Math.abs(x - startHandleX);
      const distEnd = Math.abs(x - endHandleX);
      const handleMaxY = isTouch ? RULER_HEIGHT : 13;
      const handleHitPx = isTouch ? 18 : 8;
      if (y <= handleMaxY && (distStart <= handleHitPx || distEnd <= handleHitPx)) {
        const edge: 'start' | 'end' = distEnd <= distStart ? 'end' : 'start';
        updateCanvasCursor('cursor-ew-resize');
        setActiveDrag({
          kind: 'ruler-range',
          pointerId: e.pointerId,
          edge,
          anchorStartMeasure: safeTargetRange.startMeasure,
          anchorEndMeasure: safeTargetRange.endMeasure,
        });
        return;
      }

      const anchorPlayheadTick = safePlayheadTick;
      const snapped = interactiveCtrl ? interactiveCtrl.snapTick(rawTick) : Math.round(rawTick);
      controller.setPlayheadTick(snapped);
      updateCanvasCursor('cursor-ew-resize');
      setActiveDrag({
        kind: 'ruler-scrub',
        pointerId: e.pointerId,
        anchorPlayheadTick,
      });
      return;
    }

    if (!interactiveCtrl) return;

    // 2. Left Keyboard Click/Tap -> Audition Pitch
    if (x < KEYBOARD_WIDTH && y >= RULER_HEIGHT && y < gridBottom) {
      const pitch = Math.max(
        0,
        Math.min(127, 127 - Math.floor((y - RULER_HEIGHT + scrollPxY) / rowHeight)),
      );
      interactiveCtrl.auditionPitch(pitch);
      return;
    }

    // 3. Bottom Velocity Lane Drag (Mouse, Stylus & Touch)
    if (x >= KEYBOARD_WIDTH && y >= gridBottom) {
      updateCanvasCursor('cursor-ns-resize');
      const updates = new Map<string, number>();
      applyVelocityAtPoint(
        activeTrack,
        safeSelectedNoteIds,
        canvasSize.height,
        scrollTickX,
        pixelsPerTick,
        safePpqn,
        x,
        y,
        updates,
      );
      setActiveDrag({
        kind: 'velocity',
        pointerId: e.pointerId,
        trackId: activeTrack.id,
        notesSnapshotRef: Array.isArray(activeTrack.notes) ? activeTrack.notes : [],
        updates,
      });
      setDragRenderTick((c) => c + 1);
      return;
    }

    // 4. Main Note Grid Interaction (Unified Mouse, Stylus & Touch — `COMPASS DTM` Modeless Smart Pointer)
    if (x >= KEYBOARD_WIDTH && y >= RULER_HEIGHT && y < gridBottom) {
      const rawTick = Math.max(0, scrollTickX + (x - KEYBOARD_WIDTH) / pixelsPerTick);
      const pitch = Math.max(
        0,
        Math.min(127, 127 - Math.floor((y - RULER_HEIGHT + scrollPxY) / rowHeight)),
      );
      const hit = findNoteAt(rawTick, pitch, isTouch);
      const activeNotesSnapshot = Array.isArray(activeTrack.notes) ? activeTrack.notes : [];

      const liveSnapSelected = interactiveCtrl.getSnapshot()?.selectedNoteIds;
      const currentSelectedIds: ReadonlySet<string> =
        liveSnapSelected instanceof Set ? liveSnapSelected : safeSelectedNoteIds;

      if (hit) {
        // On touch in `draw` mode, defer tap-to-delete to `pointerup` so 2-finger pinch/pan never deletes notes ([R3-MED-04])
        if (isTouch && toolMode === 'draw') {
          setActiveDrag({
            kind: 'touch-draw-tap',
            pointerId: e.pointerId,
            trackId: activeTrack.id,
            notesSnapshotRef: activeNotesSnapshot,
            startClientX: e.clientX,
            startClientY: e.clientY,
            pitch,
            rawTick,
            hitNoteId: hit.note.id,
          });
          return;
        }

        const initialSelection = new Set(currentSelectedIds);
        const alreadySelected = currentSelectedIds.has(hit.note.id);
        let nextSelection = new Set(currentSelectedIds);
        let toggleDeselectOnClick = false;
        let isolateSelectionOnClick = false;
        if (e.shiftKey) {
          if (alreadySelected) {
            if (!hit.nearRightEdge) {
              toggleDeselectOnClick = true;
            }
          } else {
            nextSelection.add(hit.note.id);
            interactiveCtrl.setSelectedNotes(nextSelection);
          }
        } else if (!alreadySelected) {
          nextSelection = new Set([hit.note.id]);
          interactiveCtrl.setSelectedNotes(nextSelection);
        } else if (currentSelectedIds.size > 1 && !hit.nearRightEdge) {
          isolateSelectionOnClick = true;
        }

        const movingNotes: Array<{
          id: string;
          pitch: number;
          startTick: number;
          durationTicks: number;
          velocity: number;
        }> = [];
        for (const selId of nextSelection) {
          const found = activeNoteById.get(selId);
          if (found) {
            movingNotes.push({
              id: found.id,
              pitch: found.pitch,
              startTick: found.startTick,
              durationTicks: found.durationTicks,
              velocity: found.velocity,
            });
          }
        }

        if (!hit.nearRightEdge) {
          interactiveCtrl.auditionPitch(hit.note.pitch);
        }

        updateCanvasCursor(hit.nearRightEdge ? 'cursor-ew-resize' : 'cursor-grabbing');
        setActiveDrag({
          kind: hit.nearRightEdge ? 'resize' : 'move',
          pointerId: e.pointerId,
          trackId: activeTrack.id,
          notesSnapshotRef: activeNotesSnapshot,
          anchorNoteId: hit.note.id,
          startClientX: e.clientX,
          startClientY: e.clientY,
          initialNotes: movingNotes,
          initialSelection,
          toggleDeselectOnClick,
          isolateSelectionOnClick,
          fineResize: e.shiftKey,
          hasMovedPointer: false,
          currentDeltaTick: 0,
          currentDeltaPitch: 0,
        });
        setDragRenderTick((c) => c + 1);
        return;
      }

      if (toolMode === 'draw') {
        if (
          activeNotesSnapshot.length >= MAX_NOTES_PER_TRACK ||
          totalWorkspaceNotes >= MAX_WORKSPACE_NOTES
        ) {
          return;
        }
        if (isTouch) {
          // Defer touch note creation to `pointerup` so a 2nd finger landing cancels cleanly without phantom notes ([R3-MED-04])
          setActiveDrag({
            kind: 'touch-draw-tap',
            pointerId: e.pointerId,
            trackId: activeTrack.id,
            notesSnapshotRef: activeNotesSnapshot,
            startClientX: e.clientX,
            startClientY: e.clientY,
            pitch,
            rawTick,
            hitNoteId: null,
          });
        } else {
          const maxProjectTicks = MAX_SMF_QUARTER_NOTES * safePpqn;
          const snappedStartTick = Math.min(
            Math.max(0, maxProjectTicks - 1),
            interactiveCtrl.snapTick(rawTick),
          );
          const initialDurationTicks = Math.max(
            1,
            Math.min(interactiveCtrl.getSnapStepTicks(), maxProjectTicks - snappedStartTick),
          );
          interactiveCtrl.auditionPitch(pitch);
          updateCanvasCursor('cursor-ew-resize');
          setActiveDrag({
            kind: 'draw-create',
            pointerId: e.pointerId,
            trackId: activeTrack.id,
            notesSnapshotRef: activeNotesSnapshot,
            startClientX: e.clientX,
            startClientY: e.clientY,
            pitch,
            rawTick,
            snappedStartTick,
            initialDurationTicks,
            velocity: 100,
            fineResize: e.shiftKey,
            hasMovedPointer: false,
            currentDeltaTick: 0,
          });
          setDragRenderTick((c) => c + 1);
        }
      } else {
        const initialSelection = new Set(currentSelectedIds);
        if (!e.shiftKey) {
          interactiveCtrl.setSelectedNotes([]);
        }
        const worldY = y - RULER_HEIGHT + scrollPxY;
        setActiveDrag({
          kind: 'marquee',
          pointerId: e.pointerId,
          trackId: activeTrack.id,
          notesSnapshotRef: activeNotesSnapshot,
          startClientX: e.clientX,
          startClientY: e.clientY,
          additive: e.shiftKey,
          initialSelection,
          startWorldX: rawTick,
          startWorldY: worldY,
          currentWorldX: rawTick,
          currentWorldY: worldY,
          hasExceededDragThreshold: false,
        });
        setDragRenderTick((c) => c + 1);
      }
    }
  };

  const handleCanvasDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (
      toolMode === 'draw' ||
      !interactiveCtrl ||
      !activeTrack ||
      !Number.isFinite(e.clientX) ||
      !Number.isFinite(e.clientY)
    ) {
      return;
    }
    const { x, y } = getCanvasCoords(e.clientX, e.clientY);
    const gridBottom = canvasSize.height - computeVelocityLaneHeight(canvasSize.height);
    if (x < KEYBOARD_WIDTH || y < RULER_HEIGHT || y >= gridBottom) return;

    const rawTick = Math.max(0, scrollTickX + (x - KEYBOARD_WIDTH) / pixelsPerTick);
    const pitch = Math.max(
      0,
      Math.min(127, 127 - Math.floor((y - RULER_HEIGHT + scrollPxY) / rowHeight)),
    );
    const hit = findNoteAt(rawTick, pitch);
    if (hit) {
      interactiveCtrl.setSelectedNotes([hit.note.id]);
      interactiveCtrl.deleteSelectedNotes();
    } else {
      interactiveCtrl.createNoteAt(pitch, rawTick);
    }
    onProjectMutated?.();
  };

  const handleCanvasContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (dragRef.current !== null) {
      cancelActiveDragRef.current();
      return;
    }
    if (
      !interactiveCtrl ||
      !activeTrack ||
      !Number.isFinite(e.clientX) ||
      !Number.isFinite(e.clientY)
    ) {
      return;
    }
    const { x, y } = getCanvasCoords(e.clientX, e.clientY);
    const gridBottom = canvasSize.height - computeVelocityLaneHeight(canvasSize.height);
    if (x < KEYBOARD_WIDTH || y < RULER_HEIGHT || y >= gridBottom) return;

    const rawTick = Math.max(0, scrollTickX + (x - KEYBOARD_WIDTH) / pixelsPerTick);
    const pitch = Math.max(
      0,
      Math.min(127, 127 - Math.floor((y - RULER_HEIGHT + scrollPxY) / rowHeight)),
    );
    const hit = findNoteAt(rawTick, pitch);
    if (!hit) return;

    const remainingSelected: string[] = [];
    for (const selId of safeSelectedNoteIds) {
      if (selId !== hit.note.id) {
        remainingSelected.push(selId);
      }
    }
    interactiveCtrl.setSelectedNotes([hit.note.id]);
    interactiveCtrl.deleteSelectedNotes();
    if (remainingSelected.length > 0) {
      interactiveCtrl.setSelectedNotes(remainingSelected);
    }
    onProjectMutated?.();
  };

  // Ref holding latest pointer context so global `pointermove`/`pointerup`/`pointercancel` attach strictly ONCE on mount
  const pointerContextRef = useRef({
    controller,
    interactiveCtrl,
    activeTrack,
    activeNoteById,
    canvasHeight: canvasSize.height,
    pixelsPerBeat,
    pixelsPerTick,
    ppqn: safePpqn,
    beatsPerMeasure: session?.beatsPerMeasure,
    targetRange: safeTargetRange,
    toolMode,
    rowHeight,
    scrollTickX,
    scrollPxY,
    selectedNoteIds: safeSelectedNoteIds,
    findNoteAt,
    updateCanvasCursor,
    onProjectMutated,
  });
  pointerContextRef.current = {
    controller,
    interactiveCtrl,
    activeTrack,
    activeNoteById,
    canvasHeight: canvasSize.height,
    pixelsPerBeat,
    pixelsPerTick,
    ppqn: safePpqn,
    beatsPerMeasure: session?.beatsPerMeasure,
    targetRange: safeTargetRange,
    toolMode,
    rowHeight,
    scrollTickX,
    scrollPxY,
    selectedNoteIds: safeSelectedNoteIds,
    findNoteAt,
    updateCanvasCursor,
    onProjectMutated,
  };
  cancelActiveDragRef.current = () => {
    const drag = dragRef.current;
    if (!drag) return;
    setActiveDrag(null);
    const ctx = pointerContextRef.current;
    if (drag.kind === 'ruler-range') {
      const cur = ctx.controller.getSnapshot().targetRange;
      if (
        cur.startMeasure !== drag.anchorStartMeasure ||
        cur.endMeasure !== drag.anchorEndMeasure
      ) {
        ctx.controller.setTargetMeasureRange(drag.anchorStartMeasure, drag.anchorEndMeasure);
        ctx.onProjectMutated?.();
      }
    } else if (drag.kind === 'ruler-scrub') {
      if (ctx.controller.getSnapshot().playheadTick !== drag.anchorPlayheadTick) {
        ctx.controller.setPlayheadTick(drag.anchorPlayheadTick);
      }
    } else if (
      (drag.kind === 'marquee' || drag.kind === 'move' || drag.kind === 'resize') &&
      ctx.activeTrack?.id === drag.trackId &&
      Array.isArray(ctx.activeTrack.notes) &&
      areNoteSnapshotsIdentical(drag.notesSnapshotRef, ctx.activeTrack.notes)
    ) {
      const liveIds = Array.from(drag.initialSelection).filter((id) => ctx.activeNoteById.has(id));
      ctx.interactiveCtrl?.setSelectedNotes(liveIds);
    }
    ctx.updateCanvasCursor(ctx.toolMode === 'draw' ? 'cursor-crosshair' : 'cursor-default');
    setDragRenderTick((c) => c + 1);
  };

  useEffect(() => {
    const onPointerMove = (e: PointerEvent) => {
      if (!Number.isFinite(e.clientX) || !Number.isFinite(e.clientY)) return;
      const ctx = pointerContextRef.current;

      if (e.pointerType === 'touch' && activeTouchPointersRef.current.has(e.pointerId)) {
        activeTouchPointersRef.current.set(e.pointerId, {
          clientX: e.clientX,
          clientY: e.clientY,
        });
        const gesture = multiTouchGestureRef.current;
        if (activeTouchPointersRef.current.size >= 2 && gesture) {
          const pts = Array.from(activeTouchPointersRef.current.values());
          const p0 = pts[0];
          const p1 = pts[1];
          const midX = (p0.clientX + p1.clientX) * 0.5;
          const midY = (p0.clientY + p1.clientY) * 0.5;
          const span = Math.max(24, Math.hypot(p1.clientX - p0.clientX, p1.clientY - p0.clientY));
          const zoomRatio = span / Math.max(1, gesture.startSpanX);
          const nextPpb = clampPixelsPerBeat(
            gesture.startPixelsPerBeat * zoomRatio,
            gesture.startPixelsPerBeat,
          );
          const safePpqnVal =
            Number.isFinite(ctx.ppqn) && ctx.ppqn >= 1 ? Math.min(32767, ctx.ppqn) : 480;
          const oldPxPerTick = clampPixelsPerBeat(gesture.startPixelsPerBeat, 64) / safePpqnVal;
          const nextPxPerTick = nextPpb / safePpqnVal;
          const rect = canvasRef.current?.getBoundingClientRect();
          const startGridOffsetX =
            rect && Number.isFinite(rect.left)
              ? Math.max(0, gesture.startMidX - rect.left - KEYBOARD_WIDTH)
              : Math.max(0, gesture.startMidX - KEYBOARD_WIDTH);
          const curGridOffsetX =
            rect && Number.isFinite(rect.left)
              ? Math.max(0, midX - rect.left - KEYBOARD_WIDTH)
              : Math.max(0, midX - KEYBOARD_WIDTH);
          const anchorTick = gesture.startScrollTickX + startGridOffsetX / oldPxPerTick;
          const nextScrollTick = clampScrollTickX(
            anchorTick - curGridOffsetX / nextPxPerTick,
            safePpqnVal,
            gesture.startScrollTickX,
          );

          const dy = gesture.startMidY - midY;
          const gridHeight = Math.max(
            1,
            ctx.canvasHeight - RULER_HEIGHT - computeVelocityLaneHeight(ctx.canvasHeight),
          );
          const maxScrollY = Math.max(0, TOTAL_PITCHES * ctx.rowHeight - gridHeight);
          const nextScrollY = clampScrollPxY(
            gesture.startScrollPxY + dy,
            maxScrollY,
            gesture.startScrollPxY,
          );

          ctx.pixelsPerBeat = nextPpb;
          ctx.pixelsPerTick = nextPxPerTick;
          ctx.scrollTickX = nextScrollTick;
          ctx.scrollPxY = nextScrollY;
          setPixelsPerBeat(nextPpb);
          setScrollTickX(nextScrollTick);
          setScrollPxY(nextScrollY);
          return;
        }
      }

      const drag = dragRef.current;
      if (!drag) {
        if (e.pointerType !== 'touch' && e.target === canvasRef.current && ctx.activeTrack) {
          const { x, y } = getCanvasCoords(e.clientX, e.clientY);
          const gridBottom = ctx.canvasHeight - computeVelocityLaneHeight(ctx.canvasHeight);
          if (y < RULER_HEIGHT && x >= KEYBOARD_WIDTH) {
            const startHandleX =
              KEYBOARD_WIDTH + (ctx.targetRange.startTick - ctx.scrollTickX) * ctx.pixelsPerTick;
            const endHandleX =
              KEYBOARD_WIDTH + (ctx.targetRange.endTick - ctx.scrollTickX) * ctx.pixelsPerTick;
            if (y <= 13 && (Math.abs(x - startHandleX) <= 8 || Math.abs(x - endHandleX) <= 8)) {
              ctx.updateCanvasCursor('cursor-ew-resize');
            } else {
              ctx.updateCanvasCursor('cursor-pointer');
            }
          } else if (x < KEYBOARD_WIDTH && y >= RULER_HEIGHT && y < gridBottom) {
            ctx.updateCanvasCursor('cursor-pointer');
          } else if (x >= KEYBOARD_WIDTH && y >= gridBottom) {
            ctx.updateCanvasCursor('cursor-ns-resize');
          } else if (x >= KEYBOARD_WIDTH && y >= RULER_HEIGHT && y < gridBottom) {
            if (e.altKey) {
              ctx.updateCanvasCursor('cursor-grab');
            } else {
              const rawTick = Math.max(
                0,
                ctx.scrollTickX + (x - KEYBOARD_WIDTH) / ctx.pixelsPerTick,
              );
              const pitch = Math.max(
                0,
                Math.min(127, 127 - Math.floor((y - RULER_HEIGHT + ctx.scrollPxY) / ctx.rowHeight)),
              );
              const hit = ctx.findNoteAt(rawTick, pitch, false);
              if (hit?.nearRightEdge) {
                ctx.updateCanvasCursor('cursor-ew-resize');
              } else if (hit) {
                ctx.updateCanvasCursor('cursor-grab');
              } else {
                ctx.updateCanvasCursor(
                  ctx.toolMode === 'draw' ? 'cursor-crosshair' : 'cursor-default',
                );
              }
            }
          } else {
            ctx.updateCanvasCursor('cursor-default');
          }
        }
        return;
      }

      if (drag.pointerId !== e.pointerId) return;

      if (
        e.pointerType !== 'touch' &&
        pointerDownButtonsRef.current > 0 &&
        typeof e.buttons === 'number' &&
        e.buttons === 0
      ) {
        cancelActiveDragRef.current();
        return;
      }

      if (drag.kind === 'touch-draw-tap') {
        if (Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY) > 10) {
          setActiveDrag(null);
        }
        return;
      }

      if (drag.kind === 'pan') {
        const dx = drag.startClientX - e.clientX;
        const dy = drag.startClientY - e.clientY;
        const gridHeight = Math.max(
          1,
          ctx.canvasHeight - RULER_HEIGHT - computeVelocityLaneHeight(ctx.canvasHeight),
        );
        const maxScrollY = Math.max(0, TOTAL_PITCHES * ctx.rowHeight - gridHeight);
        setScrollTickX(
          clampScrollTickX(
            drag.startScrollTickX + dx / ctx.pixelsPerTick,
            ctx.ppqn,
            drag.startScrollTickX,
          ),
        );
        setScrollPxY(clampScrollPxY(drag.startScrollPxY + dy, maxScrollY, drag.startScrollPxY));
        return;
      }

      if (drag.kind === 'ruler-scrub') {
        const { x } = getCanvasCoords(e.clientX, e.clientY);
        const rawTick = Math.max(0, ctx.scrollTickX + (x - KEYBOARD_WIDTH) / ctx.pixelsPerTick);
        const snapped = ctx.interactiveCtrl ? ctx.interactiveCtrl.snapTick(rawTick) : Math.round(rawTick);
        ctx.controller.setPlayheadTick(snapped);
        return;
      }

      if (drag.kind === 'ruler-range') {
        const { x } = getCanvasCoords(e.clientX, e.clientY);
        const rawTick = Math.max(0, ctx.scrollTickX + (x - KEYBOARD_WIDTH) / ctx.pixelsPerTick);
        const safeBeats =
          typeof ctx.beatsPerMeasure === 'number' &&
          Number.isFinite(ctx.beatsPerMeasure) &&
          ctx.beatsPerMeasure > 0
            ? Math.max(0.0625, Math.min(256, ctx.beatsPerMeasure))
            : 4;
        const measureTicks = Math.max(1, Math.round(ctx.ppqn * safeBeats));
        let nextStart = ctx.targetRange.startMeasure;
        let nextEnd = ctx.targetRange.endMeasure;
        if (drag.edge === 'start') {
          const boundaryIdx = Math.max(0, Math.min(1023, Math.round(rawTick / measureTicks)));
          nextStart = boundaryIdx + 1;
          nextEnd = Math.max(nextStart, Math.min(nextStart + 7, drag.anchorEndMeasure));
        } else {
          const boundaryIdx = Math.max(1, Math.min(1031, Math.round(rawTick / measureTicks)));
          nextStart = drag.anchorStartMeasure;
          nextEnd = Math.max(nextStart, Math.min(nextStart + 7, boundaryIdx));
        }
        if (
          nextStart !== ctx.targetRange.startMeasure ||
          nextEnd !== ctx.targetRange.endMeasure
        ) {
          ctx.controller.setTargetMeasureRange(nextStart, nextEnd);
          ctx.onProjectMutated?.();
        }
        return;
      }

      if (!ctx.interactiveCtrl || !ctx.activeTrack) return;

      if (drag.kind === 'draw-create') {
        const dxPx = e.clientX - drag.startClientX;
        const dyPx = e.clientY - drag.startClientY;
        if (!drag.hasMovedPointer) {
          if (!(Math.hypot(dxPx, dyPx) >= DRAG_DEADZONE_PX)) {
            return;
          }
          drag.hasMovedPointer = true;
        }
        drag.fineResize = e.shiftKey;
        const rawDeltaTicks = dxPx / ctx.pixelsPerTick;
        const baseEndTick = drag.snappedStartTick + drag.initialDurationTicks;
        const nextDeltaTick = e.shiftKey
          ? Math.round(rawDeltaTicks)
          : ctx.interactiveCtrl.snapTick(
              Math.max(drag.snappedStartTick + 1, baseEndTick + rawDeltaTicks),
            ) - baseEndTick;
        drag.currentDeltaTick = Number.isFinite(nextDeltaTick) ? nextDeltaTick : 0;
        setDragRenderTick((c) => c + 1);
        return;
      }

      if (drag.kind === 'move' || drag.kind === 'resize') {
        const dxPx = e.clientX - drag.startClientX;
        const dyPx = e.clientY - drag.startClientY;
        if (!drag.hasMovedPointer) {
          if (!(Math.hypot(dxPx, dyPx) >= DRAG_DEADZONE_PX)) {
            return;
          }
          drag.hasMovedPointer = true;
        }
        if (drag.kind === 'resize') {
          drag.fineResize = e.shiftKey;
        }
        const rawDeltaTicks = dxPx / ctx.pixelsPerTick;
        const anchorInit =
          drag.initialNotes.find((n) => n.id === drag.anchorNoteId) ?? drag.initialNotes[0];
        let nextDeltaTick: number;
        if (e.shiftKey || !anchorInit) {
          nextDeltaTick = Math.round(rawDeltaTicks);
        } else if (drag.kind === 'move') {
          const snappedAnchorStart = ctx.interactiveCtrl.snapTick(
            Math.max(0, anchorInit.startTick + rawDeltaTicks),
          );
          nextDeltaTick = snappedAnchorStart - anchorInit.startTick;
        } else {
          const anchorEndTick = anchorInit.startTick + anchorInit.durationTicks;
          const snappedAnchorEnd = ctx.interactiveCtrl.snapTick(
            Math.max(anchorInit.startTick + 1, anchorEndTick + rawDeltaTicks),
          );
          nextDeltaTick = snappedAnchorEnd - anchorEndTick;
        }
        const nextDeltaPitch = drag.kind === 'move' ? -Math.round(dyPx / ctx.rowHeight) : 0;
        if (drag.kind === 'move') {
          const liveNotes = drag.initialNotes.filter((init) => ctx.activeNoteById.has(init.id));
          const groupClamped = computeClampedGroupMoveDeltas(
            liveNotes,
            nextDeltaTick,
            nextDeltaPitch,
            ctx.ppqn,
          );
          drag.currentDeltaTick = groupClamped.deltaTick;
          drag.currentDeltaPitch = groupClamped.deltaPitch;
        } else {
          drag.currentDeltaTick = Number.isFinite(nextDeltaTick) ? nextDeltaTick : 0;
          drag.currentDeltaPitch = 0;
        }
        setDragRenderTick((c) => c + 1);
      } else if (drag.kind === 'marquee') {
        if (
          !drag.hasExceededDragThreshold &&
          Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY) > 5
        ) {
          drag.hasExceededDragThreshold = true;
        }
        const { x, y } = getCanvasCoords(e.clientX, e.clientY);
        drag.currentWorldX = clampScrollTickX(
          ctx.scrollTickX + (x - KEYBOARD_WIDTH) / ctx.pixelsPerTick,
          ctx.ppqn,
          drag.startWorldX,
        );
        const rawWorldY = y - RULER_HEIGHT + ctx.scrollPxY;
        drag.currentWorldY = Number.isFinite(rawWorldY)
          ? Math.max(0, Math.min(TOTAL_PITCHES * ctx.rowHeight, rawWorldY))
          : drag.startWorldY;
        setDragRenderTick((c) => c + 1);
      } else if (drag.kind === 'velocity') {
        const { x, y } = getCanvasCoords(e.clientX, e.clientY);
        applyVelocityAtPoint(
          ctx.activeTrack,
          ctx.selectedNoteIds,
          ctx.canvasHeight,
          ctx.scrollTickX,
          ctx.pixelsPerTick,
          ctx.ppqn,
          x,
          y,
          drag.updates,
        );
        setDragRenderTick((c) => c + 1);
      }
    };

    const cleanupTouchPointer = (pointerId: number) => {
      activeTouchPointersRef.current.delete(pointerId);
      if (activeTouchPointersRef.current.size < 2) {
        multiTouchGestureRef.current = null;
      } else {
        const ctx = pointerContextRef.current;
        const pts = Array.from(activeTouchPointersRef.current.values());
        const p0 = pts[0];
        const p1 = pts[1];
        multiTouchGestureRef.current = {
          startMidX: (p0.clientX + p1.clientX) * 0.5,
          startMidY: (p0.clientY + p1.clientY) * 0.5,
          startSpanX: Math.max(24, Math.hypot(p1.clientX - p0.clientX, p1.clientY - p0.clientY)),
          startScrollTickX: ctx.scrollTickX,
          startScrollPxY: ctx.scrollPxY,
          startPixelsPerBeat: ctx.pixelsPerBeat,
        };
      }
    };

    const abortAllPointerInteractions = () => {
      activeTouchPointersRef.current.clear();
      multiTouchGestureRef.current = null;
      pointerDownButtonsRef.current = 0;
      if (dragRef.current !== null) {
        cancelActiveDragRef.current();
      }
    };

    const onVisibilityChange = () => {
      if (document.hidden) {
        abortAllPointerInteractions();
      }
    };

    const onPointerCancel = (e: PointerEvent) => {
      cleanupTouchPointer(e.pointerId);
      if (dragRef.current && dragRef.current.pointerId === e.pointerId) {
        cancelActiveDragRef.current();
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      cleanupTouchPointer(e.pointerId);
      const drag = dragRef.current;
      const ctx = pointerContextRef.current;
      if (!drag || drag.pointerId !== e.pointerId) return;
      setActiveDrag(null);
      ctx.updateCanvasCursor(ctx.toolMode === 'draw' ? 'cursor-crosshair' : 'cursor-default');

      if (
        drag.kind === 'pan' ||
        drag.kind === 'ruler-scrub' ||
        drag.kind === 'ruler-range' ||
        !ctx.interactiveCtrl ||
        !ctx.activeTrack
      ) {
        setDragRenderTick((c) => c + 1);
        return;
      }

      if (drag.kind === 'draw-create') {
        if (ctx.activeTrack.id !== drag.trackId) {
          setDragRenderTick((c) => c + 1);
          return;
        }
        const commitSnapTicks = ctx.interactiveCtrl.getSnapStepTicks();
        const commitMinResizableTicks = Math.max(2, Math.round((ctx.ppqn * 24) / 480));
        const minResizeFloor = drag.fineResize ? commitMinResizableTicks : commitSnapTicks;
        const maxProjectTicks = MAX_SMF_QUARTER_NOTES * ctx.ppqn;
        const snappedStart = Math.min(
          Math.max(0, maxProjectTicks - 1),
          ctx.interactiveCtrl.snapTick(drag.rawTick),
        );
        const baseDur = Math.max(1, Math.min(commitSnapTicks, maxProjectTicks - snappedStart));
        const noteMinResize = Math.max(1, Math.min(baseDur, minResizeFloor));
        const maxNoteDur = Math.max(1, maxProjectTicks - snappedStart);
        const finalDuration = drag.hasMovedPointer
          ? Math.max(noteMinResize, Math.min(maxNoteDur, baseDur + drag.currentDeltaTick))
          : baseDur;
        ctx.interactiveCtrl.createNoteAt(drag.pitch, drag.rawTick, finalDuration, drag.velocity, {
          silent: true,
        });
        ctx.onProjectMutated?.();
        setDragRenderTick((c) => c + 1);
        return;
      }

      if (drag.kind === 'touch-draw-tap') {
        if (
          ctx.activeTrack.id === drag.trackId &&
          Number.isFinite(e.clientX) &&
          Number.isFinite(e.clientY) &&
          Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY) <= 10
        ) {
          if (drag.hitNoteId) {
            if (ctx.activeNoteById.has(drag.hitNoteId)) {
              ctx.interactiveCtrl.setSelectedNotes([drag.hitNoteId]);
              ctx.interactiveCtrl.deleteSelectedNotes();
              ctx.onProjectMutated?.();
            }
          } else {
            ctx.interactiveCtrl.createNoteAt(drag.pitch, drag.rawTick);
            ctx.onProjectMutated?.();
          }
        }
        setDragRenderTick((c) => c + 1);
        return;
      }

      if (drag.kind === 'move' || drag.kind === 'resize') {
        if (ctx.activeTrack.id !== drag.trackId) {
          setDragRenderTick((c) => c + 1);
          return;
        }
        const liveInitialNotes = drag.initialNotes.filter((init) =>
          ctx.activeNoteById.has(init.id),
        );
        if (liveInitialNotes.length === 0) {
          setDragRenderTick((c) => c + 1);
          return;
        }
        const groupMove =
          drag.kind === 'move'
            ? computeClampedGroupMoveDeltas(
                liveInitialNotes,
                drag.currentDeltaTick,
                drag.currentDeltaPitch,
                ctx.ppqn,
              )
            : { deltaTick: drag.currentDeltaTick, deltaPitch: 0 };
        if (groupMove.deltaTick !== 0 || groupMove.deltaPitch !== 0) {
          const commitSnapTicks = ctx.interactiveCtrl.getSnapStepTicks();
          const commitMinResizableTicks = Math.max(2, Math.round((ctx.ppqn * 24) / 480));
          const minResizeFloor = drag.fineResize ? commitMinResizableTicks : commitSnapTicks;
          const maxProjectTicks = MAX_SMF_QUARTER_NOTES * ctx.ppqn;
          const updates = liveInitialNotes.map((init) => {
            const noteMinResize = Math.max(1, Math.min(init.durationTicks, minResizeFloor));
            const maxNoteDur = Math.max(1, maxProjectTicks - init.startTick);
            return {
              id: init.id,
              pitch:
                drag.kind === 'move'
                  ? Math.max(0, Math.min(127, init.pitch + groupMove.deltaPitch))
                  : init.pitch,
              startTick:
                drag.kind === 'move'
                  ? Math.max(0, init.startTick + groupMove.deltaTick)
                  : init.startTick,
              durationTicks:
                drag.kind === 'resize'
                  ? Math.max(
                      noteMinResize,
                      Math.min(maxNoteDur, init.durationTicks + groupMove.deltaTick),
                    )
                  : init.durationTicks,
              velocity: init.velocity,
            };
          });
          ctx.interactiveCtrl.transformNotes(
            updates,
            drag.kind === 'move' ? 'ノート移動' : 'ノート長変更',
          );
          ctx.onProjectMutated?.();
        } else if (!drag.hasMovedPointer) {
          if (drag.toggleDeselectOnClick) {
            const nextSel = new Set(ctx.selectedNoteIds);
            nextSel.delete(drag.anchorNoteId);
            ctx.interactiveCtrl.setSelectedNotes(nextSel);
          } else if (drag.isolateSelectionOnClick) {
            ctx.interactiveCtrl.setSelectedNotes([drag.anchorNoteId]);
          }
        }
      } else if (drag.kind === 'marquee') {
        const movedPx =
          Number.isFinite(e.clientX) && Number.isFinite(e.clientY)
            ? Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY)
            : 0;
        if (!drag.hasExceededDragThreshold && movedPx <= 5) {
          // Single-click on empty grid snaps & moves playhead (`PianoRollView.java` parity)
          const snapped = ctx.interactiveCtrl.snapTick(drag.startWorldX);
          ctx.controller.setPlayheadTick(snapped);
        } else {
          const minTick = Math.min(drag.startWorldX, drag.currentWorldX);
          const maxTick = Math.max(drag.startWorldX, drag.currentWorldX);
          const minY = Math.min(drag.startWorldY, drag.currentWorldY);
          const maxY = Math.max(drag.startWorldY, drag.currentWorldY);
          const highestP = 127 - Math.floor(minY / ctx.rowHeight);
          const lowestP = 127 - Math.floor(maxY / ctx.rowHeight);

          const overlapping = queryOverlappingNotesInSnapshot(
            ctx.activeTrack,
            minTick - 1,
            maxTick + 1,
          );
          const hits = overlapping
            .filter(
              (n) =>
                n.endTick >= minTick &&
                n.startTick <= maxTick &&
                n.pitch >= lowestP &&
                n.pitch <= highestP,
            )
            .map((n) => n.id);
          if (drag.additive) {
            const merged = new Set(drag.initialSelection);
            for (const id of hits) merged.add(id);
            ctx.interactiveCtrl.setSelectedNotes(merged);
          } else {
            ctx.interactiveCtrl.setSelectedNotes(hits);
          }
        }
      } else if (drag.kind === 'velocity' && drag.updates.size > 0) {
        // O(1) per updated note via `activeNoteById` Map lookup
        const updates: Array<{
          id: string;
          pitch: number;
          startTick: number;
          durationTicks: number;
          velocity: number;
        }> = [];
        for (const [id, velocity] of drag.updates.entries()) {
          const existing = ctx.activeNoteById.get(id);
          if (existing) {
            updates.push({
              id,
              pitch: existing.pitch,
              startTick: existing.startTick,
              durationTicks: existing.durationTicks,
              velocity,
            });
          }
        }
        if (updates.length > 0) {
          ctx.interactiveCtrl.transformNotes(updates, 'ベロシティ変更');
          ctx.onProjectMutated?.();
        }
      }
      setDragRenderTick((c) => c + 1);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerCancel);
    window.addEventListener('blur', abortAllPointerInteractions);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerCancel);
      window.removeEventListener('blur', abortAllPointerInteractions);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [applyVelocityAtPoint, getCanvasCoords]);

  // Imperative `{ passive: false }` wheel listener so `e.preventDefault()` executes cleanly
  // without Chromium passive-listener console errors (`[R2-HIGH-02]`), with COMPASS cursor-anchored zoom
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !hasInteractiveCanvas) return;

    const onNativeWheel = (e: WheelEvent) => {
      e.preventDefault();
      const ctx = pointerContextRef.current;
      const rawDx = Number.isFinite(e.deltaX)
        ? Math.max(-MAX_WHEEL_DELTA_PX, Math.min(MAX_WHEEL_DELTA_PX, e.deltaX))
        : 0;
      const rawDy = Number.isFinite(e.deltaY)
        ? Math.max(-MAX_WHEEL_DELTA_PX, Math.min(MAX_WHEEL_DELTA_PX, e.deltaY))
        : 0;

      const rect = canvas.getBoundingClientRect();
      const hasValidClient =
        Number.isFinite(e.clientX) &&
        Number.isFinite(e.clientY) &&
        Number.isFinite(rect.left) &&
        Number.isFinite(rect.top) &&
        rect.width > 0 &&
        rect.height > 0;
      const localX = hasValidClient
        ? Math.max(KEYBOARD_WIDTH, Math.min(rect.width, e.clientX - rect.left))
        : KEYBOARD_WIDTH;
      const localY = hasValidClient ? e.clientY - rect.top : RULER_HEIGHT + 120;

      if (e.ctrlKey || e.metaKey) {
        if (rawDy === 0) return;
        if (e.shiftKey) {
          // Cursor-anchored vertical pitch-row zoom (`Ctrl + Shift + Wheel`)
          const oldRh = clampRowHeight(ctx.rowHeight, 16);
          const nextRh = clampRowHeight(oldRh + (rawDy < 0 ? 2 : -2), oldRh);
          if (nextRh === oldRh) return;
          const gridHeight = Math.max(
            1,
            ctx.canvasHeight - RULER_HEIGHT - computeVelocityLaneHeight(ctx.canvasHeight),
          );
          const relGridY = Math.max(0, Math.min(gridHeight, localY - RULER_HEIGHT));
          const worldRowFromTop = (ctx.scrollPxY + relGridY) / oldRh;
          const nextMaxScrollY = Math.max(0, TOTAL_PITCHES * nextRh - gridHeight);
          const nextScrollY = clampScrollPxY(
            worldRowFromTop * nextRh - relGridY,
            nextMaxScrollY,
            ctx.scrollPxY,
          );
          ctx.rowHeight = nextRh;
          ctx.scrollPxY = nextScrollY;
          setRowHeight(nextRh);
          setScrollPxY(nextScrollY);
          return;
        }

        // Cursor-anchored horizontal timeline zoom (`Ctrl + Wheel`)
        const factor = rawDy < 0 ? 1.15 : 1 / 1.15;
        const safePpqnVal =
          Number.isFinite(ctx.ppqn) && ctx.ppqn >= 1 ? Math.min(32767, ctx.ppqn) : 480;
        const oldPpb = clampPixelsPerBeat(ctx.pixelsPerBeat, 64);
        const nextPpb = clampPixelsPerBeat(oldPpb * factor, oldPpb);
        if (nextPpb === oldPpb) return;
        const oldPxPerTick = oldPpb / safePpqnVal;
        const nextPxPerTick = nextPpb / safePpqnVal;
        const gridOffsetX = Math.max(0, localX - KEYBOARD_WIDTH);
        const anchorTick = ctx.scrollTickX + gridOffsetX / oldPxPerTick;
        const nextScrollTick = clampScrollTickX(
          anchorTick - gridOffsetX / nextPxPerTick,
          safePpqnVal,
          ctx.scrollTickX,
        );
        ctx.pixelsPerBeat = nextPpb;
        ctx.pixelsPerTick = nextPxPerTick;
        ctx.scrollTickX = nextScrollTick;
        setPixelsPerBeat(nextPpb);
        setScrollTickX(nextScrollTick);
        return;
      }

      if (e.shiftKey || Math.abs(rawDx) > Math.abs(rawDy)) {
        const deltaPx = e.shiftKey ? rawDy : rawDx;
        if (deltaPx === 0) return;
        const safePxPerTick =
          Number.isFinite(ctx.pixelsPerTick) && ctx.pixelsPerTick > 0
            ? ctx.pixelsPerTick
            : 64 / Math.max(1, ctx.ppqn);
        const baseX = clampScrollTickX(ctx.scrollTickX, ctx.ppqn, 0);
        const nextX = clampScrollTickX(baseX + deltaPx / safePxPerTick, ctx.ppqn, baseX);
        ctx.scrollTickX = nextX;
        setScrollTickX(nextX);
      } else {
        if (rawDy === 0) return;
        const gridHeight = Math.max(
          1,
          ctx.canvasHeight - RULER_HEIGHT - computeVelocityLaneHeight(ctx.canvasHeight),
        );
        const maxScrollY = Math.max(0, TOTAL_PITCHES * ctx.rowHeight - gridHeight);
        const baseY = clampScrollPxY(ctx.scrollPxY, maxScrollY, 0);
        const nextY = clampScrollPxY(baseY + rawDy, maxScrollY, baseY);
        ctx.scrollPxY = nextY;
        setScrollPxY(nextY);
      }
    };

    canvas.addEventListener('wheel', onNativeWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', onNativeWheel);
    };
  }, [hasInteractiveCanvas]);

  useEffect(() => {
    if (prevHasInteractiveCanvasRef.current === hasInteractiveCanvas) {
      return;
    }
    prevHasInteractiveCanvasRef.current = hasInteractiveCanvas;
    const switchModality = standbySwitchModalityRef.current;
    standbySwitchModalityRef.current = null;
    const effectiveModality = switchModality ?? lastInputModalityRef.current;
    if (!isActive || typeof document === 'undefined') {
      return;
    }
    const activeEl = document.activeElement;
    if (!activeEl || activeEl === document.body) {
      const launchBtn =
        pianoRollRootRef.current?.querySelector<HTMLButtonElement>(
          '[data-testid="launch-compass-ai-button"]',
        ) ?? null;
      if (launchBtn) {
        if (hasInteractiveCanvas) {
          if (effectiveModality === 'pointer') {
            focusButtonWithPointerModality(launchBtn);
          } else {
            launchBtn.focus();
          }
        } else if (effectiveModality === 'keyboard') {
          launchBtn.focus();
        }
      }
    }
  }, [focusButtonWithPointerModality, hasInteractiveCanvas, isActive]);

  // Fallback view when external Cubase / Ableton bridge is active (`LSP`)
  if (!interactiveCtrl) {
    if (isSecondaryPopout) {
      return (
        <div
          ref={pianoRollRootRef}
          data-testid="popout-external-daw-standby"
          className="flex-1 min-h-0 w-full max-w-full overflow-y-auto overflow-x-hidden bg-transparent px-4 sm:px-6 py-8 text-zinc-100"
        >
          <div className="mx-auto max-w-xl space-y-6">
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-zinc-100">
                外部DAW接続中 ({safeControllerName})
              </h2>
              <p className="mt-2 text-sm text-zinc-400 leading-relaxed max-w-md">
                現在メインウィンドウは外部DAWブリッジに接続されています。対象トラックや小節範囲の設定はメインウィンドウで行い、内蔵ピアノロールを編集する場合はデモスタジオへ切り替えてください。
              </p>
            </div>

            <div className="border-t border-zinc-800/90 pt-5 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={onLaunchCompassAi}
                aria-keyshortcuts="Shift+G"
                data-testid="launch-compass-ai-button"
                className="inline-flex min-h-[44px] items-center gap-2 rounded-full bg-zinc-100 px-5 text-sm font-semibold text-zinc-950 hover:bg-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <span>COMPASS AI を起動</span>
                <kbd className="hidden sm:inline-block font-mono text-xs text-zinc-700">Shift+G</kbd>
              </button>
              <button
                type="button"
                disabled={Boolean(isGenerating)}
                aria-description={isGenerating ? '生成処理中は切り替えできません' : undefined}
                data-tooltip={isGenerating ? '生成処理中は切り替えできません' : undefined}
                data-tooltip-when-disabled="true"
                data-testid="popout-switch-to-demo-button"
                onClick={(e) => {
                  standbySwitchModalityRef.current = isPointerClickEvent(e)
                    ? 'pointer'
                    : 'keyboard';
                  onSwitchToDemoController?.();
                }}
                className="min-h-[44px] rounded-full border border-[#82828B] bg-studio-900/80 px-5 text-sm text-zinc-100 hover:border-zinc-300 disabled:border-[#82828B]/40 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
              >
                内蔵ピアノロールへ切り替え
              </button>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div
        ref={pianoRollRootRef}
        className="flex-1 min-h-0 w-full max-w-full overflow-y-auto overflow-x-hidden bg-transparent px-4 sm:px-6 py-8 text-zinc-100"
      >
        <div className="mx-auto max-w-2xl space-y-6">
          <div>
            <h2 className="text-xl font-semibold tracking-tight text-zinc-100">
              外部DAWブリッジ接続モード ({safeControllerName})
            </h2>
            <p className="mt-2 text-sm text-zinc-400 leading-relaxed max-w-md">
              外部DAWコントローラーへ送出する対象小節・テンポ・拍子・生成候補をここで制御します。内蔵ピアノロールで編集する場合はデモスタジオへ切り替えてください。
            </p>
          </div>

          {/* External DAW Target Range, Track, Tempo & Meter Controls */}
          <div className="border-t border-zinc-800/90 pt-5 space-y-4">
            <div className="flex flex-wrap items-center gap-4">
              {safeTracks.length > 0 && (
                <div className="flex items-center gap-2 min-w-0 max-w-full">
                  <label
                    htmlFor="external-daw-track-select"
                    className="text-xs font-mono text-zinc-300 shrink-0"
                  >
                    対象トラック
                  </label>
                  <div className="relative min-w-0 max-w-[220px]">
                    <select
                      id="external-daw-track-select"
                      value={activeTrack?.id ?? safeTracks[0]?.id ?? ''}
                      onChange={(e) => {
                        controller.setActiveTrack(e.target.value);
                        onProjectMutated?.();
                      }}
                      className="min-h-[44px] w-full max-w-full appearance-none truncate rounded-xl border border-[#82828B] bg-zinc-950/90 pl-3 pr-8 font-mono text-xs text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                    >
                      {safeTracks.map((trk, idx) => (
                        <option key={`${idx}-${trk.id}`} value={trk.id}>
                          {typeof trk.name === 'string' && trk.name ? trk.name : trk.id}
                        </option>
                      ))}
                    </select>
                    <ChevronDown
                      className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                      strokeWidth={1.75}
                    />
                  </div>
                </div>
              )}

              <div className="flex items-center gap-1.5">
                <span className="text-xs font-mono text-zinc-300">対象小節</span>
                <input
                  type="number"
                  min={1}
                  max={1024}
                  aria-label="生成開始小節"
                  value={startBarDraft !== null ? startBarDraft : String(safeStartMeasure)}
                  onFocus={() => {
                    cancelStartBarDraftRef.current = false;
                    startBarAnchorEndRef.current = safeEndMeasure;
                    rangeFocusAnchorRef.current = { start: safeStartMeasure, end: safeEndMeasure };
                    setStartBarDraft(String(safeStartMeasure));
                  }}
                  onChange={(e) => {
                    const raw = e.target.value;
                    const isFocused = document.activeElement === e.currentTarget;
                    if (!isFocused) {
                      const s = Math.max(1, Math.min(1024, Math.round(Number(raw) || 1)));
                      const end = Math.max(s, Math.min(s + 7, safeEndMeasure));
                      controller.setTargetMeasureRange(s, end);
                      onProjectMutated?.();
                      return;
                    }
                    setStartBarDraft(raw);
                    const parsed = Number(raw);
                    if (raw.trim() !== '' && Number.isFinite(parsed) && parsed >= 1 && parsed <= 1024) {
                      const s = Math.round(parsed);
                      const end = Math.max(s, Math.min(s + 7, startBarAnchorEndRef.current));
                      controller.setTargetMeasureRange(s, end);
                      onProjectMutated?.();
                    }
                  }}
                  onBlur={() => {
                    if (cancelStartBarDraftRef.current) {
                      cancelStartBarDraftRef.current = false;
                      setStartBarDraft(null);
                      return;
                    }
                    if (startBarDraft !== null) {
                      const trimmed = startBarDraft.trim();
                      if (trimmed !== '' && Number.isFinite(Number(trimmed))) {
                        const s = Math.max(1, Math.min(1024, Math.round(Number(trimmed))));
                        const end = Math.max(s, Math.min(s + 7, startBarAnchorEndRef.current));
                        const curRange = controller.getSnapshot().targetRange;
                        if (curRange.startMeasure !== s || curRange.endMeasure !== end) {
                          controller.setTargetMeasureRange(s, end);
                          onProjectMutated?.();
                        }
                      }
                      setStartBarDraft(null);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (isImeComposing(e)) return;
                    if (e.key === 'Enter') {
                      e.currentTarget.blur();
                    } else if (e.key === 'Escape') {
                      e.preventDefault();
                      e.stopPropagation();
                      cancelStartBarDraftRef.current = true;
                      setStartBarDraft(null);
                      const anchor = rangeFocusAnchorRef.current;
                      const curRange = controller.getSnapshot().targetRange;
                      if (curRange.startMeasure !== anchor.start || curRange.endMeasure !== anchor.end) {
                        controller.setTargetMeasureRange(anchor.start, anchor.end);
                        onProjectMutated?.();
                      }
                      e.currentTarget.blur();
                      cancelStartBarDraftRef.current = false;
                    }
                  }}
                  className="min-h-[44px] w-16 rounded-xl border border-[#82828B] bg-zinc-950/90 px-2 text-center font-mono text-xs text-zinc-100 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                />
                <span className="text-zinc-400 font-mono" aria-hidden="true">–</span>
                <input
                  type="number"
                  min={safeStartMeasure}
                  max={safeStartMeasure + 7}
                  aria-label="生成終了小節"
                  value={endBarDraft !== null ? endBarDraft : String(safeEndMeasure)}
                  onFocus={() => {
                    cancelEndBarDraftRef.current = false;
                    rangeFocusAnchorRef.current = { start: safeStartMeasure, end: safeEndMeasure };
                    setEndBarDraft(String(safeEndMeasure));
                  }}
                  onChange={(e) => {
                    const raw = e.target.value;
                    const isFocused = document.activeElement === e.currentTarget;
                    if (!isFocused) {
                      const rawEnd = Math.max(
                        safeStartMeasure,
                        Math.round(Number(raw) || safeStartMeasure),
                      );
                      controller.setTargetMeasureRange(
                        safeStartMeasure,
                        Math.min(safeStartMeasure + 7, rawEnd),
                      );
                      onProjectMutated?.();
                      return;
                    }
                    setEndBarDraft(raw);
                    const parsed = Number(raw);
                    if (
                      raw.trim() !== '' &&
                      Number.isFinite(parsed) &&
                      parsed >= safeStartMeasure &&
                      parsed <= safeStartMeasure + 7
                    ) {
                      controller.setTargetMeasureRange(safeStartMeasure, Math.round(parsed));
                      onProjectMutated?.();
                    }
                  }}
                  onBlur={() => {
                    if (cancelEndBarDraftRef.current) {
                      cancelEndBarDraftRef.current = false;
                      setEndBarDraft(null);
                      return;
                    }
                    if (endBarDraft !== null) {
                      const trimmed = endBarDraft.trim();
                      if (trimmed !== '' && Number.isFinite(Number(trimmed))) {
                        const clampedEnd = Math.max(
                          safeStartMeasure,
                          Math.min(safeStartMeasure + 7, Math.round(Number(trimmed))),
                        );
                        const curRange = controller.getSnapshot().targetRange;
                        if (curRange.startMeasure !== safeStartMeasure || curRange.endMeasure !== clampedEnd) {
                          controller.setTargetMeasureRange(safeStartMeasure, clampedEnd);
                          onProjectMutated?.();
                        }
                      }
                      setEndBarDraft(null);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (isImeComposing(e)) return;
                    if (e.key === 'Enter') {
                      e.currentTarget.blur();
                    } else if (e.key === 'Escape') {
                      e.preventDefault();
                      e.stopPropagation();
                      cancelEndBarDraftRef.current = true;
                      setEndBarDraft(null);
                      const anchor = rangeFocusAnchorRef.current;
                      const curRange = controller.getSnapshot().targetRange;
                      if (curRange.startMeasure !== anchor.start || curRange.endMeasure !== anchor.end) {
                        controller.setTargetMeasureRange(anchor.start, anchor.end);
                        onProjectMutated?.();
                      }
                      e.currentTarget.blur();
                      cancelEndBarDraftRef.current = false;
                    }
                  }}
                  className="min-h-[44px] w-16 rounded-xl border border-[#82828B] bg-zinc-950/90 px-2 text-center font-mono text-xs text-zinc-100 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                />
              </div>

              <div className="flex items-center gap-1.5">
                <label htmlFor="pianoroll-bpm" className="text-xs font-mono text-zinc-300">
                  BPM
                </label>
                <input
                  id="pianoroll-bpm"
                  type="number"
                  min={20}
                  max={300}
                  value={bpmDraft !== null ? bpmDraft : String(safeBpm)}
                  onFocus={() => {
                    cancelBpmDraftRef.current = false;
                    bpmFocusAnchorRef.current = safeBpm;
                    setBpmDraft(String(safeBpm));
                  }}
                  onChange={(e) => {
                    const raw = e.target.value;
                    const isFocused = document.activeElement === e.currentTarget;
                    if (!isFocused) {
                      controller.setBpm(Number(raw) || 120);
                      onProjectMutated?.();
                      return;
                    }
                    setBpmDraft(raw);
                    const parsed = Number(raw);
                    if (raw.trim() !== '' && Number.isFinite(parsed) && parsed >= 20 && parsed <= 300) {
                      controller.setBpm(parsed);
                      onProjectMutated?.();
                    }
                  }}
                  onBlur={() => {
                    if (cancelBpmDraftRef.current) {
                      cancelBpmDraftRef.current = false;
                      setBpmDraft(null);
                      return;
                    }
                    if (bpmDraft !== null) {
                      const trimmed = bpmDraft.trim();
                      if (trimmed !== '' && Number.isFinite(Number(trimmed))) {
                        const prevBpm = controller.getSnapshot().bpm;
                        controller.setBpm(Number(trimmed));
                        if (controller.getSnapshot().bpm !== prevBpm) {
                          onProjectMutated?.();
                        }
                      }
                      setBpmDraft(null);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (isImeComposing(e)) return;
                    if (e.key === 'Enter') {
                      e.currentTarget.blur();
                    } else if (e.key === 'Escape') {
                      e.preventDefault();
                      e.stopPropagation();
                      cancelBpmDraftRef.current = true;
                      setBpmDraft(null);
                      if (controller.getSnapshot().bpm !== bpmFocusAnchorRef.current) {
                        controller.setBpm(bpmFocusAnchorRef.current);
                        onProjectMutated?.();
                      }
                      e.currentTarget.blur();
                      cancelBpmDraftRef.current = false;
                    }
                  }}
                  className="min-h-[44px] w-16 rounded-xl border border-[#82828B] bg-zinc-950/90 px-2.5 text-center font-mono text-xs text-zinc-100 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                />
              </div>

              <div className="relative">
                <select
                  aria-label="拍子記号"
                  data-testid="pianoroll-timesig-select"
                  value={safeTimeSigLabel}
                  onChange={(e) => {
                    const [numStr, denStr] = e.target.value.split('/');
                    const num = Number(numStr);
                    const den = Number(denStr);
                    if (Number.isFinite(num) && Number.isFinite(den) && num >= 1 && den >= 1) {
                      controller.setTimeSignature(num, den);
                      onProjectMutated?.();
                    }
                  }}
                  className="min-h-[44px] min-w-[66px] appearance-none rounded-xl border border-[#82828B] bg-zinc-950/90 pl-2.5 pr-7 font-mono text-xs text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                >
                  {!TIME_SIGNATURE_PRESETS.includes(safeTimeSigLabel) && (
                    <option value={safeTimeSigLabel}>{safeTimeSigLabel}</option>
                  )}
                  {TIME_SIGNATURE_PRESETS.map((ts) => (
                    <option key={ts} value={ts}>
                      {ts}
                    </option>
                  ))}
                </select>
                <ChevronDown
                  className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                  strokeWidth={1.75}
                />
              </div>
            </div>

            {safeCandidates.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="font-mono text-xs text-zinc-300">生成候補:</span>
                {safeCandidates.map((cand, idx) => {
                  const safeCandIdx =
                    typeof cand.index === 'number' &&
                    Number.isFinite(cand.index) &&
                    cand.index >= 0
                      ? Math.trunc(cand.index)
                      : idx;
                  const safeActiveCandIdx =
                    typeof session?.activeCandidateIndex === 'number' &&
                    Number.isFinite(session.activeCandidateIndex)
                      ? Math.trunc(session.activeCandidateIndex)
                      : 0;
                  const isSelectedCand = safeActiveCandIdx === safeCandIdx;
                  return (
                    <button
                      key={`${idx}-${safeCandIdx}`}
                      type="button"
                      aria-pressed={isSelectedCand}
                      onClick={() => {
                        void controller
                          .selectGeneratedCandidate(safeCandIdx)
                          .then(() => onProjectMutated?.());
                      }}
                      className={`min-h-[44px] min-w-[44px] rounded-full px-4 font-mono text-xs border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                        isSelectedCand
                          ? 'border-sky-400 bg-sky-950/40 text-sky-200 font-medium'
                          : 'border-[#82828B] bg-studio-900/90 text-zinc-200 hover:border-zinc-300'
                      }`}
                    >
                      候補 {safeCandIdx + 1}
                    </button>
                  );
                })}
                {session?.latestReason && typeof session.latestReason === 'object' && (
                  <span className="font-mono text-xs text-zinc-400 ml-1">
                    Task:{' '}
                    {typeof session.latestReason.task === 'string' && session.latestReason.task
                      ? session.latestReason.task
                      : 'meta'}
                    {typeof session.latestReason.key === 'string' && session.latestReason.key
                      ? ` | Key: ${session.latestReason.key}`
                      : ''}
                    {typeof session.latestReason.gen_measure_count === 'number' &&
                    Number.isFinite(session.latestReason.gen_measure_count) &&
                    session.latestReason.gen_measure_count > 0
                      ? ` | Bars: ${Math.round(session.latestReason.gen_measure_count)}`
                      : ''}
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="border-t border-zinc-800/90 pt-5 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={onLaunchCompassAi}
              aria-keyshortcuts="Shift+G"
              data-testid="launch-compass-ai-button"
              className="inline-flex min-h-[44px] items-center gap-2 rounded-full bg-zinc-100 px-5 text-sm font-semibold text-zinc-950 hover:bg-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              <span>COMPASS AI を起動</span>
              <kbd className="hidden sm:inline-block font-mono text-xs text-zinc-700">Shift+G</kbd>
            </button>
            <button
              type="button"
              disabled={Boolean(isGenerating)}
              aria-description={isGenerating ? '生成処理中は切り替えできません' : undefined}
              data-tooltip={isGenerating ? '生成処理中は切り替えできません' : undefined}
              data-tooltip-when-disabled="true"
              onClick={(e) => {
                standbySwitchModalityRef.current = isPointerClickEvent(e) ? 'pointer' : 'keyboard';
                onSwitchToDemoController?.();
              }}
              className="min-h-[44px] rounded-full border border-[#82828B] bg-studio-900/80 px-5 text-sm text-zinc-100 hover:border-zinc-300 disabled:border-[#82828B]/40 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            >
              内蔵ピアノロールへ切り替え
            </button>
            {(safeCandidates.length > 0 ||
              safeTracks.some((t) => Array.isArray(t.notes) && t.notes.length > 0)) && (
              <button
                type="button"
                onClick={handleExportMidi}
                className="min-h-[44px] rounded-full border border-[#82828B] bg-studio-900/80 px-5 text-sm text-zinc-200 hover:border-zinc-300 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
              >
                生成済みMIDIを保存
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={pianoRollRootRef}
      className="flex flex-1 [@media(min-width:768px)_and_(min-height:520px)]:min-h-0 w-full max-w-full flex-col overflow-x-hidden overflow-y-visible [@media(min-width:768px)_and_(min-height:520px)]:overflow-y-hidden bg-transparent text-zinc-100"
    >
      {/* Hidden File Input for `MIDIをアップロード` */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".mid,.midi,audio/midi,audio/x-midi"
        onChange={handleMidiFileSelected}
        className="hidden"
        data-testid="midi-file-input"
      />

      {/* Top Studio Piano Roll Toolbar (Icon-First Universal Actions + Explicit Domain Labels) */}
      <div
        aria-hidden={isImportModalRendered ? 'true' : undefined}
        {...(isImportModalRendered ? { inert: '' } : {})}
        className={`relative z-20 flex flex-wrap items-center 2xl:justify-between gap-1.5 border-b border-zinc-800/85 bg-studio-900/80 px-2.5 sm:px-3 py-1.5 ${
          isImportModalRendered ? 'pointer-events-none select-none' : ''
        }`}
      >
        {/* Left Group: Track Rack Toggle + MIDI I/O + Transport + Tempo/Meter */}
        <div className="contents 2xl:flex 2xl:flex-wrap 2xl:items-center 2xl:gap-1.5">
          <button
            type="button"
            data-testid="toggle-track-rack-button"
            aria-pressed={showTrackRack}
            aria-label="トラック一覧の表示切替"
            data-tooltip="トラック一覧"
            data-tooltip-align="start"
            onClick={() => {
              userToggledTrackRackRef.current = true;
              setShowTrackRack((v) => !v);
            }}
            className={`inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
              showTrackRack
                ? 'border-sky-400 bg-sky-950/35 text-sky-200'
                : 'border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100'
            }`}
          >
            <PanelLeft className="h-4 w-4" strokeWidth={1.75} />
          </button>

          <button
            ref={uploadMidiButtonRef}
            type="button"
            aria-label="MIDIをアップロード"
            onClick={(e) => {
              uploadOpenedViaPointerRef.current = isPointerClickEvent(e);
              fileInputRef.current?.click();
            }}
            data-testid="upload-midi-button"
            className="inline-flex min-h-[44px] min-w-[44px] items-center gap-1.5 rounded-full border border-[#82828B] bg-zinc-900/90 px-3 text-xs font-medium text-zinc-100 hover:border-zinc-300 hover:bg-zinc-800 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            <Upload className="h-4 w-4 shrink-0" strokeWidth={1.75} />
            <span>MIDI読込</span>
          </button>

          <button
            type="button"
            data-testid="load-sample-arrangement-button"
            aria-label="サンプルアレンジを読込"
            data-tooltip="サンプル4小節を読込"
            onClick={(e) => {
              const viaPointer = isPointerClickEvent(e);
              midiUploadSeqRef.current += 1;
              const hasExistingNotes = safeTracks.some(
                (t) => Array.isArray(t.notes) && t.notes.length > 0,
              );
              if (hasExistingNotes) {
                openImportModal(
                  {
                    fileName: 'compass_sample_arrangement.mid',
                    parsed: StandardMidiCodec.createDemoMidiPreset(),
                    isSamplePreset: true,
                  },
                  e.currentTarget,
                  viaPointer,
                );
                return;
              }
              interactiveCtrl.loadBuiltInDemoArrangement();
              const snap = interactiveCtrl.getSnapshot();
              if (snap.tracks[0]) {
                centerViewportOnNotes(snap.tracks[0].notes, 0);
              }
              setUploadNotice('検証用サンプルアレンジ (4小節 Piano + Sax) を読み込みました。');
              onProjectMutated?.();
            }}
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center gap-1.5 rounded-full border border-[#82828B] bg-zinc-950/90 px-2.5 text-xs font-medium text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            <Music className="h-4 w-4 shrink-0" strokeWidth={1.75} />
            <span>サンプル</span>
          </button>

          <button
            type="button"
            onClick={handleExportMidi}
            aria-label="MIDIを書き出し (Ctrl+S)"
            data-tooltip="MIDI書き出し (Ctrl+S)"
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center gap-1.5 rounded-full border border-[#82828B] bg-zinc-950/90 px-2.5 text-xs font-medium text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            <Download className="h-4 w-4 shrink-0" strokeWidth={1.75} />
            <span>保存</span>
          </button>

          <div className="hidden 2xl:block h-5 w-px bg-zinc-800 mx-0.5" />

          {/* Transport Controls */}
          <button
            type="button"
            onClick={() => {
              if (safeIsPlaying) interactiveCtrl.pause();
              else void interactiveCtrl.play();
            }}
            aria-label={safeIsPlaying ? '一時停止' : '再生'}
            data-tooltip={safeIsPlaying ? '一時停止 (Space)' : '再生 (Space)'}
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-100 hover:border-zinc-300 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            {safeIsPlaying ? (
              <Pause className="h-4 w-4" strokeWidth={1.75} />
            ) : (
              <Play className="h-4 w-4" strokeWidth={1.75} />
            )}
          </button>

          <button
            type="button"
            onClick={() => interactiveCtrl.stop()}
            aria-label="停止して開始位置へ戻る"
            data-tooltip="停止して先頭へ"
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            <Square className="h-4 w-4" strokeWidth={1.75} />
          </button>

          <div className="flex items-center gap-1">
            <label htmlFor="pianoroll-bpm" className="text-xs font-mono text-zinc-400">
              BPM
            </label>
            <input
              id="pianoroll-bpm"
              type="number"
              min={20}
              max={300}
              value={bpmDraft !== null ? bpmDraft : String(safeBpm)}
              onFocus={() => {
                cancelBpmDraftRef.current = false;
                bpmFocusAnchorRef.current = safeBpm;
                setBpmDraft(String(safeBpm));
              }}
              onChange={(e) => {
                const raw = e.target.value;
                const isFocused = document.activeElement === e.currentTarget;
                if (!isFocused) {
                  interactiveCtrl.setBpm(Number(raw) || 120);
                  onProjectMutated?.();
                  return;
                }
                setBpmDraft(raw);
                const parsed = Number(raw);
                if (raw.trim() !== '' && Number.isFinite(parsed) && parsed >= 20 && parsed <= 300) {
                  interactiveCtrl.setBpm(parsed);
                  onProjectMutated?.();
                }
              }}
              onBlur={() => {
                if (cancelBpmDraftRef.current) {
                  cancelBpmDraftRef.current = false;
                  setBpmDraft(null);
                  return;
                }
                if (bpmDraft !== null) {
                  const trimmed = bpmDraft.trim();
                  if (trimmed !== '' && Number.isFinite(Number(trimmed))) {
                    const prevBpm = interactiveCtrl.getSnapshot().bpm;
                    interactiveCtrl.setBpm(Number(trimmed));
                    if (interactiveCtrl.getSnapshot().bpm !== prevBpm) {
                      onProjectMutated?.();
                    }
                  }
                  setBpmDraft(null);
                }
              }}
              onKeyDown={(e) => {
                if (isImeComposing(e)) return;
                if (e.key === 'Enter') {
                  e.currentTarget.blur();
                } else if (e.key === 'Escape') {
                  e.preventDefault();
                  e.stopPropagation();
                  cancelBpmDraftRef.current = true;
                  setBpmDraft(null);
                  if (interactiveCtrl.getSnapshot().bpm !== bpmFocusAnchorRef.current) {
                    interactiveCtrl.setBpm(bpmFocusAnchorRef.current);
                    onProjectMutated?.();
                  }
                  e.currentTarget.blur();
                  cancelBpmDraftRef.current = false;
                }
              }}
              className="min-h-[44px] min-w-[44px] w-12 rounded-xl border border-[#82828B] bg-zinc-950/90 px-1 text-center font-mono text-xs text-zinc-100 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            />
          </div>

          <div className="relative">
            <select
              aria-label="拍子記号"
              data-testid="pianoroll-timesig-select"
              value={safeTimeSigLabel}
              onChange={(e) => {
                const [numStr, denStr] = e.target.value.split('/');
                const num = Number(numStr);
                const den = Number(denStr);
                if (Number.isFinite(num) && Number.isFinite(den) && num >= 1 && den >= 1) {
                  controller.setTimeSignature(num, den);
                  onProjectMutated?.();
                }
              }}
              className="min-h-[44px] min-w-[60px] appearance-none rounded-xl border border-[#82828B] bg-zinc-950/90 pl-2 pr-6 font-mono text-xs text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            >
              {!TIME_SIGNATURE_PRESETS.includes(safeTimeSigLabel) && (
                <option value={safeTimeSigLabel}>{safeTimeSigLabel}</option>
              )}
              {TIME_SIGNATURE_PRESETS.map((ts) => (
                <option key={ts} value={ts}>
                  {ts}
                </option>
              ))}
            </select>
            <ChevronDown
              className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
              strokeWidth={1.75}
            />
          </div>
        </div>

        {/* Right Group: Optional Pencil Toggle, Snap + Triplet, Target Range + Range Sel, Undo/Redo, and Primary `COMPASS AI を起動` */}
        <div className="contents 2xl:flex 2xl:flex-wrap 2xl:items-center 2xl:gap-1.5">
          <button
            type="button"
            aria-pressed={toolMode === 'draw'}
            aria-label="ペン入力"
            data-tooltip="ペン入力切替"
            onClick={() => {
              setToolMode((prev) => {
                const next = prev === 'draw' ? 'select' : 'draw';
                updateCanvasCursor(next === 'draw' ? 'cursor-crosshair' : 'cursor-default');
                return next;
              });
            }}
            className={`inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
              toolMode === 'draw'
                ? 'border-sky-400 bg-sky-950/40 text-sky-200'
                : 'border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100'
            }`}
          >
            <Pencil className="h-4 w-4" strokeWidth={1.75} />
          </button>

          <div className="relative">
            <select
              aria-label="グリッドクオンタイズ"
              value={safeQuantDiv}
              onChange={(e) => {
                interactiveCtrl.setQuantize(
                  Number(e.target.value) as 4 | 8 | 16 | 32,
                  Boolean(session?.isTriplet),
                );
                onProjectMutated?.();
              }}
              className="min-h-[44px] min-w-[64px] appearance-none rounded-xl border border-[#82828B] bg-zinc-950/90 pl-2 pr-6 font-mono text-xs text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            >
              <option value={4}>1/4</option>
              <option value={8}>1/8</option>
              <option value={16}>1/16</option>
              <option value={32}>1/32</option>
            </select>
            <ChevronDown
              className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
              strokeWidth={1.75}
            />
          </div>

          <button
            type="button"
            aria-pressed={Boolean(session?.isTriplet)}
            aria-label="3連符グリッド切替"
            data-tooltip="3連符グリッド切替"
            onClick={() => {
              interactiveCtrl.setQuantize(safeQuantDiv, !session?.isTriplet);
              onProjectMutated?.();
            }}
            className={`inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border px-2.5 font-mono text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
              session?.isTriplet
                ? 'border-sky-400 bg-sky-950/40 text-sky-200 font-medium'
                : 'border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100'
            }`}
          >
            3連
          </button>

          <div className="flex items-center gap-1">
            <span className="text-xs font-mono text-zinc-400">
              <span className="hidden 2xl:inline">対象</span>小節
            </span>
            <input
              type="number"
              min={1}
              max={1024}
              aria-label="生成開始小節"
              value={startBarDraft !== null ? startBarDraft : String(safeStartMeasure)}
              onFocus={() => {
                cancelStartBarDraftRef.current = false;
                startBarAnchorEndRef.current = safeEndMeasure;
                rangeFocusAnchorRef.current = { start: safeStartMeasure, end: safeEndMeasure };
                setStartBarDraft(String(safeStartMeasure));
              }}
              onChange={(e) => {
                const raw = e.target.value;
                const isFocused = document.activeElement === e.currentTarget;
                if (!isFocused) {
                  const s = Math.max(1, Math.min(1024, Math.round(Number(raw) || 1)));
                  const end = Math.max(s, Math.min(s + 7, safeEndMeasure));
                  interactiveCtrl.setTargetMeasureRange(s, end);
                  onProjectMutated?.();
                  return;
                }
                setStartBarDraft(raw);
                const parsed = Number(raw);
                if (raw.trim() !== '' && Number.isFinite(parsed) && parsed >= 1 && parsed <= 1024) {
                  const s = Math.round(parsed);
                  const end = Math.max(s, Math.min(s + 7, startBarAnchorEndRef.current));
                  interactiveCtrl.setTargetMeasureRange(s, end);
                  onProjectMutated?.();
                }
              }}
              onBlur={() => {
                if (cancelStartBarDraftRef.current) {
                  cancelStartBarDraftRef.current = false;
                  setStartBarDraft(null);
                  return;
                }
                if (startBarDraft !== null) {
                  const trimmed = startBarDraft.trim();
                  if (trimmed !== '' && Number.isFinite(Number(trimmed))) {
                    const s = Math.max(1, Math.min(1024, Math.round(Number(trimmed))));
                    const end = Math.max(s, Math.min(s + 7, startBarAnchorEndRef.current));
                    const curRange = interactiveCtrl.getSnapshot().targetRange;
                    if (curRange.startMeasure !== s || curRange.endMeasure !== end) {
                      interactiveCtrl.setTargetMeasureRange(s, end);
                      onProjectMutated?.();
                    }
                  }
                  setStartBarDraft(null);
                }
              }}
              onKeyDown={(e) => {
                if (isImeComposing(e)) return;
                if (e.key === 'Enter') {
                  e.currentTarget.blur();
                } else if (e.key === 'Escape') {
                  e.preventDefault();
                  e.stopPropagation();
                  cancelStartBarDraftRef.current = true;
                  setStartBarDraft(null);
                  const anchor = rangeFocusAnchorRef.current;
                  const curRange = interactiveCtrl.getSnapshot().targetRange;
                  if (curRange.startMeasure !== anchor.start || curRange.endMeasure !== anchor.end) {
                    interactiveCtrl.setTargetMeasureRange(anchor.start, anchor.end);
                    onProjectMutated?.();
                  }
                  e.currentTarget.blur();
                  cancelStartBarDraftRef.current = false;
                }
              }}
              className="min-h-[44px] min-w-[44px] w-12 rounded-xl border border-[#82828B] bg-zinc-950/90 px-1 text-center font-mono text-xs text-zinc-100 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            />
            <span className="text-zinc-400 font-mono" aria-hidden="true">–</span>
            <input
              type="number"
              min={safeStartMeasure}
              max={safeStartMeasure + 7}
              aria-label="生成終了小節"
              value={endBarDraft !== null ? endBarDraft : String(safeEndMeasure)}
              onFocus={() => {
                cancelEndBarDraftRef.current = false;
                rangeFocusAnchorRef.current = { start: safeStartMeasure, end: safeEndMeasure };
                setEndBarDraft(String(safeEndMeasure));
              }}
              onChange={(e) => {
                const raw = e.target.value;
                const isFocused = document.activeElement === e.currentTarget;
                if (!isFocused) {
                  const rawEnd = Math.max(
                    safeStartMeasure,
                    Math.round(Number(raw) || safeStartMeasure),
                  );
                  interactiveCtrl.setTargetMeasureRange(
                    safeStartMeasure,
                    Math.min(safeStartMeasure + 7, rawEnd),
                  );
                  onProjectMutated?.();
                  return;
                }
                setEndBarDraft(raw);
                const parsed = Number(raw);
                if (
                  raw.trim() !== '' &&
                  Number.isFinite(parsed) &&
                  parsed >= safeStartMeasure &&
                  parsed <= safeStartMeasure + 7
                ) {
                  interactiveCtrl.setTargetMeasureRange(safeStartMeasure, Math.round(parsed));
                  onProjectMutated?.();
                }
              }}
              onBlur={() => {
                if (cancelEndBarDraftRef.current) {
                  cancelEndBarDraftRef.current = false;
                  setEndBarDraft(null);
                  return;
                }
                if (endBarDraft !== null) {
                  const trimmed = endBarDraft.trim();
                  if (trimmed !== '' && Number.isFinite(Number(trimmed))) {
                    const clampedEnd = Math.max(
                      safeStartMeasure,
                      Math.min(safeStartMeasure + 7, Math.round(Number(trimmed))),
                    );
                    const curRange = interactiveCtrl.getSnapshot().targetRange;
                    if (curRange.startMeasure !== safeStartMeasure || curRange.endMeasure !== clampedEnd) {
                      interactiveCtrl.setTargetMeasureRange(safeStartMeasure, clampedEnd);
                      onProjectMutated?.();
                    }
                  }
                  setEndBarDraft(null);
                }
              }}
              onKeyDown={(e) => {
                if (isImeComposing(e)) return;
                if (e.key === 'Enter') {
                  e.currentTarget.blur();
                } else if (e.key === 'Escape') {
                  e.preventDefault();
                  e.stopPropagation();
                  cancelEndBarDraftRef.current = true;
                  setEndBarDraft(null);
                  const anchor = rangeFocusAnchorRef.current;
                  const curRange = interactiveCtrl.getSnapshot().targetRange;
                  if (curRange.startMeasure !== anchor.start || curRange.endMeasure !== anchor.end) {
                    interactiveCtrl.setTargetMeasureRange(anchor.start, anchor.end);
                    onProjectMutated?.();
                  }
                  e.currentTarget.blur();
                  cancelEndBarDraftRef.current = false;
                }
              }}
              className="min-h-[44px] min-w-[44px] w-12 rounded-xl border border-[#82828B] bg-zinc-950/90 px-1 text-center font-mono text-xs text-zinc-100 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            />
          </div>

          <button
            type="button"
            onClick={handleSelectNotesInTargetRange}
            aria-label="対象小節内のノートを選択"
            data-tooltip="対象小節のノートを一括選択"
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center gap-1.5 rounded-full border border-[#82828B] bg-zinc-950/90 px-2.5 text-xs font-medium text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            <BoxSelect className="h-4 w-4 shrink-0" strokeWidth={1.75} />
            <span>範囲選択</span>
          </button>

          <button
            ref={undoButtonRef}
            type="button"
            disabled={!safeCanUndo}
            onClick={(e) => {
              const shouldHandoffFocus =
                typeof document !== 'undefined' &&
                document.activeElement === undoButtonRef.current &&
                !isPointerClickEvent(e);
              interactiveCtrl.undo();
              if (shouldHandoffFocus && !interactiveCtrl.getSnapshot().canUndo) {
                pendingToolbarFocusTargetRef.current = 'redo';
              }
              onProjectMutated?.();
            }}
            aria-label="元に戻す"
            aria-description={!safeCanUndo ? '元に戻す操作がありません (Ctrl+Z)' : undefined}
            data-tooltip={safeCanUndo ? '元に戻す (Ctrl+Z)' : '元に戻す操作がありません (Ctrl+Z)'}
            data-tooltip-when-disabled="true"
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 disabled:border-[#82828B]/45 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            <Undo2 className="h-4 w-4" strokeWidth={1.75} />
          </button>

          <button
            ref={redoButtonRef}
            type="button"
            disabled={!safeCanRedo}
            onClick={(e) => {
              const shouldHandoffFocus =
                typeof document !== 'undefined' &&
                document.activeElement === redoButtonRef.current &&
                !isPointerClickEvent(e);
              interactiveCtrl.redo();
              if (shouldHandoffFocus && !interactiveCtrl.getSnapshot().canRedo) {
                pendingToolbarFocusTargetRef.current = 'undo';
              }
              onProjectMutated?.();
            }}
            aria-label="やり直す"
            aria-description={!safeCanRedo ? 'やり直す操作がありません (Ctrl+Y)' : undefined}
            data-tooltip={safeCanRedo ? 'やり直す (Ctrl+Y)' : 'やり直す操作がありません (Ctrl+Y)'}
            data-tooltip-when-disabled="true"
            data-tooltip-align="end"
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 disabled:border-[#82828B]/45 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            <Redo2 className="h-4 w-4" strokeWidth={1.75} />
          </button>

          {/* Strictly Single Primary Solid White Action in Piano Roll View */}
          <button
            ref={launchCompassAiButtonRef}
            type="button"
            onClick={onLaunchCompassAi}
            aria-keyshortcuts="Shift+G"
            data-testid="launch-compass-ai-button"
            className="xl:ml-auto 2xl:ml-0 inline-flex min-h-[44px] min-w-[44px] items-center gap-2 rounded-full bg-zinc-100 px-3.5 text-xs font-semibold tracking-wide text-zinc-950 hover:bg-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            <span>COMPASS AI を起動</span>
            <kbd className="hidden 2xl:inline-block font-mono text-xs text-zinc-700">Shift+G</kbd>
          </button>
        </div>
      </div>

      {/* Status / Candidate Switcher / CoT Reason Strip */}
      {(uploadNotice || safeCandidates.length > 0) && (
        <div
          ref={secondaryStripRef}
          data-tooltip-strip="true"
          aria-hidden={isImportModalRendered ? 'true' : undefined}
          {...(isImportModalRendered ? { inert: '' } : {})}
          className={`relative z-10 flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800/85 bg-studio-900/65 px-3 sm:px-4 py-1.5 text-xs ${
            isImportModalRendered ? 'pointer-events-none select-none' : ''
          }`}
        >
          {safeCandidates.length > 0 ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-zinc-400">生成候補:</span>
              {safeCandidates.map((cand, idx) => {
                const safeCandIdx =
                  typeof cand.index === 'number' &&
                  Number.isFinite(cand.index) &&
                  cand.index >= 0
                    ? Math.trunc(cand.index)
                    : idx;
                const safeActiveCandIdx =
                  typeof session?.activeCandidateIndex === 'number' &&
                  Number.isFinite(session.activeCandidateIndex)
                    ? Math.trunc(session.activeCandidateIndex)
                    : 0;
                const isSelectedCand = safeActiveCandIdx === safeCandIdx;
                return (
                  <button
                    key={`${idx}-${safeCandIdx}`}
                    type="button"
                    aria-pressed={isSelectedCand}
                    onClick={() => {
                      void interactiveCtrl
                        .selectGeneratedCandidate(safeCandIdx)
                        .then(() => {
                          setCandidateCenterSeq((s) => s + 1);
                          onProjectMutated?.();
                        });
                    }}
                    className={`min-h-[44px] min-w-[44px] rounded-full px-4 font-mono text-xs border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                      isSelectedCand
                        ? 'border-sky-400 bg-sky-950/40 text-sky-200 font-medium'
                        : 'border-[#82828B] bg-studio-900/90 text-zinc-200 hover:border-zinc-300'
                    }`}
                  >
                    候補 {safeCandIdx + 1}
                  </button>
                );
              })}
              {session?.latestReason && typeof session.latestReason === 'object' && (
                <span className="font-mono text-xs text-zinc-400 ml-2">
                  Task:{' '}
                  {typeof session.latestReason.task === 'string' && session.latestReason.task
                    ? session.latestReason.task
                    : 'meta'}
                  {typeof session.latestReason.key === 'string' && session.latestReason.key
                    ? ` | Key: ${session.latestReason.key}`
                    : ''}
                  {typeof session.latestReason.gen_measure_count === 'number' &&
                  Number.isFinite(session.latestReason.gen_measure_count) &&
                  session.latestReason.gen_measure_count > 0
                    ? ` | Bars: ${Math.round(session.latestReason.gen_measure_count)}`
                    : ''}
                  {Array.isArray(session.latestReason.genre)
                    ? (() => {
                        const cleanGenres = session.latestReason.genre.filter(
                          (g): g is string => typeof g === 'string' && g.length > 0,
                        );
                        return cleanGenres.length > 0 ? ` | Genre: ${cleanGenres.join(', ')}` : '';
                      })()
                    : ''}
                </span>
              )}
            </div>
          ) : (
            <span className="text-zinc-200 flex-1 min-w-0 break-words">{uploadNotice}</span>
          )}

          {uploadNotice && (
            <div className="flex items-center gap-2 ml-auto">
              {safeCandidates.length > 0 && (
                <span className="text-zinc-300 break-words">{uploadNotice}</span>
              )}
              <button
                type="button"
                onClick={(e) => {
                  const viaPointer = isPointerClickEvent(e);
                  const hadFocus =
                    typeof document !== 'undefined' && document.activeElement === e.currentTarget;
                  if (!hadFocus) {
                    setUploadNotice(null);
                    return;
                  }
                  if (safeCandidates.length > 0) {
                    const stripEl = secondaryStripRef.current;
                    const candBtn =
                      stripEl?.querySelector<HTMLButtonElement>('button[aria-pressed="true"]') ??
                      stripEl?.querySelector<HTMLButtonElement>('button[aria-pressed]') ??
                      null;
                    setUploadNotice(null);
                    if (candBtn && !candBtn.disabled) {
                      if (viaPointer) {
                        focusButtonWithPointerModality(candBtn);
                      } else {
                        candBtn.focus();
                      }
                      return;
                    }
                  } else {
                    setUploadNotice(null);
                  }
                  const uploadBtn = uploadMidiButtonRef.current;
                  if (uploadBtn && !uploadBtn.disabled) {
                    if (viaPointer) {
                      focusButtonWithPointerModality(uploadBtn);
                    } else {
                      uploadBtn.focus();
                    }
                  }
                }}
                aria-label="通知を閉じる"
                data-tooltip="通知を閉じる"
                data-tooltip-align="end"
                className="inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
              >
                <X className="h-4 w-4" strokeWidth={1.75} />
              </button>
            </div>
          )}
        </div>
      )}

      {/* Main Split Workspace: Track Rack + Interactive High-DPI Piano Roll Canvas */}
      <div
        aria-hidden={isImportModalRendered ? 'true' : undefined}
        {...(isImportModalRendered ? { inert: '' } : {})}
        className={`relative z-[5] flex flex-1 [@media(min-width:768px)_and_(min-height:520px)]:min-h-0 flex-col [@media(min-width:768px)_and_(min-height:520px)]:flex-row [@media(min-width:768px)_and_(min-height:520px)]:overflow-hidden ${
          isImportModalRendered ? 'pointer-events-none select-none' : ''
        }`}
      >
        {showTrackRack && (
          <aside
            ref={trackRackRef}
            aria-label="トラックラック"
            className="w-full [@media(min-width:768px)_and_(min-height:520px)]:w-64 shrink-0 border-b [@media(min-width:768px)_and_(min-height:520px)]:border-b-0 [@media(min-width:768px)_and_(min-height:520px)]:border-r border-zinc-800/85 bg-studio-900/60 flex flex-col max-h-44 [@media(min-width:768px)_and_(min-height:520px)]:max-h-none scroll-pt-16 overflow-x-hidden overflow-y-auto"
          >
            <div
              data-tooltip-sticky-header="true"
              className="sticky top-0 z-20 flex items-center justify-between border-b border-zinc-800/85 bg-studio-900 px-3 py-2"
            >
              <span className="text-xs font-medium text-zinc-300">
                トラック ({safeTracks.length})
              </span>
              <button
                type="button"
                disabled={safeTracks.length >= 64}
                aria-label="トラックを追加"
                aria-description={
                  safeTracks.length >= 64 ? '最大64トラックまで追加できます' : undefined
                }
                data-tooltip={
                  safeTracks.length >= 64 ? '最大64トラックまで追加できます' : 'トラックを追加'
                }
                data-tooltip-when-disabled="true"
                data-tooltip-align="end"
                onClick={(e) => {
                  const shouldHandoffFocus =
                    typeof document !== 'undefined' &&
                    document.activeElement === e.currentTarget &&
                    !isPointerClickEvent(e);
                  const rackEl = trackRackRef.current;
                  let firstVisibleTrackId: string | null = null;
                  if (shouldHandoffFocus && rackEl) {
                    const stickyHdr = rackEl.querySelector('[data-tooltip-sticky-header="true"]');
                    const minVisibleTop =
                      stickyHdr instanceof HTMLElement
                        ? stickyHdr.getBoundingClientRect().bottom - 4
                        : -Infinity;
                    const trackBtns = rackEl.querySelectorAll('button[data-track-select-id]');
                    for (let i = 0; i < trackBtns.length; i += 1) {
                      const tb = trackBtns[i];
                      if (tb instanceof HTMLButtonElement && !tb.disabled) {
                        const r = tb.getBoundingClientRect();
                        if (r.height === 0 || r.top >= minVisibleTop) {
                          firstVisibleTrackId = tb.getAttribute('data-track-select-id');
                          break;
                        }
                      }
                    }
                  }
                  const createdId = interactiveCtrl.addTrack();
                  const nextSnap = interactiveCtrl.getSnapshot();
                  if (shouldHandoffFocus && nextSnap.tracks.length >= 64) {
                    restoreFocusTrackPointerModalityRef.current = false;
                    restoreFocusTrackIdRef.current =
                      firstVisibleTrackId ||
                      nextSnap.tracks[0]?.id ||
                      (typeof createdId === 'string' && createdId) ||
                      nextSnap.activeTrackId ||
                      null;
                  }
                  onProjectMutated?.();
                }}
                className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 disabled:border-[#82828B]/40 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
              >
                <Plus className="h-4 w-4" strokeWidth={1.75} />
              </button>
            </div>

            <div className="divide-y divide-zinc-800/80">
              {(() => {
                const seenRowIds = new Set<string>();
                return safeTracks.map((trk, idx) => {
                  const rowKey = seenRowIds.has(trk.id) ? `${trk.id}__dup_${idx}` : trk.id;
                  seenRowIds.add(trk.id);
                  const isActiveTrack = trk.id === session?.activeTrackId;
                  const safeTrkName =
                    typeof trk.name === 'string' && trk.name ? trk.name : trk.id;
                  const safeTrkColor =
                    typeof trk.colorHex === 'string' && trk.colorHex ? trk.colorHex : '#38BDF8';
                  const safeTrkNotesCount = Array.isArray(trk.notes) ? trk.notes.length : 0;
                  const safeTrkMuted = Boolean(trk.isMuted);
                  const safeTrkSoloed = Boolean(trk.isSoloed);
                  const isRenamingThisTrack = editingTrackId === trk.id;
                  return (
                    <div
                      key={rowKey}
                      className={`p-2.5 transition-colors ${
                        isActiveTrack ? 'bg-sky-950/20' : 'bg-transparent hover:bg-zinc-900/45'
                      }`}
                    >
                    <div className="flex items-center justify-between gap-1">
                      {isRenamingThisTrack ? (
                        <input
                          type="text"
                          autoFocus
                          aria-label={`${safeTrkName} トラック名編集`}
                          value={editingTrackDraft}
                          onChange={(e) => setEditingTrackDraft(e.target.value)}
                          onBlur={() => {
                            trackRenameOpenedViaPointerRef.current = false;
                            if (cancelTrackRenameRef.current) {
                              cancelTrackRenameRef.current = false;
                              setEditingTrackId(null);
                              return;
                            }
                            const trimmed = editingTrackDraft.trim();
                            if (trimmed && trimmed !== safeTrkName) {
                              interactiveCtrl.renameTrack(trk.id, trimmed);
                              onProjectMutated?.();
                            }
                            setEditingTrackId(null);
                          }}
                          onKeyDown={(e) => {
                            if (isImeComposing(e)) return;
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              cancelTrackRenameRef.current = false;
                              restoreFocusTrackPointerModalityRef.current =
                                trackRenameOpenedViaPointerRef.current;
                              trackRenameOpenedViaPointerRef.current = false;
                              restoreFocusTrackIdRef.current = trk.id;
                              const trimmed = editingTrackDraft.trim();
                              if (trimmed && trimmed !== safeTrkName) {
                                interactiveCtrl.renameTrack(trk.id, trimmed);
                                onProjectMutated?.();
                              }
                              setEditingTrackId(null);
                            } else if (e.key === 'Escape') {
                              e.preventDefault();
                              e.stopPropagation();
                              cancelTrackRenameRef.current = true;
                              restoreFocusTrackPointerModalityRef.current =
                                trackRenameOpenedViaPointerRef.current;
                              trackRenameOpenedViaPointerRef.current = false;
                              restoreFocusTrackIdRef.current = trk.id;
                              setEditingTrackId(null);
                            }
                          }}
                          className="min-h-[44px] flex-1 min-w-0 rounded-xl border border-sky-400 bg-zinc-950 px-2.5 text-xs text-zinc-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                        />
                      ) : (
                        <button
                          type="button"
                          data-track-select-id={trk.id}
                          aria-pressed={isActiveTrack}
                          aria-keyshortcuts="F2"
                          data-tooltip="選択 / ダブルクリック・F2で名前変更"
                          data-tooltip-align="start"
                          onPointerDown={(e) => {
                            lastTrackBtnPointerTypeRef.current = e.pointerType || 'mouse';
                          }}
                          onPointerCancel={() => {
                            lastTrackBtnPointerTypeRef.current = 'mouse';
                          }}
                          onKeyDown={(e) => {
                            if (isImeComposing(e)) return;
                            if (e.key === 'F2') {
                              e.preventDefault();
                              e.stopPropagation();
                              cancelTrackRenameRef.current = false;
                              trackRenameOpenedViaPointerRef.current = false;
                              setEditingTrackId(trk.id);
                              setEditingTrackDraft(safeTrkName);
                            } else if (e.key === 'Enter' || e.key === ' ') {
                              lastTrackBtnPointerTypeRef.current = 'keyboard';
                            }
                          }}
                          onClick={() => {
                            const ptrType = lastTrackBtnPointerTypeRef.current;
                            lastTrackBtnPointerTypeRef.current = 'mouse';
                            if (ptrType === 'touch' && isActiveTrack) {
                              cancelTrackRenameRef.current = false;
                              trackRenameOpenedViaPointerRef.current = true;
                              setEditingTrackId(trk.id);
                              setEditingTrackDraft(safeTrkName);
                              return;
                            }
                            interactiveCtrl.setActiveTrack(trk.id);
                            onProjectMutated?.();
                          }}
                          onDoubleClick={() => {
                            cancelTrackRenameRef.current = false;
                            trackRenameOpenedViaPointerRef.current = true;
                            setEditingTrackId(trk.id);
                            setEditingTrackDraft(safeTrkName);
                          }}
                          className={`flex min-h-[44px] flex-1 items-center gap-2 rounded-xl border px-2 text-left min-w-0 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                            isActiveTrack
                              ? 'border-sky-400 bg-sky-950/45 text-zinc-100'
                              : 'border-[#82828B] bg-zinc-950/75 text-zinc-200 hover:border-zinc-300 hover:bg-zinc-900/75'
                          }`}
                        >
                          <span
                            className="h-3.5 w-1.5 shrink-0 rounded-full"
                            style={{ backgroundColor: safeTrkColor }}
                          />
                          <div className="min-w-0">
                            <div className="text-xs font-medium text-zinc-100 truncate">
                              {safeTrkName}
                            </div>
                            <div
                              className={`font-mono text-xs truncate ${
                                isActiveTrack ? 'text-sky-200' : 'text-zinc-400'
                              }`}
                            >
                              {safeTrkNotesCount} notes{isActiveTrack ? ' · 対象' : ''}
                            </div>
                          </div>
                        </button>
                      )}

                      <button
                        type="button"
                        aria-pressed={safeTrkMuted}
                        onClick={() => {
                          interactiveCtrl.toggleTrackMute(trk.id);
                          onProjectMutated?.();
                        }}
                        aria-label={`${safeTrkName} ミュート切替`}
                        data-tooltip="ミュート"
                        className={`min-h-[44px] min-w-[44px] rounded-full font-mono text-xs border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                          safeTrkMuted
                            ? 'border-zinc-300 bg-zinc-800 text-zinc-100 font-semibold'
                            : 'border-[#82828B] bg-zinc-950/90 text-zinc-300 hover:border-zinc-300 hover:text-zinc-100'
                        }`}
                      >
                        M
                      </button>

                      <button
                        type="button"
                        aria-pressed={safeTrkSoloed}
                        onClick={() => {
                          interactiveCtrl.toggleTrackSolo(trk.id);
                          onProjectMutated?.();
                        }}
                        aria-label={`${safeTrkName} ソロ切替`}
                        data-tooltip="ソロ"
                        data-tooltip-align="end"
                        className={`min-h-[44px] min-w-[44px] rounded-full font-mono text-xs border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                          safeTrkSoloed
                            ? 'border-sky-400 bg-sky-950/50 text-sky-200 font-semibold'
                            : 'border-[#82828B] bg-zinc-950/90 text-zinc-300 hover:border-zinc-300 hover:text-zinc-100'
                        }`}
                      >
                        S
                      </button>
                    </div>

                    <div className="mt-2 flex items-center gap-2">
                      <div className="relative flex-1 min-w-0">
                        <select
                          aria-label={`${safeTrkName} 楽器選択`}
                          value={
                            INSTRUMENT_OPTIONS.some((o) => o.value === trk.instrument)
                              ? trk.instrument
                              : 'PIANO'
                          }
                          onChange={(e) => {
                            interactiveCtrl.setTrackInstrument(
                              trk.id,
                              e.target.value as TrackInstrument,
                            );
                            onProjectMutated?.();
                          }}
                          className="min-h-[44px] w-full appearance-none truncate rounded-xl border border-[#82828B] bg-zinc-950/90 pl-3 pr-8 font-mono text-xs text-zinc-200 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                        >
                          {INSTRUMENT_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                        <ChevronDown
                          className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                          strokeWidth={1.75}
                        />
                      </div>

                      {safeTracks.length > 1 && (
                        <button
                          type="button"
                          onClick={(e) => {
                            const shouldHandoffFocus =
                              typeof document !== 'undefined' &&
                              document.activeElement === e.currentTarget &&
                              !isPointerClickEvent(e);
                            interactiveCtrl.deleteTrack(trk.id);
                            const nextSnap = interactiveCtrl.getSnapshot();
                            const adjacentTrack =
                              nextSnap.tracks[
                                Math.min(idx, Math.max(0, nextSnap.tracks.length - 1))
                              ] ?? null;
                            if (
                              editingTrackId === trk.id ||
                              (shouldHandoffFocus &&
                                editingTrackId !== null &&
                                editingTrackId === adjacentTrack?.id)
                            ) {
                              trackRenameOpenedViaPointerRef.current = false;
                              setEditingTrackId(null);
                            }
                            if (shouldHandoffFocus) {
                              restoreFocusTrackPointerModalityRef.current = false;
                              restoreFocusTrackIdRef.current =
                                adjacentTrack?.id ||
                                nextSnap.activeTrackId ||
                                nextSnap.tracks[0]?.id ||
                                null;
                            }
                            onProjectMutated?.();
                          }}
                          aria-label={`${safeTrkName} を削除`}
                          data-tooltip="トラックを削除"
                          data-tooltip-align="end"
                          data-tooltip-pos="top"
                          data-tooltip-prefer-pos="top"
                          className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-300 hover:border-red-400 hover:text-red-300 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400/70"
                        >
                          <Trash2 className="h-4 w-4" strokeWidth={1.75} />
                        </button>
                      )}
                    </div>
                  </div>
                  );
                });
              })()}
            </div>
          </aside>
        )}

        {/* Right Canvas Container (Transparent so Bottom Crimson-Purple Aurora Glows Through) */}
        <div
          ref={canvasContainerRef}
          className="relative flex-1 min-w-0 min-h-[176px] [@media(max-height:519px)]:max-h-[200px] [@media(max-width:767px)_and_(min-height:520px)]:min-h-[240px] [@media(min-width:768px)_and_(min-height:520px)]:min-h-0 overflow-hidden bg-transparent select-none"
        >
          <canvas
            ref={canvasRef}
            onPointerDown={handleCanvasPointerDown}
            onDoubleClick={handleCanvasDoubleClick}
            onContextMenu={handleCanvasContextMenu}
            data-testid="pianoroll-canvas"
            className={`absolute inset-0 block w-full h-full touch-none ${canvasCursor}`}
            aria-label="ピアノロールエディタキャンバス"
          />
        </div>
      </div>

      {/* Bottom COMPASS Telemetry & Direct-Manipulation Guide Bar */}
      <div
        data-testid="pianoroll-telemetry-bar"
        aria-hidden={isImportModalRendered ? 'true' : undefined}
        {...(isImportModalRendered ? { inert: '' } : {})}
        className={`flex shrink-0 h-7 sm:h-8 [@media(max-height:500px)]:h-6 items-center justify-between gap-4 border-t border-zinc-800/85 bg-studio-900/55 px-3 sm:px-3.5 font-mono text-xs text-zinc-400 ${
          isImportModalRendered ? 'pointer-events-none select-none' : ''
        }`}
      >
        <span className="truncate text-zinc-200">{selectedNotesTelemetry}</span>
        <span className="hidden lg:inline truncate text-zinc-400">
          ダブルクリック: 作成/削除 · 右クリック: 削除 · Alt+ドラッグ: パン · Ctrl+ホイール: ズーム
        </span>
      </div>

      {/* Multi-Track Import Modal */}
      {isImportModalRendered && pendingImport && (
        <MidiTrackImportModal
          fileName={pendingImport.fileName}
          parsedMidi={pendingImport.parsed}
          isSamplePreset={pendingImport.isSamplePreset}
          restoreFocusTo={modalTriggerFocusRef.current}
          fallbackFocusTo={uploadMidiButtonRef.current ?? launchCompassAiButtonRef.current}
          allowFocusRestoreRef={isActiveAndInteractiveRef}
          onRestoreFocus={(el, meta) => {
            const openedViaPointer = modalOpenedViaPointerRef.current;
            modalOpenedViaPointerRef.current = false;
            const isPointerRestore =
              meta.closeSource === 'pointer-button' ||
              (meta.closeSource === 'escape' &&
                openedViaPointer &&
                !meta.keyboardUsedInsideModal);
            if (isPointerRestore && el instanceof HTMLButtonElement) {
              focusButtonWithPointerModality(el);
            }
          }}
          onConfirm={handleConfirmModalImport}
          onCancel={closeImportModal}
        />
      )}
    </div>
  );
};

function VEL_INNER_MAX(totalVelHeight: number): number {
  return Math.max(16, totalVelHeight - 18);
}
