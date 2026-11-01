/**
 * Concrete Reversible Note Editing Commands (matches Java COMPASS `ux/pianoroll/commands/*`).
 */

import {
  cloneCanonicalIdSetMutable,
  DawTrack,
  EMPTY_CANONICAL_ID_SET,
  isPreparedCandidateBatch,
  normalizeOrReuseCanonicalIdSet,
  normalizeTrackIdIterable,
  normalizeTrackNoteArray,
  PreparedCandidateBatch,
} from '../DawTrack';
import { isDomainProxy, MidiNote, truncateWellFormedUtf16 } from '../MidiNote';
import { IDawCommand } from './CommandManager';

function sanitizeCommandDescription(raw: unknown, fallback: string): string {
  if (typeof raw !== 'string') return fallback;
  const clean = truncateWellFormedUtf16(raw.trim(), 128).trim();
  return clean || fallback;
}

export class AddNotesCommand implements IDawCommand {
  public readonly description: string;
  private readonly _track: DawTrack;
  private readonly _notes: readonly MidiNote[];
  private _actuallyInsertedIds: ReadonlySet<string> = new Set();

  constructor(track: DawTrack, notes: readonly MidiNote[], description = 'ノート追加') {
    this._track = track;
    this._notes = normalizeTrackNoteArray(notes);
    this.description = sanitizeCommandDescription(description, 'ノート追加');
  }

  public get targetTrackId(): string {
    return this._track.id;
  }

  public get actuallyInsertedIds(): ReadonlySet<string> {
    return this._actuallyInsertedIds;
  }

  public execute(): void {
    this._actuallyInsertedIds = new Set();
    if (this._notes.length === 0) return;
    const inserted = this._track.addNotes(this._notes);
    if (inserted.length > 0) {
      this._actuallyInsertedIds = new Set(inserted.map((n) => n.id));
    }
  }

  public undo(): void {
    if (this._actuallyInsertedIds.size === 0) return;
    this._track.removeNotesByIds(this._actuallyInsertedIds);
    this._actuallyInsertedIds = new Set();
  }
}

export class DeleteNotesCommand implements IDawCommand {
  public readonly description = 'ノート削除';
  private readonly _track: DawTrack;
  private readonly _ids: ReadonlySet<string>;
  private _removedNotes: MidiNote[] = [];

  constructor(track: DawTrack, noteIds: Iterable<string>) {
    this._track = track;
    this._ids = normalizeTrackIdIterable(noteIds);
  }

  public get targetTrackId(): string {
    return this._track.id;
  }

  public execute(): void {
    this._removedNotes = this._track.removeNotesByIds(this._ids);
  }

  public undo(): void {
    if (this._removedNotes.length > 0) {
      this._track.addNotes(this._removedNotes);
    }
  }
}

export class TransformNotesCommand implements IDawCommand {
  public readonly description: string;
  private readonly _track: DawTrack;
  private readonly _afterById: ReadonlyMap<string, MidiNote>;
  private readonly _onExecuteHook?: () => void;
  private readonly _onUndoHook?: () => void;
  private _effectiveBeforeById: ReadonlyMap<string, MidiNote> = new Map();

  constructor(
    track: DawTrack,
    beforeNotes: readonly MidiNote[],
    afterNotes: readonly MidiNote[],
    description = 'ノート編集',
    hooks?: {
      onExecute?: () => void;
      onUndo?: () => void;
    },
  ) {
    this._track = track;
    normalizeTrackNoteArray(beforeNotes);
    const normAfter = normalizeTrackNoteArray(afterNotes);
    this._afterById = new Map(normAfter.map((n) => [n.id, n]));
    this.description = sanitizeCommandDescription(description, 'ノート編集');
    this._onExecuteHook = typeof hooks?.onExecute === 'function' ? hooks.onExecute : undefined;
    this._onUndoHook = typeof hooks?.onUndo === 'function' ? hooks.onUndo : undefined;
  }

  public get targetTrackId(): string {
    return this._track.id;
  }

  public execute(): void {
    this._effectiveBeforeById = new Map();
    if (this._afterById.size === 0) {
      return;
    }
    const capturedBefore = new Map<string, MidiNote>();
    const currentNotes = this._track.toSnapshot().notes;
    for (let i = 0; i < currentNotes.length; i += 1) {
      const cn = currentNotes[i];
      if (this._afterById.has(cn.id)) {
        capturedBefore.set(cn.id, cn);
      }
    }
    if (capturedBefore.size === 0) {
      return;
    }
    this._effectiveBeforeById = capturedBefore;
    this._track.updateNotes(this._afterById);
    this._onExecuteHook?.();
  }

  public undo(): void {
    if (this._effectiveBeforeById.size === 0) return;
    this._track.updateNotes(this._effectiveBeforeById);
    this._effectiveBeforeById = new Map();
    this._onUndoHook?.();
  }
}

export class ReplaceRangeNotesCommand implements IDawCommand {
  public readonly description: string;
  private readonly _track: DawTrack;
  private readonly _rangeStartTick: number;
  private readonly _rangeEndTick: number;
  private readonly _insertedNotes: readonly MidiNote[] | PreparedCandidateBatch;
  private readonly _extraIdsToRemove: ReadonlySet<string>;
  private readonly _onExecuteHook?: (actuallyInsertedIds: ReadonlySet<string>) => void;
  private readonly _onUndoHook?: () => void;

  private _actuallyInsertedIds: ReadonlySet<string> = EMPTY_CANONICAL_ID_SET;
  private _removedNotes: MidiNote[] = [];
  private _truncatedBeforeById = new Map<string, MidiNote>();

  constructor(
    track: DawTrack,
    rangeStartTick: number,
    rangeEndTick: number,
    insertedNotes: readonly MidiNote[] | PreparedCandidateBatch,
    description = 'AI生成フレーズ配置',
    extraIdsToRemove?: ReadonlySet<string>,
    hooks?: {
      onExecute?: (actuallyInsertedIds: ReadonlySet<string>) => void;
      onUndo?: () => void;
    },
  ) {
    this._track = track;
    const safeStart =
      typeof rangeStartTick === 'number' && Number.isFinite(rangeStartTick)
        ? Math.max(0, Math.round(rangeStartTick))
        : 0;
    const safeEnd =
      typeof rangeEndTick === 'number' && Number.isFinite(rangeEndTick)
        ? Math.max(safeStart, Math.round(rangeEndTick))
        : safeStart;
    this._rangeStartTick = safeStart;
    this._rangeEndTick = safeEnd;
    this._insertedNotes = isPreparedCandidateBatch(insertedNotes)
      ? insertedNotes
      : normalizeTrackNoteArray(insertedNotes);
    this._extraIdsToRemove = normalizeOrReuseCanonicalIdSet(extraIdsToRemove);
    let execHook: ((actuallyInsertedIds: ReadonlySet<string>) => void) | undefined;
    let undoHook: (() => void) | undefined;
    if (hooks && typeof hooks === 'object' && !isDomainProxy(hooks)) {
      try {
        const ed = Object.getOwnPropertyDescriptor(hooks, 'onExecute');
        if (ed && 'value' in ed && typeof ed.value === 'function') {
          execHook = ed.value as (actuallyInsertedIds: ReadonlySet<string>) => void;
        }
        const ud = Object.getOwnPropertyDescriptor(hooks, 'onUndo');
        if (ud && 'value' in ud && typeof ud.value === 'function') {
          undoHook = ud.value as () => void;
        }
      } catch {
        // Ignore hostile/revoked hooks object
      }
    }
    this._onExecuteHook = execHook;
    this._onUndoHook = undoHook;
    this.description = sanitizeCommandDescription(description, 'AI生成フレーズ配置');
  }

  public get targetTrackId(): string {
    return this._track.id;
  }

  public get actuallyInsertedIds(): ReadonlySet<string> {
    return this._actuallyInsertedIds;
  }

  public cloneMutableInsertedIds(): Set<string> {
    return cloneCanonicalIdSetMutable(this._actuallyInsertedIds);
  }

  public execute(): void {
    const res = this._track.replaceRangeNotesAtomic(
      this._rangeStartTick,
      this._rangeEndTick,
      this._insertedNotes,
      this._extraIdsToRemove,
    );
    this._removedNotes = res.removedNotes;
    this._truncatedBeforeById = res.truncatedBeforeById;
    this._actuallyInsertedIds = res.actuallyInsertedIds;
    try {
      this._onExecuteHook?.(this._actuallyInsertedIds);
    } catch {
      // Never abort command execution after state commit if optional hook throws
    }
  }

  public undo(): void {
    if (this._actuallyInsertedIds.size > 0) {
      this._track.removeNotesByIds(this._actuallyInsertedIds);
      this._actuallyInsertedIds = EMPTY_CANONICAL_ID_SET;
    }
    if (this._truncatedBeforeById.size > 0) {
      this._track.updateNotes(this._truncatedBeforeById);
    }
    if (this._removedNotes.length > 0) {
      this._track.addNotes(this._removedNotes);
    }
    try {
      this._onUndoHook?.();
    } catch {
      // Never abort command undo after state rollback if optional hook throws
    }
  }
}
