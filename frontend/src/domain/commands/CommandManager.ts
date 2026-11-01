/**
 * Command Pattern Interface & Bounded Undo/Redo Stack (`oop-extensible-craft`).
 */

export interface IDawCommand {
  readonly description: string;
  readonly targetTrackId?: string;
  execute(): void;
  undo(): void;
}

export class CommandManager {
  private readonly _maxHistory: number;
  private _undoStack: IDawCommand[] = [];
  private _redoStack: IDawCommand[] = [];

  constructor(maxHistory = 100) {
    this._maxHistory = Math.max(10, maxHistory);
  }

  public execute(command: IDawCommand): void {
    command.execute();
    this._undoStack.push(command);
    if (this._undoStack.length > this._maxHistory) {
      this._undoStack.shift();
    }
    this._redoStack = [];
  }

  public canUndo(): boolean {
    return this._undoStack.length > 0;
  }

  public canRedo(): boolean {
    return this._redoStack.length > 0;
  }

  public peekUndoDescription(): string | null {
    return this._undoStack.length > 0
      ? this._undoStack[this._undoStack.length - 1].description
      : null;
  }

  public peekRedoDescription(): string | null {
    return this._redoStack.length > 0
      ? this._redoStack[this._redoStack.length - 1].description
      : null;
  }

  public undo(): boolean {
    const cmd = this._undoStack.pop();
    if (!cmd) return false;
    cmd.undo();
    this._redoStack.push(cmd);
    return true;
  }

  public redo(): boolean {
    const cmd = this._redoStack.pop();
    if (!cmd) return false;
    cmd.execute();
    this._undoStack.push(cmd);
    return true;
  }

  public clear(): void {
    this._undoStack = [];
    this._redoStack = [];
  }

  public pruneTrack(trackId: string): void {
    if (!trackId) return;
    this._undoStack = this._undoStack.filter((cmd) => cmd.targetTrackId !== trackId);
    this._redoStack = this._redoStack.filter((cmd) => cmd.targetTrackId !== trackId);
  }
}
