/**
 * Open/Closed Registry for Pluggable DAW Controllers (`oop-extensible-craft`).
 * New DAW integrations (Cubase, Ableton Live, Logic Pro, Reaper, etc.) register here
 * without modifying any View or Model Engine client code.
 */

import { isDomainProxy } from '../domain/MidiNote';
import { IDawController } from './IDawController';

export class DawControllerRegistry {
  private readonly _controllers = new Map<string, IDawController>();
  private _activeId: string | null = null;
  private _pendingId: string | null = null;
  private _connectedId: string | null = null;
  private _notifiedId: string | null = null;
  private _activationSeq = 0;
  private _transitionChain: Promise<void> = Promise.resolve();
  private readonly _listeners = new Set<(active: IDawController) => void>();

  public register(controller: IDawController): void {
    this._controllers.set(controller.id, controller);
    if (this._activeId === null) {
      this._activeId = controller.id;
      this._pendingId = controller.id;
      this._notifiedId = controller.id;
    }
  }

  public listAll(): readonly IDawController[] {
    return Array.from(this._controllers.values());
  }

  public get(id: string): IDawController {
    const found = this._controllers.get(id);
    if (!found) {
      throw new Error(`未登録のDAWコントローラーです: ${id}`);
    }
    return found;
  }

  public getActive(): IDawController {
    if (!this._activeId) {
      throw new Error('有効なDAWコントローラーが登録されていません。');
    }
    return this.get(this._activeId);
  }

  public getPendingId(): string | null {
    return this._pendingId ?? this._activeId;
  }

  public async activate(id: string): Promise<IDawController> {
    const next = this.get(id);
    const seq = ++this._activationSeq;
    const prevCommittedId = this._activeId;
    const prevPendingId = this._pendingId ?? this._activeId;
    this._pendingId = id;
    for (const prevId of [prevPendingId, prevCommittedId]) {
      if (prevId && prevId !== id) {
        try {
          this._controllers.get(prevId)?.pause();
        } catch {
          // Isolate synchronous pause errors during optimistic stop
        }
      }
    }

    const notifyListeners = (ctrl: IDawController): void => {
      this._notifiedId = ctrl.id;
      for (const listener of Array.from(this._listeners)) {
        try {
          listener(ctrl);
        } catch {
          // Isolate faulty subscriber so remaining listeners still receive active controller changes
        }
      }
    };

    const runStep = async (): Promise<void> => {
      if (seq !== this._activationSeq) {
        return;
      }
      const rollbackId = this._notifiedId ?? this._connectedId ?? prevCommittedId;
      try {
        if (this._connectedId && this._connectedId !== id) {
          const prevConnected = this._controllers.get(this._connectedId);
          this._connectedId = null;
          if (prevConnected) {
            try {
              prevConnected.pause();
            } catch {
              // Isolate pause failure before disconnect
            }
            try {
              await prevConnected.disconnect();
            } catch {
              // Isolate disconnect failure so switching away from a faulted controller never bricks recovery
            }
          }
        }
        if (seq !== this._activationSeq) {
          return;
        }
        await next.connect();
        this._connectedId = id;
        if (seq !== this._activationSeq) {
          return;
        }
        this._activeId = id;
        this._pendingId = id;
        notifyListeners(next);
      } catch (err) {
        if (seq === this._activationSeq) {
          const fallbackId =
            rollbackId && this._controllers.has(rollbackId)
              ? rollbackId
              : (this._controllers.keys().next().value ?? null);
          this._activeId = fallbackId;
          this._pendingId = fallbackId;
          const rollbackCtrl = fallbackId ? (this._controllers.get(fallbackId) ?? null) : null;
          if (rollbackCtrl && this._connectedId !== rollbackCtrl.id) {
            try {
              await rollbackCtrl.connect();
              this._connectedId = rollbackCtrl.id;
            } catch {
              // Best-effort rollback reconnect
            }
          }
          if (seq === this._activationSeq && rollbackCtrl && this._notifiedId !== rollbackCtrl.id) {
            notifyListeners(rollbackCtrl);
          }
        }
        throw err;
      }
    };

    const stepPromise = this._transitionChain.then(runStep, runStep);
    this._transitionChain = stepPromise.catch(() => {});
    await stepPromise;
    let seenSeq = seq;
    while (seenSeq !== this._activationSeq) {
      seenSeq = this._activationSeq;
      await this._transitionChain;
    }
    return this.getActive();
  }

  public subscribeActiveChange(listener: (active: IDawController) => void): () => void {
    if (typeof listener !== 'function' || isDomainProxy(listener)) {
      return () => {};
    }
    this._listeners.add(listener);
    return () => {
      this._listeners.delete(listener);
    };
  }
}
