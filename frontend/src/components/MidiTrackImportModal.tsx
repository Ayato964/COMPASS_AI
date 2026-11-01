/**
 * Multi-Track MIDI Import Modal (`MidiTrackImportModal.tsx`).
 * Modeled after Java COMPASS `MidiHandler` track selection workflow, redesigned with
 * strict `anti-ai-web-design` rules: monochrome studio palette, single solid primary CTA,
 * zero emojis, >= 44x44px targets.
 */

import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Check, X } from 'lucide-react';
import {
  extractCanonicalParsedMidiTracks,
  MAX_PARSED_SMF_TRACKS,
  ParsedMidiFile,
} from '../domain/midi/StandardMidiCodec';
import { useAdaptiveTooltipAlignment } from '../utils/useAdaptiveTooltipAlignment';

export const MAX_MODAL_IMPORT_SELECTION = 64;

export type MidiModalCloseSource = 'pointer-button' | 'keyboard-button' | 'escape' | 'external';

interface MidiTrackImportModalProps {
  readonly fileName: string;
  readonly parsedMidi: ParsedMidiFile;
  readonly isSamplePreset?: boolean;
  readonly restoreFocusTo?: HTMLElement | null;
  readonly fallbackFocusTo?: HTMLElement | null;
  readonly allowFocusRestoreRef?: React.RefObject<boolean>;
  readonly onRestoreFocus?: (
    restoredEl: HTMLElement,
    meta: { closeSource: MidiModalCloseSource; keyboardUsedInsideModal: boolean },
  ) => void;
  readonly onConfirm: (selectedIndices: readonly number[], replaceExisting: boolean) => void;
  readonly onCancel: () => void;
}

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])';

export const MidiTrackImportModal: React.FC<MidiTrackImportModalProps> = ({
  fileName,
  parsedMidi,
  isSamplePreset = false,
  restoreFocusTo = null,
  fallbackFocusTo = null,
  allowFocusRestoreRef,
  onRestoreFocus,
  onConfirm,
  onCancel,
}) => {
  useAdaptiveTooltipAlignment();
  const safeTracks = extractCanonicalParsedMidiTracks(parsedMidi, MAX_PARSED_SMF_TRACKS);
  const safeFileName = typeof fileName === 'string' && fileName ? fileName : 'imported.mid';
  const safeBpm =
    typeof parsedMidi?.bpm === 'number' && Number.isFinite(parsedMidi.bpm)
      ? Math.max(20, Math.min(300, Math.round(parsedMidi.bpm * 10) / 10))
      : 120;
  const safePpqn =
    typeof parsedMidi?.ppqn === 'number' &&
    Number.isFinite(parsedMidi.ppqn) &&
    parsedMidi.ppqn >= 1
      ? Math.min(32767, Math.round(parsedMidi.ppqn))
      : 480;

  const dialogRef = useRef<HTMLDivElement | null>(null);
  const confirmButtonRef = useRef<HTMLButtonElement | null>(null);
  const [selected, setSelected] = useState<Set<number>>(() => {
    const initial = new Set<number>();
    for (
      let idx = 0;
      idx < safeTracks.length && initial.size < MAX_MODAL_IMPORT_SELECTION;
      idx += 1
    ) {
      initial.add(safeTracks[idx].canonicalIndex);
    }
    return initial;
  });
  const [replaceExisting, setReplaceExisting] = useState<boolean>(true);

  const restoreFocusPropRef = useRef<HTMLElement | null>(restoreFocusTo);
  restoreFocusPropRef.current = restoreFocusTo;
  const fallbackFocusPropRef = useRef<HTMLElement | null>(fallbackFocusTo);
  fallbackFocusPropRef.current = fallbackFocusTo;
  const allowFocusRestorePropRef = useRef<React.RefObject<boolean> | undefined>(
    allowFocusRestoreRef,
  );
  allowFocusRestorePropRef.current = allowFocusRestoreRef;
  const onRestoreFocusPropRef = useRef<
    | ((
        el: HTMLElement,
        meta: { closeSource: MidiModalCloseSource; keyboardUsedInsideModal: boolean },
      ) => void)
    | undefined
  >(onRestoreFocus);
  onRestoreFocusPropRef.current = onRestoreFocus;
  const capturedRestoreTargetRef = useRef<HTMLElement | null>(null);
  const lastDialogPointerBtnRef = useRef<HTMLButtonElement | null>(null);
  const closeSourceRef = useRef<MidiModalCloseSource>('external');
  const keyboardUsedInsideModalRef = useRef<boolean>(false);

  const recordButtonCloseModality = (e: React.MouseEvent<HTMLButtonElement>): void => {
    const viaPointer =
      e.detail > 0 || lastDialogPointerBtnRef.current === e.currentTarget;
    closeSourceRef.current = viaPointer ? 'pointer-button' : 'keyboard-button';
    lastDialogPointerBtnRef.current = null;
  };

  // Move focus inside dialog synchronously on commit (`useLayoutEffect`) so background `aria-hidden`
  // containers never retain a focused descendant (`[R2-MED-02]`), while restoring focus in passive
  // `useEffect` cleanup AFTER sibling `removeAttribute('aria-hidden')` updates commit (`[R3-MED-01]`).
  useLayoutEffect(() => {
    const dialogEl = dialogRef.current;
    if (
      !capturedRestoreTargetRef.current &&
      document.activeElement instanceof HTMLElement &&
      document.activeElement !== document.body &&
      !dialogEl?.contains(document.activeElement)
    ) {
      capturedRestoreTargetRef.current = document.activeElement;
    }
    const initialTarget =
      confirmButtonRef.current && !confirmButtonRef.current.disabled
        ? confirmButtonRef.current
        : dialogEl?.querySelector<HTMLElement>(FOCUSABLE_SELECTOR) ?? dialogEl;
    initialTarget?.focus();
  }, []);

  useEffect(() => {
    return () => {
      if (dialogRef.current && dialogRef.current.isConnected) {
        return; // Skip React.StrictMode simulated DEV unmount while modal is still open
      }
      if (allowFocusRestorePropRef.current && allowFocusRestorePropRef.current.current === false) {
        return;
      }
      const isFocusableCandidate = (el: HTMLElement | null | undefined): el is HTMLElement =>
        Boolean(
          el &&
            document.contains(el) &&
            !el.matches(':disabled') &&
            !el.closest('[aria-hidden="true"], [hidden], [inert]'),
        );
      const candidates: Array<HTMLElement | null | undefined> = [
        restoreFocusPropRef.current,
        capturedRestoreTargetRef.current,
        fallbackFocusPropRef.current,
        document.querySelector<HTMLElement>(
          '[data-testid="upload-midi-button"]:not([disabled]), [data-testid="launch-compass-ai-button"]:not([disabled])',
        ),
      ];
      const target = candidates.find(isFocusableCandidate) ?? null;
      if (target) {
        onRestoreFocusPropRef.current?.(target, {
          closeSource: closeSourceRef.current,
          keyboardUsedInsideModal: keyboardUsedInsideModalRef.current,
        });
        if (document.activeElement !== target) {
          target.focus();
        }
      }
    };
  }, []);

  useEffect(() => {
    let pointerFallbackTimer: number | null = null;
    const clearPointerTimer = () => {
      if (pointerFallbackTimer !== null) {
        window.clearTimeout(pointerFallbackTimer);
        pointerFallbackTimer = null;
      }
    };
    const clearDialogPointerBtn = () => {
      clearPointerTimer();
      lastDialogPointerBtnRef.current = null;
    };
    const onPointerUpCapture = (ev: PointerEvent) => {
      const pressed = lastDialogPointerBtnRef.current;
      if (!pressed) return;
      const targetEl = ev.target instanceof Element ? ev.target : null;
      if (!targetEl || !pressed.contains(targetEl) || pressed.disabled || !pressed.isConnected) {
        clearDialogPointerBtn();
        return;
      }
      clearPointerTimer();
      pointerFallbackTimer = window.setTimeout(() => {
        pointerFallbackTimer = null;
        lastDialogPointerBtnRef.current = null;
      }, 120);
    };
    const onCaptureKeyDown = (ev: KeyboardEvent) => {
      clearDialogPointerBtn();
      if (ev.key !== 'Escape' && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        keyboardUsedInsideModalRef.current = true;
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeSourceRef.current = 'escape';
        onCancel();
        return;
      }
      if (e.key === 'Tab') {
        keyboardUsedInsideModalRef.current = true;
        const dialogEl = dialogRef.current;
        if (!dialogEl) return;
        const focusables = Array.from(
          dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
        );
        if (focusables.length === 0) {
          e.preventDefault();
          dialogEl.focus();
          return;
        }
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey) {
          if (!active || active === first || !dialogEl.contains(active)) {
            e.preventDefault();
            last.focus();
          }
        } else if (!active || active === last || !dialogEl.contains(active)) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener('pointerup', onPointerUpCapture, true);
    window.addEventListener('pointercancel', clearDialogPointerBtn, true);
    window.addEventListener('click', clearDialogPointerBtn);
    window.addEventListener('keydown', onCaptureKeyDown, true);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      clearPointerTimer();
      window.removeEventListener('pointerup', onPointerUpCapture, true);
      window.removeEventListener('pointercancel', clearDialogPointerBtn, true);
      window.removeEventListener('click', clearDialogPointerBtn);
      window.removeEventListener('keydown', onCaptureKeyDown, true);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [onCancel]);

  const toggleIndex = (index: number) => {
    setSelected((prev) => {
      if (prev.has(index)) {
        const next = new Set(prev);
        next.delete(index);
        return next;
      }
      if (prev.size >= MAX_MODAL_IMPORT_SELECTION) {
        return prev;
      }
      const next = new Set(prev);
      next.add(index);
      return next;
    });
  };

  return (
    <div
      ref={dialogRef}
      tabIndex={-1}
      className="fixed inset-0 z-50 flex justify-center overflow-y-auto bg-black/80 px-2 sm:px-4 py-4 sm:py-6 [@media(max-height:560px)]:py-2 [@media(max-height:500px)]:py-2 focus:outline-none"
      role="dialog"
      aria-modal="true"
      aria-labelledby="midi-import-modal-title"
      data-testid="midi-import-modal"
    >
      <div
        data-testid="midi-import-modal-card"
        className="my-auto flex max-h-[calc(100dvh-2rem)] sm:max-h-[calc(100dvh-3rem)] [@media(max-height:560px)]:max-h-[calc(100dvh-1rem)] [@media(max-height:500px)]:max-h-[calc(100dvh-1rem)] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-studio-900 text-zinc-100 shadow-sm"
      >
        <div className="relative z-20 flex shrink-0 items-center justify-between gap-2 sm:gap-3 border-b border-zinc-800 px-3 sm:px-5 py-3.5 [@media(max-height:560px)]:py-1.5 [@media(max-height:500px)]:py-1.5">
          <div className="min-w-0 flex-1">
            <h2
              id="midi-import-modal-title"
              className="text-base font-semibold tracking-tight text-zinc-100 truncate"
            >
              {isSamplePreset ? 'サンプルアレンジの読み込み設定' : 'MIDIトラックの読み込み設定'}
            </h2>
            <p className="mt-0.5 text-xs text-zinc-400 font-mono truncate w-full">
              {isSamplePreset
                ? `内蔵サンプル (4小節 Piano + Sax) — ${safeBpm} BPM / PPQN ${safePpqn}`
                : `${safeFileName} — ${safeBpm} BPM / PPQN ${safePpqn}`}
            </p>
          </div>
          <button
            type="button"
            onPointerDown={(e) => {
              if (e.button === 0) lastDialogPointerBtnRef.current = e.currentTarget;
            }}
            onClick={(e) => {
              recordButtonCloseModality(e);
              onCancel();
            }}
            className="inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-full border border-[#82828B] bg-zinc-950/90 text-zinc-300 hover:border-zinc-300 hover:bg-zinc-800/80 hover:text-zinc-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            aria-label="閉じる"
            data-tooltip="閉じる (Esc)"
            data-tooltip-align="end"
          >
            <X className="h-4 w-4" strokeWidth={1.75} />
          </button>
        </div>

        <div
          data-testid="midi-import-scroll-body"
          className="min-h-0 flex-1 overflow-y-auto px-3 sm:px-5 py-4 space-y-4 [@media(max-height:560px)]:py-1.5 [@media(max-height:560px)]:space-y-1.5 [@media(max-height:500px)]:py-1.5 [@media(max-height:500px)]:space-y-1.5"
        >
          <p className="text-sm text-zinc-300 leading-relaxed max-w-md [@media(max-height:560px)]:hidden [@media(max-height:500px)]:hidden">
            ピアノロールへ配置するトラックを選択してください。
          </p>
          {(safeTracks.length > MAX_MODAL_IMPORT_SELECTION ||
            selected.size >= MAX_MODAL_IMPORT_SELECTION) && (
            <p
              data-testid="midi-import-selection-cap-hint"
              className="text-xs [@media(max-height:560px)]:text-[11px] text-zinc-400 leading-relaxed [@media(max-height:560px)]:leading-tight [@media(max-height:500px)]:leading-tight max-w-md"
            >
              ワークスペース上限により、一度に読み込めるのは最大 {MAX_MODAL_IMPORT_SELECTION}{' '}
              トラックです（選択中: {selected.size} / {safeTracks.length}）。
            </p>
          )}

          <div className="divide-y divide-zinc-800 overflow-hidden rounded-xl border border-zinc-800">
            {safeTracks.map((track, idx) => {
              const safeIndex = track.canonicalIndex;
              const safeProgram =
                typeof track.program === 'number' && Number.isFinite(track.program)
                  ? Math.max(0, Math.min(127, Math.trunc(track.program)))
                  : 0;
              const safeNoteCount = Array.isArray(track.notes) ? track.notes.length : 0;
              const safeTrackName =
                typeof track.name === 'string' && track.name ? track.name : `Track ${idx + 1}`;
              const safeTrackInst =
                typeof track.instrument === 'string' && track.instrument
                  ? track.instrument
                  : 'PIANO';
              const checked = selected.has(safeIndex);
              const selectionCapReached = !checked && selected.size >= MAX_MODAL_IMPORT_SELECTION;
              return (
                <label
                  key={`${idx}-${safeIndex}`}
                  className={`flex min-h-[48px] [@media(max-height:560px)]:min-h-[44px] [@media(max-height:500px)]:min-h-[44px] items-center justify-between gap-3 py-2.5 [@media(max-height:560px)]:py-1.5 [@media(max-height:500px)]:py-1.5 px-2.5 transition-colors ${
                    selectionCapReached
                      ? 'cursor-not-allowed opacity-50'
                      : 'cursor-pointer hover:bg-zinc-900/70'
                  }`}
                >
                  <div className="flex flex-1 items-center gap-3 min-w-0">
                    <span className="relative inline-flex h-5 w-5 shrink-0 items-center justify-center">
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={selectionCapReached}
                        onChange={() => toggleIndex(safeIndex)}
                        className={`peer h-5 w-5 appearance-none rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 disabled:cursor-not-allowed ${
                          checked
                            ? 'border-sky-400 bg-sky-400'
                            : 'border-[#82828B] bg-zinc-950 hover:border-zinc-300'
                        }`}
                      />
                      {checked && (
                        <Check
                          className="pointer-events-none absolute h-3.5 w-3.5 text-zinc-950"
                          strokeWidth={2.5}
                        />
                      )}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-zinc-100 truncate">
                        {safeTrackName}
                      </div>
                      <div className="text-xs text-zinc-400 font-mono truncate">
                        {safeTrackInst} / Prog {safeProgram} / {safeNoteCount} notes
                      </div>
                    </div>
                  </div>
                </label>
              );
            })}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5 sm:gap-3 border-t border-zinc-800 px-3 sm:px-5 py-2.5 sm:py-3 [@media(max-height:560px)]:py-1.5 [@media(max-height:500px)]:py-1.5">
          <button
            type="button"
            aria-pressed={replaceExisting}
            onClick={() => setReplaceExisting(true)}
            className={`min-h-[44px] min-w-[44px] flex-1 sm:flex-none rounded-full px-2 sm:px-4 text-[11px] sm:text-xs font-medium border truncate transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
              replaceExisting
                ? 'border-sky-400 bg-sky-950/35 text-sky-200'
                : 'border-[#82828B] bg-transparent text-zinc-300 hover:border-zinc-300'
            }`}
          >
            既存トラックを置換
          </button>
          <button
            type="button"
            aria-pressed={!replaceExisting}
            onClick={() => setReplaceExisting(false)}
            className={`min-h-[44px] min-w-[44px] flex-1 sm:flex-none rounded-full px-2 sm:px-4 text-[11px] sm:text-xs font-medium border truncate transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
              !replaceExisting
                ? 'border-sky-400 bg-sky-950/35 text-sky-200'
                : 'border-[#82828B] bg-transparent text-zinc-300 hover:border-zinc-300'
            }`}
          >
            既存トラックに追加
          </button>
        </div>

        <div className="relative z-20 flex shrink-0 items-center justify-end gap-1.5 sm:gap-3 border-t border-zinc-800 px-3 sm:px-5 py-2.5 sm:py-3.5 [@media(max-height:560px)]:py-1.5 [@media(max-height:500px)]:py-1.5">
          <button
            type="button"
            onPointerDown={(e) => {
              if (e.button === 0) lastDialogPointerBtnRef.current = e.currentTarget;
            }}
            onClick={(e) => {
              recordButtonCloseModality(e);
              onCancel();
            }}
            className="min-h-[44px] min-w-[44px] shrink-0 rounded-full border border-[#82828B] px-2.5 sm:px-4 text-xs sm:text-sm text-zinc-300 hover:border-zinc-300 hover:bg-zinc-800/70 hover:text-zinc-100 whitespace-nowrap transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
          >
            キャンセル
          </button>
          <button
            ref={confirmButtonRef}
            type="button"
            disabled={selected.size === 0}
            aria-description={selected.size === 0 ? '1つ以上のトラックを選択してください' : undefined}
            data-tooltip={selected.size === 0 ? '1つ以上のトラックを選択してください' : undefined}
            data-tooltip-when-disabled="true"
            data-tooltip-pos="top"
            data-tooltip-prefer-pos="top"
            data-tooltip-align="end"
            data-testid="confirm-midi-import-button"
            onPointerDown={(e) => {
              if (e.button === 0) lastDialogPointerBtnRef.current = e.currentTarget;
            }}
            onClick={(e) => {
              recordButtonCloseModality(e);
              onConfirm(Array.from(selected), replaceExisting);
            }}
            className="inline-flex min-h-[44px] min-w-[44px] shrink items-center justify-center rounded-full px-3 sm:px-5 bg-zinc-100 text-zinc-950 text-[11px] sm:text-sm font-semibold hover:bg-white disabled:border disabled:border-[#82828B]/45 disabled:bg-zinc-900 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            <span className="min-w-0 truncate pointer-events-none">選択したトラックを読み込む</span>
          </button>
        </div>
      </div>
    </div>
  );
};
