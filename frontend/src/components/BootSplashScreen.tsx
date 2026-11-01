/**
 * Boot Splash Screen (`BootSplashScreen.tsx`).
 * Displays the COMPASS AI Logo at the center of a pure dark studio canvas (`#09090B`),
 * with `Loading...` and real-time initialization logs directly underneath.
 */

import React from 'react';
import { AuroraBackdrop } from './AuroraBackdrop';
import { CompassLogo } from './CompassLogo';

export interface BootLogEntry {
  readonly id: number;
  readonly timestamp: string;
  readonly message: string;
}

interface BootSplashScreenProps {
  readonly currentAction: string;
  readonly progressPercent: number;
  readonly logs: readonly BootLogEntry[];
}

export const BootSplashScreen: React.FC<BootSplashScreenProps> = ({
  currentAction,
  progressPercent,
  logs,
}) => {
  const safePct =
    typeof progressPercent === 'number' && Number.isFinite(progressPercent)
      ? progressPercent
      : 4;
  const clamped = Math.max(4, Math.min(100, Math.round(safePct)));
  const safeAction = typeof currentAction === 'string' ? currentAction : '';
  const safeLogs = Array.isArray(logs)
    ? logs.filter((entry): entry is BootLogEntry => Boolean(entry && typeof entry === 'object'))
    : [];

  return (
    <div
      className="fixed inset-0 z-50 isolate flex flex-col items-center justify-center overflow-y-auto bg-studio-950 px-6 py-6 text-zinc-100"
      role="status"
      aria-live="polite"
    >
      <AuroraBackdrop />
      <div className="relative z-10 my-auto w-full max-w-md flex flex-col items-center">
        <CompassLogo size="lg" showSubtitle />

        <div className="mt-8 sm:mt-10 w-full flex flex-col items-center">
          <div className="flex items-center gap-2.5 text-sm font-medium tracking-wide text-zinc-200">
            <span>Loading...</span>
            <span className="font-mono text-xs text-zinc-400 tabular-nums">{clamped}%</span>
          </div>

          {/* Crisp rounded studio progress line */}
          <div className="mt-3 h-1.5 w-64 max-w-full overflow-hidden rounded-full bg-zinc-800/90">
            <div
              className="h-full rounded-full bg-zinc-100 transition-all duration-300 ease-out motion-reduce:transition-none"
              style={{ width: `${clamped}%` }}
            />
          </div>

          {/* Active processing step */}
          <p className="mt-4 text-xs text-zinc-300 text-center max-w-md leading-relaxed min-h-[20px]">
            {safeAction}
          </p>

          {/* Boot processing log history */}
          <div
            className="mt-5 w-full border-t border-zinc-800/85 pt-3 font-mono text-xs text-zinc-400 space-y-1.5 max-h-32 overflow-y-auto"
            aria-label="起動処理ログ"
          >
            {safeLogs.slice(-5).map((entry, idx) => {
              const safeKey =
                typeof entry.id === 'number' && Number.isFinite(entry.id)
                  ? `${idx}-${entry.id}`
                  : String(idx);
              const safeTs = typeof entry.timestamp === 'string' ? entry.timestamp : '';
              const safeMsg = typeof entry.message === 'string' ? entry.message : '';
              return (
                <div key={safeKey} className="flex items-baseline gap-2.5">
                  <span className="text-zinc-400 shrink-0 tabular-nums">{safeTs}</span>
                  <span className="text-zinc-300 break-words">{safeMsg}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
