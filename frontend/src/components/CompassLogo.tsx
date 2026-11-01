/**
 * Custom Precision Geometric SVG Logo for COMPASS AI (`anti-ai-web-design`).
 * Strictly monochrome vector geometry — zero emojis, zero gradients, zero neon glows.
 */

import React from 'react';

interface CompassLogoProps {
  readonly size?: 'sm' | 'md' | 'lg';
  readonly showSubtitle?: boolean;
}

export const CompassLogo: React.FC<CompassLogoProps> = ({
  size = 'md',
  showSubtitle = false,
}) => {
  const iconPx = size === 'lg' ? 64 : size === 'md' ? 36 : 24;
  const titleClass =
    size === 'lg'
      ? 'text-2xl font-semibold tracking-[0.22em] text-zinc-100'
      : size === 'md'
        ? 'text-base font-semibold tracking-[0.18em] text-zinc-100'
        : 'text-sm font-semibold tracking-[0.16em] text-zinc-100';

  return (
    <div className="inline-flex flex-col items-center select-none">
      <div className="inline-flex items-center gap-3.5">
        <svg
          width={iconPx}
          height={iconPx}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
          data-testid="compass-logo-svg"
          className="text-zinc-100 shrink-0"
        >
          {/* Outer calibrated studio ring */}
          <circle
            cx="32"
            cy="32"
            r="28"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeOpacity="0.9"
          />
          {/* Inner acoustic reference circle */}
          <circle
            cx="32"
            cy="32"
            r="19"
            stroke="currentColor"
            strokeWidth="1"
            strokeOpacity="0.32"
          />
          {/* Cardinal calibration ticks */}
          <line x1="32" y1="4" x2="32" y2="10" stroke="currentColor" strokeWidth="1.5" />
          <line x1="32" y1="54" x2="32" y2="60" stroke="currentColor" strokeWidth="1.5" strokeOpacity="0.5" />
          <line x1="4" y1="32" x2="10" y2="32" stroke="currentColor" strokeWidth="1.5" strokeOpacity="0.5" />
          <line x1="54" y1="32" x2="60" y2="32" stroke="currentColor" strokeWidth="1.5" strokeOpacity="0.5" />

          {/* Precision compass needle polygon */}
          <polygon
            points="32,13 37.5,32 32,29.5 26.5,32"
            fill="currentColor"
          />
          <polygon
            points="32,51 37.5,32 32,34.5 26.5,32"
            stroke="currentColor"
            strokeWidth="1.25"
            fill="none"
            strokeOpacity="0.55"
          />
          {/* Center pivot */}
          <circle cx="32" cy="32" r="2.2" fill="#09090B" stroke="currentColor" strokeWidth="1.4" />
        </svg>

        {size !== 'lg' && <span className={titleClass}>COMPASS AI</span>}
      </div>

      {size === 'lg' && (
        <div className="mt-5 text-center">
          <div className={titleClass}>COMPASS AI</div>
          {showSubtitle && (
            <p className="mt-1.5 text-xs tracking-[0.14em] uppercase text-zinc-400 font-mono">
              Symbolic Music Foundation Workstation
            </p>
          )}
        </div>
      )}
    </div>
  );
};
