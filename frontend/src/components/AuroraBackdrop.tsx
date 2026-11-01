/**
 * Ambient Crimson-Purple Bottom Aurora Backdrop (`AuroraBackdrop.tsx`).
 * Maintains the deep `#09090B` black studio base while emitting a calm, non-interactive
 * crimson-magenta-purple (`赤紫`) aurora glow upward from the bottom edge of the viewport.
 * Composite bottom luminance stays <= 0.032 so WCAG AA/AAA text contrast is preserved.
 */

import React from 'react';

export const AuroraBackdrop: React.FC = () => {
  return (
    <div
      data-testid="aurora-backdrop"
      aria-hidden="true"
      className="pointer-events-none select-none fixed inset-0 z-0 overflow-hidden"
    >
      <div className="absolute inset-0 bg-aurora-horizon" />
      <div className="absolute inset-0 bg-aurora-core" />
      <div className="absolute inset-0 bg-aurora-curtains" />
    </div>
  );
};
