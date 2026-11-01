/**
 * Central Audio-Reactive & AI Generation Vibrating Circle (`ResonanceCircleCanvas.tsx`).
 * - Idle: Calm, calibrated geometric studio circle (`#F4F4F5` / `#27272A` hairline on `#09090B`).
 * - Audio Playing: Vibrates dynamically in real time using waveform & spectral energy from `IAudioEngine`.
 * - AI Generating: Animates smooth harmonic standing waves while displaying the live generation step underneath.
 * Strictly follows `anti-ai-web-design`: zero neon blurs, zero rainbow gradients.
 */

import React, { useEffect, useRef } from 'react';
import { IAudioEngine } from '../audio/WebAudioEngine';

interface ResonanceCircleCanvasProps {
  readonly audioEngine: IAudioEngine;
  readonly isGenerating: boolean;
  readonly isPlayingAudio: boolean;
  readonly statusTitle: string;
  readonly statusDetail: string;
  readonly externalAudioRms?: number;
  readonly isVisible?: boolean;
}

export const ResonanceCircleCanvas: React.FC<ResonanceCircleCanvasProps> = ({
  audioEngine,
  isGenerating,
  isPlayingAudio,
  statusTitle,
  statusDetail,
  externalAudioRms = 0,
  isVisible = true,
}) => {
  const safeIsGenerating = Boolean(isGenerating);
  const safeIsPlayingAudio = Boolean(isPlayingAudio);
  const safeExternalRms =
    typeof externalAudioRms === 'number' && Number.isFinite(externalAudioRms)
      ? Math.max(0, externalAudioRms)
      : 0;
  const safeStatusTitle = typeof statusTitle === 'string' ? statusTitle : '';
  const safeStatusDetail = typeof statusDetail === 'string' ? statusDetail : '';

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef({
    isGenerating: safeIsGenerating,
    isPlayingAudio: safeIsPlayingAudio,
    externalAudioRms: safeExternalRms,
  });

  stateRef.current = {
    isGenerating: safeIsGenerating,
    isPlayingAudio: safeIsPlayingAudio,
    externalAudioRms: safeExternalRms,
  };

  useEffect(() => {
    if (!isVisible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const canvasEl: HTMLCanvasElement = canvas;
    const ctx2d: CanvasRenderingContext2D = ctx;

    const waveformBuf = new Float32Array(1024);
    const spectrumBuf = new Uint8Array(512);
    let animFrameId = 0;
    let isLoopScheduled = false;
    let phase = 0;
    let smoothedEnergy = 0;
    let lastDpr = Math.min(window.devicePixelRatio || 1, 2);
    let measuredCssSize = Math.max(
      180,
      Math.round(canvasEl.getBoundingClientRect().width || canvasEl.clientWidth || 232),
    );

    const render = () => {
      isLoopScheduled = false;
      const {
        isGenerating: genActive,
        isPlayingAudio: playingActive,
        externalAudioRms: extRms,
      } = stateRef.current;
      let rawTelemetry: {
        rms?: number;
        peak?: number;
        lowBandEnergy?: number;
        highBandEnergy?: number;
      } = { rms: 0, peak: 0, lowBandEnergy: 0, highBandEnergy: 0 };
      if (!reducedMotion && audioEngine && typeof audioEngine === 'object') {
        try {
          if (typeof audioEngine.readAcousticTelemetry === 'function') {
            const res = audioEngine.readAcousticTelemetry(waveformBuf, spectrumBuf);
            if (res && typeof res === 'object') {
              rawTelemetry = res;
            }
          }
        } catch {
          // Never crash 60fps rAF loop if a custom IAudioEngine throws at runtime
        }
      }
      const telemetryRms =
        typeof rawTelemetry?.rms === 'number' && Number.isFinite(rawTelemetry.rms)
          ? Math.max(0, rawTelemetry.rms)
          : 0;
      const telemetryHighBand =
        typeof rawTelemetry?.highBandEnergy === 'number' &&
        Number.isFinite(rawTelemetry.highBandEnergy)
          ? Math.max(0, rawTelemetry.highBandEnergy)
          : 0;
      const liveRms = reducedMotion ? 0 : Math.max(telemetryRms, extRms);
      const targetEnergy = reducedMotion
        ? 0
        : genActive
          ? Math.max(0.38, liveRms * 2.4)
          : liveRms * 3.2;
      smoothedEnergy = reducedMotion ? 0 : smoothedEnergy + (targetEnergy - smoothedEnergy) * 0.18;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssSize = measuredCssSize;
      const targetPx = Math.round(cssSize * dpr);
      if (canvasEl.width !== targetPx || canvasEl.height !== targetPx) {
        canvasEl.width = targetPx;
        canvasEl.height = targetPx;
      }

      ctx2d.save();
      ctx2d.scale(targetPx / cssSize, targetPx / cssSize);
      ctx2d.clearRect(0, 0, cssSize, cssSize);

      const cx = cssSize / 2;
      const cy = cssSize / 2;
      const baseRadius = Math.round(cssSize * (74 / 232));

      // Outer static reference guide hairline
      ctx2d.beginPath();
      ctx2d.arc(cx, cy, baseRadius + 22, 0, Math.PI * 2);
      ctx2d.strokeStyle = '#1F1F23';
      ctx2d.lineWidth = 1;
      ctx2d.stroke();

      // Inner static reference guide hairline
      ctx2d.beginPath();
      ctx2d.arc(cx, cy, baseRadius - 22, 0, Math.PI * 2);
      ctx2d.strokeStyle = '#18181B';
      ctx2d.lineWidth = 1;
      ctx2d.stroke();

      if (reducedMotion) {
        // Static accessible studio ring when `prefers-reduced-motion: reduce` is active ([R3-MED-02])
        if (genActive || playingActive) {
          ctx2d.beginPath();
          ctx2d.arc(cx, cy, baseRadius + 6, 0, Math.PI * 2);
          ctx2d.strokeStyle = genActive ? '#38BDF8' : '#52525B';
          ctx2d.globalAlpha = 0.45;
          ctx2d.lineWidth = 1.1;
          ctx2d.stroke();
          ctx2d.globalAlpha = 1;
        }
        ctx2d.beginPath();
        ctx2d.arc(cx, cy, baseRadius, 0, Math.PI * 2);
        ctx2d.strokeStyle = genActive || playingActive ? '#F4F4F5' : '#D4D4D8';
        ctx2d.lineWidth = genActive || playingActive ? 1.85 : 1.5;
        ctx2d.stroke();

        ctx2d.beginPath();
        ctx2d.arc(cx, cy, genActive ? 3 : 2, 0, Math.PI * 2);
        ctx2d.fillStyle = genActive ? '#38BDF8' : playingActive ? '#F4F4F5' : '#52525B';
        ctx2d.fill();

        ctx2d.restore();
        return;
      }

      const segments = 180;
      if (!Number.isFinite(smoothedEnergy)) smoothedEnergy = 0;
      if (!Number.isFinite(phase)) phase = 0;
      phase += genActive ? 0.055 : smoothedEnergy > 0.005 ? 0.045 : 0.008;

      const readFiniteWaveSample = (idx: number): number => {
        const rawVal = waveformBuf[idx];
        return typeof rawVal === 'number' && Number.isFinite(rawVal)
          ? Math.max(-1, Math.min(1, rawVal))
          : 0;
      };

      // Secondary harmonic trace (subtle zinc-600 line when vibrating)
      if (smoothedEnergy > 0.008 || genActive) {
        ctx2d.beginPath();
        for (let i = 0; i <= segments; i += 1) {
          const theta = (i / segments) * Math.PI * 2;
          const waveIdx = Math.min(
            waveformBuf.length - 1,
            Math.floor((i / segments) * (waveformBuf.length * 0.5)),
          );
          const sample = readFiniteWaveSample(waveIdx);

          let disp = 0;
          if (genActive) {
            disp =
              Math.sin(theta * 3 - phase * 1.4) * 5.5 +
              Math.cos(theta * 5 + phase * 2.1) * 3.5;
          } else {
            disp =
              sample * 24 * Math.sin(theta * 2 + phase) +
              Math.sin(theta * 6 - phase * 2) * (smoothedEnergy * 7);
          }

          const r = baseRadius + disp * 0.72;
          const x = cx + Math.cos(theta) * r;
          const y = cy + Math.sin(theta) * r;
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          if (i === 0) ctx2d.moveTo(x, y);
          else ctx2d.lineTo(x, y);
        }
        ctx2d.closePath();
        ctx2d.strokeStyle = genActive ? '#38BDF8' : '#52525B';
        const traceAlpha = genActive ? 0.45 : Math.max(0, Math.min(0.65, smoothedEnergy * 1.8));
        ctx2d.globalAlpha = Number.isFinite(traceAlpha) ? traceAlpha : 0.45;
        ctx2d.lineWidth = 1.1;
        ctx2d.stroke();
        ctx2d.globalAlpha = 1;
      }

      // Primary Acoustic / Inference Circle
      ctx2d.beginPath();
      for (let i = 0; i <= segments; i += 1) {
        const theta = (i / segments) * Math.PI * 2;
        // Seamless closed-loop angular envelope (`sin^2(theta/2)` avoids seam discontinuity at 0/2pi)
        const seamWindow = Math.sin(theta * 0.5);
        const waveIdx = Math.min(
          waveformBuf.length - 1,
          Math.floor((i / segments) * waveformBuf.length),
        );
        const rawSample = readFiniteWaveSample(waveIdx) * seamWindow;

        let radialOffset = 0;
        if (genActive) {
          radialOffset =
            Math.sin(theta * 4 + phase * 1.8) * 6.5 +
            Math.sin(theta * 7 - phase * 2.6) * 3.2 +
            rawSample * 16;
        } else if (smoothedEnergy > 0.002) {
          radialOffset =
            rawSample * 34 +
            Math.sin(theta * 4 + phase * 2.4) * (smoothedEnergy * 10) +
            Math.cos(theta * 8 - phase * 3.1) * (telemetryHighBand * 8);
        }

        const r = baseRadius + radialOffset;
        const x = cx + Math.cos(theta) * r;
        const y = cy + Math.sin(theta) * r;
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (i === 0) ctx2d.moveTo(x, y);
        else ctx2d.lineTo(x, y);
      }
      ctx2d.closePath();
      ctx2d.strokeStyle = genActive ? '#F4F4F5' : smoothedEnergy > 0.015 ? '#F4F4F5' : '#D4D4D8';
      ctx2d.lineWidth = genActive || smoothedEnergy > 0.015 ? 1.85 : 1.5;
      ctx2d.stroke();

      // Center precision crosshair dot
      ctx2d.beginPath();
      ctx2d.arc(cx, cy, genActive ? 3 : 2, 0, Math.PI * 2);
      ctx2d.fillStyle = genActive ? '#38BDF8' : playingActive ? '#F4F4F5' : '#52525B';
      ctx2d.fill();

      ctx2d.restore();

      const settledIdle =
        !genActive && !playingActive && smoothedEnergy < 0.0008 && liveRms < 0.0005;
      if (!settledIdle) {
        isLoopScheduled = true;
        animFrameId = window.requestAnimationFrame(render);
      }
    };

    const scheduleSingleRender = () => {
      if (isLoopScheduled) return;
      isLoopScheduled = true;
      animFrameId = window.requestAnimationFrame(render);
    };

    const updateCssSize = () => {
      const next = Math.round(canvasEl.getBoundingClientRect().width || canvasEl.clientWidth || 232);
      const nextDpr = Math.min(window.devicePixelRatio || 1, 2);
      let changed = false;
      if (next >= 120 && next !== measuredCssSize) {
        measuredCssSize = next;
        changed = true;
      }
      if (nextDpr !== lastDpr) {
        lastDpr = nextDpr;
        changed = true;
      }
      if (changed && !isLoopScheduled) {
        scheduleSingleRender();
      }
    };

    window.addEventListener('resize', updateCssSize);
    const resizeObs =
      typeof ResizeObserver !== 'undefined'
        ? new ResizeObserver(() => {
            updateCssSize();
          })
        : null;
    resizeObs?.observe(canvasEl);

    const mql =
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-reduced-motion: reduce)')
        : null;
    let reducedMotion = Boolean(mql?.matches);

    const onMotionChange = (ev: MediaQueryListEvent) => {
      reducedMotion = ev.matches;
      window.cancelAnimationFrame(animFrameId);
      isLoopScheduled = false;
      scheduleSingleRender();
    };
    if (mql) {
      if (typeof mql.addEventListener === 'function') {
        mql.addEventListener('change', onMotionChange);
      } else if (typeof mql.addListener === 'function') {
        mql.addListener(onMotionChange);
      }
    }

    scheduleSingleRender();
    return () => {
      isLoopScheduled = false;
      if (mql) {
        if (typeof mql.removeEventListener === 'function') {
          mql.removeEventListener('change', onMotionChange);
        } else if (typeof mql.removeListener === 'function') {
          mql.removeListener(onMotionChange);
        }
      }
      window.removeEventListener('resize', updateCssSize);
      resizeObs?.disconnect();
      window.cancelAnimationFrame(animFrameId);
    };
  }, [audioEngine, isVisible, safeIsGenerating, safeIsPlayingAudio, safeExternalRms]);

  return (
    <div className="flex flex-col items-center justify-center py-2 lg:py-3 select-none">
      <div className="relative flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={216}
          height={216}
          className="w-[196px] h-[196px] sm:w-[216px] sm:h-[216px]"
          aria-label="COMPASS AI 音響・推論レゾナンスサークル"
        />
      </div>

      {/* Live operational status directly beneath the circle */}
      <div className="mt-2 text-center max-w-md px-1" role="status" aria-live="polite">
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-[0.16em] text-zinc-400">
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${
              safeIsGenerating
                ? 'bg-sky-400'
                : safeIsPlayingAudio
                  ? 'bg-emerald-400'
                  : 'bg-zinc-600'
            }`}
          />
          <span>{safeStatusTitle}</span>
        </div>
        <p className="mt-1 text-sm text-zinc-300 leading-relaxed text-pretty min-h-[22px]">
          {safeStatusDetail}
        </p>
      </div>
    </div>
  );
};
