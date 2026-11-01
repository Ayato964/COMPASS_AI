/**
 * Main COMPASS AI Screen (`ControlPanelView.tsx`).
 * Displays the central audio-reactive / inference-reactive circle at the top
 * and the monochrome studio Control Panel directly underneath.
 * Strictly adheres to `anti-ai-web-design`:
 * - Zero emojis, zero decorative glyphs, zero gradients, zero neon glows
 * - Single high-contrast solid white CTA (`生成を実行`) per view (§5.3 One Primary Action)
 * - Negative-space layout with 1px zinc hairlines (zero cardification clutter)
 * - Japanese copy constrained to 35-45 characters per line (`max-w-md`)
 * - All interactive controls >= 44x44px, zero horizontal scroll at 375px
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Pause, Play, RefreshCw } from 'lucide-react';
import { IAudioEngine } from '../audio/WebAudioEngine';
import {
  DawSessionSnapshot,
  IDawController,
  resolveEffectiveSftTask,
  SftTaskMode,
} from '../controllers/IDawController';
import { evaluateGenerationMeter } from '../domain/MeasureRange';
import {
  ALL_MUSICAL_KEYS,
  ALL_SFT_GENRES,
  HardwareStatusInfo,
  ModelMetadata,
} from '../services/CompassApiClient';
import { ResonanceCircleCanvas } from './ResonanceCircleCanvas';
import { useAdaptiveTooltipAlignment } from '../utils/useAdaptiveTooltipAlignment';

export interface ControlPanelSettings {
  readonly modelName: string;
  readonly temperature: number;
  readonly topP: number;
  readonly thinking: boolean;
  readonly cotTemperature: number;
  readonly numCandidates: number;
  readonly taskMode: SftTaskMode;
  readonly conditionTrackId: string;
  readonly forceMonophonic: boolean;
  readonly noteDensity: number | null;
  readonly musicalKey: string;
  readonly selectedGenres: readonly string[];
}

interface ControlPanelViewProps {
  readonly isActive?: boolean;
  readonly audioEngine: IAudioEngine;
  readonly controllers: readonly IDawController[];
  readonly session: DawSessionSnapshot;
  readonly models: readonly ModelMetadata[];
  readonly hardwareInfo: HardwareStatusInfo | null;
  readonly settings: ControlPanelSettings;
  readonly isGenerating: boolean;
  readonly isRescanningModels: boolean;
  readonly statusTitle: string;
  readonly statusDetail: string;
  readonly generationLogs: readonly string[];
  readonly errorMessage: string | null;
  readonly onUpdateSettings: (patch: Partial<ControlPanelSettings>) => void;
  readonly onSwitchController: (controllerId: string) => void;
  readonly onRescanModels: () => void;
  readonly onTogglePlayback: () => void;
  readonly onGenerate: () => void;
  readonly onCancelGenerate: () => void;
}

const TASK_LABELS: Record<SftTaskMode, string> = {
  auto: '自動判定 (ピアノロールの前後文脈から推論)',
  meta: 'meta — 新規フレーズ生成 (メタ条件のみ)',
  meta_past: 'meta_past — 前方文脈からの続き生成',
  meta_future: 'meta_future — 後方文脈へ繋がる導入生成',
  infill: 'infill — 前後小節を繋ぐインフィリング補完',
  inst_comp: 'inst_comp — 別トラック伴奏・対旋律生成',
};

export const ControlPanelView: React.FC<ControlPanelViewProps> = ({
  isActive = true,
  audioEngine,
  controllers,
  session,
  models,
  hardwareInfo,
  settings,
  isGenerating,
  isRescanningModels,
  statusTitle,
  statusDetail,
  generationLogs,
  errorMessage,
  onUpdateSettings,
  onSwitchController,
  onRescanModels,
  onTogglePlayback,
  onGenerate,
  onCancelGenerate,
}) => {
  useAdaptiveTooltipAlignment();
  const [showStructureOptions, setShowStructureOptions] = useState(false);
  const [genreFilter, setGenreFilter] = useState('');
  const modelSelectRef = useRef<HTMLSelectElement | null>(null);
  const structureSectionRef = useRef<HTMLDivElement | null>(null);
  const structureToggleBtnRef = useRef<HTMLButtonElement | null>(null);
  const pointerPressedElementRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    let pressedFallbackTimer: number | null = null;
    const clearPressedFallbackTimer = () => {
      if (pressedFallbackTimer !== null) {
        window.clearTimeout(pressedFallbackTimer);
        pressedFallbackTimer = null;
      }
    };
    const clearPressed = () => {
      clearPressedFallbackTimer();
      pointerPressedElementRef.current = null;
    };
    const onPointerDownCapture = (ev: PointerEvent) => {
      clearPressedFallbackTimer();
      const isPrimary = ev.button === 0;
      const targetEl = ev.target instanceof Element ? ev.target : null;
      const rawBtn = targetEl?.closest('button') ?? null;
      pointerPressedElementRef.current =
        isPrimary && rawBtn instanceof HTMLButtonElement && !rawBtn.disabled ? rawBtn : null;
    };
    const onPointerUpCapture = (ev: PointerEvent) => {
      const pressed = pointerPressedElementRef.current;
      if (!pressed) return;
      const targetEl = ev.target instanceof Element ? ev.target : null;
      if (!targetEl || !pressed.contains(targetEl) || pressed.disabled || !pressed.isConnected) {
        clearPressed();
        return;
      }
      clearPressedFallbackTimer();
      pressedFallbackTimer = window.setTimeout(() => {
        pressedFallbackTimer = null;
        pointerPressedElementRef.current = null;
      }, 120);
    };
    window.addEventListener('pointerdown', onPointerDownCapture, true);
    window.addEventListener('pointerup', onPointerUpCapture, true);
    window.addEventListener('pointercancel', clearPressed, true);
    window.addEventListener('click', clearPressed);
    window.addEventListener('keydown', clearPressed, true);
    return () => {
      clearPressedFallbackTimer();
      window.removeEventListener('pointerdown', onPointerDownCapture, true);
      window.removeEventListener('pointerup', onPointerUpCapture, true);
      window.removeEventListener('pointercancel', clearPressed, true);
      window.removeEventListener('click', clearPressed);
      window.removeEventListener('keydown', clearPressed, true);
    };
  }, []);

  const isPointerClickEvent = (e: React.MouseEvent<HTMLButtonElement>): boolean => {
    if (e.detail > 0) return true;
    const pressed = pointerPressedElementRef.current;
    return pressed !== null && pressed === e.currentTarget;
  };

  const safeModels = useMemo(
    () =>
      Array.isArray(models)
        ? models.filter((m): m is ModelMetadata =>
            Boolean(m && typeof m === 'object' && typeof m.model_name === 'string'),
          )
        : [],
    [models],
  );
  const safeControllers = useMemo(
    () =>
      Array.isArray(controllers)
        ? controllers.filter((c): c is IDawController =>
            Boolean(c && typeof c === 'object' && typeof c.id === 'string'),
          )
        : [],
    [controllers],
  );
  const safeGenerationLogs = useMemo(
    () =>
      Array.isArray(generationLogs)
        ? generationLogs.filter((l): l is string => typeof l === 'string')
        : [],
    [generationLogs],
  );
  const safeSettings: Partial<ControlPanelSettings> =
    settings && typeof settings === 'object' ? settings : {};
  const safeStatusTitle = typeof statusTitle === 'string' ? statusTitle : '';
  const safeStatusDetail = typeof statusDetail === 'string' ? statusDetail : '';
  const safeErrorMessage =
    typeof errorMessage === 'string' && errorMessage.length > 0 ? errorMessage : null;
  const safeSupportsTransport = Boolean(session?.capabilities?.supportsTransportControl);
  const safeIsPlaying = Boolean(session?.isPlaying);
  const safeIsConnected = Boolean(session?.isConnected);
  const safeConnectionStatusText =
    typeof session?.connectionStatusText === 'string' ? session.connectionStatusText : '';
  const safeHwDevice =
    hardwareInfo && typeof hardwareInfo.device === 'string' && hardwareInfo.device.trim()
      ? hardwareInfo.device.toUpperCase()
      : 'CPU';
  const safeHwDtype =
    hardwareInfo && typeof hardwareInfo.dtype === 'string' && hardwareInfo.dtype.trim()
      ? hardwareInfo.dtype
      : 'float32';
  const safeHwUserDir =
    hardwareInfo && typeof hardwareInfo.user_models_dir === 'string'
      ? hardwareInfo.user_models_dir
      : '';

  const meterEvaluation = useMemo(
    () => evaluateGenerationMeter(session?.timeSigNumerator, session?.timeSigDenominator),
    [session?.timeSigNumerator, session?.timeSigDenominator],
  );

  const safeModelRaw = typeof safeSettings.modelName === 'string' ? safeSettings.modelName : '';
  const safeModelName = safeModels.some((m) => m.model_name === safeModelRaw)
    ? safeModelRaw
    : (safeModels[0]?.model_name ?? '');

  const selectedModelMeta = useMemo(
    () => safeModels.find((m) => m.model_name === safeModelName) ?? null,
    [safeModels, safeModelName],
  );

  const isFoundationModel = useMemo(
    () =>
      (typeof selectedModelMeta?.tag?.model === 'string'
        ? selectedModelMeta.tag.model.toLowerCase()
        : '') === 'foundation',
    [selectedModelMeta],
  );

  // Chain-of-Thought `<thinking>` and 42-genre SFT conditioning are supported exclusively by `sft_gen` models
  const isSftGenModel = useMemo(() => {
    const tagModel =
      typeof selectedModelMeta?.tag?.model === 'string'
        ? selectedModelMeta.tag.model.toLowerCase()
        : '';
    if (tagModel) {
      return tagModel === 'sft_gen';
    }
    return safeModelRaw.toLowerCase().includes('sft');
  }, [safeModelRaw, selectedModelMeta]);

  const safeTaskMode: SftTaskMode =
    typeof safeSettings.taskMode === 'string' && safeSettings.taskMode in TASK_LABELS
      ? safeSettings.taskMode
      : 'auto';

  const safeThinking = Boolean(safeSettings.thinking);
  const safeForceMonophonic = Boolean(safeSettings.forceMonophonic);
  const safeTemperature =
    typeof safeSettings.temperature === 'number' && Number.isFinite(safeSettings.temperature)
      ? Math.max(0.1, Math.min(2.0, safeSettings.temperature))
      : 1.0;
  const safeTopP =
    typeof safeSettings.topP === 'number' && Number.isFinite(safeSettings.topP)
      ? Math.max(0.1, Math.min(1.0, safeSettings.topP))
      : 0.95;
  const safeCotTemperature =
    typeof safeSettings.cotTemperature === 'number' && Number.isFinite(safeSettings.cotTemperature)
      ? Math.max(0.1, Math.min(1.5, safeSettings.cotTemperature))
      : 0.8;
  const safeNumCandidates =
    typeof safeSettings.numCandidates === 'number' &&
    Number.isFinite(safeSettings.numCandidates) &&
    [1, 2, 3, 4].includes(Math.round(safeSettings.numCandidates))
      ? Math.round(safeSettings.numCandidates)
      : 1;
  const safeNoteDensity: number | '' =
    typeof safeSettings.noteDensity === 'number' &&
    Number.isInteger(safeSettings.noteDensity) &&
    safeSettings.noteDensity >= 1 &&
    safeSettings.noteDensity <= 10
      ? safeSettings.noteDensity
      : '';
  const safeControllerId =
    typeof session?.controllerId === 'string' &&
    safeControllers.some((c) => c.id === session.controllerId)
      ? session.controllerId
      : (safeControllers[0]?.id ?? '');
  const safeTracks = useMemo(
    () =>
      Array.isArray(session?.tracks)
        ? session.tracks.filter((t) => Boolean(t && typeof t === 'object' && typeof t.id === 'string'))
        : [],
    [session?.tracks],
  );
  const safeActiveTrackId =
    typeof session?.activeTrackId === 'string' &&
    safeTracks.some((t) => t.id === session.activeTrackId)
      ? session.activeTrackId
      : (safeTracks[0]?.id ?? '');
  const safeCondTrackId =
    typeof safeSettings.conditionTrackId === 'string' &&
    safeSettings.conditionTrackId !== safeActiveTrackId &&
    safeTracks.some((t) => t.id === safeSettings.conditionTrackId)
      ? safeSettings.conditionTrackId
      : '';
  const safeMusicalKey =
    typeof safeSettings.musicalKey === 'string' &&
    (ALL_MUSICAL_KEYS as readonly string[]).includes(safeSettings.musicalKey)
      ? safeSettings.musicalKey
      : '';
  const safeSelectedGenres = useMemo(
    () =>
      Array.isArray(safeSettings.selectedGenres)
        ? safeSettings.selectedGenres
            .filter(
              (g): g is string =>
                typeof g === 'string' && (ALL_SFT_GENRES as readonly string[]).includes(g),
            )
            .slice(0, 3)
        : [],
    [safeSettings.selectedGenres],
  );
  const safePastNoteCount =
    typeof session?.contextAvailability?.pastNoteCount === 'number' &&
    Number.isFinite(session.contextAvailability.pastNoteCount)
      ? Math.max(0, Math.round(session.contextAvailability.pastNoteCount))
      : 0;
  const safeFutureNoteCount =
    typeof session?.contextAvailability?.futureNoteCount === 'number' &&
    Number.isFinite(session.contextAvailability.futureNoteCount)
      ? Math.max(0, Math.round(session.contextAvailability.futureNoteCount))
      : 0;
  const conditionCandidateTracks = useMemo(
    () =>
      Array.isArray(session?.contextAvailability?.conditionCandidateTracks)
        ? session.contextAvailability.conditionCandidateTracks.filter((t) =>
            Boolean(t && typeof t === 'object' && typeof t.id === 'string'),
          )
        : [],
    [session?.contextAvailability?.conditionCandidateTracks],
  );
  const hasSelectedCondContext = useMemo(
    () =>
      Boolean(
        safeCondTrackId &&
          conditionCandidateTracks.some(
            (t) => t.id === safeCondTrackId && typeof t.noteCount === 'number' && t.noteCount > 0,
          ),
      ),
    [conditionCandidateTracks, safeCondTrackId],
  );
  const hasAnyCondContext = useMemo(
    () => conditionCandidateTracks.some((t) => typeof t.noteCount === 'number' && t.noteCount > 0),
    [conditionCandidateTracks],
  );
  const effectiveResolvedTask = useMemo(
    () =>
      resolveEffectiveSftTask({
        requestedTask: safeTaskMode,
        hasPastContext:
          Boolean(session?.contextAvailability?.hasPastContext) || safePastNoteCount > 0,
        hasFutureContext:
          Boolean(session?.contextAvailability?.hasFutureContext) || safeFutureNoteCount > 0,
        hasExplicitConditionTrackSelection: Boolean(safeCondTrackId),
        hasSelectedConditionContext: hasSelectedCondContext,
        hasAnyConditionContext: hasAnyCondContext,
      }),
    [
      hasAnyCondContext,
      hasSelectedCondContext,
      safeCondTrackId,
      safeFutureNoteCount,
      safePastNoteCount,
      safeTaskMode,
      session?.contextAvailability?.hasFutureContext,
      session?.contextAvailability?.hasPastContext,
    ],
  );

  // In `MORTM45DRapper`, `task === 'meta'` supplies the complete system prompt directly, so CoT is skipped
  const isCotBypassedByMetaTask = isSftGenModel && effectiveResolvedTask === 'meta';
  const isCotEffectivelyActive =
    isSftGenModel && !isCotBypassedByMetaTask && safeThinking;

  const filteredGenres = useMemo(() => {
    const q = genreFilter.trim().toLowerCase();
    if (!q) return ALL_SFT_GENRES;
    return ALL_SFT_GENRES.filter((g) => g.includes(q));
  }, [genreFilter]);

  const toggleGenre = (genre: string) => {
    if (!isSftGenModel) return;
    const exists = safeSelectedGenres.includes(genre);
    if (exists) {
      onUpdateSettings({
        selectedGenres: safeSelectedGenres.filter((g) => g !== genre),
      });
    } else if (safeSelectedGenres.length < 3) {
      onUpdateSettings({
        selectedGenres: [...safeSelectedGenres, genre],
      });
    }
  };

  const effectiveTaskLabel =
    safeTaskMode === 'auto'
      ? `自動判定: ${effectiveResolvedTask}`
      : effectiveResolvedTask === safeTaskMode
        ? safeTaskMode
        : `${safeTaskMode} (実効: ${effectiveResolvedTask})`;

  const priorGenerationLogs = useMemo(() => {
    const heroDetail = safeStatusDetail.trim();
    if (!heroDetail) return safeGenerationLogs;
    return safeGenerationLogs.filter(
      (line) => line.replace(/^\d{2}:\d{2}:\d{2}\s+/, '').trim() !== heroDetail,
    );
  }, [safeGenerationLogs, safeStatusDetail]);

  const cotStatusBadge = !isSftGenModel
    ? 'SFT-Gen専用'
    : isCotBypassedByMetaTask
      ? 'AUTO-SKIP (meta)'
      : safeThinking
        ? 'ACTIVE'
        : 'OFF';

  const safeCtrlStartMeasure =
    typeof session?.targetRange?.startMeasure === 'number' &&
    Number.isFinite(session.targetRange.startMeasure)
      ? Math.max(1, Math.min(1024, Math.round(session.targetRange.startMeasure)))
      : 1;
  const safeCtrlEndMeasure =
    typeof session?.targetRange?.endMeasure === 'number' &&
    Number.isFinite(session.targetRange.endMeasure)
      ? Math.max(
          safeCtrlStartMeasure,
          Math.min(safeCtrlStartMeasure + 7, Math.round(session.targetRange.endMeasure)),
        )
      : Math.min(1031, safeCtrlStartMeasure + 3);

  const activeTrackSummary = useMemo(() => {
    const found = safeTracks.find((t) => t.id === safeActiveTrackId) ?? safeTracks[0];
    if (!found) return 'Track 1';
    return typeof found.name === 'string' && found.name.trim() ? found.name.trim() : found.id;
  }, [safeActiveTrackId, safeTracks]);

  return (
    <div className="w-full max-w-full overflow-x-clip bg-transparent text-zinc-100">
      <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 pt-3 pb-8 lg:py-2 [@media(min-height:860px)]:lg:py-3">
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(280px,332px)_minmax(0,1fr)] gap-6 lg:gap-8 items-start">
          {/* Left Studio Monitor Column: Sound/Generation-Reactive Circle & Primary Action */}
          <section
            aria-label="レゾナンスモニター"
            className="flex flex-col items-center lg:border-r lg:border-zinc-800/85 lg:pr-6 lg:py-1"
          >
            <ResonanceCircleCanvas
              audioEngine={audioEngine}
              isGenerating={Boolean(isGenerating)}
              isPlayingAudio={safeIsPlaying}
              statusTitle={safeStatusTitle}
              statusDetail={safeStatusDetail}
              isVisible={isActive}
            />

            {/* Quiet read-only Piano Roll target context summary (zero redundant form controls) */}
            <p
              data-testid="pianoroll-context-summary"
              className="mt-1 text-center font-mono text-xs text-zinc-400 max-w-md"
            >
              {activeTrackSummary} · {safeCtrlStartMeasure}–{safeCtrlEndMeasure}小節 · {effectiveTaskLabel}
              {(safePastNoteCount > 0 || safeFutureNoteCount > 0) &&
                ` · Past ${safePastNoteCount} / Future ${safeFutureNoteCount}`}
            </p>

            {/* Primary Action Row (Strictly One Primary Solid CTA + Icon-Only Audio Preview) */}
            <div className="mt-4 flex flex-wrap items-center justify-center gap-3 w-full">
              {isGenerating ? (
                <button
                  type="button"
                  onClick={onCancelGenerate}
                  className="inline-flex min-h-[48px] min-w-[180px] items-center justify-center rounded-full border border-[#82828B] bg-studio-900/90 px-7 text-sm font-medium text-zinc-100 hover:border-zinc-300 hover:bg-zinc-800 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                >
                  生成処理を中断
                </button>
              ) : (
                <button
                  type="button"
                  onClick={onGenerate}
                  disabled={!meterEvaluation.supported}
                  aria-describedby={
                    !meterEvaluation.supported && meterEvaluation.reason
                      ? 'unsupported-meter-banner'
                      : undefined
                  }
                  data-testid="generate-button"
                  className="inline-flex min-h-[48px] min-w-[200px] items-center justify-center rounded-full bg-zinc-100 px-8 text-sm font-semibold tracking-wide text-zinc-950 hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                >
                  生成を実行
                </button>
              )}

              <button
                type="button"
                onClick={onTogglePlayback}
                disabled={!safeSupportsTransport}
                aria-label={safeIsPlaying ? '一時停止' : 'サウンド確認'}
                aria-description={
                  !safeSupportsTransport ? '外部DAW接続中はホストDAW側で再生を操作します' : undefined
                }
                data-tooltip={
                  !safeSupportsTransport
                    ? '外部DAW接続中はホストDAW側で再生を操作します'
                    : safeIsPlaying
                      ? '一時停止'
                      : 'サウンド確認'
                }
                data-tooltip-when-disabled="true"
                className="inline-flex min-h-[48px] min-w-[48px] items-center justify-center rounded-full border border-[#82828B] bg-studio-900/80 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 disabled:border-[#82828B]/40 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
              >
                {safeIsPlaying ? (
                  <Pause className="h-4 w-4" strokeWidth={1.75} />
                ) : (
                  <Play className="h-4 w-4" strokeWidth={1.75} />
                )}
              </button>
            </div>

            {/* Prior step log feed during AI generation (excludes current hero status line to avoid redundancy) */}
            {priorGenerationLogs.length > 0 && (
              <div
                className="mt-3 w-full max-w-md border-t border-zinc-800/85 pt-2 font-mono text-xs text-zinc-400 space-y-1"
                aria-label="生成処理ログ"
                data-testid="generation-log-feed"
              >
                {priorGenerationLogs.slice(-3).map((line, idx) => (
                  <div key={`${idx}-${line}`} className="truncate text-center text-zinc-300">
                    {line}
                  </div>
                ))}
              </div>
            )}

            {/* Error Banner (Semantic Red, zero glow) */}
            {safeErrorMessage && (
              <div
                role="alert"
                className="mt-3 w-full max-w-md rounded-xl border border-red-900/80 bg-red-950/35 px-3.5 py-2.5 text-xs text-red-200 leading-relaxed"
              >
                {safeErrorMessage}
              </div>
            )}

            {/* Unsupported Meter Warning Banner (`[R4-MED-02]`, zero emojis/gradients/glow) */}
            {!meterEvaluation.supported && meterEvaluation.reason && (
              <div
                id="unsupported-meter-banner"
                role="status"
                data-testid="unsupported-meter-banner"
                className="mt-3 w-full max-w-md rounded-xl border border-amber-900/80 bg-amber-950/35 px-3.5 py-2.5 text-xs text-amber-200 leading-relaxed"
              >
                {meterEvaluation.reason}
              </div>
            )}
          </section>

          {/* Right Parameter Console — Zero-Scroll Desktop Fit with Negative Space & Hairlines */}
          <section
            aria-label="生成コントロールパネル"
            className="min-w-0 space-y-4 lg:space-y-2.5 [@media(min-height:860px)]:lg:space-y-3"
          >
            {/* Section 1: Model & Host DAW Controller */}
            <div className="border-t border-zinc-800/90 pt-5 lg:border-t-0 lg:pt-0">
              <div className="mb-1.5">
                <h2 className="text-base lg:text-[17px] leading-snug font-semibold tracking-tight text-zinc-100">
                  推論エンジンと接続先DAW
                </h2>
                <p className="mt-0.5 text-xs text-zinc-400 leading-snug max-w-xl [@media(min-width:1024px)_and_(max-height:820px)]:hidden">
                  ローカル環境で実行するMORTMモデルと、出力先のDAWコントローラーを選択します。
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-5">
                <div className="min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <label htmlFor="compass-model-select" className="text-xs font-medium text-zinc-300">
                      使用モデル
                    </label>
                    <span className="font-mono text-xs text-zinc-400">
                      {hardwareInfo ? `${safeHwDevice} / ${safeHwDtype}` : 'LOCAL ENGINE'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="relative flex-1 min-w-0">
                      <select
                        ref={modelSelectRef}
                        id="compass-model-select"
                        value={safeModelName}
                        onChange={(e) => onUpdateSettings({ modelName: e.target.value })}
                        className="min-h-[44px] w-full appearance-none truncate rounded-xl border border-[#82828B] bg-studio-900/90 pl-3.5 pr-9 text-sm text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                      >
                        {safeModels.map((m, idx) => (
                          <option key={`${idx}-${m.model_name}`} value={m.model_name}>
                            {m.model_name}
                            {m.has_weights ? ' (重み検出済)' : ' (未配置・検証可)'}
                          </option>
                        ))}
                      </select>
                      <ChevronDown
                        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                        strokeWidth={1.75}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        const shouldHandoffFocus =
                          typeof document !== 'undefined' &&
                          document.activeElement === e.currentTarget &&
                          !isPointerClickEvent(e);
                        onRescanModels();
                        if (shouldHandoffFocus) {
                          modelSelectRef.current?.focus();
                        }
                      }}
                      disabled={isRescanningModels || isGenerating}
                      aria-label="モデルフォルダを再スキャン"
                      aria-description={
                        isRescanningModels
                          ? 'モデルフォルダを再スキャン中です'
                          : isGenerating
                            ? '生成処理中は再スキャンできません'
                            : undefined
                      }
                      data-tooltip={
                        isRescanningModels
                          ? 'モデルフォルダを再スキャン中です'
                          : isGenerating
                            ? '生成処理中は再スキャンできません'
                            : 'モデルフォルダを再スキャン'
                      }
                      data-tooltip-when-disabled="true"
                      data-tooltip-align="end"
                      className="inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-full border border-[#82828B] bg-studio-900/90 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 disabled:border-[#82828B]/45 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors motion-reduce:transition-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                    >
                      <RefreshCw
                        className={`h-4 w-4 ${isRescanningModels ? 'animate-spin motion-reduce:animate-none' : ''}`}
                        strokeWidth={1.75}
                      />
                    </button>
                  </div>
                  <p className="mt-1 text-xs text-zinc-400 truncate">
                    {selectedModelMeta?.has_weights
                      ? '重みファイル検出済み'
                      : '重み未配置 (初期重みで動作検証可)'}
                    {hardwareInfo && safeHwUserDir ? ` · 配置先: ${safeHwUserDir}` : ''}
                  </p>
                </div>

                <div className="min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <label htmlFor="compass-daw-select" className="text-xs font-medium text-zinc-300">
                      DAWコントローラー (C層アダプター)
                    </label>
                    <span className="font-mono text-xs text-zinc-400">
                      {safeIsConnected ? 'CONNECTED' : 'STANDBY'}
                    </span>
                  </div>
                  <div className="relative min-w-0">
                    <select
                      id="compass-daw-select"
                      value={safeControllerId}
                      disabled={Boolean(isGenerating)}
                      onChange={(e) => onSwitchController(e.target.value)}
                      className="min-h-[44px] w-full appearance-none truncate rounded-xl border border-[#82828B] bg-studio-900/90 pl-3.5 pr-9 text-sm text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 disabled:opacity-50"
                    >
                      {safeControllers.map((ctrl, idx) => (
                        <option key={`${idx}-${ctrl.id}`} value={ctrl.id}>
                          {typeof ctrl.displayName === 'string' && ctrl.displayName
                            ? ctrl.displayName
                            : ctrl.id}
                        </option>
                      ))}
                    </select>
                    <ChevronDown
                      className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                      strokeWidth={1.75}
                    />
                  </div>
                  <p className="mt-1 text-xs text-zinc-400 truncate">
                    {safeConnectionStatusText}
                  </p>
                </div>
              </div>
            </div>

            {/* Section 2: Sampling & CoT Parameters */}
            <div className="border-t border-zinc-800/90 pt-2.5 [@media(min-height:860px)]:pt-3">
              <div className="mb-1.5">
                <h2 className="text-base lg:text-[17px] leading-snug font-semibold tracking-tight text-zinc-100">
                  サンプリングと推論パラメータ
                </h2>
                <p className="mt-0.5 text-xs text-zinc-400 leading-snug max-w-xl [@media(min-width:1024px)_and_(max-height:820px)]:hidden">
                  温度（Temperature）や核サンプリング（Top-P）を調整し、表現の多様性を制御します。
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-5 gap-y-2">
                {/* Temperature */}
                <div>
                  <div className="flex items-center justify-between gap-2 mb-0.5">
                    <label htmlFor="slider-temperature" className="text-xs font-medium text-zinc-300">
                      Temperature (温度)
                    </label>
                    <span className="font-mono text-xs text-zinc-100 tabular-nums">
                      {safeTemperature.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex min-h-[44px] items-center">
                    <input
                      id="slider-temperature"
                      type="range"
                      min={0.1}
                      max={2.0}
                      step={0.05}
                      value={safeTemperature}
                      onChange={(e) => {
                        const next = parseFloat(e.target.value);
                        if (Number.isFinite(next)) {
                          onUpdateSettings({ temperature: Math.max(0.1, Math.min(2.0, next)) });
                        }
                      }}
                      className="compass-range-slider min-h-[44px] w-full cursor-pointer rounded-full bg-transparent focus:outline-none"
                    />
                  </div>
                  <div className="flex justify-between gap-2 text-xs leading-none text-zinc-400 font-mono">
                    <span>0.10 (厳密・安定的)</span>
                    <span>2.00 (大胆・実験的)</span>
                  </div>
                </div>

                {/* Top-P */}
                <div>
                  <div className="flex items-center justify-between gap-2 mb-0.5">
                    <label htmlFor="slider-top-p" className="text-xs font-medium text-zinc-300">
                      Top-P (核サンプリング閾値)
                    </label>
                    <span className="font-mono text-xs text-zinc-100 tabular-nums">
                      {safeTopP.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex min-h-[44px] items-center">
                    <input
                      id="slider-top-p"
                      type="range"
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      value={safeTopP}
                      onChange={(e) => {
                        const next = parseFloat(e.target.value);
                        if (Number.isFinite(next)) {
                          onUpdateSettings({ topP: Math.max(0.1, Math.min(1.0, next)) });
                        }
                      }}
                      className="compass-range-slider min-h-[44px] w-full cursor-pointer rounded-full bg-transparent focus:outline-none"
                    />
                  </div>
                  <div className="flex justify-between gap-2 text-xs leading-none text-zinc-400 font-mono">
                    <span>0.10 (高確率語のみ)</span>
                    <span>0.95 (標準)</span>
                  </div>
                </div>

                {/* Chain-of-Thought Thinking Toggle + CoT Temperature */}
                <div>
                  <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 mb-1">
                    <span className="text-xs font-medium text-zinc-300">
                      Chain-of-Thought 構造推論 (Thinking)
                    </span>
                    <span className="shrink-0 font-mono text-xs text-zinc-400">
                      {cotStatusBadge}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      disabled={!isSftGenModel || isCotBypassedByMetaTask}
                      aria-pressed={isCotEffectivelyActive}
                      aria-description={
                        !isSftGenModel
                          ? 'SFT-Genモデル選択時に使用できます'
                          : isCotBypassedByMetaTask
                            ? '白紙からの生成 (meta) では構造推論は自動省略されます'
                            : undefined
                      }
                      data-tooltip={
                        !isSftGenModel
                          ? 'SFT-Genモデル選択時に使用できます'
                          : isCotBypassedByMetaTask
                            ? '白紙からの生成 (meta) では構造推論は自動省略されます'
                            : undefined
                      }
                      data-tooltip-when-disabled="true"
                      onClick={() => onUpdateSettings({ thinking: !safeThinking })}
                      className={`min-h-[44px] rounded-full px-4 text-xs font-medium border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                        isCotEffectivelyActive
                          ? 'border-sky-400 bg-sky-950/40 text-sky-200'
                          : 'border-[#82828B] bg-studio-900/90 text-zinc-200 hover:border-zinc-300'
                      } disabled:border-[#82828B]/40 disabled:text-zinc-500 disabled:cursor-not-allowed`}
                    >
                      {isCotBypassedByMetaTask
                        ? '構造推論 自動省略'
                        : safeThinking && isSftGenModel
                          ? '構造推論 ON'
                          : '構造推論 OFF'}
                    </button>

                    <div className="flex-1 min-w-[140px]">
                      <div className="flex justify-between gap-2 text-xs leading-none text-zinc-400 font-mono">
                        <label htmlFor="slider-cot-temp">CoT温度</label>
                        <span>{safeCotTemperature.toFixed(2)}</span>
                      </div>
                      <input
                        id="slider-cot-temp"
                        type="range"
                        min={0.1}
                        max={1.5}
                        step={0.05}
                        disabled={!isCotEffectivelyActive}
                        value={safeCotTemperature}
                        onChange={(e) => {
                          const next = parseFloat(e.target.value);
                          if (Number.isFinite(next)) {
                            onUpdateSettings({
                              cotTemperature: Math.max(0.1, Math.min(1.5, next)),
                            });
                          }
                        }}
                        className="compass-range-slider min-h-[44px] w-full cursor-pointer rounded-full bg-transparent focus:outline-none disabled:opacity-30"
                      />
                    </div>
                  </div>
                </div>

                {/* Number of Candidates & Polyphony Mode */}
                <div>
                  <span className="block text-xs font-medium text-zinc-300 mb-1">
                    同時生成候補数 / 単旋律制約
                  </span>
                  <div className="flex flex-wrap items-center gap-2">
                    <div
                      role="group"
                      aria-label="同時生成候補数"
                      className="inline-flex flex-wrap items-center gap-2"
                    >
                      {[1, 2, 3, 4].map((count) => {
                        const isSelected = safeNumCandidates === count;
                        return (
                          <button
                            key={count}
                            type="button"
                            aria-pressed={isSelected}
                            aria-label={`同時生成候補数 ${count}件`}
                            data-tooltip={`同時生成候補数: ${count}件`}
                            onClick={() => onUpdateSettings({ numCandidates: count })}
                            className={`min-h-[44px] min-w-[44px] rounded-full px-3 font-mono text-xs font-medium border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                              isSelected
                                ? 'border-sky-400 bg-sky-950/40 text-sky-200'
                                : 'border-[#82828B] bg-studio-900/90 text-zinc-200 hover:border-zinc-300'
                            }`}
                          >
                            {count}
                          </button>
                        );
                      })}
                    </div>

                    <button
                      type="button"
                      aria-pressed={safeForceMonophonic}
                      onClick={() => onUpdateSettings({ forceMonophonic: !safeForceMonophonic })}
                      className={`min-h-[44px] rounded-full px-3.5 text-xs font-medium border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                        safeForceMonophonic
                          ? 'border-sky-400 bg-sky-950/40 text-sky-200'
                          : 'border-[#82828B] bg-studio-900/90 text-zinc-300 hover:border-zinc-300 hover:text-zinc-100'
                      }`}
                    >
                      単旋律 (リード) 優先
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Section 3: Expandable SFT Musical Structure Directives */}
            <div
              ref={structureSectionRef}
              className="border-t border-zinc-800/90 pt-2.5 [@media(min-height:860px)]:pt-3"
            >
              <button
                ref={structureToggleBtnRef}
                type="button"
                onClick={() => setShowStructureOptions((prev) => !prev)}
                aria-expanded={showStructureOptions}
                aria-controls="sft-structure-options-panel"
                className="flex min-h-[44px] w-full items-center justify-between rounded-xl border border-[#82828B] bg-studio-900/60 px-4 text-left text-sm font-medium text-zinc-200 hover:border-zinc-300 hover:bg-zinc-900/70 hover:text-zinc-100 transition-colors motion-reduce:transition-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
              >
                <span>音楽構造・ジャンル・タスクの詳細指定 (SFT条件付け)</span>
                <ChevronDown
                  className={`h-4 w-4 shrink-0 text-zinc-400 transition-transform motion-reduce:transition-none ${
                    showStructureOptions ? 'rotate-180' : ''
                  }`}
                  strokeWidth={1.75}
                />
              </button>

              <div
                id="sft-structure-options-panel"
                hidden={!showStructureOptions}
                className={showStructureOptions ? 'mt-2 space-y-2' : 'hidden'}
              >
                {showStructureOptions && (
                  <>
                  {isFoundationModel && (
                    <p className="text-xs text-zinc-400 leading-snug">
                      Foundation モデルは自由生成モードのため、以下の条件指定は省略されます。
                    </p>
                  )}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-5 gap-y-2">
                    <div>
                      <label htmlFor="sft-task-select" className="block text-xs font-medium text-zinc-300 mb-0.5">
                        生成タスクモード
                      </label>
                      <div className="relative min-w-0">
                        <select
                          id="sft-task-select"
                          value={safeTaskMode}
                          disabled={isFoundationModel}
                          onChange={(e) => {
                            if (e.target.value in TASK_LABELS) {
                              onUpdateSettings({ taskMode: e.target.value as SftTaskMode });
                            }
                          }}
                          className="min-h-[44px] w-full appearance-none truncate rounded-xl border border-[#82828B] bg-studio-900/90 pl-3.5 pr-9 text-sm text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 disabled:opacity-50"
                        >
                          {(Object.keys(TASK_LABELS) as SftTaskMode[]).map((mode) => (
                            <option key={mode} value={mode}>
                              {TASK_LABELS[mode]}
                            </option>
                          ))}
                        </select>
                        <ChevronDown
                          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                          strokeWidth={1.75}
                        />
                      </div>
                    </div>

                    <div>
                      <label htmlFor="cond-track-select" className="block text-xs font-medium text-zinc-300 mb-0.5">
                        参照する伴奏・条件トラック (inst_comp用)
                      </label>
                      <div className="relative min-w-0">
                        <select
                          id="cond-track-select"
                          value={safeCondTrackId}
                          disabled={isFoundationModel}
                          onChange={(e) => onUpdateSettings({ conditionTrackId: e.target.value })}
                          className="min-h-[44px] w-full appearance-none truncate rounded-xl border border-[#82828B] bg-studio-900/90 pl-3.5 pr-9 text-sm text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 disabled:opacity-50"
                        >
                          <option value="">指定なし</option>
                          {safeTracks
                            .filter((t) => t.id !== safeActiveTrackId)
                            .map((t, idx) => (
                              <option key={`${idx}-${t.id}`} value={t.id}>
                                {typeof t.name === 'string' && t.name ? t.name : t.id} (
                                {typeof t.instrument === 'string' && t.instrument
                                  ? t.instrument
                                  : 'PIANO'}
                                )
                              </option>
                            ))}
                        </select>
                        <ChevronDown
                          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                          strokeWidth={1.75}
                        />
                      </div>
                    </div>

                    <div>
                      <label htmlFor="note-density-select" className="block text-xs font-medium text-zinc-300 mb-0.5">
                        音符密度 (Note Density: 1〜10)
                      </label>
                      <div className="relative min-w-0">
                        <select
                          id="note-density-select"
                          value={safeNoteDensity}
                          disabled={isFoundationModel}
                          onChange={(e) => {
                            if (e.target.value === '') {
                              onUpdateSettings({ noteDensity: null });
                              return;
                            }
                            const parsed = Number(e.target.value);
                            if (Number.isInteger(parsed) && parsed >= 1 && parsed <= 10) {
                              onUpdateSettings({ noteDensity: parsed });
                            }
                          }}
                          className="min-h-[44px] w-full appearance-none truncate rounded-xl border border-[#82828B] bg-studio-900/90 pl-3.5 pr-9 text-sm text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 disabled:opacity-50"
                        >
                          <option value="">自動決定 (CoT推論または標準密度)</option>
                          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((d) => (
                            <option key={d} value={d}>
                              密度レベル {d} {d <= 3 ? '(疎・ゆったり)' : d >= 8 ? '(高密度・技巧的)' : '(標準)'}
                            </option>
                          ))}
                        </select>
                        <ChevronDown
                          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                          strokeWidth={1.75}
                        />
                      </div>
                    </div>

                    <div>
                      <label htmlFor="musical-key-select" className="block text-xs font-medium text-zinc-300 mb-0.5">
                        調号指定 (Key)
                      </label>
                      <div className="relative min-w-0">
                        <select
                          id="musical-key-select"
                          value={safeMusicalKey}
                          disabled={isFoundationModel}
                          onChange={(e) => onUpdateSettings({ musicalKey: e.target.value })}
                          className="min-h-[44px] w-full appearance-none truncate rounded-xl border border-[#82828B] bg-studio-900/90 pl-3.5 pr-9 text-sm text-zinc-100 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 disabled:opacity-50"
                        >
                          <option value="">自動判定 (文脈またはCoT推論)</option>
                          {ALL_MUSICAL_KEYS.map((k) => (
                            <option key={k} value={k}>
                              {k}
                            </option>
                          ))}
                        </select>
                        <ChevronDown
                          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
                          strokeWidth={1.75}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Genre Filter & Chips */}
                  <div>
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-1.5">
                      <div className="flex flex-wrap items-center justify-between sm:justify-start gap-2">
                        <label htmlFor="genre-filter-input" className="text-xs font-medium text-zinc-300">
                          ジャンルタグ (
                          {isSftGenModel
                            ? `最大3件まで選択可 — 現在 ${safeSelectedGenres.length}/3`
                            : 'SFT-Gen モデル選択時に有効'}
                          )
                        </label>
                        {safeSelectedGenres.length > 0 && (
                          <button
                            type="button"
                            onClick={(e) => {
                              const shouldHandoffFocus =
                                typeof document !== 'undefined' &&
                                document.activeElement === e.currentTarget &&
                                !isPointerClickEvent(e);
                              onUpdateSettings({ selectedGenres: [] });
                              if (!shouldHandoffFocus) {
                                return;
                              }
                              const sec = structureSectionRef.current;
                              const filterInput = sec?.querySelector('#genre-filter-input');
                              if (filterInput instanceof HTMLInputElement && !filterInput.disabled) {
                                filterInput.focus();
                                return;
                              }
                              const keySelect = sec?.querySelector('#musical-key-select');
                              if (keySelect instanceof HTMLSelectElement && !keySelect.disabled) {
                                keySelect.focus();
                                return;
                              }
                              structureToggleBtnRef.current?.focus();
                            }}
                            className="min-h-[44px] rounded-full border border-[#82828B] bg-studio-900/80 px-3.5 text-xs text-zinc-300 hover:border-zinc-300 hover:bg-zinc-900/60 hover:text-zinc-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
                          >
                            選択をクリア
                          </button>
                        )}
                      </div>
                      <input
                        id="genre-filter-input"
                        type="text"
                        disabled={!isSftGenModel}
                        value={genreFilter}
                        onChange={(e) => setGenreFilter(e.target.value)}
                        placeholder="ジャンルを検索 (例: jazz, ambient)"
                        className="min-h-[44px] w-full sm:w-72 rounded-xl border border-[#82828B] bg-studio-900/90 px-3.5 text-sm text-zinc-100 placeholder:text-zinc-400 focus:border-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 disabled:opacity-40"
                      />
                    </div>
                    <div
                      data-testid="genre-chips-scroll-tray"
                      className="flex flex-wrap gap-2 max-h-32 lg:max-h-[76px] [@media(min-height:860px)]:lg:max-h-[104px] [@media(min-height:900px)]:lg:max-h-[124px] [@media(min-height:1020px)]:lg:max-h-52 overflow-y-auto p-0.5 pr-1.5"
                    >
                      {filteredGenres.length === 0 ? (
                        <p
                          role="status"
                          data-testid="genre-empty-filter-status"
                          className="py-1 text-xs text-zinc-400"
                        >
                          一致するジャンルタグがありません
                        </p>
                      ) : (
                        filteredGenres.map((genre) => {
                          const active = isSftGenModel && safeSelectedGenres.includes(genre);
                          return (
                            <button
                              key={genre}
                              type="button"
                              disabled={!isSftGenModel}
                              aria-pressed={active}
                              onClick={() => toggleGenre(genre)}
                              className={`min-h-[44px] min-w-[44px] rounded-full px-3.5 font-mono text-xs border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                                active
                                  ? 'border-sky-400 bg-sky-950/40 text-sky-200 font-medium'
                                  : 'border-[#82828B] bg-studio-900/90 text-zinc-300 hover:border-zinc-300 hover:text-zinc-100'
                              } disabled:opacity-40`}
                            >
                              {genre}
                            </button>
                          );
                        })
                      )}
                    </div>
                  </div>
                  </>
                )}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
