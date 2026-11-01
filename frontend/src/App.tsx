/**
 * COMPASS AI Root Chromium Workspace Application (`App.tsx`).
 * Connects View (`V`: BootSplashScreen, ControlPanelView, PianoRollView),
 * Pluggable DAW Controller Registry (`C`: DemoPianoRollController, Cubase, Ableton),
 * and Local MORTM 4.5 Inference Engine Client (`M`: CompassApiClient).
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { WebAudioEngine } from './audio/WebAudioEngine';
import { AuroraBackdrop } from './components/AuroraBackdrop';
import { BootLogEntry, BootSplashScreen } from './components/BootSplashScreen';
import { CompassLogo } from './components/CompassLogo';
import { ControlPanelSettings, ControlPanelView } from './components/ControlPanelView';
import { PianoRollView } from './components/PianoRollView';
import { DawControllerRegistry } from './controllers/DawControllerRegistry';
import { DemoPianoRollController } from './controllers/DemoPianoRollController';
import {
  AbletonDawController,
  CubaseDawController,
} from './controllers/ExternalDawAdapters';
import {
  DawSessionSnapshot,
  IDawController,
  isInteractivePianoRollController,
  SftTaskMode,
} from './controllers/IDawController';
import { evaluateGenerationMeter } from './domain/MeasureRange';
import {
  ALL_MUSICAL_KEYS,
  ALL_SFT_GENRES,
  CompassApiClient,
  HardwareStatusInfo,
  ModelMetadata,
} from './services/CompassApiClient';
import { WorkspaceSyncBus, WorkspaceTabId } from './services/WorkspaceSyncBus';
import { useAdaptiveTooltipAlignment } from './utils/useAdaptiveTooltipAlignment';

const MAX_GENERATION_LOGS = 40;
const VALID_TASK_MODES: ReadonlySet<string> = new Set<SftTaskMode>([
  'auto',
  'meta',
  'meta_past',
  'meta_future',
  'infill',
  'inst_comp',
]);

const FALLBACK_MODELS: readonly ModelMetadata[] = [
  {
    model_name: 'MORTM4.5D-80M-SFT-Gen',
    description: 'MORTM 4.5 Dense SFT-Gen (ローカル実行・CoT対応)',
    has_weights: false,
    tag: { model: 'sft_gen', type: 'mortm', version: '4.5' },
    rule: { input_midi: true },
  },
  {
    model_name: 'MORTM4.5E-A80M-E64-SFT-Gen',
    description: 'MORTM 4.5 MoE SFT-Gen (64 Experts・ローカル実行)',
    has_weights: false,
    tag: { model: 'sft_gen', type: 'mortm', version: '4.5' },
    rule: { input_midi: true },
  },
  {
    model_name: 'MORTM4.5D-80M',
    description: 'MORTM 4.5 Foundation (事前学習ベースモデル)',
    has_weights: false,
    tag: { model: 'foundation', type: 'mortm', version: '4.5' },
    rule: { input_midi: false },
  },
];

function nowTimeCode(): string {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

function getInitialTabFromUrl(): WorkspaceTabId {
  if (typeof window === 'undefined') return 'compass-ai';
  const params = new URLSearchParams(window.location.search);
  return params.get('popout') === '1' || params.get('tab') === 'pianoroll'
    ? 'pianoroll'
    : 'compass-ai';
}

const NON_TEXT_INPUT_TYPES = new Set([
  'button',
  'checkbox',
  'radio',
  'range',
  'file',
  'submit',
  'reset',
  'color',
]);

function isTextEntryElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable || target.closest('[contenteditable="true"]')) return true;
  const field = target.closest('input, textarea');
  if (!field) return false;
  if (field instanceof HTMLInputElement) {
    return !NON_TEXT_INPUT_TYPES.has((field.type || 'text').toLowerCase());
  }
  return true;
}

export const App: React.FC = () => {
  // Singleton Infrastructure & Controller Registry (`DIP` / `OCP`)
  const services = useMemo(() => {
    const audioEngine = new WebAudioEngine();
    const apiClient = new CompassApiClient();
    const registry = new DawControllerRegistry();
    const demoController = new DemoPianoRollController(audioEngine);
    registry.register(demoController);
    registry.register(new CubaseDawController());
    registry.register(new AbletonDawController());
    const syncBus = new WorkspaceSyncBus();
    return { audioEngine, apiClient, registry, demoController, syncBus };
  }, []);

  useAdaptiveTooltipAlignment();

  const [isSecondaryPopout, setIsSecondaryPopout] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return new URLSearchParams(window.location.search).get('popout') === '1';
  });

  const [isBooting, setIsBooting] = useState<boolean>(() => !isSecondaryPopout);
  const [bootProgress, setBootProgress] = useState<number>(8);
  const [bootAction, setBootAction] = useState<string>(
    'COMPASS AI ワークスペースを起動しています...',
  );
  const [bootLogs, setBootLogs] = useState<BootLogEntry[]>(() => [
    {
      id: 1,
      timestamp: nowTimeCode(),
      message: 'Chromium UI カーネルと WebAudio シンセサイザーを初期化中...',
    },
  ]);

  const [activeTab, setActiveTab] = useState<WorkspaceTabId>(getInitialTabFromUrl);
  const [pianoRollFocusRevision, setPianoRollFocusRevision] = useState<number>(0);
  const [candidateFocusRevision, setCandidateFocusRevision] = useState<number>(0);
  const [isPianoRollModalOpen, setIsPianoRollModalOpen] = useState<boolean>(false);

  const [activeController, setActiveController] = useState<IDawController>(() =>
    services.registry.getActive(),
  );
  const [session, setSession] = useState<DawSessionSnapshot>(() =>
    services.registry.getActive().getSnapshot(),
  );

  const [models, setModels] = useState<readonly ModelMetadata[]>(FALLBACK_MODELS);
  const [hardwareInfo, setHardwareInfo] = useState<HardwareStatusInfo | null>(null);
  const [isRescanningModels, setIsRescanningModels] = useState<boolean>(false);

  const [settings, setSettings] = useState<ControlPanelSettings>({
    modelName: 'MORTM4.5D-80M-SFT-Gen',
    temperature: 1.0,
    topP: 0.95,
    thinking: true,
    cotTemperature: 0.8,
    numCandidates: 1,
    taskMode: 'auto',
    conditionTrackId: '',
    forceMonophonic: false,
    noteDensity: null,
    musicalKey: '',
    selectedGenres: [],
  });

  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [statusTitle, setStatusTitle] = useState<string>('READY');
  const [statusDetail, setStatusDetail] = useState<string>(
    '設定を確認し「生成を実行」を押すとローカル推論を開始します。',
  );
  const [generationLogs, setGenerationLogs] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const genAbortRef = useRef<AbortController | null>(null);
  const genRunIdRef = useRef<number>(0);
  const isGeneratingRef = useRef<boolean>(false);
  const statusDetailRef = useRef<string>(statusDetail);
  const isSecondaryPopoutRef = useRef<boolean>(isSecondaryPopout);
  isGeneratingRef.current = isGenerating;
  statusDetailRef.current = statusDetail;
  isSecondaryPopoutRef.current = isSecondaryPopout;
  const logSeqRef = useRef<number>(2);

  const appendBootLog = useCallback((message: string, progress?: number) => {
    const id = logSeqRef.current++;
    setBootAction(message);
    if (typeof progress === 'number') {
      setBootProgress(progress);
    }
    setBootLogs((prev) => [...prev.slice(-29), { id, timestamp: nowTimeCode(), message }]);
  }, []);

  // Subscribe to Active Controller & Session Snapshot changes
  useEffect(() => {
    const unsubRegistry = services.registry.subscribeActiveChange((nextCtrl) => {
      setActiveController(nextCtrl);
      setSession(nextCtrl.getSnapshot());
    });
    return unsubRegistry;
  }, [services.registry]);

  useEffect(() => {
    setSession(activeController.getSnapshot());
    const unsubSession = activeController.subscribe((snap) => {
      setSession(snap);
    });
    return unsubSession;
  }, [activeController]);

  // Clean up singleton resources on unmount & clear peer generation lock on Main reload/close (`[R7-MED-02]`)
  useEffect(() => {
    const notifyIdleOnUnload = () => {
      if (!isSecondaryPopoutRef.current && isGeneratingRef.current) {
        services.syncBus.publishGenerationStatus(false, '生成処理を中断しました。');
      }
    };
    if (typeof window !== 'undefined') {
      window.addEventListener('pagehide', notifyIdleOnUnload);
      window.addEventListener('beforeunload', notifyIdleOnUnload);
    }
    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('pagehide', notifyIdleOnUnload);
        window.removeEventListener('beforeunload', notifyIdleOnUnload);
      }
      notifyIdleOnUnload();
      genAbortRef.current?.abort();
      services.demoController.pause();
      services.syncBus.dispose();
      void services.audioEngine.dispose();
    };
  }, [services]);

  // Boot Sequence: Initializes Controllers + Connects to Local MORTM FastAPI Server
  const bootStartedAsPopoutRef = useRef<boolean>(isSecondaryPopout);
  const backendInitializedRef = useRef<boolean>(false);

  useEffect(() => {
    let cancelled = false;

    const connectToBackendOnce = async (emitSplashLogs: boolean): Promise<boolean> => {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1200);
      try {
        const hw = await services.apiClient.getHardwareInfo(ctrl.signal);
        clearTimeout(timer);
        if (cancelled) return false;

        setHardwareInfo(hw);
        if (emitSplashLogs) {
          const safeBootDev =
            typeof hw?.device === 'string' && hw.device.trim()
              ? hw.device.toUpperCase()
              : 'CPU';
          const safeBootDtype =
            typeof hw?.dtype === 'string' && hw.dtype.trim() ? hw.dtype : 'float32';
          appendBootLog(
            `推論バックエンド検出: ${safeBootDev} (${safeBootDtype}) — モデルフォルダ確認完了`,
            76,
          );
        }

        const modelList = await services.apiClient.getModelInfo();
        if (!cancelled && modelList.length > 0) {
          setModels(modelList);
          if (!backendInitializedRef.current) {
            const preferred =
              modelList.find((m) => m.has_weights && m.model_name.includes('SFT')) ??
              modelList.find((m) => m.has_weights) ??
              modelList.find((m) => m.model_name === 'MORTM4.5D-80M-SFT-Gen') ??
              modelList[0];
            setSettings((prev) => ({ ...prev, modelName: preferred.model_name }));
          }
          if (emitSplashLogs) {
            appendBootLog(
              `利用可能なモデル定義 (${modelList.length}件) をロードしました。`,
              94,
            );
          }
        }
        backendInitializedRef.current = true;
        return true;
      } catch {
        clearTimeout(timer);
        return false;
      }
    };

    const unsubDesktopBoot = window.compassDesktop?.onBootLog?.((line) => {
      if (!cancelled) {
        appendBootLog(line);
      }
    });
    const unsubApiConfig = window.compassDesktop?.onApiConfigChanged?.((cfg) => {
      if (cfg.apiBaseUrl) {
        services.apiClient.setBaseUrl(cfg.apiBaseUrl);
      }
      if (typeof cfg.handshakeToken === 'string') {
        services.apiClient.setHandshakeToken(cfg.handshakeToken);
      }
      if (typeof cfg.backendReady === 'boolean') {
        services.apiClient.setDesktopBackendReady(cfg.backendReady);
      }
      if (!cancelled && (cfg.handshakeToken || cfg.backendReady)) {
        void connectToBackendOnce(false);
      }
    });

    const syncDesktopApiConfig = async () => {
      if (window.compassDesktop?.requestApiConfig) {
        const cached = window.compassDesktop.getApiConfig?.();
        const cfg =
          cached?.backendReady || cached?.handshakeToken
            ? cached
            : await window.compassDesktop.requestApiConfig();
        if (!cancelled && cfg) {
          if (cfg.apiBaseUrl) {
            services.apiClient.setBaseUrl(cfg.apiBaseUrl);
          }
          if (typeof cfg.handshakeToken === 'string') {
            services.apiClient.setHandshakeToken(cfg.handshakeToken);
          }
          if (typeof cfg.backendReady === 'boolean') {
            services.apiClient.setDesktopBackendReady(cfg.backendReady);
          }
        }
      }
    };

    const initialBootActivate =
      services.registry.getActive().id === 'demo-pianoroll'
        ? services.registry.activate('demo-pianoroll')
        : null;

    const runBoot = async () => {
      try {
        await syncDesktopApiConfig();
        if (initialBootActivate) {
          await initialBootActivate;
        }
        if (cancelled) return;
        appendBootLog(
          'DAWコントローラー (Demo Studio / Cubase / Ableton) を登録しました。',
          28,
        );

        await sleep(100);
        if (cancelled) return;
        await syncDesktopApiConfig();
        appendBootLog(
          `ローカル推論エンジン (${services.apiClient.baseUrl}) の稼働状態を確認中...`,
          52,
        );

        const maxAttempts = window.compassDesktop ? 8 : 2;
        let connected = false;

        for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
          if (cancelled) return;
          await syncDesktopApiConfig();
          if (await connectToBackendOnce(true)) {
            connected = true;
            break;
          }
          if (attempt < maxAttempts) {
            await sleep(350);
          }
        }

        if (!connected && !cancelled) {
          appendBootLog(
            'ローカル推論サーバー未接続 — スタンドアロンUI検証モードで起動します。',
            92,
          );
        }

        await sleep(240);
        if (!cancelled) {
          appendBootLog('COMPASS AI とピアノロールの準備が完了しました。', 100);
          await sleep(180);
          if (!cancelled) {
            setIsBooting(false);
          }
        }
      } catch {
        if (!cancelled) {
          setIsBooting(false);
        }
      }
    };

    if (bootStartedAsPopoutRef.current) {
      // Pop-out tab activates its initial controller synchronously in Effect 1 (`seq = 1` before Effect 2's
      // `SYNC_ACTIVE_CONTROLLER`) and fetches authenticated config + metadata silently in background (`[R14-MED-03]`)
      void initialBootActivate?.catch(() => {});
      void (async () => {
        await syncDesktopApiConfig();
        if (!cancelled) {
          await connectToBackendOnce(false);
        }
      })();
    } else {
      void runBoot();
    }

    return () => {
      cancelled = true;
      unsubDesktopBoot?.();
      unsubApiConfig?.();
    };
  }, [appendBootLog, services]);

  const broadcastCurrentProject = useCallback(
    (forceEvenWithoutPeerFlag?: unknown) => {
      const forceDesktopIpc = forceEvenWithoutPeerFlag === true;
      if (
        !forceDesktopIpc &&
        typeof window !== 'undefined' &&
        window.compassDesktop?.hasPeerWindows?.() === false
      ) {
        return;
      }
      const ctrl = services.registry.getActive();
      if (!isInteractivePianoRollController(ctrl)) return;
      const ws = ctrl.exportTrustedWorkspaceState
        ? ctrl.exportTrustedWorkspaceState()
        : ctrl.exportWorkspaceState();
      services.syncBus.publishWorkspaceState(ws, forceDesktopIpc ? { forceDesktopIpc: true } : undefined);
    },
    [services.registry, services.syncBus],
  );

  // Background cold-boot reconnection poller when Python/PyTorch takes > ~10s to finish warming up
  useEffect(() => {
    if (isBooting || hardwareInfo !== null) {
      return;
    }
    let cancelled = false;
    let activeAbort: AbortController | null = null;
    let attempts = 0;

    const timer = window.setInterval(() => {
      if (cancelled || attempts >= 60) {
        window.clearInterval(timer);
        return;
      }
      attempts += 1;
      activeAbort?.abort();
      const ctrl = new AbortController();
      activeAbort = ctrl;
      const reqTimeout = window.setTimeout(() => ctrl.abort(), 2200);

      void (async () => {
        try {
          if (
            typeof window !== 'undefined' &&
            !services.apiClient.handshakeToken &&
            !services.apiClient.desktopBackendReady &&
            window.compassDesktop?.requestApiConfig
          ) {
            const refreshed = await window.compassDesktop.requestApiConfig();
            if (refreshed?.apiBaseUrl) {
              services.apiClient.setBaseUrl(refreshed.apiBaseUrl);
            }
            if (typeof refreshed?.handshakeToken === 'string') {
              services.apiClient.setHandshakeToken(refreshed.handshakeToken);
            }
            if (typeof refreshed?.backendReady === 'boolean') {
              services.apiClient.setDesktopBackendReady(refreshed.backendReady);
            }
          }
          const hw = await services.apiClient.getHardwareInfo(ctrl.signal);
          window.clearTimeout(reqTimeout);
          if (cancelled) return;
          setHardwareInfo(hw);
          const modelList = await services.apiClient.getModelInfo();
          if (!cancelled && modelList.length > 0) {
            setModels(modelList);
            const preferred =
              modelList.find((m) => m.has_weights && m.model_name.includes('SFT')) ??
              modelList.find((m) => m.has_weights) ??
              null;
            if (preferred && !backendInitializedRef.current) {
              setSettings((prev) => ({ ...prev, modelName: preferred.model_name }));
            }
            backendInitializedRef.current = true;
          }
          window.clearInterval(timer);
        } catch {
          window.clearTimeout(reqTimeout);
        }
      })();
    }, 2500);

    return () => {
      cancelled = true;
      activeAbort?.abort();
      window.clearInterval(timer);
    };
  }, [hardwareInfo, isBooting, services.apiClient]);

  // Cross-Tab & Electron Foreground Synchronization + Pop-Out State Handshake
  const pendingInboundFocusTabRef = useRef<WorkspaceTabId | null>(null);
  const [focusTabSyncSeq, setFocusTabSyncSeq] = useState(0);
  const switchReqSeqRef = useRef(0);

  const handleCancelGenerate = useCallback(() => {
    genRunIdRef.current += 1;
    const ctrl = genAbortRef.current;
    genAbortRef.current = null;
    ctrl?.abort();
    if (isGeneratingRef.current) {
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setStatusTitle('CANCELED');
      setStatusDetail('生成処理を中断しました。');
      services.syncBus.publishGenerationStatus(false, '生成処理を中断しました。');
    }
  }, [services.syncBus]);

  const bringCompassAiToForeground = useCallback(
    (reason: string) => {
      if (!isSecondaryPopout) {
        pendingInboundFocusTabRef.current = 'compass-ai';
        setFocusTabSyncSeq((s) => s + 1);
        setActiveTab('compass-ai');
        services.syncBus.publishFocusTab('compass-ai', reason);
        return;
      }
      services.syncBus.publishFocusTab('compass-ai', reason);
      if (typeof window !== 'undefined' && !window.compassDesktop?.bringWindowToFront) {
        let focusedOpener = false;
        try {
          if (
            window.opener &&
            !window.opener.closed &&
            typeof window.opener.focus === 'function'
          ) {
            window.opener.focus();
            focusedOpener = true;
          }
        } catch {
          focusedOpener = false;
        }
        if (!focusedOpener) {
          pendingInboundFocusTabRef.current = 'compass-ai';
          setFocusTabSyncSeq((s) => s + 1);
          setIsSecondaryPopout(false);
          setActiveTab('compass-ai');
          if (isGeneratingRef.current && !genAbortRef.current) {
            isGeneratingRef.current = false;
            setIsGenerating(false);
            setStatusTitle('READY');
            setStatusDetail('設定を確認し「生成を実行」を押すとローカル推論を開始します。');
          }
          try {
            const url = new URL(window.location.href);
            url.searchParams.delete('popout');
            url.searchParams.set('tab', 'compass-ai');
            window.history.replaceState(null, '', url.toString());
          } catch {
            // Ignore URL history update errors
          }
        }
      }
    },
    [isSecondaryPopout, services.syncBus],
  );

  const bringPianoRollToForeground = useCallback(
    (reason: string) => {
      if (!isSecondaryPopout) {
        pendingInboundFocusTabRef.current = 'pianoroll';
        setFocusTabSyncSeq((s) => s + 1);
      }
      setActiveTab('pianoroll');
      if (reason === 'generation-complete') {
        setCandidateFocusRevision((r) => r + 1);
      } else {
        setPianoRollFocusRevision((r) => r + 1);
      }
      services.syncBus.publishFocusTab('pianoroll', reason);
    },
    [isSecondaryPopout, services.syncBus],
  );

  useEffect(() => {
    const unsub = services.syncBus.subscribe((msg) => {
      if (msg.type === 'FOCUS_TAB') {
        if (isSecondaryPopout && msg.tab === 'compass-ai') {
          return;
        }
        pendingInboundFocusTabRef.current = msg.tab;
        setFocusTabSyncSeq((s) => s + 1);
        setActiveTab(msg.tab);
        if (msg.tab === 'pianoroll') {
          if (msg.reason === 'generation-complete') {
            setCandidateFocusRevision((r) => r + 1);
          } else if (!(isSecondaryPopout && msg.reason === 'desktop-ipc')) {
            setPianoRollFocusRevision((r) => r + 1);
          }
        }
        if (
          typeof window !== 'undefined' &&
          (typeof navigator === 'undefined' || !/jsdom/i.test(navigator.userAgent ?? ''))
        ) {
          try {
            window.focus();
          } catch {
            // Ignore focus restrictions
          }
        }
      } else if (msg.type === 'GENERATION_STATUS') {
        setIsGenerating(msg.isGenerating);
        setStatusDetail(msg.statusDetail);
      } else if (msg.type === 'REQUEST_WORKSPACE_STATE') {
        if (!isSecondaryPopout) {
          const demoCtrl = services.demoController;
          const ws = demoCtrl.exportTrustedWorkspaceState
            ? demoCtrl.exportTrustedWorkspaceState()
            : demoCtrl.exportWorkspaceState();
          services.syncBus.publishWorkspaceState(ws, { forceDesktopIpc: true });
          services.syncBus.publishGenerationStatus(
            isGeneratingRef.current,
            statusDetailRef.current,
          );
          services.syncBus.publishActiveController(services.registry.getActive().id, {
            forceDesktopIpc: true,
          });
        }
      } else if (msg.type === 'SYNC_ACTIVE_CONTROLLER') {
        if (
          services.registry.getActive().id !== msg.controllerId ||
          services.registry.getPendingId() !== msg.controllerId
        ) {
          const reqSeq = ++switchReqSeqRef.current;
          handleCancelGenerate();
          services.registry
            .activate(msg.controllerId)
            .then((active) => {
              if (reqSeq === switchReqSeqRef.current && active.id === msg.controllerId) {
                setErrorMessage(null);
              }
            })
            .catch((err: unknown) => {
              if (reqSeq !== switchReqSeqRef.current) {
                return;
              }
              const detail =
                err instanceof Error && err.message.trim()
                  ? err.message
                  : 'DAWコントローラーの切り替えに失敗しました。';
              setErrorMessage(detail);
              try {
                const settledId = services.registry.getActive().id;
                if (settledId === 'demo-pianoroll' && msg.controllerId !== 'demo-pianoroll') {
                  services.syncBus.publishActiveController(settledId, { forceDesktopIpc: true });
                }
              } catch {
                // Ignore empty registry edge case
              }
            });
        }
      } else if (msg.type === 'SYNC_WORKSPACE_STATE') {
        const demoCtrl = services.demoController;
        const prevSnap = demoCtrl.getSnapshot();
        const prevFirstRaw = prevSnap.candidates[0]?.rawMidiBytes ?? null;
        const prevCandIdx = prevSnap.activeCandidateIndex;
        const prevCandRange = prevSnap.activeCandidateRange ?? null;
        demoCtrl.applyWorkspaceState(msg.workspaceState);
        if (services.registry.getActive() === demoCtrl) {
          const nextSnap = demoCtrl.getSnapshot();
          const nextFirstRaw = nextSnap.candidates[0]?.rawMidiBytes ?? null;
          const nextCandRange = nextSnap.activeCandidateRange ?? null;
          const firstCandChanged =
            (prevFirstRaw === null) !== (nextFirstRaw === null) ||
            (prevFirstRaw !== null &&
              nextFirstRaw !== null &&
              (prevFirstRaw.byteLength !== nextFirstRaw.byteLength ||
                prevFirstRaw.some((b, i) => b !== nextFirstRaw[i])));
          const candRangeChanged =
            (prevCandRange === null) !== (nextCandRange === null) ||
            (prevCandRange !== null &&
              nextCandRange !== null &&
              (prevCandRange.trackId !== nextCandRange.trackId ||
                prevCandRange.startTick !== nextCandRange.startTick ||
                prevCandRange.endTick !== nextCandRange.endTick));
          if (
            nextCandRange !== null &&
            (firstCandChanged || nextSnap.activeCandidateIndex !== prevCandIdx || candRangeChanged)
          ) {
            setCandidateFocusRevision((r) => r + 1);
          }
        }
      }
    });

    if (isSecondaryPopout) {
      services.syncBus.requestWorkspaceState();
    } else {
      const demoCtrl = services.demoController;
      const ws = demoCtrl.exportTrustedWorkspaceState
        ? demoCtrl.exportTrustedWorkspaceState()
        : demoCtrl.exportWorkspaceState();
      services.syncBus.publishWorkspaceState(ws, { forceDesktopIpc: true });
      if (!isGeneratingRef.current) {
        services.syncBus.publishGenerationStatus(false, statusDetailRef.current);
      }
      services.syncBus.publishActiveController(services.registry.getActive().id, {
        forceDesktopIpc: true,
      });
    }

    return unsub;
  }, [broadcastCurrentProject, handleCancelGenerate, isSecondaryPopout, services.demoController, services.registry, services.syncBus]);

  useEffect(() => {
    const targetTab = pendingInboundFocusTabRef.current;
    if (!targetTab || targetTab !== activeTab) return;
    pendingInboundFocusTabRef.current = null;
    if (targetTab === 'compass-ai') {
      const panel = document.getElementById('workspace-panel-compass-ai');
      const activeEl = document.activeElement;
      if (!activeEl || activeEl === document.body || !panel?.contains(activeEl)) {
        document.getElementById('workspace-tab-compass-ai')?.focus();
      }
    } else if (targetTab === 'pianoroll' && !isSecondaryPopout) {
      const panel = document.getElementById('workspace-panel-pianoroll');
      const activeEl = document.activeElement;
      if (!activeEl || activeEl === document.body || !panel?.contains(activeEl)) {
        document.getElementById('workspace-tab-pianoroll')?.focus();
      }
    }
  }, [activeTab, focusTabSyncSeq, isPianoRollModalOpen, isSecondaryPopout]);

  // Global `Shift + G` Shortcut -> Bring COMPASS AI (Circle + Control Panel) to Foreground
  // Excluded ONLY when typing inside a text-entry control (`input`, `textarea`, `contenteditable`)
  // so focused `<button>` / `<select>` / `[role="tab"]` controls never trap `Shift + G` (`[R2-HIGH-01]`, `[R3-MED-03]`).
  useEffect(() => {
    const onGlobalKeyDown = (e: KeyboardEvent) => {
      if (e.isComposing || e.keyCode === 229) return;
      if (!e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
      if (e.key !== 'G' && e.key !== 'g' && e.code !== 'KeyG') return;
      if (document.querySelector('[role="dialog"]')) return;
      if (isTextEntryElement(e.target)) return;

      e.preventDefault();
      bringCompassAiToForeground('shortcut-shift-g');
    };

    window.addEventListener('keydown', onGlobalKeyDown);
    return () => window.removeEventListener('keydown', onGlobalKeyDown);
  }, [bringCompassAiToForeground]);

  const handleRescanModels = useCallback(async () => {
    setIsRescanningModels(true);
    setErrorMessage(null);
    try {
      const [refreshedModels, hw] = await Promise.all([
        services.apiClient.rescanModels(),
        services.apiClient.getHardwareInfo(),
      ]);
      setHardwareInfo(hw);
      if (Array.isArray(refreshedModels) && refreshedModels.length > 0) {
        setModels(refreshedModels);
      }
      const detectedWeightsCount = Array.isArray(hw?.models_with_weights)
        ? hw.models_with_weights.length
        : 0;
      setStatusDetail(
        `モデルフォルダを再スキャンしました (重み検出: ${detectedWeightsCount}件)。`,
      );
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : 'モデルフォルダの再スキャンに失敗しました。',
      );
    } finally {
      setIsRescanningModels(false);
    }
  }, [services.apiClient]);

  // Execute Local MORTM Generation -> Animate Circle + Stream Telemetry -> Auto-Switch to Piano Roll Foreground
  const handleGenerate = useCallback(async () => {
    const meterEval = evaluateGenerationMeter(
      session.timeSigNumerator,
      session.timeSigDenominator,
    );
    if (!meterEval.supported) {
      const msg =
        meterEval.reason ??
        `現在の拍子 (${meterEval.label}) は生成でサポートされていません。`;
      setErrorMessage(msg);
      setStatusTitle('STANDBY');
      setStatusDetail(msg);
      return;
    }

    const myRunId = (genRunIdRef.current += 1);
    genAbortRef.current?.abort();
    const abortCtrl = new AbortController();
    genAbortRef.current = abortCtrl;

    setErrorMessage(null);
    isGeneratingRef.current = true;
    setIsGenerating(true);
    setStatusTitle('GENERATING');

    const pushGenLog = (line: string) => {
      if (
        abortCtrl.signal.aborted ||
        genAbortRef.current !== abortCtrl ||
        genRunIdRef.current !== myRunId
      ) {
        return;
      }
      setStatusDetail(line);
      setGenerationLogs((prev) => [
        ...prev.slice(-(MAX_GENERATION_LOGS - 1)),
        `${nowTimeCode()}  ${line}`,
      ]);
      services.syncBus.publishGenerationStatus(true, line);
    };

    const startedAt = performance.now();

    try {
      pushGenLog('対象トラックと小節範囲から前後コンテキストMIDIを抽出中...');
      const safeTaskOverride: SftTaskMode =
        typeof settings.taskMode === 'string' && VALID_TASK_MODES.has(settings.taskMode)
          ? settings.taskMode
          : 'auto';
      const liveSnap = activeController.getSnapshot();
      const liveTracks = Array.isArray(liveSnap?.tracks)
        ? liveSnap.tracks.filter(
            (t) => Boolean(t && typeof t === 'object' && typeof t.id === 'string'),
          )
        : [];
      const liveActiveTrackId =
        typeof liveSnap?.activeTrackId === 'string' &&
        liveTracks.some((t) => t.id === liveSnap.activeTrackId)
          ? liveSnap.activeTrackId
          : (liveTracks[0]?.id ?? '');
      const safeCondTrackId =
        typeof settings.conditionTrackId === 'string' &&
        settings.conditionTrackId.trim() &&
        settings.conditionTrackId !== liveActiveTrackId &&
        liveTracks.some((t) => t.id === settings.conditionTrackId)
          ? settings.conditionTrackId
          : null;
      const extracted = await activeController.extractGenerationContext({
        taskOverride: safeTaskOverride,
        conditionTrackId: safeCondTrackId,
        forceMonophonic: Boolean(settings.forceMonophonic),
      });

      if (
        abortCtrl.signal.aborted ||
        services.registry.getActive() !== activeController ||
        services.registry.getPendingId() !== activeController.id
      ) {
        if (genAbortRef.current === abortCtrl && genRunIdRef.current === myRunId) {
          isGeneratingRef.current = false;
          setIsGenerating(false);
          setStatusTitle('CANCELED');
          setStatusDetail('生成処理を中断しました。');
          services.syncBus.publishGenerationStatus(false, '生成処理を中断しました。');
        }
        return;
      }

      const safeDensityVal =
        typeof settings.noteDensity === 'number' &&
        Number.isInteger(settings.noteDensity) &&
        settings.noteDensity >= 1 &&
        settings.noteDensity <= 10
          ? settings.noteDensity
          : null;
      const densityPayload =
        safeDensityVal !== null ? { [extracted.targetMortmInstrument]: safeDensityVal } : null;

      const safeTempo =
        typeof session.bpm === 'number' && Number.isFinite(session.bpm)
          ? Math.max(20, Math.min(300, Math.round(session.bpm)))
          : 120;
      const safeNumGems =
        typeof settings.numCandidates === 'number' &&
        Number.isFinite(settings.numCandidates) &&
        [1, 2, 3, 4].includes(Math.round(settings.numCandidates))
          ? Math.round(settings.numCandidates)
          : 1;
      const safeTopP =
        typeof settings.topP === 'number' && Number.isFinite(settings.topP)
          ? Math.max(0.1, Math.min(1.0, settings.topP))
          : 0.95;
      const safeTemp =
        typeof settings.temperature === 'number' && Number.isFinite(settings.temperature)
          ? Math.max(0.1, Math.min(2.0, settings.temperature))
          : 1.0;
      const safeCotTemp =
        typeof settings.cotTemperature === 'number' && Number.isFinite(settings.cotTemperature)
          ? Math.max(0.1, Math.min(1.5, settings.cotTemperature))
          : 0.8;
      const safeMeasureCount = extracted.targetMeasureCount;

      const safeModelNameStr =
        typeof settings.modelName === 'string' && settings.modelName.trim()
          ? settings.modelName
          : (models[0]?.model_name ?? 'MORTM4.5D-80M-SFT-Gen');
      const selectedModelMeta = models.find((m) => m.model_name === safeModelNameStr) ?? null;
      const safeMetaTagModel =
        typeof selectedModelMeta?.tag?.model === 'string'
          ? selectedModelMeta.tag.model.toLowerCase()
          : '';
      const isSftGen =
        safeMetaTagModel === 'sft_gen' ||
        (!safeMetaTagModel && safeModelNameStr.toLowerCase().includes('sft'));
      const safeGenres = Array.isArray(settings.selectedGenres)
        ? settings.selectedGenres
            .filter(
              (g): g is string =>
                typeof g === 'string' && (ALL_SFT_GENRES as readonly string[]).includes(g),
            )
            .slice(0, 3)
        : [];
      const safeMusicalKey =
        typeof settings.musicalKey === 'string' &&
        (ALL_MUSICAL_KEYS as readonly string[]).includes(settings.musicalKey)
          ? settings.musicalKey
          : null;
      const safeTsNum =
        typeof session.timeSigNumerator === 'number' &&
        Number.isFinite(session.timeSigNumerator) &&
        session.timeSigNumerator >= 1
          ? Math.min(64, Math.round(session.timeSigNumerator))
          : 4;
      const safeTsDen =
        typeof session.timeSigDenominator === 'number' &&
        Number.isFinite(session.timeSigDenominator) &&
        session.timeSigDenominator >= 1
          ? Math.min(64, Math.round(session.timeSigDenominator))
          : 4;

      const result = await services.apiClient.generateMusic(
        {
          meta: {
            model_type: safeModelNameStr,
            task: extracted.resolvedTask,
            program: [extracted.targetMortmInstrument],
            tempo: safeTempo,
            num_gems: safeNumGems,
            genfield_measure: safeMeasureCount,
            gen_note_dense: densityPayload,
            p: safeTopP,
            temperature: safeTemp,
            cot_temperature: safeCotTemp,
            thinking:
              isSftGen && extracted.resolvedTask !== 'meta' ? Boolean(settings.thinking) : false,
            genre: isSftGen && safeGenres.length > 0 ? safeGenres : null,
            key: safeMusicalKey,
            time_signature_numerator: safeTsNum,
            time_signature_denominator: safeTsDen,
          },
          pastMidiBytes: extracted.pastMidiBytes,
          conditionsMidiBytes: extracted.conditionsMidiBytes,
          futureMidiBytes: extracted.futureMidiBytes,
        },
        (progressMsg) => {
          pushGenLog(progressMsg);
        },
        abortCtrl.signal,
      );

      // Ensure smooth visual feedback on the central circle even on ultra-fast local runs
      const elapsed = performance.now() - startedAt;
      if (elapsed < 380) {
        await sleep(380 - elapsed);
      }

      if (
        abortCtrl.signal.aborted ||
        services.registry.getActive() !== activeController ||
        services.registry.getPendingId() !== activeController.id
      ) {
        if (genAbortRef.current === abortCtrl && genRunIdRef.current === myRunId) {
          isGeneratingRef.current = false;
          setIsGenerating(false);
          setStatusTitle('CANCELED');
          setStatusDetail('生成処理を中断しました。');
          services.syncBus.publishGenerationStatus(false, '生成処理を中断しました。');
        }
        return;
      }

      if (
        extracted.targetSlot &&
        !activeController
          .getSnapshot()
          .tracks.some((t) => t.id === extracted.targetSlot!.trackId)
      ) {
        if (genAbortRef.current === abortCtrl && genRunIdRef.current === myRunId) {
          const skippedMsg =
            '生成対象のトラックが変更または削除されたため、配置をスキップしました。';
          isGeneratingRef.current = false;
          setIsGenerating(false);
          setStatusTitle('CANCELED');
          setStatusDetail(skippedMsg);
          setGenerationLogs((prev) => [
            ...prev.slice(-(MAX_GENERATION_LOGS - 1)),
            `${nowTimeCode()}  ${skippedMsg}`,
          ]);
          services.syncBus.publishGenerationStatus(false, skippedMsg);
        }
        return;
      }

      if (genAbortRef.current !== abortCtrl || genRunIdRef.current !== myRunId) return;

      await activeController.applyGeneratedCandidates(
        result.candidates,
        result.reasons,
        extracted.targetSlot,
      );
      broadcastCurrentProject();

      if (genAbortRef.current !== abortCtrl || genRunIdRef.current !== myRunId) return;

      const firstTracks = result.candidates[0]?.parsedMidi?.tracks;
      const placedNoteCount = Array.isArray(firstTracks)
        ? firstTracks.reduce(
            (acc, t) => acc + (Array.isArray(t?.notes) ? t.notes.length : 0),
            0,
          )
        : 0;

      const hasEmbeddedPr = Boolean(
        activeController.getSnapshot().capabilities.hasEmbeddedPianoRoll,
      );
      const doneMsg = hasEmbeddedPr
        ? `生成完了 (${result.candidates.length}候補 / ${placedNoteCount}音) — ピアノロールを最前面に表示します。`
        : `生成完了 (${result.candidates.length}候補 / ${placedNoteCount}音) — 外部DAWブリッジへ配置しました。`;
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setStatusTitle('COMPLETED');
      setStatusDetail(doneMsg);
      setGenerationLogs((prev) => [
        ...prev.slice(-(MAX_GENERATION_LOGS - 1)),
        `${nowTimeCode()}  ${doneMsg}`,
      ]);
      services.syncBus.publishGenerationStatus(false, doneMsg);

      // Bring Embedded Piano Roll to foreground (including `?popout=1` when embedded),
      // or switch local tab to External DAW Bridge without hijacking `?popout=1` (`[R7-HIGH-03]`).
      if (hasEmbeddedPr) {
        bringPianoRollToForeground('generation-complete');
      } else {
        if (!isSecondaryPopout) {
          pendingInboundFocusTabRef.current = 'pianoroll';
          setFocusTabSyncSeq((s) => s + 1);
        }
        setActiveTab('pianoroll');
      }
    } catch (err) {
      if (abortCtrl.signal.aborted) {
        if (genAbortRef.current === abortCtrl && genRunIdRef.current === myRunId) {
          isGeneratingRef.current = false;
          setIsGenerating(false);
          setStatusTitle('CANCELED');
          setStatusDetail('生成処理を中断しました。');
          services.syncBus.publishGenerationStatus(false, '生成処理を中断しました。');
        }
        return;
      }
      if (genAbortRef.current !== abortCtrl || genRunIdRef.current !== myRunId) return;
      const message =
        err instanceof Error ? err.message : 'ローカル推論中に予期しないエラーが発生しました。';
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setStatusTitle('ERROR');
      setStatusDetail('生成処理を完了できませんでした。設定またはモデル配置を確認してください。');
      setErrorMessage(message);
      services.syncBus.publishGenerationStatus(false, message);
    }
  }, [
    activeController,
    bringPianoRollToForeground,
    broadcastCurrentProject,
    isSecondaryPopout,
    models,
    services.apiClient,
    services.registry,
    services.syncBus,
    session.bpm,
    session.timeSigDenominator,
    session.timeSigNumerator,
    settings,
  ]);

  const handleSwitchController = useCallback(
    (id: string) => {
      const reqSeq = ++switchReqSeqRef.current;
      handleCancelGenerate();
      services.registry
        .activate(id)
        .then((active) => {
          if (reqSeq !== switchReqSeqRef.current) {
            return;
          }
          if (active.id === id) {
            setErrorMessage(null);
            services.syncBus.publishActiveController(active.id, { forceDesktopIpc: true });
          }
        })
        .catch((err: unknown) => {
          if (reqSeq !== switchReqSeqRef.current) {
            return;
          }
          const detail =
            err instanceof Error && err.message.trim()
              ? err.message
              : 'DAWコントローラーの切り替えに失敗しました。';
          setErrorMessage(detail);
          try {
            services.syncBus.publishActiveController(services.registry.getActive().id, {
              forceDesktopIpc: true,
            });
          } catch {
            // Ignore empty registry edge case
          }
        });
    },
    [handleCancelGenerate, services.registry, services.syncBus],
  );

  const handleOpenSeparatePianoRollWindow = () => {
    if (!activeController.getSnapshot().capabilities.hasEmbeddedPianoRoll) {
      return;
    }
    if (window.compassDesktop?.openPianoRollWindow) {
      setActiveTab('compass-ai');
      void Promise.resolve(window.compassDesktop.openPianoRollWindow()).catch(() => {});
      return;
    }
    const url = new URL(window.location.href);
    url.searchParams.set('tab', 'pianoroll');
    url.searchParams.set('popout', '1');
    const popup = window.open(url.toString(), 'compass_ai_pianoroll_tab');
    if (popup) {
      setActiveTab('compass-ai');
    }
  };

  const totalNoteCount = useMemo(
    () =>
      Array.isArray(session?.tracks)
        ? session.tracks.reduce(
            (acc, t) => acc + (Array.isArray(t?.notes) ? t.notes.length : 0),
            0,
          )
        : 0,
    [session?.tracks],
  );

  // Sync status title with live audio playback when idle
  const effectiveStatusTitle = isGenerating
    ? 'GENERATING'
    : session.isPlaying
      ? 'AUDIO ACTIVE'
      : statusTitle;

  const safeLiveBpm =
    typeof session?.bpm === 'number' && Number.isFinite(session.bpm)
      ? Math.max(20, Math.min(300, Math.round(session.bpm * 10) / 10))
      : 120;

  const effectiveStatusDetail =
    !isGenerating && session.isPlaying
      ? `再生中 (${safeLiveBpm} BPM) — 出力波形に連動して中央リングが振動しています。`
      : statusDetail;

  const isModalBlockingWorkspace =
    (isSecondaryPopout || activeTab === 'pianoroll') && isPianoRollModalOpen;

  if (isBooting) {
    return (
      <BootSplashScreen
        currentAction={bootAction}
        progressPercent={bootProgress}
        logs={bootLogs}
      />
    );
  }

  return (
    <div
      className={`relative isolate min-h-dvh ${
        isSecondaryPopout || activeTab === 'pianoroll'
          ? '[@media(min-width:768px)_and_(min-height:520px)]:h-dvh [@media(min-width:768px)_and_(min-height:520px)]:overflow-hidden'
          : ''
      } w-full max-w-full overflow-x-clip bg-studio-950 text-zinc-100 flex flex-col`}
    >
      <AuroraBackdrop />

      {/* Top Chromium Studio Tab Strip (`COMPASS AI` | `ピアノロール`, or Clean Heading in Pop-Out Window) */}
      <header
        aria-hidden={isModalBlockingWorkspace ? 'true' : undefined}
        {...(isModalBlockingWorkspace ? { inert: '' } : {})}
        className={`static [@media(min-width:768px)_and_(min-height:520px)]:sticky top-0 z-40 ${
          isModalBlockingWorkspace ? 'pointer-events-none select-none' : ''
        } flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 border-b border-zinc-800/85 bg-studio-900/90 px-3 sm:px-5 py-1.5`}
      >
        <div className={`${!isSecondaryPopout ? 'w-full sm:w-auto ' : ''}flex items-center`}>
          <CompassLogo size="sm" />
        </div>

        {isSecondaryPopout ? (
          <div
            id="popout-pianoroll-heading"
            data-testid="popout-pianoroll-badge"
            className="flex items-center gap-2 border-l border-zinc-700 pl-3 py-1 sm:mr-auto text-sm font-semibold tracking-tight text-zinc-200 select-none"
          >
            <span>ピアノロール</span>
            {totalNoteCount > 0 && (
              <span className="rounded-full bg-sky-950/60 px-2 py-0.5 font-mono text-xs leading-none text-sky-300">
                {totalNoteCount}
              </span>
            )}
          </div>
        ) : (
          <div
            role="tablist"
            aria-label="COMPASS AI ワークスペースタブ"
            onKeyDown={(e) => {
              if (e.nativeEvent.isComposing || e.keyCode === 229) return;
              if (e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return;
              let nextTab: WorkspaceTabId | null = null;
              if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                nextTab = activeTab === 'compass-ai' ? 'pianoroll' : 'compass-ai';
              } else if (e.key === 'Home') {
                nextTab = 'compass-ai';
              } else if (e.key === 'End') {
                nextTab = 'pianoroll';
              }
              if (nextTab) {
                e.preventDefault();
                setActiveTab(nextTab);
                const targetEl = document.getElementById(
                  nextTab === 'compass-ai' ? 'workspace-tab-compass-ai' : 'workspace-tab-pianoroll',
                );
                targetEl?.focus();
              }
            }}
            className="flex grow sm:grow-0 min-w-0 sm:w-auto items-center gap-1 rounded-full border border-zinc-700 bg-zinc-950/90 p-1 sm:mr-auto"
          >
            <button
              type="button"
              role="tab"
              id="workspace-tab-compass-ai"
              data-testid="tab-compass-ai"
              tabIndex={activeTab === 'compass-ai' ? 0 : -1}
              aria-selected={activeTab === 'compass-ai'}
              aria-controls="workspace-panel-compass-ai"
              onClick={() => setActiveTab('compass-ai')}
              className={`min-h-[44px] min-w-[44px] grow sm:grow-0 whitespace-nowrap rounded-full border px-2 sm:px-4 text-xs sm:text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                activeTab === 'compass-ai'
                  ? 'border-sky-400 bg-zinc-800 text-zinc-100'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              COMPASS AI
            </button>

            <button
              type="button"
              role="tab"
              id="workspace-tab-pianoroll"
              data-testid="tab-pianoroll"
              tabIndex={activeTab === 'pianoroll' ? 0 : -1}
              aria-selected={activeTab === 'pianoroll'}
              aria-controls="workspace-panel-pianoroll"
              onClick={() => setActiveTab('pianoroll')}
              className={`min-h-[44px] min-w-[44px] grow sm:grow-0 whitespace-nowrap rounded-full border px-2 sm:px-4 text-xs sm:text-sm font-medium flex items-center justify-center gap-1.5 sm:gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70 ${
                activeTab === 'pianoroll'
                  ? 'border-sky-400 bg-zinc-800 text-zinc-100'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <span>ピアノロール</span>
              {totalNoteCount > 0 && (
                <span className="rounded-full bg-sky-950/60 px-2 py-0.5 font-mono text-xs leading-none text-sky-300">
                  {totalNoteCount}
                </span>
              )}
            </button>
          </div>
        )}

        <div
          className={`${
            isSecondaryPopout ? 'hidden sm:flex' : 'flex'
          } shrink-0 items-center gap-3 text-xs text-zinc-400`}
        >
          <span className="hidden sm:inline font-mono text-xs text-zinc-400">
            {hardwareInfo
              ? typeof hardwareInfo.device === 'string' && hardwareInfo.device.trim()
                ? hardwareInfo.device.toUpperCase()
                : 'CPU'
              : 'STANDALONE'}
          </span>
          {!isSecondaryPopout && (
            <button
              type="button"
              disabled={!session.capabilities.hasEmbeddedPianoRoll}
              onClick={handleOpenSeparatePianoRollWindow}
              aria-label="別ウィンドウでピアノロールを開く"
              aria-description={
                !session.capabilities.hasEmbeddedPianoRoll
                  ? '外部DAW接続中は別ウィンドウを開けません'
                  : undefined
              }
              data-tooltip={
                session.capabilities.hasEmbeddedPianoRoll
                  ? '別ウィンドウでピアノロールを開く'
                  : '外部DAW接続中は別ウィンドウを開けません'
              }
              data-tooltip-when-disabled="true"
              data-tooltip-sm-hidden={
                session.capabilities.hasEmbeddedPianoRoll ? 'true' : undefined
              }
              data-tooltip-align="end"
              className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center gap-1.5 rounded-full border border-[#82828B] bg-zinc-950/90 px-2.5 sm:px-3.5 text-zinc-200 hover:border-zinc-300 hover:text-zinc-100 disabled:border-[#82828B]/40 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/70"
            >
              <ExternalLink className="h-4 w-4 shrink-0" strokeWidth={1.75} />
              <span className="hidden sm:inline text-xs font-medium">別ウィンドウ</span>
            </button>
          )}
        </div>
      </header>

      {/* Foreground View Panels — Both remain mounted so state, canvas & audio stay instant */}
      <main
        className={`relative ${
          isModalBlockingWorkspace ? 'z-50' : 'z-10'
        } flex-1 [@media(min-width:768px)_and_(min-height:520px)]:min-h-0 flex flex-col w-full max-w-full`}
      >
        {!isSecondaryPopout && (
          <div
            id="workspace-panel-compass-ai"
            role="tabpanel"
            aria-labelledby="workspace-tab-compass-ai"
            hidden={activeTab !== 'compass-ai'}
            className={activeTab === 'compass-ai' ? 'block flex-1' : 'hidden'}
          >
            <ControlPanelView
              isActive={activeTab === 'compass-ai'}
              audioEngine={services.audioEngine}
              controllers={services.registry.listAll()}
              session={session}
              models={models}
              hardwareInfo={hardwareInfo}
              settings={settings}
              isGenerating={isGenerating}
              isRescanningModels={isRescanningModels}
              statusTitle={effectiveStatusTitle}
              statusDetail={effectiveStatusDetail}
              generationLogs={generationLogs}
              errorMessage={errorMessage}
              onUpdateSettings={(patch) => {
                if (!patch || typeof patch !== 'object') return;
                setSettings((prev) => ({
                  ...prev,
                  modelName:
                    patch.modelName !== undefined
                      ? typeof patch.modelName === 'string' && patch.modelName.trim()
                        ? patch.modelName
                        : prev.modelName
                      : prev.modelName,
                  temperature:
                    patch.temperature !== undefined
                      ? typeof patch.temperature === 'number' && Number.isFinite(patch.temperature)
                        ? Math.max(0.1, Math.min(2.0, patch.temperature))
                        : prev.temperature
                      : prev.temperature,
                  topP:
                    patch.topP !== undefined
                      ? typeof patch.topP === 'number' && Number.isFinite(patch.topP)
                        ? Math.max(0.1, Math.min(1.0, patch.topP))
                        : prev.topP
                      : prev.topP,
                  thinking:
                    patch.thinking !== undefined ? Boolean(patch.thinking) : prev.thinking,
                  cotTemperature:
                    patch.cotTemperature !== undefined
                      ? typeof patch.cotTemperature === 'number' &&
                        Number.isFinite(patch.cotTemperature)
                        ? Math.max(0.1, Math.min(1.5, patch.cotTemperature))
                        : prev.cotTemperature
                      : prev.cotTemperature,
                  numCandidates:
                    patch.numCandidates !== undefined
                      ? typeof patch.numCandidates === 'number' &&
                        Number.isFinite(patch.numCandidates) &&
                        [1, 2, 3, 4].includes(Math.round(patch.numCandidates))
                        ? Math.round(patch.numCandidates)
                        : prev.numCandidates
                      : prev.numCandidates,
                  taskMode:
                    patch.taskMode !== undefined
                      ? typeof patch.taskMode === 'string' && VALID_TASK_MODES.has(patch.taskMode)
                        ? patch.taskMode
                        : prev.taskMode
                      : prev.taskMode,
                  conditionTrackId:
                    patch.conditionTrackId !== undefined
                      ? typeof patch.conditionTrackId === 'string'
                        ? patch.conditionTrackId
                        : prev.conditionTrackId
                      : prev.conditionTrackId,
                  forceMonophonic:
                    patch.forceMonophonic !== undefined
                      ? Boolean(patch.forceMonophonic)
                      : prev.forceMonophonic,
                  noteDensity:
                    patch.noteDensity !== undefined
                      ? patch.noteDensity === null
                        ? null
                        : typeof patch.noteDensity === 'number' &&
                            Number.isInteger(patch.noteDensity) &&
                            patch.noteDensity >= 1 &&
                            patch.noteDensity <= 10
                          ? patch.noteDensity
                          : prev.noteDensity
                      : prev.noteDensity,
                  musicalKey:
                    patch.musicalKey !== undefined
                      ? patch.musicalKey === '' ||
                        (typeof patch.musicalKey === 'string' &&
                          (ALL_MUSICAL_KEYS as readonly string[]).includes(patch.musicalKey))
                        ? patch.musicalKey
                        : prev.musicalKey
                      : prev.musicalKey,
                  selectedGenres:
                    patch.selectedGenres !== undefined
                      ? Array.isArray(patch.selectedGenres)
                        ? patch.selectedGenres
                            .filter(
                              (g): g is string =>
                                typeof g === 'string' &&
                                (ALL_SFT_GENRES as readonly string[]).includes(g),
                            )
                            .slice(0, 3)
                        : prev.selectedGenres
                      : prev.selectedGenres,
                }));
              }}
              onSwitchController={handleSwitchController}
              onRescanModels={() => {
                void handleRescanModels();
              }}
              onTogglePlayback={() => {
                if (session.isPlaying) activeController.pause();
                else void activeController.play();
              }}
              onGenerate={() => {
                void handleGenerate();
              }}
              onCancelGenerate={handleCancelGenerate}
            />
          </div>
        )}

        <div
          id="workspace-panel-pianoroll"
          role={isSecondaryPopout ? 'region' : 'tabpanel'}
          aria-labelledby={isSecondaryPopout ? 'popout-pianoroll-heading' : 'workspace-tab-pianoroll'}
          hidden={!isSecondaryPopout && activeTab !== 'compass-ai' && activeTab !== 'pianoroll' ? true : !isSecondaryPopout && activeTab !== 'pianoroll'}
          className={
            isSecondaryPopout || activeTab === 'pianoroll'
              ? 'flex flex-1 [@media(min-width:768px)_and_(min-height:520px)]:min-h-0 flex-col'
              : 'hidden'
          }
        >
          <PianoRollView
            isActive={isSecondaryPopout || activeTab === 'pianoroll'}
            isSecondaryPopout={isSecondaryPopout}
            isGenerating={isGenerating}
            controller={activeController}
            session={session}
            focusRevision={pianoRollFocusRevision}
            candidateFocusRevision={candidateFocusRevision}
            onLaunchCompassAi={() => bringCompassAiToForeground('launch-button')}
            onSwitchToDemoController={() => handleSwitchController('demo-pianoroll')}
            onProjectMutated={broadcastCurrentProject}
            onModalOpenChange={setIsPianoRollModalOpen}
          />
        </div>
      </main>
    </div>
  );
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
