# COMPASS AI — System Concept & MVC Architecture

本ドキュメントは、記号的音楽基盤モデル **MORTM 4.5 Foundation (SFT)** をあらゆるローカル環境（NVIDIA GPU / Apple Silicon Mac / CPU）で高速実行し、Chromiumベースのフロントエンドおよび各種DAW（Cubase、Ableton Live等）とシームレスに連携する **COMPASS AI** の全体概念図です。

---

## 1. 全体概念図 (MVC + ワンクリックインストーラー統合構造)

エンジニアリング知識を持たないエンドユーザーがインストーラー1つで環境構築から起動まで完結でき、MVC（Model-View-Controller）の責務分離とSOLID原則（依存性逆転の原則・開放閉鎖の原則）に完全準拠した全体構造です。

```mermaid
flowchart TB
    subgraph Installer["Zero-Config Bootstrapper (One-Click Installer / Launcher)"]
        HWDetect["Hardware & OS Detector (Windows / macOS / GPU / MPS / CPU)"]
        EnvSetup["Isolated Runtime Provisioner (Embedded Python / Venv / Dependencies)"]
        ModelFetch["Model Weight & Config Manager (MORTM 4.5 Foundation / SFT)"]
        ProcessSup["Process Supervisor (Lifecycle / Port / Zombie Prevention)"]
        HWDetect --> EnvSetup --> ModelFetch --> ProcessSup
    end

    subgraph View["V: View Layer (Chromium Frontend)"]
        UI_Shell["Chromium Application Shell (Anti-AI Web Design / 90-10 Neutral Palette)"]
        UI_PianoRoll["Interactive Context & Piano Roll View (PAST / CONST / FUTURE)"]
        UI_MetaCtrl["SFT Condition Panel (Instrument / Density / Genre / Key / CoT Thinking)"]
        UI_Shell --> UI_PianoRoll
        UI_Shell --> UI_MetaCtrl
    end

    subgraph Controller["C: Controller Layer (Application Orchestrator & Pluggable DAW Adapters)"]
        AppCtrl["Application Controller (Use-Case Orchestration / IPC & HTTP Bridge)"]
        DAWPort["<<interface>> IDAWController (Abstract Port)"]
        DAWRegistry["DAWControllerRegistry & Factory"]
        CubaseAdapter["CubaseController (MIDI Remote API / VST / Mackie / File Sync)"]
        AbletonAdapter["AbletonController (AbletonOSC / Remote Script / MIDI Bridge)"]
        StandaloneAdapter["StandaloneMIDIController (Virtual MIDI Port / Standard SMF)"]

        AppCtrl --> DAWRegistry
        DAWRegistry --> DAWPort
        CubaseAdapter -.->|implements| DAWPort
        AbletonAdapter -.->|implements| DAWPort
        StandaloneAdapter -.->|implements| DAWPort
    end

    subgraph Model["M: Model Layer (Cross-Platform MORTM 4.5 SFT Local Engine)"]
        EngineFacade["MORTMInferenceEngine (ModelController / LRU & Memory Guard)"]
        WrapperFactory["ModelWrapperFactory & Registry (foundation / sft_gen / pretrained)"]
        ExecStrategy["<<interface>> IExecutionBackend (Device / Dtype / Attention Strategy)"]
        CudaBackend["CUDA Backend (BF16/FP16 + SDPA / FlashAttn / GroupedMoE)"]
        MpsBackend["Apple Silicon MPS Backend (FP16/FP32 + PyTorch SDPA + Vectorized MoE)"]
        CpuBackend["CPU Backend (FP32/BF16 + SDPA + Torch Compile / Quantized Linear)"]
        MORTMCore["MORTM 4.5 Foundation + SFT LoRA (RoPE + RMSNorm + KV Cache + 2-Stage CoT)"]

        EngineFacade --> WrapperFactory
        WrapperFactory --> MORTMCore
        MORTMCore --> ExecStrategy
        CudaBackend -.->|implements| ExecStrategy
        MpsBackend -.->|implements| ExecStrategy
        CpuBackend -.->|implements| ExecStrategy
    end

    ProcessSup ==>|Launches & Monitors| View
    ProcessSup ==>|Launches & Monitors| Controller
    ProcessSup ==>|Launches & Monitors| Model

    View <==>|Typed JSON / Binary MIDI Stream| AppCtrl
    AppCtrl <==>|Domain Request / Response| EngineFacade
    CubaseAdapter <==>|Bidirectional DAW Sync| ExternalCubase["Cubase"]
    AbletonAdapter <==>|Bidirectional DAW Sync| ExternalAbleton["Ableton Live"]
    StandaloneAdapter <==>|MIDI / SMF| ExternalOther["Other DAWs"]
```

---

## 2. Model (M): マルチプラットフォーム・高速推論エンジンの概念図

`flash-attn` や特定CUDAバージョンへのハードコード依存を完全に排除し、実行環境（CUDA / MPS / CPU）に応じて最適な計算カーネル・データ型（`bfloat16` / `float16` / `float32`）・KVキャッシュを注入する Strategy パターン設計です。

```mermaid
flowchart LR
    subgraph InputDomain["1. Domain Input (Value Objects)"]
        GenMeta["GenerationRequest (Task / Program / Density / Genre / Key / CoT)"]
        MidiCtx["MIDI Contexts (PAST_M / CONST_M / FUTURE_M)"]
    end

    subgraph PromptBuilder["2. SFT Prompt & Token Pipeline"]
        TaskRouter["SFT Task Strategy (meta / meta_past / meta_future / infill / inst_comp)"]
        SysBuilder["Partial / Full System Tag Builder + <thinking> Trigger"]
        Tokenizer["Metric-Oriented Tokenizer (Vocab 700 / 96th Subdivision Grid)"]
        TaskRouter --> SysBuilder --> Tokenizer
    end

    subgraph RuntimeStrategy["3. Hardware Abstraction & Execution Strategy"]
        DevicePolicy["RuntimePolicyResolver"]
        DevicePolicy -->|"NVIDIA GPU"| CudaExec["CUDA Strategy (BF16/FP16 + SDPA/Flash + Pre-allocated KV Cache)"]
        DevicePolicy -->|"Apple Silicon"| MpsExec["MPS Strategy (FP16/FP32 Safe Cast + Contiguous SDPA KV Cache)"]
        DevicePolicy -->|"No GPU (CPU)"| CpuExec["CPU Strategy (FP32/BF16 + Dynamic Int8/SDPA + Fast Top-P)"]
    end

    subgraph TwoStageDecode["4. Two-Stage Autoregressive KV-Cache Decoding"]
        Prefill["Phase 1: Parallel Prompt Prefill (Batched / Unpadded or Masked SDPA)"]
        CoTDecode["Phase 2a: CoT Meta Prediction (<SYSTEM>..<TAG_END> @ cot_temperature=0.1)"]
        MusicDecode["Phase 2b: Musical Token Generation (CONST_M @ temperature -> <TE>)"]
        Prefill --> CoTDecode --> MusicDecode
    end

    subgraph OutputDomain["5. Post-Processing & Output"]
        ReasonParser["CoT Reason Extractor (X-Generation-Reason JSON)"]
        DeConverter["MIDI De-Converter (PrettyMIDI / SMF Stream)"]
    end

    InputDomain --> PromptBuilder
    PromptBuilder --> RuntimeStrategy
    RuntimeStrategy --> TwoStageDecode
    TwoStageDecode --> OutputDomain
```

---

## 3. OOP クラス階層概念図 (SOLID原則・デザインパターン構造)

`oop-extensible-craft` プロトコルに基づき、条件分岐の乱立を排し、抽象インターフェース（DIP）とレジストリ/ファクトリ（OCP）で構成されるクラス設計です。

```mermaid
classDiagram
    class IHardwareBackend {
        <<interface>>
        +device() torch.device
        +compute_dtype() torch.dtype
        +autocast_context() ContextManager
        +scaled_dot_product_attn(q, k, v, mask, is_causal) Tensor
        +grouped_moe_forward(x, w13, w2, weights, indices) Tensor
        +empty_cache() void
    }

    class CudaHardwareBackend {
        +device() cuda
        +compute_dtype() bfloat16 / float16
    }

    class MpsHardwareBackend {
        +device() mps
        +compute_dtype() float16 / float32
    }

    class CpuHardwareBackend {
        +device() cpu
        +compute_dtype() float32 / bfloat16
    }

    IHardwareBackend <|.. CudaHardwareBackend
    IHardwareBackend <|.. MpsHardwareBackend
    IHardwareBackend <|.. CpuHardwareBackend

    class AbstractModelWrapper {
        <<abstract>>
        #model_info: ModelMetadata
        #backend: IHardwareBackend
        +preprocess(context: MidiContextBundle, params: GenerationParams)* PromptBatch
        +generate(batch: PromptBatch, params: GenerationParams)* RawGenerationResult
        +postprocess(result: RawGenerationResult, save_dir: Path)* GenerationArtifact
    }

    class MORTM45PretrainedWrapper {
        +preprocess()
        +generate()
        +postprocess()
    }

    class MORTM45FoundationWrapper {
        +preprocess()
        +generate()
        +postprocess()
    }

    class MORTM45SFTGenWrapper {
        -task_registry: SFTTaskStrategyRegistry
        +preprocess()
        +generate()
        +postprocess()
    }

    AbstractModelWrapper <|-- MORTM45PretrainedWrapper
    AbstractModelWrapper <|-- MORTM45FoundationWrapper
    AbstractModelWrapper <|-- MORTM45SFTGenWrapper
    AbstractModelWrapper --> IHardwareBackend : uses (DI)

    class ISFTTaskStrategy {
        <<interface>>
        +build_prompt(tokenizer, context, params) TokenSequence
    }

    class MetaOnlyTaskStrategy
    class MetaPastTaskStrategy
    class MetaFutureTaskStrategy
    class InfillTaskStrategy
    class InstCompTaskStrategy

    ISFTTaskStrategy <|.. MetaOnlyTaskStrategy
    ISFTTaskStrategy <|.. MetaPastTaskStrategy
    ISFTTaskStrategy <|.. MetaFutureTaskStrategy
    ISFTTaskStrategy <|.. InfillTaskStrategy
    ISFTTaskStrategy <|.. InstCompTaskStrategy
    MORTM45SFTGenWrapper --> ISFTTaskStrategy : delegates

    class IDAWController {
        <<interface>>
        +connect() bool
        +capture_context() MidiContextBundle
        +inject_generated_midi(artifact: GenerationArtifact, target_track: str) bool
        +sync_transport(tempo: int, key: str) void
    }

    class CubaseController
    class AbletonController
    class StandaloneDAWController

    IDAWController <|.. CubaseController
    IDAWController <|.. AbletonController
    IDAWController <|.. StandaloneDAWController
```

---

## 4. シーケンス概念図 (DAW連携〜2段階CoT推論〜DAW反映フロー)

```mermaid
sequenceDiagram
    autonumber
    actor User as User (Composer)
    participant View as V: Chromium UI
    participant Ctrl as C: App & DAW Controller
    participant Engine as M: MORTM 4.5 Local Engine
    participant DAW as External DAW (Cubase / Ableton)

    User->>View: Select Task (e.g., infill / inst_comp) & Click Generate
    View->>Ctrl: Request Generation (GenerationParams)
    Ctrl->>DAW: Capture Selected Regions (PAST / CONST / FUTURE MIDI)
    DAW-->>Ctrl: Return MIDI Context Streams
    Ctrl->>Engine: Invoke Local Inference (Context + Params)

    Note over Engine: 1. Hardware Strategy Selection (CUDA / MPS / CPU)<br/>2. Build SFT Prompt + <thinking> Trigger<br/>3. Prefill KV Cache
    Note over Engine: 4. CoT Phase (<SYSTEM>..<TAG_END> @ T=0.1)<br/>Predicts optimal Key / Density / Genre / Length
    Note over Engine: 5. Music Phase (CONST_M @ user T -> <TE>)<br/>Contiguous Pre-allocated KV Cache Step

    Engine-->>Ctrl: Generated MIDI + Reason Metadata (X-Generation-Reason)
    Ctrl->>DAW: Insert Generated MIDI into Target Track at Bar Position
    Ctrl-->>View: Update Piano Roll & Display CoT Reason
```

---

## 5. 実装モジュール構成 (MVC Layer Mapping)

- **Model (`M` — Local MORTM 4.5 Engine)**:
  - `mortm/models/mortm.py`, `mortm/models/modules/` (`backend.py`, `attention.py`, `layers.py`, `lora_layers.py`)
  - `models/mortm/mortm45d.py`, `models/mortm/mortm45.py`, `rapper.py`, `model.py`, `app.py`, `main.py`
- **Controller (`C` — Pluggable DAW Controller Layer)**:
  - `frontend/src/controllers/IDawController.ts`, `frontend/src/controllers/DawControllerRegistry.ts`
  - `frontend/src/controllers/DemoPianoRollController.ts`, `frontend/src/controllers/ExternalDawAdapters.ts`
  - `frontend/src/domain/` (`MidiNote.ts`, `MeasureRange.ts`, `DawTrack.ts`, `commands/`, `midi/StandardMidiCodec.ts`)
- **View (`V` — Chromium TypeScript + Tailwind CSS Studio UI)**:
  - `frontend/src/components/` (`CompassLogo.tsx`, `BootSplashScreen.tsx`, `ResonanceCircleCanvas.tsx`, `ControlPanelView.tsx`, `PianoRollView.tsx`, `MidiTrackImportModal.tsx`)
  - `frontend/src/App.tsx`, `frontend/src/services/WorkspaceSyncBus.ts`, `frontend/src/audio/WebAudioEngine.ts`
  - `frontend/electron/main.cjs`, `frontend/electron/preload.cjs`, `start_compass_ai.bat`
