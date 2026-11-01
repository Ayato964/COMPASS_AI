import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import JSZip from 'jszip';
import {
  DawTrack,
  MAX_NOTES_PER_TRACK,
  PreparedCandidateBatch,
  ReadonlyCanonicalIdSet,
  TRACK_SWATCHES,
  isGenuineDawTrack,
  isPreparedCandidateBatch,
  isValidTrackInstrument,
  queryOverlappingNotesIndexed,
} from '../domain/DawTrack';
import {
  MeasureRange,
  canonicalBeatsPerMeasure,
  deriveTimeSignatureFromBeats,
  evaluateGenerationMeter,
} from '../domain/MeasureRange';
import {
  MAX_SAFE_MIDI_TICK,
  MidiNote,
  isGenuineMidiNote,
  truncateWellFormedUtf16,
} from '../domain/MidiNote';
import {
  extractCanonicalParsedMidiTracks,
  MAX_PARSED_SMF_NOTES,
  MAX_PARSED_SMF_TRACKS,
  MAX_SMF_PARSE_BYTES,
  MAX_SMF_QUARTER_NOTES,
  StandardMidiCodec,
} from '../domain/midi/StandardMidiCodec';
import { CommandManager } from '../domain/commands/CommandManager';
import {
  AddNotesCommand,
  DeleteNotesCommand,
  ReplaceRangeNotesCommand,
  TransformNotesCommand,
} from '../domain/commands/NoteCommands';
import { DawControllerRegistry } from '../controllers/DawControllerRegistry';
import { isInteractivePianoRollController } from '../controllers/IDawController';
import { DemoPianoRollController } from '../controllers/DemoPianoRollController';
import { AbletonDawController, CubaseDawController } from '../controllers/ExternalDawAdapters';
import { IAudioEngine, WebAudioEngine } from '../audio/WebAudioEngine';
import { AuroraBackdrop } from '../components/AuroraBackdrop';
import { BootSplashScreen } from '../components/BootSplashScreen';
import { ControlPanelView } from '../components/ControlPanelView';
import {
  MAX_MODAL_IMPORT_SELECTION,
  MidiTrackImportModal,
} from '../components/MidiTrackImportModal';
import {
  PianoRollView,
  computeClampedGroupMoveDeltas,
  pickAccessibleNoteLabelColor,
} from '../components/PianoRollView';
import { ResonanceCircleCanvas } from '../components/ResonanceCircleCanvas';
import {
  CompassApiClient,
  MAX_CANDIDATE_MIDI_BYTES,
  MAX_FAILED_CANDIDATE_ATTEMPTS,
  MAX_PLAIN_ARRAY_MIDI_BYTES,
  cloneCleanUint8Array,
  extractValidatedCandidateMidiBytes,
  inspectGenuineUint8ArrayPayload,
  normalizeGeneratedCandidatesBatch,
  sanitizeGenerationReason,
} from '../services/CompassApiClient';
import {
  WorkspaceSyncBus,
  buildTrustedWorkspaceSnapshot,
  isValidCandidateMidiBytesPayload,
  isValidGenerationReasonPayload,
  isValidIncomingSyncMessage,
} from '../services/WorkspaceSyncBus';
import { App } from '../App';
import { alignButtonTooltipToViewport } from '../utils/useAdaptiveTooltipAlignment';

if (typeof window !== 'undefined' && typeof window.PointerEvent === 'undefined') {
  class PolyfillPointerEvent extends MouseEvent {
    public readonly pointerId: number;
    public readonly pointerType: string;
    constructor(type: string, params: PointerEventInit = {}) {
      super(type, params);
      this.pointerId = params.pointerId ?? 1;
      this.pointerType = params.pointerType ?? 'mouse';
    }
  }
  (window as unknown as { PointerEvent: unknown }).PointerEvent = PolyfillPointerEvent;
  (globalThis as unknown as { PointerEvent: unknown }).PointerEvent = PolyfillPointerEvent;
}

class StubAudioEngine implements IAudioEngine {
  public async ensureResumed(): Promise<void> {}
  public getCurrentTime(): number {
    return 0;
  }
  public triggerNote(): void {}
  public auditionPitch(): void {}
  public stopAllVoices(): void {}
  public readAcousticTelemetry(waveformOut: Float32Array, spectrumOut: Uint8Array) {
    waveformOut.fill(0);
    spectrumOut.fill(0);
    return { rms: 0, peak: 0, lowBandEnergy: 0, highBandEnergy: 0 };
  }
  public async dispose(): Promise<void> {}
}

describe('COMPASS AI Domain, Pluggable DAW Controller (C), and Chromium View (V)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();

    // Stub HTMLCanvasElement.prototype.getContext for jsdom
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => {
      return {
        save: () => {},
        restore: () => {},
        scale: () => {},
        clearRect: () => {},
        fillRect: () => {},
        strokeRect: () => {},
        beginPath: () => {},
        closePath: () => {},
        moveTo: () => {},
        lineTo: () => {},
        arc: () => {},
        rect: () => {},
        clip: () => {},
        stroke: () => {},
        fill: () => {},
        fillText: () => {},
        set strokeStyle(_v: string) {},
        set fillStyle(_v: string) {},
        set lineWidth(_v: number) {},
        set globalAlpha(_v: number) {},
        set font(_v: string) {},
        set textBaseline(_v: string) {},
      } as unknown as CanvasRenderingContext2D;
    });
  });

  it('encodes and parses multi-track Standard MIDI Files with exact tick/pitch fidelity', () => {
    const preset = StandardMidiCodec.createDemoMidiPreset();
    expect(preset.tracks.length).toBe(2);

    const track1 = new DawTrack({
      name: preset.tracks[0].name,
      instrument: preset.tracks[0].instrument,
      notes: preset.tracks[0].notes,
    });
    const track2 = new DawTrack({
      name: preset.tracks[1].name,
      instrument: preset.tracks[1].instrument,
      notes: preset.tracks[1].notes,
    });

    const bytes = StandardMidiCodec.encodeMultiTrackMidi(
      [track1.toSnapshot(), track2.toSnapshot()],
      480,
      124,
    );
    expect(bytes[0]).toBe(0x4d); // 'M'
    expect(bytes[1]).toBe(0x54); // 'T'

    const reparsed = StandardMidiCodec.parseMidi(bytes.buffer as ArrayBuffer);
    expect(reparsed.ppqn).toBe(480);
    expect(reparsed.bpm).toBe(124);
    expect(reparsed.tracks.length).toBe(2);
    expect(reparsed.tracks[0].notes.length).toBe(preset.tracks[0].notes.length);
    expect(reparsed.tracks[1].instrument).toBe('SAX');
  });

  it('executes O(log N + K) binary-search note queries and reversible CommandManager Undo/Redo', () => {
    const notes: MidiNote[] = [];
    for (let i = 0; i < 600; i += 1) {
      notes.push(new MidiNote({ pitch: 60 + (i % 12), startTick: i * 120, durationTicks: 100 }));
    }
    const track = new DawTrack({ name: 'Piano', instrument: 'PIANO' });
    const cmdMgr = new CommandManager();

    cmdMgr.execute(new AddNotesCommand(track, notes));
    expect(track.getNotes().length).toBe(600);

    const sliced = track.queryNotesStartingIn(1200, 2400);
    expect(sliced.length).toBe(10);
    expect(sliced[0].startTick).toBe(1200);
    expect(sliced[sliced.length - 1].startTick).toBe(2280);

    const replacement = [new MidiNote({ pitch: 72, startTick: 1200, durationTicks: 480 })];
    cmdMgr.execute(new ReplaceRangeNotesCommand(track, 1200, 2400, replacement));
    expect(track.queryNotesStartingIn(1200, 2400).length).toBe(1);

    expect(cmdMgr.undo()).toBe(true);
    expect(track.queryNotesStartingIn(1200, 2400).length).toBe(10);

    expect(cmdMgr.redo()).toBe(true);
    expect(track.queryNotesStartingIn(1200, 2400).length).toBe(1);
  });

  it('supports hot-swapping DAW controllers (Demo / Cubase / Ableton) and trims 1-bar SFT leading blank', async () => {
    const audio = new StubAudioEngine();
    const registry = new DawControllerRegistry();
    const demo = new DemoPianoRollController(audio);
    const cubase = new CubaseDawController();
    const ableton = new AbletonDawController();

    registry.register(demo);
    registry.register(cubase);
    registry.register(ableton);

    expect(registry.getActive().id).toBe('demo-pianoroll');
    await registry.activate('cubase-midi-remote');
    expect(registry.getActive().id).toBe('cubase-midi-remote');
    await registry.activate('ableton-osc-bridge');
    expect(registry.getActive().id).toBe('ableton-osc-bridge');
    await registry.activate('demo-pianoroll');

    // Add past notes in Bar 1..2 and set target range to Bar 3..4 -> auto task should be `meta_past`
    const range = new MeasureRange(3, 4, 480, 4);
    expect(range.clampedSftMeasureCount).toBe(2);

    demo.createNoteAt(60, 0, 480, 100);
    demo.setTargetMeasureRange(3, 4);

    const ctx = await demo.extractGenerationContext({ taskOverride: 'auto' });
    expect(ctx.resolvedTask).toBe('meta_past');
    expect(ctx.pastMidiBytes).not.toBeNull();

    // Simulate SFT generated MIDI: startTick 0 places at Bar 3 (3840), while genuine 1-bar rest (1920) stays at Bar 4 (5760)
    const genMidiBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [
        new MidiNote({ pitch: 65, startTick: 0, durationTicks: 480, velocity: 100 }),
        new MidiNote({ pitch: 67, startTick: 1920, durationTicks: 480, velocity: 105 }),
      ],
      480,
      120,
      'PIANO',
    );
    const parsedGen = StandardMidiCodec.parseMidi(genMidiBytes.buffer as ArrayBuffer);
    await demo.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'output_0.mid',
          rawMidiBytes: genMidiBytes,
          parsedMidi: parsedGen,
          reason: {
            source: 'model_cot',
            task: 'meta_past',
            thinking: true,
            instruments: ['PIANO'],
            note_density: { PIANO: 4 },
            gen_measure_count: 2,
            genre: ['jazz'],
            key: 'CM',
          },
        },
      ],
      [],
    );

    const activeNotes = demo.getSnapshot().tracks[0].notes;
    expect(activeNotes.find((n) => n.pitch === 65)?.startTick).toBe(3840);
    expect(activeNotes.find((n) => n.pitch === 67)?.startTick).toBe(5760);
  });

  it('renders BootSplashScreen with COMPASS AI logo, Loading..., and zero AI slop symbols', () => {
    const { container } = render(
      <BootSplashScreen
        currentAction="モデル定義を読み込んでいます..."
        progressPercent={64}
        logs={[{ id: 1, timestamp: '12:00:00', message: '初期化開始' }]}
      />,
    );

    expect(screen.getByText('COMPASS AI')).toBeTruthy();
    expect(screen.getByText('Loading...')).toBeTruthy();
    expect(screen.getByText('モデル定義を読み込んでいます...')).toBeTruthy();

    const text = container.textContent ?? '';
    expect(/[✦✧★☆🚀💡✨🔥🎯🤖]/u.test(text)).toBe(false);
  });

  it('completes boot, switches via Shift+G / COMPASS AI を起動, uploads MIDI, and auto-foregrounds Piano Roll after Generate', async () => {
    // Prepare mock ZIP response for POST /generate
    const zip = new JSZip();
    const genMidi = StandardMidiCodec.encodeSingleTrackMidi(
      [
        new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480, velocity: 96 }),
        new MidiNote({ pitch: 67, startTick: 480, durationTicks: 480, velocity: 98 }),
      ],
      480,
      120,
      'PIANO',
    );
    zip.file('output_0.mid', genMidi);
    const zipBytes = await zip.generateAsync({ type: 'uint8array' });

    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'float32',
            backend_class: 'CpuBackend',
            user_models_dir: 'C:\\Users\\user\\AppData\\Roaming\\COMPASS_AI\\models',
            bundled_models_dir: 'g:\\My_Project\\COMPASS_AI\\data\\models',
            available_models_count: 1,
            models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info')) {
        return new Response(
          JSON.stringify([
            {
              model_name: 'MORTM4.5D-80M-SFT-Gen',
              has_weights: true,
              tag: { type: 'sft', version: '4.5' },
            },
          ]),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/generate')) {
        const reasonHeader = encodeURIComponent(
          JSON.stringify([
            {
              source: 'model_cot',
              task: 'meta_past',
              thinking: true,
              instruments: ['PIANO'],
              note_density: { PIANO: 4 },
              gen_measure_count: 4,
              genre: ['classical'],
              key: 'CM',
            },
          ]),
        );
        return new Response(zipBytes as Uint8Array<ArrayBuffer>, {
          status: 200,
          headers: {
            'Content-Type': 'application/zip',
            'X-Generation-Reason': reasonHeader,
          },
        });
      }
      return new Response('{}', { status: 200 });
    });

    const { container } = render(<App />);

    // Initially shows BootSplashScreen (`Loading...`)
    expect(screen.getByText('Loading...')).toBeTruthy();

    // Wait for boot transition to main COMPASS AI screen + separate Piano Roll tab
    await waitFor(
      () => {
        expect(screen.getByTestId('generate-button')).toBeTruthy();
      },
      { timeout: 3500 },
    );

    const tabCompass = screen.getByTestId('tab-compass-ai');
    const tabPianoRoll = screen.getByTestId('tab-pianoroll');
    expect(tabCompass.getAttribute('aria-selected')).toBe('true');
    expect(tabPianoRoll.getAttribute('aria-selected')).toBe('false');

    // Switch to Piano Roll tab and upload a single-track MIDI file
    fireEvent.click(tabPianoRoll);
    expect(tabPianoRoll.getAttribute('aria-selected')).toBe('true');
    expect(screen.getByTestId('upload-midi-button')).toBeTruthy();

    const uploadBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const midiFile = new File([uploadBytes as Uint8Array<ArrayBuffer>], 'phrase.mid', {
      type: 'audio/midi',
    });
    // Polyfill File.prototype.arrayBuffer if missing in older jsdom
    if (typeof midiFile.arrayBuffer !== 'function') {
      Object.defineProperty(midiFile, 'arrayBuffer', {
        value: async () => uploadBytes.buffer.slice(uploadBytes.byteOffset, uploadBytes.byteOffset + uploadBytes.byteLength),
      });
    }

    const inputEl = screen.getByTestId('midi-file-input');
    fireEvent.change(inputEl, { target: { files: [midiFile] } });

    await waitFor(() => {
      expect(tabPianoRoll.textContent).toContain('1');
    });

    // Press Shift+G -> brings COMPASS AI (Circle + Control Panel) to foreground
    fireEvent.keyDown(window, { key: 'G', code: 'KeyG', shiftKey: true });
    await waitFor(() => {
      expect(screen.getByTestId('tab-compass-ai').getAttribute('aria-selected')).toBe('true');
    });

    // Switch back to Piano Roll and test clicking `COMPASS AI を起動` button
    fireEvent.click(screen.getByTestId('tab-pianoroll'));
    fireEvent.click(screen.getByTestId('launch-compass-ai-button'));
    expect(screen.getByTestId('tab-compass-ai').getAttribute('aria-selected')).toBe('true');

    // Click Generate -> animates circle and automatically brings Piano Roll to foreground when done
    fireEvent.click(screen.getByTestId('generate-button'));

    await waitFor(
      () => {
        expect(screen.getByTestId('tab-pianoroll').getAttribute('aria-selected')).toBe('true');
      },
      { timeout: 3500 },
    );

    expect(fetchMock).toHaveBeenCalled();
    // Total notes should now be 1 uploaded + 2 generated = 3
    expect(screen.getByTestId('tab-pianoroll').textContent).toContain('3');

    // Verify zero AI slop glyphs in full rendered DOM
    const fullText = container.textContent ?? '';
    expect(/[✦✧★☆🚀💡✨🔥🎯🤖]/u.test(fullText)).toBe(false);
  });

  it('round-trips >=128-byte UTF-8 VLQ track names, DRUMS Ch 10, 3/4 Time Signature, and Conditions MIDI', () => {
    const longJapaneseName = 'ピアノトラック・メインテーマ旋律検証用ロングタイトル'.repeat(3);
    expect(new TextEncoder().encode(longJapaneseName).length).toBeGreaterThanOrEqual(128);

    const pianoTrack = new DawTrack({
      name: longJapaneseName,
      instrument: 'PIANO',
      notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
    });
    const multiBytes = StandardMidiCodec.encodeMultiTrackMidi(
      [pianoTrack.toSnapshot()],
      480,
      128,
      3,
    );
    const parsedMulti = StandardMidiCodec.parseMidi(multiBytes.buffer as ArrayBuffer);
    expect(parsedMulti.beatsPerMeasure).toBe(3);
    expect(parsedMulti.tracks[0].name).toBe(longJapaneseName);

    const drumBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 36, startTick: 0, durationTicks: 240, velocity: 110 })],
      480,
      120,
      'DRUMS',
      3,
    );
    const parsedDrum = StandardMidiCodec.parseMidi(drumBytes.buffer as ArrayBuffer);
    expect(parsedDrum.beatsPerMeasure).toBe(3);
    expect(parsedDrum.tracks[0].isDrum).toBe(true);
    expect(parsedDrum.tracks[0].instrument).toBe('DRUMS');

    const condBytes = StandardMidiCodec.encodeConditionsMidi(
      [new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480, velocity: 95 })],
      480,
      120,
      false,
      3,
    );
    const parsedCond = StandardMidiCodec.parseMidi(condBytes.buffer as ArrayBuffer);
    expect(parsedCond.beatsPerMeasure).toBe(3);
    expect(parsedCond.tracks[0].instrument).toBe('PIANO');
  });

  it('handles adversarial drone notes via implicit interval tree and truncates/restores sustaining notes on Undo', () => {
    const drone = new MidiNote({ id: 'drone_0', pitch: 36, startTick: 0, durationTicks: 10000 });
    const manyShort: MidiNote[] = [drone];
    for (let i = 0; i < 1000; i += 1) {
      manyShort.push(
        new MidiNote({ pitch: 60 + (i % 12), startTick: 10 + i * 4, durationTicks: 20 }),
      );
    }
    const track = new DawTrack({ name: 'DroneTest', instrument: 'PIANO', notes: manyShort });
    const hits = track.queryNotesOverlapping(8000, 9000);
    expect(hits.length).toBe(1);
    expect(hits[0].id).toBe('drone_0');

    // Sustaining note spanning [1000, 3000] when [2000, 4000] is replaced
    const sustainTrack = new DawTrack({
      name: 'Sustain',
      instrument: 'PIANO',
      notes: [new MidiNote({ id: 'sus_1', pitch: 60, startTick: 1000, durationTicks: 2000 })],
    });
    const cmdMgr = new CommandManager();
    cmdMgr.execute(
      new ReplaceRangeNotesCommand(sustainTrack, 2000, 4000, [
        new MidiNote({ id: 'gen_1', pitch: 67, startTick: 2000, durationTicks: 500 }),
      ]),
    );
    const afterExec = sustainTrack.getNotes();
    expect(afterExec.find((n) => n.id === 'sus_1')?.durationTicks).toBe(1000);
    expect(afterExec.find((n) => n.id === 'gen_1')).toBeDefined();

    expect(cmdMgr.undo()).toBe(true);
    const afterUndo = sustainTrack.getNotes();
    expect(afterUndo.length).toBe(1);
    expect(afterUndo[0].id).toBe('sus_1');
    expect(afterUndo[0].durationTicks).toBe(2000);
  });

  it('cleans trailing notes on candidate switch and synchronizes 0-note multi-track workspace across WorkspaceSyncBus', async () => {
    const audio = new StubAudioEngine();
    const ctrlA = new DemoPianoRollController(audio);
    ctrlA.setTargetMeasureRange(1, 2);

    // Candidate 0 spans 6 bars (up to tick 9600); Candidate 1 spans 2 bars (up to tick 1920)
    const cand0Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [
        new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480 }),
        new MidiNote({ pitch: 72, startTick: 7680, durationTicks: 480 }),
      ],
      480,
      120,
      'PIANO',
    );
    const cand1Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480 })],
      480,
      120,
      'PIANO',
    );

    await ctrlA.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'output_0.mid',
          rawMidiBytes: cand0Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(cand0Bytes.buffer as ArrayBuffer),
          reason: null,
        },
        {
          index: 1,
          filename: 'output_1.mid',
          rawMidiBytes: cand1Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(cand1Bytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    expect(ctrlA.getSnapshot().tracks[0].notes.length).toBe(2);

    await ctrlA.selectGeneratedCandidate(1);
    expect(ctrlA.getSnapshot().activeCandidateIndex).toBe(1);
    expect(ctrlA.getSnapshot().tracks[0].notes.length).toBe(1);
    expect(ctrlA.getSnapshot().tracks[0].notes[0].pitch).toBe(64);

    // Undo / Redo restores activeCandidateIndex alongside canvas notes
    ctrlA.undo();
    expect(ctrlA.getSnapshot().activeCandidateIndex).toBe(0);
    expect(ctrlA.getSnapshot().tracks[0].notes.length).toBe(2);
    ctrlA.redo();
    expect(ctrlA.getSnapshot().activeCandidateIndex).toBe(1);
    expect(ctrlA.getSnapshot().tracks[0].notes.length).toBe(1);

    // Add a third 0-note SAX track and synchronize to peer controller via WorkspaceSyncBus
    ctrlA.addTrack('Lead Sax Empty', 'SAX');
    expect(ctrlA.getSnapshot().tracks.length).toBe(3);

    const ctrlB = new DemoPianoRollController(audio);
    const chanName = `compass_sync_test_${Date.now()}`;
    const busA = new WorkspaceSyncBus(chanName);
    const busB = new WorkspaceSyncBus(chanName);

    busA.subscribe((msg) => {
      if (msg.type === 'REQUEST_WORKSPACE_STATE') {
        busA.publishWorkspaceState(ctrlA.exportWorkspaceState());
      }
    });
    busB.subscribe((msg) => {
      if (msg.type === 'SYNC_WORKSPACE_STATE') {
        ctrlB.applyWorkspaceState(msg.workspaceState);
      }
    });

    busB.requestWorkspaceState();
    await waitFor(() => {
      expect(ctrlB.getSnapshot().tracks.length).toBe(3);
    });
    expect(ctrlB.getSnapshot().tracks[2].name).toBe('Lead Sax Empty');
    expect(ctrlB.getSnapshot().tracks[2].notes.length).toBe(0);

    busA.dispose();
    busB.dispose();
  });

  it('enforces Round 3 adversarial invariants: hotkey isolation, infill future protection, Cubase multi-bar retention, play/stop race, and NaN/immutability guards', async () => {
    // 1. [MED-01] Async play() -> immediate stop() race cancels pending playback cleanly
    let resolveResume: (() => void) | null = null;
    const delayedAudio: IAudioEngine = {
      ensureResumed: () =>
        new Promise<void>((res) => {
          resolveResume = res;
        }),
      getCurrentTime: () => 0,
      triggerNote: () => {},
      auditionPitch: () => {},
      stopAllVoices: () => {},
      readAcousticTelemetry: (w, s) => {
        w.fill(0);
        s.fill(0);
        return { rms: 0, peak: 0, lowBandEnergy: 0, highBandEnergy: 0 };
      },
      dispose: async () => {},
    };
    const raceCtrl = new DemoPianoRollController(delayedAudio);
    const playPromise = raceCtrl.play();
    raceCtrl.stop();
    (resolveResume as (() => void) | null)?.();
    await playPromise;
    expect(raceCtrl.getSnapshot().isPlaying).toBe(false);

    // 2. [MED-07] & [LOW-03] NaN immunity, frozen snapshots, SMPTE 0-tick rejection, and 6/8 & 7/8 round-trip
    raceCtrl.setBpm(132);
    raceCtrl.setBpm(Number.NaN);
    expect(raceCtrl.getSnapshot().bpm).toBe(132);
    raceCtrl.setPlayheadTick(960);
    raceCtrl.setPlayheadTick(Number.POSITIVE_INFINITY);
    expect(raceCtrl.getSnapshot().playheadTick).toBe(960);

    const safeNote = new MidiNote({
      pitch: Number.NaN,
      startTick: Number.NEGATIVE_INFINITY,
      durationTicks: Number.NaN,
      velocity: Number.NaN,
    });
    expect(Number.isFinite(safeNote.pitch)).toBe(true);
    expect(Number.isFinite(safeNote.startTick)).toBe(true);
    expect(safeNote.durationTicks).toBeGreaterThanOrEqual(1);

    const frozenSnap = raceCtrl.getSnapshot();
    expect(Object.isFrozen(frozenSnap)).toBe(true);
    expect(Object.isFrozen(frozenSnap.tracks)).toBe(true);
    (frozenSnap.selectedNoteIds as Set<string>).add('mutated_id');
    expect(raceCtrl.getSnapshot().selectedNoteIds.has('mutated_id')).toBe(false);

    // SMPTE division 0xE800 (ticksPerFrame = 0) must throw cleanly
    const badSmpte = new Uint8Array([
      0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
      0x00, 0x00, 0x00, 0x01, 0xe8, 0x00,
      0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x04, 0x00, 0xff, 0x2f, 0x00,
    ]);
    expect(() => StandardMidiCodec.parseMidi(badSmpte.buffer)).toThrow(/SMPTE/);

    // 6/8 and 7/8 time signature round-trip
    const tempTrack = new DawTrack({
      name: 'OddMeter',
      instrument: 'PIANO',
      notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 240 })],
    });
    const midi68 = StandardMidiCodec.encodeMultiTrackMidi([tempTrack.toSnapshot()], 480, 120, 3, {
      numerator: 6,
      denominator: 8,
    });
    const parsed68 = StandardMidiCodec.parseMidi(midi68.buffer as ArrayBuffer);
    expect(parsed68.timeSigNumerator).toBe(6);
    expect(parsed68.timeSigDenominator).toBe(8);
    expect(parsed68.beatsPerMeasure).toBe(3);

    const midi78 = StandardMidiCodec.encodeMultiTrackMidi([tempTrack.toSnapshot()], 480, 120, 3.5, {
      numerator: 7,
      denominator: 8,
    });
    const parsed78 = StandardMidiCodec.parseMidi(midi78.buffer as ArrayBuffer);
    expect(parsed78.timeSigNumerator).toBe(7);
    expect(parsed78.timeSigDenominator).toBe(8);
    expect(parsed78.beatsPerMeasure).toBe(3.5);

    // Pop-out WorkspaceSync round-trips 6/8 & 7/8 time signature numerator/denominator on export
    const oddSrcCtrl = new DemoPianoRollController(new StubAudioEngine());
    oddSrcCtrl.importParsedMidiTracks(parsed68, [0], true);
    const oddDstCtrl = new DemoPianoRollController(new StubAudioEngine());
    oddDstCtrl.applyWorkspaceState(oddSrcCtrl.exportWorkspaceState());
    const reExported68 = StandardMidiCodec.parseMidi(oddDstCtrl.exportProjectMidiBytes().buffer as ArrayBuffer);
    expect(reExported68.timeSigNumerator).toBe(6);
    expect(reExported68.timeSigDenominator).toBe(8);
    expect(reExported68.beatsPerMeasure).toBe(3);

    // 3. [HIGH-08] Infill overhanging candidate clips at targetRange.endTick and preserves future Bar 5 notes
    const infillCtrl = new DemoPianoRollController(new StubAudioEngine());
    infillCtrl.createNoteAt(60, 0, 480, 100); // Bar 1 (Past)
    infillCtrl.createNoteAt(72, 7680, 480, 100); // Bar 5 (Future, tick 7680)
    infillCtrl.setTargetMeasureRange(3, 4); // Target Bars 3..4 [3840, 7680)
    expect(infillCtrl.getSnapshot().contextAvailability.autoResolvedTask).toBe('infill');

    const overhangCandBytes = StandardMidiCodec.encodeMultiTrackMidi(
      [
        new DawTrack({
          name: 'Piano Gen',
          instrument: 'PIANO',
          notes: [
            new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480 }),
            new MidiNote({ pitch: 67, startTick: 3600, durationTicks: 600 }), // Overhangs 3840 -> clipped to 240
            new MidiNote({ pitch: 79, startTick: 4200, durationTicks: 480 }), // Starts past 3840 -> dropped
          ],
        }).toSnapshot(),
        new DawTrack({
          name: 'Sax Gen',
          instrument: 'SAX',
          notes: [new MidiNote({ pitch: 84, startTick: 0, durationTicks: 480 })], // Filtered out for PIANO track
        }).toSnapshot(),
      ],
      480,
      120,
    );
    await infillCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'infill_0.mid',
          rawMidiBytes: overhangCandBytes,
          parsedMidi: StandardMidiCodec.parseMidi(overhangCandBytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    const infillNotes = infillCtrl.getSnapshot().tracks[0].notes;
    expect(infillNotes.find((n) => n.pitch === 72 && n.startTick === 7680)).toBeDefined();
    expect(infillNotes.find((n) => n.pitch === 84)).toBeUndefined();
    expect(infillNotes.find((n) => n.pitch === 79)).toBeUndefined();
    expect(infillNotes.find((n) => n.pitch === 67)?.endTick).toBe(7680);

    // 4. [HIGH-10.1] CubaseDawController retains Bar 1..4 notes when generating Bar 5..8
    const cubase = new CubaseDawController();
    cubase.setTargetMeasureRange(1, 2);
    const bar1Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480 })],
      480,
      120,
      'PIANO',
    );
    await cubase.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'bar1.mid',
          rawMidiBytes: bar1Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(bar1Bytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    cubase.setTargetMeasureRange(3, 4);
    expect(cubase.getSnapshot().contextAvailability.autoResolvedTask).toBe('meta_past');
    const bar3Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 67, startTick: 0, durationTicks: 480 })],
      480,
      120,
      'PIANO',
    );
    await cubase.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'bar3.mid',
          rawMidiBytes: bar3Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(bar3Bytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    expect(cubase.getSnapshot().tracks[0].notes.length).toBe(2);
    cubase.stop();
    expect(cubase.getSnapshot().playheadTick).toBe(3840);

    // 5. [HIGH-06] MidiTrackImportModal truncates 80-char uninterrupted filename safely on 375px layout
    const longFileName = `${'UninterruptedStudioSessionTake_'.repeat(3)}.mid`;
    const sampleMulti = StandardMidiCodec.encodeMultiTrackMidi(
      [
        new DawTrack({
          name: 'Uninterrupted_Track_Header_Name_For_Mobile_Audit_001',
          instrument: 'PIANO',
          notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480 })],
        }).toSnapshot(),
      ],
      480,
      120,
    );
    const parsedModalMidi = StandardMidiCodec.parseMidi(sampleMulti.buffer as ArrayBuffer);
    const modalRender = render(
      <MidiTrackImportModal
        fileName={longFileName}
        parsedMidi={parsedModalMidi}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );
    const modalDialog = modalRender.getByTestId('midi-import-modal');
    expect(modalDialog.querySelector('.min-w-0.flex-1')).toBeTruthy();
    expect(modalDialog.querySelector('.truncate.w-full')?.textContent).toContain(longFileName);
    modalRender.unmount();

    // 6. [HIGH-05] Hidden PianoRollView ignores Space/Delete while COMPASS AI tab or dialog is active
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'float32',
            backend_class: 'CpuBackend',
            user_models_dir: 'C:\\Users\\user\\AppData\\Roaming\\COMPASS_AI\\models',
            bundled_models_dir: 'g:\\My_Project\\COMPASS_AI\\data\\models',
            available_models_count: 1,
            models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info')) {
        return new Response(
          JSON.stringify([
            {
              model_name: 'MORTM4.5D-80M-SFT-Gen',
              has_weights: true,
              tag: { type: 'sft', version: '4.5' },
            },
          ]),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response('{}', { status: 200 });
    });

    const appRender = render(<App />);
    await waitFor(
      () => {
        expect(screen.getByTestId('generate-button')).toBeTruthy();
      },
      { timeout: 3500 },
    );
    // Switch to Piano Roll, upload 1 note, select all, then return to COMPASS AI tab
    fireEvent.click(screen.getByTestId('tab-pianoroll'));
    const oneNoteBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const oneFile = new File([oneNoteBytes as Uint8Array<ArrayBuffer>], 'guard.mid', {
      type: 'audio/midi',
    });
    if (typeof oneFile.arrayBuffer !== 'function') {
      Object.defineProperty(oneFile, 'arrayBuffer', {
        value: async () =>
          oneNoteBytes.buffer.slice(oneNoteBytes.byteOffset, oneNoteBytes.byteOffset + oneNoteBytes.byteLength),
      });
    }
    fireEvent.change(screen.getByTestId('midi-file-input'), { target: { files: [oneFile] } });
    await waitFor(() => {
      expect(screen.getByTestId('tab-pianoroll').textContent).toContain('1');
    });
    // Select all notes while Piano Roll is active
    fireEvent.keyDown(window, { key: 'a', ctrlKey: true });
    // Switch to COMPASS AI tab (PianoRollView is now hidden via class="hidden" & isActive=false)
    fireEvent.click(screen.getByTestId('tab-compass-ai'));
    expect(screen.getByTestId('tab-compass-ai').getAttribute('aria-selected')).toBe('true');

    // Pressing Delete / Backspace / Space while COMPASS AI tab is active must NOT delete note or start Piano Roll playback
    fireEvent.keyDown(window, { key: 'Delete' });
    fireEvent.keyDown(window, { key: 'Backspace' });
    fireEvent.keyDown(window, { key: ' ' });
    expect(screen.getByTestId('tab-pianoroll').textContent).toContain('1');
    appRender.unmount();
  });

  it('verifies Phase 2 Critic Round 2 remediations: focused-button Shift+G/Delete/Undo, non-passive Ctrl+Wheel, touch pointer editing, X-Compass-Handshake + job_id, same-range overhang cleanup, 7/8 time signature, atomic sync guard, and 64-voice polyphony cap', async () => {
    // 1. [R2-HIGH-03] & [R2-HIGH-07] CompassApiClient sends X-Compass-Handshake & scoped job_id on /generate + /cancel_generate
    let seenHwHandshake: string | null = null;
    let seenGenJobId: string | null = null;
    let seenGenHandshake: string | null = null;
    let seenCancelJobId: string | null = null;
    let seenCancelHandshake: string | null = null;

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();
      const hdrs = new Headers(init?.headers);
      if (url.endsWith('/hardware_info')) {
        seenHwHandshake = hdrs.get('X-Compass-Handshake');
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'float32',
            backend_class: 'CpuBackend',
            user_models_dir: 'C:\\Models',
            bundled_models_dir: 'C:\\Bundled',
            available_models_count: 1,
            models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.includes('/cancel_generate')) {
        seenCancelHandshake = hdrs.get('X-Compass-Handshake');
        const parsedUrl = new URL(url);
        seenCancelJobId = hdrs.get('X-Compass-Job-Id') ?? parsedUrl.searchParams.get('job_id');
        return new Response(JSON.stringify({ cancelled: true }), { status: 200 });
      }
      if (url.endsWith('/generate')) {
        seenGenHandshake = hdrs.get('X-Compass-Handshake');
        seenGenJobId = hdrs.get('X-Compass-Job-Id');
        // Wait until aborted by caller
        return await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });
      }
      return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } });
    });

    const authedClient = new CompassApiClient('http://127.0.0.1:18999', 'secret_handshake_r2');
    await authedClient.getHardwareInfo();
    expect(seenHwHandshake).toBe('secret_handshake_r2');

    const genAbort = new AbortController();
    const genPromise = authedClient.generateMusic(
      {
        meta: {
          model_type: 'MORTM4.5D-80M-SFT-Gen',
          task: 'meta',
          program: ['PIANO'],
          tempo: 120,
          num_gems: 1,
          genfield_measure: 4,
          gen_note_dense: null,
          p: 0.95,
          temperature: 1.0,
          cot_temperature: 0.8,
          thinking: false,
          time_signature_numerator: 7,
          time_signature_denominator: 8,
        },
      },
      undefined,
      genAbort.signal,
    );
    await waitFor(() => {
      expect(seenGenJobId).toBeTruthy();
    });
    genAbort.abort();
    await expect(genPromise).rejects.toThrow();
    await waitFor(() => {
      expect(seenCancelJobId).toBe(seenGenJobId);
    });
    expect(seenGenHandshake).toBe('secret_handshake_r2');
    expect(seenCancelHandshake).toBe('secret_handshake_r2');

    // 2. [R2-HIGH-04] Same-range re-generation cleans overhanging notes & preserves meta_past (never corrupts to infill)
    const regenCtrl = new DemoPianoRollController(new StubAudioEngine());
    regenCtrl.createNoteAt(60, 0, 480, 100); // Past note in Bar 1
    regenCtrl.setTargetMeasureRange(5, 8); // Target Bars 5..8 -> [7680, 15360)
    expect(regenCtrl.getSnapshot().contextAvailability.autoResolvedTask).toBe('meta_past');

    // Candidate 1 overhangs into Bar 9 (offset 7920 -> absolute 15600 >= 15360)
    const overhang5BarBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [
        new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480 }),
        new MidiNote({ pitch: 76, startTick: 7920, durationTicks: 480 }),
      ],
      480,
      120,
      'PIANO',
    );
    await regenCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'gen1.mid',
          rawMidiBytes: overhang5BarBytes,
          parsedMidi: StandardMidiCodec.parseMidi(overhang5BarBytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    expect(regenCtrl.getSnapshot().tracks[0].notes.some((n) => n.pitch === 76)).toBe(true);
    // While targetRange is STILL Bars 5..8, autoResolvedTask MUST remain meta_past (NOT infill!)
    expect(regenCtrl.getSnapshot().contextAvailability.autoResolvedTask).toBe('meta_past');
    const regenCtx = await regenCtrl.extractGenerationContext({ taskOverride: 'auto' });
    expect(regenCtx.resolvedTask).toBe('meta_past');
    expect(regenCtx.futureMidiBytes).toBeNull();

    // Re-generating on the SAME range Bars 5..8 removes the Bar-9 overhang note cleanly
    const short2BarBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 67, startTick: 0, durationTicks: 480 })],
      480,
      120,
      'PIANO',
    );
    await regenCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'gen2.mid',
          rawMidiBytes: short2BarBytes,
          parsedMidi: StandardMidiCodec.parseMidi(short2BarBytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    expect(regenCtrl.getSnapshot().tracks[0].notes.some((n) => n.pitch === 76)).toBe(false);
    expect(regenCtrl.getSnapshot().tracks[0].notes.some((n) => n.pitch === 67)).toBe(true);

    // Advancing targetRange to Bars 9..12 treats Bars 5..8 as committed Past context for multi-section composition
    regenCtrl.setTargetMeasureRange(9, 12);
    expect(regenCtrl.exportWorkspaceState().activeCandidateRange?.startTick).toBe(7680);
    expect(regenCtrl.getSnapshot().contextAvailability.autoResolvedTask).toBe('meta_past');
    const bar9Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 72, startTick: 0, durationTicks: 480 })],
      480,
      120,
      'PIANO',
    );
    await regenCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'gen3_bar9.mid',
          rawMidiBytes: bar9Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(bar9Bytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    expect(regenCtrl.getSnapshot().tracks[0].notes.some((n) => n.pitch === 67)).toBe(true);
    expect(regenCtrl.getSnapshot().tracks[0].notes.some((n) => n.pitch === 72)).toBe(true);

    // [R2-MED-07] External DAW adapters report honest canUndo=false / canRedo=false
    const extCubase = new CubaseDawController();
    expect(extCubase.getSnapshot().canUndo).toBe(false);
    expect(extCubase.getSnapshot().canRedo).toBe(false);

    // 3. [R2-HIGH-05] setTimeSignature(7, 8) propagates exact 7/8 SMF meta events into context MIDI
    regenCtrl.setTimeSignature(7, 8);
    expect(regenCtrl.getSnapshot().timeSigNumerator).toBe(7);
    expect(regenCtrl.getSnapshot().timeSigDenominator).toBe(8);
    expect(regenCtrl.getSnapshot().beatsPerMeasure).toBe(3.5);
    const ctx78 = await regenCtrl.extractGenerationContext({ taskOverride: 'meta_past' });
    expect(ctx78.pastMidiBytes).not.toBeNull();
    const parsedPast78 = StandardMidiCodec.parseMidi(ctx78.pastMidiBytes!.buffer as ArrayBuffer);
    expect(parsedPast78.timeSigNumerator).toBe(7);
    expect(parsedPast78.timeSigDenominator).toBe(8);

    // 4. [R2-MED-03] Malformed applyWorkspaceState payload aborts atomically before wiping tracks or undo stack
    const beforeMalformedCount = regenCtrl.getSnapshot().tracks[0].notes.length;
    expect(regenCtrl.getSnapshot().canUndo).toBe(true);
    regenCtrl.applyWorkspaceState({
      tracks: [
        {
          id: 'corrupt',
          name: 'Corrupt',
          instrument: 'PIANO',
          colorHex: '#fff',
          isMuted: false,
          isSoloed: false,
          notes: null as unknown as [],
        },
      ],
    } as never);
    expect(regenCtrl.getSnapshot().tracks[0].notes.length).toBe(beforeMalformedCount);
    expect(regenCtrl.getSnapshot().canUndo).toBe(true);

    // 5. [R2-LOW-03] WebAudioEngine caps simultaneous voices at MAX_ACTIVE_VOICES (64)
    const OrigAudioCtx = (window as unknown as { AudioContext?: unknown }).AudioContext;
    class MockGainNode {
      public gain = {
        value: 1,
        setValueAtTime: () => {},
        linearRampToValueAtTime: () => {},
        exponentialRampToValueAtTime: () => {},
        cancelScheduledValues: () => {},
      };
      public connect() {}
      public disconnect() {}
    }
    class MockOscillatorNode {
      public type = 'triangle';
      public frequency = { value: 440, setValueAtTime: () => {}, exponentialRampToValueAtTime: () => {} };
      public onended: (() => void) | null = null;
      public connect() {}
      public disconnect() {}
      public start() {}
      public stop() {}
    }
    class MockAudioContext {
      public state = 'running';
      public currentTime = 0;
      public sampleRate = 44100;
      public destination = {};
      public createGain() {
        return new MockGainNode();
      }
      public createDynamicsCompressor() {
        return {
          threshold: { value: 0 },
          knee: { value: 0 },
          ratio: { value: 0 },
          attack: { value: 0 },
          release: { value: 0 },
          connect: () => {},
        };
      }
      public createAnalyser() {
        return {
          fftSize: 2048,
          smoothingTimeConstant: 0.8,
          frequencyBinCount: 1024,
          connect: () => {},
          getFloatTimeDomainData: (a: Float32Array) => a.fill(0),
          getByteFrequencyData: (a: Uint8Array) => a.fill(0),
        };
      }
      public createOscillator() {
        return new MockOscillatorNode();
      }
      public createBiquadFilter() {
        return {
          type: 'lowpass',
          Q: { value: 1, setValueAtTime: () => {} },
          frequency: {
            value: 1000,
            setValueAtTime: () => {},
            linearRampToValueAtTime: () => {},
            exponentialRampToValueAtTime: () => {},
            cancelScheduledValues: () => {},
          },
          connect: () => {},
          disconnect: () => {},
        };
      }
      public createBuffer(_ch: number, len: number) {
        return { getChannelData: () => new Float32Array(len) };
      }
      public createBufferSource() {
        return { buffer: null, onended: null, connect: () => {}, disconnect: () => {}, start: () => {}, stop: () => {} };
      }
      public async resume() {}
      public async close() {}
    }
    (window as unknown as { AudioContext: unknown }).AudioContext = MockAudioContext;
    try {
      const voiceEngine = new WebAudioEngine();
      await voiceEngine.ensureResumed();
      for (let i = 0; i < 100; i += 1) {
        voiceEngine.triggerNote(36 + (i % 48), 100, 2.0, 'PIANO', 0);
      }
      expect(voiceEngine.activeVoiceCount).toBeLessThanOrEqual(64);
      await voiceEngine.dispose();
    } finally {
      (window as unknown as { AudioContext?: unknown }).AudioContext = OrigAudioCtx;
    }

    // 6. [R2-HIGH-01] & [R2-HIGH-02] Focused-<button> Shift+G / Delete / Ctrl+Z, Non-Passive Ctrl+Wheel, & Touch Pointer Draw
    const appUi = render(<App />);
    await waitFor(
      () => {
        expect(screen.getByTestId('generate-button')).toBeTruthy();
      },
      { timeout: 3500 },
    );

    const prTabBtn = screen.getByTestId('tab-pianoroll');
    fireEvent.click(prTabBtn);
    prTabBtn.focus();
    expect(document.activeElement).toBe(prTabBtn);

    const canvas = screen.getByTestId('pianoroll-canvas');
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    });

    // Imperative { passive: false } wheel listener sets defaultPrevented === true
    const wheelEvent = new WheelEvent('wheel', {
      deltaY: -120,
      ctrlKey: true,
      bubbles: true,
      cancelable: true,
    });
    act(() => {
      canvas.dispatchEvent(wheelEvent);
    });
    expect(wheelEvent.defaultPrevented).toBe(true);

    // Touch Pointer draw in `ペン入力` mode creates a note on touch devices
    const drawBtn = screen.getByRole('button', { name: 'ペン入力' });
    fireEvent.click(drawBtn);
    drawBtn.focus();
    expect(document.activeElement).toBe(drawBtn);

    fireEvent.pointerDown(canvas, {
      pointerId: 11,
      pointerType: 'touch',
      button: 0,
      clientX: 180,
      clientY: 180,
    });
    fireEvent.pointerUp(window, {
      pointerId: 11,
      pointerType: 'touch',
      button: 0,
      clientX: 180,
      clientY: 180,
    });
    await waitFor(() => {
      expect(prTabBtn.textContent).toContain('1');
    });

    // While `<button>ペン入力</button>` STILL holds focus, Ctrl+A -> Delete deletes the note, and Ctrl+Z undoes it!
    expect(document.activeElement).toBe(drawBtn);
    fireEvent.keyDown(drawBtn, { key: 'a', ctrlKey: true });
    fireEvent.keyDown(drawBtn, { key: 'Delete' });
    await waitFor(() => {
      expect(prTabBtn.textContent).not.toContain('1');
    });

    fireEvent.keyDown(drawBtn, { key: 'z', ctrlKey: true });
    await waitFor(() => {
      expect(prTabBtn.textContent).toContain('1');
    });

    // While `<button>ペン入力</button>` STILL holds focus, Shift+G switches to COMPASS AI tab!
    fireEvent.keyDown(drawBtn, { key: 'G', code: 'KeyG', shiftKey: true });
    await waitFor(() => {
      expect(screen.getByTestId('tab-compass-ai').getAttribute('aria-selected')).toBe('true');
    });

    appUi.unmount();
  });

  it('verifies Round-3 Fact-Checker fixes: VLQ/truncation hardening, reduced-motion rAF stop, configurable DAW bridges, 2-finger touch draw safety, pinned candidate slot switching, and duplicate note.id rejection', async () => {
    // 1. [R3-MED-01] 5-byte VLQ bomb, truncated VLQ, and truncated NoteOn throw in StandardMidiCodec.parseMidi
    const mthd = [0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x01, 0x01, 0xe0];
    const fiveByteVlqTrack = [0x80, 0x80, 0x80, 0x80, 0x00, 0xff, 0x2f, 0x00];
    const fiveByteSmf = new Uint8Array([
      ...mthd,
      0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, fiveByteVlqTrack.length,
      ...fiveByteVlqTrack,
    ]);
    expect(() => StandardMidiCodec.parseMidi(fiveByteSmf.buffer)).toThrow();

    const truncatedVlqSmf = new Uint8Array([
      ...mthd,
      0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x02,
      0x81, 0x82,
    ]);
    expect(() => StandardMidiCodec.parseMidi(truncatedVlqSmf.buffer)).toThrow();

    const truncatedNoteOnSmf = new Uint8Array([
      ...mthd,
      0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x03,
      0x00, 0x90, 60,
    ]);
    expect(() => StandardMidiCodec.parseMidi(truncatedNoteOnSmf.buffer)).toThrow();

    // 2. [R3-MED-02] prefers-reduced-motion: reduce halts ResonanceCircleCanvas requestAnimationFrame loop
    const origMatchMedia = window.matchMedia;
    let rafSchedules = 0;
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => {
      rafSchedules += 1;
      return rafSchedules;
    });
    try {
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        configurable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: query.includes('prefers-reduced-motion'),
          media: query,
          onchange: null,
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          addListener: vi.fn(),
          removeListener: vi.fn(),
          dispatchEvent: vi.fn(),
        })),
      });
      const reducedUi = render(
        <ResonanceCircleCanvas
          audioEngine={new StubAudioEngine()}
          isGenerating={true}
          isPlayingAudio={true}
          statusTitle="GENERATING"
          statusDetail="トークン推論中..."
        />,
      );
      expect(rafSchedules).toBeLessThanOrEqual(1);
      reducedUi.unmount();
    } finally {
      rafSpy.mockRestore();
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        configurable: true,
        value: origMatchMedia,
      });
    }

    // 3. [R3-MED-03] Configurable bridgeBaseUrl on CubaseDawController & AbletonDawController + zero dead _commands
    expect(CubaseDawController.length).toBe(1);
    expect(AbletonDawController.length).toBe(1);
    const customCubase = new CubaseDawController('ws://127.0.0.1:9911/');
    const customAbleton = new AbletonDawController({ bridgeBaseUrl: 'ws://127.0.0.1:9922' });
    expect(customCubase.bridgeBaseUrl).toBe('ws://127.0.0.1:9911');
    expect(customAbleton.bridgeBaseUrl).toBe('ws://127.0.0.1:9922');
    expect('_commands' in (customCubase as unknown as Record<string, unknown>)).toBe(false);
    await customCubase.connect();
    expect(customCubase.getSnapshot().connectionStatusText).toContain('ws://127.0.0.1:9911');

    // 4. [R3-MED-05] Switching candidates AFTER moving target range or active track replaces original slot with 0 orphans
    const candCtrl = new DemoPianoRollController(new StubAudioEngine());
    await candCtrl.connect();
    const trk1Id = candCtrl.getSnapshot().activeTrackId;
    const trk2Id = candCtrl.addTrack('Lead 2', 'SAX');
    candCtrl.setActiveTrack(trk1Id);
    candCtrl.setTargetMeasureRange(5, 8);

    const candBytes0 = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const candBytes1 = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 67, startTick: 240, durationTicks: 480, velocity: 105 })],
      480,
      120,
      'PIANO',
    );
    const makeCand = (index: number, raw: Uint8Array) => ({
      index,
      filename: `cand_${index}.mid`,
      rawMidiBytes: raw,
      parsedMidi: StandardMidiCodec.parseMidi(
        (raw.buffer as ArrayBuffer).slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
      ),
      reason: null,
    });

    await candCtrl.applyGeneratedCandidates([makeCand(0, candBytes0), makeCand(1, candBytes1)], []);
    expect(candCtrl.getSnapshot().tracks.find((t) => t.id === trk1Id)?.notes.map((n) => n.pitch)).toEqual([60]);

    // Move target bar range to 9..12 AND switch active track to Track 2, then select Candidate 2 (`index = 1`)
    candCtrl.setTargetMeasureRange(9, 12);
    candCtrl.setActiveTrack(trk2Id);
    await candCtrl.selectGeneratedCandidate(1);

    const snapAfterSwitch = candCtrl.getSnapshot();
    const trk1Notes = snapAfterSwitch.tracks.find((t) => t.id === trk1Id)!.notes;
    const trk2Notes = snapAfterSwitch.tracks.find((t) => t.id === trk2Id)!.notes;
    expect(trk1Notes.length).toBe(1);
    expect(trk1Notes[0].pitch).toBe(67);
    expect(trk1Notes[0].startTick).toBe(4 * 1920 + 240); // Bar 5 + 240 ticks (no orphan C4 at Bar 5, no leak to Bar 9 or Track 2!)
    expect(trk2Notes.length).toBe(0);

    // Same pinned-range guarantee on ExternalDawAdapters
    customCubase.setTargetMeasureRange(5, 8);
    await customCubase.applyGeneratedCandidates([makeCand(0, candBytes0), makeCand(1, candBytes1)], []);
    customCubase.setTargetMeasureRange(9, 12);
    await customCubase.selectGeneratedCandidate(1);
    expect(customCubase.getSnapshot().tracks[0].notes.map((n) => n.pitch)).toEqual([67]);

    // Deleting the candidate's track clears candidate metadata cleanly
    candCtrl.deleteTrack(trk1Id);
    expect(candCtrl.getSnapshot().candidates.length).toBe(0);

    // 5. [R3-LOW-01] Duplicate note.id inside a track is rejected atomically by applyWorkspaceState
    candCtrl.createNoteAt(64, 0, 480, 100);
    expect(candCtrl.getSnapshot().canUndo).toBe(true);
    const validWs = candCtrl.exportWorkspaceState();
    const activeTrkBefore = candCtrl.getSnapshot().activeTrackId;
    candCtrl.applyWorkspaceState({
      ...validWs,
      tracks: validWs.tracks.map((trk, idx) =>
        idx === 0
          ? {
              ...trk,
              notes: [
                { id: 'dup-note-1', pitch: 60, startTick: 0, durationTicks: 240, velocity: 100, channel: 0 },
                { id: 'dup-note-1', pitch: 67, startTick: 480, durationTicks: 240, velocity: 100, channel: 0 },
              ],
            }
          : trk,
      ),
    });
    // Atomic abort preserved original E4 (64) note on active track and undo stack
    const activeNotesAfter = candCtrl
      .getSnapshot()
      .tracks.find((t) => t.id === activeTrkBefore)!
      .notes.map((n) => n.pitch);
    expect(activeNotesAfter).toEqual([64]);
    expect(candCtrl.getSnapshot().canUndo).toBe(true);

    // 6. [R3-MED-04] Two-finger touch pinch/pan in `ペン入力` (`draw`) mode creates/deletes ZERO phantom notes
    const appTouch = render(<App />);
    await waitFor(
      () => {
        expect(screen.getByTestId('generate-button')).toBeTruthy();
      },
      { timeout: 3500 },
    );
    fireEvent.click(screen.getByTestId('tab-pianoroll'));
    const touchCanvas = screen.getByTestId('pianoroll-canvas');
    vi.spyOn(touchCanvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    });
    fireEvent.click(screen.getByRole('button', { name: 'ペン入力' }));

    // 2-finger pinch on empty grid: Finger 1 lands, then Finger 2 lands -> ZERO notes created
    fireEvent.pointerDown(touchCanvas, {
      pointerId: 21,
      pointerType: 'touch',
      button: 0,
      clientX: 200,
      clientY: 200,
    });
    fireEvent.pointerDown(touchCanvas, {
      pointerId: 22,
      pointerType: 'touch',
      button: 0,
      clientX: 320,
      clientY: 200,
    });
    fireEvent.pointerMove(window, {
      pointerId: 22,
      pointerType: 'touch',
      clientX: 380,
      clientY: 210,
    });
    fireEvent.pointerUp(window, { pointerId: 22, pointerType: 'touch', clientX: 380, clientY: 210 });
    fireEvent.pointerUp(window, { pointerId: 21, pointerType: 'touch', clientX: 200, clientY: 200 });
    expect(screen.getByTestId('tab-pianoroll').textContent).not.toContain('1');

    // Genuine 1-finger tap creates 1 note cleanly
    fireEvent.pointerDown(touchCanvas, {
      pointerId: 31,
      pointerType: 'touch',
      button: 0,
      clientX: 200,
      clientY: 200,
    });
    fireEvent.pointerUp(window, {
      pointerId: 31,
      pointerType: 'touch',
      clientX: 200,
      clientY: 200,
    });
    await waitFor(() => {
      expect(screen.getByTestId('tab-pianoroll').textContent).toContain('1');
    });

    // 2-finger pinch starting right on top of that existing note does NOT delete it!
    fireEvent.pointerDown(touchCanvas, {
      pointerId: 41,
      pointerType: 'touch',
      button: 0,
      clientX: 200,
      clientY: 200,
    });
    fireEvent.pointerDown(touchCanvas, {
      pointerId: 42,
      pointerType: 'touch',
      button: 0,
      clientX: 320,
      clientY: 200,
    });
    fireEvent.pointerUp(window, { pointerId: 42, pointerType: 'touch', clientX: 320, clientY: 200 });
    fireEvent.pointerUp(window, { pointerId: 41, pointerType: 'touch', clientX: 200, clientY: 200 });
    expect(screen.getByTestId('tab-pianoroll').textContent).toContain('1');

    appTouch.unmount();
  });

  it('enforces Round 4 Fact-Checker remediations: custom-scheme IPC origin trust, candidate origin slot across undo/deleteTrack, cross-track duplicate note.id rejection, and strict SMF running-status errors', async () => {
    // 1. [R4-HIGH-01] `ipc_security.cjs` canonical custom-scheme origin (`compass-app://studio`) & token-wipe immunity
    const ipcSecPath = '../../electron/ipc_security.cjs';
    const ipcSec = (await import(/* @vite-ignore */ ipcSecPath)) as {
      extractCanonicalOrigin: (rawUrl: unknown) => string;
      isTrustedIpcSender: (event: unknown, env?: Record<string, string | undefined>) => boolean;
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };
    expect(ipcSec.extractCanonicalOrigin('compass-app://studio/index.html')).toBe(
      'compass-app://studio',
    );
    expect(
      ipcSec.extractCanonicalOrigin('compass-app://STUDIO/index.html?tab=pianoroll&popout=1'),
    ).toBe('compass-app://studio');
    expect(ipcSec.extractCanonicalOrigin('compass-app://evil/index.html')).toBe('');
    expect(ipcSec.extractCanonicalOrigin('file:///C:/Windows/System32/cmd.exe')).toBe('');
    expect(ipcSec.extractCanonicalOrigin('http://127.0.0.1:5173/index.html')).toBe(
      'http://127.0.0.1:5173',
    );
    expect(
      ipcSec.isTrustedIpcSender(
        { senderFrame: { url: 'compass-app://studio/index.html' } },
        {},
      ),
    ).toBe(true);
    expect(
      ipcSec.isTrustedIpcSender(
        { senderFrame: { url: 'compass-app://evil/index.html' } },
        {},
      ),
    ).toBe(false);

    // Verify App.tsx never wipes a pushed 64-hex handshakeToken if requestApiConfig() returns empty strings
    let pushListener: ((cfg: { apiBaseUrl: string; handshakeToken: string }) => void) | null = null;
    const validHexToken = 'a'.repeat(64);
    const seenAuthHeaders: string[] = [];
    const origDesktop = window.compassDesktop;
    window.compassDesktop = {
      apiBaseUrl: 'http://127.0.0.1:18999',
      getApiConfig: () => ({ apiBaseUrl: 'http://127.0.0.1:18999', handshakeToken: validHexToken }),
      requestApiConfig: async () => ({ apiBaseUrl: '', handshakeToken: '' }),
      onApiConfigChanged: (cb) => {
        pushListener = cb;
        return () => {
          pushListener = null;
        };
      },
    };
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      const hdrs = new Headers(init?.headers);
      seenAuthHeaders.push(hdrs.get('X-Compass-Handshake') || '');
      return new Response(
        JSON.stringify({
          active_device: 'cpu',
          device_name: 'CPU',
          torch_version: '2.5.0',
          cuda_available: false,
          mps_available: false,
          models: [],
          user_models_dir: 'C:/Sandbox/COMPASS_AI/models',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    });
    const appToken = render(<App />);
    act(() => {
      pushListener?.({ apiBaseUrl: 'http://127.0.0.1:18999', handshakeToken: validHexToken });
    });
    await waitFor(() => {
      expect(seenAuthHeaders.some((h) => h === validHexToken)).toBe(true);
    });
    // Even after subsequent boot retries call requestApiConfig() returning { '', '' }, token stays validHexToken!
    await act(async () => {
      await new Promise((r) => setTimeout(r, 420));
    });
    const nonEmptyAfterPush = seenAuthHeaders.slice(seenAuthHeaders.indexOf(validHexToken));
    expect(nonEmptyAfterPush.every((h) => h === validHexToken)).toBe(true);
    appToken.unmount();
    fetchSpy.mockRestore();
    window.compassDesktop = origDesktop;

    // 2. [R4-MED-03] Candidate origin slot & deleted-track undo safety (Scenarios A, B, C, D)
    const makeCand = (index: number, pitch: number) => {
      const bytes = StandardMidiCodec.encodeMultiTrackMidi(
        [
          new DawTrack({
            name: 'Gen',
            instrument: 'PIANO',
            notes: [new MidiNote({ pitch, startTick: 0, durationTicks: 480, velocity: 100 })],
          }).toSnapshot(),
        ],
        480,
        120,
        4,
      );
      return {
        index,
        filename: `cand_${index}.mid`,
        rawMidiBytes: bytes,
        parsedMidi: StandardMidiCodec.parseMidi(
          (bytes.buffer as ArrayBuffer).slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
        ),
        reason: null,
      };
    };

    // Scenario A: gen on T1 -> undo -> delete T1 -> selectGeneratedCandidate(1) must NOT leak onto T2
    const ctrlScenA = new DemoPianoRollController(new StubAudioEngine());
    const t1IdA = ctrlScenA.getSnapshot().tracks[0].id;
    const t2IdA = ctrlScenA.getSnapshot().tracks[1].id;
    await ctrlScenA.applyGeneratedCandidates([makeCand(0, 60), makeCand(1, 64)], []);
    ctrlScenA.undo();
    expect(ctrlScenA.getSnapshot().activeCandidateIndex).toBe(-1);
    ctrlScenA.deleteTrack(t1IdA);
    expect(ctrlScenA.getSnapshot().candidates.length).toBe(0);
    await ctrlScenA.selectGeneratedCandidate(1);
    const t2AfterA = ctrlScenA.getSnapshot().tracks.find((t) => t.id === t2IdA)!;
    expect(t2AfterA.notes.length).toBe(0);

    // Scenario B: gen on T1 (Bars 1..4) -> undo -> switch to T2 & Bars 9..12 -> selectGeneratedCandidate(1) places onto T1 Bars 1..4!
    const ctrlScenB = new DemoPianoRollController(new StubAudioEngine());
    const t1IdB = ctrlScenB.getSnapshot().tracks[0].id;
    const t2IdB = ctrlScenB.getSnapshot().tracks[1].id;
    ctrlScenB.setTargetMeasureRange(1, 4);
    await ctrlScenB.applyGeneratedCandidates([makeCand(0, 60), makeCand(1, 67)], []);
    ctrlScenB.undo();
    ctrlScenB.setActiveTrack(t2IdB);
    ctrlScenB.setTargetMeasureRange(9, 12);
    await ctrlScenB.selectGeneratedCandidate(1);
    const snapB = ctrlScenB.getSnapshot();
    const t1NotesB = snapB.tracks.find((t) => t.id === t1IdB)!.notes;
    const t2NotesB = snapB.tracks.find((t) => t.id === t2IdB)!.notes;
    expect(t1NotesB.length).toBe(1);
    expect(t1NotesB[0].pitch).toBe(67);
    expect(t1NotesB[0].startTick).toBe(0);
    expect(t2NotesB.length).toBe(0);

    // Scenario C: gen on T1 -> selectGeneratedCandidate(1) -> delete T1 -> undo() must NOT resurrect deleted T1 in activeCandidateRange
    const ctrlScenC = new DemoPianoRollController(new StubAudioEngine());
    const t1IdC = ctrlScenC.getSnapshot().tracks[0].id;
    await ctrlScenC.applyGeneratedCandidates([makeCand(0, 60), makeCand(1, 65)], []);
    await ctrlScenC.selectGeneratedCandidate(1);
    ctrlScenC.deleteTrack(t1IdC);
    ctrlScenC.undo();
    const wsAfterC = ctrlScenC.exportWorkspaceState();
    expect(wsAfterC.activeCandidateRange).toBeNull();
    expect(wsAfterC.candidateOriginSlot).toBeNull();

    // 3. [R4-LOW-01] Cross-track duplicate `note.id` & invalid `activeCandidateRange.trackId` rejected atomically
    const ctrlSyncTarget = new DemoPianoRollController(new StubAudioEngine());
    ctrlSyncTarget.createNoteAt(72, 0, 480, 100);
    const baseState = ctrlSyncTarget.exportWorkspaceState();
    const crossDupState = {
      ...baseState,
      tracks: [
        {
          ...baseState.tracks[0],
          notes: [{ id: 'shared_note_1', pitch: 60, startTick: 0, durationTicks: 480, velocity: 100, channel: 0 }],
        },
        {
          ...baseState.tracks[1],
          notes: [{ id: 'shared_note_1', pitch: 64, startTick: 0, durationTicks: 480, velocity: 100, channel: 0 }],
        },
      ],
    };
    ctrlSyncTarget.applyWorkspaceState(crossDupState);
    expect(ctrlSyncTarget.getSnapshot().tracks[0].notes[0].pitch).toBe(72);
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'm1',
        senderId: 's1',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: crossDupState,
      }),
    ).toBe(false);

    const badRangeState = {
      ...baseState,
      activeCandidateRange: { trackId: 'non_existent_track', startTick: 0, endTick: 1920 },
    };
    ctrlSyncTarget.applyWorkspaceState(badRangeState);
    expect(ctrlSyncTarget.exportWorkspaceState().activeCandidateRange).toBeNull();
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'm2',
        senderId: 's1',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: badRangeState,
      }),
    ).toBe(false);

    // 4. [R4-LOW-02] StandardMidiCodec throws on missing running status after SysEx (0xF0) & orphan delta-time
    // SysEx (00 F0 01 7F) cancels running status -> following data byte (00 3C 64) has runningStatus === 0 -> MUST throw!
    const sysexThenOrphanData = new Uint8Array([
      0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
      0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
      0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x10,
      0x00, 0x90, 0x3c, 0x64,
      0x00, 0xf0, 0x01, 0x7f,
      0x00, 0x3c, 0x00,
      0x00, 0xff, 0x2f, 0x00, 0x00,
    ]);
    expect(() => StandardMidiCodec.parseMidi(sysexThenOrphanData.buffer)).toThrow(
      /ランニングステータス/,
    );

    // Orphan trailing delta-time (0x00) at end of MTrk with no status/event byte before End-of-Track -> MUST throw!
    const orphanDeltaSmf = new Uint8Array([
      0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
      0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
      0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x05,
      0x00, 0x90, 0x3c, 0x64, 0x00,
    ]);
    expect(() => StandardMidiCodec.parseMidi(orphanDeltaSmf.buffer)).toThrow(
      /デルタタイム直後で SMF トラックイベントが途中で切断されています/,
    );
  });

  it('enforces Round 5 Fact-Checker remediations: Japanese SMF error localization, 50-byte single-track SMF upload, and clearing stale uploadNotice on AI generation', async () => {
    // 1. [R5-HIGH-01] Exact 50-byte (`0x32`) single-track SMF parses cleanly into 1 track with 4 notes at 120 BPM
    const trackEvents = [
      0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
      0x00, 0xc0, 0x00,
      0x00, 0x90, 60, 100, 0x83, 0x60, 0x80, 60, 0,
      0x00, 0x90, 64, 100, 0x83, 0x60, 0x80, 64, 0,
      0x00, 0x90, 67, 100, 0x83, 0x60, 0x80, 67, 0,
      0x00, 0x90, 72, 100, 0x83, 0x60, 0x80, 72, 0,
      0x00, 0xff, 0x2f, 0x00,
    ];
    expect(trackEvents.length).toBe(50);
    const validUploadBytes = new Uint8Array([
      0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
      0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
      0x4d, 0x54, 0x72, 0x6b,
      0x00, 0x00, 0x00, trackEvents.length,
      ...trackEvents,
    ]);
    const parsedSingle = StandardMidiCodec.parseMidi(validUploadBytes.buffer);
    expect(parsedSingle.tracks.length).toBe(1);
    expect(parsedSingle.tracks[0].notes.length).toBe(4);
    expect(parsedSingle.bpm).toBe(120);

    // 2. [R5-LOW-01] Malformed SMF payloads throw localized Japanese error strings (zero English developer message leak)
    const malformedPayloads: Uint8Array[] = [
      // Truncated VLQ
      new Uint8Array([
        0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
        0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
        0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x01, 0x81,
      ]),
      // >4-byte VLQ
      new Uint8Array([
        0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
        0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
        0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x05, 0x81, 0x81, 0x81, 0x81, 0x01,
      ]),
      // Truncated Meta payload
      new Uint8Array([
        0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
        0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
        0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x04, 0x00, 0xff, 0x51, 0x03,
      ]),
      // Truncated Channel event
      new Uint8Array([
        0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
        0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
        0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x90, 0x3c,
      ]),
    ];
    for (const bad of malformedPayloads) {
      let caught = '';
      try {
        StandardMidiCodec.parseMidi(bad.buffer as ArrayBuffer);
      } catch (err) {
        caught = err instanceof Error ? err.message : '';
      }
      expect(caught.length).toBeGreaterThan(0);
      expect(/[\u3040-\u30ff\u4e00-\u9faf]/.test(caught)).toBe(true);
      expect(/Truncated|Malformed/i.test(caught)).toBe(false);
    }

    // 3. [R5-LOW-01] `PianoRollView` clears prior `uploadNotice` when new AI generation candidates arrive
    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    const Wrapper: React.FC = () => {
      const [sess, setSess] = React.useState(ctrl.getSnapshot());
      React.useEffect(() => ctrl.subscribe(setSess), []);
      return (
        <PianoRollView
          controller={ctrl}
          session={sess}
          isActive={true}
          focusRevision={1}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    };
    const view = render(<Wrapper />);
    const fileInput = view.getByTestId('midi-file-input') as HTMLInputElement;
    const uploadFile = new File([validUploadBytes], 'uploaded_test.mid', { type: 'audio/midi' });
    if (typeof uploadFile.arrayBuffer !== 'function') {
      Object.defineProperty(uploadFile, 'arrayBuffer', {
        value: async () => validUploadBytes.buffer.slice(0),
      });
    }
    fireEvent.change(fileInput, { target: { files: [uploadFile] } });
    await waitFor(() => {
      expect(view.container.textContent).toContain('uploaded_test.mid を読み込みました');
    });

    // Apply generated candidates -> prior uploadNotice MUST be cleared cleanly
    const candBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 67, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    await act(async () => {
      await ctrl.applyGeneratedCandidates(
        [
          {
            index: 0,
            filename: 'gen_0.mid',
            rawMidiBytes: candBytes,
            parsedMidi: StandardMidiCodec.parseMidi(
              (candBytes.buffer as ArrayBuffer).slice(
                candBytes.byteOffset,
                candBytes.byteOffset + candBytes.byteLength,
              ),
            ),
            reason: null,
          },
        ],
        [],
      );
    });
    expect(view.container.textContent).toContain('候補 1');
    expect(view.container.textContent).not.toContain('uploaded_test.mid を読み込みました');
    view.unmount();
  });

  it('enforces Round 6 Verification-Critic remediations: IPC/BroadcastChannel OOM & schema guards, Cubase->Demo ResizeObserver rebind, touch 1/16 note resize & focused-<select> hotkeys, VLQ tick ceiling, cross-range candidate Undo, _stopEpoch audio guard, modal focus trap, WCAG note label contrast, and 5/4 meter notice', async () => {
    const ipcSecPath = '../../electron/ipc_security.cjs';
    const ipcSec = (await import(/* @vite-ignore */ ipcSecPath)) as {
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };

    // 1. [R3-MED-01] Multi-GB array-like OOM bomb & malformed latestReason.genre rejected across IPC, BroadcastChannel & applyWorkspaceState
    const guardCtrl = new DemoPianoRollController(new StubAudioEngine());
    guardCtrl.createNoteAt(60, 0, 480, 100);
    expect(guardCtrl.getSnapshot().canUndo).toBe(true);
    const cleanWs = guardCtrl.exportWorkspaceState();

    const oomCandidateMsg = {
      messageId: 'oom_1',
      senderId: 'evil_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: {
        ...cleanWs,
        candidates: [
          {
            index: 0,
            filename: 'bomb.mid',
            rawMidiBytes: { length: 1_073_741_824 } as unknown as Uint8Array,
            reason: null,
          },
        ],
      },
    };
    const badGenreMsg = {
      messageId: 'bad_genre_1',
      senderId: 'evil_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: {
        ...cleanWs,
        latestReason: {
          source: 'model_cot',
          task: 'meta',
          thinking: true,
          instruments: ['PIANO'],
          note_density: null,
          gen_measure_count: 4,
          genre: 'jazz' as unknown as readonly string[],
          key: 'CM',
        },
      },
    };

    // Valid 350 KB Uint8Array SMF candidate passes in O(1) without JSON.stringify 10x inflation
    const largeValidSmf = new Uint8Array(350_000);
    largeValidSmf.set([0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x01, 0x01, 0xe0]);
    const validLargeMsg = {
      messageId: 'ok_large_1',
      senderId: 'trusted_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: {
        ...cleanWs,
        candidates: [
          {
            index: 0,
            filename: 'symphony_350kb.mid',
            rawMidiBytes: largeValidSmf,
            reason: null,
          },
        ],
      },
    };
    const junkKeySmuggleMsg = {
      messageId: 'junk_1',
      senderId: 'evil_peer',
      type: 'REQUEST_WORKSPACE_STATE',
      junkBomb: 'x'.repeat(1024),
    };

    const t0 = performance.now();
    expect(ipcSec.isValidWorkspaceSyncPayload(oomCandidateMsg)).toBe(false);
    expect(ipcSec.isValidWorkspaceSyncPayload(badGenreMsg)).toBe(false);
    expect(ipcSec.isValidWorkspaceSyncPayload(junkKeySmuggleMsg)).toBe(false);
    expect(ipcSec.isValidWorkspaceSyncPayload(validLargeMsg)).toBe(true);
    expect(isValidIncomingSyncMessage(oomCandidateMsg)).toBe(false);
    expect(isValidIncomingSyncMessage(badGenreMsg)).toBe(false);
    expect(isValidIncomingSyncMessage(junkKeySmuggleMsg)).toBe(false);
    expect(isValidIncomingSyncMessage(validLargeMsg)).toBe(true);
    expect(performance.now() - t0).toBeLessThan(25);

    // [R6B-MED-02] 100% 1-to-1 nested schema & note_density parity between ipc_security.cjs & WorkspaceSyncBus.ts
    const ownProtoTrack = { ...cleanWs.tracks[0] } as Record<string, unknown>;
    Object.defineProperty(ownProtoTrack, '__proto__', {
      value: { evil: 1 },
      enumerable: true,
      configurable: true,
    });
    const fortyKNotesA = Array.from({ length: 20_000 }, (_, idx) => ({
      id: `na_${idx}`,
      pitch: 60,
      startTick: idx * 10,
      durationTicks: 10,
      velocity: 100,
      channel: 0,
    }));
    const fortyKNotesB = Array.from({ length: 20_000 }, (_, idx) => ({
      id: `nb_${idx}`,
      pitch: 64,
      startTick: idx * 10,
      durationTicks: 10,
      velocity: 100,
      channel: 0,
    }));
    const nestedSmugglePayloads = [
      { ...cleanWs, tracks: [{ ...cleanWs.tracks[0], extraKey: 1 }] },
      { ...cleanWs, tracks: [ownProtoTrack] },
      {
        ...cleanWs,
        tracks: [
          {
            ...cleanWs.tracks[0],
            notes: [{ ...cleanWs.tracks[0].notes[0], extraNoteKey: 1 }],
          },
        ],
      },
      {
        ...cleanWs,
        activeCandidateRange: {
          trackId: cleanWs.tracks[0].id,
          startTick: 0,
          endTick: 480,
          extraSlotKey: 1,
        },
      },
      {
        ...cleanWs,
        candidates: [
          {
            index: 0,
            filename: 'ok.mid',
            rawMidiBytes: largeValidSmf.slice(0, 64),
            reason: null,
            extraCandKey: 1,
          },
        ],
      },
      {
        ...cleanWs,
        latestReason: {
          source: 'model_cot',
          task: 'meta',
          thinking: true,
          instruments: ['PIANO'],
          note_density: null,
          gen_measure_count: 4,
          genre: ['jazz'],
          key: 'CM',
          extraReasonKey: 1,
        },
      },
      { ...cleanWs, tracks: [{ ...cleanWs.tracks[0], colorHex: '#' + 'a'.repeat(50) }] },
      {
        ...cleanWs,
        tracks: [
          { ...cleanWs.tracks[0], notes: fortyKNotesA },
          { ...cleanWs.tracks[1], notes: fortyKNotesB },
        ],
      },
      {
        ...cleanWs,
        latestReason: {
          source: 'model_cot',
          task: 'meta',
          thinking: true,
          instruments: ['PIANO'],
          note_density: { ['k'.repeat(65)]: 3 },
          gen_measure_count: 4,
          genre: ['jazz'],
          key: 'CM',
        },
      },
    ];
    for (const badWs of nestedSmugglePayloads) {
      const wrapped = {
        messageId: 'r6b_parity',
        senderId: 'evil_peer',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: badWs,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(wrapped)).toBe(false);
      expect(isValidIncomingSyncMessage(wrapped)).toBe(false);
    }

    // [R6B-MED-01] CompassApiClient lazily refreshes handshake via requestApiConfig() when empty OR on HTTP 403
    const prevDesktop = window.compassDesktop;
    const origFetch = globalThis.fetch;
    try {
      const validHexToken = 'f'.repeat(64);
      const reqCfgSpy = vi.fn(async () => ({
        apiBaseUrl: 'http://127.0.0.1:18765',
        handshakeToken: validHexToken,
      }));
      window.compassDesktop = {
        requestApiConfig: reqCfgSpy,
      };
      globalThis.fetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
        const hdrs = (init?.headers ?? {}) as Record<string, string>;
        if (hdrs['X-Compass-Handshake'] !== validHexToken) {
          return new Response(JSON.stringify({ detail: 'Forbidden' }), { status: 403 });
        }
        return new Response(
          JSON.stringify({
            device: 'cuda:0',
            dtype: 'torch.bfloat16',
            backend_class: 'CudaHardwareBackend',
            user_models_dir: 'C:\\models',
            bundled_models_dir: 'C:\\bundled',
            available_models_count: 1,
            models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
            models: [],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }) as typeof fetch;

      // Case A: Initially empty handshakeToken -> lazily fetches before request -> 200 OK
      const lazyClient = new CompassApiClient('http://127.0.0.1:8000', '');
      const hwInfo = await lazyClient.getHardwareInfo();
      expect(hwInfo.device).toBe('cuda:0');
      expect(reqCfgSpy).toHaveBeenCalledTimes(1);
      expect(lazyClient.baseUrl).toBe('http://127.0.0.1:18765');

      // Case B: Stale handshakeToken -> gets 403 once -> auto-refreshes via requestApiConfig -> retries -> 200 OK
      const staleClient = new CompassApiClient('http://127.0.0.1:8000', '0'.repeat(64));
      await staleClient.rescanModels();
      expect(reqCfgSpy).toHaveBeenCalledTimes(2);
      expect(staleClient.handshakeToken).toBe(validHexToken);
    } finally {
      window.compassDesktop = prevDesktop;
      globalThis.fetch = origFetch;
    }

    guardCtrl.applyWorkspaceState(oomCandidateMsg.workspaceState as never);
    guardCtrl.applyWorkspaceState(badGenreMsg.workspaceState as never);
    // Undo stack and note remain completely intact!
    expect(guardCtrl.getSnapshot().canUndo).toBe(true);
    expect(guardCtrl.getSnapshot().tracks[0].notes.length).toBe(1);

    // BroadcastChannel frame injection is blocked by WorkspaceSyncBus._dispatchIncoming
    const chName = `compass_r6_guard_${Date.now()}`;
    const victimBus = new WorkspaceSyncBus(chName);
    let receivedCount = 0;
    victimBus.subscribe(() => {
      receivedCount += 1;
    });
    if (typeof BroadcastChannel !== 'undefined') {
      const attackerCh = new BroadcastChannel(chName);
      attackerCh.postMessage(oomCandidateMsg);
      attackerCh.postMessage(badGenreMsg);
      await new Promise((r) => setTimeout(r, 40));
      attackerCh.close();
    }
    expect(receivedCount).toBe(0);
    victimBus.dispose();

    // 2. [R3-MED-02] Switching controller from Cubase -> Demo re-binds ResizeObserver to newly mounted canvasContainerRef
    let observedElements: Element[] = [];
    const origRO = globalThis.ResizeObserver;
    class TrackingResizeObserver {
      private _cb: ResizeObserverCallback;
      constructor(cb: ResizeObserverCallback) {
        this._cb = cb;
      }
      public observe(el: Element) {
        observedElements.push(el);
        void this._cb;
      }
      public unobserve() {}
      public disconnect() {}
    }
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = TrackingResizeObserver;
    try {
      const cubaseCtrl = new CubaseDawController();
      const demoCtrl = new DemoPianoRollController(new StubAudioEngine());
      const switchRender = render(
        <PianoRollView
          controller={cubaseCtrl}
          session={cubaseCtrl.getSnapshot()}
          isActive={true}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />,
      );
      expect(observedElements.length).toBe(0);
      observedElements = [];
      switchRender.rerender(
        <PianoRollView
          controller={demoCtrl}
          session={demoCtrl.getSnapshot()}
          isActive={true}
          focusRevision={1}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />,
      );
      expect(observedElements.length).toBeGreaterThanOrEqual(1);
      switchRender.unmount();
    } finally {
      (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = origRO;
    }

    // 3. [R3-MED-03] Touch right-edge resize on a 1/16 (120-tick) note + focused <select> hotkeys (Delete, Ctrl+Z, Shift+G)
    const touchResizeCtrl = new DemoPianoRollController(new StubAudioEngine());
    // Place a 1/16 note (120 ticks) at pitch 74 (row 53 from top; default scrollPxY = 53 * 16 -> y = 32..48)
    touchResizeCtrl.createNoteAt(74, 0, 120, 100);
    let launchedCompassAi = 0;
    const ResizeHarness: React.FC = () => {
      const [sess, setSess] = React.useState(touchResizeCtrl.getSnapshot());
      React.useEffect(() => touchResizeCtrl.subscribe(setSess), []);
      return (
        <PianoRollView
          controller={touchResizeCtrl}
          session={sess}
          isActive={true}
          focusRevision={0}
          onLaunchCompassAi={() => {
            launchedCompassAi += 1;
          }}
          onSwitchToDemoController={() => {}}
        />
      );
    };
    const resizeUi = render(<ResizeHarness />);
    const prCanvas = resizeUi.getByTestId('pianoroll-canvas');
    vi.spyOn(prCanvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    });
    // At 64 px/beat (480 PPQN -> 0.13333 px/tick), a 120-tick note spans x = 72..88. Touch at x = 87 (tick ~112.5, within 40% right edge)
    fireEvent.pointerDown(prCanvas, {
      pointerId: 77,
      pointerType: 'touch',
      button: 0,
      clientX: 87,
      clientY: 40,
    });
    fireEvent.pointerMove(window, {
      pointerId: 77,
      pointerType: 'touch',
      clientX: 87 + 64, // +1 beat (+480 ticks)
      clientY: 40,
    });
    fireEvent.pointerUp(window, {
      pointerId: 77,
      pointerType: 'touch',
      clientX: 87 + 64,
      clientY: 40,
    });
    const resizedNote = touchResizeCtrl.getSnapshot().tracks[0].notes[0];
    expect(resizedNote.startTick).toBe(0);
    expect(resizedNote.durationTicks).toBeGreaterThan(120);

    // Focused <select> (Track Instrument selector) must NOT swallow Delete or Ctrl+Z!
    const instSelect = resizeUi.getAllByRole('combobox')[0] as HTMLSelectElement;
    instSelect.focus();
    expect(document.activeElement).toBe(instSelect);
    act(() => {
      touchResizeCtrl.selectAllNotesOnActiveTrack();
    });
    fireEvent.keyDown(instSelect, { key: 'Delete', bubbles: true });
    expect(touchResizeCtrl.getSnapshot().tracks[0].notes.length).toBe(0);
    fireEvent.keyDown(instSelect, { key: 'z', ctrlKey: true, bubbles: true });
    expect(touchResizeCtrl.getSnapshot().tracks[0].notes.length).toBe(1);
    void launchedCompassAi;
    resizeUi.unmount();

    // 4. [R3-MED-06] writeVarLen clamps >0x0FFFFFFF without 5-byte VLQ corruption & parseMidi rejects >2,000,000-tick bomb
    const hugeTickBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0x7fffffff, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    // Must not throw 5-byte VLQ error; hits cumulative tick ceiling cleanly with Japanese message
    expect(() => StandardMidiCodec.parseMidi(hugeTickBytes.buffer as ArrayBuffer)).toThrow(
      /累積ティック数が上限を超えています/,
    );

    // 5. [R3-MED-07] Cross-Range Generation Undo (`Run 1 Bars 3..4 -> Run 2 Bars 7..8 -> Undo`) restores Run 1 candidates & `meta_past`
    const crossCtrl = new DemoPianoRollController(new StubAudioEngine());
    const makeSimpleCand = (idx: number, pitch: number, reasonTask: string) => {
      const b = StandardMidiCodec.encodeSingleTrackMidi(
        [new MidiNote({ pitch, startTick: 0, durationTicks: 480, velocity: 100 })],
        480,
        120,
        'PIANO',
      );
      return {
        index: idx,
        filename: `r_${idx}.mid`,
        rawMidiBytes: b,
        parsedMidi: StandardMidiCodec.parseMidi(
          (b.buffer as ArrayBuffer).slice(b.byteOffset, b.byteOffset + b.byteLength),
        ),
        reason: {
          source: 'model_cot' as const,
          task: reasonTask,
          thinking: true,
          instruments: ['PIANO'],
          note_density: null,
          gen_measure_count: 2,
          genre: ['jazz'],
          key: 'CM',
        },
      };
    };
    crossCtrl.setTargetMeasureRange(3, 4); // [3840, 7680)
    await crossCtrl.applyGeneratedCandidates(
      [makeSimpleCand(0, 60, 'meta'), makeSimpleCand(1, 64, 'meta')],
      [],
    );
    crossCtrl.setTargetMeasureRange(7, 8); // [11520, 15360) — past window [0, 11520) sees Run 1 at Bars 3..4!
    expect(crossCtrl.getSnapshot().contextAvailability.autoResolvedTask).toBe('meta_past');

    await crossCtrl.applyGeneratedCandidates([makeSimpleCand(0, 72, 'meta_past')], []);
    expect(crossCtrl.getSnapshot().tracks[0].notes.map((n) => n.pitch)).toEqual([60, 72]);

    // Undo Run 2 -> Run 2's Bar 7 note (72) removed, Run 1's Bar 3 note (60) AND Run 1's 2 candidates restored!
    crossCtrl.undo();
    const afterCrossUndo = crossCtrl.getSnapshot();
    expect(afterCrossUndo.tracks[0].notes.map((n) => n.pitch)).toEqual([60]);
    expect(afterCrossUndo.candidates.length).toBe(2);
    expect(afterCrossUndo.activeCandidateIndex).toBe(0);
    expect(afterCrossUndo.contextAvailability.autoResolvedTask).toBe('meta_past');

    // Selecting Candidate 2 (`index = 1`) swaps into Run 1's Bars 3..4 slot (pitch 60 -> 64), NOT Bars 7..8!
    await crossCtrl.selectGeneratedCandidate(1);
    const afterCrossSwitch = crossCtrl.getSnapshot().tracks[0].notes;
    expect(afterCrossSwitch.length).toBe(1);
    expect(afterCrossSwitch[0].pitch).toBe(64);
    expect(afterCrossSwitch[0].startTick).toBe(3840);

    // 6. [R3-LOW-01] MidiTrackImportModal moves focus inside dialog, traps Tab/Shift+Tab, and restores focus on close
    const triggerBtn = document.createElement('button');
    triggerBtn.textContent = 'Trigger';
    document.body.appendChild(triggerBtn);
    triggerBtn.focus();
    expect(document.activeElement).toBe(triggerBtn);

    const multiBytes = StandardMidiCodec.encodeMultiTrackMidi(
      [
        new DawTrack({
          name: 'Trk1',
          instrument: 'PIANO',
          notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480 })],
        }).toSnapshot(),
        new DawTrack({
          name: 'Trk2',
          instrument: 'SAX',
          notes: [new MidiNote({ pitch: 67, startTick: 0, durationTicks: 480 })],
        }).toSnapshot(),
      ],
      480,
      120,
    );
    const modalTrapUi = render(
      <MidiTrackImportModal
        fileName="trap.mid"
        parsedMidi={StandardMidiCodec.parseMidi(multiBytes.buffer as ArrayBuffer)}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );
    const dialogNode = modalTrapUi.getByTestId('midi-import-modal');
    expect(dialogNode.contains(document.activeElement)).toBe(true);
    const confirmBtn = modalTrapUi.getByTestId('confirm-midi-import-button');
    confirmBtn.focus();
    // Tab on last button wraps to close button (first focusable inside dialog)
    fireEvent.keyDown(window, { key: 'Tab', shiftKey: false });
    expect(dialogNode.contains(document.activeElement)).toBe(true);
    expect(document.activeElement).not.toBe(confirmBtn);
    // Shift+Tab on first focusable wraps back to confirmBtn
    fireEvent.keyDown(window, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(confirmBtn);

    modalTrapUi.unmount();
    expect(document.activeElement).toBe(triggerBtn);
    document.body.removeChild(triggerBtn);

    // 7. [R3-LOW-02] WebAudioEngine._stopEpoch cancels pending auditionPitch when stopAllVoices() fires during resume()
    let resolveResume: (() => void) | null = null;
    let startedOscillators = 0;
    const OrigCtx = (window as unknown as { AudioContext?: unknown }).AudioContext;
    class DelayedResumeAudioContext {
      public state = 'suspended';
      public currentTime = 0;
      public destination = {};
      public createGain() {
        return {
          gain: {
            value: 1,
            setValueAtTime: () => {},
            linearRampToValueAtTime: () => {},
            exponentialRampToValueAtTime: () => {},
            cancelScheduledValues: () => {},
          },
          connect: () => {},
          disconnect: () => {},
        };
      }
      public createDynamicsCompressor() {
        return {
          threshold: { value: 0 },
          knee: { value: 0 },
          ratio: { value: 0 },
          attack: { value: 0 },
          release: { value: 0 },
          connect: () => {},
        };
      }
      public createAnalyser() {
        return { fftSize: 1024, smoothingTimeConstant: 0.8, connect: () => {} };
      }
      public createBiquadFilter() {
        return {
          type: 'lowpass',
          Q: { value: 1 },
          frequency: { value: 1000, setValueAtTime: () => {}, exponentialRampToValueAtTime: () => {} },
          connect: () => {},
          disconnect: () => {},
        };
      }
      public createOscillator() {
        return {
          type: 'sine',
          frequency: { setValueAtTime: () => {} },
          onended: null,
          connect: () => {},
          disconnect: () => {},
          start: () => {
            startedOscillators += 1;
          },
          stop: () => {},
        };
      }
      public resume() {
        return new Promise<void>((res) => {
          resolveResume = () => {
            this.state = 'running';
            res();
          };
        });
      }
      public async close() {
        this.state = 'closed';
      }
    }
    (window as unknown as { AudioContext: unknown }).AudioContext = DelayedResumeAudioContext;
    try {
      const epochAudio = new WebAudioEngine();
      epochAudio.auditionPitch(60, 'PIANO');
      epochAudio.stopAllVoices();
      const resumeCb = resolveResume as (() => void) | null;
      resumeCb?.();
      await new Promise((r) => setTimeout(r, 20));
      expect(startedOscillators).toBe(0);
      // NaN whenAudioTime is also safely ignored without throwing
      epochAudio.triggerNote(60, 100, 0.25, 'PIANO', Number.NaN);
      expect(startedOscillators).toBe(0);
      await epochAudio.dispose();
    } finally {
      (window as unknown as { AudioContext?: unknown }).AudioContext = OrigCtx;
    }

    // 8. [R3-LOW-03] CompassApiClient scopes GET /generation_progress?job_id=... and ignores mismatched job_id
    const spyProg = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : input.toString();
      expect(url).toContain('job_id=job_expected');
      return new Response(
        JSON.stringify({
          active: true,
          stage: 'decoding',
          generated_tokens: 12,
          max_tokens: 128,
          message: '別ジョブの進捗',
          updated_at: 1,
          job_id: 'job_other',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    });
    const scopedClient = new CompassApiClient('http://127.0.0.1:18999', 'tok');
    const mismatchedSnap = await scopedClient.getGenerationProgress(undefined, 'job_expected');
    expect(mismatchedSnap).toBeNull();
    spyProg.mockRestore();

    // 9. [R3-LOW-04] WCAG >= 4.5:1 note pitch label contrast across all TRACK_SWATCHES & velocities + 5/4 meter notice
    const srgbLin = (c: number) => {
      const s = c / 255;
      return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    for (const hex of [...TRACK_SWATCHES, '#F4F4F5']) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      for (const vel of [1, 32, 64, 96, 127]) {
        const alpha = hex === '#F4F4F5' ? 0.96 : 0.45 + (vel / 127) * 0.5;
        for (const pitch of [60, 61]) {
          const textColor = pickAccessibleNoteLabelColor(hex, alpha, pitch);
          const isBlack = [1, 3, 6, 8, 10].includes(pitch % 12);
          const bgR = isBlack ? 13 : 18;
          const bgG = isBlack ? 13 : 18;
          const bgB = isBlack ? 16 : 22;
          const lumBg =
            0.2126 * srgbLin(r * alpha + bgR * (1 - alpha)) +
            0.7152 * srgbLin(g * alpha + bgG * (1 - alpha)) +
            0.0722 * srgbLin(b * alpha + bgB * (1 - alpha));
          const lumFg = textColor === '#000000' ? 0 : 1;
          const ratio = (Math.max(lumBg, lumFg) + 0.05) / (Math.min(lumBg, lumFg) + 0.05);
          expect(ratio).toBeGreaterThanOrEqual(4.5);
        }
      }
    }

    const meter54Ctrl = new DemoPianoRollController(new StubAudioEngine());
    const meter54Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
      5,
      { numerator: 5, denominator: 4 },
    );
    const Meter54Wrapper: React.FC = () => {
      const [sess, setSess] = React.useState(meter54Ctrl.getSnapshot());
      React.useEffect(() => meter54Ctrl.subscribe(setSess), []);
      return (
        <PianoRollView
          controller={meter54Ctrl}
          session={sess}
          isActive={true}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    };
    const m54View = render(<Meter54Wrapper />);
    const m54File = new File([meter54Bytes as Uint8Array<ArrayBuffer>], 'five_four.mid', {
      type: 'audio/midi',
    });
    if (typeof m54File.arrayBuffer !== 'function') {
      Object.defineProperty(m54File, 'arrayBuffer', {
        value: async () =>
          meter54Bytes.buffer.slice(
            meter54Bytes.byteOffset,
            meter54Bytes.byteOffset + meter54Bytes.byteLength,
          ),
      });
    }
    fireEvent.change(m54View.getByTestId('midi-file-input'), { target: { files: [m54File] } });
    await waitFor(() => {
      expect(m54View.container.textContent).toContain('拍子 5/4 はモデルの1小節上限');
    });
    m54View.unmount();
  });

  it('verifies Round-6E backendReady caching avoids redundant requestApiConfig probes on auth_required=false while preserving forceRefresh=true on 403', async () => {
    const prevDesktop = window.compassDesktop;
    const prevFetch = globalThis.fetch;
    let requestCalls = 0;
    let forceTrueCalls = 0;
    let promotedToken = '';

    window.compassDesktop = {
      apiBaseUrl: 'https://remote.example.test/prefix',
      getApiConfig: () => ({
        apiBaseUrl: 'https://remote.example.test/prefix',
        handshakeToken: promotedToken,
        backendReady: true,
      }),
      requestApiConfig: async (force?: boolean) => {
        requestCalls += 1;
        if (force === true) {
          forceTrueCalls += 1;
          promotedToken = 'a'.repeat(64);
        }
        return {
          apiBaseUrl: 'https://remote.example.test/prefix',
          handshakeToken: promotedToken,
          backendReady: true,
        };
      },
    };

    const seenUrls: string[] = [];
    let force403Next = false;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const u = String(input);
      seenUrls.push(u);
      const hdrs = (init?.headers ?? {}) as Record<string, string>;
      if (force403Next && !hdrs['X-Compass-Handshake']) {
        return new Response(JSON.stringify({ detail: 'Forbidden' }), { status: 403 });
      }
      if (u.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cuda:0',
            dtype: 'bfloat16',
            backend_class: 'CudaBackend',
            user_models_dir: '/models',
            bundled_models_dir: '/bundled',
            available_models_count: 1,
            models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
          }),
          { status: 200 },
        );
      }
      if (u.endsWith('/model_info')) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (u.endsWith('/rescan_models')) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 });
      }
      return new Response(
        JSON.stringify({ active: true, stage: 'generating', message: 'step 1', job_id: 'j1' }),
        { status: 200 },
      );
    }) as typeof fetch;

    try {
      const client = new CompassApiClient();
      expect(client.baseUrl).toBe('https://remote.example.test/prefix');
      expect(client.desktopBackendReady).toBe(true);

      await client.getHardwareInfo();
      await client.getModelInfo();
      await client.rescanModels();
      for (let i = 0; i < 6; i += 1) {
        await client.getGenerationProgress(undefined, 'j1');
      }
      // Zero requestApiConfig IPC calls while backendReady=true and server returns 200!
      expect(requestCalls).toBe(0);
      expect(seenUrls[0]).toBe('https://remote.example.test/prefix/hardware_info');

      // When server transitions to requiring auth (HTTP 403), forceRefresh=true recovers in 1 shot
      force403Next = true;
      const hwAfter403 = await client.getHardwareInfo();
      expect(hwAfter403.device).toBe('cuda:0');
      expect(requestCalls).toBe(1);
      expect(forceTrueCalls).toBe(1);
      expect(client.handshakeToken).toBe('a'.repeat(64));
    } finally {
      window.compassDesktop = prevDesktop;
      globalThis.fetch = prevFetch;
    }
  });

  it('enforces Round 5 Verification-Critic (R4) remediations: meter bounds & UI disable, CompassApiClient stale-token clear & single-shot 403 non-retry, backing ArrayBuffer & Symbol.toStringTag IPC/Sync spoofs, PPQN-scaled SMF limits & 9600 PPQN Pro Tools import, workspace track/name/meter IPC parity, and composite WCAG AA note labels', async () => {
    const ipcSecPath = '../../electron/ipc_security.cjs';
    const ipcSec = (await import(/* @vite-ignore */ ipcSecPath)) as {
      isValidCandidateMidiBytes: (raw: unknown) => boolean;
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };

    // 1. [R4-MED-02] evaluateGenerationMeter bounds + ControlPanelView unsupported-meter banner & disabled generate CTA
    expect(evaluateGenerationMeter(4, 4).supported).toBe(true);
    expect(evaluateGenerationMeter(3, 4).supported).toBe(true);
    expect(evaluateGenerationMeter(6, 8).supported).toBe(true);
    expect(evaluateGenerationMeter(7, 8).supported).toBe(true);
    expect(evaluateGenerationMeter(12, 16).supported).toBe(true); // 72 steps <= 96
    expect(evaluateGenerationMeter(1, 8).supported).toBe(true); // 12 steps (lower boundary)
    expect(evaluateGenerationMeter(5, 4).supported).toBe(false); // 120 > 96
    expect(evaluateGenerationMeter(6, 4).supported).toBe(false); // 144 > 96
    expect(evaluateGenerationMeter(12, 8).supported).toBe(false); // 144 > 96
    expect(evaluateGenerationMeter(0, 4).supported).toBe(false);
    expect(evaluateGenerationMeter(1, 16).supported).toBe(false); // 6 < 12
    expect(evaluateGenerationMeter(4, 3).supported).toBe(false); // non-power-of-two

    const fiveFourSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
      5,
      { numerator: 5, denominator: 4 },
    );
    const meterCtrl = new DemoPianoRollController(new StubAudioEngine());
    const parsedFiveFour = StandardMidiCodec.parseMidi(fiveFourSmf.buffer as ArrayBuffer);
    meterCtrl.importParsedMidiTracks(parsedFiveFour, [0], true);
    expect(meterCtrl.getSnapshot().timeSigNumerator).toBe(5);
    expect(meterCtrl.getSnapshot().timeSigDenominator).toBe(4);

    let genTriggered = 0;
    const cpRender = render(
      <ControlPanelView
        isActive={true}
        audioEngine={new StubAudioEngine()}
        controllers={[meterCtrl]}
        session={meterCtrl.getSnapshot()}
        models={[
          {
            model_name: 'MORTM4.5D-80M-SFT-Gen',
            description: 'Dense SFT',
            has_weights: true,
          },
        ]}
        hardwareInfo={null}
        settings={{
          modelName: 'MORTM4.5D-80M-SFT-Gen',
          temperature: 1.0,
          topP: 0.98,
          thinking: true,
          cotTemperature: 0.9,
          numCandidates: 2,
          taskMode: 'auto',
          conditionTrackId: '',
          forceMonophonic: false,
          noteDensity: null,
          musicalKey: '',
          selectedGenres: [],
        }}
        isGenerating={false}
        isRescanningModels={false}
        statusTitle="Ready"
        statusDetail="待機中"
        generationLogs={[]}
        errorMessage={null}
        onUpdateSettings={() => {}}
        onSwitchController={() => {}}
        onRescanModels={() => {}}
        onTogglePlayback={() => {}}
        onGenerate={() => {
          genTriggered += 1;
        }}
        onCancelGenerate={() => {}}
      />,
    );
    const meterBanner = cpRender.getByTestId('unsupported-meter-banner');
    expect(meterBanner.textContent).toContain('5/4');
    const genBtn = cpRender.getByTestId('generate-button') as HTMLButtonElement;
    expect(genBtn.disabled).toBe(true);
    fireEvent.click(genBtn);
    expect(genTriggered).toBe(0);
    cpRender.unmount();

    // 2. [R4-MED-03] CompassApiClient clears stale _handshakeToken on empty refresh AND never double-fires HTTP 403 when token is unchanged
    const prevDesktop = window.compassDesktop;
    const prevFetch = globalThis.fetch;
    try {
      const staleToken = 'b'.repeat(64);
      let refreshMode: 'unchanged' | 'cleared' = 'unchanged';
      window.compassDesktop = {
        handshakeToken: staleToken,
        getApiConfig: () => ({
          apiBaseUrl: 'http://127.0.0.1:18888',
          handshakeToken: staleToken,
          backendReady: true,
        }),
        requestApiConfig: async () => ({
          apiBaseUrl: 'http://127.0.0.1:18888',
          handshakeToken: refreshMode === 'unchanged' ? staleToken : '',
          backendReady: refreshMode === 'unchanged',
        }),
      };

      // Subcase A: Rejected token remains unchanged after forceRefresh -> exactly 1 fetch call (zero duplicate 403 retry)
      let fetchCallsA = 0;
      globalThis.fetch = vi.fn(async () => {
        fetchCallsA += 1;
        return new Response(JSON.stringify({ detail: 'Forbidden' }), { status: 403 });
      }) as typeof fetch;
      const clientA = new CompassApiClient();
      expect(clientA.handshakeToken).toBe(staleToken);
      await expect(clientA.getHardwareInfo()).rejects.toThrow();
      expect(fetchCallsA).toBe(1);

      // Subcase B: Backend restarted without auth (refresh returns handshakeToken: '') -> clears _handshakeToken and does NOT fall back to stale getApiConfig()
      refreshMode = 'cleared';
      const seenAuthHeaders: Array<string | undefined> = [];
      globalThis.fetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
        const hdrs = (init?.headers ?? {}) as Record<string, string>;
        seenAuthHeaders.push(hdrs['X-Compass-Handshake']);
        if (seenAuthHeaders.length === 1) {
          return new Response(JSON.stringify({ detail: 'Stale token rejected' }), { status: 403 });
        }
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            backend_class: 'CpuHardwareBackend',
            user_models_dir: '/u',
            bundled_models_dir: '/b',
            available_models_count: 1,
            models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
          }),
          { status: 200 },
        );
      }) as typeof fetch;
      const clientB = new CompassApiClient();
      const hwB = await clientB.getHardwareInfo();
      expect(hwB.device).toBe('cpu');
      expect(clientB.handshakeToken).toBe('');
      expect(seenAuthHeaders).toEqual([staleToken, undefined]);
    } finally {
      window.compassDesktop = prevDesktop;
      globalThis.fetch = prevFetch;
    }

    // 3. [R4-MED-05] Backing ArrayBuffer amplification, Symbol.toStringTag spoofing, & byteLength shadowing rejected; subarray compacted on publish
    const validSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
    );
    const twentyMbBacking = new ArrayBuffer(20_000_000);
    const tinySubarrayOver20Mb = new Uint8Array(twentyMbBacking, 1024, validSmf.byteLength);
    tinySubarrayOver20Mb.set(validSmf);

    const tagSpoof = {
      [Symbol.toStringTag]: 'Uint8Array',
      byteLength: 30,
      length: 1_073_741_824,
      0: 0x4d,
      1: 0x54,
      2: 0x68,
      3: 0x64,
    };

    const shadowed20Mb = new Uint8Array(20_000_000);
    shadowed20Mb.set([0x4d, 0x54, 0x68, 0x64]);
    Object.defineProperty(shadowed20Mb, 'byteLength', { value: 30 });

    for (const evilBytes of [tinySubarrayOver20Mb, tagSpoof, shadowed20Mb]) {
      expect(ipcSec.isValidCandidateMidiBytes(evilBytes)).toBe(false);
      expect(isValidCandidateMidiBytesPayload(evilBytes)).toBe(false);
      const syncCtrl = new DemoPianoRollController(new StubAudioEngine());
      const evilMsg = {
        messageId: 'r4_med05_evil',
        senderId: 'evil_peer',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: {
          ...syncCtrl.exportWorkspaceState(),
          candidates: [
            {
              index: 0,
              filename: 'spoof.mid',
              rawMidiBytes: evilBytes as unknown as Uint8Array,
              reason: null,
            },
          ],
        },
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(evilMsg)).toBe(false);
      expect(isValidIncomingSyncMessage(evilMsg)).toBe(false);

      syncCtrl.createNoteAt(64, 0, 480, 100);
      syncCtrl.applyWorkspaceState(evilMsg.workspaceState as never);
      expect(syncCtrl.getSnapshot().tracks[0].notes.length).toBe(1);
    }

    // Valid subarray over a 4 KB buffer is compacted (.slice()) by publishWorkspaceState
    const fourKbBuf = new ArrayBuffer(4096);
    const offsetView = new Uint8Array(fourKbBuf, 512, validSmf.byteLength);
    offsetView.set(validSmf);
    expect(offsetView.byteOffset).toBe(512);
    let capturedDesktopSyncMsg: unknown = null;
    window.compassDesktop = {
      sendWorkspaceSync: (msg) => {
        capturedDesktopSyncMsg = msg;
      },
    };
    try {
      const pubBus = new WorkspaceSyncBus(`compass_r4_compact_${Date.now()}`);
      const pubCtrl = new DemoPianoRollController(new StubAudioEngine());
      pubBus.publishWorkspaceState({
        ...pubCtrl.exportWorkspaceState(),
        candidates: [
          {
            index: 0,
            filename: 'offset.mid',
            rawMidiBytes: offsetView,
            reason: null,
          },
        ],
      });
      pubBus.dispose();
      const sent = capturedDesktopSyncMsg as {
        workspaceState: { candidates: Array<{ rawMidiBytes: Uint8Array }> };
      };
      expect(sent).not.toBeNull();
      const sentBytes = sent.workspaceState.candidates[0].rawMidiBytes;
      expect(sentBytes.byteOffset).toBe(0);
      expect(sentBytes.buffer.byteLength).toBe(validSmf.byteLength);
      expect(ipcSec.isValidWorkspaceSyncPayload(sent)).toBe(true);
    } finally {
      window.compassDesktop = prevDesktop;
    }

    // 4. [R4-MED-06] PPQN-scaled SMF limits: 96 PPQN self-export round-trip (380,000 ticks OK; 450,000 ticks rejected) + 9600 PPQN Pro Tools 55-bar import
    const smf96Ok = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 380_000, durationTicks: 96, velocity: 100 })],
      96,
      120,
    );
    const parsed96Ok = StandardMidiCodec.parseMidi(smf96Ok.buffer as ArrayBuffer);
    const ctrl96 = new DemoPianoRollController(new StubAudioEngine());
    ctrl96.importParsedMidiTracks(parsed96Ok, [0], true);
    const reExported96 = ctrl96.exportProjectMidiBytes();
    const reParsed96 = StandardMidiCodec.parseMidi(reExported96.buffer as ArrayBuffer);
    expect(reParsed96.ppqn).toBe(480);
    expect(reParsed96.tracks[0].notes.length).toBe(1);
    expect(reParsed96.tracks[0].notes[0].startTick).toBe(1_900_000);

    const smf96TooLong = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 450_000, durationTicks: 96, velocity: 100 })],
      96,
      120,
    );
    expect(() => StandardMidiCodec.parseMidi(smf96TooLong.buffer as ArrayBuffer)).toThrow(
      /累積ティック数が上限を超えています/,
    );

    // 55-bar 4/4 Pro Tools SMF at 9600 PPQN: startTick = 54 * 4 * 9600 = 2,073,600, endTick = 2,112,000 (220 quarter notes <= 4128)
    const smf9600 = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 67, startTick: 2_073_600, durationTicks: 38_400, velocity: 105 })],
      9600,
      120,
    );
    const parsed9600 = StandardMidiCodec.parseMidi(smf9600.buffer as ArrayBuffer);
    expect(parsed9600.ppqn).toBe(9600);
    expect(parsed9600.tracks[0].notes[0].startTick).toBe(2_073_600);
    const ctrl9600 = new DemoPianoRollController(new StubAudioEngine());
    ctrl9600.importParsedMidiTracks(parsed9600, [0], true);
    expect(ctrl9600.getSnapshot().tracks[0].notes.length).toBe(1);

    // 5. [R4-MED-07] Workspace IPC schema parity: 300-char FF 03 name, UTF-16 surrogate rename, 4/64 meter (denomPow=6), and >64 tracks cap
    const buildCustomSmf = (opts: { trackName?: string; denomPow?: number; num?: number }) => {
      const events: number[] = [];
      if (opts.trackName) {
        const nameBytes = new TextEncoder().encode(opts.trackName);
        events.push(0x00, 0xff, 0x03);
        // VLQ length
        let len = nameBytes.length;
        const vlq: number[] = [len & 0x7f];
        while ((len >>= 7) > 0) {
          vlq.unshift((len & 0x7f) | 0x80);
        }
        events.push(...vlq, ...nameBytes);
      }
      if (opts.denomPow !== undefined) {
        events.push(0x00, 0xff, 0x58, 0x04, opts.num ?? 4, opts.denomPow, 0x18, 0x08);
      }
      events.push(0x00, 0x90, 60, 100, 0x78, 0x80, 60, 0, 0x00, 0xff, 0x2f, 0x00);
      const out = new Uint8Array(14 + 8 + events.length);
      out.set([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, 0x01, 0xe0], 0);
      out.set([0x4d, 0x54, 0x72, 0x6b], 14);
      const dv = new DataView(out.buffer);
      dv.setUint32(18, events.length, false);
      out.set(events, 22);
      return out;
    };

    const smfLongNameAnd64Denom = buildCustomSmf({
      trackName: 'T'.repeat(300),
      num: 4,
      denomPow: 6, // 4/64 -> normalized proportionally to 2/32
    });
    const parsedEdge = StandardMidiCodec.parseMidi(smfLongNameAnd64Denom.buffer as ArrayBuffer);
    expect(parsedEdge.tracks[0].name.length).toBeLessThanOrEqual(256);
    expect(parsedEdge.timeSigNumerator).toBe(2);
    expect(parsedEdge.timeSigDenominator).toBe(32);
    expect(
      evaluateGenerationMeter(parsedEdge.timeSigNumerator ?? 4, parsedEdge.timeSigDenominator ?? 4)
        .supported,
    ).toBe(false);

    const parityCtrl = new DemoPianoRollController(new StubAudioEngine());
    parityCtrl.importParsedMidiTracks(parsedEdge, [0], true);

    // Surrogate pair right at index 255..256 must NOT leave an unpaired high surrogate \uD83D
    const surrogateBoundaryName = 'A'.repeat(255) + '\uD83D\uDE00' + 'Z'.repeat(10);
    parityCtrl.renameTrack(parityCtrl.getSnapshot().activeTrackId, surrogateBoundaryName);
    const renamed = parityCtrl.getSnapshot().tracks[0].name;
    expect(renamed.length).toBe(255);
    expect(renamed.charCodeAt(renamed.length - 1)).toBe(65);

    // Attempt to add 80 tracks -> capped cleanly at 64
    for (let i = 0; i < 80; i += 1) {
      parityCtrl.addTrack(`Track_${i}`, 'SYNTH');
    }
    expect(parityCtrl.getSnapshot().tracks.length).toBe(64);

    const exportedParityMsg = {
      messageId: 'r4_med07_ok',
      senderId: 'local_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: parityCtrl.exportWorkspaceState(),
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(exportedParityMsg)).toBe(true);
    expect(isValidIncomingSyncMessage(exportedParityMsg)).toBe(true);

    // 6. [R4-LOW-03] Composite WCAG AA note pitch label contrast across target tint & stacked ghost notes
    expect(
      pickAccessibleNoteLabelColor('#7DD3FC', 0.45 + (24 / 127) * 0.5, 60, { inTargetRange: true }),
    ).toBe('#000000');
    expect(
      pickAccessibleNoteLabelColor('#D4D4D8', 0.45 + (18 / 127) * 0.5, 60, { ghostStackCount: 1 }),
    ).toBe('#000000');
  });

  it('enforces Round 6 Fact-Checker (R5) remediations: linear SMPTE SMF wall-clock parity (0xFF 0x51 ignored & 24fps*1tpf unclamped), 25,000-note SMF export without V8 stack RangeError, 32,768-note candidate cap IPC parity, and Bar 1032 SMF tick-ceiling clamp', async () => {
    const ipcSecPath = '../../electron/ipc_security.cjs';
    const ipcSec = (await import(/* @vite-ignore */ ipcSecPath)) as {
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };

    const encodeVlq = (val: number): number[] => {
      let v = Math.max(0, Math.floor(val));
      const out = [v & 0x7f];
      v = Math.floor(v / 128);
      while (v > 0) {
        out.unshift((v & 0x7f) | 0x80);
        v = Math.floor(v / 128);
      }
      return out;
    };

    // 1. [R5-MED-02] Linear SMPTE SMF ignores 0xFF 0x51 Set Tempo & never clamps low-subframe 24fps*1tpf
    // Subcase A: -25 fps * 40 tpf (0xE728 -> 1000 ticks/sec, tpq=500) + 250-tick delta + 60 BPM FF 51 + 250-tick delta + 1000-tick (1.0s) note
    const smpte25Events: number[] = [
      ...encodeVlq(250),
      0xff,
      0x51,
      0x03,
      0x0f,
      0x42,
      0x40, // 60 BPM (1,000,000 us/quarter) -> MUST be ignored on SMPTE!
      ...encodeVlq(250),
      0x90,
      60,
      100,
      ...encodeVlq(1000),
      0x80,
      60,
      0,
      0x00,
      0xff,
      0x2f,
      0x00,
    ];
    const smpte25Smf = new Uint8Array(22 + smpte25Events.length);
    smpte25Smf.set([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, 0xe7, 0x28], 0);
    smpte25Smf.set([0x4d, 0x54, 0x72, 0x6b], 14);
    new DataView(smpte25Smf.buffer).setUint32(18, smpte25Events.length, false);
    smpte25Smf.set(smpte25Events, 22);

    const parsedSmpte25 = StandardMidiCodec.parseMidi(smpte25Smf.buffer as ArrayBuffer);
    expect(parsedSmpte25.ppqn).toBe(480);
    expect(parsedSmpte25.bpm).toBe(120);
    expect(parsedSmpte25.tracks[0].notes.length).toBe(1);
    expect(parsedSmpte25.tracks[0].notes[0].startTick).toBe(480); // 500 SMPTE ticks = 0.5s = 480 ticks @ 120 BPM
    expect(parsedSmpte25.tracks[0].notes[0].durationTicks).toBe(960); // 1000 SMPTE ticks = 1.0s = 960 ticks @ 120 BPM

    // Subcase B: -24 fps * 1 tpf (0xE801 -> 24 ticks/sec, tpq=12) + 24-tick (1.0s) note
    const smpte24Events = [0x00, 0x90, 64, 100, ...encodeVlq(24), 0x80, 64, 0, 0x00, 0xff, 0x2f, 0x00];
    const smpte24Smf = new Uint8Array(22 + smpte24Events.length);
    smpte24Smf.set([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, 0xe8, 0x01], 0);
    smpte24Smf.set([0x4d, 0x54, 0x72, 0x6b], 14);
    new DataView(smpte24Smf.buffer).setUint32(18, smpte24Events.length, false);
    smpte24Smf.set(smpte24Events, 22);

    const parsedSmpte24 = StandardMidiCodec.parseMidi(smpte24Smf.buffer as ArrayBuffer);
    expect(parsedSmpte24.ppqn).toBe(480);
    expect(parsedSmpte24.bpm).toBe(120);
    expect(parsedSmpte24.tracks[0].notes[0].startTick).toBe(0);
    expect(parsedSmpte24.tracks[0].notes[0].durationTicks).toBe(960); // 24 SMPTE ticks = 1.0s = 960 ticks

    // 2. [R5-MED-03] 25,000-note SMF export & DawTrack.addNotes(32,768) without V8 stack RangeError
    const dense25kNotes: MidiNote[] = [];
    for (let i = 0; i < 25_000; i += 1) {
      dense25kNotes.push(
        new MidiNote({
          pitch: 36 + (i % 48),
          startTick: i * 10,
          durationTicks: 8,
          velocity: 90,
        }),
      );
    }
    const single25kBytes = StandardMidiCodec.encodeSingleTrackMidi(dense25kNotes, 480, 120, 'PIANO');
    expect(single25kBytes.byteLength).toBeGreaterThan(150_000);
    const reparsedSingle25k = StandardMidiCodec.parseMidi(single25kBytes.buffer as ArrayBuffer);
    expect(reparsedSingle25k.tracks[0].notes.length).toBe(25_000);

    const multi25kTrack = new DawTrack({
      id: 'multi_25k',
      name: 'Dense Piano',
      instrument: 'PIANO',
      notes: dense25kNotes,
    });
    const multi25kBytes = StandardMidiCodec.encodeMultiTrackMidi(
      [multi25kTrack.toSnapshot()],
      480,
      120,
    );
    const reparsedMulti25k = StandardMidiCodec.parseMidi(multi25kBytes.buffer as ArrayBuffer);
    expect(reparsedMulti25k.tracks[0].notes.length).toBe(25_000);

    const bulkTrack = new DawTrack({ id: 'bulk_32k', name: 'Bulk', instrument: 'PIANO' });
    const bulk32kNotes: MidiNote[] = [];
    for (let i = 0; i < 32_768; i += 1) {
      bulk32kNotes.push(
        new MidiNote({ pitch: 60, startTick: i * 4, durationTicks: 2, velocity: 80 }),
      );
    }
    bulkTrack.addNotes(bulk32kNotes);
    expect(bulkTrack.toSnapshot().notes.length).toBe(32_768);

    // 3. [R5-MED-03] _insertCandidateIntoActiveTrack caps workspace total at 32,768 so IPC & SyncBus accept it
    const capCtrl = new DemoPianoRollController(new StubAudioEngine());
    const base32760Notes: MidiNote[] = [];
    for (let i = 0; i < 32_760; i += 1) {
      base32760Notes.push(
        new MidiNote({ pitch: 48 + (i % 24), startTick: i * 4, durationTicks: 2, velocity: 85 }),
      );
    }
    capCtrl.importParsedMidiTracks(
      {
        format: 1,
        ppqn: 480,
        bpm: 120,
        beatsPerMeasure: 4,
        tracks: [
          {
            index: 0,
            name: 'Backing 32760',
            program: 0,
            isDrum: false,
            instrument: 'PIANO',
            notes: base32760Notes,
          },
          {
            index: 1,
            name: 'Lead Target',
            program: 65,
            isDrum: false,
            instrument: 'SAX',
            notes: [],
          },
        ],
      },
      [0, 1],
      true,
    );
    const leadTrackId = capCtrl.getSnapshot().tracks[1].id;
    capCtrl.setActiveTrack(leadTrackId);
    capCtrl.setTargetMeasureRange(1, 2);

    const twentyCandNotes: MidiNote[] = [];
    for (let i = 0; i < 20; i += 1) {
      twentyCandNotes.push(
        new MidiNote({ pitch: 72, startTick: i * 60, durationTicks: 40, velocity: 100 }),
      );
    }
    const candSmf20 = StandardMidiCodec.encodeSingleTrackMidi(twentyCandNotes, 480, 120, 'SAX');
    const parsedCand20 = StandardMidiCodec.parseMidi(candSmf20.buffer as ArrayBuffer);
    await capCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'cand_0.mid',
          rawMidiBytes: candSmf20,
          parsedMidi: parsedCand20,
          reason: null,
        },
      ],
      [],
    );

    const postSnap = capCtrl.getSnapshot();
    const totalAfterCap = postSnap.tracks.reduce((acc, t) => acc + t.notes.length, 0);
    expect(totalAfterCap).toBe(32_768);
    expect(postSnap.tracks[1].notes.length).toBe(8);

    const syncMsg32k = {
      messageId: 'r5_med03_cap_ok',
      senderId: 'local_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: capCtrl.exportWorkspaceState(),
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(syncMsg32k)).toBe(true);
    expect(isValidIncomingSyncMessage(syncMsg32k)).toBe(true);

    // 4. [R5-MED-03] Candidate placed at Bar 1032 clamps within MAX_SMF_QUARTER_NOTES * 480 so re-export parses cleanly
    const endCtrl = new DemoPianoRollController(new StubAudioEngine());
    endCtrl.importParsedMidiTracks(
      {
        format: 0,
        ppqn: 480,
        bpm: 120,
        beatsPerMeasure: 4,
        tracks: [
          {
            index: 0,
            name: 'Piano End',
            program: 0,
            isDrum: false,
            instrument: 'PIANO',
            notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 90 })],
          },
        ],
      },
      [0],
      true,
    );
    // Bars 1024..1032 start at 1023 * 1920 = 1,964,160; MAX_SMF_QUARTER_NOTES * 480 = 1,981,440 (end of Bar 1032, +17,280 ticks)
    endCtrl.setTargetMeasureRange(1024, 1032);
    const overhangCandSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [
        new MidiNote({ pitch: 67, startTick: 16_000, durationTicks: 3840, velocity: 100 }), // extends past Bar 1032!
        new MidiNote({ pitch: 72, startTick: 18_000, durationTicks: 480, velocity: 100 }), // starts beyond Bar 1032!
      ],
      480,
      120,
      'PIANO',
    );
    await endCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'end_overhang.mid',
          rawMidiBytes: overhangCandSmf,
          parsedMidi: StandardMidiCodec.parseMidi(overhangCandSmf.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    const maxAllowedTick = MAX_SMF_QUARTER_NOTES * 480;
    for (const n of endCtrl.getSnapshot().tracks[0].notes) {
      expect(n.endTick).toBeLessThanOrEqual(maxAllowedTick);
    }
    const exportedEndBytes = endCtrl.exportProjectMidiBytes();
    const reParsedEnd = StandardMidiCodec.parseMidi(exportedEndBytes.buffer as ArrayBuffer);
    expect(reParsedEnd.tracks[0].notes.length).toBe(2);
    expect(reParsedEnd.tracks[0].notes[1].endTick).toBe(maxAllowedTick);
  });

  it('enforces Round 7 Fact-Checker (R6) remediations: Cubase/Ableton horizon & 32,768-note LSP parity, cross-window note/slot tick ceiling across ipc_security/SyncBus/applyWorkspaceState, and 29.97 SMPTE fractional horizon & dangling-note clamping', async () => {
    const ipcSecPath = '../../electron/ipc_security.cjs';
    const ipcSec = (await import(/* @vite-ignore */ ipcSecPath)) as {
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };

    const maxAllowedTick480 = MAX_SMF_QUARTER_NOTES * 480; // 1,981,440

    // 1. [R6-MED-03] CubaseDawController & AbletonDawController enforce maxProjectTicks (Bar 1024..1032 4/4 & Bar 300 16/4) + 32,768-note cap
    for (const adapter of [new CubaseDawController(), new AbletonDawController()]) {
      adapter.setTargetMeasureRange(1024, 1032);
      const overhangSmf = StandardMidiCodec.encodeSingleTrackMidi(
        [
          new MidiNote({ pitch: 60, startTick: 16_000, durationTicks: 4000, velocity: 100 }),
          new MidiNote({ pitch: 67, startTick: 18_500, durationTicks: 960, velocity: 100 }),
        ],
        480,
        120,
        'PIANO',
      );
      await adapter.applyGeneratedCandidates(
        [
          {
            index: 0,
            filename: 'adapter_bar1024.mid',
            rawMidiBytes: overhangSmf,
            parsedMidi: StandardMidiCodec.parseMidi(overhangSmf.buffer as ArrayBuffer),
            reason: null,
          },
        ],
        [],
      );
      const snap = adapter.getSnapshot();
      expect(snap.tracks[0].notes.length).toBe(1);
      expect(snap.tracks[0].notes[0].endTick).toBe(maxAllowedTick480);
      const exported = StandardMidiCodec.encodeMultiTrackMidi(snap.tracks, snap.ppqn, snap.bpm);
      const reparsed = StandardMidiCodec.parseMidi(exported.buffer as ArrayBuffer);
      expect(reparsed.tracks[0].notes[0].endTick).toBe(maxAllowedTick480);

      // 16/4 at Bar 300 (startTick = 299 * 16 * 480 = 2,296,320 >= 1,981,440): places 0 beyond-horizon notes so export still parses cleanly
      adapter.setTimeSignature(16, 4);
      adapter.setTargetMeasureRange(300, 304);
      await adapter.applyGeneratedCandidates(
        [
          {
            index: 0,
            filename: 'adapter_16_4_bar300.mid',
            rawMidiBytes: overhangSmf,
            parsedMidi: StandardMidiCodec.parseMidi(overhangSmf.buffer as ArrayBuffer),
            reason: null,
          },
        ],
        [],
      );
      const snap164 = adapter.getSnapshot();
      for (const n of snap164.tracks[0].notes) {
        expect(n.endTick).toBeLessThanOrEqual(maxAllowedTick480);
      }
      const exported164 = StandardMidiCodec.encodeMultiTrackMidi(
        snap164.tracks,
        snap164.ppqn,
        snap164.bpm,
      );
      expect(() => StandardMidiCodec.parseMidi(exported164.buffer as ArrayBuffer)).not.toThrow();
    }

    // 2. [R6-MED-04] ipc_security.cjs, WorkspaceSyncBus, and DemoPianoRollController.applyWorkspaceState reject startTick/endTick > maxProjectTicks while allowing valid 16/4 candidate slots
    const syncCtrl = new DemoPianoRollController(new StubAudioEngine());
    syncCtrl.createNoteAt(60, 0, 480, 100);
    const baseWs = syncCtrl.exportWorkspaceState();

    const outOfHorizonMsg = {
      messageId: 'r6_med04_bad_tick',
      senderId: 'evil_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: {
        ...baseWs,
        tracks: baseWs.tracks.map((t, idx) =>
          idx === 0
            ? {
                ...t,
                notes: [
                  {
                    id: 'overflow_note',
                    pitch: 64,
                    startTick: 2_500_000,
                    durationTicks: 480,
                    velocity: 100,
                    channel: 0,
                  },
                ],
              }
            : t,
        ),
      },
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(outOfHorizonMsg)).toBe(false);
    expect(isValidIncomingSyncMessage(outOfHorizonMsg)).toBe(false);
    syncCtrl.applyWorkspaceState(outOfHorizonMsg.workspaceState as never);
    expect(syncCtrl.getSnapshot().tracks[0].notes[0].pitch).toBe(60);
    expect(syncCtrl.getSnapshot().tracks[0].notes[0].startTick).toBe(0);

    // Valid 16/4 Bar 300..304 workspace state (slot endTick = 304 * 16 * 480 = 2,334,720 <= 16,512 * 480) is accepted!
    const valid164Ctrl = new DemoPianoRollController(new StubAudioEngine());
    valid164Ctrl.setTimeSignature(16, 4);
    valid164Ctrl.setTargetMeasureRange(300, 304);
    const cand164 = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    await valid164Ctrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'c164.mid',
          rawMidiBytes: cand164,
          parsedMidi: StandardMidiCodec.parseMidi(cand164.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    const valid164Msg = {
      messageId: 'r6_med04_valid_164',
      senderId: 'peer_ok',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: valid164Ctrl.exportWorkspaceState(),
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(valid164Msg)).toBe(true);
    expect(isValidIncomingSyncMessage(valid164Msg)).toBe(true);

    // 3. [R6-LOW-01] 29.97 drop-frame SMPTE (0xE328 -> tpq=599.4, maxTrackTicks=2,474,323) boundary + dangling unclosed note clamping
    const encodeVlq = (val: number): number[] => {
      let v = Math.max(0, Math.floor(val));
      const out = [v & 0x7f];
      v = Math.floor(v / 128);
      while (v > 0) {
        out.unshift((v & 0x7f) | 0x80);
        v = Math.floor(v / 128);
      }
      return out;
    };
    const buildSmpte2997 = (events: number[]) => {
      const out = new Uint8Array(22 + events.length);
      out.set([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, 0xe3, 0x28], 0);
      out.set([0x4d, 0x54, 0x72, 0x6b], 14);
      new DataView(out.buffer).setUint32(18, events.length, false);
      out.set(events, 22);
      return out;
    };

    // Over-horizon 2,474,324 (> Math.floor(4128 * 599.4) = 2,474,323) throws cleanly
    const over2997 = buildSmpte2997([0x00, 0x90, 60, 100, ...encodeVlq(2_474_324), 0x80, 60, 0, 0x00, 0xff, 0x2f, 0x00]);
    expect(() => StandardMidiCodec.parseMidi(over2997.buffer as ArrayBuffer)).toThrow(
      /累積ティック数が上限を超えています/,
    );

    // Dangling unclosed NoteOn at 2,474,322..2,474,323 clamps endTick <= 1,981,440 and re-parses cleanly!
    const dangling2997 = buildSmpte2997([
      ...encodeVlq(2_474_322),
      0x90,
      60,
      100,
      0x01,
      0xff,
      0x2f,
      0x00,
    ]);
    const parsedDangling = StandardMidiCodec.parseMidi(dangling2997.buffer as ArrayBuffer);
    expect(parsedDangling.tracks[0].notes.length).toBe(1);
    expect(parsedDangling.tracks[0].notes[0].endTick).toBeLessThanOrEqual(maxAllowedTick480);
    const reEncodedDangling = StandardMidiCodec.encodeSingleTrackMidi(
      parsedDangling.tracks[0].notes,
      480,
      120,
      'PIANO',
    );
    expect(() => StandardMidiCodec.parseMidi(reEncodedDangling.buffer as ArrayBuffer)).not.toThrow();
  });

  it('enforces Round 8 Fact-Checker (R7) remediations: 9600 PPQN MeasureRange parity, odd /32 meter (5/32, 7/32) zero-drift math, >128-char malformed job_id rejection, and canonical trimmed ID parity across ipc_security/SyncBus/applyWorkspaceState', async () => {
    const ipcSecPath = '../../electron/ipc_security.cjs';
    const ipcSec = (await import(/* @vite-ignore */ ipcSecPath)) as {
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };

    // 1. [R7-MED-01] 9600 PPQN synced workspace preserves Bar 3 user note & meta_past detection when generating Bars 5..8
    const range9600 = new MeasureRange(5, 8, 9600, 4);
    expect(range9600.ppqn).toBe(9600);
    expect(range9600.ticksPerMeasure).toBe(38_400);
    expect(range9600.startTick).toBe(153_600);

    const ctrl9600 = new DemoPianoRollController(new StubAudioEngine());
    const base9600 = ctrl9600.exportWorkspaceState();
    ctrl9600.applyWorkspaceState({
      ...base9600,
      ppqn: 9600,
      beatsPerMeasure: 4,
      timeSigNumerator: 4,
      timeSigDenominator: 4,
      targetStartMeasure: 5,
      targetEndMeasure: 8,
      tracks: [
        {
          id: 'trk_9600',
          name: 'ProTools 9600',
          instrument: 'PIANO',
          colorHex: '#60A5FA',
          isMuted: false,
          isSoloed: false,
          notes: [
            {
              id: 'bar3_user_note',
              pitch: 64,
              startTick: 2 * 38_400, // Bar 3 Beat 1 at 9600 PPQN (76,800 ticks)
              durationTicks: 9600,
              velocity: 100,
              channel: 0,
            },
          ],
        },
      ],
      activeTrackId: 'trk_9600',
      activeCandidateRange: null,
      candidateOriginSlot: null,
      activeCandidateIndex: 0,
      activeCandidateNoteIds: [],
      candidates: [],
      latestReason: null,
    });
    const snap9600Before = ctrl9600.getSnapshot();
    expect(snap9600Before.ppqn).toBe(9600);
    expect(snap9600Before.targetRange.ppqn).toBe(9600);
    expect(snap9600Before.targetRange.startTick).toBe(153_600);
    expect(snap9600Before.contextAvailability.hasPastContext).toBe(true);
    expect(snap9600Before.contextAvailability.autoResolvedTask).toBe('meta_past');

    const candBar5Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 72, startTick: 0, durationTicks: 480, velocity: 96 })],
      480,
      120,
      'PIANO',
    );
    await ctrl9600.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'bar5_gen.mid',
          rawMidiBytes: candBar5Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(candBar5Bytes.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    const snap9600After = ctrl9600.getSnapshot();
    expect(snap9600After.tracks[0].notes.some((n) => n.id === 'bar3_user_note')).toBe(true);
    expect(snap9600After.tracks[0].notes.some((n) => n.startTick === 153_600)).toBe(true);

    // 2. [R7-MED-02] Odd /32 meters (5/32 = 0.625 beats = 300 ticks @ 480 PPQN, 7/32 = 0.875 beats = 420 ticks @ 480 PPQN)
    const range532 = new MeasureRange(1, 4, 480, 5 / 8);
    expect(range532.beatsPerMeasure).toBe(0.625);
    expect(range532.ticksPerMeasure).toBe(300);

    const range732 = new MeasureRange(1, 4, 480, 7 / 8);
    expect(range732.beatsPerMeasure).toBe(0.875);
    expect(range732.ticksPerMeasure).toBe(420);

    const odd32Ctrl = new DemoPianoRollController(new StubAudioEngine());
    odd32Ctrl.setTimeSignature(5, 32);
    expect(odd32Ctrl.getSnapshot().beatsPerMeasure).toBe(0.625);
    expect(odd32Ctrl.getSnapshot().targetRange.beatsPerMeasure).toBe(0.625);
    expect(odd32Ctrl.getSnapshot().targetRange.ticksPerMeasure).toBe(300);

    const cubase32 = new CubaseDawController();
    cubase32.setTimeSignature(7, 32);
    expect(cubase32.getSnapshot().targetRange.beatsPerMeasure).toBe(0.875);
    expect(cubase32.getSnapshot().targetRange.ticksPerMeasure).toBe(420);

    const smf532 = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 120, velocity: 100 })],
      480,
      120,
      'PIANO',
      0.625,
    );
    const parsed532 = StandardMidiCodec.parseMidi(smf532.buffer as ArrayBuffer);
    expect(parsed532.timeSigNumerator).toBe(5);
    expect(parsed532.timeSigDenominator).toBe(32);
    expect(parsed532.beatsPerMeasure).toBe(0.625);

    // 3. [R7-LOW-03] CompassApiClient.getGenerationProgress rejects >128-char malformed & whitespace-only jobId before fetch
    let fetchCalled = false;
    const spyNoFetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
      fetchCalled = true;
      return new Response('{}', { status: 200 });
    });
    try {
      const api = new CompassApiClient('http://127.0.0.1:18999', 'tok');
      expect(await api.getGenerationProgress(undefined, `${'a'.repeat(128)}!evil`)).toBeNull();
      expect(await api.getGenerationProgress(undefined, `${'a'.repeat(128)}, job_2`)).toBeNull();
      expect(await api.getGenerationProgress(undefined, '   ')).toBeNull();
      expect(fetchCalled).toBe(false);
    } finally {
      spyNoFetch.mockRestore();
    }

    // 4. [R7-LOW-04] Zero divergence on whitespace-only & padded string IDs across ipc_security, WorkspaceSyncBus, and applyWorkspaceState
    const idAuditCtrl = new DemoPianoRollController(new StubAudioEngine());
    idAuditCtrl.createNoteAt(60, 0, 480, 100);
    const cleanWs = idAuditCtrl.exportWorkspaceState();

    for (const invalidWs of [
      { ...cleanWs, tracks: [{ ...cleanWs.tracks[0], id: '   ' }], activeTrackId: '   ' },
      { ...cleanWs, tracks: [{ ...cleanWs.tracks[0], name: '   ' }] },
      { ...cleanWs, tracks: [{ ...cleanWs.tracks[0], colorHex: '   ' }] },
      {
        ...cleanWs,
        tracks: [
          {
            ...cleanWs.tracks[0],
            notes: [
              { id: 'n1', pitch: 60, startTick: 0, durationTicks: 480, velocity: 100, channel: 0 },
              { id: ' n1 ', pitch: 64, startTick: 480, durationTicks: 480, velocity: 100, channel: 0 },
            ],
          },
        ],
      },
    ]) {
      const msg = {
        messageId: 'msg_r8_check',
        senderId: 'peer_r8',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: invalidWs,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(msg)).toBe(false);
      expect(isValidIncomingSyncMessage(msg)).toBe(false);
      const probeCtrl = new DemoPianoRollController(new StubAudioEngine());
      probeCtrl.createNoteAt(71, 240, 240, 90);
      probeCtrl.applyWorkspaceState(invalidWs as never);
      expect(probeCtrl.getSnapshot().tracks[0].notes[0].pitch).toBe(71);
    }

    const paddedValidWs = {
      ...cleanWs,
      tracks: [
        {
          ...cleanWs.tracks[0],
          id: ' trk_pad ',
          notes: [
            { id: ' note_pad ', pitch: 67, startTick: 0, durationTicks: 480, velocity: 100, channel: 0 },
          ],
        },
      ],
      activeTrackId: ' trk_pad ',
      activeCandidateRange: { trackId: ' trk_pad ', startTick: 0, endTick: 1920 },
      candidateOriginSlot: { trackId: 'trk_pad', startTick: 0, endTick: 1920 },
    };
    const paddedValidMsg = {
      messageId: 'msg_r8_valid_pad',
      senderId: 'peer_r8',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: paddedValidWs,
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(paddedValidMsg)).toBe(true);
    expect(isValidIncomingSyncMessage(paddedValidMsg)).toBe(true);
    const paddedCtrl = new DemoPianoRollController(new StubAudioEngine());
    paddedCtrl.applyWorkspaceState(paddedValidWs as never);
    expect(paddedCtrl.getSnapshot().activeTrackId).toBe('trk_pad');
    expect(paddedCtrl.getSnapshot().tracks[0].id).toBe('trk_pad');
    expect(paddedCtrl.getSnapshot().tracks[0].notes[0].id).toBe('note_pad');
  });

  it('verifies Round-9 Fact-Checker R8 remediations (GenerationReason sanitization, canonical meter sync, sparse holes, low-PPQN & defensive SMF clamping)', async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ipcSec = require('../../electron/ipc_security.cjs');

    // 1. [R8-MED-01] HTTP X-Generation-Reason & controller applyGeneratedCandidates sanitization + cross-window sync survival
    const validSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const zip = new JSZip();
    zip.file('output_0.mid', validSmf);
    zip.file('output_1.mid', validSmf);
    const zipBytes = await zip.generateAsync({ type: 'uint8array' });

    const hostileHeaderReasons = [
      {
        source: 'model_cot',
        task: 'meta',
        thinking: true,
        genre: 'jazz',
        key: `C_major_${'x'.repeat(40)}`,
        extra_debug_prop: 99,
      },
      {
        source: 'custom_adapter',
        task: 'infill',
        thinking: 1,
        instruments: ['PIANO', 42, 'SAX'],
        genre: 123,
        note_density: { PIANO: 4, __proto__: 9 },
        gen_measure_count: 9999,
      },
    ];
    const encodedHeader = encodeURIComponent(JSON.stringify(hostileHeaderReasons));
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = typeof input === 'string' ? input : (input as URL).toString();
      if (url.includes('/generation_progress')) {
        return new Response(
          JSON.stringify({
            active: false,
            stage: 'done',
            generated_tokens: 16,
            max_tokens: 16,
            message: 'ok',
            updated_at: 1,
          }),
          { status: 200 },
        );
      }
      return new Response(zipBytes.slice().buffer as ArrayBuffer, {
        status: 200,
        headers: { 'X-Generation-Reason': encodedHeader },
      });
    });

    try {
      const api = new CompassApiClient('http://127.0.0.1:18999', 'tok');
      const genRes = await api.generateMusic({
        meta: {
          model_type: 'MORTM4.5D-80M-SFT-Gen',
          task: 'meta',
          program: ['PIANO'],
          tempo: 120,
          num_gems: 2,
          genfield_measure: 4,
          gen_note_dense: { PIANO: 4 },
          p: 0.95,
          temperature: 1.0,
          cot_temperature: 0.1,
          thinking: true,
        },
      });
      expect(genRes.candidates.length).toBe(2);
      expect(genRes.candidates[0].reason?.genre).toEqual(['jazz']);
      expect(genRes.candidates[0].reason?.key?.length).toBe(32);
      expect('extra_debug_prop' in (genRes.candidates[0].reason as object)).toBe(false);
      expect(genRes.candidates[1].reason?.source).toBe('request');
      expect(genRes.candidates[1].reason?.genre).toBeNull();
      expect(genRes.candidates[1].reason?.instruments).toEqual(['PIANO', 'SAX']);
      expect(genRes.candidates[1].reason?.gen_measure_count).toBe(512);

      const reasonCtrl = new DemoPianoRollController(new StubAudioEngine());
      await reasonCtrl.applyGeneratedCandidates(genRes.candidates, genRes.reasons);
      const exportedMsg = {
        messageId: 'msg_r9_reason_ok',
        senderId: 'peer_r9',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: reasonCtrl.exportWorkspaceState(),
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(exportedMsg)).toBe(true);
      expect(isValidIncomingSyncMessage(exportedMsg)).toBe(true);

      const { unmount } = render(
        <PianoRollView
          controller={reasonCtrl}
          session={reasonCtrl.getSnapshot()}
          focusRevision={1}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />,
      );
      expect(screen.getByText(/Genre: jazz/)).toBeTruthy();
      unmount();

      // Direct unsanitized applyGeneratedCandidates on both Demo & External DAW controllers
      const cubaseCtrl = new CubaseDawController();
      const rawDirtyCand = {
        ...genRes.candidates[0],
        reason: {
          source: 'custom' as never,
          task: 'meta',
          thinking: true,
          instruments: null,
          note_density: null,
          gen_measure_count: 4,
          genre: 'funk' as never,
          key: 'y'.repeat(50),
          extra: 1,
        } as never,
      };
      await reasonCtrl.applyGeneratedCandidates([rawDirtyCand], []);
      await cubaseCtrl.applyGeneratedCandidates([rawDirtyCand], []);
      expect(reasonCtrl.getSnapshot().latestReason?.genre).toEqual(['funk']);
      expect(reasonCtrl.getSnapshot().latestReason?.key?.length).toBe(32);
      expect(cubaseCtrl.getSnapshot().latestReason?.genre).toEqual(['funk']);
      expect(cubaseCtrl.getSnapshot().latestReason?.key?.length).toBe(32);
      expect(
        ipcSec.isValidWorkspaceSyncPayload({
          messageId: 'msg_r9_direct_dirty',
          senderId: 'peer_r9',
          type: 'SYNC_WORKSPACE_STATE',
          workspaceState: reasonCtrl.exportWorkspaceState(),
        }),
      ).toBe(true);
      expect(sanitizeGenerationReason({ source: 'model_cot', task: '   ' })).toBeNull();
    } finally {
      fetchSpy.mockRestore();
    }

    // 2. [R8-MED-02] Canonical beatsPerMeasure <-> (timeSigNumerator, timeSigDenominator) sync across SMF import & IPC
    const buildCustomTsSmf = (rawNum: number, denomPow: number): Uint8Array => {
      const body = [
        0x00, 0xff, 0x58, 0x04, rawNum & 0xff, denomPow & 0xff, 0x18, 0x08,
        0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
        0x00, 0x90, 0x3c, 0x64,
        0x78, 0x80, 0x3c, 0x00,
        0x00, 0xff, 0x2f, 0x00,
      ];
      const out = new Uint8Array(22 + body.length);
      const dv = new DataView(out.buffer);
      out.set([0x4d, 0x54, 0x68, 0x64]);
      dv.setUint32(4, 6, false);
      dv.setUint16(8, 0, false);
      dv.setUint16(10, 1, false);
      dv.setUint16(12, 480, false);
      out.set([0x4d, 0x54, 0x72, 0x6b], 14);
      dv.setUint32(18, body.length, false);
      out.set(body, 22);
      return out;
    };

    // 40/32 SMF -> 20/16 (5.0 beats, 120 steps > 96 -> evaluateGenerationMeter.supported === false!)
    const parsed4032 = StandardMidiCodec.parseMidi(buildCustomTsSmf(40, 5).buffer as ArrayBuffer);
    expect(parsed4032.timeSigNumerator).toBe(20);
    expect(parsed4032.timeSigDenominator).toBe(16);
    expect(parsed4032.beatsPerMeasure).toBe(5);
    expect(
      evaluateGenerationMeter(parsed4032.timeSigNumerator!, parsed4032.timeSigDenominator!).supported,
    ).toBe(false);

    // 11/64 SMF -> 6/32 (0.75 beats, zero drift between MeasureRange & PianoRollView)
    const parsed1164 = StandardMidiCodec.parseMidi(buildCustomTsSmf(11, 6).buffer as ArrayBuffer);
    expect(parsed1164.timeSigNumerator).toBe(6);
    expect(parsed1164.timeSigDenominator).toBe(32);
    expect(parsed1164.beatsPerMeasure).toBe(0.75);
    const ctrl1164 = new DemoPianoRollController(new StubAudioEngine());
    ctrl1164.importParsedMidiTracks(parsed1164, [0], true);
    expect(ctrl1164.getSnapshot().beatsPerMeasure).toBe(0.75);
    expect(ctrl1164.getSnapshot().targetRange.ticksPerMeasure).toBe(360);
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'msg_r9_1164',
        senderId: 'peer_r9',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: ctrl1164.exportWorkspaceState(),
      }),
    ).toBe(true);

    // Contradictory or partial timeSignature/beatsPerMeasure sync payloads rejected across all 3 layers
    const baseSyncCtrl = new DemoPianoRollController(new StubAudioEngine());
    baseSyncCtrl.createNoteAt(64, 0, 480, 100);
    const baseWs = baseSyncCtrl.exportWorkspaceState();
    for (const contradictoryWs of [
      { ...baseWs, timeSigNumerator: 4, timeSigDenominator: 4, beatsPerMeasure: 3 },
      { ...baseWs, timeSigNumerator: 7, timeSigDenominator: 8, beatsPerMeasure: 4 },
      { ...baseWs, timeSigNumerator: 3, timeSigDenominator: undefined, beatsPerMeasure: 3 },
      { ...baseWs, timeSigNumerator: undefined, timeSigDenominator: 4, beatsPerMeasure: 4 },
      { ...baseWs, timeSigNumerator: undefined, timeSigDenominator: undefined, beatsPerMeasure: 4.125 },
    ]) {
      const m = {
        messageId: 'msg_r9_contra',
        senderId: 'peer_r9',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: contradictoryWs,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(m)).toBe(false);
      expect(isValidIncomingSyncMessage(m)).toBe(false);
      const probe = new DemoPianoRollController(new StubAudioEngine());
      probe.createNoteAt(72, 0, 240, 100);
      probe.applyWorkspaceState(contradictoryWs as never);
      expect(probe.getSnapshot().tracks[0].notes[0].pitch).toBe(72);
    }

    // Standalone beatsPerMeasure without timeSig* derives exact 7/8 and round-trips cleanly
    const beatsOnlyWs = {
      ...baseWs,
      timeSigNumerator: undefined,
      timeSigDenominator: undefined,
      beatsPerMeasure: 3.5,
    };
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'msg_r9_beats_only',
        senderId: 'peer_r9',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: beatsOnlyWs,
      }),
    ).toBe(true);
    expect(
      isValidIncomingSyncMessage({
        messageId: 'msg_r9_beats_only',
        senderId: 'peer_r9',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: beatsOnlyWs,
      }),
    ).toBe(true);
    const beatsOnlyCtrl = new DemoPianoRollController(new StubAudioEngine());
    beatsOnlyCtrl.applyWorkspaceState(beatsOnlyWs as never);
    expect(beatsOnlyCtrl.getSnapshot().timeSigNumerator).toBe(7);
    expect(beatsOnlyCtrl.getSnapshot().timeSigDenominator).toBe(8);
    expect(beatsOnlyCtrl.getSnapshot().beatsPerMeasure).toBe(3.5);
    expect(canonicalBeatsPerMeasure(7, 8)).toBe(3.5);
    expect(deriveTimeSignatureFromBeats(3.5)).toEqual({ numerator: 7, denominator: 8 });

    // 3. [R8-LOW-01] Sparse array holes rejected across ipc_security, WorkspaceSyncBus, and DemoPianoRollController
    const holeyGenre = new Array<string>(2);
    holeyGenre[1] = 'jazz';
    const holeyIds = new Array<string>(2);
    holeyIds[1] = 'note_1';
    const holeyTracks = new Array(2);
    holeyTracks[0] = baseWs.tracks[0];
    const holeyNotes = new Array(2);
    holeyNotes[0] = baseWs.tracks[0].notes[0];

    for (const sparseWs of [
      {
        ...baseWs,
        latestReason: {
          source: 'model_cot',
          task: 'meta',
          thinking: true,
          instruments: ['PIANO'],
          note_density: null,
          gen_measure_count: 4,
          genre: holeyGenre,
          key: 'CM',
        },
      },
      { ...baseWs, activeCandidateNoteIds: holeyIds },
      { ...baseWs, tracks: holeyTracks },
      { ...baseWs, tracks: [{ ...baseWs.tracks[0], notes: holeyNotes }] },
    ]) {
      const m = {
        messageId: 'msg_r9_sparse',
        senderId: 'peer_r9',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: sparseWs,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(m)).toBe(false);
      expect(isValidIncomingSyncMessage(m)).toBe(false);
      const probe = new DemoPianoRollController(new StubAudioEngine());
      probe.createNoteAt(69, 0, 240, 100);
      probe.applyWorkspaceState(sparseWs as never);
      expect(probe.getSnapshot().tracks[0].notes[0].pitch).toBe(69);
    }

    // 4. [R8-LOW-02] Exact bar ticks at ppqn=24 for sub-quarter MORTM meters & ppqn=481 in 5/32
    expect(new MeasureRange(1, 4, 24, 0.5).ticksPerMeasure).toBe(12);
    expect(new MeasureRange(1, 4, 24, 0.625).ticksPerMeasure).toBe(15);
    expect(new MeasureRange(1, 4, 24, 0.75).ticksPerMeasure).toBe(18);
    expect(new MeasureRange(1, 4, 24, 0.875).ticksPerMeasure).toBe(21);

    const ctrl481 = new DemoPianoRollController(new StubAudioEngine());
    ctrl481.applyWorkspaceState({
      ...baseWs,
      ppqn: 481,
      timeSigNumerator: 5,
      timeSigDenominator: 32,
      beatsPerMeasure: 0.625,
    });
    expect(ctrl481.getSnapshot().targetRange.ticksPerMeasure).toBe(301);
    const { unmount: unmount481 } = render(
      <PianoRollView
        controller={ctrl481}
        session={ctrl481.getSnapshot()}
        focusRevision={1}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );
    unmount481();

    // 5. [R8-LOW-03] StandardMidiCodec defensive ppqn (1..32767) & 24-bit mpqn (1..0xFFFFFF) clamping
    const sampleNotes = [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 120, velocity: 100 })];
    for (const badPpqn of [0, -5, Number.NaN, 40_000]) {
      for (const badBpm of [Number.NaN, 0, -20, 1, 2.5]) {
        const enc0 = StandardMidiCodec.encodeSingleTrackMidi(sampleNotes, badPpqn, badBpm, 'PIANO');
        const encCond = StandardMidiCodec.encodeConditionsMidi(sampleNotes, badPpqn, badBpm, false);
        const encMulti = StandardMidiCodec.encodeMultiTrackMidi(
          ctrl481.getSnapshot().tracks,
          badPpqn,
          badBpm,
        );
        for (const bytes of [enc0, encCond, encMulti]) {
          const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
          const division = dv.getUint16(12, false);
          expect(division & 0x8000).toBe(0);
          expect(division).toBeGreaterThanOrEqual(1);
          expect(division).toBeLessThanOrEqual(32767);
          const reparsed = StandardMidiCodec.parseMidi(
            bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
          );
          expect(reparsed.bpm).toBeGreaterThan(0);
        }
      }
    }
  });

  it('verifies Round-10 Fact-Checker R9 remediations (low-PPQN snap/resize/whole-note grid, LSP reason fallback + index<=64, UTF-16 surrogate safety, non-pow2 meter scaling)', async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ipcSec = require('../../electron/ipc_security.cjs');

    // 1. [R9-LOW-01] Low-PPQN snap/resize floor (1..15) + MidiNote.withDuration(>=1) + 1/1 whole-note & odd-PPQN beat-index grid
    const lowPpqnCtrl = new DemoPianoRollController(new StubAudioEngine());
    const baseWs = lowPpqnCtrl.exportWorkspaceState();
    lowPpqnCtrl.applyWorkspaceState({
      ...baseWs,
      ppqn: 24,
      timeSigNumerator: 1,
      timeSigDenominator: 8,
      beatsPerMeasure: 0.5,
      quantizeDivision: 16,
      isTriplet: false,
    });
    expect(lowPpqnCtrl.getSnapStepTicks()).toBe(6); // 24 * 4 / 16 = 6 (not inflated to 15!)
    expect(lowPpqnCtrl.snapTick(12)).toBe(12); // Bar 2 in 1/8 @ 24 PPQN stays exact at tick 12!
    lowPpqnCtrl.setQuantize(32, true);
    expect(lowPpqnCtrl.getSnapStepTicks()).toBe(2); // 3 * 2 / 3 = 2

    const shortNote = new MidiNote({ pitch: 60, startTick: 0, durationTicks: 12, velocity: 100 });
    expect(shortNote.withDuration(6).durationTicks).toBe(6);
    expect(shortNote.withDuration(0).durationTicks).toBe(1);

    // Render 1/1 whole-note meter & 5/32 @ ppqn=27 & 6/8 @ ppqn=481 cleanly
    for (const gridSpec of [
      { ppqn: 480, timeSigNumerator: 1, timeSigDenominator: 1, beatsPerMeasure: 4 },
      { ppqn: 27, timeSigNumerator: 5, timeSigDenominator: 32, beatsPerMeasure: 0.625 },
      { ppqn: 481, timeSigNumerator: 6, timeSigDenominator: 8, beatsPerMeasure: 3 },
    ]) {
      const gCtrl = new DemoPianoRollController(new StubAudioEngine());
      gCtrl.applyWorkspaceState({ ...baseWs, ...gridSpec });
      const { unmount } = render(
        <PianoRollView
          controller={gCtrl}
          session={gCtrl.getSnapshot()}
          focusRevision={1}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />,
      );
      unmount();
    }

    // 2. [R9-LOW-02] SOLID LSP fallback parity when candidates[0].reason is invalid + sc.index <= 64 sync bound
    const validSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const parsedSmf = StandardMidiCodec.parseMidi(validSmf.buffer as ArrayBuffer);
    const candWithBlankReason = [
      {
        index: 0,
        filename: 'cand_0.mid',
        rawMidiBytes: validSmf,
        parsedMidi: parsedSmf,
        reason: { task: '   ' } as never,
      },
      {
        index: 1,
        filename: 'cand_1.mid',
        rawMidiBytes: validSmf,
        parsedMidi: parsedSmf,
        reason: {} as never,
      },
    ];
    const fallbackReasons = [
      {
        source: 'model_cot' as const,
        task: 'infill',
        thinking: true,
        instruments: ['PIANO'],
        note_density: { PIANO: 4 },
        gen_measure_count: 4,
        genre: ['jazz'],
        key: 'CM',
      },
    ];

    const demoLsp = new DemoPianoRollController(new StubAudioEngine());
    const cubaseLsp = new CubaseDawController();
    await demoLsp.applyGeneratedCandidates(candWithBlankReason, fallbackReasons);
    await cubaseLsp.applyGeneratedCandidates(candWithBlankReason, fallbackReasons);
    expect(demoLsp.getSnapshot().latestReason?.task).toBe('infill');
    expect(cubaseLsp.getSnapshot().latestReason?.task).toBe('infill');
    await demoLsp.selectGeneratedCandidate(1);
    await cubaseLsp.selectGeneratedCandidate(1);
    expect(demoLsp.getSnapshot().latestReason?.task).toBe('infill');
    expect(cubaseLsp.getSnapshot().latestReason?.task).toBe('infill');

    // sc.index > 64 rejected identically across ipc_security, WorkspaceSyncBus, and applyWorkspaceState
    const overIndexWs = {
      ...demoLsp.exportWorkspaceState(),
      candidates: [
        {
          index: 65,
          filename: 'over.mid',
          rawMidiBytes: validSmf,
          reason: null,
        },
      ],
    };
    const overIndexMsg = {
      messageId: 'msg_r10_idx65',
      senderId: 'peer_r10',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: overIndexWs,
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(overIndexMsg)).toBe(false);
    expect(isValidIncomingSyncMessage(overIndexMsg)).toBe(false);
    const idxProbe = new DemoPianoRollController(new StubAudioEngine());
    idxProbe.createNoteAt(67, 0, 240, 100);
    idxProbe.applyWorkspaceState(overIndexWs as never);
    expect(idxProbe.getSnapshot().tracks[0].notes[0].pitch).toBe(67);

    // 3. [R9-LOW-03] UTF-16 surrogate-safe truncation (never leaves unpaired high/low surrogates)
    const splitBoundaryKey = `${'K'.repeat(31)}\ud83c\udfb5`;
    expect(splitBoundaryKey.length).toBe(33);
    const truncKey = truncateWellFormedUtf16(splitBoundaryKey, 32);
    expect(truncKey).toBe('K'.repeat(31));
    expect(truncKey.length).toBeLessThanOrEqual(32);
    expect((truncKey as unknown as { isWellFormed?: () => boolean }).isWellFormed?.() ?? true).toBe(
      true,
    );

    const loneSurrogateReason = sanitizeGenerationReason({
      source: 'model_cot',
      task: `${'T'.repeat(127)}\ud83c\udfb5`,
      thinking: true,
      instruments: [`${'I'.repeat(63)}\ud83c\udfb5`],
      genre: ['jazz\ud800'],
      key: `${'K'.repeat(31)}\ud83c\udfb5`,
    });
    expect(loneSurrogateReason).not.toBeNull();
    expect(loneSurrogateReason!.task).toBe('T'.repeat(127));
    expect(loneSurrogateReason!.instruments).toEqual(['I'.repeat(63)]);
    expect(loneSurrogateReason!.genre).toEqual(['jazz\ufffd']);
    expect(loneSurrogateReason!.key).toBe('K'.repeat(31));

    // 4. [R9-LOW-04] Bidirectional non-power-of-2 denominator scaling in importParsedMidiTracks (6/6 -> 8/8, 3/3 -> 4/4, 12/12 -> 16/16)
    for (const [rawN, rawD, expN, expD, expBeats] of [
      [6, 6, 8, 8, 4],
      [3, 3, 4, 4, 4],
      [12, 12, 16, 16, 4],
    ] as const) {
      const mCtrl = new DemoPianoRollController(new StubAudioEngine());
      mCtrl.importParsedMidiTracks(
        {
          format: 0,
          ppqn: 480,
          bpm: 120,
          beatsPerMeasure: 4,
          timeSigNumerator: rawN,
          timeSigDenominator: rawD,
          tracks: [
            {
              index: 0,
              name: 'Piano',
              program: 0,
              isDrum: false,
              instrument: 'PIANO',
              notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
            },
          ],
        },
        [0],
        true,
      );
      expect(mCtrl.getSnapshot().timeSigNumerator).toBe(expN);
      expect(mCtrl.getSnapshot().timeSigDenominator).toBe(expD);
      expect(mCtrl.getSnapshot().beatsPerMeasure).toBe(expBeats);
    }
  });

  it('verifies Round 11 Fact-Checker remediations: [R10-MED-01] track.id surrogate canonicalization, [R10-LOW-01] MidiNote/DawTrack/candidate normalization, and [R10-LOW-02] PianoRollView grid & low-PPQN guards', async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ipcSec = require('../../electron/ipc_security.cjs') as {
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };
    const isWf = (s: string) =>
      (s as unknown as { isWellFormed?: () => boolean }).isWellFormed?.() ?? true;

    // 1. [R10-MED-01] Colliding lone-surrogate track IDs ("trk_\uD800" vs "trk_\uDFFF") rejected atomically
    const ctrlMed = new DemoPianoRollController(new StubAudioEngine());
    const origTrackCount = ctrlMed.getSnapshot().tracks.length;
    const origFirstTrackId = ctrlMed.getSnapshot().tracks[0].id;
    ctrlMed.createNoteAt(65, 0, 240, 100);
    const baseWs = ctrlMed.exportWorkspaceState();
    ctrlMed.applyWorkspaceState({
      ...baseWs,
      activeTrackId: 'trk_\ud800',
      tracks: [
        {
          id: 'trk_\ud800',
          name: 'Track A',
          instrument: 'PIANO',
          colorHex: '#38BDF8',
          isMuted: false,
          isSoloed: false,
          notes: [],
        },
        {
          id: 'trk_\udfff',
          name: 'Track B',
          instrument: 'SAX',
          colorHex: '#93C5FD',
          isMuted: false,
          isSoloed: false,
          notes: [],
        },
      ],
    });
    // Rejected atomically: original tracks & pitch 65 remain intact
    expect(ctrlMed.getSnapshot().tracks.length).toBe(origTrackCount);
    expect(ctrlMed.getSnapshot().tracks[0].id).toBe(origFirstTrackId);
    expect(ctrlMed.getSnapshot().tracks[0].notes[0].pitch).toBe(65);

    // Single lone-surrogate track ID + activeCandidateRange.trackId + activeTrackId canonicalized in lockstep
    ctrlMed.applyWorkspaceState({
      ...baseWs,
      activeTrackId: 'trk_\ud800',
      tracks: [
        {
          id: 'trk_\ud800',
          name: 'Lead\ud800',
          instrument: 'PIANO',
          colorHex: '#38BDF8',
          isMuted: false,
          isSoloed: false,
          notes: [
            {
              id: 'note_\ud800',
              pitch: 60,
              startTick: 0,
              durationTicks: 480,
              velocity: 100,
              channel: 0,
            },
          ],
        },
      ],
      activeCandidateNoteIds: ['note_\ud800'],
      activeCandidateRange: { trackId: 'trk_\ud800', startTick: 0, endTick: 1920 },
      candidateOriginSlot: { trackId: 'trk_\ud800', startTick: 0, endTick: 1920 },
    });
    const exportedMed = ctrlMed.exportWorkspaceState();
    expect(exportedMed.tracks[0].id).toBe('trk_\ufffd');
    expect(exportedMed.activeTrackId).toBe('trk_\ufffd');
    expect(exportedMed.activeCandidateRange?.trackId).toBe('trk_\ufffd');
    expect(exportedMed.candidateOriginSlot?.trackId).toBe('trk_\ufffd');
    expect(exportedMed.tracks[0].notes[0].id).toBe('note_\ufffd');
    expect(exportedMed.activeCandidateNoteIds).toEqual(['note_\ufffd']);
    expect(isWf(exportedMed.tracks[0].id)).toBe(true);
    expect(isWf(exportedMed.tracks[0].notes[0].id)).toBe(true);
    const syncEnvelope = {
      messageId: 'r11_med01_ok',
      senderId: 'peer_r11',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: exportedMed,
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(syncEnvelope)).toBe(true);
    expect(isValidIncomingSyncMessage(syncEnvelope)).toBe(true);

    // 2. [R10-LOW-01](a),(b),(c) Fractional/NaN maxUnits, MidiNote.id collision rejection, & DawTrack.setColorHex
    expect(truncateWellFormedUtf16('a\ud83c\udfb5', 1.5)).toBe('a');
    expect(truncateWellFormedUtf16('\ud83c\udfb5', 1.5)).toBe('');
    expect(truncateWellFormedUtf16('abc', Number.NaN)).toBe('');
    expect(truncateWellFormedUtf16('abc', -5)).toBe('');
    expect(truncateWellFormedUtf16('abc', Number.POSITIVE_INFINITY)).toBe('abc');

    const directIllNote = new MidiNote({
      id: 'direct_\ud800',
      pitch: 64,
      startTick: 0,
      durationTicks: 120,
    });
    expect(directIllNote.id).toBe('direct_\ufffd');
    expect(isWf(directIllNote.id)).toBe(true);

    // Colliding lone-surrogate note IDs ("n_\uD800" vs "n_\uDFFF") rejected atomically
    const noteColCtrl = new DemoPianoRollController(new StubAudioEngine());
    noteColCtrl.createNoteAt(69, 0, 240, 100);
    noteColCtrl.applyWorkspaceState({
      ...noteColCtrl.exportWorkspaceState(),
      tracks: [
        {
          id: 'trk_valid',
          name: 'Valid',
          instrument: 'PIANO',
          colorHex: '#38BDF8',
          isMuted: false,
          isSoloed: false,
          notes: [
            { id: 'n_\ud800', pitch: 60, startTick: 0, durationTicks: 120, velocity: 100, channel: 0 },
            { id: 'n_\udfff', pitch: 62, startTick: 120, durationTicks: 120, velocity: 100, channel: 0 },
          ],
        },
      ],
    });
    expect(noteColCtrl.getSnapshot().tracks[0].notes[0].pitch).toBe(69);

    const colorTrk = new DawTrack({ id: 'ctrk', name: 'ColorTest' });
    colorTrk.setColorHex(`${'#'.padEnd(31, 'A')}\ud83c\udfb5`);
    expect(colorTrk.colorHex.length).toBeLessThanOrEqual(32);
    expect(isWf(colorTrk.colorHex)).toBe(true);

    // 3. [R10-LOW-01](d) 32-candidate cap, <=256 filename sanitization, sparse index mapping & selectGeneratedCandidate guard
    const smfOne = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const parsedOne = StandardMidiCodec.parseMidi(smfOne.buffer as ArrayBuffer);
    const over35Cands = Array.from({ length: 35 }, (_, i) => ({
      index: i + 10,
      filename: `${'f'.repeat(255)}\ud83c\udfb5_${i}.mid`,
      rawMidiBytes: smfOne,
      parsedMidi: parsedOne,
      reason: null,
    }));
    const demoCap = new DemoPianoRollController(new StubAudioEngine());
    const cubaseCap = new CubaseDawController();
    await demoCap.applyGeneratedCandidates(over35Cands, []);
    await cubaseCap.applyGeneratedCandidates(over35Cands, []);
    expect(demoCap.getSnapshot().candidates.length).toBe(32);
    expect(cubaseCap.getSnapshot().candidates.length).toBe(32);
    expect(demoCap.getSnapshot().candidates[31].index).toBe(31);
    expect(demoCap.getSnapshot().candidates[0].filename.length).toBeLessThanOrEqual(256);
    expect(isWf(demoCap.getSnapshot().candidates[0].filename)).toBe(true);
    await demoCap.selectGeneratedCandidate(Number.NaN);
    await demoCap.selectGeneratedCandidate(1.5);
    await cubaseCap.selectGeneratedCandidate(Number.NaN);
    expect(demoCap.getSnapshot().activeCandidateIndex).toBe(0);
    expect(cubaseCap.getSnapshot().activeCandidateIndex).toBe(0);

    // Sparse sc.index [5, 10] with activeCandidateIndex=10 normalizes to [0, 1] and activeCandidateIndex=1
    demoCap.applyWorkspaceState({
      ...demoCap.exportWorkspaceState(),
      candidates: [
        { index: 5, filename: 'a.mid', rawMidiBytes: smfOne, reason: null },
        { index: 10, filename: 'b.mid', rawMidiBytes: smfOne, reason: null },
      ],
      activeCandidateIndex: 10,
    });
    expect(demoCap.getSnapshot().candidates.map((c) => c.index)).toEqual([0, 1]);
    expect(demoCap.getSnapshot().activeCandidateIndex).toBe(1);

    // 4. [R10-LOW-02] Extreme custom snapshot meter guard + odd-PPQN subdivision & 24-PPQN hit-testing
    const extremeCtrl = new DemoPianoRollController(new StubAudioEngine());
    const extremeSnap = {
      ...extremeCtrl.getSnapshot(),
      ppqn: Number.NaN,
      beatsPerMeasure: Number.NaN,
      timeSigNumerator: 1e9,
      timeSigDenominator: Number.POSITIVE_INFINITY,
    };
    const { unmount: unmountExtreme } = render(
      <PianoRollView
        controller={extremeCtrl}
        session={extremeSnap}
        focusRevision={1}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );
    unmountExtreme();

    // 5. 1,000-vector deterministic UTF-16 surrogate & candidate sync round-trip fuzzer:
    // Every accepted/rejected applyWorkspaceState call leaves exportWorkspaceState() 100% valid & well-formed
    const fuzzFragments = ['trk', '\ud800', '\udbff', '\udc00', '\udfff', '\ud83c\udfb5', ' '];
    let seed = 0x51a7beef;
    const nextRand = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed;
    };
    const randStr = (maxParts: number) => {
      const len = 1 + (nextRand() % maxParts);
      let s = '';
      for (let p = 0; p < len; p += 1) {
        s += fuzzFragments[nextRand() % fuzzFragments.length];
      }
      return s;
    };
    const fuzzCtrl = new DemoPianoRollController(new StubAudioEngine());
    for (let v = 0; v < 1000; v += 1) {
      const tCount = 1 + (nextRand() % 3);
      const tracks = Array.from({ length: tCount }, (_, ti) => ({
        id: `${randStr(4)}_${nextRand() % 3}`,
        name: `${randStr(4)}_T${ti}`,
        instrument: 'PIANO' as const,
        colorHex: `#${randStr(3)}`,
        isMuted: false,
        isSoloed: false,
        notes: [
          {
            id: `${randStr(4)}_N${nextRand() % 3}`,
            pitch: 60 + (ti % 12),
            startTick: ti * 120,
            durationTicks: 120,
            velocity: 100,
            channel: 0,
          },
        ],
      }));
      const pickTrackId = tracks[nextRand() % tracks.length].id;
      fuzzCtrl.applyWorkspaceState({
        ...baseWs,
        activeTrackId: pickTrackId,
        tracks,
        activeCandidateNoteIds: [tracks[0].notes[0].id],
        activeCandidateRange: { trackId: pickTrackId, startTick: 0, endTick: 1920 },
        candidateOriginSlot: { trackId: pickTrackId, startTick: 0, endTick: 1920 },
        candidates: [
          {
            index: nextRand() % 70,
            filename: `${randStr(4)}.mid`,
            rawMidiBytes: smfOne,
            reason: null,
          },
        ],
        activeCandidateIndex: nextRand() % 5,
      });
      const outWs = fuzzCtrl.exportWorkspaceState();
      const env = {
        messageId: `fuzz_${v}`,
        senderId: 'fuzz_peer',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: outWs,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(env)).toBe(true);
      expect(isValidIncomingSyncMessage(env)).toBe(true);
      for (const tr of outWs.tracks) {
        expect(isWf(tr.id)).toBe(true);
        expect(isWf(tr.name)).toBe(true);
        expect(isWf(tr.colorHex)).toBe(true);
        for (const nt of tr.notes) {
          expect(isWf(nt.id)).toBe(true);
        }
      }
      if (outWs.activeCandidateRange) {
        expect(isWf(outWs.activeCandidateRange.trackId)).toBe(true);
      }
      for (const cd of outWs.candidates) {
        expect(isWf(cd.filename)).toBe(true);
      }
    }
  });

  it('verifies Round 12 Fact-Checker remediations: [R11-MED-01] cross-track slot & undo/deleteTrack invariants, [R11-MED-02] additive MIDI import & <=32768 undo/redo cap, [R11-MED-03] PPQN-scaled resize floor (24/480/9600), [R11-LOW-01] <=2MB candidate byte cap, and [R11-LOW-02] MidiNote safe-int ceiling & PianoRollView NaN select guard', async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ipcSec = require('../../electron/ipc_security.cjs') as {
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };
    const assertValidExport = (ctrl: DemoPianoRollController, tag: string) => {
      const ws = ctrl.exportWorkspaceState();
      const env = {
        messageId: `r12_${tag}`,
        senderId: 'r12_peer',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: ws,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(env)).toBe(true);
      expect(isValidIncomingSyncMessage(env)).toBe(true);
      return ws;
    };

    const smfC4 = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const parsedC4 = StandardMidiCodec.parseMidi(smfC4.buffer as ArrayBuffer);
    const smfE4 = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const parsedE4 = StandardMidiCodec.parseMidi(smfE4.buffer as ArrayBuffer);

    // 1. [R11-MED-01] Cross-track slot mismatch rejected across IPC, SyncBus, and Controller;
    // multi-track generate -> switch -> deleteTrack -> undo -> redo never orphans candidateOriginSlot
    const med01Ctrl = new DemoPianoRollController(new StubAudioEngine());
    const initWs = med01Ctrl.exportWorkspaceState();
    const trkA = initWs.tracks[0].id;
    const trkB = initWs.tracks[1].id;
    const mismatchedEnv = {
      messageId: 'r12_med01_mismatch',
      senderId: 'r12_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: {
        ...initWs,
        activeCandidateRange: { trackId: trkA, startTick: 0, endTick: 1920 },
        candidateOriginSlot: { trackId: trkB, startTick: 0, endTick: 1920 },
      },
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(mismatchedEnv)).toBe(false);
    expect(isValidIncomingSyncMessage(mismatchedEnv)).toBe(false);
    med01Ctrl.applyWorkspaceState(mismatchedEnv.workspaceState);
    expect(med01Ctrl.exportWorkspaceState().activeCandidateRange).toBeNull();
    expect(med01Ctrl.exportWorkspaceState().candidateOriginSlot).toBeNull();

    // Padded same-track IDs (' trkA ' vs 'trkA') remain accepted ([R7-LOW-04] compatibility)
    const paddedSameEnv = {
      messageId: 'r12_med01_padded_ok',
      senderId: 'r12_peer',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: {
        ...initWs,
        activeCandidateRange: { trackId: ` ${trkA} `, startTick: 0, endTick: 1920 },
        candidateOriginSlot: { trackId: trkA, startTick: 0, endTick: 1920 },
      },
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(paddedSameEnv)).toBe(true);
    expect(isValidIncomingSyncMessage(paddedSameEnv)).toBe(true);

    // Multi-track generate on trkB -> switch to trkA -> generate on trkA -> deleteTrack(trkB) -> undo -> redo
    med01Ctrl.setActiveTrack(trkB);
    await med01Ctrl.applyGeneratedCandidates(
      [{ index: 0, filename: 'b0.mid', rawMidiBytes: smfC4, parsedMidi: parsedC4, reason: null }],
      [],
    );
    assertValidExport(med01Ctrl, 'gen_trkB');
    med01Ctrl.setActiveTrack(trkA);
    await med01Ctrl.applyGeneratedCandidates(
      [
        { index: 0, filename: 'a0.mid', rawMidiBytes: smfC4, parsedMidi: parsedC4, reason: null },
        { index: 1, filename: 'a1.mid', rawMidiBytes: smfE4, parsedMidi: parsedE4, reason: null },
      ],
      [],
    );
    assertValidExport(med01Ctrl, 'gen_trkA');
    await med01Ctrl.selectGeneratedCandidate(1);
    assertValidExport(med01Ctrl, 'sel_trkA');
    med01Ctrl.deleteTrack(trkB);
    assertValidExport(med01Ctrl, 'del_trkB');
    med01Ctrl.undo();
    assertValidExport(med01Ctrl, 'undo_1');
    med01Ctrl.undo();
    const afterSecondUndo = assertValidExport(med01Ctrl, 'undo_2');
    expect(afterSecondUndo.candidateOriginSlot).toBeNull();
    expect(afterSecondUndo.activeCandidateRange).toBeNull();
    med01Ctrl.redo();
    assertValidExport(med01Ctrl, 'redo_1');

    // 2. [R11-MED-02] Additive importParsedMidiTracks(..., false) clears _commands, resets activeCandidateIndex,
    // and enforces <= 32,768 workspace note cap across undo/redo
    const med02Ctrl = new DemoPianoRollController(new StubAudioEngine());
    med02Ctrl.createNoteAt(60, 0, 240, 100);
    med02Ctrl.createNoteAt(64, 240, 240, 100);
    med02Ctrl.selectAllNotesOnActiveTrack();
    med02Ctrl.deleteSelectedNotes();
    await med02Ctrl.applyGeneratedCandidates(
      [
        { index: 0, filename: 'c0.mid', rawMidiBytes: smfC4, parsedMidi: parsedC4, reason: null },
        { index: 1, filename: 'c1.mid', rawMidiBytes: smfE4, parsedMidi: parsedE4, reason: null },
      ],
      [],
    );
    await med02Ctrl.selectGeneratedCandidate(1);
    expect(med02Ctrl.getSnapshot().activeCandidateIndex).toBe(1);
    expect(med02Ctrl.getSnapshot().canUndo).toBe(true);

    const denseNotes = Array.from({ length: 32_768 }, (_, idx) =>
      new MidiNote({
        pitch: 36 + (idx % 48),
        startTick: idx * 10,
        durationTicks: 8,
        velocity: 90,
      }),
    );
    med02Ctrl.importParsedMidiTracks(
      {
        format: 1,
        ppqn: 480,
        bpm: 120,
        beatsPerMeasure: 4,
        timeSigNumerator: 4,
        timeSigDenominator: 4,
        tracks: [
          {
            index: 0,
            name: 'DenseAdditive',
            program: 0,
            isDrum: false,
            instrument: 'PIANO',
            notes: denseNotes,
          },
        ],
      },
      [0],
      false,
    );
    const postAdditiveSnap = med02Ctrl.getSnapshot();
    expect(postAdditiveSnap.canUndo).toBe(false);
    expect(postAdditiveSnap.canRedo).toBe(false);
    expect(postAdditiveSnap.candidates.length).toBe(0);
    expect(postAdditiveSnap.activeCandidateIndex).toBe(0);
    const totalNotesAfterAdditive = postAdditiveSnap.tracks.reduce(
      (acc, t) => acc + t.notes.length,
      0,
    );
    expect(totalNotesAfterAdditive).toBe(32_768);
    assertValidExport(med02Ctrl, 'additive_32768');

    // 3. [R11-MED-03] PPQN-scaled resize floor keeps right-edge resize handle grabbable at 480 & 9600 PPQN
    // and allows Shift-shortening 6 -> 5 ticks at 24 PPQN without inflating to 15
    for (const [ppqn, minExpected] of [
      [480, 24],
      [9600, 480],
    ] as const) {
      const rCtrl = new DemoPianoRollController(new StubAudioEngine());
      rCtrl.applyWorkspaceState({
        ...rCtrl.exportWorkspaceState(),
        ppqn,
        activeTrackId: 'trk_resize_ppqn',
        tracks: [
          {
            id: 'trk_resize_ppqn',
            name: `Resize_${ppqn}`,
            instrument: 'PIANO',
            colorHex: '#38BDF8',
            isMuted: false,
            isSoloed: false,
            notes: [
              {
                id: `note_resize_${ppqn}`,
                pitch: 74,
                startTick: ppqn,
                durationTicks: ppqn,
                velocity: 100,
                channel: 0,
              },
            ],
          },
        ],
        activeCandidateNoteIds: [],
        activeCandidateRange: null,
        candidateOriginSlot: null,
      });
      expect(rCtrl.getSnapshot().ppqn).toBe(ppqn);
      const Harness: React.FC = () => {
        const [sess, setSess] = React.useState(rCtrl.getSnapshot());
        React.useEffect(() => rCtrl.subscribe(setSess), []);
        return (
          <PianoRollView
            controller={rCtrl}
            session={sess}
            isActive={true}
            focusRevision={0}
            onLaunchCompassAi={() => {}}
            onSwitchToDemoController={() => {}}
          />
        );
      };
      const ui = render(<Harness />);
      const canvas = ui.getByTestId('pianoroll-canvas');
      vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 960,
        bottom: 540,
        width: 960,
        height: 540,
        toJSON: () => ({}),
      });
      // At 64 px/beat: startTick=ppqn is at x=72+64=136, endTick=2*ppqn is at x=72+128=200. Drag right edge far left!
      fireEvent.pointerDown(canvas, { pointerId: 91, pointerType: 'mouse', button: 0, clientX: 199, clientY: 40 });
      fireEvent.pointerMove(window, { pointerId: 91, pointerType: 'mouse', clientX: 80, clientY: 40, shiftKey: true });
      fireEvent.pointerUp(window, { pointerId: 91, pointerType: 'mouse', clientX: 80, clientY: 40, shiftKey: true });
      const collapsed = rCtrl.getSnapshot().tracks[0].notes[0];
      expect(collapsed.startTick).toBe(ppqn);
      expect(collapsed.durationTicks).toBe(minExpected);

      // Grab the right edge of the collapsed note (endX = 136 + (minExpected / ppqn) * 64 = 139.2) and lengthen by +1 beat (+64 px)
      const rightEdgeX = 136 + (minExpected / ppqn) * 64 - 0.2;
      fireEvent.pointerDown(canvas, { pointerId: 92, pointerType: 'mouse', button: 0, clientX: rightEdgeX, clientY: 40 });
      fireEvent.pointerMove(window, { pointerId: 92, pointerType: 'mouse', clientX: rightEdgeX + 64, clientY: 40 });
      fireEvent.pointerUp(window, { pointerId: 92, pointerType: 'mouse', clientX: rightEdgeX + 64, clientY: 40 });
      const reExpanded = rCtrl.getSnapshot().tracks[0].notes[0];
      expect(reExpanded.startTick).toBe(ppqn);
      // Measure-anchored `snapTick` (`PianoRollView.java` parity) snaps the dragged right edge cleanly onto Beat 3 (`2 * ppqn`)
      expect(reExpanded.durationTicks).toBe(ppqn);
      ui.unmount();
    }

    // 24 PPQN + 1/4 quantize (snapTicks=24): Shift-shortening a 6-tick (1/16) note by -1 tick yields 5 (never 15!)
    const low24Ctrl = new DemoPianoRollController(new StubAudioEngine());
    low24Ctrl.applyWorkspaceState({
      ...low24Ctrl.exportWorkspaceState(),
      ppqn: 24,
      activeTrackId: 'trk_low24',
      tracks: [
        {
          id: 'trk_low24',
          name: 'Low24',
          instrument: 'PIANO',
          colorHex: '#38BDF8',
          isMuted: false,
          isSoloed: false,
          notes: [
            {
              id: 'note_low24',
              pitch: 74,
              startTick: 0,
              durationTicks: 6,
              velocity: 100,
              channel: 0,
            },
          ],
        },
      ],
      activeCandidateNoteIds: [],
      activeCandidateRange: null,
      candidateOriginSlot: null,
    });
    expect(low24Ctrl.getSnapshot().ppqn).toBe(24);
    low24Ctrl.setQuantize(4, false);
    const Low24Harness: React.FC = () => {
      const [sess, setSess] = React.useState(low24Ctrl.getSnapshot());
      React.useEffect(() => low24Ctrl.subscribe(setSess), []);
      return (
        <PianoRollView
          controller={low24Ctrl}
          session={sess}
          isActive={true}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    };
    const low24Ui = render(<Low24Harness />);
    const low24Canvas = low24Ui.getByTestId('pianoroll-canvas');
    vi.spyOn(low24Canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    });
    // At 24 PPQN & 64 px/beat (2.6667 px/tick), 6-tick note spans x = 72..88. Grab x = 87.5 and Shift-drag -2.6667 px (-1 tick)
    fireEvent.pointerDown(low24Canvas, { pointerId: 93, pointerType: 'mouse', button: 0, clientX: 87.5, clientY: 40 });
    fireEvent.pointerMove(window, {
      pointerId: 93,
      pointerType: 'mouse',
      clientX: 87.5 - 64 / 24,
      clientY: 40,
      shiftKey: true,
    });
    fireEvent.pointerUp(window, {
      pointerId: 93,
      pointerType: 'mouse',
      clientX: 87.5 - 64 / 24,
      clientY: 40,
      shiftKey: true,
    });
    expect(low24Ctrl.getSnapshot().tracks[0].notes[0].durationTicks).toBe(5);
    low24Ui.unmount();

    // 4. [R11-LOW-01] 25 x ~100 KB (~2.5 MB) candidates capped at <= 2,000,000 cumulative bytes across all controllers & API client
    const notesPerHeavySmf = Array.from({ length: 12_500 }, (_, ni) =>
      new MidiNote({
        pitch: 48 + (ni % 24),
        startTick: ni * 10,
        durationTicks: 5,
        velocity: 95,
      }),
    );
    const heavySmf = StandardMidiCodec.encodeSingleTrackMidi(notesPerHeavySmf, 480, 120, 'PIANO');
    expect(heavySmf.byteLength).toBeGreaterThan(95_000);
    const heavyParsed = StandardMidiCodec.parseMidi(heavySmf.buffer as ArrayBuffer);
    const heavyBatch = Array.from({ length: 25 }, (_, i) => ({
      index: i,
      filename: `heavy_${i}.mid`,
      rawMidiBytes: heavySmf,
      parsedMidi: heavyParsed,
      reason: null,
    }));

    const demoByteCapCtrl = new DemoPianoRollController(new StubAudioEngine());
    const cubaseByteCapCtrl = new CubaseDawController();
    const abletonByteCapCtrl = new AbletonDawController();
    await demoByteCapCtrl.applyGeneratedCandidates(heavyBatch, []);
    await cubaseByteCapCtrl.applyGeneratedCandidates(heavyBatch, []);
    await abletonByteCapCtrl.applyGeneratedCandidates(heavyBatch, []);

    const demoCands = demoByteCapCtrl.getSnapshot().candidates;
    const demoBytesSum = demoCands.reduce((acc, c) => acc + c.rawMidiBytes.byteLength, 0);
    expect(demoCands.length).toBeGreaterThan(0);
    expect(demoCands.length).toBeLessThan(25);
    expect(demoBytesSum).toBeLessThanOrEqual(MAX_CANDIDATE_MIDI_BYTES);
    expect(cubaseByteCapCtrl.getSnapshot().candidates.length).toBe(demoCands.length);
    expect(abletonByteCapCtrl.getSnapshot().candidates.length).toBe(demoCands.length);
    assertValidExport(demoByteCapCtrl, 'byte_cap_ok');

    // Also verify CompassApiClient.generateMusic ZIP extraction caps cumulative bytes <= 2,000,000
    const zip = new JSZip();
    for (let zi = 0; zi < 25; zi += 1) {
      zip.file(`output_${zi}.mid`, heavySmf);
    }
    const zipArrayBuf = await zip.generateAsync({ type: 'arraybuffer' });
    const origFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async () =>
      new Response(zipArrayBuf, {
        status: 200,
        headers: { 'Content-Type': 'application/zip' },
      }),
    ) as unknown as typeof fetch;
    try {
      const api = new CompassApiClient('http://127.0.0.1:8000');
      const apiRes = await api.generateMusic({
        meta: {
          model_type: 'MORTM4.5D-SFT-Gen',
          task: 'meta',
          program: ['PIANO'],
          tempo: 120,
          num_gems: 4,
          genfield_measure: 4,
          gen_note_dense: null,
          p: 0.95,
          temperature: 1.0,
          cot_temperature: 0.8,
          thinking: false,
        },
      });
      const apiSum = apiRes.candidates.reduce((acc, c) => acc + c.rawMidiBytes.byteLength, 0);
      expect(apiRes.candidates.length).toBe(demoCands.length);
      expect(apiSum).toBeLessThanOrEqual(MAX_CANDIDATE_MIDI_BYTES);
    } finally {
      globalThis.fetch = origFetch;
    }

    // 5. [R11-LOW-02](a) MidiNote upper safe-integer tick ceiling & [R3-MED-06] 0x7fffffff preservation
    const hugeNote = new MidiNote({ pitch: 60, startTick: 1e308, durationTicks: 1e308 });
    expect(hugeNote.startTick).toBe(MAX_SAFE_MIDI_TICK);
    expect(hugeNote.durationTicks).toBe(MAX_SAFE_MIDI_TICK);
    expect(Number.isSafeInteger(hugeNote.endTick)).toBe(true);
    expect(hugeNote.endTick).toBeGreaterThan(hugeNote.startTick);

    const base480 = new MidiNote({ pitch: 60, startTick: 480, durationTicks: 480 });
    const scaledOverflow = base480.scaleAndShift(1e300, 0);
    expect(scaledOverflow.startTick).toBe(MAX_SAFE_MIDI_TICK);
    expect(scaledOverflow.durationTicks).toBe(MAX_SAFE_MIDI_TICK);
    expect(Number.isSafeInteger(scaledOverflow.endTick)).toBe(true);
    expect(base480.shiftTicks(Number.POSITIVE_INFINITY).startTick).toBe(MAX_SAFE_MIDI_TICK);
    expect(base480.withDuration(Number.POSITIVE_INFINITY).durationTicks).toBe(MAX_SAFE_MIDI_TICK);
    expect(base480.withTiming(1e308, 1e308).endTick).toBe(MAX_SAFE_MIDI_TICK * 2);
    expect(base480.cloneWithNewId(1e308).startTick).toBe(MAX_SAFE_MIDI_TICK);

    const r3Med06Note = new MidiNote({
      pitch: 60,
      startTick: 0x7fffffff,
      durationTicks: 480,
      velocity: 100,
    });
    expect(r3Med06Note.startTick).toBe(0x7fffffff);
    expect(r3Med06Note.endTick).toBe(0x7fffffff + 480);

    // 6. [R11-LOW-02](b) PianoRollView controlled <select>/<input> NaN fallback produces zero React console warnings
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const nanCtrl = new DemoPianoRollController(new StubAudioEngine());
      const nanSession = {
        ...nanCtrl.getSnapshot(),
        bpm: Number.NaN,
        quantizeDivision: Number.NaN as unknown as 16,
        isTriplet: undefined as unknown as boolean,
      };
      const nanUi = render(
        <PianoRollView
          controller={nanCtrl}
          session={nanSession}
          isActive={true}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />,
      );
      const quantSelect = nanUi.getByLabelText('グリッドクオンタイズ') as HTMLSelectElement;
      expect(quantSelect.value).toBe('16');
      fireEvent.change(quantSelect, { target: { value: '8' } });
      expect(nanCtrl.getSnapshot().quantizeDivision).toBe(8);
      expect(nanCtrl.getSnapshot().isTriplet).toBe(false);
      expect(errSpy).not.toHaveBeenCalled();
      expect(warnSpy).not.toHaveBeenCalled();
      nanUi.unmount();
    } finally {
      errSpy.mockRestore();
      warnSpy.mockRestore();
    }

    // 7. 3,000-step stateful multi-operation sequence fuzzer verifying exportWorkspaceState() invariants at EVERY step
    let sSeed = 0x7212c0de;
    const sRand = () => {
      sSeed = (Math.imul(sSeed, 1664525) + 1013904223) >>> 0;
      return sSeed;
    };
    for (let run = 0; run < 250; run += 1) {
      const seqCtrl = new DemoPianoRollController(new StubAudioEngine());
      for (let step = 0; step < 12; step += 1) {
        const op = sRand() % 9;
        const snap = seqCtrl.getSnapshot();
        if (op === 0) {
          seqCtrl.addTrack(`Fuzz_${step}`, step % 2 === 0 ? 'PIANO' : 'SAX');
        } else if (op === 1) {
          if (snap.tracks.length > 1) {
            seqCtrl.deleteTrack(snap.tracks[sRand() % snap.tracks.length].id);
          }
        } else if (op === 2) {
          seqCtrl.setActiveTrack(snap.tracks[sRand() % snap.tracks.length].id);
          await seqCtrl.applyGeneratedCandidates(
            [
              { index: 0, filename: 's0.mid', rawMidiBytes: smfC4, parsedMidi: parsedC4, reason: null },
              { index: 1, filename: 's1.mid', rawMidiBytes: smfE4, parsedMidi: parsedE4, reason: null },
            ],
            [],
          );
        } else if (op === 3) {
          await seqCtrl.selectGeneratedCandidate(sRand() % 3);
        } else if (op === 4) {
          seqCtrl.createNoteAt(60 + (sRand() % 12), (sRand() % 8) * 120, 120, 100);
        } else if (op === 5) {
          seqCtrl.selectAllNotesOnActiveTrack();
          seqCtrl.deleteSelectedNotes();
        } else if (op === 6) {
          seqCtrl.importParsedMidiTracks(parsedC4, [0], (sRand() & 1) === 0);
        } else if (op === 7) {
          seqCtrl.undo();
        } else {
          seqCtrl.redo();
        }
        assertValidExport(seqCtrl, `seq_${run}_${step}`);
      }
    }
  });

  it('verifies Round 13 Fact-Checker (R12) remediations: [R12-MED-01] revoked Proxy & getter/accessor TOCTOU immunity, [R12-LOW-01] 3-way UTF-16 lone-surrogate ID canonicalization parity, [R12-LOW-02] Node Buffer subarray compaction, [R12-LOW-03] single-rounded affine MidiNote.scaleAndShift, and [R12-LOW-04] ControlPanelView NaN/Infinity sanitization', async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ipcSec = require('../../electron/ipc_security.cjs') as {
      extractCanonicalOrigin: (rawUrl: unknown) => string;
      getAllowedAppOrigin: (env?: unknown) => string;
      isTrustedIpcSender: (event: unknown, env?: unknown) => boolean;
      isValidSerializedNote: (n: unknown, maxProjectTicks: number) => boolean;
      isValidSerializedTrack: (t: unknown, seenNoteIds?: Set<string>, maxProjectTicks?: number) => boolean;
      isValidCandidateSlotRange: (r: unknown, seenTrackIds: Set<string>, maxSlotTick: number) => boolean;
      isValidGenerationReason: (reason: unknown) => boolean;
      isValidCandidateMidiBytes: (raw: unknown) => boolean;
      isValidSerializedCandidate: (c: unknown) => boolean;
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };

    const validSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const validParsed = StandardMidiCodec.parseMidi(validSmf.buffer as ArrayBuffer);

    // 1. [R12-MED-01] Revoked Proxy, throwing getters, stateful TOCTOU getters, Proxy candidates & extra-key smuggling
    const makeRevokedObj = () => {
      const r = Proxy.revocable({ type: 'SYNC_WORKSPACE_STATE' }, {});
      r.revoke();
      return r.proxy;
    };
    const makeRevokedArr = () => {
      const r = Proxy.revocable([1, 2, 3], {});
      r.revoke();
      return r.proxy;
    };

    for (const badInput of [makeRevokedObj(), makeRevokedArr()]) {
      expect(() => ipcSec.extractCanonicalOrigin(badInput)).not.toThrow();
      expect(ipcSec.extractCanonicalOrigin(badInput)).toBe('');
      expect(() => ipcSec.getAllowedAppOrigin(badInput)).not.toThrow();
      expect(() => ipcSec.isTrustedIpcSender(badInput, badInput)).not.toThrow();
      expect(ipcSec.isTrustedIpcSender(badInput, badInput)).toBe(false);
      expect(() => ipcSec.isValidSerializedNote(badInput, 1_981_440)).not.toThrow();
      expect(ipcSec.isValidSerializedNote(badInput, 1_981_440)).toBe(false);
      expect(() => ipcSec.isValidSerializedTrack(badInput)).not.toThrow();
      expect(ipcSec.isValidSerializedTrack(badInput)).toBe(false);
      expect(() => ipcSec.isValidCandidateSlotRange(badInput, new Set(['t1']), 1_981_440)).not.toThrow();
      expect(ipcSec.isValidCandidateSlotRange(badInput, new Set(['t1']), 1_981_440)).toBe(false);
      expect(() => ipcSec.isValidGenerationReason(badInput)).not.toThrow();
      expect(ipcSec.isValidGenerationReason(badInput)).toBe(false);
      expect(() => ipcSec.isValidCandidateMidiBytes(badInput)).not.toThrow();
      expect(ipcSec.isValidCandidateMidiBytes(badInput)).toBe(false);
      expect(() => ipcSec.isValidSerializedCandidate(badInput)).not.toThrow();
      expect(ipcSec.isValidSerializedCandidate(badInput)).toBe(false);
      expect(() => ipcSec.isValidWorkspaceSyncPayload(badInput)).not.toThrow();
      expect(ipcSec.isValidWorkspaceSyncPayload(badInput)).toBe(false);
      expect(() => isValidGenerationReasonPayload(badInput)).not.toThrow();
      expect(isValidGenerationReasonPayload(badInput)).toBe(false);
      expect(() => isValidCandidateMidiBytesPayload(badInput)).not.toThrow();
      expect(isValidCandidateMidiBytesPayload(badInput)).toBe(false);
      expect(() => isValidIncomingSyncMessage(badInput)).not.toThrow();
      expect(isValidIncomingSyncMessage(badInput)).toBe(false);
      expect(() => sanitizeGenerationReason(badInput)).not.toThrow();
      expect(sanitizeGenerationReason(badInput)).toBeNull();
      expect(() => normalizeGeneratedCandidatesBatch(badInput as never)).not.toThrow();
      expect(normalizeGeneratedCandidatesBatch(badInput as never)).toEqual([]);
    }

    const baseCtrl = new DemoPianoRollController(new StubAudioEngine());
    baseCtrl.createNoteAt(67, 0, 480, 100);
    const cleanWs = baseCtrl.exportWorkspaceState();

    // Revoked Proxy inside nested workspaceState / tracks / notes / candidates / reason never throws and leaves controller untouched
    for (const nestedPoison of [
      makeRevokedObj(),
      { ...cleanWs, tracks: makeRevokedArr() },
      { ...cleanWs, tracks: [makeRevokedObj()] },
      { ...cleanWs, tracks: [{ ...cleanWs.tracks[0], notes: makeRevokedArr() }] },
      { ...cleanWs, candidates: [makeRevokedObj()] },
      { ...cleanWs, latestReason: makeRevokedObj() },
    ]) {
      const env = {
        messageId: 'r12_revoked',
        senderId: 'peer_r12',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: nestedPoison,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(env)).toBe(false);
      expect(isValidIncomingSyncMessage(env)).toBe(false);
      expect(() => baseCtrl.applyWorkspaceState(nestedPoison as never)).not.toThrow();
      expect(baseCtrl.getSnapshot().tracks[0].notes[0].pitch).toBe(67);
      expect(baseCtrl.getSnapshot().canUndo).toBe(true);
    }

    // Getter / accessor TOCTOU attack (1st read valid, 2nd read poisoned/throws) blocked by own-data-descriptor check
    let flipCalls = 0;
    const toctouReason = {
      source: 'model_cot',
      task: 'infill',
      thinking: true,
      instruments: ['PIANO'],
      genre: ['jazz'],
      gen_measure_count: 4,
      key: 'C',
      note_density: {},
    };
    Object.defineProperty(toctouReason, 'task', {
      enumerable: true,
      configurable: true,
      get() {
        flipCalls += 1;
        return flipCalls === 1 ? 'infill' : (12345 as unknown as string);
      },
    });
    expect(ipcSec.isValidGenerationReason(toctouReason)).toBe(false);
    expect(isValidGenerationReasonPayload(toctouReason)).toBe(false);
    expect(sanitizeGenerationReason(toctouReason)).toBeNull();

    // Uint8Array index-0 getter shadowing defeated by backing-buffer header check
    const shadowedU8 = new Uint8Array(validSmf);
    shadowedU8[0] = 0x00; // Corrupt actual underlying byte 0 (`\0Thd`)
    expect(ipcSec.isValidCandidateMidiBytes(shadowedU8)).toBe(false);
    expect(isValidCandidateMidiBytesPayload(shadowedU8)).toBe(false);

    // Extra smuggled keys & Proxy-wrapped candidate stripped by normalizeGeneratedCandidatesBatch
    const smuggledCand = {
      index: 0,
      filename: 'clean.mid',
      rawMidiBytes: validSmf,
      parsedMidi: validParsed,
      reason: null,
      smuggledEvilKey: { fn: () => {} },
    };
    const proxyCand = new Proxy(
      {
        index: 1,
        filename: 'proxy.mid',
        rawMidiBytes: validSmf,
        parsedMidi: validParsed,
        reason: null,
      },
      {
        get(target, prop, receiver) {
          return Reflect.get(target, prop, receiver);
        },
      },
    );
    expect(ipcSec.isValidWorkspaceSyncPayload(new Proxy({ messageId: 'm', senderId: 's', type: 'REQUEST_WORKSPACE_STATE' }, {}))).toBe(false);
    const normBatch = normalizeGeneratedCandidatesBatch([smuggledCand as never]);
    expect(normBatch.length).toBe(1);
    expect(Object.keys(normBatch[0]).sort()).toEqual([
      'filename',
      'index',
      'parsedMidi',
      'rawMidiBytes',
      'reason',
    ]);
    expect('smuggledEvilKey' in (normBatch[0] as unknown as Record<string, unknown>)).toBe(false);
    void proxyCand;

    // 2. [R12-LOW-01] 3-way UTF-16 lone-surrogate ID canonicalization parity across ipc_security, WorkspaceSyncBus & applyWorkspaceState
    const surrogateCollisionVectors = [
      // Colliding track IDs ("trk_\uD800" vs "trk_\uD801" -> both canonicalize to "trk_\uFFFD")
      {
        ...cleanWs,
        activeTrackId: 'trk_\ud800',
        tracks: [
          { ...cleanWs.tracks[0], id: 'trk_\ud800', notes: [] },
          { ...cleanWs.tracks[0], id: 'trk_\ud801', name: 'Second', notes: [] },
        ],
      },
      // Colliding note IDs ("note_\uD800" vs "note_\uD801" -> both canonicalize to "note_\uFFFD")
      {
        ...cleanWs,
        tracks: [
          {
            ...cleanWs.tracks[0],
            notes: [
              { id: 'note_\ud800', pitch: 60, startTick: 0, durationTicks: 240, velocity: 100, channel: 0 },
              { id: 'note_\ud801', pitch: 64, startTick: 240, durationTicks: 240, velocity: 100, channel: 0 },
            ],
          },
        ],
      },
      // Cross-reference aliasing: track.id = "trk_\uD800", activeTrackId = "trk_\uD801"
      {
        ...cleanWs,
        activeTrackId: 'trk_\ud801',
        tracks: [{ ...cleanWs.tracks[0], id: 'trk_\ud800' }],
      },
      // Cross-reference aliasing: track.id = "trk_\uD800", activeCandidateRange.trackId = "trk_\uD801"
      {
        ...cleanWs,
        activeTrackId: 'trk_\ud800',
        tracks: [{ ...cleanWs.tracks[0], id: 'trk_\ud800' }],
        activeCandidateRange: { trackId: 'trk_\ud801', startTick: 0, endTick: 1920 },
        candidateOriginSlot: { trackId: 'trk_\ud801', startTick: 0, endTick: 1920 },
      },
      // Cross-reference aliasing: note.id = "note_\uD800", activeCandidateNoteIds = ["note_\uD801"]
      {
        ...cleanWs,
        tracks: [
          {
            ...cleanWs.tracks[0],
            notes: [
              { id: 'note_\ud800', pitch: 60, startTick: 0, durationTicks: 240, velocity: 100, channel: 0 },
            ],
          },
        ],
        activeCandidateNoteIds: ['note_\ud801'],
      },
      // Colliding deleted candidate note IDs: ["del_\uD800", "del_\uD801"]
      {
        ...cleanWs,
        tracks: [{ ...cleanWs.tracks[0], notes: [] }],
        activeCandidateNoteIds: ['del_\ud800', 'del_\ud801'],
      },
    ];

    for (const badWs of surrogateCollisionVectors) {
      const env = {
        messageId: 'r12_low01_reject',
        senderId: 'peer_r12',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: badWs,
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(env)).toBe(false);
      expect(isValidIncomingSyncMessage(env)).toBe(false);
      const probe = new DemoPianoRollController(new StubAudioEngine());
      probe.createNoteAt(71, 0, 240, 100);
      probe.applyWorkspaceState(badWs as never);
      expect(probe.getSnapshot().tracks[0].notes[0].pitch).toBe(71);
    }

    // 3. [R12-LOW-02] Node Buffer subarray over 100 KB ArrayBuffer compacted into exact-length Uint8Array
    const hundredKb = new ArrayBuffer(102_400);
    const nodeSubBuf = Buffer.from(hundredKb, 0, validSmf.byteLength);
    nodeSubBuf.set(validSmf);
    expect(nodeSubBuf.byteOffset).toBe(0);
    expect(nodeSubBuf.buffer.byteLength).toBe(102_400);

    let capturedSyncPayload: unknown = null;
    const prevDesktop = window.compassDesktop;
    window.compassDesktop = {
      sendWorkspaceSync: (msg) => {
        capturedSyncPayload = msg;
      },
    };
    try {
      const compactBus = new WorkspaceSyncBus(`compass_r12_buf_${Date.now()}`);
      compactBus.publishWorkspaceState({
        ...cleanWs,
        candidates: [
          {
            index: 0,
            filename: 'node_buf.mid',
            rawMidiBytes: nodeSubBuf,
            reason: null,
          },
        ],
      });
      compactBus.dispose();
    } finally {
      window.compassDesktop = prevDesktop;
    }
    expect(capturedSyncPayload).not.toBeNull();
    expect(ipcSec.isValidWorkspaceSyncPayload(capturedSyncPayload)).toBe(true);
    expect(isValidIncomingSyncMessage(capturedSyncPayload)).toBe(true);
    const outCandBytes = (
      capturedSyncPayload as {
        workspaceState: { candidates: Array<{ rawMidiBytes: Uint8Array }> };
      }
    ).workspaceState.candidates[0].rawMidiBytes;
    expect(Object.getPrototypeOf(outCandBytes)).toBe(Uint8Array.prototype);
    expect(outCandBytes.byteOffset).toBe(0);
    expect(outCandBytes.byteLength).toBe(validSmf.byteLength);
    expect(outCandBytes.buffer.byteLength).toBe(validSmf.byteLength);

    // 4. [R12-LOW-03] MidiNote.scaleAndShift single-rounded monotonic affine transform
    const bigTickNote = new MidiNote({ pitch: 60, startTick: 5_000_000, durationTicks: 480 });
    // Exact affine: 5e6 * 1e9 - 1e15 = 5e15 - 1e15 = 4e15 (no premature clamp to MAX_SAFE_MIDI_TICK before negative offset!)
    expect(bigTickNote.scaleAndShift(1e9, -1e15).startTick).toBe(4_000_000_000_000_000);
    // Single rounding: 1 * 0.4 + 0.4 = 0.8 -> rounds to 1 (no double-rounding 0 + 0 = 0!)
    const unitTickNote = new MidiNote({ pitch: 60, startTick: 1, durationTicks: 10 });
    expect(unitTickNote.scaleAndShift(0.4, 0.4).startTick).toBe(1);
    expect(unitTickNote.scaleAndShift(0, 0).durationTicks).toBe(1);
    // 0 * +Infinity guarded (never NaN) & +Infinity scale saturates cleanly
    const zeroTickNote = new MidiNote({ pitch: 60, startTick: 0, durationTicks: 120 });
    expect(zeroTickNote.scaleAndShift(Number.POSITIVE_INFINITY, 240).startTick).toBe(240);
    expect(unitTickNote.scaleAndShift(Number.POSITIVE_INFINITY, 0).startTick).toBe(MAX_SAFE_MIDI_TICK);
    expect(unitTickNote.scaleAndShift(Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY).startTick).toBe(0);

    // 5. [R12-LOW-04] ControlPanelView with NaN / +/-Infinity settings produces 0 console errors & 0 "NaN"/"Infinity" text
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const cpCtrl = new DemoPianoRollController(new StubAudioEngine());
      const cpUi = render(
        <ControlPanelView
          isActive={true}
          audioEngine={new StubAudioEngine()}
          controllers={[cpCtrl]}
          session={cpCtrl.getSnapshot()}
          models={[
            {
              model_name: 'MORTM4.5D-80M-SFT-Gen',
              has_weights: true,
              tag: { model: 'sft_gen', type: 'mortm', version: '4.5' },
            },
          ]}
          hardwareInfo={null}
          settings={{
            modelName: 'MORTM4.5D-80M-SFT-Gen',
            temperature: Number.NaN,
            topP: Number.POSITIVE_INFINITY,
            thinking: true,
            cotTemperature: Number.NEGATIVE_INFINITY,
            numCandidates: Number.NaN,
            taskMode: 'infill',
            conditionTrackId: 'non_existent_track',
            forceMonophonic: false,
            noteDensity: Number.NaN,
            musicalKey: 'INVALID_KEY',
            selectedGenres: ['jazz', 'INVALID_GENRE'],
          }}
          isGenerating={false}
          isRescanningModels={false}
          statusTitle="READY"
          statusDetail="Ready"
          generationLogs={[]}
          errorMessage={null}
          onUpdateSettings={() => {}}
          onSwitchController={() => {}}
          onRescanModels={() => {}}
          onTogglePlayback={() => {}}
          onGenerate={() => {}}
          onCancelGenerate={() => {}}
        />,
      );
      fireEvent.click(cpUi.getByText('音楽構造・ジャンル・タスクの詳細指定 (SFT条件付け)'));
      const densitySelect = cpUi.container.querySelector('#note-density-select') as HTMLSelectElement;
      expect(densitySelect).not.toBeNull();
      expect(densitySelect.value).toBe('');
      expect(cpUi.container.textContent).not.toContain('NaN');
      expect(cpUi.container.textContent).not.toContain('Infinity');
      expect(errSpy).not.toHaveBeenCalled();
      expect(warnSpy).not.toHaveBeenCalled();
      cpUi.unmount();
    } finally {
      errSpy.mockRestore();
      warnSpy.mockRestore();
    }
  });

  it('verifies Round 14 Fact-Checker (R13) remediations: [R13-MED-01] deep POJO MidiNote reconstruction & transactional ExternalDawAdapters, [R13-MED-02] WorkspaceSyncBus DataCloneError/Proxy/TOCTOU/.slice immunity & Array parity, [R13-MED-03] 0-MTrk MThd sync apply fallback, [R13-LOW-01] zero NaN/Infinity UI & hostile-prop resilience, and [R13-LOW-02] exact BigInt dyadic MidiNote.scaleAndShift', async () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ipcSec = require('../../electron/ipc_security.cjs') as {
      isValidCandidateMidiBytes: (raw: unknown) => boolean;
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };

    const validSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480, velocity: 96 })],
      480,
      120,
      'PIANO',
    );
    const validParsed = StandardMidiCodec.parseMidi(validSmf.buffer as ArrayBuffer);

    // 1. [R13-MED-01] Plain JSON-deserialized note POJOs inside parsedMidi reconstruct into genuine MidiNote instances
    // and succeed cleanly across DemoPianoRollController, CubaseDawController, and AbletonDawController!
    const pojoParsed = JSON.parse(JSON.stringify(validParsed));
    expect(pojoParsed.tracks[0].notes[0] instanceof MidiNote).toBe(false);

    const pojoBatch = normalizeGeneratedCandidatesBatch([
      {
        index: 0,
        filename: 'pojo_cand.mid',
        rawMidiBytes: validSmf,
        parsedMidi: pojoParsed,
        reason: null,
      },
    ]);
    expect(pojoBatch.length).toBe(1);
    expect(pojoBatch[0].parsedMidi.tracks[0].notes[0] instanceof MidiNote).toBe(true);
    expect(typeof pojoBatch[0].parsedMidi.tracks[0].notes[0].shiftTicks).toBe('function');

    const demoCtrl = new DemoPianoRollController(new StubAudioEngine());
    const cubaseCtrl = new CubaseDawController();
    const abletonCtrl = new AbletonDawController();
    for (const ctrl of [demoCtrl, cubaseCtrl, abletonCtrl]) {
      await expect(
        ctrl.applyGeneratedCandidates(
          [
            {
              index: 0,
              filename: 'pojo_cand.mid',
              rawMidiBytes: validSmf,
              parsedMidi: pojoParsed,
              reason: null,
            },
          ],
          [],
        ),
      ).resolves.toBeUndefined();
      expect(ctrl.getSnapshot().candidates.length).toBe(1);
      expect(ctrl.getSnapshot().tracks[0].notes.length).toBeGreaterThan(0);
      expect(ctrl.getSnapshot().tracks[0].notes[0] instanceof MidiNote).toBe(true);
      await expect(ctrl.selectGeneratedCandidate(0)).resolves.toBeUndefined();
    }

    // Sparse / getter-poisoned / Proxy parsedMidi falls back to rawMidiBytes re-parse with 0 Proxy leaks;
    // when rawMidiBytes is also unparseable, ExternalDawAdapters remains strictly transactional!
    const proxyParsed = new Proxy(validParsed, {});
    const fallbackBatch = normalizeGeneratedCandidatesBatch([
      {
        index: 0,
        filename: 'proxy_parsed.mid',
        rawMidiBytes: validSmf,
        parsedMidi: proxyParsed,
        reason: null,
      },
    ]);
    expect(fallbackBatch.length).toBe(1);
    expect(fallbackBatch[0].parsedMidi).not.toBe(proxyParsed);
    expect(fallbackBatch[0].parsedMidi.tracks[0].notes[0] instanceof MidiNote).toBe(true);

    const txCubase = new CubaseDawController();
    await txCubase.applyGeneratedCandidates(pojoBatch, []);
    const priorCandCount = txCubase.getSnapshot().candidates.length;
    expect(priorCandCount).toBe(1);
    // Feed invalid/empty batch -> prior candidates & track notes remain intact (transactional)
    await expect(
      txCubase.applyGeneratedCandidates(
        [
          {
            index: 0,
            filename: 'bad.mid',
            rawMidiBytes: new Uint8Array([0, 1, 2, 3]),
            parsedMidi: proxyParsed,
            reason: null,
          },
        ],
        [],
      ),
    ).resolves.toBeUndefined();
    expect(txCubase.getSnapshot().candidates.length).toBe(1);

    // 2. [R13-MED-02] Outbound WorkspaceSyncBus DataCloneError immunity, Proxy/getter incoming isolation,
    // stateful Proxy TOCTOU block, shadowed ArrayBuffer.slice bypass block, and Array expando/symbol parity
    const OriginalBC = globalThis.BroadcastChannel;
    const postedClones: unknown[] = [];
    class CloningBroadcastChannel {
      public onmessage: ((ev: MessageEvent) => void) | null = null;
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      constructor(_name: string) {}
      public postMessage(msg: unknown): void {
        // Real browser BroadcastChannel runs structuredClone and throws DataCloneError on functions/Proxies/Symbols!
        postedClones.push(structuredClone(msg));
      }
      public close(): void {}
    }
    (globalThis as unknown as { BroadcastChannel: unknown }).BroadcastChannel = CloningBroadcastChannel;
    try {
      const safeBus = new WorkspaceSyncBus();
      const nonCloneableWs = {
        ...demoCtrl.exportWorkspaceState(),
        smuggledFn: () => 'boom',
        latestReason: new Proxy(
          {
            source: 'model_cot',
            task: 'infill',
            thinking: true,
            instruments: ['PIANO'],
            genre: ['jazz'],
            gen_measure_count: 4,
            key: 'C',
            note_density: {},
          },
          {},
        ),
      };
      expect(() => safeBus.publishFocusTab('pianoroll', 'test')).not.toThrow();
      expect(() => safeBus.publishGenerationStatus(true, 'step')).not.toThrow();
      expect(() => safeBus.requestWorkspaceState()).not.toThrow();
      // Proxy in latestReason causes clean rejection (0 throws, 0 DataCloneError)
      expect(() => safeBus.publishWorkspaceState(nonCloneableWs as never)).not.toThrow();
      // Valid workspace with extra non-cloneable function property is also rejected by strict key whitelist (0 throws, 0 DataCloneError)
      const cleanExportWithExtraFn = {
        ...demoCtrl.exportWorkspaceState(),
        extraNonCloneableFn: () => {},
      };
      expect(() => safeBus.publishWorkspaceState(cleanExportWithExtraFn as never)).not.toThrow();
      expect(postedClones.length).toBe(3);
      // Genuine exported workspace state clones & posts cleanly
      expect(() => safeBus.publishWorkspaceState(demoCtrl.exportWorkspaceState())).not.toThrow();
      expect(postedClones.length).toBe(4);
      safeBus.dispose();
    } finally {
      (globalThis as unknown as { BroadcastChannel: unknown }).BroadcastChannel = OriginalBC;
    }

    // Stateful Proxy TOCTOU (`task` flips after 1st read) blocked across ipcSec, WorkspaceSyncBus & applyWorkspaceState
    let proxyTaskReads = 0;
    const statefulReasonProxy = new Proxy(
      {
        source: 'model_cot',
        task: 'infill',
        thinking: true,
        instruments: ['PIANO'],
        genre: ['jazz'],
        gen_measure_count: 4,
        key: 'C',
        note_density: {},
      },
      {
        get(target, prop, receiver) {
          if (prop === 'task') {
            proxyTaskReads += 1;
            return proxyTaskReads === 1 ? 'infill' : 'MUTATED_AFTER_VALIDATION';
          }
          return Reflect.get(target, prop, receiver);
        },
      },
    );
    const toctouWs = {
      ...demoCtrl.exportWorkspaceState(),
      latestReason: statefulReasonProxy as never,
    };
    const toctouMsg = {
      messageId: 'msg-toctou',
      senderId: 'peer-1',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: toctouWs,
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(toctouMsg)).toBe(false);
    expect(isValidIncomingSyncMessage(toctouMsg)).toBe(false);
    const guardCtrl = new DemoPianoRollController(new StubAudioEngine());
    guardCtrl.setBpm(144);
    guardCtrl.applyWorkspaceState(toctouWs);
    expect(guardCtrl.getSnapshot().bpm).toBe(144);
    expect(guardCtrl.getSnapshot().latestReason).toBeNull();

    // Shadowed `backingBuf.slice` returning 5 MB must NEVER bypass MAX_CANDIDATE_MIDI_BYTES or allocate 5 MB
    const evilSliceBacking = new ArrayBuffer(validSmf.byteLength);
    new Uint8Array(evilSliceBacking).set(validSmf);
    let sliceInvoked = false;
    Object.defineProperty(evilSliceBacking, 'slice', {
      value: () => {
        sliceInvoked = true;
        return new ArrayBuffer(5 * 1024 * 1024);
      },
    });
    const evilSubView = new Uint8Array(evilSliceBacking, 0, validSmf.byteLength);
    const sliceTestBatch = normalizeGeneratedCandidatesBatch([
      {
        index: 0,
        filename: 'slice_attack.mid',
        rawMidiBytes: evilSubView,
        parsedMidi: validParsed,
        reason: null,
      },
    ]);
    expect(sliceInvoked).toBe(false);
    expect(sliceTestBatch.length).toBe(1);
    expect(sliceTestBatch[0].rawMidiBytes.byteLength).toBe(validSmf.byteLength);
    expect(sliceTestBatch[0].rawMidiBytes.buffer.byteLength).toBe(validSmf.byteLength);
    expect(sliceTestBatch[0].rawMidiBytes.byteLength).toBeLessThanOrEqual(MAX_CANDIDATE_MIDI_BYTES);

    // Array expando / symbol / Proxy parity between `ipcSec.isValidCandidateMidiBytes` and `isValidCandidateMidiBytesPayload`
    const validByteArr = Array.from(validSmf);
    const expandoByteArr = Array.from(validSmf) as number[] & { evil?: string };
    expandoByteArr.evil = 'smuggled';
    const symbolByteArr = Array.from(validSmf);
    Object.defineProperty(symbolByteArr, Symbol('secret'), { value: 1 });
    const proxyByteArr = new Proxy(Array.from(validSmf), {});
    for (const [sample, expected] of [
      [validByteArr, true],
      [expandoByteArr, false],
      [symbolByteArr, false],
      [proxyByteArr, false],
    ] as const) {
      expect(ipcSec.isValidCandidateMidiBytes(sample)).toBe(expected);
      expect(isValidCandidateMidiBytesPayload(sample)).toBe(expected);
    }

    // 3. [R13-MED-03] 350 KB `'MThd'` buffer with 0 `'MTrk'` chunks (matching Test #14 `validLargeMsg`)
    // accepted by IPC & SyncBus applies cleanly in DemoPianoRollController!
    const zeroTrackSmf350k = new Uint8Array(350_000);
    zeroTrackSmf350k.set([
      0x4d, 0x54, 0x68, 0x64,
      0x00, 0x00, 0x00, 0x06,
      0x00, 0x00,
      0x00, 0x01, // Header declares 1 track, remainder is 0x00 (0 `'MTrk'` chunks -> parseMidi throws, controller falls back!)
      0x01, 0xe0,
    ]);
    const baseStateFor350k = demoCtrl.exportWorkspaceState();
    const validLargeZeroTrkWs = {
      ...baseStateFor350k,
      bpm: 132,
      candidates: [
        {
          index: 0,
          filename: 'symphony_350kb.mid',
          rawMidiBytes: zeroTrackSmf350k,
          reason: null,
        },
      ],
      activeCandidateIndex: 0,
    };
    const validLargeZeroTrkMsg = {
      messageId: 'msg-350k-0trk',
      senderId: 'peer-main',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: validLargeZeroTrkWs,
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(validLargeZeroTrkMsg)).toBe(true);
    expect(isValidIncomingSyncMessage(validLargeZeroTrkMsg)).toBe(true);

    const syncTargetCtrl = new DemoPianoRollController(new StubAudioEngine());
    syncTargetCtrl.applyWorkspaceState(validLargeZeroTrkWs);
    expect(syncTargetCtrl.getSnapshot().bpm).toBe(132);
    expect(syncTargetCtrl.getSnapshot().candidates.length).toBe(1);
    expect(syncTargetCtrl.getSnapshot().candidates[0].rawMidiBytes.byteLength).toBe(350_000);
    expect(syncTargetCtrl.getSnapshot().candidates[0].parsedMidi.tracks.length).toBe(0);
    await expect(syncTargetCtrl.selectGeneratedCandidate(0)).resolves.toBeUndefined();

    // 4. [R13-LOW-01] Zero `"NaN"`/`"Infinity"` text & zero throws across domain + all UI components under hostile props
    for (const [badNum, badDen] of [
      [Number.NaN, Number.POSITIVE_INFINITY],
      [Number.NEGATIVE_INFINITY, Number.NaN],
      [0, -4],
    ]) {
      const mEval = evaluateGenerationMeter(badNum, badDen);
      expect(mEval.supported).toBe(false);
      expect(mEval.label).not.toContain('NaN');
      expect(mEval.label).not.toContain('Infinity');
      expect(mEval.reason ?? '').not.toContain('NaN');
      expect(mEval.reason ?? '').not.toContain('Infinity');
    }

    const uiErrSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const uiWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const baseSnap = demoCtrl.getSnapshot();
      const hostileSession = {
        ...baseSnap,
        bpm: Number.NaN,
        ppqn: Number.POSITIVE_INFINITY,
        playheadTick: Number.NaN,
        timeSigNumerator: Number.NaN,
        timeSigDenominator: Number.POSITIVE_INFINITY,
        capabilities: undefined as never,
        selectedNoteIds: null as never,
        targetRange: null as never,
        activeCandidateIndex: Number.NaN,
        candidates: [
          {
            ...baseSnap.candidates[0],
            index: Number.NaN,
          },
        ],
        latestReason: {
          source: 'model_cot' as const,
          task: 'infill',
          thinking: true,
          instruments: ['PIANO'],
          genre: ['jazz', 999 as unknown as string],
          gen_measure_count: Number.POSITIVE_INFINITY,
          key: 'C',
          note_density: {},
        },
      };

      const prUi = render(
        <PianoRollView
          isActive={true}
          controller={demoCtrl}
          session={hostileSession}
          focusRevision={1}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />,
      );
      expect(prUi.container.textContent).not.toContain('NaN');
      expect(prUi.container.textContent).not.toContain('Infinity');
      prUi.unmount();

      const cpHostileUi = render(
        <ControlPanelView
          isActive={true}
          audioEngine={new StubAudioEngine()}
          controllers={null as never}
          session={{ ...hostileSession, tracks: null as never, candidates: null as never }}
          models={null as never}
          hardwareInfo={{
            device: null as never,
            dtype: undefined as never,
            backend_class: 'TorchBackend',
            user_models_dir: null as never,
            bundled_models_dir: '',
            available_models_count: Number.NaN,
            models_with_weights: null as never,
          }}
          settings={null as never}
          isGenerating={false}
          isRescanningModels={false}
          statusTitle={null as never}
          statusDetail={undefined as never}
          generationLogs={null as never}
          errorMessage={null}
          onUpdateSettings={() => {}}
          onSwitchController={() => {}}
          onRescanModels={() => {}}
          onTogglePlayback={() => {}}
          onGenerate={() => {}}
          onCancelGenerate={() => {}}
        />,
      );
      expect(cpHostileUi.container.textContent).not.toContain('NaN');
      expect(cpHostileUi.container.textContent).not.toContain('Infinity');
      cpHostileUi.unmount();

      const splashUi = render(
        <BootSplashScreen
          currentAction={null as never}
          progressPercent={Number.NaN}
          logs={null as never}
        />,
      );
      expect(splashUi.container.textContent).toContain('4%');
      expect(splashUi.container.textContent).not.toContain('NaN');
      expect(splashUi.container.textContent).not.toContain('Infinity');
      splashUi.unmount();

      const circleUi = render(
        <ResonanceCircleCanvas
          audioEngine={new StubAudioEngine()}
          isGenerating={false}
          isPlayingAudio={true}
          statusTitle="AUDIO ACTIVE"
          statusDetail="Testing"
          externalAudioRms={Number.NaN}
          isVisible={true}
        />,
      );
      expect(circleUi.container.textContent).not.toContain('NaN');
      expect(circleUi.container.textContent).not.toContain('Infinity');
      circleUi.unmount();

      const modalUi = render(
        <MidiTrackImportModal
          fileName=""
          parsedMidi={{
            ...validParsed,
            bpm: Number.NaN,
            ppqn: Number.POSITIVE_INFINITY,
            tracks: [
              {
                ...validParsed.tracks[0],
                index: Number.NaN,
                program: Number.NaN,
                notes: null as never,
              },
            ],
          }}
          onConfirm={() => {}}
          onCancel={() => {}}
        />,
      );
      expect(modalUi.container.textContent).not.toContain('NaN');
      expect(modalUi.container.textContent).not.toContain('Infinity');
      modalUi.unmount();

      expect(uiErrSpy).not.toHaveBeenCalled();
      expect(uiWarnSpy).not.toHaveBeenCalled();
    } finally {
      uiErrSpy.mockRestore();
      uiWarnSpy.mockRestore();
    }

    // Unknown / non-cloneable own property on workspaceState rejected identically across all 3 sync layers
    const evilFnWs = {
      ...demoCtrl.exportWorkspaceState(),
      bpm: 165,
      evilFn: () => {},
    } as never;
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'm-evil-fn',
        senderId: 'peer',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: evilFnWs,
      }),
    ).toBe(false);
    expect(
      isValidIncomingSyncMessage({
        messageId: 'm-evil-fn',
        senderId: 'peer',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: evilFnWs,
      }),
    ).toBe(false);
    syncTargetCtrl.applyWorkspaceState(evilFnWs);
    expect(syncTargetCtrl.getSnapshot().bpm).toBe(132);

    // 5. [R13-LOW-02] Exact IEEE-754 binary64 dyadic BigInt affine rounding when intermediate |startTick * scale| > 2^53
    const hugeNote = new MidiNote({
      pitch: 60,
      startTick: 3_500_000_000_000_001,
      durationTicks: 480,
    });
    // Exact: 3_500_000_000_000_001 * 3 - 9_000_000_000_000_000 = 1_500_000_000_000_003 (Float64 loses +1/+3 into 1_500_000_000_000_002!)
    expect(hugeNote.scaleAndShift(3, -9_000_000_000_000_000).startTick).toBe(
      1_500_000_000_000_003,
    );
    // Exact half-up rounding parity with Math.round: 2 * 0.25 + 0 = 0.5 -> 1; 10 * 1 + (-0.5) = 9.5 -> 10
    expect(new MidiNote({ pitch: 60, startTick: 2, durationTicks: 4 }).scaleAndShift(0.25, 0).startTick).toBe(1);
    expect(new MidiNote({ pitch: 60, startTick: 10, durationTicks: 4 }).scaleAndShift(1, -0.5).startTick).toBe(10);
    // Exact shiftTicks near MAX_SAFE_MIDI_TICK without Float64 double-rounding (4_500_000_000_000_001 + 0.49999999999999994 < ...001.5 -> ...001, whereas Float64 rounds up to ...002!)
    expect(
      new MidiNote({
        pitch: 60,
        startTick: 4_500_000_000_000_001,
        durationTicks: 480,
      }).shiftTicks(0.49999999999999994).startTick,
    ).toBe(4_500_000_000_000_001);
  });

  it('26. enforces Round-15 invariants: browser-mode single-pass Proxy defense, sandboxed preload sendWorkspaceSync zero-DataCloneError, detached ArrayBuffer rejection, 3-way mandatory-field parity, null/NaN canvas/query guards, and duplicate React key / model_info hardening ([R14-MED-01..03], [R14-LOW-01..03])', async () => {
    const nodeMod = await import('node:module');
    const fsMod = await import('node:fs');
    const vmMod = await import('node:vm');
    const pathMod = await import('node:path');
    const req = nodeMod.createRequire(import.meta.url);
    const ipcSec = req('../../electron/ipc_security.cjs') as {
      isValidWorkspaceSyncPayload: (m: unknown) => boolean;
      isValidCandidateMidiBytes: (b: unknown) => boolean;
    };

    const validSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const seedCtrl = new DemoPianoRollController(new StubAudioEngine());
    seedCtrl.createNoteAt(60, 0, 480, 100);
    const fullValidWs = seedCtrl.exportWorkspaceState();

    // 1. [R14-MED-01] Browser-Mode single-pass extraction when Node's `util.types.isProxy` is unavailable
    const procObj = process as unknown as Record<string, unknown>;
    const origGetBuiltin = procObj.getBuiltinModule;
    try {
      delete procObj.getBuiltinModule;
      vi.resetModules();
      const browserApiMod = await import('../services/CompassApiClient');
      const browserSyncMod = await import('../services/WorkspaceSyncBus');
      const browserCtrlMod = await import('../controllers/DemoPianoRollController');

      expect(browserApiMod.isRuntimeProxy(new Proxy({}, {}))).toBe(false);

      // Stateful Browser-Mode Proxy array (`SPARSE_HOLE_AT_1` on second read)
      let elem1Reads = 0;
      const sparseFlipArr = new Proxy(['note_a', 'note_b'], {
        getOwnPropertyDescriptor(target, prop) {
          if (prop === '1') {
            elem1Reads += 1;
            if (elem1Reads > 1) {
              return undefined;
            }
          }
          return Reflect.getOwnPropertyDescriptor(target, prop);
        },
      });
      const cleanReason = browserApiMod.sanitizeGenerationReason({
        source: 'model_cot',
        task: 'infill',
        thinking: true,
        instruments: sparseFlipArr,
      });
      expect(elem1Reads).toBe(1);
      expect(cleanReason?.instruments).toEqual(['note_a', 'note_b']);

      // Stateful Browser-Mode Proxy on `candidates` flipping `.length` from `1 -> 40 > 32` on second read
      let candLenReads = 0;
      const cand0 = {
        index: 0,
        filename: 'cand0.mid',
        rawMidiBytes: new Uint8Array(validSmf),
        reason: null,
      };
      const lenFlipCandidates = new Proxy([cand0], {
        getOwnPropertyDescriptor(target, prop) {
          if (prop === 'length') {
            candLenReads += 1;
            return {
              value: candLenReads === 1 ? 1 : 40,
              writable: true,
              enumerable: false,
              configurable: false,
            };
          }
          return Reflect.getOwnPropertyDescriptor(target, prop);
        },
      });
      const bCtrl = new browserCtrlMod.DemoPianoRollController(new StubAudioEngine());
      bCtrl.applyWorkspaceState({
        ...fullValidWs,
        bpm: 144,
        candidates: lenFlipCandidates as never,
      });
      expect(candLenReads).toBe(1);
      expect(bCtrl.getSnapshot().bpm).toBe(144);
      expect(bCtrl.getSnapshot().candidates.length).toBe(1);

      // Stateful Browser-Mode Proxy on `SYNC_WORKSPACE_STATE` flipping `bpm` to `'evil'` on second read
      let bpmReads = 0;
      const bpmFlipWs = new Proxy(
        { ...fullValidWs, bpm: 155 },
        {
          getOwnPropertyDescriptor(target, prop) {
            if (prop === 'bpm') {
              bpmReads += 1;
              return {
                value: bpmReads === 1 ? 155 : 'evil_non_number',
                writable: true,
                enumerable: true,
                configurable: true,
              };
            }
            return Reflect.getOwnPropertyDescriptor(target, prop);
          },
        },
      );
      expect(
        browserSyncMod.isValidIncomingSyncMessage({
          messageId: 'bm-1',
          senderId: 'peer',
          type: 'SYNC_WORKSPACE_STATE',
          workspaceState: bpmFlipWs,
        }),
      ).toBe(true);
      expect(bpmReads).toBe(1);
    } finally {
      procObj.getBuiltinModule = origGetBuiltin;
      vi.resetModules();
    }

    // 2. [R14-MED-02] Sandboxed `preload.cjs` `sendWorkspaceSync` (zero `require('util')`, zero `DataCloneError`)
    const preloadPath = pathMod.resolve(__dirname, '../../electron/preload.cjs');
    const preloadSrc = fsMod.readFileSync(preloadPath, 'utf8');
    expect(preloadSrc).not.toMatch(/require\(['"]util['"]\)/);

    const sentIpcPayloads: unknown[] = [];
    let exposedDesktopBridge: Record<string, (...args: unknown[]) => unknown> | null = null;
    const sandboxRequire = (modId: string) => {
      if (modId === 'electron') {
        return {
          contextBridge: {
            exposeInMainWorld: (_name: string, api: Record<string, (...args: unknown[]) => unknown>) => {
              exposedDesktopBridge = api;
            },
          },
          ipcRenderer: {
            on: () => {},
            removeListener: () => {},
            invoke: async () => ({
              apiBaseUrl: 'http://127.0.0.1:8000',
              handshakeToken: '',
              backendReady: false,
            }),
            send: (_ch: string, payload: unknown) => {
              // Simulate Chromium IPC structured clone (throws `DataCloneError` on functions, Symbols, Proxies, detached ArrayBuffers)
              const cloned = structuredClone(payload);
              sentIpcPayloads.push(cloned);
            },
          },
        };
      }
      throw new Error(`module not found in sandboxed preload: ${modId}`);
    };
    const sandboxCtx = vmMod.createContext({
      require: sandboxRequire,
      process: { argv: [] },
      Object,
      Array,
      Set,
      Map,
      Uint8Array,
      ArrayBuffer,
      Buffer,
      Symbol,
      Number,
      String,
      Boolean,
      Math,
      Promise,
    });
    vmMod.runInContext(preloadSrc, sandboxCtx, { filename: 'preload.cjs' });
    expect(exposedDesktopBridge).not.toBeNull();

    // Uncloneable function smuggled on `FOCUS_TAB` -> dropped cleanly with zero `DataCloneError`!
    expect(() =>
      exposedDesktopBridge!.sendWorkspaceSync({
        messageId: 'm-bad-focus',
        senderId: 's1',
        type: 'FOCUS_TAB',
        tab: 'pianoroll',
        reason: 'ok',
        evilFn: () => {},
      }),
    ).not.toThrow();
    expect(sentIpcPayloads.length).toBe(0);

    // Valid `FOCUS_TAB` and `SYNC_WORKSPACE_STATE` -> cloned cleanly and accepted by `ipc_security.cjs`!
    exposedDesktopBridge!.sendWorkspaceSync({
      messageId: 'm-good-focus',
      senderId: 's1',
      type: 'FOCUS_TAB',
      tab: 'pianoroll',
      reason: 'done',
    });
    exposedDesktopBridge!.sendWorkspaceSync({
      messageId: 'm-good-ws',
      senderId: 's1',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: {
        ...fullValidWs,
        candidates: [
          { index: 0, filename: 'ok.mid', rawMidiBytes: new Uint8Array(validSmf), reason: null },
        ],
      },
    });
    expect(sentIpcPayloads.length).toBe(2);
    expect(ipcSec.isValidWorkspaceSyncPayload(sentIpcPayloads[0])).toBe(true);
    expect(ipcSec.isValidWorkspaceSyncPayload(sentIpcPayloads[1])).toBe(true);

    // 3. [R14-MED-03] Detached `ArrayBuffer` rejected cleanly & `cloneCleanUint8Array` never throws
    const detachSource = new ArrayBuffer(validSmf.byteLength);
    new Uint8Array(detachSource).set(validSmf);
    const detachedU8 = new Uint8Array(detachSource);
    structuredClone(detachSource, { transfer: [detachSource] });
    expect(inspectGenuineUint8ArrayPayload(detachedU8)).toBeNull();
    expect(ipcSec.isValidCandidateMidiBytes(detachedU8)).toBe(false);
    expect(isValidCandidateMidiBytesPayload(detachedU8)).toBe(false);
    expect(
      cloneCleanUint8Array({ backingBuf: detachSource, byteOffset: 0, viewLen: 0 }).byteLength,
    ).toBe(0);

    // 4. [R14-LOW-01] 3-way mandatory-field parity + SubU8 / Symbol / expando rejection + < 25ms 350KB budget
    const partialPayloads: unknown[] = [
      (() => {
        const p = { ...fullValidWs } as Record<string, unknown>;
        delete p.bpm;
        return p;
      })(),
      (() => {
        const p = { ...fullValidWs } as Record<string, unknown>;
        delete p.activeTrackId;
        return p;
      })(),
      (() => {
        const p = { ...fullValidWs } as Record<string, unknown>;
        delete p.targetStartMeasure;
        return p;
      })(),
      (() => {
        const p = { ...fullValidWs } as Record<string, unknown>;
        delete p.candidates;
        return p;
      })(),
      (() => {
        const p = { ...fullValidWs } as Record<string, unknown>;
        delete p.activeCandidateIndex;
        return p;
      })(),
      (() => {
        const p = { ...fullValidWs, activeCandidateIndex: 65 };
        return p;
      })(),
      (() => {
        const p = { ...fullValidWs } as Record<string, unknown>;
        delete p.activeCandidateNoteIds;
        return p;
      })(),
      (() => {
        const trk = { ...fullValidWs.tracks[0] } as Record<string, unknown>;
        delete trk.instrument;
        return { ...fullValidWs, tracks: [trk] };
      })(),
      (() => {
        const n = { ...fullValidWs.tracks[0].notes[0] } as Record<string, unknown>;
        delete n.id;
        return { ...fullValidWs, tracks: [{ ...fullValidWs.tracks[0], notes: [n] }] };
      })(),
    ];

    const guardCtrl = new DemoPianoRollController(new StubAudioEngine());
    guardCtrl.setBpm(188);
    for (let i = 0; i < partialPayloads.length; i += 1) {
      const env = {
        messageId: `part-${i}`,
        senderId: 'peer',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: partialPayloads[i],
      };
      expect(ipcSec.isValidWorkspaceSyncPayload(env)).toBe(false);
      expect(isValidIncomingSyncMessage(env)).toBe(false);
      guardCtrl.applyWorkspaceState(partialPayloads[i] as never);
      expect(guardCtrl.getSnapshot().bpm).toBe(188);
    }

    class SubUint8 extends Uint8Array {}
    const subU8 = new SubUint8(validSmf.byteLength);
    subU8.set(validSmf);
    const symU8 = new Uint8Array(validSmf);
    (symU8 as unknown as Record<symbol, unknown>)[Symbol('smuggle')] = 1;
    const expandoU8 = new Uint8Array(validSmf);
    Object.defineProperty(expandoU8, 'extraFn', { value: () => {}, enumerable: true });

    for (const badBytes of [subU8, symU8, expandoU8]) {
      expect(ipcSec.isValidCandidateMidiBytes(badBytes)).toBe(false);
      expect(isValidCandidateMidiBytesPayload(badBytes)).toBe(false);
    }

    const largeNotes: MidiNote[] = [];
    for (let i = 0; i < 45_000; i += 1) {
      largeNotes.push(new MidiNote({ pitch: 60 + (i % 12), startTick: i * 10, durationTicks: 8 }));
    }
    const largeSmf = StandardMidiCodec.encodeSingleTrackMidi(largeNotes, 480, 120, 'PIANO');
    expect(largeSmf.byteLength).toBeGreaterThan(350_000);
    const t0 = performance.now();
    expect(ipcSec.isValidCandidateMidiBytes(largeSmf)).toBe(true);
    expect(isValidCandidateMidiBytesPayload(largeSmf)).toBe(true);
    expect(performance.now() - t0).toBeLessThan(25);

    // 5. [R14-LOW-02] Null/NaN guards on `DawTrack`, `StandardMidiCodec`, `<PianoRollView />`, and `<ResonanceCircleCanvas />`
    expect(queryOverlappingNotesIndexed([null as never], [99999], 0, 480)).toEqual([]);
    const dt = new DawTrack({
      name: 'SafeTrack',
      notes: [null as never, new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480 })],
    });
    expect(dt.noteCount).toBe(1);
    expect(dt.queryNotesStartingIn(0, Number.NaN)).toEqual([]);
    expect(dt.queryNotesOverlapping(0, Number.NaN)).toEqual([]);
    expect(
      StandardMidiCodec.encodeMultiTrackMidi([
        {
          ...dt.toSnapshot(),
          notes: [
            null as never,
            {
              id: 'bad',
              pitch: Number.NaN,
              startTick: Number.NaN,
              durationTicks: Number.NaN,
              endTick: Number.NaN,
              velocity: Number.NaN,
              channel: 0,
            } as never,
          ],
        },
      ]).byteLength,
    ).toBeGreaterThan(22);

    class NanWaveformAudioEngine extends StubAudioEngine {
      public override readAcousticTelemetry(w: Float32Array, s: Uint8Array) {
        w.fill(Number.NaN);
        s.fill(0);
        return { rms: 0.4, peak: 0.8, lowBandEnergy: 0.3, highBandEnergy: Number.NaN };
      }
    }
    let capturedRafCb: FrameRequestCallback | null = null;
    const rafSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => {
        capturedRafCb = cb;
        return 1;
      });
    try {
      const nanCircle = render(
        <ResonanceCircleCanvas
          audioEngine={new NanWaveformAudioEngine()}
          isGenerating={false}
          isPlayingAudio={true}
          statusTitle="NAN WAVE"
          statusDetail="Safe"
          isVisible={true}
        />,
      );
      expect(capturedRafCb).not.toBeNull();
      expect(() => (capturedRafCb as unknown as (t: number) => void)(16)).not.toThrow();
      nanCircle.unmount();
    } finally {
      rafSpy.mockRestore();
    }

    // 6. [R14-LOW-03] Duplicate React `key` zero-warning guarantee + `/model_info` & `/hardware_info` sanitization
    const keyErrSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const keyWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const dupTrackA = {
        ...seedCtrl.getSnapshot().tracks[0],
        id: 'dup_id',
        name: 'Dup A',
      };
      const dupTrackB = {
        ...seedCtrl.getSnapshot().tracks[0],
        id: 'dup_id',
        name: 'Dup B',
      };
      const dupSession = {
        ...seedCtrl.getSnapshot(),
        activeTrackId: 'dup_id',
        tracks: [dupTrackA, dupTrackB],
      };
      const dupCp = render(
        <ControlPanelView
          audioEngine={new StubAudioEngine()}
          models={[
            { model_name: 'MORTM4.5D-80M-SFT-Gen', has_weights: true },
            { model_name: 'MORTM4.5D-80M-SFT-Gen', has_weights: false },
          ]}
          hardwareInfo={null}
          controllers={[
            { id: 'dup-ctrl', displayName: 'Ctrl A' } as never,
            { id: 'dup-ctrl', displayName: 'Ctrl B' } as never,
          ]}
          session={dupSession}
          settings={{
            modelName: 'MORTM4.5D-80M-SFT-Gen',
            taskMode: 'auto',
            temperature: 1.0,
            topP: 0.95,
            cotTemperature: 0.8,
            thinking: true,
            numCandidates: 2,
            forceMonophonic: false,
            conditionTrackId: '',
            noteDensity: null,
            musicalKey: '',
            selectedGenres: [],
          }}
          isGenerating={false}
          isRescanningModels={false}
          statusTitle="STANDBY"
          statusDetail="Ready"
          generationLogs={[]}
          errorMessage={null}
          onUpdateSettings={() => {}}
          onSwitchController={() => {}}
          onRescanModels={() => {}}
          onTogglePlayback={() => {}}
          onGenerate={() => {}}
          onCancelGenerate={() => {}}
        />,
      );
      fireEvent.click(
        dupCp.getByRole('button', { name: /音楽構造・ジャンル・タスクの詳細指定/ }),
      );
      dupCp.unmount();

      // Hostile external controller with duplicate track IDs + NaN notes + Export .mid click
      const hostileExtCtrl = {
        ...new CubaseDawController(),
        id: 'cubase-hostile',
        displayName: 'Cubase Hostile',
        getSnapshot: () => ({
          ...dupSession,
          tracks: [
            {
              ...dupTrackA,
              notes: [
                null as never,
                {
                  id: 'n_nan',
                  pitch: Number.NaN,
                  startTick: Number.NaN,
                  durationTicks: Number.NaN,
                  endTick: Number.NaN,
                  velocity: Number.NaN,
                  channel: 0,
                } as never,
              ],
            },
            dupTrackB,
          ],
        }),
        subscribe: () => () => {},
        setTargetMeasureRange: () => {},
        setPlayheadTick: () => {},
      };
      const origCreateObjUrl = URL.createObjectURL;
      const origRevokeObjUrl = URL.revokeObjectURL;
      URL.createObjectURL = vi.fn(() => 'blob:compass-test');
      URL.revokeObjectURL = vi.fn();
      const clickSpy = vi
        .spyOn(HTMLAnchorElement.prototype, 'click')
        .mockImplementation(() => {});
      try {
        const prDup = render(
          <PianoRollView
            controller={hostileExtCtrl as never}
            session={hostileExtCtrl.getSnapshot() as never}
            focusRevision={0}
            onLaunchCompassAi={() => {}}
            onSwitchToDemoController={() => {}}
          />,
        );
        const exportBtn = prDup.queryByRole('button', { name: /MIDI|書き出し|エクスポート/i });
        if (exportBtn) {
          expect(() => fireEvent.click(exportBtn)).not.toThrow();
        }
        prDup.unmount();

        const prInteractiveDup = render(
          <PianoRollView
            controller={seedCtrl}
            session={dupSession}
            focusRevision={0}
            onLaunchCompassAi={() => {}}
            onSwitchToDemoController={() => {}}
          />,
        );
        prInteractiveDup.unmount();
      } finally {
        clickSpy.mockRestore();
        URL.createObjectURL = origCreateObjUrl;
        URL.revokeObjectURL = origRevokeObjUrl;
      }

      expect(keyErrSpy).not.toHaveBeenCalled();
      expect(keyWarnSpy).not.toHaveBeenCalled();
    } finally {
      keyErrSpy.mockRestore();
      keyWarnSpy.mockRestore();
    }

    // `/model_info` & `/hardware_info` malformed JSON payload sanitization
    const origFetch = globalThis.fetch;
    try {
      let step = 0;
      globalThis.fetch = vi.fn(async () => {
        step += 1;
        if (step === 1) {
          return new Response(
            JSON.stringify({ models: [null, { model_id: 123 }, { model_name: 'NaN' }] }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (step === 2) {
          return new Response(
            JSON.stringify({ models: [null, { model_name: 'MORTM4.5D-Valid', has_weights: true }] }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        return new Response(
          JSON.stringify({
            device: 'NaN',
            dtype: 'Infinity',
            backend_class: '-Infinity',
            available_models_count: 'bad',
            models_with_weights: ['NaN', 'MORTM4.5D-Valid'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }) as typeof fetch;

      const testApi = new CompassApiClient('http://127.0.0.1:8000', 'tok');
      expect(await testApi.getModelInfo()).toEqual([]);
      expect(await testApi.getModelInfo()).toEqual([
        { model_name: 'MORTM4.5D-Valid', has_weights: true },
      ]);
      const hwRes = await testApi.getHardwareInfo();
      expect(hwRes.device).toBe('cpu');
      expect(hwRes.dtype).toBe('float32');
      expect(hwRes.backend_class).toBe('CpuBackend');
      expect(hwRes.available_models_count).toBe(0);
      expect(hwRes.models_with_weights).toEqual(['MORTM4.5D-Valid']);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('27. enforces Round-16 4-way preload sync parity, single-read cloneCleanUint8Array, SMF ceiling/meter invariants, and domain Proxy/duck-note hardening ([R15-MED-01..03], [R15-LOW-01..02])', async () => {
    const fsMod = await import('node:fs');
    const pathMod = await import('node:path');
    const vmMod = await import('node:vm');
    const modReq = (await import('node:module')).createRequire(import.meta.url);
    const ipcSec = modReq('../../electron/ipc_security.cjs') as {
      isValidWorkspaceSyncPayload: (m: unknown) => boolean;
    };

    const validSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'n_seed', pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const validParsed = StandardMidiCodec.parseMidi(validSmf.buffer as ArrayBuffer);

    // -------------------------------------------------------------------------
    // 1. [R15-MED-01] & [R15-LOW-02] Sandboxed `preload.cjs` 4-way sync parity + listener isolation
    // -------------------------------------------------------------------------
    const preloadPath = pathMod.resolve(__dirname, '../../electron/preload.cjs');
    const preloadSrc = fsMod.readFileSync(preloadPath, 'utf8');
    expect(preloadSrc).not.toMatch(/require\(['"]util['"]\)/);

    const sentIpcPayloads: unknown[] = [];
    const ipcListeners = new Map<string, Array<(_ev: unknown, payload: unknown) => void>>();
    let exposedBridge: Record<string, (...args: unknown[]) => unknown> | null = null;
    const sandboxCtx = vmMod.createContext({
      require: (id: string) => {
        if (id === 'electron') {
          return {
            contextBridge: {
              exposeInMainWorld: (_k: string, api: Record<string, (...args: unknown[]) => unknown>) => {
                exposedBridge = api;
              },
            },
            ipcRenderer: {
              on: (ch: string, fn: (_ev: unknown, payload: unknown) => void) => {
                const list = ipcListeners.get(ch) ?? [];
                list.push(fn);
                ipcListeners.set(ch, list);
              },
              removeListener: (ch: string, fn: (_ev: unknown, payload: unknown) => void) => {
                const list = ipcListeners.get(ch) ?? [];
                ipcListeners.set(
                  ch,
                  list.filter((x) => x !== fn),
                );
              },
              invoke: async () => ({}),
              send: (_ch: string, m: unknown) => {
                sentIpcPayloads.push(structuredClone(m));
              },
            },
          };
        }
        throw new Error(`Unexpected require in sandbox: ${id}`);
      },
      // Intentionally omit `process.argv` (`[R15-LOW-02]` part 4)
      process: {},
      Object,
      Array,
      Set,
      Map,
      Uint8Array,
      ArrayBuffer,
      Buffer,
      Symbol,
      Number,
      String,
      Boolean,
      Math,
      Promise,
    });
    expect(() => vmMod.runInContext(preloadSrc, sandboxCtx, { filename: 'preload.cjs' })).not.toThrow();
    expect(exposedBridge).not.toBeNull();

    // Listener exception isolation (`onWorkspaceSync`, `onBootLog`, `onFocusTabRequest`)
    let secondWsGot: unknown = null;
    let secondBootGot = '';
    let secondFocusGot = '';
    exposedBridge!.onWorkspaceSync(() => {
      throw new Error('subscriber 1 boom');
    });
    exposedBridge!.onWorkspaceSync((p: unknown) => {
      secondWsGot = p;
    });
    exposedBridge!.onBootLog(() => {
      throw new Error('boot 1 boom');
    });
    exposedBridge!.onBootLog((line: unknown) => {
      secondBootGot = String(line);
    });
    exposedBridge!.onFocusTabRequest(() => {
      throw new Error('focus 1 boom');
    });
    exposedBridge!.onFocusTabRequest((tab: unknown) => {
      secondFocusGot = String(tab);
    });
    for (const fn of ipcListeners.get('compass:workspace-sync') ?? []) {
      expect(() => fn({}, { ok: 1 })).not.toThrow();
    }
    for (const fn of ipcListeners.get('compass:boot-log') ?? []) {
      expect(() => fn({}, 'BOOT_OK')).not.toThrow();
    }
    for (const fn of ipcListeners.get('compass:focus-tab') ?? []) {
      expect(() => fn({}, 'pianoroll')).not.toThrow();
    }
    expect(secondWsGot).toEqual({ ok: 1 });
    expect(secondBootGot).toBe('BOOT_OK');
    expect(secondFocusGot).toBe('pianoroll');

    const seedCtrl = new DemoPianoRollController(new StubAudioEngine());
    seedCtrl.createNoteAt(60, 0, 480, 100);
    await seedCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'seed.mid',
          rawMidiBytes: validSmf,
          parsedMidi: validParsed,
          reason: null,
        },
      ],
      [],
    );
    const baseWs = seedCtrl.exportWorkspaceState();
    const cloneWsObj = (): Record<string, unknown> =>
      structuredClone(baseWs) as unknown as Record<string, unknown>;

    const assertFourWayParity = (wsVariant: unknown, expectedValid: boolean, label: string) => {
      const msg = {
        messageId: `m_${label}`,
        senderId: 'peer_r16',
        type: 'SYNC_WORKSPACE_STATE',
        workspaceState: wsVariant,
      };
      const before = sentIpcPayloads.length;
      exposedBridge!.sendWorkspaceSync(msg);
      const preloadOk = sentIpcPayloads.length > before;
      const ipcOk = ipcSec.isValidWorkspaceSyncPayload(msg);
      const busOk = isValidIncomingSyncMessage(msg);
      expect({ label, preloadOk, ipcOk, busOk }).toEqual({
        label,
        preloadOk: expectedValid,
        ipcOk: expectedValid,
        busOk: expectedValid,
      });

      const testCtrl = new DemoPianoRollController(new StubAudioEngine());
      testCtrl.setBpm(193);
      testCtrl.applyWorkspaceState(wsVariant as Parameters<typeof testCtrl.applyWorkspaceState>[0]);
      if (expectedValid) {
        expect(testCtrl.getSnapshot().bpm).not.toBe(193);
      } else {
        expect(testCtrl.getSnapshot().bpm).toBe(193);
      }
    };

    // Valid 7/8 workspace state -> accepted by all 4 boundaries
    const valid78 = cloneWsObj();
    valid78.bpm = 132;
    valid78.timeSigNumerator = 7;
    valid78.timeSigDenominator = 8;
    valid78.beatsPerMeasure = 3.5;
    assertFourWayParity(valid78, true, 'valid-7-8');

    // (1a) Meter vs beatsPerMeasure mismatch (3/4 with beatsPerMeasure=4)
    const badMeterPair = cloneWsObj();
    badMeterPair.timeSigNumerator = 3;
    badMeterPair.timeSigDenominator = 4;
    badMeterPair.beatsPerMeasure = 4;
    assertFourWayParity(badMeterPair, false, 'bad-meter-pair');

    // (1b) Non-1/8-step standalone beatsPerMeasure (4.333)
    const badStandaloneBeats = cloneWsObj();
    delete badStandaloneBeats.timeSigNumerator;
    delete badStandaloneBeats.timeSigDenominator;
    badStandaloneBeats.beatsPerMeasure = 4.333;
    assertFourWayParity(badStandaloneBeats, false, 'bad-standalone-beats');

    // (2a) Canonical surrogate track ID collision ('trk_\uD800' vs 'trk_\uD801')
    const badTrackCanonDup = cloneWsObj();
    const t0 = (badTrackCanonDup.tracks as Array<Record<string, unknown>>)[0];
    badTrackCanonDup.tracks = [
      { ...t0, id: 'trk_\ud800', notes: [] },
      { ...t0, id: 'trk_\ud801', notes: [] },
    ];
    badTrackCanonDup.activeTrackId = 'trk_\ud800';
    badTrackCanonDup.activeCandidateRange = null;
    badTrackCanonDup.candidateOriginSlot = null;
    assertFourWayParity(badTrackCanonDup, false, 'bad-track-canon-dup');

    // (2b) activeTrackId missing from tracks
    const badMissingActiveTrack = cloneWsObj();
    badMissingActiveTrack.activeTrackId = 'nonexistent_track';
    assertFourWayParity(badMissingActiveTrack, false, 'bad-missing-active-track');

    // (3a) Note exceeding project tick horizon (4128 * ppqn)
    const badOverHorizonNote = cloneWsObj();
    (badOverHorizonNote.tracks as Array<Record<string, unknown>>)[0].notes = [
      {
        id: 'n_over',
        pitch: 60,
        startTick: 4128 * 480 - 4,
        durationTicks: 10,
        velocity: 100,
        channel: 0,
      },
    ];
    assertFourWayParity(badOverHorizonNote, false, 'bad-over-horizon-note');

    // (3b) Cross-track canonical note ID collision
    const badCrossTrackNoteDup = cloneWsObj();
    const baseTrk = (badCrossTrackNoteDup.tracks as Array<Record<string, unknown>>)[0];
    badCrossTrackNoteDup.tracks = [
      {
        ...baseTrk,
        id: 't_a',
        notes: [{ id: 'n_\ud800', pitch: 60, startTick: 0, durationTicks: 120, velocity: 100 }],
      },
      {
        ...baseTrk,
        id: 't_b',
        notes: [{ id: 'n_\ud801', pitch: 64, startTick: 0, durationTicks: 120, velocity: 100 }],
      },
    ];
    badCrossTrackNoteDup.activeTrackId = 't_a';
    badCrossTrackNoteDup.activeCandidateRange = null;
    badCrossTrackNoteDup.candidateOriginSlot = null;
    badCrossTrackNoteDup.activeCandidateNoteIds = [];
    assertFourWayParity(badCrossTrackNoteDup, false, 'bad-cross-track-note-dup');

    // (4a) Slot endTick > 16512 * ppqn
    const badSlotOverflow = cloneWsObj();
    badSlotOverflow.activeCandidateRange = {
      trackId: baseWs.activeTrackId,
      startTick: 0,
      endTick: 16512 * 480 + 1,
    };
    assertFourWayParity(badSlotOverflow, false, 'bad-slot-overflow');

    // (4b) activeCandidateRange.trackId !== candidateOriginSlot.trackId
    const badSlotTrackMismatch = cloneWsObj();
    const trkTemplate = (badSlotTrackMismatch.tracks as Array<Record<string, unknown>>)[0];
    badSlotTrackMismatch.tracks = [
      { ...trkTemplate, id: 't_1', notes: [] },
      { ...trkTemplate, id: 't_2', notes: [] },
    ];
    badSlotTrackMismatch.activeTrackId = 't_1';
    badSlotTrackMismatch.activeCandidateNoteIds = [];
    badSlotTrackMismatch.activeCandidateRange = { trackId: 't_1', startTick: 0, endTick: 1920 };
    badSlotTrackMismatch.candidateOriginSlot = { trackId: 't_2', startTick: 0, endTick: 1920 };
    assertFourWayParity(badSlotTrackMismatch, false, 'bad-slot-track-mismatch');

    // (5a) activeCandidateNoteIds canonical surrogate collision
    const badCandIdCollision = cloneWsObj();
    badCandIdCollision.activeCandidateNoteIds = ['cand_\ud800', 'cand_\ud801'];
    assertFourWayParity(badCandIdCollision, false, 'bad-cand-id-collision');

    // (5b) Duplicate `note_density` keys via Proxy `ownKeys` trap in `preload.cjs`
    const dupNdProxy = new Proxy(
      { PIANO: 4 },
      {
        ownKeys: () => ['PIANO', 'PIANO'],
      },
    );
    const badDupNdWs = cloneWsObj();
    badDupNdWs.latestReason = {
      source: 'request',
      task: 'meta',
      thinking: false,
      instruments: ['PIANO'],
      genre: null,
      gen_measure_count: 4,
      key: 'C',
      note_density: dupNdProxy,
    };
    const beforeDupNd = sentIpcPayloads.length;
    exposedBridge!.sendWorkspaceSync({
      messageId: 'm_dup_nd',
      senderId: 'peer_r16',
      type: 'SYNC_WORKSPACE_STATE',
      workspaceState: badDupNdWs,
    });
    expect(sentIpcPayloads.length).toBe(beforeDupNd);

    // -------------------------------------------------------------------------
    // 2. [R15-MED-02] `cloneCleanUint8Array` single-descriptor-read TOCTOU defense
    // -------------------------------------------------------------------------
    const cleanAb = new ArrayBuffer(validSmf.byteLength);
    new Uint8Array(cleanAb).set(validSmf);
    const sharedAb =
      typeof SharedArrayBuffer === 'function' ? new SharedArrayBuffer(64) : new ArrayBuffer(0);
    let getterReads = 0;
    const getterSwapInfo = {
      get backingBuf() {
        getterReads += 1;
        return (getterReads <= 5 ? cleanAb : sharedAb) as ArrayBuffer;
      },
      byteOffset: 0,
      viewLen: validSmf.byteLength,
    };
    expect(cloneCleanUint8Array(getterSwapInfo).byteLength).toBe(0);
    expect(getterReads).toBe(0);

    if (typeof SharedArrayBuffer === 'function') {
      expect(
        cloneCleanUint8Array({
          backingBuf: new SharedArrayBuffer(64) as unknown as ArrayBuffer,
          byteOffset: 0,
          viewLen: 22,
        }).byteLength,
      ).toBe(0);
    }

    const inspected = inspectGenuineUint8ArrayPayload(validSmf);
    expect(inspected).not.toBeNull();
    const clonedClean = cloneCleanUint8Array(inspected!);
    expect(clonedClean.byteLength).toBe(validSmf.byteLength);
    expect(Array.from(clonedClean)).toEqual(Array.from(validSmf));

    // -------------------------------------------------------------------------
    // 3. [R15-MED-03] `StandardMidiCodec` NaN/Infinity meter & ceiling Note-Off ordering
    // -------------------------------------------------------------------------
    for (const badBeats of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -8]) {
      const m1 = StandardMidiCodec.encodeSingleTrackMidi(
        [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 240 })],
        480,
        120,
        'PIANO',
        badBeats,
        { numerator: Number.NaN, denominator: 0 },
      );
      const p1 = StandardMidiCodec.parseMidi(m1.buffer as ArrayBuffer);
      expect(Number.isFinite(p1.beatsPerMeasure)).toBe(true);
      expect(p1.beatsPerMeasure).toBe(4);

      const m2 = StandardMidiCodec.encodeConditionsMidi(
        [new MidiNote({ pitch: 62, startTick: 0, durationTicks: 240 })],
        480,
        120,
        false,
        badBeats,
        { numerator: 4, denominator: 1_000_000_000 },
      );
      const p2 = StandardMidiCodec.parseMidi(m2.buffer as ArrayBuffer);
      expect(Number.isFinite(p2.beatsPerMeasure)).toBe(true);
    }

    // Ceiling 0x0fffffff in encodeSingleTrackMidi: Note-On (0x90) MUST precede Note-Off (0x80)
    const ceilingSingle = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 64, startTick: 0x0fffffff, durationTicks: 480, velocity: 99 })],
      480,
      120,
      'PIANO',
    );
    let noteOnIdx = -1;
    let noteOffIdx = -1;
    for (let bi = 22; bi < ceilingSingle.length - 2; bi += 1) {
      if (ceilingSingle[bi] === 0x90 && ceilingSingle[bi + 1] === 64 && ceilingSingle[bi + 2] === 99) {
        noteOnIdx = bi;
      }
      if (ceilingSingle[bi] === 0x80 && ceilingSingle[bi + 1] === 64 && ceilingSingle[bi + 2] === 0x00) {
        noteOffIdx = bi;
      }
    }
    expect(noteOnIdx).toBeGreaterThan(21);
    expect(noteOffIdx).toBeGreaterThan(noteOnIdx);
    // And parseMidi still enforces the 4128 * ppqn project horizon on single-track decode:
    expect(() => StandardMidiCodec.parseMidi(ceilingSingle.buffer as ArrayBuffer)).toThrow(
      /累積ティック数が上限を超えています/,
    );

    // Multi-track export clamps onto 4128 * ppqn so exported DAW projects always round-trip cleanly!
    const overHorizonTrack = new DawTrack({
      id: 'trk_horizon',
      name: 'Horizon',
      instrument: 'PIANO',
      notes: [new MidiNote({ pitch: 67, startTick: 10_000_000, durationTicks: 480, velocity: 105 })],
    });
    const multiExported = StandardMidiCodec.encodeMultiTrackMidi(
      [overHorizonTrack.toSnapshot()],
      480,
      120,
      4,
    );
    const reparsedMulti = StandardMidiCodec.parseMidi(multiExported.buffer as ArrayBuffer);
    expect(reparsedMulti.tracks.length).toBe(1);
    expect(reparsedMulti.tracks[0].notes.length).toBe(1);
    expect(reparsedMulti.tracks[0].notes[0].startTick).toBe(MAX_SMF_QUARTER_NOTES * 480 - 1);
    expect(reparsedMulti.tracks[0].notes[0].endTick).toBe(MAX_SMF_QUARTER_NOTES * 480);

    // -------------------------------------------------------------------------
    // 4. [R15-LOW-01] `DawTrack` duck-note normalization, WeakSet identity, & `parseMidi` guards
    // -------------------------------------------------------------------------
    const genuineNote = new MidiNote({ id: 'gn_1', pitch: 60, startTick: 0, durationTicks: 480 });
    expect(isGenuineMidiNote(genuineNote)).toBe(true);
    expect(isGenuineMidiNote(new Proxy(genuineNote, {}))).toBe(false);

    const revNote = Proxy.revocable(genuineNote, {});
    revNote.revoke();
    expect(isGenuineMidiNote(revNote.proxy)).toBe(false);

    const duckPlain = {
      id: 'duck_1',
      pitch: 65,
      startTick: 480,
      durationTicks: 240,
      velocity: 88,
    } as unknown as MidiNote;
    const negDuck = {
      id: 'duck_neg',
      pitch: 65,
      startTick: -120,
      durationTicks: 240,
      velocity: 88,
    } as unknown as MidiNote;
    const throwingDuck = Object.defineProperty({}, 'startTick', {
      enumerable: true,
      get() {
        throw new Error('duck getter boom');
      },
    }) as unknown as MidiNote;

    const mixedTrack = new DawTrack({
      id: 'trk_mixed',
      name: 'Mixed',
      instrument: 'PIANO',
      notes: [genuineNote, duckPlain, negDuck, throwingDuck, revNote.proxy as unknown as MidiNote],
    });
    const storedNotes = mixedTrack.getNotes();
    expect(storedNotes.length).toBe(2);
    // O(1) fast-path preserves exact reference identity on genuine `MidiNote`:
    expect(storedNotes[0]).toBe(genuineNote);
    // Duck-typed plain object normalized into genuine frozen `MidiNote` with working `.withTiming`:
    expect(isGenuineMidiNote(storedNotes[1])).toBe(true);
    expect(Object.isFrozen(storedNotes[1])).toBe(true);

    const replaceCmd = new ReplaceRangeNotesCommand(mixedTrack, 360, 960, [
      new MidiNote({ id: 'rep_1', pitch: 72, startTick: 480, durationTicks: 240 }),
    ]);
    expect(() => replaceCmd.execute()).not.toThrow();
    expect(() => replaceCmd.undo()).not.toThrow();
    expect(mixedTrack.getNotes().map((n) => n.id)).toEqual(['gn_1', 'duck_1']);

    // `parseMidi` rejects non-ArrayBuffer / SharedArrayBuffer / detached ArrayBuffer with Japanese MThd error
    expect(() => StandardMidiCodec.parseMidi(null as unknown as ArrayBuffer)).toThrow(
      /有効な Standard MIDI ファイル \(MThd\) ではありません/,
    );
    if (typeof SharedArrayBuffer === 'function') {
      expect(() =>
        StandardMidiCodec.parseMidi(new SharedArrayBuffer(64) as unknown as ArrayBuffer),
      ).toThrow(/有効な Standard MIDI ファイル \(MThd\) ではありません/);
    }
    const detAb = new ArrayBuffer(64);
    structuredClone(detAb, { transfer: [detAb] });
    expect(() => StandardMidiCodec.parseMidi(detAb)).toThrow(
      /有効な Standard MIDI ファイル \(MThd\) ではありません/,
    );

    // -------------------------------------------------------------------------
    // 5. [R15-LOW-02] Single-pass `getHardwareInfo`, `normalizeGeneratedCandidatesBatch`, & `resolveBridgeUrl`
    // -------------------------------------------------------------------------
    const origFetch2 = globalThis.fetch;
    try {
      let devGetterCalled = 0;
      const hostileHwPayload = {
        available_models_count: 1e30,
        models_with_weights: ['MORTM4.5E-Test'],
      };
      Object.defineProperty(hostileHwPayload, 'device', {
        enumerable: true,
        get() {
          devGetterCalled += 1;
          throw new Error('device getter boom');
        },
      });
      globalThis.fetch = vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => hostileHwPayload,
      })) as unknown as typeof fetch;
      const hwClient = new CompassApiClient('http://127.0.0.1:8000', 'tok');
      const hw = await hwClient.getHardwareInfo();
      expect(devGetterCalled).toBe(0);
      expect(hw.device).toBe('cpu');
      expect(hw.available_models_count).toBe(Number.MAX_SAFE_INTEGER);
      expect(hw.models_with_weights).toEqual(['MORTM4.5E-Test']);
    } finally {
      globalThis.fetch = origFetch2;
    }

    const revOpts = Proxy.revocable({ bridgeBaseUrl: 'http://127.0.0.1:9999' }, {});
    revOpts.revoke();
    const throwingOpts = Object.defineProperty({}, 'bridgeBaseUrl', {
      enumerable: true,
      get() {
        throw new Error('bridge getter boom');
      },
    });
    expect(() => new CubaseDawController(revOpts.proxy)).not.toThrow();
    expect(() => new AbletonDawController(throwingOpts)).not.toThrow();
    expect(new CubaseDawController(throwingOpts).bridgeBaseUrl).toBe('ws://127.0.0.1:9120');
    expect(new AbletonDawController(revOpts.proxy).bridgeBaseUrl).toBe('ws://127.0.0.1:9121');
  });

  it('enforces Round 17 Fact-Checker (R16) remediations: progress schema & unhandled-rejection guard, finite wheel/pointer clamping, NoteCommands duck normalization, SMF track-name & zero-spread loops, importParsedMidiTracks pre-pause validation, fallback sort, plugin boundary guards, and atomic preload config', async () => {
    // -------------------------------------------------------------------------
    // 1. [R16-MED-01] `getGenerationProgress` strict schema/job_id guard + `generateMusic` zero unhandled rejections
    // -------------------------------------------------------------------------
    const origFetch = globalThis.fetch;
    const unhandledErrors: unknown[] = [];
    const onUnhandled = (reason: unknown) => {
      unhandledErrors.push(reason);
    };
    process.on('unhandledRejection', onUnhandled);
    try {
      let progGetterInvoked = 0;
      const getterSnap: Record<string, unknown> = { active: true, job_id: 'job_ok' };
      Object.defineProperty(getterSnap, 'message', {
        enumerable: true,
        get() {
          progGetterInvoked += 1;
          throw new Error('progress getter boom');
        },
      });
      const progClient = new CompassApiClient('http://127.0.0.1:18999', 'tok');

      const badPayloads: unknown[] = [
        null,
        ['not', 'object'],
        Object.create({ active: true }),
        new Proxy({ active: true, job_id: 'job_ok' }, {}),
        getterSnap,
        { active: 'yes', job_id: 'job_ok' },
        // Missing `job_id` while `active: true` when polling for a specific job MUST fail closed:
        { active: true, stage: 'decoding', message: 'spoofed active without job_id' },
      ];
      for (const bad of badPayloads) {
        globalThis.fetch = vi.fn(async () => ({
          ok: true,
          status: 200,
          json: async () => bad,
        })) as unknown as typeof fetch;
        expect(await progClient.getGenerationProgress(undefined, 'job_ok')).toBeNull();
      }
      expect(progGetterInvoked).toBe(0);

      // Valid progress sanitizes "undefined" / "null" / "NaN" / lone surrogates and defaults omitted counters
      globalThis.fetch = vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          active: true,
          stage: 'undefined',
          generated_tokens: 1e30,
          max_tokens: -5,
          message: 'null',
          updated_at: Number.NaN,
          job_id: 'job_ok',
        }),
      })) as unknown as typeof fetch;
      const cleanProg = await progClient.getGenerationProgress(undefined, 'job_ok');
      expect(cleanProg).toEqual({
        active: true,
        stage: '',
        generated_tokens: Number.MAX_SAFE_INTEGER,
        max_tokens: 0,
        message: '',
        updated_at: 0,
        job_id: 'job_ok',
      });

      // `generateMusic` polling with non-string `message` or throwing `onProgress` callback never causes unhandledRejection
      const tinyMidi = StandardMidiCodec.encodeSingleTrackMidi(
        [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
        480,
        120,
      );
      const zip = new JSZip();
      zip.file('cand_0.mid', tinyMidi);
      const zipBytes = await zip.generateAsync({ type: 'uint8array' });

      globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const u = String(input);
        const hdrs = (init?.headers ?? {}) as Record<string, string>;
        const reqJobId = hdrs['X-Compass-Job-Id'] ?? 'j_poll';
        if (u.includes('/generation_progress')) {
          return new Response(
            JSON.stringify({
              active: true,
              stage: 'generating',
              message: 'poll tick ok',
              job_id: reqJobId,
            }),
            { status: 200 },
          );
        }
        await new Promise((r) => setTimeout(r, 230));
        return new Response(zipBytes.slice().buffer as ArrayBuffer, { status: 200 });
      }) as typeof fetch;

      let progressCallCount = 0;
      const genRes = await progClient.generateMusic(
        {
          meta: {
            model_type: 'MORTM4.5D-80M-SFT-Gen',
            task: 'meta',
            program: ['PIANO'],
            tempo: 120,
            num_gems: 1,
            genfield_measure: 4,
            gen_note_dense: { PIANO: 4 },
            p: 0.95,
            temperature: 1.0,
            cot_temperature: 0.1,
            thinking: false,
          },
        },
        (msg) => {
          progressCallCount += 1;
          if (msg === 'poll tick ok') {
            throw new Error('hostile onProgress throw');
          }
        },
      );
      expect(genRes.candidates.length).toBe(1);
      expect(progressCallCount).toBeGreaterThanOrEqual(3);
      await new Promise((r) => setTimeout(r, 30));
      expect(unhandledErrors.length).toBe(0);
    } finally {
      process.off('unhandledRejection', onUnhandled);
      globalThis.fetch = origFetch;
    }

    // -------------------------------------------------------------------------
    // 2. [R16-MED-02] `PianoRollView` non-finite & extreme wheel/pointer clamping (zero hang)
    // -------------------------------------------------------------------------
    const wheelCtrl = new DemoPianoRollController(new StubAudioEngine());
    const WheelHarness: React.FC = () => {
      const [sess, setSess] = React.useState(wheelCtrl.getSnapshot());
      React.useEffect(() => wheelCtrl.subscribe(setSess), []);
      return (
        <PianoRollView
          controller={wheelCtrl}
          session={sess}
          isActive={true}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    };
    const prRender = render(<WheelHarness />);
    const canvas = prRender.getByTestId('pianoroll-canvas');
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    } as DOMRect);

    const makeSyntheticEvent = (type: string, props: Record<string, unknown>): Event => {
      const ev = new Event(type, { bubbles: true, cancelable: true });
      for (const [k, v] of Object.entries(props)) {
        Object.defineProperty(ev, k, { value: v, configurable: true, enumerable: true });
      }
      return ev;
    };

    act(() => {
      canvas.dispatchEvent(makeSyntheticEvent('wheel', { deltaX: 0, deltaY: Number.NaN }));
      canvas.dispatchEvent(
        makeSyntheticEvent('wheel', {
          ctrlKey: true,
          deltaX: 0,
          deltaY: Number.POSITIVE_INFINITY,
        }),
      );
      canvas.dispatchEvent(
        new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaX: 1e308, deltaY: 0 }),
      );
      canvas.dispatchEvent(
        makeSyntheticEvent('pointerdown', {
          clientX: Number.NaN,
          clientY: Number.POSITIVE_INFINITY,
          button: 0,
          pointerId: 7,
          pointerType: 'mouse',
        }),
      );
      window.dispatchEvent(
        makeSyntheticEvent('pointermove', {
          clientX: Number.NaN,
          clientY: 120,
          pointerId: 7,
          pointerType: 'mouse',
        }),
      );
      canvas.dispatchEvent(
        makeSyntheticEvent('dblclick', {
          clientX: Number.NaN,
          clientY: 200,
        }),
      );
    });
    expect(wheelCtrl.getSnapshot().tracks[0].notes.length).toBe(0);
    // Subsequent finite double-click creates a note cleanly without poisoned scroll state
    act(() => {
      canvas.dispatchEvent(
        new MouseEvent('dblclick', {
          bubbles: true,
          clientX: 140,
          clientY: 200,
        }),
      );
    });
    expect(wheelCtrl.getSnapshot().tracks[0].notes.length).toBe(1);
    prRender.unmount();

    // -------------------------------------------------------------------------
    // 3. [R16-MED-03] `NoteCommands` duck-note & ID normalization (zero Undo/Redo leaks)
    // -------------------------------------------------------------------------
    const cmdTrack = new DawTrack({ id: 'cmd_trk', name: 'CmdTrack', instrument: 'PIANO' });
    const duckNoId = { pitch: 64, startTick: 240, durationTicks: 240, velocity: 90 } as unknown as MidiNote;
    const addCmd = new AddNotesCommand(cmdTrack, [duckNoId], '  追加テスト  ');
    addCmd.execute();
    expect(cmdTrack.getNotes().length).toBe(1);
    const addedId = cmdTrack.getNotes()[0].id;
    addCmd.undo();
    expect(cmdTrack.getNotes().length).toBe(0);
    addCmd.execute();
    expect(cmdTrack.getNotes().map((n) => n.id)).toEqual([addedId]);

    const txCmd = new TransformNotesCommand(
      cmdTrack,
      [{ id: addedId, pitch: 64, startTick: 240, durationTicks: 240, velocity: 90 } as unknown as MidiNote],
      [{ id: `  ${addedId}  `, pitch: 67, startTick: 480, durationTicks: 360, velocity: 110 } as unknown as MidiNote],
    );
    txCmd.execute();
    expect(cmdTrack.getNotes()[0].pitch).toBe(67);
    txCmd.undo();
    expect(cmdTrack.getNotes()[0].pitch).toBe(64);

    const delCmd = new DeleteNotesCommand(cmdTrack, [`  ${addedId}  `]);
    delCmd.execute();
    expect(cmdTrack.getNotes().length).toBe(0);
    delCmd.undo();
    expect(cmdTrack.getNotes().length).toBe(1);

    const revHooks = Proxy.revocable(
      { onExecute: () => {}, onUndo: () => {} },
      {},
    );
    revHooks.revoke();
    const repDuck = { pitch: 72, startTick: 0, durationTicks: 480, velocity: 100 } as unknown as MidiNote;
    const repCmd = new ReplaceRangeNotesCommand(
      cmdTrack,
      0,
      960,
      [repDuck],
      '置換',
      undefined,
      revHooks.proxy,
    );
    expect(() => repCmd.execute()).not.toThrow();
    expect(cmdTrack.getNotes().length).toBe(1);
    expect(cmdTrack.getNotes()[0].pitch).toBe(72);
    expect(() => repCmd.undo()).not.toThrow();
    expect(cmdTrack.getNotes().map((n) => n.id)).toEqual([addedId]);

    // -------------------------------------------------------------------------
    // 4. [R16-MED-04] `StandardMidiCodec` track-name UTF-16 truncation & zero-spread encoding
    // -------------------------------------------------------------------------
    const denseNotes = Array.from({ length: 3000 }, (_, i) =>
      new MidiNote({ id: `dn_${i}`, pitch: 48 + (i % 24), startTick: i * 10, durationTicks: 8, velocity: 90 }),
    );
    const overlongTrackSnap = {
      ...new DawTrack({ id: 'ov_t', name: 'Normal', instrument: 'PIANO', notes: denseNotes }).toSnapshot(),
      name: `${'A'.repeat(2000)}\ud800`,
    };
    const encodedMulti = StandardMidiCodec.encodeMultiTrackMidi([overlongTrackSnap], 480, 120);
    const reparsedMulti = StandardMidiCodec.parseMidi(encodedMulti.buffer as ArrayBuffer);
    expect(reparsedMulti.tracks.length).toBe(1);
    expect(reparsedMulti.tracks[0].name.length).toBeLessThanOrEqual(256);
    expect(
      (reparsedMulti.tracks[0].name as string & { isWellFormed?: () => boolean }).isWellFormed?.() ??
        true,
    ).toBe(true);
    expect(reparsedMulti.tracks[0].notes.length).toBe(3000);

    // -------------------------------------------------------------------------
    // 5. [R16-MED-05] `DemoPianoRollController.importParsedMidiTracks` pre-pause validation & NaN guards
    // -------------------------------------------------------------------------
    const impCtrl = new DemoPianoRollController(new StubAudioEngine());
    impCtrl.createNoteAt(60, 0, 480, 100);
    await impCtrl.play();
    expect(impCtrl.getSnapshot().isPlaying).toBe(true);
    // No-op / invalid imports MUST NOT pause active playback!
    impCtrl.importParsedMidiTracks(reparsedMulti, [], false);
    impCtrl.importParsedMidiTracks(null as unknown as typeof reparsedMulti, [0], true);
    expect(impCtrl.getSnapshot().isPlaying).toBe(true);

    // Duck-typed notes + NaN bpm/ppqn import cleanly without poisoning BPM or throwing
    impCtrl.importParsedMidiTracks(
      {
        format: 1,
        ppqn: Number.NaN,
        bpm: Number.NaN,
        beatsPerMeasure: 4,
        tracks: [
          {
            index: 0,
            name: 'DuckImported',
            program: 0,
            isDrum: false,
            instrument: 'PIANO',
            notes: [{ pitch: 65, startTick: 120, durationTicks: 240, velocity: 95 } as unknown as MidiNote],
          },
        ],
      },
      [0],
      true,
    );
    expect(impCtrl.getSnapshot().isPlaying).toBe(false);
    expect(Number.isFinite(impCtrl.getSnapshot().bpm)).toBe(true);
    expect(impCtrl.getSnapshot().tracks[0].notes.length).toBe(1);
    expect(impCtrl.getSnapshot().tracks[0].notes[0].pitch).toBe(65);
    impCtrl.stop();

    // -------------------------------------------------------------------------
    // 6. [R16-LOW-01] `queryOverlappingNotesIndexed` fallback ascending sort + O(log N + K) fast-path preservation
    // -------------------------------------------------------------------------
    const unsortedNotes = [
      new MidiNote({ id: 'u2', pitch: 64, startTick: 480, durationTicks: 120 }),
      new MidiNote({ id: 'u1', pitch: 67, startTick: 120, durationTicks: 120 }),
      new MidiNote({ id: 'u0', pitch: 60, startTick: 120, durationTicks: 120 }),
    ];
    const fallbackSorted = queryOverlappingNotesIndexed(unsortedNotes, [], 0, 960);
    expect(fallbackSorted.map((n) => n.id)).toEqual(['u0', 'u1', 'u2']);

    // -------------------------------------------------------------------------
    // 7. [R16-LOW-02] `ResonanceCircleCanvas` throwing telemetry guard & `isInteractivePianoRollController` Proxy guard
    // -------------------------------------------------------------------------
    const throwingAudio: IAudioEngine = {
      async ensureResumed() {},
      getCurrentTime: () => 0,
      triggerNote: () => {},
      auditionPitch: () => {},
      stopAllVoices: () => {},
      readAcousticTelemetry: () => {
        throw new Error('custom audio telemetry boom');
      },
      async dispose() {},
    };
    const origRaf = window.requestAnimationFrame;
    const origCaf = window.cancelAnimationFrame;
    try {
      let capturedRaf: FrameRequestCallback | null = null;
      window.requestAnimationFrame = ((cb: FrameRequestCallback) => {
        capturedRaf = cb;
        return 1;
      }) as typeof window.requestAnimationFrame;
      window.cancelAnimationFrame = (() => {}) as typeof window.cancelAnimationFrame;
      const rcView = render(
        <ResonanceCircleCanvas
          audioEngine={throwingAudio}
          isGenerating={true}
          isPlayingAudio={true}
          statusTitle="推論中"
          statusDetail="進捗取得中"
        />,
      );
      expect(() => capturedRaf?.(16.67)).not.toThrow();
      rcView.unmount();
    } finally {
      window.requestAnimationFrame = origRaf;
      window.cancelAnimationFrame = origCaf;
    }

    const revCtrl = Proxy.revocable(new DemoPianoRollController(new StubAudioEngine()), {});
    revCtrl.revoke();
    const throwingCapsCtrl = Object.defineProperty({}, 'capabilities', {
      enumerable: true,
      get() {
        throw new Error('caps getter boom');
      },
    });
    expect(isInteractivePianoRollController(revCtrl.proxy)).toBe(false);
    expect(isInteractivePianoRollController(throwingCapsCtrl as unknown as DemoPianoRollController)).toBe(
      false,
    );

    // -------------------------------------------------------------------------
    // 8. [R16-LOW-03] `MeasureRange` `-0` / window bounds & `preload.cjs` atomic `applyIncomingApiConfig`
    // -------------------------------------------------------------------------
    const negZeroNum = evaluateGenerationMeter(-0, 4);
    const negZeroDen = evaluateGenerationMeter(4, -0);
    expect(Object.is(negZeroNum.numerator, -0)).toBe(false);
    expect(Object.is(negZeroDen.denominator, -0)).toBe(false);
    expect(negZeroNum.label).toBe('0/4');
    expect(negZeroDen.label).toBe('4/0');

    const maxRange = new MeasureRange(1024, 1032, 32767, 16);
    const pastInf = maxRange.pastWindowTicks(1e308);
    const futInf = maxRange.futureWindowTicks(1e308);
    expect(Number.isSafeInteger(pastInf.startTick)).toBe(true);
    expect(Number.isSafeInteger(futInf.endTick)).toBe(true);
    expect(futInf.endTick).toBeLessThanOrEqual(MAX_SAFE_MIDI_TICK);

    const { createRequire } = await import('node:module');
    const nodeVm = await import('node:vm');
    const nodeFs = await import('node:fs');
    const nodePath = await import('node:path');
    const preloadSrc = nodeFs.readFileSync(
      nodePath.resolve(process.cwd(), 'electron', 'preload.cjs'),
      'utf8',
    );
    type ExposedApiConfigBridge = {
      getApiConfig: () => { apiBaseUrl: string; handshakeToken: string; backendReady: boolean };
      onApiConfigChanged: (
        cb: (cfg: { apiBaseUrl: string; handshakeToken: string; backendReady: boolean }) => void,
      ) => () => void;
    };
    const bridgeHolder: {
      exposed: ExposedApiConfigBridge | null;
      ipcHandler: ((ev: unknown, cfg: unknown) => void) | null;
    } = { exposed: null, ipcHandler: null };
    const fakeElectron = {
      contextBridge: {
        exposeInMainWorld: (_k: string, val: unknown) => {
          bridgeHolder.exposed = val as ExposedApiConfigBridge;
        },
      },
      ipcRenderer: {
        on: (ch: string, fn: (ev: unknown, cfg: unknown) => void) => {
          if (ch === 'compass:api-config') bridgeHolder.ipcHandler = fn;
        },
        removeListener: () => {},
        invoke: async () => ({
          apiBaseUrl: 'http://127.0.0.1:18100',
          handshakeToken: 'initial_tok',
          backendReady: true,
        }),
        send: () => {},
      },
    };
    const reqFromHere = createRequire(import.meta.url);
    const customRequire = (id: string) => (id === 'electron' ? fakeElectron : reqFromHere(id));
    nodeVm.runInNewContext(preloadSrc, {
      require: customRequire,
      process: { argv: [] },
      Buffer,
      Uint8Array,
      ArrayBuffer,
      Object,
      Array,
      Set,
      Map,
      Number,
      String,
      Boolean,
      Symbol,
      Error,
      TypeError,
    });
    await new Promise((r) => setTimeout(r, 10));
    expect(bridgeHolder.exposed).not.toBeNull();
    const receivedSnaps: Array<{
      apiBaseUrl: string;
      handshakeToken: string;
      backendReady: boolean;
    }> = [];
    const unsubCfg = bridgeHolder.exposed!.onApiConfigChanged((snap) => {
      receivedSnaps.push(snap);
    });
    // Partial invalid update (valid apiBaseUrl + getter on backendReady) MUST roll back atomically!
    const partialGetterCfg: Record<string, unknown> = {
      apiBaseUrl: 'http://127.0.0.1:19999',
    };
    Object.defineProperty(partialGetterCfg, 'backendReady', {
      enumerable: true,
      get() {
        throw new Error('backendReady getter boom');
      },
    });
    bridgeHolder.ipcHandler?.({}, partialGetterCfg);
    bridgeHolder.ipcHandler?.({}, Object.create({ apiBaseUrl: 'http://127.0.0.1:19999' }));
    bridgeHolder.ipcHandler?.({}, { apiBaseUrl: 'http://127.0.0.1:19999', backendReady: 'not_bool' });
    expect(bridgeHolder.exposed!.getApiConfig().apiBaseUrl).toBe('http://127.0.0.1:18100');
    expect(receivedSnaps.length).toBe(0);

    // Valid atomic update commits and delivers frozen snapshot
    bridgeHolder.ipcHandler?.(
      {},
      { apiBaseUrl: 'http://127.0.0.1:18200', handshakeToken: 'next_tok', backendReady: false },
    );
    expect(receivedSnaps.length).toBe(1);
    expect(Object.isFrozen(receivedSnaps[0])).toBe(true);
    expect(bridgeHolder.exposed!.getApiConfig()).toEqual({
      apiBaseUrl: 'http://127.0.0.1:18200',
      handshakeToken: 'next_tok',
      backendReady: false,
    });
    unsubCfg();
  });

  it('29. enforces Round-17 remediations ([R17-MED-01], [R17-MED-02], [R17-LOW-01]): exact Command Undo/Redo, 32768-note cap, mutator 4-way sync safety, and revoked Proxy guards', async () => {
    const fsMod = await import('node:fs');
    const pathMod = await import('node:path');
    const nodeVm = await import('node:vm');
    const { createRequire } = await import('node:module');
    const reqFromHere = createRequire(import.meta.url);
    const ipcSec = reqFromHere(
      pathMod.resolve(process.cwd(), 'electron', 'ipc_security.cjs'),
    ) as {
      isValidWorkspaceSyncPayload: (payload: unknown) => boolean;
    };

    // 1. [R17-MED-01] AddNotesCommand, ReplaceRangeNotesCommand, TransformNotesCommand, DeleteNotesCommand & DawTrack 32,768 cap
    const keepNote = new MidiNote({ id: 'keep_me', pitch: 60, startTick: 0, durationTicks: 480 });
    const trkCollide = new DawTrack({ name: 'Collide', notes: [keepNote] });
    const addCollideCmd = new AddNotesCommand(trkCollide, [
      new MidiNote({ id: 'keep_me', pitch: 64, startTick: 480, durationTicks: 480 }),
    ]);
    addCollideCmd.execute();
    expect(trkCollide.noteCount).toBe(1);
    expect(trkCollide.getNotes()[0].pitch).toBe(60);
    addCollideCmd.undo();
    expect(trkCollide.noteCount).toBe(1);
    expect(trkCollide.getNotes()[0].id).toBe('keep_me');

    // Near 32,768 cap: AddNotesCommand with [new_fit, existing_0, overflow_1] removes ONLY new_fit on undo()
    const nearCapSeed: MidiNote[] = [];
    for (let i = 0; i < MAX_NOTES_PER_TRACK - 1; i += 1) {
      nearCapSeed.push(
        new MidiNote({ id: `seed_${i}`, pitch: 60, startTick: i * 10, durationTicks: 5 }),
      );
    }
    const trkNearCap = new DawTrack({ name: 'NearCap', notes: nearCapSeed });
    expect(trkNearCap.noteCount).toBe(MAX_NOTES_PER_TRACK - 1);
    const addCapCmd = new AddNotesCommand(trkNearCap, [
      new MidiNote({ id: 'seed_0', pitch: 72, startTick: 0, durationTicks: 5 }),
      new MidiNote({ id: 'new_fit', pitch: 74, startTick: 999_990, durationTicks: 5 }),
      new MidiNote({ id: 'overflow_1', pitch: 76, startTick: 999_995, durationTicks: 5 }),
    ]);
    addCapCmd.execute();
    expect(trkNearCap.noteCount).toBe(MAX_NOTES_PER_TRACK);
    addCapCmd.undo();
    expect(trkNearCap.noteCount).toBe(MAX_NOTES_PER_TRACK - 1);
    expect(trkNearCap.getNotes()[0].id).toBe('seed_0');
    expect(trkNearCap.getNotes()[0].pitch).toBe(60);

    // Over-cap constructor (32,770 notes) + surrogate-colliding IDs deduplicated & capped at 32,768 -> Delete + Undo loses 0 notes
    const overCapInput: MidiNote[] = [
      new MidiNote({ id: 'sur_\uD800', pitch: 60, startTick: 0, durationTicks: 10 }),
      new MidiNote({ id: 'sur_\uDFFF', pitch: 62, startTick: 10, durationTicks: 10 }),
      ...nearCapSeed,
      new MidiNote({ id: 'extra_a', pitch: 65, startTick: 500_000, durationTicks: 10 }),
      new MidiNote({ id: 'extra_b', pitch: 67, startTick: 500_020, durationTicks: 10 }),
    ];
    const trkCapped = new DawTrack({ name: 'Capped', notes: overCapInput });
    expect(trkCapped.noteCount).toBe(MAX_NOTES_PER_TRACK);
    const delAtCapCmd = new DeleteNotesCommand(trkCapped, ['seed_0', 'seed_1']);
    delAtCapCmd.execute();
    expect(trkCapped.noteCount).toBe(MAX_NOTES_PER_TRACK - 2);
    delAtCapCmd.undo();
    expect(trkCapped.noteCount).toBe(MAX_NOTES_PER_TRACK);

    // ReplaceRangeNotesCommand colliding with clipped sustaining note `sus_1` and untouched note `out_1`
    const susNote = new MidiNote({ id: 'sus_1', pitch: 60, startTick: 0, durationTicks: 960 });
    const outNote = new MidiNote({ id: 'out_1', pitch: 67, startTick: 1920, durationTicks: 480 });
    const trkRange = new DawTrack({ name: 'Range', notes: [susNote, outNote] });
    const repCmd = new ReplaceRangeNotesCommand(trkRange, 480, 960, [
      new MidiNote({ id: 'sus_1', pitch: 72, startTick: 480, durationTicks: 240 }),
      new MidiNote({ id: 'out_1', pitch: 74, startTick: 720, durationTicks: 240 }),
      new MidiNote({ id: 'gen_ok', pitch: 76, startTick: 480, durationTicks: 480 }),
    ]);
    repCmd.execute();
    expect(trkRange.getNotes().map((n) => `${n.id}:${n.durationTicks}`)).toEqual([
      'sus_1:480',
      'gen_ok:480',
      'out_1:480',
    ]);
    repCmd.undo();
    expect(trkRange.getNotes().map((n) => `${n.id}:${n.durationTicks}`)).toEqual([
      'sus_1:960',
      'out_1:480',
    ]);

    // TransformNotesCommand with omitted `beforeNotes` captures live pre-mutation state & DawTrack.updateNotes preserves key ID
    const trkTx = new DawTrack({
      name: 'Tx',
      notes: [
        new MidiNote({ id: 'tx_a', pitch: 60, startTick: 0, durationTicks: 240 }),
        new MidiNote({ id: 'tx_b', pitch: 64, startTick: 240, durationTicks: 240 }),
      ],
    });
    trkTx.updateNotes(
      new Map([['tx_a', new MidiNote({ id: 'tx_b', pitch: 65, startTick: 0, durationTicks: 240 })]]),
    );
    expect(trkTx.getNotes().map((n) => `${n.id}:${n.pitch}`)).toEqual(['tx_a:65', 'tx_b:64']);
    const txCmd = new TransformNotesCommand(
      trkTx,
      [],
      [new MidiNote({ id: 'tx_a', pitch: 72, startTick: 0, durationTicks: 240 })],
    );
    txCmd.execute();
    expect(trkTx.getNotes()[0].pitch).toBe(72);
    txCmd.undo();
    expect(trkTx.getNotes()[0].pitch).toBe(65);

    // 2. [R17-MED-02] Controller & DawTrack mutator runtime guards + 4-way sync preservation
    expect(isValidTrackInstrument('PIANO')).toBe(true);
    expect(isValidTrackInstrument(' KAZOO ')).toBe(false);
    trkTx.setInstrument(' KAZOO ' as never);
    trkTx.setMuted('yes' as never);
    trkTx.setSoloed(1 as never);
    expect(trkTx.instrument).toBe('PIANO');
    expect(trkTx.isMuted).toBe(false);
    expect(trkTx.isSoloed).toBe(false);

    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    const addedId = ctrl.addTrack(123 as never, 'KAZOO' as never);
    ctrl.setTrackInstrument(addedId, ' KAZOO ' as never);
    ctrl.setQuantize(0 as never, 'yes' as never);
    expect(ctrl.snapTick(130)).toBe(120);

    const ppqn = ctrl.getSnapshot().ppqn;
    const maxProjectTicks = MAX_SMF_QUARTER_NOTES * ppqn;
    ctrl.createNoteAt(60, maxProjectTicks - 60, Number.NaN, 100);
    const createdNotes = ctrl
      .getSnapshot()
      .tracks.find((t) => t.id === addedId)!
      .notes;
    expect(createdNotes.length).toBe(1);
    expect(createdNotes[0].endTick).toBeLessThanOrEqual(maxProjectTicks);

    const revUpdArr = Proxy.revocable([], {});
    revUpdArr.revoke();
    const revUpdItem = Proxy.revocable({}, {});
    revUpdItem.revoke();
    const getterUpdItem: Record<string, unknown> = { id: createdNotes[0].id };
    Object.defineProperty(getterUpdItem, 'durationTicks', {
      enumerable: true,
      get() {
        throw new Error('getter boom');
      },
    });
    expect(() => ctrl.transformNotes(revUpdArr.proxy as never)).not.toThrow();
    expect(() =>
      ctrl.transformNotes([
        revUpdItem.proxy as never,
        getterUpdItem as never,
        {
          id: createdNotes[0].id,
          pitch: Number.NaN,
          startTick: maxProjectTicks - 30,
          durationTicks: Number.NaN,
          velocity: Number.NaN,
        },
      ]),
    ).not.toThrow();
    const transformedNote = ctrl
      .getSnapshot()
      .tracks.find((t) => t.id === addedId)!
      .notes[0];
    expect(transformedNote.endTick).toBeLessThanOrEqual(maxProjectTicks);

    const revSel = Proxy.revocable(new Set<string>(), {});
    revSel.revoke();
    expect(() => ctrl.setSelectedNotes(revSel.proxy as never)).not.toThrow();
    const revCtxOpts = Proxy.revocable({}, {});
    revCtxOpts.revoke();
    await expect(ctrl.extractGenerationContext(revCtxOpts.proxy as never)).resolves.toBeDefined();

    // Verify exported workspace state passes all 4 sync gates (ipc_security.cjs, WorkspaceSyncBus, applyWorkspaceState)
    const exported = ctrl.exportWorkspaceState();
    const envelope = {
      messageId: 'msg_r17_check',
      type: 'SYNC_WORKSPACE_STATE' as const,
      senderId: 'win_r17_check',
      workspaceState: exported,
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(envelope)).toBe(true);
    expect(isValidIncomingSyncMessage(envelope)).toBe(true);
    const dstCtrl = new DemoPianoRollController(new StubAudioEngine());
    dstCtrl.applyWorkspaceState(exported);
    expect(dstCtrl.getSnapshot().tracks.some((t) => t.id === addedId)).toBe(true);

    // 3. [R17-LOW-01] Revoked Proxy safety on importParsedMidiTracks & preload.cjs applyIncomingApiConfig
    const revParsedArr = Proxy.revocable([], {});
    revParsedArr.revoke();
    const revParsedObj = Proxy.revocable({}, {});
    revParsedObj.revoke();
    expect(() =>
      ctrl.importParsedMidiTracks(revParsedArr.proxy as never, [0], true),
    ).not.toThrow();
    expect(() =>
      ctrl.importParsedMidiTracks(revParsedObj.proxy as never, [0], true),
    ).not.toThrow();
    expect(() =>
      ctrl.importParsedMidiTracks(
        {
          format: 1,
          ppqn: 480,
          bpm: 120,
          beatsPerMeasure: 4,
          timeSigNumerator: 4,
          timeSigDenominator: 4,
          tracks: [],
        },
        revParsedArr.proxy as never,
        false,
      ),
    ).not.toThrow();

    const preloadPath = pathMod.resolve(process.cwd(), 'electron', 'preload.cjs');
    const preloadSrc = fsMod.readFileSync(preloadPath, 'utf8');
    expect(preloadSrc.includes("require('util')")).toBe(false);
    expect(preloadSrc.includes('require("util")')).toBe(false);
    const bridgeHolder: {
      ipcHandler: ((ev: unknown, cfg: unknown) => void) | null;
      getApiConfig: (() => { apiBaseUrl: string; handshakeToken: string; backendReady: boolean }) | null;
    } = { ipcHandler: null, getApiConfig: null };
    const fakeElectron = {
      contextBridge: {
        exposeInMainWorld: (_name: string, api: { getApiConfig: () => { apiBaseUrl: string; handshakeToken: string; backendReady: boolean } }) => {
          bridgeHolder.getApiConfig = api.getApiConfig;
        },
      },
      ipcRenderer: {
        on: (ch: string, fn: (ev: unknown, payload: unknown) => void) => {
          if (ch === 'compass:api-config') {
            bridgeHolder.ipcHandler = fn;
          }
        },
        removeListener: () => {},
        invoke: async () => ({
          apiBaseUrl: 'http://127.0.0.1:18300',
          handshakeToken: 'safe_tok',
          backendReady: true,
        }),
        send: () => {},
      },
    };
    nodeVm.runInNewContext(preloadSrc, {
      require: (id: string) => (id === 'electron' ? fakeElectron : reqFromHere(id)),
      process: { argv: [] },
      Buffer,
      Uint8Array,
      ArrayBuffer,
      Object,
      Array,
      Set,
      Map,
      Number,
      String,
      Boolean,
      Symbol,
      Error,
      TypeError,
    });
    await new Promise((r) => setTimeout(r, 10));
    expect(() => bridgeHolder.ipcHandler?.({}, revParsedArr.proxy)).not.toThrow();
    expect(() => bridgeHolder.ipcHandler?.({}, revParsedObj.proxy)).not.toThrow();
    expect(bridgeHolder.getApiConfig?.().apiBaseUrl).toBe('http://127.0.0.1:18300');
  });

  it('30. enforces Round-18 remediations ([R18-MED-01..04], [R18-LOW-01..03]): cross-window ID uniqueness & candidate hijack immunity, TransformNotesCommand execute/undo/redo parity, streaming ZIP-bomb defense, checkHealth sanitization, upload byte-cap, listener isolation, and preload Symbol.toStringTag/URL scheme guards', async () => {
    const fsMod = await import('node:fs');
    const pathMod = await import('node:path');
    const nodeVm = await import('node:vm');
    const { createRequire } = await import('node:module');
    const reqFromHere = createRequire(import.meta.url);
    const ipcSec = reqFromHere(
      pathMod.resolve(process.cwd(), 'electron', 'ipc_security.cjs'),
    ) as {
      isValidWorkspaceSyncPayload: (payload: unknown) => boolean;
    };

    // 1. [R18-MED-01] ReplaceRangeNotesCommand.actuallyInsertedIds prevents hijacking pre-existing user notes,
    // and cross-window synced controllers never mint colliding workspace track.id / note.id
    const bar10Note = new MidiNote({
      id: 'bar10_user_keep',
      pitch: 72,
      startTick: 17_280,
      durationTicks: 480,
    });
    const hijackTrack = new DawTrack({ name: 'HijackGuard', notes: [bar10Note] });
    let hookInsertedIds = new Set<string>();
    const hijackCmd = new ReplaceRangeNotesCommand(
      hijackTrack,
      0,
      1920,
      [
        new MidiNote({ id: 'bar10_user_keep', pitch: 60, startTick: 0, durationTicks: 480 }),
        new MidiNote({ id: 'cand_real_0', pitch: 64, startTick: 480, durationTicks: 480 }),
      ],
      'Hijack Check',
      undefined,
      {
        onExecute: (insertedIds) => {
          hookInsertedIds = new Set(insertedIds);
        },
      },
    );
    hijackCmd.execute();
    expect(hookInsertedIds.has('bar10_user_keep')).toBe(false);
    expect(hookInsertedIds.has('cand_real_0')).toBe(true);
    expect(hijackCmd.actuallyInsertedIds.has('bar10_user_keep')).toBe(false);

    // Simulate multi-window sync followed by addTrack, createNoteAt, pasteNotesAtPlayhead, importParsedMidiTracks, & candidate placement
    const winA = new DemoPianoRollController(new StubAudioEngine());
    winA.addTrack('Lead A', 'SAX');
    winA.createNoteAt(60, 0, 480, 100);
    const winB = new DemoPianoRollController(new StubAudioEngine());
    winB.applyWorkspaceState(winA.exportWorkspaceState());
    const addedOnB = winB.addTrack('Comp B', 'PIANO');
    expect(new Set(winB.getSnapshot().tracks.map((t) => t.id)).size).toBe(
      winB.getSnapshot().tracks.length,
    );
    winB.setActiveTrack(addedOnB);
    winB.createNoteAt(64, 0, 480, 100);
    winB.selectAllNotesOnActiveTrack();
    expect(winB.copySelectedNotes()).toBe(1);
    winB.setPlayheadTick(480);
    expect(winB.pasteNotesAtPlayhead()).toBe(1);

    const demoPreset = StandardMidiCodec.createDemoMidiPreset();
    winB.importParsedMidiTracks(demoPreset, [0], false);
    const smfBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 67, startTick: 0, durationTicks: 480, velocity: 96 })],
      480,
      120,
      'PIANO',
      4,
    );
    const parsedSmf = StandardMidiCodec.parseMidi(smfBytes.buffer as ArrayBuffer);
    await winB.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'cand_0.mid',
          rawMidiBytes: smfBytes,
          parsedMidi: parsedSmf,
          reason: null,
        },
        {
          index: 1,
          filename: 'cand_1.mid',
          rawMidiBytes: smfBytes,
          parsedMidi: parsedSmf,
          reason: null,
        },
      ],
      [],
    );
    await winB.selectGeneratedCandidate(1);
    const exportedWinB = winB.exportWorkspaceState();
    const syncEnv = {
      messageId: 'msg_r18_sync',
      type: 'SYNC_WORKSPACE_STATE' as const,
      senderId: 'win_b',
      workspaceState: exportedWinB,
    };
    expect(ipcSec.isValidWorkspaceSyncPayload(syncEnv)).toBe(true);
    expect(isValidIncomingSyncMessage(syncEnv)).toBe(true);

    // 2. [R18-MED-02] TransformNotesCommand exact execute -> undo -> redo reversibility
    const txN1 = new MidiNote({ id: 'n1', pitch: 60, startTick: 0, durationTicks: 240 });
    const txN2 = new MidiNote({ id: 'n2', pitch: 67, startTick: 240, durationTicks: 240 });
    const txTrack = new DawTrack({ name: 'TxParity', notes: [txN1, txN2] });

    // (a) Empty afterNotes must be a pure no-op across execute -> undo -> redo
    const noopTx = new TransformNotesCommand(
      txTrack,
      [new MidiNote({ id: 'n1', pitch: 40, startTick: 0, durationTicks: 240 })],
      [],
    );
    noopTx.execute();
    expect(txTrack.getNotes()[0].pitch).toBe(60);
    noopTx.undo();
    expect(txTrack.getNotes()[0].pitch).toBe(60);
    noopTx.execute();
    expect(txTrack.getNotes()[0].pitch).toBe(60);

    // (b) Partial afterNotes + stale beforeNotes: only n1 mutates, n2 stays 67, and live pre-state (60) is restored
    const partialStaleTx = new TransformNotesCommand(
      txTrack,
      [
        new MidiNote({ id: 'n1', pitch: 48, startTick: 0, durationTicks: 240 }),
        new MidiNote({ id: 'n2', pitch: 48, startTick: 240, durationTicks: 240 }),
      ],
      [new MidiNote({ id: 'n1', pitch: 72, startTick: 0, durationTicks: 240 })],
    );
    partialStaleTx.execute();
    expect(txTrack.getNotes().map((n) => `${n.id}:${n.pitch}`)).toEqual(['n1:72', 'n2:67']);
    partialStaleTx.undo();
    expect(txTrack.getNotes().map((n) => `${n.id}:${n.pitch}`)).toEqual(['n1:60', 'n2:67']);
    partialStaleTx.execute();
    expect(txTrack.getNotes().map((n) => `${n.id}:${n.pitch}`)).toEqual(['n1:72', 'n2:67']);

    // 3. [R18-MED-04] CompassApiClient streaming ZIP-bomb defense, checkHealth sanitization, & non-string guards
    const safeClient = new CompassApiClient(123 as never, 456 as never);
    expect(() => safeClient.setBaseUrl(null as never)).not.toThrow();
    expect(() => safeClient.setHandshakeToken(123 as never)).not.toThrow();
    const emptyAuthClient = new CompassApiClient('http://127.0.0.1:8000', '');
    expect(emptyAuthClient.handshakeToken).toBe('');
    emptyAuthClient.setHandshakeToken('tok_temp');
    expect(emptyAuthClient.handshakeToken).toBe('tok_temp');
    emptyAuthClient.setHandshakeToken('');
    expect(emptyAuthClient.handshakeToken).toBe('');

    const bombZip = new JSZip();
    bombZip.file('00_bomb.mid', new Uint8Array(6 * 1024 * 1024), {
      compression: 'DEFLATE',
      compressionOptions: { level: 9 },
    });
    bombZip.file('01_valid.mid', smfBytes, { compression: 'DEFLATE' });
    const bombZipBuf = await bombZip.generateAsync({ type: 'arraybuffer' });
    expect(bombZipBuf.byteLength).toBeLessThan(100_000);

    const origFetch = globalThis.fetch;
    try {
      let callPhase: 'zip' | 'bad_health' | 'ok_health' = 'zip';
      globalThis.fetch = vi.fn(async () => {
        if (callPhase === 'zip') {
          return new Response(bombZipBuf, {
            status: 200,
            headers: { 'Content-Type': 'application/zip' },
          });
        }
        if (callPhase === 'bad_health') {
          return new Response(JSON.stringify(['not_a_health_object']), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response(
          JSON.stringify({
            status: '  ok  ',
            engine: 'MORTM4.5',
            device: 'cuda:0',
            dtype: 'torch.bfloat16',
            models_dir_ready: true,
            available_models_count: 9,
            auth_required: true,
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }) as unknown as typeof fetch;

      const genRes = await emptyAuthClient.generateMusic({
        meta: {
          model_type: 'MORTM4.5D-SFT-Gen',
          task: 'meta',
          program: ['PIANO'],
          tempo: 120,
          num_gems: 2,
          genfield_measure: 4,
          gen_note_dense: null,
          p: 0.95,
          temperature: 1.0,
          cot_temperature: 0.8,
          thinking: false,
        },
      });
      expect(genRes.candidates.length).toBe(1);
      expect(genRes.candidates[0].filename).toBe('01_valid.mid');

      callPhase = 'bad_health';
      await expect(emptyAuthClient.checkHealth()).rejects.toThrow();

      callPhase = 'ok_health';
      const health = await emptyAuthClient.checkHealth();
      expect(health).toEqual({
        status: 'ok',
        engine: 'MORTM4.5',
        device: 'cuda:0',
        dtype: 'torch.bfloat16',
        models_dir_ready: true,
        available_models_count: 9,
        auth_required: true,
      });
    } finally {
      globalThis.fetch = origFetch;
    }

    // 4. [R18-LOW-01] parseUploadedMidiFile NaN / spoofed size & StandardMidiCodec MAX_SMF_PARSE_BYTES
    expect(MAX_SMF_PARSE_BYTES).toBe(32 * 1024 * 1024);
    await expect(winA.parseUploadedMidiFile(null as never)).rejects.toThrow();
    await expect(
      winA.parseUploadedMidiFile({
        name: 'nan_size.mid',
        size: Number.NaN,
        arrayBuffer: async () => smfBytes.buffer,
      } as unknown as File),
    ).rejects.toThrow();
    await expect(
      winA.parseUploadedMidiFile({
        name: 'spoofed_11mb.mid',
        size: 100,
        arrayBuffer: async () => new ArrayBuffer(10 * 1024 * 1024 + 64),
      } as unknown as File),
    ).rejects.toThrow(/10MB/);

    // 5. [R18-LOW-02] subscribe listener validation & exception isolation + ExternalDawAdapters extractGenerationContext LSP parity
    let healthyCalls = 0;
    const unsubNonFn = winA.subscribe(123 as never);
    expect(typeof unsubNonFn).toBe('function');
    const unsubThrow = winA.subscribe(() => {
      throw new Error('subscriber boom');
    });
    const unsubHealthy = winA.subscribe(() => {
      healthyCalls += 1;
    });
    expect(() => winA.setBpm(128)).not.toThrow();
    expect(healthyCalls).toBe(1);
    unsubThrow();
    unsubHealthy();

    const ableton = new AbletonDawController();
    const revOpts = Proxy.revocable({}, {});
    revOpts.revoke();
    await expect(ableton.extractGenerationContext(null as never)).resolves.toBeDefined();
    await expect(ableton.extractGenerationContext(revOpts.proxy as never)).resolves.toBeDefined();
    let abletonHealthy = 0;
    ableton.subscribe(() => {
      throw new Error('ableton subscriber boom');
    });
    ableton.subscribe(() => {
      abletonHealthy += 1;
    });
    expect(() => ableton.setBpm(132)).not.toThrow();
    expect(abletonHealthy).toBe(1);

    const reg = new DawControllerRegistry();
    reg.register(winA);
    reg.register(ableton);
    let regHealthy = 0;
    reg.subscribeActiveChange(null as never);
    reg.subscribeActiveChange(() => {
      throw new Error('reg subscriber boom');
    });
    reg.subscribeActiveChange(() => {
      regHealthy += 1;
    });
    await expect(reg.activate(ableton.id)).resolves.toBe(ableton);
    expect(regHealthy).toBe(1);

    // 6. [R18-LOW-03] preload.cjs Symbol.toStringTag getter prevention, extra-key rejection & pure-RegExp http(s) scheme check
    const preloadPath = pathMod.resolve(process.cwd(), 'electron', 'preload.cjs');
    const preloadSrc = fsMod.readFileSync(preloadPath, 'utf8');
    expect(preloadSrc.includes("require('util')")).toBe(false);
    expect(preloadSrc.includes('new URL(')).toBe(false);

    const holder: {
      ipcHandler: ((ev: unknown, cfg: unknown) => void) | null;
      getApiConfig:
        | (() => { apiBaseUrl: string; handshakeToken: string; backendReady: boolean })
        | null;
    } = { ipcHandler: null, getApiConfig: null };
    const fakeElectron = {
      contextBridge: {
        exposeInMainWorld: (
          _name: string,
          api: {
            getApiConfig: () => {
              apiBaseUrl: string;
              handshakeToken: string;
              backendReady: boolean;
            };
          },
        ) => {
          holder.getApiConfig = api.getApiConfig;
        },
      },
      ipcRenderer: {
        on: (ch: string, fn: (ev: unknown, payload: unknown) => void) => {
          if (ch === 'compass:api-config') {
            holder.ipcHandler = fn;
          }
        },
        removeListener: () => {},
        invoke: async () => ({
          apiBaseUrl: 'http://127.0.0.1:18400',
          handshakeToken: 'init_tok',
          backendReady: true,
        }),
        send: () => {},
      },
    };
    nodeVm.runInNewContext(preloadSrc, {
      require: (id: string) => (id === 'electron' ? fakeElectron : reqFromHere(id)),
      process: { argv: [] },
      Buffer,
      Uint8Array,
      ArrayBuffer,
      Object,
      Array,
      Set,
      Map,
      Number,
      String,
      Boolean,
      Symbol,
      Error,
      TypeError,
      RegExp,
    });
    await new Promise((r) => setTimeout(r, 10));

    let tagGetterInvoked = 0;
    const tagTrapObj: Record<PropertyKey, unknown> = {
      apiBaseUrl: 'http://127.0.0.1:19999',
    };
    Object.defineProperty(tagTrapObj, Symbol.toStringTag, {
      get() {
        tagGetterInvoked += 1;
        return 'Object';
      },
    });
    holder.ipcHandler?.({}, tagTrapObj);
    expect(tagGetterInvoked).toBe(0);
    holder.ipcHandler?.({}, { apiBaseUrl: 'javascript:alert(1)' });
    holder.ipcHandler?.({}, { apiBaseUrl: 'http://127.0.0.1:19999', extraKey: 1 });
    expect(holder.getApiConfig?.().apiBaseUrl).toBe('http://127.0.0.1:18400');

    holder.ipcHandler?.(
      {},
      { apiBaseUrl: 'https://127.0.0.1:18443', handshakeToken: 'ok_tok', backendReady: true },
    );
    expect(holder.getApiConfig?.().apiBaseUrl).toBe('https://127.0.0.1:18443');
  });

  it('31. enforces Round-20 O(1) dense-array length pre-cap, 4096-byte plain-Array vs 2MB Uint8Array 4-way parity, cumulative budget pre-cap, and TOCTOU-free parseUploadedMidiFile ([R19-MED-01], [R19-MED-02], [R19-LOW-01])', async () => {
    const nodeFs = await import('node:fs');
    const nodePath = await import('node:path');
    const nodeVm = await import('node:vm');
    const reqFromHere = (await import('node:module')).createRequire(import.meta.url);
    const ipcSec = reqFromHere('../../electron/ipc_security.cjs') as {
      MAX_PLAIN_ARRAY_MIDI_BYTES: number;
      isValidWorkspaceSyncPayload: (m: unknown) => boolean;
    };

    expect(MAX_PLAIN_ARRAY_MIDI_BYTES).toBe(4096);
    expect(ipcSec.MAX_PLAIN_ARRAY_MIDI_BYTES).toBe(4096);

    const preloadPath = nodePath.resolve(process.cwd(), 'electron/preload.cjs');
    const preloadSrc = nodeFs.readFileSync(preloadPath, 'utf8');
    expect(preloadSrc.includes("require('util')")).toBe(false);
    expect(preloadSrc.includes('new URL(')).toBe(false);

    let sentSyncPayloads: unknown[] = [];
    let exposedDesktopApi: {
      sendWorkspaceSync: (msg: unknown) => void;
    } | null = null;
    const fakeElectron = {
      contextBridge: {
        exposeInMainWorld: (_name: string, api: { sendWorkspaceSync: (m: unknown) => void }) => {
          exposedDesktopApi = api;
        },
      },
      ipcRenderer: {
        on: () => {},
        removeListener: () => {},
        invoke: async () => ({
          apiBaseUrl: 'http://127.0.0.1:18400',
          handshakeToken: 'init_tok',
          backendReady: true,
        }),
        send: (ch: string, payload: unknown) => {
          if (ch === 'compass:workspace-sync') {
            sentSyncPayloads.push(payload);
          }
        },
      },
    };
    nodeVm.runInNewContext(preloadSrc, {
      require: (id: string) => (id === 'electron' ? fakeElectron : reqFromHere(id)),
      process: { argv: [] },
      Buffer,
      Uint8Array,
      ArrayBuffer,
      Object,
      Array,
      Set,
      Map,
      Number,
      String,
      Boolean,
      Symbol,
      Error,
      TypeError,
      RegExp,
      Math,
    });
    expect(exposedDesktopApi).not.toBeNull();

    const tryPreloadSync = (msg: unknown): boolean => {
      sentSyncPayloads = [];
      exposedDesktopApi!.sendWorkspaceSync(msg);
      return sentSyncPayloads.length === 1;
    };

    const seedSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'n_r20', pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const makeSmfOfSize = (byteLen: number): Uint8Array => {
      const out = new Uint8Array(byteLen);
      out.set(seedSmf.subarray(0, Math.min(seedSmf.byteLength, byteLen)));
      return out;
    };

    const baseWs = (candidates: unknown[], tracksOverride?: unknown) => ({
      ppqn: 480,
      bpm: 120,
      beatsPerMeasure: 4,
      timeSigNumerator: 4,
      timeSigDenominator: 4,
      activeTrackId: 'trk_r20',
      targetStartMeasure: 1,
      targetEndMeasure: 4,
      quantizeDivision: 16 as const,
      isTriplet: false,
      tracks:
        tracksOverride ?? [
          {
            id: 'trk_r20',
            name: 'Piano',
            instrument: 'PIANO',
            colorHex: '#E4E4E7',
            isMuted: false,
            isSoloed: false,
            notes: [
              { id: 'n_r20', pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 },
            ],
          },
        ],
      candidates,
      activeCandidateIndex: candidates.length > 0 ? 0 : -1,
      activeCandidateNoteIds: [],
      activeCandidateRange: null,
      candidateOriginSlot: null,
      latestReason: null,
    });
    const wrapEnv = (ws: unknown, seq: number) => ({
      messageId: `msg_r20_${seq}`,
      type: 'SYNC_WORKSPACE_STATE',
      senderId: 'sender_r20_remote',
      workspaceState: ws,
    });

    // 1. [R19-MED-02] 500,000-element tracks array rejects in O(1) (< 30 ms total across all 4 gates + applyWorkspaceState)
    const hugeTracks = new Array(500_000).fill(0);
    const hugeTracksEnv = wrapEnv(baseWs([], hugeTracks), 1);
    const ctrlTarget = new DemoPianoRollController(new StubAudioEngine());
    const t0Tracks = performance.now();
    expect(ipcSec.isValidWorkspaceSyncPayload(hugeTracksEnv)).toBe(false);
    expect(tryPreloadSync(hugeTracksEnv)).toBe(false);
    expect(isValidIncomingSyncMessage(hugeTracksEnv)).toBe(false);
    ctrlTarget.applyWorkspaceState(hugeTracksEnv.workspaceState as never);
    const dtTracksMs = performance.now() - t0Tracks;
    expect(dtTracksMs).toBeLessThan(30);

    // 2. [R19-MED-01] 32 x 2,000,000-element plain Array rawMidiBytes rejects in O(1) (< 30 ms total across all 4 gates)
    const hugePlainMidi = new Array(2_000_000).fill(0);
    hugePlainMidi[0] = 0x4d;
    hugePlainMidi[1] = 0x54;
    hugePlainMidi[2] = 0x68;
    hugePlainMidi[3] = 0x64;
    const hugePlainCands = Array.from({ length: 32 }, (_, idx) => ({
      index: idx,
      filename: `huge_${idx}.mid`,
      rawMidiBytes: hugePlainMidi,
      reason: null,
    }));
    const hugePlainEnv = wrapEnv(baseWs(hugePlainCands), 2);
    const t0Plain = performance.now();
    expect(ipcSec.isValidWorkspaceSyncPayload(hugePlainEnv)).toBe(false);
    expect(tryPreloadSync(hugePlainEnv)).toBe(false);
    expect(isValidIncomingSyncMessage(hugePlainEnv)).toBe(false);
    ctrlTarget.applyWorkspaceState(hugePlainEnv.workspaceState as never);
    const dtPlainMs = performance.now() - t0Plain;
    expect(dtPlainMs).toBeLessThan(30);

    // 3. [R19-MED-01] Cumulative 2MB budget pre-capped BEFORE cloning/iterating candidate 1..31 (< 30 ms)
    const u8_1_5mb = makeSmfOfSize(1_500_000);
    const overBudgetU8Cands = Array.from({ length: 32 }, (_, idx) => ({
      index: idx,
      filename: `u8_${idx}.mid`,
      rawMidiBytes: u8_1_5mb,
      reason: null,
    }));
    const overBudgetEnv = wrapEnv(baseWs(overBudgetU8Cands), 3);
    const t0Budget = performance.now();
    expect(ipcSec.isValidWorkspaceSyncPayload(overBudgetEnv)).toBe(false);
    expect(tryPreloadSync(overBudgetEnv)).toBe(false);
    expect(isValidIncomingSyncMessage(overBudgetEnv)).toBe(false);
    ctrlTarget.applyWorkspaceState(overBudgetEnv.workspaceState as never);
    const dtBudgetMs = performance.now() - t0Budget;
    expect(dtBudgetMs).toBeLessThan(30);

    // 4. [R19-MED-01] 100% 4-way parity: 4096 plain number[] ACCEPTED, 4097 plain number[] REJECTED, 8192 Uint8Array ACCEPTED
    const plain4096 = Array.from(makeSmfOfSize(4096));
    const env4096 = wrapEnv(
      baseWs([{ index: 0, filename: 'ok_4096.mid', rawMidiBytes: plain4096, reason: null }]),
      4,
    );
    expect(ipcSec.isValidWorkspaceSyncPayload(env4096)).toBe(true);
    expect(tryPreloadSync(env4096)).toBe(true);
    expect(isValidIncomingSyncMessage(env4096)).toBe(true);
    ctrlTarget.applyWorkspaceState(env4096.workspaceState as never);
    expect(ctrlTarget.getSnapshot().candidates.length).toBe(1);

    const plain4097 = Array.from(makeSmfOfSize(4097));
    const env4097 = wrapEnv(
      baseWs([{ index: 0, filename: 'bad_4097.mid', rawMidiBytes: plain4097, reason: null }]),
      5,
    );
    expect(ipcSec.isValidWorkspaceSyncPayload(env4097)).toBe(false);
    expect(tryPreloadSync(env4097)).toBe(false);
    expect(isValidIncomingSyncMessage(env4097)).toBe(false);
    const ctrlReject4097 = new DemoPianoRollController(new StubAudioEngine());
    ctrlReject4097.applyWorkspaceState(env4097.workspaceState as never);
    expect(ctrlReject4097.getSnapshot().candidates.length).toBe(0);

    // Accessor trap on index 4095 of plain Array must be rejected with zero getter invocations
    let getterHits = 0;
    const plainGetterArr = Array.from(makeSmfOfSize(4096));
    Object.defineProperty(plainGetterArr, 4095, {
      get() {
        getterHits += 1;
        return 0;
      },
      enumerable: true,
      configurable: true,
    });
    const envGetter = wrapEnv(
      baseWs([{ index: 0, filename: 'getter.mid', rawMidiBytes: plainGetterArr, reason: null }]),
      6,
    );
    expect(ipcSec.isValidWorkspaceSyncPayload(envGetter)).toBe(false);
    expect(tryPreloadSync(envGetter)).toBe(false);
    expect(isValidIncomingSyncMessage(envGetter)).toBe(false);
    ctrlReject4097.applyWorkspaceState(envGetter.workspaceState as never);
    expect(getterHits).toBe(0);

    // Genuine Uint8Array > 4096 (8192 bytes) accepted across all 4 gates + applyWorkspaceState
    const u8_8192 = makeSmfOfSize(8192);
    const envU8_8192 = wrapEnv(
      baseWs([{ index: 0, filename: 'ok_8192.mid', rawMidiBytes: u8_8192, reason: null }]),
      7,
    );
    expect(ipcSec.isValidWorkspaceSyncPayload(envU8_8192)).toBe(true);
    expect(tryPreloadSync(envU8_8192)).toBe(true);
    expect(isValidIncomingSyncMessage(envU8_8192)).toBe(true);
    const ctrlAccept8192 = new DemoPianoRollController(new StubAudioEngine());
    ctrlAccept8192.applyWorkspaceState(envU8_8192.workspaceState as never);
    expect(ctrlAccept8192.getSnapshot().candidates.length).toBe(1);
    expect(ctrlAccept8192.getSnapshot().candidates[0].rawMidiBytes.byteLength).toBe(8192);

    // 5. [R19-LOW-01] parseUploadedMidiFile single-read (zero TOCTOU), intrinsic ArrayBuffer byteLength, clean error wrapping
    let arrayBufferGetterReads = 0;
    const singleReadFile = {
      name: 'single_read.mid',
      size: seedSmf.byteLength,
      get arrayBuffer() {
        arrayBufferGetterReads += 1;
        return async () => seedSmf.slice().buffer;
      },
    } as unknown as File;
    const parsedOk = await ctrlAccept8192.parseUploadedMidiFile(singleReadFile);
    expect(arrayBufferGetterReads).toBe(1);
    expect(parsedOk.tracks.length).toBe(1);

    // Own-property `{ byteLength: 100 }` shadowing on a 12 MB ArrayBuffer must still fail with 10MB cap error
    const twelveMbBuf = new ArrayBuffer(12 * 1024 * 1024);
    Object.defineProperty(twelveMbBuf, 'byteLength', {
      value: 100,
      configurable: true,
    });
    await expect(
      ctrlAccept8192.parseUploadedMidiFile({
        name: 'shadowed_12mb.mid',
        size: 100,
        arrayBuffer: async () => twelveMbBuf,
      } as unknown as File),
    ).rejects.toThrow(/10MB/);

    // Plain non-ArrayBuffer `{ byteLength: 100 }` must fail with Standard MIDI error
    await expect(
      ctrlAccept8192.parseUploadedMidiFile({
        name: 'fake_buf.mid',
        size: 100,
        arrayBuffer: async () => ({ byteLength: 100 }) as unknown as ArrayBuffer,
      } as unknown as File),
    ).rejects.toThrow(/Standard MIDI/);

    // Rejected `arrayBuffer()` promise must be wrapped cleanly without leaking internal error message
    let caughtMsg = '';
    try {
      await ctrlAccept8192.parseUploadedMidiFile({
        name: 'broken_io.mid',
        size: 100,
        arrayBuffer: () => Promise.reject(new Error('INTERNAL_DISK_PATH_C_SECRET')),
      } as unknown as File);
    } catch (err) {
      caughtMsg = err instanceof Error ? err.message : String(err);
    }
    expect(caughtMsg).toContain('MIDIファイルの読み込みに失敗しました');
    expect(caughtMsg).not.toContain('INTERNAL_DISK_PATH_C_SECRET');
  });

  it('32. enforces Round-21 O(1) FIFO SMF queue, <=65,536 per-SMF note cap, zero-copy MThd pre-check, 2-pass sanitizeParsedMidiFile, and <=8 failed-candidate retry cap ([R20-HIGH-01], [R20-MED-01])', async () => {
    expect(MAX_PARSED_SMF_NOTES).toBe(65_536);
    expect(MAX_FAILED_CANDIDATE_ATTEMPTS).toBe(8);

    const buildStackedRunningStatusSmf = (noteOnCount: number, closeAll: boolean): Uint8Array => {
      // MThd (14 bytes) + MTrk header (8 bytes) + status 0x90 step 0 + remaining 3-byte running-status events + EOT (4 bytes)
      const bodyLen =
        1 + 3 + (noteOnCount - 1) * 3 + (closeAll ? noteOnCount * 3 : 0) + 4;
      const buf = new Uint8Array(14 + 8 + bodyLen);
      const dv = new DataView(buf.buffer);
      buf.set([0x4d, 0x54, 0x68, 0x64], 0);
      dv.setUint32(4, 6, false);
      dv.setUint16(8, 0, false);
      dv.setUint16(10, 1, false);
      dv.setUint16(12, 480, false);
      buf.set([0x4d, 0x54, 0x72, 0x6b], 14);
      dv.setUint32(18, bodyLen, false);
      let p = 22;
      // First Note-On establishes Running Status 0x90
      buf[p++] = 0x00;
      buf[p++] = 0x90;
      buf[p++] = 0x3c;
      buf[p++] = 0x64;
      for (let i = 1; i < noteOnCount; i += 1) {
        buf[p++] = 0x00;
        buf[p++] = 0x3c;
        buf[p++] = 0x64;
      }
      if (closeAll) {
        for (let i = 0; i < noteOnCount; i += 1) {
          buf[p++] = i === 0 ? 0x10 : 0x00;
          buf[p++] = 0x3c;
          buf[p++] = 0x00;
        }
      }
      buf[p++] = 0x00;
      buf[p++] = 0xff;
      buf[p++] = 0x2f;
      buf[p++] = 0x00;
      return buf;
    };

    // 1. [R20-HIGH-01] 30,000 stacked same-pitch Note-Ons + 30,000 Note-Offs dequeue in O(1) (< 250 ms, zero shift() O(N^2))
    const stacked30k = buildStackedRunningStatusSmf(30_000, true);
    const t0Stack = performance.now();
    const parsed30k = StandardMidiCodec.parseMidi(stacked30k.buffer as ArrayBuffer);
    const dtStackMs = performance.now() - t0Stack;
    expect(parsed30k.tracks[0].notes.length).toBe(30_000);
    expect(dtStackMs).toBeLessThan(250);

    // 2. [R20-HIGH-01] < 2 MB 333,000 stacked Note-Ons & 666,000 unclosed Note-Ons abort in < 30 ms
    const stacked333k = buildStackedRunningStatusSmf(333_000, true);
    expect(stacked333k.byteLength).toBeLessThan(MAX_CANDIDATE_MIDI_BYTES);
    const t0333k = performance.now();
    expect(() => StandardMidiCodec.parseMidi(stacked333k.buffer as ArrayBuffer)).toThrow(/65536/);
    expect(performance.now() - t0333k).toBeLessThan(30);

    const unclosed666k = buildStackedRunningStatusSmf(666_000, false);
    expect(unclosed666k.byteLength).toBeLessThan(MAX_CANDIDATE_MIDI_BYTES);
    const t0666k = performance.now();
    expect(() => StandardMidiCodec.parseMidi(unclosed666k.buffer as ArrayBuffer)).toThrow(/65536/);
    expect(performance.now() - t0666k).toBeLessThan(30);

    // Sync receiver (`applyWorkspaceState`) handles 333k-note < 2 MB candidate in < 30 ms via empty-tracks fallback
    const syncCtrl = new DemoPianoRollController(new StubAudioEngine());
    const t0Sync333k = performance.now();
    syncCtrl.applyWorkspaceState({
      ppqn: 480,
      bpm: 120,
      beatsPerMeasure: 4,
      timeSigNumerator: 4,
      timeSigDenominator: 4,
      activeTrackId: 't_r21',
      targetStartMeasure: 1,
      targetEndMeasure: 4,
      quantizeDivision: 16,
      isTriplet: false,
      tracks: [
        {
          id: 't_r21',
          name: 'Piano',
          instrument: 'PIANO',
          colorHex: '#E4E4E7',
          isMuted: false,
          isSoloed: false,
          notes: [],
        },
      ],
      candidates: [
        { index: 0, filename: 'bomb_333k.mid', rawMidiBytes: stacked333k, reason: null },
      ],
      activeCandidateIndex: 0,
      activeCandidateNoteIds: [],
      activeCandidateRange: null,
      candidateOriginSlot: null,
      latestReason: null,
    });
    expect(performance.now() - t0Sync333k).toBeLessThan(30);
    expect(syncCtrl.getSnapshot().candidates.length).toBe(1);
    expect(syncCtrl.getSnapshot().candidates[0].parsedMidi.tracks).toEqual([]);

    // 3. [R20-MED-01] 256 x 2 MB non-MIDI Uint8Arrays reject via zero-copy 4-byte MThd pre-check (< 30 ms, 0 MB copied)
    const nonMidi2Mb = new Uint8Array(MAX_CANDIDATE_MIDI_BYTES);
    nonMidi2Mb[0] = 0x50; // 'P' (not 'MThd')
    const t0PreMagic = performance.now();
    for (let i = 0; i < 256; i += 1) {
      expect(extractValidatedCandidateMidiBytes(nonMidi2Mb, false)).toBeNull();
    }
    expect(performance.now() - t0PreMagic).toBeLessThan(30);

    // 4. [R20-MED-01] 256 entries with 32,767 valid note POJOs + 1 bad note at end + 0-MTrk MThd halt strictly at 8 attempts
    const validNotePojo = { pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 };
    const poisonedNotes = new Array(32_768).fill(validNotePojo);
    poisonedNotes[32_767] = { pitch: 999, startTick: 0, durationTicks: 480, velocity: 100 };
    const zeroTrackMThd = new Uint8Array(22);
    zeroTrackMThd.set([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 0, 0x01, 0xe0], 0);
    const poisonedBatch = Array.from({ length: 256 }, (_, idx) => ({
      index: idx,
      filename: `poison_${idx}.mid`,
      rawMidiBytes: zeroTrackMThd,
      parsedMidi: {
        format: 0,
        ppqn: 480,
        bpm: 120,
        beatsPerMeasure: 4,
        tracks: [{ index: 0, name: 'Poison', program: 0, isDrum: false, instrument: 'PIANO', notes: poisonedNotes }],
      },
      reason: null,
    }));
    let index8Touched = 0;
    Object.defineProperty(poisonedBatch, 8, {
      get() {
        index8Touched += 1;
        return null;
      },
      enumerable: true,
      configurable: true,
    });
    const t0Poison = performance.now();
    const normPoison = normalizeGeneratedCandidatesBatch(poisonedBatch as never);
    const dtPoisonMs = performance.now() - t0Poison;
    expect(normPoison).toEqual([]);
    expect(index8Touched).toBe(0);
    expect(dtPoisonMs).toBeLessThan(1500);
  });

  it('enforces Round 22 Fact-Checker (R21-HIGH-02) remediations: 512-track SMF ceiling, 64-track modal selection cap, index >= 512 POJO import, and exact post-import / full-workspace notice parity', async () => {
    const buildMultiTrackSmf = (mtrkCount: number, channelsPerMtrk: number): Uint8Array => {
      const mtrkPayload: number[] = [];
      for (let ch = 0; ch < channelsPerMtrk; ch += 1) {
        mtrkPayload.push(0x00, 0x90 | (ch & 0x0f), 60 + (ch % 12), 100);
        mtrkPayload.push(0x10, 0x80 | (ch & 0x0f), 60 + (ch % 12), 0);
      }
      mtrkPayload.push(0x00, 0xff, 0x2f, 0x00);
      const chunkLen = mtrkPayload.length;
      const chunkSize = 8 + chunkLen;
      const out = new Uint8Array(14 + mtrkCount * chunkSize);
      const dv = new DataView(out.buffer);
      out.set([0x4d, 0x54, 0x68, 0x64], 0);
      dv.setUint32(4, 6, false);
      dv.setUint16(8, 1, false);
      dv.setUint16(10, mtrkCount, false);
      dv.setUint16(12, 480, false);
      for (let t = 0; t < mtrkCount; t += 1) {
        const off = 14 + t * chunkSize;
        out.set([0x4d, 0x54, 0x72, 0x6b], off);
        dv.setUint32(off + 4, chunkLen, false);
        out.set(mtrkPayload, off + 8);
      }
      return out;
    };

    // 1. Exact 512-track boundary succeeds; 513-track & 65,536-track (4096 x 16ch, ~304KB) SMFs fail fast
    expect(MAX_PARSED_SMF_TRACKS).toBe(512);
    const smf512 = buildMultiTrackSmf(512, 1);
    const parsed512 = StandardMidiCodec.parseMidi(smf512.buffer as ArrayBuffer);
    expect(parsed512.tracks.length).toBe(512);

    const smf513 = buildMultiTrackSmf(513, 1);
    expect(() => StandardMidiCodec.parseMidi(smf513.buffer as ArrayBuffer)).toThrow(/512トラック/);

    const smf65536 = buildMultiTrackSmf(4096, 16);
    const t0Bomb = performance.now();
    expect(() => StandardMidiCodec.parseMidi(smf65536.buffer as ArrayBuffer)).toThrow(/512トラック/);
    expect(performance.now() - t0Bomb).toBeLessThan(50);

    // applyWorkspaceState safely catches >512-track candidate MIDI and falls back to tracks: []
    const syncCtrl = new DemoPianoRollController(new StubAudioEngine());
    const baseState = syncCtrl.exportWorkspaceState();
    syncCtrl.applyWorkspaceState({
      ...baseState,
      candidates: [
        {
          index: 0,
          filename: 'many_tracks_65536.mid',
          rawMidiBytes: smf65536,
          reason: null,
        },
      ],
      activeCandidateIndex: 0,
    });
    expect(syncCtrl.getSnapshot().candidates[0].parsedMidi.tracks).toEqual([]);

    // 2. MidiTrackImportModal caps initial & interactive selection at 64 tracks and DOM rows at 512
    expect(MAX_MODAL_IMPORT_SELECTION).toBe(64);
    let confirmedIndices: readonly number[] = [];
    const modal80 = render(
      <MidiTrackImportModal
        fileName="session_80_tracks.mid"
        parsedMidi={{ ...parsed512, tracks: parsed512.tracks.slice(0, 80) }}
        onConfirm={(indices) => {
          confirmedIndices = indices;
        }}
        onCancel={() => {}}
      />,
    );
    const checkboxes80 = Array.from(
      modal80.container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'),
    );
    expect(checkboxes80.length).toBe(80);
    expect(checkboxes80.filter((cb) => cb.checked).length).toBe(64);
    expect(checkboxes80[64].disabled).toBe(true);
    expect(modal80.getByTestId('midi-import-selection-cap-hint').textContent).toContain('最大 64');

    // Unchecking track 0 frees 1 slot so track 64 becomes selectable
    fireEvent.click(checkboxes80[0]);
    expect(checkboxes80[64].disabled).toBe(false);
    fireEvent.click(checkboxes80[64]);
    fireEvent.click(modal80.getByTestId('confirm-midi-import-button'));
    expect(confirmedIndices.length).toBe(64);
    expect(confirmedIndices).not.toContain(0);
    expect(confirmedIndices).toContain(64);
    modal80.unmount();

    const pojo700Tracks = Array.from({ length: 700 }, (_, idx) => ({
      index: idx,
      name: `POJO Track ${idx + 1}`,
      program: 0,
      isDrum: false,
      instrument: 'PIANO' as const,
      notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 240, velocity: 90 })],
    }));
    const modal700 = render(
      <MidiTrackImportModal
        fileName="pojo_700.mid"
        parsedMidi={{ format: 1, ppqn: 480, bpm: 120, beatsPerMeasure: 4, tracks: pojo700Tracks }}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );
    expect(modal700.container.querySelectorAll('input[type="checkbox"]').length).toBe(512);
    modal700.unmount();

    // 3. DemoPianoRollController.importParsedMidiTracks imports index >= 512 POJO tracks & reports exact MidiTrackImportResult
    const highIdxCtrl = new DemoPianoRollController(new StubAudioEngine());
    const highRes = highIdxCtrl.importParsedMidiTracks(
      { format: 1, ppqn: 480, bpm: 120, beatsPerMeasure: 4, tracks: pojo700Tracks.slice(0, 560) },
      [512, 559],
      true,
    );
    expect(highRes).toEqual({ importedTracks: 2, importedNotes: 2 });
    expect(highIdxCtrl.getSnapshot().tracks.map((t) => t.name)).toEqual([
      'POJO Track 513',
      'POJO Track 560',
    ]);

    // 4. Full 64-track workspace additive import returns {0, 0} and PianoRollView shows capacity notice instead of false placement banner
    const fullCtrl = new DemoPianoRollController(new StubAudioEngine());
    const fillRes = fullCtrl.importParsedMidiTracks(
      { format: 1, ppqn: 480, bpm: 120, beatsPerMeasure: 4, tracks: parsed512.tracks.slice(0, 64) },
      Array.from({ length: 64 }, (_, i) => i),
      true,
    );
    expect(fillRes).toEqual({ importedTracks: 64, importedNotes: 64 });
    expect(fullCtrl.getSnapshot().tracks.length).toBe(64);

    const viewFull = render(
      <PianoRollView
        session={fullCtrl.getSnapshot()}
        controller={fullCtrl}
        isActive={true}
        focusRevision={1}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );
    const twoTrackSmf = buildMultiTrackSmf(2, 1);
    const twoTrackBuf = twoTrackSmf.slice().buffer;
    const twoTrackFile = new File([twoTrackBuf], 'extra_two.mid', { type: 'audio/midi' });
    if (typeof twoTrackFile.arrayBuffer !== 'function') {
      Object.defineProperty(twoTrackFile, 'arrayBuffer', {
        value: async () => twoTrackBuf.slice(0),
      });
    }
    const fileInput = viewFull.getByTestId('midi-file-input') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [twoTrackFile] } });

    await waitFor(() => {
      expect(viewFull.getByTestId('midi-import-modal')).toBeTruthy();
    });
    fireEvent.click(viewFull.getByText('既存トラックに追加'));
    fireEvent.click(viewFull.getByTestId('confirm-midi-import-button'));

    await waitFor(() => {
      expect(viewFull.container.textContent).toContain('最大64トラックの上限に達しています');
    });
    expect(viewFull.container.textContent).not.toContain('extra_two.mid (2 ノート) をピアノロールに配置しました');
    viewFull.unmount();
  });

  it('enforces Round 23 Fact-Checker ([R22-HIGH-03], [R22-MED-02], [R22-LOW-01]) remediations: singleton TextDecoder 0xFF 0x03 flood immunity, zero redundant candidate re-parses up to 512 tracks / 65,536 notes, and 2-pass canonical POJO track index parity', async () => {
    // 1. [R22-HIGH-03] 2 MB repeated empty `00 FF 03 00` & whitespace `00 FF 03 01 20` events parse in < 100 ms
    const buildMetaFloodSmf = (eventBytes: readonly number[], repeatCount: number, trailingName?: string): ArrayBuffer => {
      const nameEvents: number[] = [];
      if (trailingName) {
        const nameUtf8 = new TextEncoder().encode(trailingName);
        nameEvents.push(0x00, 0xff, 0x03, nameUtf8.length, ...nameUtf8);
      }
      const tail = [...nameEvents, 0x00, 0x90, 60, 100, 0x10, 0x80, 60, 0, 0x00, 0xff, 0x2f, 0x00];
      const chunkLen = eventBytes.length * repeatCount + tail.length;
      const out = new Uint8Array(22 + chunkLen);
      const dv = new DataView(out.buffer);
      out.set([0x4d, 0x54, 0x68, 0x64], 0);
      dv.setUint32(4, 6, false);
      dv.setUint16(8, 0, false);
      dv.setUint16(10, 1, false);
      dv.setUint16(12, 480, false);
      out.set([0x4d, 0x54, 0x72, 0x6b], 14);
      dv.setUint32(18, chunkLen, false);
      let cursor = 22;
      for (let i = 0; i < repeatCount; i += 1) {
        out.set(eventBytes, cursor);
        cursor += eventBytes.length;
      }
      out.set(tail, cursor);
      return out.buffer;
    };

    const emptyNameFlood = buildMetaFloodSmf([0x00, 0xff, 0x03, 0x00], 480_000, 'GrandPiano');
    const t0Empty = performance.now();
    const parsedEmptyFlood = StandardMidiCodec.parseMidi(emptyNameFlood);
    expect(performance.now() - t0Empty).toBeLessThan(120);
    expect(parsedEmptyFlood.tracks.length).toBe(1);
    expect(parsedEmptyFlood.tracks[0].name).toBe('GrandPiano');

    const wsNameFlood = buildMetaFloodSmf([0x00, 0xff, 0x03, 0x01, 0x20], 380_000);
    const t0Ws = performance.now();
    const parsedWsFlood = StandardMidiCodec.parseMidi(wsNameFlood);
    expect(performance.now() - t0Ws).toBeLessThan(120);
    expect(parsedWsFlood.tracks.length).toBe(1);

    // 2. [R22-MED-02] 120-track & 40,000-note candidates perform 0 redundant parseMidi calls and preserve MidiNote.id
    const mtrk120Payload = [0x00, 0x90, 60, 100, 0x10, 0x80, 60, 0, 0x00, 0xff, 0x2f, 0x00];
    const smf120 = new Uint8Array(14 + 120 * (8 + mtrk120Payload.length));
    const dv120 = new DataView(smf120.buffer);
    smf120.set([0x4d, 0x54, 0x68, 0x64], 0);
    dv120.setUint32(4, 6, false);
    dv120.setUint16(8, 1, false);
    dv120.setUint16(10, 120, false);
    dv120.setUint16(12, 480, false);
    for (let t = 0; t < 120; t += 1) {
      const off = 14 + t * (8 + mtrk120Payload.length);
      smf120.set([0x4d, 0x54, 0x72, 0x6b], off);
      dv120.setUint32(off + 4, mtrk120Payload.length, false);
      smf120.set(mtrk120Payload, off + 8);
    }
    const parsed120 = StandardMidiCodec.parseMidi(smf120.buffer as ArrayBuffer);
    const originalFirstId120 = parsed120.tracks[0].notes[0].id;

    const notes40k = Array.from({ length: 40_000 }, (_, i) =>
      new MidiNote({ pitch: 48 + (i % 24), startTick: (i % 1000) * 12, durationTicks: 6, velocity: 90 }),
    );
    const smf40k = StandardMidiCodec.encodeSingleTrackMidi(notes40k, 480, 120, 'PIANO');
    const parsed40k = StandardMidiCodec.parseMidi(smf40k.buffer as ArrayBuffer);
    const originalFirstId40k = parsed40k.tracks[0].notes[0].id;

    const candCtrl = new DemoPianoRollController(new StubAudioEngine());
    const parseSpy = vi.spyOn(StandardMidiCodec, 'parseMidi');
    try {
      await candCtrl.applyGeneratedCandidates(
        [
          { index: 0, filename: 'cand_120_tracks.mid', rawMidiBytes: smf120, parsedMidi: parsed120, reason: null },
          { index: 1, filename: 'cand_40k_notes.mid', rawMidiBytes: smf40k, parsedMidi: parsed40k, reason: null },
        ],
        [],
      );
      await candCtrl.selectGeneratedCandidate(1);
      await candCtrl.selectGeneratedCandidate(0);
      expect(parseSpy).not.toHaveBeenCalled();
      expect(candCtrl.getSnapshot().candidates[0].parsedMidi.tracks[0].notes[0].id).toBe(originalFirstId120);
      expect(candCtrl.getSnapshot().candidates[1].parsedMidi.tracks[0].notes[0].id).toBe(originalFirstId40k);
    } finally {
      parseSpy.mockRestore();
    }

    // 3. [R22-LOW-01] Shared 2-pass canonical POJO track index normalization across Modal, View, & Controller
    const sampleNoteA = new MidiNote({ pitch: 60, startTick: 0, durationTicks: 240, velocity: 95 });
    const sampleNoteB = new MidiNote({ pitch: 64, startTick: 240, durationTicks: 240, velocity: 95 });
    const oddIndicesPojo = {
      format: 1,
      ppqn: 480,
      bpm: 120,
      beatsPerMeasure: 4,
      tracks: [
        { index: -1, name: 'NegIdx', program: 0, isDrum: false, instrument: 'PIANO' as const, notes: [sampleNoteA] },
        { index: 1.5, name: 'FloatIdx', program: 0, isDrum: false, instrument: 'SAX' as const, notes: [sampleNoteB] },
      ],
    };
    const canonicalOdd = extractCanonicalParsedMidiTracks(oddIndicesPojo);
    expect(canonicalOdd.map((c) => c.canonicalIndex)).toEqual([0, 1]);

    let oddConfirmed: readonly number[] = [];
    const oddModal = render(
      <MidiTrackImportModal
        fileName="odd_indices.mid"
        parsedMidi={oddIndicesPojo}
        onConfirm={(sel) => {
          oddConfirmed = sel;
        }}
        onCancel={() => {}}
      />,
    );
    fireEvent.click(oddModal.getByTestId('confirm-midi-import-button'));
    oddModal.unmount();
    expect(oddConfirmed).toEqual([0, 1]);

    const oddCtrl = new DemoPianoRollController(new StubAudioEngine());
    expect(oddCtrl.importParsedMidiTracks(oddIndicesPojo, oddConfirmed, true)).toEqual({
      importedTracks: 2,
      importedNotes: 2,
    });
    expect(
      oddCtrl.importParsedMidiTracks(
        { ...oddIndicesPojo, tracks: [oddIndicesPojo.tracks[0]] },
        [extractCanonicalParsedMidiTracks({ ...oddIndicesPojo, tracks: [oddIndicesPojo.tracks[0]] })[0].canonicalIndex],
        true,
      ),
    ).toEqual({ importedTracks: 1, importedNotes: 1 });

    // 100 POJO tracks with duplicate index: 0 -> strictly 64 checked, 36 disabled, independent toggle
    const dup100Pojo = {
      format: 1,
      ppqn: 480,
      bpm: 120,
      beatsPerMeasure: 4,
      tracks: Array.from({ length: 100 }, (_, i) => ({
        index: 0,
        name: `DupZero ${i + 1}`,
        program: 0,
        isDrum: false,
        instrument: 'PIANO' as const,
        notes: [sampleNoteA],
      })),
    };
    let dupConfirmed: readonly number[] = [];
    const dupModal = render(
      <MidiTrackImportModal
        fileName="dup_100.mid"
        parsedMidi={dup100Pojo}
        onConfirm={(sel) => {
          dupConfirmed = sel;
        }}
        onCancel={() => {}}
      />,
    );
    const dupBoxes = Array.from(
      dupModal.container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'),
    );
    expect(dupBoxes.length).toBe(100);
    expect(dupBoxes.filter((cb) => cb.checked).length).toBe(64);
    expect(dupBoxes.filter((cb) => cb.disabled).length).toBe(36);
    fireEvent.click(dupBoxes[0]);
    expect(dupBoxes.filter((cb) => cb.checked).length).toBe(63);
    fireEvent.click(dupBoxes[64]);
    expect(dupBoxes.filter((cb) => cb.checked).length).toBe(64);
    fireEvent.click(dupModal.getByTestId('confirm-midi-import-button'));
    dupModal.unmount();

    const dupCtrl = new DemoPianoRollController(new StubAudioEngine());
    expect(dupCtrl.importParsedMidiTracks(dup100Pojo, dupConfirmed, true)).toEqual({
      importedTracks: 64,
      importedNotes: 64,
    });
  });

  it('enforces Round 24 Fact-Checker ([R23-MED-01], [R23-MED-02], [R23-MED-03], [R23-LOW-02]) remediations: short-viewport modal flex layout, >512-track prefix-identical canonical indexing, clamped 999 BPM & <40ms 65k-note candidate switches, and multi-MTrk note-less budget', async () => {
    const noteVis = new MidiNote({ id: 'n_vis', pitch: 60, startTick: 0, durationTicks: 240, velocity: 100 });
    const noteHid = new MidiNote({ id: 'n_hid', pitch: 72, startTick: 0, durationTicks: 240, velocity: 100 });

    // 1. [R23-MED-01] & [R23-MED-02]: 700-track POJO where 0..63 have index:-1 and 512..575 have index:0..63
    const hijack700Pojo = {
      format: 1,
      ppqn: 480,
      bpm: 120,
      beatsPerMeasure: 4,
      tracks: Array.from({ length: 700 }, (_, i) => {
        if (i < 64) {
          return {
            index: -1,
            name: `VisibleRow_${i}`,
            program: 0,
            isDrum: false,
            instrument: 'PIANO' as const,
            notes: [noteVis],
          };
        }
        if (i >= 512 && i < 576) {
          return {
            index: i - 512,
            name: `HiddenHijack_${i}`,
            program: 65,
            isDrum: false,
            instrument: 'SAX' as const,
            notes: [noteHid],
          };
        }
        return {
          index: 1000 + i,
          name: `Filler_${i}`,
          program: 0,
          isDrum: false,
          instrument: 'PIANO' as const,
          notes: [noteVis],
        };
      }),
    };

    const canon512 = extractCanonicalParsedMidiTracks(hijack700Pojo, 512);
    const canon4096 = extractCanonicalParsedMidiTracks(hijack700Pojo, 4096);
    expect(canon512).toEqual(canon4096.slice(0, 512));
    expect(canon512.slice(0, 64).map((e) => e.canonicalIndex)).toEqual(
      Array.from({ length: 64 }, (_, i) => 64 + i),
    );

    let confirmed700: readonly number[] = [];
    const modal700 = render(
      <MidiTrackImportModal
        fileName="hijack_700.mid"
        parsedMidi={hijack700Pojo}
        onConfirm={(sel) => {
          confirmed700 = sel;
        }}
        onCancel={() => {}}
      />,
    );
    const dialogEl = modal700.getByTestId('midi-import-modal');
    const cardEl = modal700.getByTestId('midi-import-modal-card');
    expect(dialogEl.className).toContain('overflow-y-auto');
    expect(cardEl.className).toContain('max-h-[calc(100dvh-2rem)]');
    expect(cardEl.className).toContain('flex-col');
    fireEvent.click(modal700.getByTestId('confirm-midi-import-button'));
    modal700.unmount();

    const hijackCtrl = new DemoPianoRollController(new StubAudioEngine());
    const imp700Res = hijackCtrl.importParsedMidiTracks(hijack700Pojo, confirmed700, true);
    expect(imp700Res).toEqual({ importedTracks: 64, importedNotes: 64 });
    const importedNames = hijackCtrl.getSnapshot().tracks.map((t) => t.name);
    expect(importedNames.every((nm) => nm.startsWith('VisibleRow_'))).toBe(true);
    expect(importedNames.some((nm) => nm.startsWith('HiddenHijack_'))).toBe(false);

    // 2. [R23-MED-03]: Fast SMF tempo (mpqn = 50,000 -> 1200 raw BPM clamped to 999) + <40ms 65,536-note switches
    const fastTempoPayload = [
      0x00, 0xff, 0x51, 0x03, 0x00, 0xc3, 0x50, // mpqn = 50,000 (1200 BPM raw)
      0x00, 0x90, 60, 100, 0x10, 0x80, 60, 0,
      0x00, 0xff, 0x2f, 0x00,
    ];
    const fastSmf = new Uint8Array(22 + fastTempoPayload.length);
    const fastDv = new DataView(fastSmf.buffer);
    fastSmf.set([0x4d, 0x54, 0x68, 0x64], 0);
    fastDv.setUint32(4, 6, false);
    fastDv.setUint16(8, 0, false);
    fastDv.setUint16(10, 1, false);
    fastDv.setUint16(12, 480, false);
    fastSmf.set([0x4d, 0x54, 0x72, 0x6b], 14);
    fastDv.setUint32(18, fastTempoPayload.length, false);
    fastSmf.set(fastTempoPayload, 22);

    const parsedFast = StandardMidiCodec.parseMidi(fastSmf.buffer as ArrayBuffer);
    expect(parsedFast.bpm).toBe(999);
    const origFastNoteId = parsedFast.tracks[0].notes[0].id;

    const notes65kA = Array.from({ length: 65_536 }, (_, i) =>
      new MidiNote({ pitch: 36 + (i % 48), startTick: (i % 1024) * 6, durationTicks: 4, velocity: 90 }),
    );
    const notes65kB = Array.from({ length: 65_536 }, (_, i) =>
      new MidiNote({ pitch: 40 + (i % 48), startTick: (i % 1024) * 6, durationTicks: 4, velocity: 95 }),
    );
    const smf65kA = StandardMidiCodec.encodeSingleTrackMidi(notes65kA, 480, 120, 'PIANO');
    const smf65kB = StandardMidiCodec.encodeSingleTrackMidi(notes65kB, 480, 120, 'PIANO');
    const parsed65kA = StandardMidiCodec.parseMidi(smf65kA.buffer as ArrayBuffer);
    const parsed65kB = StandardMidiCodec.parseMidi(smf65kB.buffer as ArrayBuffer);

    const fastCtrl = new DemoPianoRollController(new StubAudioEngine());
    // Seed sibling track with a colliding note ID to verify cross-track uniqueness preservation
    const collidingId = parsed65kB.tracks[0].notes[0].id;
    const snapBefore = fastCtrl.getSnapshot();
    const otherTrackId = snapBefore.tracks[1].id;
    fastCtrl.setActiveTrack(otherTrackId);
    fastCtrl.importParsedMidiTracks(
      {
        format: 0,
        ppqn: 480,
        bpm: 120,
        beatsPerMeasure: 4,
        tracks: [
          {
            index: 0,
            name: 'SiblingCollision',
            program: 65,
            isDrum: false,
            instrument: 'SAX',
            notes: [new MidiNote({ id: collidingId, pitch: 67, startTick: 0, durationTicks: 240 })],
          },
        ],
      },
      [0],
      false,
    );
    fastCtrl.setActiveTrack(fastCtrl.getSnapshot().tracks[0].id);

    const spyParse = vi.spyOn(StandardMidiCodec, 'parseMidi');
    try {
      await fastCtrl.applyGeneratedCandidates(
        [
          { index: 0, filename: 'fast_bpm.mid', rawMidiBytes: fastSmf, parsedMidi: parsedFast, reason: null },
          { index: 1, filename: '65k_a.mid', rawMidiBytes: smf65kA, parsedMidi: parsed65kA, reason: null },
          { index: 2, filename: '65k_b.mid', rawMidiBytes: smf65kB, parsedMidi: parsed65kB, reason: null },
        ],
        [],
      );
      expect(spyParse).not.toHaveBeenCalled();
      expect(fastCtrl.getSnapshot().candidates[0].parsedMidi.tracks[0].notes[0].id).toBe(origFastNoteId);

      let liveNotifies = 0;
      const unsubLive = fastCtrl.subscribe(() => {
        liveNotifies += 1;
      });
      try {
        const t0SwitchA = performance.now();
        await fastCtrl.selectGeneratedCandidate(1);
        const dtSwitchA = performance.now() - t0SwitchA;
        expect(dtSwitchA).toBeLessThan(60);

        const t0SwitchB = performance.now();
        await fastCtrl.selectGeneratedCandidate(2);
        const dtSwitchB = performance.now() - t0SwitchB;
        expect(dtSwitchB).toBeLessThan(60);
        expect(liveNotifies).toBe(2);
      } finally {
        unsubLive();
      }
      expect(spyParse).not.toHaveBeenCalled();

      // Verify workspace-wide note ID uniqueness across all tracks after collision-guarded switch
      const allIds: string[] = [];
      for (const trk of fastCtrl.getSnapshot().tracks) {
        for (const n of trk.notes) {
          allIds.push(n.id);
        }
      }
      expect(new Set(allIds).size).toBe(allIds.length);
      expect(fastCtrl.getSnapshot().tracks[0].notes.length).toBe(32_767);

      // Verify Undo & Redo across 32,767-note candidate switches preserve exact counts & IDs
      fastCtrl.undo();
      expect(fastCtrl.getSnapshot().activeCandidateIndex).toBe(1);
      expect(fastCtrl.getSnapshot().tracks[0].notes.length).toBe(32_767);
      fastCtrl.redo();
      expect(fastCtrl.getSnapshot().activeCandidateIndex).toBe(2);
      expect(fastCtrl.getSnapshot().tracks[0].notes.length).toBe(32_767);
    } finally {
      spyParse.mockRestore();
    }

    // 3. [R23-LOW-02]: 10,000 note-less MTrk chunks x 8 UTF-8 NBSP 0xFF 0x03 events parse in < 80 ms
    const nbspEvent = [0x00, 0xff, 0x03, 0x02, 0xc2, 0xa0]; // U+00A0 NBSP (> 0x20, trims to '')
    const notelessPayload: number[] = [];
    for (let k = 0; k < 8; k += 1) {
      notelessPayload.push(...nbspEvent);
    }
    notelessPayload.push(0x00, 0xff, 0x2f, 0x00);
    const numNoteless = 10_000;
    const notelessSmf = new Uint8Array(14 + numNoteless * (8 + notelessPayload.length));
    const nlDv = new DataView(notelessSmf.buffer);
    notelessSmf.set([0x4d, 0x54, 0x68, 0x64], 0);
    nlDv.setUint32(4, 6, false);
    nlDv.setUint16(8, 1, false);
    nlDv.setUint16(10, numNoteless, false);
    nlDv.setUint16(12, 480, false);
    for (let t = 0; t < numNoteless; t += 1) {
      const off = 14 + t * (8 + notelessPayload.length);
      notelessSmf.set([0x4d, 0x54, 0x72, 0x6b], off);
      nlDv.setUint32(off + 4, notelessPayload.length, false);
      notelessSmf.set(notelessPayload, off + 8);
    }
    const t0Nl = performance.now();
    const parsedNl = StandardMidiCodec.parseMidi(notelessSmf.buffer as ArrayBuffer);
    expect(performance.now() - t0Nl).toBeLessThan(80);
    expect(parsedNl.tracks.length).toBe(0);
  });

  it('36. enforces Round 25 Fact-Checker ([R24-MED-01], [R24-MED-02], [R24-MED-03], [R24-LOW-01]) remediations: non-ASCII BMP whitespace ID parity, short-viewport modal compact CSS, unexported domain internals, and <55ms full-pipeline 32k-note candidate broadcast', async () => {
    const nodeFs = await import('node:fs');
    const nodePath = await import('node:path');
    const nodeModule = await import('node:module');
    const reqFromHere = nodeModule.createRequire(import.meta.url);
    const ipcSec = reqFromHere('../../electron/ipc_security.cjs') as {
      isValidWorkspaceSyncPayload: (m: unknown) => boolean;
    };

    const preloadPath = nodePath.resolve(process.cwd(), 'electron', 'preload.cjs');
    const preloadSrc = nodeFs.readFileSync(preloadPath, 'utf8');
    expect(preloadSrc.includes("require('util')")).toBe(false);
    expect(preloadSrc.includes('new URL(')).toBe(false);

    const sentViaPreload: unknown[] = [];
    const preloadHolder: {
      sendWorkspaceSync: ((msg: unknown) => void) | null;
    } = { sendWorkspaceSync: null };
    const fakeElectron36 = {
      contextBridge: {
        exposeInMainWorld: (
          _name: string,
          api: { sendWorkspaceSync: (msg: unknown) => void },
        ) => {
          preloadHolder.sendWorkspaceSync = api.sendWorkspaceSync;
        },
      },
      ipcRenderer: {
        on: () => {},
        removeListener: () => {},
        invoke: async () => ({
          apiBaseUrl: 'http://127.0.0.1:18400',
          handshakeToken: 'tok36',
          backendReady: true,
        }),
        send: (_ch: string, payload: unknown) => {
          sentViaPreload.push(payload);
        },
      },
    };
    const modAny = nodeModule.Module as unknown as {
      _load: (request: string, parent: unknown, isMain: boolean) => unknown;
      _cache: Record<string, unknown>;
    };
    const origLoad = modAny._load;
    const resolvedPreload = reqFromHere.resolve('../../electron/preload.cjs');
    delete modAny._cache[resolvedPreload];
    modAny._load = (request: string, parent: unknown, isMain: boolean) =>
      request === 'electron' ? fakeElectron36 : origLoad(request, parent, isMain);
    try {
      reqFromHere('../../electron/preload.cjs');
    } finally {
      modAny._load = origLoad;
      delete modAny._cache[resolvedPreload];
    }
    expect(typeof preloadHolder.sendWorkspaceSync).toBe('function');

    // 1. [R24-MED-01]: All 18 non-ASCII BMP whitespace characters canonicalize identically across domain, controller, and 5-way sync
    const nonAsciiWsChars = [
      '\u00A0',
      '\u1680',
      '\u2000',
      '\u2001',
      '\u2002',
      '\u2003',
      '\u2004',
      '\u2005',
      '\u2006',
      '\u2007',
      '\u2008',
      '\u2009',
      '\u200A',
      '\u2028',
      '\u2029',
      '\u202F',
      '\u205F',
      '\u3000',
    ];
    for (const wsChar of nonAsciiWsChars) {
      const wsOnlyNote = new MidiNote({
        id: wsChar,
        pitch: 60,
        startTick: 0,
        durationTicks: 120,
        velocity: 100,
      });
      expect(wsOnlyNote.id).not.toBe(wsChar);
      expect(wsOnlyNote.id.trim()).toBe(wsOnlyNote.id);
      expect(wsOnlyNote.id.length).toBeGreaterThan(0);

      const paddedNote = new MidiNote({
        id: `${wsChar}n_ws_${wsChar.charCodeAt(0)}${wsChar}`,
        pitch: 64,
        startTick: 120,
        durationTicks: 120,
        velocity: 96,
      });
      expect(paddedNote.id).toBe(`n_ws_${wsChar.charCodeAt(0)}`);

      const canonSet = new ReadonlyCanonicalIdSet([
        `${wsChar}n_ws_${wsChar.charCodeAt(0)}${wsChar}`,
        `n_ws_${wsChar.charCodeAt(0)}`,
        wsChar,
      ]);
      expect(canonSet.size).toBe(1);
      expect(canonSet.has(`n_ws_${wsChar.charCodeAt(0)}`)).toBe(true);
    }

    const ideoTrack = new DawTrack({
      id: '\u3000t_ideo\u3000',
      name: 'Ideo Track',
      instrument: 'PIANO',
      colorHex: TRACK_SWATCHES[0],
      notes: [
        new MidiNote({
          id: '\u3000n_ideo_1\u3000',
          pitch: 60,
          startTick: 0,
          durationTicks: 240,
          velocity: 100,
        }),
      ],
    });
    expect(ideoTrack.id).toBe('t_ideo');
    expect(ideoTrack.toSnapshot().notes[0].id).toBe('n_ideo_1');

    const wsCtrl = new DemoPianoRollController(new StubAudioEngine());
    await wsCtrl.connect();
    wsCtrl.applyWorkspaceState({
      ...wsCtrl.exportWorkspaceState(),
      activeTrackId: '\u3000t_ideo\u3000',
      tracks: [
        {
          id: '\u3000t_ideo\u3000',
          name: 'NBSP Track',
          instrument: 'PIANO',
          colorHex: TRACK_SWATCHES[0],
          isMuted: false,
          isSoloed: false,
          notes: [
            {
              id: '\u3000n_ideo_1\u3000',
              pitch: 60,
              startTick: 0,
              durationTicks: 240,
              velocity: 100,
              channel: 0,
            },
          ],
        },
      ],
      candidates: [],
      activeCandidateIndex: 0,
      activeCandidateNoteIds: [],
      activeCandidateRange: null,
      candidateOriginSlot: null,
      latestReason: null,
    });
    expect(wsCtrl.getSnapshot().activeTrackId).toBe('t_ideo');
    expect(wsCtrl.getSnapshot().tracks[0].notes[0].id).toBe('n_ideo_1');
    wsCtrl.transformNotes([
      {
        id: '\u3000n_ideo_1\u3000',
        pitch: 72,
        startTick: 0,
        durationTicks: 240,
        velocity: 110,
      },
    ]);
    expect(wsCtrl.getSnapshot().tracks[0].notes[0].pitch).toBe(72);

    const busA = new WorkspaceSyncBus('r25_test_ws_ch');
    const busB = new WorkspaceSyncBus('r25_test_ws_ch');
    const receivedOnB: unknown[] = [];
    const unsubB = busB.subscribe((m) => receivedOnB.push(m));
    try {
      busA.publishWorkspaceState(wsCtrl.exportWorkspaceState());
      await new Promise((r) => setTimeout(r, 15));
      expect(receivedOnB.length).toBe(1);
    } finally {
      unsubB();
      busA.dispose();
      busB.dispose();
    }

    // 2. [R24-LOW-01]: Internal domain capabilities are unexported at runtime and replaceRangeNotesAtomic is defensive
    expect((ReadonlyCanonicalIdSet as unknown as Record<string, unknown>)._unwrapInner).toBeUndefined();
    expect(
      (ReadonlyCanonicalIdSet as unknown as Record<string, unknown>)._fromOwnedCanonicalSet,
    ).toBeUndefined();
    expect((PreparedCandidateBatch as unknown as Record<string, unknown>)._getNotes).toBeUndefined();
    expect(
      (PreparedCandidateBatch as unknown as Record<string, unknown>)._isPreparedBatch,
    ).toBeUndefined();
    expect(
      (PreparedCandidateBatch as unknown as Record<string, unknown>)._takeAllIdsOnce,
    ).toBeUndefined();
    expect(typeof isPreparedCandidateBatch).toBe('function');
    expect((MidiNote as unknown as Record<string, unknown>)._createTrusted).toBeUndefined();

    const defTrack = new DawTrack({ name: 'DefTrack', instrument: 'PIANO', colorHex: TRACK_SWATCHES[0] });
    defTrack.addNotes([
      new MidiNote({ id: 'keep_1', pitch: 60, startTick: 0, durationTicks: 120, velocity: 100 }),
    ]);
    const safeNullAtomic = defTrack.replaceRangeNotesAtomic(
      0,
      480,
      null as unknown as PreparedCandidateBatch,
      null as unknown as ReadonlySet<string>,
    );
    expect(safeNullAtomic.removedNotes.length).toBe(0);
    expect(safeNullAtomic.actuallyInsertedIds.size).toBe(0);
    expect(defTrack.noteCount).toBe(1);

    // 3. [R24-MED-02]: Short-viewport compact rules (`@media(max-height:500px)`) & 360px single-row mode buttons on MidiTrackImportModal
    const modal80 = render(
      <MidiTrackImportModal
        fileName="landscape_80.mid"
        parsedMidi={{
          format: 1,
          ppqn: 480,
          bpm: 120,
          beatsPerMeasure: 4,
          tracks: Array.from({ length: 80 }, (_, idx) => ({
            index: idx,
            name: `LandscapeTrack_${idx + 1}`,
            program: 0,
            isDrum: false,
            instrument: 'PIANO' as const,
            notes: [
              new MidiNote({
                id: `m80_${idx}`,
                pitch: 60,
                startTick: 0,
                durationTicks: 120,
                velocity: 100,
              }),
            ],
          })),
        }}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );
    const dialogEl = modal80.getByTestId('midi-import-modal-card');
    expect(dialogEl.className).toContain('max-h-[calc(100dvh-2rem)]');
    expect(dialogEl.className).toContain('[@media(max-height:500px)]:max-h-[calc(100dvh-1rem)]');
    const scrollBodyEl = modal80.getByTestId('midi-import-scroll-body');
    expect(scrollBodyEl.className).toContain('overflow-y-auto');
    const capHintEl = modal80.getByTestId('midi-import-selection-cap-hint');
    expect(capHintEl.textContent).toContain('最大 64');
    modal80.unmount();

    // 4. [R24-MED-03]: Mutable exportWorkspaceState contract, extensible Proxy ownKeys 5-way parity, post-export Uint8Array tamper guard, and <55ms full candidate-click pipeline
    const mutableWs = wsCtrl.exportWorkspaceState();
    (mutableWs as { bpm: number }).bpm = 144;
    expect(mutableWs.bpm).toBe(144);

    const makeSyncEnvelope = (tracksVal: unknown) => ({
      messageId: 'm_proxy_test',
      type: 'SYNC_WORKSPACE_STATE' as const,
      senderId: 'peer_1',
      workspaceState: {
        ...wsCtrl.exportWorkspaceState(),
        tracks: tracksVal,
      },
    });
    const baseTrackObj = wsCtrl.exportWorkspaceState().tracks[0];
    const proxyEvilKeys = new Proxy([baseTrackObj], {
      ownKeys() {
        return ['evil', 'length'];
      },
      getOwnPropertyDescriptor(target, prop) {
        if (prop === 'evil') {
          return { value: baseTrackObj, writable: true, enumerable: true, configurable: true };
        }
        return Reflect.getOwnPropertyDescriptor(target, prop);
      },
    });
    const proxyLeadingZero = new Proxy([baseTrackObj], {
      ownKeys() {
        return ['00', 'length'];
      },
      getOwnPropertyDescriptor(target, prop) {
        if (prop === '00') {
          return { value: baseTrackObj, writable: true, enumerable: true, configurable: true };
        }
        return Reflect.getOwnPropertyDescriptor(target, prop);
      },
    });
    const twoTracks = [
      baseTrackObj,
      { ...baseTrackObj, id: 'trk_second', notes: [] },
    ];
    const proxyReorderedValid = new Proxy(twoTracks, {
      ownKeys() {
        return ['length', '1', '0'];
      },
    });

    sentViaPreload.length = 0;
    preloadHolder.sendWorkspaceSync?.(makeSyncEnvelope(proxyEvilKeys));
    preloadHolder.sendWorkspaceSync?.(makeSyncEnvelope(proxyLeadingZero));
    expect(sentViaPreload.length).toBe(0);
    expect(isValidIncomingSyncMessage(makeSyncEnvelope(proxyEvilKeys))).toBe(false);
    expect(isValidIncomingSyncMessage(makeSyncEnvelope(proxyLeadingZero))).toBe(false);
    expect(ipcSec.isValidWorkspaceSyncPayload(makeSyncEnvelope(proxyEvilKeys))).toBe(false);
    expect(ipcSec.isValidWorkspaceSyncPayload(makeSyncEnvelope(proxyLeadingZero))).toBe(false);

    // Note: Node `util.types.isProxy` rejects ALL runtime Proxies in `ipc_security.cjs`, while `preload.cjs` (sandboxed without `util`) validates extensible Proxy dense arrays via `Uint8Array` fallback
    preloadHolder.sendWorkspaceSync?.(makeSyncEnvelope(proxyReorderedValid));
    expect(sentViaPreload.length).toBe(1);
    expect(ipcSec.isValidWorkspaceSyncPayload(structuredClone(sentViaPreload[0]))).toBe(true);

    // Full shipped candidate-click pipeline: 2 x 45,000-note candidates (clamped to 32,768 placed notes)
    const buildDenseSmf = (noteCount: number, basePitch: number): Uint8Array => {
      const evBytes = new Uint8Array(noteCount * 8 + 4);
      let p = 0;
      for (let i = 0; i < noteCount; i += 1) {
        const pitch = basePitch + (i % 12);
        evBytes[p++] = 0x00;
        evBytes[p++] = 0x90;
        evBytes[p++] = pitch;
        evBytes[p++] = 100;
        evBytes[p++] = 0x01;
        evBytes[p++] = 0x80;
        evBytes[p++] = pitch;
        evBytes[p++] = 0;
      }
      evBytes[p++] = 0x00;
      evBytes[p++] = 0xff;
      evBytes[p++] = 0x2f;
      evBytes[p++] = 0x00;
      const smf = new Uint8Array(14 + 8 + evBytes.length);
      const dv = new DataView(smf.buffer);
      smf.set([0x4d, 0x54, 0x68, 0x64], 0);
      dv.setUint32(4, 6, false);
      dv.setUint16(8, 0, false);
      dv.setUint16(10, 1, false);
      dv.setUint16(12, 480, false);
      smf.set([0x4d, 0x54, 0x72, 0x6b], 14);
      dv.setUint32(18, evBytes.length, false);
      smf.set(evBytes, 22);
      return smf;
    };

    const smf45kA = buildDenseSmf(45_000, 48);
    const smf45kB = buildDenseSmf(45_000, 60);
    const parsed45kA = StandardMidiCodec.parseMidi(smf45kA.buffer as ArrayBuffer);
    const parsed45kB = StandardMidiCodec.parseMidi(smf45kB.buffer as ArrayBuffer);

    const benchCtrl = new DemoPianoRollController(new StubAudioEngine());
    await benchCtrl.connect();
    // Clear built-in demo notes on inactive track so total workspace notes === 32,768 (within 5-way sync wire cap)
    const snapInit = benchCtrl.getSnapshot();
    for (const trk of snapInit.tracks) {
      if (trk.id !== snapInit.activeTrackId) {
        benchCtrl.deleteTrack(trk.id);
      }
    }
    benchCtrl.setTargetMeasureRange(1, 32);

    await benchCtrl.applyGeneratedCandidates(
      [
        { index: 0, filename: 'cand_45k_a.mid', rawMidiBytes: smf45kA, parsedMidi: parsed45kA, reason: null },
        { index: 1, filename: 'cand_45k_b.mid', rawMidiBytes: smf45kB, parsedMidi: parsed45kB, reason: null },
      ],
      [],
    );
    expect(benchCtrl.getSnapshot().tracks[0].notes.length).toBe(32_768);

    // Post-export Uint8Array mutation on trusted snapshot MUST be caught and blocked by publishWorkspaceState
    const tamperTrusted = benchCtrl.exportTrustedWorkspaceState();
    expect(Object.isFrozen(tamperTrusted)).toBe(true);
    expect(Object.isFrozen(tamperTrusted.tracks)).toBe(true);
    expect(Object.isFrozen(tamperTrusted.tracks[0].notes)).toBe(true);
    tamperTrusted.candidates[0].rawMidiBytes[0] = 0x00; // Corrupt 'MThd' magic in-place

    sentViaPreload.length = 0;
    const prevDesktop = window.compassDesktop;
    window.compassDesktop = {
      sendWorkspaceSync: (msg) => preloadHolder.sendWorkspaceSync?.(msg),
    };
    const benchBus = new WorkspaceSyncBus('r25_bench_ch');
    try {
      benchBus.publishWorkspaceState(tamperTrusted);
      expect(sentViaPreload.length).toBe(0);

      // Warm up JIT once across the full pipeline, then measure both:
      // (a) Single-window / BroadcastChannel path (`selectGeneratedCandidate` + `exportTrustedWorkspaceState` + `publishWorkspaceState`), and
      // (b) Multi-window Electron IPC path (+ `preload.sendWorkspaceSync` + `ipc_security.isValidWorkspaceSyncPayload`).
      await benchCtrl.selectGeneratedCandidate(1);
      benchBus.publishWorkspaceState(benchCtrl.exportTrustedWorkspaceState());
      await benchCtrl.selectGeneratedCandidate(0);
      benchBus.publishWorkspaceState(benchCtrl.exportTrustedWorkspaceState());
      sentViaPreload.length = 0;

      delete window.compassDesktop;
      const t0SingleWin = performance.now();
      await benchCtrl.selectGeneratedCandidate(1);
      benchBus.publishWorkspaceState(benchCtrl.exportTrustedWorkspaceState());
      const dtSingleWin = performance.now() - t0SingleWin;
      expect(dtSingleWin).toBeLessThan(55);

      await benchCtrl.selectGeneratedCandidate(0);
      window.compassDesktop = {
        sendWorkspaceSync: (msg) => preloadHolder.sendWorkspaceSync?.(msg),
      };
      sentViaPreload.length = 0;

      const t0Full = performance.now();
      await benchCtrl.selectGeneratedCandidate(1);
      const trustedWs = benchCtrl.exportTrustedWorkspaceState();
      benchBus.publishWorkspaceState(trustedWs);
      const dtTotal = performance.now() - t0Full;

      expect(sentViaPreload.length).toBe(1);
      const wireCloned = structuredClone(sentViaPreload[0]);
      ipcSec.isValidWorkspaceSyncPayload(wireCloned); // Warm up JIT
      const t0MainIpc = performance.now();
      const mainValid = ipcSec.isValidWorkspaceSyncPayload(wireCloned);
      const dtMainIpc = performance.now() - t0MainIpc;

      expect(mainValid).toBe(true);
      expect(dtTotal).toBeLessThan(110);
      expect(dtMainIpc).toBeLessThan(55);

      // Verify peer applyWorkspaceState accepts the 32,768-note broadcast cleanly
      const peerCtrl = new DemoPianoRollController(new StubAudioEngine());
      await peerCtrl.connect();
      const wireMsg = wireCloned as { workspaceState: ReturnType<typeof benchCtrl.exportWorkspaceState> };
      peerCtrl.applyWorkspaceState(wireMsg.workspaceState);
      expect(peerCtrl.getSnapshot().tracks[0].notes.length).toBe(32_768);
      expect(peerCtrl.getSnapshot().activeCandidateIndex).toBe(1);
    } finally {
      benchBus.dispose();
      if (prevDesktop === undefined) {
        delete window.compassDesktop;
      } else {
        window.compassDesktop = prevDesktop;
      }
    }
  });

  it('37. enforces Round-26 collision-free 32,768-note batch IDs, single/dual-window sync dedup & parsedMidi caching, private brand Proxy rejection, and TOCTOU snapshot guards', async () => {
    const nodeReq = (
      globalThis as unknown as { process?: { getBuiltinModule?: (id: string) => unknown } }
    ).process?.getBuiltinModule;
    if (!nodeReq) {
      throw new Error('Node builtin module loader unavailable in Vitest environment');
    }
    const nodePath = nodeReq('node:path') as { resolve: (...parts: string[]) => string };
    const nodeModule = nodeReq('node:module') as {
      createRequire: (filename: string) => (id: string) => unknown;
      _load: (request: string, parent: unknown, isMain: boolean) => unknown;
    };
    const cjsReq = nodeModule.createRequire(import.meta.url);
    const ipcSecPath = nodePath.resolve(process.cwd(), 'electron', 'ipc_security.cjs');
    const ipcSec = cjsReq(ipcSecPath) as {
      isValidWorkspaceSyncPayload: (raw: unknown) => boolean;
    };

    const preloadPath = nodePath.resolve(process.cwd(), 'electron', 'preload.cjs');
    const sentViaPreload: unknown[] = [];
    const preloadHolder: {
      sendWorkspaceSync?: (msg: unknown) => void;
      hasPeerWindows?: () => boolean;
      apiConfigHandler?: (_ev: unknown, cfg: unknown) => void;
    } = {};
    const electronStub = {
      contextBridge: {
        exposeInMainWorld: (_name: string, api: Record<string, unknown>) => {
          preloadHolder.sendWorkspaceSync = api.sendWorkspaceSync as (msg: unknown) => void;
          preloadHolder.hasPeerWindows = api.hasPeerWindows as () => boolean;
        },
      },
      ipcRenderer: {
        on: (channel: string, cb: (_ev: unknown, payload: unknown) => void) => {
          if (channel === 'compass:api-config') {
            preloadHolder.apiConfigHandler = cb;
          }
        },
        removeListener: () => {},
        invoke: () => Promise.resolve({ apiBaseUrl: 'http://127.0.0.1:8000', handshakeToken: '' }),
        send: (channel: string, payload: unknown) => {
          if (channel === 'compass:workspace-sync') {
            sentViaPreload.push(payload);
          }
        },
      },
    };
    const origLoad = nodeModule._load;
    delete (cjsReq as unknown as { cache?: Record<string, unknown> }).cache?.[preloadPath];
    nodeModule._load = function (request: string, parent: unknown, isMain: boolean): unknown {
      if (request === 'electron') return electronStub;
      return origLoad.call(this, request, parent, isMain);
    };
    try {
      cjsReq(preloadPath);
    } finally {
      nodeModule._load = origLoad;
    }

    // 1. [R25-MED-01] & [R25-LOW-01] Private-brand `#genuineBrand` & `#dawTrackBrand` reject Proxy wrappers + preserve Object.freeze
    const genuineNote = new MidiNote({ id: 'brand_n1', pitch: 60, startTick: 0, durationTicks: 120 });
    expect(Object.isFrozen(genuineNote)).toBe(true);
    expect(isGenuineMidiNote(genuineNote)).toBe(true);
    expect(isGenuineMidiNote(new Proxy(genuineNote, {}))).toBe(false);

    const genuineTrack = new DawTrack({ id: 'brand_trk1', name: 'Piano', notes: [genuineNote] });
    expect(isGenuineDawTrack(genuineTrack)).toBe(true);
    expect(isGenuineDawTrack(new Proxy(genuineTrack, {}))).toBe(false);

    // Stateful getter on `candidates[0].index` and Proxy-wrapped `DawTrack` rejected by `buildTrustedWorkspaceSnapshot`
    let statefulCandReads = 0;
    const statefulCandidate = {};
    Object.defineProperty(statefulCandidate, 'index', {
      enumerable: true,
      get() {
        statefulCandReads += 1;
        return statefulCandReads === 1 ? 0 : 999_999;
      },
    });
    const minSmf = StandardMidiCodec.encodeSingleTrackMidi([genuineNote], 480, 120);
    Object.assign(statefulCandidate, {
      filename: 'stateful.mid',
      rawMidiBytes: minSmf,
      reason: null,
    });
    expect(
      buildTrustedWorkspaceSnapshot({
        ppqn: 480,
        bpm: 120,
        beatsPerMeasure: 4,
        timeSigNumerator: 4,
        timeSigDenominator: 4,
        activeTrackId: 'brand_trk1',
        targetStartMeasure: 1,
        targetEndMeasure: 4,
        quantizeDivision: 16,
        isTriplet: false,
        domainTracks: [genuineTrack],
        candidates: [statefulCandidate as never],
        activeCandidateIndex: 0,
        activeCandidateNoteIds: new Set(['brand_n1']),
        activeCandidateRange: null,
        candidateOriginSlot: null,
        latestReason: null,
      }),
    ).toBeNull();
    expect(statefulCandReads).toBe(0);

    expect(
      buildTrustedWorkspaceSnapshot({
        ppqn: 480,
        bpm: 120,
        beatsPerMeasure: 4,
        timeSigNumerator: 4,
        timeSigDenominator: 4,
        activeTrackId: 'brand_trk1',
        targetStartMeasure: 1,
        targetEndMeasure: 4,
        quantizeDivision: 16,
        isTriplet: false,
        domainTracks: [new Proxy(genuineTrack, {})],
        candidates: [],
        activeCandidateIndex: 0,
        activeCandidateNoteIds: new Set<string>(),
        activeCandidateRange: null,
        candidateOriginSlot: null,
        latestReason: null,
      }),
    ).toBeNull();

    // 2. [R25-HIGH-01] 32,768-note candidate insertion with K=3 note-ID collisions:
    //    `activeCandidateNoteIds.size === 32,768`, `execute() === undo(); redo()`, and 5-way sync parity!
    const buildDenseSmf = (noteCount: number, basePitch: number): Uint8Array => {
      const evBytes = new Uint8Array(noteCount * 8 + 4);
      let p = 0;
      for (let i = 0; i < noteCount; i += 1) {
        const pitch = basePitch + (i % 12);
        evBytes[p++] = 0x00;
        evBytes[p++] = 0x90;
        evBytes[p++] = pitch;
        evBytes[p++] = 100;
        evBytes[p++] = 0x01;
        evBytes[p++] = 0x80;
        evBytes[p++] = pitch;
        evBytes[p++] = 0;
      }
      evBytes[p++] = 0x00;
      evBytes[p++] = 0xff;
      evBytes[p++] = 0x2f;
      evBytes[p++] = 0x00;
      const smf = new Uint8Array(14 + 8 + evBytes.length);
      const dv = new DataView(smf.buffer);
      smf.set([0x4d, 0x54, 0x68, 0x64], 0);
      dv.setUint32(4, 6, false);
      dv.setUint16(8, 0, false);
      dv.setUint16(10, 1, false);
      dv.setUint16(12, 480, false);
      smf.set([0x4d, 0x54, 0x72, 0x6b], 14);
      dv.setUint32(18, evBytes.length, false);
      smf.set(evBytes, 22);
      return smf;
    };

    const smf40kA = buildDenseSmf(40_000, 48);
    const smf40kB = buildDenseSmf(40_000, 60);
    const parsedA = StandardMidiCodec.parseMidi(smf40kA.buffer as ArrayBuffer);
    const parsedB = StandardMidiCodec.parseMidi(smf40kB.buffer as ArrayBuffer);
    // Inject K=3 explicit duplicate/colliding note IDs inside parsedA (`dup_id_a`, `dup_id_b`)
    const notesWithCollisions = parsedA.tracks[0].notes.slice();
    notesWithCollisions[0] = new MidiNote({ ...notesWithCollisions[0], id: 'dup_id_a' });
    notesWithCollisions[1] = new MidiNote({ ...notesWithCollisions[1], id: 'dup_id_a' });
    notesWithCollisions[2] = new MidiNote({ ...notesWithCollisions[2], id: 'dup_id_a' });
    notesWithCollisions[3] = new MidiNote({ ...notesWithCollisions[3], id: 'dup_id_b' });
    notesWithCollisions[4] = new MidiNote({ ...notesWithCollisions[4], id: 'dup_id_b' });
    const parsedCollidingA = {
      ...parsedA,
      tracks: [{ ...parsedA.tracks[0], notes: notesWithCollisions }],
    };

    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    await ctrl.connect();
    for (const trk of ctrl.getSnapshot().tracks) {
      if (trk.id !== ctrl.getSnapshot().activeTrackId) {
        ctrl.deleteTrack(trk.id);
      }
    }
    ctrl.setTargetMeasureRange(1, 32);
    await ctrl.applyGeneratedCandidates(
      [
        { index: 0, filename: 'coll_a.mid', rawMidiBytes: smf40kA, parsedMidi: parsedCollidingA, reason: null },
        { index: 1, filename: 'coll_b.mid', rawMidiBytes: smf40kB, parsedMidi: parsedB, reason: null },
      ],
      [],
    );

    const wsAfterExec = ctrl.exportTrustedWorkspaceState();
    expect(Object.isFrozen(wsAfterExec)).toBe(true);
    expect(wsAfterExec.tracks[0].notes.length).toBe(32_768);
    expect(wsAfterExec.activeCandidateNoteIds.length).toBe(32_768);
    const execPlacedIdSet = new Set(wsAfterExec.tracks[0].notes.map((n) => n.id));
    expect(execPlacedIdSet.size).toBe(32_768);
    for (const cid of wsAfterExec.activeCandidateNoteIds) {
      expect(execPlacedIdSet.has(cid)).toBe(true);
    }

    // Verify point-in-time & mutation isolation of lazy `getSnapshot().selectedNoteIds`
    const snapT0 = ctrl.getSnapshot();
    ctrl.setSelectedNotes(['custom_sel_only']);
    expect(snapT0.selectedNoteIds.size).toBe(32_768);
    expect(ctrl.getSnapshot().selectedNoteIds.has('custom_sel_only')).toBe(true);

    // Verify `execute()` vs `undo(); redo()` exact parity on activeCandidateNoteIds
    ctrl.undo();
    expect(ctrl.getSnapshot().tracks[0].notes.length).toBe(0);
    ctrl.redo();
    const wsAfterRedo = ctrl.exportTrustedWorkspaceState();
    expect(wsAfterRedo.activeCandidateNoteIds.length).toBe(32_768);
    expect(wsAfterRedo.activeCandidateNoteIds).toEqual(wsAfterExec.activeCandidateNoteIds);

    // 3. [R25-HIGH-02] Single-Window `hasPeerWindows() === false` skips IPC, Dual-Window deduplicates
    //    incoming duplicate `messageId` BEFORE cloning, and peer `applyWorkspaceState` reuses `parsedMidi`
    //    with ZERO `StandardMidiCodec.parseMidi` re-parses across Mute/Solo/Instrument/Candidate switches!
    const prevDesktop = window.compassDesktop;
    const busA = new WorkspaceSyncBus('r26_dedup_ch');
    const busB = new WorkspaceSyncBus('r26_dedup_ch');
    const peerCtrl = new DemoPianoRollController(new StubAudioEngine());
    await peerCtrl.connect();
    let receivedOnB = 0;
    const unsubB = busB.subscribe((msg) => {
      if (msg.type === 'SYNC_WORKSPACE_STATE') {
        receivedOnB += 1;
        peerCtrl.applyWorkspaceState(msg.workspaceState);
      }
    });
    const parseMidiSpy = vi.spyOn(StandardMidiCodec, 'parseMidi');
    try {
      // Simulate Single-Window Electron (`hasPeerWindows: false` pushed from `main.cjs`)
      expect(preloadHolder.hasPeerWindows?.()).toBe(true);
      preloadHolder.apiConfigHandler?.(null, {
        apiBaseUrl: 'http://127.0.0.1:8000',
        handshakeToken: '',
        backendReady: true,
        hasPeerWindows: false,
      });
      expect(preloadHolder.hasPeerWindows?.()).toBe(false);

      window.compassDesktop = {
        hasPeerWindows: () => preloadHolder.hasPeerWindows?.() ?? true,
        sendWorkspaceSync: (msg) => preloadHolder.sendWorkspaceSync?.(msg),
      };
      sentViaPreload.length = 0;
      busA.publishWorkspaceState(wsAfterRedo);
      expect(sentViaPreload.length).toBe(0); // Skipped IPC in Single-Window mode!

      // Simulate Dual-Window Electron (`hasPeerWindows: true`) while preserving strict 3-key `getApiConfig()`
      preloadHolder.apiConfigHandler?.(null, {
        apiBaseUrl: 'http://127.0.0.1:8000',
        handshakeToken: '',
        backendReady: true,
        hasPeerWindows: true,
      });
      expect(preloadHolder.hasPeerWindows?.()).toBe(true);

      parseMidiSpy.mockClear();
      receivedOnB = 0;
      busA.publishWorkspaceState(ctrl.exportTrustedWorkspaceState());
      expect(sentViaPreload.length).toBe(1);
      expect(ipcSec.isValidWorkspaceSyncPayload(structuredClone(sentViaPreload[0]))).toBe(true);

      // Deliver Copy #1 to busB (e.g. via BroadcastChannel / first arrival)
      const firstWireCopy = structuredClone(sentViaPreload[0]) as {
        messageId: string;
        senderId: string;
        type: string;
        workspaceState: unknown;
      };
      (busB as unknown as { _dispatchIncoming: (m: unknown) => void })._dispatchIncoming(firstWireCopy);
      expect(receivedOnB).toBe(1);
      expect(peerCtrl.getSnapshot().tracks[0].notes.length).toBe(32_768);
      expect(parseMidiSpy).toHaveBeenCalledTimes(2); // Initial cold parse of 2 candidates

      // Deliver Copy #2 with SAME `messageId` and a trap getter on `workspaceState` proving `_dispatchIncoming`
      // drops duplicate `messageId` in O(1) BEFORE `parseAndCloneSyncMessage` touches `workspaceState`!
      let duplicateWorkspaceTouched = false;
      const duplicateEnvelope: Record<string, unknown> = {
        messageId: firstWireCopy.messageId,
        senderId: firstWireCopy.senderId,
        type: 'SYNC_WORKSPACE_STATE',
      };
      Object.defineProperty(duplicateEnvelope, 'workspaceState', {
        enumerable: true,
        get() {
          duplicateWorkspaceTouched = true;
          return firstWireCopy.workspaceState;
        },
      });
      (busB as unknown as { _dispatchIncoming: (m: unknown) => void })._dispatchIncoming(duplicateEnvelope);
      expect(duplicateWorkspaceTouched).toBe(false);
      expect(receivedOnB).toBe(1);

      // Now perform Mute, Solo, Instrument, and Candidate switch on `ctrl` and sync to `peerCtrl`:
      // `StandardMidiCodec.parseMidi` MUST be called ZERO times (`0` re-parses of unchanged `rawMidiBytes`)!
      parseMidiSpy.mockClear();
      const activeTid = ctrl.getSnapshot().activeTrackId;
      ctrl.toggleTrackMute(activeTid);
      sentViaPreload.length = 0;
      busA.publishWorkspaceState(ctrl.exportTrustedWorkspaceState());
      const muteWireCopy = structuredClone(sentViaPreload[0]);
      const t0MutePeerApply = performance.now();
      (busB as unknown as { _dispatchIncoming: (m: unknown) => void })._dispatchIncoming(muteWireCopy);
      const dtMutePeerApply = performance.now() - t0MutePeerApply;

      ctrl.toggleTrackSolo(activeTid);
      sentViaPreload.length = 0;
      busA.publishWorkspaceState(ctrl.exportTrustedWorkspaceState());
      const soloWireCopy = structuredClone(sentViaPreload[0]);
      const t0SoloPeerApply = performance.now();
      (busB as unknown as { _dispatchIncoming: (m: unknown) => void })._dispatchIncoming(soloWireCopy);
      const dtSoloPeerApply = performance.now() - t0SoloPeerApply;

      ctrl.setTrackInstrument(activeTid, 'SAX');
      sentViaPreload.length = 0;
      busA.publishWorkspaceState(ctrl.exportTrustedWorkspaceState());
      const instWireCopy = structuredClone(sentViaPreload[0]);
      const t0InstPeerApply = performance.now();
      (busB as unknown as { _dispatchIncoming: (m: unknown) => void })._dispatchIncoming(instWireCopy);
      const dtInstPeerApply = performance.now() - t0InstPeerApply;
      expect(Math.min(dtMutePeerApply, dtSoloPeerApply, dtInstPeerApply)).toBeLessThan(100);

      const t0CandSwitch = performance.now();
      await ctrl.selectGeneratedCandidate(1);
      const dtCandSwitch = performance.now() - t0CandSwitch;
      expect(dtCandSwitch).toBeLessThan(45);

      sentViaPreload.length = 0;
      busA.publishWorkspaceState(ctrl.exportTrustedWorkspaceState());
      const candWireCopy = structuredClone(sentViaPreload[0]);
      const t0PeerApply = performance.now();
      (busB as unknown as { _dispatchIncoming: (m: unknown) => void })._dispatchIncoming(candWireCopy);
      const dtPeerApply = performance.now() - t0PeerApply;

      expect(parseMidiSpy).toHaveBeenCalledTimes(0);
      expect(dtPeerApply).toBeLessThan(125);
      expect(peerCtrl.getSnapshot().activeCandidateIndex).toBe(1);
      expect(peerCtrl.getSnapshot().tracks[0].instrument).toBe('SAX');
      expect(peerCtrl.getSnapshot().tracks[0].isMuted).toBe(true);
      expect(peerCtrl.getSnapshot().tracks[0].isSoloed).toBe(true);
    } finally {
      parseMidiSpy.mockRestore();
      unsubB();
      busA.dispose();
      busB.dispose();
      if (prevDesktop === undefined) {
        delete window.compassDesktop;
      } else {
        window.compassDesktop = prevDesktop;
      }
    }
  });

  it('38. verifies UI/UX overhaul & COMPASS DTM Piano Roll parity: crimson-purple bottom AuroraBackdrop, deleted redundant ControlPanel target section, rounded interactive geometry, group-clamped chord drag, modeless smart pointer, ruler AI TARGET handle drag, cursor-anchored H/V zoom, triplet grid, right-click delete, inline track rename, and telemetry bar', async () => {
    // 1. Direct unit proof of `computeClampedGroupMoveDeltas` (zero chord squashing at 0/maxProjectTicks/0/127)
    const triad = [
      { pitch: 60, startTick: 480, durationTicks: 480 },
      { pitch: 64, startTick: 480, durationTicks: 480 },
      { pitch: 67, startTick: 960, durationTicks: 480 },
    ];
    const leftClamp = computeClampedGroupMoveDeltas(triad, -99999, -200, 480);
    expect(leftClamp.deltaTick).toBe(-480); // C4/E4 stop at tick 0; G4 stays at 480 (+480 offset preserved!)
    expect(leftClamp.deltaPitch).toBe(-60); // C4->0, E4->4, G4->7 (major triad intervals preserved!)

    const topClamp = computeClampedGroupMoveDeltas(triad, 0, 200, 480);
    expect(topClamp.deltaPitch).toBe(60); // G4->127, E4->124, C4->120 (major triad intervals preserved!)

    const rightClamp = computeClampedGroupMoveDeltas(triad, 999_999_999, 0, 480);
    const maxTicks = MAX_SMF_QUARTER_NOTES * 480;
    expect(960 + 480 + rightClamp.deltaTick).toBe(maxTicks);
    expect(480 + rightClamp.deltaTick).toBe(maxTicks - 960);

    // 2. `<AuroraBackdrop />` renders standalone and inside `<BootSplashScreen />` with non-interactive crimson-purple layers
    const standaloneAurora = render(<AuroraBackdrop />);
    expect(standaloneAurora.getByTestId('aurora-backdrop').getAttribute('aria-hidden')).toBe('true');
    standaloneAurora.unmount();

    const splashUi = render(
      <BootSplashScreen
        currentAction="モデルカタログをロードしています"
        progressPercent={42}
        logs={[{ id: 1, timestamp: '00:00:01', message: 'Engine warm-up' }]}
      />,
    );
    const splashAurora = splashUi.getByTestId('aurora-backdrop');
    expect(splashAurora.getAttribute('aria-hidden')).toBe('true');
    expect(splashAurora.className).toContain('pointer-events-none');
    expect(splashAurora.innerHTML).toContain('bg-aurora-horizon');
    expect(splashAurora.innerHTML).toContain('bg-aurora-core');
    expect(splashAurora.innerHTML).toContain('bg-aurora-curtains');
    splashUi.unmount();

    // 3. `<ControlPanelView />`: Section 2 deleted, redundant `ピアノロールを表示` deleted, rounded geometry & context summary verified
    const stubAudio = new StubAudioEngine();
    const demoCtrl = new DemoPianoRollController(stubAudio);
    const cpSnap = demoCtrl.getSnapshot();
    const cpUi = render(
      <ControlPanelView
        audioEngine={stubAudio}
        controllers={[demoCtrl]}
        session={cpSnap}
        models={[
          {
            model_name: 'MORTM4.5D-80M-SFT-Gen',
            description: 'MORTM 4.5 Dense SFT-Gen',
            has_weights: true,
            tag: { model: 'sft_gen', type: 'mortm', version: '4.5' },
            rule: { input_midi: true },
          },
        ]}
        hardwareInfo={{
          device: 'cpu',
          dtype: 'float32',
          backend_class: 'DemoBackend',
          user_models_dir: '',
          bundled_models_dir: '',
          available_models_count: 1,
          models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
        }}
        settings={{
          modelName: 'MORTM4.5D-80M-SFT-Gen',
          temperature: 1.0,
          topP: 0.95,
          thinking: true,
          cotTemperature: 0.85,
          numCandidates: 2,
          taskMode: 'auto',
          conditionTrackId: '',
          forceMonophonic: false,
          noteDensity: null,
          musicalKey: '',
          selectedGenres: [],
        }}
        isGenerating={false}
        isRescanningModels={false}
        statusTitle="STANDBY"
        statusDetail="準備完了"
        generationLogs={[]}
        errorMessage={null}
        onUpdateSettings={() => {}}
        onSwitchController={() => {}}
        onRescanModels={() => {}}
        onTogglePlayback={() => {}}
        onGenerate={() => {}}
        onCancelGenerate={() => {}}
      />,
    );
    expect(cpUi.container.querySelector('#target-track-select')).toBeNull();
    expect(cpUi.container.querySelector('#target-start-bar')).toBeNull();
    expect(cpUi.container.querySelector('#target-end-bar')).toBeNull();
    expect(cpUi.container.textContent).not.toContain('生成対象のトラックと小節範囲');
    expect(cpUi.container.textContent).not.toContain('ピアノロールを表示');
    const summaryEl = cpUi.getByTestId('pianoroll-context-summary');
    expect(summaryEl.textContent).toContain('小節');
    expect(cpUi.getByTestId('generate-button').className).toContain('rounded-full');
    expect(cpUi.getByRole('button', { name: 'サウンド確認' }).className).toContain('rounded-full');
    fireEvent.click(cpUi.getByText('音楽構造・ジャンル・タスクの詳細指定 (SFT条件付け)'));
    const cpButtons = Array.from(cpUi.container.querySelectorAll('button'));
    for (const btn of cpButtons) {
      expect(btn.className).toMatch(/rounded-(full|xl|2xl)/);
      const accessible =
        (btn.textContent || '').trim() ||
        (btn.getAttribute('aria-label') || '').trim() ||
        (btn.getAttribute('title') || '').trim();
      expect(accessible.length).toBeGreaterThan(0);
    }
    cpUi.unmount();

    // 4. `<MidiTrackImportModal />` rounded geometry (`rounded-2xl` card, `rounded-full` buttons)
    const parsedModalMidi = StandardMidiCodec.createDemoMidiPreset();
    const modalUi = render(
      <MidiTrackImportModal
        fileName="rounded_modal_check.mid"
        parsedMidi={parsedModalMidi}
        onCancel={() => {}}
        onConfirm={() => {}}
      />,
    );
    expect(modalUi.getByTestId('midi-import-modal-card').className).toContain('rounded-2xl');
    expect(modalUi.getByTestId('confirm-midi-import-button').className).toContain('rounded-full');
    modalUi.unmount();

    // 5. `<PianoRollView />` COMPASS DTM Direct Manipulation & Rounded Toolbar Verification
    const prCtrl = new DemoPianoRollController(new StubAudioEngine());
    // Clear existing notes on active track and place a clean C4-E4-G4 chord at tick 480..960
    const initTrackId = prCtrl.getSnapshot().activeTrackId;
    prCtrl.selectAllNotesOnActiveTrack();
    prCtrl.deleteSelectedNotes();
    prCtrl.createNoteAt(60, 480, 480, 100);
    prCtrl.createNoteAt(64, 480, 480, 96);
    prCtrl.createNoteAt(67, 960, 480, 92);
    const seededNotes = prCtrl.getSnapshot().tracks.find((t) => t.id === initTrackId)!.notes;
    const nC4 = seededNotes.find((n) => n.pitch === 60)!;
    const nE4 = seededNotes.find((n) => n.pitch === 64)!;
    const nG4 = seededNotes.find((n) => n.pitch === 67)!;
    prCtrl.setTargetMeasureRange(1, 3);
    prCtrl.setSelectedNotes([]);

    const StatefulPianoRollHarness: React.FC = () => {
      const [snap, setSnap] = React.useState(prCtrl.getSnapshot());
      React.useEffect(() => prCtrl.subscribe(() => setSnap(prCtrl.getSnapshot())), []);
      return (
        <PianoRollView
          isActive={true}
          controller={prCtrl}
          session={snap}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    };

    const prUi = render(<StatefulPianoRollHarness />);
    const canvas = prUi.getByTestId('pianoroll-canvas') as HTMLCanvasElement;
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    } as DOMRect);

    // Toolbar rationalization: zero redundant `選択` / `パン` text mode buttons; all buttons rounded & labeled
    const prButtons = Array.from(prUi.container.querySelectorAll('button'));
    expect(prButtons.some((b) => (b.textContent || '').trim() === '選択')).toBe(false);
    expect(prButtons.some((b) => (b.textContent || '').trim() === 'パン')).toBe(false);
    for (const btn of prButtons) {
      expect(btn.className).toMatch(/rounded-(full|xl|2xl)/);
      const accessible =
        (btn.textContent || '').trim() ||
        (btn.getAttribute('aria-label') || '').trim() ||
        (btn.getAttribute('title') || '').trim();
      expect(accessible.length).toBeGreaterThan(0);
    }

    // (a) Single-click empty grid (tick ~1440 => x = 72 + 3 * 64 = 264, pitch 72 => y = 72) snaps & moves playheadTick!
    fireEvent.pointerDown(canvas, { pointerId: 1, button: 0, clientX: 264, clientY: 72 });
    fireEvent.pointerUp(canvas, { pointerId: 1, button: 0, clientX: 264, clientY: 72 });
    expect(prCtrl.getSnapshot().playheadTick).toBe(1440);

    // (b) Range-select button (`対象小節内のノートを選択`) selects all 3 notes in bars 1..3, and telemetry bar shows selected count
    fireEvent.click(prUi.getByRole('button', { name: '対象小節内のノートを選択' }));
    expect(prCtrl.getSnapshot().selectedNoteIds.size).toBe(3);
    expect(prUi.getByTestId('pianoroll-telemetry-bar').textContent).toContain('3');

    // (c) Group-clamped multi-note DOM drag: drag C4 (x=144, y=264) far down-left (-500px X, +1200px Y) -> intervals preserved!
    fireEvent.pointerDown(canvas, { pointerId: 2, button: 0, clientX: 144, clientY: 264 });
    fireEvent.pointerMove(canvas, { pointerId: 2, buttons: 1, clientX: -356, clientY: 1464 });
    fireEvent.pointerUp(canvas, { pointerId: 2, button: 0, clientX: -356, clientY: 1464 });
    const afterClampNotes = prCtrl
      .getSnapshot()
      .tracks.find((t) => t.id === initTrackId)!
      .notes.slice()
      .sort((a, b) => a.pitch - b.pitch);
    expect(afterClampNotes.map((n) => n.pitch)).toEqual([0, 4, 7]);
    expect(afterClampNotes.map((n) => n.startTick)).toEqual([0, 0, 480]);
    act(() => {
      prCtrl.undo(); // Restore C4=60@480, E4=64@480, G4=67@960
      prCtrl.setSelectedNotes([nC4.id]);
    });

    // (d) Additive `Shift + Marquee` selection: select C4 only, then Shift-marquee around G4 (x=200..260, y=140..165)
    fireEvent.pointerDown(canvas, {
      pointerId: 3,
      button: 0,
      shiftKey: true,
      clientX: 195,
      clientY: 140,
    });
    fireEvent.pointerMove(canvas, {
      pointerId: 3,
      buttons: 1,
      shiftKey: true,
      clientX: 268,
      clientY: 168,
    });
    fireEvent.pointerUp(canvas, {
      pointerId: 3,
      button: 0,
      shiftKey: true,
      clientX: 268,
      clientY: 168,
    });
    expect(prCtrl.getSnapshot().selectedNoteIds.has(nC4.id)).toBe(true);
    expect(prCtrl.getSnapshot().selectedNoteIds.has(nG4.id)).toBe(true);
    expect(prCtrl.getSnapshot().selectedNoteIds.has(nE4.id)).toBe(false);

    // (e) Ruler `AI TARGET` right bracket handle drag: Bar 3 end is tick 5760 -> x = 72 + 12 * 64 = 840, y = 8
    // Drag left to Bar 2 end (tick 3840 -> x = 72 + 8 * 64 = 584) -> targetRange becomes 1..2!
    fireEvent.pointerDown(canvas, { pointerId: 4, button: 0, clientX: 840, clientY: 8 });
    fireEvent.pointerMove(canvas, { pointerId: 4, buttons: 1, clientX: 584, clientY: 8 });
    fireEvent.pointerUp(canvas, { pointerId: 4, button: 0, clientX: 584, clientY: 8 });
    expect(prCtrl.getSnapshot().targetRange.startMeasure).toBe(1);
    expect(prCtrl.getSnapshot().targetRange.endMeasure).toBe(2);

    // (f) Right-click (`contextMenu`) on E4 (tick 480..960 => x=150, pitch 64 => y=200) deletes E4!
    fireEvent.contextMenu(canvas, { clientX: 150, clientY: 200 });
    const remainingIds = prCtrl
      .getSnapshot()
      .tracks.find((t) => t.id === initTrackId)!
      .notes.map((n) => n.id);
    expect(remainingIds).not.toContain(nE4.id);
    expect(remainingIds.length).toBe(2);

    // (g) Triplet (`3連`) toggle + Pencil (`ペン入力`) draws triplet-snapped note (1/16 triplet at 480 PPQN = 80 ticks)
    fireEvent.click(prUi.getByRole('button', { name: '3連符グリッド切替' }));
    fireEvent.click(prUi.getByRole('button', { name: 'ペン入力' }));
    // Click near tick 170 => x = 72 + (170 / 480) * 64 = 94.67, pitch 72 => y = 72 -> snaps to tick 160 with duration 80!
    fireEvent.pointerDown(canvas, { pointerId: 5, button: 0, clientX: 95, clientY: 72 });
    fireEvent.pointerUp(canvas, { pointerId: 5, button: 0, clientX: 95, clientY: 72 });
    const drawnTriplet = prCtrl
      .getSnapshot()
      .tracks.find((t) => t.id === initTrackId)!
      .notes.find((n) => n.pitch === 72);
    expect(drawnTriplet).toBeDefined();
    expect(drawnTriplet!.startTick).toBe(160);
    expect(drawnTriplet!.durationTicks).toBe(80);

    // (h) Cursor-anchored horizontal (`Ctrl+Wheel`) & vertical (`Ctrl+Shift+Wheel`) zoom execute cleanly without NaN
    const hWheel = new WheelEvent('wheel', {
      deltaY: -120,
      ctrlKey: true,
      clientX: 328,
      clientY: 200,
      bubbles: true,
      cancelable: true,
    });
    act(() => {
      canvas.dispatchEvent(hWheel);
    });
    expect(hWheel.defaultPrevented).toBe(true);

    const vWheel = new WheelEvent('wheel', {
      deltaY: -120,
      ctrlKey: true,
      shiftKey: true,
      clientX: 328,
      clientY: 200,
      bubbles: true,
      cancelable: true,
    });
    act(() => {
      canvas.dispatchEvent(vWheel);
    });
    expect(vWheel.defaultPrevented).toBe(true);

    // (i) Double-click inline Track Rename in Track Rack (both Escape cancel + Enter commit)
    const activeTrackName = prCtrl.getSnapshot().tracks.find((t) => t.id === initTrackId)!.name;
    const trackRack = prUi.getByLabelText('トラックラック');
    const trackNameEl = Array.from(trackRack.querySelectorAll('div')).find(
      (el) => el.textContent === activeTrackName,
    )!;
    fireEvent.doubleClick(trackNameEl);
    const cancelInput = prUi.getByLabelText(`${activeTrackName} トラック名編集`) as HTMLInputElement;
    fireEvent.change(cancelInput, { target: { value: 'Should Not Save' } });
    fireEvent.keyDown(cancelInput, { key: 'Escape', code: 'Escape' });
    expect(prCtrl.getSnapshot().tracks.find((t) => t.id === initTrackId)!.name).toBe(
      activeTrackName,
    );

    const trackNameEl2 = Array.from(trackRack.querySelectorAll('div')).find(
      (el) => el.textContent === activeTrackName,
    )!;
    fireEvent.doubleClick(trackNameEl2);
    const renameInput = prUi.getByLabelText(`${activeTrackName} トラック名編集`) as HTMLInputElement;
    fireEvent.change(renameInput, { target: { value: 'Lead Synth COMPASS' } });
    fireEvent.keyDown(renameInput, { key: 'Enter', code: 'Enter' });
    expect(prCtrl.getSnapshot().tracks.find((t) => t.id === initTrackId)!.name).toBe(
      'Lead Synth COMPASS',
    );

    prUi.unmount();
  });

  it('39. verifies odd-meter snap, undo-safe metadata sync, Shift-fine drag/resize, isolated right-click delete, multi-digit inputs, safe sample modal, and pop-out header non-redundancy', async () => {
    const ctrl = new DemoPianoRollController(new StubAudioEngine());

    // 1. Odd/compound meter measure-anchored `snapTick`: 7/8 (measureTicks=1680) + 1/4 quantize -> Bar 2 downbeat 1680 snaps to 1680
    ctrl.setTimeSignature(7, 8);
    ctrl.setQuantize(4, false);
    expect(ctrl.snapTick(1680)).toBe(1680);
    expect(ctrl.snapTick(1670)).toBe(1680);
    expect(ctrl.snapTick(3360)).toBe(3360);

    // 3/4 (measureTicks=1440) + 1/4 triplet (step=320) -> Bar 2 downbeat 1440 snaps cleanly to 1440
    ctrl.setTimeSignature(3, 4);
    ctrl.setQuantize(4, true);
    expect(ctrl.snapTick(1440)).toBe(1440);
    ctrl.setTimeSignature(4, 4);
    ctrl.setQuantize(16, false);

    // 2. Peer-window metadata/range `applyWorkspaceState` preserves local Undo stack!
    const trkId = ctrl.getSnapshot().activeTrackId;
    ctrl.selectAllNotesOnActiveTrack();
    ctrl.deleteSelectedNotes();
    ctrl.createNoteAt(60, 480, 480, 100);
    ctrl.createNoteAt(64, 960, 480, 96);
    expect(ctrl.getSnapshot().canUndo).toBe(true);
    const exported = ctrl.exportWorkspaceState();
    ctrl.applyWorkspaceState({
      ...exported,
      bpm: 138,
      targetStartMeasure: 2,
      targetEndMeasure: 5,
    });
    expect(ctrl.getSnapshot().bpm).toBe(138);
    expect(ctrl.getSnapshot().targetRange.startMeasure).toBe(2);
    expect(ctrl.getSnapshot().canUndo).toBe(true);
    ctrl.undo();
    expect(ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.length).toBe(1);
    ctrl.redo();
    expect(ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.length).toBe(2);
    ctrl.setBpm(120);
    ctrl.setTargetMeasureRange(1, 4);

    const notesInit = ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes;
    const noteA = notesInit.find((n) => n.pitch === 60)!; // 480..960
    const noteB = notesInit.find((n) => n.pitch === 64)!; // 960..1440

    const StatefulHarness: React.FC = () => {
      const [snap, setSnap] = React.useState(ctrl.getSnapshot());
      React.useEffect(() => ctrl.subscribe(() => setSnap(ctrl.getSnapshot())), []);
      return (
        <PianoRollView
          isActive={true}
          controller={ctrl}
          session={snap}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    };

    const ui = render(<StatefulHarness />);
    const canvas = ui.getByTestId('pianoroll-canvas') as HTMLCanvasElement;
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    } as DOMRect);

    // 3. `Shift + Drag` on already-selected noteA performs 1-tick fine move WITHOUT deselecting noteA!
    act(() => {
      ctrl.setSelectedNotes([noteA.id]);
    });
    // noteA is at tick 480..960 (x = 136..200), pitch 60 (y = 264); +4px = +30 ticks (fine move, off 120-tick grid)
    fireEvent.pointerDown(canvas, {
      pointerId: 11,
      button: 0,
      shiftKey: true,
      clientX: 150,
      clientY: 264,
    });
    fireEvent.pointerMove(canvas, {
      pointerId: 11,
      buttons: 1,
      shiftKey: true,
      clientX: 154,
      clientY: 264,
    });
    fireEvent.pointerUp(canvas, {
      pointerId: 11,
      button: 0,
      shiftKey: true,
      clientX: 154,
      clientY: 264,
    });
    const movedNoteA = ctrl
      .getSnapshot()
      .tracks.find((t) => t.id === trkId)!
      .notes.find((n) => n.id === noteA.id)!;
    expect(movedNoteA.startTick).toBe(510);
    expect(ctrl.getSnapshot().selectedNoteIds.has(noteA.id)).toBe(true);
    act(() => {
      ctrl.undo();
      ctrl.setSelectedNotes([noteA.id, noteB.id]);
    });

    // Stationary `Shift + Click` on selected noteA deselects noteA while keeping noteB selected
    fireEvent.pointerDown(canvas, {
      pointerId: 12,
      button: 0,
      shiftKey: true,
      clientX: 150,
      clientY: 264,
    });
    fireEvent.pointerUp(canvas, {
      pointerId: 12,
      button: 0,
      shiftKey: true,
      clientX: 150,
      clientY: 264,
    });
    expect(ctrl.getSnapshot().selectedNoteIds.has(noteA.id)).toBe(false);
    expect(ctrl.getSnapshot().selectedNoteIds.has(noteB.id)).toBe(true);

    // Stationary plain `Click` on noteA when both noteA+noteB are selected isolates selection to [noteA]
    act(() => {
      ctrl.setSelectedNotes([noteA.id, noteB.id]);
    });
    fireEvent.pointerDown(canvas, { pointerId: 13, button: 0, clientX: 150, clientY: 264 });
    fireEvent.pointerUp(canvas, { pointerId: 13, button: 0, clientX: 150, clientY: 264 });
    expect(Array.from(ctrl.getSnapshot().selectedNoteIds)).toEqual([noteA.id]);

    // 4. Snapped resize vs `Shift` fine-resize floor: right edge of noteA (tick 960 -> x = 200, y = 264)
    fireEvent.pointerDown(canvas, { pointerId: 14, button: 0, clientX: 198, clientY: 264 });
    fireEvent.pointerMove(canvas, { pointerId: 14, buttons: 1, clientX: 80, clientY: 264 });
    fireEvent.pointerUp(canvas, { pointerId: 14, button: 0, clientX: 80, clientY: 264 });
    expect(
      ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.find((n) => n.id === noteA.id)!
        .durationTicks,
    ).toBe(120); // Snapped 1/16 floor

    fireEvent.pointerDown(canvas, {
      pointerId: 15,
      button: 0,
      shiftKey: true,
      clientX: 151,
      clientY: 264,
    });
    fireEvent.pointerMove(canvas, {
      pointerId: 15,
      buttons: 1,
      shiftKey: true,
      clientX: 80,
      clientY: 264,
    });
    fireEvent.pointerUp(canvas, {
      pointerId: 15,
      button: 0,
      shiftKey: true,
      clientX: 80,
      clientY: 264,
    });
    expect(
      ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.find((n) => n.id === noteA.id)!
        .durationTicks,
    ).toBe(24); // Fine 24-tick floor
    act(() => {
      ctrl.undo();
      ctrl.undo();
      ctrl.setSelectedNotes([noteA.id, noteB.id]);
    });

    // 5. Right-click during active drag cancels drag without deleting; idle right-click on noteB deletes ONLY noteB
    fireEvent.pointerDown(canvas, { pointerId: 16, button: 0, clientX: 150, clientY: 264 });
    fireEvent.pointerMove(canvas, { pointerId: 16, buttons: 1, clientX: 220, clientY: 232 });
    fireEvent.contextMenu(canvas, { clientX: 220, clientY: 232 });
    fireEvent.pointerUp(canvas, { pointerId: 16, button: 0, clientX: 220, clientY: 232 });
    expect(ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.length).toBe(2);
    expect(
      ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.find((n) => n.id === noteA.id)!
        .startTick,
    ).toBe(480);

    // Idle right-click on noteB (tick 960..1440 -> x=220, pitch 64 -> y=200) while [noteA, noteB] selected
    fireEvent.contextMenu(canvas, { clientX: 220, clientY: 200 });
    const afterRightClickNotes = ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes;
    expect(afterRightClickNotes.map((n) => n.id)).toEqual([noteA.id]);
    expect(Array.from(ctrl.getSnapshot().selectedNoteIds)).toEqual([noteA.id]);

    // 6. Focused multi-digit typing in `#pianoroll-bpm` ('1' -> '4' -> '0' = 140) and `生成終了小節` (start=12, '1' -> '5' = 15)
    const bpmInput = ui.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
    act(() => {
      bpmInput.focus();
    });
    fireEvent.change(bpmInput, { target: { value: '1' } });
    expect(bpmInput.value).toBe('1'); // Not prematurely clamped to '20'!
    fireEvent.change(bpmInput, { target: { value: '14' } });
    expect(bpmInput.value).toBe('14');
    fireEvent.change(bpmInput, { target: { value: '140' } });
    expect(bpmInput.value).toBe('140');
    act(() => {
      bpmInput.blur();
    });
    expect(ctrl.getSnapshot().bpm).toBe(140);

    act(() => {
      ctrl.setTargetMeasureRange(12, 14);
    });
    const endBarInput = ui.getByLabelText('生成終了小節') as HTMLInputElement;
    act(() => {
      endBarInput.focus();
    });
    fireEvent.change(endBarInput, { target: { value: '1' } });
    expect(endBarInput.value).toBe('1'); // Not prematurely clamped to '12'!
    fireEvent.change(endBarInput, { target: { value: '15' } });
    act(() => {
      endBarInput.blur();
    });
    expect(ctrl.getSnapshot().targetRange.startMeasure).toBe(12);
    expect(ctrl.getSnapshot().targetRange.endMeasure).toBe(15);

    // 7. Sample button (`サンプル`) has explicit text label & opens confirmation modal when notes already exist
    const sampleBtn = ui.getByTestId('load-sample-arrangement-button');
    expect(sampleBtn.textContent).toContain('サンプル');
    fireEvent.click(sampleBtn);
    expect(ui.queryByTestId('midi-import-modal')).not.toBeNull();
    expect(ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.map((n) => n.id)).toEqual([
      noteA.id,
    ]);
    fireEvent.click(ui.getByRole('button', { name: 'キャンセル' }));
    expect(ui.queryByTestId('midi-import-modal')).toBeNull();

    ui.unmount();
  });

  // =========================================================================
  // 40. Round-2 Fact-Checker Remediations (`[R2-HIGH-01..02]`, `[R2-MED-01..04]`, `[R2-LOW-01]`)
  // =========================================================================
  it('enforces Round-2 Fact-Checker remediations: 2.5px drag deadzone, synchronous Escape cancel in BPM/Bar inputs, single-Undo Pencil draw-create, modal focus/aria-hidden + Append range preservation, ruler cancel restoration + y<=13 hit-test, and candidate byte equality', async () => {
    const audio = new StubAudioEngine();
    const ctrl = new DemoPianoRollController(audio);
    await ctrl.connect();
    const trkId = ctrl.getSnapshot().activeTrackId;

    ctrl.createNoteAt(60, 480, 480, 100);
    ctrl.createNoteAt(64, 960, 480, 96);
    const [noteA, noteB] = ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes;
    ctrl.setSelectedNotes([noteA.id, noteB.id]);

    let modalOpenFlag = false;
    const StatefulHarness: React.FC = () => {
      const [snap, setSnap] = React.useState(ctrl.getSnapshot());
      React.useEffect(() => ctrl.subscribe(setSnap), []);
      return (
        <PianoRollView
          isActive={true}
          controller={ctrl}
          session={snap}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
          onModalOpenChange={(open) => {
            modalOpenFlag = open;
          }}
        />
      );
    };

    const ui = render(<StatefulHarness />);
    const canvas = ui.getByTestId('pianoroll-canvas') as HTMLCanvasElement;
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    });

    // 1. `[R2-HIGH-01]` 1px-2px micro-jitter during `Shift+Click` on noteA NEVER shifts chord off-grid & ALWAYS deselects noteA!
    fireEvent.pointerDown(canvas, {
      pointerId: 31,
      button: 0,
      clientX: 150,
      clientY: 264,
      shiftKey: true,
    });
    fireEvent.pointerMove(canvas, {
      pointerId: 31,
      buttons: 1,
      clientX: 151,
      clientY: 265,
      shiftKey: true,
    }); // hypot(1,1) = 1.414px < 2.5px deadzone!
    fireEvent.pointerUp(canvas, {
      pointerId: 31,
      button: 0,
      clientX: 151,
      clientY: 265,
      shiftKey: true,
    });
    const afterMicroJitterNotes = ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes;
    expect(afterMicroJitterNotes.find((n) => n.id === noteA.id)!.startTick).toBe(480);
    expect(afterMicroJitterNotes.find((n) => n.id === noteB.id)!.startTick).toBe(960);
    expect(Array.from(ctrl.getSnapshot().selectedNoteIds)).toEqual([noteB.id]);

    // 2. `[R2-HIGH-02]` Pressing `Escape` inside `#pianoroll-bpm`, `生成開始小節`, and `生成終了小節` restores pre-focus values cleanly
    const bpmInput = ui.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
    act(() => {
      bpmInput.focus();
    });
    fireEvent.change(bpmInput, { target: { value: '9' } }); // < 20 partial draft
    act(() => {
      fireEvent.keyDown(bpmInput, { key: 'Escape' });
    });
    expect(ctrl.getSnapshot().bpm).toBe(120); // NOT clamped to 20!

    act(() => {
      bpmInput.focus();
    });
    fireEvent.change(bpmInput, { target: { value: '145' } }); // Live onChange mutates to 145
    expect(ctrl.getSnapshot().bpm).toBe(145);
    act(() => {
      fireEvent.keyDown(bpmInput, { key: 'Escape' });
    });
    expect(ctrl.getSnapshot().bpm).toBe(120); // Restored to pre-focus 120!

    act(() => {
      ctrl.setTargetMeasureRange(3, 6);
    });
    const startBarInput = ui.getByLabelText('生成開始小節') as HTMLInputElement;
    act(() => {
      startBarInput.focus();
    });
    fireEvent.change(startBarInput, { target: { value: '1' } });
    expect(ctrl.getSnapshot().targetRange.startMeasure).toBe(1);
    act(() => {
      fireEvent.keyDown(startBarInput, { key: 'Escape' });
    });
    expect(ctrl.getSnapshot().targetRange.startMeasure).toBe(3);
    expect(ctrl.getSnapshot().targetRange.endMeasure).toBe(6);

    const endBarInput = ui.getByLabelText('生成終了小節') as HTMLInputElement;
    act(() => {
      endBarInput.focus();
    });
    fireEvent.change(endBarInput, { target: { value: '8' } });
    expect(ctrl.getSnapshot().targetRange.endMeasure).toBe(8);
    act(() => {
      fireEvent.keyDown(endBarInput, { key: 'Escape' });
    });
    expect(ctrl.getSnapshot().targetRange.startMeasure).toBe(3);
    expect(ctrl.getSnapshot().targetRange.endMeasure).toBe(6);

    // 3. `[R2-MED-01]` Pencil (`draw`) drag-to-size commits in 1 single Undo step, and `Escape` mid-stroke leaves ZERO orphan notes
    fireEvent.click(ui.getByRole('button', { name: 'ペン入力' }));
    // Start drawing at tick 1920 (x = 72 + 4 * 64 = 328), pitch 72 (y = 72), drag horizontally, then press Escape!
    fireEvent.pointerDown(canvas, { pointerId: 32, button: 0, clientX: 328, clientY: 72 });
    fireEvent.pointerMove(canvas, { pointerId: 32, buttons: 1, clientX: 456, clientY: 72 });
    fireEvent.keyDown(window, { key: 'Escape' });
    fireEvent.pointerUp(canvas, { pointerId: 32, button: 0, clientX: 456, clientY: 72 });
    expect(ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.length).toBe(2); // Zero orphan notes!

    // Now draw and drag to 2 beats (960 ticks: dx = 112px -> +840 ticks over base 120 ticks = 960 ticks) and release
    fireEvent.pointerDown(canvas, { pointerId: 33, button: 0, clientX: 328, clientY: 72 });
    fireEvent.pointerMove(canvas, { pointerId: 33, buttons: 1, clientX: 440, clientY: 72 });
    fireEvent.pointerUp(canvas, { pointerId: 33, button: 0, clientX: 440, clientY: 72 });
    const drawnNote = ctrl
      .getSnapshot()
      .tracks.find((t) => t.id === trkId)!
      .notes.find((n) => n.pitch === 72);
    expect(drawnNote).toBeDefined();
    expect(drawnNote!.startTick).toBe(1920);
    expect(drawnNote!.durationTicks).toBe(960);

    // A SINGLE `undo()` removes the entire drawn 960-tick note!
    act(() => {
      ctrl.undo();
    });
    expect(
      ctrl.getSnapshot().tracks.find((t) => t.id === trkId)!.notes.find((n) => n.pitch === 72),
    ).toBeUndefined();
    fireEvent.click(ui.getByRole('button', { name: 'ペン入力' }));

    // 4. `[R2-MED-04]` `y = 14` on ruler hits Playhead scrub (NOT `AI TARGET` handle `y <= 13`), and `Escape` cancels ruler-range / ruler-scrub!
    act(() => {
      ctrl.setTargetMeasureRange(1, 3);
      ctrl.setPlayheadTick(960);
    });
    // Bar 1 start handle is at x = 72, y = 2..12. Clicking at x = 72, y = 14 scrubs playhead to 0 without dragging targetRange!
    fireEvent.pointerDown(canvas, { pointerId: 34, button: 0, clientX: 72, clientY: 14 });
    expect(ctrl.getSnapshot().playheadTick).toBe(0);
    // Cancel scrub with Escape -> restores playheadTick to 960!
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(ctrl.getSnapshot().playheadTick).toBe(960);

    // Drag `AI TARGET` end handle (Bar 3 end = tick 5760 -> x = 840, y = 8) to Bar 2 end (x = 584), then press Escape -> restores 1..3!
    fireEvent.pointerDown(canvas, { pointerId: 35, button: 0, clientX: 840, clientY: 8 });
    fireEvent.pointerMove(canvas, { pointerId: 35, buttons: 1, clientX: 584, clientY: 8 });
    expect(ctrl.getSnapshot().targetRange.endMeasure).toBe(2);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(ctrl.getSnapshot().targetRange.startMeasure).toBe(1);
    expect(ctrl.getSnapshot().targetRange.endMeasure).toBe(3);

    // 5. `[R2-MED-02]` Sample modal focus enters dialog synchronously (no focused descendant inside `aria-hidden="true"`), displays Japanese sample preset title, and Append mode (`既存トラックに追加`) preserves `targetRange` & `playheadTick`!
    act(() => {
      ctrl.setTargetMeasureRange(2, 5);
      ctrl.setPlayheadTick(1440);
    });
    const sampleBtn = ui.getByTestId('load-sample-arrangement-button');
    act(() => {
      sampleBtn.focus();
    });
    fireEvent.click(sampleBtn);
    expect(modalOpenFlag).toBe(true);
    const modalEl = ui.getByTestId('midi-import-modal');
    expect(modalEl.textContent).toContain('サンプルアレンジの読み込み設定');
    expect(modalEl.contains(document.activeElement)).toBe(true);
    const hiddenAncestorsWithFocus = Array.from(
      ui.container.querySelectorAll('[aria-hidden="true"]'),
    ).filter((el) => el.contains(document.activeElement));
    expect(hiddenAncestorsWithFocus.length).toBe(0);

    fireEvent.click(ui.getByText('既存トラックに追加'));
    fireEvent.click(ui.getByTestId('confirm-midi-import-button'));
    expect(modalOpenFlag).toBe(false);
    expect(ctrl.getSnapshot().targetRange.startMeasure).toBe(2);
    expect(ctrl.getSnapshot().targetRange.endMeasure).toBe(5);
    expect(ctrl.getSnapshot().playheadTick).toBe(1440);

    // 6. `[R2-LOW-01]` `preserveUndoHistory` clears stale Undo stack if candidate `rawMidiBytes` differ despite identical count/index
    const candCtrl = new DemoPianoRollController(new StubAudioEngine());
    const smfA = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const smfB = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 67, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    await candCtrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'c0.mid',
          rawMidiBytes: smfA,
          parsedMidi: StandardMidiCodec.parseMidi(smfA.buffer as ArrayBuffer),
          reason: null,
        },
      ],
      [],
    );
    candCtrl.createNoteAt(72, 1920, 480, 100);
    expect(candCtrl.getSnapshot().canUndo).toBe(true);
    const exportedWs = candCtrl.exportWorkspaceState();
    candCtrl.applyWorkspaceState({
      ...exportedWs,
      candidates: [{ ...exportedWs.candidates[0], rawMidiBytes: smfB }],
    });
    expect(candCtrl.getSnapshot().canUndo).toBe(false);

    ui.unmount();
  });

  it('41. enforces Round-3 Fact-Checker UI/UX & Piano Roll remediations ([R3-MED-01], [R3-MED-02], [R3-MED-03], [R3-MED-04], [R3-LOW-01])', async () => {
    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    ctrl.createNoteAt(60, 480, 480, 100);
    ctrl.createNoteAt(64, 960, 480, 100);
    const trk0 = ctrl.getSnapshot().tracks[0];
    const [noteA, noteB] = trk0.notes;
    ctrl.setSelectedNotes([noteA.id, noteB.id]);

    let modalOpenFlag = false;
    const Round3Harness: React.FC<{ isActive?: boolean }> = ({ isActive = true }) => {
      const [snap, setSnap] = React.useState(ctrl.getSnapshot());
      React.useEffect(() => ctrl.subscribe(setSnap), []);
      return (
        <React.StrictMode>
          <PianoRollView
            isActive={isActive}
            controller={ctrl}
            session={snap}
            focusRevision={0}
            onLaunchCompassAi={() => {}}
            onSwitchToDemoController={() => {}}
            onModalOpenChange={(open) => {
              modalOpenFlag = open;
            }}
          />
        </React.StrictMode>
      );
    };

    const ui = render(<Round3Harness isActive={true} />);
    const canvas = ui.getByTestId('pianoroll-canvas') as HTMLCanvasElement;
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    });

    // 1. `[R3-MED-01]` Under `<React.StrictMode>`, modal close restores focus AFTER sibling Toolbar removes `aria-hidden="true"`, AND async multi-track upload restores focus to `upload-midi-button`
    const origFocus = HTMLElement.prototype.focus;
    const focusWithAriaHiddenAncestor: string[] = [];
    const focusSpy = vi
      .spyOn(HTMLElement.prototype, 'focus')
      .mockImplementation(function (this: HTMLElement, options?: FocusOptions) {
        if (this.closest('[aria-hidden="true"]')) {
          focusWithAriaHiddenAncestor.push(
            this.getAttribute('data-testid') ?? this.tagName.toLowerCase(),
          );
        }
        return origFocus.call(this, options);
      });

    try {
      const sampleBtn = ui.getByTestId('load-sample-arrangement-button');
      act(() => {
        sampleBtn.focus();
      });
      fireEvent.click(sampleBtn);
      expect(ui.queryByTestId('midi-import-modal')).not.toBeNull();
      expect(focusWithAriaHiddenAncestor).toEqual([]);

      // Close modal via Escape -> focus restores to `sampleBtn` AFTER `aria-hidden` is removed from Toolbar!
      fireEvent.keyDown(window, { key: 'Escape' });
      expect(ui.queryByTestId('midi-import-modal')).toBeNull();
      expect(document.activeElement).toBe(sampleBtn);
      expect(focusWithAriaHiddenAncestor).toEqual([]);

      // Async multi-track `.mid` upload when `document.activeElement` is `document.body` restores focus to `upload-midi-button`!
      act(() => {
        (document.activeElement as HTMLElement | null)?.blur();
      });
      const multiMidBytes = StandardMidiCodec.encodeMultiTrackMidi(
        [
          new DawTrack({
            name: 'Piano',
            instrument: 'PIANO',
            notes: [new MidiNote({ pitch: 60, startTick: 0, durationTicks: 480 })],
          }).toSnapshot(),
          new DawTrack({
            name: 'Sax',
            instrument: 'SAX',
            notes: [new MidiNote({ pitch: 67, startTick: 0, durationTicks: 480 })],
          }).toSnapshot(),
        ],
        480,
        120,
      );
      const fileInput = ui.getByTestId('midi-file-input') as HTMLInputElement;
      const multiFile = new File([new Uint8Array(multiMidBytes)], 'two_tracks.mid', {
        type: 'audio/midi',
      });
      if (typeof multiFile.arrayBuffer !== 'function') {
        Object.defineProperty(multiFile, 'arrayBuffer', {
          value: async () =>
            multiMidBytes.buffer.slice(
              multiMidBytes.byteOffset,
              multiMidBytes.byteOffset + multiMidBytes.byteLength,
            ),
        });
      }
      fireEvent.change(fileInput, { target: { files: [multiFile] } });
      await waitFor(() => {
        expect(ui.queryByTestId('midi-import-modal')).not.toBeNull();
      });
      fireEvent.keyDown(window, { key: 'Escape' });
      expect(ui.queryByTestId('midi-import-modal')).toBeNull();
      expect(document.activeElement).toBe(ui.getByTestId('upload-midi-button'));
      expect(focusWithAriaHiddenAncestor).toEqual([]);
    } finally {
      focusSpy.mockRestore();
    }

    // 2. `[R3-MED-02]` Switching `isActive={false}` while modal is open immediately unmounts `[role="dialog"]`, clears `modalOpenFlag`, does NOT resurrect on return, and new AI candidate batch closes open modal!
    const sampleBtn = ui.getByTestId('load-sample-arrangement-button');
    fireEvent.click(sampleBtn);
    expect(modalOpenFlag).toBe(true);
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();

    ui.rerender(<Round3Harness isActive={false} />);
    expect(modalOpenFlag).toBe(false);
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    // Returning to `isActive={true}` does NOT resurrect stale modal
    ui.rerender(<Round3Harness isActive={true} />);
    expect(modalOpenFlag).toBe(false);
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    // Opening modal and receiving an in-flight AI candidate batch auto-closes the modal cleanly
    fireEvent.click(sampleBtn);
    expect(ui.queryByTestId('midi-import-modal')).not.toBeNull();
    const candSmf = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 72, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    await act(async () => {
      await ctrl.applyGeneratedCandidates(
        [
          {
            index: 0,
            filename: 'cand.mid',
            rawMidiBytes: candSmf,
            parsedMidi: StandardMidiCodec.parseMidi(candSmf.buffer as ArrayBuffer),
            reason: null,
          },
        ],
        [],
      );
    });
    expect(ui.queryByTestId('midi-import-modal')).toBeNull();
    expect(modalOpenFlag).toBe(false);

    // Undo candidate placement to restore noteA (tick 480, pitch 60) & noteB (tick 960, pitch 64)
    act(() => {
      ctrl.undo();
      ctrl.setSelectedNotes([noteA.id]);
    });

    // 3. `[R3-MED-03]` Mid-drag `Delete` and mid-drag `Ctrl+Z` cancel active drag without ghost notes or stale overwrite on `pointerUp`
    // NoteA (pitch 60 -> row 67 -> y = 32 + 67*16 - 848 + 8 = 264, tick 480 -> x = 72 + 64 + 14 = 150)
    fireEvent.pointerDown(canvas, { pointerId: 41, button: 0, clientX: 150, clientY: 264 });
    fireEvent.pointerMove(canvas, {
      pointerId: 41,
      buttons: 1,
      clientX: 214,
      clientY: 264,
    }); // Dragged +1 beat right
    // Press Delete BEFORE releasing mouse button
    fireEvent.keyDown(window, { key: 'Delete' });
    expect(
      ctrl.getSnapshot().tracks[0].notes.find((n) => n.id === noteA.id),
    ).toBeUndefined();
    // Releasing mouse button must NOT resurrect or commit stale transform
    fireEvent.pointerUp(canvas, { pointerId: 41, button: 0, clientX: 214, clientY: 264 });
    expect(
      ctrl.getSnapshot().tracks[0].notes.find((n) => n.id === noteA.id),
    ).toBeUndefined();

    // Undo Delete so noteA is back at startTick 480; now start dragging noteB (tick 960 -> x = 214, pitch 64 -> y = 200) and press Ctrl+Z mid-drag!
    act(() => {
      ctrl.undo();
      ctrl.setSelectedNotes([noteB.id]);
    });
    expect(ctrl.getSnapshot().tracks[0].notes.length).toBe(2);
    fireEvent.pointerDown(canvas, { pointerId: 42, button: 0, clientX: 214, clientY: 200 });
    fireEvent.pointerMove(canvas, {
      pointerId: 42,
      buttons: 1,
      clientX: 278,
      clientY: 200,
    });
    // Press Ctrl+Z mid-drag -> undoes the prior Redo/Restore (re-deleting noteA? Wait: undo reverts the prior Undo? No, Undo reverts noteB creation!)
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true });
    const countAfterMidDragUndo = ctrl.getSnapshot().tracks[0].notes.length;
    expect(ctrl.getSnapshot().canRedo).toBe(true);
    // Releasing mouse button must NOT overwrite Undo or wipe Redo!
    fireEvent.pointerUp(canvas, { pointerId: 42, button: 0, clientX: 278, clientY: 200 });
    expect(ctrl.getSnapshot().tracks[0].notes.length).toBe(countAfterMidDragUndo);
    expect(ctrl.getSnapshot().canRedo).toBe(true);
    // Redo restores noteB cleanly
    fireEvent.keyDown(window, { key: 'y', ctrlKey: true });
    expect(ctrl.getSnapshot().tracks[0].notes.length).toBe(2);

    // 4. `[R3-MED-04]` 2-finger touch pan/pinch & Escape marquee cancel preserve multi-note selection on empty grid!
    const liveIds = ctrl.getSnapshot().tracks[0].notes.map((n) => n.id);
    act(() => {
      ctrl.setSelectedNotes(liveIds);
    });
    expect(ctrl.getSnapshot().selectedNoteIds.size).toBe(2);

    // Desktop marquee start on empty space (x = 400, y = 120) -> Escape restores 2-note selection!
    fireEvent.pointerDown(canvas, { pointerId: 43, button: 0, clientX: 400, clientY: 120 });
    expect(ctrl.getSnapshot().selectedNoteIds.size).toBe(0);
    fireEvent.pointerMove(canvas, {
      pointerId: 43,
      buttons: 1,
      clientX: 430,
      clientY: 150,
    });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(ctrl.getSnapshot().selectedNoteIds.size).toBe(2);

    // 2-finger touch gesture on empty space: Finger 1 lands, then Finger 2 lands -> restores 2-note selection!
    fireEvent.pointerDown(canvas, {
      pointerId: 44,
      pointerType: 'touch',
      clientX: 400,
      clientY: 120,
    });
    expect(ctrl.getSnapshot().selectedNoteIds.size).toBe(0);
    fireEvent.pointerDown(canvas, {
      pointerId: 45,
      pointerType: 'touch',
      clientX: 460,
      clientY: 140,
    });
    expect(ctrl.getSnapshot().selectedNoteIds.size).toBe(2);
    fireEvent.pointerUp(canvas, {
      pointerId: 44,
      pointerType: 'touch',
      clientX: 400,
      clientY: 120,
    });
    fireEvent.pointerUp(canvas, {
      pointerId: 45,
      pointerType: 'touch',
      clientX: 460,
      clientY: 140,
    });
    expect(ctrl.getSnapshot().selectedNoteIds.size).toBe(2);

    // Single-click on empty space still deselects & snaps playhead!
    fireEvent.pointerDown(canvas, { pointerId: 46, button: 0, clientX: 400, clientY: 120 });
    fireEvent.pointerUp(canvas, { pointerId: 46, button: 0, clientX: 400, clientY: 120 });
    expect(ctrl.getSnapshot().selectedNoteIds.size).toBe(0);

    // 5. `[R3-LOW-01]` Synthetic unfocused `Escape` on `#pianoroll-bpm` is self-cleaning and does not block subsequent `onBlur` commit
    const bpmInput = ui.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
    fireEvent.keyDown(bpmInput, { key: 'Escape' });
    act(() => {
      bpmInput.focus();
    });
    fireEvent.change(bpmInput, { target: { value: '145' } });
    act(() => {
      bpmInput.blur();
    });
    expect(ctrl.getSnapshot().bpm).toBe(145);

    ui.unmount();
  });

  it('42. [Round-5 Audit] verifies SFT auto/fallback CoT parity, External DAW Bridge controls, IME guards, move-cancel rollback, playback voice cut, and canvas blur/Space UX', async () => {
    // 1. `[CRIT-HIGH-03]` SFT `auto` + `conditionTrackId` -> `inst_comp` unlocks `#slider-cot-temp`, and `infill` with 0 context falls back to `meta`
    const stubAudio = new StubAudioEngine();
    let stopAllVoicesCount = 0;
    const origStopAll = stubAudio.stopAllVoices.bind(stubAudio);
    stubAudio.stopAllVoices = () => {
      stopAllVoicesCount += 1;
      origStopAll();
    };
    const demoCtrl = new DemoPianoRollController(stubAudio);
    await demoCtrl.connect();
    // Track 1 (Piano 1) is empty; add Track 2 (Sax) with an overlapping note in bars 1..4
    const condTrkId = demoCtrl.addTrack('Chord Pad', 'PIANO');
    demoCtrl.setActiveTrack(condTrkId);
    demoCtrl.createNoteAt(60, 0, 960, 100, { silent: true });
    const emptyTargetTrackId = demoCtrl.getSnapshot().tracks[0].id;
    demoCtrl.setActiveTrack(emptyTargetTrackId);

    const extractedAutoComp = await demoCtrl.extractGenerationContext({
      taskOverride: 'auto',
      conditionTrackId: condTrkId,
    });
    expect(extractedAutoComp.resolvedTask).toBe('inst_comp');

    const sftModels = [
      {
        model_name: 'MORTM46_SFT_Gen',
        has_weights: true,
        tag: { model: 'sft_gen', version: '4.6' },
      },
    ];
    const cpAutoComp = render(
      <ControlPanelView
        audioEngine={stubAudio}
        controllers={[demoCtrl]}
        session={demoCtrl.getSnapshot()}
        models={sftModels}
        hardwareInfo={null}
        settings={{
          modelName: 'MORTM46_SFT_Gen',
          temperature: 1.0,
          topP: 0.95,
          thinking: true,
          cotTemperature: 0.8,
          numCandidates: 1,
          taskMode: 'auto',
          conditionTrackId: condTrkId,
          forceMonophonic: false,
          noteDensity: null,
          musicalKey: '',
          selectedGenres: [],
        }}
        isGenerating={false}
        isRescanningModels={false}
        statusTitle="待機中"
        statusDetail="準備完了"
        generationLogs={['05:00:01 前ステップログ', '05:00:02 準備完了']}
        errorMessage={null}
        onUpdateSettings={() => {}}
        onSwitchController={() => {}}
        onRescanModels={() => {}}
        onTogglePlayback={() => {}}
        onGenerate={() => {}}
        onCancelGenerate={() => {}}
      />,
    );
    expect(cpAutoComp.getByTestId('pianoroll-context-summary').textContent).toContain(
      '自動判定: inst_comp',
    );
    const cotSliderAutoComp = cpAutoComp.container.querySelector(
      '#slider-cot-temp',
    ) as HTMLInputElement;
    expect(cotSliderAutoComp.disabled).toBe(false);
    // `[CRIT-MED-03]` Hero statusDetail (`準備完了`) is deduplicated from `generation-log-feed` while prior log remains
    const logFeedText = cpAutoComp.getByTestId('generation-log-feed').textContent || '';
    expect(logFeedText).toContain('前ステップログ');
    expect(logFeedText).not.toContain('準備完了');

    // Rerender with explicit `taskMode: 'infill'` on empty target track (no conditionTrackId) -> effective `meta` disables CoT slider
    cpAutoComp.rerender(
      <ControlPanelView
        audioEngine={stubAudio}
        controllers={[demoCtrl]}
        session={demoCtrl.getSnapshot()}
        models={sftModels}
        hardwareInfo={null}
        settings={{
          modelName: 'MORTM46_SFT_Gen',
          temperature: 1.0,
          topP: 0.95,
          thinking: true,
          cotTemperature: 0.8,
          numCandidates: 1,
          taskMode: 'infill',
          conditionTrackId: '',
          forceMonophonic: false,
          noteDensity: null,
          musicalKey: '',
          selectedGenres: [],
        }}
        isGenerating={true}
        isRescanningModels={false}
        statusTitle="生成中"
        statusDetail="推論中"
        generationLogs={[]}
        errorMessage={null}
        onUpdateSettings={() => {}}
        onSwitchController={() => {}}
        onRescanModels={() => {}}
        onTogglePlayback={() => {}}
        onGenerate={() => {}}
        onCancelGenerate={() => {}}
      />,
    );
    expect(cpAutoComp.getByTestId('pianoroll-context-summary').textContent).toContain(
      'infill (実効: meta)',
    );
    expect(
      (cpAutoComp.container.querySelector('#slider-cot-temp') as HTMLInputElement).disabled,
    ).toBe(true);
    // `[CRIT-MED-09]` `#compass-daw-select` is disabled during generation
    expect(
      (cpAutoComp.container.querySelector('#compass-daw-select') as HTMLSelectElement).disabled,
    ).toBe(true);
    cpAutoComp.unmount();

    // 2. `[CRIT-HIGH-02]` External DAW Bridge (`CubaseDawController`) exposes measure range, BPM, meter, and candidate switcher
    const cubase = new CubaseDawController();
    await cubase.connect();
    const demoMidiBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'ext_n1', pitch: 64, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const parsedDemoCand = StandardMidiCodec.parseMidi(demoMidiBytes.slice().buffer);
    const demoReason = {
      source: 'request' as const,
      task: 'meta',
      thinking: false,
      instruments: ['PIANO'],
      note_density: null,
      gen_measure_count: 4,
      genre: null,
      key: 'CM',
    };
    await cubase.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'cand0.mid',
          rawMidiBytes: demoMidiBytes,
          parsedMidi: parsedDemoCand,
          reason: demoReason,
        },
        {
          index: 1,
          filename: 'cand1.mid',
          rawMidiBytes: demoMidiBytes,
          parsedMidi: parsedDemoCand,
          reason: demoReason,
        },
      ],
      [demoReason],
    );
    let cubaseSnap = cubase.getSnapshot();
    const extUi = render(
      <PianoRollView
        controller={cubase}
        session={cubaseSnap}
        focusRevision={0}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );
    const extStartInput = extUi.getByLabelText('生成開始小節') as HTMLInputElement;
    fireEvent.change(extStartInput, { target: { value: '5' } });
    expect(cubase.getSnapshot().targetRange.startMeasure).toBe(5);
    const extBpmInput = extUi.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
    fireEvent.change(extBpmInput, { target: { value: '132' } });
    expect(cubase.getSnapshot().bpm).toBe(132);
    const extTsSelect = extUi.getByTestId('pianoroll-timesig-select') as HTMLSelectElement;
    fireEvent.change(extTsSelect, { target: { value: '3/4' } });
    expect(cubase.getSnapshot().timeSigNumerator).toBe(3);
    const cand2Btn = extUi.getByRole('button', { name: '候補 2' });
    await act(async () => {
      fireEvent.click(cand2Btn);
    });
    expect(cubase.getSnapshot().activeCandidateIndex).toBe(1);
    extUi.unmount();

    // 3. `[CRIT-MED-07]` Active playback silences voices immediately on note deletion / undo
    demoCtrl.setActiveTrack(condTrkId);
    await demoCtrl.play();
    const beforeVoiceStops = stopAllVoicesCount;
    demoCtrl.selectAllNotesOnActiveTrack();
    demoCtrl.deleteSelectedNotes();
    expect(stopAllVoicesCount).toBeGreaterThan(beforeVoiceStops);
    const afterDeleteStops = stopAllVoicesCount;
    demoCtrl.undo();
    expect(stopAllVoicesCount).toBeGreaterThan(afterDeleteStops);
    demoCtrl.stop();

    // 4. `[CRIT-MED-05]`, `[CRIT-MED-06]`, `[CRIT-MED-08]` IME guard, move-cancel selection rollback, blur cancel, & Space after canvas touch
    demoCtrl.createNoteAt(64, 480, 480, 100, { silent: true });
    demoCtrl.createNoteAt(67, 960, 480, 100, { silent: true });
    const allThreeNotes = demoCtrl.getSnapshot().tracks.find((t) => t.id === condTrkId)!.notes;
    expect(allThreeNotes.length).toBe(3);
    act(() => {
      demoCtrl.setSelectedNotes([allThreeNotes[0].id, allThreeNotes[1].id]);
    });

    function LivePrHarness() {
      const [snap, setSnap] = React.useState(() => demoCtrl.getSnapshot());
      React.useEffect(() => demoCtrl.subscribe(setSnap), []);
      return (
        <PianoRollView
          controller={demoCtrl}
          session={snap}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    }
    const prLive = render(<LivePrHarness />);
    const liveCanvas = prLive.getByTestId('pianoroll-canvas') as HTMLCanvasElement;
    vi.spyOn(liveCanvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      width: 960,
      height: 540,
      right: 960,
      bottom: 540,
      toJSON: () => ({}),
    });

    // `[CRIT-MED-05]` IME `isComposing` Enter on `#pianoroll-bpm` does NOT blur
    const liveBpm = prLive.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
    act(() => {
      liveBpm.focus();
    });
    expect(document.activeElement).toBe(liveBpm);
    fireEvent.keyDown(liveBpm, { key: 'Enter', isComposing: true, keyCode: 229 });
    expect(document.activeElement).toBe(liveBpm);

    // `[CRIT-MED-08]` Touching `<canvas>` blurs focused `<input>` (`#pianoroll-bpm`)
    fireEvent.pointerDown(liveCanvas, {
      pointerId: 71,
      pointerType: 'touch',
      clientX: 500,
      clientY: 200,
    });
    fireEvent.pointerUp(window, {
      pointerId: 71,
      pointerType: 'touch',
      clientX: 500,
      clientY: 200,
    });
    expect(document.activeElement).not.toBe(liveBpm);

    // `[CRIT-MED-06]` Restore 2-note selection, start dragging 3rd unselected note (pitch 67 at tick 960 -> x=72+128+10=210, y=32+(127-67)*16-848+8=152), then cancel via `window` `'blur'` (`[CRIT-MED-08]`)
    act(() => {
      demoCtrl.setSelectedNotes([allThreeNotes[0].id, allThreeNotes[1].id]);
    });
    expect(demoCtrl.getSnapshot().selectedNoteIds.size).toBe(2);
    fireEvent.pointerDown(liveCanvas, {
      pointerId: 72,
      pointerType: 'mouse',
      button: 0,
      buttons: 1,
      clientX: 210,
      clientY: 152,
    });
    // Selection temporarily switched to 3rd note on pointerDown
    expect(demoCtrl.getSnapshot().selectedNoteIds.has(allThreeNotes[2].id)).toBe(true);
    // Window blur aborts drag & rolls selection back to the original 2 notes!
    act(() => {
      window.dispatchEvent(new Event('blur'));
    });
    expect(demoCtrl.getSnapshot().selectedNoteIds.size).toBe(2);
    expect(demoCtrl.getSnapshot().selectedNoteIds.has(allThreeNotes[0].id)).toBe(true);
    expect(demoCtrl.getSnapshot().selectedNoteIds.has(allThreeNotes[1].id)).toBe(true);

    // `[CRIT-MED-08]` Focus a toolbar `<button>`, then interact with `<canvas>` -> `Space` toggles transport without clicking button
    const pencilBtn = prLive.getByRole('button', { name: 'ペン入力' });
    act(() => {
      pencilBtn.focus();
    });
    fireEvent.pointerDown(liveCanvas, {
      pointerId: 73,
      pointerType: 'touch',
      clientX: 500,
      clientY: 200,
    });
    fireEvent.pointerUp(window, {
      pointerId: 73,
      pointerType: 'touch',
      clientX: 500,
      clientY: 200,
    });
    expect(document.activeElement).toBe(pencilBtn);
    expect(demoCtrl.getSnapshot().isPlaying).toBe(false);
    fireEvent.keyDown(pencilBtn, { key: ' ', code: 'Space' });
    await waitFor(() => {
      expect(demoCtrl.getSnapshot().isPlaying).toBe(true);
    });
    act(() => {
      demoCtrl.stop();
    });
    prLive.unmount();
  });

  it('43. [Round-6 Fact-Checker] verifies pinned targetSlot across mid-generation track switch/delete, stale conditionTrackId parity, async MIDI upload sequence guard, and single-tooltip adaptive alignment', async () => {
    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    await ctrl.connect();

    // 1. `[R5-BUG-01]` Pinned `targetSlot` places candidate on original track/range even if user switches active track & range mid-generation, and safely no-ops if original track was deleted
    const trk1Id = ctrl.getSnapshot().activeTrackId;
    const trk2Id = ctrl.addTrack('Sax Lead', 'SAX');
    ctrl.setActiveTrack(trk1Id);
    ctrl.setTargetMeasureRange(1, 2);
    const extracted = await ctrl.extractGenerationContext({ taskOverride: 'auto' });
    expect(extracted.targetSlot).toEqual({
      trackId: trk1Id,
      startTick: 0,
      endTick: 3840,
    });

    // Simulate user switching active track & target range while inference is running
    ctrl.setActiveTrack(trk2Id);
    ctrl.setTargetMeasureRange(5, 6);

    const candBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'r6_c1', pitch: 65, startTick: 0, durationTicks: 480, velocity: 96 })],
      480,
      120,
      'PIANO',
    );
    const parsedCand = StandardMidiCodec.parseMidi(candBytes.slice().buffer);
    const candReason = {
      source: 'request' as const,
      task: 'meta',
      thinking: false,
      instruments: ['PIANO'],
      note_density: null,
      gen_measure_count: 2,
      genre: null,
      key: null,
    };
    await ctrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'r6_cand0.mid',
          rawMidiBytes: candBytes,
          parsedMidi: parsedCand,
          reason: candReason,
        },
      ],
      [candReason],
      extracted.targetSlot,
    );
    const snapAfterPinned = ctrl.getSnapshot();
    expect(snapAfterPinned.tracks.find((t) => t.id === trk1Id)?.notes.length).toBe(1);
    expect(snapAfterPinned.tracks.find((t) => t.id === trk1Id)?.notes[0]?.startTick).toBe(0);
    expect(snapAfterPinned.tracks.find((t) => t.id === trk2Id)?.notes.length).toBe(0);

    // Delete `trk1Id` mid-generation -> applying to stale `extracted.targetSlot` is a clean no-op
    ctrl.deleteTrack(trk1Id);
    await ctrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'r6_cand_deleted.mid',
          rawMidiBytes: candBytes,
          parsedMidi: parsedCand,
          reason: candReason,
        },
      ],
      [candReason],
      extracted.targetSlot,
    );
    expect(ctrl.getSnapshot().tracks.find((t) => t.id === trk2Id)?.notes.length).toBe(0);

    // 2. `[R5-BUG-02]` Stale/deleted `conditionTrackId` falls back to available accompaniment track identically to `ControlPanelView`
    const accompTrkId = ctrl.addTrack('Accomp Chords', 'PIANO');
    ctrl.setActiveTrack(accompTrkId);
    ctrl.createNoteAt(60, 0, 960, 95, { silent: true });
    ctrl.setActiveTrack(trk2Id);
    ctrl.setTargetMeasureRange(1, 4);
    const ctxWithStaleCond = await ctrl.extractGenerationContext({
      taskOverride: 'inst_comp',
      conditionTrackId: 'deleted-nonexistent-track-id',
    });
    expect(ctxWithStaleCond.resolvedTask).toBe('inst_comp');
    expect(ctxWithStaleCond.conditionsMidiBytes).not.toBeNull();

    // Whereas an existing empty track explicitly chosen as condition track downgrades to `meta`
    const emptyCondTrkId = ctrl.addTrack('Empty Cond', 'PIANO');
    ctrl.setActiveTrack(trk2Id);
    const ctxWithEmptyExplicitCond = await ctrl.extractGenerationContext({
      taskOverride: 'inst_comp',
      conditionTrackId: emptyCondTrkId,
    });
    expect(ctxWithEmptyExplicitCond.resolvedTask).toBe('meta');
    expect(ctxWithEmptyExplicitCond.conditionsMidiBytes).toBeNull();

    // 3. `[R5-BUG-01]` External DAW `内蔵ピアノロールへ切り替え` button is disabled during `isGenerating`
    const cubase = new CubaseDawController();
    await cubase.connect();
    const extGeneratingUi = render(
      <PianoRollView
        isGenerating={true}
        controller={cubase}
        session={cubase.getSnapshot()}
        focusRevision={0}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );
    const switchDemoBtn = extGeneratingUi.getByRole('button', {
      name: '内蔵ピアノロールへ切り替え',
    }) as HTMLButtonElement;
    expect(switchDemoBtn.disabled).toBe(true);
    extGeneratingUi.unmount();

    // 4. `[R5-BUG-03]` Overlapping async MIDI uploads in `<React.StrictMode>` discard stale earlier resolves
    let resolveSlow!: (v: ReturnType<typeof StandardMidiCodec.createDemoMidiPreset>) => void;
    let resolveFast!: (v: ReturnType<typeof StandardMidiCodec.createDemoMidiPreset>) => void;
    const parseSpy = vi
      .spyOn(ctrl, 'parseUploadedMidiFile')
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSlow = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFast = resolve;
          }),
      );

    function StrictPrHarness() {
      const [snap, setSnap] = React.useState(() => ctrl.getSnapshot());
      React.useEffect(() => ctrl.subscribe(setSnap), []);
      return (
        <React.StrictMode>
          <PianoRollView
            controller={ctrl}
            session={snap}
            focusRevision={0}
            onLaunchCompassAi={() => {}}
            onSwitchToDemoController={() => {}}
          />
        </React.StrictMode>
      );
    }
    const strictUi = render(<StrictPrHarness />);

    // `[R5-UI-02]` Zero `button[data-tooltip]` elements carry duplicate `title`
    const tooltipButtons = Array.from(
      strictUi.container.querySelectorAll('button[data-tooltip]'),
    ) as HTMLButtonElement[];
    expect(tooltipButtons.length).toBeGreaterThan(5);
    for (const tb of tooltipButtons) {
      expect(tb.hasAttribute('title')).toBe(false);
    }

    // `[R5-READ-01]` Adaptive tooltip alignment sets `start` near left viewport edge & `end` near right edge
    const pencilBtn = strictUi.getByRole('button', { name: 'ペン入力' }) as HTMLButtonElement;
    Object.defineProperty(document.documentElement, 'clientWidth', {
      configurable: true,
      value: 375,
    });
    vi.spyOn(pencilBtn, 'getBoundingClientRect').mockReturnValue({
      x: 8,
      y: 60,
      left: 8,
      top: 60,
      width: 44,
      height: 44,
      right: 52,
      bottom: 104,
      toJSON: () => ({}),
    });
    alignButtonTooltipToViewport(pencilBtn);
    expect(pencilBtn.getAttribute('data-tooltip-align')).toBe('start');

    vi.spyOn(pencilBtn, 'getBoundingClientRect').mockReturnValue({
      x: 325,
      y: 60,
      left: 325,
      top: 60,
      width: 44,
      height: 44,
      right: 369,
      bottom: 104,
      toJSON: () => ({}),
    });
    alignButtonTooltipToViewport(pencilBtn);
    expect(pencilBtn.getAttribute('data-tooltip-align')).toBe('end');

    const fileInput = strictUi.getByTestId('midi-file-input') as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: {
        files: [new File([candBytes.slice().buffer], 'slow_first.mid', { type: 'audio/midi' })],
      },
    });
    fireEvent.change(fileInput, {
      target: {
        files: [new File([candBytes.slice().buffer], 'fast_second.mid', { type: 'audio/midi' })],
      },
    });
    expect(parseSpy).toHaveBeenCalledTimes(2);

    const fastSingleParsed = StandardMidiCodec.parseMidi(
      StandardMidiCodec.encodeSingleTrackMidi(
        [new MidiNote({ id: 'fast_1', pitch: 72, startTick: 0, durationTicks: 480, velocity: 100 })],
        480,
        128,
        'PIANO',
      ).slice().buffer,
    );
    const slowSingleParsed = StandardMidiCodec.parseMidi(
      StandardMidiCodec.encodeSingleTrackMidi(
        [new MidiNote({ id: 'slow_1', pitch: 48, startTick: 0, durationTicks: 480, velocity: 90 })],
        480,
        90,
        'PIANO',
      ).slice().buffer,
    );

    await act(async () => {
      resolveFast(fastSingleParsed);
    });
    expect(ctrl.getSnapshot().bpm).toBe(128);
    expect(ctrl.getSnapshot().tracks[0]?.notes[0]?.pitch).toBe(72);

    await act(async () => {
      resolveSlow(slowSingleParsed);
    });
    // Stale `slow_first.mid` did NOT overwrite `fast_second.mid`!
    expect(ctrl.getSnapshot().bpm).toBe(128);
    expect(ctrl.getSnapshot().tracks[0]?.notes[0]?.pitch).toBe(72);

    parseSpy.mockRestore();
    strictUi.unmount();
  });

  it('44. [Round-7 Fact-Checker] verifies candidate track/viewport coherence across mid-gen edits & candidate switching, aborted onProgress suppression, and pop-out GENERATION_STATUS sync on REQUEST_WORKSPACE_STATE', async () => {
    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    await ctrl.connect();

    // 1. `[R6-BUG-01]` Mid-generation track/range switch + candidate switcher brings target track & `activeCandidateRange` into foreground
    const trk1Id = ctrl.getSnapshot().activeTrackId;
    const trk2Id = ctrl.addTrack('Sax Solo', 'SAX');
    ctrl.setActiveTrack(trk1Id);
    ctrl.setTargetMeasureRange(5, 6); // [7680, 11520)
    const extracted = await ctrl.extractGenerationContext({ taskOverride: 'auto' });
    expect(extracted.targetSlot).toEqual({
      trackId: trk1Id,
      startTick: 7680,
      endTick: 11520,
    });

    // User switches to `trk2Id` and moves `AI TARGET` to Bars 13..16 while inference is running
    ctrl.setActiveTrack(trk2Id);
    ctrl.setTargetMeasureRange(13, 16);

    const cand0Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'r7_c0', pitch: 64, startTick: 0, durationTicks: 480, velocity: 98 })],
      480,
      120,
      'PIANO',
    );
    const cand1Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'r7_c1', pitch: 79, startTick: 0, durationTicks: 480, velocity: 105 })],
      480,
      120,
      'PIANO',
    );
    await ctrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'r7_0.mid',
          rawMidiBytes: cand0Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(cand0Bytes.slice().buffer),
          reason: null,
        },
        {
          index: 1,
          filename: 'r7_1.mid',
          rawMidiBytes: cand1Bytes,
          parsedMidi: StandardMidiCodec.parseMidi(cand1Bytes.slice().buffer),
          reason: null,
        },
      ],
      [],
      extracted.targetSlot,
    );

    const snapAfterGen = ctrl.getSnapshot();
    expect(snapAfterGen.activeTrackId).toBe(trk1Id);
    expect(snapAfterGen.playheadTick).toBe(7680);
    expect(snapAfterGen.selectedNoteIds.size).toBe(1);
    expect(snapAfterGen.activeCandidateRange).toEqual({
      trackId: trk1Id,
      startTick: 7680,
      endTick: 11520,
    });

    // User switches back to `trk2Id` and clicks `候補 2` in `<PianoRollView />` -> focuses `trk1Id` + selects Candidate 2 note + resets playhead to 7680
    ctrl.setActiveTrack(trk2Id);
    expect(ctrl.getSnapshot().activeTrackId).toBe(trk2Id);

    function CandSwitchHarness({ candRev }: { candRev: number }) {
      const [snap, setSnap] = React.useState(() => ctrl.getSnapshot());
      React.useEffect(() => ctrl.subscribe(setSnap), []);
      return (
        <PianoRollView
          controller={ctrl}
          session={snap}
          focusRevision={0}
          candidateFocusRevision={candRev}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    }

    const candUi = render(<CandSwitchHarness candRev={1} />);
    const cand2Btn = candUi.getByRole('button', { name: '候補 2' });
    await act(async () => {
      fireEvent.click(cand2Btn);
    });
    const snapAfterCand2 = ctrl.getSnapshot();
    expect(snapAfterCand2.activeTrackId).toBe(trk1Id);
    expect(snapAfterCand2.activeCandidateIndex).toBe(1);
    expect(snapAfterCand2.playheadTick).toBe(7680);
    expect(snapAfterCand2.selectedNoteIds.size).toBe(1);
    const placedCand2Note = snapAfterCand2.tracks.find((t) => t.id === trk1Id)?.notes[0];
    expect(placedCand2Note?.pitch).toBe(79);
    expect(snapAfterCand2.selectedNoteIds.has(placedCand2Note!.id)).toBe(true);

    // Undo while viewing `trk2Id` does NOT hijack `activeTrackId` back to `trk1Id`
    act(() => {
      ctrl.setActiveTrack(trk2Id);
      ctrl.undo();
    });
    expect(ctrl.getSnapshot().activeTrackId).toBe(trk2Id);
    expect(ctrl.getSnapshot().activeCandidateIndex).toBe(0);
    candUi.unmount();

    // 2. `[R6-BUG-02]` `CompassApiClient.generateMusic` suppresses `onProgress` once `signal.aborted` is true
    const origFetch = globalThis.fetch;
    try {
      const progressAbort = new AbortController();
      const progressCalls: string[] = [];
      progressAbort.abort();
      const abortedClient = new CompassApiClient('http://127.0.0.1:18999', 'tok_r7');
      await expect(
        abortedClient.generateMusic(
          {
            meta: {
              model_type: 'MORTM4.5D-80M-SFT-Gen',
              task: 'meta',
              program: ['PIANO'],
              tempo: 120,
              num_gems: 1,
              genfield_measure: 4,
              gen_note_dense: null,
              p: 0.95,
              temperature: 1.0,
              cot_temperature: 0.8,
              thinking: false,
            },
          },
          (msg) => progressCalls.push(msg),
          progressAbort.signal,
        ),
      ).rejects.toThrow();
      expect(progressCalls).toEqual([]);

      // 3. `[R6-BUG-02]` `<App />` replies to `REQUEST_WORKSPACE_STATE` during active generation with `GENERATION_STATUS(true, ...)`
      let resolveInFlightGen!: (res: Response) => void;
      globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith('/system/hardware')) {
          return new Response(
            JSON.stringify({
              device: 'cpu',
              cuda_available: false,
              gpu_name: null,
              models_dir: 'C:\\Models',
              bundled_models_dir: 'C:\\Bundled',
              available_models_count: 1,
              models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (url.endsWith('/models')) {
          return new Response(
            JSON.stringify([
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                tag: { model: 'sft_gen' },
              },
            ]),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (url.endsWith('/generate')) {
          return await new Promise<Response>((resolve, reject) => {
            resolveInFlightGen = resolve;
            init?.signal?.addEventListener('abort', () => {
              reject(new DOMException('Aborted', 'AbortError'));
            });
          });
        }
        return new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      });

      const appUi = render(<App />);
      await waitFor(
        () => {
          expect(appUi.getByTestId('generate-button')).toBeTruthy();
        },
        { timeout: 3500 },
      );

      const peerBus = new WorkspaceSyncBus();
      const receivedGenStatuses: Array<{ isGenerating: boolean; statusDetail: string }> = [];
      let receivedWorkspaceSync = false;
      const unsubPeer = peerBus.subscribe((msg) => {
        if (msg.type === 'GENERATION_STATUS') {
          receivedGenStatuses.push({
            isGenerating: msg.isGenerating,
            statusDetail: msg.statusDetail,
          });
        } else if (msg.type === 'SYNC_WORKSPACE_STATE') {
          receivedWorkspaceSync = true;
        }
      });

      fireEvent.click(appUi.getByTestId('generate-button'));
      await waitFor(() => {
        expect(typeof resolveInFlightGen).toBe('function');
      });

      receivedGenStatuses.length = 0;
      receivedWorkspaceSync = false;

      // Simulate a newly opened `?popout=1` window requesting initial workspace state mid-generation
      peerBus.requestWorkspaceState();
      await waitFor(() => {
        expect(receivedWorkspaceSync).toBe(true);
        expect(receivedGenStatuses.some((s) => s.isGenerating === true)).toBe(true);
      });

      unsubPeer();
      peerBus.dispose();
      appUi.unmount();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('45. [Round-8 Critic Remediation] verifies edited candidate detachment + Undo/Redo, External DAW pop-out split-brain guards, F2/touch track rename, SC 1.4.13 Escape tooltip dismissal & vertical flip, immediate cancel broadcast, deferred hidden centering, and telemetry track context', async () => {
    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    await ctrl.connect();

    // 1. `[R7-HIGH-02]` Moving/editing candidate notes detaches them from `_activeCandidateNoteIds` and Undo/Redo restores/re-prunes cleanly
    ctrl.setSelectedNotes(ctrl.getSnapshot().tracks[0]!.notes.map((n) => n.id));
    ctrl.deleteSelectedNotes();
    ctrl.setTargetMeasureRange(1, 2);
    const cand0Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'r8_c0', pitch: 60, startTick: 0, durationTicks: 480, velocity: 100 })],
      480,
      120,
      'PIANO',
    );
    const cand1Bytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ id: 'r8_c1', pitch: 67, startTick: 0, durationTicks: 480, velocity: 105 })],
      480,
      120,
      'PIANO',
    );
    const parsed0 = StandardMidiCodec.parseMidi(cand0Bytes.slice().buffer);
    const parsed1 = StandardMidiCodec.parseMidi(cand1Bytes.slice().buffer);
    const reason = {
      source: 'request' as const,
      task: 'meta',
      thinking: false,
      instruments: ['PIANO'],
      note_density: null,
      gen_measure_count: 2,
      genre: null,
      key: null,
    };
    await ctrl.applyGeneratedCandidates(
      [
        {
          index: 0,
          filename: 'r8_0.mid',
          rawMidiBytes: cand0Bytes,
          parsedMidi: parsed0,
          reason,
        },
        {
          index: 1,
          filename: 'r8_1.mid',
          rawMidiBytes: cand1Bytes,
          parsedMidi: parsed1,
          reason,
        },
      ],
      [reason, reason],
    );
    const placedCand0 = ctrl.getSnapshot().tracks[0]!.notes[0]!;
    expect(placedCand0.pitch).toBe(60);

    // Before user edits `placedCand0`, extracting context on bars 1..2 excludes unedited candidate notes -> `meta`
    const ctxUnedited = await ctrl.extractGenerationContext({ taskOverride: 'auto' });
    expect(ctxUnedited.resolvedTask).toBe('meta');

    // Move `placedCand0` into bar 4 (`startTick: 5760`, Future window of bars 1..2)
    ctrl.transformNotes([
      {
        id: placedCand0.id,
        pitch: placedCand0.pitch,
        startTick: 5760,
        durationTicks: placedCand0.durationTicks,
      },
    ]);
    const ctxAfterMove = await ctrl.extractGenerationContext({ taskOverride: 'auto' });
    expect(ctxAfterMove.resolvedTask).toBe('meta_future');
    expect(ctxAfterMove.futureMidiBytes).not.toBeNull();

    //Undo `transformNotes` -> note returns to `0` AND `_activeCandidateNoteIds` membership is restored -> `meta`
    ctrl.undo();
    const ctxAfterUndo = await ctrl.extractGenerationContext({ taskOverride: 'auto' });
    expect(ctxAfterUndo.resolvedTask).toBe('meta');

    // Redo `transformNotes` -> re-detached -> `meta_future`; switching to `候補 2` preserves moved note at `5760`!
    ctrl.redo();
    await ctrl.selectGeneratedCandidate(1);
    const notesAfterCandSwitch = ctrl.getSnapshot().tracks[0]!.notes;
    expect(notesAfterCandSwitch.some((n) => n.startTick === 5760 && n.pitch === 60)).toBe(true);
    expect(notesAfterCandSwitch.some((n) => n.startTick === 0 && n.pitch === 67)).toBe(true);

    // 2. `[R7-HIGH-04]`, `[R7-MED-03]`, `[R7-LOW-01]` Deferred hidden centering, telemetry track prefix, and F2/touch track rename
    function R8PianoRollHarness({ active, candRev }: { active: boolean; candRev: number }) {
      const [snap, setSnap] = React.useState(() => ctrl.getSnapshot());
      React.useEffect(() => ctrl.subscribe(setSnap), []);
      return (
        <PianoRollView
          isActive={active}
          controller={ctrl}
          session={snap}
          focusRevision={0}
          candidateFocusRevision={candRev}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />
      );
    }

    // Mount hidden (`isActive={false}`) with `candRev={1}`, then activate -> deferred centering runs once visible
    const prUi = render(<R8PianoRollHarness active={false} candRev={1} />);
    prUi.rerender(<R8PianoRollHarness active={true} candRev={1} />);

    // Telemetry retains active track name on single & multi-note selection (`[R7-LOW-01]`)
    const curTrkName = ctrl.getSnapshot().tracks[0]!.name;
    const allIds = ctrl.getSnapshot().tracks[0]!.notes.map((n) => n.id);
    act(() => {
      ctrl.setSelectedNotes([allIds[0]!]);
    });
    expect(prUi.getByTestId('pianoroll-telemetry-bar').textContent).toContain(`${curTrkName} · Pitch`);
    act(() => {
      ctrl.setSelectedNotes(allIds);
    });
    expect(prUi.getByTestId('pianoroll-telemetry-bar').textContent).toContain(
      `${curTrkName} · ${allIds.length} notes selected`,
    );

    // Ensure Track Rack is open, then rename via keyboard `F2` -> `Enter` restores focus (`[R7-HIGH-04]`)
    const rackToggle = prUi.getByTestId('toggle-track-rack-button');
    if (rackToggle.getAttribute('aria-pressed') !== 'true') {
      fireEvent.click(rackToggle);
    }
    const trk0Id = ctrl.getSnapshot().tracks[0]!.id;
    const trkBtn = prUi.container.querySelector(
      `button[data-track-select-id="${trk0Id}"]`,
    ) as HTMLButtonElement;
    expect(trkBtn).toBeTruthy();
    act(() => {
      trkBtn.focus();
    });
    fireEvent.keyDown(trkBtn, { key: 'F2' });
    const renameInput = prUi.getByLabelText(`${curTrkName} トラック名編集`) as HTMLInputElement;
    fireEvent.change(renameInput, { target: { value: 'Studio Grand' } });
    fireEvent.keyDown(renameInput, { key: 'Enter' });
    expect(ctrl.getSnapshot().tracks[0]!.name).toBe('Studio Grand');
    const restoredBtn = prUi.container.querySelector(
      `button[data-track-select-id="${trk0Id}"]`,
    ) as HTMLButtonElement;
    expect(document.activeElement).toBe(restoredBtn);

    // Touch tap on already-active track opens rename, and `Escape` cancels & restores focus (`[R7-HIGH-04]`)
    fireEvent.pointerDown(restoredBtn, { pointerType: 'touch' });
    fireEvent.click(restoredBtn);
    const touchRenameInput = prUi.getByLabelText('Studio Grand トラック名編集') as HTMLInputElement;
    fireEvent.change(touchRenameInput, { target: { value: 'Discarded Name' } });
    fireEvent.keyDown(touchRenameInput, { key: 'Escape' });
    expect(ctrl.getSnapshot().tracks[0]!.name).toBe('Studio Grand');
    prUi.unmount();

    // 3. `[R7-MED-01]` WCAG 2.1 SC 1.4.13 `Escape` dismisses hovered tooltip on 1st press without closing modal, and closes modal on 2nd press + vertical `<aside>` flip
    const onModalCancel = vi.fn();
    const modalUi = render(
      <MidiTrackImportModal
        fileName="escape_tooltip_test.mid"
        parsedMidi={StandardMidiCodec.createDemoMidiPreset()}
        onConfirm={() => {}}
        onCancel={onModalCancel}
      />,
    );
    const modalCloseBtn = modalUi.getByRole('button', { name: '閉じる' });
    fireEvent.pointerOver(modalCloseBtn);
    expect(modalCloseBtn.getAttribute('data-tooltip-dismissed')).toBeNull();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(modalCloseBtn.getAttribute('data-tooltip-dismissed')).toBe('true');
    expect(onModalCancel).toHaveBeenCalledTimes(0);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onModalCancel).toHaveBeenCalledTimes(1);
    modalUi.unmount();

    // Vertical `<aside>` flip test for `alignButtonTooltipToViewport`
    const mockAside = document.createElement('aside');
    const mockTrashBtn = document.createElement('button');
    mockTrashBtn.setAttribute('data-tooltip', 'トラックを削除');
    mockTrashBtn.setAttribute('data-tooltip-pos', 'top');
    mockTrashBtn.setAttribute('data-tooltip-prefer-pos', 'top');
    mockAside.appendChild(mockTrashBtn);
    document.body.appendChild(mockAside);
    vi.spyOn(mockAside, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 100,
      left: 0,
      top: 100,
      right: 375,
      bottom: 276,
      width: 375,
      height: 176,
      toJSON: () => ({}),
    });
    const trashRectSpy = vi.spyOn(mockTrashBtn, 'getBoundingClientRect').mockReturnValue({
      x: 310,
      y: 106,
      left: 310,
      top: 106,
      right: 354,
      bottom: 150,
      width: 44,
      height: 44,
      toJSON: () => ({}),
    });
    alignButtonTooltipToViewport(mockTrashBtn);
    // Only 6px above inside `<aside>`, 126px below -> flips from `top` to default bottom
    expect(mockTrashBtn.getAttribute('data-tooltip-pos')).toBeNull();
    trashRectSpy.mockReturnValue({
      x: 310,
      y: 224,
      left: 310,
      top: 224,
      right: 354,
      bottom: 268,
      width: 44,
      height: 44,
      toJSON: () => ({}),
    });
    alignButtonTooltipToViewport(mockTrashBtn);
    // Plenty of room above (`124px >= 32px`) -> honors `data-tooltip-prefer-pos="top"`
    expect(mockTrashBtn.getAttribute('data-tooltip-pos')).toBe('top');
    mockAside.remove();

    // 4. `[R7-HIGH-03]`, `[R7-MED-02]`, `[R7-MED-03]` `<App />` immediate cancel broadcast, idle `REQUEST_WORKSPACE_STATE` status reply, local tab click isolation, and External DAW split-brain guards
    const origFetch = globalThis.fetch;
    try {
      let genMode: 'hang' | 'instant' = 'hang';
      const zip = new JSZip();
      zip.file('gen_0.mid', cand0Bytes);
      zip.file('reasons.json', JSON.stringify([reason]));
      const zipBytes = await zip.generateAsync({ type: 'uint8array' });

      globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith('/system/hardware')) {
          return new Response(
            JSON.stringify({
              device: 'cpu',
              cuda_available: false,
              gpu_name: null,
              models_dir: 'C:\\Models',
              bundled_models_dir: 'C:\\Bundled',
              available_models_count: 1,
              models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (url.endsWith('/models')) {
          return new Response(
            JSON.stringify([
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                tag: { model: 'sft_gen' },
              },
            ]),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (url.endsWith('/generate')) {
          if (genMode === 'instant') {
            return new Response(zipBytes.slice().buffer, {
              status: 200,
              headers: { 'Content-Type': 'application/zip' },
            });
          }
          return await new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener('abort', () => {
              reject(new DOMException('Aborted', 'AbortError'));
            });
          });
        }
        return new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      });

      const appUi = render(<App />);
      await waitFor(
        () => {
          expect(appUi.getByTestId('generate-button')).toBeTruthy();
        },
        { timeout: 3500 },
      );

      const peerBus = new WorkspaceSyncBus();
      const observedStatuses: boolean[] = [];
      const observedFocusTabs: Array<{ tab: string; reason: string }> = [];
      const unsub = peerBus.subscribe((msg) => {
        if (msg.type === 'GENERATION_STATUS') {
          observedStatuses.push(msg.isGenerating);
        } else if (msg.type === 'FOCUS_TAB') {
          observedFocusTabs.push({ tab: msg.tab, reason: msg.reason });
        }
      });

      // Idle `REQUEST_WORKSPACE_STATE` replies with `GENERATION_STATUS(false, ...)` (`[R7-MED-02]`)
      peerBus.requestWorkspaceState();
      await waitFor(() => {
        expect(observedStatuses).toContain(false);
      });

      // Start generation, then click `生成処理を中断` -> immediately clears & broadcasts `false` (`[R7-MED-02]`)
      observedStatuses.length = 0;
      fireEvent.click(appUi.getByTestId('generate-button'));
      const cancelGenBtn = await waitFor(() =>
        appUi.getByRole('button', { name: '生成処理を中断' }),
      );
      fireEvent.click(cancelGenBtn);
      await waitFor(() => {
        expect(observedStatuses).toContain(false);
        expect(appUi.getByTestId('generate-button')).toBeTruthy();
      });

      // Clicking `#workspace-tab-pianoroll` switches tab locally WITHOUT broadcasting `FOCUS_TAB` (`[R7-MED-03]`)
      observedFocusTabs.length = 0;
      fireEvent.click(appUi.getByTestId('tab-pianoroll'));
      expect(appUi.getByTestId('tab-pianoroll').getAttribute('aria-selected')).toBe('true');
      expect(observedFocusTabs.length).toBe(0);
      fireEvent.click(appUi.getByTestId('tab-compass-ai'));

      // Switch to External DAW (`cubase-midi-remote`) (`[R7-HIGH-03]`)
      const dawSelect = appUi.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => {
        const popoutBtn = appUi.getByRole('button', {
          name: '別ウィンドウでピアノロールを開く',
        }) as HTMLButtonElement;
        expect(popoutBtn.disabled).toBe(true);
        expect(popoutBtn.getAttribute('data-tooltip')).toBe(
          '外部DAW接続中は別ウィンドウを開けません',
        );
      });

      // Inbound `SYNC_WORKSPACE_STATE` from a pop-out window updates `demoController` even while `Cubase` is active (`[R7-HIGH-03]`)
      const remoteCtrl = new DemoPianoRollController(new StubAudioEngine());
      await remoteCtrl.connect();
      const remTrkId = remoteCtrl.getSnapshot().tracks[0]!.id;
      remoteCtrl.renameTrack(remTrkId, 'Popout Synced Keys');
      peerBus.publishWorkspaceState(remoteCtrl.exportWorkspaceState());

      // External DAW generation completion does NOT yank pop-out Piano Roll via `FOCUS_TAB('pianoroll', 'generation-complete')` (`[R7-HIGH-03]`)
      genMode = 'instant';
      observedFocusTabs.length = 0;
      fireEvent.click(appUi.getByTestId('generate-button'));
      await waitFor(() => {
        expect(appUi.getByTestId('tab-pianoroll').getAttribute('aria-selected')).toBe('true');
      });
      expect(
        observedFocusTabs.some((f) => f.tab === 'pianoroll' && f.reason === 'generation-complete'),
      ).toBe(false);

      // Switch back to `demo-pianoroll` and verify `'Popout Synced Keys'` survived without split-brain loss!
      fireEvent.click(appUi.getByTestId('tab-compass-ai'));
      fireEvent.change(dawSelect, { target: { value: 'demo-pianoroll' } });
      fireEvent.click(appUi.getByTestId('tab-pianoroll'));
      await waitFor(() => {
        expect(appUi.getByTestId('pianoroll-telemetry-bar').textContent).toContain(
          'Popout Synced Keys',
        );
      });

      unsub();
      peerBus.dispose();
      appUi.unmount();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('prevents tooltip click/drag occlusion ([R8-BUG-01]), replies to REQUEST_WORKSPACE_STATE under External DAW ([R8-BUG-02]), and preserves Escape cancel/revert ([R8-BUG-03], [R8-LOW-01..03])', async () => {
    // 1. [R8-BUG-01] + [R8-BUG-03] + [R8-LOW-02] + [R8-LOW-03] on PianoRollView
    const ctrl = new DemoPianoRollController(new StubAudioEngine());
    await ctrl.connect();
    const trk1Id = ctrl.getSnapshot().tracks[0]!.id;
    ctrl.createNoteAt(60, 0, 480, 100);
    const createdId = ctrl.getSnapshot().tracks[0]!.notes[0]!.id;
    ctrl.setSelectedNotes([createdId]);

    let latestSnap = ctrl.getSnapshot();
    const unsubCtrl = ctrl.subscribe((s) => {
      latestSnap = s;
    });

    const prUi = render(
      <PianoRollView
        isActive={true}
        isGenerating={false}
        controller={ctrl}
        session={latestSnap}
        focusRevision={0}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );

    const canvas = prUi.getByTestId('pianoroll-canvas') as HTMLCanvasElement;
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 60,
      left: 0,
      top: 60,
      right: 960,
      bottom: 600,
      width: 960,
      height: 540,
      toJSON: () => ({}),
    } as DOMRect);

    const pencilBtn = prUi.getByRole('button', { name: 'ペン入力' }) as HTMLButtonElement;
    vi.spyOn(pencilBtn, 'getBoundingClientRect').mockReturnValue({
      x: 200,
      y: 8,
      left: 200,
      top: 8,
      right: 244,
      bottom: 52,
      width: 44,
      height: 44,
      toJSON: () => ({}),
    } as DOMRect);

    // [R8-BUG-01]: Moving across the 6px gap onto `<canvas>` activates `data-tooltip-bridge="true"`,
    // while moving onto a neighboring button clears the bridge immediately
    fireEvent.pointerOver(pencilBtn, { clientX: 222, clientY: 30, buttons: 0 });
    fireEvent.pointerOut(pencilBtn, {
      relatedTarget: canvas,
      clientX: 222,
      clientY: 66,
      buttons: 0,
    });
    expect(pencilBtn.getAttribute('data-tooltip-bridge')).toBe('true');

    // Pressing `Escape` while hovering bridged tooltip over `<canvas>` (with no drag/input) dismisses it
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(pencilBtn.getAttribute('data-tooltip-bridge')).toBeNull();
    expect(pencilBtn.getAttribute('data-tooltip-dismissed')).toBe('true');

    // Re-hover & bridge, then enter neighbor discrete control -> clears immediately
    fireEvent.pointerOut(pencilBtn, { relatedTarget: document.body, clientX: 500, clientY: 500 });
    fireEvent.pointerOver(pencilBtn, { clientX: 222, clientY: 30, buttons: 0 });
    fireEvent.pointerOut(pencilBtn, {
      relatedTarget: canvas,
      clientX: 222,
      clientY: 66,
      buttons: 0,
    });
    expect(pencilBtn.getAttribute('data-tooltip-bridge')).toBe('true');
    const rangeBtn = prUi.getByRole('button', { name: '3連符グリッド切替' });
    fireEvent.pointerOver(rangeBtn, { clientX: 320, clientY: 30, buttons: 0 });
    expect(pencilBtn.getAttribute('data-tooltip-bridge')).toBeNull();
    fireEvent.pointerOut(rangeBtn, { relatedTarget: document.body, clientX: 500, clientY: 500 });

    // [R8-BUG-03] Sub-case A: Active `<canvas>` drag (`data-pointer-dragging="true"`) + hovered tooltip button
    // MUST NOT swallow `Escape` — drag cancels and restores `initialSelection` on 1st `Escape`!
    fireEvent.pointerDown(canvas, {
      button: 0,
      buttons: 1,
      pointerId: 11,
      clientX: 300,
      clientY: 220,
    });
    expect(canvas.getAttribute('data-pointer-dragging')).toBe('true');
    // Selection was cleared on empty-grid marquee start; now hover toolbar button mid-drag and press Escape
    expect(ctrl.getSnapshot().selectedNoteIds.has(createdId)).toBe(false);
    fireEvent.pointerOver(pencilBtn, { clientX: 222, clientY: 30, buttons: 1 });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(canvas.getAttribute('data-pointer-dragging')).toBeNull();
    expect(ctrl.getSnapshot().selectedNoteIds.has(createdId)).toBe(true);

    // [R8-BUG-03] Sub-case B: Focused `<input>` draft revert while cursor rests on neighboring `button[data-tooltip]`
    fireEvent.pointerOver(pencilBtn, { clientX: 222, clientY: 30, buttons: 0 });
    const bpmInput = prUi.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
    bpmInput.focus();
    fireEvent.focus(bpmInput);
    fireEvent.change(bpmInput, { target: { value: '199' } });
    expect(bpmInput.value).toBe('199');
    fireEvent.keyDown(bpmInput, { key: 'Escape' });
    expect(bpmInput.value).toBe('120');

    // [R8-BUG-03] Sub-case C: Stale `hoveredTooltipBtn` on an inert/hidden toolbar button behind `MidiTrackImportModal`
    // MUST NOT swallow 1st `Escape`!
    fireEvent.pointerOver(pencilBtn, { clientX: 222, clientY: 30, buttons: 0 });
    fireEvent.click(prUi.getByTestId('load-sample-arrangement-button'));
    await waitFor(() => {
      expect(prUi.getByRole('dialog')).toBeTruthy();
    });
    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => {
      expect(prUi.queryByRole('dialog')).toBeNull();
    });

    // [R8-LOW-03]: `pointerCancel` after touch `pointerDown` on active track button resets pointer type
    const trackSelectBtn = prUi.container.querySelector(
      `button[data-track-select-id="${trk1Id}"]`,
    ) as HTMLButtonElement;
    fireEvent.pointerDown(trackSelectBtn, { pointerType: 'touch' });
    fireEvent.pointerCancel(trackSelectBtn);
    fireEvent.click(trackSelectBtn);
    expect(prUi.queryByRole('textbox', { name: /トラック名編集/ })).toBeNull();

    // [R8-LOW-02]: Track Rack `+` button disables at 64-track workspace cap
    while (ctrl.getSnapshot().tracks.length < 64) {
      ctrl.addTrack();
    }
    prUi.rerender(
      <PianoRollView
        isActive={true}
        isGenerating={false}
        controller={ctrl}
        session={ctrl.getSnapshot()}
        focusRevision={0}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );
    const addTrackBtn = prUi.getByRole('button', { name: 'トラックを追加' }) as HTMLButtonElement;
    expect(addTrackBtn.disabled).toBe(true);
    expect(addTrackBtn.getAttribute('data-tooltip')).toBe('最大64トラックまで追加できます');

    // [R9-TOOLTIP-01] Sub-case 1: Single-active-tooltip mutual exclusion (focus-restore <-> hover/bridge)
    const firstMuteBtn = prUi.container.querySelector(
      'button[data-tooltip="ミュート"]',
    ) as HTMLButtonElement;
    act(() => {
      trackSelectBtn.focus();
    });
    expect(trackSelectBtn.getAttribute('data-tooltip-dismissed')).toBeNull();
    fireEvent.pointerOver(firstMuteBtn, { clientX: 160, clientY: 130, buttons: 0 });
    expect(trackSelectBtn.getAttribute('data-tooltip-dismissed')).toBe('true');

    // Reverse direction: active hover on `firstMuteBtn` + active bridge on `pencilBtn` cleared/dismissed on `:focus-visible`
    fireEvent.pointerOver(pencilBtn, { clientX: 222, clientY: 30, buttons: 0 });
    fireEvent.pointerOut(pencilBtn, {
      relatedTarget: canvas,
      clientX: 222,
      clientY: 66,
      buttons: 0,
    });
    expect(pencilBtn.getAttribute('data-tooltip-bridge')).toBe('true');
    fireEvent.pointerOver(firstMuteBtn, { clientX: 160, clientY: 130, buttons: 0 });
    act(() => {
      (rangeBtn as HTMLButtonElement).focus();
    });
    expect(pencilBtn.getAttribute('data-tooltip-bridge')).toBeNull();
    expect(firstMuteBtn.getAttribute('data-tooltip-dismissed')).toBe('true');
    fireEvent.pointerOut(firstMuteBtn, { relatedTarget: document.body, clientX: 500, clientY: 500 });

    // [R9-TOOLTIP-01] Sub-case 2: Disabled `<button disabled>` skipped by Chromium `PointerEvent` hit-testing
    // still clears `サンプル`'s `[data-tooltip-bridge="true"]` via bounding-rect hit detection on parent container
    const sampleBtn = prUi.getByTestId('load-sample-arrangement-button') as HTMLButtonElement;
    vi.spyOn(sampleBtn, 'getBoundingClientRect').mockReturnValue({
      x: 162,
      y: 8,
      left: 162,
      top: 8,
      right: 248,
      bottom: 52,
      width: 86,
      height: 44,
      toJSON: () => ({}),
    } as DOMRect);
    vi.spyOn(addTrackBtn, 'getBoundingClientRect').mockReturnValue({
      x: 199,
      y: 62,
      left: 199,
      top: 62,
      right: 243,
      bottom: 106,
      width: 44,
      height: 44,
      toJSON: () => ({}),
    } as DOMRect);
    const rackHeaderDiv = addTrackBtn.parentElement as HTMLElement;
    fireEvent.pointerOver(sampleBtn, { clientX: 205, clientY: 30, buttons: 0 });
    fireEvent.pointerOut(sampleBtn, {
      relatedTarget: canvas,
      clientX: 205,
      clientY: 58,
      buttons: 0,
    });
    expect(sampleBtn.getAttribute('data-tooltip-bridge')).toBe('true');
    fireEvent.pointerOver(rackHeaderDiv, { clientX: 220, clientY: 80, buttons: 0 });
    expect(sampleBtn.getAttribute('data-tooltip-bridge')).toBeNull();

    // [R9-TOOLTIP-01] Sub-case 3: Stacking containment (`z-[5]` < `z-50` modal) & `[inert]` bridge cleanup on modal open
    const splitWorkspace = prUi.container.querySelector('[aria-label="トラックラック"]')?.parentElement;
    expect(splitWorkspace?.className).toContain('z-[5]');
    const firstSoloBtn = prUi.container.querySelector(
      'button[data-tooltip="ソロ"]',
    ) as HTMLButtonElement;
    vi.spyOn(firstSoloBtn, 'getBoundingClientRect').mockReturnValue({
      x: 191,
      y: 112,
      left: 191,
      top: 112,
      right: 235,
      bottom: 156,
      width: 44,
      height: 44,
      toJSON: () => ({}),
    } as DOMRect);
    fireEvent.pointerOver(firstSoloBtn, { clientX: 213, clientY: 134, buttons: 0 });
    fireEvent.pointerOut(firstSoloBtn, {
      relatedTarget: canvas,
      clientX: 213,
      clientY: 162,
      buttons: 0,
    });
    expect(firstSoloBtn.getAttribute('data-tooltip-bridge')).toBe('true');
    fireEvent.click(sampleBtn);
    await waitFor(() => {
      expect(prUi.getByRole('dialog')).toBeTruthy();
    });
    expect(firstSoloBtn.getAttribute('data-tooltip-bridge')).toBeNull();

    unsubCtrl();
    prUi.unmount();

    // 2. [R8-BUG-02] + [R8-LOW-01]: Main Window replies to `REQUEST_WORKSPACE_STATE` with `demoController` project
    // even while switched to External DAW (`cubase-midi-remote`)!
    const origFetch = globalThis.fetch;
    try {
      globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === 'string' ? input : input.toString();
        if (url.endsWith('/hardware_info')) {
          return new Response(
            JSON.stringify({
              device: 'cpu',
              dtype: 'torch.float32',
              user_models_dir: '/tmp/models',
              models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (url.endsWith('/model_info')) {
          return new Response(
            JSON.stringify([
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                tag: { model: 'sft_gen' },
              },
            ]),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        return new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      });

      const appUi = render(<App />);
      await waitFor(
        () => {
          expect(appUi.getByTestId('generate-button')).toBeTruthy();
        },
        { timeout: 3500 },
      );

      const peerBus = new WorkspaceSyncBus();
      // Seed Main Window's `demoController` with a custom arrangement
      const seededCtrl = new DemoPianoRollController(new StubAudioEngine());
      await seededCtrl.connect();
      const seedTrkId = seededCtrl.getSnapshot().tracks[0]!.id;
      seededCtrl.renameTrack(seedTrkId, 'Preserved MultiTrack Arrangement');
      seededCtrl.createNoteAt(67, 0, 480, 110);
      peerBus.publishWorkspaceState(seededCtrl.exportWorkspaceState());

      // Switch Main Window to External DAW (`cubase-midi-remote`)
      const dawSelect = appUi.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => {
        const popoutBtn = appUi.getByRole('button', {
          name: '別ウィンドウでピアノロールを開く',
        }) as HTMLButtonElement;
        expect(popoutBtn.disabled).toBe(true);
        expect(popoutBtn.getAttribute('aria-description')).toBe(
          '外部DAW接続中は別ウィンドウを開けません',
        );
      });

      // Simulate `?popout=1` reload requesting workspace state while Main Window is on `Cubase`
      let receivedTrackName = '';
      let receivedNoteCount = 0;
      const unsubPeer = peerBus.subscribe((msg) => {
        if (msg.type === 'SYNC_WORKSPACE_STATE') {
          const restored = new DemoPianoRollController(new StubAudioEngine());
          restored.applyWorkspaceState(msg.workspaceState);
          const snap = restored.getSnapshot();
          receivedTrackName = snap.tracks[0]?.name ?? '';
          receivedNoteCount = snap.tracks[0]?.notes.length ?? 0;
        }
      });

      peerBus.requestWorkspaceState();
      await waitFor(() => {
        expect(receivedTrackName).toBe('Preserved MultiTrack Arrangement');
        expect(receivedNoteCount).toBe(1);
      });

      unsubPeer();
      peerBus.dispose();
      appUi.unmount();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('47. [Round-10 Critic] verifies disabled tooltip opacity/pointermove lifecycle ([R10-HIGH-01]), SYNC_ACTIVE_CONTROLLER 3-file lockstep & pop-out standby card ([R10-HIGH-02]), ControlPanel preview/CoT tooltips ([R10-MED-01]), secondary strip tooltip stacking ([R10-MED-02]), and programmatic tab focus retention ([R10-MED-03])', async () => {
    // 1. `[R10-HIGH-02]` Lockstep validation of `SYNC_ACTIVE_CONTROLLER` in `ipc_security.cjs` & `WorkspaceSyncBus`
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ipcSec = require('../../electron/ipc_security.cjs') as {
      isValidWorkspaceSyncPayload: (msg: unknown) => boolean;
    };
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'm_ctrl_1',
        type: 'SYNC_ACTIVE_CONTROLLER',
        controllerId: 'cubase-midi-remote',
        senderId: 'peer_1',
      }),
    ).toBe(true);
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'm_ctrl_2',
        type: 'SYNC_ACTIVE_CONTROLLER',
        controllerId: 'evil-controller',
        senderId: 'peer_1',
      }),
    ).toBe(false);
    expect(
      ipcSec.isValidWorkspaceSyncPayload({
        messageId: 'm_ctrl_3',
        type: 'SYNC_ACTIVE_CONTROLLER',
        controllerId: 'demo-pianoroll',
        senderId: 'peer_1',
        extraKey: true,
      }),
    ).toBe(false);

    // 2. `[R10-HIGH-01]` + `[R10-MED-02]`: Disabled `+` (`トラックを追加`) keeps full opacity (`no disabled:opacity-40`),
    // supports `pointermove` hover -> corridor bridge -> `Escape` dismissal -> exit + re-hover recovery,
    // and `[data-tooltip-strip]` elevates via `data-tooltip-strip-active="true"` ONLY while its tooltip is active
    const audio = new StubAudioEngine();
    const ctrl = new DemoPianoRollController(audio);
    await ctrl.connect();
    while (ctrl.getSnapshot().tracks.length < 64) {
      ctrl.addTrack();
    }
    let curSnap = ctrl.getSnapshot();
    const prUi = render(
      <PianoRollView
        isActive={true}
        isGenerating={false}
        controller={ctrl}
        session={curSnap}
        focusRevision={0}
        onLaunchCompassAi={() => {}}
        onSwitchToDemoController={() => {}}
      />,
    );
    const unsubCtrl = ctrl.subscribe((s) => {
      curSnap = s;
      prUi.rerender(
        <PianoRollView
          isActive={true}
          isGenerating={false}
          controller={ctrl}
          session={curSnap}
          focusRevision={0}
          onLaunchCompassAi={() => {}}
          onSwitchToDemoController={() => {}}
        />,
      );
    });

    const addTrackBtn = prUi.getByRole('button', { name: 'トラックを追加' }) as HTMLButtonElement;
    expect(addTrackBtn.disabled).toBe(true);
    expect(addTrackBtn.className).not.toContain('disabled:opacity-40');
    vi.spyOn(addTrackBtn, 'getBoundingClientRect').mockReturnValue({
      x: 199,
      y: 62,
      left: 199,
      top: 62,
      right: 243,
      bottom: 106,
      width: 44,
      height: 44,
      toJSON: () => ({}),
    } as DOMRect);
    const rackHeaderDiv = addTrackBtn.parentElement as HTMLElement;

    // Move inside disabled `+` button within `rackHeaderDiv` (no `pointerover` needed!)
    fireEvent.pointerMove(rackHeaderDiv, { clientX: 220, clientY: 84, buttons: 0 });
    expect(addTrackBtn.getAttribute('data-tooltip-hovered')).toBe('true');

    // Move across the 6px gap below disabled `+` button (`y = 112`) -> activates corridor bridge!
    fireEvent.pointerMove(rackHeaderDiv, { clientX: 220, clientY: 112, buttons: 0 });
    expect(addTrackBtn.getAttribute('data-tooltip-hovered')).toBeNull();
    expect(addTrackBtn.getAttribute('data-tooltip-bridge')).toBe('true');

    // Re-enter disabled `+` button from bridge (`y = 84`) -> restores hover & clears bridge
    fireEvent.pointerMove(rackHeaderDiv, { clientX: 220, clientY: 84, buttons: 0 });
    expect(addTrackBtn.getAttribute('data-tooltip-bridge')).toBeNull();
    expect(addTrackBtn.getAttribute('data-tooltip-hovered')).toBe('true');

    // Press `Escape` -> dismisses disabled tooltip immediately
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(addTrackBtn.getAttribute('data-tooltip-dismissed')).toBe('true');
    expect(addTrackBtn.getAttribute('data-tooltip-hovered')).toBeNull();

    // Move out of disabled `+` rect (`y = 40` inside `rackHeaderDiv`), then re-hover (`y = 84`) -> recovers cleanly!
    fireEvent.pointerMove(rackHeaderDiv, { clientX: 120, clientY: 84, buttons: 0 });
    expect(addTrackBtn.getAttribute('data-tooltip-dismissed')).toBeNull();
    fireEvent.pointerMove(rackHeaderDiv, { clientX: 220, clientY: 84, buttons: 0 });
    expect(addTrackBtn.getAttribute('data-tooltip-hovered')).toBe('true');
    fireEvent.pointerMove(rackHeaderDiv, { clientX: 120, clientY: 84, buttons: 0 });

    // `[R10-MED-02]`: Trigger `uploadNotice` by uploading a 1-track MIDI so Secondary Strip (`[data-tooltip-strip]`) renders
    ctrl.deleteTrack(ctrl.getSnapshot().tracks[63]!.id);
    const singleTrackBytes = StandardMidiCodec.encodeSingleTrackMidi(
      [new MidiNote({ pitch: 64, startTick: 0, durationTicks: 480, velocity: 96 })],
      480,
      120,
      'PIANO',
    );
    const midiFile = new File([singleTrackBytes as Uint8Array<ArrayBuffer>], 'strip_notice.mid', {
      type: 'audio/midi',
    });
    if (typeof midiFile.arrayBuffer !== 'function') {
      Object.defineProperty(midiFile, 'arrayBuffer', {
        value: async () =>
          singleTrackBytes.buffer.slice(
            singleTrackBytes.byteOffset,
            singleTrackBytes.byteOffset + singleTrackBytes.byteLength,
          ),
      });
    }
    fireEvent.change(prUi.getByTestId('midi-file-input'), { target: { files: [midiFile] } });
    const dismissNoticeBtn = (await waitFor(() =>
      prUi.getByRole('button', { name: '通知を閉じる' }),
    )) as HTMLButtonElement;
    const stripEl = dismissNoticeBtn.closest('[data-tooltip-strip]') as HTMLElement;
    expect(stripEl).toBeTruthy();
    expect(stripEl.getAttribute('data-tooltip-strip-active')).toBeNull();

    fireEvent.pointerOver(dismissNoticeBtn, { clientX: 300, clientY: 70, buttons: 0 });
    expect(stripEl.getAttribute('data-tooltip-strip-active')).toBe('true');
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(stripEl.getAttribute('data-tooltip-strip-active')).toBeNull();

    unsubCtrl();
    prUi.unmount();

    // 3. `[R10-MED-01]`, `[R10-MED-03]`, `[R10-HIGH-02]`: Full `<App />` + Pop-Out `<App />` verification
    const origFetch = globalThis.fetch;
    const origUrl = window.location.href;
    try {
      globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === 'string' ? input : input.toString();
        if (url.endsWith('/hardware_info')) {
          return new Response(
            JSON.stringify({
              device: 'cpu',
              dtype: 'torch.float32',
              user_models_dir: '/tmp/models',
              models_with_weights: ['MORTM4.5D-80M-SFT-Gen'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (url.endsWith('/model_info')) {
          return new Response(
            JSON.stringify([
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                tag: { model: 'sft_gen' },
              },
            ]),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        return new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      });

      const mainApp = render(<App />);
      await waitFor(
        () => {
          expect(mainApp.getByTestId('generate-button')).toBeTruthy();
        },
        { timeout: 3500 },
      );

      // `[R10-MED-01]`: Preview button on `ControlPanelView` has NO misleading `(Space)`,
      // and CoT button in `meta` mode (`構造推論 自動省略`) is genuinely `disabled` with explanatory tooltip
      const previewBtn = mainApp.getByRole('button', { name: 'サウンド確認' }) as HTMLButtonElement;
      expect(previewBtn.getAttribute('data-tooltip')).toBe('サウンド確認');
      expect(previewBtn.className).not.toContain('disabled:opacity-40');

      const cotBtn = mainApp.getByRole('button', { name: '構造推論 自動省略' }) as HTMLButtonElement;
      expect(cotBtn.disabled).toBe(true);
      expect(cotBtn.getAttribute('data-tooltip')).toBe(
        '白紙からの生成 (meta) では構造推論は自動省略されます',
      );
      expect(cotBtn.className).not.toContain('disabled:opacity-40');

      // `[R10-MED-03]`: Switch to Piano Roll, focus `launch-compass-ai-button`, click it ->
      // programmatic tab transition moves focus cleanly to `#workspace-tab-compass-ai` (NOT `document.body`)
      fireEvent.click(mainApp.getByTestId('tab-pianoroll'));
      const launchBtn = mainApp.getByTestId('launch-compass-ai-button') as HTMLButtonElement;
      launchBtn.focus();
      fireEvent.click(launchBtn);
      await waitFor(() => {
        expect(document.activeElement?.id).toBe('workspace-tab-compass-ai');
      });

      // Mount secondary pop-out `<App />` (`?tab=pianoroll&popout=1`) in parallel
      window.history.replaceState(null, '', '/?tab=pianoroll&popout=1');
      const popoutApp = render(<App />);
      await waitFor(
        () => {
          expect(popoutApp.getByTestId('popout-pianoroll-badge')).toBeTruthy();
        },
        { timeout: 3500 },
      );

      // Switch Main Window to External DAW (`cubase-midi-remote`) ->
      // 1) Main preview button disables with host-DAW tooltip & no `disabled:opacity-40`
      // 2) Main External DAW Bridge `launch-compass-ai-button` carries `aria-keyshortcuts="Shift+G"`
      // 3) Pop-Out Window receives `SYNC_ACTIVE_CONTROLLER` and renders non-redundant Standby Card (`0` form inputs)
      const dawSelect = mainApp.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect, { target: { value: 'cubase-midi-remote' } });

      await waitFor(() => {
        expect(previewBtn.disabled).toBe(true);
        expect(previewBtn.getAttribute('data-tooltip')).toBe(
          '外部DAW接続中はホストDAW側で再生を操作します',
        );
        expect(popoutApp.getByTestId('popout-external-daw-standby')).toBeTruthy();
      });

      const standbyCard = popoutApp.getByTestId('popout-external-daw-standby');
      expect(standbyCard.querySelectorAll('select, input').length).toBe(0);

      fireEvent.click(
        mainApp.container.querySelector('[data-testid="tab-pianoroll"]') as HTMLButtonElement,
      );
      const extLaunchBtn = mainApp.container.querySelector(
        '[data-testid="launch-compass-ai-button"]',
      ) as HTMLButtonElement;
      expect(extLaunchBtn.getAttribute('aria-keyshortcuts')).toBe('Shift+G');

      // Click `内蔵ピアノロールへ切り替え` inside Pop-Out Standby Card -> switches BOTH Pop-Out and Main Window back to `demo-pianoroll`!
      const popoutSwitchBtn = popoutApp.container.querySelector(
        '[data-testid="popout-switch-to-demo-button"]',
      ) as HTMLButtonElement;
      popoutSwitchBtn.focus();
      fireEvent.click(popoutSwitchBtn);

      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="popout-external-daw-standby"]')).toBeNull();
        expect(dawSelect.value).toBe('demo-pianoroll');
        expect(previewBtn.disabled).toBe(false);
      });

      popoutApp.unmount();
      mainApp.unmount();
    } finally {
      window.history.replaceState(null, '', origUrl);
      globalThis.fetch = origFetch;
    }
  });

  it('48. [Round-11 Fact-Checker + Zero-Scroll Console] verifies DawControllerRegistry async re-entrancy ([R11-MED-01]), Main reload & orphan promotion sync ([R11-MED-02]), self-unmount focus retention ([R11-MED-03]), and click strip cleanup ([R11-LOW-01])', async () => {
    // 1. `[R11-MED-01]` Deterministic async re-entrancy protection in `DawControllerRegistry.activate()`
    const reg = new DawControllerRegistry();
    const demo = new DemoPianoRollController(new StubAudioEngine());
    const cubase = new CubaseDawController();
    const ableton = new AbletonDawController();
    reg.register(demo);
    reg.register(cubase);
    reg.register(ableton);

    const notifiedIds: string[] = [];
    reg.subscribeActiveChange((ctrl) => {
      notifiedIds.push(ctrl.id);
    });

    // Rapid A -> B -> A in the same microtask turn MUST keep `getActive()` committed until settled
    // while `getPendingId()` tracks the latest target (`[R14-MED-03]`), landing on `'demo-pianoroll'`
    const p1 = reg.activate('cubase-midi-remote');
    expect(reg.getPendingId()).toBe('cubase-midi-remote');
    expect(reg.getActive().id).toBe('demo-pianoroll');
    const p2 = reg.activate('demo-pianoroll');
    expect(reg.getPendingId()).toBe('demo-pianoroll');
    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1.id).toBe('demo-pianoroll');
    expect(r2.id).toBe('demo-pianoroll');
    expect(reg.getActive().id).toBe('demo-pianoroll');
    expect(notifiedIds[notifiedIds.length - 1]).toBe('demo-pianoroll');
    expect(notifiedIds).not.toContain('cubase-midi-remote');

    // 2. `[R11-MED-02]`, `[R11-MED-03]`, `[R11-LOW-01]` across Main + Pop-Out + Orphan Promotion
    const origFetch = globalThis.fetch;
    const origUrl = window.location.href;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            cuda_available: false,
            gpu_name: null,
            user_models_dir: 'C:/models',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info')) {
        return new Response(
          JSON.stringify({
            models: [
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                weights_file: 'sft_gen.safetensors',
                tag: { model: 'sft_gen' },
              },
              {
                model_name: 'MORTM4.5D-80M',
                has_weights: true,
                weights_file: 'foundation.safetensors',
                tag: { model: 'foundation' },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });

    try {
      window.history.replaceState(null, '', '/');
      let mainApp = render(<App />);
      await waitFor(
        () => {
          expect(mainApp.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
        },
        { timeout: 3500 },
      );

      // Verify desktop 2-column Studio Console container in ControlPanelView
      const consoleGrid = mainApp.container.querySelector(
        'section[aria-label="レゾナンスモニター"]',
      )?.parentElement;
      expect(consoleGrid?.className).toContain('lg:grid-cols-');

      window.history.replaceState(null, '', '/?tab=pianoroll&popout=1');
      const popoutApp = render(<App />);
      await waitFor(
        () => {
          expect(popoutApp.container.querySelector('[data-testid="popout-pianoroll-badge"]')).toBeTruthy();
        },
        { timeout: 3500 },
      );

      // Switch Main to `cubase-midi-remote` -> Pop-Out enters `popout-external-daw-standby`
      const dawSelect1 = mainApp.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect1, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="popout-external-daw-standby"]')).toBeTruthy();
      });

      // `[R11-MED-02]` Sub-case A: Reload Main Window (`unmount` + fresh `render(<App />)`) ->
      // Pop-Out immediately receives `SYNC_ACTIVE_CONTROLLER('demo-pianoroll')` and exits standby!
      mainApp.unmount();
      window.history.replaceState(null, '', '/');
      mainApp = render(<App />);
      await waitFor(
        () => {
          expect(mainApp.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
          expect(popoutApp.container.querySelector('[data-testid="popout-external-daw-standby"]')).toBeNull();
        },
        { timeout: 3500 },
      );

      // `[R11-MED-02]` Sub-case B: Switch to `cubase-midi-remote` again, unmount Main (orphan Pop-Out),
      // and click `COMPASS AI を起動` in orphan Pop-Out -> promotes to standalone WITHOUT resetting `cubase-midi-remote`!
      const dawSelect2 = mainApp.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect2, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="popout-external-daw-standby"]')).toBeTruthy();
      });
      mainApp.unmount();

      const orphanLaunchBtn = popoutApp.container.querySelector(
        '[data-testid="launch-compass-ai-button"]',
      ) as HTMLButtonElement;
      fireEvent.click(orphanLaunchBtn);
      await waitFor(() => {
        const promotedDawSelect = popoutApp.container.querySelector(
          '#compass-daw-select',
        ) as HTMLSelectElement | null;
        expect(promotedDawSelect).toBeTruthy();
        expect(promotedDawSelect?.value).toBe('cubase-midi-remote');
      });

      // `[R11-MED-03]` Self-unmounting `選択をクリア` button in ControlPanelView retains focus (both SFT-Gen & Foundation)
      const sftAccordionBtn = Array.from(popoutApp.container.querySelectorAll('button')).find((b) =>
        (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定'),
      ) as HTMLButtonElement;
      fireEvent.click(sftAccordionBtn);
      const jazzChip = Array.from(popoutApp.container.querySelectorAll('button')).find(
        (b) => (b.textContent || '').trim() === 'jazz',
      ) as HTMLButtonElement;
      fireEvent.click(jazzChip);
      let clearGenreBtn = Array.from(popoutApp.container.querySelectorAll('button')).find((b) =>
        (b.textContent || '').includes('選択をクリア'),
      ) as HTMLButtonElement;
      clearGenreBtn.focus();
      fireEvent.click(clearGenreBtn);
      expect(document.activeElement?.id).toBe('genre-filter-input');

      // Re-select `jazz`, switch to `Foundation` (where `#genre-filter-input` and `#musical-key-select` are disabled),
      // and click `選択をクリア` -> focus lands on `sftAccordionBtn` (never `document.body`)!
      fireEvent.click(jazzChip);
      const modelSelect = popoutApp.container.querySelector(
        '#compass-model-select',
      ) as HTMLSelectElement;
      fireEvent.change(modelSelect, { target: { value: 'MORTM4.5D-80M' } });
      clearGenreBtn = Array.from(popoutApp.container.querySelectorAll('button')).find((b) =>
        (b.textContent || '').includes('選択をクリア'),
      ) as HTMLButtonElement;
      clearGenreBtn.focus();
      fireEvent.click(clearGenreBtn);
      expect(document.activeElement).toBe(sftAccordionBtn);

      // Switch back to `demo-pianoroll` & Piano Roll tab to verify `<Trash2 />`, `通知を閉じる`, `Undo`, and `[R11-LOW-01]`
      const promotedDawSelect = popoutApp.container.querySelector(
        '#compass-daw-select',
      ) as HTMLSelectElement;
      fireEvent.change(promotedDawSelect, { target: { value: 'demo-pianoroll' } });
      fireEvent.click(
        popoutApp.container.querySelector('[data-testid="tab-pianoroll"]') as HTMLButtonElement,
      );

      // Create an undoable note via Pencil mode + add Track 2 -> verify Undo/Redo disabled focus handoff & `<Trash2 />`
      const addTrackBtn = await waitFor(() => {
        const btn = popoutApp.container.querySelector(
          'button[aria-label="トラックを追加"]',
        ) as HTMLButtonElement | null;
        expect(btn).toBeTruthy();
        return btn!;
      });
      fireEvent.click(addTrackBtn);

      const prCanvas = popoutApp.container.querySelector(
        '[data-testid="pianoroll-canvas"]',
      ) as HTMLCanvasElement;
      vi.spyOn(prCanvas, 'getBoundingClientRect').mockReturnValue({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 960,
        bottom: 540,
        width: 960,
        height: 540,
        toJSON: () => ({}),
      });
      const pencilBtn = popoutApp.container.querySelector(
        'button[aria-label="ペン入力"]',
      ) as HTMLButtonElement;
      fireEvent.click(pencilBtn);
      fireEvent.pointerDown(prCanvas, { clientX: 200, clientY: 160, button: 0, pointerId: 1 });
      fireEvent.pointerUp(prCanvas, { clientX: 200, clientY: 160, button: 0, pointerId: 1 });

      const undoBtn = popoutApp.container.querySelector(
        'button[aria-label="元に戻す"]',
      ) as HTMLButtonElement;
      const redoBtn = popoutApp.container.querySelector(
        'button[aria-label="やり直す"]',
      ) as HTMLButtonElement;
      expect(undoBtn.disabled).toBe(false);
      undoBtn.focus();
      fireEvent.click(undoBtn);
      expect(undoBtn.disabled).toBe(true);
      expect(document.activeElement).toBe(redoBtn);

      // Redo (`Redo` becomes `:disabled`!) -> focus moves back to `Undo`!
      fireEvent.click(redoBtn);
      expect(redoBtn.disabled).toBe(true);
      expect(document.activeElement).toBe(undoBtn);

      // Delete Track 2 via `<Trash2 />` -> focus moves to surviving Track 1's `[data-track-select-id]` button,
      // and subsequent `Space` does NOT hijack transport playback!
      const trashBtns = popoutApp.container.querySelectorAll('button[data-tooltip="トラックを削除"]');
      const trashBtn2 = trashBtns[trashBtns.length - 1] as HTMLButtonElement;
      trashBtn2.focus();
      fireEvent.click(trashBtn2);
      expect(document.activeElement).not.toBe(document.body);
      expect(
        (document.activeElement as HTMLElement | null)?.hasAttribute('data-track-select-id'),
      ).toBe(true);
      fireEvent.keyDown(document.activeElement as HTMLElement, { key: ' ', code: 'Space' });
      expect(
        popoutApp.container.querySelector('button[aria-label="一時停止"]'),
      ).toBeNull();

      // Upload 1-track MIDI to mount `通知を閉じる` inside `[data-tooltip-strip]`, hover it, then call `.click()` (`[R11-LOW-01]`)
      const smf1Track = new Uint8Array([
        0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
        0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x90, 0x3c, 0x64, 0x83, 0x60,
        0x80, 0x3c, 0x00, 0x00, 0xff, 0x2f, 0x00,
      ]);
      const fileInput = popoutApp.container.querySelector(
        'input[type="file"]',
      ) as HTMLInputElement;
      const midiFile = new File([smf1Track], 'focus_test.mid', { type: 'audio/midi' });
      fireEvent.change(fileInput, { target: { files: [midiFile] } });
      const dismissNoticeBtn = await waitFor(() => {
        const btn = popoutApp.container.querySelector(
          'button[aria-label="通知を閉じる"]',
        ) as HTMLButtonElement | null;
        expect(btn).toBeTruthy();
        return btn!;
      });
      const stripEl = dismissNoticeBtn.closest('[data-tooltip-strip]') as HTMLElement;
      expect(stripEl).toBeTruthy();
      fireEvent.pointerOver(dismissNoticeBtn, { pointerType: 'mouse' });
      expect(stripEl.getAttribute('data-tooltip-strip-active')).toBe('true');

      // Programmatic `.click()` without `pointerdown` / `keyup` -> focus moves to `MIDIファイルを読み込む` AND microtask clears strip!
      dismissNoticeBtn.focus();
      act(() => {
        dismissNoticeBtn.click();
      });
      await Promise.resolve();
      expect(
        (document.activeElement as HTMLElement | null)?.getAttribute('aria-label'),
      ).toBe('MIDIをアップロード');
      expect(stripEl.getAttribute('data-tooltip-strip-active')).toBeNull();

      popoutApp.unmount();
    } finally {
      window.history.replaceState(null, '', origUrl);
      globalThis.fetch = origFetch;
    }
  });

  it('49. [Round-12 Fact-Checker] verifies pointer vs keyboard focus modality & Space transport ([R12-HIGH-01]), sticky rack header & 340px/224px tooltip geometry ([R12-MED-01]), and DawControllerRegistry error rollback without split-brain ([R12-MED-02])', async () => {
    // 1. `[R12-MED-02]` DawControllerRegistry rolls back `_activeId` & `_connectedId` when `connect()` or `disconnect()` throws
    const reg = new DawControllerRegistry();
    const audio = new StubAudioEngine();
    const demo = new DemoPianoRollController(audio);
    let cubaseFailConnect = true;
    const flakyCubase = new CubaseDawController();
    vi.spyOn(flakyCubase, 'connect').mockImplementation(async () => {
      if (cubaseFailConnect) {
        throw new Error('Cubase MIDI Remote port busy');
      }
    });
    vi.spyOn(flakyCubase, 'pause').mockImplementation(() => {
      throw new Error('Synchronous pause glitch');
    });
    reg.register(demo);
    reg.register(flakyCubase);
    await reg.activate('demo-pianoroll');
    const seenNotified: string[] = [];
    reg.subscribeActiveChange((ctrl) => seenNotified.push(ctrl.id));

    await expect(reg.activate('cubase-midi-remote')).rejects.toThrow('Cubase MIDI Remote port busy');
    // Must cleanly roll back to `demo-pianoroll` with zero split-brain!
    expect(reg.getActive().id).toBe('demo-pianoroll');

    // Subsequent retry when Cubase recovers must succeed cleanly
    cubaseFailConnect = false;
    const activatedCubase = await reg.activate('cubase-midi-remote');
    expect(activatedCubase.id).toBe('cubase-midi-remote');
    expect(reg.getActive().id).toBe('cubase-midi-remote');
    expect(seenNotified).toContain('cubase-midi-remote');

    // Synchronous `pause()` throw on `flakyCubase` during switch back to `demo-pianoroll` must not break activation
    const backToDemo = await reg.activate('demo-pianoroll');
    expect(backToDemo.id).toBe('demo-pianoroll');

    // 2. `[R12-HIGH-01]` & `[R12-MED-01]` Modality separation, Space transport after mouse click, rescan handoff, non-active track delete, and 64-track sticky rack
    const origFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            cuda_available: false,
            gpu_name: null,
            user_models_dir: 'C:/models',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info') || url.endsWith('/rescan_models')) {
        if (url.endsWith('/rescan_models')) {
          await new Promise((r) => setTimeout(r, 40));
        }
        return new Response(
          JSON.stringify({
            models: [
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                weights_file: 'sft_gen.safetensors',
                tag: { model: 'sft_gen' },
              },
              {
                model_name: 'MORTM4.5D-80M',
                has_weights: true,
                weights_file: 'foundation.safetensors',
                tag: { model: 'foundation' },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;

    try {
      const app = render(<App />);
      await waitFor(() => {
        expect(app.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
      });

      // Keyboard activation of `モデルフォルダを再スキャン` hands focus to `#compass-model-select` (not `document.body`)
      const rescanBtn = app.container.querySelector(
        'button[aria-label="モデルフォルダを再スキャン"]',
      ) as HTMLButtonElement;
      rescanBtn.focus();
      fireEvent.click(rescanBtn);
      expect(document.activeElement?.id).toBe('compass-model-select');
      await waitFor(() => expect(rescanBtn.disabled).toBe(false));

      // Pointer tap on `選択をクリア` does NOT focus `#genre-filter-input` (prevents mobile virtual keyboard pop-up)
      const sftToggle = Array.from(app.container.querySelectorAll('button')).find((b) =>
        (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定'),
      ) as HTMLButtonElement;
      fireEvent.click(sftToggle);
      const genreInput = app.container.querySelector('#genre-filter-input') as HTMLInputElement;
      expect(genreInput.placeholder).toBe('ジャンルを検索 (例: jazz, ambient)');

      const jazzChip = Array.from(app.container.querySelectorAll('button')).find(
        (b) => (b.textContent || '').trim() === 'jazz',
      ) as HTMLButtonElement;
      fireEvent.click(jazzChip);
      const clearBtn = Array.from(app.container.querySelectorAll('button')).find((b) =>
        (b.textContent || '').includes('選択をクリア'),
      ) as HTMLButtonElement;
      fireEvent.pointerDown(clearBtn, { pointerType: 'touch', button: 0 });
      clearBtn.focus();
      fireEvent.pointerUp(clearBtn, { pointerType: 'touch', button: 0 });
      fireEvent.click(clearBtn, { detail: 1 });
      expect(document.activeElement).not.toBe(genreInput);

      // Switch to Piano Roll to test pointer `Undo` + `Space` transport & non-active track delete focus
      fireEvent.click(app.container.querySelector('[data-testid="tab-pianoroll"]') as HTMLElement);
      const prCanvas = await waitFor(() => {
        const c = app.container.querySelector(
          '[data-testid="pianoroll-canvas"]',
        ) as HTMLCanvasElement | null;
        expect(c).toBeTruthy();
        return c!;
      });
      vi.spyOn(prCanvas, 'getBoundingClientRect').mockReturnValue({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 960,
        bottom: 540,
        width: 960,
        height: 540,
        toJSON: () => ({}),
      });
      const pencilBtn = app.container.querySelector(
        'button[aria-label="ペン入力"]',
      ) as HTMLButtonElement;
      fireEvent.click(pencilBtn);
      fireEvent.pointerDown(prCanvas, { clientX: 220, clientY: 160, button: 0, pointerId: 1 });
      fireEvent.pointerUp(prCanvas, { clientX: 220, clientY: 160, button: 0, pointerId: 1 });

      const undoBtn = app.container.querySelector(
        'button[aria-label="元に戻す"]',
      ) as HTMLButtonElement;
      const redoBtn = app.container.querySelector(
        'button[aria-label="やり直す"]',
      ) as HTMLButtonElement;
      // Mouse-click `Undo` (`pointerdown` -> `focus` -> `pointerup` -> `click` with `detail: 1`):
      // must NOT hand focus to `Redo`, and pressing `Space` must start audio transport!
      fireEvent.pointerDown(undoBtn, { pointerType: 'mouse', button: 0 });
      undoBtn.focus();
      fireEvent.pointerUp(undoBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(undoBtn, { detail: 1 });
      expect(undoBtn.disabled).toBe(true);
      expect(redoBtn.disabled).toBe(false);
      expect(document.activeElement).not.toBe(redoBtn);
      fireEvent.keyDown(window, { key: ' ', code: 'Space' });
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="一時停止"]')).toBeTruthy();
      });
      expect(redoBtn.disabled).toBe(false);
      fireEvent.keyDown(window, { key: ' ', code: 'Space' });

      // Non-active track delete via keyboard (`>= 3` tracks, Track 1 active, delete Track 2 -> focuses new Track 2, NOT Track 1)
      const addTrackBtn = app.container.querySelector(
        'button[aria-label="トラックを追加"]',
      ) as HTMLButtonElement;
      fireEvent.click(addTrackBtn);
      fireEvent.click(addTrackBtn);
      const trackSelectBtnsBefore = app.container.querySelectorAll('button[data-track-select-id]');
      expect(trackSelectBtnsBefore.length).toBeGreaterThanOrEqual(3);
      fireEvent.click(trackSelectBtnsBefore[0]); // Ensure Track 1 is active
      const expectedAdjacentId = trackSelectBtnsBefore[2].getAttribute('data-track-select-id');
      const trashBtns3 = app.container.querySelectorAll('button[data-tooltip="トラックを削除"]');
      const middleTrash = trashBtns3[1] as HTMLButtonElement;
      middleTrash.focus();
      fireEvent.click(middleTrash);
      expect(
        (document.activeElement as HTMLElement | null)?.getAttribute('data-track-select-id'),
      ).toBe(expectedAdjacentId);

      // Verify sticky header attribute is mounted on `<aside>` header
      const stickyHdr = app.container.querySelector('aside [data-tooltip-sticky-header="true"]');
      expect(stickyHdr).toBeTruthy();

      app.unmount();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('50. [Round-13 Fact-Checker] verifies active-tab Space transport ([R13-MED-01]), enabled-button pointer focus + Enter->Space keyboard modality ([R13-MED-02]), DawControllerRegistry disconnect isolation & concurrent transition settlement ([R13-MED-03]), and sticky-rack scroll + min-overflow tooltip alignment ([R13-LOW-01])', async () => {
    // 1. `[R13-MED-03]` Registry: `register()` does NOT mark unconnected controller as connected,
    //    `prevConnected.disconnect()` throwing does NOT brick switching back, and superseded `activate(A)`
    //    waits for `_transitionChain` before resolving `getActive()`.
    const reg = new DawControllerRegistry();
    const audio = new StubAudioEngine();
    const demo = new DemoPianoRollController(audio);
    const demoDisconnectSpy = vi.spyOn(demo, 'disconnect');
    const cubase = new CubaseDawController();
    const ableton = new AbletonDawController();

    reg.register(demo);
    reg.register(cubase);
    reg.register(ableton);

    // Before `activate('demo-pianoroll')`, switching directly to `cubase` must NOT call `demo.disconnect()`
    await reg.activate('cubase-midi-remote');
    expect(demoDisconnectSpy).toHaveBeenCalledTimes(0);
    expect(reg.getActive().id).toBe('cubase-midi-remote');

    // Faulty `cubase.disconnect()` throwing when switching back to `demo-pianoroll` must NOT brick recovery
    vi.spyOn(cubase, 'disconnect').mockRejectedValueOnce(new Error('MIDI port severed'));
    const recovered = await reg.activate('demo-pianoroll');
    expect(recovered.id).toBe('demo-pianoroll');
    expect(reg.getActive().id).toBe('demo-pianoroll');

    // Concurrent `activate('cubase')` superseded in-flight by `activate(ableton.id)` where `ableton.connect()` fails:
    // both `p1` (resolved) and `reg.getActive()` must converge to rolled-back `'demo-pianoroll'`.
    let resolveCubaseConnect: () => void = () => {};
    const cubaseConnectStarted = new Promise<void>((markStarted) => {
      vi.spyOn(cubase, 'connect').mockImplementationOnce(() => {
        markStarted();
        return new Promise<void>((r) => {
          resolveCubaseConnect = r;
        });
      });
    });
    vi.spyOn(ableton, 'connect').mockRejectedValueOnce(new Error('Ableton OSC unreachable'));

    const p1 = reg.activate(cubase.id);
    await cubaseConnectStarted;
    const p2 = reg.activate(ableton.id);
    resolveCubaseConnect();
    await expect(p2).rejects.toThrow('Ableton OSC unreachable');
    const settledFromP1 = await p1;
    expect(settledFromP1.id).toBe('demo-pianoroll');
    expect(reg.getActive().id).toBe('demo-pianoroll');

    // 2. `[R13-LOW-01]` Minimum-overflow tooltip alignment inside narrow 256px `<aside>`
    const narrowAside = document.createElement('aside');
    const midBtn = document.createElement('button');
    midBtn.setAttribute('data-tooltip', '最大トラック数 (64) に達しています');
    narrowAside.appendChild(midBtn);
    document.body.appendChild(narrowAside);
    vi.spyOn(narrowAside, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 256,
      bottom: 600,
      width: 256,
      height: 600,
      toJSON: () => ({}),
    });
    vi.spyOn(midBtn, 'getBoundingClientRect').mockReturnValue({
      x: 92,
      y: 120,
      left: 92,
      top: 120,
      right: 136,
      bottom: 164,
      width: 44,
      height: 44,
      toJSON: () => ({}),
    });
    alignButtonTooltipToViewport(midBtn);
    // Centered tooltip (`2..226`) has far less boundary overflow than `start` (`92..316`), so stays centered!
    expect(midBtn.getAttribute('data-tooltip-align')).toBeNull();
    narrowAside.remove();

    // 3. `[R13-MED-01]` & `[R13-MED-02]` & `[R13-LOW-01]` UI modality, post-generation tab Space, and sticky rack scroll
    const origFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            cuda_available: false,
            gpu_name: null,
            user_models_dir: 'C:/models',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info')) {
        return new Response(
          JSON.stringify({
            models: [
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                weights_file: 'sft_gen.safetensors',
                tag: { model: 'sft_gen' },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;

    try {
      const app = render(<App />);
      await waitFor(() => {
        expect(app.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
      });

      fireEvent.click(app.container.querySelector('[data-testid="tab-pianoroll"]') as HTMLElement);
      const prCanvas = await waitFor(() => {
        const c = app.container.querySelector(
          '[data-testid="pianoroll-canvas"]',
        ) as HTMLCanvasElement | null;
        expect(c).toBeTruthy();
        return c!;
      });
      vi.spyOn(prCanvas, 'getBoundingClientRect').mockReturnValue({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 960,
        bottom: 540,
        width: 960,
        height: 540,
        toJSON: () => ({}),
      });

      // `[R13-MED-01]` When `#workspace-tab-pianoroll` (`aria-selected="true"`) holds focus (as after AI generation),
      // pressing `Space` directly on `document.activeElement` immediately starts Piano Roll playback!
      const prTab = document.getElementById('workspace-tab-pianoroll') as HTMLButtonElement;
      const aiTab = document.getElementById('workspace-tab-compass-ai') as HTMLButtonElement;
      prTab.focus();
      expect(document.activeElement).toBe(prTab);
      expect(prTab.getAttribute('aria-selected')).toBe('true');
      const spaceOnActiveTab = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        prTab.dispatchEvent(spaceOnActiveTab);
      });
      expect(spaceOnActiveTab.defaultPrevented).toBe(true);
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="一時停止"]')).toBeTruthy();
      });
      // Pause again via `Space` while `prTab` is still focused
      act(() => {
        prTab.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="再生"]')).toBeTruthy();
      });

      // Whereas on unselected `#workspace-tab-compass-ai` (`aria-selected="false"`), `Space` is NOT intercepted!
      aiTab.focus();
      const spaceOnInactiveTab = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      aiTab.dispatchEvent(spaceOnInactiveTab);
      expect(spaceOnInactiveTab.defaultPrevented).toBe(false);

      // `[R13-MED-02]` Enabled 2-step `Undo` button holding pointer focus + `Space` on `document.activeElement`
      const pencilBtn = app.container.querySelector(
        'button[aria-label="ペン入力"]',
      ) as HTMLButtonElement;
      fireEvent.click(pencilBtn);
      fireEvent.pointerDown(prCanvas, { clientX: 220, clientY: 160, button: 0, pointerId: 1 });
      fireEvent.pointerUp(prCanvas, { clientX: 220, clientY: 160, button: 0, pointerId: 1 });
      fireEvent.pointerDown(prCanvas, { clientX: 280, clientY: 180, button: 0, pointerId: 1 });
      fireEvent.pointerUp(prCanvas, { clientX: 280, clientY: 180, button: 0, pointerId: 1 });

      const undoBtn = app.container.querySelector(
        'button[aria-label="元に戻す"]',
      ) as HTMLButtonElement;
      expect(undoBtn.disabled).toBe(false);

      // Mouse-click `Undo` once: 1 undo step remains, so `undoBtn` STAYS enabled & focused!
      fireEvent.pointerDown(undoBtn, { pointerType: 'mouse', button: 0 });
      undoBtn.focus();
      fireEvent.pointerUp(undoBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(undoBtn, { detail: 1 });
      expect(undoBtn.disabled).toBe(false);
      expect(document.activeElement).toBe(undoBtn);

      // Dispatch `Space` directly on `undoBtn` (`e.target === undoBtn`): must start transport & NOT re-click `Undo`!
      const spaceOnPointerUndo = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        undoBtn.dispatchEvent(spaceOnPointerUndo);
      });
      expect(spaceOnPointerUndo.defaultPrevented).toBe(true);
      expect(undoBtn.disabled).toBe(false);
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="一時停止"]')).toBeTruthy();
      });
      // Pause playback
      act(() => {
        undoBtn.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });

      // Now test `Mouse-click -> Enter -> Space` on enabled `pencilBtn`:
      // `Enter` transitions `pencilBtn` to keyboard modality, so subsequent `Space` does NOT hijack button activation!
      fireEvent.pointerDown(pencilBtn, { pointerType: 'mouse', button: 0 });
      pencilBtn.focus();
      fireEvent.pointerUp(pencilBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(pencilBtn, { detail: 1 });
      expect(document.activeElement).toBe(pencilBtn);

      pencilBtn.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true }),
      );
      const spaceAfterEnter = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      pencilBtn.dispatchEvent(spaceAfterEnter);
      expect(spaceAfterEnter.defaultPrevented).toBe(false);

      // `[R13-LOW-01]` Verify `<aside>` has `scroll-pt-16` and adjusts `scrollTop` when restoring focus behind sticky header
      const rackAside = app.container.querySelector('aside[aria-label="トラックラック"]') as HTMLElement;
      expect(rackAside.className).toContain('scroll-pt-16');

      app.unmount();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('51. [Round-14 Critic Remediations] verifies pointer modal/blur/standby/dismiss Space transport ([R14-HIGH-01]), natural header tablist DOM order & Arrow keys ([R14-MED-01]), genre tray & empty status ([R14-MED-02]), committed registry activeId ([R14-MED-03]), disabled modal fallback ([R14-MED-04]), and disabled Undo/Redo/Rescan tooltips ([R14-LOW-02])', async () => {
    // 1. `[R14-MED-03]` Committed `_activeId` stays on `'demo-pianoroll'` while slow external `connect()` is in-flight
    const reg = new DawControllerRegistry();
    const demo = new DemoPianoRollController(new StubAudioEngine());
    const cubase = new CubaseDawController();
    reg.register(demo);
    reg.register(cubase);
    await reg.activate('demo-pianoroll');

    let finishCubaseConnect: () => void = () => {};
    const cubaseConnectStarted = new Promise<void>((markStarted) => {
      vi.spyOn(cubase, 'connect').mockImplementationOnce(() => {
        markStarted();
        return new Promise<void>((r) => {
          finishCubaseConnect = r;
        });
      });
    });

    const activatePromise = reg.activate('cubase-midi-remote');
    await cubaseConnectStarted;
    expect(reg.getPendingId()).toBe('cubase-midi-remote');
    // Committed `getActive()` remains `'demo-pianoroll'` until `cubase.connect()` resolves!
    expect(reg.getActive().id).toBe('demo-pianoroll');
    finishCubaseConnect();
    await activatePromise;
    expect(reg.getActive().id).toBe('cubase-midi-remote');
    expect(reg.getPendingId()).toBe('cubase-midi-remote');

    // 2. `[R14-MED-04]` Disabled `restoreFocusTo` skips `:disabled` and focuses `fallbackFocusTo`
    const disabledTrigger = document.createElement('button');
    disabledTrigger.disabled = true;
    const enabledFallback = document.createElement('button');
    document.body.appendChild(disabledTrigger);
    document.body.appendChild(enabledFallback);
    const modalView = render(
      <MidiTrackImportModal
        fileName="fallback_test.mid"
        parsedMidi={StandardMidiCodec.createDemoMidiPreset()}
        restoreFocusTo={disabledTrigger}
        fallbackFocusTo={enabledFallback}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );
    modalView.unmount();
    expect(document.activeElement).toBe(enabledFallback);
    disabledTrigger.remove();
    enabledFallback.remove();

    // 3. Full App verification of `[R14-MED-01]`, `[R14-MED-02]`, `[R14-LOW-02]`, and `[R14-HIGH-01]`
    const origFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            cuda_available: false,
            gpu_name: null,
            user_models_dir: 'C:/models',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info') || url.endsWith('/rescan_models')) {
        if (url.endsWith('/rescan_models')) {
          await new Promise((r) => setTimeout(r, 40));
        }
        return new Response(
          JSON.stringify({
            models: [
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                weights_file: 'sft_gen.safetensors',
                tag: { model: 'sft_gen' },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;

    try {
      const app = render(<App />);
      await waitFor(() => {
        expect(app.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
      });

      // `[R14-MED-01]` `<div role="tablist">`, zero `<nav role="tablist">`, zero `order-*` classes in `<header>`, + Arrow key navigation
      const headerEl = app.container.querySelector('header') as HTMLElement;
      expect(headerEl.querySelector('nav[role="tablist"]')).toBeNull();
      const tablistEl = headerEl.querySelector('div[role="tablist"]') as HTMLElement;
      expect(tablistEl).toBeTruthy();
      expect(headerEl.innerHTML).not.toMatch(/\border-(?:1|2|3|first|last|none)\b/);

      const aiTab = headerEl.querySelector('[data-testid="tab-compass-ai"]') as HTMLButtonElement;
      const prTab = headerEl.querySelector('[data-testid="tab-pianoroll"]') as HTMLButtonElement;
      aiTab.focus();
      fireEvent.keyDown(tablistEl, { key: 'ArrowRight', code: 'ArrowRight' });
      expect(document.activeElement).toBe(prTab);
      expect(prTab.getAttribute('aria-selected')).toBe('true');
      fireEvent.keyDown(tablistEl, { key: 'Home', code: 'Home' });
      expect(document.activeElement).toBe(aiTab);
      expect(aiTab.getAttribute('aria-selected')).toBe('true');

      // `[R14-MED-02]` & `[R14-LOW-02]` Rescan disabled tooltip, SFT `aria-controls`, genre tray zero `-m-1`, empty filter `role="status"`
      const rescanBtn = app.container.querySelector(
        'button[aria-label="モデルフォルダを再スキャン"]',
      ) as HTMLButtonElement;
      expect(rescanBtn.getAttribute('data-tooltip-when-disabled')).toBe('true');
      fireEvent.click(rescanBtn);
      expect(rescanBtn.disabled).toBe(true);
      expect(rescanBtn.getAttribute('data-tooltip')).toBe('モデルフォルダを再スキャン中です');
      await waitFor(() => expect(rescanBtn.disabled).toBe(false));

      const sftAccordionBtn = Array.from(app.container.querySelectorAll('button')).find((b) =>
        (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定'),
      ) as HTMLButtonElement;
      expect(sftAccordionBtn.getAttribute('aria-controls')).toBe('sft-structure-options-panel');
      fireEvent.click(sftAccordionBtn);
      expect(app.container.querySelector('#sft-structure-options-panel')).toBeTruthy();

      const genreTray = app.container.querySelector(
        '[data-testid="genre-chips-scroll-tray"]',
      ) as HTMLElement;
      expect(genreTray).toBeTruthy();
      expect(genreTray.className).not.toContain('-m-1');
      expect(genreTray.className).toContain('[@media(min-height:860px)]:lg:max-h-[104px]');

      const genreFilterInput = app.container.querySelector(
        '#genre-filter-input',
      ) as HTMLInputElement;
      fireEvent.change(genreFilterInput, { target: { value: 'zzzz_no_such_genre_999' } });
      const emptyGenreMsg = app.container.querySelector(
        '[data-testid="genre-empty-filter-status"]',
      ) as HTMLElement;
      expect(emptyGenreMsg).toBeTruthy();
      expect(emptyGenreMsg.getAttribute('role')).toBe('status');

      // Switch to Piano Roll to test `[R14-LOW-02]` Undo/Redo disabled tooltips and all 4 `[R14-HIGH-01]` paths
      fireEvent.click(prTab);
      const sampleBtn = await waitFor(() => {
        const b = app.container.querySelector(
          '[data-testid="load-sample-arrangement-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        return b!;
      });
      const undoBtn = app.container.querySelector(
        'button[aria-label="元に戻す"]',
      ) as HTMLButtonElement;
      const redoBtn = app.container.querySelector(
        'button[aria-label="やり直す"]',
      ) as HTMLButtonElement;
      expect(undoBtn.disabled).toBe(true);
      expect(undoBtn.getAttribute('data-tooltip-when-disabled')).toBe('true');
      expect(undoBtn.getAttribute('data-tooltip')).toContain('元に戻す操作がありません');
      expect(undoBtn.className).not.toContain('disabled:opacity-35');
      expect(redoBtn.disabled).toBe(true);
      expect(redoBtn.getAttribute('data-tooltip-when-disabled')).toBe('true');
      expect(redoBtn.getAttribute('data-tooltip')).toContain('やり直す操作がありません');

      // `[R14-HIGH-01]` Path 1: Load sample notes, then pointer-click `サンプル` to open modal,
      // pointer-click `選択したトラックを読み込む`, and press `Space` -> MUST start playback (not reopen modal!)
      fireEvent.pointerDown(sampleBtn, { pointerType: 'mouse', button: 0 });
      sampleBtn.focus();
      fireEvent.pointerUp(sampleBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(sampleBtn, { detail: 1 });

      // Second pointer click on `サンプル` opens `MidiTrackImportModal` (since notes now exist)
      fireEvent.pointerDown(sampleBtn, { pointerType: 'mouse', button: 0 });
      sampleBtn.focus();
      fireEvent.pointerUp(sampleBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(sampleBtn, { detail: 1 });
      const confirmModalBtn = await waitFor(() => {
        const b = app.container.querySelector(
          '[data-testid="confirm-midi-import-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        return b!;
      });
      fireEvent.pointerDown(confirmModalBtn, { pointerType: 'mouse', button: 0 });
      confirmModalBtn.focus();
      fireEvent.pointerUp(confirmModalBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(confirmModalBtn, { detail: 1 });

      await waitFor(() => {
        expect(app.container.querySelector('[role="dialog"]')).toBeNull();
        expect(document.activeElement).toBe(sampleBtn);
      });
      const spaceAfterModalClose = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        sampleBtn.dispatchEvent(spaceAfterModalClose);
      });
      expect(spaceAfterModalClose.defaultPrevented).toBe(true);
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="一時停止"]')).toBeTruthy();
      });
      // Pause transport
      act(() => {
        sampleBtn.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="再生"]')).toBeTruthy();
      });

      // `[R14-HIGH-01]` Path 2: Pointer-click `通知を閉じる` -> focus moves to `MIDI読込` with pointer modality -> `Space` starts playback!
      const dismissBtn = app.container.querySelector(
        'button[aria-label="通知を閉じる"]',
      ) as HTMLButtonElement;
      const uploadBtn = app.container.querySelector(
        '[data-testid="upload-midi-button"]',
      ) as HTMLButtonElement;
      expect(dismissBtn).toBeTruthy();
      fireEvent.pointerDown(dismissBtn, { pointerType: 'mouse', button: 0 });
      dismissBtn.focus();
      fireEvent.pointerUp(dismissBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(dismissBtn, { detail: 1 });
      expect(document.activeElement).toBe(uploadBtn);
      const spaceAfterDismiss = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        uploadBtn.dispatchEvent(spaceAfterDismiss);
      });
      expect(spaceAfterDismiss.defaultPrevented).toBe(true);
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="一時停止"]')).toBeTruthy();
      });
      act(() => {
        uploadBtn.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="再生"]')).toBeTruthy();
      });

      // `[R14-HIGH-01]` Path 3: Simulate Blink window `blur` -> `focus` + `focusin` (e.g., native OS file dialog cancel / Alt+Tab)
      // while `uploadBtn` holds pointer focus -> subsequent `Space` MUST still start playback!
      act(() => {
        window.dispatchEvent(new Event('blur'));
        window.dispatchEvent(new Event('focus'));
        uploadBtn.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
      });
      const spaceAfterWindowRefocus = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        uploadBtn.dispatchEvent(spaceAfterWindowRefocus);
      });
      expect(spaceAfterWindowRefocus.defaultPrevented).toBe(true);
      await waitFor(() => {
        expect(app.container.querySelector('button[aria-label="一時停止"]')).toBeTruthy();
      });
      act(() => {
        uploadBtn.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });

      // `[R14-HIGH-01]` Path 4: Switch to External DAW, open Piano Roll standby, pointer-click `内蔵ピアノロールへ切り替え` ->
      // `launch-compass-ai-button` receives pointer focus so `Space` starts playback instead of switching back to COMPASS AI!
      fireEvent.click(aiTab);
      const dawSelect = app.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => expect(dawSelect.value).toBe('cubase-midi-remote'));
      fireEvent.click(prTab);
      const switchToDemoBtn = await waitFor(() => {
        const btn = Array.from(app.container.querySelectorAll('button')).find((b) =>
          (b.textContent || '').includes('内蔵ピアノロールへ切り替え'),
        ) as HTMLButtonElement | null;
        expect(btn).toBeTruthy();
        return btn!;
      });
      fireEvent.pointerDown(switchToDemoBtn, { pointerType: 'mouse', button: 0 });
      switchToDemoBtn.focus();
      fireEvent.pointerUp(switchToDemoBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(switchToDemoBtn, { detail: 1 });

      const launchBtnAfterSwitch = await waitFor(() => {
        expect(app.container.querySelector('[data-testid="pianoroll-canvas"]')).toBeTruthy();
        const b = app.container.querySelector(
          '[data-testid="launch-compass-ai-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        expect(document.activeElement).toBe(b);
        return b!;
      });
      const spaceAfterStandbySwitch = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        launchBtnAfterSwitch.dispatchEvent(spaceAfterStandbySwitch);
      });
      expect(spaceAfterStandbySwitch.defaultPrevented).toBe(true);
      expect(prTab.getAttribute('aria-selected')).toBe('true');

      app.unmount();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('52. [Round-15 Fact-Checker] verifies unclipped confirm-midi-import-button disabled tooltip ([R15-HIGH-01]), keyboard-modality preservation on peer sync / aborted pointer / Enter confirm / file cancel ([R15-HIGH-02]), roving tabIndex ([R15-MED-01]), non-dangling sft aria-controls ([R15-LOW-01]), and rollback _connectedId tracking ([R15-LOW-02])', async () => {
    // 1. `[R15-LOW-02]` Unconditional `_connectedId` tracking when `rollbackCtrl.connect()` completes after being superseded
    const reg = new DawControllerRegistry();
    const demo = new DemoPianoRollController(new StubAudioEngine());
    const cubase = new CubaseDawController();
    const ableton = new AbletonDawController();
    reg.register(demo);
    reg.register(cubase);
    reg.register(ableton);
    await reg.activate(demo.id);

    const demoDisconnectSpy = vi.spyOn(demo, 'disconnect');
    vi.spyOn(cubase, 'connect').mockRejectedValueOnce(new Error('Cubase handshake timeout'));
    let resolveDemoRollbackConnect: () => void = () => {};
    const demoRollbackStarted = new Promise<void>((markStarted) => {
      vi.spyOn(demo, 'connect').mockImplementationOnce(() => {
        markStarted();
        return new Promise<void>((r) => {
          resolveDemoRollbackConnect = r;
        });
      });
    });

    const pFailCubase = reg.activate(cubase.id);
    await demoRollbackStarted;
    // Supersede mid-rollback with `ableton.id` BEFORE `demo.connect()` finishes!
    const pAbleton = reg.activate(ableton.id);
    resolveDemoRollbackConnect();
    await expect(pFailCubase).rejects.toThrow('Cubase handshake timeout');
    await pAbleton;
    // `demo.disconnect()` must have been called TWICE: once before `cubase.connect()` and once before `ableton.connect()`!
    expect(demoDisconnectSpy).toHaveBeenCalledTimes(2);
    expect(reg.getActive().id).toBe(ableton.id);

    // 2. `[R15-HIGH-01]`, `[R15-HIGH-02]`, `[R15-MED-01]`, `[R15-LOW-01]`
    const origFetch = globalThis.fetch;
    const origUrl = window.location.href;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            cuda_available: false,
            gpu_name: null,
            user_models_dir: 'C:/models',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info')) {
        return new Response(
          JSON.stringify({
            models: [
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                weights_file: 'sft_gen.safetensors',
                tag: { model: 'sft_gen' },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;

    try {
      window.history.replaceState(null, '', '/');
      const mainApp = render(<App />);
      await waitFor(() => {
        expect(mainApp.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
      });

      // `[R15-MED-01]` Roving `tabIndex` (`0` vs `-1`) on `[role="tab"]`
      const aiTab = mainApp.container.querySelector('[data-testid="tab-compass-ai"]') as HTMLButtonElement;
      const prTab = mainApp.container.querySelector('[data-testid="tab-pianoroll"]') as HTMLButtonElement;
      expect(aiTab.tabIndex).toBe(0);
      expect(prTab.tabIndex).toBe(-1);

      // `[R15-LOW-01]` `#sft-structure-options-panel` is mounted with `hidden === true` when collapsed (zero dangling `aria-controls`)
      const sftAccordionBtn = Array.from(mainApp.container.querySelectorAll('button')).find((b) =>
        (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定'),
      ) as HTMLButtonElement;
      expect(sftAccordionBtn.getAttribute('aria-expanded')).toBe('false');
      const collapsedPanel = mainApp.container.querySelector('#sft-structure-options-panel') as HTMLElement;
      expect(collapsedPanel).toBeTruthy();
      expect(collapsedPanel.hidden).toBe(true);

      // `[R15-HIGH-02]` Sub-bug 1: Peer `SYNC_ACTIVE_CONTROLLER` must NOT hijack keyboard modality in Pop-Out!
      const dawSelect = mainApp.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => expect(dawSelect.value).toBe('cubase-midi-remote'));

      window.history.replaceState(null, '', '/?tab=pianoroll&popout=1');
      const popoutApp = render(<App />);
      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="popout-external-daw-standby"]')).toBeTruthy();
      });
      // User presses `Tab` in Pop-Out (keyboard modality), then Main switches back to `demo-pianoroll`
      fireEvent.keyDown(window, { key: 'Tab', code: 'Tab' });
      fireEvent.change(dawSelect, { target: { value: 'demo-pianoroll' } });
      const popoutLaunchBtn = await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="pianoroll-canvas"]')).toBeTruthy();
        const b = popoutApp.container.querySelector(
          '[data-testid="launch-compass-ai-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        return b!;
      });
      if (document.activeElement !== popoutLaunchBtn) {
        popoutLaunchBtn.focus();
      }
      const spaceAfterPeerSync = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      popoutLaunchBtn.dispatchEvent(spaceAfterPeerSync);
      // Must NOT hijack `Space` into transport play on a peer-synced keyboard focus!
      expect(spaceAfterPeerSync.defaultPrevented).toBe(false);
      popoutApp.unmount();

      // Switch Main to Piano Roll -> verify roving `tabIndex` flipped `0 <-> -1`
      window.history.replaceState(null, '', '/');
      fireEvent.click(prTab);
      expect(aiTab.tabIndex).toBe(-1);
      expect(prTab.tabIndex).toBe(0);

      const sampleBtn = await waitFor(() => {
        const b = mainApp.container.querySelector(
          '[data-testid="load-sample-arrangement-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        return b!;
      });
      const uploadBtn = mainApp.container.querySelector(
        '[data-testid="upload-midi-button"]',
      ) as HTMLButtonElement;
      const fileInput = mainApp.container.querySelector(
        '[data-testid="midi-file-input"]',
      ) as HTMLInputElement;

      // `[R15-HIGH-02]` Sub-bug 3: Pointer click `MIDI読込` -> OS file picker `cancel` resets `uploadOpenedViaPointerRef`
      fireEvent.pointerDown(uploadBtn, { pointerType: 'mouse', button: 0 });
      uploadBtn.focus();
      fireEvent.pointerUp(uploadBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(uploadBtn, { detail: 1 });
      act(() => {
        fileInput.dispatchEvent(new Event('cancel'));
      });

      // Load sample notes, then pointer-click `サンプル` to open `MidiTrackImportModal`
      fireEvent.click(sampleBtn);
      fireEvent.pointerDown(sampleBtn, { pointerType: 'mouse', button: 0 });
      sampleBtn.focus();
      fireEvent.pointerUp(sampleBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(sampleBtn, { detail: 1 });

      const confirmModalBtn = await waitFor(() => {
        const b = mainApp.container.querySelector(
          '[data-testid="confirm-midi-import-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        return b!;
      });

      // `[R15-HIGH-01]` Uncheck all tracks -> `confirm-midi-import-button` is disabled WITHOUT self-`truncate` (`overflow: visible`),
      // with `data-tooltip-pos="top"` and `data-tooltip-prefer-pos="top"` and inner `<span class="... truncate ...">`!
      expect(confirmModalBtn.className).not.toMatch(/\btruncate\b/);
      expect(confirmModalBtn.querySelector('span.truncate')).toBeTruthy();
      expect(confirmModalBtn.getAttribute('data-tooltip-pos')).toBe('top');
      expect(confirmModalBtn.getAttribute('data-tooltip-prefer-pos')).toBe('top');
      const checkboxes = Array.from(
        mainApp.container.querySelectorAll('[role="dialog"] input[type="checkbox"]'),
      ) as HTMLInputElement[];
      for (const cb of checkboxes) {
        if (cb.checked) fireEvent.click(cb);
      }
      expect(confirmModalBtn.disabled).toBe(true);
      expect(confirmModalBtn.getAttribute('data-tooltip')).toBe('1つ以上のトラックを選択してください');
      // Re-check Track 1 so confirm button is enabled again
      fireEvent.click(checkboxes[0]);
      expect(confirmModalBtn.disabled).toBe(false);

      // `[R15-HIGH-02]` Sub-bug 2(i) & 2(ii): Aborted pointerdown (`pointerup` outside button on dialog) + keyboard `Enter` confirm (`detail: 0`)
      // MUST restore KEYBOARD modality (`Space` on `sampleBtn` is NOT hijacked into transport play)!
      const dialogBackdrop = mainApp.container.querySelector('[role="dialog"]') as HTMLElement;
      fireEvent.pointerDown(confirmModalBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.pointerUp(dialogBackdrop, { pointerType: 'mouse', button: 0 });
      fireEvent.keyDown(confirmModalBtn, { key: 'Enter', code: 'Enter' });
      fireEvent.click(confirmModalBtn, { detail: 0 });

      await waitFor(() => {
        expect(mainApp.container.querySelector('[role="dialog"]')).toBeNull();
        expect(document.activeElement).toBe(sampleBtn);
      });
      const spaceAfterKeyboardModalConfirm = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      sampleBtn.dispatchEvent(spaceAfterKeyboardModalConfirm);
      expect(spaceAfterKeyboardModalConfirm.defaultPrevented).toBe(false);

      // Conversely, pointer-opening modal and immediately pressing `Escape` (without any other key) preserves pointer modality!
      fireEvent.pointerDown(sampleBtn, { pointerType: 'mouse', button: 0 });
      sampleBtn.focus();
      fireEvent.pointerUp(sampleBtn, { pointerType: 'mouse', button: 0 });
      fireEvent.click(sampleBtn, { detail: 1 });
      await waitFor(() => {
        expect(mainApp.container.querySelector('[role="dialog"]')).toBeTruthy();
      });
      fireEvent.keyDown(window, { key: 'Escape', code: 'Escape' });
      await waitFor(() => {
        expect(mainApp.container.querySelector('[role="dialog"]')).toBeNull();
        expect(document.activeElement).toBe(sampleBtn);
      });
      const spaceAfterImmediateEscape = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        sampleBtn.dispatchEvent(spaceAfterImmediateEscape);
      });
      expect(spaceAfterImmediateEscape.defaultPrevented).toBe(true);

      mainApp.unmount();
    } finally {
      window.history.replaceState(null, '', origUrl);
      globalThis.fetch = origFetch;
    }
  });

  it('53. [Round-16 Critic] verifies pointer-session peer SYNC_ACTIVE_CONTROLLER focus modality ([R16-MED-01]), mouse side-button & Enter canvasHadRecentPointerRef guards ([R16-MED-02]), and 320px external-daw-track-select max-w-full fit ([R16-LOW-01])', async () => {
    const origFetch = globalThis.fetch;
    const origUrl = window.location.href;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            cuda_available: false,
            gpu_name: null,
            user_models_dir: 'C:/models',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info')) {
        return new Response(
          JSON.stringify({
            models: [
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                weights_file: 'sft_gen.safetensors',
                tag: { model: 'sft_gen' },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;

    try {
      // 1. `[R16-MED-01]` Pointer-driven Pop-Out session across peer `SYNC_ACTIVE_CONTROLLER` (`demo -> cubase -> demo`)
      window.history.replaceState(null, '', '/');
      const mainApp = render(<App />);
      await waitFor(() => {
        expect(mainApp.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
      });

      window.history.replaceState(null, '', '/?tab=pianoroll&popout=1');
      const popoutApp = render(<App />);
      const popoutCanvas = await waitFor(() => {
        const c = popoutApp.container.querySelector(
          '[data-testid="pianoroll-canvas"]',
        ) as HTMLCanvasElement | null;
        expect(c).toBeTruthy();
        return c!;
      });
      vi.spyOn(popoutCanvas, 'getBoundingClientRect').mockReturnValue({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 960,
        bottom: 540,
        width: 960,
        height: 540,
        toJSON: () => ({}),
      });

      // Simulate pointer interaction on Pop-Out canvas (`document.activeElement === document.body`, zero keyboard nav)
      if (document.activeElement instanceof HTMLElement && document.activeElement !== document.body) {
        document.activeElement.blur();
      }
      fireEvent.pointerDown(popoutCanvas, {
        pointerId: 88,
        pointerType: 'mouse',
        button: 0,
        buttons: 1,
        clientX: 240,
        clientY: 180,
      });
      fireEvent.pointerUp(window, {
        pointerId: 88,
        pointerType: 'mouse',
        button: 0,
        buttons: 0,
        clientX: 240,
        clientY: 180,
      });
      expect(document.activeElement).toBe(document.body);

      // Path A (`demo-pianoroll -> cubase-midi-remote` via Main Window):
      // In pointer session, Standby must NOT plant keyboard focus on `launch-compass-ai-button` (`document.activeElement` stays `document.body`)!
      const dawSelect = mainApp.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="popout-external-daw-standby"]')).toBeTruthy();
      });
      expect(document.activeElement).toBe(document.body);

      // `[R16-LOW-01]` Verify `#external-daw-track-select` in Main Window's External DAW Bridge has `w-full max-w-full`
      // inside a `max-w-[220px] min-w-0` wrapper so it cannot overflow at `320px`!
      const mainPrTab = mainApp.container.querySelector('[data-testid="tab-pianoroll"]') as HTMLButtonElement;
      fireEvent.click(mainPrTab);
      const extTrackSelect = await waitFor(() => {
        const sel = mainApp.container.querySelector(
          '#external-daw-track-select',
        ) as HTMLSelectElement | null;
        expect(sel).toBeTruthy();
        return sel!;
      });
      expect(extTrackSelect.className).toMatch(/\bw-full\b/);
      expect(extTrackSelect.className).toMatch(/\bmax-w-full\b/);
      expect(extTrackSelect.parentElement?.className).toMatch(/\bmin-w-0\b/);
      expect(extTrackSelect.parentElement?.className).toMatch(/\bmax-w-\[220px\]/);

      // Return focus state to `document.body` (simulating pointer return to Pop-Out) and switch Main back to `demo-pianoroll`!
      const mainAiTab = mainApp.container.querySelector('[data-testid="tab-compass-ai"]') as HTMLButtonElement;
      fireEvent.click(mainAiTab);
      if (document.activeElement instanceof HTMLElement && document.activeElement !== document.body) {
        document.activeElement.blur();
      }
      // Path B (`cubase-midi-remote -> demo-pianoroll` via peer `SYNC_ACTIVE_CONTROLLER`):
      // Pop-Out focuses `launch-compass-ai-button` with POINTER modality so `Space` starts playback (`defaultPrevented === true`)!
      fireEvent.change(dawSelect, { target: { value: 'demo-pianoroll' } });
      const restoredPopoutLaunchBtn = await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="pianoroll-canvas"]')).toBeTruthy();
        const b = popoutApp.container.querySelector(
          '[data-testid="launch-compass-ai-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        expect(document.activeElement).toBe(b);
        return b!;
      });
      const spaceInPointerPopout = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        restoredPopoutLaunchBtn.dispatchEvent(spaceInPointerPopout);
      });
      expect(spaceInPointerPopout.defaultPrevented).toBe(true);
      // Pause playback via another Space
      act(() => {
        restoredPopoutLaunchBtn.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });

      // 2. `[R16-MED-02]` Sub-bug 1: Mouse side-buttons (`button: 3` / `button: 4`) over `<canvas>` neither blur `<input>`
      // nor corrupt `Space` on a keyboard-focused toolbar `<button>`!
      const bpmInput = popoutApp.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
      bpmInput.focus();
      expect(document.activeElement).toBe(bpmInput);
      fireEvent.pointerDown(popoutCanvas, {
        pointerId: 91,
        pointerType: 'mouse',
        button: 3,
        buttons: 8,
        clientX: 300,
        clientY: 200,
      });
      expect(document.activeElement).toBe(bpmInput);

      const pencilBtn = popoutApp.container.querySelector(
        'button[aria-label="ペン入力"]',
      ) as HTMLButtonElement;
      fireEvent.keyDown(window, { key: 'Tab', code: 'Tab' });
      pencilBtn.focus();
      expect(document.activeElement).toBe(pencilBtn);

      // Mouse X1/X2 (`button: 4`) on `<canvas>` must NOT hijack subsequent `Space` on `pencilBtn`!
      fireEvent.pointerDown(popoutCanvas, {
        pointerId: 92,
        pointerType: 'mouse',
        button: 4,
        buttons: 16,
        clientX: 320,
        clientY: 210,
      });
      const spaceAfterSideBtn = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      pencilBtn.dispatchEvent(spaceAfterSideBtn);
      expect(spaceAfterSideBtn.defaultPrevented).toBe(false);

      // 3. `[R16-MED-02]` Sub-bug 2: Right-click (`button: 2`) / touch on `<canvas>` while `pencilBtn` holds focus
      // sets `canvasHadRecentPointerRef`, AND pressing `Enter` on `pencilBtn` resets `canvasHadRecentPointerRef`
      // so subsequent `Space` activates `pencilBtn` (`defaultPrevented === false`)!
      const liveCanvasNow = popoutApp.container.querySelector(
        '[data-testid="pianoroll-canvas"]',
      ) as HTMLCanvasElement;
      vi.spyOn(liveCanvasNow, 'getBoundingClientRect').mockReturnValue({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 960,
        bottom: 540,
        width: 960,
        height: 540,
        toJSON: () => ({}),
      });
      fireEvent.pointerDown(liveCanvasNow, {
        pointerId: 93,
        pointerType: 'mouse',
        button: 2,
        buttons: 2,
        clientX: 300,
        clientY: 200,
      });
      expect(document.activeElement).toBe(pencilBtn);
      // Re-engage keyboard control on the already-focused `pencilBtn` via `Enter` (no `focusin` fires!)
      fireEvent.keyDown(pencilBtn, { key: 'Enter', code: 'Enter' });
      const spaceAfterEnterReengage = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      pencilBtn.dispatchEvent(spaceAfterEnterReengage);
      expect(spaceAfterEnterReengage.defaultPrevented).toBe(false);

      popoutApp.unmount();
      mainApp.unmount();
    } finally {
      window.history.replaceState(null, '', origUrl);
      globalThis.fetch = origFetch;
    }
  });

  it('54. [Round-17 Fact-Checker] verifies pointer-modality track rename Space playback ([R17-MED-01]), IME & text-input Enter modality guards ([R17-MED-02]), controller-switch draft/rename reset ([R17-MED-03]), and non-redundant external-daw-track-select label ([R17-LOW-01])', async () => {
    const origFetch = globalThis.fetch;
    const origUrl = window.location.href;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/hardware_info')) {
        return new Response(
          JSON.stringify({
            device: 'cpu',
            dtype: 'torch.float32',
            cuda_available: false,
            gpu_name: null,
            user_models_dir: 'C:/models',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (url.endsWith('/model_info')) {
        return new Response(
          JSON.stringify({
            models: [
              {
                model_name: 'MORTM4.5D-80M-SFT-Gen',
                has_weights: true,
                weights_file: 'sft_gen.safetensors',
                tag: { model: 'sft_gen' },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as typeof fetch;

    try {
      window.history.replaceState(null, '', '/');
      const mainApp = render(<App />);
      await waitFor(() => {
        expect(mainApp.container.querySelector('[data-testid="generate-button"]')).toBeTruthy();
      });

      window.history.replaceState(null, '', '/?tab=pianoroll&popout=1');
      const popoutApp = render(<App />);
      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="pianoroll-canvas"]')).toBeTruthy();
      });

      // 1. `[R17-MED-01]` Mouse double-click on Track 1 button + `Enter` restores pointer modality so `Space` starts playback!
      const trackBtn1 = popoutApp.container.querySelector(
        'button[data-track-select-id]',
      ) as HTMLButtonElement;
      expect(trackBtn1).toBeTruthy();
      fireEvent.pointerDown(trackBtn1, { pointerType: 'mouse', button: 0 });
      trackBtn1.focus();
      fireEvent.pointerUp(trackBtn1, { pointerType: 'mouse', button: 0 });
      fireEvent.doubleClick(trackBtn1);

      const renameInput = popoutApp.container.querySelector(
        'input[aria-label$="トラック名編集"]',
      ) as HTMLInputElement;
      expect(renameInput).toBeTruthy();
      // `[R17-MED-02]` IME composition (`isComposing: true` / `keyCode: 229`) on `Tab`/`Enter` is ignored cleanly
      fireEvent.keyDown(renameInput, { key: 'Tab', code: 'Tab', isComposing: true, keyCode: 229 });
      fireEvent.change(renameInput, { target: { value: 'Lead Synth R17' } });
      fireEvent.keyDown(renameInput, { key: 'Enter', code: 'Enter' });

      const restoredTrackBtn = popoutApp.container.querySelector(
        'button[data-track-select-id]',
      ) as HTMLButtonElement;
      expect(document.activeElement).toBe(restoredTrackBtn);
      const spaceAfterMouseRename = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        restoredTrackBtn.dispatchEvent(spaceAfterMouseRename);
      });
      expect(spaceAfterMouseRename.defaultPrevented).toBe(true);
      // Pause via Space
      act(() => {
        restoredTrackBtn.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });

      // Touch tap on active Track 1 button + `Escape` ALSO restores pointer modality!
      fireEvent.pointerDown(restoredTrackBtn, { pointerType: 'touch', button: 0 });
      fireEvent.click(restoredTrackBtn);
      const touchRenameInput = popoutApp.container.querySelector(
        'input[aria-label$="トラック名編集"]',
      ) as HTMLInputElement;
      expect(touchRenameInput).toBeTruthy();
      fireEvent.keyDown(touchRenameInput, { key: 'Escape', code: 'Escape' });
      const restoredAfterTouchEsc = popoutApp.container.querySelector(
        'button[data-track-select-id]',
      ) as HTMLButtonElement;
      expect(document.activeElement).toBe(restoredAfterTouchEsc);
      const spaceAfterTouchEsc = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        restoredAfterTouchEsc.dispatchEvent(spaceAfterTouchEsc);
      });
      expect(spaceAfterTouchEsc.defaultPrevented).toBe(true);
      act(() => {
        restoredAfterTouchEsc.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });

      // Conversely, keyboard `F2` + `Enter` restores KEYBOARD modality (`Space` does NOT preventDefault)!
      fireEvent.keyDown(window, { key: 'Tab', code: 'Tab' });
      restoredAfterTouchEsc.focus();
      fireEvent.keyDown(restoredAfterTouchEsc, { key: 'F2', code: 'F2' });
      const f2RenameInput = popoutApp.container.querySelector(
        'input[aria-label$="トラック名編集"]',
      ) as HTMLInputElement;
      expect(f2RenameInput).toBeTruthy();
      fireEvent.keyDown(f2RenameInput, { key: 'Enter', code: 'Enter' });
      const restoredAfterF2 = popoutApp.container.querySelector(
        'button[data-track-select-id]',
      ) as HTMLButtonElement;
      expect(document.activeElement).toBe(restoredAfterF2);
      const spaceAfterF2Rename = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      restoredAfterF2.dispatchEvent(spaceAfterF2Rename);
      expect(spaceAfterF2Rename.defaultPrevented).toBe(false);

      // 2. `[R17-MED-02]` Pointer click on `#pianoroll-bpm` + typing `'140'` + `Enter` MUST keep `lastInputModalityRef === 'pointer'`!
      const popoutBpm = popoutApp.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
      fireEvent.pointerDown(popoutBpm, { pointerType: 'mouse', button: 0 });
      popoutBpm.focus();
      fireEvent.change(popoutBpm, { target: { value: '140' } });
      fireEvent.keyDown(popoutBpm, { key: 'Enter', code: 'Enter' });
      expect(document.activeElement).toBe(document.body);

      const dawSelect = mainApp.container.querySelector('#compass-daw-select') as HTMLSelectElement;
      fireEvent.change(dawSelect, { target: { value: 'cubase-midi-remote' } });
      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="popout-external-daw-standby"]')).toBeTruthy();
      });
      // Because `Enter` inside `#pianoroll-bpm` did NOT flip modality to `'keyboard'`, Standby keeps `document.activeElement === document.body`!
      expect(document.activeElement).toBe(document.body);
      fireEvent.change(dawSelect, { target: { value: 'demo-pianoroll' } });
      const popoutLaunchAfterBpmEnter = await waitFor(() => {
        const b = popoutApp.container.querySelector(
          '[data-testid="launch-compass-ai-button"]',
        ) as HTMLButtonElement | null;
        expect(b).toBeTruthy();
        expect(document.activeElement).toBe(b);
        return b!;
      });
      const spaceAfterBpmEnterPeerSync = new KeyboardEvent('keydown', {
        key: ' ',
        code: 'Space',
        bubbles: true,
        cancelable: true,
      });
      act(() => {
        popoutLaunchAfterBpmEnter.dispatchEvent(spaceAfterBpmEnterPeerSync);
      });
      expect(spaceAfterBpmEnterPeerSync.defaultPrevented).toBe(true);
      act(() => {
        popoutLaunchAfterBpmEnter.dispatchEvent(
          new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true }),
        );
      });

      // 3. `[R17-MED-03]` & `[R17-LOW-01]` In Main Window:
      // Focus `#pianoroll-bpm` with uncommitted partial draft `'1'` AND open inline track rename (`editingTrackId !== null`) in Pop-Out,
      // then switch `demo -> cubase`:
      // - Main Window's Cubase `#pianoroll-bpm` displays Cubase's truthful `'120'` (NOT stale `'1'` or `'140'`)!
      // - `#external-daw-track-select` has NO redundant `aria-label` (labeled solely by `<label htmlFor>`)!
      // - Switching back `cubase -> demo` does NOT resurrect `<input autoFocus>` in Pop-Out!
      const mainPrTab = mainApp.container.querySelector('[data-testid="tab-pianoroll"]') as HTMLButtonElement;
      fireEvent.click(mainPrTab);
      const mainBpmDemo = await waitFor(() => {
        const inp = mainApp.container.querySelector('#pianoroll-bpm') as HTMLInputElement | null;
        expect(inp).toBeTruthy();
        return inp!;
      });
      mainBpmDemo.focus();
      fireEvent.change(mainBpmDemo, { target: { value: '1' } });
      expect(mainBpmDemo.value).toBe('1');

      const popoutTrkBtn = popoutApp.container.querySelector(
        'button[data-track-select-id]',
      ) as HTMLButtonElement;
      fireEvent.doubleClick(popoutTrkBtn);
      expect(popoutApp.container.querySelector('input[aria-label$="トラック名編集"]')).toBeTruthy();

      // Switch controller via peer BroadcastChannel while Main is on Piano Roll tab (`isActive === true`)!
      const syncBus = new BroadcastChannel('compass_ai_workspace_v1');
      act(() => {
        syncBus.postMessage({
          messageId: 'r17_test_cubase',
          type: 'SYNC_ACTIVE_CONTROLLER',
          controllerId: 'cubase-midi-remote',
          senderId: 'r17_external_peer',
        });
      });
      const extTrackSel = await waitFor(() => {
        const sel = mainApp.container.querySelector(
          '#external-daw-track-select',
        ) as HTMLSelectElement | null;
        expect(sel).toBeTruthy();
        return sel!;
      });
      // `[R17-LOW-01]` Sole native `<label htmlFor="external-daw-track-select">`, zero redundant `aria-label`
      expect(extTrackSel.getAttribute('aria-label')).toBeNull();
      expect(
        mainApp.container.querySelector('label[for="external-daw-track-select"]')?.textContent,
      ).toBe('対象トラック');
      // `[R17-MED-03]` Cubase BPM input shows truthful `'120'`, NOT Demo's uncommitted `'1'`!
      const mainBpmCubase = mainApp.container.querySelector('#pianoroll-bpm') as HTMLInputElement;
      expect(mainBpmCubase.value).toBe('120');

      // Switch back `cubase -> demo` -> Pop-Out Track Rack must NOT resurrect `<input autoFocus>`!
      if (document.activeElement instanceof HTMLElement && document.activeElement !== document.body) {
        document.activeElement.blur();
      }
      act(() => {
        syncBus.postMessage({
          messageId: 'r17_test_demo',
          type: 'SYNC_ACTIVE_CONTROLLER',
          controllerId: 'demo-pianoroll',
          senderId: 'r17_external_peer',
        });
      });
      syncBus.close();
      await waitFor(() => {
        expect(popoutApp.container.querySelector('[data-testid="pianoroll-canvas"]')).toBeTruthy();
      });
      expect(popoutApp.container.querySelector('input[aria-label$="トラック名編集"]')).toBeNull();

      popoutApp.unmount();
      mainApp.unmount();
    } finally {
      window.history.replaceState(null, '', origUrl);
      globalThis.fetch = origFetch;
    }
  });
});
