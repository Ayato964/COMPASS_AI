/**
 * Web Audio Polyphonic Synthesizer & Real-Time Acoustic Analyser (`oop-extensible-craft`).
 * Routes all synthesized notes through a master `DynamicsCompressorNode` -> `AnalyserNode`
 * so the central COMPASS AI circle vibrates organically in response to live audio waveforms.
 */

import { TrackInstrument } from '../domain/DawTrack';

export interface AcousticTelemetry {
  readonly rms: number;
  readonly peak: number;
  readonly lowBandEnergy: number;
  readonly highBandEnergy: number;
}

export interface IAudioEngine {
  triggerNote(
    pitch: number,
    velocity: number,
    durationSec: number,
    instrument: TrackInstrument,
    whenAudioTime?: number,
  ): void;
  auditionPitch(pitch: number, instrument: TrackInstrument): void;
  stopAllVoices(): void;
  getCurrentTime(): number;
  ensureResumed(): Promise<void>;
  readAcousticTelemetry(outWaveform: Float32Array, outSpectrum: Uint8Array): AcousticTelemetry;
  dispose(): Promise<void>;
}

function midiPitchToFreq(pitch: number): number {
  return 440 * Math.pow(2, (pitch - 69) / 12);
}

export class WebAudioEngine implements IAudioEngine {
  public static readonly MAX_ACTIVE_VOICES = 64;

  private _ctx: AudioContext | null = null;
  private _masterGain: GainNode | null = null;
  private _analyser: AnalyserNode | null = null;
  private _activeStops = new Set<() => void>();
  private _stopEpoch = 0;
  private _disposed = false;

  public get activeVoiceCount(): number {
    return this._activeStops.size;
  }

  private _ensureGraph(): AudioContext | null {
    if (this._disposed) return null;
    if (this._ctx) return this._ctx;
    const AudioCtxCtor =
      typeof window !== 'undefined'
        ? window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
        : undefined;
    if (!AudioCtxCtor) return null;

    const ctx = new AudioCtxCtor();
    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -16;
    compressor.knee.value = 18;
    compressor.ratio.value = 4;
    compressor.attack.value = 0.004;
    compressor.release.value = 0.18;

    const masterGain = ctx.createGain();
    masterGain.gain.value = 0.72;

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.78;

    masterGain.connect(compressor);
    compressor.connect(analyser);
    analyser.connect(ctx.destination);

    this._ctx = ctx;
    this._masterGain = masterGain;
    this._analyser = analyser;
    return ctx;
  }

  public async ensureResumed(): Promise<void> {
    if (this._disposed) return;
    const ctx = this._ensureGraph();
    if (ctx && ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch {
        // Ignore autoplay restriction until user gesture
      }
    }
  }

  public getCurrentTime(): number {
    const ctx = this._ensureGraph();
    return ctx ? ctx.currentTime : performance.now() / 1000;
  }

  public auditionPitch(pitch: number, instrument: TrackInstrument): void {
    if (this._disposed || !Number.isFinite(pitch)) return;
    const epochAtRequest = this._stopEpoch;
    void this.ensureResumed().then(() => {
      if (this._disposed || this._stopEpoch !== epochAtRequest) return;
      this.triggerNote(pitch, 96, 0.24, instrument);
    });
  }

  public triggerNote(
    pitch: number,
    velocity: number,
    durationSec: number,
    instrument: TrackInstrument,
    whenAudioTime?: number,
  ): void {
    if (
      this._disposed ||
      !Number.isFinite(pitch) ||
      !Number.isFinite(durationSec) ||
      (whenAudioTime !== undefined && !Number.isFinite(whenAudioTime))
    ) {
      return;
    }
    const ctx = this._ensureGraph();
    if (!ctx || !this._masterGain) return;

    const start = Math.max(ctx.currentTime, whenAudioTime ?? ctx.currentTime);
    const dur = Math.max(0.05, Math.min(12.0, durationSec));
    const amp = Math.max(0.03, Math.min(1.0, (Number.isFinite(velocity) ? velocity : 96) / 127)) * 0.28;
    const freq = midiPitchToFreq(Math.max(0, Math.min(127, pitch)));

    const voiceGain = ctx.createGain();
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';

    const oscPrimary = ctx.createOscillator();
    const oscHarmonic = ctx.createOscillator();
    const harmonicGain = ctx.createGain();

    if (instrument === 'SAX' || instrument === 'VIOLIN') {
      oscPrimary.type = 'sawtooth';
      oscHarmonic.type = 'triangle';
      oscHarmonic.frequency.setValueAtTime(freq, start);
      harmonicGain.gain.value = 0.35;
      filter.frequency.setValueAtTime(Math.min(4800, freq * 5.2), start);
      filter.Q.value = 1.6;

      // Expressive reed/bow attack & sustain
      const attack = Math.min(0.045, dur * 0.25);
      const release = 0.14;
      voiceGain.gain.setValueAtTime(0.0001, start);
      voiceGain.gain.linearRampToValueAtTime(amp, start + attack);
      voiceGain.gain.setValueAtTime(amp * 0.86, start + Math.max(attack, dur - 0.02));
      voiceGain.gain.exponentialRampToValueAtTime(0.0001, start + dur + release);

      oscPrimary.frequency.setValueAtTime(freq, start);
      oscPrimary.connect(filter);
      oscHarmonic.connect(harmonicGain);
      harmonicGain.connect(filter);
      filter.connect(voiceGain);
      voiceGain.connect(this._masterGain);

      const stopTime = start + dur + release + 0.02;
      this._scheduleVoiceLifecycle(
        oscPrimary,
        oscHarmonic,
        harmonicGain,
        filter,
        voiceGain,
        start,
        stopTime,
      );
      return;
    }

    if (instrument === 'BASS') {
      oscPrimary.type = 'triangle';
      oscHarmonic.type = 'sine';
      oscPrimary.frequency.setValueAtTime(freq, start);
      oscHarmonic.frequency.setValueAtTime(freq * 2, start);
      harmonicGain.gain.value = 0.22;
      filter.frequency.setValueAtTime(Math.min(1800, freq * 4), start);

      const release = 0.12;
      voiceGain.gain.setValueAtTime(0.0001, start);
      voiceGain.gain.linearRampToValueAtTime(amp * 1.15, start + 0.008);
      voiceGain.gain.exponentialRampToValueAtTime(amp * 0.45, start + dur);
      voiceGain.gain.exponentialRampToValueAtTime(0.0001, start + dur + release);

      oscPrimary.connect(filter);
      oscHarmonic.connect(harmonicGain);
      harmonicGain.connect(filter);
      filter.connect(voiceGain);
      voiceGain.connect(this._masterGain);

      this._scheduleVoiceLifecycle(
        oscPrimary,
        oscHarmonic,
        harmonicGain,
        filter,
        voiceGain,
        start,
        start + dur + release + 0.02,
      );
      return;
    }

    // Default Studio Grand Piano / Poly Synth voice
    oscPrimary.type = instrument === 'SYNTH' ? 'sawtooth' : 'triangle';
    oscHarmonic.type = 'sine';
    oscPrimary.frequency.setValueAtTime(freq, start);
    oscHarmonic.frequency.setValueAtTime(freq * 2, start);
    harmonicGain.gain.setValueAtTime(0.28, start);
    harmonicGain.gain.exponentialRampToValueAtTime(0.04, start + Math.min(0.6, dur));

    filter.frequency.setValueAtTime(Math.min(7500, freq * 8.5), start);
    filter.frequency.exponentialRampToValueAtTime(Math.max(600, freq * 2.4), start + dur + 0.15);

    const release = 0.22;
    voiceGain.gain.setValueAtTime(0.0001, start);
    voiceGain.gain.linearRampToValueAtTime(amp, start + 0.006);
    voiceGain.gain.exponentialRampToValueAtTime(Math.max(0.002, amp * 0.32), start + dur);
    voiceGain.gain.exponentialRampToValueAtTime(0.0001, start + dur + release);

    oscPrimary.connect(filter);
    oscHarmonic.connect(harmonicGain);
    harmonicGain.connect(filter);
    filter.connect(voiceGain);
    voiceGain.connect(this._masterGain);

    this._scheduleVoiceLifecycle(
      oscPrimary,
      oscHarmonic,
      harmonicGain,
      filter,
      voiceGain,
      start,
      start + dur + release + 0.02,
    );
  }

  public stopAllVoices(): void {
    this._stopEpoch += 1;
    for (const stopFn of this._activeStops) {
      try {
        stopFn();
      } catch {
        // Ignore already-stopped oscillator
      }
    }
    this._activeStops.clear();
  }

  public async dispose(): Promise<void> {
    this._disposed = true;
    this.stopAllVoices();
    const ctx = this._ctx;
    this._ctx = null;
    this._masterGain = null;
    this._analyser = null;
    if (ctx && ctx.state !== 'closed') {
      try {
        await ctx.close();
      } catch {
        // Ignore already-closed AudioContext
      }
    }
  }

  public readAcousticTelemetry(
    outWaveform: Float32Array,
    outSpectrum: Uint8Array,
  ): AcousticTelemetry {
    if (!this._analyser || !this._ctx || this._ctx.state !== 'running') {
      outWaveform.fill(0);
      outSpectrum.fill(0);
      return { rms: 0, peak: 0, lowBandEnergy: 0, highBandEnergy: 0 };
    }

    this._analyser.getFloatTimeDomainData(outWaveform as Float32Array<ArrayBuffer>);
    this._analyser.getByteFrequencyData(outSpectrum as Uint8Array<ArrayBuffer>);

    let sumSq = 0;
    let peak = 0;
    for (let i = 0; i < outWaveform.length; i += 1) {
      const s = outWaveform[i];
      sumSq += s * s;
      const abs = Math.abs(s);
      if (abs > peak) peak = abs;
    }
    const rms = Math.sqrt(sumSq / Math.max(1, outWaveform.length));

    const splitIdx = Math.max(1, Math.floor(outSpectrum.length * 0.18));
    let lowSum = 0;
    for (let i = 0; i < splitIdx; i += 1) {
      lowSum += outSpectrum[i];
    }
    let highSum = 0;
    for (let i = splitIdx; i < outSpectrum.length; i += 1) {
      highSum += outSpectrum[i];
    }

    return {
      rms,
      peak,
      lowBandEnergy: lowSum / (splitIdx * 255),
      highBandEnergy: highSum / (Math.max(1, outSpectrum.length - splitIdx) * 255),
    };
  }

  private _scheduleVoiceLifecycle(
    osc1: OscillatorNode,
    osc2: OscillatorNode,
    harmonicGain: GainNode,
    filter: BiquadFilterNode,
    gain: GainNode,
    startTime: number,
    stopTime: number,
  ): void {
    const disconnectAll = () => {
      try {
        osc1.disconnect();
        osc2.disconnect();
        harmonicGain.disconnect();
        filter.disconnect();
        gain.disconnect();
      } catch {
        // Safe cleanup
      }
    };

    const forceStop = () => {
      try {
        gain.gain.cancelScheduledValues(0);
        gain.gain.value = 0;
        osc1.stop();
        osc2.stop();
      } catch {
        // Ignore already-stopped oscillator
      }
      disconnectAll();
    };

    // Enforce 64-voice polyphony cap via O(1) FIFO oldest-voice stealing ([R2-LOW-03])
    while (this._activeStops.size >= WebAudioEngine.MAX_ACTIVE_VOICES) {
      const oldestStop = this._activeStops.values().next().value;
      if (!oldestStop) break;
      this._activeStops.delete(oldestStop);
      try {
        oldestStop();
      } catch {
        // Ignore already-stopped voice
      }
    }

    this._activeStops.add(forceStop);
    osc1.onended = () => {
      this._activeStops.delete(forceStop);
      disconnectAll();
    };

    osc1.start(startTime);
    osc2.start(startTime);
    osc1.stop(stopTime);
    osc2.stop(stopTime);
  }
}
