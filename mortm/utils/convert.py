import copy
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
from pretty_midi.pretty_midi import Instrument, Note, PrettyMIDI

from mortm.train.custom_token import (
    ChordToken,
    ShiftTimeContainer,
    Token,
    validate_supported_time_signature,
)
from mortm.train.tokenizer import Tokenizer
from mortm.utils.key import get_key_dict


@dataclass(frozen=True)
class InstrumentConstraint:
    programs: Tuple[int, ...]
    is_polyphonic: bool


INSTRUMENT_RULES: Dict[str, InstrumentConstraint] = {
    "PIANO": InstrumentConstraint(programs=(0, 1, 2, 3, 4, 5, 6), is_polyphonic=True),
    "SAX": InstrumentConstraint(programs=(64, 65, 66, 67), is_polyphonic=False),
}

T = TypeVar("T")


def split_sequence_measure(sequence: np.ndarray, measure_length: int, measure_token_id: int) -> List[np.ndarray]:
    """
    Splits a token ID sequence by measure boundaries (<SME>) into chunks of `measure_length` measures.
    """
    sme_indices = np.where(sequence == measure_token_id)[0]
    if len(sme_indices) == 0:
        return [sequence]

    chunks: List[np.ndarray] = []
    for i in range(0, len(sme_indices), measure_length):
        start_index = sme_indices[i]
        next_chunk_sme_index_pos = i + measure_length
        if next_chunk_sme_index_pos < len(sme_indices):
            end_index = sme_indices[next_chunk_sme_index_pos]
            chunk = sequence[start_index:end_index]
        else:
            chunk = sequence[start_index:]
        chunks.append(chunk)

    return chunks


def convert_str_int_program(program_list: List[str]) -> List[Tuple[int, ...]]:
    pl: List[Tuple[int, ...]] = []
    for p in program_list:
        if p == "SAX":
            pl.append((64, 65, 66, 67, 68))
        elif p == "PIANO":
            pl.append((0, 1, 2, 3, 4, 5, 6))
    return pl


class _AbstractConverter(ABC):
    def __init__(self, instance: Generic[T], directory: Optional[str], file_name: Optional[str | List[str]]):
        self.instance = instance
        self.directory = directory
        self.file_name = file_name
        self.is_error = False
        self.error_reason: str = "不明なエラー"

    def __call__(self, *args, **kwargs):
        self.convert(*args, **kwargs)

    @abstractmethod
    def save(self, save_directory: str) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def convert(self, *args, **kwargs):
        pass


class _AbstractMidiConverter(_AbstractConverter):
    def __init__(
        self,
        instance: Generic[T],
        tokenizer: Tokenizer,
        directory: Optional[str],
        file_name: Optional[str],
        program_list: List[str],
        key: Optional[str] = None,
        inst_list_with_name: Optional[List[Tuple[Instrument, str]]] = None,
        midi_data: Optional[PrettyMIDI] = None,
        default_time_signature: Tuple[int, int] = (4, 4),
        force_time_signature: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(instance, directory, file_name)
        self.token_converter: List[Token] = tokenizer.music_token_list
        self.tokenizer = tokenizer
        self.key_dict = None
        self.key = key
        if not isinstance(default_time_signature, (tuple, list)) or len(default_time_signature) != 2:
            raise ValueError(f"指定された拍子記号 ({default_time_signature}) が不正です。")
        def_num, def_den = validate_supported_time_signature(
            default_time_signature[0], default_time_signature[1]
        )
        self.time_signature: Tuple[int, int] = (def_num, def_den)
        self.quarters_per_bar: float = float(def_num) * (4.0 / float(def_den))
        self.file_time_signature: Optional[Tuple[int, int]] = None
        self.has_explicit_time_signature: bool = False

        validated_force_ts: Optional[Tuple[int, int]] = None
        if force_time_signature is not None:
            if not isinstance(force_time_signature, (tuple, list)) or len(force_time_signature) != 2:
                raise ValueError(f"指定された拍子記号 ({force_time_signature}) が不正です。")
            validated_force_ts = validate_supported_time_signature(
                force_time_signature[0], force_time_signature[1]
            )

        if midi_data is not None:
            self.midi_data: PrettyMIDI = midi_data
        else:
            try:
                midi_path = os.path.join(directory, file_name) if directory and file_name else (file_name or "")
                self.midi_data = PrettyMIDI(midi_path)
            except Exception:
                self.is_error = True
                self.error_reason = "MIDIのロードができませんでした。"

        if not self.is_error:
            if validated_force_ts is None:
                for t_s in self.midi_data.time_signature_changes:
                    try:
                        validate_supported_time_signature(t_s.numerator, t_s.denominator)
                    except ValueError as exc:
                        self.is_error = True
                        self.error_reason = str(exc)
                        break
                if not self.is_error and self.midi_data.time_signature_changes:
                    ts0 = self.midi_data.time_signature_changes[0]
                    self.file_time_signature = (int(ts0.numerator), int(ts0.denominator))
                    self.has_explicit_time_signature = True
            elif self.midi_data.time_signature_changes:
                ts0 = self.midi_data.time_signature_changes[0]
                self.file_time_signature = (int(ts0.numerator), int(ts0.denominator))
                self.has_explicit_time_signature = True

            if not self.is_error:
                if validated_force_ts is not None:
                    f_num, f_den = validated_force_ts
                    self.time_signature = (f_num, f_den)
                    self.quarters_per_bar = float(f_num) * (4.0 / float(f_den))
                elif self.file_time_signature is not None:
                    self.time_signature = self.file_time_signature
                    self.quarters_per_bar = float(self.file_time_signature[0]) * (
                        4.0 / float(self.file_time_signature[1])
                    )

        if not self.is_error:
            self.tempo_change_time, self.tempo = self.midi_data.get_tempo_changes()
            if len(self.tempo) == 0:
                self.tempo_change_time = np.array([0.0])
                self.tempo = np.array([120.0])
            else:
                self.tempo = np.clip(np.nan_to_num(self.tempo, nan=120.0, posinf=300.0, neginf=20.0), 20.0, 300.0)

            if inst_list_with_name is not None:
                self.inst_list_with_name = inst_list_with_name
                self.inst_list = [inst for inst, _ in self.inst_list_with_name]
                self.program_list = [name for _, name in self.inst_list_with_name]
            else:
                self.inst_list_with_name, found_program_names = self.is_in_correct_inst(program_list)
                self.inst_list = [inst for inst, _ in self.inst_list_with_name]
                self.program_list = found_program_names

            if len(self.inst_list) == 0 and not self.is_error:
                self.is_error = True
                self.error_reason = "欲しい楽器がMIDIにありません。"
            elif not self.is_error and key is None:
                if len(self.inst_list) > 0:
                    midi_path = os.path.join(directory, file_name) if directory and file_name else None
                    self.key = get_key_dict(midi_path, window_measures=999, midi_data=self.midi_data)
                    self.key_dict = self.key.get("segments", [])
            elif key is not None:
                if isinstance(key, str):
                    if "major" in key:
                        self.key = f"{key.split(' major')[0]}M"
                    elif "minor" in key:
                        self.key = f"{key.split(' minor')[0]}m"
                    else:
                        self.key = key
                else:
                    self.key = key

    def is_in_correct_inst(self, program_list: List[str]) -> Tuple[List[Tuple[Instrument, str]], List[str]]:
        int_program_groups = convert_str_int_program(program_list)

        inst_by_program: Dict[int, List[Instrument]] = {}
        for inst in self.midi_data.instruments:
            if not inst.is_drum:
                inst_by_program.setdefault(inst.program, []).append(inst)

        found_instruments_with_name: List[Tuple[Instrument, str]] = []
        found_program_names: List[str] = []
        added_instrument_ids: set = set()

        for name, group in zip(program_list, int_program_groups):
            candidate_instrument = None
            for program_num in group:
                if program_num in inst_by_program:
                    for inst in inst_by_program[program_num]:
                        if id(inst) not in added_instrument_ids:
                            candidate_instrument = inst
                            break
                if candidate_instrument:
                    break

            if candidate_instrument:
                candidate_instrument.notes.sort(key=lambda note: note.start)
                found_instruments_with_name.append((candidate_instrument, name))
                added_instrument_ids.add(id(candidate_instrument))
                if name not in found_program_names:
                    found_program_names.append(name)

        return found_instruments_with_name, found_program_names

    def get_midi_change_scale(self, scale_up_key: int) -> PrettyMIDI:
        midi = copy.deepcopy(self.midi_data)
        for ins in midi.instruments:
            if not ins.is_drum:
                for note in ins.notes:
                    pitch = note.pitch + scale_up_key
                    while pitch > 127:
                        pitch -= 12
                    while pitch < 0:
                        pitch += 12
                    note.pitch = int(pitch)
        return midi

    def get_tempo(self, start: float) -> float:
        idx = int(np.searchsorted(self.tempo_change_time, start, side="right")) - 1
        if idx < 0:
            idx = 0
        return float(self.tempo[idx])


def apply_sustain_pedal(inst: Instrument) -> List[Note]:
    """
    Applies CC64 sustain pedal extension and retrigger truncation to an instrument's notes.
    Uses binary search (O(N log P)) for sustain period lookup.
    """
    cc64 = sorted(
        [cc for cc in inst.control_changes if cc.number == 64],
        key=lambda c: c.time,
    )
    if not cc64:
        return list(inst.notes)

    sustain_starts: List[float] = []
    sustain_ends: List[float] = []
    sus_on: Optional[float] = None
    for cc in cc64:
        if cc.value >= 64 and sus_on is None:
            sus_on = cc.time
        elif cc.value < 64 and sus_on is not None:
            sustain_starts.append(sus_on)
            sustain_ends.append(cc.time)
            sus_on = None

    if not sustain_starts:
        return list(inst.notes)

    starts_arr = np.asarray(sustain_starts, dtype=np.float64)
    ends_arr = np.asarray(sustain_ends, dtype=np.float64)

    def _get_sustain_off(t: float) -> Optional[float]:
        idx = int(np.searchsorted(starts_arr, t, side="right")) - 1
        if 0 <= idx < len(ends_arr) and starts_arr[idx] <= t < ends_arr[idx]:
            return float(ends_arr[idx])
        return None

    notes = sorted(inst.notes, key=lambda n: n.start)
    result: List[Note] = []
    active_sustained: Dict[int, int] = {}

    for note in notes:
        p = note.pitch
        if p in active_sustained:
            idx = active_sustained.pop(p)
            old = result[idx]
            if old.end > note.start:
                result[idx] = Note(
                    velocity=old.velocity,
                    pitch=old.pitch,
                    start=old.start,
                    end=note.start,
                )

        sus_off = _get_sustain_off(note.end)
        if sus_off is not None:
            result.append(
                Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=sus_off,
                )
            )
            active_sustained[p] = len(result) - 1
        else:
            result.append(
                Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
            )

    result.sort(key=lambda n: n.start)
    return result


class MIDI2Seq(_AbstractMidiConverter):
    """Converts PrettyMIDI instrument tracks into MORTM token sequences."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        directory: Optional[str],
        file_name: Optional[str],
        program_list: List[str],
        key: Optional[str] = None,
        inst_list_with_name: Optional[List[Tuple[Instrument, str]]] = None,
        midi_data: Optional[PrettyMIDI] = None,
        default_time_signature: Tuple[int, int] = (4, 4),
        force_time_signature: Optional[Tuple[int, int]] = None,
        max_clip_tokens: int = 8192,
    ):
        super().__init__(
            MIDI2Seq,
            tokenizer,
            directory,
            file_name,
            program_list,
            key=key,
            inst_list_with_name=inst_list_with_name,
            midi_data=midi_data,
            default_time_signature=default_time_signature,
            force_time_signature=force_time_signature,
        )
        self.max_clip_tokens: int = max(4, int(max_clip_tokens))
        self.aya_node: Dict[str, np.ndarray] = {}

    def convert(self, *args, **kwargs):
        if not self.is_error:
            container = [
                ShiftTimeContainer(0, self.tempo[0], True, quarters_per_bar=self.quarters_per_bar)
                for _ in self.inst_list
            ]
            note_counts = [0 for _ in self.inst_list]
            s_e = self.tokenizer.get_length_tuple("s")

            for i, (inst, program) in enumerate(self.inst_list_with_name):
                original_notes = inst.notes
                bounded_notes = (
                    sorted(original_notes, key=lambda n: n.start)[:4096]
                    if len(original_notes) > 4096
                    else original_notes
                )
                inst.notes = bounded_notes
                inst.notes = apply_sustain_pedal(inst)
                if len(inst.notes) > 4096:
                    inst.notes = inst.notes[:4096]

                inst_clip, note_count, _ = self.convert_inst(program, inst, container[i], note_counts[i])

                inst.notes = original_notes
                note_counts[i] = note_count

                if s_e is not None and np.any((inst_clip >= s_e[0]) & (inst_clip <= s_e[1])):
                    self.aya_node[program] = inst_clip

    def convert_inst(
        self,
        program: str,
        inst: Instrument,
        container: ShiftTimeContainer,
        note_count: int,
        max_clip_tokens: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, bool]:
        clip: List[int] = []
        clip_count = 0
        loop_steps = 0
        eff_max_clip_tokens = max(4, int(max_clip_tokens if max_clip_tokens is not None else self.max_clip_tokens))
        max_loop_steps = 16384
        back_note: Optional[Note] = None

        non_chord_converters = [c for c in self.token_converter if not isinstance(c, ChordToken)]

        while clip_count < 256 and len(clip) < eff_max_clip_tokens and loop_steps < max_loop_steps:
            loop_steps += 1
            if note_count >= len(inst.notes):
                return np.array(clip, dtype=np.int64), note_count, True
            note: Note = inst.notes[note_count]
            tempo = self.get_tempo(note.start)
            container.tempo = tempo
            is_continue_note = False
            step_tokens: List[int] = []
            step_sme_inc = 0

            for conv in non_chord_converters:
                token = conv(inst=inst, back_notes=back_note, note=note, tempo=tempo, container=container)
                if token is not None:
                    if conv.token_type == "<SME>":
                        step_sme_inc += 1
                    if not container.is_velocity and conv.token_type == "v":
                        continue

                    token_id = self.tokenizer.get(token)
                    if token_id is not None:
                        step_tokens.append(token_id)

                    if conv.token_type == "<BLANK>":
                        is_continue_note = True
                        break

                if container.is_error:
                    self.is_error = True
                    self.error_reason = "MIDIの変換中にエラーが発生しました。"
                    break

            if self.is_error:
                break

            if clip_count + step_sme_inc >= 256 or (
                len(clip) > 0 and len(clip) + len(step_tokens) > eff_max_clip_tokens
            ):
                return np.array(clip, dtype=np.int64), note_count, False

            clip_count += step_sme_inc
            clip.extend(step_tokens)

            if not is_continue_note:
                note_count += 1
                back_note = note

        return np.array(clip, dtype=np.int64), note_count, False

    def save(self, save_directory: str) -> Tuple[bool, str]:
        if not self.is_error:
            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node.values())}
            if len(array_dict) > 0:
                np.savez(f"{save_directory}/{self.file_name}", **array_dict)
                return True, "処理が正常に終了しました。"
            return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        return False, self.error_reason


class MIDIConverter(_AbstractMidiConverter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        directory: Optional[str],
        file_name: Optional[str],
        program_list: List[str],
        use_midi2seq: bool = True,
        use_midi2seq_with_chord: bool = False,
        all_chords: Optional[List[str]] = None,
        all_chord_timestamps: Optional[List[float]] = None,
        key: Optional[str] = None,
        midi_data: Optional[PrettyMIDI] = None,
        default_time_signature: Tuple[int, int] = (4, 4),
        force_time_signature: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            MIDIConverter,
            tokenizer,
            directory,
            file_name,
            program_list,
            key=key,
            midi_data=midi_data,
            default_time_signature=default_time_signature,
            force_time_signature=force_time_signature,
        )
        self.midi2seq: Optional[MIDI2Seq] = None
        self.midi2seq_with_chord = None
        if not self.is_error and use_midi2seq:
            self.midi2seq = MIDI2Seq(
                tokenizer,
                directory,
                file_name,
                program_list,
                key=self.key,
                inst_list_with_name=self.inst_list_with_name,
                midi_data=self.midi_data,
                default_time_signature=self.time_signature,
                force_time_signature=self.time_signature,
            )

    def make_system_prompt(self, time: float, found_list: List[Tuple[str, Optional[int]]], call_function=None) -> List[int]:
        prompt = [
            self.tokenizer.get("<EOS>"),
            self.tokenizer.get("<SYSTEM>"),
        ]
        for p_name, density_id in found_list:
            prompt.append(self.tokenizer.get(f"<INST_{p_name}>"))
            if density_id is not None:
                prompt.append(density_id)

        if call_function is not None:
            call_function(prompt)

        if self.key_dict:
            key_str = self.get_key(time)
        elif isinstance(self.key, dict):
            key_str = self.get_key(time)
        else:
            key_str = self.key or "Unknown"

        key_tok = self.tokenizer.get(f"k_{key_str}")
        if key_tok is None:
            key_tok = self.tokenizer.get("k_Unknown")
        prompt.append(key_tok)
        prompt.append(self.tokenizer.get("<TAG_END>"))
        return prompt

    def convert(self, *args, **kwargs):
        if self.is_error:
            return
        if self.midi2seq is not None:
            self.midi2seq.convert()

    def save(self, save_directory: str) -> Tuple[bool, str]:
        if self.is_error or self.midi2seq is None:
            return False, self.error_reason or "MIDI2Seq data is not available."
        return self.midi2seq.save(save_directory)

    def get_key(self, time: float) -> str:
        if not self.key_dict:
            return "Unknown"
        target_k = None
        for k in self.key_dict:
            if k['start_time'] <= time < k['end_time']:
                target_k = k
                break
        if target_k is None:
            if self.key_dict and self.key_dict[-1]['end_time'] >= time:
                target_k = self.key_dict[-1]
            else:
                return "Unknown"

        if target_k.get('key_name') == "Unknown":
            return "Unknown"
        tonic = target_k.get('tonic')
        if not tonic or tonic == "Unknown":
            return "Unknown"

        if '-' in tonic:
            tonic = tonic.replace('-', 'b')
        mode = 'M' if target_k.get('mode') == 'major' else 'm'
        candidate_key = f"{tonic}{mode}"

        if f"k_{candidate_key}" in self.tokenizer.tokens:
            return candidate_key

        enharmonic_map = {
            'C#': 'Db', 'Db': 'C#',
            'D#': 'Eb', 'Eb': 'D#',
            'F#': 'Gb', 'Gb': 'F#',
            'G#': 'Ab', 'Ab': 'G#',
            'A#': 'Bb', 'Bb': 'A#',
        }
        if tonic in enharmonic_map:
            alt_candidate = f"{enharmonic_map[tonic]}{mode}"
            if f"k_{alt_candidate}" in self.tokenizer.tokens:
                return alt_candidate

        return "Unknown"
