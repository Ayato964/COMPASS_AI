from typing import Tuple, Union
import numpy as np
from pretty_midi import Note, PrettyMIDI, TimeSignature
import pretty_midi.instrument as inst
from torch import Tensor

from mortm.train.custom_token import (
    ShiftTimeContainer,
    ct_beat_to_time,
    validate_supported_time_signature,
)
from mortm.train.tokenizer import Tokenizer


def ct_token_to_midi(
    tokenizer: Tokenizer,
    seq: Union[Tensor, np.ndarray, list],
    save_directory: str,
    tempo: float = 120.0,
    time_signature: Tuple[int, int] = (4, 4),
) -> PrettyMIDI:
    """
    Converts a MORTM token sequence back into a standard MIDI file (.mid).
    Optimized to O(N) direct token dispatch (replacing the 30-converter inner loop)
    and hardened against leading tag variations, meter changes (`quarters_per_bar`), and empty tracks.
    """
    if isinstance(seq, Tensor):
        token_ids = seq.detach().cpu().tolist()
    elif isinstance(seq, np.ndarray):
        token_ids = seq.tolist()
    else:
        token_ids = list(seq)

    # Preserve backwards compatibility with callers that include a leading <MGEN> tag
    # without dropping <INST_PIANO>/<INST_SAX> when the caller already stripped <MGEN>.
    if token_ids:
        first_tok = tokenizer.rev_get(int(token_ids[0]))
        if first_tok in ("<MGEN>", "<CGEN>", "<EOS>", "<TAG_END>"):
            token_ids = token_ids[1:]

    if time_signature is None:
        ts_num, ts_den = 4, 4
    else:
        if not isinstance(time_signature, (tuple, list)) or len(time_signature) != 2:
            raise ValueError(f"指定された拍子記号 ({time_signature}) が不正です。")
        ts_num, ts_den = validate_supported_time_signature(time_signature[0], time_signature[1])
    quarters_per_bar = float(ts_num) * (4.0 / float(ts_den))

    tempo_val = float(tempo) if tempo and tempo > 0 else 120.0
    midi = PrettyMIDI(resolution=480, initial_tempo=tempo_val)
    midi.time_signature_changes.append(TimeSignature(ts_num, ts_den, 0.0))
    container = ShiftTimeContainer(0.0, tempo_val, True, quarters_per_bar=quarters_per_bar)
    current_inst = None
    note = Note(pitch=60, velocity=100, start=0.0, end=0.0)
    te_id = tokenizer.get("<TE>")

    for raw_id in token_ids:
        tid = int(raw_id)
        if tid == te_id:
            break

        token = tokenizer.rev_get(tid)
        if not token or token == "<PAD>":
            continue

        if token in ("<INST_PIANO>", "<INST_SAX>"):
            if current_inst is not None and len(current_inst.notes) > 0:
                midi.instruments.append(current_inst)
            container.inst = token
            prog = container.get_inst() or 1
            current_inst = inst.Instrument(program=prog, is_drum=False)
            container = ShiftTimeContainer(0.0, tempo_val, True, quarters_per_bar=quarters_per_bar)
        elif token == "<SME>":
            container.shift()
        elif "_" in token:
            prefix, val_str = token.split("_", 1)
            if prefix == "s":
                max_step = max(0, int(container.steps_per_bar) - 1)
                clamped_step = min(float(max_step), max(0.0, float(val_str)))
                shift = ct_beat_to_time(clamped_step, tempo_val)
                note.start = container.measure_start_time + shift
            elif prefix == "p":
                note.pitch = max(0, min(127, int(float(val_str))))
            elif prefix == "v":
                note.velocity = max(1, min(127, int(float(val_str))))
            elif prefix == "d":
                duration = ct_beat_to_time(float(val_str), tempo_val)
                note.end = note.start + max(duration, 0.01)
                if current_inst is None:
                    current_inst = inst.Instrument(program=1, is_drum=False)
                current_inst.notes.append(note)
                note = Note(pitch=60, velocity=100, start=0.0, end=0.0)

    if current_inst is None:
        current_inst = inst.Instrument(program=1, is_drum=False)
    midi.instruments.append(current_inst)
    midi.write(save_directory)
    return midi
