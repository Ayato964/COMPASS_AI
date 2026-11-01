from abc import abstractmethod
from decimal import Decimal
import math
import numbers
from typing import Any, Optional, Tuple
from pretty_midi import Note, Instrument as PrettyMidiInstrument
from mortm.train.utils.chord_midi import ChordMidi


def _coerce_time_sig_int(val: Any) -> int:
    if val is None or isinstance(val, (bool, str, bytes, bytearray)):
        raise ValueError("指定された拍子記号が不正です。")
    if type(val).__name__ in ("bool_", "ndarray", "Tensor"):
        raise ValueError("指定された拍子記号が不正です。")
    try:
        if getattr(val, "ndim", 0) != 0:
            raise ValueError("指定された拍子記号が不正です。")
        if not isinstance(val, (int, float, Decimal, numbers.Real)):
            raise ValueError("指定された拍子記号が不正です。")
        scalar = (
            val.item()
            if hasattr(val, "item") and type(val) not in (int, float, Decimal)
            else val
        )
        if (
            scalar is None
            or isinstance(scalar, (bool, str, bytes, bytearray))
            or type(scalar).__name__ in ("bool_", "ndarray", "Tensor")
            or not isinstance(scalar, (int, float, Decimal, numbers.Real))
        ):
            raise ValueError("指定された拍子記号が不正です。")
        if isinstance(scalar, float):
            if not math.isfinite(scalar) or scalar < -1_048_576.0 or scalar > 1_048_576.0:
                raise ValueError("指定された拍子記号が不正です。")
        elif isinstance(scalar, Decimal):
            if not scalar.is_finite() or scalar < -1_048_576 or scalar > 1_048_576:
                raise ValueError("指定された拍子記号が不正です。")
        elif type(scalar) is int:
            if scalar < -1_048_576 or scalar > 1_048_576:
                raise ValueError("指定された拍子記号が不正です。")
        else:
            f_probe = float(scalar)
            if not math.isfinite(f_probe) or f_probe < -1_048_576.0 or f_probe > 1_048_576.0:
                raise ValueError("指定された拍子記号が不正です。")
        iv = int(scalar)
        eq_res = iv == scalar
        is_exact = eq_res if type(eq_res) is bool else False
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("指定された拍子記号が不正です。") from exc
    if not is_exact:
        raise ValueError("指定された拍子記号が不正です。")
    return iv


def validate_supported_time_signature(numerator: Any, denominator: Any) -> Tuple[int, int]:
    """
    Validates `(numerator, denominator)` against MORTM's pretrained 24-steps-per-quarter-note
    `s_0..s_96` measure vocabulary (`12 <= steps_per_bar <= 96`, power-of-2 denominator in
    `{1, 2, 4, 8, 16, 32}`) using scalar-safe, overflow-safe integer arithmetic
    (`[R4-MED-02]`, `[R5-LOW-01]`, `[R6-LOW-02]`).
    """
    num = _coerce_time_sig_int(numerator)
    den = _coerce_time_sig_int(denominator)
    if den <= 0 or den > 32 or (den & (den - 1)) != 0 or num <= 0:
        raise ValueError("指定された拍子記号が不正です。")
    if num > 1024:
        raise ValueError("指定された拍子記号はモデルの1小節上限 (96ステップ) を超えています。")
    # Exact integer arithmetic: for `den in {1, 2, 4, 8, 16, 32}`, `96 % den == 0` always holds.
    steps_per_bar = num * (96 // den)
    if steps_per_bar < 12 or steps_per_bar > 96:
        if steps_per_bar > 96:
            raise ValueError(
                f"指定された拍子記号 ({num}/{den}, {steps_per_bar} steps) はモデルの1小節上限 (96ステップ) を超えています。"
            )
        raise ValueError(
            f"指定された拍子記号 ({num}/{den}, {steps_per_bar} steps) はモデルの1小節下限 (12ステップ) 未満です。"
        )
    return num, den

roots = [
    'C', 'C#', 'D', 'D#', 'E', 'E#', 'F', 'F#', 'G', 'G#',
    'A', 'A#', 'B', 'Cb', 'Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bb'
]
bases = [
    '/C', '/C#', '/D', '/D#', '/E', '/E#', '/F', '/F#', '/G', '/G#',
    '/A', '/A#', '/B', '/B#', '/Db', '/Eb', '/Fb', '/Gb', '/Ab', '/Bb',
    '/Cb', 'None'
]
qualities = [
    'None', 'm', '7', 'maj7', 'm7', 'dim', 'aug', 'm7b5', 'b5',
    'sus2', 'sus4', '6', 'm6', '9', 'maj9', 'm9', '11', '13', 'add9'
]

key = [
    'CM', 'DM', 'EM', 'FM', 'GM', 'AM', 'BM', 'C#M', 'D#M', 'F#M', 'G#M', 'A#M',
    'DbM', 'EbM', 'GbM', 'AbM', 'BbM', 'Unknown',
    'Cm', 'Dm', 'Em', 'Fm', 'Gm', 'Am', 'Bm', 'C#m', 'D#m', 'F#m', 'G#m', 'A#m',
    'Dbm', 'Ebm', 'Gbm', 'Abm', 'Bbm'
]

_SORTED_ROOTS = sorted(roots, key=lambda x: -len(x))
_QUALITY_SET = frozenset(qualities)


def parse_chord(chord: str) -> Tuple[str, str, str]:
    """
    コード文字列を root, quality, base に分解する。
    例: "Em/G" -> ("E", "m", "/G")
    """
    if '/' in chord:
        core, bass_note = chord.split('/', 1)
        base = '/' + bass_note
    else:
        core = chord
        base = 'None'

    for r in _SORTED_ROOTS:
        if core.startswith(r):
            root = r
            quality = core[len(r):] or 'None'
            break
    else:
        raise ValueError(f"Unknown root in chord '{chord}'")

    if quality not in _QUALITY_SET:
        quality = 'None'

    return root, quality, base


def ct_time_to_beat(time: float, tempo: float) -> int:
    """Convert time (seconds) to 96-subdivision beat index with grid-boundary rounding."""
    safe_tempo = max(20.0, min(300.0, float(tempo) if math.isfinite(tempo) and tempo > 0 else 120.0))
    b4 = 60.0 / safe_tempo
    measure = b4 * 4.0
    b96 = measure / 96.0
    beat, _ = calc_time_to_beat(time, b96)
    return beat


def ct_beat_to_time(beat: float, tempo: float) -> float:
    """Convert 96-subdivision beat index to time (seconds)."""
    safe_tempo = max(20.0, min(300.0, float(tempo) if math.isfinite(tempo) and tempo > 0 else 120.0))
    b4 = 60.0 / safe_tempo
    measure = b4 * 4.0
    b96 = measure / 96.0
    return float(beat) * b96


def calc_time_to_beat(time: float, beat_time: float) -> Tuple[int, float]:
    if beat_time <= 0.0 or not math.isfinite(time) or not math.isfinite(beat_time):
        return 0, 0.0
    exact = time / beat_time
    nearest = round(exact)
    if abs(exact - nearest) <= 1e-3:
        return int(nearest), 0.0
    main_beat = int(math.floor(exact))
    sub_time = max(0.0, (exact - main_beat) * beat_time)
    return main_beat, sub_time


def _get_symbol(token: str) -> int:
    split = token.split("_")
    return int(float(split[-1]))


class ShiftTimeContainer:
    def __init__(self, time: float, tempo: float, is_velocity: bool, quarters_per_bar: float = 4.0):
        safe_tempo = max(20.0, min(300.0, float(tempo) if math.isfinite(tempo) and tempo > 0 else 120.0))
        qpb = float(quarters_per_bar) if (math.isfinite(quarters_per_bar) and quarters_per_bar > 0) else 4.0
        self.quarters_per_bar = qpb
        self.steps_per_bar = max(1, min(96, int(round(qpb * 24.0))))
        self._initial_time = float(time)
        self._initial_tempo = safe_tempo
        self._constant_tempo_shifts: Optional[int] = 0
        self.measure_start_time = float(time)
        self.shift_measure = False
        self.is_error = False
        self.is_velocity = is_velocity
        self.tempo = safe_tempo
        self.is_code_mode = False
        self.inst: Optional[str] = None

    def shift(self):
        safe_tempo = max(20.0, min(300.0, float(self.tempo) if math.isfinite(self.tempo) and self.tempo > 0 else 120.0))
        bar_seconds = (60.0 / safe_tempo) * self.quarters_per_bar
        if self._constant_tempo_shifts is not None and abs(safe_tempo - self._initial_tempo) <= 1e-9:
            self._constant_tempo_shifts += 1
            self.measure_start_time = self._initial_time + self._constant_tempo_shifts * bar_seconds
        else:
            self._constant_tempo_shifts = None
            self.measure_start_time += bar_seconds
        self.shift_measure = True

    def get_inst(self) -> Optional[int]:
        if self.inst is None:
            return None
        if self.inst == "<INST_SAX>":
            return 66
        elif self.inst == "<INST_PIANO>":
            return 1
        else:
            return 1


class Token:
    def __init__(self, token_type: str, convert_type: int):
        self.token_type = token_type
        self.token_position = 0
        self.convert_type = convert_type
        self.start = 0
        self.end = 0

    def get_token_length_tuple(self) -> Tuple[int, int]:
        return self.start, self.end

    @abstractmethod
    def get_token(self, *args, **kwargs) -> int | str | None:
        pass

    @abstractmethod
    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    @abstractmethod
    def _set_tokens(self, tokens: dict):
        pass

    def set_tokens(self, tokens: dict):
        self.start = len(tokens)
        self._set_tokens(tokens)
        self.end = len(tokens) - 1

    @abstractmethod
    def is_my_token(self, seq):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SpecialToken(Token):
    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def _set_tokens(self, tokens: dict):
        tokens[self.token_type] = len(tokens)

    def is_my_token(self, seq):
        pass

    @abstractmethod
    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def __call__(
        self,
        inst: PrettyMidiInstrument = None,
        back_notes: Note = None,
        note: Note = None,
        token: str = None,
        tempo: int = 120,
        container: ShiftTimeContainer = None,
        *args,
        **kwargs,
    ):
        if self.convert_type == 0:
            return self.get_token(inst=inst, back_notes=back_notes, note=note, tempo=tempo, container=container)
        else:
            self.de_convert(token, back_notes, note, tempo, container)
            return self.token_type


class MusicToken(Token):
    def __call__(
        self,
        inst: PrettyMidiInstrument = None,
        back_notes: Note = None,
        note: Note = None,
        token: str = None,
        tempo: int = 120,
        container: ShiftTimeContainer = None,
        *args,
        **kwargs,
    ):
        if self.convert_type == 0:
            k = self.get_token(inst=inst, back_notes=back_notes, note=note, tempo=tempo, container=container)
            return f"{self.token_type}_{k}" if k is not None else None
        else:
            if token is None:
                return None
            split = token.split("_")
            if split[0] == self.token_type:
                self.de_convert(split[1], back_notes, note, tempo, container)
                return split[0]
            else:
                return None

    @abstractmethod
    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    @abstractmethod
    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    @abstractmethod
    def _set_tokens(self, tokens: dict):
        pass

    @abstractmethod
    def is_my_token(self, seq):
        pass


class ChordToken(Token):
    def __call__(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer, token=None, *args, **kwargs):
        if self.convert_type == 0:
            token_c = self.get_token(note=note, chords=chords, container=container)
            if token_c is not None:
                return f"{self.token_type}_{token_c}"
            else:
                return None
        else:
            t_s = token.split("_")
            return t_s[1]

    def __init__(self, token_type: str, convert_type: int):
        super().__init__(token_type, convert_type)

    @abstractmethod
    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> int | str | None:
        pass


class TrackEnd(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<TE>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        if inst.notes[-1] == note:
            return self.token_type
        return None


class MeasureToken(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<SME>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        note_tick = ct_time_to_beat(note.start, tempo)
        container_tick = ct_time_to_beat(container.measure_start_time, tempo)
        steps_per_bar = getattr(container, "steps_per_bar", 96)
        if note_tick - container_tick >= steps_per_bar:
            container.shift()
            return self.token_type
        return None

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        if number == self.token_type:
            container.shift()


class Blank(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<BLANK>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        container_tick = ct_time_to_beat(container.measure_start_time, tempo)
        note_tick = ct_time_to_beat(note.start, tempo)
        steps_per_bar = getattr(container, "steps_per_bar", 96)
        if note_tick - container_tick >= steps_per_bar:
            return self.token_type
        return None


class SequenceEnd(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<ESEQ>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        return None


class MGen(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<MGEN>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        return None


class Meta(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<META>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class CGen(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<CGEN>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class CLS(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<CLS>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class ConstMelody(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<CONST_M>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class ConstChord(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<CONST_C>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class Motif(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<Motif>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class GenMotif(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<MotifGen>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class EOS(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<EOS>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class System(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<SYSTEM>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class TagEnd(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<TAG_END>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class PastMelody(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<PAST_M>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class FutureMelody(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<FUTURE_M>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class GenMeasureCount(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<GEN_MEASURE_COUNT>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i in range(1, 9):
            tokens[f'<GEN_MEASURE_COUNT_{i}>'] = base + i - 1


class NoteDense(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<NOTE_DENSE>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i in range(1, 11):
            tokens[f'<NOTE_DENSE_{i}>'] = base + i - 1


class Genre(SpecialToken):
    """ジャンルトークン (MIDICaps 42 genres, sorted fixed order)."""
    GENRES = [
        "80s", "90s", "alternative", "ambient", "blues", "celtic", "chillout",
        "classical", "country", "dance", "drumnbass", "easylistening", "electronic",
        "electropop", "experimental", "folk", "funk", "hiphop", "house", "indie",
        "instrumentalpop", "instrumentalrock", "jazz", "jazzfusion", "latin", "lounge",
        "metal", "newage", "orchestral", "pop", "popfolk", "poprock", "punkrock",
        "reggae", "rock", "soundtrack", "swing", "symphonic", "synthpop", "techno",
        "trance", "world",
    ]

    def __init__(self, convert_type: int):
        super().__init__("<GENRE>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, g in enumerate(self.GENRES):
            tokens[f'<GENRE_{g}>'] = base + i


class Thinking(SpecialToken):
    """CoT (Chain-of-Thought) trigger <thinking>."""
    def __init__(self, convert_type: int):
        super().__init__("<thinking>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class AnalysisTrigger(SpecialToken):
    """Analysis task triggers: <KEY>, <DENCE>, <GENRE>, <LENGTH>."""
    TRIGGERS = ["<KEY>", "<DENCE>", "<GENRE>", "<LENGTH>"]

    def __init__(self, convert_type: int):
        super().__init__("<ANALYSIS>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, t in enumerate(self.TRIGGERS):
            tokens[t] = base + i


class Key(MusicToken):
    def __init__(self, convert_type: int):
        super().__init__("k", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, k in enumerate(key):
            tokens[f'k_{k}'] = base + i

    def is_my_token(self, seq):
        pass


class Instrument(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<INST>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        if number is not None and "INST" in str(number):
            container.inst = str(number)

    def _set_tokens(self, tokens: dict):
        tokens['<INST_SAX>'] = len(tokens)
        tokens['<INST_PIANO>'] = len(tokens)


class Eval(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<EVAL>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class Human(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<HUMAN>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class AI(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<AI>", convert_type)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class StartRE(MusicToken):
    def _set_tokens(self, tokens: dict):
        max_length = 96
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f's_{i}'] = tokens_length + i

    def de_convert(self, number: int | str, back_note, note: Note, tempo, container: ShiftTimeContainer):
        steps_per_bar = max(1, int(getattr(container, "steps_per_bar", 96)))
        clamped_step = min(float(steps_per_bar - 1), max(0.0, float(number)))
        shift = ct_beat_to_time(clamped_step, tempo)
        note.start = container.measure_start_time + shift

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo, container: ShiftTimeContainer) -> int:
        time = max(0.0, note.start - container.measure_start_time)
        shift_time = max(0, ct_time_to_beat(time, tempo))
        steps_per_bar = max(1, int(getattr(container, "steps_per_bar", 96)))
        return int(min(steps_per_bar - 1, shift_time % steps_per_bar))

    def is_my_token(self, seq):
        pass


class Duration(MusicToken):
    def _set_tokens(self, tokens: dict):
        max_length = 96 * 3
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'd_{i}'] = tokens_length + i

    def de_convert(self, number: int | str, back_note, note: Note, tempo, container: ShiftTimeContainer):
        duration = ct_beat_to_time(float(number), tempo)
        note.end = note.start + duration

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo, container: ShiftTimeContainer) -> int:
        start = ct_time_to_beat(note.start, tempo)
        end = ct_time_to_beat(note.end, tempo)
        d = int(max(abs(end - start), 1))
        if d > 96 * 3:
            d = 96 * 3 - 1
        return d

    def is_my_token(self, seq):
        pass


class Pitch(MusicToken):
    def is_my_token(self, seq):
        pass

    def _set_tokens(self, tokens: dict):
        max_length = 127
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'p_{i}'] = tokens_length + i

    def de_convert(self, number: int | str, back_note, note: Note, tempo, container: ShiftTimeContainer):
        note.pitch = int(number)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo, container: ShiftTimeContainer) -> int:
        return int(note.pitch)


class Velocity(MusicToken):
    def is_my_token(self, seq):
        pass

    def _set_tokens(self, tokens: dict):
        max_length = 127
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'v_{i}'] = tokens_length + i

    def de_convert(self, number: int | str, back_note, note: Note, tempo, container: ShiftTimeContainer):
        note.velocity = int(number)

    def get_token(self, inst: PrettyMidiInstrument, back_notes: Note, note: Note, tempo, container: ShiftTimeContainer) -> int:
        return int(note.velocity)


class ChordRoot(ChordToken):
    def __init__(self, convert_type: int):
        super().__init__('CR', convert_type)

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, r in enumerate(roots):
            tokens[f'CR_{r}'] = base + i

    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> Optional[str]:
        p = chords.get_chord(note.start)
        if p is not None:
            c, _ = p
            if not c.is_called:
                root, _, _ = parse_chord(c.chord)
                return root
        return None

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def is_my_token(self, seq):
        pass


class ChordQuality(ChordToken):
    def __init__(self, convert_type: int):
        super().__init__('CQ', convert_type)

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, q in enumerate(qualities):
            tokens[f'CQ_{q}'] = base + i

    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> Optional[str]:
        p = chords.get_chord(note.start)
        if p is not None:
            c, _ = p
            if not c.is_called:
                _, q, _ = parse_chord(c.chord)
                return q
        return None

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def is_my_token(self, seq):
        pass


class ChordBass(ChordToken):
    def __init__(self, convert_type: int):
        super().__init__('CB', convert_type)

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, r in enumerate(bases):
            tokens[f'CB_{r}'] = base + i

    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> Optional[str]:
        p = chords.get_chord(note.start)
        if p is not None:
            c, _ = p
            if not c.is_called:
                _, _, base = parse_chord(c.chord)
                c.is_called = True
                return base
        return None

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def is_my_token(self, seq):
        pass


class ChordShiftRE(ChordToken):
    def __init__(self, token_type: str, convert_type: int):
        super().__init__(token_type, convert_type)

    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> int | str | None:
        if container.is_code_mode:
            now_c, _ = chords.get_chord(note.start)
            time = now_c.time_stamp - container.measure_start_time
            shift_time = ct_time_to_beat(time, container.tempo)
            steps_per_bar = max(1, int(getattr(container, "steps_per_bar", 96)))
            return int(min(steps_per_bar - 1, shift_time % steps_per_bar))
        return None

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def _set_tokens(self, tokens: dict):
        pass

    def is_my_token(self, seq):
        pass
