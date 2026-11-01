import json
from typing import Dict, List, Optional, Tuple

from mortm import constants
from mortm.train.custom_token import (
    AI,
    AnalysisTrigger,
    Blank,
    CGen,
    ChordBass,
    ChordQuality,
    ChordRoot,
    ChordShiftRE,
    ConstChord,
    ConstMelody,
    Duration,
    EOS,
    Eval,
    FutureMelody,
    GenMeasureCount,
    GenMotif,
    Genre,
    Human,
    Instrument,
    Key,
    MGen,
    MeasureToken,
    Meta,
    Motif,
    NoteDense,
    PastMelody,
    Pitch,
    SequenceEnd,
    StartRE,
    System,
    TagEnd,
    Thinking,
    Token,
    TrackEnd,
    Velocity,
)

PITCH_TYPE = 'p'
VELOCITY_TYPE = 'v'
DURATION_TYPE = 'd'

START_TYPE = 's'
SHIFT_TYPE = 'h'
MEASURE_TYPE = 'm'

TO_TOKEN = 0
TO_MUSIC = 1


def get_token_converter_pro(convert: int, old_version: bool = True) -> List[Token]:
    register: List[Token] = [
        EOS(convert),
        TagEnd(convert),
        PastMelody(convert),
        ConstChord(convert),
        ConstMelody(convert),
        FutureMelody(convert),
        Motif(convert),
        System(convert),
        Instrument(convert),
        GenMotif(convert),
        MGen(convert),
        CGen(convert),
        Key(convert),
        MeasureToken(convert),
        Blank(convert),
        StartRE(START_TYPE, convert),
        ChordShiftRE(START_TYPE, convert),
        ChordRoot(convert),
        ChordQuality(convert),
        ChordBass(convert),
        Pitch(PITCH_TYPE, convert),
        Duration(DURATION_TYPE, convert),
        SequenceEnd(convert),
        TrackEnd(convert),
        Meta(convert),
        GenMeasureCount(convert),
        NoteDense(convert),
        Genre(convert),            # 42 genres (IDs 647..688)
        Thinking(convert),         # CoT trigger <thinking> (ID 689)
        AnalysisTrigger(convert),  # <KEY>, <DENCE>, <GENRE>, <LENGTH> (IDs 690..693)
    ]
    return register


def get_token_converter_melody_only(convert: int) -> List[Token]:
    return [
        EOS(convert),
        MGen(convert),
        CGen(convert),
        Key(convert),
        MeasureToken(convert),
        Blank(convert),
        StartRE(START_TYPE, convert),
        Pitch(PITCH_TYPE, convert),
        Duration(DURATION_TYPE, convert),
        SequenceEnd(convert),
        TrackEnd(convert),
    ]


def bertm_converter(convert: int) -> List[Token]:
    register: List[Token] = get_token_converter_pro(convert)
    register.append(Eval(convert))
    register.append(Human(convert))
    register.append(AI(convert))
    return register


def omega_converter(convert: int) -> List[Token]:
    """Converter for MORTM 4.5 generation preview models (vocab_size=650, IDs 0..646)."""
    return get_token_converter_pro(convert)[:-3]


class Tokenizer:
    """
    MORTM Tokenizer with O(1) token & converter lookup.
    """

    def __init__(self, music_token: List[Token], load_data: Optional[str] = None):
        self.music_token_list = music_token
        self._converter_by_type: Dict[str, Token] = {}

        if load_data is None:
            self.tokens: Dict[str, int] = {constants.PADDING_TOKEN: 0}
            for t in music_token:
                t.set_tokens(self.tokens)
                if t.token_type not in self._converter_by_type:
                    self._converter_by_type[t.token_type] = t

            self.token_max: Dict[int, int] = {i: 0 for i in range(len(self.tokens))}
            self.rev_tokens: Optional[Dict[int, str]] = {v: k for k, v in self.tokens.items()}
        else:
            with open(load_data, 'r', encoding='utf-8') as file:
                self.tokens = json.load(file)
            for t in music_token:
                dummy: Dict[str, int] = {}
                t._set_tokens(dummy)
                ids = [self.tokens[k] for k in dummy.keys() if k in self.tokens]
                if ids:
                    t.start = min(ids)
                    t.end = max(ids)
                else:
                    t.start = len(self.tokens)
                    t._set_tokens(self.tokens)
                    t.end = len(self.tokens) - 1
                if t.token_type not in self._converter_by_type:
                    self._converter_by_type[t.token_type] = t
            self.token_max = {i: 0 for i in range(len(self.tokens))}
            self.rev_tokens = {v: k for k, v in self.tokens.items()}

    def rev_get(self, token_id: int) -> str:
        if self.rev_tokens is None:
            self.rev_tokens = {v: k for k, v in self.tokens.items()}
        return self.rev_tokens.get(int(token_id), constants.PADDING_TOKEN)

    def get(self, token: Optional[str]) -> Optional[int]:
        if token is not None:
            if token not in self.tokens and "<" not in token:
                sp = token.split('_')
                length = self.get_length(sp[0])
                self.tokens[token] = length
                if self.rev_tokens is not None:
                    self.rev_tokens[length] = token
            return self.tokens.get(token)
        return None

    def get_length(self, token_type: str) -> Optional[int]:
        conv = self._converter_by_type.get(token_type)
        if conv is not None:
            p = conv.token_position
            conv.token_position += 1
            return p
        return None

    def get_length_tuple(self, token_type: str) -> Optional[Tuple[int, int]]:
        conv = self._converter_by_type.get(token_type)
        if conv is not None:
            return conv.get_token_length_tuple()
        return None

    def save(self, save_directory: str):
        with open(f"{save_directory}/vocab_list.json", 'w', encoding='utf-8') as file:
            json.dump(self.tokens, file)
        with open(f"{save_directory}/vocab_max.json", 'w', encoding='utf-8') as file:
            json.dump(self.token_max, file)

    def mode(self, to: int = TO_MUSIC):
        for li in self.music_token_list:
            li.convert_type = to
        if to == TO_MUSIC and self.rev_tokens is None:
            self.rev_tokens = {v: k for k, v in self.tokens.items()}

    def begin_token(self, token_type: str):
        if self.rev_tokens is None:
            self.rev_tokens = {v: k for k, v in self.tokens.items()}
        for i in range(len(self.rev_tokens)):
            if i in self.rev_tokens and token_type in self.rev_tokens[i]:
                return i
        return "NOT FOUND"

    def end_token(self, token_type: str):
        if self.rev_tokens is None:
            self.rev_tokens = {v: k for k, v in self.tokens.items()}
        for i in range(len(self.rev_tokens)):
            token = len(self.rev_tokens) - 1 - i
            if token in self.rev_tokens and token_type in self.rev_tokens[token]:
                return token
        return None

    def get_token_converter(self, token_type: str) -> Optional[Token]:
        return self._converter_by_type.get(token_type)
