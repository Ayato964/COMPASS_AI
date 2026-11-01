"""
COMPASS AI Model Wrapper Interfaces, Metadata Schemas, and Factory (SOLID / Template Method & Factory Patterns).
100% compatible with MozartAPI v2 (`GenerateMeta`, `AbstractModelRapper`, `ModelRapperFactory`).
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, Field, model_validator
import torch

from mortm.models.modules.backend import HardwareBackendFactory, IHardwareBackend
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import DefaultObserver
from mortm.models.mortm import MORTM
from mortm.train.custom_token import validate_supported_time_signature


class GenerateMeta(BaseModel):
    """
    Request metadata schema for `POST /generate`.
    Supports all MozartAPI v1/v2 fields (`model_type`, `program`, `genfield_measure`, `gen_note_dense`, `p`)
    as well as convenient aliases (`model_name`, `target_instruments`, `gen_measure_count`, `note_density`, `top_p`).
    """

    model_type: str = Field("", description="使用するモデルの識別名")
    program: List[Union[str, int]] = Field(default_factory=lambda: ["PIANO"])
    tempo: int = 120
    task: str = "Meta2MIDI"
    key: Optional[str] = None

    num_gems: int = 1
    genfield_measure: Optional[int] = 8
    gen_note_dense: Optional[Union[int, Dict[str, int]]] = Field(default_factory=lambda: {"PIANO": 4})
    p: float = 0.95
    temperature: float = 1.0
    chord_item: Optional[List[str]] = None
    chord_times: Optional[List[float]] = None
    split_measure: int = 999
    ai_continue_mode: bool = False

    # MORTM 4.5D / 4.5E SFT-Gen fields
    genre: Optional[Union[str, List[str]]] = None
    thinking: bool = True
    cot_temperature: float = 0.1

    # Optional context measure slicing & time signature ([R2-HIGH-05])
    start_measure: Optional[int] = 0
    measure_count: Optional[int] = None
    future_start_measure: Optional[int] = 0
    future_measure_count: Optional[int] = None
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if not d.get("model_type") and d.get("model_name"):
            d["model_type"] = d["model_name"]
        if "target_instruments" in d and d["target_instruments"] and "program" not in data:
            d["program"] = d["target_instruments"]
        if "gen_measure_count" in d and "genfield_measure" not in data:
            d["genfield_measure"] = d["gen_measure_count"]
        if "note_density" in d and "gen_note_dense" not in data:
            d["gen_note_dense"] = d["note_density"]
        if "top_p" in d and d["top_p"] is not None and "p" not in data:
            d["p"] = d["top_p"]
        if isinstance(d.get("genre"), str):
            d["genre"] = [d["genre"]]
        return d

    @model_validator(mode="after")
    def validate_values(self):
        if not self.model_type:
            raise ValueError("model_type (または model_name) は必須です")
        if not (1 <= self.tempo <= 500):
            raise ValueError("tempo must be between 1 and 500")
        if not (1 <= self.num_gems <= 16):
            raise ValueError("num_gems must be between 1 and 16")
        if self.genfield_measure is not None and not (1 <= self.genfield_measure <= 64):
            raise ValueError("genfield_measure must be between 1 and 64")
        if not (0.0 < self.p <= 1.0):
            raise ValueError("p must be in (0, 1]")
        if not (0.0 < self.temperature <= 10.0):
            raise ValueError("temperature must be in (0, 10.0]")
        if not (0.0 < self.cot_temperature <= 10.0):
            raise ValueError("cot_temperature must be in (0, 10.0]")
        if self.start_measure is not None and not (0 <= self.start_measure <= 1024):
            raise ValueError("start_measure must be between 0 and 1024")
        if self.measure_count is not None and not (1 <= self.measure_count <= 64):
            raise ValueError("measure_count must be between 1 and 64")
        if self.future_start_measure is not None and not (0 <= self.future_start_measure <= 1024):
            raise ValueError("future_start_measure must be between 0 and 1024")
        if self.future_measure_count is not None and not (1 <= self.future_measure_count <= 64):
            raise ValueError("future_measure_count must be between 1 and 64")
        if (self.chord_item is None) ^ (self.chord_times is None):
            raise ValueError("chord_item と chord_times は両方指定するか、両方省略してください")
        if self.chord_item is not None and len(self.chord_item) != len(self.chord_times):
            raise ValueError("chord_item と chord_times の長さが一致していません")
        validate_supported_time_signature(
            self.time_signature_numerator, self.time_signature_denominator
        )
        return self


# Alias for callers importing MetaData
MetaData = GenerateMeta


class AbstractModelRapper(ABC):
    """
    Abstract Template-Method base class for all MORTM model wrappers:
    `preprocessing` -> `generate` -> `postprocessing`.
    Provides shared safe checkpoint loading (`weights_only=True`), FP32 LoRA folding before
    device/dtype casting, and shared instrument/density/measure normalization helpers.
    """

    def __init__(self, model_info: dict, backend: Optional[IHardwareBackend] = None):
        self.meta = model_info
        self.backend: IHardwareBackend = backend or HardwareBackendFactory.create("auto")
        self.last_reason: Optional[List[Dict[str, Any]]] = None
        self.merged_lora_layers: int = 0
        self.model: MORTM = self._load_model()

    def _load_model(self) -> MORTM:
        model_path = self.meta["model_folder_path"]
        config_path = os.path.join(model_path, "config.json")
        model_pth_path = os.path.join(model_path, "model.pth")

        args = MORTMArgs(config_path)
        state: Optional[Dict[str, torch.Tensor]] = None

        if os.path.exists(model_pth_path):
            # Safe deserialization (CWE-502): weights_only=True prevents arbitrary pickle code execution
            raw_state = torch.load(model_pth_path, map_location="cpu", weights_only=True)
            if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
                raw_state = raw_state["state_dict"]
            if not isinstance(raw_state, dict):
                raise ValueError("モデルの重みファイル (model.pth) の state_dict 形式が不正です。")

            state = {}
            for k, v in raw_state.items():
                if not isinstance(v, torch.Tensor):
                    continue
                clean_key = k.replace("module.", "", 1)
                state[clean_key] = v

            # Detect if an MoE LoRA checkpoint was already pre-folded into stacked w13/w2 tensors
            if args.use_moe_decoder and args.use_ffn_lora:
                has_stacked = any(k.endswith(".ffn.w13") for k in state)
                has_experts = any(".ffn.experts." in k for k in state)
                if has_stacked and not has_experts:
                    args.lora_merged = True
        elif os.environ.get("COMPASS_ALLOW_UNINITIALIZED_MODEL", "1") != "1":
            raise FileNotFoundError(
                "モデル重みファイル (model.pth) が見つかりません。AppData/Roaming/COMPASS_AI/models 内に配置してください。"
            )

        progress = DefaultObserver(self.backend)
        model = MORTM(args, progress=progress, backend=self.backend)
        if state is not None:
            model.load_state_dict(state, strict=False)

        # Fold LoRA weights in-place in FP32 BEFORE casting to target compute_dtype (BF16/FP16)
        self.merged_lora_layers = model.merge_lora()
        model.to(device=self.backend.device, dtype=self.backend.compute_dtype)
        model.eval()
        return model

    @staticmethod
    def normalize_program_name(value: Union[str, int]) -> str:
        if isinstance(value, int):
            if 0 <= value <= 6:
                return "PIANO"
            if 64 <= value <= 68:
                return "SAX"
            raise ValueError(f"Unsupported MIDI program id: {value}. PIANO(0-6) / SAX(64-68) のみ対応です。")

        text = str(value).strip()
        if text.isdigit():
            return AbstractModelRapper.normalize_program_name(int(text))

        upper = text.upper()
        aliases = {
            "PIANO": "PIANO",
            "ACOUSTIC_GRAND_PIANO": "PIANO",
            "GRANDPIANO": "PIANO",
            "SAX": "SAX",
            "ALTO_SAX": "SAX",
            "ALTOSAX": "SAX",
            "TENOR_SAX": "SAX",
            "TENORSAX": "SAX",
        }
        if upper not in aliases:
            raise ValueError(f"Unsupported program token: {value}. PIANO / SAX のみ対応です。")
        return aliases[upper]

    def normalize_programs(
        self,
        programs: Optional[Sequence[Union[str, int]]],
        *,
        require_non_empty: bool = False,
    ) -> List[str]:
        if not programs:
            if require_non_empty:
                raise ValueError("program が空です")
            return []
        normalized = [self.normalize_program_name(p) for p in programs]
        deduped: List[str] = []
        for p in normalized:
            if p not in deduped:
                deduped.append(p)
        if require_non_empty and not deduped:
            raise ValueError("program が空です")
        return deduped

    @staticmethod
    def clamp_measure_count(value: int) -> int:
        return max(1, min(int(value), 8))

    @staticmethod
    def clamp_density(value: int) -> int:
        return max(1, min(int(value), 10))

    @staticmethod
    def resolve_time_signature(meta: GenerateMeta) -> Tuple[int, int]:
        """
        Validates that `(time_signature_numerator, time_signature_denominator)` fits within MORTM's
        pretrained `s_0..s_96` 24-steps-per-quarter-note measure grid (`12 <= steps_per_bar <= 96`,
        supporting `2/4`, `3/8`, `5/8`, `3/4`, `6/8`, `7/8`, `4/4`, etc. without shifting vocabulary IDs,
        while cleanly rejecting >96-step or <12-step meters such as `5/4`, `6/4`, `12/8`, `0/4`, `1/16`
        with `ValueError` -> `HTTP 400`).
        """
        raw_num = getattr(meta, "time_signature_numerator", None)
        raw_den = getattr(meta, "time_signature_denominator", None)
        num = 4 if raw_num is None else raw_num
        den = 4 if raw_den is None else raw_den
        return validate_supported_time_signature(num, den)

    @staticmethod
    def build_out_of_bar_shift_token_ids(
        tokenizer: Any,
        time_sig: Tuple[int, int] = (4, 4),
    ) -> List[int]:
        """
        Returns vocabulary token IDs for out-of-bar shift tokens (`s_{steps_per_bar}..s_96`)
        so autoregressive decoding in non-4/4 meters (`3/4`, `2/4`, `6/8`, `7/8`, `5/8`, `3/8`)
        as well as `s_96` boundary overflow is masked to `-inf` at logit sampling (`[R3-HIGH-01]`).
        """
        num = int(time_sig[0]) if time_sig and int(time_sig[0]) > 0 else 4
        den = int(time_sig[1]) if time_sig and int(time_sig[1]) > 0 else 4
        quarters_per_bar = float(num) * (4.0 / float(den))
        steps_per_bar = max(12, min(96, int(round(quarters_per_bar * 24.0))))
        tok_dict = getattr(tokenizer, "tokens", None)
        out: List[int] = []
        for step in range(steps_per_bar, 97):
            key = f"s_{step}"
            tid = tok_dict.get(key) if isinstance(tok_dict, dict) else tokenizer.get(key)
            if tid is not None:
                out.append(int(tid))
        return out

    @staticmethod
    def has_explicit_time_signature(meta: GenerateMeta) -> bool:
        fields_set = getattr(meta, "model_fields_set", set()) or set()
        return (
            "time_signature_numerator" in fields_set
            or "time_signature_denominator" in fields_set
        )

    @classmethod
    def resolve_unified_time_signature(
        cls,
        meta: GenerateMeta,
        midi_paths: Sequence[Optional[str]] = (),
    ) -> Tuple[int, int]:
        """
        Pre-resolves a single unified `(numerator, denominator)` meter across `meta` and all
        active context MIDI files BEFORE any prompt tokenization occurs (`[R4-MED-02]`):
        - Validates `meta` time signature (`12 <= steps_per_bar <= 96`, power-of-2 denominator).
        - Validates all SMF `time_signature_changes` in every active context MIDI.
        - When `meta` explicitly sets `time_signature_*`, returns `requested_ts` so all context
          blocks, logit masks, and SMF output share the requested meter grid.
        - When `meta` omits `time_signature_*`, inherits the unique explicit SMF time signature
          detected across any context MIDI (even if sibling context MIDIs omit `0xFF 0x58`),
          rejecting conflicting explicit meters (`ValueError` -> `HTTP 400`).
        """
        from pretty_midi import PrettyMIDI

        requested_ts = cls.resolve_time_signature(meta)
        explicit_meta_ts = cls.has_explicit_time_signature(meta)

        seen_paths: List[str] = []
        for p in midi_paths:
            if p and p not in seen_paths:
                seen_paths.append(p)

        distinct_file_meters: List[Tuple[int, int]] = []
        for p in seen_paths:
            try:
                pm = PrettyMIDI(p)
            except Exception as exc:
                raise ValueError("MIDIのロードができませんでした。") from exc

            if not explicit_meta_ts:
                for t_s in pm.time_signature_changes:
                    meter = validate_supported_time_signature(t_s.numerator, t_s.denominator)
                    if meter not in distinct_file_meters:
                        distinct_file_meters.append(meter)

        if explicit_meta_ts:
            return requested_ts

        if len(distinct_file_meters) > 1:
            formatted = ", ".join(f"{n}/{d}" for n, d in distinct_file_meters)
            raise ValueError(
                f"コンテキストMIDI間で異なる拍子記号 ({formatted}) が混在しています。拍子記号を統一するか meta で明示指定してください。"
            )
        if len(distinct_file_meters) == 1:
            return distinct_file_meters[0]
        return requested_ts

    @staticmethod
    def trim_music_tokens_on_event_boundary(
        tokenizer: Any,
        inner: Sequence[int],
        budget: int,
        *,
        keep_tail: bool,
    ) -> List[int]:
        """
        Slices inner instrument music tokens strictly on atomic event boundaries (`<SME> [<BLANK>]` or
        complete `(s_*, p_*, [v_*], d_*)` note tuples) so context truncation NEVER cuts mid-note (`[R2-HIGH-06]`),
        and strips any trailing incomplete note tuple even when `len(inner) <= budget` (`[R3-HIGH-02]`).
        """
        if budget <= 0 or not inner:
            return []

        sme_id = tokenizer.get("<SME>")
        blank_id = tokenizer.get("<BLANK>")
        s_range = tokenizer.get_length_tuple("s")
        p_range = tokenizer.get_length_tuple("p")
        v_range = tokenizer.get_length_tuple("v")
        d_range = tokenizer.get_length_tuple("d")

        def _in_range(val: int, rng: Optional[Tuple[int, int]]) -> bool:
            return rng is not None and rng[0] <= val <= rng[1]

        events: List[List[int]] = []
        n = len(inner)
        i = 0
        while i < n:
            tid = int(inner[i])
            if sme_id is not None and tid == sme_id:
                if blank_id is not None and i + 1 < n and int(inner[i + 1]) == blank_id:
                    events.append([tid, int(inner[i + 1])])
                    i += 2
                else:
                    events.append([tid])
                    i += 1
            elif _in_range(tid, s_range):
                ev = [tid]
                i += 1
                completed_note = False
                while i < n:
                    nt = int(inner[i])
                    if (sme_id is not None and nt == sme_id) or _in_range(nt, s_range):
                        break
                    ev.append(nt)
                    i += 1
                    if _in_range(nt, d_range):
                        completed_note = True
                        break
                if completed_note:
                    events.append(ev)
            elif _in_range(tid, p_range) or _in_range(tid, v_range) or _in_range(tid, d_range):
                # Drop any pre-existing orphan mid-note attribute token
                i += 1
            elif blank_id is not None and tid == blank_id:
                # Drop any orphan <BLANK> not immediately preceded by <SME> ([R4-MED-01])
                i += 1
            else:
                events.append([tid])
                i += 1

        if not events:
            return []

        bar_marker_ids = {t for t in (sme_id, blank_id) if t is not None}

        if not keep_tail:
            out: List[int] = []
            for ev in events:
                if len(out) + len(ev) > budget:
                    break
                out.extend(ev)
            while out and out[-1] in bar_marker_ids:
                out.pop()
            if not out or all(t in bar_marker_ids for t in out):
                return []
            return out

        kept_rev: List[List[int]] = []
        total = 0
        for ev in reversed(events):
            if total + len(ev) > budget:
                break
            kept_rev.append(ev)
            total += len(ev)
        kept = list(reversed(kept_rev))

        if len(kept) < len(events) and sme_id is not None:
            for idx_ev, ev in enumerate(kept):
                if ev[0] == sme_id:
                    after = [tok for sub in kept[idx_ev + 1 :] for tok in sub]
                    while after and after[-1] in bar_marker_ids:
                        after.pop()
                    if any(_in_range(t, s_range) for t in after):
                        return after
                    break

        flat_kept = [tok for sub in kept for tok in sub]
        while flat_kept and flat_kept[-1] in bar_marker_ids:
            flat_kept.pop()
        if not flat_kept or all(t in bar_marker_ids for t in flat_kept):
            return []
        return flat_kept

    @abstractmethod
    def preprocessing(
        self,
        past_midi: Optional[str],
        const_midi: Optional[str],
        future_midi: Optional[str],
        meta: GenerateMeta,
    ) -> dict:
        pass

    @abstractmethod
    def generate(self, **kwargs) -> dict:
        pass

    @abstractmethod
    def postprocessing(self, save_directory: str, **kwargs) -> list:
        pass


# Backwards-compatible alias
ModelRapper = AbstractModelRapper


class ModelRapperFactory:
    """
    Open/Closed Factory for creating `AbstractModelRapper` instances.
    Resolves either by explicit `model_name` registration OR dynamically by `tag.model`
    (`"foundation"`, `"sft_gen"`, `"pretrained"`, `"generation"`), allowing users to drop
    any newly trained SFT/Foundation model folder into `AppData/Roaming/COMPASS_AI/models`
    without modifying code.
    """

    def __init__(self):
        self._rappers_by_name: Dict[str, type] = {}
        self._rappers_by_tag: Dict[str, type] = {}

    def register_rapper(self, model_type_keyword: str, rapper_class: type):
        self._rappers_by_name[model_type_keyword] = rapper_class

    def register_tag_rapper(self, tag_model_type: str, rapper_class: type):
        self._rappers_by_tag[tag_model_type.lower()] = rapper_class

    def create_rapper(self, model_info: dict, backend: Optional[IHardwareBackend] = None) -> AbstractModelRapper:
        model_name = model_info.get("model_name", "")
        if model_name in self._rappers_by_name:
            return self._rappers_by_name[model_name](model_info, backend=backend)

        tag_model = str(model_info.get("tag", {}).get("model", "")).lower()
        if tag_model in self._rappers_by_tag:
            return self._rappers_by_tag[tag_model](model_info, backend=backend)

        raise ValueError(f"適切なモデルラッパーが見つかりません: model_name='{model_name}', tag.model='{tag_model}'")
