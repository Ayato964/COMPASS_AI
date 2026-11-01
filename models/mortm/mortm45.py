"""
MORTM 4.5 Legacy Preview Wrapper (`pretrained` / `generation` models: Flash / Pro).
Upgraded for cross-platform execution (CUDA / Apple Silicon MPS / CPU) and in-place LoRA folding.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from mortm.train.tokenizer import (
    TO_MUSIC,
    TO_TOKEN,
    Tokenizer,
    get_token_converter_pro,
    omega_converter,
)
from mortm.utils.convert import MIDIConverter
from mortm.utils.de_convert import ct_token_to_midi
from rapper import AbstractModelRapper, GenerateMeta

_TASK_ALIASES = {
    "meta": "Meta2MIDI",
    "metapast": "Meta2MIDI",
    "metafuture": "Meta2MIDI",
    "infill": "Meta2MIDI",
    "instcomp": "Chord2MIDI",
    "unconditional": "Meta2MIDI",
    "meta2midi": "Meta2MIDI",
    "prompt2midi": "Meta2MIDI",
    "generate": "Meta2MIDI",
    "melodygen": "Meta2MIDI",
    "chord2midi": "Chord2MIDI",
    "midi2chord": "MIDI2Chord",
    "midi2meta": "MIDI2Meta",
    "metagen": "MIDI2Meta",
}


class MORTM45Rapper(AbstractModelRapper):
    @staticmethod
    def _normalize_task(task: Optional[str]) -> str:
        if task is None:
            return "Meta2MIDI"
        key = "".join(ch for ch in task if ch.isalnum()).lower()
        if key not in _TASK_ALIASES:
            raise ValueError(f"Unsupported task: {task}")
        return _TASK_ALIASES[key]

    def _normalize_key(self, tokenizer: Tokenizer, raw_key: Optional[str]) -> Optional[str]:
        if not raw_key:
            return None
        k = str(raw_key).strip()
        if not k or k.lower() == "auto":
            return None
        if "major" in k.lower():
            k = f"{k.split()[0]}M"
        elif "minor" in k.lower():
            k = f"{k.split()[0]}m"
        if f"k_{k}" in tokenizer.tokens:
            return k[:32]
        return "Unknown"

    def _resolve_density(self, meta: GenerateMeta, program_name: str) -> int:
        dense = meta.gen_note_dense
        if dense is None:
            return 4
        if isinstance(dense, dict):
            for key in (program_name, program_name.upper(), program_name.lower()):
                if key in dense and dense[key] is not None:
                    return self.clamp_density(dense[key])
            if len(dense) == 1:
                val = next(iter(dense.values()))
                return self.clamp_density(val if val is not None else 4)
            return 4
        return self.clamp_density(dense)

    def _build_system_prompt(
        self,
        tokenizer: Tokenizer,
        meta: GenerateMeta,
        program_names: List[str],
        *,
        include_dense: bool,
        include_measure_count: bool,
    ) -> np.ndarray:
        prompt: List[int] = [tokenizer.get("<EOS>"), tokenizer.get("<SYSTEM>")]
        for program_name in program_names:
            prompt.append(tokenizer.get(f"<INST_{program_name}>"))
            if include_dense:
                dense = self._resolve_density(meta, program_name)
                prompt.append(tokenizer.get(f"<NOTE_DENSE_{dense}>"))

        if include_measure_count:
            eff_measures = meta.genfield_measure if meta.genfield_measure is not None else 8
            prompt.append(tokenizer.get(f"<GEN_MEASURE_COUNT_{self.clamp_measure_count(eff_measures)}>"))

        normalized_key = self._normalize_key(tokenizer, getattr(meta, "key", None))
        if normalized_key:
            k_id = tokenizer.get(f"k_{normalized_key}")
            if k_id is not None:
                prompt.append(k_id)

        prompt.append(tokenizer.get("<TAG_END>"))
        return np.asarray(prompt, dtype=np.int64)

    def _load_midi_node_dict(
        self,
        tokenizer: Tokenizer,
        midi_path: str,
        program_names: List[str],
        key: Optional[str],
        default_time_signature: Tuple[int, int] = (4, 4),
        force_time_signature: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Tuple[int, int]]]:
        converter = MIDIConverter(
            tokenizer,
            os.path.dirname(midi_path),
            os.path.basename(midi_path),
            program_list=program_names,
            key=key or "Unknown",
            use_midi2seq=True,
            use_midi2seq_with_chord=False,
            default_time_signature=default_time_signature,
            force_time_signature=force_time_signature,
        )
        converter.convert()
        if converter.is_error:
            raise ValueError(converter.error_reason)
        if converter.midi2seq is None:
            raise ValueError("MIDI2Seq converter was not initialized")
        detected_ts: Optional[Tuple[int, int]] = (
            converter.file_time_signature if converter.has_explicit_time_signature else None
        )
        return converter.midi2seq.aya_node, detected_ts

    def get_context(
        self,
        tokenizer: Tokenizer,
        node_dict: dict,
        key_token: str,
        max_tokens_per_inst: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ) -> List[int]:
        if max_total_tokens is not None and int(max_total_tokens) < 5:
            return []

        cr_range = tokenizer.get_length_tuple("CR")
        cq_range = tokenizer.get_length_tuple("CQ")
        cb_range = tokenizer.get_length_tuple("CB")
        te_id = tokenizer.get("<TE>")
        eseq_id = tokenizer.get("<ESEQ>")

        max_pos = max(32, int(getattr(self.model.args, "position_length", 8192)))
        num_insts = max(1, len(node_dict))
        keep_tail = key_token == "<PAST_M>"

        if max_total_tokens is None:
            inst_budget_default = max_tokens_per_inst or max(16, (max_pos // 4) // num_insts)
            framed_default: List[int] = []
            for program_name, inst in node_dict.items():
                clean_inst: List[int] = []
                for t_id in inst:
                    t_val = int(t_id)
                    if t_val in (te_id, eseq_id):
                        continue
                    if (
                        cr_range[0] <= t_val <= cr_range[1]
                        or cq_range[0] <= t_val <= cq_range[1]
                        or cb_range[0] <= t_val <= cb_range[1]
                    ):
                        continue
                    clean_inst.append(t_val)
                clean_inst = self.trim_music_tokens_on_event_boundary(
                    tokenizer, clean_inst, inst_budget_default, keep_tail=keep_tail
                )
                if clean_inst:
                    framed_default.append(tokenizer.get(f"<INST_{program_name}>"))
                    framed_default.extend(clean_inst)
                    framed_default.append(tokenizer.get("<ESEQ>"))
            if not framed_default:
                return []
            return [tokenizer.get(key_token)] + framed_default + [tokenizer.get("<TAG_END>")]

        remaining_budget = int(max_total_tokens) - 2  # reserve <KEY_TOKEN> and <TAG_END>
        remaining_insts = len(node_dict)
        framed_insts: List[int] = []

        for program_name, inst in node_dict.items():
            if remaining_budget < 3:
                break
            fit_insts = max(1, min(remaining_insts, remaining_budget // 5))
            frame_budget = max(3, remaining_budget // fit_insts)
            inst_budget = max(1, frame_budget - 2)

            clean_inst_bounded: List[int] = []
            for t_id in inst:
                t_val = int(t_id)
                if t_val in (te_id, eseq_id):
                    continue
                if (
                    cr_range[0] <= t_val <= cr_range[1]
                    or cq_range[0] <= t_val <= cq_range[1]
                    or cb_range[0] <= t_val <= cb_range[1]
                ):
                    continue
                clean_inst_bounded.append(t_val)

            clean_inst_bounded = self.trim_music_tokens_on_event_boundary(
                tokenizer, clean_inst_bounded, inst_budget, keep_tail=keep_tail
            )

            if clean_inst_bounded and len(clean_inst_bounded) + 2 <= remaining_budget:
                framed_insts.append(tokenizer.get(f"<INST_{program_name}>"))
                framed_insts.extend(clean_inst_bounded)
                framed_insts.append(tokenizer.get("<ESEQ>"))
                remaining_budget -= len(clean_inst_bounded) + 2
            remaining_insts -= 1

        if not framed_insts:
            return []

        return [tokenizer.get(key_token)] + framed_insts + [tokenizer.get("<TAG_END>")]

    def _build_midi_context(
        self,
        tokenizer: Tokenizer,
        midi_path: str,
        key_token: str,
        program_names: List[str],
        key: Optional[str],
        max_total_tokens: Optional[int] = None,
        default_time_signature: Tuple[int, int] = (4, 4),
        force_time_signature: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        extract_programs = ["PIANO", "SAX"] if key_token == "<CONST_M>" else program_names
        node_dict, detected_ts = self._load_midi_node_dict(
            tokenizer,
            midi_path,
            extract_programs,
            key,
            default_time_signature=default_time_signature,
            force_time_signature=force_time_signature,
        )
        return (
            np.asarray(
                self.get_context(tokenizer, node_dict, key_token, max_total_tokens=max_total_tokens),
                dtype=np.int64,
            ),
            detected_ts,
        )

    def _build_ai_continue_seed(
        self,
        tokenizer: Tokenizer,
        node_dict: dict,
        max_seed_tokens: Optional[int] = None,
    ) -> np.ndarray:
        if len(node_dict) != 1:
            raise ValueError("ai_continue_mode は現在 1 トラック入力のみ対応です。")

        te_id = tokenizer.get("<TE>")
        eseq_id = tokenizer.get("<ESEQ>")
        tag_end_id = tokenizer.get("<TAG_END>")
        mgen_id = tokenizer.get("<MGEN>")
        cgen_id = tokenizer.get("<CGEN>")

        cr_range = tokenizer.get_length_tuple("CR")
        cq_range = tokenizer.get_length_tuple("CQ")
        cb_range = tokenizer.get_length_tuple("CB")

        inst_tok_id: Optional[int] = None
        inner_tokens: List[int] = []
        for program_name, inst in node_dict.items():
            inst_tok_id = tokenizer.get(f"<INST_{program_name}>")
            for t_id in inst:
                t_id = int(t_id)
                if cr_range[0] <= t_id <= cr_range[1] or cq_range[0] <= t_id <= cq_range[1] or cb_range[0] <= t_id <= cb_range[1]:
                    continue
                if t_id in {te_id, eseq_id, tag_end_id, mgen_id, cgen_id}:
                    continue
                inner_tokens.append(t_id)

        if inst_tok_id is None or len(inner_tokens) == 0:
            raise ValueError("ai_continue_mode 用の seed が空です。past_midi を確認してください。")
        max_pos = max(32, int(getattr(self.model.args, "position_length", 8192)))
        seed_cap = max(2, int(max_seed_tokens)) if max_seed_tokens is not None else max(8, max_pos // 2)
        inner_cap = max(1, seed_cap - 1)
        inner_tokens = self.trim_music_tokens_on_event_boundary(
            tokenizer, inner_tokens, inner_cap, keep_tail=True
        )
        if not inner_tokens:
            return np.asarray([inst_tok_id], dtype=np.int64)
        seed = [inst_tok_id] + inner_tokens
        return np.asarray(seed, dtype=np.int64)

    def _extract_from_generation_start(self, tokenizer: Tokenizer, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.int64)
        mgen_id = tokenizer.get("<MGEN>")
        cgen_id = tokenizer.get("<CGEN>")
        te_id = tokenizer.get("<TE>")

        gen_pos = np.where((seq == mgen_id) | (seq == cgen_id))[0]
        if len(gen_pos) == 0:
            raise ValueError("生成系列内に <MGEN>/<CGEN> が見つかりません。")

        start = int(gen_pos[0])
        te_pos = np.where(seq == te_id)[0]
        end = int(te_pos[0]) + 1 if len(te_pos) > 0 else len(seq)
        return seq[start:end]

    def _build_pretrained_prompt(
        self,
        tokenizer: Tokenizer,
        past_midi: Optional[str],
        const_midi: Optional[str],
        future_midi: Optional[str],
        meta: GenerateMeta,
        program_names: List[str],
        default_time_signature: Tuple[int, int] = (4, 4),
        force_time_signature: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        task = self._normalize_task(meta.task)

        max_pos = max(32, int(getattr(self.model.args, "position_length", 8192)))
        min_gen_reserve = min(512, max(8, max_pos // 4))
        max_prompt_total = max(8, max_pos - min_gen_reserve)
        detected_context_ts: Optional[Tuple[int, int]] = None

        if task == "MIDI2Meta":
            if const_midi is None:
                raise ValueError("MIDI2Meta では conditions_midi が必須です")
            prompt = np.asarray([tokenizer.get("<EOS>")], dtype=np.int64)
            const_seq, det_ts = self._build_midi_context(
                tokenizer,
                const_midi,
                "<CONST_M>",
                program_names,
                meta.key,
                max_total_tokens=max(0, max_prompt_total - 2),
                default_time_signature=default_time_signature,
                force_time_signature=force_time_signature,
            )
            if det_ts is not None:
                detected_context_ts = det_ts
            return (
                np.concatenate([prompt, const_seq, np.asarray([tokenizer.get("<META>")], dtype=np.int64)]),
                detected_context_ts,
            )

        if task == "MIDI2Chord":
            if const_midi is None:
                raise ValueError("MIDI2Chord では conditions_midi が必須です")

        include_dense = task != "MIDI2Chord"
        include_measure_count = future_midi is None
        prompt = self._build_system_prompt(
            tokenizer,
            meta,
            program_names,
            include_dense=include_dense,
            include_measure_count=include_measure_count,
        )

        use_continue_seed = task == "Meta2MIDI" and bool(meta.ai_continue_mode)
        if use_continue_seed and past_midi is None:
            raise ValueError("ai_continue_mode=True の場合は past_midi が必要です")

        has_past_ctx = past_midi is not None and not meta.ai_continue_mode
        has_future_ctx = future_midi is not None
        has_const_ctx = const_midi is not None
        remaining_blocks = int(has_past_ctx) + int(has_future_ctx) + int(has_const_ctx) + int(use_continue_seed)
        remaining_context = max(0, max_prompt_total - len(prompt) - 1)

        def _alloc_block_budget() -> int:
            nonlocal remaining_blocks
            if remaining_blocks <= 0 or remaining_context <= 0:
                return 0
            fit_blocks = max(1, min(remaining_blocks, max(1, remaining_context // 7)))
            return remaining_context // fit_blocks

        if has_past_ctx:
            b_budget = _alloc_block_budget()
            ctx_arr, det_ts = self._build_midi_context(
                tokenizer,
                past_midi,
                "<PAST_M>",
                program_names,
                meta.key,
                max_total_tokens=b_budget,
                default_time_signature=default_time_signature,
                force_time_signature=force_time_signature,
            )
            if det_ts is not None and detected_context_ts is None:
                detected_context_ts = det_ts
            if ctx_arr.size > 0:
                prompt = np.concatenate([prompt, ctx_arr])
                remaining_context = max(0, remaining_context - int(ctx_arr.size))
            remaining_blocks -= 1

        if has_future_ctx:
            b_budget = _alloc_block_budget()
            ctx_arr, det_ts = self._build_midi_context(
                tokenizer,
                future_midi,
                "<FUTURE_M>",
                program_names,
                meta.key,
                max_total_tokens=b_budget,
                default_time_signature=default_time_signature,
                force_time_signature=force_time_signature,
            )
            if det_ts is not None and detected_context_ts is None:
                detected_context_ts = det_ts
            if ctx_arr.size > 0:
                prompt = np.concatenate([prompt, ctx_arr])
                remaining_context = max(0, remaining_context - int(ctx_arr.size))
            remaining_blocks -= 1

        if has_const_ctx:
            b_budget = _alloc_block_budget()
            ctx_arr, det_ts = self._build_midi_context(
                tokenizer,
                const_midi,
                "<CONST_M>",
                program_names,
                meta.key,
                max_total_tokens=b_budget,
                default_time_signature=default_time_signature,
                force_time_signature=force_time_signature,
            )
            if det_ts is not None and detected_context_ts is None:
                detected_context_ts = det_ts
            if ctx_arr.size > 0:
                prompt = np.concatenate([prompt, ctx_arr])
                remaining_context = max(0, remaining_context - int(ctx_arr.size))
            remaining_blocks -= 1

        start_token = "<CGEN>" if task == "MIDI2Chord" else "<MGEN>"
        prompt = np.concatenate([prompt, np.asarray([tokenizer.get(start_token)], dtype=np.int64)])

        if use_continue_seed:
            node_dict, det_ts = self._load_midi_node_dict(
                tokenizer,
                past_midi,
                program_names,
                meta.key,
                default_time_signature=default_time_signature,
                force_time_signature=force_time_signature,
            )
            if det_ts is not None and detected_context_ts is None:
                detected_context_ts = det_ts
            seed = self._build_ai_continue_seed(
                tokenizer, node_dict, max_seed_tokens=max(2, remaining_context)
            )
            if len(prompt) + len(seed) < max_pos:
                prompt = np.concatenate([prompt, seed])
            remaining_blocks -= 1

        if len(prompt) >= max_pos:
            raise ValueError("コンテキストMIDIのトークン長がモデルの上限を超えています。")

        return prompt, detected_context_ts

    def preprocessing(self, past_midi, const_midi, future_midi, meta: GenerateMeta) -> dict:
        program_names = self.normalize_programs(meta.program, require_non_empty=True)
        task = self._normalize_task(meta.task)
        model_tag = self.meta["tag"]["model"]
        if model_tag == "pretrained":
            tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
        elif model_tag == "generation":
            tokenizer = Tokenizer(omega_converter(TO_TOKEN))
        else:
            raise ValueError(f"Unsupported model tag: {model_tag}")

        normalized_key = self._normalize_key(tokenizer, meta.key)
        eff_density = {p: self._resolve_density(meta, p) for p in program_names}
        eff_measures = meta.genfield_measure if meta.genfield_measure is not None else 8
        self.last_reason = [
            {
                "source": "request",
                "task": task,
                "thinking": False,
                "instruments": program_names,
                "note_density": eff_density,
                "gen_measure_count": self.clamp_measure_count(eff_measures),
                "genre": None,
                "key": normalized_key,
            }
            for _ in range(meta.num_gems)
        ]

        if model_tag == "pretrained":
            if task == "MIDI2Meta":
                active_context_midis = [const_midi]
            else:
                use_continue_seed = task == "Meta2MIDI" and bool(meta.ai_continue_mode)
                active_context_midis = []
                if (past_midi is not None and not meta.ai_continue_mode) or use_continue_seed:
                    active_context_midis.append(past_midi)
                if future_midi is not None:
                    active_context_midis.append(future_midi)
                if const_midi is not None:
                    active_context_midis.append(const_midi)
            effective_time_sig = self.resolve_unified_time_signature(meta, active_context_midis)
            sequences = []
            for _ in range(meta.num_gems):
                seq_arr, _ = self._build_pretrained_prompt(
                    tokenizer,
                    past_midi,
                    const_midi,
                    future_midi,
                    meta,
                    program_names,
                    default_time_signature=effective_time_sig,
                    force_time_signature=effective_time_sig,
                )
                sequences.append(seq_arr)
            return {
                "meta": meta,
                "task": task,
                "sequence": np.stack(sequences, axis=0),
                "tokenizer": tokenizer,
                "effective_time_sig": effective_time_sig,
            }

        if model_tag == "generation":
            effective_time_sig = self.resolve_unified_time_signature(
                meta, [past_midi, future_midi, const_midi]
            )
            device = self.backend.device
            max_pos = max(32, int(getattr(self.model.args, "position_length", 8192)))
            min_gen_reserve = min(512, max(8, max_pos // 4))
            max_prompt_total = max(8, max_pos - min_gen_reserve)
            sequences = []

            for _ in range(meta.num_gems):
                prompt = self._build_system_prompt(
                    tokenizer,
                    meta,
                    program_names,
                    include_dense=True,
                    include_measure_count=(future_midi is None),
                )
                remaining_blocks = int(past_midi is not None) + int(future_midi is not None) + int(const_midi is not None)
                remaining_context = max(0, max_prompt_total - len(prompt) - 1)

                def _alloc_gen_block() -> int:
                    if remaining_blocks <= 0 or remaining_context <= 0:
                        return 0
                    fit_blocks = max(1, min(remaining_blocks, max(1, remaining_context // 7)))
                    return remaining_context // fit_blocks

                if past_midi is not None:
                    ctx_arr, _ = self._build_midi_context(
                        tokenizer,
                        past_midi,
                        "<PAST_M>",
                        program_names,
                        meta.key,
                        max_total_tokens=_alloc_gen_block(),
                        default_time_signature=effective_time_sig,
                        force_time_signature=effective_time_sig,
                    )
                    if ctx_arr.size > 0:
                        prompt = np.concatenate([prompt, ctx_arr])
                        remaining_context = max(0, remaining_context - int(ctx_arr.size))
                    remaining_blocks -= 1
                if future_midi is not None:
                    ctx_arr, _ = self._build_midi_context(
                        tokenizer,
                        future_midi,
                        "<FUTURE_M>",
                        program_names,
                        meta.key,
                        max_total_tokens=_alloc_gen_block(),
                        default_time_signature=effective_time_sig,
                        force_time_signature=effective_time_sig,
                    )
                    if ctx_arr.size > 0:
                        prompt = np.concatenate([prompt, ctx_arr])
                        remaining_context = max(0, remaining_context - int(ctx_arr.size))
                    remaining_blocks -= 1
                if const_midi is not None:
                    ctx_arr, _ = self._build_midi_context(
                        tokenizer,
                        const_midi,
                        "<CONST_M>",
                        program_names,
                        meta.key,
                        max_total_tokens=_alloc_gen_block(),
                        default_time_signature=effective_time_sig,
                        force_time_signature=effective_time_sig,
                    )
                    if ctx_arr.size > 0:
                        prompt = np.concatenate([prompt, ctx_arr])
                        remaining_context = max(0, remaining_context - int(ctx_arr.size))
                    remaining_blocks -= 1

                prompt = np.concatenate([prompt, np.asarray([tokenizer.get("<MGEN>")], dtype=np.int64)])
                if len(prompt) >= max_pos:
                    raise ValueError("コンテキストMIDIのトークン長がモデルの上限を超えています。")
                sequences.append(torch.tensor(prompt, dtype=torch.long, device=device))

            return {
                "meta": meta,
                "task": task,
                "sequence": sequences,
                "tokenizer": tokenizer,
                "effective_time_sig": effective_time_sig,
            }

        raise ValueError(f"Unsupported model tag: {model_tag}")

    def generate(self, **kwargs) -> dict:
        meta: GenerateMeta = kwargs["meta"]
        tokenizer: Tokenizer = kwargs["tokenizer"]
        task: str = kwargs["task"]
        sequence = kwargs["sequence"]
        effective_time_sig: Tuple[int, int] = kwargs.get("effective_time_sig") or self.resolve_time_signature(meta)

        if isinstance(sequence, list):
            src_tensor = pad_sequence(sequence, batch_first=True, padding_value=tokenizer.get("<PAD>"))
        else:
            src_tensor = sequence

        forbidden_ids = (
            self.build_out_of_bar_shift_token_ids(tokenizer, effective_time_sig)
            if task != "MIDI2Meta"
            else []
        )

        full_sequences, (_, generated_parts) = self.model.top_sampling_measure_kv_cache(
            tokenizer=tokenizer,
            src=src_tensor,
            p=meta.p,
            temperature=meta.temperature,
            forbidden_token_ids=forbidden_ids,
            cancel_event=kwargs.get("cancel_event"),
            run_id=kwargs.get("run_id"),
        )
        return {
            "meta": meta,
            "task": task,
            "tokenizer": tokenizer,
            "effective_time_sig": effective_time_sig,
            "full_sequences": full_sequences,
            "generated_parts": generated_parts,
        }

    def postprocessing(self, save_directory: str, **kwargs) -> list:
        meta: GenerateMeta = kwargs["meta"]
        tokenizer: Tokenizer = kwargs["tokenizer"]
        task: str = kwargs["task"]

        tokenizer.mode(TO_MUSIC)
        full_sequences = kwargs["full_sequences"]
        generated_parts = kwargs["generated_parts"]

        time_sig = kwargs.get("effective_time_sig") or self.resolve_time_signature(meta)
        quarters_per_bar = float(time_sig[0]) * (4.0 / float(time_sig[1]))
        outputs = []
        for i in range(len(full_sequences)):
            if task == "MIDI2Chord":
                outputs.append(self._parse_chords(tokenizer, generated_parts[i], meta.tempo, quarters_per_bar=quarters_per_bar))
            elif task == "MIDI2Meta":
                outputs.append(self._parse_metadata(tokenizer, generated_parts[i]))
            else:
                if meta.ai_continue_mode:
                    seq_tokens = self._extract_from_generation_start(tokenizer, full_sequences[i])
                else:
                    seq_tokens = generated_parts[i]

                seq_list = (
                    seq_tokens.detach().cpu().tolist()
                    if hasattr(seq_tokens, "detach")
                    else (seq_tokens.tolist() if hasattr(seq_tokens, "tolist") else list(seq_tokens))
                )
                # Preserve leading `<SME>` tokens so genuine 1-bar rests (`[<INST_*>, <SME>, s_0, ...]`)
                # keep Bar 1 notes at Bar 1 instead of collapsing onto Bar 0 (`[R2-MED-07]`).
                output_path = os.path.join(save_directory, f"output_{i}.mid")
                ct_token_to_midi(
                    tokenizer,
                    seq_list,
                    output_path,
                    tempo=meta.tempo,
                    time_signature=time_sig,
                )
                outputs.append(output_path)

        return outputs

    def _parse_chords(self, tokenizer: Tokenizer, seq, tempo: int, quarters_per_bar: float = 4.0):
        b4 = 60.0 / tempo
        measure = b4 * quarters_per_bar
        b96 = (b4 * 4.0) / 96.0
        max_step = max(0, int(round(quarters_per_bar * 24.0)) - 1)
        measure_start_time = 0.0
        current_time = 0.0
        chords = []
        current_root = None
        current_quality = None
        current_bass = None

        def try_commit():
            nonlocal current_root, current_quality, current_bass
            if current_root is not None:
                qual = current_quality if current_quality and current_quality != "None" else ""
                bass = current_bass if current_bass and current_bass != "None" else ""
                chords.append({"time": round(current_time, 3), "chord": f"{current_root}{qual}{bass}"})
                current_root = None
                current_quality = None
                current_bass = None

        if hasattr(seq, "flatten"):
            seq = seq.flatten()

        for token_id in seq:
            t_id = int(token_id.item()) if hasattr(token_id, "item") else int(token_id)
            token = tokenizer.rev_get(t_id)
            if not token:
                continue
            if token in {"<TE>", "<EOS>", "<TAG_END>"}:
                break
            if token == "<SME>":
                measure_start_time += measure
            elif token.startswith("s_"):
                try_commit()
                beat = min(float(max_step), max(0.0, float(token.split("_")[1])))
                current_time = measure_start_time + beat * b96
            elif token.startswith("CR_"):
                current_root = token.split("_", 1)[1]
            elif token.startswith("CQ_"):
                current_quality = token.split("_", 1)[1]
            elif token.startswith("CB_"):
                current_bass = token.split("_", 1)[1]

        try_commit()
        return chords

    def _parse_metadata(self, tokenizer: Tokenizer, seq):
        meta_dict = {}
        instruments = []
        densities = {}
        current_inst = None

        if hasattr(seq, "flatten"):
            seq = seq.flatten()

        for token_id in seq:
            t_id = int(token_id.item()) if hasattr(token_id, "item") else int(token_id)
            token = tokenizer.rev_get(t_id)
            if not token:
                continue
            if token in {"<TE>", "<TAG_END>"}:
                break
            if token in {"<EOS>", "<SYSTEM>", "<META>", "<MGEN>", "<CGEN>"}:
                continue
            if token.startswith("k_"):
                meta_dict["key"] = token[2:]
            elif token.startswith("<INST_"):
                current_inst = token[6:-1]
                instruments.append(current_inst)
            elif token.startswith("<NOTE_DENSE_"):
                density_val = token[12:-1]
                if current_inst is not None:
                    densities[current_inst] = density_val
            elif token.startswith("<GEN_MEASURE_COUNT_"):
                meta_dict["gen_measure_count"] = token[19:-1]

        if instruments:
            meta_dict["instruments"] = instruments
        if densities:
            meta_dict["note_density"] = densities
        return meta_dict
