"""
MORTM 4.5D (Dense) & 4.5E (64-Expert MoE) Foundation & SFT-Gen Model Wrapper (`MORTM45DRapper`).
Implements:
1. Foundation unconditional generation (`tag.model == "foundation"`)
2. SFT-Gen 5-task conditional generation (`meta`, `meta_past`, `meta_future`, `infill`, `inst_comp`)
3. Deterministic Chain-of-Thought (`<thinking>` + `<MGEN> <SYSTEM>` trigger)
4. 2-Stage Temperature Sampling (`cot_temperature` for `<SYSTEM>..<TAG_END>`, `temperature` for MIDI)
5. Structured `last_reason` metadata extraction (`source: "model_cot" | "request" | "foundation"`)
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from mortm.train.custom_token import Genre
from mortm.train.tokenizer import (
    TO_MUSIC,
    TO_TOKEN,
    Tokenizer,
    get_token_converter_pro,
)
from mortm.utils.convert import MIDIConverter
from mortm.utils.de_convert import ct_token_to_midi
from rapper import AbstractModelRapper, GenerateMeta

_SFT_TASK_ALIASES = {
    "meta": "meta",
    "meta2midi": "meta",
    "prompt2midi": "meta",
    "generate": "meta",
    "melodygen": "meta",
    "metapast": "meta_past",
    "meta_past": "meta_past",
    "metafuture": "meta_future",
    "meta_future": "meta_future",
    "infill": "infill",
    "instcomp": "inst_comp",
    "inst_comp": "inst_comp",
}


class MORTM45DRapper(AbstractModelRapper):
    """
    Unified Wrapper for MORTM 4.5D (Dense) and MORTM 4.5E (64-Expert MoE)
    Foundation (`tag.model == "foundation"`) and SFT-Gen (`tag.model == "sft_gen"`) models
    across CUDA, Apple Silicon MPS, and CPU.
    """

    VALID_GENRES = frozenset(Genre.GENRES)

    def _normalize_key(self, tokenizer: Tokenizer, raw_key: Optional[str]) -> Optional[str]:
        if not raw_key:
            return None
        k = raw_key.strip()
        if "major" in k.lower():
            k = f"{k.split()[0]}M"
        elif "minor" in k.lower():
            k = f"{k.split()[0]}m"
        if f"k_{k}" in tokenizer.tokens:
            return k
        return "Unknown"

    def _normalize_genres(self, raw_genre: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        if not raw_genre:
            return None
        raw_list = [raw_genre] if isinstance(raw_genre, str) else list(raw_genre)
        cleaned: List[str] = []
        for item in raw_list:
            if not item:
                continue
            g = str(item).strip().lower()
            if g not in self.VALID_GENRES:
                raise ValueError(
                    f"Unsupported genre '{item}'. Valid genres ({len(self.VALID_GENRES)}): {sorted(self.VALID_GENRES)}"
                )
            if g not in cleaned:
                cleaned.append(g)
        return cleaned if cleaned else None

    def _resolve_task(
        self,
        raw_task: Optional[str],
        past_midi: Optional[str],
        const_midi: Optional[str],
        future_midi: Optional[str],
    ) -> str:
        norm_key = "".join(ch for ch in (raw_task or "meta") if ch.isalnum() or ch == "_").lower()
        if norm_key not in _SFT_TASK_ALIASES:
            raise ValueError(
                f"Unsupported SFT task: '{raw_task}'. Supported tasks: meta, meta_past, meta_future, infill, inst_comp"
            )
        resolved = _SFT_TASK_ALIASES[norm_key]

        # If caller left task at default ("Meta2MIDI" -> "meta") but supplied MIDI files, infer the matching SFT task
        if resolved == "meta" and norm_key in ("meta2midi", "prompt2midi", "generate", "melodygen"):
            if past_midi and future_midi:
                return "infill"
            if const_midi:
                return "inst_comp"
            if past_midi:
                return "meta_past"
            if future_midi:
                return "meta_future"
        return resolved

    def _resolve_density_map(
        self,
        meta: GenerateMeta,
        instruments: List[str],
        default_density: Optional[int],
    ) -> Dict[str, Optional[int]]:
        density_map: Dict[str, Optional[int]] = {inst: default_density for inst in instruments}
        dense = meta.gen_note_dense
        if dense is None:
            return density_map

        if isinstance(dense, int):
            clamped = self.clamp_density(dense)
            for inst in instruments:
                density_map[inst] = clamped
        elif isinstance(dense, dict):
            for inst in instruments:
                matched = None
                for k in (inst, inst.upper(), inst.lower()):
                    if k in dense and dense[k] is not None:
                        matched = self.clamp_density(dense[k])
                        break
                if matched is not None:
                    density_map[inst] = matched
        return density_map

    def _extract_context_midi_tokens(
        self,
        tokenizer: Tokenizer,
        midi_path: str,
        start_measure: int = 0,
        measure_count: Optional[int] = None,
        key_hint: Optional[str] = None,
        default_time_signature: Tuple[int, int] = (4, 4),
        force_time_signature: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[int], List[str], Optional[Tuple[int, int]]]:
        """
        Converts a context MIDI file into framed instrument token sequences:
        `[<INST_PROG>, ..., <ESEQ>, ...]` without chord tokens.
        Correctly handles both 1-measure MIDI files (`len(sme_indices) == 0`) and multi-measure
        MIDI files where Measure 0 notes precede `sme_indices[0]`, and enforces a unified
        bar grid across multi-context / explicit-override inputs (`[R3-HIGH-01]`, `[R4-MED-02]`).
        """
        converter = MIDIConverter(
            tokenizer,
            os.path.dirname(midi_path),
            os.path.basename(midi_path),
            program_list=["PIANO", "SAX"],
            use_midi2seq=True,
            use_midi2seq_with_chord=False,
            key=key_hint or "Unknown",
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

        sme_id = tokenizer.get("<SME>")
        blank_id = tokenizer.get("<BLANK>")
        eseq_id = tokenizer.get("<ESEQ>")
        te_id = tokenizer.get("<TE>")
        s_range = tokenizer.get_length_tuple("s")
        cr_range = tokenizer.get_length_tuple("CR")
        cq_range = tokenizer.get_length_tuple("CQ")
        cb_range = tokenizer.get_length_tuple("CB")

        context_tokens: List[int] = []
        found_programs: List[str] = []

        for prog_name, seq in converter.midi2seq.aya_node.items():
            if len(seq) == 0:
                continue

            # In MIDI2Seq, Measure 0 starts at index 0, and each <SME> marks the start of Measure 1, 2, ...
            # Note: MIDI2Seq stops emitting <SME> after an instrument's final sounding note, so if
            # `start_measure >= total_measures` that instrument is silent in the requested window (`[R5-MED-01]`).
            sme_indices = np.where(seq == sme_id)[0]
            total_measures = len(sme_indices) + 1
            raw_start = max(0, 0 if start_measure is None else int(start_measure))
            if raw_start >= total_measures:
                continue
            s_idx = raw_start

            begin_pos = 0 if s_idx == 0 else int(sme_indices[s_idx - 1]) + 1
            if measure_count is not None:
                if int(measure_count) <= 0:
                    continue
                last_measure_idx = s_idx + int(measure_count) - 1
                if last_measure_idx < len(sme_indices):
                    end_pos = int(sme_indices[last_measure_idx])
                else:
                    end_pos = len(seq)
            else:
                end_pos = len(seq)

            sliced = seq[begin_pos:end_pos]
            clean_slice: List[int] = []
            for t_id in sliced:
                t_val = int(t_id)
                if t_val in (te_id, eseq_id):
                    continue
                if (
                    cr_range[0] <= t_val <= cr_range[1]
                    or cq_range[0] <= t_val <= cq_range[1]
                    or cb_range[0] <= t_val <= cb_range[1]
                ):
                    continue
                clean_slice.append(t_val)

            # In MORTM grammar, <BLANK> only exists inside ['<SME>', '<BLANK>'] for empty bars k >= 1;
            # an empty local Bar 0 after slicing at sme_indices[s_idx - 1] + 1 has no leading <BLANK> ([R4-MED-01]).
            while clean_slice and blank_id is not None and clean_slice[0] == blank_id:
                clean_slice.pop(0)

            clean_slice = AbstractModelRapper.trim_music_tokens_on_event_boundary(
                tokenizer, clean_slice, len(clean_slice), keep_tail=False
            )
            has_note = s_range is not None and any(
                s_range[0] <= t <= s_range[1] for t in clean_slice
            )
            inst_tok = tokenizer.get(f"<INST_{prog_name}>")
            if inst_tok is not None and len(clean_slice) > 0 and has_note:
                context_tokens.append(inst_tok)
                context_tokens.extend(clean_slice)
                context_tokens.append(eseq_id)
                found_programs.append(prog_name)

        return context_tokens, found_programs, detected_ts

    @staticmethod
    def _trim_framed_context_tokens(
        tokenizer: Tokenizer,
        tokens: List[int],
        max_budget: int,
        *,
        keep_tail: bool,
    ) -> List[int]:
        """
        Caps framed `[<INST_...>, ..., <ESEQ>]` context token blocks so adversarial or multi-hundred-bar
        context MIDIs never exceed `self.model.args.position_length`, while strictly preserving balanced
        `<INST_*>` ... `<ESEQ>` framing even under ultra-tight multi-track budgets (`[R3-MED-06]`).
        """
        if max_budget < 3:
            return []

        eseq_id = tokenizer.get("<ESEQ>")
        inst_ids = {
            t for t in (tokenizer.get("<INST_PIANO>"), tokenizer.get("<INST_SAX>")) if t is not None
        }

        frames: List[Tuple[int, List[int]]] = []
        idx = 0
        n = len(tokens)
        while idx < n:
            if tokens[idx] in inst_ids:
                inst_tok = tokens[idx]
                idx += 1
                inner: List[int] = []
                while idx < n and tokens[idx] != eseq_id and tokens[idx] not in inst_ids:
                    inner.append(tokens[idx])
                    idx += 1
                if idx < n and tokens[idx] == eseq_id:
                    idx += 1
                frames.append((inst_tok, inner))
            else:
                idx += 1

        if not frames or eseq_id is None:
            return []

        trimmed: List[int] = []
        remaining_frames = len(frames)
        for inst_tok, inner in frames:
            remaining_budget = max_budget - len(trimmed)
            if remaining_budget < 3:
                break
            fit_frames = max(1, min(remaining_frames, remaining_budget // 3))
            frame_budget = max(3, remaining_budget // fit_frames)
            inner_budget = frame_budget - 2
            trimmed_inner = AbstractModelRapper.trim_music_tokens_on_event_boundary(
                tokenizer, inner, inner_budget, keep_tail=keep_tail
            )
            if not trimmed_inner:
                remaining_frames -= 1
                continue
            if len(trimmed_inner) + 2 <= remaining_budget:
                trimmed.append(inst_tok)
                trimmed.extend(trimmed_inner)
                trimmed.append(int(eseq_id))
            remaining_frames -= 1
        return trimmed

    def _parse_cot_reason(
        self,
        tokenizer: Tokenizer,
        cot_token_ids: Sequence[int],
        task_name: str,
        fallback_insts: List[str],
    ) -> Dict[str, Any]:
        """
        Parses the `<SYSTEM> ... <TAG_END>` CoT token sequence emitted by the model
        into a structured dictionary for `X-Generation-Reason`.
        """
        instruments: List[str] = []
        note_density: Dict[str, Optional[int]] = {}
        gen_measure_count: Optional[int] = None
        key_val: Optional[str] = None
        genres: List[str] = []
        last_inst: Optional[str] = None

        for tid in cot_token_ids:
            tok = tokenizer.rev_get(int(tid))
            if not tok:
                continue
            if tok.startswith("<INST_") and tok.endswith(">"):
                inst_name = tok[len("<INST_") : -1]
                if inst_name not in instruments:
                    instruments.append(inst_name)
                note_density.setdefault(inst_name, None)
                last_inst = inst_name
            elif tok.startswith("<NOTE_DENSE_") and tok.endswith(">"):
                try:
                    d_val = int(tok[len("<NOTE_DENSE_") : -1])
                    if last_inst is not None:
                        note_density[last_inst] = d_val
                except ValueError:
                    pass
            elif tok.startswith("<GEN_MEASURE_COUNT_") and tok.endswith(">"):
                try:
                    gen_measure_count = int(tok[len("<GEN_MEASURE_COUNT_") : -1])
                except ValueError:
                    pass
            elif tok.startswith("k_"):
                key_val = tok[2:]
            elif tok.startswith("<GENRE_") and tok.endswith(">"):
                g_name = tok[len("<GENRE_") : -1]
                if g_name not in genres:
                    genres.append(g_name)

        if not instruments:
            instruments = list(fallback_insts)
            for inst_name in instruments:
                note_density.setdefault(inst_name, None)

        return {
            "source": "model_cot",
            "task": task_name,
            "thinking": True,
            "instruments": instruments,
            "note_density": note_density,
            "gen_measure_count": gen_measure_count,
            "genre": genres if genres else None,
            "key": key_val,
        }

    def preprocessing(
        self,
        past_midi: Optional[str],
        const_midi: Optional[str],
        future_midi: Optional[str],
        meta: GenerateMeta,
    ) -> dict:
        requested_ts = self.resolve_time_signature(meta)
        effective_time_sig = requested_ts
        tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
        model_tag = self.meta.get("tag", {}).get("model", "sft_gen")

        # 1. Foundation unconditional mode (<EOS>)
        if model_tag == "foundation":
            eos_id = tokenizer.get("<EOS>")
            sequences = [np.asarray([eos_id], dtype=np.int64) for _ in range(meta.num_gems)]
            self.last_reason = [
                {
                    "source": "foundation",
                    "task": "unconditional",
                    "thinking": False,
                    "instruments": None,
                    "note_density": None,
                    "gen_measure_count": None,
                    "genre": None,
                    "key": None,
                }
                for _ in range(meta.num_gems)
            ]
            return {
                "meta": meta,
                "task": "unconditional",
                "model_tag": "foundation",
                "sequence": np.stack(sequences, axis=0),
                "tokenizer": tokenizer,
                "use_cot": False,
                "target_insts": ["PIANO"],
                "effective_time_sig": effective_time_sig,
            }

        # 2. SFT-Gen 5-task conditional mode
        task_str = self._resolve_task(meta.task, past_midi, const_midi, future_midi)

        eff_past_midi = past_midi
        eff_future_midi = future_midi if future_midi is not None else (past_midi if task_str == "meta_future" else None)
        eff_const_midi = const_midi if const_midi is not None else (past_midi if task_str == "inst_comp" else None)

        if task_str == "meta_past" and eff_past_midi is None:
            raise ValueError("Task 'meta_past' requires 'past_midi'.")
        if task_str == "meta_future" and eff_future_midi is None:
            raise ValueError("Task 'meta_future' requires 'future_midi' (or 'past_midi').")
        if task_str == "infill" and (eff_past_midi is None or eff_future_midi is None):
            raise ValueError("Task 'infill' requires both 'past_midi' and 'future_midi'.")
        if task_str == "inst_comp" and eff_const_midi is None:
            raise ValueError("Task 'inst_comp' requires 'conditions_midi' (or 'past_midi').")

        active_context_midis: List[Optional[str]] = []
        if task_str in ("meta_past", "infill") and eff_past_midi is not None:
            active_context_midis.append(eff_past_midi)
        if task_str in ("meta_future", "infill") and eff_future_midi is not None:
            active_context_midis.append(eff_future_midi)
        if task_str == "inst_comp" and eff_const_midi is not None:
            active_context_midis.append(eff_const_midi)

        effective_time_sig = self.resolve_unified_time_signature(meta, active_context_midis)

        past_tokens: List[int] = []
        future_tokens: List[int] = []
        const_tokens: List[int] = []
        context_progs: List[str] = []

        if task_str in ("meta_past", "infill") and eff_past_midi is not None:
            past_tokens, context_progs, _ = self._extract_context_midi_tokens(
                tokenizer,
                eff_past_midi,
                start_measure=meta.start_measure or 0,
                measure_count=meta.measure_count,
                key_hint=meta.key,
                default_time_signature=effective_time_sig,
                force_time_signature=effective_time_sig,
            )

        if task_str in ("meta_future", "infill") and eff_future_midi is not None:
            f_start = meta.future_start_measure if future_midi is not None else (meta.start_measure or 0)
            f_count = meta.future_measure_count if future_midi is not None else meta.measure_count
            future_tokens, fut_progs, _ = self._extract_context_midi_tokens(
                tokenizer,
                eff_future_midi,
                start_measure=f_start or 0,
                measure_count=f_count,
                key_hint=meta.key,
                default_time_signature=effective_time_sig,
                force_time_signature=effective_time_sig,
            )
            if not context_progs:
                context_progs = fut_progs

        if task_str == "inst_comp" and eff_const_midi is not None:
            const_tokens, context_progs, _ = self._extract_context_midi_tokens(
                tokenizer,
                eff_const_midi,
                start_measure=meta.start_measure or 0,
                measure_count=meta.measure_count,
                key_hint=meta.key,
                default_time_signature=effective_time_sig,
                force_time_signature=effective_time_sig,
            )

        def _reconcile_task_for_context(
            curr_task: str,
            p_toks: List[int],
            f_toks: List[int],
            c_toks: List[int],
        ) -> str:
            if curr_task == "infill":
                if p_toks and f_toks:
                    return "infill"
                if p_toks:
                    return "meta_past"
                if f_toks:
                    return "meta_future"
                return "meta"
            if curr_task == "meta_past" and not p_toks:
                return "meta"
            if curr_task == "meta_future" and not f_toks:
                return "meta"
            if curr_task == "inst_comp" and not c_toks:
                return "meta"
            return curr_task

        # Reconcile `task_str` when silent / out-of-range context slices produce `[]` (`[R6-MED-02]`, `[R7-LOW-02]`)
        # so zero-context runs never emit an OOD `<thinking>` CoT prompt without conditioning blocks,
        # while preserving `orig_is_inst_comp` so complementary target instrument selection never inverts.
        orig_is_inst_comp = task_str == "inst_comp"
        task_str = _reconcile_task_for_context(task_str, past_tokens, future_tokens, const_tokens)

        requested_insts = self.normalize_programs(meta.program, require_non_empty=False)
        eff_genres = self._normalize_genres(meta.genre)
        tag_end_id = tokenizer.get("<TAG_END>")
        system_id = tokenizer.get("<SYSTEM>")
        mgen_id = tokenizer.get("<MGEN>")

        def _build_header_for_task(resolved_task: str):
            meta_only = resolved_task == "meta"
            if requested_insts:
                t_insts = requested_insts
            elif (resolved_task == "inst_comp" or orig_is_inst_comp) and context_progs:
                t_insts = (
                    ["SAX"]
                    if "PIANO" in context_progs and "SAX" not in context_progs
                    else ["PIANO"]
                )
            elif context_progs:
                t_insts = context_progs
            else:
                t_insts = ["PIANO"]

            d_map = self._resolve_density_map(
                meta,
                t_insts,
                default_density=4 if meta_only else None,
            )
            e_mc = (
                self.clamp_measure_count(meta.genfield_measure)
                if meta.genfield_measure is not None
                else (8 if meta_only else None)
            )
            e_key = (
                self._normalize_key(tokenizer, meta.key)
                if meta.key is not None
                else ("Unknown" if meta_only else None)
            )
            cot_flag = bool(meta.thinking) and not meta_only

            hdr: List[int] = [
                tokenizer.get("<EOS>"),
                system_id,
            ]
            for inst_name in t_insts:
                inst_id = tokenizer.get(f"<INST_{inst_name}>")
                if inst_id is not None:
                    hdr.append(inst_id)
                d_val = d_map.get(inst_name)
                if d_val is not None:
                    d_id = tokenizer.get(f"<NOTE_DENSE_{d_val}>")
                    if d_id is not None:
                        hdr.append(d_id)

            if e_mc is not None:
                mc_id = tokenizer.get(f"<GEN_MEASURE_COUNT_{e_mc}>")
                if mc_id is not None:
                    hdr.append(mc_id)

            if eff_genres:
                for g_name in eff_genres:
                    g_id = tokenizer.get(f"<GENRE_{g_name}>")
                    if g_id is not None:
                        hdr.append(g_id)

            if e_key is not None:
                k_id = tokenizer.get(f"k_{e_key}")
                if k_id is not None:
                    hdr.append(k_id)

            if cot_flag:
                thinking_id = tokenizer.get("<thinking>")
                if thinking_id is not None:
                    hdr.append(thinking_id)

            hdr.append(tag_end_id)
            return hdr, t_insts, d_map, e_mc, e_key, cot_flag

        prompt, target_insts, density_map, eff_measure_count, eff_key, use_cot = (
            _build_header_for_task(task_str)
        )

        # Bound context blocks so total prompt stays comfortably within model position_length ([CRIT-03])
        max_pos = max(32, int(getattr(self.model.args, "position_length", 8192)))
        gen_reserve = min(512, max(8, max_pos // 4))
        active_blocks = sum(
            1
            for flag, blk in (
                (task_str in ("meta_past", "infill"), past_tokens),
                (task_str in ("meta_future", "infill"), future_tokens),
                (task_str == "inst_comp", const_tokens),
            )
            if flag and blk
        )
        if active_blocks > 0:
            fixed_overhead = len(prompt) + (2 * active_blocks) + (2 if use_cot else 1)
            avail_context = max(0, max_pos - gen_reserve - fixed_overhead)
            per_block_budget = avail_context // active_blocks
            if task_str in ("meta_past", "infill") and past_tokens:
                past_tokens = self._trim_framed_context_tokens(
                    tokenizer, past_tokens, per_block_budget, keep_tail=True
                )
            if task_str in ("meta_future", "infill") and future_tokens:
                future_tokens = self._trim_framed_context_tokens(
                    tokenizer, future_tokens, per_block_budget, keep_tail=False
                )
            if task_str == "inst_comp" and const_tokens:
                const_tokens = self._trim_framed_context_tokens(
                    tokenizer, const_tokens, per_block_budget, keep_tail=False
                )
            post_trim_task = _reconcile_task_for_context(
                task_str, past_tokens, future_tokens, const_tokens
            )
            if post_trim_task != task_str:
                task_str = post_trim_task
                prompt, target_insts, density_map, eff_measure_count, eff_key, use_cot = (
                    _build_header_for_task(task_str)
                )

        # Append Context Blocks
        if task_str in ("meta_past", "infill") and past_tokens:
            prompt.append(tokenizer.get("<PAST_M>"))
            prompt.extend(past_tokens)
            prompt.append(tag_end_id)

        if task_str in ("meta_future", "infill") and future_tokens:
            prompt.append(tokenizer.get("<FUTURE_M>"))
            prompt.extend(future_tokens)
            prompt.append(tag_end_id)

        if task_str == "inst_comp" and const_tokens:
            prompt.append(tokenizer.get("<CONST_M>"))
            prompt.extend(const_tokens)
            prompt.append(tag_end_id)

        # Append Generation Trigger <MGEN> (+ deterministic <SYSTEM> when CoT is active)
        prompt.append(mgen_id)
        if use_cot:
            prompt.append(system_id)

        if len(prompt) >= max_pos:
            raise ValueError("コンテキストMIDIのトークン長がモデルの上限を超えています。")

        prompt_arr = np.asarray(prompt, dtype=np.int64)
        sequences = [prompt_arr.copy() for _ in range(meta.num_gems)]

        self.last_reason = [
            {
                "source": "request",
                "task": task_str,
                "thinking": use_cot,
                "instruments": target_insts,
                "note_density": density_map,
                "gen_measure_count": eff_measure_count,
                "genre": eff_genres,
                "key": eff_key,
            }
            for _ in range(meta.num_gems)
        ]

        return {
            "meta": meta,
            "task": task_str,
            "model_tag": "sft_gen",
            "sequence": np.stack(sequences, axis=0),
            "tokenizer": tokenizer,
            "use_cot": use_cot,
            "target_insts": target_insts,
            "effective_time_sig": effective_time_sig,
        }

    def generate(self, **kwargs) -> dict:
        meta: GenerateMeta = kwargs["meta"]
        tokenizer: Tokenizer = kwargs["tokenizer"]
        task: str = kwargs["task"]
        model_tag: str = kwargs["model_tag"]
        sequence = kwargs["sequence"]
        use_cot: bool = kwargs.get("use_cot", False)
        target_insts: List[str] = kwargs.get("target_insts", ["PIANO"])
        effective_time_sig: Tuple[int, int] = kwargs.get("effective_time_sig") or self.resolve_time_signature(meta)

        system_id = tokenizer.get("<SYSTEM>")
        tag_end_id = tokenizer.get("<TAG_END>")
        te_id = tokenizer.get("<TE>")
        eseq_id = tokenizer.get("<ESEQ>")

        end_tokens = [te_id, eseq_id] if model_tag == "foundation" else [te_id]
        forbidden_ids = self.build_out_of_bar_shift_token_ids(tokenizer, effective_time_sig)

        full_sequences, (prompt_parts, generated_parts) = self.model.top_sampling_measure_kv_cache(
            tokenizer=tokenizer,
            src=sequence,
            p=meta.p,
            temperature=meta.temperature,
            cot_temperature=meta.cot_temperature if use_cot else None,
            system_token_id=system_id if use_cot else None,
            tag_end_token_id=tag_end_id if use_cot else None,
            end_tokens=end_tokens,
            forbidden_token_ids=forbidden_ids,
            cancel_event=kwargs.get("cancel_event"),
            run_id=kwargs.get("run_id"),
        )

        return {
            "meta": meta,
            "task": task,
            "model_tag": model_tag,
            "tokenizer": tokenizer,
            "use_cot": use_cot,
            "target_insts": target_insts,
            "effective_time_sig": effective_time_sig,
            "full_sequences": full_sequences,
            "prompt_parts": prompt_parts,
            "generated_parts": generated_parts,
        }

    def postprocessing(self, save_directory: str, **kwargs) -> List[str]:
        meta: GenerateMeta = kwargs["meta"]
        tokenizer: Tokenizer = kwargs["tokenizer"]
        task: str = kwargs["task"]
        model_tag: str = kwargs["model_tag"]
        use_cot: bool = kwargs.get("use_cot", False)
        target_insts: List[str] = kwargs.get("target_insts", ["PIANO"])
        full_sequences = kwargs["full_sequences"]
        prompt_parts = kwargs.get("prompt_parts", [])

        tokenizer.mode(TO_MUSIC)

        system_id = tokenizer.get("<SYSTEM>")
        tag_end_id = tokenizer.get("<TAG_END>")
        mgen_id = tokenizer.get("<MGEN>")
        te_id = tokenizer.get("<TE>")
        eos_id = tokenizer.get("<EOS>")

        time_sig = kwargs.get("effective_time_sig") or self.resolve_time_signature(meta)
        outputs: List[str] = []
        updated_reasons: List[Dict[str, Any]] = []

        for i in range(len(full_sequences)):
            seq_arr = np.asarray(full_sequences[i], dtype=np.int64).flatten()

            if model_tag == "foundation":
                music_tokens = seq_arr[1:] if (len(seq_arr) > 0 and seq_arr[0] == eos_id) else seq_arr
                output_path = os.path.join(save_directory, f"output_{i}.mid")
                ct_token_to_midi(
                    tokenizer,
                    music_tokens,
                    output_path,
                    tempo=meta.tempo,
                    time_signature=time_sig,
                )
                outputs.append(output_path)
                continue

            # SFT-Gen: locate the prompt's generation trigger <MGEN> (within prompt_parts first)
            prompt_len = len(prompt_parts[i]) if i < len(prompt_parts) else len(seq_arr)
            prompt_mgen = np.where(seq_arr[:prompt_len] == mgen_id)[0]
            if len(prompt_mgen) > 0:
                after_mgen = seq_arr[int(prompt_mgen[-1]) + 1 :].tolist()
            else:
                all_mgen = np.where(seq_arr == mgen_id)[0]
                if len(all_mgen) > 0:
                    after_mgen = seq_arr[int(all_mgen[0]) + 1 :].tolist()
                else:
                    after_mgen = np.asarray(kwargs["generated_parts"][i], dtype=np.int64).flatten().tolist()

            if te_id in after_mgen:
                after_mgen = after_mgen[: after_mgen.index(te_id)]

            music_tokens_list = after_mgen
            spontaneous_cot = (
                bool(meta.thinking)
                and task != "meta"
                and bool(after_mgen)
                and after_mgen[0] == system_id
                and tag_end_id in after_mgen
            )
            if use_cot or spontaneous_cot:
                if tag_end_id in after_mgen:
                    cot_end_idx = after_mgen.index(tag_end_id)
                    cot_slice = after_mgen[: cot_end_idx + 1]
                    music_tokens_list = after_mgen[cot_end_idx + 1 :]
                else:
                    cot_slice = after_mgen[: min(8, len(after_mgen))]
                    music_tokens_list = after_mgen[1:] if len(after_mgen) > 1 else after_mgen

                updated_reasons.append(
                    self._parse_cot_reason(tokenizer, cot_slice, task_name=task, fallback_insts=target_insts)
                )
            elif self.last_reason and i < len(self.last_reason):
                updated_reasons.append(self.last_reason[i])

            output_path = os.path.join(save_directory, f"output_{i}.mid")
            ct_token_to_midi(
                tokenizer,
                music_tokens_list,
                output_path,
                tempo=meta.tempo,
                time_signature=time_sig,
            )
            outputs.append(output_path)

        if updated_reasons:
            self.last_reason = updated_reasons

        return outputs
