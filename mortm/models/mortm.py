"""
MORTM 4.5 / 4.5D / 4.5E Autoregressive Symbolic Music Foundation & SFT Model.
100% parameter-name compatible with MORTM checkpoints (`embedding`, `decoder`, `Wout`),
with zero-sync Contiguous SDPA KV Cache, pre-allocated O(1) token buffers,
2-stage Chain-of-Thought (CoT) temperature sampling, and cross-platform execution
across NVIDIA CUDA, Apple Silicon MPS, and Multi-Core CPUs.
"""

import threading
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mortm.models.modules.backend import HardwareBackendFactory, IHardwareBackend
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.layers import MORTMDecoder, PMA
from mortm.models.modules.lora_layers import merge_lora_weights_inplace
from mortm.models.modules.progress import (
    DefaultObserver,
    GenerationCancelledError,
    LearningProgress,
    _LearnProgress,
)
from mortm.train.tokenizer import Tokenizer


def top_p_sampling(logits: Tensor, p: float = 0.95, temperature: Union[float, Tensor] = 1.0) -> Tensor:
    """
    Vectorized Nucleus (Top-P) Sampling supporting scalar or per-batch-item `[B, 1]` temperature.
    Casts logits to FP32 before temperature scaling to prevent FP16 overflow when temperature is small.
    """
    logits_f32 = logits.float()
    if isinstance(temperature, Tensor):
        temp = torch.clamp(temperature.float(), min=1e-5)
        scaled_logits = logits_f32 / temp
    else:
        if float(temperature) <= 0.0:
            return torch.argmax(logits_f32, dim=-1)
        scaled_logits = logits_f32 / max(float(temperature), 1e-5)

    probs = F.softmax(scaled_logits, dim=-1)

    if 0.0 < p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_probs_to_remove = cumulative_probs > p
        sorted_probs_to_remove[..., 1:] = sorted_probs_to_remove[..., :-1].clone()
        sorted_probs_to_remove[..., 0] = False

        probs_to_keep = sorted_probs.masked_fill(sorted_probs_to_remove, 0.0)
        probs_sum = probs_to_keep.sum(dim=-1, keepdim=True)
        renormalized_probs = probs_to_keep / (probs_sum + 1e-9)

        sampled_next_indices = torch.multinomial(renormalized_probs, num_samples=1)
        sampled_original_indices = torch.gather(sorted_indices, dim=-1, index=sampled_next_indices)
        return sampled_original_indices.squeeze(-1)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.squeeze(-1)


class MORTM(nn.Module, _LearnProgress):
    """
    Main class for the MORTM 4.5 / 4.5D / 4.5E model.
    Preserves exact `embedding`, `decoder`, `Wout`, `softmax` parameter hierarchy.
    """

    def __init__(
        self,
        args: MORTMArgs,
        progress: Optional[LearningProgress] = None,
        backend: Optional[IHardwareBackend] = None,
    ):
        super().__init__()
        self.args = args
        self.backend: IHardwareBackend = backend or HardwareBackendFactory.create("auto")
        self.progress = progress or DefaultObserver(self.backend)

        self.e_layer = args.e_layer
        self.d_layer = args.d_layer
        self.num_heads = args.num_heads
        self.d_model = args.d_model
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout

        self.decoder = MORTMDecoder(args, progress=self.progress)
        self.embedding = nn.Embedding(args.vocab_size, self.d_model, padding_idx=0)
        self.Wout = nn.Linear(self.d_model, args.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    @property
    def embed(self) -> nn.Embedding:
        return self.embedding

    def merge_lora(self) -> int:
        """Folds all LoRA adapters (Attention, Gate, FFN, MoE) into base weights in-place."""
        return merge_lora_weights_inplace(self)

    def reset_kv_cache(self) -> None:
        self.decoder.reset_cache()

    def is_end_point(self, x: Tensor, end_tokens: List[int]) -> bool:
        mask = torch.zeros_like(x, dtype=torch.bool)
        for e in end_tokens:
            if e is not None:
                mask = mask | (x == int(e))
        return bool(mask.any(dim=1).all())

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        is_save_cache: bool = False,
        tgt_mask: Optional[Tensor] = None,
        input_padding_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tensor:
        save_cache = is_save_cache or use_cache
        eff_mask = padding_mask if padding_mask is not None else input_padding_mask

        if x.dim() == 1:
            x = x.unsqueeze(1)

        emb = self.embedding(x)
        target_dtype = self.decoder.layers[0].self_attention.qkv_block.qkv_weight.weight.dtype
        if emb.dtype != target_dtype:
            emb = emb.to(dtype=target_dtype)

        with self.backend.autocast_context():
            out = self.decoder(
                tgt=emb,
                tgt_is_causal=is_causal,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                is_save_cache=save_cache,
                key_padding_mask=eff_mask,
                position_ids=position_ids,
            )

        score = self.Wout(out.to(dtype=self.Wout.weight.dtype)).float()
        return score

    def top_p_sampling(self, logits: Tensor, p: float = 0.9, temperature: Union[float, Tensor] = 1.0) -> Tensor:
        return top_p_sampling(logits, p=p, temperature=temperature)

    @torch.inference_mode()
    def top_sampling_measure_kv_cache(
        self,
        tokenizer: Optional[Tokenizer] = None,
        src: Optional[Union[Tensor, np.ndarray, list]] = None,
        p: float = 0.95,
        temperature: float = 1.0,
        print_log: bool = False,
        max_tokens: Optional[int] = None,
        cot_temperature: Optional[float] = None,
        system_token_id: Optional[int] = None,
        tag_end_token_id: Optional[int] = None,
        end_tokens: Optional[List[int]] = None,
        forbidden_token_ids: Optional[Sequence[int]] = None,
        cancel_event: Optional[threading.Event] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        High-Speed Zero-Sync KV-Cached Autoregressive Generator.
        Supports both `top_sampling_measure_kv_cache(tokenizer, src, ...)` and
        `top_sampling_measure_kv_cache(src=..., tokenizer=..., cot_temperature=...)`.
        Returns `(np_all_tokens, (prompt_list, generated_list))` for 100% MozartAPI compatibility.
        """
        # Handle positional `(src, ...)` if caller passed tensor/array as first argument
        if src is None and not isinstance(tokenizer, Tokenizer):
            src = tokenizer
            tokenizer = None

        if src is None:
            raise ValueError("`src` prompt sequence must be provided.")

        self.eval()
        device = next(self.parameters()).device
        compute_dtype = self.decoder.layers[0].self_attention.qkv_block.qkv_weight.weight.dtype

        pad_id = int(self.embedding.padding_idx) if self.embedding.padding_idx is not None else 0

        if isinstance(src, np.ndarray):
            src_t = torch.as_tensor(src, dtype=torch.long, device=device)
        elif isinstance(src, (list, tuple)):
            if len(src) == 0:
                raise ValueError("`src` prompt sequence must not be empty.")
            if isinstance(src[0], (list, tuple, np.ndarray, Tensor)):
                rows = [torch.as_tensor(r, dtype=torch.long, device=device).reshape(-1) for r in src]
                max_row_len = max((int(r.numel()) for r in rows), default=0)
                if max_row_len <= 0:
                    raise ValueError("コンテキストMIDIの有効トークンが空です。")
                src_t = torch.full((len(rows), max_row_len), pad_id, dtype=torch.long, device=device)
                for b_idx, row_t in enumerate(rows):
                    r_len = int(row_t.numel())
                    if r_len > 0:
                        src_t[b_idx, max_row_len - r_len :] = row_t
            else:
                src_t = torch.tensor(src, dtype=torch.long, device=device)
        else:
            src_t = src.to(device=device, dtype=torch.long)

        if src_t.dim() == 1:
            src_t = src_t.unsqueeze(0)

        batch_size, raw_src_len = src_t.shape
        raw_valid_mask = src_t != pad_id
        has_prompt_pad = not bool(raw_valid_mask.all())

        if has_prompt_pad:
            valid_counts = raw_valid_mask.long().sum(dim=-1)
            max_valid_len = int(valid_counts.max().item()) if batch_size > 0 else 0
            if max_valid_len <= 0:
                raise ValueError("コンテキストMIDIの有効トークンが空です。")
            left_padded = torch.full((batch_size, max_valid_len), pad_id, dtype=torch.long, device=device)
            for b_idx in range(batch_size):
                row_valid = src_t[b_idx][raw_valid_mask[b_idx]]
                v_len = int(row_valid.numel())
                if v_len > 0:
                    left_padded[b_idx, max_valid_len - v_len :] = row_valid
            src_t = left_padded
            has_prompt_pad = not bool((src_t != pad_id).all())

        batch_size, src_len = src_t.shape
        max_pos = max(32, int(self.args.position_length))
        if src_len >= max_pos:
            raise ValueError(
                f"コンテキストMIDIのトークン長 ({src_len}) がモデル上限 ({max_pos}) を超えています。"
            )

        avail_budget = max(1, max_pos - src_len)
        budget = max(1, min(int(max_tokens), avail_budget)) if max_tokens is not None else min(1600, avail_budget)
        total_capacity = src_len + budget + 32

        prompt_padding_mask = src_t != pad_id
        position_ids = (prompt_padding_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        full_valid_mask: Optional[Tensor] = None
        decode_pos: Optional[Tensor] = None
        if has_prompt_pad:
            full_valid_mask = torch.ones((batch_size, src_len + budget), dtype=torch.bool, device=device)
            full_valid_mask[:, :src_len] = prompt_padding_mask
            decode_pos = prompt_padding_mask.long().sum(dim=-1)

        if tokenizer is not None:
            if system_token_id is None:
                system_token_id = tokenizer.get("<SYSTEM>")
            if tag_end_token_id is None:
                tag_end_token_id = tokenizer.get("<TAG_END>")
            if end_tokens is None:
                te_tok = tokenizer.get("<TE>")
                end_tokens = [te_tok] if te_tok is not None else []

        use_two_stage_temp = (
            cot_temperature is not None
            and system_token_id is not None
            and tag_end_token_id is not None
        )
        in_cot = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if use_two_stage_temp:
            in_cot = src_t[:, -1] == int(system_token_id)

        def _select_temperature() -> Union[float, Tensor]:
            if not use_two_stage_temp or not bool(in_cot.any()):
                return temperature
            if bool(in_cot.all()):
                return float(cot_temperature)
            return torch.where(
                in_cot,
                torch.tensor(float(cot_temperature), device=device, dtype=torch.float32),
                torch.tensor(float(temperature), device=device, dtype=torch.float32),
            ).unsqueeze(1)

        end_token_set = [int(t) for t in (end_tokens or []) if t is not None]
        end_tensor = torch.tensor(end_token_set, dtype=torch.long, device=device) if end_token_set else None
        vocab_dim = int(self.Wout.out_features)
        masked_id_set = {pad_id} if 0 <= pad_id < vocab_dim else set()
        if forbidden_token_ids is not None:
            for ft in forbidden_token_ids:
                if ft is not None and 0 <= int(ft) < vocab_dim:
                    masked_id_set.add(int(ft))
        forbidden_tensor = (
            torch.tensor(sorted(masked_id_set), dtype=torch.long, device=device)
            if masked_id_set
            else None
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        finish_step = torch.full((batch_size,), src_len + budget, dtype=torch.long, device=device)

        try:
            self.decoder.allocate_kv_cache(
                batch_size=batch_size,
                max_seqlen=total_capacity,
                device=device,
                dtype=compute_dtype,
            )
            token_buffer = torch.zeros((batch_size, src_len + budget), dtype=torch.long, device=device)
            token_buffer[:, :src_len] = src_t
            cur_len = src_len

            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelledError("Generation cancelled before prefill.")

            # 1. Batched Prefill
            logits = self.forward(
                src_t,
                padding_mask=prompt_padding_mask if has_prompt_pad else None,
                position_ids=position_ids,
                is_causal=True,
                is_save_cache=True,
            )
            next_token_logits = logits[:, -1, :].clone()
            if forbidden_tensor is not None:
                next_token_logits[:, forbidden_tensor] = float("-inf")
            next_tokens = top_p_sampling(next_token_logits, p=p, temperature=_select_temperature())

            token_buffer[:, cur_len] = next_tokens
            cur_len += 1

            if use_two_stage_temp:
                in_cot = torch.where(next_tokens == int(system_token_id), torch.ones_like(in_cot), in_cot)
                in_cot = torch.where(next_tokens == int(tag_end_token_id), torch.zeros_like(in_cot), in_cot)

            if end_tensor is not None:
                newly_done = torch.isin(next_tokens, end_tensor) & (~finished)
                finish_step = torch.where(newly_done, torch.full_like(finish_step, cur_len), finish_step)
                finished |= newly_done

            # 2. Autoregressive Decode Loop
            while cur_len < src_len + budget and not bool(finished.all()):
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelledError("Generation cancelled during token decoding.")

                step_pad_mask = full_valid_mask[:, : cur_len + 1] if full_valid_mask is not None else None
                step_pos_ids = decode_pos.unsqueeze(1) if decode_pos is not None else None

                logits = self.forward(
                    next_tokens.unsqueeze(1),
                    padding_mask=step_pad_mask,
                    position_ids=step_pos_ids,
                    is_causal=False,
                    is_save_cache=True,
                )
                if decode_pos is not None:
                    decode_pos = decode_pos + 1

                next_token_logits = logits[:, -1, :].clone()
                if forbidden_tensor is not None:
                    next_token_logits[:, forbidden_tensor] = float("-inf")
                next_tokens = top_p_sampling(next_token_logits, p=p, temperature=_select_temperature())

                if bool(finished.any()):
                    next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)

                token_buffer[:, cur_len] = next_tokens
                cur_len += 1
                gen_steps = cur_len - src_len
                if gen_steps == 1 or gen_steps % 8 == 0:
                    self.progress.report_generation_step(gen_steps, budget, run_id=run_id)

                if use_two_stage_temp:
                    in_cot = torch.where(next_tokens == int(system_token_id), torch.ones_like(in_cot), in_cot)
                    in_cot = torch.where(next_tokens == int(tag_end_token_id), torch.zeros_like(in_cot), in_cot)

                if end_tensor is not None:
                    newly_done = torch.isin(next_tokens, end_tensor) & (~finished)
                    finish_step = torch.where(newly_done, torch.full_like(finish_step, cur_len), finish_step)
                    finished |= newly_done
        finally:
            self.decoder.reset_cache()

        all_tokens_cpu = token_buffer[:, :cur_len].detach().cpu().numpy()
        mgen_id = tokenizer.get("<MGEN>") if tokenizer is not None else -1
        cgen_id = tokenizer.get("<CGEN>") if tokenizer is not None else -1
        meta_id = tokenizer.get("<META>") if tokenizer is not None else -1

        np_all_tokens: List[np.ndarray] = []
        prompt_list: List[np.ndarray] = []
        generated_list: List[np.ndarray] = []

        for b in range(batch_size):
            end_idx = int(min(int(finish_step[b].item()), cur_len))
            row = all_tokens_cpu[b, :end_idx]
            prompt_part = row[:src_len]
            non_pad_prompt = prompt_part[prompt_part != pad_id]
            prompt_list.append(non_pad_prompt)

            # Locate generation trigger (<MGEN>, <CGEN>, or <META>) within the prompt first
            prompt_gen_indices = np.where(
                (prompt_part == mgen_id) | (prompt_part == cgen_id) | (prompt_part == meta_id)
            )[0]
            if len(prompt_gen_indices) > 0:
                gen_start = int(prompt_gen_indices[-1])
                gen_part = row[gen_start:]
            else:
                # Foundation unconditional (<EOS>) or custom prompt
                gen_part = row[src_len:]

            generated_list.append(gen_part)
            np_all_tokens.append(np.concatenate([non_pad_prompt, row[src_len:]]))

        return np_all_tokens, (prompt_list, generated_list)

    def get_param(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        if not getattr(self.args, "use_moe_decoder", False):
            return total_params, total_params

        first_layer = self.decoder.layers[0]
        moe_module = first_layer.ffn
        if getattr(moe_module, "stacked", False):
            one_expert_params = sum(
                p.numel() // self.args.num_experts
                for name, p in moe_module.named_parameters(recurse=False)
                if name in ("w13", "w2", "b13", "b2")
            )
        elif hasattr(moe_module, "experts"):
            one_expert_params = sum(p.numel() for p in moe_module.experts[0].parameters())
        else:
            return total_params, total_params

        num_inactive = max(0, self.args.num_experts - self.args.topk_experts)
        total_inactive = num_inactive * one_expert_params * self.args.d_layer
        return total_params, total_params - total_inactive


class ClassificationMORTM(MORTM):
    def __init__(self, args: MORTMArgs, class_num: int, progress: Optional[LearningProgress] = None, pma_out_dim: Optional[int] = None):
        super().__init__(args, progress)
        pod = pma_out_dim if pma_out_dim is not None else self.d_model * 2
        self.pma_out_dim = pod
        self.pma = PMA(self.d_model, pod)
        self.classifier = nn.Sequential(
            nn.Linear(pod, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, class_num),
        )
