"""
MORTM 4.5 / 4.5D / 4.5E Decoder, MoE, Gate, and Expert Layers.
100% state_dict-compatible with MORTM checkpoints (`self_attention`, `norm1`, `norm3`,
`gate.gate_proj`, `shared_experts`, `w13` `[E, D, 2*FF]`, `w2` `[E, FF, D]`),
with built-in LoRA weight folding and zero-sync `_routed_bmm` MoE decoding.
"""

import math
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import LayerNorm

from mortm.models.modules.attention import FlashCrossAttentionM, FlashSelfAttentionM, flash_attn_varlen_kvpacked_func
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.lora_layers import LoRALinear


class NormTanh(nn.Module):
    def __init__(self, normalized_shape: int, alpha_init_value: float = 0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = torch.tanh(self.alpha * x)
        x = x * self.weight + self.bias
        return x.to(dtype=dtype)


class FFN(nn.Module):
    def __init__(self, d_model: int, ff_d: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_d)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_d, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout1(F.relu(self.linear1(x))))


class Expert(nn.Module):
    def __init__(self, args: MORTMArgs):
        super().__init__()
        if not args.use_ffn_lora:
            self.w1 = nn.Linear(args.d_model, args.dim_feedforward, bias=args.use_bias)
            self.w2 = nn.Linear(args.dim_feedforward, args.d_model, bias=args.use_bias)
            self.w3 = nn.Linear(args.d_model, args.dim_feedforward, bias=args.use_bias)
        else:
            self.w1 = LoRALinear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w2 = LoRALinear(args.dim_feedforward, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w3 = LoRALinear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SharedExpert(nn.Module):
    def __init__(self, args: MORTMArgs):
        super().__init__()
        use_shared_lora = getattr(args, "use_shared_expert_lora", None)
        if use_shared_lora is None:
            use_shared_lora = getattr(args, "use_shared_ffn_lora", args.use_ffn_lora)
        if not use_shared_lora:
            self.w1 = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
            self.w2 = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
            self.w3 = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
        else:
            self.w1 = LoRALinear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w2 = LoRALinear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w3 = LoRALinear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Sigmoid / Softmax-gated MoE router with auxiliary-loss-free load balancing.
    Preserves exact `_FP32_BUFFERS` and buffer names for 100% checkpoint compatibility.
    """

    _FP32_BUFFERS = (
        "routing_bias",
        "ema_counts",
        "ema_counts_sq",
        "ema_counts_long",
        "pending_counts",
        "diag_gap",
        "diag_distortion",
        "diag_aff_loss",
    )

    def _apply(self, *args, **kwargs):
        mod = super()._apply(*args, **kwargs)
        for name in Gate._FP32_BUFFERS:
            buf = mod._buffers.get(name)
            if buf is not None and buf.is_floating_point() and buf.dtype is not torch.float32:
                mod._buffers[name] = buf.float()
        return mod

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in Gate._FP32_BUFFERS:
            k = prefix + name
            if k in state_dict and state_dict[k].is_floating_point():
                state_dict[k] = state_dict[k].float()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(self, args: MORTMArgs, layer_id: int, route_scale: float = 1.0):
        super().__init__()
        n_exp = args.num_experts
        self.topk = args.topk_experts
        self.n_exp = n_exp
        self.score_func = args.score_type
        self.layer_id = layer_id
        self.route_scale = route_scale

        self.ema_decay = getattr(args, "ema_decay", 0.97)
        self.long_decay = getattr(args, "ema_decay_long", 0.999)
        self.manual_step = getattr(args, "gate_manual_step", False)
        self.bias_gain_ref = getattr(args, "bias_gain_ref", 1.2)
        self.update_every = getattr(args, "bias_update_every", 1)

        if getattr(args, "use_gate_lora", False):
            self.gate_proj = LoRALinear(
                args.d_model,
                n_exp,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                bias=False,
            )
        else:
            self.gate_proj = nn.Linear(args.d_model, n_exp, bias=False)

        self.register_buffer("routing_bias", torch.zeros(n_exp, dtype=torch.float32))
        self.register_buffer("ema_counts", torch.zeros(n_exp, dtype=torch.float32))
        self.register_buffer("ema_init", torch.tensor(False))
        self.register_buffer("bias_enabled", torch.tensor(bool(args.use_gate_bias)))

        self.register_buffer("pending_counts", torch.zeros(n_exp, dtype=torch.float32), persistent=False)
        self.register_buffer("ema_counts_sq", torch.zeros(n_exp, dtype=torch.float32))
        self.register_buffer("ema_counts_long", torch.zeros(n_exp, dtype=torch.float32))
        self.register_buffer("zero_streak", torch.zeros(n_exp, dtype=torch.long))
        self.register_buffer("cv_alert", torch.tensor(False))
        self.register_buffer("dead_alert", torch.tensor(False))
        self.register_buffer("dead_clear_streak", torch.zeros((), dtype=torch.long))
        for k in ("diag_gap", "diag_distortion", "diag_aff_loss"):
            self.register_buffer(k, torch.tensor(float("nan")))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.gate_proj(x).float()
        scores = logits.softmax(dim=-1) if self.score_func == "softmax" else logits.sigmoid()

        routing_scores = scores + self.routing_bias if self.bias_enabled else scores
        _, indices = torch.topk(routing_scores, self.topk, dim=-1)

        weights = scores.gather(dim=-1, index=indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        weights = weights * self.route_scale
        return weights.to(dtype=x.dtype), indices


class _DispatchGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, token_ids: Tensor, inv: Tensor, k: int):
        ctx.save_for_backward(inv)
        ctx.k = k
        ctx.n_tok = x.size(0)
        return x.index_select(0, token_ids)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (inv,) = ctx.saved_tensors
        g = grad_out.index_select(0, inv).view(ctx.n_tok, ctx.k, -1).sum(dim=1)
        return g, None, None, None


class MoE(nn.Module):
    """
    Routed MoE Layer with Stacked Expert Weights:
      `w13`: `[E, D, 2 * FF]`
      `w2` : `[E, FF, D]`

    Features a 3-tier execution engine:
    1. `_routed_bmm`: Batched `torch.bmm` for autoregressive decoding (`T * k <= 64`)
       across CUDA, Apple Silicon MPS, and CPU — ZERO `.cpu().tolist()` syncs and ZERO Python loops!
    2. `_routed_grouped`: `torch._grouped_mm` for large prefill batches on CUDA BF16/FP16.
    3. `_routed_loop`: Fallback for large non-CUDA prefill batches.
    """

    _GROUPED_DTYPES = (torch.bfloat16, torch.float16)

    def __init__(self, args: MORTMArgs, layer_id: int = 0, route_scale: float = 1.0):
        super().__init__()
        self.dim = args.d_model
        self.d_ff = args.dim_feedforward
        self.n_routed_experts = args.num_experts
        self.n_activated_experts = args.topk_experts
        self.gate = Gate(args, layer_id, route_scale=route_scale)
        self.shared_experts = SharedExpert(args)

        self.use_lora = bool(args.use_ffn_lora) and not bool(getattr(args, "lora_merged", False))
        self.use_bias = bool(args.use_bias)
        self.grouped_min_experts = getattr(args, "moe_grouped_min_experts", 16)

        if self.use_lora:
            self.experts = nn.ModuleList([Expert(args) for _ in range(self.n_routed_experts)])
            self.stacked = False
            return

        self.stacked = True
        E, D, FF = self.n_routed_experts, self.dim, self.d_ff
        self.w13 = nn.Parameter(torch.empty(E, D, 2 * FF))
        self.w2 = nn.Parameter(torch.empty(E, FF, D))
        if self.use_bias:
            self.b13 = nn.Parameter(torch.empty(E, 2 * FF))
            self.b2 = nn.Parameter(torch.empty(E, D))
        self._reset_expert_parameters()

    @torch.no_grad()
    def _reset_expert_parameters(self) -> None:
        bw = 1.0 / math.sqrt(self.dim)
        bo = 1.0 / math.sqrt(self.d_ff)
        self.w13.uniform_(-bw, bw)
        self.w2.uniform_(-bo, bo)
        if self.use_bias:
            self.b13.uniform_(-bw, bw)
            self.b2.uniform_(-bo, bo)

    @torch.no_grad()
    def merge_lora_inplace(self) -> bool:
        """
        If `use_ffn_lora=True` (`stacked=False`), folds each `Expert`'s LoRA weights
        and stacks them into contiguous `w13` (`[E, D, 2*FF]`) and `w2` (`[E, FF, D]`)
        tensors so decoding can use the zero-sync `_routed_bmm` fast path!
        """
        if self.stacked or not hasattr(self, "experts"):
            return False

        for exp in self.experts:
            for sub in (exp.w1, exp.w2, exp.w3):
                if hasattr(sub, "merge_lora_inplace"):
                    sub.merge_lora_inplace()

        E, D, FF = self.n_routed_experts, self.dim, self.d_ff
        ref_param = self.experts[0].w1.weight
        w13_stack = torch.empty((E, D, 2 * FF), dtype=ref_param.dtype, device=ref_param.device)
        w2_stack = torch.empty((E, FF, D), dtype=ref_param.dtype, device=ref_param.device)
        if self.use_bias:
            b13_stack = torch.empty((E, 2 * FF), dtype=ref_param.dtype, device=ref_param.device)
            b2_stack = torch.empty((E, D), dtype=ref_param.dtype, device=ref_param.device)

        for idx, exp in enumerate(self.experts):
            # nn.Linear weight is [out, in], so transpose to [in, out]
            w13_stack[idx, :, :FF] = exp.w1.weight.t()
            w13_stack[idx, :, FF:] = exp.w3.weight.t()
            w2_stack[idx] = exp.w2.weight.t()
            if self.use_bias and exp.w1.bias is not None:
                b13_stack[idx, :FF] = exp.w1.bias
                b13_stack[idx, FF:] = exp.w3.bias
                b2_stack[idx] = exp.w2.bias

        self.register_parameter("w13", nn.Parameter(w13_stack, requires_grad=False))
        self.register_parameter("w2", nn.Parameter(w2_stack, requires_grad=False))
        if self.use_bias:
            self.register_parameter("b13", nn.Parameter(b13_stack, requires_grad=False))
            self.register_parameter("b2", nn.Parameter(b2_stack, requires_grad=False))
        del self.experts
        self.stacked = True
        return True

    def _use_grouped(self, x: Tensor) -> bool:
        return (
            self.stacked
            and self.n_routed_experts >= self.grouped_min_experts
            and x.is_cuda
            and x.dtype in self._GROUPED_DTYPES
            and hasattr(torch, "_grouped_mm")
        )

    def _routed_bmm(self, x_flat: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        """
        Zero-sync, zero-loop vectorized batched matmul (`torch.bmm`) path for
        autoregressive decoding and small batches (`T * k <= 64`).
        `w13` is `[E, D, 2*FF]` and `w2` is `[E, FF, D]`.
        """
        T, D = x_flat.shape
        k = self.n_activated_experts
        N = T * k

        flat_indices = indices.reshape(-1)  # [N]
        x_rep = x_flat.unsqueeze(1).expand(T, k, D).reshape(N, 1, D)

        w13_sel = self.w13.index_select(0, flat_indices).to(dtype=x_flat.dtype)  # [N, D, 2*FF]
        h = torch.bmm(x_rep, w13_sel)  # [N, 1, 2*FF]
        if self.use_bias:
            h = h + self.b13.index_select(0, flat_indices).unsqueeze(1).to(dtype=h.dtype)

        a, b = h.chunk(2, dim=-1)
        hidden = F.silu(a) * b  # [N, 1, FF]

        w2_sel = self.w2.index_select(0, flat_indices).to(dtype=x_flat.dtype)  # [N, FF, D]
        out = torch.bmm(hidden, w2_sel).squeeze(1)  # [N, D]
        if self.use_bias:
            out = out + self.b2.index_select(0, flat_indices).to(dtype=out.dtype)

        out = out.view(T, k, D) * weights.reshape(T, k, 1).to(dtype=out.dtype)
        return out.sum(dim=1)

    @staticmethod
    def _route(indices: Tensor, n_exp: int, k: int):
        sorted_expert, order = torch.sort(indices.reshape(-1))
        token_ids = order // k
        offsets = torch.bincount(sorted_expert, minlength=n_exp).cumsum(0).to(torch.int32)
        return sorted_expert, order, token_ids, offsets

    def _routed_grouped(self, x_flat: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        k = self.n_activated_experts
        n_tok = x_flat.size(0)
        sorted_expert, order, token_ids, offsets = self._route(indices, self.n_routed_experts, k)

        inv = torch.empty_like(order)
        inv.scatter_(0, order, torch.arange(order.numel(), device=order.device))

        xs = _DispatchGather.apply(x_flat, token_ids, inv, k)
        h = torch._grouped_mm(xs, self.w13.to(dtype=xs.dtype), offs=offsets)
        if self.use_bias:
            h = h + self.b13.index_select(0, sorted_expert).to(dtype=h.dtype)
        a, b = h.chunk(2, dim=-1)
        out = torch._grouped_mm((F.silu(a) * b).contiguous(), self.w2.to(dtype=xs.dtype), offs=offsets)
        if self.use_bias:
            out = out + self.b2.index_select(0, sorted_expert).to(dtype=out.dtype)

        out = out.index_select(0, inv).view(n_tok, k, -1)
        return (out * weights.reshape(n_tok, k, 1)).sum(dim=1)

    def _routed_loop(self, x_flat: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        k = self.n_activated_experts
        assign_expert = indices.reshape(-1)
        order = torch.argsort(assign_expert)
        assign_expert = assign_expert[order]
        assign_weight = weights.reshape(-1)[order]
        token_ids = order // k

        counts = torch.bincount(assign_expert, minlength=self.n_routed_experts)
        boundaries = counts.cumsum(0).cpu().tolist()

        y_flat = torch.zeros_like(x_flat)
        start = 0
        for expert_id, end in enumerate(boundaries):
            if end == start:
                continue
            cur_token_ids = token_ids[start:end]
            cur_weights = assign_weight[start:end].unsqueeze(-1)
            expert_in = x_flat.index_select(0, cur_token_ids)

            if self.stacked:
                w13 = self.w13[expert_id].to(dtype=expert_in.dtype)
                h = expert_in @ w13
                if self.use_bias:
                    h = h + self.b13[expert_id].to(dtype=h.dtype)
                a, b = h.chunk(2, dim=-1)
                expert_out = (F.silu(a) * b) @ self.w2[expert_id].to(dtype=expert_in.dtype)
                if self.use_bias:
                    expert_out = expert_out + self.b2[expert_id].to(dtype=expert_out.dtype)
            else:
                expert_out = self.experts[expert_id](expert_in)

            y_flat.index_add_(0, cur_token_ids, (expert_out * cur_weights).to(y_flat.dtype))
            start = end
        return y_flat

    def forward(self, x: Tensor) -> Tensor:
        weights, indices = self.gate(x)
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        T = x_flat.size(0)

        if self.stacked and T * self.n_activated_experts <= 64:
            y = self._routed_bmm(x_flat, weights, indices)
        elif self._use_grouped(x_flat):
            y = self._routed_grouped(x_flat, weights, indices)
        else:
            y = self._routed_loop(x_flat, weights, indices)

        z = self.shared_experts(x_flat)
        return (y + z).view(orig_shape)


class MORTMDecoderLayer(nn.Module):
    def __init__(self, args: MORTMArgs, layer_id: int, progress=None):
        super().__init__()
        self.n_head = args.num_heads
        self.args = args
        self.d_model = args.d_model
        self.layer_id = layer_id
        self.self_attention = FlashSelfAttentionM(args, progress=progress)

        if args.use_cross_attention:
            self.cross_attention = FlashCrossAttentionM(args, progress=progress)

        if args.use_moe_decoder:
            self.ffn = MoE(args, layer_id, route_scale=getattr(args, "route_scale", 1.0))
        else:
            if args.use_silu:
                self.ffn = Expert(args)
            else:
                self.ffn = FFN(args.d_model, args.dim_feedforward, args.dropout)

        if args.normalize_type == "tanh":
            self.norm1 = NormTanh(args.d_model)
            if args.use_cross_attention:
                self.norm2 = NormTanh(args.d_model)
            self.norm3 = NormTanh(args.d_model)
        elif args.normalize_type == "layernorm":
            self.norm1 = LayerNorm(args.d_model, eps=1e-5, bias=True)
            if args.use_cross_attention:
                self.norm2 = LayerNorm(args.d_model, eps=1e-5, bias=True)
            self.norm3 = LayerNorm(args.d_model, eps=1e-5, bias=True)
        elif args.normalize_type == "rmsnorm":
            self.norm1 = nn.RMSNorm(args.d_model, eps=1e-5)
            if args.use_cross_attention:
                self.norm2 = nn.RMSNorm(args.d_model, eps=1e-5)
            self.norm3 = nn.RMSNorm(args.d_model, eps=1e-5)

        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout3 = nn.Dropout(args.dropout)

    def forward(
        self,
        tgt: Tensor,
        tgt_is_causal: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        indices: Optional[Tensor] = None,
        is_save_cache: bool = False,
        encoder_x: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        y = tgt
        y = y + self.self_block(
            self.norm1(y),
            tgt_is_causal,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            indices=indices,
            is_save_cache=is_save_cache,
            key_padding_mask=key_padding_mask,
            position_ids=position_ids,
        )
        if self.args.use_cross_attention:
            y = y + self.cross_block(
                self.norm2(y),
                encoder_x,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen_k,
            )
        y = y + self.ff_block(self.norm3(y))
        return y

    def self_block(
        self,
        y: Tensor,
        is_causal: bool,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        indices: Optional[Tensor] = None,
        is_save_cache: bool = False,
        key_padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        y = self.self_attention(
            y,
            is_causal=is_causal,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            indices=indices,
            is_save_cache=is_save_cache,
            key_padding_mask=key_padding_mask,
            position_ids=position_ids,
        )
        return self.dropout1(y)

    def cross_block(
        self,
        x: Tensor,
        encoder_x: Tensor,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> Tensor:
        y = self.cross_attention(
            x,
            encoder_x,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        return self.dropout2(y)

    def ff_block(self, y: Tensor) -> Tensor:
        return self.dropout3(self.ffn(y))


class MORTMDecoder(nn.Module):
    def __init__(self, args: MORTMArgs, progress=None):
        super().__init__()
        self.num_layer = args.d_layer
        self.layers = nn.ModuleList([MORTMDecoderLayer(args, i, progress) for i in range(self.num_layer)])
        if args.normalize_type == "tanh":
            self.norm = NormTanh(args.d_model)
        elif args.normalize_type == "layernorm":
            self.norm = LayerNorm(args.d_model, eps=1e-5, bias=True)
        elif args.normalize_type == "rmsnorm":
            self.norm = nn.RMSNorm(args.d_model, eps=1e-5)

    def allocate_kv_cache(self, batch_size: int, max_seqlen: int, device: torch.device, dtype: torch.dtype) -> None:
        for layer in self.layers:
            layer.self_attention.allocate_kv_cache(batch_size, max_seqlen, device=device, dtype=dtype)

    def reset_cache(self) -> None:
        for layer in self.layers:
            layer.self_attention.reset_cache()

    def forward(
        self,
        tgt: Tensor,
        tgt_is_causal: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        indices: Optional[Tensor] = None,
        is_save_cache: bool = False,
        encoder_x: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                tgt_is_causal=tgt_is_causal,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                indices=indices,
                is_save_cache=is_save_cache,
                encoder_x=encoder_x,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
                key_padding_mask=key_padding_mask,
                position_ids=position_ids,
            )
        return self.norm(output)


class PMA(nn.Module):
    def __init__(self, d_model: int, out_dim: int, h_dim: int = 64):
        super().__init__()
        assert out_dim % h_dim == 0
        self.num_heads = out_dim // h_dim
        self.head_dim = h_dim
        self.out_dim = out_dim
        self.d_model = d_model

        self.q_seed = nn.Parameter(torch.empty(1, out_dim))
        nn.init.normal_(self.q_seed, std=math.sqrt(1.0 / d_model))
        self.w_kv = nn.Linear(d_model, out_dim * 2)
        self.w_o = nn.Linear(out_dim, out_dim)

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        batch = cu_seqlens.numel() - 1
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen_k = int(seqlens.max().item())

        kv = self.w_kv(x).view(-1, 2, self.num_heads, self.head_dim)
        q = self.q_seed.expand(batch, -1).view(batch, self.num_heads, self.head_dim).to(kv.dtype)
        cu_seqlens_q = torch.arange(batch + 1, device=x.device, dtype=torch.int32)

        out = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens.to(torch.int32),
            1,
            max_seqlen_k,
        )
        return self.w_o(out.view(batch, self.out_dim))


class MILPool(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None, r_init: float = 2.0):
        super().__init__()
        h = hidden or d_model
        self.scorer = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, 1))
        self.log_r = nn.Parameter(torch.tensor(float(math.log(r_init))))

    def forward(self, x: Tensor, cu_seqlens: Tensor, return_scores: bool = False):
        s = self.scorer(x).squeeze(-1).float()
        r = self.log_r.exp().clamp(0.05, 20.0)
        cu = cu_seqlens.detach().to("cpu").tolist()
        out = []
        for i in range(len(cu) - 1):
            lo, hi = int(cu[i]), int(cu[i + 1])
            if hi <= lo:
                out.append(s.new_zeros(()))
                continue
            seg = s[lo:hi]
            out.append((torch.logsumexp(r * seg, dim=0) - math.log(float(hi - lo))) / r)
        bag = torch.stack(out)
        return (bag, s) if return_scores else bag
