"""
Hardware-Agnostic Multi-Head Self/Cross Attention with Zero-Sync Contiguous SDPA KV Cache.
100% parameter-name compatible with MORTM 4.5 / 4.5D / 4.5E checkpoints (`qkv_block.qkv_weight`,
`qkv_block.W_o`, RoPE `base=10000`, and ALiBi slopes), while eliminating mandatory `flash_attn`
and `loralib` dependencies, `.item()` GPU-to-CPU syncs, and per-step `.permute()` copies.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.lora_layers import LoRALinear


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unpad_input(hidden_states: Tensor, attention_mask: Tensor):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() > 0 else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    unpadded = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
    return unpadded, indices, cu_seqlens, max_seqlen_in_batch, seqlens_in_batch


def pad_input(hidden_states: Tensor, indices: Tensor, batch: int, seqlen: int) -> Tensor:
    output = torch.zeros(
        batch * seqlen,
        *hidden_states.shape[1:],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def get_alibi_slopes(n_heads: int) -> list:
    def get_slopes_power_of_2(n: int) -> list:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra = get_alibi_slopes(2 * closest_power_of_2)[0::2]
    slopes.extend(extra[: n_heads - closest_power_of_2])
    return slopes


def _build_alibi_bias(
    q_len: int,
    k_len: int,
    alibi_slopes: Tensor,
    causal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    q_pos = torch.arange(k_len - q_len, k_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    rel_pos = -torch.abs(q_pos - k_pos).to(torch.float32)
    bias = alibi_slopes.to(device=device, dtype=torch.float32)[:, None, None] * rel_pos[None, :, :]
    if causal:
        causal_mask = k_pos > q_pos
        bias = bias.masked_fill(causal_mask[None, :, :], float("-inf"))
    return bias.unsqueeze(0).to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
        if q.dim() == 4 and cos.dim() == 2:
            cos = cos.unsqueeze(1).unsqueeze(1)
            sin = sin.unsqueeze(1).unsqueeze(1)
        elif q.dim() == 4 and cos.dim() == 3:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        elif q.dim() == 3 and cos.dim() == 2:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 5000, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_position_embeddings = seq_len
        t = torch.arange(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward_at_pos(self, x: Tensor, pos: int) -> Tuple[Tensor, Tensor]:
        """O(1) zero-sync RoPE lookup at integer `pos` for single-token decoding."""
        if pos >= self.max_position_embeddings or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(max(pos + 512, self.max_position_embeddings * 2), x.device, x.dtype)
        cos = self.cos_cached[pos : pos + 1].view(1, 1, 1, -1).to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[pos : pos + 1].view(1, 1, 1, -1).to(device=x.device, dtype=x.dtype)
        return cos, sin

    def forward(self, x: Tensor, seq_len: int = 2048) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_position_embeddings or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(max(seq_len + 512, self.max_position_embeddings), x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )


def flash_attn_varlen_kvpacked_func(
    q: Tensor,
    kv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tensor:
    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()
    outs = []
    for i in range(len(cu_q) - 1):
        sq, eq = int(cu_q[i]), int(cu_q[i + 1])
        sk, ek = int(cu_k[i]), int(cu_k[i + 1])
        if eq <= sq:
            continue
        if ek <= sk:
            outs.append(torch.zeros((eq - sq, q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype))
            continue
        q_i = q[sq:eq].permute(1, 0, 2).unsqueeze(0)
        k_i, v_i = kv[sk:ek].unbind(dim=1)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0)
        o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        outs.append(o_i.squeeze(0).permute(1, 0, 2))
    if not outs:
        return torch.empty_like(q)
    return torch.cat(outs, dim=0)


class QKVLinear(nn.Module):
    """
    Packed QKV and Output Projection matching MORTM checkpoint parameter names:
    `qkv_weight` (3 * d_model, d_model) and `W_o` (d_model, d_model).
    Uses built-in `LoRALinear` when `args.use_attn_lora` is enabled.
    """

    def __init__(self, args: MORTMArgs, use_cross_attention: bool = False):
        super().__init__()
        self.num_heads = args.num_heads
        self.head_dim = args.d_model // args.num_heads
        self.use_cross_attention = use_cross_attention

        if not use_cross_attention:
            if not args.use_attn_lora:
                self.qkv_weight = nn.Linear(args.d_model, 3 * args.d_model, bias=False)
                self.W_o = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
            else:
                self.qkv_weight = LoRALinear(
                    args.d_model,
                    3 * args.d_model,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    bias=False,
                )
                self.W_o = LoRALinear(
                    args.d_model,
                    args.d_model,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    bias=args.use_bias,
                )
        else:
            self.q_weight = nn.Linear(args.d_model, args.d_model, bias=True)
            self.kv_weight = nn.Linear(args.d_model, 2 * args.d_model, bias=True)
            self.W_o = nn.Linear(args.d_model, args.d_model)

    def forward(self, q: Tensor, kv: Optional[Tensor] = None):
        if not self.use_cross_attention:
            qkv = self.qkv_weight(q)
            new_shape = qkv.shape[:-1] + (3, self.num_heads, self.head_dim)
            return qkv.view(new_shape)
        else:
            q_out = self.q_weight(q)
            q_out = q_out.view(q_out.shape[:-1] + (self.num_heads, self.head_dim))
            kv_out = self.kv_weight(kv)
            kv_out = kv_out.view(kv_out.shape[:-1] + (2, self.num_heads, self.head_dim))
            return q_out, kv_out

    def comp(self, o: Tensor) -> Tensor:
        return self.W_o(o)


class FlashSelfAttentionM(nn.Module):
    """
    High-Speed Multi-Head Self-Attention with Contiguous SDPA KV Cache (`[B, H, MaxSeqLen, D]`).
    Supports both RoPE (`args.use_rope=True`) and ALiBi (`args.use_rope=False`),
    and both batched 3D `[B, S, D]` and varlen 2D `[Total, D]` inputs.
    """

    def __init__(self, args: MORTMArgs, progress=None):
        super().__init__()
        self.args = args
        self.num_heads = args.num_heads
        self.head_dim = args.d_model // args.num_heads
        self.d_model = args.d_model
        self.drop = args.dropout

        self.qkv_block = QKVLinear(args)

        # Contiguous SDPA KV cache: [B, H, MaxSeqLen, D]
        self.k_cache: Optional[Tensor] = None
        self.v_cache: Optional[Tensor] = None
        self.kv_cache: Optional[Tuple[Tensor, Tensor]] = None
        self.cache_seqlens: Optional[Tensor] = None
        self._cache_pos: int = 0

        device = progress.get_device() if progress else _get_default_device()

        if self.args.use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=args.position_length,
                base=getattr(args, "rope_theta", 10000.0),
                device=device,
            )
            self.alibi_slopes = None
        else:
            self.register_buffer(
                "alibi_slopes",
                torch.tensor(get_alibi_slopes(args.num_heads), dtype=torch.float32, device=device),
                persistent=False,
            )
            self.rotary_emb = None

    def allocate_kv_cache(self, batch_size: int, max_seqlen: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (batch_size, self.num_heads, max_seqlen, self.head_dim)
        self.k_cache = torch.empty(shape, device=device, dtype=dtype)
        self.v_cache = torch.empty(shape, device=device, dtype=dtype)
        self.kv_cache = (self.k_cache, self.v_cache)
        self.cache_seqlens = torch.zeros(batch_size, device=device, dtype=torch.int32)
        self._cache_pos = 0

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.kv_cache = None
        self.cache_seqlens = None
        self._cache_pos = 0

    def forward(
        self,
        x: Tensor,
        is_causal: bool = True,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        indices: Optional[Tensor] = None,
        is_save_cache: bool = False,
        key_padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        target_dtype = self.qkv_block.qkv_weight.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        # =====================================================================
        # Fast Path A: Batched 3D Tensor [Batch, SeqLen, Dim] (Prefill & Decode)
        # Zero unpad_input / pad_input overhead, zero .item() GPU syncs!
        # =====================================================================
        if cu_seqlens is None and x.dim() == 3:
            B, S_q, _ = x.shape
            qkv = self.qkv_block(q=x)  # [B, S_q, 3, H, D]
            q, k, v = qkv.unbind(2)    # each [B, S_q, H, D]

            # Single-Token Decode with Active Contiguous KV Cache (Zero GPU Sync)
            if is_save_cache and S_q == 1 and self.k_cache is not None and self._cache_pos > 0:
                pos = self._cache_pos
                if self.args.use_rope:
                    if position_ids is None:
                        cos, sin = self.rotary_emb.forward_at_pos(v, pos)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)
                    else:
                        rope_len = max(pos + 1, int(position_ids.max().item()) + 1)
                        cos, sin = self.rotary_emb(v, seq_len=rope_len)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

                q_t = q.transpose(1, 2)  # [B, H, 1, D]
                k_t = k.transpose(1, 2)  # [B, H, 1, D]
                v_t = v.transpose(1, 2)  # [B, H, 1, D]

                if pos >= self.k_cache.shape[2]:
                    new_max = self.k_cache.shape[2] * 2
                    new_k = torch.empty((self.k_cache.shape[0], self.num_heads, new_max, self.head_dim), dtype=self.k_cache.dtype, device=self.k_cache.device)
                    new_v = torch.empty((self.v_cache.shape[0], self.num_heads, new_max, self.head_dim), dtype=self.v_cache.dtype, device=self.v_cache.device)
                    new_k[:, :, :pos, :] = self.k_cache[:, :, :pos, :]
                    new_v[:, :, :pos, :] = self.v_cache[:, :, :pos, :]
                    self.k_cache = new_k
                    self.v_cache = new_v
                    self.kv_cache = (self.k_cache, self.v_cache)

                self.k_cache[:B, :, pos : pos + 1, :] = k_t
                self.v_cache[:B, :, pos : pos + 1, :] = v_t
                end_pos = pos + 1
                self._cache_pos = end_pos
                if self.cache_seqlens is not None:
                    self.cache_seqlens[:B] = end_pos

                k_full = self.k_cache[:B, :, :end_pos, :]
                v_full = self.v_cache[:B, :, :end_pos, :]

                attn_mask = None
                if not self.args.use_rope and self.alibi_slopes is not None:
                    attn_mask = _build_alibi_bias(1, end_pos, self.alibi_slopes, False, q_t.device, q_t.dtype)
                    if key_padding_mask is not None:
                        valid_mask = key_padding_mask[:, :end_pos].bool()
                        if not valid_mask.all():
                            pad_bias = torch.zeros((B, 1, 1, end_pos), dtype=q_t.dtype, device=q_t.device)
                            pad_bias.masked_fill_(~valid_mask.view(B, 1, 1, end_pos), float("-inf"))
                            attn_mask = attn_mask + pad_bias
                elif key_padding_mask is not None:
                    valid_mask = key_padding_mask[:, :end_pos].bool()
                    if not valid_mask.all():
                        attn_mask = valid_mask.view(B, 1, 1, end_pos)

                out = F.scaled_dot_product_attention(
                    q_t,
                    k_full,
                    v_full,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
                out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
                return self.qkv_block.comp(out)

            # Batched Prefill or Non-Cached 3D Forward
            if self.args.use_rope:
                if position_ids is None:
                    position_ids = torch.arange(S_q, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
                    rope_len = S_q
                else:
                    rope_len = max(S_q, int(position_ids.max().item()) + 1) if position_ids.numel() > 0 else S_q
                cos, sin = self.rotary_emb(v, seq_len=rope_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

            q_t = q.transpose(1, 2)  # [B, H, S_q, D]
            k_t = k.transpose(1, 2)  # [B, H, S_q, D]
            v_t = v.transpose(1, 2)  # [B, H, S_q, D]

            if is_save_cache:
                if self.k_cache is None or self.k_cache.shape[0] < B or self.k_cache.shape[2] < S_q:
                    alloc_len = max(self.args.position_length + 512, S_q + 1024)
                    self.allocate_kv_cache(B, alloc_len, x.device, q_t.dtype)
                self.k_cache[:B, :, :S_q, :] = k_t
                self.v_cache[:B, :, :S_q, :] = v_t
                self._cache_pos = S_q
                if self.cache_seqlens is not None:
                    self.cache_seqlens[:B] = S_q

            attn_mask = None
            use_causal = is_causal and S_q > 1
            if not self.args.use_rope and self.alibi_slopes is not None:
                attn_mask = _build_alibi_bias(S_q, S_q, self.alibi_slopes, use_causal, q_t.device, q_t.dtype)
                if key_padding_mask is not None:
                    valid_mask = key_padding_mask.bool()
                    if not valid_mask.all():
                        pad_bias = torch.zeros((B, 1, 1, S_q), dtype=q_t.dtype, device=q_t.device)
                        pad_bias.masked_fill_(~valid_mask.view(B, 1, 1, S_q), float("-inf"))
                        attn_mask = attn_mask + pad_bias
                        # Prevent softmax([-inf, ...]) -> NaN on leading left-pad query positions
                        row_has_valid = torch.isfinite(attn_mask).any(dim=-1, keepdim=True)
                        if not bool(row_has_valid.all()):
                            diag_zero = torch.full((1, 1, S_q, S_q), float("-inf"), dtype=q_t.dtype, device=q_t.device)
                            diag_idx = torch.arange(S_q, device=q_t.device)
                            diag_zero[:, :, diag_idx, diag_idx] = 0.0
                            attn_mask = torch.where(row_has_valid, attn_mask, diag_zero)
                use_causal = False
            elif key_padding_mask is not None:
                valid_mask = key_padding_mask.bool()
                if not valid_mask.all():
                    pad_mask = valid_mask.view(B, 1, 1, S_q)
                    if use_causal:
                        causal_mask = torch.ones((S_q, S_q), dtype=torch.bool, device=x.device).tril()
                        attn_mask = causal_mask.view(1, 1, S_q, S_q) & pad_mask
                    else:
                        attn_mask = pad_mask.expand(B, 1, S_q, S_q)
                    # Prevent softmax(all-False) -> NaN on leading left-pad query positions
                    row_has_valid = attn_mask.any(dim=-1, keepdim=True)
                    if not bool(row_has_valid.all()):
                        eye_mask = torch.eye(S_q, dtype=torch.bool, device=x.device).view(1, 1, S_q, S_q)
                        attn_mask = torch.where(row_has_valid, attn_mask, eye_mask)
                    use_causal = False

            drop_p = self.drop if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=attn_mask,
                dropout_p=drop_p,
                is_causal=use_causal,
            )
            out = out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
            return self.qkv_block.comp(out)

        # =====================================================================
        # Path B: Legacy 2D Decode `[Batch, Dim]` or Packed Varlen `cu_seqlens`
        # =====================================================================
        if cu_seqlens is None and x.dim() == 2:
            # 2D single-step decode [B, D] -> delegate to 3D fast path and squeeze back to [B, D]
            out_3d = self.forward(
                x.unsqueeze(1),
                is_causal=is_causal,
                is_save_cache=is_save_cache,
            )
            return out_3d.squeeze(1)

        # Varlen `cu_seqlens` path (training / heterogeneous prompt lengths)
        qkv = self.qkv_block(q=x)  # [Total, 3, H, D]
        q, k, v = qkv.unbind(1)
        cu_list = cu_seqlens.detach().cpu().tolist()
        B_var = len(cu_list) - 1

        if self.args.use_rope:
            pos_list = [
                torch.arange(0, int(cu_list[i + 1]) - int(cu_list[i]), device=x.device, dtype=torch.long)
                for i in range(B_var)
            ]
            pos_ids = torch.cat(pos_list) if pos_list else torch.empty(0, device=x.device, dtype=torch.long)
            cos, sin = self.rotary_emb(v, seq_len=int(max_seqlen or x.shape[0]))
            q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

        if is_save_cache:
            alloc_len = max(self.args.position_length + 512, int(max_seqlen or 0) + 1024)
            if self.k_cache is None or self.k_cache.shape[0] < B_var or self.k_cache.shape[2] < alloc_len:
                self.allocate_kv_cache(B_var, alloc_len, x.device, q.dtype)
            for i in range(B_var):
                s_i, e_i = int(cu_list[i]), int(cu_list[i + 1])
                l_i = e_i - s_i
                if l_i > 0:
                    self.k_cache[i, :, :l_i, :] = k[s_i:e_i].permute(1, 0, 2)
                    self.v_cache[i, :, :l_i, :] = v[s_i:e_i].permute(1, 0, 2)
                self.cache_seqlens[i] = l_i
            self._cache_pos = int(cu_list[1] - cu_list[0]) if B_var > 0 else 0

        outs = []
        drop_p = self.drop if self.training else 0.0
        for i in range(B_var):
            s_i, e_i = int(cu_list[i]), int(cu_list[i + 1])
            if e_i <= s_i:
                continue
            q_i = q[s_i:e_i].permute(1, 0, 2).unsqueeze(0)
            k_i = k[s_i:e_i].permute(1, 0, 2).unsqueeze(0)
            v_i = v[s_i:e_i].permute(1, 0, 2).unsqueeze(0)
            L_i = e_i - s_i
            if not self.args.use_rope and self.alibi_slopes is not None:
                attn_mask = _build_alibi_bias(L_i, L_i, self.alibi_slopes, is_causal, q_i.device, q_i.dtype)
                o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, attn_mask=attn_mask, dropout_p=drop_p, is_causal=False)
            else:
                o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=drop_p, is_causal=is_causal)
            outs.append(o_i.squeeze(0).permute(1, 0, 2))

        out_cat = torch.cat(outs, dim=0) if outs else torch.empty((0, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        out_flat = out_cat.contiguous().view(-1, self.d_model)
        return self.qkv_block.comp(out_flat)


class FlashCrossAttentionM(nn.Module):
    def __init__(self, args: MORTMArgs, progress=None):
        super().__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None
        self.args = args
        self.embed_dim = args.d_model
        self.qkv_block = QKVLinear(args, use_cross_attention=True)
        self.drop = args.dropout

    def forward(
        self,
        x: Tensor,
        encoder_x: Tensor,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> Tensor:
        target_dtype = self.qkv_block.q_weight.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        if encoder_x.dtype != target_dtype:
            encoder_x = encoder_x.to(dtype=target_dtype)

        if cu_seqlens_q is not None:
            q, kv = self.qkv_block(q=x, kv=encoder_x)
            out = flash_attn_varlen_kvpacked_func(
                q=q,
                kv=kv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q or 1,
                max_seqlen_k=max_seqlen_k or 1,
                causal=False,
                dropout_p=(self.drop if self.training else 0.0),
            )
            out = out.contiguous().view(-1, self.embed_dim)
            return self.qkv_block.comp(out)

        q, kv = self.qkv_block(q=x, kv=encoder_x)
        if q.dim() == 3:
            q = q.unsqueeze(0)
            kv = kv.unsqueeze(0)
        k, v = kv.unbind(dim=2)
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)
        o_t = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=(self.drop if self.training else 0.0), is_causal=False)
        out = o_t.permute(0, 2, 1, 3).contiguous().view(-1, self.embed_dim)
        return self.qkv_block.comp(out)
