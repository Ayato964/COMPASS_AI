import json
import os
from typing import Optional


class MORTMArgs:
    """
    Configuration Value Object for MORTM 4.5 / 4.5D / 4.5E / 5 models.
    100% compatible with all `MORTM` JSON configurations (`80M.json`, `160M.json`,
    `A80M_E64.json`, `MORTM45_Flash`, `MORTM45_Pro`, `MORTM45D_80M_sft_gen`).
    """

    def __init__(self, json_directory: Optional[str] = None, log_scale: bool = False, force_debug_attention: bool = False, **kwargs):
        if json_directory is not None:
            with open(json_directory, 'r', encoding='utf-8') as f:
                data: dict = json.load(f)
        else:
            data = dict(kwargs)

        self.name = "MORTM"
        self.vocab_size: int = data.get('vocab_size') or 128
        self.d_layer: int = data.get('d_layer') or data.get('num_layer') or 12
        self.num_layer: int = self.d_layer
        self.e_layer: int = data.get('e_layer') or 12
        self.num_heads: int = data['num_heads']
        self.d_model: int = data['d_model']
        self.dim_feedforward: int = data.get('dim_feedforward') or data.get('d_ff') or 2048
        self.d_ff: int = self.dim_feedforward
        self.dropout: float = float(data.get('dropout', 0.1))
        self.position_length: int = data.get('position_length') or data.get('max_position_embeddings') or 512
        self.max_position_embeddings: int = self.position_length
        self.min_length: int = data.get('min_length') or 90
        self.num_experts: int = data.get('num_experts') or 12
        self.topk_experts: int = data.get('topk_experts') or data.get('topk') or 2
        self.topk: int = self.topk_experts
        self.num_groups: int = data.get('num_groups') or 1
        self.topk_groups: int = data.get('topk_groups') or 1
        self.route_scale: float = float(data.get('route_scale') or 1.0)
        self.score_type: str = data.get('score_type') or data.get('score_func') or "softmax"
        self.score_func: str = self.score_type
        self.use_moe_encoder: bool = False if data.get('use_moe_encoder') is None else bool(data['use_moe_encoder'])
        self.use_moe_decoder: bool = True if data.get('use_moe_decoder') is None else bool(data['use_moe_decoder'])
        self.is_not_flash = data.get("is_not_flash")
        self.use_silu: bool = False if data.get('use_silu') is None else bool(data['use_silu'])
        self.use_rope: bool = False if data.get('use_rope') is None else bool(data['use_rope'])
        self.rope_theta: float = float(data.get('rope_theta', 10000.0))
        self.use_cross_attention: bool = False if data.get('use_cross_attention') is None else bool(data['use_cross_attention'])
        self.use_bias: bool = True if data.get('use_bias') is None else bool(data['use_bias'])
        self.use_gate_bias: bool = True if data.get('use_gate_bias') is None else bool(data['use_gate_bias'])

        self.normalize_type: str = "tanh" if data.get('norm_type') is None else str(data['norm_type'])
        self.rms_norm_eps: float = float(data.get('rms_norm_eps', 1e-5))

        self.use_attn_lora: bool = False if data.get('use_attn_lora') is None else bool(data['use_attn_lora'])
        self.use_gate_lora: bool = False if data.get('use_gate_lora') is None else bool(data['use_gate_lora'])
        self.use_ffn_lora: bool = False if data.get('use_ffn_lora') is None else bool(data['use_ffn_lora'])
        _shared_lora = data.get('use_shared_expert_lora')
        if _shared_lora is None:
            _shared_lora = data.get('use_shared_ffn_lora', False)
        self.use_shared_expert_lora: bool = bool(_shared_lora)
        self.use_lora: bool = bool(data.get('use_lora', False)) or self.use_attn_lora or self.use_gate_lora or self.use_ffn_lora or self.use_shared_expert_lora
        self.lora_r: int = int(data.get('lora_r') or 8)
        self.lora_alpha: int = int(data.get('lora_alpha') or 16)
        self.lora_dropout: float = float(data.get('lora_dropout', 0.0))
        self.lora_merged: bool = bool(data.get('lora_merged', False))

        # Disable accidental `debug_attention: true` in production inference unless explicitly forced
        env_debug = os.environ.get("MORTM_DEBUG_ATTENTION", "0") == "1"
        raw_debug = bool(data.get('debug_attention', False))
        self.debug_attention: bool = raw_debug if (force_debug_attention or env_debug) else False


class MORTM5Args(MORTMArgs):
    def __init__(self, json_directory: str):
        super().__init__(json_directory)
        with open(json_directory, 'r', encoding='utf-8') as f:
            data: dict = json.load(f)
            self.name = "MORTM5"
            self.roll_w = 24 if data.get("roll_w") is None else data["roll_w"]
            self.roll_h = 128 if data.get("roll_h") is None else data["roll_h"]
            self.track_size = 4 if data.get("track_size") is None else data["track_size"]
            self.first_channel = 64 if data.get("first_channel") is None else data["first_channel"]
            self.max_channel = 256 if data.get("max_channel") is None else data["max_channel"]
            self.bottle_channel = 128 if data.get("bottle_channel") is None else data["bottle_channel"]
            self.encoder_wout = 512 if data.get("encoder_wout") is None else data["encoder_wout"]
            self.vq_groups = 4 if data.get("vq_groups") is None else data["vq_groups"]
            self.encoder_layer = 2 if data.get("encoder_layer") is None else data["encoder_layer"]
            self.codebook_size = 1024 if data.get("codebook_size") is None else data["codebook_size"]
            self.use_pre_vq_norm = True if data.get("use_pre_vq_norm") is None else bool(data["use_pre_vq_norm"])
