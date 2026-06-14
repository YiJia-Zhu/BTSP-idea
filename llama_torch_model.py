from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaTorchConfig:
    vocab_size: int
    hidden_size: int = 128
    intermediate_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    dropout: float = 0.0


def llama1b_body_config(vocab_size: int, max_position_embeddings: int) -> LlamaTorchConfig:
    """Llama-1B body dimensions from the local Llama-3.2-1B config, without loading weights."""
    return LlamaTorchConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
    )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos[:seq_len].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaTorchConfig) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.kv_repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        k = repeat_kv(k, self.kv_repeats)
        v = repeat_kv(v, self.kv_repeats)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(y)


class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaTorchConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaTorchConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = LlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = LlamaMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaTorchCausalLM(nn.Module):
    def __init__(self, cfg: LlamaTorchConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return logits, loss

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
