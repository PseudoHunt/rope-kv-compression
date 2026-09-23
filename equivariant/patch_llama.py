"""Swap LlamaAttention.forward (per instance) for the compressed variant.

Methods (all at the same K budget of rho * n*d_h real floats per token per layer):
  off  uncompressed
  a    real PCA of stacked pre-RoPE keys (rank 2R), reconstruct, then RoPE       [SALS/Palu-style]
  b    per-frequency real cross-head PCA (eig Re Sigma_i), global top-R            [RoRoPE constraint]
  bp   same as b, uniform R/P channels per frequency                              [RoRoPE-as-published allocation]
  c    per-frequency complex cross-head PCA (eig Sigma_i), global top-R           [this method]
  cp   same as c, uniform allocation                                              [ablation]
  d    real PCA of stacked post-RoPE keys (rank 2R)                              [Eigen-Attention-style]
"""
import copy
import types
from dataclasses import dataclass

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from .attention import attention_core
from .compressor import build_compressor, latent_inv_freq, packing_matrix

LATENT = {"b": (True, "global"), "bp": (True, "uniform"), "c": (False, "global"), "cp": (False, "uniform")}
METHODS = ["off", "a", "b", "bp", "c", "cp", "d"]


@dataclass
class KCState:
    method: str
    proj: torch.Tensor = None  # a, d: [n*dh, n*dh] U_r U_r^T
    B: torch.Tensor = None  # latent: [n, 2R, dh]
    rotary: LlamaRotaryEmbedding = None
    comp: object = None


def latent_rotary(model_rotary, inv_freq):
    rot = copy.deepcopy(model_rotary)
    rot.inv_freq = inv_freq.to(model_rotary.inv_freq.device, model_rotary.inv_freq.dtype)
    rot.original_inv_freq = rot.inv_freq
    return rot


def make_state(method, R, layer_stats, model_rotary, n, dh, device):
    """layer_stats: dict with 'Sigma' [P,n,n] complex, 'U_pre'/'U_post' [n*dh, >=2R] (desc. eigvecs)."""
    if method == "off":
        return KCState("off")
    if method in ("a", "d"):
        U = layer_stats["U_pre" if method == "a" else "U_post"]
        assert 2 * R <= U.shape[1], f"rank {2 * R} exceeds stored basis ({U.shape[1]} columns)"
        U = U[:, : 2 * R].to(device, torch.float32)
        return KCState(method, proj=U @ U.T)
    real, alloc = LATENT[method]
    comp = build_compressor(layer_stats["Sigma"], R, real=real, allocation=alloc)
    B = packing_matrix(comp, n, dh).to(device)
    rot = latent_rotary(model_rotary, latent_inv_freq(comp, model_rotary.inv_freq.cpu()))
    return KCState(method, B=B, rotary=rot, comp=comp)


def _forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None,
             cache_position=None, position_ids=None, **kwargs):
    return attention_core(self, hidden_states, position_embeddings, attention_mask, position_ids,
                          past_key_value=past_key_value, cache_position=cache_position)


def patch_model(model):
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(_forward, layer.self_attn)
    return model


def set_method(model, method, rho=None, stats=None, layers=None):
    """Configure every layer (or only `layers`) to `method`; others go to 'off'."""
    cfg = model.config
    n, dh = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    R = None if rho is None else int(round(rho * n * dh / 2))
    for li, layer in enumerate(model.model.layers):
        on = method != "off" and (layers is None or li in layers)
        layer.self_attn.kc = (make_state(method, R, stats[li], model.model.rotary_emb, n, dh,
                                         layer.self_attn.q_proj.weight.device) if on else KCState("off"))
