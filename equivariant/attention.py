"""Eager attention for every arm. V is always per-head and uncompressed.

latent arms (b, b', c, c'): each head's absorbed query q~_h = B_h q_h (runtime, not a materialized
W_Q') attends to one shared rotated latent c = sum_h B_h k_h (runtime, not a materialized W_K').
Both are RoPE'd with the model's own apply_rotary_pos_emb using cos/sin from a LlamaRotaryEmbedding
whose inv_freq is the per-channel latent table. The latent is cached already rotated.
Softmax scale stays module.scaling = 1/sqrt(d_h) = 1/sqrt(128), not 1/sqrt(2R).

projection arms (a, d): keys replaced by their rank-truncated PCA reconstruction, before RoPE (a)
or after RoPE (d); standard per-head attention afterwards. (Fidelity simulation: the cache holds
reconstructed keys, so it does not actually save memory.)
"""
import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def softmax_attend(q, k, v, mask, scaling):
    """q [b,n,s,e], k [b,n|1,S,e], v [b,n,S,dh]. Returns (out [b,s,n,dh], weights fp32)."""
    w = torch.matmul(q, k.transpose(-1, -2)) * scaling
    if mask is not None:
        w = w + mask[:, :, :, : k.shape[-2]]
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32)
    out = torch.matmul(w.to(v.dtype), v)
    return out.transpose(1, 2).contiguous(), w


def attention_core(module, hidden_states, position_embeddings, attention_mask, position_ids,
                   past_key_value=None, cache_position=None):
    """Returns (o_proj output, attention weights fp32). module.kc selects the arm."""
    kc = getattr(module, "kc", None)
    method = "off" if kc is None else kc.method
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    dtype = hidden_states.dtype

    q = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings

    if method in ("off", "a", "d"):
        if method == "a":
            b, n, s, dh = k.shape
            kf = k.transpose(1, 2).reshape(b, s, n * dh).float() @ kc.proj
            k = kf.to(dtype).view(b, s, n, dh).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if method == "d":
            b, n, s, dh = k.shape
            kf = k.transpose(1, 2).reshape(b, s, n * dh).float() @ kc.proj
            k = kf.to(dtype).view(b, s, n, dh).transpose(1, 2)
    else:
        B = kc.B  # [n, 2R, dh] fp32
        q = torch.einsum("bhsd,hed->bhse", q.float(), B)
        k = torch.einsum("bhsd,hed->bse", k.float(), B).unsqueeze(1)  # one shared latent "head"
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        lcos, lsin = kc.rotary(k, position_ids)  # model's own RoPE code, latent inv_freq
        q, k = apply_rotary_pos_emb(q, k, lcos, lsin)
        q, k = q.to(dtype), k.to(dtype)

    if past_key_value is not None:
        k, v = past_key_value.update(k, v, module.layer_idx, {"cache_position": cache_position})

    out, w = softmax_attend(q, k, v, attention_mask, module.scaling)
    out = module.o_proj(out.reshape(*input_shape, -1))
    return out, w
