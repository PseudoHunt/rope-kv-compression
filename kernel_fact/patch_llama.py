"""Swap LlamaAttention.forward (per instance) for attention through a factorized RoPE kernel.

Every compressed arm shares one cache per layer: c = U_r^T k (stacked pre-RoPE keys, NOT rotated),
stored as fp32 [b, 1, S, r+1] with the key's absolute position in the last channel (exact in fp32),
so gaps Δ = qpos - kpos stay correct under DynamicCache and incremental decoding. V per-head,
uncompressed. Softmax scale = module.scaling = 1/sqrt(d_h).

  off    unmodified model (RoPE, full keys)
  exact  decode k̂ = B_h^T c, apply the model's own RoPE at the stored positions (= previous task's (a))
  kf     scores via Φ, Γ (mode 'absorbed' = §2.4 literally; 'factor' = same algebra on decoded keys)
"""
import types
from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .absorb import uv, to_complex
from .attention import absorbed_scores, factor_scores, factor_scores_fast, gaps


@dataclass
class KFState:
    method: str = "off"
    U: torch.Tensor = None  # [n*dh, r] fp32
    Phi: torch.Tensor = None  # [n|1, L, p] complex
    Gamma: torch.Tensor = None  # [n|1, p, P] complex
    mode: str = "factor"  # 'absorbed' | 'factor' (Gate 1-3) | 'fast' (Gate 4)
    rotary: nn.Module = None
    cache_decoded: bool = False  # Gate 4 simulation: cache k̂ = B_h^T c (identical scores; decoded once per key)
    sink_k: int = 0  # post-hoc SINK-k variant: keys at absolute positions < sink_k are scored with the EXACT path


def _qkv(module, h):
    shp = (*h.shape[:-1], -1, module.head_dim)
    return tuple(proj(h).view(shp).transpose(1, 2) for proj in (module.q_proj, module.k_proj, module.v_proj))


def exact_scores(rotary, q, khat, qpos, kpos):
    """EXACT path: model RoPE on pre-RoPE queries and decoded keys. q [b,n,Tq,dh], khat [b,n,Tk,dh],
    qpos [b,Tq], kpos [b,Tk] -> [b, n, Tq, Tk] (no causal masking)."""
    qc, qs = rotary(q, qpos)
    kc, ks = rotary(khat, kpos)
    qr, _ = apply_rotary_pos_emb(q, q, qc, qs)
    _, kr = apply_rotary_pos_emb(khat, khat, kc, ks)
    return qr @ kr.transpose(-1, -2)


def sink_combine(s_kf, s_exact, qpos, kpos, k):
    """SINK-k: causal keys at absolute positions < k take the EXACT score, all others keep s_kf.
    s_* [b, n, Tq, Tk], qpos [b, Tq], kpos [b, Tk]."""
    use = (kpos[:, None, :] < k) & (kpos[:, None, :] <= qpos[:, :, None])  # [b, Tq, Tk]
    return torch.where(use[:, None], s_exact, s_kf)


def kf_attention(module, hidden_states, position_embeddings, attention_mask, position_ids,
                 past_key_value=None, cache_position=None):
    """Returns (o_proj output, attention weights fp32 [b, n, Tq, Tk])."""
    s, v, b, S = layer_scores(module, hidden_states, position_embeddings, position_ids,
                              past_key_value, cache_position)
    s = s * module.scaling
    if attention_mask is not None:
        s = s + attention_mask[:, :, :, : s.shape[-1]].float()
    w = nn.functional.softmax(s, dim=-1, dtype=torch.float32)
    out = torch.matmul(w.to(v.dtype), v).transpose(1, 2).reshape(b, S, -1)
    return module.o_proj(out), w


def layer_scores(module, hidden_states, position_embeddings, position_ids, past_key_value=None,
                 cache_position=None):
    """Unscaled pre-softmax scores [b, n, Tq, Tk] for the module's current KFState (causal -inf
    already applied on the compressed paths), plus the (cached) values."""
    st = module.kf
    q, k, v = _qkv(module, hidden_states)
    b, n, S, dh = q.shape
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    position_ids = position_ids.expand(b, -1)

    if st.method == "off":
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, module.layer_idx, {"cache_position": cache_position})
        s = (q.float() @ k.float().transpose(-1, -2))
    elif st.cache_decoded:
        kf = k.transpose(1, 2).reshape(b, S, n * dh).to(st.U.dtype)
        khat = ((kf @ st.U) @ st.U.T).view(b, S, n, dh).transpose(1, 2)  # [b, n, S, dh]
        key = torch.cat([khat, position_ids.to(khat.dtype)[:, None, :, None].expand(b, n, S, 1)], -1)
        if past_key_value is not None:
            key, v = past_key_value.update(key, v, module.layer_idx, {"cache_position": cache_position})
        Khat, kpos = key[..., :dh], key[:, 0, :, dh].round().long()
        qf = q.to(st.U.dtype)
        if st.method == "exact":
            s = exact_scores(st.rotary, qf, Khat, position_ids, kpos)
        else:
            if S == 1:
                s = batched_gather_scores(qf, Khat, st.Phi, st.Gamma, position_ids, kpos)
            else:
                s = torch.stack([factor_scores_fast(qf[bi], Khat[bi], st.Phi, st.Gamma, position_ids[bi], kpos[bi])
                                 for bi in range(b)])
            if st.sink_k > 0:
                s = sink_combine(s, exact_scores(st.rotary, qf, Khat, position_ids, kpos), position_ids, kpos, st.sink_k)
    else:
        c = k.transpose(1, 2).reshape(b, S, n * dh).to(st.U.dtype) @ st.U  # [b, S, r]
        key = torch.cat([c, position_ids.to(c.dtype).unsqueeze(-1)], -1).unsqueeze(1)
        if past_key_value is not None:
            key, v = past_key_value.update(key, v, module.layer_idx, {"cache_position": cache_position})
        r = st.U.shape[1]
        C, kpos = key[:, 0, :, :r], key[:, 0, :, r].round().long()
        B = st.U.view(n, dh, r).transpose(1, 2)  # [n, r, dh]
        s = []
        for bi in range(b):
            qb, qpos = q[bi].to(st.U.dtype), position_ids[bi]
            khat = (C[bi] @ st.U.T).view(-1, n, dh).transpose(0, 1)  # [n, Tk, dh]
            if st.method == "exact":
                sb = exact_scores(st.rotary, qb[None], khat[None], qpos[None], kpos[bi][None])[0]
            else:
                if st.mode == "absorbed":
                    sb = absorbed_scores(qb, C[bi], B, st.Phi, st.Gamma, qpos, kpos[bi])
                else:
                    sb = factor_scores(qb, khat, st.Phi, st.Gamma, qpos, kpos[bi])
                if st.sink_k > 0:
                    se = exact_scores(st.rotary, qb[None], khat[None], qpos[None], kpos[bi][None])
                    sb = sink_combine(sb[None], se, qpos[None], kpos[bi][None], st.sink_k)[0]
            s.append(sb)
        s = torch.stack(s)
    return s, v, b, S


def batched_gather_scores(q, khat, Phi, Gamma, qpos, kpos):
    """Batched over samples with per-sample positions (decode / left padding). q [b,n,Tq,dh],
    khat [b,n,Tk,dh], qpos [b,Tq], kpos [b,Tk]. Real form (§2.3). -> [b, n, Tq, Tk]."""
    w = to_complex(q)
    D = (qpos[:, :, None] - kpos[:, None, :])  # [b, Tq, Tk]
    neg = D < 0
    D = D.clamp_min(0)
    PhR, PhI = Phi.real.to(q.dtype), Phi.imag.to(q.dtype)  # [n|1, L, p]
    kT = khat.transpose(-1, -2)
    acc = torch.zeros(*q.shape[:3], khat.shape[2], dtype=q.dtype, device=q.device)
    for k in range(Gamma.shape[-2]):
        u, v = uv(w * Gamma[:, k].to(w.dtype)[None, :, None, :])
        pr = PhR[:, :, k][:, D].permute(1, 0, 2, 3)  # [b, n|1, Tq, Tk]
        pi = PhI[:, :, k][:, D].permute(1, 0, 2, 3)
        acc += pr * (u @ kT) - pi * (v @ kT)
    return acc.masked_fill(neg[:, None], float("-inf"))


def _forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None,
             cache_position=None, position_ids=None, **kwargs):
    return kf_attention(self, hidden_states, position_embeddings, attention_mask, position_ids,
                        past_key_value=past_key_value, cache_position=cache_position)


def patch_model(model):
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(_forward, layer.self_attn)
        layer.self_attn.kf = KFState()
    return model


def set_states(model, states):
    """states: list (per layer) of KFState; the model's rotary module is attached for 'exact'."""
    for layer, st in zip(model.model.layers, states):
        st.rotary = model.model.rotary_emb
        layer.self_attn.kf = st
