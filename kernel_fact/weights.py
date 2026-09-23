"""Calibration weights for the kernel factorization, per (layer, head), from C4-train only.

  pi[l,h,Δ]    attention-weighted gap distribution: mean over queries of the uncompressed model's
               attention mass at gap Δ (sums to 1 per query, so pi sums to 1).
  w2, z2       E|w_{h,i}|^2, E|z_{h,i}|^2 over all tokens  -> default ε = w2 * z2
  wa, za       E|w_{h,i}|, E|z_{h,i}|                       -> MHA2MLA 2-norm score (PR-2n)
  joint        E_q Σ_j a_qj |w_{q,i}|^2 |z_{j,i}|^2         -> ablation ε (attention-weighted, joint)
Saved to cache/kf_weights.pt.
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from equivariant.data import CACHE, MODEL, calib_tokens


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=64)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
    cfg = model.config
    nL, n = cfg.num_hidden_layers, cfg.num_attention_heads
    dh = cfg.hidden_size // n
    P, T = dh // 2, 4096
    dev = "cuda"
    acc = {k: torch.zeros(nL, n, T if k == "pi" else P, dtype=torch.float64, device=dev)
           for k in ["pi", "w2", "z2", "wa", "za", "joint"]}
    pos = torch.arange(T, device=dev)
    cos, sin = model.model.rotary_emb(torch.zeros(1, device=dev), pos[None])
    gap_idx = (pos[:, None] - pos[None, :]).clamp_min(0)  # [q, Δ] -> key index q - Δ
    valid = pos[None, :] <= pos[:, None]  # Δ <= q
    causal = torch.full((T, T), float("-inf"), device=dev).triu(1)
    qbuf = {}

    def qhook(l):
        def f(mod, inp, out):
            qbuf[l] = out
        return f

    def khook(l):
        def f(mod, inp, out):
            q = qbuf.pop(l)[0].float().view(T, n, dh).transpose(0, 1)  # [n, T, dh] pre-RoPE
            k = out[0].float().view(T, n, dh).transpose(0, 1)
            qr, kr = apply_rotary_pos_emb(q[None], k[None], cos, sin)
            A = torch.softmax((qr[0] @ kr[0].transpose(-1, -2)) * dh ** -0.5 + causal, -1)  # [n, T, T]
            acc["pi"][l] += (A.gather(-1, gap_idx.expand(n, -1, -1)) * valid).sum(1).double()
            w2 = q[..., :P] ** 2 + q[..., P:] ** 2
            z2 = k[..., :P] ** 2 + k[..., P:] ** 2
            acc["w2"][l] += w2.sum(1).double()
            acc["z2"][l] += z2.sum(1).double()
            acc["wa"][l] += w2.sqrt().sum(1).double()
            acc["za"][l] += z2.sqrt().sum(1).double()
            acc["joint"][l] += (w2 * (A @ z2)).sum(1).double()
        return f

    hs = []
    for l, layer in enumerate(model.model.layers):
        hs.append(layer.self_attn.q_proj.register_forward_hook(qhook(l)))
        hs.append(layer.self_attn.k_proj.register_forward_hook(khook(l)))
    toks = calib_tokens()[: args.n_seq]
    t0 = time.time()
    for i in range(len(toks)):
        model(toks[i : i + 1].to(dev), use_cache=False, logits_to_keep=1)
        if i % 8 == 0:
            print(f"{i}/{len(toks)} {time.time() - t0:.0f}s", flush=True)
    for h in hs:
        h.remove()
    N = len(toks) * T
    out = {k: (v / N).cpu() for k, v in acc.items()}
    out["n_tokens"] = N
    torch.save(out, f"{CACHE}/kf_weights.pt")
    pi = out["pi"]
    print("pi row sums (should be 1):", pi.sum(-1).min().item(), pi.sum(-1).max().item())
    mean_gap = (pi * torch.arange(T)).sum(-1)
    print("mean attended gap per layer (median over heads):", mean_gap.median(-1).values.round().tolist())


if __name__ == "__main__":
    main()
