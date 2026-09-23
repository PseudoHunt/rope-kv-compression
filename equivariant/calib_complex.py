"""Gate 2: calibration statistics and energy diagnostic.

One pass over the calibration tokens with forward hooks on every layer's k_proj (pre-RoPE keys)
accumulates, per layer, in fp64:
    C_pre  = sum_t k_t k_t^T            (stacked pre-RoPE keys, 4096 x 4096)
    C_post = sum_t RoPE(k_t) RoPE(k_t)^T (post-RoPE, model's own apply_rotary_pos_emb)
Every per-frequency complex covariance Sigma_i is a sub-block of C_pre (complex_cov_from_real).
The same accumulation on held-out C4-validation sequences gives out-of-sample energies (the bases are
fit on calibration only).

Outputs: cache/calib_stats.pt (Sigma, top-2048 eigvecs of C_pre / C_post, spectra), results/gate2_*.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .compressor import build_compressor, complex_cov_from_real, rank_alloc, retained_energy
from .data import CACHE, MODEL, calib_tokens, heldout_tokens

ROOT = "/home/jl_fs/rope_equiv"
RHOS = [0.5, 0.25, 0.125]
LATENT = {"b": (True, "global"), "bp": (True, "uniform"), "c": (False, "global"), "cp": (False, "uniform")}


@torch.no_grad()
def accumulate(model, tokens, bs=2):
    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    dev = model.device
    C_pre = torch.zeros(L, D, D, dtype=torch.float64, device=dev)
    C_post = torch.zeros_like(C_pre)
    n, dh = model.config.num_attention_heads, D // model.config.num_attention_heads
    S = tokens.shape[1]
    pos = torch.arange(S, device=dev)[None]
    cos, sin = model.model.rotary_emb(torch.zeros(1, dtype=torch.float32, device=dev), pos)

    def hook(li):
        def f(mod, inp, out):
            k = out.float()  # [b, S, D]
            kh = k.view(k.shape[0], S, n, dh).transpose(1, 2)
            kr, _ = apply_rotary_pos_emb(kh, kh, cos, sin)
            kr = kr.transpose(1, 2).reshape(-1, D).double()
            k = k.reshape(-1, D).double()
            C_pre[li].addmm_(k.T, k)
            C_post[li].addmm_(kr.T, kr)
        return f

    hs = [l.self_attn.k_proj.register_forward_hook(hook(i)) for i, l in enumerate(model.model.layers)]
    t0 = time.time()
    for i in range(0, len(tokens), bs):
        model(tokens[i : i + bs].to(dev), use_cache=False, logits_to_keep=1)
        if i % 64 == 0:
            print(f"  {i}/{len(tokens)}  {time.time() - t0:.0f}s", flush=True)
    for h in hs:
        h.remove()
    N = tokens.numel()
    return C_pre / N, C_post / N


def freq_block_idx(n, dh, device):
    """[P, 2n] indices of pair i's coordinates across all heads in the stacked (head-major) key."""
    P = dh // 2
    h = torch.arange(n, device=device) * dh
    i = torch.arange(P, device=device)[:, None]
    return torch.cat([h[None] + i, h[None] + i + P], 1)


def blockdiag_energy(C, H, idx, r):
    """Diagnostic, NOT RoPE-exact in general: unconstrained real PCA inside each frequency's 2n x 2n
    block (all heads' (x_i, y_i)), eigenvalues pooled across frequencies, top r real dims."""
    Cb = C[idx[:, :, None], idx[:, None, :]]
    Hb = H[idx[:, :, None], idx[:, None, :]]
    ev, U = torch.linalg.eigh(Cb)
    order = torch.sort(-ev.reshape(-1), stable=True).indices[:r]
    f, j = order // ev.shape[1], order % ev.shape[1]
    u = U[f, :, j]  # [r, 2n]
    held = torch.einsum("ra,rab,rb->", u, Hb[f], u).item()
    return ev.reshape(-1)[order].sum().item(), held


def energies(Cp, Cq, Hp, Hq, n, dh):
    """Per-layer energy fractions for every method and rho; in-sample (Cp/Cq) and held-out (Hp/Hq)."""
    rows, alloc, extra = [], {}, {}
    ep, Up = torch.linalg.eigh(Cp)
    eq, Uq = torch.linalg.eigh(Cq)
    ep, Up, eq, Uq = ep.flip(0), Up.flip(1), eq.flip(0), Uq.flip(1)
    Sig, Sig_h = complex_cov_from_real(Cp, n, dh).cpu(), complex_cov_from_real(Hp, n, dh).cpu()
    tot, tot_h = Cp.trace().item(), Hp.trace().item()
    extra["trace_pre_post_rel_diff"] = abs(tot - Cq.trace().item()) / tot
    extra["im_over_abs_fro"] = (Sig.imag.norm() / Sig.norm()).item()
    for rho in RHOS:
        r = int(round(rho * n * dh))  # real dims for (a),(d)
        R = r // 2  # complex channels for latent methods
        E = {
            "a": (ep[:r].sum().item() / tot, (Up[:, :r] * (Hp @ Up[:, :r])).sum().item() / tot_h),
            "d": (eq[:r].sum().item() / tot, (Uq[:, :r] * (Hq @ Uq[:, :r])).sum().item() / tot_h),
        }
        idx = freq_block_idx(n, dh, Cp.device)
        e_in, e_out = blockdiag_energy(Cp, Hp, idx, r)
        E["e"] = (e_in / tot, e_out / tot_h)
        e_in, e_out = blockdiag_energy(Cq, Hq, idx, r)
        E["d_bd"] = (e_in / tot, e_out / tot_h)
        for m, (real, al) in LATENT.items():
            comp = build_compressor(Sig, R, real=real, allocation=al)
            E[m] = (retained_energy(Sig, comp) / tot, retained_energy(Sig_h, comp) / tot_h)
            if m == "c":
                alloc[rho] = rank_alloc(comp, dh // 2).tolist()
        for m, (e_in, e_out) in E.items():
            rows.append(dict(rho=rho, method=m, energy=e_in, energy_heldout=e_out))
    return rows, alloc, extra, Up[:, :2048].float().cpu(), Uq[:, :2048].float().cpu(), ep.cpu(), Sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib", type=int, default=512)
    ap.add_argument("--n_held", type=int, default=50)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(f"{ROOT}/results", exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    n, D = model.config.num_attention_heads, model.config.hidden_size
    dh = D // n
    print("calibration", flush=True)
    Cp, Cq = accumulate(model, calib_tokens(args.n_calib, 4096))
    print("held-out", flush=True)
    Hp, Hq = accumulate(model, heldout_tokens(args.n_held, 4096))
    del model
    torch.cuda.empty_cache()

    rows, allocs, extras = [], {}, {}
    stats = []
    for l in range(Cp.shape[0]):
        r, alloc, extra, Up, Uq, ep, Sig = energies(Cp[l], Cq[l], Hp[l], Hq[l], n, dh)
        for x in r:
            rows.append(dict(layer=l, **x))
        allocs[l], extras[l] = alloc, extra
        stats.append(dict(Sigma=Sig, U_pre=Up, U_post=Uq, eig_pre=ep))
        e = {x["method"]: x["energy"] for x in r if x["rho"] == 0.25}
        print(f"L{l:2d} rho=.25 " + " ".join(f"{k}={v:.3f}" for k, v in e.items())
              + f"  |Im|/|S|={extra['im_over_abs_fro']:.3f}", flush=True)
    torch.save(stats, f"{CACHE}/calib_stats.pt")

    import csv
    with open(f"{ROOT}/results/gate2_energy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(rank_alloc_c=allocs, extras=extras), open(f"{ROOT}/results/gate2_meta.json", "w"))


if __name__ == "__main__":
    main()
