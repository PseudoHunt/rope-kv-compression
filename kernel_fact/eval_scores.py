"""Gate 2: (i) spectrum of the weighted kernel W per (layer, head); (ii) attention-weighted score MSE
of every arm against the uncompressed model on held-out C4-validation.

(ii) evaluation. Per sequence, Q query positions are sampled uniformly (unbiased for E_q). All arms
share the key cache c = U_r^T k; with decoded keys ẑ = B_h^T c, every arm's score is
    s_X(q, Δ) = Re Σ_i w_{q,i} conj(ẑ_{q-Δ, i}) K̃_X[Δ, i],   K̃_X = Φ_X Γ_X,
algebraically identical to the absorbed form (Gate 1 test_three_score_paths_agree_truncated). It is
evaluated in the gap domain: M'[q, Δ, i] = w_{q,i} conj(ẑ_{q-Δ,i}) is built once per layer and one
batched complex matmul against the stacked K̃_X of all arms gives every arm's scores.
Reference s_ref: model's own RoPE on the uncompressed pre-RoPE q, k; a = its causal softmax.
    MSE_w(X) = E_q Σ_j a_qj (s_X - s_ref)^2      MSE_u(X) = E_q mean_{j<=q} (s_X - s_ref)^2
Also recorded: s_EXACT(kernel form) vs the model's RoPE on ẑ (instrument floor; must be << MSE_EXACT).
"""
import argparse
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from equivariant.data import CACHE, MODEL, c4_chunks
from .arms import layer_arms
from .absorb import to_complex

ROOT = "/home/jl_fs/rope_equiv"


def spectrum(Wt, inv_freq):
    from . import factorize as F

    rows = []
    E = F.kernel_E(inv_freq, 4096)
    for l in range(Wt["pi"].shape[0]):
        pi, eps = F.regularize(Wt["pi"][l], Wt["w2"][l] * Wt["z2"][l])
        W = pi.sqrt()[..., None] * E * eps.sqrt()[..., None, :]
        sv2 = torch.linalg.svdvals(W) ** 2
        cum = sv2.cumsum(-1) / sv2.sum(-1, keepdim=True)
        mean_gap = (Wt["pi"][l] * torch.arange(4096)).sum(-1)
        for h in range(W.shape[0]):
            rows.append(dict(layer=l, head=h, mean_gap=mean_gap[h].item(),
                             p99=int((cum[h] < 0.99).sum()) + 1, p999=int((cum[h] < 0.999).sum()) + 1,
                             **{f"cum{p}": cum[h, p - 1].item() for p in [4, 8, 16, 32]}))
    return pd.DataFrame(rows)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=64)
    ap.add_argument("--Q", type=int, default=128)
    ap.add_argument("--rhos", default="0.25,0.125")
    ap.add_argument("--hchunk", type=int, default=8)
    args = ap.parse_args()
    rhos = [float(x) for x in args.rhos.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=dev).eval()
    cfg = model.config
    nL, n = cfg.num_hidden_layers, cfg.num_attention_heads
    dh = cfg.hidden_size // n
    P, T = dh // 2, 4096
    inv_freq = model.model.rotary_emb.inv_freq.cpu()
    Wt = torch.load(f"{CACHE}/kf_weights.pt")
    stats = torch.load(f"{CACHE}/calib_stats.pt")

    sp = spectrum(Wt, inv_freq)
    sp.to_csv(f"{ROOT}/results/kf_gate2_spectrum.csv", index=False)
    print("spectrum: median p99 / p999 per layer:",
          sp.groupby("layer").p99.median().tolist(), sp.groupby("layer").p999.median().tolist(), flush=True)

    # stacked kernels per layer and rho: K [n, L, P, C] complex64
    ps = [4, 8, 16, 32]
    names, K = {}, {}
    t0 = time.time()
    for l in range(nL):
        arms = layer_arms(inv_freq, Wt, l, ps)  # arms do not depend on rho; same set at every rho
        # keep only the small factors on CPU (the stacked K̃ is ~2.3 GB/layer; container RAM is 120 GB)
        fac = [(Ph.to(torch.complex64), G.to(torch.complex64)) for Ph, G in arms.values()]
        for rho in rhos:
            names[(l, rho)], K[(l, rho)] = list(arms), fac
    print(f"built arms in {time.time() - t0:.0f}s", flush=True)
    print("arms per layer:", len(names[(0, 0.25)]), flush=True)

    pos = torch.arange(T, device=dev)
    cos, sin = model.model.rotary_emb(torch.zeros(1, device=dev), pos[None])
    buf = {}

    def hook(l, which):
        def f(mod, inp, out):
            buf[(l, which)] = out[0].view(T, n, dh).transpose(0, 1)  # [n, T, dh] fp16
        return f

    hs = []
    for l, layer in enumerate(model.model.layers):
        hs.append(layer.self_attn.q_proj.register_forward_hook(hook(l, "q")))
        hs.append(layer.self_attn.k_proj.register_forward_hook(hook(l, "k")))

    acc = {}  # (l, rho) -> dict of [n, C] sums
    floor = {}
    toks = c4_chunks("validation", args.n_seq, T)
    g = torch.Generator().manual_seed(0)
    Dg = torch.arange(T, device=dev)
    t0 = time.time()
    for si in range(len(toks)):
        model(toks[si : si + 1].to(dev), use_cache=False, logits_to_keep=1)
        qpos = torch.randint(0, T, (args.Q,), generator=g).to(dev)
        gidx = (qpos[:, None] - Dg[None, :]).clamp_min(0)  # [Q, L] key index at gap Δ
        valid = Dg[None, :] <= qpos[:, None]  # [Q, L]
        for l in range(nL):
            q, k = buf[(l, "q")].float(), buf[(l, "k")].float()
            qs = q[:, qpos]  # [n, Q, dh]
            qr, _ = apply_rotary_pos_emb(qs[None], qs[None], cos[:, qpos], sin[:, qpos])
            _, kr = apply_rotary_pos_emb(k[None], k[None], cos, sin)
            s_ref = torch.gather(qr[0] @ kr[0].transpose(-1, -2), -1, gidx.expand(n, -1, -1))  # [n, Q, L]
            s_ref = s_ref.masked_fill(~valid, float("-inf"))
            a = torch.softmax(s_ref * dh ** -0.5, -1)
            s_ref = s_ref.masked_fill(~valid, 0.0)
            nvalid = valid.sum(-1).float()  # [Q]
            w = to_complex(qs)  # [n, Q, P]
            kf = k.transpose(0, 1).reshape(T, n * dh)
            Kt = torch.stack([(Ph.to(dev) @ G.to(dev)).expand(n, -1, -1) for Ph, G in K[(l, rhos[0])]], -1)
            for rho in rhos:
                r = int(rho * n * dh)
                U = stats[l]["U_pre"][:, :r].to(dev)
                khat = ((kf @ U) @ U.T).view(T, n, dh).transpose(0, 1)  # [n, T, dh]
                zc = to_complex(khat).conj()
                C = Kt.shape[-1]
                sw = torch.zeros(n, C, dtype=torch.float64, device=dev)
                su = torch.zeros_like(sw)
                for h0 in range(0, n, args.hchunk):
                    hsl = slice(h0, h0 + args.hchunk)
                    M = w[hsl, :, None, :] * zc[hsl][:, gidx]  # [h, Q, L, P]
                    M = M.masked_fill(~valid[None, :, :, None], 0).permute(0, 2, 1, 3).contiguous()  # [h, L, Q, P]
                    S = torch.matmul(M, Kt[hsl]).real.permute(0, 2, 1, 3)  # [h, Q, L, C]
                    del M
                    err2 = (S - s_ref[hsl, :, :, None]) ** 2 * valid[None, :, :, None]
                    sw[hsl] += (err2 * a[hsl, :, :, None]).sum((1, 2)).double()
                    su[hsl] += (err2.sum(2) / nvalid[None, :, None]).sum(1).double()
                    if rho == rhos[0] and h0 == 0:
                        # instrument floor: kernel-form EXACT vs the model's RoPE on the decoded keys
                        _, khr = apply_rotary_pos_emb(khat[None], khat[None], cos, sin)
                        sm = torch.gather(qr[0, hsl] @ khr[0, hsl].transpose(-1, -2), -1,
                                          gidx.expand(S.shape[0], -1, -1))
                        ie = names[(l, rho)].index("EXACT")
                        fl = (((S[..., ie] - sm) ** 2 * valid) * a[hsl]).sum().item()
                        floor[l] = floor.get(l, 0.0) + fl
                    del S, err2
                key = (l, rho)
                if key not in acc:
                    acc[key] = dict(sw=sw, su=su)
                else:
                    acc[key]["sw"] += sw
                    acc[key]["su"] += su
        buf.clear()
        print(f"seq {si + 1}/{len(toks)}  {time.time() - t0:.0f}s", flush=True)
        if (si + 1) % 8 == 0 or si + 1 == len(toks):
            save(acc, names, floor, si + 1, args.Q, n)
    for h in hs:
        h.remove()


def save(acc, names, floor, nseq, Q, n):
    from .arms import split

    rows, head_rows = [], []
    for (l, rho), d in acc.items():
        sw, su = (d["sw"] / (nseq * Q)).cpu(), (d["su"] / (nseq * Q)).cpu()  # [n, C]
        for c, name in enumerate(names[(l, rho)]):
            arm, p = split(name)
            rows.append(dict(layer=l, rho=rho, arm=arm, p=p, mse_w=sw[:, c].mean().item(), mse_u=su[:, c].mean().item()))
            for h in range(n):
                head_rows.append(dict(layer=l, head=h, rho=rho, arm=arm, p=p, mse_w=sw[h, c].item(), mse_u=su[h, c].item()))
    df = pd.DataFrame(rows)
    ex = df[df.arm == "EXACT"].set_index(["layer", "rho"])
    df["rho_s"] = df.mse_w / ex.loc[list(zip(df.layer, df.rho)), "mse_w"].values
    df["rho_s_u"] = df.mse_u / ex.loc[list(zip(df.layer, df.rho)), "mse_u"].values
    df["n_seq"] = nseq
    df.to_csv(f"{ROOT}/results/kf_gate2_scoremse.csv", index=False)
    pd.DataFrame(head_rows).to_csv(f"{ROOT}/results/kf_gate2_scoremse_heads.csv", index=False)
    ex25 = ex.xs(0.25, level="rho")
    pd.DataFrame(dict(layer=list(floor), floor_w=[v / (nseq * Q * 8) for v in floor.values()],
                      exact_w=[ex25.loc[l, "mse_w"] for l in floor])).to_csv(
        f"{ROOT}/results/kf_gate2_floor.csv", index=False)


if __name__ == "__main__":
    main()
