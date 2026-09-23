"""Instrument check for the Gate 2 sanity violation "(d) post-RoPE PCA beats (c)".

Hypothesis: if the key distribution is position-independent, C_post is fully determined by C_pre:
    C_post_pred = (1/S) sum_m R_m C_pre R_m^T
               = (Cos^T Cos)∘C + (Cos^T Sin)∘(C J^T) + (Sin^T Cos)∘(J C) + (Sin^T Sin)∘(J C J^T),  / S
where R_m k = k∘cos_m + (J k)∘sin_m, J = rotate_half as a matrix. Position averaging multiplies each
cross-frequency block by a Dirichlet average of e^{jm(θi∓θk)}, which is ~0 unless both frequencies are
slow. So if (d) - (c) is real, the predicted C_post reproduces the measured (d) energy, and restricting
the prediction to the fast pairs removes the (d) - (c) gap.
"""
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half

from .compressor import build_compressor, complex_cov_from_real, retained_energy
from .data import MODEL, calib_tokens

LAYERS = [0, 15, 31]


def topk_frac(C, r):
    ev = torch.linalg.eigvalsh(C).flip(0)
    return (ev[:r].sum() / ev.sum()).item()


@torch.no_grad()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
    n, D = 32, 4096
    dh, P, S = 128, 64, 4096
    pos = torch.arange(S, device="cuda")[None]
    cos, sin = model.model.rotary_emb(torch.zeros(1, device="cuda"), pos)  # [1, S, dh]
    acc = {l: [torch.zeros(D, D, dtype=torch.float64, device="cuda") for _ in range(2)] for l in LAYERS}

    def hook(l):
        def f(mod, inp, out):
            kh = out.float().view(1, S, n, dh).transpose(1, 2)
            kr, _ = apply_rotary_pos_emb(kh, kh, cos, sin)
            k, kr = out.double().reshape(-1, D), kr.transpose(1, 2).reshape(-1, D).double()
            acc[l][0].addmm_(k.T, k)
            acc[l][1].addmm_(kr.T, kr)
        return f

    hs = [model.model.layers[l].self_attn.k_proj.register_forward_hook(hook(l)) for l in LAYERS]
    toks = calib_tokens()[:64]
    for i in range(len(toks)):
        model(toks[i : i + 1].cuda(), use_cache=False, logits_to_keep=1)
    for h in hs:
        h.remove()

    # cos/sin tiled over heads -> [S, D]; J = rotate_half as a matrix (J k = rotate_half(k))
    Cs, Sn = cos[0].double().repeat(1, n), sin[0].double().repeat(1, n)
    J = rotate_half(torch.eye(dh, dtype=torch.float64, device="cuda")).T  # column j = rotate_half(e_j)
    J = torch.block_diag(*[J] * n)
    theta = model.model.rotary_emb.inv_freq.double().cuda()
    slow = (theta * S < 2 * torch.pi)  # pairs completing < 1 turn in context
    print(f"slow pairs (θ·{S} < 2π): i >= {int(slow.nonzero().min())}  ({int(slow.sum())} of {P})")
    pair_of = torch.arange(D, device="cuda") % dh % P
    fast_coords = ~slow[pair_of]

    for l in LAYERS:
        C, Cq = (x / (len(toks) * S) for x in acc[l])
        pred = ((Cs.T @ Cs) * C + (Cs.T @ Sn) * (C @ J.T) + (Sn.T @ Cs) * (J @ C) + (Sn.T @ Sn) * (J @ C @ J.T)) / S
        rel = ((pred - Cq).norm() / Cq.norm()).item()
        Sig = complex_cov_from_real(C, n, dh).cpu()
        e_slow = (Sig.diagonal(dim1=1, dim2=2).real.sum(1)[slow.cpu()].sum() / C.trace()).item()
        print(f"\nlayer {l}: ||C_post_pred - C_post|| / ||C_post|| = {rel:.4f};  energy share of slow pairs = {e_slow:.3f}")
        for rho in [0.5, 0.25, 0.125]:
            r = int(rho * D)
            c = retained_energy(Sig, build_compressor(Sig, r // 2)) / C.trace().item()
            # same comparison restricted to fast pairs only, at the same *fraction* of their dims
            f = fast_coords
            rf = int(rho * f.sum().item())
            Sf = Sig[~slow.cpu()]
            cf = retained_energy(Sf, build_compressor(Sf, rf // 2)) / Sf.diagonal(dim1=1, dim2=2).real.sum().item()
            print(f"  rho={rho}: d_meas={topk_frac(Cq, r):.4f} d_pred={topk_frac(pred, r):.4f} c={c:.4f} | "
                  f"fast pairs only: d_meas={topk_frac(Cq[f][:, f], rf):.4f} c={cf:.4f}")


if __name__ == "__main__":
    main()
