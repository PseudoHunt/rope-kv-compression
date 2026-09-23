"""All arms for one layer: name -> (Φ [n|1, L, p], Γ [n|1, p, P]) complex128.

EXACT     Γ = I, Φ = E                         (reconstruct + RoPE; previous task's arm (a))
OPT       per-head weighted SVD, π and ε = E|w|^2 E|z|^2
OPT-unw   π uniform, ε ≡ 1 (one factorization for all heads)
PR-hi     p-1 fastest pairs keep RoPE, rest NoPE (MHA2MLA S_high, training-free)
PR-en     per-head top-(p-1) pairs by ε (brief's energy selection)
PR-2n     per-head top-(p-1) pairs by E|w|·E|z| (MHA2MLA S_2-norm score, training-free)
FOLD-mean contiguous equal-ε groups, ε-weighted mean θ per group (not TransMLA's published FreqFold)
NOPE      p = 1, no RoPE
ablations OPT-joint (ε = attention-weighted joint E[|w|^2 |z|^2]); OPT-layer (π, ε shared per layer)
"""
from . import factorize as F

MAIN = ["OPT", "OPT-unw", "PR-hi", "PR-en", "PR-2n", "FOLD-mean"]
ABLATIONS = ["OPT-joint", "OPT-layer"]


def layer_arms(inv_freq, Wt, l, ps, L=4096, ablations=True):
    pi, eps = Wt["pi"][l], Wt["w2"][l] * Wt["z2"][l]  # [n, L], [n, P]
    n, P = eps.shape
    b = lambda t: t if t[0].dim() == 3 else (t[0][None], t[1][None])
    arms = {"EXACT": b(F.exact(inv_freq, L)), "NOPE": b(F.nope(P, L))}
    o = F.opt_multi(inv_freq, pi, eps, ps, L_out=L)
    ou = F.opt_unweighted(inv_freq, None, L, ps=ps)
    if ablations:
        oj = F.opt_multi(inv_freq, pi, Wt["joint"][l], ps, L_out=L)
        ol = F.opt_multi(inv_freq, pi.mean(0), eps.mean(0), ps, L_out=L)
    for p in ps:
        arms[f"OPT/{p}"] = o[p]
        arms[f"OPT-unw/{p}"] = b(ou[p])
        arms[f"PR-hi/{p}"] = b(F.pr_hi(inv_freq, p, L))
        arms[f"PR-en/{p}"] = F.pr_select(inv_freq, eps, p, L)
        arms[f"PR-2n/{p}"] = F.pr_select(inv_freq, Wt["wa"][l] * Wt["za"][l], p, L)
        arms[f"FOLD-mean/{p}"] = F.fold_mean(inv_freq, eps, p, L)
        if ablations:
            arms[f"OPT-joint/{p}"] = oj[p]
            arms[f"OPT-layer/{p}"] = b(ol[p])
    return arms


def split(name):
    arm, _, p = name.partition("/")
    return arm, (int(p) if p else {"EXACT": 64, "NOPE": 1}[arm])
