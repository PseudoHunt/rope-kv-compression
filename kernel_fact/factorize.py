"""Low-rank factorizations of the RoPE positional kernel E[Δ, i] = exp(j Δ θ_i).

Score of head h at gap Δ = m - n >= 0 (pre-RoPE query w, key z, complex pair form):
    s(Δ) = Re Σ_i w_i conj(z_i) E[Δ, i]  ≈  Re Σ_k Φ[Δ, k] <w ⊙ Γ_k, z>,   E ≈ Φ Γ.
Every constructor returns (Φ [..., L, p], Γ [..., p, P]) in complex128. A leading batch dim
(e.g. heads) is carried through when π / ε have one.
"""
import torch


def kernel_E(inv_freq, L):
    """E [L, P] complex128. θ are the model's inv_freq values (fp32) promoted to fp64."""
    d = torch.arange(L, dtype=torch.float64)[:, None]
    return torch.exp(1j * d * inv_freq.double().cpu()[None, :])


def regularize(pi, eps, eta=1e-3):
    """π' = (1-η) π/Σπ + η/L ;  ε floored at 1e-8 max ε. Works on [..., L] / [..., P]."""
    pi = pi.double()
    L = pi.shape[-1]
    pi = (1 - eta) * pi / pi.sum(-1, keepdim=True) + eta / L
    eps = eps.double()
    eps = torch.maximum(eps, 1e-8 * eps.amax(-1, keepdim=True))
    return pi, eps


def ls_refit(E, Gamma, eps, ridge=1e-8):
    """ε-weighted LS for every row of E given Γ: Φ = E D Γ^H (Γ D Γ^H + λI)^{-1}, λ relative."""
    D = eps.to(torch.complex128)
    G = (Gamma * D.unsqueeze(-2)) @ Gamma.mH  # [..., p, p]
    lam = ridge * G.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
    G = G + lam * torch.eye(G.shape[-1], dtype=G.dtype)
    rhs = (E * D.unsqueeze(-2)) @ Gamma.mH  # [..., L, p]
    return torch.linalg.solve(G.mT, rhs.mT).mT  # Φ G = rhs  <=>  G^T Φ^T = rhs^T


def opt_multi(inv_freq, pi, eps, ps, L_out=None, eta=1e-3):
    """Weighted-SVD optimum of Σ_Δ π'(Δ) Σ_i ε_i |E - ΦΓ|^2 (rank-1 weights -> exact via SVD),
    with Φ then LS-refit for every gap 0..L_out-1 (L_out may exceed the calibration window).
    One SVD serves every rank in `ps`. Returns {p: (Φ, Γ)}."""
    pi, eps = regularize(pi, eps, eta)
    L = pi.shape[-1]
    E = kernel_E(inv_freq, L)
    W = pi.sqrt().unsqueeze(-1) * E * eps.sqrt().unsqueeze(-2)
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
    E_out = E if (L_out or L) == L else kernel_E(inv_freq, L_out)
    out = {}
    for p in ps:
        Gamma = Vh[..., :p, :] / eps.sqrt().unsqueeze(-2)
        out[p] = (ls_refit(E_out, Gamma, eps), Gamma)
    return out


def opt(inv_freq, pi, eps, p, L_out=None, eta=1e-3):
    return opt_multi(inv_freq, pi, eps, [p], L_out, eta)[p]


def opt_unweighted(inv_freq, p, L, n_heads=None, ps=None):
    """ps given -> {p: (Φ, Γ)} from one SVD; else the single rank p."""
    shape = () if n_heads is None else (n_heads,)
    P = inv_freq.shape[0]
    res = opt_multi(inv_freq, torch.ones(*shape, L), torch.ones(*shape, P), ps or [p], eta=0.0)
    return res if ps else res[p]


def exact(inv_freq, L):
    P = inv_freq.shape[0]
    return kernel_E(inv_freq, L), torch.eye(P, dtype=torch.complex128)


def nope(P, L):
    return torch.ones(L, 1, dtype=torch.complex128), torch.ones(1, P, dtype=torch.complex128)


def partial_rope(inv_freq, keep, L):
    """keep: [..., p-1] long pair indices that keep their own rotation; every other pair is NoPE
    (shares one constant column Φ = 1). p = keep.shape[-1] + 1."""
    P = inv_freq.shape[0]
    batch, k1 = keep.shape[:-1], keep.shape[-1]
    Gamma = torch.zeros(*batch, k1 + 1, P, dtype=torch.complex128)
    Gamma.scatter_(-1, keep.unsqueeze(-1), 1.0)
    Gamma[..., k1, :] = 1.0 - Gamma[..., :k1, :].sum(-2)  # indicator of the NoPE pairs
    d = torch.arange(L, dtype=torch.float64)
    theta = inv_freq.double().cpu()[keep]  # [..., p-1]
    Phi = torch.exp(1j * d[:, None] * theta.unsqueeze(-2))  # [..., L, p-1]
    Phi = torch.cat([Phi, torch.ones(*batch, L, 1, dtype=torch.complex128)], -1)
    return Phi, Gamma


def pr_hi(inv_freq, p, L):
    """MHA2MLA S_high: the p-1 fastest pairs (smallest i; θ decreases with i), same for all heads."""
    return partial_rope(inv_freq, torch.arange(p - 1), L)


def pr_select(inv_freq, score, p, L):
    """Per-head top-(p-1) pairs by `score` [..., P] (ε for PR-en; E|w|·E|z| for PR-2n)."""
    keep = torch.sort(torch.topk(score.double(), p - 1, dim=-1).indices, -1).values
    return partial_rope(inv_freq, keep, L)


def fold_groups(eps, p):
    """Contiguous groups of pairs with (approximately) equal total ε. [..., P] -> group id [..., P]."""
    e = eps.double()
    mid = e.cumsum(-1) - e / 2
    return torch.clamp((p * mid / e.sum(-1, keepdim=True)).floor().long(), max=p - 1)


def fold_mean(inv_freq, eps, p, L):
    """FOLD-mean: group g uses one column exp(j Δ θ~_g), θ~_g = ε-weighted mean θ of the group.
    (Not TransMLA's published FreqFold, which picks grid frequencies after a per-group PCA.)"""
    P = inv_freq.shape[0]
    g = fold_groups(eps, p)
    Gamma = torch.nn.functional.one_hot(g, p).mT.to(torch.complex128)  # [..., p, P]
    e = eps.double()
    th = inv_freq.double().cpu()
    num = (Gamma.real * (e * th).unsqueeze(-2)).sum(-1)
    den = (Gamma.real * e.unsqueeze(-2)).sum(-1)
    theta_g = num / den.clamp_min(1e-300)  # empty groups: θ~ = 0, Γ row = 0 (inert)
    d = torch.arange(L, dtype=torch.float64)
    Phi = torch.exp(1j * d[:, None] * theta_g.unsqueeze(-2))
    return Phi, Gamma


def weighted_residual(Phi, Gamma, inv_freq, pi, eps, eta=1e-3):
    """Surrogate Σ_Δ π'(Δ) Σ_i ε_i |E - ΦΓ|^2 over the calibration window."""
    pi, eps = regularize(pi, eps, eta)
    L = pi.shape[-1]
    R = kernel_E(inv_freq, L) - Phi[..., :L, :] @ Gamma
    return (pi.unsqueeze(-1) * eps.unsqueeze(-2) * R.abs() ** 2).sum((-2, -1))
