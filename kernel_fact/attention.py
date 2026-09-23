"""Scores through a factorized RoPE kernel, and the eager attention that uses them.

Three algebraically identical evaluations of s_h(q, j) = Re Σ_k Φ[Δ,k] <w_q ⊙ Γ_k, z_j>,
Δ = qpos - kpos (query minus key position). Positions with Δ < 0 are never read from Φ; their
scores are -inf (causal).
  absorbed_scores : the method as specified (§2.4 boxed): 2p real dots of length r against the
                    cached latent c, with ũ = B_h u, ṽ = B_h v absorbed into the query.
  factor_scores   : Σ_k Φ[Δ,k] ((w ⊙ Γ_k) ẑ^H) on the decoded keys ẑ = B_h^T c (d_h-length dots,
                    cheaper in eager PyTorch; used for evaluation).
  kernel_scores   : Re Σ_i w_i conj(ẑ_i) K̃[Δ,i] with K̃ = ΦΓ (gather form; Gate 2).
Tests assert all three agree, and that Γ = I, Φ = E reproduces the model's own RoPE scores.
"""
import torch

from .absorb import absorbed_queries, to_complex


def gaps(qpos, kpos):
    d = qpos[:, None] - kpos[None, :]
    return d.clamp_min(0), d < 0


def _gather_phi(Phi, D):
    """Phi [n|1, L, p], D [Tq, Tk] -> [n|1, p, Tq, Tk]."""
    return Phi[:, D].permute(0, 3, 1, 2)


def absorbed_scores(q, c, B, Phi, Gamma, qpos, kpos):
    """q [n, Tq, dh]; c [Tk, r]; B [n, r, dh]; Φ [n|1, L, p]; Γ [n|1, p, P]. -> [n, Tq, Tk]."""
    D, neg = gaps(qpos, kpos)
    ut, vt = absorbed_queries(q, Gamma, B)  # [n, p, Tq, r]
    Ud, Vd = ut @ c.T, vt @ c.T  # [n, p, Tq, Tk]
    Ph = _gather_phi(Phi, D)
    s = (Ph.real * Ud - Ph.imag * Vd).sum(1)
    return s.masked_fill(neg, float("-inf"))


def factor_scores(q, khat, Phi, Gamma, qpos, kpos, chunk=1024):
    """q [n, Tq, dh], khat [n, Tk, dh] (decoded pre-RoPE keys). Query-chunked. -> [n, Tq, Tk] fp32."""
    up = lambda x: x if x.dtype in (torch.float32, torch.float64) else x.float()
    w, z = to_complex(up(q)), to_complex(up(khat))
    cd = torch.complex64 if w.dtype == torch.complex64 else torch.complex128
    Phi, Gamma = Phi.to(cd), Gamma.to(cd)
    n, Tq = w.shape[:2]
    out = torch.empty(n, Tq, z.shape[1], dtype=w.real.dtype, device=w.device)
    zH = z.conj().mT  # [n, P, Tk]
    for s0 in range(0, Tq, chunk):
        sl = slice(s0, s0 + chunk)
        D, neg = gaps(qpos[sl], kpos)
        acc = torch.zeros(n, D.shape[0], D.shape[1], dtype=cd, device=w.device)
        for k in range(Gamma.shape[-2]):
            G = (w[:, sl] * Gamma[:, k].unsqueeze(1)) @ zH  # [n, tq, Tk]
            acc += Phi[:, D, k] * G
        out[:, sl] = acc.real.masked_fill(neg, float("-inf"))
    return out


def _contiguous(pos):
    return bool((pos[1:] - pos[:-1] == 1).all()) if len(pos) > 1 else True


def factor_scores_fast(q, khat, Phi, Gamma, qpos, kpos, chunk=1024):
    """Same function as factor_scores, organised for speed (Gate 4):
      - real form (§2.3): Re/Im <w ⊙ Γ_k, ẑ> = u_k · k̂ / v_k · k̂, two real matmuls per factor
        (tensor cores when TF32 is enabled by the caller);
      - when query and key positions are contiguous ranges (prefill), Φ_k[q - j] is Toeplitz: with the
        query block reversed it is a zero-copy as_strided view of a 1-D slice of Φ_k (no gather).
    Falls back to an index gather otherwise (decode, padding)."""
    from .absorb import uv

    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, khat = q.to(dt), khat.to(dt)
    w = to_complex(q)
    n, Tq, _ = q.shape
    Tk = khat.shape[1]
    L = Phi.shape[-2]
    PhR = Phi.real.to(dt).transpose(-1, -2).contiguous()  # [n|1, p, L]
    PhI = Phi.imag.to(dt).transpose(-1, -2).contiguous()
    kT = khat.transpose(-1, -2)  # [n, dh, Tk]
    toeplitz = _contiguous(qpos) and _contiguous(kpos)
    out = torch.empty(n, Tq, Tk, dtype=dt, device=q.device)
    for s0 in range(0, Tq, chunk):
        sl = slice(s0, s0 + chunk)
        tq = w[:, sl].shape[1]
        D, neg = gaps(qpos[sl], kpos)
        acc = torch.zeros(n, tq, Tk, dtype=dt, device=q.device)
        if toeplitz:
            # rows in reversed order a' = tq-1-a: Φ[c + a - b] = ψ[a' + b], ψ[t] = Φ[c + tq - 1 - t]
            c = int(qpos[s0] - kpos[0])
            t = torch.arange(tq + Tk - 1, device=q.device)
            idx = (c + tq - 1 - t).clamp(0, L - 1)
        for k in range(Gamma.shape[-2]):
            a = w[:, sl] * Gamma[:, k].to(w.dtype).unsqueeze(1)  # [n, tq, P]
            u, v = uv(a)  # [n, tq, dh]
            if toeplitz:
                u, v = u.flip(1), v.flip(1)
            Su, Sv = u @ kT, v @ kT  # [n, tq, Tk]
            if toeplitz:
                pr, pi = PhR[:, k, idx], PhI[:, k, idx]  # [n|1, tq+Tk-1]
                st = pr.stride()
                pr = pr.as_strided((pr.shape[0], tq, Tk), (st[0], 1, 1))
                pi = pi.as_strided((pi.shape[0], tq, Tk), (st[0], 1, 1))
            else:
                pr, pi = PhR[:, k][:, D], PhI[:, k][:, D]
            acc += pr * Su - pi * Sv
        if toeplitz:
            acc = acc.flip(1)
        out[:, sl] = acc.masked_fill(neg, float("-inf"))
    return out


def kernel_scores(q, khat, Ktil, qpos, kpos):
    """Ktil [n|1, L, P] = ΦΓ. -> [n, Tq, Tk]."""
    D, neg = gaps(qpos, kpos)
    w, z = to_complex(q), to_complex(khat)
    M = w[:, :, None, :] * z.conj()[:, None, :, :]  # [n, Tq, Tk, P]
    s = (M * Ktil[:, D]).real.sum(-1)
    return s.masked_fill(neg, float("-inf"))
