"""RoPE-equivariant key compressors: per-frequency cross-head (complex or real) PCA.

Notation (HF Llama `rotate_half` layout): head dim d_h, P = d_h/2 pairs; pair i of head h is the
coordinate pair (i, i+P) and is represented as z_{h,i} = k_h[i] + j k_h[i+P].

A compressor is a list of R latent channels. Channel t has a source frequency i(t) and a vector
V[t] in C^n (n = heads); its latent value is c_t = V[t]^H z_{:, i(t)}. Because RoPE multiplies all
of z_{:, i} by the same scalar e^{j m theta_i}, c_t rotates by e^{j m theta_{i(t)}}: the latent can
be RoPE'd with a custom inv_freq table and cached already rotated.

Real packing of the latent ("virtual head", width 2R, rotate_half layout): slot t = Re c_t,
slot t+R = Im c_t. Both key and query absorption are the same real map per head:
    B_h in R^{2R x d_h},  c = sum_h B_h k_h,   q~_h = B_h q_h
since conj(V)·z and w·conj(V) have identical real 2x2 forms. Then q~_h · c = sum_t Re(q~_{h,t} conj(c_t))
approximates q_h · k_h.
"""
from dataclasses import dataclass

import torch


def complex_cov_from_real(C, n, dh):
    """Per-frequency complex cross-head covariance from the real key covariance.

    C: [n*dh, n*dh] uncentered E[k k^T] of stacked pre-RoPE keys (head-major).
    Returns Sigma: [P, n, n] complex, Sigma_i[h,g] = E[z_{h,i} conj(z_{g,i})]
                 = E[x_h x_g + y_h y_g] + j E[y_h x_g - x_h y_g],  x = k[i], y = k[i+P].
    """
    P = dh // 2
    C4 = C.view(n, dh, n, dh)
    # torch.diagonal over the two within-head coordinate axes -> [n, n, P], then -> [P, n, n]
    d = lambda a, b: torch.diagonal(C4[:, a : a + P, :, b : b + P], dim1=1, dim2=3).permute(2, 0, 1)
    xx, yy, yx, xy = d(0, 0), d(P, P), d(P, 0), d(0, P)
    S = torch.complex(xx + yy, yx - xy)
    return (S + S.mH) / 2


@dataclass
class Compressor:
    freq_idx: torch.Tensor  # [R] long, source pair of each latent channel
    V: torch.Tensor  # [R, n] complex, channel vectors (orthonormal within a frequency)
    eig: torch.Tensor  # [R] float, calibration energy of each channel (eigenvalue)

    @property
    def R(self):
        return len(self.freq_idx)


def build_compressor(Sigma, R, real=False, allocation="global"):
    """Per-frequency PCA of Sigma [P, n, n].

    real=False -> method (c): eigenvectors of Sigma (complex PCA).
    real=True  -> method (b): eigenvectors of Re(Sigma), used as a real orthogonal cross-head mix
                 applied identically to both coordinates of the pair (RoRoPE constraint).
    allocation='global'  -> pool all P*n eigenvalues, keep the top R (ties broken by (freq, index)).
    allocation='uniform' -> R/P channels per frequency (RoRoPE-as-published keeps a fixed m per
                 frequency: this is (b') for real=True, (c') for real=False).
    """
    Sigma = Sigma.to(torch.complex128)
    P, n, _ = Sigma.shape
    M = torch.complex(Sigma.real, torch.zeros_like(Sigma.real)) if real else Sigma
    evals, evecs = torch.linalg.eigh(M)  # ascending
    evals, evecs = evals.flip(-1), evecs.flip(-1)  # [P, n], [P, n(coord), n(component)]
    if allocation == "global":
        order = torch.sort(-evals.reshape(-1), stable=True).indices[:R]
    elif allocation == "uniform":
        assert R % P == 0 and R // P <= n
        m = R // P
        order = (torch.arange(P)[:, None] * n + torch.arange(m)[None, :]).reshape(-1)
    else:
        raise ValueError(allocation)
    order = torch.sort(order).values  # channels ordered by (frequency, component)
    fi, ci = order // n, order % n
    V = evecs[fi, :, ci]  # [R, n]
    return Compressor(freq_idx=fi, V=V, eig=evals[fi, ci])


def retained_energy(Sigma, comp):
    """sum_t V_t^H Sigma_{i(t)} V_t (real energy captured, same units as tr E[k k^T])."""
    S = Sigma.to(torch.complex128)[comp.freq_idx]  # [R, n, n]
    V = comp.V.to(torch.complex128)
    return torch.einsum("rh,rhg,rg->r", V.conj(), S, V).real.sum().item()


def rank_alloc(comp, P):
    return torch.bincount(comp.freq_idx, minlength=P)


def packing_matrix(comp, n, dh, dtype=torch.float32):
    """B: [n, 2R, dh] real map. latent c = sum_h B[h] @ k_h ; absorbed query q~_h = B[h] @ q_h.

    conj(v) (x + j y) = (vr x + vi y) + j (vr y - vi x)  -> rows t (Re) and t+R (Im).
    (x + j y) conj(v) is the same expression, so one matrix serves keys and queries.
    """
    P = dh // 2
    R = comp.R
    vr, vi = comp.V.real.T.to(dtype), comp.V.imag.T.to(dtype)  # [n, R]
    t, i = torch.arange(R), comp.freq_idx
    B = torch.zeros(n, 2 * R, dh, dtype=dtype)
    B[:, t, i] = vr
    B[:, t, i + P] = vi
    B[:, R + t, i] = -vi
    B[:, R + t, i + P] = vr
    return B


def latent_inv_freq(comp, inv_freq):
    """Custom inv_freq of length R: channel t rotates at its source pair's frequency."""
    return inv_freq[comp.freq_idx.to(inv_freq.device)]
