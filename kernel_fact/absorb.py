"""Real forms (§2.3) and query absorption of the key basis (§2.4).

For a = α + jβ ∈ C^P and a real key k (z = x + j y, x = k[:P], y = k[P:]):
    Re<a, z> = u·k,  u = [α, β];      Im<a, z> = v·k,  v = [β, -α]
    Re(φ <a, z>) = Re φ (u·k) - Im φ (v·k)
Key cache c = U_r^T k (stacked heads, pre-RoPE, never rotated); head h decodes k_h ≈ B_h^T c with
B_h = U_r[h*d_h:(h+1)*d_h, :]^T ∈ R^{r x d_h}, so u·k_h ≈ (B_h u)·c.
"""
import torch


def to_complex(x):
    P = x.shape[-1] // 2
    return torch.complex(x[..., :P], x[..., P:])


def uv(a):
    """a complex [..., P] -> (u, v) real [..., 2P]."""
    return torch.cat([a.real, a.imag], -1), torch.cat([a.imag, -a.real], -1)


def head_decoders(U_r, n, dh):
    """U_r [n*dh, r] -> B [n, r, dh]."""
    return U_r.view(n, dh, -1).transpose(1, 2)


def absorbed_queries(q, Gamma, B):
    """q [n, T, dh] pre-RoPE queries; Γ [n|1, p, P]; B [n, r, dh].
    Returns ũ, ṽ [n, p, T, r]: ũ_{h,k} = B_h u(w_h ⊙ Γ_k), ṽ likewise."""
    w = to_complex(q)  # [n, T, P]
    a = w.unsqueeze(1) * Gamma.to(w.dtype).unsqueeze(2)  # [n, p, T, P]
    u, v = uv(a)  # [n, p, T, dh]
    return torch.einsum("hptd,hrd->hptr", u, B), torch.einsum("hptd,hrd->hptr", v, B)
