"""Gate 1: synthetic correctness. CPU. Every RoPE application uses the model's own code
(LlamaRotaryEmbedding.inv_freq / forward and apply_rotary_pos_emb)."""
import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (LlamaAttention, LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb)

from equivariant.compressor import (Compressor, build_compressor, complex_cov_from_real,
                                    latent_inv_freq, packing_matrix, retained_energy)
from equivariant.patch_llama import METHODS, latent_rotary, patch_model, set_method

MODEL = "/home/jl_fs/models/llama-2-7b-chat"
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def cfg():
    return LlamaConfig.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def rot(cfg):
    return LlamaRotaryEmbedding(cfg).double()


def rope(rot, x, pos):
    """Model RoPE of x [n, s, dh] at positions pos [s]."""
    cos, sin = rot(x, pos[None])
    out, _ = apply_rotary_pos_emb(x[None], x[None], cos, sin)
    return out[0]


def to_z(k):  # [..., dh] -> [..., P] complex
    P = k.shape[-1] // 2
    return torch.complex(k[..., :P], k[..., P:])


def model_angle(inv_freq, m):
    """LlamaRotaryEmbedding.forward computes m*theta in fp32 (autocast disabled, .float()); mirror it."""
    return (inv_freq.float() * float(m)).double()


def rand_psd_sigma(P, n, rank=None):
    A = torch.randn(P, n, rank or n, dtype=torch.complex128)
    return A @ A.mH


def latent_scores(B, lrot, q, k, qpos, kpos):
    """q, k: [n, s, dh] pre-RoPE. Scores [n, sq, sk] via absorbed queries vs shared latent."""
    qt = torch.einsum("hsd,hed->hse", q, B)
    c = torch.einsum("hsd,hed->se", k, B)[None]
    qc, qs = lrot(qt, qpos[None])
    kc, ks = lrot(c, kpos[None])
    qt, _ = apply_rotary_pos_emb(qt[None], qt[None], qc, qs)
    c, _ = apply_rotary_pos_emb(c[None], c[None], kc, ks)
    return qt[0] @ c[0].transpose(-1, -2)


# 1
def test_packing():
    q, k = torch.randn(128), torch.randn(128)
    assert torch.allclose(q @ k, (to_z(q) * to_z(k).conj()).real.sum())


# 2
def test_rope_is_complex_multiply(rot):
    k = torch.randn(32, 1, 128)
    for m in [0, 1, 7, 1000, 4095]:
        kr = rope(rot, k, torch.tensor([m]))
        expect = to_z(k) * torch.exp(1j * model_angle(rot.inv_freq, m))
        assert torch.allclose(to_z(kr), expect, atol=1e-10)


def random_comp(n=32, P=64, R=300):
    fi = torch.sort(torch.randint(0, P, (R,))).values
    V = torch.randn(R, n, dtype=torch.complex128)
    return Compressor(fi, V, torch.ones(R))


def equivariance_err(B, comp, rot, cfg_rot_latent, k, m):
    lhs = torch.einsum("hsd,hed->se", rope(rot, k, torch.tensor([m])), B)  # compress(RoPE_m(k))
    c = torch.einsum("hsd,hed->se", k, B)[None]
    cos, sin = cfg_rot_latent(c, torch.tensor([[m]]))
    rhs, _ = apply_rotary_pos_emb(c[None], c[None], cos, sin)  # RoPE^latent_m(compress(k))
    return (lhs - rhs[0, 0]).abs().max().item()


# 3
def test_equivariance(rot):
    comp = random_comp()
    B = packing_matrix(comp, 32, 128, torch.float64)
    lrot = latent_rotary(rot, latent_inv_freq(comp, rot.inv_freq))
    k = torch.randn(32, 1, 128)
    for m in [1, 13, 999, 4095]:
        assert equivariance_err(B, comp, rot, lrot, k, m) < 1e-9
    # packing agrees with the complex definition c_t = V_t^H z_{:, i(t)}
    c = torch.einsum("hsd,hed->se", k, B)[0]
    zc = torch.einsum("rh,hr->r", comp.V.conj(), to_z(k[:, 0])[:, comp.freq_idx])
    assert torch.allclose(torch.complex(c[: comp.R], c[comp.R:]), zc)


# 4
def test_full_rank_exact(rot):
    n, P = 32, 64
    comp = build_compressor(rand_psd_sigma(P, n), R=n * P)  # r_i = n for every frequency
    assert comp.R == 2048
    B = packing_matrix(comp, n, 128, torch.float64)
    lrot = latent_rotary(rot, latent_inv_freq(comp, rot.inv_freq))
    q, k = torch.randn(n, 5, 128), torch.randn(n, 7, 128)
    qpos, kpos = torch.randint(0, 4096, (5,)), torch.randint(0, 4096, (7,))
    ref = rope(rot, q, qpos) @ rope(rot, k, kpos).transpose(-1, -2)
    got = latent_scores(B, lrot, q, k, qpos, kpos)
    assert torch.allclose(got, ref, atol=1e-8)
    # fp32 at the brief's tolerance
    B32, l32 = B.float(), latent_rotary(rot.float(), latent_inv_freq(comp, rot.inv_freq.float()))
    got32 = latent_scores(B32, l32, q.float(), k.float(), qpos, kpos)
    assert torch.allclose(got32.double(), ref, atol=1e-4 * ref.abs().max().item())


def test_truncated_scores_match_complex_formula(rot):
    """Truncated rank: latent scores == sum_i Re(w e^{jm} conj(zhat e^{jn})) with zhat = V V^H z."""
    n, P = 32, 64
    comp = build_compressor(rand_psd_sigma(P, n), R=512)
    B = packing_matrix(comp, n, 128, torch.float64)
    lrot = latent_rotary(rot, latent_inv_freq(comp, rot.inv_freq))
    q, k = torch.randn(n, 1, 128), torch.randn(n, 1, 128)
    qpos, kpos = torch.tensor([3000]), torch.tensor([17])
    zq, zk = to_z(q[:, 0]), to_z(k[:, 0])  # [n, P]
    zhat = torch.zeros_like(zk)
    for t in range(comp.R):
        i, v = comp.freq_idx[t], comp.V[t]
        zhat[:, i] += v * (v.conj() @ zk[:, i])
    eq, ek = torch.exp(1j * model_angle(rot.inv_freq, 3000)), torch.exp(1j * model_angle(rot.inv_freq, 17))
    ref = (zq * eq * (zhat * ek).conj()).real.sum(-1)
    got = latent_scores(B, lrot, q, k, qpos, kpos)[:, 0, 0]
    assert torch.allclose(got, ref, atol=1e-8)


# 5
def test_negative_control_real_linear_breaks_equivariance(rot):
    comp = random_comp(R=64)
    B = packing_matrix(comp, 32, 128, torch.float64)
    lrot = latent_rotary(rot, latent_inv_freq(comp, rot.inv_freq))
    k = torch.randn(32, 1, 128)
    assert equivariance_err(B, comp, rot, lrot, k, 999) < 1e-9
    # channel 0: real-linear but not complex-linear -- Re from head 0's x, Im from head 1's x
    Bbad = B.clone()
    t, i, R = 0, comp.freq_idx[0].item(), comp.R
    Bbad[:, t], Bbad[:, R + t] = 0, 0
    Bbad[0, t, i] = 1.0
    Bbad[1, R + t, i] = 0.7
    assert equivariance_err(Bbad, comp, rot, lrot, k, 999) > 1e-3


# 6
def test_energy_ordering():
    for _ in range(20):
        S = rand_psd_sigma(1, 16)[0]
        ec = torch.linalg.eigvalsh(S).flip(0)
        eb = torch.linalg.eigvalsh(S.real).flip(0)
        for r in range(1, 17):
            assert ec[:r].sum() >= eb[:r].sum() - 1e-9
    S = rand_psd_sigma(64, 32)
    for R in [64, 256, 512, 1024]:
        c, b = build_compressor(S, R), build_compressor(S, R, real=True)
        bp, cp = build_compressor(S, R, real=True, allocation="uniform"), build_compressor(S, R, allocation="uniform")
        E = {k: retained_energy(S, x) for k, x in dict(c=c, b=b, bp=bp, cp=cp).items()}
        assert E["c"] >= E["b"] - 1e-6 and E["c"] >= E["cp"] - 1e-6 and E["b"] >= E["bp"] - 1e-6
        assert E["cp"] >= E["bp"] - 1e-6
        assert math.isclose(E["c"], c.eig.sum().item(), rel_tol=1e-9)  # eig = retained energy
        assert math.isclose(E["b"], b.eig.sum().item(), rel_tol=1e-9)  # Im part adds nothing for real V


def test_global_allocation_is_optimal():
    """Global top-R pooling == best split of R across frequencies (brute force on a small case)."""
    S = rand_psd_sigma(3, 4)
    ev = torch.linalg.eigvalsh(S).flip(-1)
    for R in range(1, 13):
        best = max(sum(ev[i, :r].sum().item() for i, r in enumerate(rs))
                   for rs in [(a, b, R - a - b) for a in range(5) for b in range(5) if 0 <= R - a - b <= 4])
        assert math.isclose(retained_energy(S, build_compressor(S, R)), best, rel_tol=1e-9)


# 7
def test_scale(cfg):
    attn = LlamaAttention(cfg, 0)
    assert math.isclose(attn.scaling, 1 / math.sqrt(128), rel_tol=1e-12)


# 8
def test_layout(rot):
    for i in [0, 5, 63]:
        for m in [1, 100]:
            e = torch.zeros(1, 1, 128)
            e[0, 0, i] = 1
            out = rope(rot, e, torch.tensor([m]))[0, 0]
            th = model_angle(rot.inv_freq, m)[i].item()
            expect = torch.zeros(128)
            expect[i], expect[i + 64] = math.cos(th), math.sin(th)
            assert torch.allclose(out, expect, atol=1e-10)


def test_complex_cov_from_real():
    n, dh, N = 4, 16, 50
    K = torch.randn(N, n, dh)
    C = K.reshape(N, -1).T @ K.reshape(N, -1) / N
    z = to_z(K)  # [N, n, P]
    direct = torch.einsum("thi,tgi->ihg", z, z.conj()) / N
    assert torch.allclose(complex_cov_from_real(C, n, dh), direct, atol=1e-12)


# ---- whole-model tests on a tiny random Llama (patch_llama + attention + cache paths) ----

@pytest.fixture(scope="module")
def tiny():
    torch.set_default_dtype(torch.float32)
    c = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=2,
                    intermediate_size=128, vocab_size=100, max_position_embeddings=512,
                    attn_implementation="eager")
    torch.manual_seed(1)
    m = LlamaForCausalLM(c).eval()
    n, dh = 4, 16
    stats = []
    for _ in range(2):
        A = torch.randn(n * dh, n * dh, dtype=torch.float64)
        C = A @ A.T
        U = torch.linalg.eigh(C).eigenvectors.flip(-1).float()
        stats.append(dict(Sigma=complex_cov_from_real(C, n, dh), U_pre=U, U_post=U))
    ids = torch.randint(0, 100, (2, 40))
    with torch.no_grad():
        ref = m(ids).logits
    patch_model(m)
    yield m, stats, ids, ref
    torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("method", METHODS)
def test_tiny_full_rank_equals_reference(tiny, method):
    """rho=1 (2R = n*dh = 64 != d_h = 16): every arm must reproduce the unpatched logits. For the
    latent arms this also pins the softmax scale to 1/sqrt(d_h) (1/sqrt(2R) would differ)."""
    m, stats, ids, ref = tiny
    if method in ("bp", "cp"):
        pytest.skip("uniform allocation at full rank is identical to global")
    set_method(m, method, rho=1.0, stats=stats)
    with torch.no_grad():
        got = m(ids).logits
    assert (got - ref).abs().max() < 1e-4, method


@pytest.mark.parametrize("method", ["c", "b", "a", "d"])
def test_tiny_truncated_differs(tiny, method):
    m, stats, ids, ref = tiny
    set_method(m, method, rho=0.25, stats=stats)
    with torch.no_grad():
        got = m(ids).logits
    assert (got - ref).abs().max() > 1e-3  # truncation actually does something


@pytest.mark.parametrize("method", ["c", "bp", "a", "off"])
def test_tiny_cached_decode_matches_full_forward(tiny, method):
    """Latent cached already rotated, new token rotated by its own absolute position."""
    m, stats, ids, _ = tiny
    set_method(m, method, rho=0.25, stats=stats)
    with torch.no_grad():
        full = m(ids).logits
        cache = DynamicCache()
        out = m(ids[:, :30], past_key_values=cache, use_cache=True)
        steps = [out.logits]
        for t in range(30, 40):
            out = m(ids[:, t: t + 1], past_key_values=out.past_key_values, use_cache=True)
            steps.append(out.logits)
    inc = torch.cat(steps, 1)
    assert (inc - full).abs().max() < 1e-4
    if method == "c":
        k0 = out.past_key_values.layers[0].keys if hasattr(out.past_key_values, "layers") else out.past_key_values.key_cache[0]
        assert k0.shape == (2, 1, 40, 16)  # [b, 1 shared latent, seq, 2R], R = 0.25*64/2 = 8
