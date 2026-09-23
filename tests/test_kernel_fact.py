"""Gate 1 (kernel factorization): synthetic correctness. CPU. Every reference path uses the model's own
LlamaRotaryEmbedding / apply_rotary_pos_emb; baselines are checked against direct implementations
(the model's RoPE with a modified inv_freq)."""
import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (LlamaAttention, LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb)

from equivariant.patch_llama import latent_rotary
from kernel_fact import factorize as F
from kernel_fact.absorb import head_decoders, to_complex, uv
from kernel_fact.attention import absorbed_scores, factor_scores, kernel_scores
from kernel_fact.patch_llama import KFState, patch_model, set_states

MODEL = "/home/jl_fs/models/llama-2-7b-chat"
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
n, dh, P, L = 32, 128, 64, 4096


@pytest.fixture(scope="module")
def rot():
    return LlamaRotaryEmbedding(LlamaConfig.from_pretrained(MODEL)).double()


def rope(rot, x, pos):
    """Model RoPE of x [n, s, dh] at positions pos [s]."""
    cos, sin = rot(x, pos[None])
    out, _ = apply_rotary_pos_emb(x[None], x[None], cos, sin)
    return out[0]


def ref_scores(rot, q, k, qpos, kpos):
    s = rope(rot, q, qpos) @ rope(rot, k, kpos).transpose(-1, -2)
    return s.masked_fill(qpos[:, None] < kpos[None, :], float("-inf"))


def close(a, b, rel=1e-4):
    fin = torch.isfinite(b)
    assert torch.equal(fin, torch.isfinite(a))
    err = (a[fin] - b[fin]).abs().max().item()
    assert err <= rel * b[fin].abs().max().item(), err


def rand_setup(Tq=6, Tk=9):
    q, k = torch.randn(n, Tq, dh), torch.randn(n, Tk, dh)
    kpos = torch.sort(torch.randperm(4000)[:Tk]).values
    qpos = torch.cat([kpos[-3:], torch.randint(4000, 4096, (Tq - 3,))])  # includes Δ = 0 and Δ < 0
    return q, k, qpos, kpos


def rand_weights(h=n):
    pi = torch.rand(h, L) * torch.exp(-torch.arange(L) / torch.randint(10, 2000, (h, 1)))
    eps = torch.rand(h, P) ** 3 + 1e-3
    return pi, eps


# 1
def test_real_conversion():
    for _ in range(10):
        a = torch.randn(P, dtype=torch.complex128)
        k = torch.randn(dh)
        phi = torch.randn((), dtype=torch.complex128)
        lhs = (phi * (a * to_complex(k).conj()).sum()).real
        u, v = uv(a)
        assert torch.allclose(lhs, phi.real * (u @ k) - phi.imag * (v @ k))


# 2
def test_exact_endpoint_no_compression(rot):
    q, k, qpos, kpos = rand_setup()
    Phi, Gamma = F.exact(rot.inv_freq, L)
    I = torch.eye(n * dh)
    c = k.transpose(0, 1).reshape(-1, n * dh)  # [Tk, n*dh], B_h = identity slice
    ref = ref_scores(rot, q, k, qpos, kpos)
    close(absorbed_scores(q, c, head_decoders(I, n, dh), Phi[None], Gamma[None], qpos, kpos), ref)
    close(factor_scores(q, k, Phi[None], Gamma[None], qpos, kpos), ref)
    close(kernel_scores(q, k, (Phi @ Gamma)[None], qpos, kpos), ref)
    # fp32 at the brief's tolerance
    q32, k32 = q.float(), k.float()
    ref32 = ref_scores(rot.float(), q32, k32, qpos, kpos)
    close(factor_scores(q32, k32, Phi[None].to(torch.complex64), Gamma[None].to(torch.complex64), qpos, kpos),
          ref32.double())


# 3
def test_exact_endpoint_with_key_compression(rot):
    q, k, qpos, kpos = rand_setup()
    U = torch.linalg.qr(torch.randn(n * dh, 1024)).Q
    c = k.transpose(0, 1).reshape(-1, n * dh) @ U  # cache
    khat = (c @ U.T).view(-1, n, dh).transpose(0, 1)
    ref = ref_scores(rot, q, khat, qpos, kpos)  # reconstruct-then-RoPE = arm (a)
    Phi, Gamma = F.exact(rot.inv_freq, L)
    close(absorbed_scores(q, c, head_decoders(U, n, dh), Phi[None], Gamma[None], qpos, kpos), ref)


# 4
def test_weighted_svd_full_rank_is_exact(rot):
    pi, eps = rand_weights(4)
    Phi, Gamma = F.opt(rot.inv_freq, pi, eps, p=P, L_out=6000)  # beyond the calibration window too
    E = F.kernel_E(rot.inv_freq, 6000)
    assert (Phi @ Gamma - E).abs().max() < 1e-7  # LS ridge (1e-8 relative) biases by exactly ~1e-8
    q, k, qpos, kpos = rand_setup()
    q, k = q[:4], k[:4]
    close(factor_scores(q, k, Phi, Gamma, qpos, kpos), ref_scores(rot, q, k, qpos, kpos))


def direct_with_inv_freq(rot, inv_freq_h, q, k, qpos, kpos):
    """Direct implementation: the model's RoPE with a per-head modified inv_freq table."""
    out = []
    for h in range(q.shape[0]):
        r = latent_rotary(rot, inv_freq_h[h])
        out.append(ref_scores(r, q[h:h + 1], k[h:h + 1], qpos, kpos)[0])
    return torch.stack(out)


# 5
@pytest.mark.parametrize("p", [4, 16])
def test_baselines_match_direct_implementations(rot, p):
    H = 3
    q, k, qpos, kpos = rand_setup()
    q, k = q[:H], k[:H]
    _, eps = rand_weights(H)
    th = rot.inv_freq.double()
    # PR-hi: RoPE on pairs 0..p-2, θ = 0 (NoPE) elsewhere
    Phi, Gamma = F.pr_hi(rot.inv_freq, p, L)
    mod = th.clone()
    mod[p - 1:] = 0
    close(factor_scores(q, k, Phi[None], Gamma[None], qpos, kpos),
          direct_with_inv_freq(rot, mod.expand(H, -1), q, k, qpos, kpos), rel=1e-4)  # model RoPE computes m·θ in fp32
    # PR-en: per-head top-ε pairs keep RoPE
    Phi, Gamma = F.pr_select(rot.inv_freq, eps, p, L)
    keep = torch.topk(eps, p - 1, -1).indices
    mod = torch.zeros(H, P).scatter(-1, keep, th.expand(H, -1).gather(-1, keep))
    close(factor_scores(q, k, Phi, Gamma, qpos, kpos), direct_with_inv_freq(rot, mod, q, k, qpos, kpos), rel=1e-4)  # model RoPE computes m·θ in fp32
    # FOLD-mean: every pair's θ replaced by its group's ε-weighted mean θ
    Phi, Gamma = F.fold_mean(rot.inv_freq, eps, p, L)
    g = F.fold_groups(eps, p)
    mod = torch.zeros(H, P)
    for h in range(H):
        for gi in g[h].unique():
            m = g[h] == gi
            mod[h, m] = (eps[h, m] * th[m]).sum() / eps[h, m].sum()
    close(factor_scores(q, k, Phi, Gamma, qpos, kpos), direct_with_inv_freq(rot, mod, q, k, qpos, kpos), rel=1e-4)  # model RoPE computes m·θ in fp32


# 6
def test_opt_is_optimal_on_surrogate(rot):
    pi, eps = rand_weights(6)
    res = F.weighted_residual
    prev = None
    for p in [2, 4, 8, 16, 32, 48]:
        Phi, Gamma = F.opt(rot.inv_freq, pi, eps, p)
        r_opt = res(Phi, Gamma, rot.inv_freq, pi, eps)
        for Phb, Gb in [F.pr_hi(rot.inv_freq, p, L), F.pr_select(rot.inv_freq, eps, p, L),
                        F.fold_mean(rot.inv_freq, eps, p, L), F.opt_unweighted(rot.inv_freq, p, L)]:
            Phb = Phb if Phb.dim() == 3 else Phb[None]
            Gb = Gb if Gb.dim() == 3 else Gb[None]
            assert (r_opt <= res(Phb, Gb, rot.inv_freq, pi, eps) * (1 + 1e-9)).all()
        if prev is not None:
            assert (r_opt <= prev * (1 + 1e-9)).all()
        prev = r_opt
    # Eckart-Young: residual = sum of discarded squared singular values of W
    p_ = 8
    pr, er = F.regularize(pi, eps)
    W = pr.sqrt()[..., None] * F.kernel_E(rot.inv_freq, L) * er.sqrt()[..., None, :]
    sv = torch.linalg.svdvals(W)
    Phi, Gamma = F.opt(rot.inv_freq, pi, eps, p_)
    assert torch.allclose(res(Phi, Gamma, rot.inv_freq, pi, eps), (sv[:, p_:] ** 2).sum(-1), rtol=1e-8)


# 7
def test_ls_refit_finite_on_unseen_gaps(rot):
    pi = torch.zeros(2, L)
    pi[:, :100] = torch.rand(2, 100)
    eps = torch.rand(2, P) + 1e-3
    for eta in [1e-3, 0.0]:
        Phi, Gamma = F.opt(rot.inv_freq, pi, eps, 16, L_out=16384, eta=eta)
        assert torch.isfinite(Phi).all() and torch.isfinite(Gamma).all()
    # refit leaves the SVD solution unchanged on the calibration window (Γ D Γ^H = I by construction)
    pr, er = F.regularize(pi, eps)
    W = pr.sqrt()[..., None] * F.kernel_E(rot.inv_freq, L) * er.sqrt()[..., None, :]
    U_, S_, Vh = torch.linalg.svd(W, full_matrices=False)
    Phi_svd = U_[..., :16] * S_[..., None, :16] / pr.sqrt()[..., None]
    Phi, _ = F.opt(rot.inv_freq, pi, eps, 16)
    assert torch.allclose(Phi, Phi_svd, atol=1e-8)


# 8
def test_negative_control_shifted_phi(rot):
    q, k, qpos, kpos = rand_setup()
    Phi, Gamma = F.exact(rot.inv_freq, L + 1)
    shifted = Phi[1:]  # row Δ now holds E[Δ+1]
    ref = ref_scores(rot, q, k, qpos, kpos)
    with pytest.raises(AssertionError):
        close(factor_scores(q, k, shifted[None], Gamma[None], qpos, kpos), ref)
    # and the gap sign: using n - m (conjugated kernel) must fail too
    with pytest.raises(AssertionError):
        close(factor_scores(q, k, Phi.conj()[None], Gamma[None], qpos, kpos), ref)


# 9
def test_scale_and_causality(rot):
    assert math.isclose(LlamaAttention(LlamaConfig.from_pretrained(MODEL), 0).scaling, 1 / math.sqrt(128), rel_tol=1e-12)
    q, k, qpos, kpos = rand_setup()
    Phi, Gamma = F.opt(rot.inv_freq, *rand_weights(n), 8)
    neg = qpos[:, None] < kpos[None, :]
    assert neg.any() and (~neg).any()
    for s in [factor_scores(q, k, Phi, Gamma, qpos, kpos), kernel_scores(q, k, Phi @ Gamma, qpos, kpos)]:
        assert torch.isneginf(s[:, neg]).all() and torch.isfinite(s[:, ~neg]).all()


def test_three_score_paths_agree_truncated(rot):
    q, k, qpos, kpos = rand_setup()
    U = torch.linalg.qr(torch.randn(n * dh, 512)).Q
    c = k.transpose(0, 1).reshape(-1, n * dh) @ U
    khat = (c @ U.T).view(-1, n, dh).transpose(0, 1)
    Phi, Gamma = F.opt(rot.inv_freq, *rand_weights(n), 8)
    a = absorbed_scores(q, c, head_decoders(U, n, dh), Phi, Gamma, qpos, kpos)
    close(factor_scores(q, khat, Phi, Gamma, qpos, kpos), a, rel=1e-10)
    close(kernel_scores(q, khat, Phi @ Gamma, qpos, kpos), a, rel=1e-10)


# 10 -- tiny random Llama through the patched attention
@pytest.fixture(scope="module")
def tiny():
    torch.set_default_dtype(torch.float32)
    cfg = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=2,
                      intermediate_size=128, vocab_size=100, max_position_embeddings=512,
                      attn_implementation="eager")
    torch.manual_seed(1)
    m = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 100, (2, 40))
    with torch.no_grad():
        ref = m(ids).logits
    patch_model(m)
    yield m, ids, ref
    torch.set_default_dtype(torch.float64)


def tiny_states(m, method, p, r, mode="factor"):
    inv = m.model.rotary_emb.inv_freq
    nh, d = 4, 16
    torch.manual_seed(2)
    out = []
    for _ in range(2):
        U = torch.linalg.qr(torch.randn(nh * d, nh * d)).Q[:, :r].float()
        if method == "exact":
            out.append(KFState("exact", U=U))
            continue
        pi = torch.rand(nh, 64)
        eps = torch.rand(nh, d // 2) + 0.1
        Phi, Gamma = F.opt(inv, pi, eps, p)
        out.append(KFState("kf", U=U, Phi=Phi.to(torch.complex64), Gamma=Gamma.to(torch.complex64), mode=mode))
    return out


@pytest.mark.parametrize("method,mode", [("exact", "factor"), ("kf", "factor"), ("kf", "absorbed")])
def test_tiny_full_rank_equals_reference(tiny, method, mode):
    m, ids, ref = tiny
    set_states(m, tiny_states(m, method, p=8, r=64, mode=mode))  # p = P, U orthogonal full rank
    with torch.no_grad():
        got = m(ids).logits
    assert (got - ref).abs().max() < 1e-4


@pytest.mark.parametrize("method,mode", [("kf", "factor"), ("kf", "absorbed"), ("exact", "factor")])
def test_tiny_cached_decode_matches_full_forward(tiny, method, mode):
    m, ids, ref = tiny
    set_states(m, tiny_states(m, method, p=3, r=24, mode=mode))
    with torch.no_grad():
        full = m(ids).logits
        out = m(ids[:, :30], past_key_values=DynamicCache(), use_cache=True)
        steps = [out.logits]
        for t in range(30, 40):
            out = m(ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            steps.append(out.logits)
    assert (torch.cat(steps, 1) - full).abs().max() < 1e-4
    assert (full - ref).abs().max() > 1e-3  # compression active
    pkv = out.past_key_values
    keys0 = pkv.layers[0].keys if hasattr(pkv, "layers") else pkv.key_cache[0]
    assert keys0.shape == (2, 1, 40, 25)  # unrotated latent c (r=24) + position channel
    assert torch.equal(keys0[0, 0, :, -1], torch.arange(40.0))


def test_ls_refit_matches_lstsq_for_general_gamma(rot):
    """Γ D Γ^H ≠ I here (unlike OPT), so a transpose/conjugation slip in the solve is visible."""
    eps = torch.rand(3, P) + 0.05
    Gamma = torch.randn(3, 6, P, dtype=torch.complex128)
    E = F.kernel_E(rot.inv_freq, 50)
    Phi = F.ls_refit(E, Gamma, eps, ridge=0.0)
    for h in range(3):
        sw = eps[h].sqrt().to(torch.complex128)
        # min_φ Σ_i ε_i |E[Δ,i] - (φ Γ)_i|^2  ==  lstsq((Γ diag √ε)^T, (E diag √ε)^T)
        ref = torch.linalg.lstsq((Gamma[h] * sw).mT, (E * sw).mT).solution.mT
        assert torch.allclose(Phi[h], ref, atol=1e-9)


@pytest.mark.parametrize("contig", [True, False])
def test_fast_scores_match_factor_scores(rot, contig):
    """factor_scores_fast (real form + Toeplitz view) == factor_scores, contiguous and scattered positions,
    per-head and shared Φ, and query chunking that splits the block."""
    from kernel_fact.attention import factor_scores_fast
    H, Tq, Tk = 4, 37, 53
    q, k = torch.randn(H, Tq, dh, dtype=torch.float64), torch.randn(H, Tk, dh, dtype=torch.float64)
    if contig:
        kpos = torch.arange(100, 100 + Tk)
        qpos = kpos[-Tq:]
    else:
        kpos = torch.sort(torch.randperm(4000)[:Tk]).values
        qpos = torch.sort(torch.randint(0, 4096, (Tq,))).values
    for Phi, Gamma in [F.opt(rot.inv_freq, *rand_weights(H), 8), [x[None] for x in F.pr_hi(rot.inv_freq, 8, L)]]:
        ref = factor_scores(q, k, Phi, Gamma, qpos, kpos)
        for ch in [1024, 16]:
            close(factor_scores_fast(q, k, Phi, Gamma, qpos, kpos, chunk=ch), ref, rel=1e-10)


@pytest.mark.parametrize("method", ["kf", "exact"])
def test_tiny_gate4_paths(tiny, method):
    """Gate 4 paths (mode 'fast' + decoded-key cache + batched decode) == Gate 1-3 path, full forward
    and cached decode; and left-padded batched generation through EXACT at full rank == unmodified model."""
    m, ids, ref = tiny
    st_ref = tiny_states(m, method, p=3, r=24)
    set_states(m, st_ref)
    with torch.no_grad():
        full_ref = m(ids).logits
    st_fast = tiny_states(m, method, p=3, r=24, mode="fast")
    for st in st_fast:
        st.cache_decoded = True
    set_states(m, st_fast)
    with torch.no_grad():
        full = m(ids).logits
        out = m(ids[:, :30], past_key_values=DynamicCache(), use_cache=True)
        steps = [out.logits]
        for t in range(30, 40):
            out = m(ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            steps.append(out.logits)
    assert (full - full_ref).abs().max() < 1e-4
    assert (torch.cat(steps, 1) - full_ref).abs().max() < 1e-4


def test_tiny_left_padded_generation_exact_full_rank(tiny):
    m, ids, ref = tiny
    prompts = [ids[0, :12], ids[1, :20]]
    L_ = max(len(x) for x in prompts)
    inp = torch.stack([torch.cat([torch.zeros(L_ - len(x), dtype=torch.long), x]) for x in prompts])
    mask = torch.stack([torch.cat([torch.zeros(L_ - len(x)), torch.ones(len(x))]) for x in prompts]).long()
    gen = dict(max_new_tokens=8, do_sample=False, pad_token_id=0)
    for layer in m.model.layers:
        layer.self_attn.kf = KFState("off")
    with torch.no_grad():
        base = m.generate(inp, attention_mask=mask, **gen)
    sts = tiny_states(m, "kf", p=8, r=64, mode="fast")  # p = P, full-rank U: exact
    for st in sts:
        st.cache_decoded = True
    set_states(m, sts)
    with torch.no_grad():
        got = m.generate(inp, attention_mask=mask, **gen)
    assert torch.equal(got, base)
