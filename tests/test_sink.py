"""Post-hoc SINK-k variant: keys at positions < k are scored by the EXACT path, all others by OPT."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kernel_fact import factorize as F
from kernel_fact.patch_llama import KFState, layer_scores, patch_model, set_states


@pytest.fixture(scope="module")
def tiny64():
    cfg = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=2,
                      intermediate_size=128, vocab_size=100, max_position_embeddings=512,
                      attn_implementation="eager")
    torch.manual_seed(3)
    m = LlamaForCausalLM(cfg).double().eval()
    patch_model(m)
    return m


def states(m, method, p=3, r=24, mode="factor", sink_k=0, decoded=False):
    torch.manual_seed(4)
    U = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64)).Q[:, :r]
    Phi, Gamma = F.opt(m.model.rotary_emb.inv_freq, torch.rand(4, 64), torch.rand(4, 8) + 0.1, p)
    st = KFState(method, U=U, Phi=Phi, Gamma=Gamma, mode=mode, sink_k=sink_k, cache_decoded=decoded)
    st.rotary = m.model.rotary_emb
    return st


def scores(m, st, h, pos):
    attn = m.model.layers[0].self_attn
    attn.kf = st
    pe = m.model.rotary_emb(h, pos)
    return layer_scores(attn, h, pe, pos)[0]


@pytest.mark.parametrize("mode,decoded", [("factor", False), ("absorbed", False), ("fast", True)])
@pytest.mark.parametrize("k", [1, 4])
def test_sink_keys_exact_rest_opt(tiny64, mode, decoded, k):
    m = tiny64
    torch.manual_seed(5)
    h = torch.randn(1, 30, 64, dtype=torch.float64)
    pos = torch.arange(30)[None]
    s_ex = scores(m, states(m, "exact", mode=mode, decoded=decoded), h, pos)
    s_opt = scores(m, states(m, "kf", mode=mode, decoded=decoded), h, pos)
    s_snk = scores(m, states(m, "kf", mode=mode, decoded=decoded, sink_k=k), h, pos)
    causal = pos[0][None, :] <= pos[0][:, None]  # [Tq, Tk]
    sink = causal & (pos[0][None, :] < k)
    rest = causal & ~sink
    assert (s_snk[..., sink] - s_ex[..., sink]).abs().max() < 1e-10
    assert (s_snk[..., rest] - s_opt[..., rest]).abs().max() < 1e-10
    assert (s_ex[..., sink] - s_opt[..., sink]).abs().max() > 1e-3  # the two paths really differ there


@pytest.mark.parametrize("mode,decoded", [("factor", False), ("absorbed", False), ("fast", True)])
def test_sink0_is_plain_opt(tiny64, mode, decoded):
    m = tiny64
    torch.manual_seed(6)
    h = torch.randn(1, 25, 64, dtype=torch.float64)
    pos = torch.arange(25)[None]
    s_opt = scores(m, states(m, "kf", mode=mode, decoded=decoded), h, pos)
    s_k0 = scores(m, states(m, "kf", mode=mode, decoded=decoded, sink_k=0), h, pos)
    assert torch.equal(s_opt, s_k0)


def test_sink_cached_decode_matches_full_forward(tiny64):
    m = tiny64
    set_states(m, [states(m, "kf", mode="fast", decoded=True, sink_k=4) for _ in range(2)])
    ids = torch.randint(0, 100, (2, 30))
    with torch.no_grad():
        full = m(ids).logits
        out = m(ids[:, :20], past_key_values=DynamicCache(), use_cache=True)
        steps = [out.logits]
        for t in range(20, 30):
            out = m(ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            steps.append(out.logits)
    assert (torch.cat(steps, 1) - full).abs().max() < 1e-10
