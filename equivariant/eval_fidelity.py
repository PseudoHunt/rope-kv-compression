"""Gate 3: per-layer score/output fidelity, single-layer substitution.

For each sequence the uncompressed model runs once; a pre-hook captures every layer's attention inputs
(post-input-layernorm hidden states, RoPE cos/sin, mask, position ids). Then, per layer and arm, the
layer's attention is recomputed with only that layer's K compressed (V uncompressed, all inputs
exact), and compared to the uncompressed recomputation:
  KL   = mean over heads and queries of KL(p_ref || p_arm) over keys
  relL2 = ||y - y_ref||_F / ||y_ref||_F, y = attention block output after o_proj, whole sequence
"""
import argparse
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import attention_core
from .data import CACHE, MODEL, heldout_tokens
from .patch_llama import KCState, make_state

ROOT = "/home/jl_fs/rope_equiv"


def gsm8k_prompts(tok, n):
    from datasets import load_dataset

    tr = load_dataset("openai/gsm8k", "main", split="train", cache_dir="/home/jl_fs/hf/datasets")
    te = load_dataset("openai/gsm8k", "main", split="test", cache_dir="/home/jl_fs/hf/datasets")
    shots = "".join(f"Question: {tr[i]['question']}\nAnswer: {tr[i]['answer']}\n\n" for i in range(8))
    return [tok(shots + f"Question: {te[i]['question']}\nAnswer:", return_tensors="pt").input_ids for i in range(n)]


@torch.no_grad()
def capture(model, ids):
    caps = {}

    def pre(li):
        def f(mod, args, kwargs):
            caps[li] = dict(h=kwargs["hidden_states"], pe=kwargs["position_embeddings"],
                            mask=kwargs["attention_mask"], pos=kwargs.get("position_ids"))
        return f

    hs = [l.self_attn.register_forward_pre_hook(pre(i), with_kwargs=True) for i, l in enumerate(model.model.layers)]
    model(ids.to(model.device), use_cache=False, logits_to_keep=1)
    for h in hs:
        h.remove()
    return caps


def kl(p, q):
    return (p * (torch.log(p.clamp_min(1e-30)) - torch.log(q.clamp_min(1e-30)))).sum(-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--methods", default="a,b,bp,c,d")
    ap.add_argument("--rhos", default="0.5,0.25,0.125")
    args = ap.parse_args()
    methods, rhos = args.methods.split(","), [float(x) for x in args.rhos.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    cfg = model.config
    n, dh = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    stats = torch.load(f"{CACHE}/calib_stats.pt")
    layers = model.model.layers
    states = {(l, m, r): make_state(m, int(round(r * n * dh / 2)), stats[l], model.model.rotary_emb, n, dh, "cuda")
              for l in range(len(layers)) for m in methods for r in rhos}

    data = [("c4", s[None]) for s in heldout_tokens(args.n, 4096)] + [("gsm8k", x) for x in gsm8k_prompts(tok, args.n)]
    rows = []
    for si, (src, ids) in enumerate(data):
        caps = capture(model, ids)
        for l, layer in enumerate(layers):
            attn = layer.self_attn
            c = caps[l]
            attn.kc = KCState("off")
            y_ref, w_ref = attention_core(attn, c["h"], c["pe"], c["mask"], c["pos"])
            for m in methods:
                for r in rhos:
                    attn.kc = states[(l, m, r)]
                    y, w = attention_core(attn, c["h"], c["pe"], c["mask"], c["pos"])
                    rows.append(dict(src=src, seq=si, layer=l, method=m, rho=r, kl=kl(w_ref, w),
                                     relL2=((y.float() - y_ref.float()).norm() / y_ref.float().norm()).item()))
                    del w
            del w_ref
            attn.kc = KCState("off")
        print(f"{si + 1}/{len(data)} {src} len={ids.shape[1]}", flush=True)
        if (si + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(f"{ROOT}/results/gate3_fidelity_raw.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(f"{ROOT}/results/gate3_fidelity_raw.csv", index=False)
    summ = df.groupby(["src", "rho", "method", "layer"])[["kl", "relL2"]].mean().reset_index()
    summ.to_csv(f"{ROOT}/results/gate3_fidelity.csv", index=False)
    print(df.groupby(["src", "rho", "method"])[["kl", "relL2"]].mean().unstack("method").round(4).to_string())


if __name__ == "__main__":
    main()
