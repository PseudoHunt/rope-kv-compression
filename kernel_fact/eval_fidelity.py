"""Gate 3: attention-output fidelity, single-layer substitution (reuses the previous task's harness).

Per sequence the uncompressed model runs once; a pre-hook captures every layer's attention inputs.
Then, per layer and arm, that layer's attention is recomputed through kf_attention (shared cache
c = U_r^T k, V uncompressed) and compared with the unmodified attention on the same inputs:
  relL2 = ||y - y_ref||_F / ||y_ref||_F   (attention block output after o_proj)
  KL    = mean over heads, queries of KL(p_ref || p_arm)
Arms at rho = 25%: EXACT; OPT, PR-hi, PR-en, PR-2n, FOLD-mean at p in {8, 16, 32}; NOPE.
"""
import argparse
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from equivariant.data import CACHE, MODEL, c4_chunks
from equivariant.eval_fidelity import capture, gsm8k_prompts, kl
from .arms import layer_arms
from .patch_llama import KFState, kf_attention

ROOT = "/home/jl_fs/rope_equiv"
ARMS = ["EXACT", "NOPE"] + [f"{a}/{p}" for a in ["OPT", "PR-hi", "PR-en", "PR-2n", "FOLD-mean"] for p in [8, 16, 32]]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n4096", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    cfg = model.config
    n, dh = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    inv_freq = model.model.rotary_emb.inv_freq.cpu()
    Wt = torch.load(f"{CACHE}/kf_weights.pt")
    stats = torch.load(f"{CACHE}/calib_stats.pt")
    r = int(args.rho * n * dh)
    rot = model.model.rotary_emb
    states = {}
    t0 = time.time()
    for l in range(cfg.num_hidden_layers):
        U = stats[l]["U_pre"][:, :r].float().cuda()
        arms = layer_arms(inv_freq, Wt, l, [8, 16, 32], ablations=False)
        for name in ARMS:
            if name == "EXACT":
                states[(l, name)] = KFState("exact", U=U, rotary=rot)
            else:
                Ph, G = arms[name]
                states[(l, name)] = KFState("kf", U=U, Phi=Ph.to(torch.complex64).cuda(),
                                            Gamma=G.to(torch.complex64).cuda(), rotary=rot)
    print(f"states built {time.time() - t0:.0f}s", flush=True)

    c4 = c4_chunks("validation", args.n, 4096)
    data = ([("c4-2048", s[None, :2048]) for s in c4] + [("c4-4096", s[None]) for s in c4[: args.n4096]]
            + [("gsm8k", x) for x in gsm8k_prompts(tok, args.n)])
    rows = []
    t0 = time.time()
    for si, (src, ids) in enumerate(data):
        caps = capture(model, ids)
        for l, layer in enumerate(model.model.layers):
            attn, c = layer.self_attn, caps[l]
            attn.kf = KFState("off")
            y_ref, w_ref = kf_attention(attn, c["h"], c["pe"], c["mask"], c["pos"])
            for name in ARMS:
                attn.kf = states[(l, name)]
                y, w = kf_attention(attn, c["h"], c["pe"], c["mask"], c["pos"])
                rows.append(dict(src=src, seq=si, layer=l, arm=name, kl=kl(w_ref, w),
                                 relL2=((y.float() - y_ref.float()).norm() / y_ref.float().norm()).item()))
                del w, y
            del w_ref
            attn.kf = KFState("off")
        print(f"{si + 1}/{len(data)} {src} len={ids.shape[1]} {time.time() - t0:.0f}s", flush=True)
        if (si + 1) % 10 == 0 or si + 1 == len(data):
            pd.DataFrame(rows).to_csv(f"{ROOT}/results/kf_gate3_fidelity_raw{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
