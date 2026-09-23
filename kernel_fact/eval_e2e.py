"""Gate 4: end-to-end evaluation with every layer's K going through one arm (V uncompressed, eager).

  python -m kernel_fact.eval_e2e ppl       --arm OPT --p 16 --rho 0.25 [--ctx 2048 --max_chunks 48]
  python -m kernel_fact.eval_e2e gsm8k     --arm OPT --p 16 --rho 0.25
  python -m kernel_fact.eval_e2e longbench --arm OPT --p 16 --rho 0.25 [--n_per 50]

arm 'off' = uncompressed model through the same patched attention (same score precision as the arms).
All arms: mode 'fast' + decoded-key cache (scores identical to the absorbed form; Gate 1 tests),
TF32 matmuls for every arm including 'off'. One JSON line per run is appended to results/kf_gate4_<task>.jsonl.
"""
import argparse
import json
import re
import string
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from equivariant.data import CACHE, MODEL
from .arms import layer_arms
from .patch_llama import KFState, patch_model, set_states

ROOT = "/home/jl_fs/rope_equiv"
LB = f"{CACHE}/longbench"


def build_states(model, arm, p, rho):
    cfg = model.config
    n, dh = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    if arm == "off":
        return [KFState("off") for _ in range(cfg.num_hidden_layers)]
    inv_freq = model.model.rotary_emb.inv_freq.cpu()
    Wt = torch.load(f"{CACHE}/kf_weights.pt")
    stats = torch.load(f"{CACHE}/calib_stats.pt")
    r = int(rho * n * dh)
    out = []
    for l in range(cfg.num_hidden_layers):
        U = stats[l]["U_pre"][:, :r].float().cuda()
        if arm == "EXACT":
            out.append(KFState("exact", U=U, mode="fast", cache_decoded=True))
            continue
        name = "NOPE" if arm == "NOPE" else f"{arm}/{p}"
        Ph, G = layer_arms(inv_freq, Wt, l, [p] if arm != "NOPE" else [4], ablations=False)[name]
        out.append(KFState("kf", U=U, Phi=Ph.to(torch.complex64).cuda(), Gamma=G.to(torch.complex64).cuda(),
                           mode="fast", cache_decoded=True))
    return out


@torch.no_grad()
def run_ppl(model, tok, args):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir="/home/jl_fs/hf/datasets")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n_chunks = len(ids) // args.ctx
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    nll, cnt = 0.0, 0
    for c in range(n_chunks):
        x = ids[c * args.ctx : (c + 1) * args.ctx][None].cuda()
        loss = model(x, labels=x, use_cache=False).loss.float().item()
        nll += loss * (args.ctx - 1)
        cnt += args.ctx - 1
    return dict(ppl=float(torch.tensor(nll / cnt).exp()), ctx=args.ctx, n_chunks=n_chunks, tokens=cnt,
                total_test_tokens=len(ids))


def gsm_num(s):
    s = s.replace(",", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


@torch.no_grad()
def run_gsm8k(model, tok, args):
    from datasets import load_dataset

    tr = load_dataset("openai/gsm8k", "main", split="train", cache_dir="/home/jl_fs/hf/datasets")
    te = load_dataset("openai/gsm8k", "main", split="test", cache_dir="/home/jl_fs/hf/datasets")
    shots = "".join(f"Question: {tr[i]['question']}\nAnswer: {tr[i]['answer']}\n\n" for i in range(8))
    n = args.n_gsm or len(te)
    tok.padding_side = "left"
    tok.pad_token_id = 0
    strict = flex = 0
    preds = []
    for b0 in range(0, n, args.bs):
        qs = te[b0 : b0 + args.bs]
        enc = tok([shots + f"Question: {q}\nAnswer:" for q in qs["question"]], return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=0,
                             stop_strings=["Question:"], tokenizer=tok)
        for i, o in enumerate(out[:, enc.input_ids.shape[1]:]):
            text = tok.decode(o, skip_special_tokens=True).split("Question:")[0]
            gold = gsm_num(qs["answer"][i].split("####")[-1])
            m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
            s_ok = m is not None and gsm_num(m.group(1)) == gold
            nums = re.findall(r"-?[\d,]*\.?\d+", text)
            f_ok = bool(nums) and gsm_num(nums[-1]) == gold
            strict += s_ok
            flex += f_ok
            preds.append(dict(i=b0 + i, pred=text, gold=gold, strict=bool(s_ok), flex=bool(f_ok)))
        print(f"  gsm8k {b0 + len(qs['question'])}/{n} strict={strict / (b0 + len(qs['question'])):.3f}", flush=True)
    return dict(strict=strict / n, flexible=flex / n, n=n, shots=8, max_new_tokens=256), preds


# ---- LongBench (official prompts, truncation, and metrics for the 4 subsets used) ----
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1_score(pred, gt):
    p, g = normalize_answer(pred).split(), normalize_answer(gt).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def rouge_score(pred, gt):
    from rouge import Rouge

    try:
        return Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


def retrieval_score(pred, gt):
    gid = re.findall(r"Paragraph (\d+)", gt)[0]
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(x == gid for x in nums) / len(nums)


LB_SETS = {"qasper": qa_f1_score, "hotpotqa": qa_f1_score, "qmsum": rouge_score,
           "passage_retrieval_en": retrieval_score}


@torch.no_grad()
def run_longbench(model, tok, args):
    d2p = json.load(open(f"{LB}/d2p.json"))
    d2m = json.load(open(f"{LB}/d2m.json"))
    res, preds = {}, []
    for ds, metric in LB_SETS.items():
        rows = [json.loads(x) for x in open(f"{LB}/data/{ds}.jsonl")][: args.n_per]
        tot = 0.0
        for j, ex in enumerate(rows):
            prompt = d2p[ds].format(**ex)
            ids = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(ids) > 3500:  # LongBench llama2-7b-chat-4k: middle truncation to 3500 tokens
                half = 1750
                prompt = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)
            prompt = f"[INST]{prompt}[/INST]"
            enc = tok(prompt, truncation=False, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=d2m[ds], do_sample=False, num_beams=1)
            pred = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            score = max(metric(pred, gt) for gt in ex["answers"])
            tot += score
            preds.append(dict(ds=ds, i=j, pred=pred, score=score))
        res[ds] = 100 * tot / len(rows)
        print(f"  {ds}: {res[ds]:.2f}", flush=True)
    res["avg"] = sum(res[d] for d in LB_SETS) / len(LB_SETS)
    res["n_per"] = args.n_per
    return res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["ppl", "gsm8k", "longbench"])
    ap.add_argument("--arm", default="off")
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=0)
    ap.add_argument("--n_gsm", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_per", type=int, default=50)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True  # every arm, including 'off'
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    patch_model(model)
    set_states(model, build_states(model, args.arm, args.p, args.rho))
    t0 = time.time()
    cfg = dict(task=args.task, arm=args.arm, p=None if args.arm in ("off", "EXACT", "NOPE") else args.p,
               rho=None if args.arm == "off" else args.rho)
    if args.task == "ppl":
        res, preds = run_ppl(model, tok, args), None
    elif args.task == "gsm8k":
        res, preds = run_gsm8k(model, tok, args)
    else:
        res, preds = run_longbench(model, tok, args)
    rec = dict(**cfg, **res, seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    with open(f"{ROOT}/results/kf_gate4_{args.task}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if preds is not None:
        tag = f"{args.arm}_{cfg['p']}_{cfg['rho']}"
        with open(f"{ROOT}/results/kf_gate4_{args.task}_preds_{tag}.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in preds)


if __name__ == "__main__":
    main()
