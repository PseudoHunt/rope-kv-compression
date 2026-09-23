# BASELINE_FREEZE: uncompressed Llama-2-7B-chat

Frozen 2026-09-23. All compressed arms are compared against these numbers. Source: `results/kf_gate4_ppl.jsonl`, rows with `"arm": "off"`.

## Configuration

- **Weights:** `NousResearch/Llama-2-7b-chat-hf` (ungated mirror of Llama-2-7b-chat), fp16.
- **Attention:** eager, through the same patched attention as the compressed arms (`kernel_fact/patch_llama.py`, `method="off"`). This path is plain RoPE on full keys. Scores are fp32 with **TF32 matmuls** (as for every Gate 4 arm), so the comparison stays like-for-like.
- **Data:** WikiText-2 raw, test split, joined with `"\n\n"` and tokenized once: 341,469 tokens. Non-overlapping chunks of `ctx` tokens, with loss on tokens 2..ctx of each chunk. No stride and no BOS insertion.
- **Arm subset:** the first 48 × 2048 or 24 × 4096 chunks (about 98k tokens). Every compressed arm is evaluated on this subset, so compare arms against the **subset** row.
- **Seeds:** none needed (deterministic forward); `torch.manual_seed(0)` is set anyway.
- **Command:** `python -m kernel_fact.eval_e2e ppl --arm off --ctx {2048|4096} [--max_chunks {48|24}]`

## WikiText-2 perplexity

| ctx | full test set (chunks) | arm subset (chunks) |
|---|---|---|
| 2048 | **6.9425** (166) | **6.8695** (48) |
| 4096 | **6.4920** (83) | **6.4125** (24) |

## Not frozen yet

GSM8K (8-shot CoT, strict/flexible) and the 4 LongBench subsets (qasper, hotpotqa, qmsum, passage_retrieval_en; 50 examples each) have **not run**. The Gate 4 queue was paused before its `gsm8k` phase. Their harness is `kernel_fact/eval_e2e.py` (`gsm8k`, `longbench`), smoke-tested only.
