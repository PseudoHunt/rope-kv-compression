# RoPE and low-rank KV-cache compression: two gated studies on Llama-2-7B-chat

Training-free, calibration-only experiments on compressing the **key** cache of a RoPE model while keeping attention exact or near-exact. Every study is run as a sequence of pre-registered gates (prior art → synthetic correctness → cheap diagnostic → fidelity → end-to-end) with binding kill conditions; every outcome, including kills, is recorded.

| Study | Idea | Outcome | Write-up |
|---|---|---|---|
| 1. `equivariant/` | Compressors that **commute with RoPE exactly**: per-frequency complex-linear cross-head PCA; cache the latent already rotated, attend with plain MQA+RoPE, no reconstruction | **Stopped at Gate 3.** Output error 2.69× that of pre-RoPE PCA at 25% K budget. Key low-rankness is mostly *cross-frequency*, which no RoPE-exact linear map can use; complex vs real mixing adds only ~1.5 energy points | [`docs/FINDINGS.md`](docs/FINDINGS.md), [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) |
| 2. `kernel_fact/` | **Factorize the positional kernel** E[Δ,i] = e^{jΔθ_i} ≈ ΦΓ (rank p). Scores become p position-free query–key products times a scalar table, so an unconstrained low-rank key basis absorbs into the query. Optimal (Φ, Γ) is a closed-form π/ε-weighted SVD; partial RoPE and frequency folding are special cases | **Gates 0–3 pass.** At p = 16 (4× fewer K-side MACs than reconstruction): score MSE 1.14× and attention-output error 1.10× that of reconstruct-then-RoPE; best special case (MHA2MLA-style 2-norm partial RoPE) 1.72×. **Gate 4 partial** (WikiText-2, ρ = 25%, ctx 2048): EXACT +0.48% PPL over uncompressed; OPT +0.77% over EXACT at p = 32, +2.49% at p = 16, +1.37% at p = 16 with layers 0–1 exact (post-hoc); training-free PR-2n at p = 16 collapses (PPL 93) | [`docs/FINDINGS_KERNEL.md`](docs/FINDINGS_KERNEL.md), [`docs/PRIOR_ART_KERNEL.md`](docs/PRIOR_ART_KERNEL.md) |

## Layout

```
equivariant/   study 1: calibration stats (Σ_i), compressors (a)-(d), eager attention, Gate 2/3 scripts
kernel_fact/   study 2: factorize.py (E, weighted SVD, LS refit, PR/FOLD/NOPE), absorb.py (§2.3-2.4 real forms),
               attention.py (absorbed / factor / kernel / fast score paths), patch_llama.py (shared unrotated cache),
               weights.py (π, ε), eval_scores.py (Gate 2), eval_fidelity.py (Gate 3), eval_e2e.py + run_gate4.sh (Gate 4)
tests/         CPU correctness suites (every reference path uses the model's own RoPE code)
results/       CSVs, logs and figures for every gate
docs/          findings and prior-art reviews
```

## Reproduce

Requires a GPU with ≥ 40 GB, `torch`, `transformers` (5.x), `datasets`, `pandas`, `matplotlib`, `rouge`. Weights: `NousResearch/Llama-2-7b-chat-hf` (ungated mirror of Llama-2-7b-chat). Paths in `equivariant/data.py` point at the original machine; edit `HF`, `MODEL`, `CACHE` for yours. Calibration data (C4 train shard 0) and held-out data (C4 validation shard 0) are rebuilt automatically; `cache/` is not versioned.

```bash
python -m pytest tests -q                                  # Gate 1, both studies (CPU)
python -m equivariant.calib_complex                        # study 1, Gate 2 (also writes the key-PCA bases study 2 uses)
python -m equivariant.eval_fidelity --n 50                 # study 1, Gate 3
python -m kernel_fact.weights                              # study 2: π, ε from C4-train
python -m kernel_fact.eval_scores && python -m kernel_fact.gate2_report      # study 2, Gate 2
python -m kernel_fact.eval_fidelity && python -m kernel_fact.gate3_report    # study 2, Gate 3
bash kernel_fact/run_gate4.sh baseline|ppl25|gsm8k|ppl125|longbench PR-2n    # study 2, Gate 4
```

No latency is reported anywhere: the implementations are eager PyTorch for correctness, not speed.
