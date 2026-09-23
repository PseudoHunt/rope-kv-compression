# FINDINGS: absorbing K through RoPE by factorizing the positional kernel

**Model:** Llama-2-7B-chat (MHA, 32 × 128, 32 layers), fp16 weights. **Date:** 2026-09-23.
**Status:** Gates 0–3 PASS. Gate 4 running (see §Gate 4).

| Gate | Pre-registered condition | Result | Verdict |
|---|---|---|---|
| 0 Prior art | Kill if a paper already factorizes the RoPE kernel in low rank to absorb a low-rank key basis | None found. Closest: SPE (Liutkus 2021) splits (query pos, key pos) with random features for linear attention; no KV cache, no pretrained RoPE, no closed-form weighted optimum. See `PRIOR_ART_KERNEL.md` | **PROCEED** |
| 1 Synthetic | All tests pass | 10 required tests plus extras (three score paths agree, fast/batched Gate 4 paths, left-padded generation) = 23 kernel tests, 47 incl. previous suite. All 9 injected bugs caught (mutation check: gap sign, v sign, pair layout, LS-refit transpose, Γ weighting, FOLD weighting, PR NoPE row, softmax scale, cached positions) | **PASS** |
| 2 Score MSE, ρ=25% | Pass: median ρ_s(OPT,16) ≤ 1.44 and ≤ 2.0 on ≥ 28/32 layers. Kill: median > 2.0 | median **1.142**, ≤ 2.0 on **29/32** | **PASS** |
| 2 Dominance | OPT ≤ min(PR-hi, PR-en, FOLD-mean) on ≥ 80% of layers, p ∈ {4,8,16} | **81% / 97% / 100%** | **PASS** |
| 3 Output fidelity | Pass: median relL2(OPT,16)/relL2(EXACT) ≤ 1.2. Kill: > 1.5 | **1.098** (C4-2048 1.094, C4-4096 1.105, GSM8K 1.106) | **PASS** |
| 4 End-to-end | EXACT ≥ OPT(p) ≥ special cases; smallest p within 1 pt / 1% PPL of EXACT | running | — |

## Baseline corrections from Gate 0 (applied)

- **PR-hi** matches MHA2MLA S_high: the fastest p−1 pairs, same for every head; the other pairs lose rotation (NoPE) but stay in the cache.
- **PR-en** is the brief's ε selection. **PR-2n** was added: MHA2MLA's actual S_2-norm score, E|q_i|·E|k_i|, selected per head. The two behave almost identically (Gate 2: 2.40 vs 2.33 at p=16; Gate 3: 1.76 vs 1.72).
- **FOLD** is renamed **FOLD-mean**: contiguous groups with equal ε, each using its ε-weighted mean θ. TransMLA's published FreqFold is *not* a special case of ΦΓ in general. It runs a real PCA inside each group, keeps grid frequencies, and changes the cache. It cannot share this task's cache, so it is not reproduced here and we do not claim it as a special case.
- All published baselines fine-tune. Every special-case arm here is a **training-free** version of the published selection rule.

## Gate 2 detail (64 held-out C4-val × 4096 tokens, 128 sampled queries/seq; weights from 64 C4-train seqs)

Median over layers of ρ_s = MSE_X / MSE_EXACT (attention-weighted score MSE):

| ρ=25% | p=4 | 8 | 16 | 32 |
|---|---|---|---|---|
| **OPT** | 3.12 | 1.59 | **1.14** | 1.02 |
| OPT-joint (ε ablation) | 3.13 | 1.61 | 1.15 | 1.02 |
| OPT-layer (shared per layer) | 4.96 | 2.10 | 1.38 | 1.02 |
| OPT-unw (π, ε uniform) | 19.3 | 13.1 | 7.00 | 3.74 |
| PR-2n | 6.32 | 3.82 | 2.33 | 1.51 |
| PR-en | 6.32 | 3.87 | 2.40 | 1.52 |
| FOLD-mean | 12.3 | 10.9 | 9.19 | 6.99 |
| PR-hi | 16.2 | 16.1 | 15.3 | 9.58 |
| NOPE (p=1) | 16.3 | | | |

At ρ=12.5%, OPT is 2.34 / 1.39 / 1.09 / 1.01 at p = 4 / 8 / 16 / 32.

- **p\*_ℓ** (smallest p with ρ_s ≤ 1.44): 8 on 3 layers, 16 on 26 layers, 32 on layer 2, and more than 32 on layers 0 and 1.
- **Spectrum** of the weighted kernel W: median p99 = 5 and p99.9 = 21 over (layer, head). Layers 0–1 are much harder: median p99 = 34 and 24.
- **Instrument floor:** the kernel-form EXACT vs the model's own RoPE on the decoded keys is ≤ 0.18% of MSE_EXACT.
- **Weighting.** π carries it: removing both weights costs 6×, while the joint-ε ablation changes nothing. Per-head factorization beats per-layer sharing (1.14 vs 1.38). The unweighted (uniform over keys) ρ_s of OPT-16 is 2.14: OPT trades accuracy at rarely-attended gaps for accuracy where attention mass is. Gate 3 shows this does not create spurious attention (output error 1.10×).
- **Attention sinks dominate π.** The median attended gap is 1100–1700 in most layers, largely mass on the BOS key at gap Δ = q, so π asks for accuracy across the whole window. This was not changed, per pre-registration.

## Gate 3 detail (single-layer substitution, ρ=25%; 50 C4-val@2048, 10 C4-val@4096, 50 GSM8K 8-shot prompts)

Median over layers of relL2(arm)/relL2(EXACT); EXACT mean relL2 = 0.188, which reproduces the previous task's arm (a) at 0.184:

| p | OPT | PR-2n | PR-en | FOLD-mean | PR-hi |
|---|---|---|---|---|---|
| 8 | 1.42 | 2.36 | 2.42 | 4.60 | 6.99 |
| 16 | **1.10** | 1.72 | 1.76 | 4.33 | 6.85 |
| 32 | 1.01 | 1.33 | 1.34 | 3.21 | 4.66 |

NOPE is 7.03. The ranking is the same as Gate 2, so the surrogate predicts real output error well.

**Early layers are the weak spot.** EXACT is almost lossless in layers 0–1 (relL2 0.0003 and 0.0055), so every kernel approximation dominates there. OPT-16's absolute relL2 there is 0.04 / 0.10 / 0.15 (layers 0 / 1 / 2): below mid-layer EXACT error (~0.19), but ratios of 140× / 17× / 2×. Per-layer p allocation (§7 of the brief, out of scope) is the obvious fix.

## Implementation notes / deviations

- **Evaluation algebra.** Gates 2–3 evaluate scores as Σ_k Φ[Δ,k]·<w ⊙ Γ_k, ẑ> on the decoded key ẑ = B_hᵀc, which is algebraically identical to the absorbed form (§2.4). A Gate 1 test asserts the absorbed, factor and kernel forms agree to 1e-10. Gate 2 evaluates in the gap domain with sampled queries (unbiased for E_q).
- **Gate 4 speed path.** Gate 4 uses a real-form, Toeplitz-view implementation of the same function (tested equal to 1e-10) and caches decoded pre-RoPE keys k̂ (identical scores; memory savings are analytic, not measured). TF32 matmuls are used for **every** arm including uncompressed, which gives 3e-4 relative score error vs fp32.
- **Φ table size.** L × p complex per (layer, head): 0.5 MB at L=4096, p=16 in fp32, so 0.5 GB for the model. Per-layer sharing cuts it 32× at a cost of 1.14 → 1.38 in ρ_s.
- **Environment.** The container has a 120 GB cgroup RAM cap (not the 1 TB `free` shows). One run was OOM-killed; kernel stacks are now built per layer on the GPU.
