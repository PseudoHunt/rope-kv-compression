# FINDINGS: RoPE-equivariant KV compression via per-frequency complex cross-head PCA

**Model:** Llama-2-7B-chat (MHA, 32 heads × 128, 32 layers), fp16. **Date:** 2026-09-23.
**Outcome: STOPPED at Gate 3 (pre-registered fail).** Gate 4 was not run.

| Gate | Pre-registered condition | Result | Verdict |
|---|---|---|---|
| 0 Prior art | Kill if anyone already does complex-linear per-frequency cross-head compression with exact RoPE | No one does. RoRoPE (TransMLA) is real-orthogonal, i.e. our (b). Tucker Attention's latent RoPE is learned and unstructured. See `PRIOR_ART.md` | **PROCEED** |
| 1 Synthetic | All 8 tests pass | 24 pass (8 required plus 16 extra). All 6 injected bugs are caught | **PASS** |
| 2 Energy, ρ=25% | Pass: a−c ≤ 3 on a majority of layers. Kill: a−c > 10 on a majority | ≤3 on 2 layers, 3–10 on 22, >10 on 8. Median 9.6 | **GREY → Gate 3** |
| 2 Novelty | Report if c−b < 2 on most layers | c−b < 2 on **28/32** layers (median +1.5) | **Complex adds little over real** |
| 2 Sanity | c ≥ b on every layer; c > d on most layers | c ≥ b on 32/32 ✓. c > d on **0/32** ✗. Explained below and not a bug | Expectation wrong |
| 3 Fidelity | (c) mean relL2 within 1.5× of (a) | **2.69×** at ρ=25% (2.92× at 50%, 2.38× at 12.5%). 0/32 layers within 1.5× | **FAIL → stop** |

## Headline

The RoPE-exact family (block-diagonal over frequencies, complex-linear inside each block) cannot
reach the low-rankness that Llama-2 keys actually have. That low-rankness is **mostly cross-frequency**.

At ρ = 25%, (c) keeps 88.9% of key energy on held-out data. Unconstrained pre-RoPE PCA (a) keeps
97.7%. So (c) discards 4.8× more energy. Attention-output error follows √(lost energy) for every arm,
so (c) ends up with 2.7× the output error of (a). The method does exactly what the theory says:
it is exact at full rank (relL2 0.0017, which is fp16 noise, on the real model) and needs no key
reconstruction. The family is simply too small.

## Gate 2 detail: energy decomposition (held-out, mean over layers)

| ρ | (a) pre-RoPE PCA | (d) post-RoPE PCA | (e)* per-freq block, any real map | (c) complex, global | (c′) complex, uniform | (b) real, global | (b′) real, uniform (RoRoPE alloc.) |
|---|---|---|---|---|---|---|---|
| 50% | 99.5 | 96.2 | 95.6 | **94.9** | 93.6 | 94.2 | 92.5 |
| 25% | 97.7 | 91.8 | 90.4 | **88.9** | 87.4 | 87.4 | 85.6 |
| 12.5% | 95.3 | 88.1 | 85.9 | **83.8** | 82.2 | 81.5 | 79.7 |

\*(e) is a diagnostic, not RoPE-exact: an unconstrained real 2n×2n PCA inside each frequency block.
In-sample and held-out values agree to within 0.1 point (512 × 4096 calibration tokens).

Where the (a)−(c) gap comes from at ρ=25% (median over layers):
- **Cross-frequency structure (a→e): about 8.0 points.** No RoPE-exact linear map can reach it.
- **Complex-linearity inside a frequency (e→c): about 1.4 points.**
- **Global rank allocation over uniform (c vs c′): +1.4 points.** This is as large as the whole complex-vs-real gain.
- **Complex over real (c vs b): +1.5 points.** Against RoRoPE-as-published allocation (b′), (c) gains +3.3 points, but about half of that is the allocation.

**Novelty implication.** The complex extension of RoRoPE is real but small: Im Σᵢ is large
(‖Im Σ‖/‖Σ‖ = 0.52–0.68), yet it moves the top eigen-energy by only 1–2 points. The larger
practical gain over TransMLA's RoRoPE is global rank allocation plus keeping exact RoPE on every channel.

Rank allocation (`results/gate2_rank_alloc.png`): (c) puts most channels on the 18 slow pairs
(i ≥ 46, θᵢ·4096 < 2π). Those pairs carry about 57% of key energy in the middle and late layers.
Fast pairs (i ≲ 20) get r_i ≈ 0–2 in most layers after layer 1.

## Sanity violation: (d) post-RoPE PCA beats (c) on every layer

The brief expected (d) < (c) because RoPE inflates post-RoPE rank. It does inflate it, since
(d) < (a) by about 6 points, but (d) still beats (c) by about 3 points. This was treated as an
instrument bug until tested (`equivariant/gate2_dcheck.py`, `results/gate2_dcheck.txt`):

1. **Prediction matches.** If the key distribution does not depend on position, C_post is determined
   by C_pre: C_post = mean_m R_m C_pre R_mᵀ. The prediction from C_pre alone matches the measured
   C_post to within 0.7–3.7% relative Frobenius error. It reproduces the measured (d) energy to
   within 0.2 points on layers 0, 15 and 31 at every ρ. So the (d) number is not a measurement
   artifact. It follows from C_pre plus RoPE geometry.
2. **Mechanism.** Position-averaging multiplies each cross-frequency block by a Dirichlet average of
   e^{jm(θᵢ ∓ θₖ)}. Only the complex-linear diagonal blocks survive exactly, and those are what (c)
   uses. Terms between **slow** frequencies also survive, because they barely rotate within 4096
   tokens. (d) can exploit those terms. No exact method can.
3. **Fast pairs only.** Restricting to the fast pairs cuts the d−c gap from about 3.8 to about
   1.3 points (layer 15, ρ=25%). The rest comes from near-boundary frequencies that are only
   partially averaged out.

This is the §8 low-frequency remark showing up as a measurable effect: the part of key structure
that is "approximately equivariant" (slow band) is worth about 3 energy points on this model.
(d) converts it into fidelity as well: Gate 3 relL2 0.385 vs 0.496 for (c).

## Gate 3 detail: single-layer substitution, 50 held-out C4 (4096 tokens) + 50 GSM8K 8-shot prompts (~1.4k tokens)

| ρ | relL2 a | d | c | b | b′ | c/a | KL a | KL c |
|---|---|---|---|---|---|---|---|---|
| 50% | 0.108 | 0.250 | 0.314 | 0.335 | 0.424 | 2.92 | 0.006 | 0.037 |
| 25% | 0.184 | 0.385 | 0.496 | 0.529 | 0.619 | 2.69 | 0.016 | 0.095 |
| 12.5% | 0.264 | 0.484 | 0.630 | 0.671 | 0.751 | 2.38 | 0.030 | 0.167 |

- C4 and GSM8K give the same ratios (2.67 vs 2.72 at ρ=25%).
- The ranking a > d > c > b > b′ holds at every ρ, on nearly every layer, and matches the energy ranking.
- relL2 / √(1 − energy) is 1.2–1.6 for every arm. There is no method-specific fidelity penalty beyond lost energy.

**Lesson for pre-registration.** Gate 2's "points of retained energy" was the wrong scale.
Fidelity tracks the **ratio of lost energy**. A 9.6-point gap at 97.7% vs 88.9% means 4.8× the lost
energy, which fails a 1.5× relL2 bar. A lost-energy-ratio criterion (≤ 2.25× for a 1.5× relL2 bar)
would have killed this at Gate 2: the ratio is 4.8× at ρ=25%, 10.7× at 50% and 3.4× at 12.5%.

## What was done differently from the brief (deviations)

- **Missing SALS repo.** No SALS repo existed on the machine, so the C4 calibration was rebuilt
  (`equivariant/data.py`): 512 × 4096 tokens from C4 **train** shard 00000; held-out data is 50 × 4096
  from C4 **validation** shard 00000; documents are EOS-joined and every chunk starts with BOS.
  Pre-RoPE keys come from forward hooks on `k_proj`.
- **Weights.** `meta-llama` is gated here, so the weights are the ungated mirror
  `NousResearch/Llama-2-7b-chat-hf` (same checkpoint).
- **Runtime maps.** W_Q′ and W_K′ are not materialized. The absorbed query q̃_h = B_h q_h and the
  latent c = Σ_h B_h k_h are computed at runtime from pre-RoPE q and k with one real packing tensor
  B ∈ ℝ^{n×2R×d_h}. The same tensor serves both, because conj(v)·z and w·conj(v) have identical real
  2×2 forms.
- **(a) and (d) are fidelity simulations.** The cache holds reconstructed or projected full keys,
  so no memory is actually saved.
- **Extra arms.** Added (b′) = RoRoPE-as-published allocation (uniform R/64 channels per frequency,
  RoPE on all of them, no NoPE path or fine-tuning) and (c′) = complex with uniform allocation, to
  separate allocation effects from complex-vs-real. Added diagnostics (e) and (d restricted to blocks).
- **KL direction.** KL(p_ref ‖ p_arm), averaged over heads and queries.
- **relL2 target.** Measured on the attention block output after o_proj.
- **Not created.** `BASELINE_FREEZE.md`: Gate 4 was never reached, so no uncompressed WikiText-2 / GSM8K numbers were frozen.

## Recommendations (not run; each changes the method, so each needs its own pre-registration)

1. **Query-aware per-frequency PCA.** Energy-optimal allocation spends most rank on slow pairs
   because keys are large there, not necessarily because scores depend on them. Whitening Σᵢ by
   per-head query power, D_i = diag(E|w_{h,i}|²), gives V_i = D_i^{-1/2} · eig(D_i^{1/2} Σ_i D_i^{1/2}).
   This is still complex-linear per frequency, so still exactly RoPE-equivariant. It is the one
   remaining lever inside the exact family. Check it against a query-aware (a), because Palu and
   SALS-style baselines benefit from the same trick.
2. **Hybrid exact + approximate.** Handle the slow band (i ≥ 46, 57% of energy) with an approximate
   cross-frequency map (FreqFold-style or (d)-style; bounded error since θᵢ·L is small) and the fast
   band with (c). The d_check numbers suggest this could recover most of the 3-point (d)−(c) gap, at
   the cost of exactness in the slow band. This is the brief's §7 FreqFold ablation, now with a
   measured motivation.
3. **Don't pursue as-is.** At every tested ρ, the exact family is 2.4–2.9× worse than
   reconstruct-then-RoPE (a) on output fidelity. The systems advantage (a stock MQA+RoPE kernel,
   no reconstruction) would have to be worth that accuracy cost.

## Reproduce

```
source /home/jl_fs/venv/bin/activate; cd /home/jl_fs/rope_equiv; export HF_HOME=/home/jl_fs/hf
python -m pytest tests -q                               # Gate 1 (CPU)
python -m equivariant.calib_complex                     # Gate 2 (~10 min A100) -> results/gate2_energy.csv, cache/calib_stats.pt
python -m equivariant.gate2_dcheck                      # (d)>(c) instrument check
python -m equivariant.eval_fidelity --n 50              # Gate 3 -> results/gate3_fidelity*.csv
python -m equivariant.gate2_plots                       # all figures
```
