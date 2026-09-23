# Prior-art kill-gate check for "OPT" (low-rank factorization of the RoPE gap×frequency kernel)

Date: 2026-09-23. Checked by: web search, arXiv abstract/HTML/PDF text, and the official code for TransMLA (`MuLabPKU/TransMLA`, `TransMLA_NeurIPS_2025/transmla/*.py`) and MHA2MLA (`JT-Ushio/MHA2MLA`, `src/mha2mla/*.py`), cloned on this date.

**The idea being checked (OPT).** Take the fixed matrix E[Δ,i] = e^{jΔθ_i} (L gaps × P frequencies). Factor it as E ≈ ΦΓ with rank p. Then s(Δ) ≈ Re Σ_k Φ[Δ,k]⟨w⊙Γ_k, z⟩, a sum of p **position-free** query–key inner products, each multiplied by one scalar from a table. Because of this:

- the unrotated key can be compressed with any basis U_r (cross-head, cross-frequency PCA);
- the cache stores c = U_r^T k;
- the decoder is absorbed into the query, so each key costs 2p length-r dot products;
- (Φ, Γ) is the closed-form weighted SVD of diag(√π) E diag(√ε), where π(Δ) is the attention-weighted gap distribution and ε_i = E|w_i|²E|z_i|²;
- no training is needed.

**Kill criterion.** A paper already computes a (weighted) low-rank factorization of the RoPE kernel (gap × frequency, or equivalent) in order to absorb a low-rank key basis into the query.

---

## 1. SPE: Liutkus, Cífka, Wu, Şimşekli, Yang, Richard, "Relative Positional Encoding for Transformers with Linear Complexity", ICML 2021 (arXiv 2105.08399)

**What they factorize.** SPE works with a per-feature-dimension relative kernel P_d(m,n). The target attention is
A = exp( Σ_d diag(q_{:,d}) P_d diag(k_{:,d}) / √D )  (Eq. 9).
They read P_d as a cross-covariance, "P_d(m,n) = E[Q_d(m) K_d(n)]" (Eq. 11). They draw R random replicas so that P_d ≈ Q_d K_d^T / R (Eq. 13), then build
Q̂ = Σ_d diag(q_{:,d}) Q_d / (DR)^{1/4} and K̂ = Σ_d diag(k_{:,d}) K_d / (DR)^{1/4} (Eqs. 16–17).
The cross terms Q_d K_{d'≠d}^T vanish only **in expectation**: "for large R, the cross-terms … are negligible due to independence".
- **sineSPE** (Eqs. 18–21) sets P_d(m,n) = Σ_{k=1}^K λ²_{kd} cos(2π f_{kd}(m−n) + θ_{kd}). This is written as P_d = Ω(M,f_d,θ_d) diag(λ̈_d)² Ω(N,f_d,0)^T, a rank-2K **position(m) × position(n)** Vandermonde-type factorization. The paper notes that it reduces to the Vandermonde decomposition of a PSD Toeplitz matrix. The frequencies, phases and weights "can be trained through stochastic gradient descent".
- **convSPE** (Eq. 22) sets Q_d = Z_d ∗ Φ^Q_d and K_d = Z_d ∗ Φ^K_d. These are learned filters applied to Gaussian white noise. The resulting kernel is Toeplitz and vanishes at large lags.
- **Gated SPE** (Eqs. 23–25) mixes a content-only term into the kernel.
- **Purpose:** RPE for **linear-complexity attention** (Performer, linear Transformer). "To our knowledge, it is the first RPE strategy that is compatible with O(N) Transformers." Models are trained from scratch on LRA and music. RoPE/RoFormer is not mentioned.

**Delta from OPT.**
1. **The axis being factorized is different.** SPE splits the query position m from the key position n, P(m−n) ≈ Σ_r a_r(m) b_r(n), so that position can be baked into the keys (linear attention requires this). OPT splits the **gap Δ from the frequency index i**, E[Δ,i] ≈ Σ_k Φ[Δ,k]Γ[k,i]. The keys carry **no** position. Position enters only through a scalar table Φ[Δ,k] when each score is computed. OPT is therefore a softmax-decoding or KV-cache technique and not a linear-attention one.
2. SPE keeps D separate kernels, one per dimension, and never mixes across d. The cross-dimension terms cancel only stochastically. OPT mixes deterministically across frequencies through Γ, with no randomness.
3. SPE learns its kernels by SGD from scratch. OPT approximates a **fixed, pretrained** RoPE kernel after training, with a closed-form optimum under a query-aware score-energy weighting (π(Δ), ε_i).
4. SPE has no KV cache, no low-rank key basis and no absorption.

Note: RoPE itself is exactly sineSPE with K = 1 per pair and a known f. SPE applied to RoPE would therefore reproduce it exactly and compress nothing.

**Status: the closest conceptual precedent for "factorize the positional kernel". It is a near-miss, so PROCEED.**

## 2. TransMLA FreqFold (Meng et al., arXiv 2502.07864 v5, NeurIPS 2025)

**Paper.** App. C.1 says FreqFold approximates "numerically similar RoPE base frequencies as being effectively identical … M-D-FreqFold might treat them all as a single, representative frequency θ*". App. C.4 warns that "overly aggressive FreqFold … can degrade performance, as the loss introduced by approximation of nearby dimensions can outweigh the benefit". **The paper does not say how θ* is chosen.** The code answers this.

**Code (`partial_rope.py`, `converter.py`, `lora_qkv.py`).**
- **Group boundaries.** The code reshapes `view(b,n,kvh,2, head_dim//2//freqfold, freqfold//collapse, collapse)`. The pair index is therefore j = i·M + f'·c + c', and group i is the block of **M adjacent pair indices {iM, …, iM+M−1}**. The blocks are contiguous and fixed-size, with M a power of 2.
- **PCA.** For each group, a real covariance H is built from the kvh·M "features" (heads × in-group frequencies), with real and imaginary halves stacked as extra samples. H gets 1% mean-diagonal damping and is solved with `torch.linalg.eigh`. The same real U is applied to both halves.
- **Representative frequency.** At inference the code calls `apply_rotary_pos_emb(..., cos[:,:,::collapse], sin[:,:,::collapse], rope_head=1)`. Here `collapse` c = head_dim / qk_mqa_dim, and "auto" gives 128/64 = **2**. By default RoPE is applied only to the first 64 latent dims, which hold 32 complex pairs. After the permute `(collapse, kvh, i, f', 2)`, latent pair p = i·(M/c) + f' holds the **f'-th principal component of group i**. It is rotated by the p-th entry of the subsampled grid, i.e. θ_{c·p} = **θ_{iM + c·f'}**. So:
  - each group keeps **M/c** components, not M, when c > 1;
  - component f' is rotated at an **original grid frequency** of the group, assigned in **eigenvalue order**. The top PC gets the group's highest frequency θ_{iM}, the next gets θ_{iM+c}, and so on;
  - there is **no mean frequency, no weighting and no refit** of any positional quantity. The kept components and the query absorption (`q_rope = q · k_b_rope`) are the only fitted objects.
- **M.** "auto" searches M ∈ {c, 2c, 4c, …} by greedy WikiText PPL until the PPL stops improving. The README lists M = 8 for Llama-2-7B and M = 4 for Llama-3-8B, Qwen2.5 and Llama-3.2-1B, all with 512+64.
- **Everything else.** All other components have their RoPE stripped and become NoPE. They are then norm-balanced, joint-PCA'd with V, and fine-tuned for about 6B tokens.

**Relation to E ≈ ΦΓ.**
- **c = M (one component per group).** FreqFold is exactly a special case of OPT: Γ_k is the hard 0/1 indicator of group k, Φ[:,k] = e^{jΔθ_{kM}} (the highest frequency of the group, unoptimized), and the key basis is restricted to be block-diagonal (one vector per group). OPT optimizes Φ and Γ against π and ε and leaves the key basis unconstrained.
- **c < M (the default).** Each retained PC has its own frequency. The group kernel diag(e^{jΔθ_{iM..iM+M−1}}) is replaced by U D(Δ) U^T, where U holds the PCs and D their assigned grid frequencies. This is a **non-diagonal, key-basis-coupled** approximation and is **not** in OPT's diagonal-Γ family. It is also a heuristic without an optimality criterion; the assignment is by eigen-rank. It is not a KILL either way.

## 3. MHA2MLA (Ji et al., ACL 2025, arXiv 2502.14837)

**Paper definitions (Sec. 3).**
- S_high = {k | 0 ≤ k < r}, "the r fastest-rotating (high-frequency) subspaces".
- S_low = {k | d_h/2 − r ≤ k < d_h/2}.
- S_uniform = {⌊k·d_h/(2r)⌋ | 0 ≤ k < r}.
- S_2-norm = top-r of ‖q^{[2k,2k+1]}‖·‖k^{[2k,2k+1]}‖, aggregated over sequences per head. For GQA, "scores are averaged within each GQA group" and shared.
- "Non-selected subspaces (k ∉ S) become NoPE dimensions". These are **kept** with RoPE removed, not dropped. NoPE-K and V are then factorized by SVD_split or SVD_joint (the joint SVD of [W_k,nope, W_v] is best).
- The model is **fine-tuned** on 0.6–1% of pretraining tokens.

**Code (`src/mha2mla/2_norm.py`, `patch_func.py`).**
- `cal_2_norm` reshapes the head into (2, d_h/2), transposes, and takes the L2 norm over the (x, y) pair. This gives the **per-token pair magnitude |q_i|** of the **pre-RoPE** `q_proj`/`k_proj` outputs.
- The code averages over tokens and then over samples, so the score is **E[|q_i|]·E[|k_i|]** (a mean of magnitudes, **not** E|q_i|²·E|k_i|²).
- The score is computed per (layer, head). For GQA it is **summed** over the query heads of a group (equivalent to averaging for ranking). The pairs are ranked, and a pair is kept if rank < r/2 in the code's dim units.
- The selection is per KV head and differs between heads. S_high, S_low and S_uniform are the same for all heads. The uniform variant has a `uniform_start_point` offset.

**In OPT terms.** Partial RoPE with kept set S is E ≈ ΦΓ where Φ = [e^{jΔθ_i}]_{i∈S} ∪ [1] (a constant NoPE column), Γ is hard 0/1, and p = |S|+1. It is exact on S and replaces the rest by θ = 0. It is a special case that is not optimized, and it is fine-tuned.

## 4. Barbero et al., "Round and Round We Go! What makes Rotary Positional Encodings useful?" (ICLR 2025, arXiv 2410.06205)

- **Findings.** Gemma-7B uses the **highest** frequencies to build robust positional heads (diagonal and previous-token). It "greatly prefers to use the lowest frequencies", which the authors suspect carry semantic information. Theorem 6.1 argues these channels are not robust over long contexts.
- **p-RoPE (Sec. 6.1).** It truncates the **lowest** frequencies, i.e. makes them NoPE channels. p is "the fraction of RoPE 'kept'" (p = 0 is NoPE, p = 1 is RoPE).
- **Evaluation.** It is evaluated only by **training Gemma-2B from scratch** on Wiki and FlanV2 with an 8k context. 0.75-RoPE gives 4.4414 PPL against 4.4627 for RoPE and beats "0.75-RoPE reversed" (dropping the high frequencies). App. D relates it to RoPE_partial.
- **Relation to OPT.** p-RoPE equals MHA2MLA S_high + NoPE, with no KV compression. The finding supports OPT's premise that the low-frequency columns of E are nearly constant over the relevant gap range, so the weighted SVD will merge them into few components. It is not a factorization method.

## 5. Other related work

- **TriAttention (arXiv 2604.04921, Mao et al., MIT/ZJU/NVIDIA, Apr 2026).** Pre-RoPE Q and K concentrate around fixed centers, so logit(Δ) ≈ Σ_f [a_f cos(ω_fΔ) + b_f sin(ω_fΔ)]. The key score is S_trig(k,Δ) = Σ_f ‖E[q_f]‖·‖k_f‖·cos(ω_fΔ+φ_f), plus a norm term weighted by the mean resultant length R_f. It is used for **token eviction** (top-B keys, every 128 steps) and is training-free with calibration. It does not factorize E and does not compress the key dimension. It is conceptually close to OPT's query-aware Δ-weighting and uses the same trigonometric series view.
- **Tucker Attention (arXiv 2603.30033, Klein, Kusch, Sager, Schnake, Schotthöfer).** A Tucker factorization of the attention weight tensors, which contains GQA, MLA and MHA as special cases. It uses "latent RoPE", meaning rotations are applied in the learned latent key dimension r₃. It is **trained from scratch** and does not factorize the RoPE kernel.
- **KV-Latent (arXiv 2507.11273, ACL 2025).** Shrinks the head dims and resamples the RoPE frequencies (Formula 11: denser low-frequency sampling, high frequencies avoided). It uses a two-stage distillation and training step with about 1% of pretraining tokens. It re-designs the frequencies rather than approximating the original kernel.
- **Palu (arXiv 2407.21118, ICLR 2025).** Low-rank projection of W_k/W_v with a cached latent. For RoPE, "Palu dynamically reconstructs the keys" with a kernel that "fuses the key reconstruction, applying RoPE, and the final multiplication with query". Absorption is used **only** for non-RoPE attention. The cost is r·d_h reconstruction per key, which OPT replaces with 2p·r.
- **SALS (arXiv 2510.24273, NeurIPS 2025).** A shared latent of pre-RoPE q and k. Tokens are selected with a **RoPE-free** score s_j = q̃_{:r*}^T k̃_{j,:r*}. The selected tokens are reconstructed and exact RoPE attention is run on them. It is training-free. In OPT terms, the selection proxy is the trivial p = 1 factorization with Φ ≡ 1 and Γ ≡ 1. That proxy is used only for selection.
- **A²ATS (arXiv 2502.12665).** Windowed RoPE: u_ij = q_i R_{i−j} k_j^T if i−j < w, and q_i R_b k_j^T otherwise, with w = 64 and b = 2048 fixed. So q̃ = q R_b and k̃ = k, which is position-free. On top of this it uses query-aware VQ keys (a Cholesky-whitened k-means). The approximate scores are used **only for top-K retrieval**, followed by exact attention on the CPU. It needs offline codebook calibration but no training. In OPT terms, the far field is a hand-picked rank-1 Φ ≡ 1, Γ = e^{jbθ}. This is the closest KV-cache precedent for "replace the gap dependence by a position-free surrogate so that keys can be compressed without position". The delta: it is rank-1 and not optimized, b is heuristic, and it is used only as a retrieval proxy.
- **LRPE (Qin et al., TMLR 2023, arXiv 2307.09270).** Canonical form f_rel = q^H W_{t−s} k (Eq. 12). LRPE requires W_{t−s} = M_s^H M_t with unitary M (Eqs. 13–14), so that it decomposes for **linear** attention. This is an exact position×position factorization. RoPE is one member of the family. It is trained from scratch and has no low-rank approximation and no KV compression. Near-miss of the same type as SPE.
- **"Fast RoPE Attention" / "RoPE Attention Can Be Trained in Almost Linear Time" (arXiv 2412.17316 and follow-ups).** Theoretical complexity results based on polynomial methods, low-rank approximation of the n×n attention matrix, and FFT. They do not factorize E for a KV cache.
- **Loki (arXiv 2406.02542).** PCA of the keys for approximate top-k scoring. The full cache is kept and exact attention runs on the selected keys.
- **ShadowKV (arXiv 2410.21465).** Per-prompt SVD of pre-RoPE keys and reconstruction of the selected sparse tokens.
- **LRQK (arXiv 2510.23649).** Post-RoPE rank-r factors computed at prefill and used as proxy scores for selection.
- **xKV (arXiv 2503.18893).** Cross-layer SVD with reconstruction.
- **ReCalKV (arXiv 2505.24357).** Head reordering, grouped key SVD and reconstruction.
- **LoRC (arXiv 2410.03111).** SVD of the KV weights. The abstract mentions RoPE-specific handling, which I did not read in detail.
- **SAKI (arXiv 2608.03228).** "codes are pre-RoPE with rotation applied after reconstruction". Its §9 lists "rotation-averaged covariances" as future work.
- **STAR-KV (arXiv 2606.08382).** The PDF text never mentions RoPE.
- **KQ-SVD (arXiv 2512.05916).** Does not discuss RoPE.
- **None of the above factorize E.**
- **EliteKV (arXiv 2503.01586).** Greedy per-head selection of RoPE frequency chunks, with the rest made linear (NoPE), plus J-LRD and uptraining (0.6%). A partial-RoPE special case.
- **RAP (arXiv 2602.02599).** Prunes whole RoPE pairs so that absorption is restored.
- **Grouped Value Attention (arXiv 2609.13285, Sep 2026).** A learned content-key map absorbed into the query plus a small decoupled RoPE channel. It is trained. DeepSeek-style decoupling, not factorization.
- **Could not find:** a paper titled "RoPE is the bottleneck". The phrase occurs only as a motivation sentence in RAP, STAR-KV and similar papers.
- **KVSink (arXiv 2508.04257).** About attention sinks in KV quantization. Irrelevant.
- **SVD-LLM V2.** Weight compression. I did not open it; from general knowledge it has no RoPE-kernel content, so this one is unverified.

---

## 6. Searches performed

Web searches (24):
1. `TriAttention arXiv 2604.04921`
2. `Tucker Attention arXiv 2603.30033`
3. `low-rank RoPE kernel KV cache compression absorb key projection into query`
4. `RoPE factorization KV cache low-rank without reconstruction arXiv 2025`
5. `"separable" rotary position embedding approximation low rank attention KV cache`
6. `rotary kernel SVD frequency gap matrix low-rank approximation attention`
7. `RoPE Vandermonde low-rank approximation relative position attention`
8. `matrix absorption RoPE MLA conversion pretrained model position-dependent absorb`
9. `positional kernel low rank decomposition attention query-aware weighted SVD rotary`
10. `"Linearized Relative Positional Encoding" Qin 2023 LRPE unitary`
11. `"RoPE" bottleneck low-rank key cache "weight absorption" decoding without reconstructing keys 2026`
12. `KV-Latent 2507.11273 frequency-aware rotary positional embedding dimensional-level reduction`
13. `SALS sparse attention latent space 2510.24273 RoPE-free latent reconstruction`
14. `KVSink arXiv KV cache`
15. `"rotary" "low-rank" approximation "exp(i" gap frequency matrix KV cache compression query absorbed "training-free" 2026`
16. `RoPE-compatible low-rank KV cache compression cross-head PCA pre-RoPE keys weight absorption exact rotation equivariant`
17. `Round and Round We Go What makes Rotary Positional Encodings useful p-RoPE Barbero ICLR 2025`
18. `"Grouped Value Attention" on-demand key reconstruction 2609.13285`
19. `absorb RoPE into query low-rank latent keys without decoupled RoPE training-free MLA conversion 2026 exact`
20. `Loki low-rank keys PCA sparse attention RoPE; ShadowKV pre-RoPE low-rank key cache; xKV cross-layer SVD`
21. `RoPE attention "polynomial" OR "Chebyshev" approximation of rotation relative distance low-rank KV compression latent keys decoding`
22. `rotary position embedding separable approximation sum of products distance frequency`
23. `RoPE frequency merging KV cache compression training-free shared frequency approximation low-rank keys`
24. `attention distance distribution weighted approximation RoPE rotation matrix low-rank compression "relative distance" KV cache LLM 2026`

Full text read (arXiv abs, HTML or PDF):
- 2105.08399 (PDF, full)
- 2502.07864v5 (HTML, App. C)
- 2502.14837 (HTML)
- 2410.06205v3 (PDF, full)
- 2604.04921, 2603.30033, 2507.11273, 2407.21118, 2510.24273, 2502.12665, 2608.03228v2, 2510.23649, 2512.05916 (HTML)
- 2307.09270 (PDF, Sec. 3)
- 2606.08382 (PDF, grep)
- 2604.09742 (abs)

Code read: TransMLA `partial_rope.py`, `converter.py`, `lora_qkv.py` and README; MHA2MLA `patch_func.py` and `2_norm.py`.

## 7. Implications for baseline construction (PR-hi, PR-en, FOLD)

1. **FOLD.** The published FreqFold does **not** use a mean or weighted θ*. It works as follows:
   - groups are M adjacent pair indices;
   - a real Re(Σ) PCA is run per group across the KV heads, with 1% damping;
   - the top **M/c** PCs (c = head_dim/qk_mqa_dim, default 2) are kept;
   - PC f' is rotated at the original grid frequency **θ_{iM + c·f'}**, assigned in eigenvalue order;
   - the rotation lives in a single shared RoPE head;
   - the remaining components become NoPE (kept, not dropped);
   - M is picked by a PPL search over powers of 2 (published: 8 for Llama-2-7B, 4 for Llama-3/Qwen2.5).

   A FOLD that uses one mean θ per group, or refits anything, is a *variant*. Name it FOLD-mean, or add the faithful FOLD-TransMLA. Only c = M is an exact special case of E ≈ ΦΓ.
2. **PR-hi.** This should be MHA2MLA S_high: pairs 0..r−1, the same for all heads. The non-selected pairs become **NoPE (θ = 0, still stored and compressed)**, not zero. This equals Barbero's p-RoPE with p = r/64. If PR-hi discards the unselected pairs, it is a harsher baseline than the published method.
3. **PR-en.** If this is meant to be MHA2MLA S_2-norm, the published score is **E_t[|q_i|]·E_t[|k_i|]** (mean pre-RoPE pair magnitudes), computed per layer and **per KV head** (summed over the GQA group), with a head-specific top-r selection. It is not ε_i = E|q_i|²E|k_i|². Either match this rule or label the ε variant separately.
4. All the published methods (MHA2MLA, TransMLA, EliteKV, KV-Latent, p-RoPE) are **fine-tuned or trained**. Training-free baselines built from their selection rules must be labelled "training-free application".

## Verdict: **PROCEED**

No paper found computes a low-rank factorization of the RoPE gap×frequency kernel E, weighted or not, optimal or not, in order to absorb a low-rank, position-free key basis into the query.

**Closest prior art and the delta:**
- **SPE** (ICML 2021) factorizes relative positional kernels (sineSPE is a rank-2K Vandermonde factorization). But its factorization splits **m from n** with stochastic features, for linear attention, with kernels learned from scratch. It has no KV cache, no key-basis absorption, no weighted optimum, and does not handle pretrained RoPE. OPT splits **Δ from frequency**, deterministically, in closed form with a query-aware weighting, applied to a frozen RoPE model.
- In the KV-cache setting, the only factorizations of E are **hand-set special cases**, none optimized and none using an unconstrained key basis:
  - partial RoPE (MHA2MLA, EliteKV, p-RoPE): hard selection plus a NoPE column;
  - FreqFold: grouped hard assignment to grid frequencies;
  - A²ATS WRoPE: a fixed rank-1 far field;
  - SALS: a Φ ≡ 1 selection proxy.

  Where these methods need exactness they either fine-tune or fall back to reconstruction (Palu, ShadowKV, SALS, A²ATS).

OPT's novelty rests on three things: (i) the gap×frequency factorization used to make **exact-form absorption** possible under RoPE, (ii) the closed-form π/ε-weighted SVD optimum, and (iii) showing that these published schemes are its suboptimal special cases. For c < M, FreqFold falls outside the diagonal-Γ family (see §2), so do not claim that it is a special case in general.
