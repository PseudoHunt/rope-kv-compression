# Prior-art check: per-frequency complex cross-head key compression with exact RoPE ("method (c)")

Date of check: 2026-09-23. Sources: arXiv abs and HTML pages (text taken from the arXiv HTML renderings), the TransMLA GitHub code (`MuLabPKU/TransMLA`, `TransMLA_NeurIPS_2025/transmla/partial_rope.py`), and web search. Quoted equation and section numbers refer to the arXiv HTML version named in each section.

Notation: for frequency i, z_i ∈ C^n stacks the pre-RoPE key pair i of all n KV heads, z = x + j·y, where x is the first half and y the second half in HF `rotate_half` layout. Σ_i = E[z_i z_i^H]. Then Re(Σ_i) = E[x xᵀ + y yᵀ] and Im(Σ_i) = E[y xᵀ − x yᵀ], which is antisymmetric.
- (b) is real cross-head mixing: one real orthogonal U_i applied to x and to y. The optimal U_i comes from the eigenvectors of Re(Σ_i).
- (c) is complex cross-head mixing: V_i ∈ C^{n×r_i}, taken from the eigenvectors of Σ_i. Rank is allocated across frequencies from the pooled eigenvalues. The cache holds c_i = V_i^H z_i, which is rotated with θ_i, and conj(V_i) is absorbed into the queries.

---

## 1. TransMLA (Meng, Tang, Tang, Yao, Sun, Zhang; arXiv 2502.07864, latest version v5 of 12 Jun 2025, NeurIPS 2025 spotlight). RoRoPE and FreqFold

**What it does.** TransMLA converts GQA to MLA. It first merges all g KV heads into one latent head of size g·d, which is exact (Sec. 4.1, Eqs. 11–15). It then applies **RoRoPE** (Sec. 4.2): for each RoPE frequency l, it collects the l-th 2-D subspace of every head and applies an orthogonal matrix U_l ∈ R^{g×g}. Eq. 19 shows that the RoPE inner product is invariant when "the same orthogonal matrix U_l [is] applied to both the real (2l−1) and imaginary (2l) components". Appendix B gives the construction. It defines σ_{x,l} = K_{x,l}ᵀK_{x,l} and σ_{y,l} = K_{y,l}ᵀK_{y,l}, "the h×h covariance matrices of the real and imaginary key components", and then solves (Eq. 32)
`max_U Tr[(Uᵀ(σ_{x,l}+σ_{y,l})U)_{:m,:m}] s.t. UᵀU = I`. The solution is "an eigendecomposition on the summed covariance matrix σ_{x,l}+σ_{y,l}". The text says explicitly that the constraint "precludes performing separate PCA on the real and imaginary parts."

**(i) Real or complex?** RoRoPE is **real-orthogonal**. σ_x + σ_y equals Re(Σ_l) in uncentered form, so RoRoPE is exactly our (b). The official code confirms this. `PartialRope.joint_complex_pca` is named "complex", but it reshapes keys to `(b, n*2, num_kv_heads*freqfold, head_dim//2//freqfold)`, which stacks the real and imaginary halves **as extra samples (rows)**. It then accumulates `H += X.mT @ X` (a real matrix), adds 1% mean-diagonal damping, calls the real `torch.linalg.eigh(H)`, and returns `torch.stack(eigen_vecs + eigen_vecs)`, so the same real U is used for both halves. Nowhere in the paper or the code is there a complex or unitary U, a Hermitian covariance, or an Im(Σ) term.

A third-party survey (Serret et al., arXiv 2604.01757, App. B.2) re-describes RoRoPE in complex notation, K̂_l = K_{2l+1} + i K_{2l+2}. It calls U_l both "unitary … ∈ R^{N×N}" and "∈ C^{N×N}", and it writes the covariance with a transpose (K̂ᵀK̂) rather than a conjugate transpose. That is inconsistent and does not match the TransMLA code. It is a restatement, not a new method, and RoPE is still applied only to the first head. A reviewer could still cite it, so our write-up should state precisely that RoRoPE as published and implemented uses Re(Σ).

**(ii) Is RoPE kept on the retained channels?** Partly. RoPE is kept **only on the top-m components per frequency, placed in the first head** ("For each frequency we keep a single principal component … represented by a standard RoPE in one attention head", Fig. 2 caption). All other rotated components are **not discarded**. Their RoPE is **stripped**: "instead of discarding all non-principal components of the key, we remove their RoPE encoding" (Sec. 4.2). They become NoPE keys and are compressed jointly with V by norm-balanced PCA (Sec. 4.3, App. D). The model is therefore not exact. The paper reports zero-training numbers (Table 1) but relies on fine-tuning with about 6B tokens to recover. The first head always gets a uniform rank of 1 per frequency, or M per group of M frequencies with FreqFold. There is no global rank allocation across frequencies. The code has a `rope_head` argument that lets more than one head keep RoPE.

**(iii) FreqFold** (Sec. 4.2, App. C). FreqFold treats M adjacent RoPE frequencies as one "effective frequency θ*". It concatenates their 2g-dimensional segments and runs a single PCA over M·2g dimensions, which is still the real Re-style construction: in code the head axis becomes `num_kv_heads*freqfold`. It then keeps M components with that shared frequency. App. C.4 states that this "introduces a degree of deviation from the original, precise RoPE formulation". In other words, FreqFold is an **approximate** merge of adjacent frequencies. Method (c) never merges frequencies. It gives extra rank to a frequency by taking more complex eigenvectors of that frequency's own Σ_i, and RoPE stays exact.

**How (c) differs.** (1) Complex-unitary rather than real-orthogonal mixing per frequency. It captures the antisymmetric Im(Σ_i) cross-head correlations, for example head A's x correlated with head B's y. For the same cache width (2r real numbers per frequency), (c) keeps at least as much energy as (b), and strictly more whenever Im(Σ_i) ≠ 0 changes the top eigenspace. (2) All retained channels keep exact RoPE, and nothing is moved to NoPE. The whole key is an MQA-with-RoPE latent, with no NoPE key path and no K/V joint PCA. (3) Rank is allocated globally over frequencies from pooled eigenvalues, instead of a fixed 1 (or M) per frequency. (4) No FreqFold approximation is needed. (5) It is training-free by design.

## 2. Tucker Attention (Klein, Kusch, Sager, Schnake, Schotthöfer; arXiv 2603.30033 v1, 31 Mar 2026)

**What it does.** The paper treats the pre-softmax weights as a 3-way tensor over (heads, query-embed, key-embed) and **trains from scratch** a Tucker-factorized attention: core 𝒞, factors U1 over heads, U2 over the query side and U3 ∈ R^{d_model×r3} over the key side. Its "latent RoPE" (Definition 3.1, Eq. 12) is
`Q̂_m ×_3 (K_n R(n,r3) R(m,r3)ᵀ)` with `K = X U3`. Here R(ℓ, r3) ∈ R^{r3×r3} is "the rotary embedding in the latent key space". The proof (Lemma B.5, Eq. 15) uses only R(m)R(n)ᵀ = R(m−n). The paper states: "our modification only changes the rotation frequency by operating in the latent dimension rather than the head dimension." Corollary 3.2.1 applies the same idea to MLA ("MLA – coupled RoPE", Eqs. 13–14 and 20), where RoPE is applied to the d_c-dimensional KV latent and all query-side matrices are fused.

**Is its latent frequency-structured and complex-linear per frequency, i.e. the same as (c)?** **No.** U3 is an arbitrary **real**, learned, unstructured projection from the residual stream. The latent gets a **fresh** standard RoPE schedule over r3 (or d_c) dimensions. Nothing relates these to the pretrained per-head frequencies θ_i, and nothing is block-structured by frequency or complex-linear per frequency. The method is an architecture trained from scratch (GPT-2, ViT, Llama-3-1B pretraining) and not a conversion. Relative to any pretrained model it would change the positional kernel. It **does not kill the project.**

**But note:** Tucker Attention's inference form is "RoPE applied on a shared cached latent, with all query matrices absorbed". That is the same *inference architecture class* as (c)'s output, an MQA with RoPE over a shared latent. The paper claims "this is the first demonstration that MLA is compatible with RoPE without requiring decoupled position encodings." Plain MQA with RoPE is also this class. So (c) cannot claim novelty for the inference architecture. Its novelty must be the **exact, training-free, closed-form conversion**: a frequency-preserving complex PCA from pretrained multi-head keys, which gives a latent whose RoPE frequencies are the original θ_i with non-uniform multiplicity r_i. Cite Tucker Attention as the from-scratch counterpart.

## 3. KV-Latent (Shi et al., ACL 2025; arXiv 2507.11273)

**What it does.** KV-Latent shrinks the per-head K/V dimensions of a pretrained model by **uniform channel down-sampling** within each head (Sec. 3, Eq. 5 and App. C). It explicitly rejects SVD because "matrix multiplication does not satisfy the commutative property" with RoPE (line of Sec. 3.1, and the conclusion calls SVD integration "highly challenging"). It then **changes the RoPE frequency sampling** for the lower-dimensional heads (Sec. 3.3.3, Eq. 11): θ_j = θ^{−2(j−1+d/8)/d} or θ^{−(j−1+3d/4)/d}, which drops high frequencies and samples low frequencies more densely. It needs two-stage fine-tuning of less than 1% of pretraining tokens. **Confirmed: it changes the positional kernel and is not exact.** It does no cross-head mixing, no PCA and no complex structure. The overlap with (c) is only the goal of a smaller rotated key dimension.

## 4. Other related work

**MHA2MLA (Ji et al., ACL 2025; arXiv 2502.14837).** MHA2MLA converts full RoPE to partial RoPE. It keeps r frequency subspaces per head, chosen by S_high, S_low, S_uniform or S_{2-norm}, where the head-wise 2-norm contribution is best (Sec. 3, Fig. 3, App. A). It removes RoPE from the rest, which become NoPE, and applies a joint SVD to NoPE-K and V (SVD_joint). It then fine-tunes with 0.3–1% of pretraining data. Retained RoPE pairs are exact, but there is no cross-head mixing: the selection is diagonal per head, which is a coordinate-subset special case of (c) with V_i restricted to selection vectors. Dropping RoPE is lossy. (c) generalizes the per-frequency selection to a complex cross-head basis and keeps RoPE everywhere.

**Palu (Chang et al., ICLR 2025; arXiv 2407.21118).** Palu uses low-rank decomposition of W_K and W_V, per head, per group of heads (G-LRD, Sec. 3.2.3) or jointly, and caches the **pre-RoPE** latent. For RoPE models the key must be **reconstructed online** and then rotated, using a fused Triton kernel (Sec. 3.3 and App. B). Fusion into queries is possible only for non-RoPE attention. The mixing is real and unstructured, with no frequency structure. (c) removes the reconstruction entirely because its map commutes with RoPE.

**SALS (arXiv 2510.24273).** SALS projects pre-RoPE Q and K of all heads into a shared single-head latent. It uses **RoPE-free** latent scores only to *select* top-k tokens, then reconstructs those tokens and applies RoPE for the final attention (Sec. 1, 3). It observes that post-RoPE keys have higher rank. The projection is real and not RoPE-commuting, and the latent scoring is approximate. It is complementary to (c): (c) could replace SALS's RoPE-free latent scoring with exact-RoPE latent scoring.

**Eigen Attention (Saxena et al., arXiv 2408.05646).** Eigen Attention does real PCA of K, Q and V activations per layer. For RoPE models (Sec. 4.3, Fig. 6), "we leave the query to be full rank and transform the key back to a high dimension before applying the RoPE rotation matrix", with U^K shared across heads. This means reconstruction, and it shows latency *penalties* on Llama-3. It is not RoPE-commuting.

**RAP: RoPE-Aligned Pruning (arXiv 2602.02599).** RAP prunes W_K at **RoPE-pair granularity**. It ranks pairs by Fisher score and keeps the top pairs per head, so the binary pair-selection matrix commutes with RoPE and "fuses into W_q offline, so … decode needs no per-step reconstruction" (Sec. 1, 3, 4). The retained pairs keep exact RoPE and it is training-light (KD). RAP is the closest *exact-RoPE, no-reconstruction* compression. However, it restricts V_i to **coordinate selection within each head**, a real 0/1 map with no cross-head mixing. That is a strict special case of (c) and also of (b). RAP is a mandatory baseline, and it is also the "rank-allocation across frequencies without mixing" ablation.

**RoFormer (Su et al., arXiv 2104.09864).** RoFormer is the source of the complex view. Sec. 3.2.1 ("A 2D case"), Eq. 12, gives f_q(x_m, m) = (W_q x_m) e^{imθ}, f_k(x_n, n) = (W_k x_n) e^{inθ}, and g = Re[(W_q x_m)(W_k x_n)^* e^{i(m−n)θ}]. Sec. 3.4.3, Eq. 35, gives (R_m W_q x_m)ᵀ(R_n W_k x_n) = Re[Σ_i q_{[2i:2i+1]} k^*_{[2i:2i+1]} e^{i(m−n)θ_i}]. The complex form alone gives the commutation used by (c): complex-linear maps commute with scalar multiplication by e^{jmθ_i}.

**Additional related work found during the search (not killers):**
- *Align Attention Heads Before Merging Them* (Jin et al., arXiv 2412.20677). This MHA→GQA method uses Procrustes alignment. For keys, Eq. 14 restricts the per-head orthogonal matrix to block-diagonal **2-D rotations per RoPE pair**, which commute with RoPE and are unit-complex phases in the complex view. Heads are then merged by mean pooling plus L0 pruning and fine-tuning. It is prior art for using *complex phases* per pair to make heads more similar, but it uses only diagonal per-head phases, no cross-head complex PCA, and needs training.
- *CommVQ* (arXiv 2506.18879). CommVQ uses a VQ codebook for keys whose 2×2 blocks have the form [[x, y], [−y, x]] (Property 1, Sec. 3–4.2). That is a complex scalar per RoPE pair, so the codebook commutes with RoPE. It is prior art for the "complex scalar commutes with RoPE" observation in KV compression, but it applies to quantization within a head, not cross-head low-rank.
- *EliteKV* (arXiv 2503.01586). RoPElite greedily keeps each head's most important frequency chunks, the rest become linear (NoPE), and a joint low-rank projection is applied to NoPE-K and V. It requires uptraining. It is a per-head selection, like MHA2MLA.
- *Effectively Compress KV Heads* (Yu et al., arXiv 2406.07056, Sec. 3.4). It uses real PCA Ψ of **post-RoPE** keys and caches K̃ = KΨᵀ, applied to the rotated query. The projection is fixed while the rotated keys vary with position, so this is not RoPE-commuting.
- *X-EcoMLA* (arXiv 2503.11132), *CARE* (arXiv 2603.17946), *MHA2MLA-VLM* (arXiv 2601.11464), *Multi-Head Low-Rank Attention* (arXiv 2603.02188), *CCA* (arXiv 2510.04476), and *GTA* all use DeepSeek-style **decoupled** RoPE (a separate shared RoPE key) or partial RoPE, together with distillation, fine-tuning or pretraining.
- *GQLA* (arXiv 2605.15250, by the TransMLA group) reuses the TransMLA pipeline, including RoRoPE, FreqFold and balanced PCA, unchanged for its RoPE part. The code (`GQLA_preprint/src/compression.py`) has only real `eigh` PCA. It does not use a complex version.
- *STAR-KV* (arXiv 2606.08382) caches pre-RoPE latents and reconstructs them in a fused kernel, in the Palu style.

---

## Searches performed

WebSearch queries:
1. `Tucker Attention RoPE latent MLA arXiv 2026`
2. `TransMLA github RoRoPE code fxmeng`
3. `KV-Latent dimensional-level KV cache reduction frequency-aware rotary positional embedding 2025`
4. `complex PCA per RoPE frequency across attention heads KV cache compression`
5. `RoPE-equivariant low-rank key compression commutes with rotary embedding unitary`
6. `X-EcoMLA upcycling pre-trained attention into MLA RoPE`
7. `GQA to MLA conversion training-free exact RoPE 2026 arxiv`
8. `"RoPE" "complex-valued" low-rank projection shared latent keys heads "per frequency" MQA conversion`
9. `latent RoPE MLA without decoupled RoPE conversion pretrained model frequency-structured latent`
10. `RoPE-compatible low-rank KV cache no reconstruction query absorption rotary frequency block-diagonal projection`
11. `Hermitian covariance RoPE key pairs eigendecomposition attention heads compression LLM`
12. `"LatentRoPE" OR "latent RoPE" KV cache compression pretrained LLM training-free`
13. `frequency-wise cross-head key compression rotary embedding complex multiplication heads shared latent 2026`
14. `improving RoRoPE TransMLA complex unitary rotation real imaginary joint PCA extension`
15. `MQA conversion from MHA preserving RoPE exactly per-frequency head mixing calibration eigenvectors KV cache arXiv`
16. `"RoPE" key compression "complex PCA" OR "complex-valued PCA" OR "complex SVD" attention heads`
17. `rotary embedding commutes with complex linear map across heads same frequency KV cache MLA conversion without fine-tuning 2026`

Full text read (arXiv HTML) and grepped for `complex|unitary|hermitian|RoRoPE|per-frequency|commute|rope`: 2502.07864v5, 2603.30033v1, 2507.11273, 2502.14837, 2407.21118, 2510.24273, 2408.05646, 2602.02599, 2104.09864v5, 2604.01757, 2605.15250, 2603.17946, 2601.11464, 2603.02188, 2503.01586, 2406.07056, 2503.11132, 2606.08382, 2412.20677, 2510.04476, 2506.18879, 2607.23054. Code read: `MuLabPKU/TransMLA`, files `TransMLA_NeurIPS_2025/transmla/partial_rope.py` and `GQLA_preprint/src/compression.py`. `git clone` was blocked, so the files were fetched with raw.githubusercontent.

Limits of this check: web search is not exhaustive. Venue-only papers without arXiv HTML, and papers from after about mid-September 2026, may be missing.

---

## Verdict: **PROCEED**

None of the papers found performs **complex-linear (unitary / Hermitian-PCA) per-frequency cross-head key compression with exact RoPE on every retained channel.**

**Closest prior work: TransMLA's RoRoPE.** RoRoPE uses the same per-frequency cross-head decomposition, but with a **real** orthogonal U_l given by the eigenvectors of σ_x + σ_y = Re(Σ_l) (App. B, Eq. 32; code `joint_complex_pca`, which is real despite its name). In other words, **RoRoPE = our (b)**. Beyond the real/complex difference, RoRoPE-as-published also differs from our (b) in how it truncates:
- It keeps RoPE only on the top m = 1 component per frequency (M per group with FreqFold).
- It **strips RoPE** from the remaining components and keeps them as NoPE keys compressed jointly with V.
- It uses approximate FreqFold and fine-tunes with about 6B tokens.

**Precise delta of (c) over prior work:**
1. Complex-unitary V_i, the eigenvectors of the Hermitian Σ_i, instead of real U_i. (c) contains (b) as a strict subset. The gain is exactly the contribution of the antisymmetric Im(Σ_i) = E[y xᵀ − x yᵀ]. **This must be measured empirically.** If Im(Σ_i) ≈ 0 in real models, (c) collapses to (b) = RoRoPE's basis, and the project's contribution shrinks to points 2–4 below. A first experiment should report the ratio ‖Im Σ_i‖/‖Σ_i‖ and the captured-energy gap between (c) and (b) at equal cache size, per layer and frequency.
2. Exact RoPE on **all** cached channels. There is no NoPE channel and no RoPE stripping, and there is no key reconstruction (unlike Palu, Eigen Attention and SALS).
3. Global rank allocation across frequencies from pooled eigenvalues. RoRoPE uses a uniform 1 or M per frequency, and MHA2MLA, EliteKV and RAP use per-head selection.
4. No frequency merging, whereas FreqFold is approximate. The method is training-free and closed-form.

**Not novel, and the write-up must concede this:**
- The inference architecture "MQA / MLA with RoPE on the shared latent, queries absorbed" already exists (Tucker Attention 2603.30033, Corollary 3.2.1, trained from scratch; also plain MQA).
- The observation that complex scalars commute with RoPE is known (RoFormer Eq. 12 and 35; CommVQ Property 1).
- Using per-pair 2-D rotations (phases) of heads before merging is known (Jin et al., 2412.20677, Eq. 14).
- Real per-frequency cross-head PCA is known (RoRoPE).

### Required baselines

- **(b′) RoRoPE-as-published.** For each layer and frequency l:
  - Form H_l = Σ_tokens (x_l x_lᵀ + y_l y_lᵀ) over KV heads, with x and y the two `rotate_half` halves. Add 0.01·mean(diag H_l)·I. Take the real `eigh` and sort descending, giving U_l.
  - Rotate W_K (both halves with the same U_l) and absorb U_l into the queries.
  - Keep RoPE on the top m components per frequency. Default m = 1 (`rope_head=1`); optionally use FreqFold M ∈ {2, 4}, which pools M adjacent frequencies into one (g·M)×(g·M) PCA and applies a single representative frequency.
  - Treat the remaining rotated components as NoPE (no RoPE).
  - Compress those NoPE components jointly with V by norm-balanced PCA to the target latent size (Balanced-KV, App. D).
  - Report the result without training, since fine-tuning is out of scope.
  - Reference implementation: `MuLabPKU/TransMLA`, `TransMLA_NeurIPS_2025/transmla/partial_rope.py` plus `converter.py`.
- **(b) Idealized real version.** Use the same Re(Σ_i) eigenvectors as (b′), but with exact RoPE on all retained components, global rank allocation, and no NoPE path. This is the clean real-vs-complex ablation against (c).
- **RAP-style pair selection** (no mixing) and **MHA2MLA S_{2-norm}** (per-head partial RoPE) as frequency-selection baselines, plus **Palu** (reconstruction) as the conventional low-rank baseline.
