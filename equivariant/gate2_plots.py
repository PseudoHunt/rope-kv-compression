"""Gate 2 and Gate 3 figures from results/gate2_energy.csv, gate2_meta.json and cache/calib_stats.pt, gate3_fidelity_raw.csv."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = "/home/jl_fs/rope_equiv"
# Fixed color per method (validated default categorical order, slots 1..7), never re-assigned.
STYLE = {
    "a": ("#2a78d6", "o", "(a) pre-RoPE PCA (needs recon.)"),
    "b": ("#eb6834", "s", "(b) per-freq real cross-head"),
    "c": ("#1baf7a", "D", "(c) per-freq complex (this)"),
    "d": ("#eda100", "^", "(d) post-RoPE PCA"),
    "bp": ("#e87ba4", "v", "(b') RoRoPE uniform alloc."),
    "e": ("#008300", "P", "(e) per-freq block, any real map*"),
    "cp": ("#4a3aa7", "X", "(c') complex, uniform alloc."),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f7f6f2", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def energy_plot(df, methods, fname, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, rho in zip(axes, [0.5, 0.25, 0.125]):
        sub = df[df.rho == rho]
        for m in methods:
            col, mk, lab = STYLE[m]
            s = sub[sub.method == m].sort_values("layer")
            ax.plot(s.layer, 100 * s.energy_heldout, color=col, lw=2, marker=mk, ms=4, label=lab)
            ax.annotate(m, (s.layer.iloc[-1], 100 * s.energy_heldout.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.set_title(f"ρ = {rho:g}  (K budget {int(rho * 4096)} floats/token/layer)", fontsize=10, color=INK)
        ax.set_xlabel("layer", fontsize=9, color=MUTED)
        style_ax(ax)
    axes[0].set_ylabel("retained key energy, held-out C4 (%)", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/{fname}", dpi=140)
    plt.close(fig)


def main():
    df = pd.read_csv(f"{ROOT}/results/gate2_energy.csv")
    meta = json.load(open(f"{ROOT}/results/gate2_meta.json"))
    energy_plot(df, ["a", "d", "c", "b", "bp"], "gate2_energy.png",
                "Gate 2: key energy retained at equal K budget (bases fit on C4-train, evaluated on held-out C4-val)")
    energy_plot(df, ["a", "e", "c", "cp", "b"], "gate2_decomposition.png",
                "Where (c) loses to (a): cross-frequency (a→e) vs complex-linearity (e→c) vs allocation (c→c')   "
                "*(e) is not RoPE-exact")

    # rank allocation heatmap, rho = 0.25
    A = np.array([meta["rank_alloc_c"][str(l)]["0.25"] for l in range(32)]).T  # [P, L]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(A, aspect="auto", cmap=SEQ, vmin=0, vmax=32, origin="lower")
    ax.set_xlabel("layer", fontsize=9, color=MUTED)
    ax.set_ylabel("RoPE pair i  (θ_i = 10000^(-i/64); high i = slow)", fontsize=9, color=MUTED)
    ax.set_title("(c) rank allocation r_i at ρ = 0.25 (R = 512 complex channels / layer)", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("r_i (of 32 heads)", fontsize=9, color=MUTED)
    ax.axhline(45.5, color=INK, lw=1, ls="--")
    ax.text(0.5, 46.2, "above: θ_i·4096 < 2π (under one full turn in context)", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/gate2_rank_alloc.png", dpi=140)
    plt.close(fig)

    # per-frequency spectrum of Sigma_i for layers 0, 15, 31
    stats = torch.load("/home/jl_fs/rope_equiv/cache/calib_stats.pt")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, l in zip(axes, [0, 15, 31]):
        ev = torch.linalg.eigvalsh(stats[l]["Sigma"]).flip(-1).numpy()  # [P, n]
        frac = ev / ev.sum(1, keepdims=True)
        im = ax.imshow(np.log10(np.clip(frac, 1e-6, None)), aspect="auto", cmap=SEQ.reversed(),
                       vmin=-5, vmax=0, origin="lower")
        ax.set_title(f"layer {l}: eig(Σ_i) / tr(Σ_i), log10", fontsize=10, color=INK)
        ax.set_xlabel("eigen-index j (of 32 heads)", fontsize=9, color=MUTED)
        ax.set_ylabel("RoPE pair i", fontsize=9, color=MUTED)
        k90 = (np.cumsum(frac, 1) < 0.9).sum(1) + 1
        ax.plot(k90 - 1, np.arange(64), color="#eb6834", lw=1.5, label="90% energy")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Cross-head redundancy per frequency (fast decay = heads share structure within a frequency)",
                 fontsize=11, color=INK)
    fig.savefig(f"{ROOT}/results/gate2_spectrum.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def gate3_plot():
    df = pd.read_csv(f"{ROOT}/results/gate3_fidelity_raw.csv")
    L = df.groupby(["rho", "method", "layer"]).relL2.mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, rho in zip(axes, [0.5, 0.25, 0.125]):
        for m in ["a", "d", "c", "b", "bp"]:
            col, mk, lab = STYLE[m]
            s = L[(L.rho == rho) & (L.method == m)].sort_values("layer")
            ax.plot(s.layer, s.relL2, color=col, lw=2, marker=mk, ms=4, label=lab)
            ax.annotate(m, (s.layer.iloc[-1], s.relL2.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        a = L[(L.rho == rho) & (L.method == "a")].sort_values("layer")
        ax.plot(a.layer, 1.5 * a.relL2, color=MUTED, lw=1, ls="--", label="1.5 × (a)  (Gate 3 pass line)")
        ax.set_title(f"ρ = {rho:g}", fontsize=10, color=INK)
        ax.set_xlabel("layer", fontsize=9, color=MUTED)
        style_ax(ax)
    axes[0].set_ylabel("attention-output relL2 (single-layer substitution)", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Gate 3: per-layer output error, 50 held-out C4 + 50 GSM8K 8-shot prompts (V uncompressed)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/gate3_fidelity.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
    gate3_plot()
