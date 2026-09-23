"""Gate 2 verdict + figures from results/kf_gate2_scoremse.csv and kf_gate2_spectrum.csv."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/home/jl_fs/rope_equiv"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
# fixed color per arm (validated default categorical order), never re-assigned
STYLE = {"OPT": ("#2a78d6", "o"), "OPT-unw": ("#eb6834", "s"), "PR-hi": ("#1baf7a", "D"),
         "PR-en": ("#eda100", "^"), "PR-2n": ("#e87ba4", "v"), "FOLD-mean": ("#008300", "P"),
         "OPT-joint": ("#4a3aa7", "X"), "OPT-layer": ("#e34948", "*")}


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def verdict(df):
    out = []
    d = df[df.rho == 0.25]
    opt16 = d[(d.arm == "OPT") & (d.p == 16)].set_index("layer").rho_s
    med = opt16.median()
    n2 = int((opt16 <= 2.0).sum())
    if med <= 1.44 and n2 >= 28:
        v = "PASS"
    elif med > 2.0:
        v = "KILL"
    else:
        v = "GREY"
    out.append(f"PRIMARY (rho=25%): median rho_s(OPT,16) = {med:.3f}; layers <= 2.0: {n2}/32  -> {v}")
    for p in [4, 8, 16, 32]:
        piv = d[d.p == p].pivot(index="layer", columns="arm", values="mse_w")
        base = piv[["PR-hi", "PR-en", "FOLD-mean"]].min(1)
        dom = (piv["OPT"] <= base).mean()
        dom2 = (piv["OPT"] <= piv[["PR-hi", "PR-en", "PR-2n", "FOLD-mean"]].min(1)).mean()
        out.append(f"DOMINANCE p={p}: OPT <= min(PR-hi, PR-en, FOLD-mean) on {dom:.0%} of layers "
                   f"(incl. PR-2n: {dom2:.0%})" + ("" if p == 32 else ("  ok" if dom >= 0.8 else "  FAILED (<80%)")))
    ps = d[d.arm == "OPT"].pivot(index="layer", columns="p", values="rho_s")
    pstar = ps.apply(lambda r: next((p for p in sorted(ps.columns) if r[p] <= 1.44), ">32"), axis=1)
    out.append("p*_l (smallest p with rho_s(OPT) <= 1.44), per layer: " + " ".join(map(str, pstar.tolist())))
    return "\n".join(out), pstar


def plots(df, sp):
    # median rho_s vs p per arm
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, rho in zip(axes, [0.25, 0.125]):
        d = df[df.rho == rho]
        for arm, (col, mk) in STYLE.items():
            s = d[d.arm == arm].groupby("p").rho_s.median()
            if len(s):
                ax.plot(s.index, s.values, color=col, marker=mk, lw=2, ms=5, label=arm)
                ax.annotate(arm, (s.index[-1], s.values[-1]), xytext=(4, 0), textcoords="offset points",
                            fontsize=7, color=INK, va="center")
        nope = d[d.arm == "NOPE"].rho_s.median()
        ax.axhline(nope, color=MUTED, lw=1, ls=":", label=f"NOPE ({nope:.0f})")
        ax.axhline(1.44, color=INK, lw=1, ls="--")
        ax.axhline(2.0, color=INK, lw=1, ls="-.")
        ax.text(4, 1.44, " pass 1.44", fontsize=7, color=INK, va="bottom")
        ax.text(4, 2.0, " kill 2.0", fontsize=7, color=INK, va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
        ax.set_xlabel("kernel rank p (EXACT = 64)", fontsize=9, color=MUTED)
        ax.set_title(f"ρ = {rho:g} (r = {int(rho * 4096)})", fontsize=10, color=INK)
        style_ax(ax)
    axes[0].set_ylabel("median over layers of ρ_s = MSE_X / MSE_EXACT", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Gate 2: attention-weighted score MSE relative to EXACT (held-out C4-val)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/kf_gate2_rho_vs_p.png", dpi=140)
    plt.close(fig)

    # spectrum: cumulative weighted-kernel energy vs p
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    cols = ["cum4", "cum8", "cum16", "cum32"]
    for ax, l in zip(axes, [0, 15, 31]):
        s = sp[sp.layer == l]
        loc, glo = s.loc[s.mean_gap.idxmin()], s.loc[s.mean_gap.idxmax()]
        med = s[cols].median()
        for row, lab, col in [(med, "median head", "#2a78d6"),
                              (loc, f"most local head {int(loc['head'])} (mean gap {loc.mean_gap:.0f})", "#eb6834"),
                              (glo, f"most global head {int(glo['head'])} (mean gap {glo.mean_gap:.0f})", "#1baf7a")]:
            ax.plot([4, 8, 16, 32], [1 - row[c] for c in cols], color=col, marker="o", lw=2, label=lab)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
        ax.axhline(1e-2, color=MUTED, lw=1, ls="--")
        ax.axhline(1e-3, color=MUTED, lw=1, ls=":")
        ax.set_title(f"layer {l}: median p99 = {s.p99.median():.0f}, p99.9 = {s.p999.median():.0f}",
                     fontsize=10, color=INK)
        ax.set_xlabel("p", fontsize=9, color=MUTED)
        style_ax(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0].set_ylabel("1 − retained weighted kernel energy", fontsize=9, color=MUTED)
    fig.suptitle("Gate 2 (i): singular-value spectrum of W = diag(√π) E diag(√ε)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/kf_gate2_spectrum.png", dpi=140)
    plt.close(fig)


def main():
    df = pd.read_csv(f"{ROOT}/results/kf_gate2_scoremse.csv")
    sp = pd.read_csv(f"{ROOT}/results/kf_gate2_spectrum.csv")
    text, _ = verdict(df)
    print(f"n_seq = {df.n_seq.iloc[0]}")
    print(text)
    for rho in [0.25, 0.125]:
        d = df[df.rho == rho]
        print(f"\nrho={rho}: median over layers of rho_s (weighted) / (unweighted)")
        a = d.pivot_table(index="arm", columns="p", values="rho_s", aggfunc="median")
        b = d.pivot_table(index="arm", columns="p", values="rho_s_u", aggfunc="median")
        print(a.round(3).to_string())
        print(b.round(3).to_string())
    fl = pd.read_csv(f"{ROOT}/results/kf_gate2_floor.csv")
    print("\ninstrument floor / MSE_EXACT (max over layers):", (fl.floor_w / fl.exact_w).max())
    print("spectrum median p99 / p999:", sp.p99.median(), sp.p999.median())
    plots(df, sp)


if __name__ == "__main__":
    main()
