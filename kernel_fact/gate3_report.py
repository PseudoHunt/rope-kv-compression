"""Gate 3 verdict + figure from results/kf_gate3_fidelity_raw.csv."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .gate2_report import INK, MUTED, STYLE, style_ax

ROOT = "/home/jl_fs/rope_equiv"


def main():
    d = pd.read_csv(f"{ROOT}/results/kf_gate3_fidelity_raw.csv")
    print("sequences per source:", d.groupby("src").seq.nunique().to_dict())
    L = d.groupby(["arm", "layer"])[["relL2", "kl"]].mean()
    ex = L.loc["EXACT"]
    ratio = {a: (L.loc[a].relL2 / ex.relL2) for a in d.arm.unique()}
    med = ratio["OPT/16"].median()
    v = "PASS" if med <= 1.2 else ("KILL" if med > 1.5 else "GREY")
    print(f"PRIMARY: median over layers relL2(OPT/16)/relL2(EXACT) = {med:.3f} -> {v}")
    for src, g in d.groupby("src"):
        Ls = g.groupby(["arm", "layer"]).relL2.mean()
        print(f"   {src}: {(Ls.loc['OPT/16'] / Ls.loc['EXACT']).median():.3f}")
    rows = []
    for a, r in ratio.items():
        rows.append(dict(arm=a, median_ratio=r.median(), max_ratio=r.max(), worst_layer=int(r.idxmax()),
                         mean_relL2=L.loc[a].relL2.mean(), mean_kl=L.loc[a].kl.mean()))
    t = pd.DataFrame(rows).sort_values("median_ratio")
    print(t.round(3).to_string(index=False))
    t.to_csv(f"{ROOT}/results/kf_gate3_summary.csv", index=False)
    for p in [8, 16, 32]:
        best = min([f"{a}/{p}" for a in ["PR-hi", "PR-en", "PR-2n", "FOLD-mean"]], key=lambda a: ratio[a].median())
        print(f"p={p}: OPT {ratio[f'OPT/{p}'].median():.3f} vs best special case {best} {ratio[best].median():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, p in zip(axes, [8, 16, 32]):
        for arm in ["OPT", "PR-en", "PR-2n", "FOLD-mean", "PR-hi"]:
            col, mk = STYLE[arm]
            r = ratio[f"{arm}/{p}"]
            ax.plot(r.index, r.values, color=col, marker=mk, ms=4, lw=2, label=arm)
            ax.annotate(arm, (r.index[-1], r.values[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK, va="center")
        rn = ratio["NOPE"]
        ax.plot(rn.index, rn.values, color=MUTED, lw=1, ls=":", label="NOPE")
        ax.axhline(1.2, color=INK, lw=1, ls="--")
        ax.axhline(1.5, color=INK, lw=1, ls="-.")
        ax.set_yscale("log")
        ax.set_title(f"p = {p}", fontsize=10, color=INK)
        ax.set_xlabel("layer", fontsize=9, color=MUTED)
        style_ax(ax)
    axes[0].set_ylabel("relL2(arm) / relL2(EXACT)  (single-layer substitution)", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Gate 3: attention-output error relative to EXACT, ρ = 25% (pass ≤ 1.2 --, kill > 1.5 -.)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/kf_gate3_fidelity.png", dpi=140)


if __name__ == "__main__":
    main()
