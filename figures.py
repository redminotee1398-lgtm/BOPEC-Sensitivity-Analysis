"""
figures.py
==========
Generates all 5 publication figures of the paper (600 dpi PNG + PDF)
into results/.

Figure 1  Theorem 2 interval bounds (linear + Cournot)
Figure 2  Sensitivity decomposition overview (4 panels)
Figure 3  Phase diagram + (B,n) sensitivity heatmap
Figure 4  Validation & computational efficiency (3 panels)
Figure 5  Policy decision-error analysis (2 panels)

Numbering matches the revised manuscript:
    Fig 1 -> Section 4 opening (Theorem 2 bounds)
    Fig 2 -> Sections 4.1 / 4.2 (decomposition)
    Fig 3 -> Section 4.2 (phase diagram)
    Fig 4 -> Section 4.3 (validation)
    Fig 5 -> Section 5.1 (decision errors)
"""

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

from bopec_core import (
    linear_sensitivity, nonlinear_sensitivity, cournot_sensitivity,
    Z_nonlinear, fd_nonlinear,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "mathtext.fontset": "cm",
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})

C = dict(direct="#2166AC", indirect="#B2182B", total="#1B7837",
         num="#F4A582", krig="#D6604D", lin="#7570B3",
         accent="#FF6600", cournot="#9467bd")


def save(fig, name):
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(RESULTS, name + ext))
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ════════════════════════════════════════════════════════════════
#  FIGURE 1 — Theorem 2 interval bounds
# ════════════════════════════════════════════════════════════════

def figure1():
    print("Figure 1 (Theorem 2 bounds) ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.32)

    # (A) linear n=3
    Bv = np.linspace(0.3, 1.7, 300)
    Sv = np.array([linear_sensitivity(B, 3)[2] for B in Bv])
    Sd = np.array([linear_sensitivity(B, 3)[0] for B in Bv])
    Si = np.array([linear_sensitivity(B, 3)[1] for B in Bv])
    BL, BU = 0.5, 1.5
    m = (Bv >= BL) & (Bv <= BU)
    LB, UB = Sv[m].min(), Sv[m].max()
    ax1.axvspan(BL, BU, alpha=0.08, color="gold", label=r"$[B_L,B_U]$")
    ax1.plot(Bv, Sd, "--", color=C["direct"], lw=1.6, label=r"$S_{\rm direct}$")
    ax1.plot(Bv, Si, ":", color=C["indirect"], lw=1.6, label=r"$S_{\rm indirect}$")
    ax1.plot(Bv, Sv, "-", color=C["total"], lw=2.4, label=r"$S_{\rm total}$")
    for val, col in ((LB, "navy"), (UB, "maroon")):
        ax1.axhline(val, color=col, lw=1.2, ls="-.", alpha=0.8)
    ax1.annotate(f"LB={LB:.3f}", (BU + .03, LB), fontsize=9, color="navy",
                 va="center", fontweight="bold")
    ax1.annotate(f"UB={UB:.3f}", (BU + .03, UB), fontsize=9, color="maroon",
                 va="center", fontweight="bold")
    ax1.set_xlabel("Policy parameter $B$"); ax1.set_ylabel(r"$dZ/dB$")
    ax1.set_title("(A) Linear system ($n=3$) — certified bounds",
                  fontweight="bold")
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.25)

    # (B) Cournot
    Bv2 = np.linspace(0.3, 2.2, 300)
    Sv2 = np.array([cournot_sensitivity(B)[2] for B in Bv2])
    Sd2 = np.array([cournot_sensitivity(B)[0] for B in Bv2])
    Si2 = np.array([cournot_sensitivity(B)[1] for B in Bv2])
    BL2, BU2 = 0.5, 2.0
    m2 = (Bv2 >= BL2) & (Bv2 <= BU2)
    LB2, UB2 = Sv2[m2].min(), Sv2[m2].max()
    ax2.axvspan(BL2, BU2, alpha=0.08, color="gold", label=r"$[B_L,B_U]$")
    ax2.plot(Bv2, Sd2, "--", color=C["direct"], lw=1.6, label=r"$S_{\rm direct}$")
    ax2.plot(Bv2, Si2, ":", color=C["indirect"], lw=1.6, label=r"$S_{\rm indirect}$")
    ax2.plot(Bv2, Sv2, "-", color=C["cournot"], lw=2.4, label=r"$S_{\rm total}$")
    for val, col in ((LB2, "navy"), (UB2, "maroon")):
        ax2.axhline(val, color=col, lw=1.2, ls="-.", alpha=0.8)
    ax2.annotate(f"LB={LB2:.3f}", (BU2 + .04, LB2), fontsize=9, color="navy",
                 va="center", fontweight="bold")
    ax2.annotate(f"UB={UB2:.3f}", (BU2 + .04, UB2), fontsize=9, color="maroon",
                 va="center", fontweight="bold")
    ax2.annotate("Welfare-improving\ncertified throughout",
                 xy=(1.25, (LB2 + UB2) / 2), fontsize=8.5, ha="center",
                 color="#006600", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e8ffe8",
                           ec="#006600", lw=0.8))
    ax2.set_xlabel("Policy parameter $B$"); ax2.set_ylabel(r"$dZ/dB$")
    ax2.set_title("(B) Cournot duopoly — welfare certification",
                  fontweight="bold")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.25)

    fig.suptitle("Figure 1.  Exact sensitivity bounds via Theorem 2",
                 fontsize=13, fontweight="bold", y=1.005)
    save(fig, "figure1_theorem2_bounds")


# ════════════════════════════════════════════════════════════════
#  FIGURE 2 — Decomposition overview (4 panels)
# ════════════════════════════════════════════════════════════════

def figure2():
    print("Figure 2 (decomposition overview) ...")
    n_vals = [3, 5, 10, 20, 50]
    lin = {n: linear_sensitivity(1.0, n) for n in n_vals}
    nl = {n: nonlinear_sensitivity(1.0, n) for n in n_vals}

    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
    pos = np.arange(len(n_vals)); w = 0.25

    # (A) linear bars
    ax = fig.add_subplot(gs[0, 0])
    for off, key, idx, lab in ((-w, "direct", 0, "Direct"),
                                (0, "indirect", 1, "Indirect"),
                                (w, "total", 2, "Total")):
        ax.bar(pos + off, [lin[n][idx] for n in n_vals], w, label=lab,
               color=C[key], edgecolor="white", zorder=3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(pos); ax.set_xticklabels(map(str, n_vals))
    ax.set_xlabel("Number of followers $n$"); ax.set_ylabel(r"$dZ/dB$")
    ax.set_title("(A) Linear — decomposition ($B=1$)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.25)

    # (B) nonlinear bars with sign-reversal highlight
    ax = fig.add_subplot(gs[0, 1])
    St_nl = [nl[n][2] for n in n_vals]
    ax.bar(pos - w, [nl[n][0] for n in n_vals], w, label="Direct",
           color=C["direct"], edgecolor="white", zorder=3)
    ax.bar(pos, [nl[n][1] for n in n_vals], w, label="Indirect",
           color=C["indirect"], edgecolor="white", zorder=3)
    ax.bar(pos + w, St_nl, w, label="Total", edgecolor="white", zorder=3,
           color=[C["accent"] if v > 0 else C["total"] for v in St_nl])
    ax.axhline(0, color="black", lw=1.0)
    for i, v in enumerate(St_nl):
        if v > 0:
            ax.annotate("Sign\nreversal", xy=(pos[i] + w, v),
                        xytext=(pos[i] + w + .15, v + .5), fontsize=7.5,
                        color=C["accent"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=C["accent"]))
    ax.set_xticks(pos); ax.set_xticklabels(map(str, n_vals))
    ax.set_xlabel("Number of followers $n$"); ax.set_ylabel(r"$dZ/dB$")
    ax.set_title(r"(B) Nonlinear — sign reversal at $n\geq 20$",
                 fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.25)

    # (C) trajectories
    ax = fig.add_subplot(gs[1, 0])
    Bv = np.linspace(0.3, 2.2, 150)
    cmap = plt.cm.plasma
    for k, n in enumerate([3, 10, 20, 50]):
        tot = [nonlinear_sensitivity(B, n)[2] for B in Bv]
        ax.plot(Bv, tot, color=cmap(0.15 + 0.7 * k / 3), lw=1.8,
                label=f"$n={n}$")
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("Policy parameter $B$")
    ax.set_ylabel(r"$S_{\rm total}(B)$")
    ax.set_title(r"(C) Nonlinear trajectories $S_{\rm total}(B)$",
                 fontweight="bold")
    ax.legend(ncol=2); ax.grid(alpha=0.25)

    # (D) dominance ratio
    ax = fig.add_subplot(gs[1, 1])
    nn = np.array(n_vals)
    r_lin = [abs(lin[n][1]) / abs(lin[n][2]) * 100 for n in n_vals]
    r_nl = [abs(nl[n][1]) / (abs(nl[n][0]) + abs(nl[n][1])) * 100
            for n in n_vals]
    ax.plot(nn, r_lin, "s--", color=C["lin"], lw=1.8, ms=7,
            markeredgecolor="white", label="Linear")
    ax.plot(nn, r_nl, "D-", color=C["indirect"], lw=1.8, ms=7,
            markeredgecolor="white", label="Nonlinear")
    ax.fill_between(nn, r_nl, alpha=0.12, color=C["indirect"])
    ax.axhline(50, color="grey", lw=0.8, ls=":", label="50% threshold")
    ax.set_xlabel("Number of followers $n$")
    ax.set_ylabel(r"$|S_{\rm indirect}|/|S_{\rm total}|$  (%)")
    ax.set_title("(D) Indirect dominance ratio", fontweight="bold")
    ax.set_ylim(0, 115); ax.legend(); ax.grid(alpha=0.25)

    fig.suptitle("Figure 2.  Sensitivity decomposition: linear vs. nonlinear",
                 fontsize=13, fontweight="bold", y=1.005)
    save(fig, "figure2_decomposition")


# ════════════════════════════════════════════════════════════════
#  FIGURE 3 — Phase diagram + heatmap
# ════════════════════════════════════════════════════════════════

def figure3():
    print("Figure 3 (phase diagram + heatmap) ...")
    n_arr = np.arange(3, 51)
    Sd = np.empty(len(n_arr)); Si = np.empty(len(n_arr)); St = np.empty(len(n_arr))
    for k, n in enumerate(n_arr):
        d, i, t, _, _ = nonlinear_sensitivity(1.0, n)
        Sd[k], Si[k], St[k] = d, i, t
    sc = np.where(np.diff(np.sign(St)))[0]
    n_crit = float(n_arr[sc[0]]) + abs(St[sc[0]]) / (abs(St[sc[0]]) + abs(St[sc[0] + 1])) \
        if len(sc) else None

    B_h = np.linspace(0.3, 2.2, 50)
    n_h = np.arange(3, 51, 2)
    Zh = np.empty((len(n_h), len(B_h)))
    for i, n in enumerate(n_h):
        for j, B in enumerate(B_h):
            Zh[i, j] = nonlinear_sensitivity(B, n)[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.32)

    ax1.plot(n_arr, Sd, "--", color=C["direct"], lw=1.8, label=r"$S_{\rm direct}$")
    ax1.plot(n_arr, Si, ":", color=C["indirect"], lw=1.8, label=r"$S_{\rm indirect}$")
    ax1.plot(n_arr, St, "-", color=C["total"], lw=2.4, label=r"$S_{\rm total}$")
    ax1.axhline(0, color="black", lw=0.9)
    if n_crit:
        ax1.axvline(n_crit, color=C["accent"], lw=1.4, ls="-.",
                    label=f"$n^*\\approx{n_crit:.1f}$")
        ax1.fill_betweenx([St.min() - 1, St.max() + 1], n_crit, 51,
                          alpha=0.06, color=C["accent"])
    ax1.set_xlabel("Number of followers $n$")
    ax1.set_ylabel("Sensitivity component")
    ax1.set_title("(A) Phase diagram — sign reversal ($B=1$)",
                  fontweight="bold")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.25); ax1.set_xlim(3, 50)

    vmax = max(abs(Zh.min()), abs(Zh.max()))
    im = ax2.imshow(Zh, aspect="auto", origin="lower",
                    extent=[B_h[0], B_h[-1], n_h[0], n_h[-1]],
                    cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0,
                                                     vmax=vmax))
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04,
                 label=r"$dZ/dB$")
    cs = ax2.contour(B_h, n_h, Zh, levels=[0], colors="black",
                     linewidths=1.5)
    ax2.clabel(cs, fmt=r"$dZ/dB=0$", fontsize=8)
    ax2.set_xlabel("Policy parameter $B$")
    ax2.set_ylabel("Number of followers $n$")
    ax2.set_title(r"(B) Sensitivity map $S_{\rm total}(B,n)$",
                  fontweight="bold")

    fig.suptitle("Figure 3.  Sign-reversal phase diagram and $(B,n)$ map",
                 fontsize=13, fontweight="bold", y=1.008)
    save(fig, "figure3_phase_heatmap")
    return n_crit


# ════════════════════════════════════════════════════════════════
#  FIGURE 4 — Validation & efficiency
# ════════════════════════════════════════════════════════════════

def figure4():
    print("Figure 4 (validation & efficiency) ... (this takes ~2 min)")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    def kriging_nl(B, n, n_pts=300):
        Bt = np.linspace(max(0.1, B - 1.5), B + 1.5, n_pts).reshape(-1, 1)
        Zt = np.array([Z_nonlinear(b[0], n) for b in Bt])
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
            n_restarts_optimizer=5, alpha=1e-6, normalize_y=True)
        gp.fit(Bt, Zt)
        h = 1e-5
        return (gp.predict([[B + h]])[0] - gp.predict([[B - h]])[0]) / (2 * h)

    n_vals = [3, 5, 10, 20, 50]
    tot, fd, kg, e_fd, e_kg, sp_kg = [], [], [], [], [], []
    for n in n_vals:
        t0 = time.perf_counter(); _, _, St, _, _ = nonlinear_sensitivity(1.0, n)
        t_ana = time.perf_counter() - t0
        st_fd = fd_nonlinear(1.0, n)
        t0 = time.perf_counter(); st_kg = kriging_nl(1.0, n)
        t_kg = time.perf_counter() - t0
        tot.append(St); fd.append(st_fd); kg.append(st_kg)
        e_fd.append(abs(St - st_fd)); e_kg.append(abs(St - st_kg))
        sp_kg.append(t_kg / max(t_ana, 1e-9))
        print(f"   n={n}: done")

    nn = np.array(n_vals)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.35)

    ax = axes[0]
    ax.plot(nn, tot, "s-", color=C["total"], ms=8, lw=2, label="Analytical")
    ax.plot(nn, fd, "D--", color=C["num"], ms=7, lw=1.6, label="Finite diff.")
    ax.plot(nn, kg, "o:", color=C["krig"], ms=7, lw=1.6, label="Kriging")
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$dZ/dB$")
    ax.set_title("(A) Three-way validation", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.semilogy(nn, e_fd, "D--", color=C["num"], ms=7, lw=1.8,
                label="Finite diff.")
    ax.semilogy(nn, e_kg, "o:", color=C["krig"], ms=7, lw=1.8,
                label="Kriging")
    ax.set_xlabel("$n$"); ax.set_ylabel("Absolute error")
    ax.set_title("(B) Approximation error (log)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    ax.semilogy(nn, sp_kg, "s-", color=C["krig"], ms=8, lw=2,
                label="vs. Kriging")
    for x_, y_ in zip(nn, sp_kg):
        ax.annotate(f"{y_:.0f}x", (x_, y_), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8,
                    fontweight="bold", color=C["krig"])
    ax.set_xlabel("$n$"); ax.set_ylabel("Speedup (x)")
    ax.set_title("(C) Computational speedup", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25, which="both")

    fig.suptitle("Figure 4.  Validation and efficiency — nonlinear system",
                 fontsize=13, fontweight="bold", y=1.008)
    save(fig, "figure4_validation")


# ════════════════════════════════════════════════════════════════
#  FIGURE 5 — Decision-error analysis
# ════════════════════════════════════════════════════════════════

def figure5():
    print("Figure 5 (decision errors) ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.35)

    Bv = np.linspace(0.3, 2.2, 150)
    n = 20
    Sd = np.array([nonlinear_sensitivity(B, n)[0] for B in Bv])
    St = np.array([nonlinear_sensitivity(B, n)[2] for B in Bv])
    ax1.plot(Bv, Sd, "--", color=C["direct"], lw=1.8,
             label=r"$S_{\rm direct}$ (naive)")
    ax1.plot(Bv, St, "-", color=C["total"], lw=2.4,
             label=r"$S_{\rm total}$ (Theorem 1)")
    ax1.fill_between(Bv, Sd, St, where=St > Sd, alpha=0.12,
                     color=C["indirect"], label="Indirect correction")
    wrong = Sd * St < 0
    ax1.fill_between(Bv, min(Sd.min(), St.min()) - 1,
                     max(Sd.max(), St.max()) + 1, where=wrong, alpha=0.08,
                     color="red", label="Decision-error region")
    ax1.axhline(0, color="black", lw=0.9)
    ax1.set_xlabel("Policy parameter $B$"); ax1.set_ylabel(r"$dZ/dB$")
    ax1.set_title(r"(A) Naive vs. correct sensitivity ($n=20$)",
                  fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5); ax1.grid(alpha=0.25)

    n_vals = [5, 10, 20, 50]
    pos = np.arange(len(n_vals)); w = 0.35
    Sd_v = [nonlinear_sensitivity(1.0, n)[0] for n in n_vals]
    St_v = [nonlinear_sensitivity(1.0, n)[2] for n in n_vals]
    ax2.bar(pos - w / 2, Sd_v, w, label=r"$S_{\rm direct}$ (naive)",
            color=C["direct"], alpha=0.75, edgecolor="white", zorder=3)
    ax2.bar(pos + w / 2, St_v, w, label=r"$S_{\rm total}$ (correct)",
            color=[C["accent"] if v > 0 else C["total"] for v in St_v],
            edgecolor="white", zorder=3)
    ax2.axhline(0, color="black", lw=1.0)
    for i, (sd, st) in enumerate(zip(Sd_v, St_v)):
        if sd * st < 0:
            ax2.annotate("Wrong\ndirection", xy=(pos[i] + w / 2, st),
                         xytext=(pos[i] + w / 2 + .3, st + .5), fontsize=8,
                         color="red", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="red"))
    ax2.set_xticks(pos); ax2.set_xticklabels([f"$n={n}$" for n in n_vals])
    ax2.set_ylabel(r"$dZ/dB$  ($B=1$)")
    ax2.set_title(r"(B) Decision error from $S_{\rm direct}$ alone",
                  fontweight="bold")
    ax2.legend(); ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Figure 5.  Policy decision errors when ignoring "
                 "indirect equilibrium effects",
                 fontsize=13, fontweight="bold", y=1.005)
    save(fig, "figure5_decision_error")


if __name__ == "__main__":
    figure1()
    figure2()
    n_crit = figure3()
    figure4()
    figure5()
    print(f"\nAll figures saved to {RESULTS}/")
    if n_crit:
        print(f"Critical sign-reversal size: n* ~ {n_crit:.1f}")
