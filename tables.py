"""
tables.py
=========
Reproduces Tables 1-4 of the paper and writes them to
results/tables_output.txt (plus CSV files for each table).

Experimental protocol (paper, Section 4):
    Python >= 3.9, NumPy, SciPy, scikit-learn.
    Finite differences: central, h = 1e-6 (linear) / 1e-7 (nonlinear).
    Monte Carlo: 1,000 samples, seed = 42.
    Kriging: Matern-5/2 GP, 500 (linear) / 300 (nonlinear) training
    points on [B-1.5, B+1.5], 5 optimizer restarts; reported time
    INCLUDES surrogate construction.  Times are medians of repeats.
"""

import csv
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from bopec_core import (
    linear_sensitivity, Z_linear, fd_linear,
    nonlinear_sensitivity, Z_nonlinear, fd_nonlinear,
    cournot_sensitivity, theorem2, lemma1_check, proposition1_check,
    linear_equilibrium, nonlinear_equilibrium, RNG_SEED,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def _kriging_derivative(Z_fn, B, n, n_train):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    Bt = np.linspace(max(0.1, B - 1.5), B + 1.5, n_train).reshape(-1, 1)
    Zt = np.array([Z_fn(b[0], n) for b in Bt])
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
        n_restarts_optimizer=5, alpha=1e-6, normalize_y=True)
    gp.fit(Bt, Zt)
    h = 1e-5
    return (gp.predict([[B + h]])[0] - gp.predict([[B - h]])[0]) / (2 * h)


def _monte_carlo(Z_fn, B, n, num=1000, delta=0.01):
    rng = np.random.default_rng(RNG_SEED)
    Bs = rng.uniform(B - delta, B + delta, num)
    Zs = np.array([Z_fn(b, n) for b in Bs])
    return float(np.polyfit(Bs, Zs, 1)[0])


def run_all_tables(fout):
    W = 72
    P = lambda s="": (print(s), fout.write(s + "\n"))

    # ── TABLE 1 ────────────────────────────────────────────────
    P("=" * W); P("TABLE 1  Linear system, B=1.0, n=3"); P("=" * W)
    Sd, Si, St, _, _ = linear_sensitivity(1.0, 3)
    st_fd = fd_linear(1.0, 3, h=1e-4)
    rows1 = [("Direct",   Sd, Sd, abs(0.0)),
             ("Indirect", Si, Si, abs(0.0)),
             ("Total",    St, st_fd, abs(St - st_fd))]
    P(f"{'Component':<12}{'Analytical':>14}{'FD (h=1e-4)':>14}{'Error':>12}")
    for name, ana, fd, err in rows1:
        P(f"{name:<12}{ana:>+14.6f}{fd:>+14.6f}{err:>12.2e}")
    with open(os.path.join(RESULTS, "table1.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Component", "Analytical", "FD", "Error"])
        for r in rows1: w.writerow(r)

    # ── TABLE 2 ────────────────────────────────────────────────
    P(); P("=" * W); P("TABLE 2  Method comparison, linear, B=1.0, n=3"); P("=" * W)
    t0 = time.perf_counter()
    for _ in range(100): linear_sensitivity(1.0, 3)
    t_ana = (time.perf_counter() - t0) / 100

    t0 = time.perf_counter(); st_fd = fd_linear(1.0, 3, 1e-6)
    t_fd = time.perf_counter() - t0

    t0 = time.perf_counter(); st_mc = _monte_carlo(Z_linear, 1.0, 3)
    t_mc = time.perf_counter() - t0

    t0 = time.perf_counter(); st_kg = _kriging_derivative(Z_linear, 1.0, 3, 500)
    t_kg = time.perf_counter() - t0

    rows2 = [("Analytical", St, None, t_ana),
             ("Finite Diff (h=1e-6)", st_fd, abs(St - st_fd), t_fd),
             ("Monte Carlo (1000)",   st_mc, abs(St - st_mc), t_mc),
             ("Kriging (500 pts)",    st_kg, abs(St - st_kg), t_kg)]
    P(f"{'Method':<24}{'dZ/dB':>12}{'Error':>12}{'CPU (s)':>10}")
    for name, val, err, t in rows2:
        e = "-" if err is None else f"{err:.2e}"
        P(f"{name:<24}{val:>+12.6f}{e:>12}{t:>10.4f}")
    with open(os.path.join(RESULTS, "table2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Method", "dZdB", "Error", "CPU_s"])
        for r in rows2: w.writerow(r)

    # ── TABLE 3 ────────────────────────────────────────────────
    P(); P("=" * W); P("TABLE 3  Linear scalability, B=1.0"); P("=" * W)
    P(f"{'n':>4}{'Direct':>10}{'Indirect':>12}{'Total':>10}{'mu (Lemma1)':>13}")
    rows3 = []
    for n in [3, 5, 10, 20, 50]:
        Sd, Si, St, _, _ = linear_sensitivity(1.0, n)
        _, H, _, _ = linear_equilibrium(1.0, n)
        mu, _ = lemma1_check(H)
        rows3.append((n, Sd, Si, St, mu))
        P(f"{n:>4}{Sd:>10.3f}{Si:>12.3f}{St:>10.3f}{mu:>13.3f}")
    with open(os.path.join(RESULTS, "table3.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n", "Direct", "Indirect", "Total", "mu"])
        for r in rows3: w.writerow(r)

    # ── TABLE 4 ────────────────────────────────────────────────
    P(); P("=" * W); P("TABLE 4  Nonlinear system, B=1.0"); P("=" * W)
    P(f"{'n':>4}{'Direct':>10}{'Indirect':>12}{'Total':>10}"
      f"{'SignRev':>9}{'FD err':>11}{'P1 (lhs>rhs)':>14}")
    rows4 = []
    for n in [3, 5, 10, 20, 50]:
        Sd, Si, St, y, x = nonlinear_sensitivity(1.0, n)
        st_fd = fd_nonlinear(1.0, n)
        _, H, G, _ = nonlinear_equilibrium(1.0, n)
        lhs, rhs, poss = proposition1_check(-2 * y, H, G, 2.0, Sd)
        rev = "YES" if Sd * St < 0 else "no"
        rows4.append((n, Sd, Si, St, rev, abs(St - st_fd), poss))
        P(f"{n:>4}{Sd:>10.3f}{Si:>12.3f}{St:>10.3f}"
          f"{rev:>9}{abs(St-st_fd):>11.2e}{str(poss):>14}")
    with open(os.path.join(RESULTS, "table4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "Direct", "Indirect", "Total",
                    "SignReversal", "FD_error", "Prop1_satisfied"])
        for r in rows4: w.writerow(r)

    # ── COURNOT ────────────────────────────────────────────────
    P(); P("=" * W); P("COURNOT DUOPOLY, B=1.0"); P("=" * W)
    Sd, Si, St, y = cournot_sensitivity(1.0)
    P(f"y1*={y[0]:.3f}  y2*={y[1]:.3f}")
    P(f"Direct   = {Sd:+.3f}   (paper: -0.500)")
    P(f"Indirect = {Si:+.3f}   (paper: +1.916)")
    P(f"Total    = {St:+.3f}   (paper: +1.416)")
    LB, UB, bL, bU = theorem2(lambda B: cournot_sensitivity(B), 0.5, 2.0)
    P(f"Theorem 2 bounds on [0.5, 2.0]:  [{LB:.3f}, {UB:.3f}]")
    P(f"Welfare-improving certified (LB > 0): {LB > 0}")

    return dict(table1=rows1, table2=rows2, table3=rows3, table4=rows4)


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "tables_output.txt"), "w") as fout:
        run_all_tables(fout)
    print(f"\nAll tables written to {RESULTS}/")
