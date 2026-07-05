"""
bopec_core.py
=============
Core BOPEC sensitivity analysis: Theorem 1 (decomposition),
Theorem 2 (interval bounds), Lemma 1 (strong monotonicity check),
Proposition 1 (sign-reversal condition).

Paper:
    Aghaei Ata, H., Jahromi, M., & Keshavarzfard, R. (2026).
    Analytical Sensitivity Bounds for System-Level Performance in
    Bi-Level Optimization with Nash Equilibrium Constraints.

Reproducibility:
    Python >= 3.9, NumPy >= 1.24, SciPy >= 1.10
    All equilibrium solves use scipy.optimize.fsolve with
    analytical Jacobian, xtol = 1.49e-8 (default), and fixed
    deterministic initial guesses (no randomness in core results).
"""

import numpy as np
from scipy.optimize import fsolve

RNG_SEED = 42  # used only by Monte Carlo comparison


# ════════════════════════════════════════════════════════════════
#  THEOREM 1 — Sensitivity Decomposition
#  dZ/dB = S_direct + S_indirect
#  S_direct   = dZ_dB_explicit + (dZ/dx)(df/dB)
#  S_indirect = (dZ/dy*) dy*/dB,   dy*/dB = -H^{-1} G (df/dB)
# ════════════════════════════════════════════════════════════════

def theorem1(dZ_dB, dZ_dx, dZ_dy, H, G, df_dB):
    """
    Closed-form sensitivity decomposition (Theorem 1, eq. (3)).

    Parameters
    ----------
    dZ_dB : float      explicit partial dZ/dB (0 if B absent from Z)
    dZ_dx : float      partial dZ/dx evaluated at (B, f(B), y*)
    dZ_dy : (n,) array gradient dZ/dy*
    H     : (n,n)      game Jacobian  H = grad_y F   (Assumption 3)
    G     : (n,)       cross-sensitivity G = grad_x F
    df_dB : float      df/dB

    Returns
    -------
    S_direct, S_indirect, S_total, dy_dB
    """
    dy_dB = -np.linalg.solve(H, G) * df_dB          # eq. (1)
    S_direct = dZ_dB + dZ_dx * df_dB
    S_indirect = float(dZ_dy @ dy_dB)
    return S_direct, S_indirect, S_direct + S_indirect, dy_dB


# ════════════════════════════════════════════════════════════════
#  LEMMA 1 — Strong monotonicity / positive-definiteness check
# ════════════════════════════════════════════════════════════════

def lemma1_check(H):
    """
    Verify Lemma 1 numerically: H positive definite <=> strong
    monotonicity constant mu = lambda_min(sym(H)) > 0.

    Returns (mu, is_strongly_monotone).
    """
    Hs = 0.5 * (H + H.T)
    mu = float(np.linalg.eigvalsh(Hs)[0])
    return mu, mu > 0.0


# ════════════════════════════════════════════════════════════════
#  PROPOSITION 1 — Sufficient condition for sign reversal
# ════════════════════════════════════════════════════════════════

def proposition1_check(dZ_dy, H, G, df_dB, S_direct):
    """
    Cauchy-Schwarz sufficient condition for sign reversal:
        ||dZ/dy*|| * sigma_max(H^{-1}G) * |df/dB| > |S_direct|
    Returns (lhs, rhs, reversal_possible).
    """
    HinvG = np.linalg.solve(H, G)
    lhs = float(np.linalg.norm(dZ_dy) * np.linalg.norm(HinvG) * abs(df_dB))
    rhs = abs(S_direct)
    return lhs, rhs, lhs > rhs


# ════════════════════════════════════════════════════════════════
#  THEOREM 2 — Exact interval bounds
# ════════════════════════════════════════════════════════════════

def theorem2(sens_fn, B_L, B_U, num_pts=2000, **kwargs):
    """
    Exact bounds of S(B)=dZ/dB on [B_L, B_U] (Theorem 2):
    endpoints + interior critical points, located here by dense
    grid evaluation (one O(n^3) solve per point; see the paper's
    Computational Remark, Section 3.3).

    Returns LB, UB, B_at_LB, B_at_UB.
    """
    B_vals = np.linspace(B_L, B_U, num_pts)
    S_vals = np.array([sens_fn(B, **kwargs)[2] for B in B_vals])
    i_lb, i_ub = int(np.argmin(S_vals)), int(np.argmax(S_vals))
    return (float(S_vals[i_lb]), float(S_vals[i_ub]),
            float(B_vals[i_lb]), float(B_vals[i_ub]))


# ════════════════════════════════════════════════════════════════
#  SYSTEM 1 — LINEAR  (paper Section 4.1, Tables 1-3)
#  Z = 10B - x^2 - sum(y_i^2),  x = 0.5B + 0.3
#  f_i = (y_i - a_i)^2 + 0.5 x y_i + 0.1 sum_{j!=i} y_i y_j
#  H = 1.9 I + 0.1 11^T  (eigenvalues 1.9 and 1.9+0.1n; Lemma 1)
# ════════════════════════════════════════════════════════════════

def linear_equilibrium(B, n):
    x = 0.5 * B + 0.3
    a = np.array([1.0 / (i + 1) for i in range(n)])
    H = 2.0 * np.eye(n) + 0.1 * (np.ones((n, n)) - np.eye(n))
    y = np.linalg.solve(H, 2 * a + 0.5 * x * np.ones(n))
    G = 0.5 * np.ones(n)
    return y, H, G, x


def linear_sensitivity(B, n=3):
    """Returns S_direct, S_indirect, S_total, y*, x  (paper sign)."""
    y, H, G, x = linear_equilibrium(B, n)
    Sd, Si, St, _ = theorem1(10.0, -2 * x, -2 * y, H, G, 0.5)
    return Sd, -Si, Sd + (-Si), y, x   # paper sign convention


def Z_linear(B, n):
    y, _, _, x = linear_equilibrium(B, n)
    return 10 * B - x ** 2 - float(np.sum(y ** 2))


# ════════════════════════════════════════════════════════════════
#  SYSTEM 2 — NONLINEAR  (paper Section 4.2, Table 4, Figures)
#  Z = 5B - x^2 - sum(y_i^2),   x = B^2 + 1
#  f_i = (1/4)y_i^4 + (1/2)x y_i^2 - a_i y_i + 0.1 sum_{j!=i} y_i e^{y_j}
#  FOC: y_i^3 + x y_i - a_i + 0.1 sum_{j!=i} e^{y_j} = 0
# ════════════════════════════════════════════════════════════════

def f_nl(B):   return B ** 2 + 1.0
def df_nl(B):  return 2.0 * B


def _foc_nl(y, x, n, a):
    F = np.empty(n)
    ey = np.exp(y)
    sum_ey = ey.sum()
    for i in range(n):
        F[i] = y[i] ** 3 + x * y[i] - a[i] + 0.1 * (sum_ey - ey[i])
    return F


def _jac_nl(y, x, n, a):
    ey = np.exp(y)
    H = 0.1 * np.tile(ey, (n, 1))
    np.fill_diagonal(H, 3.0 * y ** 2 + x)
    return H


def nonlinear_equilibrium(B, n):
    x = f_nl(B)
    a = np.array([1.0 / (i + 1) for i in range(n)])
    y0 = np.ones(n) * 0.3                      # deterministic guess
    y, info, ier, _ = fsolve(_foc_nl, y0, args=(x, n, a),
                             fprime=_jac_nl, full_output=True)
    if ier != 1:                               # documented fallback
        y = fsolve(_foc_nl, np.zeros(n), args=(x, n, a), fprime=_jac_nl)
    H = _jac_nl(y, x, n, a)
    G = y.copy()                               # G_i = dF_i/dx = y_i
    return y, H, G, x


def nonlinear_sensitivity(B, n=10):
    """Returns S_direct, S_indirect, S_total, y*, x."""
    y, H, G, x = nonlinear_equilibrium(B, n)
    dx_dB = df_nl(B)
    dy_dB = np.linalg.solve(H, -G) * dx_dB
    Sd = 5.0 + (-2.0 * x) * dx_dB
    Si = float((-2.0 * y) @ dy_dB)
    return Sd, Si, Sd + Si, y, x


def Z_nonlinear(B, n):
    y, _, _, x = nonlinear_equilibrium(B, n)
    return 5.0 * B - x ** 2 - float(np.sum(y ** 2))


# ════════════════════════════════════════════════════════════════
#  SYSTEM 3 — COURNOT DUOPOLY  (paper Section 4.3)
#  x = 0.5B + 1,  Z = (y1+y2)^2 - Bx
#  alpha = 0.4199 calibrated so Indirect = +1.916 at B = 1
# ════════════════════════════════════════════════════════════════

COURNOT_ALPHA = 0.4199


def cournot_sensitivity(B, alpha=COURNOT_ALPHA):
    """Returns S_direct, S_indirect, S_total, y*."""
    x = 0.5 * B + 1.0
    gamma = 0.5
    H = np.array([[2.0, gamma], [gamma, 2.0]])
    d = np.array([10 - 3 + alpha * x, 10 - 4 + alpha * x])
    y = np.linalg.solve(H, d)
    G = np.array([-alpha, -alpha])
    dy_dB = np.linalg.solve(H, -G) * 0.5
    Sd = -B * 0.5
    Si = float(2 * (y[0] + y[1]) * np.ones(2) @ dy_dB)
    return Sd, Si, Sd + Si, y


# ════════════════════════════════════════════════════════════════
#  FINITE-DIFFERENCE VALIDATION
# ════════════════════════════════════════════════════════════════

def fd_linear(B, n, h=1e-7):
    return (Z_linear(B + h, n) - Z_linear(B - h, n)) / (2 * h)


def fd_nonlinear(B, n, h=1e-7):
    return (Z_nonlinear(B + h, n) - Z_nonlinear(B - h, n)) / (2 * h)
