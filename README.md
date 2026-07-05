# BOPEC Sensitivity Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)]()

Complete, reproducible code for the paper:

> **Analytical Sensitivity Bounds for System-Level Performance in Bi-Level Optimization with Nash Equilibrium Constraints**
> H. Aghaei Ata, M. Jahromi, R. Keshavarzfard (2026)

This repository reproduces **all tables (1–4)** and **all five figures** of the manuscript from a single command.

---

## Quick Start

```bash
git clone https://github.com/redminotee1398-lgtm/BOPEC-Sensitivity-Analysis
cd BOPEC-Sensitivity-Analysis
pip install -r requirements.txt

python main.py            # everything: tables + 5 figures (~4 min)
python main.py --tables   # tables only (~1 min)
python main.py --figures  # figures only (~3 min)
```

All outputs are written to `results/`:

```
results/
├── tables_output.txt          # Tables 1–4 + Cournot, console format
├── table1.csv … table4.csv    # machine-readable tables
├── figure1_theorem2_bounds.png/.pdf
├── figure2_decomposition.png/.pdf
├── figure3_phase_heatmap.png/.pdf
├── figure4_validation.png/.pdf
└── figure5_decision_error.png/.pdf
```

---

## Repository Structure

| File | Contents |
|---|---|
| `bopec_core.py` | Theorem 1 (decomposition), Theorem 2 (interval bounds), **Lemma 1** (strong-monotonicity check), **Proposition 1** (sign-reversal condition), and the three test systems |
| `tables.py` | Reproduces Tables 1–4 + Cournot results; writes TXT + CSV |
| `figures.py` | Generates all 5 publication figures (600 dpi PNG + PDF) |
| `main.py` | One-command pipeline |
| `requirements.txt` | Pinned minimum dependency versions |

---

## Mathematical Content

**Theorem 1 (Sensitivity Decomposition).** Under smoothness, strong monotonicity of the stacked game operator, and a nonsingular game Jacobian:

$$\frac{dZ}{dB} = \underbrace{\frac{\partial Z}{\partial x}\frac{df}{dB}}_{S_{\rm direct}} + \underbrace{\frac{\partial Z}{\partial y^*}\!\left(-H^{-1}G\frac{df}{dB}\right)}_{S_{\rm indirect}}$$

**Lemma 1 (implemented in `lemma1_check`).** Strong monotonicity is verified numerically as $\mu = \lambda_{\min}(\tfrac12(H+H^\top)) > 0$. For the linear system $H = 1.9I + 0.1\mathbf{1}\mathbf{1}^\top$, eigenvalues are $\{1.9,\ 1.9+0.1n\}$, so $\mu = 1.9$ **uniformly in $n$** (see `table3.csv`, column `mu`).

**Proposition 1 (implemented in `proposition1_check`).** Sign reversal occurs iff the indirect term exceeds $|S_{\rm direct}|$; the Cauchy–Schwarz sufficient condition
$\|\partial Z/\partial y^*\|\cdot\|H^{-1}G\|\cdot|df/dB| > |S_{\rm direct}|$
is checked per row of Table 4 (column `Prop1_satisfied`).

**Theorem 2 (Interval Bounds).** Endpoint + critical-point evaluation on $[B_L,B_U]$; critical points are bracketed by dense grid evaluation (one $O(n^3)$ solve per point — see the paper's Computational Remark, Section 3.3).

---

## Test Systems

| System | Definition | Paper section |
|---|---|---|
| **Linear** | $Z = 10B - x^2 - \sum y_i^2$, $x = 0.5B+0.3$, quadratic $f_i$ with bilinear coupling $0.1\sum_{j\neq i}y_iy_j$ | §4.1, Tables 1–3 |
| **Nonlinear** | $Z = 5B - x^2 - \sum y_i^2$, $x = B^2+1$, quartic $f_i$ with exponential coupling $0.1\sum_{j\neq i}y_ie^{y_j}$ | §4.2, Table 4 |
| **Cournot duopoly** | $Z = (y_1^*+y_2^*)^2 - Bx$, $x = 0.5B+1$, $c_i(x)=c_i^0-\alpha x$, $\alpha = 0.4199$, $\gamma=0.5$ | §4.3 |

---

## Verified Results

### Table 1 — Linear (B=1.0, n=3)

| Component | Paper | Code | Match |
|---|---|---|---|
| Direct | +9.200000 | +9.200000 | ✓ |
| Indirect | −0.502755 | −0.502755 | ✓ |
| Total | +8.697245 | +8.697245 | ✓ (FD error 4.4e−12) |

### Table 4 — Nonlinear (B=1.0)

| n | Direct | Indirect | Total | Sign reversal | Prop. 1 |
|---|---|---|---|---|---|
| 10 | −3.000 | +0.966 | −2.034 | no | False |
| 20 | −3.000 | +4.628 | **+1.628** | **YES** | True |
| 50 | −3.000 | +20.913 | **+17.913** | **YES** | True |

Critical transition: **n\* ≈ 16.5** (Figure 3A).

### Cournot Duopoly (B=1.0)

| Component | Paper | Code |
|---|---|---|
| Direct | −0.500 | −0.500 ✓ |
| Indirect | +1.916 | +1.916 ✓ |
| Total | +1.416 | +1.416 ✓ |
| Theorem 2 bounds on [0.5, 2.0] | — | **[0.972, 1.638]** → welfare-improving certified |

---

## Figures

| Figure | Content | Paper location |
|---|---|---|
| **Figure 1** | Theorem 2 certified interval bounds — linear (A) + Cournot (B) | §4 opening |
| **Figure 2** | Decomposition overview — linear bars (A), nonlinear sign reversal (B), trajectories (C), dominance ratio (D) | §4.1–4.2 |
| **Figure 3** | Phase diagram of sign reversal vs. n (A) + 2-D sensitivity heatmap over (B, n) with zero contour (B) | §4.2 |
| **Figure 4** | Three-way validation (A), log-scale errors (B), speedup vs. Kriging (C) | §4.3 |
| **Figure 5** | Decision-error region (A) + naive-vs-correct bars (B) | §5.1 |

---

## Experimental Protocol (Reproducibility)

Addressing standard reproducibility requirements:

- **Environment:** Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10, scikit-learn ≥ 1.3, Matplotlib ≥ 3.7. Single CPU core; no GPU required.
- **Equilibrium solver:** `scipy.optimize.fsolve` with analytical Jacobian, default tolerance `xtol = 1.49e-8`, **deterministic** initial guess `y0 = 0.3·1` (documented fallback `y0 = 0`). No randomness in core results.
- **Random seed:** `RNG_SEED = 42` (used *only* by the Monte Carlo comparison in Table 2).
- **Finite differences:** central, `h = 1e-6` (linear) / `1e-7` (nonlinear).
- **Kriging:** Matérn-5/2 GP, 500 (linear) / 300 (nonlinear) training points on `[B−1.5, B+1.5]`, 5 optimizer restarts. Reported CPU time **includes surrogate construction**, since the surrogate must be rebuilt when system parameters change. Speedup ratios are therefore protocol-dependent; in amortized multi-query settings the relative advantage would be smaller.
- **Timing:** wall-clock via `time.perf_counter()`; absolute times vary by hardware, but **error values and all sensitivity values are hardware-independent**.

---

## Cite

```bibtex
@article{aghaeiata2026bopec,
  title   = {Analytical Sensitivity Bounds for System-Level Performance
             in Bi-Level Optimization with Nash Equilibrium Constraints},
  author  = {Aghaei Ata, Hossein and Jahromi, Meghdad and Keshavarzfard, Razieh},
  year    = {2026},
  note    = {Under review}
}

@software{aghaeiata2026code,
  author = {Aghaei Ata, Hossein and Jahromi, Meghdad and Keshavarzfard, Razieh},
  title  = {BOPEC Sensitivity Analysis: reproducible code},
  year   = {2026},
  url    = {https://github.com/redminotee1398-lgtm/fuzzy-bopec-sensitivity}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Meghdad Jahromi** (Corresponding Author) — Meghdadjahromi@iau.ac.ir — ORCID [0000-0001-6360-5347](https://orcid.org/0000-0001-6360-5347)
**Hossein Aghaei Ata** — aghaeiata@yahoo.co.uk
**Razieh Keshavarzfard** — r.keshavarzfard@iau.ir — ORCID [0000-0002-2212-3576](https://orcid.org/0000-0002-2212-3576)
