"""
main.py
=======
Full reproduction pipeline: all tables + all 5 figures.

Usage:
    python main.py               # everything (tables + figures)
    python main.py --tables      # tables only (fast, ~1 min)
    python main.py --figures     # figures only (~3-4 min)

Outputs are written to results/:
    tables_output.txt, table1.csv ... table4.csv,
    figure1_*.png/pdf ... figure5_*.png/pdf
"""

import sys
import io
import os
import argparse

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", action="store_true", help="tables only")
    p.add_argument("--figures", action="store_true", help="figures only")
    args = p.parse_args()
    run_tables = not args.figures or args.tables
    run_figs = not args.tables or args.figures

    print("=" * 70)
    print("BOPEC Sensitivity Analysis — full reproduction")
    print("Aghaei Ata, Jahromi, Keshavarzfard (2026)")
    print("=" * 70)

    if run_tables:
        from tables import run_all_tables
        with open(os.path.join(RESULTS, "tables_output.txt"), "w") as fout:
            run_all_tables(fout)

    if run_figs:
        import figures
        figures.figure1()
        figures.figure2()
        figures.figure3()
        figures.figure4()
        figures.figure5()

    print("\n" + "=" * 70)
    print(f"DONE — all outputs in {RESULTS}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
