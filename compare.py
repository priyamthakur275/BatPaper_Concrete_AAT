"""Prints the hybrid vs non-hybrid comparison from saved result files (res_*.json, ga_long_*.json).
Usage: python compare.py            (uses saved_results/ folder)
       python compare.py .          (uses res_*.json in the current folder)
"""
import sys, glob, json, statistics as st, os
d = sys.argv[1] if len(sys.argv) > 1 else "saved_results"
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "res_*.json")))]
G = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "ga_long_*.json")))]
if not R: sys.exit("No res_*.json found in " + d)
rows = []
for key, name, hy in (("MLR", "Linear regression", False), ("ANN", "ANN (back-propagation)", False), ("ANN-GA", "GA-ANN (300 gens)", True), ("ANFIS", "ANFIS (8 rules)", True)):
    rows.append((name, hy, st.mean(r[key]["r2"] for r in R), st.mean(r[key]["rmse"] for r in R), st.mean(r[key]["mae"] for r in R)))
if G and len(G) == len(R):
    rows.insert(3, ("GA-ANN (3000 gens)", True, st.mean(g["r2"] for g in G), st.mean(g["rmse"] for g in G), st.mean(g["mae"] for g in G)))
print(f"\nHYBRID vs NON-HYBRID  (mean of {len(R)} split(s), test set {R[0]['n_test']} samples)\n")
print(f"{'Model':26}{'Type':12}{'R2':>8}{'RMSE':>9}{'MAE':>8}")
for n, hy, r2, rm, ma in sorted(rows, key=lambda x: x[3]):
    print(f"{n:26}{'Hybrid' if hy else 'Non-hybrid':12}{r2:8.3f}{rm:9.2f}{ma:8.2f}")
bh = min((r for r in rows if r[1]), key=lambda x: x[3]); bn = min((r for r in rows if not r[1]), key=lambda x: x[3])
gain = (1 - bh[3] / bn[3]) * 100
print(f"\nBest hybrid     : {bh[0]}  (RMSE {bh[3]:.2f})")
print(f"Best non-hybrid : {bn[0]}  (RMSE {bn[3]:.2f})")
if bh[3] < bn[3]: print(f"RESULT: the best HYBRID model ({bh[0]}) is better than the best NON-HYBRID model ({bn[0]}): {gain:.1f}% lower RMSE.")
else: print(f"RESULT: the best non-hybrid model ({bn[0]}) is better than the best hybrid model ({bh[0]}) in these runs.")
