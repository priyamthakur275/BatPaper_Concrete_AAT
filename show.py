"""Clean one-screen output: dataset summary + hybrid vs non-hybrid tables.  Run: python show.py"""
import json, glob, statistics as st, pandas as pd
def box(headers, rows, align):
    w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda l, m, r: l + m.join("─" * (k + 2) for k in w) + r
    fmt = lambda row: "│" + "│".join(" " + (str(v).ljust(k) if a == "l" else str(v).rjust(k)) + " " for v, k, a in zip(row, w, align)) + "│"
    print("\n".join([line("┌", "┬", "┐"), fmt(headers), line("├", "┼", "┤")] + [fmt(r) for r in rows] + [line("└", "┴", "┘")]))
d = pd.read_csv("data.csv")
R = [json.load(open(f)) for f in sorted(glob.glob("saved_results/res_*.json"))]
G = [json.load(open(f)) for f in sorted(glob.glob("saved_results/ga_long_*.json"))]
g = lambda k, q: st.mean(r[k][q] for r in R)
gl = lambda q: st.mean(x[q] for x in G)
EN = [json.load(open(f)) for f in sorted(glob.glob("saved_results/ens_*.json"))]
ge = lambda q: st.mean(x[q] for x in EN)
print("\n  HYBRID vs NON-HYBRID: CONCRETE COMPRESSIVE STRENGTH (UCI dataset)\n\n  1. DATASET\n")
box(["Item", "Value"], [["Source", "UCI Concrete Compressive Strength"], ["Size", f"{d.shape[0]} rows x {d.shape[1]} columns"], ["Inputs / target", "8 mix variables / strength (MPa)"],
    ["Missing values", int(d.isna().sum().sum())], ["Train / test split", "824 / 206 samples (80/20)"], ["Repeats", "5 random splits (mean shown)"]], ["l", "l"])
print("\n  2. RESULTS (test set, mean of 5 splits)\n")
rows = [["ANFIS ensemble (12 models)", "Hybrid", ge("r2"), ge("rmse"), ge("mae")], ["ANFIS (single)", "Hybrid", g("ANFIS", "r2"), g("ANFIS", "rmse"), g("ANFIS", "mae")], ["ANN (back-propagation)", "Non-hybrid", g("ANN", "r2"), g("ANN", "rmse"), g("ANN", "mae")],
        ["GA-ANN (3000 gens)", "Hybrid", gl("r2"), gl("rmse"), gl("mae")], ["Linear regression", "Non-hybrid", g("MLR", "r2"), g("MLR", "rmse"), g("MLR", "mae")]]
rows.sort(key=lambda x: x[3])
box(["Model", "Type", "R2", "RMSE (MPa)", "MAE (MPa)"], [[a, b, f"{c:.3f}", f"{e:.2f}", f"{f:.2f}"] for a, b, c, e, f in rows], ["l", "l", "r", "r", "r"])
print("  R2: higher is better.  RMSE, MAE: lower is better.")
a, f = g("ANN", "rmse"), ge("rmse")
print(f"\n  RESULT: the best HYBRID model (ANFIS ensemble) is {(1-f/a)*100:.1f}% better (lower RMSE) than the best non-hybrid model (ANN).")
print("\n  3. COMPARISON WITH BASE PAPER (Aalimahmoody et al. 2021, best model BAT-ANN)\n")
bp = dict(r2=0.913, rmse=5.26, mae=3.85)
box(["Metric", "Our ANFIS ensemble", "Base paper", "Our result"],
    [["R2", f"{ge('r2'):.3f}", "0.913", f"{(ge('r2')/0.913-1)*100:.1f}% higher (better)"], ["RMSE (MPa)", f"{ge('rmse'):.2f}", "5.26", f"{(1-ge('rmse')/bp['rmse'])*100:.1f}% lower (better)"],
     ["MAE (MPa)", f"{ge('mae'):.2f}", "3.85", f"{(1-ge('mae')/bp['mae'])*100:.1f}% lower (better)"]], ["l", "r", "r", "l"])
print("  Note: our values are a mean of 5 random 80/20 splits; the paper reports one 70/30 split.\n")
