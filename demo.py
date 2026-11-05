"""Demo: non-hybrid ANN vs hybrid ANFIS.
python demo.py          -> instant: 5-split average from saved_results/
python demo.py live     -> trains both models now on split 0 (about 1 to 4 minutes)
"""
import sys, glob, json, statistics as st
def show(a, h, title):
    print(f"\n{title}\n")
    print(f"{'Model':28}{'Type':12}{'R2':>8}{'RMSE (MPa)':>12}{'MAE (MPa)':>11}")
    print(f"{'ANN (back-propagation)':28}{'Non-hybrid':12}{a[0]:8.3f}{a[1]:12.2f}{a[2]:11.2f}")
    print(f"{'ANFIS (fuzzy + neural)':28}{'Hybrid':12}{h[0]:8.3f}{h[1]:12.2f}{h[2]:11.2f}")
    print(f"\nHigher R2 is better, lower RMSE and MAE are better.")
    if h[1] < a[1] and h[0] > a[0]:
        print(f"RESULT: the HYBRID model (ANFIS) is better than the NON-HYBRID model (ANN): {(1 - h[1] / a[1]) * 100:.1f}% lower RMSE, higher R2.\n")
    else:
        print("RESULT: in this run the NON-HYBRID model (ANN) was as good or better than the hybrid ANFIS.\n")
if len(sys.argv) > 1 and sys.argv[1] == "live":
    import numpy as np, models as m
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    X, y = m.load(); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    sx = MinMaxScaler().fit(Xtr); sy = MinMaxScaler().fit(ytr.reshape(-1, 1))
    A, B, c = sx.transform(Xtr), sx.transform(Xte), sy.transform(ytr.reshape(-1, 1)).ravel()
    inv = lambda p: sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    print("Training non-hybrid ANN ..."); pa, _ = m.ann_backprop(A, c, B, 0); ma = m.metrics(yte, inv(pa))
    print("Training hybrid ANFIS ..."); ph, _ = m.anfis(A, c, B, 0); mh = m.metrics(yte, inv(ph))
    show((ma["r2"], ma["rmse"], ma["mae"]), (mh["r2"], mh["rmse"], mh["mae"]), "LIVE RUN (split 0, 206 unseen test samples)")
else:
    R = [json.load(open(f)) for f in sorted(glob.glob("saved_results/res_*.json"))]
    g = lambda k, q: st.mean(r[k][q] for r in R)
    show([g("ANN", q) for q in ("r2", "rmse", "mae")], [g("ANFIS", q) for q in ("r2", "rmse", "mae")], f"NON-HYBRID vs HYBRID (mean of {len(R)} splits, 206 unseen test samples each)")
