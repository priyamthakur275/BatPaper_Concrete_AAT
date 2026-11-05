"""ANFIS ensemble (hybrid): average of 12 ANFIS models with 24-72 fuzzy rules. Usage: python run_ens.py <seed> ..."""
import sys, json, time, numpy as np
import models as M
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
RULES = (24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72)

def run(seed):
    X, y = M.load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sx = MinMaxScaler().fit(Xtr); sy = MinMaxScaler().fit(ytr.reshape(-1, 1))
    a, b = sx.transform(Xtr), sx.transform(Xte); c = sy.transform(ytr.reshape(-1, 1)).ravel()
    t = time.time(); P = []
    for i, R in enumerate(RULES):
        p, _ = M.anfis(a, c, b, seed * 10 + i, rules=R); P.append(p)
    p = sy.inverse_transform(np.mean(P, 0).reshape(-1, 1)).ravel()
    out = dict(M.metrics(yte, p), seconds=time.time() - t, members=len(RULES), params=int(sum(r * 17 for r in RULES)))
    print(seed, {k: round(v, 3) for k, v in out.items() if k in ("r2", "rmse", "mae")}, flush=True)
    json.dump(out, open(f"saved_results/ens_{seed}.json", "w"))

if __name__ == "__main__":
    for s in map(int, sys.argv[1:]): run(s)
