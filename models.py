"""Hybrid vs non-hybrid soft computing models for concrete compressive strength (UCI dataset, Yeh 1998)."""
import os, time, warnings
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

H = 8   # hidden neurons of the shared 8-H-1 network

def load():
    d = pd.read_csv("data.csv"); return d.iloc[:, :8].to_numpy(float), d.iloc[:, 8].to_numpy(float)

def metrics(y, p):
    return {"r2": r2_score(y, p), "rmse": mean_squared_error(y, p) ** 0.5, "mae": mean_absolute_error(y, p)}

# ---------- shared 8-H-1 network (tanh hidden, linear output) ----------
NW = 8 * H + H + H + 1
def unpack(w):
    i = 0; W1 = w[i:i+8*H].reshape(8, H); i += 8*H; b1 = w[i:i+H]; i += H; W2 = w[i:i+H]; i += H; b2 = w[i]
    return W1, b1, W2, b2
def forward(w, X):
    W1, b1, W2, b2 = unpack(w); return np.tanh(X @ W1 + b1) @ W2 + b2

# ---------- non-hybrid: ANN trained by back-propagation (Adam) ----------
def ann_backprop(Xtr, ytr, Xte, seed, epochs=400):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(8,)), tf.keras.layers.Dense(H, activation="tanh"), tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
    m.fit(Xtr, ytr, epochs=epochs, batch_size=32, verbose=0)
    return m.predict(Xte, verbose=0).ravel(), m.count_params()

# ---------- hybrid 1: ANN whose weights are evolved by a genetic algorithm ----------
def ann_ga(Xtr, ytr, Xte, seed, pop=80, gens=300):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (pop, NW))
    fit = lambda w: np.mean((forward(w, Xtr) - ytr) ** 2)
    f = np.array([fit(w) for w in P])
    for g in range(gens):
        new = [P[f.argmin()].copy()]                                   # elitism
        while len(new) < pop:
            a = min(rng.integers(0, pop, 3), key=lambda i: f[i]); b = min(rng.integers(0, pop, 3), key=lambda i: f[i])
            al = rng.random(NW); child = al * P[a] + (1 - al) * P[b]   # blend crossover
            mask = rng.random(NW) < 0.1; child[mask] += rng.normal(0, 0.3, mask.sum())  # mutation
            new.append(child)
        P = np.array(new); f = np.array([fit(w) for w in P])
    best = P[f.argmin()]
    return forward(best, Xte), NW

# ---------- hybrid 2: ANFIS (Takagi-Sugeno neuro-fuzzy), rules initialised by clustering ----------
def anfis(Xtr, ytr, Xte, seed, rules=8, epochs=300):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    km = KMeans(rules, n_init=10, random_state=seed).fit(Xtr)
    c = tf.Variable(km.cluster_centers_, dtype=tf.float32)                      # membership centres (rules x 8)
    s = tf.Variable(np.full((rules, 8), 0.35), dtype=tf.float32)                # membership widths
    A = tf.Variable(tf.random.normal((rules, 8), stddev=0.1)); B = tf.Variable(tf.zeros((rules,)))  # linear consequents
    opt = tf.keras.optimizers.Adam(0.01)
    Xt, yt = tf.constant(Xtr, tf.float32), tf.constant(ytr, tf.float32)
    def call(X):
        mu = tf.exp(-0.5 * tf.reduce_sum(((X[:, None, :] - c[None]) / (tf.abs(s)[None] + 1e-3)) ** 2, axis=2))  # firing strength (min/product AND)
        w = mu / (tf.reduce_sum(mu, 1, keepdims=True) + 1e-9)                    # normalised
        return tf.reduce_sum(w * (X @ tf.transpose(A) + B), axis=1)
    n = len(Xtr)
    for ep in range(epochs):
        idx = np.random.default_rng(seed + ep).permutation(n)
        for k in range(0, n, 64):
            b = idx[k:k+64]
            with tf.GradientTape() as t: loss = tf.reduce_mean((call(tf.gather(Xt, b)) - tf.gather(yt, b)) ** 2)
            g = t.gradient(loss, [c, s, A, B]); opt.apply_gradients(zip(g, [c, s, A, B]))
    return call(tf.constant(Xte, tf.float32)).numpy(), int(rules * 8 * 2 + rules * 9)

def run(seed):
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sx = MinMaxScaler().fit(Xtr); sy = MinMaxScaler().fit(ytr.reshape(-1, 1))
    Xtr_s, Xte_s = sx.transform(Xtr), sx.transform(Xte); ytr_s = sy.transform(ytr.reshape(-1, 1)).ravel()
    out = {"seed": seed, "n_train": len(ytr), "n_test": len(yte)}
    inv = lambda p: sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    t = time.time(); m = LinearRegression().fit(Xtr_s, ytr_s); out["MLR"] = dict(metrics(yte, inv(m.predict(Xte_s))), params=9, seconds=time.time() - t, hybrid=False)
    for name, fn, hy in (("ANN", ann_backprop, False), ("ANN-GA", ann_ga, True), ("ANFIS", anfis, True)):
        t = time.time(); p, k = fn(Xtr_s, ytr_s, Xte_s, seed)
        out[name] = dict(metrics(yte, inv(p)), params=int(k), seconds=time.time() - t, hybrid=hy)
        print(seed, name, {a: round(b, 3) for a, b in out[name].items() if a in ("r2", "rmse", "mae", "seconds")}, flush=True)
    return out

if __name__ == "__main__":
    import sys, json
    seed = int(sys.argv[1]); r = run(seed); json.dump(r, open(f"res_{seed}.json", "w"))
