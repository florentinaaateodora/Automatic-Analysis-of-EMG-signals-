import os, re, glob
import numpy as np
import pandas as pd


DATA_DIR = r"D:\AN 1 MASTER\SEM 1\TB\sEmg_databases\sEmg_databases"
WIN_LEN = 512
STEP = 256
KCLASS = 3

GROUPA = [0, 1, 2, 3]
GROUPB = [4, 5, 6, 7]


def find_files(root):
    return sorted(glob.glob(os.path.join(root, "**", "Subiect_*_*_r.npy"), recursive=True))

def parse_name(path):
    base = os.path.basename(path)
    m = re.match(r"Subiect_(\d+)_(\d+)_r\.npy$", base)
    if not m:
        raise ValueError(f"Nume fisier neasteptat: {base}")
    return int(m.group(1)), int(m.group(2))  # subject, class

def load_trial(path):
    x = np.load(path)
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] <= 16 and x.shape[1] > x.shape[0]:
        x = x.T
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    return x  # (N,8)

def preprocess(x):
    return x - np.mean(x, axis=0, keepdims=True)  # DC removal

def windowing(x, win_len, step):
    N, C = x.shape
    if N < win_len:
        return np.empty((0, win_len, C))
    starts = np.arange(0, N - win_len + 1, step)
    idx = starts[:, None] + np.arange(win_len)[None, :]
    return x[idx, :]  # (Nw, win_len, C)

def KAs(XA, XB, eps=1e-12):
    X1 = max(abs(float(XA)), abs(float(XB)))
    return abs(float(XA) - float(XB)) / (X1 + eps) * 100.0


# ====== MAIN ======
files = find_files(DATA_DIR)
if len(files) == 0:
    raise FileNotFoundError("Nu am gasit fisiere Subiect_*_*_r.npy")

rows = []

for p in files:
    sid, cls = parse_name(p)
    if not (0 <= cls < KCLASS):
        continue

    x = preprocess(load_trial(p))
    W = windowing(x, WIN_LEN, STEP)
    if W.shape[0] == 0:
        continue

    # --- RMS pe ferestre, pe canal ---
    rms_w = np.sqrt(np.mean(W * W, axis=1))     # (Nw, 8)
    rms_ch = np.mean(rms_w, axis=0)             # (8,)

    # --- Skewness pe ferestre, pe canal ---
    mu = np.mean(W, axis=1, keepdims=True)      # (Nw,1,8)
    sig = np.std(W, axis=1, keepdims=True) + 1e-12
    z = (W - mu) / sig
    sk_w = np.mean(z**3, axis=1)                # (Nw,8)
    sk_ch = np.mean(sk_w, axis=0)               # (8,)

    # --- XA/XB = media pe grup ---
    XA_rms = float(np.mean(rms_ch[GROUPA]))
    XB_rms = float(np.mean(rms_ch[GROUPB]))
    kas_rms = KAs(XA_rms, XB_rms)

    XA_sk = float(np.mean(sk_ch[GROUPA]))
    XB_sk = float(np.mean(sk_ch[GROUPB]))
    kas_sk = KAs(XA_sk, XB_sk)

    rows.append({
        "subject": sid,
        "class": cls,
        "file": os.path.basename(p),
        "XA_RMS": XA_rms,
        "XB_RMS": XB_rms,
        "KAs_RMS_percent": kas_rms,
        "XA_Skew": XA_sk,
        "XB_Skew": XB_sk,
        "KAs_Skew_percent": kas_sk
    })

df_trials = pd.DataFrame(rows)

# --- BY CLASS ---
df_rms = (df_trials.groupby("class")["KAs_RMS_percent"]
          .agg(["count", "mean", "std", "min", "max"])
          .reset_index()
          .rename(columns={"mean": "KAs_mean", "std": "KAs_std"}))

df_sk = (df_trials.groupby("class")["KAs_Skew_percent"]
         .agg(["count", "mean", "std", "min", "max"])
         .reset_index()
         .rename(columns={"mean": "KAs_mean", "std": "KAs_std"}))


df_trials.to_csv("Asymmetry_TRIALLEVEL_RMS_SKEW.csv", index=False)
df_rms.to_csv("Asymmetry_BYCLASS_RMS.csv", index=False)
df_sk.to_csv("Asymmetry_BYCLASS_SKEW.csv", index=False)


print("\nBY CLASS (RMS):")
print(df_rms)
print("\nBY CLASS (SKEW):")
print(df_sk)
print("\nGROUPA:", GROUPA, "GROUPB:", GROUPB)
