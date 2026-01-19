import os
import re
import glob
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical


def getUA(out, tar):
    kclass = out.shape[1]
    vn = np.sum(tar, axis=0)
    aux = tar - out
    wn = np.sum((aux + np.abs(aux)) // 2, axis=0)
    cn = vn - wn
    return np.round(np.sum(cn / np.maximum(vn, 1)) / kclass * 100, 1)


def getWA(out, tar):
    db_size = out.shape[0]
    out = np.argmax(out, axis=1)
    tar = np.argmax(tar, axis=1)
    hits = np.sum(out == tar)
    return np.round(hits / db_size * 100, 1)


def find_npy_files(root_dir):
    pattern = os.path.join(root_dir, "**", "Subiect_*_*_r.npy")
    return sorted(glob.glob(pattern, recursive=True))


_subject_re = re.compile(r"Subiect_(\d+)_(\d+)_r\.npy$")


def parse_subject_and_label(fname, kclass):
    base = os.path.basename(fname)
    m = _subject_re.match(base)
    if not m:
        raise ValueError(f"Unexpected file: {base}")
    sid = int(m.group(1))
    y = int(m.group(2))
    if not (0 <= y < kclass):
        return sid, None
    return sid, y


def load_trial(path):
    x = np.load(path)
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] <= 16 and x.shape[1] > x.shape[0]:
        x = x.T
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(f"ndim={x.ndim} unsupported: {path} shape={x.shape}")
    return x


def preprocess_trial(x):
    return x - np.mean(x, axis=0, keepdims=True)


def window_trial(x, win_len=512, step=256):
    n, c = x.shape
    if n < win_len:
        return np.empty((0, win_len, c), dtype=x.dtype)
    starts = np.arange(0, n - win_len + 1, step, dtype=int)
    idx = starts[:, None] + np.arange(win_len)[None, :]
    return x[idx, :]


def _features_from_windows(w):
    w = np.asarray(w)

    mav = np.mean(np.abs(w), axis=1)
    wl = np.sum(np.abs(np.diff(w, axis=1)), axis=1)
    rms = np.sqrt(np.mean(w ** 2, axis=1))

    std = np.std(w, axis=1)
    alpha = 0.01 * std

    x0 = w[:, :-1, :]
    x1 = w[:, 1:, :]
    zcr = np.sum(((np.abs(x1 - x0) >= alpha[:, None, :]) & ((x1 * x0) < 0)), axis=1)

    xkm1 = w[:, :-2, :]
    xk = w[:, 1:-1, :]
    xkp1 = w[:, 2:, :]
    ssc = np.sum((((xk - xkm1) * (xk - xkp1)) >= alpha[:, None, :]), axis=1)

    mu = np.mean(w, axis=1)
    sig = np.std(w, axis=1) + 1e-12
    skew = np.mean((((w - mu[:, None, :]) / sig[:, None, :]) ** 3), axis=1)

    feats = np.concatenate([mav, wl, zcr, ssc, rms, skew], axis=1)
    return feats.astype(np.float32, copy=False)


def build_window_dataset(root_dir, win_len=512, step=256, kclass=3):
    files = find_npy_files(root_dir)
    print("Trials loaded:", len(files))
    if not files:
        raise FileNotFoundError("Files not found")

    labels = []
    subjects = []
    for p in files:
        sid, y = parse_subject_and_label(p, kclass)
        if y is None:
            continue
        labels.append(y)
        subjects.append(sid)

    labels = np.asarray(labels, dtype=int)
    subjects = np.asarray(subjects, dtype=int)
    
    if subjects.size:
        print("Subject id min/max:", int(np.min(subjects)), int(np.max(subjects)))

    x0 = load_trial(files[0])
    w0 = window_trial(preprocess_trial(x0), win_len=win_len, step=step)


    x_list, y_list, g_list, t_list = [], [], [], []

    trial_id = 0
    for p in files:
        sid, y = parse_subject_and_label(p, kclass)
        if y is None:
            trial_id += 1
            continue

        x = preprocess_trial(load_trial(p))
        w = window_trial(x, win_len=win_len, step=step)
        if w.shape[0] == 0:
            trial_id += 1
            continue

        feats = _features_from_windows(w)
        x_list.append(feats)
        y_list.append(np.full((feats.shape[0],), y, dtype=int))
        g_list.append(np.full((feats.shape[0],), sid, dtype=int))
        t_list.append(np.full((feats.shape[0],), trial_id, dtype=int))

        trial_id += 1

    if not x_list:
        raise RuntimeError("Features can't be generated.")

    X = np.vstack(x_list).astype(np.float32, copy=False)
    Y = np.concatenate(y_list).astype(int, copy=False)
    G = np.concatenate(g_list).astype(int, copy=False)
    T = np.concatenate(t_list).astype(int, copy=False)

    print("Window-level dataset:", X.shape, Y.shape, "unique trials:", int(np.unique(T).size))
    return X, Y, G, T


def aggregate_to_trials(proba_windows, y_windows, trial_ids):
    uniq = np.unique(trial_ids)
    proba_trial = np.empty((uniq.size, proba_windows.shape[1]), dtype=np.float32)
    y_trial = np.empty((uniq.size,), dtype=int)

    for i, t in enumerate(uniq):
        idx = np.where(trial_ids == t)[0]
        pw = proba_windows[idx]
        proba_trial[i] = np.mean(pw, axis=0)
        y_trial[i] = int(np.bincount(y_windows[idx]).argmax())

    return proba_trial, y_trial


def build_fcnn(fin, kclass, struct=(64, 32, 16)):
    K.clear_session()
    inp = Input(shape=(fin,), name="IL")
    h = Dense(struct[0], name="HL1")(inp)
    h = Activation("relu", name="HL1_A")(h)
    h = Dense(struct[1], name="HL2")(h)
    h = Activation("relu", name="HL2_A")(h)
    h = Dense(struct[2], name="HL3")(h)
    h = Activation("relu", name="HL3_A")(h)
    h = Dense(kclass, name="OL")(h)
    out = Activation("softmax", name="OL_A")(h)
    return Model(inputs=inp, outputs=out, name="FCNN")


if __name__ == "__main__":
    plt.close("all")

    DATA_DIR = r"D:\AN 1 MASTER\SEM 1\TB\sEmg_databases\sEmg_databases"
    
    files_demo = find_npy_files(DATA_DIR)
    x_demo = load_trial(files_demo[0])
    x_demo_p = preprocess_trial(x_demo)
    
    plt.figure(figsize=(10, 6))
    for ch in range(x_demo.shape[1]):
        plt.subplot(x_demo.shape[1], 1, ch + 1)
        plt.plot(x_demo[:, ch], linewidth=0.8)
        plt.plot(x_demo_p[:, ch], linewidth=0.8)
        plt.ylabel(f"ch{ch}")
        if ch == 0:
            plt.title("sEMG example – raw vs preprocessed (DC removed)")
        if ch < x_demo.shape[1] - 1:
            plt.xticks([])
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.show()
    
    Kclass = 3
    WIN_LEN = 512
    STEP = 256

    X, Y, G, T = build_window_dataset(DATA_DIR, win_len=WIN_LEN, step=STEP, kclass=Kclass)
    Fin = X.shape[1]
    print("Feature dim:", Fin)

    FCNN_struct = (64, 32, 16)
    OPT = "adam"

    metrix = []
    cm_train = np.zeros((Kclass, Kclass), dtype=np.int64)
    cm_val = np.zeros((Kclass, Kclass), dtype=np.int64)

    gkf = GroupKFold(n_splits=5)

    for fold_idx, (idx_train, idx_val) in enumerate(gkf.split(X, Y, groups=G), start=1):
        x_train, y_train, t_train = X[idx_train], Y[idx_train], T[idx_train]
        x_val, y_val, t_val = X[idx_val], Y[idx_val], T[idx_val]

        scaler = StandardScaler().fit(x_train)
        x_train_n = scaler.transform(x_train)
        x_val_n = scaler.transform(x_val)

        model = build_fcnn(Fin, Kclass, struct=FCNN_struct)
        print("=" * 50)
        print(f"Fold {fold_idx}")
        print("=" * 50)

        model.compile(optimizer=OPT, loss="categorical_crossentropy", metrics=["accuracy"])

        early = EarlyStopping(monitor="val_loss", patience=20, verbose=1)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
                tmp_path = f.name

            ckpt = ModelCheckpoint(tmp_path, monitor="val_loss", save_best_only=True, verbose=1)

            hist = model.fit(
                x_train_n,
                to_categorical(y_train, Kclass),
                batch_size=128,
                epochs=500,
                validation_data=(x_val_n, to_categorical(y_val, Kclass)),
                callbacks=[early, ckpt],
                verbose=1,
            )

            del model
            model = load_model(tmp_path)

            out_train_w = model.predict(x_train_n, batch_size=256, verbose=1)
            out_val_w = model.predict(x_val_n, batch_size=256, verbose=1)

            proba_train_t, y_train_t = aggregate_to_trials(out_train_w, y_train, t_train)
            proba_val_t, y_val_t = aggregate_to_trials(out_val_w, y_val, t_val)

            out_train_t = np.argmax(proba_train_t, axis=1)
            out_val_t = np.argmax(proba_val_t, axis=1)

            ua_train = getUA(to_categorical(out_train_t, Kclass), to_categorical(y_train_t, Kclass))
            wa_train = getWA(to_categorical(out_train_t, Kclass), to_categorical(y_train_t, Kclass))
            ua_val = getUA(to_categorical(out_val_t, Kclass), to_categorical(y_val_t, Kclass))
            wa_val = getWA(to_categorical(out_val_t, Kclass), to_categorical(y_val_t, Kclass))

            metrix += [ua_train, wa_train, ua_val, wa_val]

            cm_train += confusion_matrix(y_train_t, out_train_t, labels=list(range(Kclass)))
            cm_val += confusion_matrix(y_val_t, out_val_t, labels=list(range(Kclass)))

            # plt.figure()
            # plt.plot(hist.history["loss"])
            # plt.plot(hist.history["val_loss"])
            # plt.xlabel("Epoch")
            # plt.ylabel("Loss")
            # plt.legend(["Train. loss", "Val. loss"], loc="upper right")

        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    ua_train_avg = np.round(np.mean(metrix[0::4]), 1)
    wa_train_avg = np.round(np.mean(metrix[1::4]), 1)
    ua_val_avg = np.round(np.mean(metrix[2::4]), 1)
    wa_val_avg = np.round(np.mean(metrix[3::4]), 1)

    print("=" * 50)
    print("Cross-validation results")
    print("=" * 50)
    print("")
    print(
        "Metrics:\n"
        f"-> UA (train) = {ua_train_avg:.1f}%\n"
        f"-> WA (train) = {wa_train_avg:.1f}%\n"
        f"-> UA (val) = {ua_val_avg:.1f}%\n"
        f"-> WA (val) = {wa_val_avg:.1f}%"
    )

    ConfusionMatrixDisplay(cm_train).plot(values_format=".0f")
    ConfusionMatrixDisplay(cm_val).plot(values_format=".0f")
    plt.show()
