import numpy as np
import pandas as pd
import ast
import os
import math
import matplotlib.pyplot as plt
from test import test_metrics as tm

np.random.seed(1)

def parse_tokens(s):
    if isinstance(s, list): return s
    if isinstance(s, str):
        try: return ast.literal_eval(s)
        except: return [int(x) for x in s.split(',')]
    return []

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def cross_entropy(pred, y):
    return -np.log(pred[np.arange(len(y)), y] + 1e-12).mean()

def accuracy(pred, y):
    return np.mean(pred == y)

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

# -----------------------------------------
# LSTM Cell
# -----------------------------------------
class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        limit = 1 / math.sqrt(hidden_dim)
        self.W = np.random.uniform(-limit, limit, (input_dim + hidden_dim, 4 * hidden_dim))
        self.b = np.zeros((1, 4 * hidden_dim))

    def forward(self, x_t, h_prev, c_prev):
        concat = np.concatenate([x_t, h_prev], axis=1)
        gates = concat @ self.W + self.b

        hdim = self.hidden_dim
        i = sigmoid(gates[:, :hdim])
        f = sigmoid(gates[:, hdim:2*hdim])
        o = sigmoid(gates[:, 2*hdim:3*hdim])
        g = np.tanh(gates[:, 3*hdim:])

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        cache = (x_t, h_prev, c_prev, i, f, o, g, c, concat)
        return h, c, cache

    def backward(self, dh, dc, cache):
        x_t, h_prev, c_prev, i, f, o, g, c, concat = cache
        hdim = self.hidden_dim

        do = dh * np.tanh(c)
        dc += dh * o * (1 - np.tanh(c)**2)
        df = dc * c_prev
        di = dc * g
        dg = dc * i

        di_input = di * i * (1 - i)
        df_input = df * f * (1 - f)
        do_input = do * o * (1 - o)
        dg_input = dg * (1 - g**2)

        dG = np.hstack([di_input, df_input, do_input, dg_input])

        dW = concat.T @ dG
        db = dG.sum(axis=0, keepdims=True)

        dconcat = dG @ self.W.T
        dx = dconcat[:, :x_t.shape[1]]
        dh_prev = dconcat[:, x_t.shape[1]:]
        dc_prev = dc * f

        return dx, dh_prev, dc_prev, dW, db

# -----------------------------------------
# BiLSTM
# -----------------------------------------
class BiLSTM:
    def __init__(self, vocab_size=4, embed_dim=16, hidden_dim=32, num_classes=4):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        limit = 1 / math.sqrt(embed_dim)
        self.Emb = np.random.uniform(-limit, limit, (vocab_size, embed_dim))

        self.fwd = LSTMCell(embed_dim, hidden_dim)
        self.bwd = LSTMCell(embed_dim, hidden_dim)

        self.W_out = np.random.uniform(-0.1, 0.1, (2 * hidden_dim, num_classes))
        self.b_out = np.zeros((1, num_classes))

    # ---------- SAVE / LOAD ----------
    def save(self, path=None):
        # Save weights to the same directory as bi_lstm.py
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "bilstm_weights.npz")
        np.savez(path,
                 Emb=self.Emb,
                 Wf=self.fwd.W, bf=self.fwd.b,
                 Wb=self.bwd.W, bb=self.bwd.b,
                 Wout=self.W_out, bout=self.b_out)

    def load(self, path=None):
        # Load weights from the same directory as bi_lstm.py
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "bilstm_weights.npz")
        if not os.path.exists(path):
            print("No saved weights found. Starting fresh.")
            return
        data = np.load(path, allow_pickle=True)
        self.Emb = data["Emb"]
        self.fwd.W = data["Wf"]; self.fwd.b = data["bf"]
        self.bwd.W = data["Wb"]; self.bwd.b = data["bb"]
        self.W_out = data["Wout"]; self.b_out = data["bout"]
        print("Loaded pre-trained weights.")

    # ---------- FORWARD ----------
    def forward(self, X):
        seq_len, batch = X.shape
        X_emb = self.Emb[X]

        h_f = np.zeros((seq_len, batch, self.hidden_dim))
        h_prev = np.zeros((batch, self.hidden_dim))
        c_prev = np.zeros((batch, self.hidden_dim))
        fwd_cache = []

        for t in range(seq_len):
            h_prev, c_prev, cache = self.fwd.forward(X_emb[t], h_prev, c_prev)
            h_f[t] = h_prev
            fwd_cache.append(cache)

        h_b = np.zeros((seq_len, batch, self.hidden_dim))
        h_prev = np.zeros((batch, self.hidden_dim))
        c_prev = np.zeros((batch, self.hidden_dim))
        bwd_cache = []

        for t in reversed(range(seq_len)):
            h_prev, c_prev, cache = self.bwd.forward(X_emb[t], h_prev, c_prev)
            h_b[t] = h_prev
            bwd_cache.append(cache)

        bwd_cache.reverse()

        H_final = np.concatenate([h_f[-1], h_b[0]], axis=1)
        logits = H_final @ self.W_out + self.b_out
        probs = softmax(logits)

        return probs, (fwd_cache, bwd_cache, X_emb, h_f, h_b)

    # ---------- BACKWARD ----------
    def backward(self, probs, y, cache, X):
        seq_len, batch = X.shape
        fwd_cache, bwd_cache, X_emb, h_f, h_b = cache

        dlogits = probs.copy()
        dlogits[np.arange(batch), y] -= 1
        dlogits /= batch

        dW_out = np.concatenate([h_f[-1], h_b[0]], axis=1).T @ dlogits
        db_out = dlogits.sum(axis=0, keepdims=True)

        dh_f_last = dlogits @ self.W_out[:self.hidden_dim].T
        dh_b_first = dlogits @ self.W_out[self.hidden_dim:].T

        dEmb = np.zeros_like(self.Emb)
        dW_f = np.zeros_like(self.fwd.W); db_f = np.zeros_like(self.fwd.b)
        dW_b = np.zeros_like(self.bwd.W); db_b = np.zeros_like(self.bwd.b)

        dh_f = dh_f_last
        dc_f = np.zeros((batch, self.hidden_dim))

        dh_b = dh_b_first
        dc_b = np.zeros((batch, self.hidden_dim))

        for t in reversed(range(seq_len)):
            dx, dh_f, dc_f, dW, db = self.fwd.backward(dh_f, dc_f, fwd_cache[t])
            dW_f += dW; db_f += db
            for i in range(batch):
                dEmb[X[t, i]] += dx[i]

        for t in range(seq_len):
            dx, dh_b, dc_b, dW, db = self.bwd.backward(dh_b, dc_b, bwd_cache[t])
            dW_b += dW; db_b += db
            for i in range(batch):
                dEmb[X[t, i]] += dx[i]

        return dEmb, dW_f, db_f, dW_b, db_b, dW_out, db_out

    # ---------- UPDATE ----------
    def step(self, grads, lr=0.001):
        dEmb, dWf, dbf, dWb, dbb, dWo, dbo = grads

        self.Emb -= lr * dEmb
        self.fwd.W -= lr * dWf; self.fwd.b -= lr * dbf
        self.bwd.W -= lr * dWb; self.bwd.b -= lr * dbb
        self.W_out -= lr * dWo; self.b_out -= lr * dbo

# -----------------------------------------
# Load Dataset
# -----------------------------------------
def load_data(file):
    df = pd.read_csv(file)

    # Normalize labels (strip spaces, lowercase)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    label_map = {
        "promoter": 0,
        "cds": 1,
        "intergenic": 2,
        "terminator": 3
    }

    # Identify invalid labels
    bad = df[~df["label"].isin(label_map.keys())]

    if len(bad) > 0:
        print("WARNING: Found invalid labels:")
        print(bad["label"].value_counts())

        # Option 1: remove bad rows
        df = df[df["label"].isin(label_map.keys())].reset_index(drop=True)

        # Option 2 (if you prefer): replace invalid labels with "intergenic"
        # df.loc[~df["label"].isin(label_map.keys()), "label"] = "intergenic"

    # Map to integers (SAFE)
    df["label_int"] = df["label"].map(label_map).astype(np.int32)
    y = df["label_int"].values

    # Parse tokens
    X_raw = [parse_tokens(s) for s in df["tokens"]]

    max_len = max(len(s) for s in X_raw)
    X = np.zeros((max_len, len(X_raw)), dtype=np.int32)

    for i, seq in enumerate(X_raw):
        X[:len(seq), i] = seq

    return X, y


# -----------------------------------------
# Training
# -----------------------------------------
def train():
    Xtr, Ytr = load_data("test/train.csv")
    Xval, Yval = load_data("test/val.csv")

    model = BiLSTM()
    model.load()

    lr = 0.001
    batch = 16
    epochs = 10

    N = Xtr.shape[1]

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for ep in range(epochs):
        perm = np.random.permutation(N)

        epoch_losses = []
        epoch_accs = []

        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            Xb = Xtr[:, idx]
            Yb = Ytr[idx]

            probs, cache = model.forward(Xb)
            loss = cross_entropy(probs, Yb)
            epoch_losses.append(loss)

            preds = np.argmax(probs, axis=1)
            epoch_accs.append(accuracy(preds, Yb))

            grads = model.backward(probs, Yb, cache, Xb)
            model.step(grads, lr)

        # Record training metrics for this epoch
        train_loss = np.mean(epoch_losses)
        train_acc = np.mean(epoch_accs)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_epoch_losses = []
        val_epoch_accs = []
        
        val_N = Xval.shape[1]
        for i in range(0, val_N, batch):
             Xvb = Xval[:, i:i+batch]
             Yvb = Yval[i:i+batch]
             v_probs, _ = model.forward(Xvb)
             v_loss = cross_entropy(v_probs, Yvb)
             val_epoch_losses.append(v_loss)
             
             v_preds = np.argmax(v_probs, axis=1)
             val_epoch_accs.append(accuracy(v_preds, Yvb))
        
        val_loss = np.mean(val_epoch_losses)
        val_acc = np.mean(val_epoch_accs)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {ep+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    model.save()

    # -----------------------------------------
    # Plotting
    # -----------------------------------------
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', marker='o')
    plt.plot(val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.savefig("bi_lstm_graph.png")
    print("Plots saved to bi_lstm_graph.png")

if __name__ == "__main__":
    train()
    X_test, Y_test = load_data("test/test.csv")  # adjust path if needed

    model = BiLSTM()
    model.load()  # load trained weights

    # -----------------------------
    # Forward Pass to Get Predictions
    # -----------------------------
    probs, _ = model.forward(X_test)
    Y_pred = np.argmax(probs, axis=1)

    # Convert probabilities to a list of dicts for top-k and AUC
    prob_list = []
    classes = np.unique(Y_test)
    for p in probs:
        prob_list.append({c: float(p[c]) for c in classes})

    # -----------------------------
    # Compute and Print Metrics
    # -----------------------------
    print("\n================ Test Metrics ================\n")
    tm.compute_metrics(Y_test, Y_pred, prob_list)