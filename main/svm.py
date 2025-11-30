import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss

def generate_ngrams(sequence, n=6):
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

def vectorize(ngrams_list, vocab_index):
    vec = np.zeros(len(vocab_index))
    for ng in ngrams_list:
        if ng in vocab_index:
            vec[vocab_index[ng]] += 1
    return vec

def build_vocab(sequences, n=6):
    vocab = set()
    for seq in sequences:
        for ng in generate_ngrams(seq, n):
            vocab.add(ng)
    vocab = list(vocab)
    vocab_index = {ng: i for i, ng in enumerate(vocab)}
    return vocab_index

def vectorize_dataset(df, vocab_index, n=6):
    X = []
    for seq in df["sequence"]:
        ng = generate_ngrams(seq, n)
        X.append(vectorize(ng, vocab_index))
    return np.array(X)

class SVM:
    def __init__(self, lr=0.001, C=5.0, epochs=2000):
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 1, 1, -1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.epochs):
            scores = X.dot(self.w) + self.b
            margin = y * scores
            mis = margin < 1
            dw = self.w - self.C * np.dot(X[mis].T, y[mis])
            db = -self.C * np.sum(y[mis])
            self.w -= self.lr * dw
            self.b -= self.lr * db
            self.lr *= 0.9995

    def decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

class MultiClassSVM:
    def __init__(self, lr=0.001, C=5.0, epochs=2000):
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}
        for c in self.classes:
            print(f"Class: {c}")
            y_binary = np.where(y == c, 1, -1)
            svm = SVM(lr=self.lr, C=self.C, epochs=self.epochs)
            svm.fit(X, y_binary)
            self.models[c] = svm

    def predict(self, X):
        scores = np.vstack([self.models[c].decision_function(X) for c in self.classes]).T
        return self.classes[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = np.vstack([self.models[c].decision_function(X) for c in self.classes]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Load data using absolute paths
base_path = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(base_path, "test", "train.csv"))
val = pd.read_csv(os.path.join(base_path, "test", "val.csv"))
test = pd.read_csv(os.path.join(base_path, "test", "test.csv"))

vocab_index = build_vocab(train["sequence"], n=6)

X_train = vectorize_dataset(train, vocab_index, n=6)
y_train = train["label"].values

X_val = vectorize_dataset(val, vocab_index, n=6)
y_val = val["label"].values

X_test = vectorize_dataset(test, vocab_index, n=6)
y_test = test["label"].values

# --- Graph Generation ---
print("Generating learning curves...")
train_sizes = np.linspace(0.1, 1.0, 5)  # 5 points
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for frac in train_sizes:
    # Subset
    limit = int(len(X_train) * frac)
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    
    # Fit
    svm_model = MultiClassSVM(lr=0.001, C=5.0, epochs=2000)
    svm_model.fit(X_sub, y_sub)
    
    # Predict
    y_train_pred_sub = svm_model.predict(X_sub)
    y_train_prob = svm_model.predict_proba(X_sub)
    
    y_val_pred_sub = svm_model.predict(X_val)
    y_val_prob = svm_model.predict_proba(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_sub, y_train_pred_sub)
    val_acc = accuracy_score(y_val, y_val_pred_sub)
    
    train_loss = log_loss(y_sub, y_train_prob, labels=svm_model.classes)
    val_loss = log_loss(y_val, y_val_prob, labels=svm_model.classes)
    
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Frac {frac:.1f}: Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}")

# Plotting
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_losses, 'o-', label='Train Loss')
plt.plot(train_sizes, val_losses, 'o-', label='Val Loss')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Log Loss')
plt.title('Learning Curve - Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_sizes, train_accs, 'o-', label='Train Accuracy')
plt.plot(train_sizes, val_accs, 'o-', label='Val Accuracy')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Accuracy')
plt.legend()

plt.savefig('svm_training_results.png')
print("Graphs saved to svm_training_results.png")

# Final model training
print("\nTraining final model on full training set...")
model = MultiClassSVM(lr=0.001, C=5.0, epochs=2000)
model.fit(X_train, y_train)

y = y_test
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = np.mean(y_pred == y)
print(f"\nTest Accuracy: {accuracy}")
