import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
import test_metrics as tm
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
    vocab = sorted(list(vocab))
    vocab_index = {ng: i for i, ng in enumerate(vocab)}
    return vocab_index

def vectorize_dataset(df, vocab_index, n=6):
    X = []
    for seq in df["sequence"]:
        ng = generate_ngrams(seq, n)
        X.append(vectorize(ng, vocab_index))
    return np.array(X)

class SVM:
    def __init__(self, lr=0.01, C=5.0, epochs=500, batch_size=256, early_stopping_tol=1e-4):
        self.lr = lr
        self.initial_lr = lr
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_tol = early_stopping_tol

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 1, 1, -1)
        self.w = np.zeros(n_features)
        self.b = 0
        
        prev_w = np.copy(self.w)
        
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                scores = X_batch.dot(self.w) + self.b
                margin = y_batch * scores
                mis = margin < 1
                
                dw = self.w - self.C * np.dot(X_batch[mis].T, y_batch[mis])
                db = -self.C * np.sum(y_batch[mis])
                
                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            # Decay learning rate
            self.lr *= 0.995
            
            # Early stopping check every 10 epochs
            if epoch % 10 == 0 and epoch > 0:
                w_change = np.linalg.norm(self.w - prev_w)
                if w_change < self.early_stopping_tol:
                    print(f"  Early stopping at epoch {epoch}")
                    break
                prev_w = np.copy(self.w)

    def decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

class MultiClassSVM:
    def __init__(self, lr=0.01, C=5.0, epochs=500, batch_size=256):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}
        for c in self.classes:
            print(f"Class: {c}", end=" ")
            y_binary = np.where(y == c, 1, -1)
            svm = SVM(lr=self.lr, C=self.C, epochs=self.epochs, batch_size=self.batch_size)
            svm.fit(X, y_binary)
            self.models[c] = svm
            print("✓")

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
print("Generating learning curves (reduced for speed)...")
train_sizes = np.linspace(0.2, 1.0, 3)  # Reduced to 3 points for speed
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for frac in train_sizes:
    # Subset
    limit = int(len(X_train) * frac)
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    
    # Fit (reduced epochs for learning curve)
    svm_model = MultiClassSVM(lr=0.01, C=5.0, epochs=200, batch_size=256)
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
model_path = 'svm_model.pkl'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded. Skipping training.")
else:
    print("No existing model found. Training new model...")
    model = MultiClassSVM(lr=0.01, C=5.0, epochs=500, batch_size=256)
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

y = y_test
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = np.mean(y_pred == y)
print(f"\nTest Accuracy: {accuracy}")

# Convert probabilities to list of dicts for test_metrics
y_prob_list = []
classes = model.classes
for prob_array in y_prob:
    prob_dict = {
        classes[i]: float(prob_array[i]) 
        for i in range(len(prob_array))
    }
    y_prob_list.append(prob_dict)

# Compute comprehensive metrics
tm.compute_metrics(y, y_pred, y_prob_list)
