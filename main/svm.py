import numpy as np
import pandas as pd

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

train = pd.read_csv("test/train.csv")
test = pd.read_csv("test/test.csv")

vocab_index = build_vocab(train["sequence"], n=6)

X_train = vectorize_dataset(train, vocab_index, n=6)
y_train = train["label"].values

X_test = vectorize_dataset(test, vocab_index, n=6)
y_test = test["label"].values

model = MultiClassSVM(lr=0.001, C=5.0, epochs=2000)
model.fit(X_train, y_train)

y = y_test
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = np.mean(y_pred == y)
print(accuracy)
