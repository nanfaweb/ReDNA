import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
import test_metrics as tm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
import pickle
import os

np.set_printoptions(legacy='1.25')

# N-gram extraction
def generate_ngrams(sequence, n=6):
	return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

# Vectorization (bag-of-n-grams)
def vectorize(ngrams_list, vocab_index):
	vec = np.zeros(len(vocab_index))
	for ng in ngrams_list:
		if ng in vocab_index:
			vec[vocab_index[ng]] += 1
	return vec

# Naive Bayes
class NaiveBayes:
	def __init__(self, alpha=1.0):
		self.alpha = alpha
	
	def fit(self, X, y):
		self.classes = np.unique(y)
		self.class_count = dict.fromkeys(self.classes, 0)
		self.feature_count = {c:np.zeros(X.shape[1]) for c in self.classes}
		
		for xi, yi in zip(X, y):
			self.class_count[yi] += 1
			self.feature_count[yi] += xi
		
		self.class_log_prior_ = {}
		self.feature_log_prob_ = {}
		total_samples = len(y)
		for c in self.classes:
			self.class_log_prior_[c] = np.log(self.class_count[c]/total_samples)
			self.feature_log_prob_[c] = np.log(
				(self.feature_count[c]+self.alpha) /
				(np.sum(self.feature_count[c])+self.alpha*X.shape[1])
			)
	
	def predict(self, X):
		y_pred = []
		for xi in X:
			scores = {c: self.class_log_prior_[c]+np.sum(xi*self.feature_log_prob_[c]) for c in self.classes}
			y_pred.append(max(scores, key=scores.get))
		return np.array(y_pred)
	
	def predict_prob(self, X):
		prob_list = []
		for xi in X:
			log_scores = {c: self.class_log_prior_[c]+np.sum(xi*self.feature_log_prob_[c]) for c in self.classes}
			# softmax
			max_log = max(log_scores.values())
			exp_scores = {c: np.exp(log_scores[c]-max_log) for c in self.classes}
			total = sum(exp_scores.values())
			probs = {c: exp_scores[c]/total for c in self.classes}
			prob_list.append(probs)
		return prob_list
	
# Load data
base_path = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(base_path, 'test', 'train.csv'))
val   = pd.read_csv(os.path.join(base_path, 'test', 'val.csv'))
test  = pd.read_csv(os.path.join(base_path, 'test', 'test.csv'))

n = 6
train['ngrams'] = train['sequence'].apply(lambda x: generate_ngrams(x, n))
val['ngrams']   = val['sequence'].apply(lambda x: generate_ngrams(x, n))
test['ngrams']  = test['sequence'].apply(lambda x: generate_ngrams(x, n))

# Build vocabulary
vocab = set()
for seq in train['ngrams']:
	vocab.update(seq)
vocab = sorted(list(vocab))
vocab_index = {k:i for i,k in enumerate(vocab)}

X_train = np.array([vectorize(seq, vocab_index) for seq in train['ngrams']])
X_val   = np.array([vectorize(seq, vocab_index) for seq in val['ngrams']])
X_test  = np.array([vectorize(seq, vocab_index) for seq in test['ngrams']])

y_train = train['label'].values
y_val   = val['label'].values
y_test  = test['label'].values

# --- Graph Generation ---
print("Generating learning curves...")
train_sizes = np.linspace(0.1, 1.0, 5) # 5 points
train_losses = []
val_losses = []
train_accs = []
val_accs = []

def prob_list_to_array(plist, classes):
    arr = np.zeros((len(plist), len(classes)))
    for i, p in enumerate(plist):
        for j, c in enumerate(classes):
            arr[i, j] = p.get(c, 0.0)
    return arr

for frac in train_sizes:
    # Subset
    limit = int(len(X_train) * frac)
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    
    # Fit
    nb = NaiveBayes(alpha=1.0)
    nb.fit(X_sub, y_sub)
    
    # Predict
    y_train_pred_sub = nb.predict(X_sub)
    y_train_prob_list = nb.predict_prob(X_sub)
    
    y_val_pred_sub = nb.predict(X_val)
    y_val_prob_list = nb.predict_prob(X_val)
    
    y_train_prob_arr = prob_list_to_array(y_train_prob_list, nb.classes)
    y_val_prob_arr = prob_list_to_array(y_val_prob_list, nb.classes)
    
    # Metrics
    train_acc = accuracy_score(y_sub, y_train_pred_sub)
    val_acc = accuracy_score(y_val, y_val_pred_sub)
    
    train_loss = log_loss(y_sub, y_train_prob_arr, labels=nb.classes)
    val_loss = log_loss(y_val, y_val_prob_arr, labels=nb.classes)
    
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

plt.savefig('naive_bayes_training_results.png')
print("Graphs saved to naive_bayes_training_results.png")

# Fit model (using your vectorized X_train/X_test)
model_path = 'naive_bayes_model.pkl'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded. Skipping training.")
else:
    print("No existing model found. Training new model...")
    model = NaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

# Predictions
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_prob(X_test)

# Compute metrics
tm.compute_metrics(y_test, y_test_pred, y_test_proba);

