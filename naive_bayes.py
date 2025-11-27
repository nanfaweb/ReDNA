import pandas as pd
import numpy as np

np.set_printoptions(legacy='1.25')

# Load data
train = pd.read_csv("train.csv")
val   = pd.read_csv("val.csv")
test  = pd.read_csv("test.csv")

# Accuracy
def accuracy(y_true, y_pred):
	return np.mean(y_true==y_pred)

# Confusion Matrix
def confusion_matrix(y_true, y_pred):
	classes = np.unique(y_true)
	class_idx = {c:i for i,c in enumerate(classes)}
	cm = np.zeros((len(classes), len(classes)), dtype=int)
	for t,p in zip(y_true, y_pred):
		cm[class_idx[t], class_idx[p]] += 1
	return cm, classes

# Precision, Recall, F1
def precision_recall_f1(y_true, y_pred):
	classes = np.unique(y_true)
	report = {}
	for c in classes:
		tp = np.sum((y_pred==c) & (y_true==c))
		fp = np.sum((y_pred==c) & (y_true!=c))
		fn = np.sum((y_pred!=c) & (y_true==c))
		precision = tp/(tp+fp) if (tp+fp)>0 else 0
		recall = tp/(tp+fn) if (tp+fn)>0 else 0
		f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
		support = np.sum(y_true==c)
		report[c] = {'precision': round(precision, 3), 'recall': round(recall, 3), 'f1': round(f1, 3), 'support': round(support, 3)}
	return report

# Exact Match (all predictions equal)
def exact_match(y_true, y_pred):
	return np.mean(y_true == y_pred)

# Top-K Accuracy
def top_k_accuracy(y_true, prob_list, k=3):
	correct = 0
	for true_label, probs in zip(y_true, prob_list):
		topk = sorted(probs, key=probs.get, reverse=True)[:k]
		if true_label in topk:
			correct += 1
	return correct / len(y_true)

# AUC (one-vs-rest for multiclass)
def auc_score(y_true, prob_list):
	from itertools import combinations
	classes = np.unique(y_true)
	aucs = []
	for c1,c2 in combinations(classes,2):
		# binary labels for class c1 vs c2
		y_bin = np.array([1 if y==c1 else 0 for y in y_true if y in [c1,c2]])
		probs_bin = np.array([p[c1] for y,p in zip(y_true, prob_list) if y in [c1,c2]])
		# ROC AUC manually
		pos = probs_bin[y_bin==1]
		neg = probs_bin[y_bin==0]
		count = sum(p>n for p in pos for n in neg)
		aucs.append(count / (len(pos)*len(neg)))
	return np.mean(aucs)

# N-gram extraction
def generate_ngrams(sequence, n=6):
	return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

n = 6
train['ngrams'] = train['sequence'].apply(lambda x: generate_ngrams(x, n))
val['ngrams']   = val['sequence'].apply(lambda x: generate_ngrams(x, n))
test['ngrams']  = test['sequence'].apply(lambda x: generate_ngrams(x, n))

# Build vocabulary
vocab = set()
for seq in train['ngrams']:
	vocab.update(seq)
vocab = list(vocab)
vocab_index = {k:i for i,k in enumerate(vocab)}

# Vectorization (bag-of-n-grams)
def vectorize(ngrams_list, vocab_index):
	vec = np.zeros(len(vocab_index))
	for ng in ngrams_list:
		if ng in vocab_index:
			vec[vocab_index[ng]] += 1
	return vec

X_train = np.array([vectorize(seq, vocab_index) for seq in train['ngrams']])
X_val   = np.array([vectorize(seq, vocab_index) for seq in val['ngrams']])
X_test  = np.array([vectorize(seq, vocab_index) for seq in test['ngrams']])

y_train = train['label'].values
y_val   = val['label'].values
y_test  = test['label'].values

# Naive Bayes from scratch
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

# Fit model (using your vectorized X_train/X_test)
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)

# Predictions
y_val_pred = nb.predict(X_val)
y_test_pred = nb.predict(X_test)
y_test_proba = nb.predict_prob(X_test)

# Compute metrics
print("Validation Accuracy:", accuracy(y_val, y_val_pred))
print("Test Accuracy:", accuracy(y_test, y_test_pred))
print("Exact Match (EM):", exact_match(y_test, y_test_pred))
print("Top-3 Accuracy:", top_k_accuracy(y_test, y_test_proba, k=3))
print("AUC:", auc_score(y_test, y_test_proba))
print("\nPrecision/Recall/F1:\n", precision_recall_f1(y_test, y_test_pred))
cm, classes = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:\n", cm)
print("Classes:", classes)

