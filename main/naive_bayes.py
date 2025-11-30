import pandas as pd
import numpy as np
from test import test_metrics as tm

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
train = pd.read_csv(r'test/train.csv')
val   = pd.read_csv(r'test/val.csv')
test  = pd.read_csv(r'test/test.csv')

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

X_train = np.array([vectorize(seq, vocab_index) for seq in train['ngrams']])
X_val   = np.array([vectorize(seq, vocab_index) for seq in val['ngrams']])
X_test  = np.array([vectorize(seq, vocab_index) for seq in test['ngrams']])

y_train = train['label'].values
y_val   = val['label'].values
y_test  = test['label'].values

# Fit model (using your vectorized X_train/X_test)
model = NaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_prob(X_test)

# Compute metrics
tm.compute_metrics(y_test, y_test_pred, y_test_proba);

