import numpy as np

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
	lines = []
	header = f"{'Class':<20}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}"
	lines.append(header)
	lines.append('-' * len(header))

	total = len(y_true)
	precisions = []
	recalls = []
	f1s = []
	supports = []

	for c in classes:
		tp = np.sum((y_pred == c) & (y_true == c))
		fp = np.sum((y_pred == c) & (y_true != c))
		fn = np.sum((y_pred != c) & (y_true == c))
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
		support = int(np.sum(y_true == c))

		report[c] = {
			'precision': round(precision, 3),
			'recall': round(recall, 3),
			'f1': round(f1, 3),
			'support': support
		}

		precisions.append(precision)
		recalls.append(recall)
		f1s.append(f1)
		supports.append(support)

		lines.append(f"{str(c):<20}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}{support:>10d}")

	# Averages
	macro_p = float(np.mean(precisions)) if precisions else 0.0
	macro_r = float(np.mean(recalls)) if recalls else 0.0
	macro_f1 = float(np.mean(f1s)) if f1s else 0.0

	weighted_p = float(np.average(precisions, weights=supports)) if total > 0 else 0.0
	weighted_r = float(np.average(recalls, weights=supports)) if total > 0 else 0.0
	weighted_f1 = float(np.average(f1s, weights=supports)) if total > 0 else 0.0

	lines.append('-' * len(header))
	lines.append(f"{'macro avg':<20}{macro_p:>10.3f}{macro_r:>10.3f}{macro_f1:>10.3f}{total:>10d}")
	lines.append(f"{'weighted avg':<20}{weighted_p:>10.3f}{weighted_r:>10.3f}{weighted_f1:>10.3f}{total:>10d}")

	# Return nicely formatted string (and keep the structured dict if needed)
	return "\n".join(lines)

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

# Compute Metrics
def compute_metrics(y, y_pred, y_prob):
	print("\nTest Accuracy:", accuracy(y, y_pred))
	print("\nExact Match (EM):", exact_match(y, y_pred))
	print("\nTop-3 Accuracy:", top_k_accuracy(y, y_prob, k=3))
	print("\nAUC:", auc_score(y, y_prob))
	print("\nPrecision/Recall/F1:\n", precision_recall_f1(y, y_pred))
	cm, classes = confusion_matrix(y, y_pred)
	print("\nConfusion Matrix:\n", cm)
	print("\nClasses:", classes)