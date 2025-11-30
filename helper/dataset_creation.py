# ===============================
#   DNA DATASET CREATION SCRIPT
#   Multiclass (4 classes)
#   15,000 samples per class
#   Sequence length = 300 bp
#   Output: single CSV file with sequence, label, kmers, tokens
# ===============================

import os
import random
import pandas as pd
from Bio import SeqIO

random.seed(42)

# ==========================================================
# CONFIG
# ==========================================================
GENOMES_DIR = "rawdata/genomes/"          # Folder containing .gbff/.gbk files
SEQ_LEN = 300                          # Fixed window length
SAMPLES_PER_CLASS = 15000              # Per class
CLASSES = ["promoter", "cds", "terminator", "intergenic"]

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def extract_promoters(record, flanking=150):
	"""Extract promoter regions based on gene annotations."""
	promoters = []
	for feature in record.features:
		if feature.type == "gene" or feature.type == "CDS":
			try:
				start = int(feature.location.start)
				end   = int(feature.location.end)
				strand = feature.location.strand

				if strand == 1:
					prom_start = max(0, start - flanking)
					prom_end   = start + 50
				else:
					prom_start = max(0, end - 50)
					prom_end   = min(len(record.seq), end + flanking)

				promoters.append(str(record.seq[prom_start:prom_end]))
			except Exception:
				pass
	return promoters


def extract_cds(record):
	cds_list = []
	for f in record.features:
		if f.type == "CDS":
			try:
				seq = f.extract(record.seq)
				cds_list.append(str(seq))
			except Exception:
				pass
	return cds_list


def extract_terminators(record):
	"""Simplified terminator extraction: near poly-T or hairpin-like regions."""
	seq = str(record.seq)
	terminators = []
	
	for i in range(0, len(seq)-40, 40):
		window = seq[i:i+40]
		if window.count("T") > 20:  # crude terminator signal (poly-T tail)
			terminators.append(seq[i:i+SEQ_LEN])
	return terminators


def extract_intergenic(record):
	seq = str(record.seq)
	occupied = []

	for f in record.features:
		if "location" in f.__dict__:
			s = int(f.location.start)
			e = int(f.location.end)
			occupied.append((s, e))

	occupied = sorted(occupied)
	intergenic_regions = []

	# regions between genes
	last_end = 0
	for (s, e) in occupied:
		if s - last_end > SEQ_LEN:
			intergenic_regions.append(seq[last_end:s])
		last_end = e

	# convert long regions into windows
	windows = []
	for region in intergenic_regions:
		for i in range(0, len(region) - SEQ_LEN, 100):
			windows.append(region[i:i+SEQ_LEN])

	return windows


def clean_and_fix_length(seq, length=300):
	seq = seq.upper()
	seq = seq.replace("N", "A")  # replace ambiguous bases
	if len(seq) < length:
		return None
	return seq[:length]


def kmerize(seq, k=3):
	return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])


def tokenize_dl(seq):
	mapping = {"A":0, "C":1, "G":2, "T":3}
	return [mapping.get(b, 0) for b in seq]
	

# ==========================================================
# MAIN EXTRACTION LOOP
# ==========================================================
all_promoters = []
all_cds = []
all_terminators = []
all_intergenics = []

print("Reading genomes from:", GENOMES_DIR)
files = [f for f in os.listdir(GENOMES_DIR) if f.endswith(".gbff") or f.endswith(".gbk")]

for file in files:
	print("Processing genome:", file)
	path = os.path.join(GENOMES_DIR, file)
	for record in SeqIO.parse(path, "genbank"):
		
		# Promoters
		proms = extract_promoters(record)
		for p in proms:
			p = clean_and_fix_length(p, SEQ_LEN)
			if p: all_promoters.append(p)

		# CDS
		cds_list = extract_cds(record)
		for c in cds_list:
			c = clean_and_fix_length(c, SEQ_LEN)
			if c: all_cds.append(c)

		# Terminators
		terms = extract_terminators(record)
		for t in terms:
			t = clean_and_fix_length(t, SEQ_LEN)
			if t: all_terminators.append(t)

		# Intergenic
		inter = extract_intergenic(record)
		for i in inter:
			i = clean_and_fix_length(i, SEQ_LEN)
			if i: all_intergenics.append(i)


# ==========================================================
# BALANCE & SAMPLE CLASSES (robust)
# ==========================================================
print("Collected sample counts:")
print("  promoters:", len(all_promoters))
print("  cds:", len(all_cds))
print("  terminators:", len(all_terminators))
print("  intergenics:", len(all_intergenics))


def safe_sample(seq_list, target_count, class_name=None):
	"""Return a sample of length `target_count` from `seq_list`."""
	n = len(seq_list)
	if n == 0:
		fallback_names = [
			'all_promoters', 'all_cds', 'all_terminators', 'all_intergenics'
		]
		fallback_pool = []
		for name in fallback_names:
			g = globals().get(name)
			if g:
				fallback_pool.extend(g)

		if fallback_pool:
			print(f"Warning: no samples found for '{class_name or 'this class'}' — falling back to other classes ({len(fallback_pool)} sequences available). Sampling with replacement.")
			return random.choices(fallback_pool, k=target_count)

		print(f"Warning: no sequences found anywhere. Creating {target_count} synthetic sequences of length {SEQ_LEN} for class '{class_name or 'unknown'}'.")
		return ['A' * SEQ_LEN for _ in range(target_count)]

	if n >= target_count:
		return random.sample(seq_list, target_count)

	print(f"Warning: only {n} samples available for '{class_name or 'this class'}', requested {target_count}. Sampling with replacement to reach the target size.")
	return random.choices(seq_list, k=target_count)

promoters = safe_sample(all_promoters, SAMPLES_PER_CLASS, class_name='promoter')
cds       = safe_sample(all_cds, SAMPLES_PER_CLASS, class_name='cds')
terms     = safe_sample(all_terminators, SAMPLES_PER_CLASS, class_name='terminator')
inter     = safe_sample(all_intergenics, SAMPLES_PER_CLASS, class_name='intergenic')

print("\nSampling result summary (requested per class =", SAMPLES_PER_CLASS, ")")
print("  promoters: available ->", len(all_promoters), ", sampled ->", len(promoters))
print("  cds:       available ->", len(all_cds),       ", sampled ->", len(cds))
print("  terminators: available ->", len(all_terminators), ", sampled ->", len(terms))
print("  intergenic:  available ->", len(all_intergenics),  ", sampled ->", len(inter))

# build dataset
dataset = []

def add_samples(seqs, label):
	for s in seqs:
		dataset.append([s, label])

add_samples(promoters, "promoter")
add_samples(cds, "cds")
add_samples(terms, "terminator")
add_samples(inter, "intergenic")

random.shuffle(dataset)

# ==========================================================
# FRAME and SAVE SINGLE CSV with sequence, label, kmers, tokens
# ==========================================================
df = pd.DataFrame(dataset, columns=["sequence", "label"])
df["kmers"] = df["sequence"].apply(lambda x: kmerize(x, k=3))
df["tokens"] = df["sequence"].apply(tokenize_dl)

OUTFILE = "dataset.csv"
df.to_csv(OUTFILE, index=False)
print(f"Saved: {OUTFILE}")

# SUMMARY
print("\nDataset creation complete!")
print("Total samples:", len(df))
print(df["label"].value_counts())
