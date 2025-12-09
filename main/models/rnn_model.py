import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ast
import os
import sys
import numpy as np

# Add test directory to path for importing test_metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test import test_metrics as tm

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
SEQ_LENGTH = 300
EMBEDDING_DIM = 16
HIDDEN_SIZE = 64
NUM_CLASSES = 4

MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "rnn_weights.pth"))
PLOT_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "graphs", "rnn_graph.png"))

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dataset Handling ---
class DNADataset(Dataset):
    def __init__(self, df):
        self.tokens = df['tokens'].tolist()
        self.labels = df['label_idx'].tolist()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Convert list to tensor
        token_seq = torch.tensor(self.tokens[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return token_seq, label

def process_dataframe(df):
    print("Processing dataframe...")
    
    # Map labels to integers if not already mapped
    # Check if label is string or int
    if df['label'].dtype == object:
        label_map = {
            'promoter': 0,
            'cds': 1,
            'terminator': 2,
            'intergenic': 3
        }
        # Filter out invalid labels if any
        df = df[df['label'].isin(label_map.keys())].copy()
        df['label_idx'] = df['label'].map(label_map)
    else:
        # Assume already int or mapped? 
        # But looking at bi_lstm.py, labels are strings.
        # Let's assume they are strings as per bi_lstm.py
        pass

    # If label_idx doesn't exist, create it (double check)
    if 'label_idx' not in df.columns:
         label_map = {
            'promoter': 0,
            'cds': 1,
            'terminator': 2,
            'intergenic': 3
        }
         df = df[df['label'].isin(label_map.keys())].copy()
         df['label_idx'] = df['label'].map(label_map)
    
    # Convert 'tokens' from string to list of ints
    print("Parsing tokens...")
    # Handle potential string representation of lists
    df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Ensure sequence length is 300
    def pad_or_truncate(tokens):
        if len(tokens) >= SEQ_LENGTH:
            return tokens[:SEQ_LENGTH]
        else:
            return tokens + [0] * (SEQ_LENGTH - len(tokens))

    df['tokens'] = df['tokens'].apply(pad_or_truncate)
    
    return df

# --- Model Architecture ---
class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Standard RNN
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        
        # RNN output
        # output shape: (batch_size, seq_length, hidden_size)
        # hn shape: (1, batch_size, hidden_size)
        output, hn = self.rnn(embedded)
        
        # We use the output of the last time step for classification
        # Alternatively could use hn[-1]
        last_step_output = output[:, -1, :] 
        
        activated = self.relu(last_step_output)
        logits = self.fc(activated)
        return logits

# --- Training and Evaluation ---
def train_model():
    # 1. Load Data
    print("Loading data from test/ directory...")
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test'))
        train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(base_path, 'val.csv'))
        test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    except FileNotFoundError:
        print(f"Error: Could not find train/val/test CSV files in '{base_path}' directory.")
        return

    # ... (rest of the train_model code)

def evaluate_model():
    print("\nEvaluating RNN Model...")
    # Load Data (Test only)
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test'))
        test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    except FileNotFoundError:
        print(f"Error: Could not find test.csv in '{base_path}' directory.")
        return

    test_df = process_dataframe(test_df)
    test_dataset = DNADataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = RNNClassifier(num_embeddings=4, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)

    # Load existing weights
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model weights from {MODEL_SAVE_PATH}...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("No existing weights found. Cannot evaluate.")
        return

    # Evaluation
    print("Evaluating on Test set...")
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            outputs = model(tokens)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Move to CPU and convert to numpy
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Map numeric labels back to string labels for test_metrics
    label_idx_to_name = {
        0: 'promoter',
        1: 'cds',
        2: 'terminator',
        3: 'intergenic'
    }
    
    y_test_str = np.array([label_idx_to_name[idx] for idx in all_labels])
    y_pred_str = np.array([label_idx_to_name[idx] for idx in all_predictions])
    
    # Convert probabilities to list of dicts format expected by test_metrics
    y_prob_list = []
    for prob_array in all_probabilities:
        prob_dict = {
            label_idx_to_name[i]: float(prob_array[i]) 
            for i in range(len(prob_array))
        }
        y_prob_list.append(prob_dict)
    
    # Compute comprehensive metrics
    tm.compute_metrics(y_test_str, y_pred_str, y_prob_list)

if __name__ == "__main__":
    train_model()
