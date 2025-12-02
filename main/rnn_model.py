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
sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
import test_metrics as tm

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
SEQ_LENGTH = 300
EMBEDDING_DIM = 16
HIDDEN_SIZE = 64
NUM_CLASSES = 4
# DATASET_PATH = 'dataset.csv' # Removed
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'rnn_weights.pth')
PLOT_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'rnn_graph.png')

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
        base_path = os.path.dirname(__file__)
        train_df = pd.read_csv(os.path.join(base_path, 'test', 'train.csv'))
        val_df = pd.read_csv(os.path.join(base_path, 'test', 'val.csv'))
        test_df = pd.read_csv(os.path.join(base_path, 'test', 'test.csv'))
    except FileNotFoundError:
        print(f"Error: Could not find train/val/test CSV files in '{os.path.join(os.path.dirname(__file__), 'test')}' directory.")
        return

    train_df = process_dataframe(train_df)
    val_df = process_dataframe(val_df)
    test_df = process_dataframe(test_df)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    train_dataset = DNADataset(train_df)
    val_dataset = DNADataset(val_df)
    test_dataset = DNADataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    # num_embeddings=4 (0,1,2,3). If we padded with 0, we might need 5 if 0 is used for padding AND a base.
    # But usually DNA is 4 bases. Let's assume the input tokens are 0-3.
    # If there are other tokens, this might crash. Let's check max token in data if possible, 
    # but for now assume 4 is safe as per prompt "tokens... [0,1,3,2]".
    # Actually, if we pad with 0 and 0 is 'A', then we are padding with 'A'. 
    # Ideally we should use a special padding token if sequences vary, but prompt implies fixed length 300.
    # We'll stick to 4 embeddings.
    model = RNNClassifier(num_embeddings=4, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load existing weights if available
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model weights from {MODEL_SAVE_PATH}...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Model weights loaded. Continuing training...")
    else:
        print("No existing weights found. Starting training from scratch...")
    
    # 4. Training Loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                outputs = model(tokens)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = correct_val / total_val
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
              
    # 5. Plotting
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Plots saved to {PLOT_SAVE_PATH}")
    
    # 6. Testing with comprehensive metrics
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
    
    # 7. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
