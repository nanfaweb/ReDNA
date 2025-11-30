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
EPOCHS = 20
SEQ_LENGTH = 300
EMBEDDING_DIM = 16
HIDDEN_SIZE = 64
NUM_CLASSES = 4
DATASET_PATH = 'dataset.csv' # Assuming script is run from project root
MODEL_SAVE_PATH = 'rnn_model_weights.pth'
PLOT_SAVE_PATH = 'training_results.png'

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

def load_and_process_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Map labels to integers
    label_map = {
        'promoter': 0,
        'cds': 1,
        'terminator': 2,
        'intergenic': 3
    }
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
            # Pad with 0? Or maybe 4 (if 0-3 are bases)? 
            # Prompt says "tokens (list of integers representing each nucleotide, e.g. [0,1,3,2,...])"
            # Usually 0-3 are A,C,G,T. Let's assume 0 padding is okay or handled by embedding.
            # However, if 0 is a valid token, padding with it might be ambiguous without a mask.
            # Given the prompt doesn't specify padding token, and usually these are fixed length 300,
            # we will just pad with 0.
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
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = load_and_process_data(DATASET_PATH)
    
    # 2. Split Data
    # 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_idx'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_idx'])
    
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
