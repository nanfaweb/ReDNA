import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import argparse
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve paths
# Current file: ReDNA/main/models/transformer_model.py
# Root relative to file: ReDNA/
BASE_DIR = Path(__file__).resolve().parents[2]
MAIN_DIR = BASE_DIR / "main"
TEST_DIR = MAIN_DIR / "test"
WEIGHTS_DIR = MAIN_DIR / "weights"
GRAPHS_DIR = MAIN_DIR / "graphs"

# Ensure output directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Add test directory to path for importing metrics
sys.path.append(str(TEST_DIR))
try:
    import test_metrics
except ImportError:
    print(f"Warning: Could not import test_metrics from {TEST_DIR}. Evaluation might fail.")

# Configuration
CONFIG = {
    'embed_dim': 128,
    'enc_layers': 3,
    'dec_layers': 2,
    'nhead': 4,
    'ff_dim': 256,
    'seq_len': 300,
    'num_classes': 4,
    'batch_size': 16,
    'epochs': 8,
    'lr': 2e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class DNADataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Parse tokens
        tokens_str = row['tokens']
        tokens = ast.literal_eval(tokens_str)
        
        # Pad or Truncate to 300
        if len(tokens) > CONFIG['seq_len']:
            tokens = tokens[:CONFIG['seq_len']]
        else:
            tokens = tokens + [0] * (CONFIG['seq_len'] - len(tokens))
            
        x = torch.tensor(tokens, dtype=torch.long)
        
        # Map label
        label_map = {'promoter': 0, 'cds': 1, 'terminator': 2, 'intergenic': 3}
        label_str = row['label']
        # Handle case where label might already be int (robustness)
        if isinstance(label_str, (int, np.integer)):
             label = label_str
        else:
             label = label_map.get(label_str, -1) # Default to -1 or error if not found? 
             # Assuming valid data.
        
        y = torch.tensor(label, dtype=torch.long)
        return x, y

class DNATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding: 4 DNA bases -> embed_dim
        self.embedding = nn.Embedding(4, CONFIG['embed_dim'])
        
        # Learned Positional Embeddings for Encoder
        self.enc_pos_embed = nn.Parameter(torch.randn(1, CONFIG['seq_len'], CONFIG['embed_dim']))
        
        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG['embed_dim'], 
            nhead=CONFIG['nhead'], 
            dim_feedforward=CONFIG['ff_dim'], 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=CONFIG['enc_layers'])
        
        # Decoder Query Token
        self.query_token = nn.Parameter(torch.randn(1, 1, CONFIG['embed_dim']))
        # Learned Positional Embedding for Decoder (as requested)
        self.dec_pos_embed = nn.Parameter(torch.randn(1, 1, CONFIG['embed_dim']))
        
        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=CONFIG['embed_dim'], 
            nhead=CONFIG['nhead'], 
            dim_feedforward=CONFIG['ff_dim'], 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=CONFIG['dec_layers'])
        
        # Classification Head
        self.classifier = nn.Linear(CONFIG['embed_dim'], CONFIG['num_classes'])
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        B = x.size(0)
        
        # Encoder Path
        embeds = self.embedding(x) # (B, 300, 128)
        embeds = embeds + self.enc_pos_embed # Add positional info
        memory = self.encoder(embeds) # (B, 300, 128)
        
        # Decoder Path
        # Expand query token to batch size
        tgt = self.query_token.expand(B, -1, -1) # (B, 1, 128)
        tgt = tgt + self.dec_pos_embed # Add positional info (broadcast)
        
        # Decode attending to memory
        out = self.decoder(tgt, memory) # (B, 1, 128)
        
        # Classification
        out = out.squeeze(1) # (B, 128)
        logits = self.classifier(out) # (B, 4)
        
        return logits

def train_model():
    print("Loading datasets...")
    train_dataset = DNADataset(TEST_DIR / "train.csv")
    val_dataset = DNADataset(TEST_DIR / "val.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = DNATransformer().to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"Starting training on {CONFIG['device']} for {CONFIG['epochs']} epochs...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        val_loss, val_acc = evaluate_loop(model, val_loader, criterion)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = WEIGHTS_DIR / "transformer_weights.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  Validation accuracy improved. Saved model to {save_path}")
            
        # Update plots every epoch
        save_plots(history)
    
    # Evaluate Final Model on Test Set
    evaluate_test_set()

def evaluate_loop(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return loss, acc

def save_plots(history):
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy Plot
    ax1.plot(epochs, history['train_acc'], label='Train Accuracy')
    ax1.plot(epochs, history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss Plot
    ax2.plot(epochs, history['train_loss'], label='Train Loss')
    ax2.plot(epochs, history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    save_path = GRAPHS_DIR / "transformer_training.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"saved training graphs to {save_path}")
    plt.close()

def evaluate_test_set():
    print("\nEvaluating on Test Set...")
    try:
        test_dataset = DNADataset(TEST_DIR / "test.csv")
    except FileNotFoundError:
        print("Test file not found.")
        return

    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = DNATransformer().to(CONFIG['device'])
    weights_path = WEIGHTS_DIR / "transformer_weights.pth"
    
    if not weights_path.exists():
        print(f"No weights found at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=CONFIG['device']))
    model.eval()
    
    y = []
    y_pred = []
    y_prob = [] # List of dicts for test_metrics
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(CONFIG['device'])
            outputs = model(inputs)
            probs_batch = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_batch = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_batch = labels.numpy()
            
            y.extend(labels_batch)
            y_pred.extend(predicted_batch)
            
            # Convert probs to list of dicts for test_metrics
            for probs in probs_batch:
                prob_dict = {i: p for i, p in enumerate(probs)}
                y_prob.append(prob_dict)
                
    # Convert lists to arrays for y and y_pred
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    # Call the test_metrics module
    if 'test_metrics' in sys.modules:
        print("Calculating metrics using test_metrics module...")
        try:
            test_metrics.compute_metrics(y, y_pred, y_prob)
        except Exception as e:
            print(f"Error calling compute_metrics: {e}")
            # Fallback evaluation if module fails
            acc = np.mean(y == y_pred)
            print(f"Fallback Test Accuracy: {acc:.4f}")
    else:
        print("test_metrics module not available.")
        print(f"Test Accuracy: {np.mean(y == y_pred):.4f}")

def main():
    parser = argparse.ArgumentParser(description='DNA Transformer Model')
    parser.add_argument('--eval', action='store_true', help='Evaluation only mode')
    args = parser.parse_args()
    
    if args.eval:
        evaluate_test_set()
    else:
        train_model()

if __name__ == "__main__":
    main()
