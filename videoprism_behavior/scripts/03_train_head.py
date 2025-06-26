import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from videoprism_behavior.src.data_utils import FeatureDataset
from videoprism_behavior.src.model import MultiHeadAttentionPooling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Runs a single training epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Mixed precision context manager
        with autocast():
            logits = model(features)
            loss = criterion(logits, labels)
        
        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()
        
        # Unscales gradients and calls optimizer.step()
        scaler.step(optimizer)
        
        # Updates the scale for next iteration
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Validating", unit="batch")
    with torch.no_grad():
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            accuracy = 100 * correct_predictions / total_samples
            pbar.set_postfix({'val_loss': loss.item(), 'val_acc': f"{accuracy:.2f}%"})

    avg_loss = total_loss / len(dataloader)
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train a classification head on pre-extracted features.")
    parser.add_argument("--annotations-file", type=str, default="videoprism_behavior/processed_data/clips/annotations.csv", help="Path to the master annotations CSV file.")
    parser.add_argument("--data-root", type=str, default="videoprism_behavior/processed_data", help="Root directory for processed data (features and clips).")
    parser.add_argument("--output-dir", type=str, default="videoprism_behavior/trained_models", help="Directory to save trained model checkpoints.")
    # Model Hyperparameters
    parser.add_argument("--feature-dim", type=int, default=768, help="Dimensionality of VideoPrism features (Base=768, Large=1024).")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads in the MAP layer.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Loading and Splitting ---
    full_dataset = FeatureDataset(annotations_file=Path(args.annotations_file), data_root=Path(args.data_root))
    
    if len(full_dataset) == 0:
        logging.error("Dataset is empty. Exiting.")
        return
        
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.annotations_df['label'][i] for i in indices]

    train_indices, val_indices = train_test_split(
        indices, test_size=args.val_split, random_state=42, stratify=labels,
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    logging.info(f"Data split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Use pin_memory=True for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    num_classes = len(full_dataset.classes)
    model = MultiHeadAttentionPooling(
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    # Compile the model for a significant speedup (PyTorch 2.0+)
    if int(torch.__version__.split('.')[0]) >= 2:
        logging.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize the gradient scaler for mixed precision
    scaler = GradScaler()
    
    # --- Training Loop ---
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Checkpointing and Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved to {best_model_path} with accuracy: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            logging.info(f"Early stopping triggered after {args.patience} epochs with no improvement.")
            break

    logging.info("Training finished.")

if __name__ == '__main__':
    main() 