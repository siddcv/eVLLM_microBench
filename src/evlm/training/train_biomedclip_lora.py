import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import json
import logging
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Optional

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from models.openCLIP_models.biomedclip_lora import BioMedCLIPLoRA
from dataloaders.jsonl_loader import JsonlDataset

class BiomedCLIPTrainingDataset(Dataset):
    """
    Dataset class for training BiomedCLIP with LoRA.
    
    Args:
        data_path: Path to the JSONL file containing training data.
        tokenizer: Tokenizer for text processing.
        preprocess: Image preprocessing function.
        max_length: Maximum sequence length for text.
    """
    
    def __init__(self, data_path: str, tokenizer, preprocess, max_length: int = 256):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load data from JSONL file."""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract image path and text
        image_path = item.get('image_path', '')
        text = item.get('text', '')
        
        # Load and preprocess image
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder if image loading fails
            image = torch.zeros(3, 224, 224)
        
        # Tokenize text
        tokens = self.tokenizer([text], context_length=self.max_length)
        
        return {
            'image': image,
            'text': tokens,
            'image_path': image_path,
            'text_raw': text
        }

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    Compute contrastive loss for image-text matching.
    
    Args:
        image_features: Image embeddings.
        text_features: Text embeddings.
        temperature: Temperature parameter for softmax.
    
    Returns:
        loss: Contrastive loss value.
    """
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Create labels (diagonal should be positive pairs)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Compute loss
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_i + loss_t) / 2

def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """
    Train for one epoch.
    
    Args:
        model: BiomedCLIP model with LoRA.
        dataloader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to run on.
        epoch: Current epoch number.
        args: Training arguments.
    
    Returns:
        avg_loss: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # Get image and text features
            image_features = model.model.encode_image(images)
            text_features = model.model.encode_text(texts)
            
            # Compute loss
            loss = contrastive_loss(image_features, text_features, temperature=args.temperature)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log to wandb if enabled
            if args.use_wandb:
                wandb.log({
                    'batch_loss': loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def validate(model, dataloader, device, epoch, args):
    """
    Validate the model.
    
    Args:
        model: BiomedCLIP model with LoRA.
        dataloader: DataLoader for validation data.
        device: Device to run on.
        epoch: Current epoch number.
        args: Training arguments.
    
    Returns:
        avg_loss: Average validation loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            # Move data to device
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            
            try:
                # Get image and text features
                image_features = model.model.encode_image(images)
                text_features = model.model.encode_text(texts)
                
                # Compute loss
                loss = contrastive_loss(image_features, text_features, temperature=args.temperature)
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train BiomedCLIP with LoRA adapters')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data JSONL file')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data JSONL file')
    
    # Model arguments
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout rate')
    parser.add_argument('--target_modules', nargs='+', 
                       default=['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                       help='Target modules for LoRA')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='biomedclip-lora',
                       help='WandB project name')
    parser.add_argument('--log_every', type=int, default=100,
                       help='Log every N batches')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("Initializing BiomedCLIP with LoRA...")
    model = BioMedCLIPLoRA(
        eval_mode=False,  # Set to False for training
        context_length=args.max_length,
        lora_config={
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'target_modules': args.target_modules,
            'lora_dropout': args.lora_dropout,
            'bias': 'none',
            'task_type': 'FEATURE_EXTRACTION'
        },
        verbose=True
    )
    
    # Set trainable parameters
    model.set_trainable_parameters(trainable=True)
    
    # Move model to device
    model.model.to(device)
    
    # Create datasets and dataloaders
    print("Creating datasets...")
    train_dataset = BiomedCLIPTrainingDataset(
        args.train_data,
        model.tokenizer,
        model.preprocess,
        max_length=args.max_length
    )
    
    val_dataset = BiomedCLIPTrainingDataset(
        args.val_data,
        model.tokenizer,
        model.preprocess,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args)
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, args)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, best_model_path)
            print(f"Best model saved to {best_model_path}")
    
    # Save LoRA adapters
    lora_save_path = output_dir / "lora_adapters"
    model.save_lora_adapters(str(lora_save_path))
    
    print("Training completed!")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 