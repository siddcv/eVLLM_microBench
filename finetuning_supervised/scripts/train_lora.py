#!/usr/bin/env python3
"""
LoRA fine-tuning script for BioMedCLIP on VQA tasks.
Implements supervised classification training with PEFT LoRA adapters.
"""

import json
import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, classification_report

# PEFT imports
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils import get_peft_model_state_dict

# OpenCLIP imports
from open_clip import create_model_from_pretrained, get_tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioMedCLIPLoRATrainer:
    """Trainer for BioMedCLIP with LoRA adapters."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.base_dir = Path(self.config['output']['base_dir'])
        self.model_dir = self.base_dir / self.config['output']['model_dir']
        self.results_dir = self.base_dir / self.config['output']['results_dir']
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.device = torch.device(self.config['model']['device'])
        self.batch_size = self.config['training']['batch_size']
        self.learning_rate = self.config['training']['learning_rate']
        self.num_epochs = self.config['training']['num_epochs']
        self.use_amp = self.config['training']['use_amp']
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        
        # Initialize LoRA config
        self.lora_config = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def setup_model(self):
        """Setup BioMedCLIP model with LoRA adapters."""
        logger.info("Loading BioMedCLIP model...")
        
        # Load base model
        model_name = self.config['model']['name']
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Freeze base model if specified
        if self.config['model']['freeze_base']:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Base model parameters frozen")
        
        # Setup LoRA configuration
        self.setup_lora_config()
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("Model setup completed")
    
    def setup_lora_config(self):
        """Setup LoRA configuration."""
        lora_config = self.config['lora']
        
        # Create target modules list
        target_modules = []
        for module_type, modules in lora_config['target_modules'].items():
            target_modules.extend(modules)
        
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # For classification
            inference_mode=False,
            r=lora_config['r']['text_encoder'],  # Use text encoder rank as default
            lora_alpha=lora_config['alpha']['text_encoder'],
            lora_dropout=lora_config['dropout'],
            target_modules=target_modules,
            bias=lora_config['bias']
        )
        
        logger.info(f"LoRA config: r={self.lora_config.r}, alpha={self.lora_config.lora_alpha}")
    
    def load_data(self, data_path: str) -> List[Dict]:
        """Load training/validation data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def create_dataloader(self, data: List[Dict], shuffle: bool = True):
        """Create DataLoader for training/validation."""
        # Group data by batch
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batches.append(batch_data)
        
        return batches
    
    def preprocess_batch(self, batch_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess a batch of data for training."""
        images = []
        questions = []
        labels = []
        
        for item in batch_data:
            # Load and preprocess image
            image_path = item['image_path']
            if os.path.exists(image_path):
                image = self.preprocess(image_path).unsqueeze(0)
                images.append(image)
                
                # Create question-answer pairs
                question = item['question']
                answer_options = item['answer_options']
                correct_idx = item['correct_answer_idx']
                
                # Create 4 separate samples (one for each answer option)
                for i, option in enumerate(answer_options):
                    full_question = f"{question} {option}"
                    questions.append(full_question)
                    labels.append(1 if i == correct_idx else 0)
        
        if not images:
            return None, None, None
        
        # Stack tensors
        images = torch.cat(images, dim=0).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Tokenize questions
        question_tokens = self.tokenizer(questions, padding=True, truncation=True, return_tensors='pt')
        question_tokens = {k: v.to(self.device) for k, v in question_tokens.items()}
        
        return images, question_tokens, labels
    
    def train_epoch(self, train_data: List[Dict], optimizer, scaler=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Create dataloader
        train_batches = self.create_dataloader(train_data, shuffle=True)
        
        progress_bar = tqdm(train_batches, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_data in progress_bar:
            # Preprocess batch
            images, question_tokens, labels = self.preprocess_batch(batch_data)
            
            if images is None:
                continue
            
            # Forward pass
            optimizer.zero_grad()
            
            if self.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, **question_tokens)
                    logits = outputs.logits
                    loss = F.cross_entropy(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(images, **question_tokens)
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels)
                
                loss.backward()
                optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (progress_bar.n + 1)
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}'
            })
        
        return total_loss / len(train_batches), correct_predictions / total_predictions
    
    def validate(self, val_data: List[Dict]) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        val_batches = self.create_dataloader(val_data, shuffle=False)
        
        with torch.no_grad():
            for batch_data in tqdm(val_batches, desc="Validation"):
                images, question_tokens, labels = self.preprocess_batch(batch_data)
                
                if images is None:
                    continue
                
                outputs = self.model(images, **question_tokens)
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_batches)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def save_model(self, organ_domain: str, task_type: str, split_ratio: str, epoch: int, val_acc: float):
        """Save the trained model."""
        model_path = self.model_dir / organ_domain / task_type / split_ratio
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        adapter_path = model_path / f"lora_adapters_epoch_{epoch}_acc_{val_acc:.4f}"
        self.model.save_pretrained(adapter_path)
        
        # Save training config
        config_path = model_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to {adapter_path}")
    
    def train(self, organ_domain: str, task_type: str, split_ratio: str):
        """Main training loop."""
        logger.info(f"Starting training for {organ_domain} {task_type} {split_ratio}")
        
        # Load data
        data_dir = self.base_dir / 'data' / 'splits' / organ_domain / task_type / split_ratio
        train_path = data_dir / 'train.json'
        val_path = data_dir / 'val.json'
        
        if not train_path.exists() or not val_path.exists():
            logger.error(f"Data files not found: {train_path}, {val_path}")
            return
        
        train_data = self.load_data(train_path)
        val_data = self.load_data(val_path)
        
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # Setup model
        self.setup_model()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.num_epochs
        )
        
        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(train_data, optimizer, scaler)
            
            # Validate
            val_loss, val_acc = self.validate(val_data)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(organ_domain, task_type, split_ratio, epoch, val_acc)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train BioMedCLIP with LoRA')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--organ-domain', type=str, required=True,
                       help='Organ domain to train on')
    parser.add_argument('--task-type', type=str, required=True, choices=['coarse', 'fine'],
                       help='Task type (coarse or fine)')
    parser.add_argument('--split-ratio', type=str, required=True,
                       help='Split ratio (0.1, 0.25, 0.5, 0.75)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BioMedCLIPLoRATrainer(args.config)
    
    # Start training
    trainer.train(args.organ_domain, args.task_type, args.split_ratio)

if __name__ == "__main__":
    main() 