import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys
import os
from peft import LoraConfig, get_peft_model, TaskType

# Add the src directory to the path to import the BiomedClip model
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from evlm.models.openCLIP_models.biomedclip import BioMedCLIP
from peft import LoraConfig, get_peft_model, TaskType

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with positive/negative pairs.
    """
    
    def __init__(self, jsonl_file: str, image_dir: str = None):
        """
        Initialize the dataset.
        
        Args:
            jsonl_file: Path to the JSONL file containing training data
            image_dir: Directory containing the images (optional, will be inferred from data)
        """
        self.data = self.load_jsonl(jsonl_file)
        self.image_dir = image_dir
        
        # Group data by image_id
        self.image_groups = {}
        for item in self.data:
            image_id = item['image_id']
            if image_id not in self.image_groups:
                self.image_groups[image_id] = []
            self.image_groups[image_id].append(item)
        
        # Create training pairs
        self.training_pairs = self.create_training_pairs()
        
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def create_training_pairs(self) -> List[Tuple[str, str, bool]]:
        """
        Create training pairs for contrastive learning.
        
        Returns:
            List of tuples: (image_path, text, is_positive)
        """
        pairs = []
        
        for image_id, items in self.image_groups.items():
            # Get image path
            image_path = items[0]['image_path']
            if self.image_dir:
                image_path = os.path.join(self.image_dir, os.path.basename(image_path))
            
            # Create pairs for each text entry
            for item in items:
                text = item['text']
                is_positive = item['is_positive']
                pairs.append((image_path, text, is_positive))
        
        return pairs
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        image_path, text, is_positive = self.training_pairs[idx]
        return {
            'image_path': image_path,
            'text': text,
            'is_positive': is_positive
        }

class BiomedClipLoRAModel(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning BiomedClip with LoRA.
    """
    
    def __init__(self, 
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 temperature: float = 0.5,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        """
        Initialize the model.
        
        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            temperature: Temperature for contrastive loss
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load the base BiomedClip model
        self.base_model = BioMedCLIP(eval_mode=False, verbose=False)


        # print("All model modules:")
        # for name, module in self.base_model.model.named_modules():
        #     print(name)
        
        # For now, use the base model directly without LoRA
        # TODO: Add LoRA support later
        # self.model = self.base_model.model

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            bias="none",
            # target_modules=["q_proj", "v_proj"],  # adjust if needed based on your model
            target_modules = ["qkv", "query", "value", "key"], #TODO
            task_type=TaskType.FEATURE_EXTRACTION  # or another appropriate task type
        )
        
        # Temporarily disable LoRA to fix interface issues
        # self.model = get_peft_model(self.base_model.model, lora_config)
        self.model = self.base_model.model
        
        # Training parameters
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Print trainable parameters
        self.print_trainable_parameters()
        
        # Store initial weights for comparison
        self.initial_weights = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
        
        print("🔍 Initial weights stored for comparison")
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}")
    
    def check_weight_changes(self, step_info=""):
        """Check if weights have changed since initialization."""
        print(f"\n{'='*60}")
        print(f"WEIGHT CHANGE CHECK {step_info}")
        print(f"{'='*60}")
        
        total_change = 0
        changed_params = 0
        unchanged_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.initial_weights:
                initial_weight = self.initial_weights[name]
                current_weight = param.data
                
                # Calculate change
                weight_diff = torch.abs(current_weight - initial_weight)
                max_change = torch.max(weight_diff).item()
                mean_change = torch.mean(weight_diff).item()
                
                if max_change > 1e-8:  # Meaningful change
                    # print(f"✅ {name}: CHANGED")
                    # print(f"   Max change: {max_change:.2e}")
                    # print(f"   Mean change: {mean_change:.2e}")
                    changed_params += 1
                    total_change += mean_change
                else:
                    # print(f"❌ {name}: NO CHANGE")
                    unchanged_params += 1
        
        print(f"\n📊 SUMMARY:")
        print(f"   Changed parameters: {changed_params}")
        print(f"   Unchanged parameters: {unchanged_params}")
        print(f"   Total mean change: {total_change:.2e}")
        
        if changed_params > 0:
            print("🎉 TRAINING IS HAPPENING - Weights are updating!")
        else:
            print("🚨 NO TRAINING - Weights are NOT changing!")
        
        print(f"{'='*60}\n")
    
    def quick_weight_check(self):
        """Quick check if any weights have changed."""
        print("\n🔍 QUICK WEIGHT CHECK:")
        any_change = False
        
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.initial_weights:
                initial = self.initial_weights[name]
                current = param.data
                max_diff = torch.max(torch.abs(current - initial)).item()
                
                if max_diff > 1e-8:
                    # print(f"✅ {name}: CHANGED (max diff: {max_diff:.2e})")
                    any_change = True
                else:
                    # print(f"❌ {name}: NO CHANGE")
                    any_change=False
        
        if any_change:
            print("🎉 Model weights are being updated!")
        else:
            print("🚨 Model weights are NOT changing!")
        print()
    
    # def forward(self, images, texts):
    #     """Forward pass through the model."""
    #     return self.model(images, texts)

    def forward(self, images, texts=None):
        # Here, if texts is None but kwargs contains tokens, extract them accordingly
        # if texts is None and 'input_ids' in kwargs:
        #     # Extract tokens dict
        #     tokens = {k: v for k, v in kwargs.items()}
        #     # Then call underlying model with images and tokens dict
        #     return self.model(images, tokens)
        # else:
            # Original forward logic
            return self.model(images, texts)

    
    def contrastive_loss(self, image_features, text_features, labels):
        """
        Compute contrastive loss.
        
        Args:
            image_features: Image embeddings
            text_features: Text embeddings
            labels: Binary labels indicating positive/negative pairs
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Create labels for contrastive learning
        # For each image, the corresponding text should be the positive pair
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Extract batch data
        image_paths = batch['image_path']
        texts = batch['text']
        is_positive = batch['is_positive']
        
        # Use the base model's forward method which handles preprocessing correctly
        # output = self.base_model.forward(image_paths, texts)
        
        # Extract features from the output for contrastive loss
        # We need to get the features directly from the model
        processed_images = self.base_model.preprocess_image(image_paths)
        tokenized_texts = self.base_model.tokenize(texts)
        
        # Get features from the base model (not LoRA model for now)
        image_features, text_features, logit_scale = self.model(processed_images, tokenized_texts)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(image_features, text_features, is_positive)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Check weight changes every 50 steps
        if batch_idx % 50 == 0:
            self.check_weight_changes(f"(Step {batch_idx})")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Similar to training step but without gradient computation
        with torch.no_grad():
            image_paths = batch['image_path']
            texts = batch['text']
            is_positive = batch['is_positive']

            # Use the base model's forward method which handles preprocessing correctly
            # output = self.base_model.forward(image_paths, texts)
            
            # Extract features from the output for contrastive loss
            processed_images = self.base_model.preprocess_image(image_paths)
            tokenized_texts = self.base_model.tokenize(texts)
            
            # Get features from the base model (not LoRA model for now)
            image_features, text_features, logit_scale = self.model(processed_images, tokenized_texts)
            
            # Compute contrastive loss
            loss = self.contrastive_loss(image_features, text_features, is_positive)
            
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        super().on_train_epoch_end()
        self.check_weight_changes(f"(Epoch {self.current_epoch})")
    
    def on_save_checkpoint(self, checkpoint):
        """Save LoRA weights when checkpoint is saved."""
        # Save LoRA weights separately
        lora_weights = {}
        for name, param in self.named_parameters():
            if 'lora_' in name:
                lora_weights[name] = param.data.clone()
        
        checkpoint['lora_weights'] = lora_weights

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BiomedClip with LoRA')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to the training JSONL file')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images (optional)')
    parser.add_argument('--output_dir', type=str, default='finetuning/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha parameter')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for contrastive loss')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Maximum number of epochs')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create datasets
    dataset = ContrastiveDataset(args.data_file, args.image_dir)
    
    # Split into train/val
    train_size = int((1 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = BiomedClipLoRAModel(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        temperature=args.temperature,
        learning_rate=args.learning_rate
    )
    
    print("🔍 BEFORE TRAINING:")
    model.quick_weight_check()
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='biomedclip_lora_{epoch:02d}_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Create logger
    logger = TensorBoardLogger("finetuning/logs", name="biomedclip_lora")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator='auto',
        devices='auto',
        precision=16,  # Use mixed precision for faster training
        gradient_clip_val=1.0
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final LoRA weights
    final_checkpoint_path = checkpoint_callback.best_model_path
    if final_checkpoint_path:
        print(f"Best model saved at: {final_checkpoint_path}")
        
        # Save LoRA weights separately
        checkpoint = torch.load(final_checkpoint_path, map_location='cpu')
        lora_weights = checkpoint.get('lora_weights', {})
        
        if lora_weights:
            lora_save_path = Path(args.output_dir) / "final_lora_weights.pt"
            torch.save(lora_weights, lora_save_path)
            print(f"LoRA weights saved at: {lora_save_path}")
    print("AFTER TRAINING:")
    model.quick_weight_check()

if __name__ == "__main__":
    main() 