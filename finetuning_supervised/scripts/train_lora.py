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
        self.learning_rate = float(self.config['training']['learning_rate'])
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
        # model_name = self.config['model']['name']
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        
        # Move to device
        self.model = self.model.to(self.device)

        # logger.info("Listing possible LoRA target modules...")
        # for name, _ in self.model.named_modules():
        #     if any(k in name.lower() for k in ["qkv", "proj", "fc1", "fc2", "query", "key", "value", "dense"]):
        #         logger.info(f"  {name}")
        
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
        # target_modules = []
        # for module_type, modules in lora_config['target_modules'].items():
        #     target_modules.extend(modules)
        target_modules = lora_config.get('target_modules', ["qkv", "proj", "fc1", "fc2", "query", "key", "value", "dense"])

        
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
    
    # def load_data(self, data_path: str) -> List[Dict]:
    #     """Load training/validation data."""
    #     logger.info(f"Loading data from 111111111111 {data_path}")
    #     with open(data_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     # print(type(data))
    #     # print(len(data))
    #     return data
    def load_data(self, data_file: str):
        """Load training data."""
        logger.info("data_path")
        
        # Define data file path
        # if task_type == 'combined':
        #     data_file = f"finetuning_supervised/data/splits/{organ_domain}/combined/{split_ratio}/train.json"
        # else:
        #     data_file = f"finetuning_supervised/data/splits/{organ_domain}/{task_type}/{split_ratio}/train.json"
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        with open(data_file, 'r') as f:
            train_data = json.load(f)
        
        logger.info(f"Loaded {len(train_data)} training samples")
        return train_data
    # def load_data(self, organ_domain, task_type, split_ratio):
    #     """Load and preprocess data."""
    #     logger.info(f"Loading data for {organ_domain} - {task_type} - {split_ratio}")
        
    #     # Load processed data
    #     data_file = f"/workspace/eVLLM_Sidd/eVLLM_microBench/finetuning_supervised/data/splits/{organ_domain}/{task_type}/{split_ratio}/train.json"
        
    #     if not os.path.exists(data_file):
    #         logger.error(f"Data file not found: {data_file}")
    #         raise FileNotFoundError(f"Data file not found: {data_file}")
        
    #     data = []
    #     with open(data_file, 'r') as f:
    #         for line in f:
    #             data.append(json.loads(line))
        
    #     logger.info(f"Loaded {len(data)} samples from {data_file}")
        
    #     # Split data
    #     train_size = int(len(data) * 0.9)
    #     train_data = data[:train_size]
    #     val_data = data[train_size:]
        
    #     logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
    #     # Create data loaders
    #     train_dataset = VQADataset(train_data, self.tokenizer, self.preprocess)
    #     val_dataset = VQADataset(val_data, self.tokenizer, self.preprocess)
        
    #     train_loader = DataLoader(
    #         train_dataset, 
    #         batch_size=self.config['training']['batch_size'],
    #         shuffle=True,
    #         collate_fn=self.collate_fn
    #     )
        
    #     val_loader = DataLoader(
    #         val_dataset, 
    #         batch_size=self.config['training']['batch_size'],
    #         shuffle=False,
    #         collate_fn=self.collate_fn
    #     )
        
    #     logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
    #     return train_loader, val_loader
    # def collate_fn(self, batch):
    #     """Custom collate function for batching."""
    #     logger.info(f"Collating batch of size {len(batch)}")
        
    #     images = []
    #     questions = []
    #     answers = []
    #     labels = []
        
    #     for item in batch:
    #         images.append(item['image'])
    #         questions.append(item['question'])
    #         answers.append(item['answers'])
    #         labels.append(item['label'])
        
    #     # Convert to tensors
    #     images = torch.stack(images)
    #     answers = torch.stack(answers)
    #     labels = torch.tensor(labels, dtype=torch.long)
        
    #     logger.info(f"Collated batch shapes: images={images.shape}, answers={answers.shape}, labels={labels.shape}")
        
    #     return {
    #         'images': images,
    #         'questions': questions,
    #         'answers': answers,
    #         'labels': labels
    #     }
    
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
            # print(item)
            image_path = item['image_path']
            if os.path.exists(image_path):
                # print("#############################################################")
                # image = self.preprocess(image_path).unsqueeze(0)
                from PIL import Image
                pil_image = Image.open(image_path).convert('RGB')
                image = self.preprocess(pil_image).unsqueeze(0)
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
        # question_tokens = self.tokenizer(questions, padding=True, truncation=True, return_tensors='pt')
        # question_tokens = {k: v.to(self.device) for k, v in question_tokens.items()}

        question_tokens = self.tokenizer(questions)
        question_tokens = torch.tensor(question_tokens).to(self.device)
        
        logger.info(f"Preprocessed batch: images={images.shape}, labels={labels.shape}, questions={len(questions)}")
        
        return images, question_tokens, labels
    
    # def train_epoch(self, train_data: List[Dict], optimizer, scaler=None):
    #     """Train for one epoch."""
    #     self.model.train()
    #     total_loss = 0.0
    #     correct_predictions = 0
    #     total_predictions = 0
        
    #     # Create dataloader
    #     train_batches = self.create_dataloader(train_data, shuffle=True)
    #     # print(type(train_batches))
    #     # print(len(train_batches))
    #     progress_bar = tqdm(train_batches, desc=f"Epoch {self.current_epoch + 1}")
        
    #     for batch_data in progress_bar:
    #         # Preprocess batch
    #         # print("BATCH DATA")
    #         # print(len(batch_data))
    #         images, question_tokens, labels = self.preprocess_batch(batch_data)
    #         if images is None:
    #             continue
            
    #         # Forward pass
    #         optimizer.zero_grad()
            
    #         if self.use_amp and scaler is not None:
    #             with torch.cuda.amp.autocast():
    #                 # outputs = self.model(images, **question_tokens)
    #                 # outputs = self.model(images, question_tokens)
    #                 # outputs = self.model.forward(images, question_tokens)
    #                 # outputs = self.model.base_model(images, question_tokens)
    #                 # image_features, text_features = self.model.base_model(images, question_tokens)
    #                 # outputs = self.model.base_model(images, question_tokens)
    #                 # logger.info(f"Model outputs type: {type(outputs)}")
    #                 # logger.info(f"Model outputs length: {len(outputs) if isinstance(outputs, (tuple, list)) else 'N/A'}")
    #                 # logger.info(f"Model outputs: {outputs}")
    #                 image_features, text_features, logit_scale = self.model.base_model(images, question_tokens)
    #                 # logits = outputs.logits
    #                 image_features = F.normalize(image_features, dim=-1)
    #                 text_features = F.normalize(text_features, dim=-1)
    #                 # Compute similarity matrix
    #                 # logits = torch.matmul(image_features, text_features.T)
    #                 logits = logit_scale * torch.matmul(image_features, text_features.T)

    #                 num_images = images.size(0)
    #                 num_questions_per_image = len(batch_data[0]['answer_options'])  # Should be 6 based on your data
    #                 num_questions = logits.size(1)
                    
    #                 # Reshape logits to group by image
    #                 logits = logits.view(num_images, num_questions_per_image, num_questions_per_image)
                    
    #                 # For each image, we want to compare it with its own answer options
    #                 # So we take the diagonal of each image's similarity matrix
    #                 batch_logits = []
    #                 for i in range(num_images):
    #                     # Get the similarity scores for this image with its own answer options
    #                     image_logits = logits[i, :, :]  # [num_answer_options, num_answer_options]
    #                     # Take the diagonal (image with its own answer options)
    #                     image_logits = torch.diag(image_logits)  # [num_answer_options]
    #                     batch_logits.append(image_logits)
                    
    #                 logits = torch.stack(batch_logits)  # [num_images, num_answer_options]
    #                 logits = logits.view(-1)  # Flatten to [num_questions]
                    
    #                 loss = F.cross_entropy(logits, labels)
                
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #         else:
    #             # outputs = self.model(images, **question_tokens)
    #             # outputs = self.model(images, question_tokens)
    #             # outputs = self.model.forward(images, question_tokens)
    #             # outputs = self.model.base_model(images, question_tokens)
    #             # image_features, text_features = self.model.base_model(images, question_tokens)
    #             # outputs = self.model.base_model(images, question_tokens)
    #             # logger.info(f"Model outputs type: {type(outputs)}")
    #             # logger.info(f"Model outputs length: {len(outputs) if isinstance(outputs, (tuple, list)) else 'N/A'}")
    #             # logger.info(f"Model outputs: {outputs}")   
    #             image_features, text_features, logit_scale = self.model.base_model(images, question_tokens)
    #             # logits = outputs.logits
    #             image_features = F.normalize(image_features, dim=-1)
    #             text_features = F.normalize(text_features, dim=-1)
                
    #             # Compute similarity matrix
    #             # logits = torch.matmul(image_features, text_features.T)
    #             logits = logit_scale * torch.matmul(image_features, text_features.T)

    #             num_images = images.size(0)
    #             num_questions_per_image = len(batch_data[0]['answer_options'])  # Should be 6 based on your data
    #             num_questions = logits.size(1)
                
    #             # Reshape logits to group by image
    #             logits = logits.view(num_images, num_questions_per_image, num_questions_per_image)
                
    #             # For each image, we want to compare it with its own answer options
    #             # So we take the diagonal of each image's similarity matrix
    #             batch_logits = []
    #             for i in range(num_images):
    #                 # Get the similarity scores for this image with its own answer options
    #                 image_logits = logits[i, :, :]  # [num_answer_options, num_answer_options]
    #                 # Take the diagonal (image with its own answer options)
    #                 image_logits = torch.diag(image_logits)  # [num_answer_options]
    #                 batch_logits.append(image_logits)
                
    #             logits = torch.stack(batch_logits)  # [num_images, num_answer_options]
    #             logits = logits.view(-1)  # Flatten to [num_questions]

                
    #             loss = F.cross_entropy(logits, labels)
                
    #             loss.backward()
    #             optimizer.step()
            
    #         # Calculate accuracy
    #         # predictions = torch.argmax(logits, dim=1)
    #         predictions = torch.argmax(logits.view(num_images, -1), dim=1)
    #         correct_predictions += (predictions == labels.view(num_images, -1)[:, 0]).sum().item()
    #         total_predictions += num_images

            
    #         correct_predictions += (predictions == labels).sum().item()
    #         total_predictions += labels.size(0)
    #         # print(total_predictions)
    #         # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            
    #         total_loss += loss.item()
            
    #         # Update progress bar
    #         avg_loss = total_loss / (progress_bar.n + 1)
    #         accuracy = correct_predictions / total_predictions
    #         progress_bar.set_postfix({
    #             'loss': f'{avg_loss:.4f}',
    #             'acc': f'{accuracy:.4f}'
    #         })
        
    #     return total_loss / len(train_batches), correct_predictions / total_predictions

    def train_epoch(self, train_data: List[Dict], optimizer, scaler=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Create dataloader
        train_batches = self.create_dataloader(train_data, shuffle=True)
        progress_bar = tqdm(train_batches, desc=f"Epoch {self.current_epoch + 1}")
        
        valid_batches = 0
        
        for batch_data in progress_bar:
            # Preprocess batch
            images, question_tokens, labels = self.preprocess_batch(batch_data)
            if images is None:
                continue
            
            valid_batches += 1
            
            # Forward pass
            optimizer.zero_grad()
            
            if self.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    # Get image features, text features, and logit scale
                    image_features, text_features, logit_scale = self.model.base_model(images, question_tokens)
                    
                    # Normalize features
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    
                    # Compute similarity matrix with temperature scaling
                    logits = logit_scale * torch.matmul(image_features, text_features.T)
                    
                    # Handle variable number of questions per image
                    num_images = images.size(0)
                    total_questions = question_tokens.size(0)
                    
                    questions_per_image = []
                    for item in batch_data:
                        num_options = len(item['answer_options'])
                        questions_per_image.append(num_options)
                    
                    max_questions = max(questions_per_image)
                    
                    # Process each image separately for classification
                    batch_logits = []
                    batch_labels = []
                    question_start = 0
                    
                    for i in range(num_images):
                        num_questions = questions_per_image[i]
                        question_end = question_start + num_questions
                        
                        # Get logits for this image with its questions
                        image_logits = logits[i, question_start:question_end]  # [num_questions]
                        image_labels = labels[question_start:question_end]  # [num_questions]
                        
                        # For classification, we want to predict which answer is correct
                        correct_idx = torch.where(image_labels == 1)[0]
                        if len(correct_idx) > 0:
                            correct_idx = correct_idx[0]  # Take the first correct answer
                            
                            # Pad logits to max_questions
                            if num_questions < max_questions:
                                padding = torch.zeros(max_questions - num_questions, device=image_logits.device)
                                image_logits = torch.cat([image_logits, padding])
                            
                            # Create classification logits and label
                            batch_logits.append(image_logits.unsqueeze(0))  # [1, max_questions]
                            batch_labels.append(correct_idx.unsqueeze(0))  # [1]
                        
                        question_start = question_end
                    
                    if not batch_logits:
                        continue
                    
                    # Stack all logits and labels
                    all_logits = torch.cat(batch_logits, dim=0)  # [num_images, max_questions]
                    all_labels = torch.cat(batch_labels, dim=0)  # [num_images]
                    
                    loss = F.cross_entropy(all_logits, all_labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Same logic for non-amp training
                image_features, text_features, logit_scale = self.model.base_model(images, question_tokens)
                
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                logits = logit_scale * torch.matmul(image_features, text_features.T)
                
                # Handle variable number of questions per image
                num_images = images.size(0)
                total_questions = question_tokens.size(0)
                
                questions_per_image = []
                for item in batch_data:
                    num_options = len(item['answer_options'])
                    questions_per_image.append(num_options)
                
                max_questions = max(questions_per_image)
                
                # Process each image separately for classification
                batch_logits = []
                batch_labels = []
                question_start = 0
                
                for i in range(num_images):
                    num_questions = questions_per_image[i]
                    question_end = question_start + num_questions
                    
                    # Get logits for this image with its questions
                    image_logits = logits[i, question_start:question_end]  # [num_questions]
                    image_labels = labels[question_start:question_end]  # [num_questions]
                    
                    # For classification, we want to predict which answer is correct
                    correct_idx = torch.where(image_labels == 1)[0]
                    if len(correct_idx) > 0:
                        correct_idx = correct_idx[0]
                        
                        # Pad logits to max_questions
                        if num_questions < max_questions:
                            padding = torch.zeros(max_questions - num_questions, device=image_logits.device)
                            image_logits = torch.cat([image_logits, padding])
                        
                        # Create classification logits and label
                        batch_logits.append(image_logits.unsqueeze(0))
                        batch_labels.append(correct_idx.unsqueeze(0))
                    
                    question_start = question_end
                
                if not batch_logits:
                    continue
                
                # Stack all logits and labels
                all_logits = torch.cat(batch_logits, dim=0)  # [num_images, max_questions]
                all_labels = torch.cat(batch_labels, dim=0)  # [num_images]
                
                loss = F.cross_entropy(all_logits, all_labels)
                
                loss.backward()
                optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(all_logits, dim=1)
            correct_predictions += (predictions == all_labels).sum().item()
            total_predictions += all_labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / valid_batches
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}'
            })
        
        return total_loss / valid_batches, correct_predictions / total_predictions



        
    # def train_epoch(self, train_data, optimizer, scaler=None):
    #     """Train for one epoch."""
    #     self.model.train()
    #     total_loss = 0.0
    #     correct_predictions = 0
    #     total_predictions = 0
        
    #     for batch_idx, batch in enumerate(tqdm(train_data, desc="Training")):
    #         # Debug: Print batch info
    #         logger.info(f"Batch {batch_idx}: batch size = {len(batch['images']) if 'images' in batch else 'N/A'}")
            
    #         # Move batch to device
    #         images = batch['images'].to(self.device)
    #         questions = batch['questions']
    #         answers = batch['answers'].to(self.device)
    #         labels = batch['labels'].to(self.device)
            
    #         # Debug: Print tensor shapes
    #         logger.info(f"Images shape: {images.shape}")
    #         logger.info(f"Answers shape: {answers.shape}")
    #         logger.info(f"Labels shape: {labels.shape}")
    #         logger.info(f"Labels values: {labels}")
            
    #         # Forward pass
    #         optimizer.zero_grad()
            
    #         try:
    #             outputs = self.model(images, questions, answers)
    #             logger.info(f"Outputs shape: {outputs.shape}")
    #             logger.info(f"Outputs values: {outputs}")
                
    #             loss = self.criterion(outputs, labels)
                
    #             # Backward pass
    #             if scaler:
    #                 scaler.scale(loss).backward()
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #             else:
    #                 loss.backward()
    #                 optimizer.step()
                
    #             total_loss += loss.item()
                
    #             # Calculate accuracy
    #             _, predicted = torch.max(outputs, 1)
    #             correct_predictions += (predicted == labels).sum().item()
    #             total_predictions += labels.size(0)
                
    #             logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}, Correct = {(predicted == labels).sum().item()}/{labels.size(0)}")
                
    #         except Exception as e:
    #             logger.error(f"Error in batch {batch_idx}: {e}")
    #             logger.error(f"Batch keys: {batch.keys()}")
    #             raise e
        
    #     # Debug: Print final stats
    #     logger.info(f"Epoch stats: total_loss = {total_loss}, correct_predictions = {correct_predictions}, total_predictions = {total_predictions}")
        
    #     if total_predictions == 0:
    #         logger.error("No predictions made during training epoch!")
    #         raise ValueError("No predictions made during training epoch")
        
    #     return total_loss / len(train_data), correct_predictions / total_predictions




    # def validate(self, val_data: List[Dict]) -> Tuple[float, float]:
    #     """Validate the model."""
    #     self.model.eval()
    #     total_loss = 0.0
    #     all_predictions = []
    #     all_labels = []
        
    #     val_batches = self.create_dataloader(val_data, shuffle=False)
        
    #     with torch.no_grad():
    #         for batch_data in tqdm(val_batches, desc="Validation"):
    #             images, question_tokens, labels = self.preprocess_batch(batch_data)
                
    #             if images is None:
    #                 continue
                
    #             # outputs = self.model(images, **question_tokens)
    #             # outputs = self.model.forward(images, question_tokens)
    #             # outputs = self.model.base_model(images, question_tokens)
    #             # image_features, text_features = self.model.base_model(images, question_tokens)
    #             image_features, text_features, logit_scale = self.model.base_model(images, question_tokens)


    #             # logits = outputs.logits
    #             image_features = F.normalize(image_features, dim=-1)
    #             text_features = F.normalize(text_features, dim=-1)
                
    #             # Compute similarity matrix
    #             # logits = torch.matmul(image_features, text_features.T)
    #             logits = logit_scale * torch.matmul(image_features, text_features.T)

    #             num_images = images.size(0)
    #             num_questions_per_image = len(batch_data[0]['answer_options'])  # Should be 6 based on your data
    #             num_questions = logits.size(1)
                
    #             # Reshape logits to group by image
    #             logits = logits.view(num_images, num_questions_per_image, num_questions_per_image)
                
    #             # For each image, we want to compare it with its own answer options
    #             # So we take the diagonal of each image's similarity matrix
    #             batch_logits = []
    #             for i in range(num_images):
    #                 # Get the similarity scores for this image with its own answer options
    #                 image_logits = logits[i, :, :]  # [num_answer_options, num_answer_options]
    #                 # Take the diagonal (image with its own answer options)
    #                 image_logits = torch.diag(image_logits)  # [num_answer_options]
    #                 batch_logits.append(image_logits)
                
    #             logits = torch.stack(batch_logits)  # [num_images, num_answer_options]
    #             logits = logits.view(-1)  # Flatten to [num_questions]

                
    #             loss = F.cross_entropy(logits, labels)
                
    #             # predictions = torch.argmax(logits, dim=1)
    #             predictions = torch.argmax(logits.view(num_images, -1), dim=1)
                
    #             total_loss += loss.item()
    #             all_predictions.extend(predictions.cpu().numpy())
    #             all_labels.extend(labels.cpu().numpy())
        
    #     avg_loss = total_loss / len(val_batches)
    #     accuracy = accuracy_score(all_labels, all_predictions)
        
    #     return avg_loss, accuracy


    def validate(self, val_data: List[Dict]) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels_list = []  # Changed variable name to avoid conflict
        
        val_batches = self.create_dataloader(val_data, shuffle=False)
        
        with torch.no_grad():
            for batch_data in tqdm(val_batches, desc="Validation"):
                images, question_tokens, labels = self.preprocess_batch(batch_data)
                
                if images is None:
                    continue
                
                # Get image features, text features, and logit scale
                image_features, text_features, logit_scale = self.model.base_model(images, question_tokens)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute similarity matrix with temperature scaling
                logits = logit_scale * torch.matmul(image_features, text_features.T)
                
                # Handle variable number of questions per image
                num_images = images.size(0)
                total_questions = question_tokens.size(0)
                
                questions_per_image = []
                for item in batch_data:
                    num_options = len(item['answer_options'])
                    questions_per_image.append(num_options)
                
                max_questions = max(questions_per_image)
                
                # Process each image separately for classification
                batch_logits = []
                batch_labels = []
                question_start = 0
                
                for i in range(num_images):
                    num_questions = questions_per_image[i]
                    question_end = question_start + num_questions
                    
                    # Get logits for this image with its questions
                    image_logits = logits[i, question_start:question_end]  # [num_questions]
                    image_labels = labels[question_start:question_end]  # [num_questions]
                    
                    # For classification, we want to predict which answer is correct
                    correct_idx = torch.where(image_labels == 1)[0]
                    if len(correct_idx) > 0:
                        correct_idx = correct_idx[0]
                        
                        # Pad logits to max_questions
                        if num_questions < max_questions:
                            padding = torch.zeros(max_questions - num_questions, device=image_logits.device)
                            image_logits = torch.cat([image_logits, padding])
                        
                        # Create classification logits and label
                        batch_logits.append(image_logits.unsqueeze(0))
                        batch_labels.append(correct_idx.unsqueeze(0))
                    
                    question_start = question_end
                
                if not batch_logits:
                    continue
                
                # Stack all logits and labels
                all_logits = torch.cat(batch_logits, dim=0)  # [num_images, max_questions]
                all_labels = torch.cat(batch_labels, dim=0)  # [num_images]
                
                loss = F.cross_entropy(all_logits, all_labels)
                
                predictions = torch.argmax(all_logits, dim=1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels_list.extend(all_labels.cpu().numpy())  # Fixed variable name
        
        avg_loss = total_loss / len(val_batches)
        accuracy = accuracy_score(all_labels_list, all_predictions)  # Fixed variable name
        
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
        
        # Normalize ratio to two-decimal string for directory naming (e.g., 0.1 -> "0.10")
        try:
            ratio_dir = f"{float(split_ratio):.2f}"
        except Exception:
            ratio_dir = split_ratio
        
        # Load data from new folder structure
        trains_dir = self.base_dir / 'data' / 'splits' / organ_domain / task_type / 'trains' / ratio_dir
        fixed_dir = self.base_dir / 'data' / 'splits' / organ_domain / task_type / 'fixed'
        train_path = trains_dir / 'train.json'
        val_path = fixed_dir / 'val.json'
        
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
            weight_decay=float(self.config['training']['weight_decay'])
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
                self.save_model(organ_domain, task_type, ratio_dir, epoch, val_acc)
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
    parser.add_argument('--task-type', type=str, required=True, choices=['coarse', 'fine','combined'],
                       help='Task type (coarse or fine or combined)')
    parser.add_argument('--split-ratio', type=str, required=True,
                       help='Split ratio (0.1, 0.25, 0.5, 0.75)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BioMedCLIPLoRATrainer(args.config)
    
    # Start training
    trainer.train(args.organ_domain, args.task_type, args.split_ratio)

if __name__ == "__main__":
    main() 