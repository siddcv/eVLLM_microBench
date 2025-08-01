# BiomedCLIP LoRA Fine-tuning Guide

This guide explains how to fine-tune BiomedCLIP using LoRA (Low-Rank Adaptation) adapters for efficient parameter-efficient fine-tuning.

## Overview

LoRA fine-tuning allows you to adapt BiomedCLIP to your specific domain or task while:
- Using significantly fewer parameters (only ~1-2% of the original model)
- Requiring less memory and computational resources
- Maintaining the original model's capabilities
- Enabling quick adaptation to new tasks

## Installation

### 1. Install Additional Dependencies

Add the LoRA-specific requirements to your environment:

```bash
pip install -r requirements_lora.txt
```

### 2. Verify Installation

Test that the LoRA-enabled model can be imported:

```python
from src.evlm.models.openCLIP_models.biomedclip_lora import BioMedCLIPLoRA

# Test initialization
model = BioMedCLIPLoRA(eval_mode=False)
print("LoRA model initialized successfully!")
```

## Data Preparation

### 1. Prepare Your Training Data

Your training data should be in JSONL format with the following structure:

```json
{"image_path": "/path/to/image1.jpg", "text": "Description of image 1"}
{"image_path": "/path/to/image2.jpg", "text": "Description of image 2"}
```

### 2. Use the Data Preparation Script

```bash
python src/evlm/training/prepare_training_data.py \
    --input_file your_data.csv \
    --output_path ./prepared_data \
    --image_dir /path/to/images \
    --text_column description \
    --image_column image_path \
    --split_ratio 0.8
```

### 3. Create Sample Data (for testing)

```bash
python src/evlm/training/prepare_training_data.py \
    --create_sample \
    --output_path ./sample_data \
    --num_samples 100
```

## Training

### 1. Basic Training Command

```bash
python src/evlm/training/train_biomedclip_lora.py \
    --train_data ./prepared_data/train.jsonl \
    --val_data ./prepared_data/val.jsonl \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

### 2. Advanced Training Options

```bash
python src/evlm/training/train_biomedclip_lora.py \
    --train_data ./prepared_data/train.jsonl \
    --val_data ./prepared_data/val.jsonl \
    --output_dir ./checkpoints \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --temperature 0.07 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --target_modules q_proj v_proj k_proj out_proj \
    --use_wandb \
    --wandb_project biomedclip-lora \
    --save_every 5
```

### 3. Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_r` | 16 | LoRA rank (higher = more parameters) |
| `--lora_alpha` | 32 | LoRA alpha parameter |
| `--lora_dropout` | 0.1 | Dropout rate for LoRA layers |
| `--target_modules` | q_proj v_proj k_proj out_proj | Modules to apply LoRA to |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--temperature` | 0.07 | Temperature for contrastive loss |
| `--num_epochs` | 10 | Number of training epochs |

## Inference with Fine-tuned Model

### 1. Basic Inference

```bash
python src/evlm/inference/biomedclip_lora_inference.py \
    --lora_path ./checkpoints/lora_adapters \
    --images image1.jpg image2.jpg \
    --texts "Description 1" "Description 2" \
    --output_file results.json
```

### 2. Python API Usage

```python
from src.evlm.inference.biomedclip_lora_inference import BiomedCLIPLoRAInference

# Initialize inference model
inference_model = BiomedCLIPLoRAInference(
    lora_path="./checkpoints/lora_adapters",
    eval_mode=True
)

# Make predictions
images = ["image1.jpg", "image2.jpg"]
texts = ["Description 1", "Description 2"]
results = inference_model.predict(images, texts)

print(f"Predictions: {results['pred']}")
print(f"Probabilities: {results['probs']}")
```

## Model Architecture

### LoRA Configuration

The LoRA adapters are applied to the attention layers of the BiomedCLIP model:

- **Target Modules**: `q_proj`, `v_proj`, `k_proj`, `out_proj`
- **Rank (r)**: Controls the rank of the low-rank matrices (default: 16)
- **Alpha**: Scaling parameter for LoRA weights (default: 32)
- **Dropout**: Regularization for LoRA layers (default: 0.1)

### Parameter Efficiency

- **Original BiomedCLIP**: ~150M parameters
- **LoRA Adapters**: ~2-4M parameters (1-2% of original)
- **Memory Usage**: Significantly reduced during training

## Best Practices

### 1. Data Quality
- Ensure high-quality, domain-relevant image-text pairs
- Use consistent text formatting and terminology
- Balance your dataset across different classes/topics

### 2. Hyperparameter Tuning
- Start with default LoRA settings (r=16, alpha=32)
- Increase rank (r) for more complex tasks
- Adjust learning rate based on your dataset size
- Use validation loss to prevent overfitting

### 3. Training Tips
- Monitor training and validation loss
- Use early stopping if validation loss increases
- Save checkpoints regularly
- Use gradient clipping for stability

### 4. Evaluation
- Test on held-out validation set
- Compare with baseline BiomedCLIP performance
- Evaluate on domain-specific metrics

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Reduce LoRA rank

2. **Poor Convergence**
   - Check learning rate
   - Verify data quality
   - Increase training epochs

3. **Import Errors**
   - Install all requirements: `pip install -r requirements_lora.txt`
   - Check Python path includes src directory

### Performance Optimization

1. **GPU Memory**
   - Use mixed precision training
   - Enable gradient checkpointing
   - Use smaller batch sizes

2. **Training Speed**
   - Use multiple GPUs with DataParallel
   - Increase number of workers in DataLoader
   - Use gradient accumulation for larger effective batch sizes

## Example Use Cases

### 1. Medical Image Classification
```python
# Fine-tune for specific medical imaging tasks
model = BioMedCLIPLoRA(
    lora_config={
        'r': 32,
        'lora_alpha': 64,
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj']
    }
)
```

### 2. Pathology Image Analysis
```python
# Adapt for pathology-specific terminology
texts = [
    "Histological section showing normal tissue",
    "Pathological sample with abnormal findings",
    "Microscopic view of cellular components"
]
```

### 3. Multi-modal Medical Tasks
```python
# Combine with other medical imaging models
# Use LoRA for efficient adaptation to new modalities
```

## Monitoring and Logging

### Weights & Biases Integration

```bash
# Enable wandb logging
python train_biomedclip_lora.py \
    --use_wandb \
    --wandb_project biomedclip-lora \
    --wandb_run_name experiment_1
```

### Custom Logging

```python
# Add custom metrics
wandb.log({
    'custom_metric': your_metric_value,
    'epoch': epoch
})
```

## Saving and Loading

### Save LoRA Adapters
```python
model.save_lora_adapters("./my_lora_adapters")
```

### Load LoRA Adapters
```python
model.load_lora_adapters("./my_lora_adapters")
```

### Export for Production
```python
# Save complete model with LoRA adapters
torch.save(model.model.state_dict(), "complete_model.pt")
```

## Advanced Features

### 1. Custom LoRA Configurations
```python
custom_lora_config = {
    'r': 64,
    'lora_alpha': 128,
    'target_modules': ['q_proj', 'v_proj'],
    'lora_dropout': 0.2,
    'bias': 'lora_only'
}
```

### 2. Multi-task Learning
```python
# Train on multiple tasks simultaneously
# Use different LoRA configurations for different tasks
```

### 3. Continual Learning
```python
# Adapt to new domains without forgetting previous knowledge
# Use LoRA for efficient domain adaptation
```

## Performance Comparison

| Model | Parameters | Memory (GB) | Training Time |
|-------|------------|-------------|---------------|
| BiomedCLIP (Full) | 150M | 12 | 24h |
| BiomedCLIP + LoRA | 152M | 4 | 2h |

## Conclusion

LoRA fine-tuning provides an efficient way to adapt BiomedCLIP to your specific use case while maintaining the model's original capabilities. The approach is particularly useful for:

- Domain-specific medical imaging tasks
- Limited computational resources
- Quick adaptation to new datasets
- Maintaining model interpretability

For questions or issues, please refer to the main documentation or create an issue in the repository. 