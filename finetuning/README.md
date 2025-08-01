# BiomedClip LoRA Fine-tuning Pipeline

This directory contains the fine-tuning pipeline for BiomedClip using LoRA (Low-Rank Adaptation) for contrastive learning on microscopy data.

## Overview

The pipeline fine-tunes BiomedClip on organ-specific datasets using contrastive learning with LoRA adapters. It supports:
- Percentage-based data sampling (25%, 50%, 75%, etc.)
- Contrastive learning with positive/negative pairs
- LoRA adaptation for efficient fine-tuning
- Model checkpointing and weight averaging

## Directory Structure

```
finetuning/
├── data/                    # Percentage-based datasets
├── checkpoints/             # Model checkpoints and LoRA weights
├── logs/                    # Training logs (TensorBoard)
├── scripts/                 # Training scripts
│   ├── prepare_data.py      # Data preparation script
│   └── train_lora.py        # Main training script
├── run_finetuning.py        # Complete pipeline runner
├── requirements.txt          # Dependencies
└── README.md               # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r finetuning/requirements.txt
```

2. Ensure you have the base BiomedClip model and data files ready.

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python finetuning/run_finetuning.py \
    --input_data organData/fineTuningData/cardiovascular/fineTuningData.jsonl \
    --percentages 25 50 75
```

### Step-by-Step

1. **Prepare Data** (optional, can be skipped with `--skip_data_prep`):
```bash
python finetuning/scripts/prepare_data.py \
    --input_file organData/fineTuningData/cardiovascular/fineTuningData.jsonl \
    --output_dir finetuning/data \
    --percentages 25 50 75
```

2. **Train Individual Model**:
```bash
python finetuning/scripts/train_lora.py \
    --data_file finetuning/data/cardiovascular_50%.jsonl \
    --output_dir finetuning/checkpoints/cardiovascular_50% \
    --lora_r 8 \
    --lora_alpha 16 \
    --batch_size 8 \
    --max_epochs 10 \
    --learning_rate 1e-4 \
    --temperature 0.5
```

## Configuration

### LoRA Parameters
- `lora_r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA alpha parameter (default: 16)
- `lora_dropout`: LoRA dropout rate (default: 0.1)

### Training Parameters
- `batch_size`: Training batch size (default: 8)
- `max_epochs`: Maximum training epochs (default: 10)
- `learning_rate`: Learning rate (default: 1e-4)
- `temperature`: Temperature for contrastive loss (default: 0.5)
- `val_split`: Validation split ratio (default: 0.1)

### Data Parameters
- `percentages`: List of data percentages to use (default: [25, 50, 75])
- `seed`: Random seed for reproducibility (default: 42)

## Data Format

The training data should be in JSONL format with the following structure:

```json
{
    "image_id": "unique_image_id",
    "image_path": "path/to/image.png",
    "text": "Description of the image",
    "is_positive": true,
    "task": "task_name",
    "caption": "caption_id"
}
```

Where:
- `image_id`: Unique identifier for the image
- `image_path`: Path to the image file
- `text`: Text description/caption
- `is_positive`: Boolean indicating if this is a positive pair
- `task`: Task identifier (for reference)
- `caption`: Caption identifier (for reference)

## Model Integration

The fine-tuned models are integrated into the existing evaluation pipeline:

1. **Model Registration**: Fine-tuned models are added to `CLIP_MODELS` in `constants.py`
2. **Model Class**: `FineTunedBioMedCLIP` class loads LoRA weights
3. **Inference**: Compatible with existing `clip_inference.py` pipeline

### Using Fine-tuned Models

```python
from src.evlm.models.openCLIP_models.biomedclip_finetuned import FineTunedBioMedCLIP

# Load fine-tuned model
model = FineTunedBioMedCLIP(
    lora_weights_path="finetuning/checkpoints/cardiovascular_50%/final_lora_weights.pt"
)

# Run inference
images = ["path/to/image1.png", "path/to/image2.png"]
texts = ["description1", "description2"]
output = model.forward(images, texts)
```

## Output Files

After training, you'll find:

1. **Checkpoints**: `finetuning/checkpoints/cardiovascular_X%/`
   - `biomedclip_lora_XX_val_loss.ckpt`: PyTorch Lightning checkpoints
   - `final_lora_weights.pt`: LoRA weights for inference

2. **Logs**: `finetuning/logs/`
   - TensorBoard logs for monitoring training

3. **Data**: `finetuning/data/`
   - `cardiovascular_X%.jsonl`: Percentage-based datasets

## Model Soup

For model soup evaluation, you can:

1. **Weight Averaging**: Average LoRA weights from different checkpoints
2. **Ensemble**: Use multiple fine-tuned models and average predictions
3. **Cross-validation**: Train on different data splits and ensemble

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Import Errors**: Ensure all dependencies are installed
3. **Data Loading Errors**: Check file paths and data format
4. **Training Divergence**: Reduce learning rate or increase temperature

### Performance Tips

1. **Mixed Precision**: Training uses FP16 for faster training
2. **Gradient Clipping**: Applied to prevent gradient explosion
3. **Early Stopping**: Prevents overfitting
4. **Checkpointing**: Saves best models based on validation loss

## Evaluation

After fine-tuning, evaluate the models using the existing pipeline:

```bash
python src/evlm/inference/model_inference_wrapper.py \
    --dataset_name "cardiovascular" \
    --model "FineTunedBioMedCLIP" \
    --output_dir "output_results"
```

## Future Enhancements

1. **Multi-organ Training**: Extend to other organ datasets
2. **Advanced LoRA**: Experiment with different LoRA configurations
3. **Model Soup**: Implement weight averaging strategies
4. **Hyperparameter Tuning**: Automated hyperparameter optimization
5. **Distributed Training**: Multi-GPU training support 