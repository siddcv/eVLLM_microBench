# BioMedCLIP LoRA Fine-tuning Pipeline

This directory contains the complete pipeline for supervised fine-tuning of BioMedCLIP using LoRA (Low-Rank Adaptation) adapters for visual question answering (VQA) tasks on microscopy data.

## Overview

The pipeline implements supervised classification training for BioMedCLIP on µ-BENCH dataset, focusing on four pathology subdomains:
- **Cardiovascular**
- **Neuropathology** 
- **Hematopathology**
- **Gastrointestinal**

## Directory Structure

```
finetuning_supervised/
├── configs/
│   └── lora_config.yaml          # Configuration file for all parameters
├── data/
│   ├── processed/                 # Processed data files
│   └── splits/                    # Train/validation splits
├── models/                        # Saved LoRA adapters
├── results/                       # Evaluation results
├── scripts/
│   ├── preprocess_data.py         # Data preprocessing script
│   ├── train_lora.py             # LoRA training script
│   ├── biomedclip_lora.py        # Modified BioMedCLIP with LoRA support
│   └── evaluate_lora.py          # Evaluation script
└── requirements.txt               # Python dependencies
```

## Installation

1. **Install dependencies:**
```bash
pip install -r finetuning_supervised/requirements.txt
```

2. **Verify CUDA availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### 1. Data Preprocessing

First, preprocess your data to create training splits:

```bash
# Process all organ domains
python finetuning_supervised/scripts/preprocess_data.py

# Process specific organ domain
python finetuning_supervised/scripts/preprocess_data.py --organ-domain cardiovascular
```

This will:
- Load JSONL files from `organData/`
- Flatten nested data structure
- Create stratified train/validation splits (10%, 25%, 50%, 75%)
- Save processed data to `finetuning_supervised/data/`

### 2. LoRA Training

Train LoRA adapters for specific configurations:

```bash
# Train on cardiovascular coarse-grained tasks with 25% data
python finetuning_supervised/scripts/train_lora.py \
    --organ-domain cardiovascular \
    --task-type coarse \
    --split-ratio 0.25

# Train on neuropathology fine-grained tasks with 50% data
python finetuning_supervised/scripts/train_lora.py \
    --organ-domain neuropathology \
    --task-type fine \
    --split-ratio 0.5
```

### 3. Evaluation

Evaluate trained models and compare with base BioMedCLIP:

```bash
# Evaluate all organ domains
python finetuning_supervised/scripts/evaluate_lora.py

# Evaluate specific organ domain
python finetuning_supervised/scripts/evaluate_lora.py --organ-domain cardiovascular

# Generate summary report only
python finetuning_supervised/scripts/evaluate_lora.py --summary-only
```

## Configuration

Edit `configs/lora_config.yaml` to customize:

### LoRA Parameters
- **Ranks**: `r.text_encoder=16`, `r.vision_encoder=32`, `r.alignment=16`
- **Alpha**: Scaling parameter for LoRA weights
- **Target modules**: Specific layers to apply LoRA to
- **Dropout**: Regularization for LoRA layers

### Training Parameters
- **Learning rate**: `1e-4` (default)
- **Batch size**: `32` (adjust based on GPU memory)
- **Epochs**: `10` (with early stopping)
- **Mixed precision**: Enabled for faster training

### Data Parameters
- **Splits**: `[0.1, 0.25, 0.5, 0.75]` (percentage of data to use)
- **Organ domains**: `[cardiovascular, neuropathology, hematopathology, gastrointestinal]`
- **Task types**: `[coarse, fine]`

## Model Architecture

### Base Model
- **BioMedCLIP**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Vision Encoder**: ViT-Base (224x224 patches)
- **Text Encoder**: PubMedBERT (256 tokens)

### LoRA Configuration
- **Target layers**: Attention layers in both encoders
- **Rank**: 16-32 (configurable per component)
- **Freeze base**: Base model parameters frozen during training
- **Supervised training**: Classification loss on question-answer pairs

## Training Process

1. **Data Format**: Each sample contains:
   - Image path
   - Question text
   - 4 answer options
   - Correct answer index
   - Question type (modality, domain, stain, etc.)

2. **Training Strategy**:
   - Supervised classification (not contrastive learning)
   - Cross-entropy loss on answer predictions
   - Early stopping based on validation accuracy
   - Mixed precision training for efficiency

3. **Validation**:
   - Per-question-type accuracy tracking
   - Confidence score monitoring
   - Best model checkpoint saving

## Evaluation Metrics

The evaluation tracks:
- **Overall accuracy**: Across all question types
- **Per-question-type accuracy**: modality, domain, stain, classification, submodality
- **Confidence scores**: Model prediction confidence
- **Cross-domain performance**: Generalization across organ domains

## Results

Results are saved in multiple formats:
- **JSON**: Detailed evaluation results
- **CSV**: Tabular format for analysis
- **Summary**: Combined statistics across all experiments

### Example Results Structure
```
results/
├── cardiovascular_evaluation_20241201_143022.json
├── cardiovascular_evaluation_20241201_143022.csv
├── neuropathology_evaluation_20241201_143022.json
├── neuropathology_evaluation_20241201_143022.csv
├── combined_evaluation_results.csv
└── evaluation_summary.json
```

## Integration with Existing Codebase

The pipeline integrates with your existing eVLLM codebase:
- **Uses existing data format**: Compatible with current JSONL files
- **Maintains evaluation pipeline**: Works with existing evaluation scripts
- **Extends BioMedCLIP**: Adds LoRA capability without breaking existing functionality

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in `lora_config.yaml`
   - Use gradient accumulation
   - Enable mixed precision training

2. **Data Loading Errors**:
   - Verify image paths exist
   - Check JSONL file format
   - Ensure proper file permissions

3. **LoRA Loading Errors**:
   - Verify adapter directory structure
   - Check model compatibility
   - Ensure PEFT version compatibility

### Performance Tips

1. **GPU Memory**: RTX 6000 Ada (48GB) can handle batch size 32-64
2. **Training Speed**: Mixed precision reduces training time by ~30%
3. **Data Loading**: Use SSD storage for faster data access
4. **Monitoring**: Use wandb for experiment tracking

## Advanced Usage

### Custom LoRA Configuration
Edit `lora_config.yaml` to experiment with:
- Different ranks for different components
- Alternative target modules
- Custom learning rates and schedules

### Multi-GPU Training
For multi-GPU setups, modify `train_lora.py` to use:
```python
model = torch.nn.DataParallel(model)
```

### Custom Evaluation
Extend `evaluate_lora.py` to add:
- Additional metrics
- Custom evaluation datasets
- Advanced visualization

## Citation

If you use this pipeline in your research, please cite:
```bibtex
@inproceedings{evllm,
  title={Evaluate Vision-LLMs},
  author={Alejandro Lozano},
  booktitle={Github},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 