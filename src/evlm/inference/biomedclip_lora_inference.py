import torch
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from models.openCLIP_models.biomedclip_lora import BioMedCLIPLoRA

class BiomedCLIPLoRAInference:
    """
    Inference class for BiomedCLIP with LoRA adapters.
    """
    
    def __init__(self, 
                 lora_path: str,
                 eval_mode: bool = True,
                 context_length: int = 256,
                 lora_config: dict = None,
                 device: str = 'auto'):
        """
        Initialize the inference model.
        
        Args:
            lora_path: Path to the LoRA adapters.
            eval_mode: Whether to set the model in evaluation mode.
            context_length: Length of input context.
            lora_config: LoRA configuration (if not loading from path).
            device: Device to use.
        """
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BioMedCLIPLoRA(
            eval_mode=eval_mode,
            context_length=context_length,
            lora_config=lora_config,
            verbose=True
        )
        
        # Load LoRA adapters if path is provided
        if lora_path:
            self.model.load_lora_adapters(lora_path)
        
        # Move model to device
        self.model.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def predict(self, images: List[str], texts: List[str]) -> Dict[str, Any]:
        """
        Make predictions using the fine-tuned model.
        
        Args:
            images: List of image paths.
            texts: List of text prompts.
        
        Returns:
            Dictionary containing predictions and probabilities.
        """
        return self.model.forward(images, texts)
    
    def encode_images(self, images: List[str]) -> torch.Tensor:
        """
        Encode images to get image features.
        
        Args:
            images: List of image paths.
        
        Returns:
            Image features tensor.
        """
        return self.model.forward_vision_only(images)
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to get text features.
        
        Args:
            texts: List of text prompts.
        
        Returns:
            Text features tensor.
        """
        tokenized_texts = self.model.tokenize(texts)
        with torch.no_grad():
            text_features = self.model.model.encode_text(tokenized_texts)
        return text_features
    
    def compute_similarity(self, images: List[str], texts: List[str]) -> torch.Tensor:
        """
        Compute similarity between images and texts.
        
        Args:
            images: List of image paths.
            texts: List of text prompts.
        
        Returns:
            Similarity scores tensor.
        """
        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(image_features, text_features.T)
        
        return similarity

def main():
    parser = argparse.ArgumentParser(description='Inference with BiomedCLIP LoRA model')
    
    parser.add_argument('--lora_path', type=str, required=True,
                       help='Path to LoRA adapters')
    parser.add_argument('--images', nargs='+', required=True,
                       help='List of image paths')
    parser.add_argument('--texts', nargs='+', required=True,
                       help='List of text prompts')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results (optional)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--context_length', type=int, default=256,
                       help='Context length for tokenization')
    
    args = parser.parse_args()
    
    # Initialize inference model
    print("Loading BiomedCLIP LoRA model...")
    inference_model = BiomedCLIPLoRAInference(
        lora_path=args.lora_path,
        eval_mode=True,
        context_length=args.context_length,
        device=args.device
    )
    
    # Make predictions
    print("Making predictions...")
    results = inference_model.predict(args.images, args.texts)
    
    # Print results
    print("\nResults:")
    print(f"Predictions: {results['pred']}")
    print(f"Probabilities shape: {results['probs'].shape}")
    
    if 'pred_prompt' in results:
        print(f"Predicted prompts: {results['pred_prompt']}")
    
    # Save results if output file is specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 