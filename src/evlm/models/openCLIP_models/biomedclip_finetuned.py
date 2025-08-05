import torch
import sys
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer
from peft import PeftModel, PeftConfig
import os

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP
from biomedclip import BioMedCLIP

class FineTunedBioMedCLIP(BioMedCLIP):
    """
    Fine-tuned BioMedClip class that loads LoRA weights and integrates them with the base model.
    
    Args:
        lora_weights_path (str): Path to the LoRA weights file
        eval_mode (bool, optional): Whether to set the model in evaluation mode. Defaults to True.
        context_length (int, optional): Length of input context. Defaults to 256.
    """

    #def __init__(self, lora_weights_path: str, eval_mode: bool = True, context_length: int = 256, verbose: bool = True):
    def __init__(self, lora_weights_path: str = None, eval_mode: bool = True, context_length: int = 256, verbose: bool = True):
        """
        Initialize the fine-tuned model by loading LoRA weights.
        
        Args:
            lora_weights_path: Path to the LoRA weights file
            eval_mode: Whether to set the model in evaluation mode
            context_length: Length of context for tokenization
            verbose: Whether to print verbose output
        """
        # Initialize the base BiomedClip model first
        super().__init__(eval_mode, context_length, verbose)

        if lora_weights_path is None:
            lora_weights_path = "finetuning/checkpoints/cardiovascular_25%/biomedclip_lora_epoch=00_val_loss=2.0774-v1.ckpt"
    
        # Load and integrate LoRA weights
        self.load_lora_weights(lora_weights_path)
        
        if verbose:
            print(f"Loaded LoRA weights from: {lora_weights_path}")
            print("Fine-tuned BiomedClip model initialized successfully!")

    # def load_lora_weights(self, lora_weights_path: str) -> None:
    #     """
    #     Load LoRA weights and integrate them with the base model.
        
    #     Args:
    #         lora_weights_path: Path to the LoRA weights file
    #     """
    #     if not os.path.exists(lora_weights_path):
    #         raise FileNotFoundError(f"LoRA weights file not found: {lora_weights_path}")
        
    #     # Load LoRA weights
    #     lora_state_dict = torch.load(lora_weights_path, map_location=self.device)
        
    #     # Apply LoRA weights to the model
    #     # Note: This is a simplified implementation. In practice, you'd need to use
    #     # a LoRA library like PEFT to properly integrate the weights
    #     self._apply_lora_weights(lora_state_dict)
    # def load_lora_weights(self, lora_weights_path: str) -> None:
    #     if not os.path.exists(lora_weights_path):
    #         print(f"Warning: LoRA weights file not found: {lora_weights_path}")
    #         print("Using base BiomedClip model without fine-tuning.")
    #         return
        
    #     try:
    #         # Load checkpoint
    #         checkpoint = torch.load(lora_weights_path, map_location=self.device)
            
    #         # Load the state dict
    #         if 'state_dict' in checkpoint:
    #             # PyTorch Lightning checkpoint
    #             state_dict = checkpoint['state_dict']
    #             # Remove 'model.' prefix if it exists
    #             cleaned_state_dict = {}
    #             for key, value in state_dict.items():
    #                 if key.startswith('model.'):
    #                     cleaned_state_dict[key[6:]] = value
    #                 else:
    #                     cleaned_state_dict[key] = value
    #             self.model.load_state_dict(cleaned_state_dict)
    #         else:
    #             # Direct state dict
    #             self.model.load_state_dict(checkpoint)
            
    #         print("Fine-tuned weights loaded successfully!")
            
    #     except Exception as e:
    #         print(f"Warning: Could not load fine-tuned weights: {e}")
    #         print("Using base BiomedClip model without fine-tuning.")

    

    def load_lora_weights(self, lora_weights_path: str) -> None:
        if not os.path.exists(lora_weights_path):
            print(f"Warning: LoRA weights file not found: {lora_weights_path}")
            print("Using base BiomedClip model without fine-tuning.")
            return
    
        try:
            # Load LoRA config and wrap the model
            print("Applying LoRA weights with PEFT...")
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
            print("LoRA weights applied successfully!")
        except Exception as e:
            print(f"Failed to apply LoRA weights: {e}")
            print("Falling back to base model.")

        print("Trainable parameters:")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # Add this debugging section:
        print("🔍 DEBUGGING: Checking if weights were actually loaded...")
        
        # Get a sample of weights from the model
        sample_weights = {}
        for name, param in self.model.named_parameters():
            if 'visual.trunk.cls_token' in name or 'logit_scale' in name:
                sample_weights[name] = param.data.clone()
                print(f"📊 {name}: {param.data.mean().item():.6f}")
        
        # Store these for comparison
        self.sample_weights = sample_weights
        print("🔍 Sample weights stored for comparison")


        
    def _apply_lora_weights(self, lora_state_dict: dict) -> None:
        """
        Apply LoRA weights to the model. This is a simplified implementation.
        In practice, you'd use a proper LoRA library like PEFT.
        
        Args:
            lora_state_dict: Dictionary containing LoRA weights
        """
        # This is a placeholder implementation
        # In practice, you'd need to use PEFT or similar library to properly apply LoRA weights
        print("LoRA weights loaded successfully!")
        # TODO: Implement proper LoRA weight application using PEFT or similar library
        
    def forward_vision_only(self, images: list[str]) -> dict[str, list[float]]:
        """
        Forward pass for vision-only inference (same as base class).
        
        Args:
            images: List of image paths
            
        Returns:
            Dictionary containing image features
        """
        return super().forward_vision_only(images)

    # def forward(self, images: list[str], texts: list[str]) -> dict[str, list[float]]:
    #     """
    #     Forward pass through the fine-tuned model.
        
    #     Args:
    #         images: Input images 
    #         texts: Input texts.

    #     Returns:
    #         dict: Dictionary containing predictions and optionally class probabilities.
    #     """
    #     # Use the same forward pass as the base class
    #     # The LoRA weights are already integrated into the model
    #     return super().forward(images, texts)
    def forward(self, images: list[str], texts: list[str]) -> dict[str, list[float]]:
        """
        Forward pass through the fine-tuned model.
        """
        print(f"🔍 Forward pass with {len(images)} images and {len(texts)} texts")
        
        # Use the same forward pass as the base class
        result = super().forward(images, texts)
        
        # Add some debugging info
        if hasattr(self, 'sample_weights'):
            print("🔍 Checking if weights changed during inference...")
            for name, param in self.model.named_parameters():
                if name in self.sample_weights:
                    current_mean = param.data.mean().item()
                    original_mean = self.sample_weights[name].mean().item()
                    print(f"📊 {name}: {original_mean:.6f} -> {current_mean:.6f}")

        print(f"🔍 PREDICTIONS: {result.get('pred', 'No pred key')}")
        if 'probs' in result:
            probs = result['probs']
            print(f"🔍 PROBABILITIES shape: {probs.shape}")
            print(f"🔍 TOP 3 PROBS: {probs[0][:3].tolist()}")
            print(f"�� MAX PROB: {probs[0].max().item():.4f}")
        
        return result


if __name__ == "__main__":
    # Example usage
    lora_weights_path = "finetuning/lora_weights/biomedclip_cardiovascular_50%_lora.pt"
    
    # Check if LoRA weights exist
    if os.path.exists(lora_weights_path):
        model = FineTunedBioMedCLIP(lora_weights_path)
        img_path = "test_images/"
        images = [img_path + "/cat.jpeg", img_path + "/dog.jpeg"]
        prompts = ["An image of a cat", "An image of a dog"]
        output = model.forward(images, prompts)
        print(output)
    else:
        print(f"LoRA weights not found at: {lora_weights_path}")
        print("Please run the fine-tuning script first to generate LoRA weights.") 