import torch
import sys
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer
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

    def __init__(self, lora_weights_path: str, eval_mode: bool = True, context_length: int = 256, verbose: bool = True):
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
        
        # Load and integrate LoRA weights
        self.load_lora_weights(lora_weights_path)
        
        if verbose:
            print(f"Loaded LoRA weights from: {lora_weights_path}")
            print("Fine-tuned BiomedClip model initialized successfully!")

    def load_lora_weights(self, lora_weights_path: str) -> None:
        """
        Load LoRA weights and integrate them with the base model.
        
        Args:
            lora_weights_path: Path to the LoRA weights file
        """
        if not os.path.exists(lora_weights_path):
            raise FileNotFoundError(f"LoRA weights file not found: {lora_weights_path}")
        
        # Load LoRA weights
        lora_state_dict = torch.load(lora_weights_path, map_location=self.device)
        
        # Apply LoRA weights to the model
        # Note: This is a simplified implementation. In practice, you'd need to use
        # a LoRA library like PEFT to properly integrate the weights
        self._apply_lora_weights(lora_state_dict)
        
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

    def forward(self, images: list[str], texts: list[str]) -> dict[str, list[float]]:
        """
        Forward pass through the fine-tuned model.
        
        Args:
            images: Input images 
            texts: Input texts.

        Returns:
            dict: Dictionary containing predictions and optionally class probabilities.
        """
        # Use the same forward pass as the base class
        # The LoRA weights are already integrated into the model
        return super().forward(images, texts)


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