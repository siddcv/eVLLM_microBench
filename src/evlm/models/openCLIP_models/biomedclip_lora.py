import torch
import sys
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn

FIXTURES_PATH = str(Path(__file__).resolve().parent)
sys.path.append(FIXTURES_PATH)

from baseclip import BaseCLIP
from biomedclip import BioMedCLIP

class BioMedCLIPLoRA(BioMedCLIP):
    """
    BioMedCLIP class with LoRA adapters for fine-tuning.
    
    Args:
        eval_mode (bool, optional): Whether to set the model in evaluation mode. Defaults to True.
        context_length (int, optional): Length of input context. Defaults to 256.
        lora_config (dict, optional): LoRA configuration parameters.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    """

    def __init__(self, 
                 eval_mode: bool = True, 
                 context_length: int = 256, 
                 lora_config: dict = None,
                 verbose: bool = True):
        """
        Initialize the BiomedCLIP model with LoRA adapters.
        
        Args:
            eval_mode: Whether to set the model in evaluation mode.
            context_length: Length of input context.
            lora_config: LoRA configuration dictionary with parameters like:
                - r: LoRA rank
                - lora_alpha: LoRA alpha parameter
                - target_modules: List of module names to apply LoRA to
                - lora_dropout: Dropout rate for LoRA layers
            verbose: Whether to print verbose output.
        """
        # Initialize the base BiomedCLIP model
        super().__init__(eval_mode, context_length, verbose)
        
        # Default LoRA configuration
        if lora_config is None:
            lora_config = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                'lora_dropout': 0.1,
                'bias': 'none',
                'task_type': TaskType.FEATURE_EXTRACTION
            }
        
        self.lora_config = lora_config
        
        # Apply LoRA adapters to the model
        self._apply_lora_adapters()
        
        if verbose:
            print(f"Applied LoRA adapters with config: {lora_config}")

    def _apply_lora_adapters(self):
        """Apply LoRA adapters to the model."""
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            lora_dropout=self.lora_config['lora_dropout'],
            bias=self.lora_config['bias'],
            task_type=self.lora_config['task_type']
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()

    def save_lora_adapters(self, save_path: str):
        """
        Save the LoRA adapters to a file.
        
        Args:
            save_path: Path to save the LoRA adapters.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"LoRA adapters saved to {save_path}")

    def load_lora_adapters(self, load_path: str):
        """
        Load LoRA adapters from a file.
        
        Args:
            load_path: Path to load the LoRA adapters from.
        """
        self.model = self.model.from_pretrained(load_path)
        print(f"LoRA adapters loaded from {load_path}")

    def set_trainable_parameters(self, trainable: bool = True):
        """
        Set which parameters are trainable.
        
        Args:
            trainable: Whether to make LoRA parameters trainable.
        """
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = trainable
            else:
                param.requires_grad = False
        
        if trainable:
            print("LoRA parameters set to trainable")
        else:
            print("All parameters set to non-trainable")

    def get_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_param

    def forward(self, images: list[str], texts: list[str]) -> dict[str, list[float]]:
        """
        Forward pass through the model with LoRA adapters.
        
        Args:
            images: Input images 
            texts: Input texts.

        Returns:
            dict: Dictionary containing predictions and optionally class probabilities.
        """
        # Use the parent class forward method
        return super().forward(images, texts)

    def forward_vision_only(self, images: list[str]) -> dict[str, list[float]]:
        """
        Forward pass for vision-only encoding with LoRA adapters.
        
        Args:
            images: Input images.

        Returns:
            dict: Dictionary containing image features.
        """
        # Use the parent class forward_vision_only method
        return super().forward_vision_only(images)


if __name__ == "__main__":
    # Test the LoRA-enabled BiomedCLIP model
    model = BioMedCLIPLoRA(
        eval_mode=False,  # Set to False for training
        lora_config={
            'r': 16,
            'lora_alpha': 32,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
            'lora_dropout': 0.1,
            'bias': 'none',
            'task_type': TaskType.FEATURE_EXTRACTION
        }
    )
    
    # Test forward pass
    img_path = "test_images/"
    images = [img_path + "/cat.jpeg", img_path + "/dog.jpeg"]
    prompts = ["An image of a cat", "An image of a dog"]
    
    output = model.forward(images, prompts)
    print("Test output:", output) 