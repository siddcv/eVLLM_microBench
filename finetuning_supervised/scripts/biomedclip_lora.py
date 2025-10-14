#!/usr/bin/env python3
"""
Modified BioMedCLIP model with LoRA adapter support.
This module extends the existing BioMedCLIP to load LoRA adapters for inference.
"""

import torch
import sys
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioMedCLIPLoRA:
    """
    BioMedCLIP model with LoRA adapter support for inference.
    Extends the base BioMedCLIP model to load and use LoRA adapters.
    """
    
    def __init__(self, eval_mode: bool = True, context_length: int = 256, verbose: bool = True):
        """
        Initialize BioMedCLIP with LoRA support.
        
        Args:
            eval_mode: Whether to set the model in evaluation mode
            context_length: Length of input context
            verbose: Whether to print verbose information
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.context_length = context_length
        self.eval = eval_mode
        self.verbose = verbose
        
        # Initialize base model
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        
        # LoRA adapter path
        self.lora_adapter_path = None
        
        if self.verbose:
            print()
            print("="*80)
            print(f"BioMedCLIPLoRA model initialized with Eval:{self.eval}, Context Length:{self.context_length}, and device:{self.device}")
            print("="*80)
            print()
    
    def load_base_model(self):
        """Load the base BioMedCLIP model."""
        logger.info("Loading base BioMedCLIP model...")

        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # self.tokenizer = get_tokenizer(model_name)
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # Move to device
        self.model = self.model.to(self.device)
        
        if self.eval:
            self.model.eval()
        
        logger.info("Base model loaded successfully")
    
    def load_lora_adapter(self, adapter_path: str):
        """
        Load LoRA adapter weights.
        
        Args:
            adapter_path: Path to the LoRA adapter directory
        """
        if self.model is None:
            self.load_base_model()
        
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        
        try:
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.lora_adapter_path = adapter_path
            self.model = self.model.to(self.device)
            if self.eval:
                self.model.eval()
            
            logger.info("LoRA adapter loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            raise
    
    def tokenize(self, captions: list[str]) -> torch.Tensor:
        """
        Tokenize captions using the BioMedCLIP tokenizer.
        
        Args:
            captions: Captions to tokenize
            
        Returns:
            Tokenized captions tensor
        """
        return self.tokenizer(captions, context_length=self.context_length).to(self.device)
    

    def preprocess_image(self, image: str | list[str], return_tensors: str = "pt") -> torch.Tensor:
        """
        Preprocess images before feeding them into the model.
        
        Args:
            image: Image file path or list of image file paths
            return_tensors: Return format ("pt" for PyTorch tensors)
            
        Returns:
            Preprocessed images tensor
        """
        from PIL import Image
        
        if return_tensors == "pt":
            if isinstance(image, str) or isinstance(image, Path):
                # Load PIL image and preprocess
                pil_image = Image.open(image).convert('RGB')
                return torch.stack([self.preprocess(pil_image)]).to(self.device)
            elif isinstance(image, list) or isinstance(image, tuple):
                # Load PIL images and preprocess
                pil_images = [Image.open(img).convert('RGB') for img in image]
                return torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        elif return_tensors == "pil":
            if isinstance(image, str) or isinstance(image, Path):
                return [Image.open(image).convert('RGB')]
            elif isinstance(image, list) or isinstance(image, tuple):
                return [Image.open(img).convert('RGB') for img in image]
    


    def forward(self, images: list[str], texts: list[str]) -> dict[str, torch.Tensor]:
            """
            Forward pass through the model.
            
            Args:
                images: Input images
                texts: Input texts
                
            Returns:
                Dictionary containing predictions and probabilities
            """
            if self.model is None:
                raise ValueError("Model not loaded. Call load_base_model() or load_lora_adapter() first.")
            
            output = {}
            processed_images = self.preprocess_image(images)
            tokenized_texts = self.tokenize(texts)
            
            with torch.no_grad():
                # Check if we have LoRA adapters loaded
                if hasattr(self.model, 'base_model'):
                    # LoRA model - call base_model
                    model_output = self.model.base_model(processed_images, tokenized_texts)
                else:
                    # Base model - call directly
                    model_output = self.model(processed_images, tokenized_texts)
                

                # Handle different possible return types
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        # Unpack carefully
                        image_features = model_output[0]
                        text_features = model_output[1]
                        logit_scale = model_output[2]
                        # logger.info(f"Successfully unpacked 3 outputs")
                    elif len(model_output) == 2:
                        # Some models return (features, logit_scale) or similar
                        image_features = model_output[0]
                        text_features = model_output[1]
                        logit_scale = torch.tensor(1.0).to(self.device)
                        logger.info(f"Successfully unpacked 2 outputs")
                    else:
                        raise ValueError(f"Unexpected number of outputs: {len(model_output)}")
                else:
                    # Single output - might be logits directly
                    logits = model_output
                    output["pred"] = torch.argmax(logits, dim=1).to("cpu")
                    output["probs"] = logits.to("cpu")
                    output["confidence"] = torch.max(logits, dim=1)[0].to("cpu")
                    
                    # Convert to lists for compatibility
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            output[key] = value.tolist()
                    
                    return output
                
                # Original logic for tuple outputs
                logits = (logit_scale * image_features @ text_features.T).detach().softmax(dim=-1)
                
                output["pred"] = torch.argmax(logits, dim=1).to("cpu")
                output["probs"] = logits.to("cpu")
                
                # Add confidence scores
                output["confidence"] = torch.max(logits, dim=1)[0].to("cpu")
                
                # Convert to lists for compatibility
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        output[key] = value.tolist()
            
            return output



    def predict_single(self, image_path: str, question: str, answer_options: list[str]) -> dict:
        """
        Predict the correct answer for a single image-question pair.
        
        Args:
            image_path: Path to the image
            question: The question text
            answer_options: List of answer options
            
        Returns:
            Dictionary with prediction results
        """
        # Create full question-answer pairs
        full_questions = [f"{question} {option}" for option in answer_options]
        
        logger.info(f"predict_single: image_path={image_path}")
        # logger.info(f"predict_single: full_questions={full_questions}")
        
        # Run inference
        try:
            logger.info(f"predict_single: calling forward with {len(full_questions)} questions")
            results = self.forward([image_path] * len(full_questions), full_questions)
            logger.info(f"predict_single: forward completed successfully")
            logger.info(f"predict_single: results keys={list(results.keys())}")
            
            # Find the predicted answer
            predicted_idx = results["pred"][0]
            confidence = results["confidence"][0]
            
            logger.info(f"predict_single: predicted_idx={predicted_idx}, confidence={confidence}")
            
            return {
                "predicted_answer": answer_options[predicted_idx],
                "predicted_idx": predicted_idx,
                "confidence": confidence,
                "all_probs": results["probs"][0]
            }
        except Exception as e:
            logger.error(f"Error in predict_single: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a default result
            return {
                "predicted_answer": answer_options[0],
                "predicted_idx": 0,
                "confidence": 0.0,
                "all_probs": [1.0/len(answer_options)] * len(answer_options)
            }
    
    def evaluate_dataset(self, data: list[dict]) -> dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            data: List of data samples with image_path, question, answer_options, correct_answer_idx
            
        Returns:
            Dictionary with evaluation results
        """
        correct_predictions = 0
        total_predictions = 0
        question_type_accuracies = {}
        confidences = []
        
        for sample in data:
            image_path = sample['image_path']
            question = sample['question']
            answer_options = sample['answer_options']
            correct_idx = sample['correct_answer_idx']
            question_type = sample.get('question_type', 'unknown')
            
            # Make prediction
            result = self.predict_single(image_path, question, answer_options)
            
            # Check if prediction is correct
            is_correct = result['predicted_idx'] == correct_idx
            if is_correct:
                correct_predictions += 1
            
            total_predictions += 1
            confidences.append(result['confidence'])
            
            # Track per-question-type accuracy
            if question_type not in question_type_accuracies:
                question_type_accuracies[question_type] = {'correct': 0, 'total': 0}
            
            question_type_accuracies[question_type]['total'] += 1
            if is_correct:
                question_type_accuracies[question_type]['correct'] += 1
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate per-question-type accuracies
        for q_type in question_type_accuracies:
            q_type_data = question_type_accuracies[q_type]
            question_type_accuracies[q_type] = q_type_data['correct'] / q_type_data['total']
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'question_type_accuracies': question_type_accuracies,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'confidences': confidences
        }

def main():
    """Example usage of BioMedCLIPLoRA."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BioMedCLIP LoRA inference')
    parser.add_argument('--adapter-path', type=str, required=True,
                       help='Path to LoRA adapter directory')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to evaluation data JSON file')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Initialize model
    model = BioMedCLIPLoRA()
    
    # Load LoRA adapter
    model.load_lora_adapter(args.adapter_path)
    
    # Load evaluation data
    import json
    with open(args.data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Run evaluation
    results = model.evaluate_dataset(eval_data)
    
    # Print results
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Average Confidence: {results['average_confidence']:.4f}")
    print("\nPer-question-type accuracies:")
    for q_type, acc in results['question_type_accuracies'].items():
        print(f"  {q_type}: {acc:.4f}")
    
    # Save results if output path specified
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    main() 