#!/usr/bin/env python3
"""
Test script to verify BiomedCLIP LoRA installation and basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        print("✓ PEFT imported successfully")
    except ImportError as e:
        print(f"✗ PEFT import failed: {e}")
        print("Please install PEFT: pip install peft")
        return False
    
    try:
        from evlm.models.openCLIP_models.biomedclip_lora import BioMedCLIPLoRA
        print("✓ BiomedCLIPLoRA imported successfully")
    except ImportError as e:
        print(f"✗ BiomedCLIPLoRA import failed: {e}")
        return False
    
    return True

def test_model_initialization():
    """Test that the LoRA model can be initialized."""
    print("\nTesting model initialization...")
    
    try:
        from evlm.models.openCLIP_models.biomedclip_lora import BioMedCLIPLoRA
        
        # Initialize model with LoRA
        model = BioMedCLIPLoRA(
            eval_mode=True,
            context_length=256,
            lora_config={
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                'lora_dropout': 0.1,
                'bias': 'none',
                'task_type': 'FEATURE_EXTRACTION'
            },
            verbose=False  # Reduce output for testing
        )
        
        print("✓ BiomedCLIPLoRA initialized successfully")
        
        # Test parameter counting
        trainable_params, total_params = model.get_trainable_parameters()
        print(f"✓ Parameter counting: {trainable_params:,} trainable, {total_params:,} total")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic model functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from evlm.models.openCLIP_models.biomedclip_lora import BioMedCLIPLoRA
        
        # Initialize model
        model = BioMedCLIPLoRA(eval_mode=True, verbose=False)
        
        # Test that model has required attributes
        assert hasattr(model, 'model'), "Model should have 'model' attribute"
        assert hasattr(model, 'tokenizer'), "Model should have 'tokenizer' attribute"
        assert hasattr(model, 'preprocess'), "Model should have 'preprocess' attribute"
        assert hasattr(model, 'lora_config'), "Model should have 'lora_config' attribute"
        
        print("✓ Model has all required attributes")
        
        # Test LoRA configuration
        expected_keys = ['r', 'lora_alpha', 'target_modules', 'lora_dropout', 'bias', 'task_type']
        for key in expected_keys:
            assert key in model.lora_config, f"LoRA config should have '{key}'"
        
        print("✓ LoRA configuration is complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== BiomedCLIP LoRA Installation Test ===\n")
    
    tests = [
        test_imports,
        test_model_initialization,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! BiomedCLIP LoRA is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your training data in JSONL format")
        print("2. Run: python src/evlm/training/train_biomedclip_lora.py --help")
        print("3. Check the documentation: docs/BIOMEDCLIP_LORA_FINETUNING.md")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements_lora.txt")
        print("2. Check that you're in the correct directory")
        print("3. Verify that the src/ directory structure is correct")

if __name__ == "__main__":
    main() 