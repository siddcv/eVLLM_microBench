#!/usr/bin/env python3
"""
Test script to validate the fine-tuning implementation.
"""

import sys
from pathlib import Path
import json
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing data loading...")
    
    # Test JSONL loading
    test_data = [
        {"image_id": "test1", "image_path": "test1.png", "text": "test text 1", "is_positive": True},
        {"image_id": "test2", "image_path": "test2.png", "text": "test text 2", "is_positive": False}
    ]
    
    # Create test file
    test_file = "finetuning/test_data.jsonl"
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # Test loading
    import sys
    sys.path.append(str(Path(__file__).parent))
    from scripts.prepare_data import load_jsonl
    loaded_data = load_jsonl(str(test_file))
    
    assert len(loaded_data) == 2, f"Expected 2 items, got {len(loaded_data)}"
    assert loaded_data[0]['image_id'] == "test1", "Data loading failed"
    
    print("✅ Data loading test passed!")
    
    # Cleanup
    Path(test_file).unlink()
    
    return True

def test_model_import():
    """Test model import functionality."""
    print("🧪 Testing model import...")
    
    try:
        from evlm.models.openCLIP_models.biomedclip import BioMedCLIP
        from evlm.models.openCLIP_models.biomedclip_finetuned import FineTunedBioMedCLIP
        print("✅ Model import test passed!")
        return True
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_constants_update():
    """Test that constants have been updated."""
    print("🧪 Testing constants update...")
    
    try:
        from evlm.inference.constants import CLIP_MODELS
        assert "FineTunedBioMedCLIP" in CLIP_MODELS, "FineTunedBioMedCLIP not found in CLIP_MODELS"
        print("✅ Constants update test passed!")
        return True
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False

def test_directory_structure():
    """Test that directory structure is correct."""
    print("🧪 Testing directory structure...")
    
    required_dirs = [
        "finetuning",
        "finetuning/data",
        "finetuning/checkpoints", 
        "finetuning/scripts",
        "finetuning/logs"
    ]
    
    required_files = [
        "finetuning/scripts/prepare_data.py",
        "finetuning/scripts/train_lora.py",
        "finetuning/run_finetuning.py",
        "finetuning/requirements.txt",
        "finetuning/README.md",
        "src/evlm/models/openCLIP_models/biomedclip_finetuned.py"
    ]
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Directory not found: {dir_path}")
            return False
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            return False
    
    print("✅ Directory structure test passed!")
    return True

def test_peft_availability():
    """Test PEFT library availability."""
    print("🧪 Testing PEFT availability...")
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        print("✅ PEFT library available!")
        return True
    except Exception as e:
        print(f"❌ PEFT library not available: {e}")
        return False

def test_pytorch_lightning_availability():
    """Test PyTorch Lightning availability."""
    print("🧪 Testing PyTorch Lightning availability...")
    
    try:
        import pytorch_lightning as pl
        print("✅ PyTorch Lightning available!")
        return True
    except ImportError:
        print("❌ PyTorch Lightning not available. Install with: pip install pytorch-lightning")
        return False

def main():
    """Run all tests."""
    print("🚀 Running fine-tuning implementation tests...\n")
    
    tests = [
        test_directory_structure,
        test_data_loading,
        test_model_import,
        test_constants_update,
        test_peft_availability,
        test_pytorch_lightning_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fine-tuning implementation is ready.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r finetuning/requirements.txt")
        print("2. Run fine-tuning: python finetuning/run_finetuning.py --input_data <your_data_file>")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    main() 