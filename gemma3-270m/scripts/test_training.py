#!/usr/bin/env python3
"""
Test script to verify training setup works correctly.
Runs a few training steps to check everything is working.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def testTrainingSetup():
    """Test the training setup with a few steps."""
    try:
        logger.info("🧪 Testing training setup...")
        
        # Import the trainer
        from train import GemmaLoRATrainer
        
        # Test with a small config
        config = {
            "model": {
                "name": "gemma3-270m",
                "path": "./models",
                "max_length": 1024,
                "vocab_size": 262144
            },
            "training": {
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "effective_batch_size": 4,
                "learning_rate": 1e-4,
                "warmup_steps": 10,
                "max_steps": 10,
                "save_steps": 5,
                "eval_steps": 5,
                "logging_steps": 1,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "lr_scheduler_type": "cosine",
                "num_train_epochs": 1
            },
            "lora": {
                "enabled": True,
                "rank": 4,
                "alpha": 8,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj"]
            },
            "data": {
                "train_file": "./data/train.jsonl",
                "validation_file": "./data/validation.jsonl",
                "test_file": "./data/test.jsonl",
                "text_column": "text",
                "label_column": "label",
                "max_seq_length": 512,
                "streaming": False,
                "shuffle_buffer_size": 100
            },
            "optimization": {
                "mixed_precision": "bf16",
                "gradient_checkpointing": False,
                "dataloader_num_workers": 1,
                "dataloader_pin_memory": False
            },
            "evaluation": {
                "metrics": ["accuracy"],
                "mteb_tasks": ["text-classification"],
                "eval_batch_size": 4
            },
            "output": {
                "output_dir": "./outputs/test",
                "save_total_limit": 1,
                "overwrite_output_dir": True,
                "push_to_hub": False
            },
            "logging": {
                "log_level": "info",
                "log_file": "./logs/test_training.log",
                "tensorboard_log_dir": "./logs/tensorboard"
            },
            "hardware": {
                "device": "auto",
                "max_memory": "8GB",
                "num_gpus": 1
            }
        }
        
        # Save test config
        testConfigPath = Path("./configs/test_config.json")
        testConfigPath.parent.mkdir(exist_ok=True)
        
        import json
        with open(testConfigPath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("✅ Test config created")
        
        # Initialize trainer
        trainer = GemmaLoRATrainer(str(testConfigPath))
        logger.info("✅ Trainer initialized successfully")
        
        # Test data loading
        logger.info(f"📊 Training data: {len(trainer.trainData)} examples")
        if trainer.valData:
            logger.info(f"📊 Validation data: {len(trainer.valData)} examples")
        
        # Test a few training steps
        logger.info("🎯 Testing training steps...")
        
        # Get a small batch
        if trainer.trainData:
            testBatch = trainer.trainData[:4]
            logger.info(f"📝 Test batch size: {len(testBatch)}")
            
            # Test training step
            loss = trainer.trainingStep(testBatch)
            logger.info(f"📊 Test training step loss: {loss:.4f}")
            
            # Test evaluation
            trainer.evaluate()
            
            # Test checkpoint saving
            trainer.saveCheckpoint(1)
            logger.info("✅ Checkpoint saved successfully")
        
        logger.info("🎉 All tests passed! Training setup is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Gemma3 270M Training Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("./models").exists():
        print("❌ Models directory not found. Please run setup first:")
        print("   cd gemma3-270m")
        print("   ./setup.sh")
        return False
    
    if not Path("./data/train.jsonl").exists():
        print("❌ Training data not found. Please generate data first:")
        print("   python scripts/generate_sample_data.py")
        return False
    
    # Run tests
    success = testTrainingSetup()
    
    if success:
        print("\n🎉 Training setup test PASSED!")
        print("✅ You can now run the full training:")
        print("   python scripts/train.py")
    else:
        print("\n❌ Training setup test FAILED!")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
