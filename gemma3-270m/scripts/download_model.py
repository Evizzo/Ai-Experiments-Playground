#!/usr/bin/env python3
"""
Download Gemma3 270M model for local training on MacBook M4.
This script downloads the model from Hugging Face and sets it up for MLX training.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import mlx.core as mx

def checkSystemRequirements():
    """Check if the system meets the requirements for training."""
    print("🔍 Checking system requirements...")
    
    # Check MLX installation
    try:
        mlx_version = mx.__version__
        print(f"✅ MLX version: {mlx_version}")
    except Exception as e:
        print(f"❌ MLX not properly installed: {e}")
        return False
    
    # Check device info
    try:
        device_info = mx.device_info()
        print(f"✅ Device info: {device_info}")
    except Exception as e:
        print(f"⚠️  Could not get device info: {e}")
    
    # Check available memory (approximate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ Available RAM: {memory_gb:.1f} GB")
        if memory_gb < 16:
            print("⚠️  Warning: Less than 16GB RAM detected. Training may be limited.")
    except ImportError:
        print("⚠️  psutil not available, cannot check RAM")
    
    return True

def downloadGemmaModel(modelName="google/gemma-3-270m", outputDir="./models"):
    """Download the Gemma3 270M model from Hugging Face."""
    print(f"📥 Downloading {modelName} to {outputDir}...")
    
    # Create output directory
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the model
        snapshot_download(
            repo_id=modelName,
            local_dir=outputDir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt", "*.md"],
            ignore_patterns=["*.h5", "*.ckpt", "*.pt", "*.pth"],
            resume_download=True
        )
        print(f"✅ Model downloaded successfully to {outputDir}")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def verifyModelFiles(modelDir):
    """Verify that the downloaded model files are complete."""
    print("🔍 Verifying model files...")
    
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    # Check for model weights (either .bin or .safetensors)
    weight_files = list(Path(modelDir).glob("*.bin")) + list(Path(modelDir).glob("*.safetensors"))
    
    if not weight_files:
        print("❌ No model weight files found (.bin or .safetensors)")
        return False
    
    print(f"✅ Found {len(weight_files)} weight files")
    
    # Check required files
    for file in required_files:
        file_path = Path(modelDir) / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ Missing {file}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download Gemma3 270M model for MLX training")
    parser.add_argument("--model", default="google/gemma-3-270m", help="Model name on Hugging Face")
    parser.add_argument("--output", default="./models", help="Output directory for model")
    parser.add_argument("--skip-checks", action="store_true", help="Skip system requirement checks")
    
    args = parser.parse_args()
    
    print("🚀 Gemma3 270M Model Downloader for MacBook M4")
    print("=" * 50)
    
    # Check system requirements
    if not args.skip_checks:
        if not checkSystemRequirements():
            print("❌ System requirements not met. Please fix the issues above.")
            sys.exit(1)
    
    # Download the model
    if downloadGemmaModel(args.model, args.output):
        # Verify the download
        if verifyModelFiles(args.output):
            print("\n🎉 Model download and verification complete!")
            print(f"📁 Model location: {os.path.abspath(args.output)}")
            print("\n📚 Next steps:")
            print("   1. Prepare your training dataset")
            print("   2. Configure training parameters")
            print("   3. Start training with: python scripts/train.py")
        else:
            print("❌ Model verification failed. Please check the download.")
            sys.exit(1)
    else:
        print("❌ Model download failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
