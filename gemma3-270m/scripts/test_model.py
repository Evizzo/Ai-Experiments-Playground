#!/usr/bin/env python3
"""
Simple script to test the Gemma3 270M model directly.
This lets you see the model working without training.
"""

import sys
from pathlib import Path

def testModelLoading():
    """Test if we can load the model and tokenizer."""
    print("🔍 Testing Model Loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import mlx.core as mx
        
        print("✅ Dependencies imported successfully")
        
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("./models")
        print("✅ Tokenizer loaded successfully")
        
        # Load model
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained("./models")
        print("✅ Model loaded successfully")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def testSimpleGeneration(tokenizer, model):
    """Test simple text generation with the model."""
    print("\n🔍 Testing Simple Text Generation...")
    
    try:
        # Simple prompt
        prompt = "Hello, my name is"
        print(f"📝 Prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"✅ Input tokenized: {inputs.input_ids.shape}")
        
        # Generate text
        print("🚀 Generating text...")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def testModelInfo(tokenizer, model):
    """Display information about the loaded model."""
    print("\n🔍 Model Information...")
    
    try:
        # Tokenizer info
        vocab_size = tokenizer.vocab_size
        print(f"📚 Vocabulary size: {vocab_size:,}")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Total parameters: {total_params:,}")
        
        # Check model type
        model_type = type(model).__name__
        print(f"🏗️  Model type: {model_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Could not get model info: {e}")
        return False

def main():
    """Main function to test the model."""
    print("🚀 Gemma3 270M Model Test")
    print("=" * 40)
    
    # Check if model exists
    model_path = Path("./models")
    if not model_path.exists():
        print("❌ Model directory not found. Please run download_model.py first.")
        return False
    
    # Test model loading
    tokenizer, model = testModelLoading()
    if tokenizer is None or model is None:
        print("❌ Failed to load model. Cannot proceed with testing.")
        return False
    
    # Test model information
    testModelInfo(tokenizer, model)
    
    # Test simple generation
    success = testSimpleGeneration(tokenizer, model)
    
    if success:
        print("\n🎉 Model test completed successfully!")
        print("✅ Your Gemma3 270M model is working correctly!")
        print("\n📚 You can now:")
        print("   1. Use the model for inference")
        print("   2. Start training with custom data")
        print("   3. Fine-tune on specific tasks")
    else:
        print("\n⚠️  Model test had some issues.")
        print("Check the errors above for more details.")
    
    return success

if __name__ == "__main__":
    try:
        import torch
        success = main()
        sys.exit(0 if success else 1)
    except ImportError:
        print("❌ PyTorch not available. Please install it first.")
        sys.exit(1)
