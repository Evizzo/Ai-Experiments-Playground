#!/usr/bin/env python3
"""
Interactive chat script for Gemma3 270M model.
Chat with the model without any RAM monitoring.
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def chatWithModel():
    """Interactive chat loop with the model."""
    print("🚀 Gemma3 270M Interactive Chat")
    print("=" * 50)
    print("💡 Type 'quit' to exit, 'clear' to clear conversation")
    print("=" * 50)
    
    try:
        logger.info("🔍 Starting dependency imports...")
        
        # Import transformers
        logger.info("📦 Importing transformers...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info("✅ Transformers imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import transformers: {e}")
            raise
        
        # Import torch
        logger.info("📦 Importing torch...")
        try:
            import torch
            logger.info(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        except ImportError as e:
            logger.error(f"❌ Failed to import torch: {e}")
            raise
        
        # Check model directory
        model_path = Path("./models")
        logger.info(f"🔍 Checking model directory: {model_path.absolute()}")
        
        if not model_path.exists():
            logger.error(f"❌ Model directory not found: {model_path}")
            print(f"❌ Model directory not found: {model_path}")
            print("💡 Please run 'python3 scripts/download_model.py' first")
            return
        
        logger.info(f"✅ Model directory found: {model_path}")
        
        # List model files
        model_files = list(model_path.glob("*"))
        logger.info(f"📁 Found {len(model_files)} files in model directory")
        for file in model_files[:5]:  # Show first 5 files
            logger.info(f"   - {file.name}")
        
        # Load tokenizer
        logger.info("📥 Loading tokenizer...")
        start_time = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained("./models")
            tokenizer_time = time.time() - start_time
            logger.info(f"✅ Tokenizer loaded successfully in {tokenizer_time:.2f}s")
            logger.info(f"📚 Vocabulary size: {tokenizer.vocab_size:,}")
        except Exception as e:
            logger.error(f"❌ Tokenizer loading failed: {e}")
            raise
        
        # Load model
        logger.info("📥 Loading model...")
        start_time = time.time()
        try:
            model = AutoModelForCausalLM.from_pretrained("./models")
            model_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {model_time:.2f}s")
            
            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"🧠 Total parameters: {total_params:,}")
            logger.info(f"🏗️  Model type: {type(model).__name__}")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
        
        conversation_history = []
        logger.info("🎉 Chat interface ready! You can start typing...")
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    logger.info("👋 User requested to quit")
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    logger.info("🧹 Conversation history cleared")
                    print("🧹 Conversation cleared!")
                    continue
                elif not user_input:
                    continue
                
                logger.info(f"📝 Processing user input: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'")
                
                # Add user message to history
                conversation_history.append(f"User: {user_input}")
                logger.info(f"📚 Conversation history now has {len(conversation_history)} messages")
                
                # Prepare context (last few messages for context)
                # Use a clearer format for the model
                if len(conversation_history) <= 2:
                    # For first message, just use the user input
                    context = f"User: {user_input}\nAssistant:"
                else:
                    # For subsequent messages, use last 2 exchanges
                    recent_messages = conversation_history[-4:]  # Last 4 messages (2 exchanges)
                    context = "\n".join(recent_messages) + "\nAssistant:"
                
                logger.info(f"🔗 Context length: {len(context)} characters")
                
                # Tokenize input
                logger.info("🔤 Tokenizing input...")
                try:
                    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024, padding=True)
                    # Fix attention mask issue
                    if inputs.get('attention_mask') is None:
                        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                    logger.info(f"✅ Input tokenized: {inputs.input_ids.shape}")
                except Exception as e:
                    logger.error(f"❌ Tokenization failed: {e}")
                    continue
                
                # Generate response
                print("🤖 Gemma is thinking...")
                start_time = time.time()
                
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs['attention_mask'],
                            max_length=inputs.input_ids.shape[1] + 300,  # Add 100 tokens
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1,  # Prevent repetition
                            no_repeat_ngram_size=3,   # Prevent 3-gram repetition
                            max_time=30.0  # 30 second timeout
                        )
                    
                    generation_time = time.time() - start_time
                    logger.info(f"✅ Generation completed in {generation_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"❌ Generation failed: {e}")
                    print(f"❌ Error generating response: {e}")
                    print("💡 Try asking a shorter question or restart the chat")
                    continue
                
                # Decode response
                try:
                    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    response = response.strip()
                    logger.info(f"✅ Response decoded: {len(response)} characters")
                except Exception as e:
                    logger.error(f"❌ Response decoding failed: {e}")
                    continue
                
                # Add response to history
                conversation_history.append(f"Gemma: {response}")
                
                # Display response
                print(f"🤖 Gemma: {response}")
                print(f"⏱️  Generated in {generation_time:.2f}s")
                
            except KeyboardInterrupt:
                logger.info("⚠️  Chat interrupted by user (Ctrl+C)")
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in chat loop: {e}")
                print(f"❌ Error: {e}")
                continue
                
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install transformers torch")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure the model is downloaded and accessible")
        print(f"💡 Check the logs above for more details")

if __name__ == "__main__":
    logger.info("🚀 Starting Gemma3 270M chat script...")
    chatWithModel()
    logger.info("🏁 Chat script finished")
