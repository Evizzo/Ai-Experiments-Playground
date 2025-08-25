# 🚀 Quick Start Guide - Gemma3 270M Training

Get up and running with Gemma3 270M training on your MacBook M4 in under 10 minutes!

## ⚡ Super Quick Start (3 commands)

```bash
# 1. Setup everything automatically
./setup.sh

# 2. Activate the environment
source venv/bin/activate

# 3. Start training with sample data
python scripts/train.py
```

## 🎯 What You'll Get

- ✅ **MLX Framework** installed and optimized for M4
- ✅ **Gemma3 270M model** downloaded from Hugging Face
- ✅ **Sample training data** generated automatically
- ✅ **LoRA training** with InfoNCE loss
- ✅ **Performance monitoring** for your M4 chip
- ✅ **MTEB evaluation** ready to go

## 📱 Hardware Requirements

- **MacBook with M4 chip** (16GB+ RAM recommended)
- **20-30GB free storage** for model and dependencies
- **Latest macOS** (Sequoia 15.x preferred)

## 🔧 Step-by-Step Setup

### 1. Run the Setup Script
```bash
./setup.sh
```
This will:
- Install Python dependencies
- Set up virtual environment
- Install MLX framework
- Create project structure

### 2. Activate Environment
```bash
source venv/bin/activate
```

### 3. Download the Model
```bash
python scripts/download_model.py
```
Downloads ~5GB Gemma3 270M model from Hugging Face.

### 4. Generate Sample Data
```bash
python scripts/generate_sample_data.py --samples 1000
```
Creates training/validation/test data for testing.

### 5. Start Training
```bash
python scripts/train.py
```
Begins LoRA training with the sample data.

## 📊 Monitor Performance

In another terminal, monitor your M4's performance:

```bash
source venv/bin/activate
python scripts/monitor_performance.py --interval 10
```

## 🎛️ Customize Training

### Change Batch Size
Edit `configs/training_config.json`:
```json
{
  "training": {
    "batch_size": 64,  // Reduce if you have less RAM
    "gradient_accumulation_steps": 32  // Increase to compensate
  }
}
```

### Use Your Own Data
```bash
python scripts/train.py --data-path /path/to/your/data.jsonl
```

Data should be in JSONL format:
```json
{"text": "Your text here", "label": 1}
{"text": "Another example", "label": 0}
```

## 🚨 Troubleshooting

### Setup Fails
```bash
# Check Python version
python3 --version  # Should be 3.9 or 3.10

# Install Xcode tools
xcode-select --install

# Retry setup
./setup.sh
```

### Out of Memory
```bash
# Edit config to reduce memory usage
# configs/training_config.json
{
  "training": {
    "batch_size": 64,  # Reduce from 128
    "max_seq_length": 1024  # Reduce from 2048
  }
}
```

### Model Download Fails
```bash
# Login to Hugging Face
huggingface-cli login

# Retry download
python scripts/download_model.py
```

## 📈 Expected Results

With M4 chip, you should see:
- **Training Speed**: 4000+ tokens/second
- **Memory Usage**: 12-14GB during training
- **Training Time**: 2-4 hours for 50k steps

## 🔍 What's Happening

1. **LoRA Training**: Only trains small adapter layers (efficient)
2. **InfoNCE Loss**: Contrastive learning for better embeddings
3. **Gradient Accumulation**: Simulates larger batch sizes
4. **MLX Optimization**: Apple Silicon-specific performance tuning

## 📚 Next Steps

After successful training:
1. **Evaluate your model**: `python scripts/evaluate.py --model-path ./outputs/checkpoint-5000`
2. **Use for inference**: Load the trained LoRA weights
3. **Fine-tune further**: Adjust hyperparameters based on results
4. **Scale up**: Increase data size and training steps

## 🆘 Need Help?

- **Check logs**: Look in `./logs/` directory
- **Review config**: Verify `configs/training_config.json`
- **Monitor performance**: Use the performance monitoring script
- **Common issues**: See the main README.md for detailed troubleshooting

---

**🎉 You're all set!** Your MacBook M4 is now ready to train Gemma3 270M models locally with MLX optimization.
