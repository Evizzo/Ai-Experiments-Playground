# Gemma3 270M Training Environment for MacBook M4

This project provides a complete setup for training Google's Gemma3 270M model locally on MacBook M4 using Jina AI's MLX framework. The setup is optimized for Apple Silicon and includes LoRA training, InfoNCE loss, and gradient accumulation.

## 🚀 Features

- **MLX Framework**: Optimized for Apple Silicon (M1/M2/M3/M4) performance
- **LoRA Training**: Efficient fine-tuning with Low-Rank Adaptation
- **InfoNCE Loss**: Contrastive learning for better embeddings
- **Gradient Accumulation**: Handle large effective batch sizes with limited memory
- **MTEB Integration**: Comprehensive evaluation using MTEB benchmarks
- **Streaming Data**: Efficient data loading for large datasets
- **Hardware Optimization**: Automatic detection and optimization for M4 chip

## 📋 Prerequisites

- **Hardware**: MacBook with M4 chip (16GB+ RAM recommended)
- **macOS**: Latest version (macOS Sequoia 15.x preferred)
- **Python**: 3.9 or 3.10
- **Xcode Command Line Tools**: For compilation support

## 🛠️ Quick Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone
cd gemma3279m

# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Download Model

```bash
python scripts/download_model.py
```

### 4. Start Training

```bash
python scripts/train.py
```

## 📁 Project Structure

```
gemma3279m/
├── setup.sh                 # Automated setup script
├── requirements.txt         # Python dependencies
├── configs/
│   └── training_config.json # Training configuration
├── scripts/
│   ├── download_model.py   # Model downloader
│   ├── train.py           # Main training script
│   └── evaluate.py        # Evaluation script
├── models/                 # Downloaded model files
├── data/                   # Training datasets
├── outputs/                # Training outputs
└── logs/                   # Training logs
```

## ⚙️ Configuration

The training configuration is in `configs/training_config.json`. Key settings:

- **Batch Size**: 128 (adjustable based on RAM)
- **Gradient Accumulation**: 16 steps
- **Effective Batch Size**: 2048
- **Learning Rate**: 1e-4
- **LoRA Rank**: 8
- **Max Sequence Length**: 2048

## 🎯 Training

### Basic Training

```bash
python scripts/train.py --config configs/training_config.json
```

### Custom Data Path

```bash
python scripts/train.py --data-path /path/to/your/data.jsonl
```

### Training Features

- **LoRA Adaptation**: Efficient parameter-efficient fine-tuning
- **InfoNCE Loss**: Contrastive learning for better representations
- **Gradient Accumulation**: Simulate larger batch sizes
- **Mixed Precision**: BF16 for memory efficiency
- **Checkpointing**: Automatic model saving and resuming

## 📊 Evaluation

### MTEB Evaluation

```bash
python scripts/evaluate.py --model-path ./outputs/checkpoint-5000
```

### Custom Tasks

```bash
python scripts/evaluate.py --model-path ./outputs/checkpoint-5000 --tasks sentence-similarity text-classification
```

## 🔧 Performance Optimization

### M4 Chip Optimization

- **Automatic Device Detection**: MLX automatically uses the best available device
- **Memory Management**: Optimized for 16GB+ RAM configurations
- **Batch Size Tuning**: Start with 128, increase if memory allows

### Memory Optimization

- **Gradient Checkpointing**: Enabled by default
- **Mixed Precision**: BF16 for reduced memory usage
- **Streaming Data**: Efficient data loading without loading entire dataset

## 📈 Expected Performance

Based on M3 Ultra benchmarks (4000 tokens/s), M4 should achieve:

- **Training Speed**: 4000+ tokens/second
- **Memory Usage**: 12-14GB during training
- **Training Time**: ~2-4 hours for 50k steps (depending on data size)

## 🐛 Troubleshooting

### Common Issues

1. **MLX Installation Failed**
   ```bash
   pip install --upgrade pip
   pip install mlx --force-reinstall
   ```

2. **Out of Memory**
   - Reduce `batch_size` in config
   - Increase `gradient_accumulation_steps`
   - Use smaller `max_seq_length`

3. **Model Download Failed**
   ```bash
   huggingface-cli login
   python scripts/download_model.py --skip-checks
   ```

### Performance Issues

- **Low Token Throughput**: Check batch size and sequence length
- **High Memory Usage**: Enable gradient checkpointing
- **Slow Training**: Verify MLX is using GPU acceleration

## 📚 Advanced Usage

### Custom LoRA Configuration

Edit `configs/training_config.json`:

```json
{
  "lora": {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj"]
  }
}
```

### Custom Evaluation

```python
from scripts.evaluate import GemmaEvaluator

evaluator = GemmaEvaluator("./models")
results = evaluator.runMTEBEvaluation(["sentence-similarity"])
```

### Data Format

Training data should be in JSONL format:

```json
{"text": "Your training text here", "label": 1}
{"text": "Another example", "label": 0}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Jina AI**: For the MLX framework and inspiration
- **Google**: For the Gemma3 270M model
- **Apple**: For the M4 chip and Metal Performance Shaders
- **MLX Team**: For the excellent MLX framework

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the MLX documentation for framework-specific questions

## 🔄 Updates

- **v1.0.0**: Initial release with basic training setup
- **v1.1.0**: Added MTEB evaluation and performance optimization
- **v1.2.0**: Enhanced LoRA configuration and memory management

---

**Note**: This setup is specifically optimized for MacBook M4 with Apple Silicon. Performance may vary on other hardware configurations.
