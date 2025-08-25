#!/bin/bash

echo "🚀 Setting up Gemma3 270M training environment for MacBook M4..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.9 or 3.10 first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📱 Python version: $PYTHON_VERSION"

# Check if Xcode Command Line Tools are installed
if ! xcode-select -p &> /dev/null; then
    echo "🔧 Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "⚠️  Please complete the Xcode installation in the popup window, then press Enter to continue..."
    read
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing it..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip

# Install all dependencies from requirements.txt
echo "📦 Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies from requirements.txt"
        echo "💡 Trying individual package installation..."
        
        # Fallback: install core packages individually
        echo "📦 Installing MLX and core dependencies..."
        python3 -m pip install mlx
        python3 -m pip install torch torchvision torchaudio
        python3 -m pip install transformers datasets tokenizers
        python3 -m pip install huggingface_hub
        python3 -m pip install mteb
        python3 -m pip install numpy scipy scikit-learn
        python3 -m pip install tqdm matplotlib seaborn
        python3 -m pip install psutil
    fi
else
    echo "⚠️  requirements.txt not found, installing core packages individually..."
    
    # Install MLX and core dependencies
    echo "📦 Installing MLX and core dependencies..."
    python3 -m pip install mlx
    python3 -m pip install torch torchvision torchaudio
    python3 -m pip install transformers datasets tokenizers
    python3 -m pip install huggingface_hub
    python3 -m pip install mteb
    python3 -m pip install numpy scipy scikit-learn
    python3 -m pip install tqdm matplotlib seaborn
    python3 -m pip install psutil
fi

# Verify MLX installation
echo "✅ Verifying MLX installation..."
python3 -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ MLX installed successfully"
else
    echo "❌ MLX installation failed"
fi

# Verify PyTorch installation
echo "✅ Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ PyTorch installed successfully"
else
    echo "❌ PyTorch installation failed"
fi

# Verify Transformers installation
echo "✅ Verifying Transformers installation..."
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Transformers installed successfully"
else
    echo "❌ Transformers installation failed"
fi

# Verify psutil installation
echo "✅ Verifying psutil installation..."
python3 -c "import psutil; print(f'psutil version: {psutil.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ psutil installed successfully"
else
    echo "❌ psutil installation failed"
fi

# Create project structure
echo "📁 Creating project structure..."
mkdir -p models
mkdir -p data
mkdir -p configs
mkdir -p scripts
mkdir -p logs

# Test basic functionality
echo "🧪 Testing basic functionality..."
python3 -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path[0]}')
try:
    import mlx.core as mx
    print(f'✅ MLX working: {mx.__version__}')
except ImportError as e:
    print(f'❌ MLX import failed: {e}')
try:
    import torch
    print(f'✅ PyTorch working: {torch.__version__}')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
try:
    import transformers
    print(f'✅ Transformers working: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')
try:
    import psutil
    print(f'✅ psutil working: {psutil.__version__}')
except ImportError as e:
    print(f'❌ psutil import failed: {e}')
"

echo ""
echo "🎉 Setup complete!"
echo "=================================================="
echo "📚 Next steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Download Gemma3 270M model: python3 scripts/download_model.py"
echo "   3. Test the model: python3 scripts/test_model.py"
echo "   4. Chat with the model: python3 scripts/chat_with_model.py"
echo "   5. Start training: python3 scripts/train.py"
echo ""
echo "💡 To activate the environment in new terminals:"
echo "   cd $(pwd)"
echo "   source venv/bin/activate"
echo "=================================================="
