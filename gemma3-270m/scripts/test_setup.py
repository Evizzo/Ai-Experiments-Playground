#!/usr/bin/env python3
"""
Test script to verify Gemma3 270M training environment setup.
Run this after setup to ensure everything is working correctly.
"""

import sys
import importlib
from pathlib import Path

def testImport(moduleName: str, packageName: str = None):
    """Test if a module can be imported."""
    try:
        if packageName:
            module = importlib.import_module(moduleName, package=packageName)
        else:
            module = importlib.import_module(moduleName)
        print(f"✅ {moduleName}: {getattr(module, '__version__', 'imported successfully')}")
        return True
    except ImportError as e:
        print(f"❌ {moduleName}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {moduleName}: {e}")
        return False

def testMLX():
    """Test MLX framework functionality."""
    print("\n🔍 Testing MLX Framework...")
    
    try:
        import mlx.core as mx
        print(f"✅ MLX version: {mx.__version__}")
        
        # Test basic operations
        x = mx.array([1, 2, 3, 4])
        y = mx.array([5, 6, 7, 8])
        z = x + y
        print(f"✅ Basic MLX operations: {z}")
        
        # Test device info
        deviceInfo = mx.device_info()
        print(f"✅ Device info: {deviceInfo}")
        
        return True
    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        return False

def testHardware():
    """Test hardware detection and capabilities."""
    print("\n🔍 Testing Hardware Detection...")
    
    try:
        import psutil
        
        # CPU info
        cpuCount = psutil.cpu_count()
        cpuPercent = psutil.cpu_percent(interval=1)
        print(f"✅ CPU: {cpuCount} cores, {cpuPercent:.1f}% usage")
        
        # Memory info
        memory = psutil.virtual_memory()
        memoryGB = memory.total / (1024**3)
        print(f"✅ Memory: {memoryGB:.1f} GB total, {memory.percent:.1f}% used")
        
        # Check if we have sufficient memory
        if memoryGB < 16:
            print("⚠️  Warning: Less than 16GB RAM detected. Training may be limited.")
        else:
            print("✅ Sufficient RAM for training")
        
        return True
    except ImportError:
        print("⚠️  psutil not available, cannot check hardware")
        return False
    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        return False

def testProjectStructure():
    """Test if project structure is properly set up."""
    print("\n🔍 Testing Project Structure...")
    
    requiredDirs = [
        "models",
        "data", 
        "configs",
        "scripts",
        "outputs",
        "logs"
    ]
    
    requiredFiles = [
        "configs/training_config.json",
        "scripts/train.py",
        "scripts/download_model.py",
        "scripts/evaluate.py",
        "requirements.txt",
        "README.md"
    ]
    
    allGood = True
    
    # Check directories
    for dirName in requiredDirs:
        dirPath = Path(dirName)
        if dirPath.exists():
            print(f"✅ Directory: {dirName}")
        else:
            print(f"❌ Missing directory: {dirName}")
            allGood = False
    
    # Check files
    for fileName in requiredFiles:
        filePath = Path(fileName)
        if filePath.exists():
            print(f"✅ File: {fileName}")
        else:
            print(f"❌ Missing file: {fileName}")
            allGood = False
    
    return allGood

def testConfiguration():
    """Test if configuration files are valid."""
    print("\n🔍 Testing Configuration...")
    
    try:
        import json
        
        # Test training config
        configPath = Path("configs/training_config.json")
        if configPath.exists():
            with open(configPath, 'r') as f:
                config = json.load(f)
            
            requiredKeys = ["model", "training", "lora", "data", "evaluation"]
            for key in requiredKeys:
                if key in config:
                    print(f"✅ Config key: {key}")
                else:
                    print(f"❌ Missing config key: {key}")
                    return False
            
            print("✅ Training configuration is valid")
            return True
        else:
            print("❌ Training configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def testDataGeneration():
    """Test data generation functionality."""
    print("\n🔍 Testing Data Generation...")
    
    try:
        # Import the data generation module
        sys.path.append(str(Path(__file__).parent))
        from generate_sample_data import generateSampleData
        
        # Generate a small sample
        result = generateSampleData(10, "./test_data")
        
        if result and Path("./test_data/train.jsonl").exists():
            print("✅ Data generation working")
            
            # Clean up test data
            import shutil
            shutil.rmtree("./test_data")
            print("✅ Test data cleaned up")
            
            return True
        else:
            print("❌ Data generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def runAllTests():
    """Run all tests and provide summary."""
    print("🚀 Gemma3 270M Environment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Core Dependencies", testCoreDependencies),
        ("MLX Framework", testMLX),
        ("Hardware Detection", testHardware),
        ("Project Structure", testProjectStructure),
        ("Configuration", testConfiguration),
        ("Data Generation", testDataGeneration)
    ]
    
    results = {}
    
    for testName, testFunc in tests:
        try:
            results[testName] = testFunc()
        except Exception as e:
            print(f"❌ {testName} test crashed: {e}")
            results[testName] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for testName, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {testName}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready for training.")
        print("\n📚 Next steps:")
        print("   1. Download the model: python scripts/download_model.py")
        print("   2. Generate sample data: python scripts/generate_sample_data.py")
        print("   3. Start training: python scripts/train.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above and fix them.")
        print("\n🔧 Common fixes:")
        print("   - Run setup again: ./setup.sh")
        print("   - Check Python version: python3 --version")
        print("   - Install Xcode tools: xcode-select --install")
    
    return passed == total

def testCoreDependencies():
    """Test core Python dependencies."""
    print("\n🔍 Testing Core Dependencies...")
    
    dependencies = [
        "torch",
        "transformers", 
        "datasets",
        "huggingface_hub",
        "mteb",
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "seaborn"
    ]
    
    allGood = True
    for dep in dependencies:
        if not testImport(dep):
            allGood = False
    
    return allGood

if __name__ == "__main__":
    success = runAllTests()
    sys.exit(0 if success else 1)
