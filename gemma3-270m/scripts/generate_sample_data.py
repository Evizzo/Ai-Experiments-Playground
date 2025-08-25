#!/usr/bin/env python3
"""
Generate sample training data for Gemma3 270M training.
This script creates synthetic data for testing the training pipeline.
"""

import json
import argparse
from pathlib import Path
import random

def generateSampleData(numSamples: int = 1000, outputDir: str = "./data"):
    """Generate sample training data for testing."""
    
    # Create output directory
    outputPath = Path(outputDir)
    outputPath.mkdir(parents=True, exist_ok=True)
    
    # Sample text templates for different categories
    positiveTexts = [
        "This product exceeded my expectations.",
        "I'm very satisfied with the quality.",
        "Excellent service and fast delivery.",
        "Highly recommend this to everyone.",
        "Outstanding performance and reliability.",
        "The best purchase I've made this year.",
        "Fantastic value for the money.",
        "Superior craftsmanship and attention to detail.",
        "Amazing results and great experience.",
        "Top-notch quality and professional service."
    ]
    
    negativeTexts = [
        "This was a complete waste of money.",
        "Terrible quality and poor service.",
        "I'm very disappointed with this purchase.",
        "Not worth the price at all.",
        "Poor performance and reliability issues.",
        "The worst product I've ever bought.",
        "Avoid this at all costs.",
        "Subpar quality and unprofessional service.",
        "Frustrating experience and poor results.",
        "Low-quality materials and bad design."
    ]
    
    neutralTexts = [
        "The product arrived on time.",
        "Standard quality for the price.",
        "It works as described.",
        "Average performance overall.",
        "The service was adequate.",
        "Nothing special, but functional.",
        "Meets basic expectations.",
        "Standard delivery and packaging.",
        "Typical results for this type of product.",
        "The experience was neither good nor bad."
    ]
    
    # Generate training data
    trainData = []
    validationData = []
    testData = []
    
    for i in range(numSamples):
        # Randomly select category and template
        category = random.choice([0, 1, 2])  # 0: negative, 1: positive, 2: neutral
        
        if category == 0:
            text = random.choice(negativeTexts)
            label = 0
        elif category == 1:
            text = random.choice(positiveTexts)
            label = 1
        else:
            text = random.choice(neutralTexts)
            label = 2
        
        # Add some variation to the text
        variations = [
            f"I think {text.lower()}",
            f"In my opinion, {text.lower()}",
            f"From my experience, {text.lower()}",
            f"Based on what I've seen, {text.lower()}",
            f"After using this, {text.lower()}",
            text,  # Keep some original
            f"Overall, {text.lower()}",
            f"To summarize, {text.lower()}"
        ]
        
        finalText = random.choice(variations)
        
        # Create data point
        dataPoint = {
            "text": finalText,
            "label": label,
            "category": ["negative", "positive", "neutral"][category],
            "id": f"sample_{i:04d}"
        }
        
        # Split data: 80% train, 10% validation, 10% test
        if i < int(0.8 * numSamples):
            trainData.append(dataPoint)
        elif i < int(0.9 * numSamples):
            validationData.append(dataPoint)
        else:
            testData.append(dataPoint)
    
    # Save data files
    trainFile = outputPath / "train.jsonl"
    validationFile = outputPath / "validation.jsonl"
    testFile = outputPath / "test.jsonl"
    
    with open(trainFile, 'w') as f:
        for item in trainData:
            f.write(json.dumps(item) + '\n')
    
    with open(validationFile, 'w') as f:
        for item in validationData:
            f.write(json.dumps(item) + '\n')
    
    with open(testFile, 'w') as f:
        for item in testData:
            f.write(json.dumps(item) + '\n')
    
    # Print statistics
    print(f"✅ Generated sample data:")
    print(f"   📁 Output directory: {outputPath.absolute()}")
    print(f"   📊 Training samples: {len(trainData)}")
    print(f"   🔍 Validation samples: {len(validationData)}")
    print(f"   🧪 Test samples: {len(testData)}")
    print(f"   📝 Total samples: {len(trainData) + len(validationData) + len(testData)}")
    
    # Show sample data
    print(f"\n📋 Sample training data:")
    for i, item in enumerate(trainData[:3]):
        print(f"   {i+1}. [{item['category']}] {item['text']}")
    
    return {
        'train': trainFile,
        'validation': validationFile,
        'test': testFile,
        'counts': {
            'train': len(trainData),
            'validation': len(validationData),
            'test': len(testData)
        }
    }

def generateDomainSpecificData(domain: str, numSamples: int = 500, outputDir: str = "./data"):
    """Generate domain-specific training data."""
    
    domains = {
        "reviews": {
            "positive": [
                "This restaurant serves delicious food.",
                "The movie was absolutely fantastic.",
                "Great book, highly recommend reading it.",
                "Amazing hotel with excellent service.",
                "Outstanding customer support team."
            ],
            "negative": [
                "The food was cold and tasteless.",
                "Boring movie, waste of time.",
                "Poorly written book, don't bother.",
                "Dirty hotel room, terrible experience.",
                "Useless customer support, no help at all."
            ]
        },
        "technical": {
            "positive": [
                "The API is well-documented and easy to use.",
                "Excellent performance optimization.",
                "Clean and maintainable code structure.",
                "Robust error handling and logging.",
                "Great integration with other systems."
            ],
            "negative": [
                "Poor API documentation, hard to understand.",
                "Performance issues and slow response times.",
                "Messy code with no clear structure.",
                "Weak error handling, crashes frequently.",
                "Difficult integration, lots of bugs."
            ]
        },
        "academic": {
            "positive": [
                "The research methodology is sound.",
                "Clear and logical argument structure.",
                "Comprehensive literature review.",
                "Strong empirical evidence provided.",
                "Well-organized and readable paper."
            ],
            "negative": [
                "Flawed research methodology.",
                "Confusing and unclear arguments.",
                "Incomplete literature review.",
                "Weak empirical evidence.",
                "Poorly organized and hard to follow."
            ]
        }
    }
    
    if domain not in domains:
        print(f"❌ Unknown domain: {domain}")
        print(f"Available domains: {list(domains.keys())}")
        return None
    
    # Create output directory
    outputPath = Path(outputDir) / domain
    outputPath.mkdir(parents=True, exist_ok=True)
    
    domainData = domains[domain]
    trainData = []
    
    for i in range(numSamples):
        # Randomly select sentiment and template
        sentiment = random.choice([0, 1])  # 0: negative, 1: positive
        templates = domainData["positive"] if sentiment == 1 else domainData["negative"]
        text = random.choice(templates)
        
        # Add variations
        variations = [
            text,
            f"I found that {text.lower()}",
            f"Based on my analysis, {text.lower()}",
            f"In conclusion, {text.lower()}",
            f"The results show that {text.lower()}"
        ]
        
        finalText = random.choice(variations)
        
        dataPoint = {
            "text": finalText,
            "label": sentiment,
            "domain": domain,
            "id": f"{domain}_{i:04d}"
        }
        
        trainData.append(dataPoint)
    
    # Save data
    outputFile = outputPath / "train.jsonl"
    with open(outputFile, 'w') as f:
        for item in trainData:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Generated {domain}-specific data:")
    print(f"   📁 Output file: {outputFile.absolute()}")
    print(f"   📊 Samples: {len(trainData)}")
    
    return outputFile

def main():
    parser = argparse.ArgumentParser(description="Generate sample training data for Gemma3 270M")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--domain", choices=["reviews", "technical", "academic"], help="Domain-specific data")
    
    args = parser.parse_args()
    
    print("🚀 Gemma3 270M Sample Data Generator")
    print("=" * 50)
    
    if args.domain:
        # Generate domain-specific data
        generateDomainSpecificData(args.domain, args.samples, args.output)
    else:
        # Generate general sample data
        generateSampleData(args.samples, args.output)
    
    print(f"\n📚 Next steps:")
    print(f"   1. Review the generated data in {args.output}")
    print(f"   2. Start training: python scripts/train.py")
    print(f"   3. Or customize the data generation for your specific use case")

if __name__ == "__main__":
    main()
