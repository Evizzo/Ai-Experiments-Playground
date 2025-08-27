#!/usr/bin/env python3
"""
Generate sample training data for Gemma3 270M fine-tuning.
Creates realistic text classification datasets for training.
"""

import json
import random
from pathlib import Path
import argparse

def generateProductReviews(numSamples: int = 1000) -> list:
    """Generate realistic product review data."""
    
    positivePhrases = [
        "excellent quality", "great value", "highly recommend", "exceeded expectations",
        "fantastic product", "amazing results", "top-notch", "outstanding performance",
        "superior craftsmanship", "premium quality", "excellent service", "fast delivery",
        "easy to use", "durable construction", "beautiful design", "perfect fit",
        "great customer service", "worth every penny", "excellent investment", "love it"
    ]
    
    negativePhrases = [
        "poor quality", "not worth the money", "disappointing", "cheap materials",
        "broke easily", "difficult to use", "poor design", "slow delivery",
        "bad customer service", "overpriced", "defective", "missing parts",
        "wrong size", "poor fit", "low quality", "unreliable", "frustrating experience",
        "waste of money", "avoid this", "terrible product"
    ]
    
    neutralPhrases = [
        "average quality", "meets expectations", "standard product", "adequate performance",
        "reasonable price", "basic functionality", "nothing special", "gets the job done",
        "typical results", "standard delivery", "average service", "functional design",
        "basic features", "standard materials", "ordinary quality", "satisfactory",
        "mediocre performance", "average experience", "standard quality", "acceptable"
    ]
    
    templates = [
        "After using this product, I found it to be {phrase}.",
        "Based on my experience, this item is {phrase}.",
        "I think this product is {phrase}.",
        "Overall, I would say it's {phrase}.",
        "To summarize, this product is {phrase}.",
        "From what I've seen, it's {phrase}.",
        "In my opinion, this is {phrase}.",
        "After testing this, I can confirm it's {phrase}.",
        "Based on the results, it's {phrase}.",
        "After careful evaluation, I found it to be {phrase}."
    ]
    
    data = []
    
    for i in range(numSamples):
        # Determine sentiment (0=negative, 1=positive, 2=neutral)
        sentiment = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        
        if sentiment == 0:  # Negative
            phrase = random.choice(negativePhrases)
            category = "negative"
        elif sentiment == 1:  # Positive
            phrase = random.choice(positivePhrases)
            category = "positive"
        else:  # Neutral
            phrase = random.choice(neutralPhrases)
            category = "neutral"
        
        template = random.choice(templates)
        text = template.format(phrase=phrase)
        
        # Add some variety with additional context
        if random.random() < 0.3:
            context = random.choice([
                " The packaging was also good.",
                " Delivery was on time.",
                " Customer service was helpful.",
                " The instructions were clear.",
                " It arrived in perfect condition."
            ])
            text += context
        
        data.append({
            "text": text,
            "label": sentiment,
            "category": category,
            "id": f"sample_{i:04d}"
        })
    
    return data

def generateSentimentData(numSamples: int = 1000) -> list:
    """Generate sentiment analysis data."""
    
    positiveSentences = [
        "I absolutely love this new feature!",
        "This is the best solution I've found.",
        "Amazing results with minimal effort.",
        "Highly recommend this approach.",
        "Exceeded all my expectations.",
        "Fantastic performance and reliability.",
        "Outstanding quality and service.",
        "This product is a game-changer.",
        "Incredible value for the price.",
        "Perfect for my needs."
    ]
    
    negativeSentences = [
        "This is the worst experience ever.",
        "Complete waste of time and money.",
        "Terrible quality and poor service.",
        "I'm extremely disappointed.",
        "Avoid this at all costs.",
        "Frustrating and unreliable.",
        "Poor design and functionality.",
        "Not worth the investment.",
        "Subpar performance overall.",
        "I regret this purchase."
    ]
    
    neutralSentences = [
        "It works as expected.",
        "Average performance overall.",
        "Meets basic requirements.",
        "Standard quality for the price.",
        "Nothing special, but functional.",
        "Typical results for this type.",
        "Adequate for basic needs.",
        "Standard delivery and service.",
        "Reasonable quality and price.",
        "Gets the job done."
    ]
    
    data = []
    
    for i in range(numSamples):
        sentiment = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        
        if sentiment == 0:
            text = random.choice(negativeSentences)
            category = "negative"
        elif sentiment == 1:
            text = random.choice(positiveSentences)
            category = "positive"
        else:
            text = random.choice(neutralSentences)
            category = "neutral"
        
        data.append({
            "text": text,
            "label": sentiment,
            "category": category,
            "id": f"sentiment_{i:04d}"
        })
    
    return data

def generateTextClassificationData(numSamples: int = 1000) -> list:
    """Generate text classification data for different topics."""
    
    topics = {
        "technology": [
            "artificial intelligence", "machine learning", "data science", "software development",
            "cloud computing", "cybersecurity", "blockchain", "internet of things",
            "virtual reality", "augmented reality", "robotics", "automation"
        ],
        "health": [
            "nutrition", "exercise", "mental health", "wellness", "fitness",
            "diet", "meditation", "sleep", "stress management", "workout",
            "healthy eating", "physical therapy"
        ],
        "business": [
            "marketing", "sales", "finance", "entrepreneurship", "management",
            "strategy", "leadership", "innovation", "customer service", "operations",
            "human resources", "project management"
        ],
        "education": [
            "online learning", "distance education", "skill development", "training",
            "academic research", "curriculum design", "student assessment", "teaching methods",
            "educational technology", "lifelong learning", "professional development"
        ]
    }
    
    data = []
    topicLabels = list(topics.keys())
    
    for i in range(numSamples):
        topic = random.choice(topicLabels)
        topicWords = topics[topic]
        
        # Create a sentence with topic-specific vocabulary
        sentence = f"This {random.choice(topicWords)} approach provides excellent results for {random.choice(topicWords)} applications."
        
        data.append({
            "text": sentence,
            "label": topicLabels.index(topic),
            "category": topic,
            "id": f"topic_{i:04d}"
        })
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate sample training data for Gemma3 270M")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--type", choices=["reviews", "sentiment", "topics", "all"], 
                       default="all", help="Type of data to generate")
    parser.add_argument("--output-dir", default="./data", help="Output directory for data files")
    
    args = parser.parse_args()
    
    # Create output directory
    outputDir = Path(args.output_dir)
    outputDir.mkdir(exist_ok=True)
    
    print(f"🚀 Generating {args.samples} samples of {args.type} data...")
    
    if args.type == "reviews" or args.type == "all":
        reviews = generateProductReviews(args.samples // 3)
        with open(outputDir / "train_reviews.jsonl", 'w') as f:
            for item in reviews:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Generated {len(reviews)} product review samples")
    
    if args.type == "sentiment" or args.type == "all":
        sentiment = generateSentimentData(args.samples // 3)
        with open(outputDir / "train_sentiment.jsonl", 'w') as f:
            for item in sentiment:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Generated {len(sentiment)} sentiment analysis samples")
    
    if args.type == "topics" or args.type == "all":
        topics = generateTextClassificationData(args.samples // 3)
        with open(outputDir / "train_topics.jsonl", 'w') as f:
            for item in topics:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Generated {len(topics)} topic classification samples")
    
    # Combine all data for main training file
    if args.type == "all":
        allData = reviews + sentiment + topics
        random.shuffle(allData)
        
        # Split into train/validation/test
        trainSize = int(len(allData) * 0.8)
        valSize = int(len(allData) * 0.1)
        
        trainData = allData[:trainSize]
        valData = allData[trainSize:trainSize + valSize]
        testData = allData[trainSize + valSize:]
        
        # Save main training file
        with open(outputDir / "train.jsonl", 'w') as f:
            for item in trainData:
                f.write(json.dumps(item) + '\n')
        
        # Save validation file
        with open(outputDir / "validation.jsonl", 'w') as f:
            for item in valData:
                f.write(json.dumps(item) + '\n')
        
        # Save test file
        with open(outputDir / "test.jsonl", 'w') as f:
            for item in testData:
                f.write(json.dumps(item) + '\n')
        
        print(f"✅ Combined dataset: {len(trainData)} train, {len(valData)} validation, {len(testData)} test")
        print(f"✅ Main training file: {outputDir / 'train.jsonl'}")
    
    print(f"🎉 Data generation complete! Files saved in {outputDir}")

if __name__ == "__main__":
    main()
