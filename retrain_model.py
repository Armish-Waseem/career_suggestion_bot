#!/usr/bin/env python3
"""
Script to retrain the ML model with updated dataset
"""

from ml_model import CareerAdviceMLModel
import os

def main():
    print("🔄 Retraining AI/ML Career Advice Model...")
    print("=" * 50)
    
    # Initialize model
    model = CareerAdviceMLModel()
    
    # Train with updated dataset
    print("📊 Loading updated dataset...")
    model.train_model("dataset/career_dataset.json")
    
    print("\n🧪 Testing the retrained model...")
    
    # Test cases
    test_cases = [
        "I love programming and computers",
        "mujhe medicine mein interest hai",
        "I want to work in business",
        "mujhe art aur design mein interest hai",
        "hello",
        "salam"
    ]
    
    for test in test_cases:
        language = model.detect_language(test)
        intent, confidence = model.predict_intent(test)
        response = model.generate_response(test, language)
        
        print(f"\nInput: {test}")
        print(f"Language: {language}")
        print(f"Intent: {intent} (confidence: {confidence:.3f})")
        print(f"Response: {response}")
        print("-" * 40)
    
    print("\n✅ Model retraining completed!")
    print("🚀 You can now start the API with: python app_ml.py")

if __name__ == "__main__":
    main()
