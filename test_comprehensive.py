#!/usr/bin/env python3
"""
Comprehensive test script for the improved AI/ML Career Advice Bot
"""

import requests
import json

def test_career_advice():
    """Test career advice functionality with new fields"""
    base_url = "http://localhost:5000"
    
    # Test cases for new career fields
    test_cases = [
        # Management & Leadership
        {
            "name": "English Management Interest",
            "input": "I want to work in management and lead teams",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Management Interest",
            "input": "mujhe management aur leadership mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Police & Army
        {
            "name": "English Police Interest",
            "input": "I want to work in police and help people",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Army Interest",
            "input": "mujhe army mein join karna hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Cooking & Culinary
        {
            "name": "English Cooking Interest",
            "input": "I love cooking and want to be a chef",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Cooking Interest",
            "input": "mujhe cooking aur food mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Philosophy
        {
            "name": "English Philosophy Interest",
            "input": "I love philosophy and deep thinking",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Philosophy Interest",
            "input": "mujhe philosophy aur ethics mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Arts & Creative
        {
            "name": "English Arts Interest",
            "input": "I want to work in arts and be creative",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Arts Interest",
            "input": "mujhe art aur painting mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Writing & Storytelling
        {
            "name": "English Writing Interest",
            "input": "I love writing and storytelling",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Writing Interest",
            "input": "mujhe writing aur books mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Film Making
        {
            "name": "English Film Interest",
            "input": "I want to make films and work in cinema",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Film Interest",
            "input": "mujhe movies aur directing mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Psychology
        {
            "name": "English Psychology Interest",
            "input": "I want to work in psychology and help people",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Psychology Interest",
            "input": "mujhe psychology aur human behavior mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Environmental Science
        {
            "name": "English Environment Interest",
            "input": "I want to work in environmental science",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Environment Interest",
            "input": "mujhe environment aur nature mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Architecture
        {
            "name": "English Architecture Interest",
            "input": "I want to work in architecture and building design",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Architecture Interest",
            "input": "mujhe architecture aur building mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Aviation
        {
            "name": "English Aviation Interest",
            "input": "I want to work in aviation and flying",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Aviation Interest",
            "input": "mujhe pilot aur flying mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Music
        {
            "name": "English Music Interest",
            "input": "I love music and want to be a musician",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Music Interest",
            "input": "mujhe music aur singing mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Fashion
        {
            "name": "English Fashion Interest",
            "input": "I want to work in fashion and clothing design",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Fashion Interest",
            "input": "mujhe fashion aur clothing mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Agriculture
        {
            "name": "English Agriculture Interest",
            "input": "I want to work in agriculture and farming",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Agriculture Interest",
            "input": "mujhe farming aur crops mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Real Estate
        {
            "name": "English Real Estate Interest",
            "input": "I want to work in real estate and property",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Real Estate Interest",
            "input": "mujhe property aur housing mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Social Work
        {
            "name": "English Social Work Interest",
            "input": "I want to work in social work and help people",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Social Work Interest",
            "input": "mujhe social work aur community mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Logistics
        {
            "name": "English Logistics Interest",
            "input": "I want to work in logistics and supply chain",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Logistics Interest",
            "input": "mujhe transportation aur logistics mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Gaming
        {
            "name": "English Gaming Interest",
            "input": "I love gaming and want to make video games",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Gaming Interest",
            "input": "mujhe gaming aur video games mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        },
        
        # Cybersecurity
        {
            "name": "English Cybersecurity Interest",
            "input": "I want to work in cybersecurity and IT security",
            "expected_language": "english",
            "expected_intent": "career_advice"
        },
        {
            "name": "Roman Urdu Cybersecurity Interest",
            "input": "mujhe cybersecurity aur hacking mein interest hai",
            "expected_language": "roman_urdu",
            "expected_intent": "career_advice"
        }
    ]
    
    print("🧪 Testing Comprehensive Career Advice Functionality")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{total_tests}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print("-" * 60)
        
        try:
            response = requests.post(
                f"{base_url}/career_advice",
                json={"user_input": test_case['input']},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check language detection
                detected_language = data.get('language_detected', '')
                expected_language = test_case['expected_language']
                language_correct = detected_language == expected_language
                
                # Check intent classification
                detected_intent = data.get('intent', '')
                expected_intent = test_case['expected_intent']
                intent_correct = detected_intent == expected_intent
                
                # Check if careers are recommended
                careers = data.get('career_recommendations', [])
                has_careers = len(careers) > 0
                
                # Check response language consistency
                response_text = data.get('message', '')
                response_language_consistent = True
                
                if expected_language == "english":
                    # Check if response contains Roman Urdu words
                    roman_urdu_words = ['aap', 'ke', 'mein', 'hun', 'kar', 'sakte', 'hain', 'aur', 'ki', 'ko']
                    if any(word in response_text.lower() for word in roman_urdu_words):
                        response_language_consistent = False
                elif expected_language == "roman_urdu":
                    # Check if response contains English words that shouldn't be there
                    english_words = ['based on', 'your interests', 'you should', 'consider these', 'career paths']
                    if any(phrase in response_text.lower() for phrase in english_words):
                        response_language_consistent = False
                
                print(f"✅ Language Detection: {detected_language} {'✓' if language_correct else '✗'}")
                print(f"✅ Intent Classification: {detected_intent} {'✓' if intent_correct else '✗'}")
                print(f"✅ Career Recommendations: {len(careers)} careers {'✓' if has_careers else '✗'}")
                print(f"✅ Language Consistency: {'✓' if response_language_consistent else '✗'}")
                print(f"✅ Response: {response_text}")
                
                if careers:
                    print(f"✅ Careers: {', '.join(careers[:3])}")
                
                # Overall test result
                if language_correct and intent_correct and has_careers and response_language_consistent:
                    print("🎉 Test PASSED")
                    passed_tests += 1
                else:
                    print("❌ Test FAILED")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the API is running on localhost:5000")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    return passed_tests, total_tests

def test_random_input():
    """Test random/garbage input handling"""
    base_url = "http://localhost:5000"
    
    random_inputs = [
        "hehe haha",
        "lol xd",
        "random text",
        "asdfghjkl",
        "123456789",
        "qwertyuiop",
        "blah blah blah",
        "nonsense words",
        "hahaha hehehe",
        "random garbage text"
    ]
    
    print("\n\n🤖 Testing Random/Garbage Input Handling")
    print("=" * 80)
    
    for random_input in random_inputs:
        print(f"\n📝 Testing: '{random_input}'")
        
        try:
            response = requests.post(
                f"{base_url}/career_advice",
                json={"user_input": random_input},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                intent = data.get('intent', '')
                careers = data.get('career_recommendations', [])
                response_text = data.get('message', '')
                
                # Should not give career advice for random input
                if intent != "career_advice" and len(careers) == 0:
                    print("✅ Correctly handled as non-career input")
                else:
                    print(f"❌ Incorrectly gave career advice: {intent} with {len(careers)} careers")
                
                print(f"Response: {response_text}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_greeting_only():
    """Test that greetings don't trigger career advice"""
    base_url = "http://localhost:5000"
    
    greeting_tests = [
        "hello",
        "hi there", 
        "salam",
        "aap kaise hain",
        "good morning",
        "good evening",
        "hey",
        "hiya"
    ]
    
    print("\n\n🤝 Testing Greeting Functionality")
    print("=" * 80)
    
    for greeting in greeting_tests:
        print(f"\n📝 Testing: '{greeting}'")
        
        try:
            response = requests.post(
                f"{base_url}/career_advice",
                json={"user_input": greeting},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                intent = data.get('intent', '')
                careers = data.get('career_recommendations', [])
                
                if intent == "greeting" and len(careers) == 0:
                    print("✅ Correctly identified as greeting")
                else:
                    print(f"❌ Incorrectly identified as {intent} with {len(careers)} careers")
                
                print(f"Response: {data.get('message', '')}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Comprehensive Testing of AI/ML Career Advice Bot")
    print("Make sure the API is running: python app_ml.py")
    print("=" * 80)
    
    # Test career advice with new fields
    passed, total = test_career_advice()
    
    # Test random input handling
    test_random_input()
    
    # Test greeting functionality
    test_greeting_only()
    
    print("\n" + "=" * 80)
    print(f"🎯 Testing Complete! Career Advice Tests: {passed}/{total} passed")
    print("🎉 The bot should now handle all career fields and random input properly!")