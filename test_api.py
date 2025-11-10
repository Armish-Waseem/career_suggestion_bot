import requests
import json

# Test the AI/ML Career Advice API
def test_career_advice_api():
    base_url = "http://localhost:5000"
    
    # Test cases
    test_cases = [
        {
            "name": "English - Technology Interest",
            "input": "I love programming and working with computers",
            "expected_language": "english"
        },
        {
            "name": "Roman Urdu - Technology Interest", 
            "input": "Mujhe computer aur programming mein interest hai",
            "expected_language": "roman_urdu"
        },
        {
            "name": "English - Healthcare Interest",
            "input": "I want to help people and work in healthcare",
            "expected_language": "english"
        },
        {
            "name": "Roman Urdu - Healthcare Interest",
            "input": "Mujhe doctor banna hai aur sehat mein kaam karna hai",
            "expected_language": "roman_urdu"
        },
        {
            "name": "English - Creative Interest",
            "input": "I enjoy art, design, and creative work",
            "expected_language": "english"
        },
        {
            "name": "Roman Urdu - Business Interest",
            "input": "Mujhe business aur marketing mein interest hai",
            "expected_language": "roman_urdu"
        }
    ]
    
    print("Testing AI/ML Career Advice API...")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        try:
            response = requests.post(
                f"{base_url}/career_advice",
                json={"user_input": test_case['input']},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success!")
                print(f"Language Detected: {data['language_detected']}")
                print(f"Total Recommendations: {data['total_recommendations']}")
                print(f"Message: {data['message']}")
                print(f"First 5 Recommendations: {data['career_recommendations'][:5]}")
                if 'ai_analysis' in data:
                    print(f"ML Model Used: {data['ai_analysis']['ml_model_used']}")
                    print(f"Processing Method: {data['ai_analysis']['processing_method']}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the API is running on localhost:5000")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Test health endpoint
    print(f"\n{'='*50}")
    print("Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

if __name__ == "__main__":
    test_career_advice_api()
