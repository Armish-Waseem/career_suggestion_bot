import requests
import json

def test_api():
    url = "http://localhost:5000/career_advice"
    
    # Test with your example
    test_data = {
        "user_input": "mujhe medicine me interest hai, hardworking hun aur logo ki madad krna acha lagta hai"
    }
    
    try:
        response = requests.post(url, json=test_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:")
            print(f"Language Detected: {result['language_detected']}")
            print(f"Message: {result['message']}")
            print(f"Top 5 Career Recommendations:")
            for i, career in enumerate(result['career_recommendations'][:5], 1):
                print(f"{i}. {career}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_api()

