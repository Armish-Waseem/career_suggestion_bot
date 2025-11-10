import requests
import json

def test_multiple_examples():
    url = "http://localhost:5000/career_advice"
    
    test_cases = [
        {
            "name": "Medicine Interest (Roman Urdu)",
            "input": "mujhe medicine me interest hai, hardworking hun aur logo ki madad krna acha lagta hai"
        },
        {
            "name": "Computer Interest (Roman Urdu)", 
            "input": "mujhe computer aur programming mein interest hai, coding karna pasand hai"
        },
        {
            "name": "Business Interest (English)",
            "input": "I love business and marketing, want to make money and help companies grow"
        },
        {
            "name": "Creative Interest (Roman Urdu)",
            "input": "mujhe art aur design mein interest hai, creative kaam karna pasand hai"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print("-" * 60)
        
        try:
            response = requests.post(url, json={"user_input": test_case['input']})
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Language: {result['language_detected']}")
                print(f"✅ Response: {result['message']}")
                print(f"✅ Top 3 Careers: {', '.join(result['career_recommendations'][:3])}")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_multiple_examples()

