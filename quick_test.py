import requests

# Test the API with different fields
test_inputs = [
    "mujhe science aur research mein interest hai",
    "mujhe law aur justice mein interest hai", 
    "mujhe agriculture mein interest hai",
    "mujhe sports aur fitness mein interest hai"
]

for test_input in test_inputs:
    try:
        response = requests.post('http://localhost:5000/career_advice', 
                               json={'user_input': test_input})
        if response.status_code == 200:
            result = response.json()
            print(f"Input: {test_input}")
            print(f"Response: {result['message']}")
            print("-" * 50)
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

