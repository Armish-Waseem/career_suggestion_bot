import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
from typing import Dict, List, Tuple
import os

class CareerAdviceMLModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        self.intent_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.career_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15
        )
        self.is_trained = False
        
        # Roman Urdu to English mapping
        self.roman_urdu_mapping = {
            "kamputar": "computer", "programming": "programming", "coding": "coding",
            "software": "software", "tech": "technology", "data": "data",
            "web": "web", "mobile": "mobile", "app": "application",
            "sehat": "health", "doctor": "doctor", "nurse": "nurse",
            "mariz": "patient", "hospital": "hospital", "dawa": "medicine",
            "karobar": "business", "marketing": "marketing", "finance": "finance",
            "management": "management", "sales": "sales", "paisa": "money",
            "fun": "art", "design": "design", "creative": "creative",
            "music": "music", "likhna": "writing", "photography": "photography",
            "parhana": "teaching", "taleem": "education", "seekhna": "learning",
            "school": "school", "talib_ilm": "student", "ilm": "knowledge",
            "tahqeeq": "research", "engineering": "engineering",
            "mechanical": "mechanical", "electrical": "electrical", "civil": "civil"
        }
        
        # Career database
        self.career_database = {
            "Software Engineer": "Develops software applications, writes code, designs systems, works with programming languages, debugging, software architecture, full-stack development, mobile apps, web applications, computer science, engineering",
            "Data Scientist": "Analyzes data, machine learning, statistical modeling, Python programming, data visualization, predictive analytics, big data, artificial intelligence, research, algorithms, mathematics, statistics",
            "AI/ML Engineer": "Machine learning, artificial intelligence, neural networks, deep learning, Python, TensorFlow, PyTorch, model training, algorithm development, automation, robotics, computer vision",
            "Doctor": "Medical practice, patient care, diagnosis, treatment, surgery, medical research, healthcare, anatomy, physiology, pharmacology, clinical skills, medicine, hospital",
            "Nurse": "Patient care, medical assistance, healthcare, clinical skills, patient monitoring, medication administration, healthcare coordination, compassion, medical knowledge, hospital",
            "Business Analyst": "Business analysis, data analysis, requirements gathering, process improvement, project management, stakeholder management, business intelligence, strategy, consulting",
            "Marketing Manager": "Marketing strategy, digital marketing, brand management, market research, advertising, social media, campaign management, analytics, leadership, brand development",
            "Graphic Designer": "Visual design, creativity, Adobe Creative Suite, branding, typography, layout design, digital art, marketing materials, artistic skills, logo design, print design",
            "Teacher": "Education, teaching, curriculum development, student development, classroom management, educational psychology, subject expertise, communication, leadership, student mentoring",
            "Mechanical Engineer": "Mechanical systems, engineering design, manufacturing, thermodynamics, materials science, CAD, problem solving, technical analysis, innovation, product development"
        }
    
    def detect_language(self, text: str) -> str:
        """Detect if text is in English or Roman Urdu"""
        roman_urdu_patterns = [
            r'\b(sehat|doctor|nurse|mariz|hospital|dawa|karobar|marketing|finance|management|sales|paisa|munafa|company|fun|design|creative|music|likhna|photography|video|parhana|taleem|seekhna|school|talib_ilm|ilm|tahqeeq|engineering|mechanical|electrical|civil|tameer|imarat|machine)\b',
            r'\b(ka|ki|ke|se|mein|par|ko|ne|koi|kuch|bahut|zaroor|shayad|bilkul|nahi|haan|theek|achha|bura|naya|purana|chota|bada|mujhe|aap|tum|ham|unhe|usne|maine|tune|hamne)\b'
        ]
        
        for pattern in roman_urdu_patterns:
            if re.search(pattern, text.lower()):
                return "roman_urdu"
        
        return "english"
    
    def translate_roman_urdu_to_english(self, text: str) -> str:
        """Convert Roman Urdu text to English equivalents"""
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.roman_urdu_mapping:
                translated_words.append(self.roman_urdu_mapping[clean_word])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for ML model"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def load_dataset(self, dataset_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Load and prepare dataset for training"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        intents = []
        careers = []
        
        # Process career advice data
        for item in data['training_data']:
            texts.append(item['text'])
            intents.append(item['intent'])
            careers.append(item['careers'])
        
        # Process conversation data
        for item in data['conversation_data']:
            texts.append(item['text'])
            intents.append(item['intent'])
            careers.append([])  # No careers for conversation intents
        
        return texts, intents, careers
    
    def train_model(self, dataset_path: str):
        """Train the ML models"""
        print("Loading dataset...")
        texts, intents, careers = self.load_dataset(dataset_path)
        
        print("Preprocessing texts...")
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        print("Training TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(processed_texts)
        
        print("Training intent classifier...")
        self.intent_classifier.fit(X, intents)
        
        print("Training career classifier...")
        # Flatten careers for multi-label classification
        all_careers = []
        career_labels = []
        for i, career_list in enumerate(careers):
            if career_list:  # Only for career_advice intent
                for career in career_list:
                    all_careers.append(processed_texts[i])
                    career_labels.append(career)
        
        if all_careers:
            X_careers = self.vectorizer.transform(all_careers)
            self.career_classifier.fit(X_careers, career_labels)
        
        self.is_trained = True
        print("Model training completed!")
        
        # Save the trained model
        self.save_model()
    
    def predict_intent(self, text: str) -> str:
        """Predict the intent of user input"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        intent = self.intent_classifier.predict(X)[0]
        confidence = self.intent_classifier.predict_proba(X).max()
        
        # Override intent if career-related keywords are detected
        career_keywords = [
            # Technology
            'programming', 'computer', 'software', 'tech', 'coding', 'data', 'ai', 'machine learning', 'web', 'mobile', 'app', 'cybersecurity', 'gaming', 'video games',
            # Healthcare
            'medicine', 'doctor', 'nurse', 'health', 'medical', 'patient', 'healthcare', 'hospital', 'therapy', 'psychology', 'psychiatrist',
            # Business & Management
            'business', 'marketing', 'finance', 'money', 'company', 'sales', 'management', 'leadership', 'teams', 'real estate', 'property',
            # Creative & Arts
            'art', 'design', 'creative', 'music', 'writing', 'photography', 'video', 'painting', 'sculpture', 'fashion', 'clothing', 'style',
            # Education
            'teaching', 'education', 'school', 'student', 'learning', 'teacher', 'professor',
            # Engineering & Construction
            'engineering', 'mechanical', 'electrical', 'civil', 'construction', 'building', 'architecture', 'machines',
            # Science & Research
            'science', 'research', 'laboratory', 'chemistry', 'physics', 'biology', 'environment', 'nature', 'conservation', 'ecology',
            # Legal
            'law', 'legal', 'justice', 'court', 'lawyer', 'judge', 'philosophy', 'ethics',
            # Sports & Fitness
            'sports', 'fitness', 'gym', 'athlete', 'coaching', 'exercise', 'athletics',
            # Media & Communication
            'media', 'journalism', 'news', 'broadcasting', 'communication', 'films', 'cinema', 'movies', 'directing', 'storytelling', 'books', 'novels',
            # Hospitality & Tourism
            'hospitality', 'hotel', 'restaurant', 'tourism', 'travel', 'cooking', 'chef', 'food', 'culinary',
            # Agriculture & Environment
            'agriculture', 'farming', 'crops', 'animals', 'food', 'veterinarian',
            # Transportation & Aviation
            'transportation', 'driving', 'pilot', 'aviation', 'logistics', 'flying', 'aircraft', 'supply chain', 'warehousing',
            # Security & Defense
            'police', 'army', 'security', 'defense', 'military',
            # Social Work
            'social work', 'helping people', 'community', 'welfare', 'ngo', 'humanitarian'
        ]
        
        # Check for career keywords in the text
        text_lower = text.lower()
        career_keyword_count = sum(1 for keyword in career_keywords if keyword in text_lower)
        
        # If multiple career keywords found, override to career_advice
        if career_keyword_count >= 1 and intent != "career_advice":
            intent = "career_advice"
            confidence = 0.9  # High confidence for career advice
        
        return intent, confidence
    
    def predict_careers(self, text: str) -> List[str]:
        """Predict career recommendations"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        # Get career probabilities
        career_probs = self.career_classifier.predict_proba(X)[0]
        career_classes = self.career_classifier.classes_
        
        # Get top 5 careers
        top_indices = np.argsort(career_probs)[-5:][::-1]
        top_careers = [career_classes[i] for i in top_indices if career_probs[i] > 0.1]
        
        # If no careers from ML model, use keyword-based matching
        if not top_careers:
            top_careers = self.get_careers_by_keywords(text)
        
        return top_careers
    
    def get_careers_by_keywords(self, text: str) -> List[str]:
        """Get career recommendations based on keyword matching"""
        text_lower = text.lower()
        careers = []
        
        # Technology keywords
        if any(word in text_lower for word in ['programming', 'computer', 'software', 'tech', 'coding', 'data', 'ai', 'machine learning', 'web', 'mobile', 'app']):
            careers.extend(['Software Engineer', 'Data Scientist', 'AI/ML Engineer', 'Web Developer', 'Mobile App Developer'])
        
        # Cybersecurity keywords
        if any(word in text_lower for word in ['cybersecurity', 'it security', 'hacking', 'protection']):
            careers.extend(['Cybersecurity Analyst', 'Information Security Manager', 'Ethical Hacker', 'Security Consultant', 'IT Security Specialist'])
        
        # Gaming keywords
        if any(word in text_lower for word in ['gaming', 'video games', 'game development']):
            careers.extend(['Game Developer', 'Game Designer', 'Game Tester', 'Game Artist', 'Game Programmer'])
        
        # Healthcare keywords
        if any(word in text_lower for word in ['medicine', 'doctor', 'nurse', 'health', 'medical', 'patient', 'healthcare', 'hospital']):
            careers.extend(['Doctor', 'Nurse', 'Pharmacist', 'Physical Therapist', 'Healthcare Administrator'])
        
        # Psychology keywords
        if any(word in text_lower for word in ['psychology', 'human behavior', 'mind', 'therapy', 'psychiatrist']):
            careers.extend(['Psychologist', 'Therapist', 'Counselor', 'Psychiatrist', 'Social Worker'])
        
        # Business keywords
        if any(word in text_lower for word in ['business', 'marketing', 'finance', 'money', 'company', 'sales']):
            careers.extend(['Business Analyst', 'Marketing Manager', 'Financial Analyst', 'Investment Banker', 'Sales Manager'])
        
        # Management keywords
        if any(word in text_lower for word in ['management', 'leadership', 'teams']):
            careers.extend(['Project Manager', 'Operations Manager', 'HR Manager', 'Business Manager', 'Team Leader'])
        
        # Real Estate keywords
        if any(word in text_lower for word in ['real estate', 'property', 'housing', 'investment']):
            careers.extend(['Real Estate Agent', 'Property Manager', 'Real Estate Developer', 'Appraiser', 'Mortgage Broker'])
        
        # Creative keywords
        if any(word in text_lower for word in ['art', 'design', 'creative', 'painting', 'sculpture']):
            careers.extend(['Artist', 'Art Director', 'Gallery Curator', 'Art Teacher', 'Art Therapist'])
        
        # Music keywords
        if any(word in text_lower for word in ['music', 'singing', 'instruments', 'composing']):
            careers.extend(['Musician', 'Singer', 'Music Producer', 'Composer', 'Music Teacher'])
        
        # Writing keywords
        if any(word in text_lower for word in ['writing', 'storytelling', 'books', 'novels']):
            careers.extend(['Writer', 'Novelist', 'Journalist', 'Content Writer', 'Screenwriter'])
        
        # Film keywords
        if any(word in text_lower for word in ['films', 'cinema', 'movies', 'directing']):
            careers.extend(['Film Director', 'Cinematographer', 'Film Editor', 'Producer', 'Screenwriter'])
        
        # Fashion keywords
        if any(word in text_lower for word in ['fashion', 'clothing', 'design', 'style']):
            careers.extend(['Fashion Designer', 'Fashion Stylist', 'Fashion Buyer', 'Textile Designer', 'Fashion Photographer'])
        
        # Education keywords
        if any(word in text_lower for word in ['teaching', 'education', 'school', 'student', 'learning', 'teacher']):
            careers.extend(['Teacher', 'Professor', 'Educational Administrator', 'Tutor', 'School Counselor'])
        
        # Engineering keywords
        if any(word in text_lower for word in ['engineering', 'mechanical', 'electrical', 'civil', 'construction', 'building', 'machines']):
            careers.extend(['Mechanical Engineer', 'Civil Engineer', 'Electrical Engineer', 'Aerospace Engineer', 'Industrial Engineer'])
        
        # Architecture keywords
        if any(word in text_lower for word in ['architecture', 'building', 'design', 'construction']):
            careers.extend(['Architect', 'Urban Planner', 'Interior Designer', 'Construction Manager', 'Building Inspector'])
        
        # Science keywords
        if any(word in text_lower for word in ['science', 'research', 'laboratory', 'chemistry', 'physics', 'biology']):
            careers.extend(['Research Scientist', 'Biologist', 'Chemist', 'Physicist', 'Data Analyst'])
        
        # Environmental keywords
        if any(word in text_lower for word in ['environment', 'nature', 'conservation', 'ecology']):
            careers.extend(['Environmental Scientist', 'Conservationist', 'Ecologist', 'Environmental Engineer', 'Wildlife Biologist'])
        
        # Law keywords
        if any(word in text_lower for word in ['law', 'legal', 'justice', 'court', 'lawyer', 'judge']):
            careers.extend(['Lawyer', 'Judge', 'Paralegal', 'Legal Assistant', 'Court Reporter'])
        
        # Philosophy keywords
        if any(word in text_lower for word in ['philosophy', 'thinking', 'ethics', 'logic']):
            careers.extend(['Philosopher', 'Ethics Consultant', 'Writer', 'Professor', 'Think Tank Researcher'])
        
        # Sports keywords
        if any(word in text_lower for word in ['sports', 'fitness', 'gym', 'athlete', 'coaching', 'exercise', 'athletics']):
            careers.extend(['Personal Trainer', 'Sports Coach', 'Athletic Trainer', 'Sports Medicine Physician', 'Fitness Instructor'])
        
        # Media keywords
        if any(word in text_lower for word in ['media', 'journalism', 'news', 'broadcasting', 'communication']):
            careers.extend(['Journalist', 'News Anchor', 'Public Relations Specialist', 'Media Producer', 'Content Creator'])
        
        # Hospitality keywords
        if any(word in text_lower for word in ['hospitality', 'hotel', 'restaurant', 'tourism', 'travel']):
            careers.extend(['Hotel Manager', 'Travel Agent', 'Event Coordinator', 'Restaurant Manager', 'Tour Guide'])
        
        # Cooking keywords
        if any(word in text_lower for word in ['cooking', 'chef', 'food', 'culinary']):
            careers.extend(['Chef', 'Restaurant Manager', 'Food Critic', 'Catering Manager', 'Pastry Chef'])
        
        # Agriculture keywords
        if any(word in text_lower for word in ['agriculture', 'farming', 'crops', 'animals']):
            careers.extend(['Farmer', 'Agricultural Engineer', 'Veterinarian', 'Agricultural Scientist', 'Farm Manager'])
        
        # Aviation keywords
        if any(word in text_lower for word in ['aviation', 'flying', 'pilot', 'aircraft']):
            careers.extend(['Pilot', 'Air Traffic Controller', 'Aircraft Engineer', 'Flight Attendant', 'Aviation Manager'])
        
        # Transportation keywords
        if any(word in text_lower for word in ['transportation', 'driving', 'logistics', 'supply chain', 'warehousing']):
            careers.extend(['Logistics Manager', 'Supply Chain Analyst', 'Warehouse Manager', 'Transportation Manager', 'Operations Analyst'])
        
        # Security keywords
        if any(word in text_lower for word in ['police', 'army', 'security', 'defense', 'military']):
            careers.extend(['Police Officer', 'Army Officer', 'Security Guard', 'Detective', 'Military Engineer'])
        
        # Social Work keywords
        if any(word in text_lower for word in ['social work', 'helping people', 'community', 'welfare', 'ngo', 'humanitarian']):
            careers.extend(['Social Worker', 'Community Organizer', 'NGO Worker', 'Counselor', 'Humanitarian Worker'])
        
        # Return top 5 unique careers
        return list(dict.fromkeys(careers))[:5]
    
    def generate_response(self, text: str, language: str) -> str:
        """Generate appropriate response based on intent and language"""
        intent, confidence = self.predict_intent(text)
        
        if intent == "career_advice":
            careers = self.predict_careers(text)
            if language == "roman_urdu":
                if careers:
                    career_list = ", ".join(careers[:3])
                    return f"Aap ke interest aur skills ke hisab se aap yeh career paths choose kar sakte hain: {career_list}. Aap ka passion aur hardworking attitude is field mein success guarantee karta hai."
                else:
                    return "Aap ke interest aur skills ke hisab se aap koi bhi field choose kar sakte hain. Aap ka passion aur hardworking attitude kisi bhi field mein success guarantee karta hai."
            else:
                if careers:
                    career_list = ", ".join(careers[:3])
                    return f"Based on your interests and skills, you should consider these career paths: {career_list}. Your passion and hardworking attitude will guarantee success in this field."
                else:
                    return "Based on your interests and skills, you can choose any field. Your passion and hardworking attitude will guarantee success in any field."
        
        elif intent == "greeting":
            if language == "roman_urdu":
                return "Salam! Main aap ka AI Career Advisor hun. Aap ki career guidance ke liye yahan hun. Aap kya janna chahte hain?"
            else:
                return "Hello! I'm your AI Career Advisor. I'm here to help you with career guidance. What would you like to know?"
        
        elif intent == "capabilities":
            if language == "roman_urdu":
                return "Main aap ke interests, skills aur goals ke base par perfect career suggest kar sakta hun. Bas mujhe bataiye aap kya pasand karte hain!"
            else:
                return "I can help you find the perfect career based on your interests, skills, and goals. Just tell me what you're passionate about!"
        
        elif intent == "help":
            if language == "roman_urdu":
                return "Bilkul! Mujhe aap ke interests, skills ya field ke bare mein bataiye, main aap ke liye career paths suggest karunga."
            else:
                return "I'd be happy to help! Tell me about your interests, skills, or what field you'd like to work in, and I'll suggest career paths for you."
        
        elif intent == "thanks":
            if language == "roman_urdu":
                return "Aap ka welcome! Main hamesha career guidance ke liye yahan hun. Kabhi bhi puch sakte hain!"
            else:
                return "You're welcome! I'm always here to help with career guidance. Feel free to ask anytime!"
        
        elif intent == "goodbye":
            if language == "roman_urdu":
                return "Khuda hafiz! Aap ki career journey mein best of luck. Kabhi bhi advice ke liye wapas aaiye!"
            else:
                return "Goodbye! Best of luck with your career journey. Come back anytime for more advice!"
        
        else:
            if language == "roman_urdu":
                return "Main aap ki career guidance ke liye yahan hun. Agar aap ko apni career ke bare mein koi help chahiye to mujhe batayiye!"
            else:
                return "I'm here to help you with career guidance. If you need any help with your career, let me know!"
    
    def save_model(self, model_path: str = "models/"):
        """Save the trained model"""
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.vectorizer, f"{model_path}vectorizer.pkl")
        joblib.dump(self.intent_classifier, f"{model_path}intent_classifier.pkl")
        joblib.dump(self.career_classifier, f"{model_path}career_classifier.pkl")
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "models/"):
        """Load the trained model"""
        try:
            self.vectorizer = joblib.load(f"{model_path}vectorizer.pkl")
            self.intent_classifier = joblib.load(f"{model_path}intent_classifier.pkl")
            self.career_classifier = joblib.load(f"{model_path}career_classifier.pkl")
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
            self.is_trained = False

# Training script
if __name__ == "__main__":
    model = CareerAdviceMLModel()
    model.train_model("dataset/career_dataset.json")
    
    # Test the model
    test_cases = [
        "I love programming",
        "mujhe medicine mein interest hai",
        "hello",
        "salam",
        "I want to help people"
    ]
    
    for test in test_cases:
        language = model.detect_language(test)
        response = model.generate_response(test, language)
        print(f"Input: {test}")
        print(f"Language: {language}")
        print(f"Response: {response}")
        print("-" * 50)
