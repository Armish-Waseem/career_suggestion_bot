from flask import Flask, request, jsonify
import re
import random
from typing import List, Dict
import json
import math
from collections import Counter
from career_data import CAREER_DATABASE

app = Flask(__name__)

# Roman Urdu to English mapping for common career-related terms
ROMAN_URDU_MAPPING = {
    "kamputar": "computer",
    "programming": "programming", 
    "coding": "coding",
    "software": "software",
    "tech": "technology",
    "data": "data",
    "web": "web",
    "mobile": "mobile",
    "app": "application",
    "digital": "digital",
    "sehat": "health",
    "doctor": "doctor",
    "nurse": "nurse",
    "mariz": "patient",
    "hospital": "hospital",
    "dawa": "medicine",
    "karobar": "business",
    "marketing": "marketing",
    "finance": "finance",
    "management": "management",
    "sales": "sales",
    "paisa": "money",
    "munafa": "profit",
    "company": "company",
    "fun": "art",
    "design": "design",
    "creative": "creative",
    "music": "music",
    "likhna": "writing",
    "photography": "photography",
    "video": "video",
    "parhana": "teaching",
    "taleem": "education",
    "seekhna": "learning",
    "school": "school",
    "talib_ilm": "student",
    "ilm": "knowledge",
    "tahqeeq": "research",
    "engineering": "engineering",
    "mechanical": "mechanical",
    "electrical": "electrical",
    "civil": "civil",
    "tameer": "construction",
    "imarat": "building",
    "machine": "machine"
}

class SimpleCareerRecommendationEngine:
    def __init__(self):
        self.career_database = CAREER_DATABASE
        self.career_names = list(CAREER_DATABASE.keys())
        self.career_descriptions = list(CAREER_DATABASE.values())
        
        # Pre-compute word frequencies for TF-IDF-like scoring
        self.career_word_counts = {}
        self.total_careers = len(self.career_database)
        
        # Build word frequency matrix
        all_words = set()
        for description in self.career_descriptions:
            words = self.preprocess_text(description)
            all_words.update(words)
        
        for i, description in enumerate(self.career_descriptions):
            words = self.preprocess_text(description)
            word_count = Counter(words)
            self.career_word_counts[i] = word_count
        
        # Calculate document frequencies
        self.doc_frequencies = {}
        for word in all_words:
            count = sum(1 for word_counts in self.career_word_counts.values() if word in word_counts)
            self.doc_frequencies[word] = count
        
        print("Simple AI/ML Career Recommendation Engine loaded successfully!")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Split into words and filter out short words
        words = [word for word in text.split() if len(word) > 2]
        return words
    
    def detect_language(self, text: str) -> str:
        """Detect if text is in English or Roman Urdu"""
        roman_urdu_patterns = [
            r'\b(sehat|doctor|nurse|mariz|hospital|dawa|karobar|marketing|finance|management|sales|paisa|munafa|company|fun|design|creative|music|likhna|photography|video|parhana|taleem|seekhna|school|talib_ilm|ilm|tahqeeq|engineering|mechanical|electrical|civil|tameer|imarat|machine)\b',
            r'\b(ka|ki|ke|se|mein|par|ko|ne|koi|kuch|bahut|zaroor|shayad|bilkul|nahi|haan|theek|achha|bura|naya|purana|chota|bada)\b'
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
            if clean_word in ROMAN_URDU_MAPPING:
                translated_words.append(ROMAN_URDU_MAPPING[clean_word])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def calculate_tfidf_score(self, user_words: List[str], career_index: int) -> float:
        """Calculate TF-IDF-like score between user input and career description"""
        career_word_count = self.career_word_counts[career_index]
        total_words_in_career = sum(career_word_count.values())
        
        score = 0.0
        for word in user_words:
            if word in career_word_count:
                # Term frequency
                tf = career_word_count[word] / total_words_in_career
                # Inverse document frequency
                idf = math.log(self.total_careers / (self.doc_frequencies[word] + 1))
                score += tf * idf
        
        return score
    
    def calculate_keyword_similarity(self, user_input: str) -> List[float]:
        """Calculate keyword-based similarity scores"""
        user_words = self.preprocess_text(user_input)
        scores = []
        
        for i in range(len(self.career_descriptions)):
            score = self.calculate_tfidf_score(user_words, i)
            scores.append(score)
        
        return scores
    
    def calculate_semantic_similarity(self, user_input: str) -> List[float]:
        """Calculate semantic similarity using word overlap and context"""
        user_words = set(self.preprocess_text(user_input))
        scores = []
        
        # Define semantic word groups
        semantic_groups = {
            'technology': ['computer', 'programming', 'software', 'tech', 'data', 'web', 'mobile', 'app', 'digital', 'cyber', 'network', 'database', 'ai', 'machine', 'learning'],
            'healthcare': ['health', 'medical', 'doctor', 'nurse', 'patient', 'hospital', 'medicine', 'therapy', 'care', 'treatment', 'clinical', 'diagnosis'],
            'business': ['business', 'marketing', 'finance', 'management', 'sales', 'money', 'profit', 'company', 'strategy', 'analysis', 'consulting'],
            'creative': ['art', 'design', 'creative', 'music', 'writing', 'photography', 'video', 'drawing', 'visual', 'artistic', 'media', 'content'],
            'education': ['teaching', 'education', 'learning', 'school', 'student', 'knowledge', 'research', 'academic', 'curriculum', 'training'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'construction', 'building', 'machine', 'technical', 'systems', 'design']
        }
        
        for description in self.career_descriptions:
            desc_words = set(self.preprocess_text(description))
            
            # Direct word overlap
            overlap = len(user_words.intersection(desc_words))
            overlap_score = overlap / len(user_words) if user_words else 0
            
            # Semantic group matching
            semantic_score = 0
            for group, keywords in semantic_groups.items():
                user_group_words = user_words.intersection(set(keywords))
                desc_group_words = desc_words.intersection(set(keywords))
                if user_group_words and desc_group_words:
                    semantic_score += len(user_group_words.intersection(desc_group_words)) / len(user_group_words)
            
            # Combined score
            combined_score = 0.7 * overlap_score + 0.3 * semantic_score
            scores.append(combined_score)
        
        return scores
    
    def get_ml_recommendations(self, user_input: str, top_k: int = 50) -> List[Dict]:
        """Get career recommendations using ML algorithms"""
        # Calculate different similarity scores
        keyword_scores = self.calculate_keyword_similarity(user_input)
        semantic_scores = self.calculate_semantic_similarity(user_input)
        
        # Create recommendations with scores
        recommendations = []
        for i, (career, description) in enumerate(self.career_database.items()):
            # Combine different scoring methods
            keyword_score = keyword_scores[i]
            semantic_score = semantic_scores[i]
            
            # Normalize scores
            keyword_score = min(keyword_score, 1.0)  # Cap at 1.0
            semantic_score = min(semantic_score, 1.0)  # Cap at 1.0
            
            # Combined score with weights
            combined_score = 0.6 * semantic_score + 0.4 * keyword_score
            
            recommendations.append({
                'career': career,
                'description': description,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top K recommendations
        return recommendations[:top_k]
    
    def cluster_careers(self, recommendations: List[Dict], n_clusters: int = 5) -> Dict:
        """Simple clustering based on career categories"""
        if len(recommendations) < n_clusters:
            return {'clusters': [recommendations]}
        
        # Define career categories
        categories = {
            'Technology': ['Software', 'Data', 'AI', 'Cybersecurity', 'Web', 'Mobile', 'DevOps', 'Cloud', 'Database', 'Network', 'UI/UX', 'Product', 'Technical', 'IT', 'System'],
            'Healthcare': ['Doctor', 'Nurse', 'Pharmacist', 'Physical', 'Dentist', 'Veterinarian', 'Medical', 'Healthcare', 'Radiologist', 'Surgeon', 'Psychologist', 'Nutritionist', 'Biomedical'],
            'Business': ['Business', 'Marketing', 'Financial', 'Investment', 'Management', 'Entrepreneur', 'Sales', 'HR', 'Operations', 'Supply', 'Accountant', 'Auditor'],
            'Creative': ['Graphic', 'Artist', 'Writer', 'Photographer', 'Video', 'Music', 'Interior', 'Fashion', 'Animator', 'Content', 'Social', 'Advertising'],
            'Education': ['Teacher', 'Professor', 'Educational', 'Curriculum', 'Librarian', 'Training', 'Academic', 'School'],
            'Engineering': ['Mechanical', 'Civil', 'Electrical', 'Chemical', 'Aerospace', 'Environmental', 'Industrial']
        }
        
        # Group careers by category
        clusters = {}
        for rec in recommendations:
            career_name = rec['career']
            assigned = False
            
            for category, keywords in categories.items():
                if any(keyword in career_name for keyword in keywords):
                    if category not in clusters:
                        clusters[category] = []
                    clusters[category].append(rec)
                    assigned = True
                    break
            
            if not assigned:
                if 'Other' not in clusters:
                    clusters['Other'] = []
                clusters['Other'].append(rec)
        
        return {'clusters': list(clusters.values())}

# Initialize the ML engine
ml_engine = SimpleCareerRecommendationEngine()

def generate_simple_response(user_input: str, top_careers: List[str], language: str) -> str:
    """Generate simple, conversational response based on user input and top careers"""
    
    if language == "roman_urdu":
        # Analyze user input for key interests
        input_lower = user_input.lower()
        
        # Check for specific fields
        if any(word in input_lower for word in ['medicine', 'doctor', 'nurse', 'health', 'sehat', 'dawa', 'mariz']):
            return f"Aap ke interest aur skills ke hisab se aap medicine ki field mein career banana chahiye. Aap doctor, nurse, ya pharmacist ban sakte hain. Aap ka helping nature aur hardworking attitude is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['computer', 'programming', 'tech', 'software', 'kamputar', 'coding']):
            return f"Aap ke interest aur skills ke hisab se aap technology ki field mein career banana chahiye. Aap software engineer, web developer, ya data scientist ban sakte hain. Aap ka technical interest is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['business', 'marketing', 'finance', 'karobar', 'paisa', 'company']):
            return f"Aap ke interest aur skills ke hisab se aap business ki field mein career banana chahiye. Aap business analyst, marketing manager, ya entrepreneur ban sakte hain. Aap ka business sense is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['art', 'design', 'creative', 'fun', 'music', 'writing', 'likhna']):
            return f"Aap ke interest aur skills ke hisab se aap creative field mein career banana chahiye. Aap graphic designer, writer, ya artist ban sakte hain. Aap ka creative talent is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['teaching', 'education', 'parhana', 'taleem', 'school', 'student']):
            return f"Aap ke interest aur skills ke hisab se aap education ki field mein career banana chahiye. Aap teacher, professor, ya educational administrator ban sakte hain. Aap ka teaching passion is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['engineering', 'mechanical', 'electrical', 'civil', 'tameer', 'machine']):
            return f"Aap ke interest aur skills ke hisab se aap engineering ki field mein career banana chahiye. Aap mechanical engineer, civil engineer, ya electrical engineer ban sakte hain. Aap ka technical interest is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['science', 'research', 'laboratory', 'scientist', 'chemistry', 'physics', 'biology', 'tahqeeq', 'lab']):
            return f"Aap ke interest aur skills ke hisab se aap science aur research ki field mein career banana chahiye. Aap research scientist, chemist, ya biologist ban sakte hain. Aap ka analytical mind is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['law', 'legal', 'lawyer', 'justice', 'court', 'legal advice', 'qanoon', 'adalat']):
            return f"Aap ke interest aur skills ke hisab se aap legal field mein career banana chahiye. Aap lawyer, judge, ya legal consultant ban sakte hain. Aap ka sense of justice is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['police', 'security', 'law enforcement', 'detective', 'police officer', 'police', 'security guard']):
            return f"Aap ke interest aur skills ke hisab se aap law enforcement ki field mein career banana chahiye. Aap police officer, detective, ya security specialist ban sakte hain. Aap ka sense of duty is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['agriculture', 'farming', 'crops', 'animals', 'kheti', 'janwar', 'farming']):
            return f"Aap ke interest aur skills ke hisab se aap agriculture ki field mein career banana chahiye. Aap agricultural engineer, farm manager, ya food scientist ban sakte hain. Aap ka love for nature is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['transportation', 'driving', 'pilot', 'aviation', 'logistics', 'transport', 'flying']):
            return f"Aap ke interest aur skills ke hisab se aap transportation ki field mein career banana chahiye. Aap pilot, truck driver, ya logistics coordinator ban sakte hain. Aap ka love for travel is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['sports', 'fitness', 'gym', 'athlete', 'coaching', 'sports', 'exercise']):
            return f"Aap ke interest aur skills ke hisab se aap sports aur fitness ki field mein career banana chahiye. Aap personal trainer, sports coach, ya fitness instructor ban sakte hain. Aap ka passion for fitness is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['media', 'journalism', 'news', 'broadcasting', 'communication', 'media', 'news']):
            return f"Aap ke interest aur skills ke hisab se aap media aur communication ki field mein career banana chahiye. Aap journalist, news anchor, ya media producer ban sakte hain. Aap ka communication skills is field ke liye perfect hai."
        
        elif any(word in input_lower for word in ['hospitality', 'hotel', 'restaurant', 'tourism', 'travel', 'customer service', 'hotel', 'restaurant']):
            return f"Aap ke interest aur skills ke hisab se aap hospitality ki field mein career banana chahiye. Aap hotel manager, chef, ya travel agent ban sakte hain. Aap ka customer service skills is field ke liye perfect hai."
        
        else:
            # Generic response based on top career
            top_career = top_careers[0] if top_careers else "career"
            return f"Aap ke interest aur skills ke hisab se aap {top_career} ki field mein career banana chahiye. Aap ka passion aur hardworking attitude is field mein success guarantee karta hai."
    
    else:
        # English responses
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['medicine', 'doctor', 'nurse', 'health', 'medical']):
            return f"Based on your interests and skills, you should build a career in the medical field. You can become a doctor, nurse, or pharmacist. Your helping nature and hardworking attitude is perfect for this field."
        
        elif any(word in input_lower for word in ['computer', 'programming', 'tech', 'software', 'coding']):
            return f"Based on your interests and skills, you should build a career in the technology field. You can become a software engineer, web developer, or data scientist. Your technical interest is perfect for this field."
        
        elif any(word in input_lower for word in ['business', 'marketing', 'finance', 'money', 'company']):
            return f"Based on your interests and skills, you should build a career in the business field. You can become a business analyst, marketing manager, or entrepreneur. Your business sense is perfect for this field."
        
        elif any(word in input_lower for word in ['art', 'design', 'creative', 'music', 'writing']):
            return f"Based on your interests and skills, you should build a career in the creative field. You can become a graphic designer, writer, or artist. Your creative talent is perfect for this field."
        
        elif any(word in input_lower for word in ['teaching', 'education', 'school', 'student']):
            return f"Based on your interests and skills, you should build a career in the education field. You can become a teacher, professor, or educational administrator. Your teaching passion is perfect for this field."
        
        elif any(word in input_lower for word in ['engineering', 'mechanical', 'electrical', 'civil', 'construction']):
            return f"Based on your interests and skills, you should build a career in the engineering field. You can become a mechanical engineer, civil engineer, or electrical engineer. Your technical interest is perfect for this field."
        
        elif any(word in input_lower for word in ['science', 'research', 'laboratory', 'scientist', 'chemistry', 'physics', 'biology']):
            return f"Based on your interests and skills, you should build a career in the science and research field. You can become a research scientist, chemist, or biologist. Your analytical mind is perfect for this field."
        
        elif any(word in input_lower for word in ['law', 'legal', 'lawyer', 'justice', 'court', 'legal advice']):
            return f"Based on your interests and skills, you should build a career in the legal field. You can become a lawyer, judge, or legal consultant. Your sense of justice is perfect for this field."
        
        elif any(word in input_lower for word in ['police', 'security', 'law enforcement', 'detective', 'police officer']):
            return f"Based on your interests and skills, you should build a career in the law enforcement field. You can become a police officer, detective, or security specialist. Your sense of duty is perfect for this field."
        
        elif any(word in input_lower for word in ['agriculture', 'farming', 'crops', 'animals']):
            return f"Based on your interests and skills, you should build a career in the agriculture field. You can become an agricultural engineer, farm manager, or food scientist. Your love for nature is perfect for this field."
        
        elif any(word in input_lower for word in ['transportation', 'driving', 'pilot', 'aviation', 'logistics', 'transport', 'flying']):
            return f"Based on your interests and skills, you should build a career in the transportation field. You can become a pilot, truck driver, or logistics coordinator. Your love for travel is perfect for this field."
        
        elif any(word in input_lower for word in ['sports', 'fitness', 'gym', 'athlete', 'coaching', 'exercise']):
            return f"Based on your interests and skills, you should build a career in the sports and fitness field. You can become a personal trainer, sports coach, or fitness instructor. Your passion for fitness is perfect for this field."
        
        elif any(word in input_lower for word in ['media', 'journalism', 'news', 'broadcasting', 'communication']):
            return f"Based on your interests and skills, you should build a career in the media and communication field. You can become a journalist, news anchor, or media producer. Your communication skills are perfect for this field."
        
        elif any(word in input_lower for word in ['hospitality', 'hotel', 'restaurant', 'tourism', 'travel', 'customer service']):
            return f"Based on your interests and skills, you should build a career in the hospitality field. You can become a hotel manager, chef, or travel agent. Your customer service skills are perfect for this field."
        
        else:
            top_career = top_careers[0] if top_careers else "career"
            return f"Based on your interests and skills, you should build a career in the {top_career} field. Your passion and hardworking attitude guarantees success in this field."

@app.route('/career_advice', methods=['POST'])
def career_advice():
    try:
        data = request.get_json()
        
        if not data or 'user_input' not in data:
            return jsonify({
                'error': 'user_input parameter is required',
                'success': False
            }), 400
        
        user_input = data['user_input']
        
        if not user_input.strip():
            return jsonify({
                'error': 'user_input cannot be empty',
                'success': False
            }), 400
        
        # Detect language
        language = ml_engine.detect_language(user_input)
        
        # Translate if needed
        if language == "roman_urdu":
            processed_input = ml_engine.translate_roman_urdu_to_english(user_input)
        else:
            processed_input = user_input
        
        # Get ML-based recommendations
        ml_recommendations = ml_engine.get_ml_recommendations(processed_input, top_k=50)
        
        # Extract just the career names for response
        career_names = [rec['career'] for rec in ml_recommendations]
        
        # Generate simple response
        simple_response = generate_simple_response(user_input, career_names[:3], language)
        
        return jsonify({
            'success': True,
            'language_detected': language,
            'message': simple_response,
            'career_recommendations': career_names[:10],  # Show only top 10
            'total_recommendations': len(career_names)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500

@app.route('/career_advice_detailed', methods=['POST'])
def career_advice_detailed():
    """Enhanced endpoint with clustering and detailed ML analysis"""
    try:
        data = request.get_json()
        
        if not data or 'user_input' not in data:
            return jsonify({
                'error': 'user_input parameter is required',
                'success': False
            }), 400
        
        user_input = data['user_input']
        
        if not user_input.strip():
            return jsonify({
                'error': 'user_input cannot be empty',
                'success': False
            }), 400
        
        # Detect language
        language = ml_engine.detect_language(user_input)
        
        # Translate if needed
        if language == "roman_urdu":
            processed_input = ml_engine.translate_roman_urdu_to_english(user_input)
        else:
            processed_input = user_input
        
        # Get ML-based recommendations
        ml_recommendations = ml_engine.get_ml_recommendations(processed_input, top_k=50)
        
        # Cluster similar careers
        clustered_recommendations = ml_engine.cluster_careers(ml_recommendations)
        
        # Generate simple response
        career_names = [rec['career'] for rec in ml_recommendations]
        simple_response = generate_simple_response(user_input, career_names[:3], language)
        
        return jsonify({
            'success': True,
            'language_detected': language,
            'message': simple_response,
            'total_recommendations': len(ml_recommendations),
            'clustered_recommendations': clustered_recommendations
        })
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'AI/ML Career Advice API is running',
        'models_loaded': {
            'custom_tfidf': 'Word frequency analysis',
            'semantic_similarity': 'Word overlap and context matching',
            'clustering': 'Category-based grouping available'
        }
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'AI/ML Career Advice API',
        'endpoints': {
            'POST /career_advice': 'Get AI-powered career recommendations',
            'POST /career_advice_detailed': 'Get detailed ML analysis with clustering',
            'GET /health': 'Health check endpoint'
        },
        'ai_features': {
            'language_detection': 'English/Roman Urdu',
            'semantic_analysis': 'Custom TF-IDF + Word Overlap',
            'keyword_extraction': 'Word frequency analysis',
            'similarity_matching': 'Semantic similarity scoring',
            'career_clustering': 'Category-based grouping',
            'ml_algorithms': 'Custom ML algorithms for career matching'
        },
        'usage': {
            'method': 'POST',
            'url': '/career_advice',
            'body': {
                'user_input': 'Your interests, hobbies, or skills (English or Roman Urdu)'
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
