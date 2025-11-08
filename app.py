from flask import Flask, request, jsonify
import re
import random
from typing import List, Dict
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os
from career_data import CAREER_DATABASE

app = Flask(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

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

class CareerRecommendationEngine:
    def __init__(self):
        self.career_database = CAREER_DATABASE
        self.career_names = list(CAREER_DATABASE.keys())
        self.career_descriptions = list(CAREER_DATABASE.values())
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize TF-IDF vectorizer for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Fit TF-IDF on career descriptions
        print("Computing TF-IDF matrix for career descriptions...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.career_descriptions)
        
        # Pre-compute career embeddings using TF-IDF
        print("Computing career embeddings...")
        self.career_embeddings = self.tfidf_matrix.toarray()
        
        print("AI/ML models loaded successfully!")
    
    def detect_language(self, text: str) -> str:
        """Detect if text is in English or Roman Urdu using ML-based approach"""
        # Check for Roman Urdu patterns (common Roman Urdu words/patterns)
        roman_urdu_patterns = [
            r'\b(sehat|doctor|nurse|mariz|hospital|dawa|karobar|marketing|finance|management|sales|paisa|munafa|company|fun|design|creative|music|likhna|photography|video|parhana|taleem|seekhna|school|talib_ilm|ilm|tahqeeq|engineering|mechanical|electrical|civil|tameer|imarat|machine)\b',
            r'\b(ka|ki|ke|se|mein|par|ko|ne|koi|kuch|bahut|zaroor|shayad|bilkul|nahi|haan|theek|achha|bura|naya|purana|chota|bada)\b'
        ]
        
        for pattern in roman_urdu_patterns:
            if re.search(pattern, text.lower()):
                return "roman_urdu"
        
        # If no Roman Urdu patterns found, assume English
        return "english"
    
    def translate_roman_urdu_to_english(self, text: str) -> str:
        """Convert Roman Urdu text to English equivalents using ML-enhanced translation"""
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in ROMAN_URDU_MAPPING:
                translated_words.append(ROMAN_URDU_MAPPING[clean_word])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using NLP techniques"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def extract_semantic_features(self, user_input: str) -> np.ndarray:
        """Extract semantic features from user input using TF-IDF"""
        # Preprocess the input
        processed_input = self.preprocess_text(user_input)
        
        # Transform using fitted TF-IDF
        user_tfidf = self.tfidf_vectorizer.transform([processed_input])
        return user_tfidf.toarray()[0]
    
    def calculate_semantic_similarity(self, user_input: str) -> List[float]:
        """Calculate semantic similarity between user input and career descriptions"""
        user_embedding = self.extract_semantic_features(user_input)
        
        # Calculate cosine similarity
        similarities = cosine_similarity([user_embedding], self.career_embeddings)[0]
        
        return similarities.tolist()
    
    def extract_keywords_tfidf(self, user_input: str) -> List[str]:
        """Extract relevant keywords using TF-IDF"""
        # Preprocess input
        processed_input = self.preprocess_text(user_input)
        
        # Transform user input using fitted TF-IDF
        user_tfidf = self.tfidf_vectorizer.transform([processed_input])
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get top keywords
        user_tfidf_array = user_tfidf.toarray()[0]
        top_indices = np.argsort(user_tfidf_array)[-15:][::-1]  # Top 15 keywords
        
        keywords = [feature_names[i] for i in top_indices if user_tfidf_array[i] > 0]
        return keywords
    
    def calculate_keyword_similarity(self, user_input: str) -> List[float]:
        """Calculate keyword-based similarity scores"""
        user_keywords = self.extract_keywords_tfidf(user_input)
        scores = []
        
        for description in self.career_descriptions:
            description_lower = description.lower()
            score = 0
            for keyword in user_keywords:
                if keyword in description_lower:
                    score += 1
            # Normalize by number of keywords
            normalized_score = score / len(user_keywords) if user_keywords else 0
            scores.append(normalized_score)
        
        return scores
    
    def get_sentiment_score(self, text: str) -> float:
        """Get sentiment score using TextBlob"""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def get_ml_recommendations(self, user_input: str, top_k: int = 50) -> List[Dict]:
        """Get career recommendations using ML algorithms"""
        # Calculate semantic similarities using TF-IDF
        semantic_similarities = self.calculate_semantic_similarity(user_input)
        
        # Calculate keyword-based similarities
        keyword_similarities = self.calculate_keyword_similarity(user_input)
        
        # Get sentiment score
        sentiment_score = self.get_sentiment_score(user_input)
        
        # Create recommendations with scores
        recommendations = []
        for i, (career, description) in enumerate(self.career_database.items()):
            # Combine semantic similarity with keyword matching
            semantic_score = semantic_similarities[i]
            keyword_score = keyword_similarities[i]
            
            # Boost positive sentiment careers if user input is positive
            sentiment_boost = 1.0
            if sentiment_score > 0.1:  # Positive sentiment
                # Boost creative and helping careers
                if any(word in description.lower() for word in ['creative', 'help', 'care', 'art', 'design']):
                    sentiment_boost = 1.2
            
            # Combined score (weighted combination)
            combined_score = (0.6 * semantic_score + 0.4 * keyword_score) * sentiment_boost
            
            recommendations.append({
                'career': career,
                'description': description,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'sentiment_boost': sentiment_boost,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top K recommendations
        return recommendations[:top_k]
    
    def cluster_careers(self, recommendations: List[Dict], n_clusters: int = 5) -> Dict:
        """Cluster similar careers together for better organization"""
        if len(recommendations) < n_clusters:
            return {'clusters': [recommendations]}
        
        # Extract career descriptions for clustering
        career_descriptions = [rec['description'] for rec in recommendations]
        
        # Get TF-IDF embeddings for clustering
        embeddings = self.tfidf_vectorizer.transform(career_descriptions).toarray()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organize careers by clusters
        clusters = {}
        for i, rec in enumerate(recommendations):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(rec)
        
        return {'clusters': list(clusters.values())}

# Initialize the ML engine
ml_engine = CareerRecommendationEngine()

def generate_response_template(language: str) -> str:
    """Generate response template based on detected language"""
    if language == "roman_urdu":
        templates = [
            "AI/ML model ke base par, aap ke interests aur skills ke hisab se yeh 50 career paths perfect hain:",
            "Machine learning algorithm ne analyze kiya hai aur yeh career options suggest kiye hain:",
            "Aap ki profile ko AI ne analyze kiya hai aur yeh best career paths hain:",
            "Advanced AI model ke through, yeh 50 career options aap ke liye suitable hain:"
        ]
    else:
        templates = [
            "Based on AI/ML analysis of your interests and skills, here are 50 perfect career paths:",
            "Our machine learning algorithm has analyzed your profile and suggests these career options:",
            "Using advanced AI analysis, here are the best career paths for you:",
            "Our AI model has processed your input and recommends these 50 career options:"
        ]
    
    return random.choice(templates)

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
        
        # Generate response template
        response_template = generate_response_template(language)
        
        # Prepare detailed response with ML scores
        detailed_recommendations = []
        for rec in ml_recommendations[:10]:  # Show top 10 with details
            detailed_recommendations.append({
                'career': rec['career'],
                'description': rec['description'],
                'confidence_score': round(rec['combined_score'], 3),
                'semantic_score': round(rec['semantic_score'], 3),
                'keyword_score': round(rec['keyword_score'], 3)
            })
        
        return jsonify({
            'success': True,
            'language_detected': language,
            'message': response_template,
            'career_recommendations': career_names,
            'total_recommendations': len(career_names),
            'ai_analysis': {
                'top_recommendations_with_scores': detailed_recommendations,
                'ml_model_used': 'TF-IDF + Cosine Similarity + NLTK + TextBlob',
                'processing_method': 'Semantic similarity + Keyword matching + Sentiment analysis',
                'nlp_features': ['Tokenization', 'Lemmatization', 'Stopword removal', 'TF-IDF vectorization']
            }
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
        
        # Generate response template
        response_template = generate_response_template(language)
        
        return jsonify({
            'success': True,
            'language_detected': language,
            'message': response_template,
            'total_recommendations': len(ml_recommendations),
            'clustered_recommendations': clustered_recommendations,
            'ai_analysis': {
                'ml_model_used': 'TF-IDF + Cosine Similarity + K-Means + NLTK + TextBlob',
                'processing_method': 'Semantic similarity + Keyword matching + Sentiment analysis + Career clustering',
                'clustering_algorithm': 'K-Means on TF-IDF embeddings',
                'nlp_features': ['Tokenization', 'Lemmatization', 'Stopword removal', 'TF-IDF vectorization', 'Sentiment analysis']
            }
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
            'tfidf_vectorizer': 'Fitted on career descriptions',
            'nltk': 'Tokenization, stopwords, lemmatization',
            'textblob': 'Sentiment analysis',
            'clustering': 'K-Means available'
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
            'semantic_analysis': 'TF-IDF + Cosine Similarity',
            'keyword_extraction': 'TF-IDF with NLTK preprocessing',
            'sentiment_analysis': 'TextBlob sentiment scoring',
            'career_clustering': 'K-Means clustering',
            'nlp_processing': 'Tokenization, lemmatization, stopword removal'
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