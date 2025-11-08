# AI/ML Career Advice Bot API

A Flask-based AI/ML API that provides intelligent career recommendations using advanced machine learning models. The API supports both English and Roman Urdu inputs and automatically detects the language to provide appropriate responses using real AI/ML algorithms.

## Features

- **Advanced AI/ML Models**: Uses Sentence Transformers, TF-IDF, and Cosine Similarity for intelligent recommendations
- **Semantic Analysis**: Deep understanding of user input using pre-trained transformer models
- **Multi-language support**: Automatically detects and responds in English or Roman Urdu
- **Comprehensive career database**: 60+ careers with detailed descriptions for ML processing
- **Smart keyword extraction**: TF-IDF-based keyword analysis for better matching
- **Career clustering**: K-Means clustering to group similar career paths
- **Confidence scoring**: ML-based confidence scores for each recommendation
- **Real-time processing**: Fast AI inference using optimized models

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the API
```bash
python app.py
```

The API will start on `http://localhost:5000`

### API Endpoints

#### 1. Get AI-Powered Career Advice
**POST** `/career_advice`

**Request Body:**
```json
{
    "user_input": "I love programming and working with computers"
}
```

**Response:**
```json
{
    "success": true,
    "language_detected": "english",
    "message": "Based on AI/ML analysis of your interests and skills, here are 50 perfect career paths:",
    "career_recommendations": [
        "Software Engineer",
        "Data Scientist",
        "AI/ML Engineer",
        ...
    ],
    "total_recommendations": 50,
    "ai_analysis": {
        "top_recommendations_with_scores": [
            {
                "career": "Software Engineer",
                "description": "Develops software applications, writes code...",
                "confidence_score": 0.892
            }
        ],
        "ml_model_used": "Sentence Transformers + TF-IDF + Cosine Similarity",
        "processing_method": "Semantic similarity + Keyword matching"
    }
}
```

#### 2. Get Detailed ML Analysis with Clustering
**POST** `/career_advice_detailed`

Returns clustered career recommendations with detailed ML analysis.

#### 3. Health Check
**GET** `/health`

#### 4. API Information
**GET** `/`

### Example Requests

**English Input:**
```bash
curl -X POST http://localhost:5000/career_advice \
  -H "Content-Type: application/json" \
  -d '{"user_input": "I enjoy helping people and working in healthcare"}'
```

**Roman Urdu Input:**
```bash
curl -X POST http://localhost:5000/career_advice \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Mujhe computer aur programming mein interest hai"}'
```

## How It Works

1. **Language Detection**: ML-based detection of English vs Roman Urdu input
2. **Text Processing**: Roman Urdu text is translated to English equivalents for ML processing
3. **Semantic Analysis**: Sentence Transformers encode user input into high-dimensional vectors
4. **Career Embeddings**: Pre-computed embeddings for all career descriptions using the same model
5. **Similarity Calculation**: Cosine similarity between user input and career embeddings
6. **Keyword Extraction**: TF-IDF analysis to extract relevant keywords from user input
7. **Score Combination**: Weighted combination of semantic similarity (70%) and keyword matching (30%)
8. **ML Ranking**: Careers are ranked by combined ML scores
9. **Clustering (Optional)**: K-Means clustering groups similar career paths together
10. **Response Generation**: AI-generated response templates based on detected language

## AI/ML Models Used

- **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic understanding
- **TF-IDF Vectorizer**: For keyword extraction and text analysis
- **Cosine Similarity**: For measuring semantic similarity between texts
- **K-Means Clustering**: For grouping similar career paths
- **Scikit-learn**: For ML algorithms and data processing

## Career Categories

The API covers 60+ careers across these categories:
- **Technology**: Software development, AI/ML, cybersecurity, etc.
- **Healthcare**: Medical professions, therapy, research, etc.
- **Business**: Management, finance, marketing, etc.
- **Creative**: Design, arts, media, etc.
- **Education**: Teaching, research, administration, etc.
- **Engineering**: Various engineering disciplines

## Error Handling

The API includes comprehensive error handling for:
- Missing or empty user input
- Invalid JSON requests
- Server errors

## Development

To run in development mode:
```bash
python app.py
```

The API runs with debug mode enabled by default for development.
