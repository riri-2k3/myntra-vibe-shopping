import os
import logging
import time
import pandas as pd
import numpy as np
import json
import re
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class VibeSearchRequest(BaseModel):
    vibe: str
    max_results: Optional[int] = 12
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category: Optional[str] = None

class Product(BaseModel):
    id: int
    title: str
    description: str = ""
    price: float
    image_url: str
    category: str
    vibe_tags: str = ""
    rating: float = Field(default=4.0, ge=0, le=5)
    reviews_count: int = Field(default=0, ge=0)
    similarity_score: float = Field(default=0.0, ge=0, le=1)

class QuizAnswersRequest(BaseModel):
    answers: Dict[str, str]

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    
class QuizQuestionsResponse(BaseModel):
    questions: List[QuizQuestion]

class QuizRecommendationResponse(BaseModel):
    primary_vibe: str
    secondary_vibe: str
    reasoning: str

# Configure Gemini AI with enhanced error handling
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
model_gemini = None

if GEMINI_API_KEY:
    try:
        logger.info("Attempting to configure Gemini AI...")
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel('gemini-1.5-flash-latest')

        test_response = model_gemini.generate_content("Test connection. Respond with 'OK'.")
        if test_response and test_response.text:
            GEMINI_AVAILABLE = True
            logger.info("✅ Gemini AI configured and tested successfully")
        else:
            logger.warning("⚠️ Gemini AI test failed - no response received")

    except Exception as e:
        logger.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        GEMINI_AVAILABLE = False
else:
    logger.warning("❌ GEMINI_API_KEY not found in environment variables")

# Global dictionary to store models and data
MODELS = {}

def _run_ml_search(request: VibeSearchRequest):
    """
    Executes the local ML-based semantic search.
    """
    try:
        model = MODELS['sentence_model']
        product_embeddings = MODELS['product_embeddings']
        df_products = MODELS['df_products']

        query_embedding = model.encode([request.vibe], convert_to_tensor=True).detach().numpy()
        similarities = cosine_similarity(query_embedding, product_embeddings)
        top_n_indices = np.argsort(similarities[0])[-request.max_results:][::-1]
        
        results = []
        for index in top_n_indices:
            product_info = df_products.iloc[index].to_dict()
            product_info['similarity_score'] = float(similarities[0][index])
            results.append(product_info)
        
        return results
    except Exception as e:
        logger.error(f"❌ ML Search Error: {str(e)}")
        return None

def _run_gemini_search(request: VibeSearchRequest, products: list):
    """
    Executes the Gemini API fallback search.
    """
    try:
        if not GEMINI_AVAILABLE or not model_gemini:
            return None
        
        # New, more descriptive prompt for Gemini
        product_descriptions = "\n".join([
            f"Product ID: {p['id']}, Title: {p['title']}, Description: {p['description']}" for p in products
        ])
        
        prompt = f"""
        Given the user's vibe: '{request.vibe}'.
        From the following list of products, identify which ones best match the vibe. 
        Product list:
        {product_descriptions}
        
        Respond with ONLY a comma-separated list of the Product IDs that match the vibe. 
        If no products match, respond with an empty string.
        """
        
        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip().replace(' ', '')
        
        # Use a more robust way to parse the response
        if response_text:
            product_ids = [int(id_str) for id_str in re.findall(r'\b\d+\b', response_text)]
        else:
            product_ids = []
        
        products_dict = {p['id']: p for p in products}
        matched_products = [products_dict[pid] for pid in product_ids if pid in products_dict]
        
        return matched_products
    except Exception as e:
        logger.error(f"❌ Gemini Fallback Error: {str(e)}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the ML model and product embeddings on application startup.
    This ensures they are only loaded once and are available for all requests.
    """
    logger.info("🚀 Starting Vibe Search API with ML model loading...")
    
    data_file = 'products_with_corpus.csv'
    embeddings_file = 'product_embeddings.npy'
    
    if not os.path.exists(data_file) or not os.path.exists(embeddings_file):
        logger.error("❌ ML search files not found. The search feature will not work.")
    else:
        MODELS['df_products'] = pd.read_csv(data_file)
        MODELS['products_list'] = MODELS['df_products'].to_dict('records')
        MODELS['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        MODELS['product_embeddings'] = np.load(embeddings_file)
        logger.info("✅ ML Model and embeddings loaded successfully.")
    
    yield
    
    logger.info("🛑 Shutting down Vibe Search API...")
    MODELS.clear()

app = FastAPI(
    title="Vibe Search API", 
    version="4.0.0",
    description="A hybrid search engine with ML and Gemini fallback.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search/vibe-hybrid", response_model=Dict[str, Any])
def vibe_search_hybrid(request: VibeSearchRequest):
    """
    Performs a hybrid search, using the local ML model with a fallback to Gemini.
    """
    search_start_time = time.time()
    fallback_threshold = 0.4 # Tune this value based on your data

    if 'sentence_model' in MODELS and 'product_embeddings' in MODELS:
        ml_results = _run_ml_search(request)
        if ml_results and ml_results[0]['similarity_score'] > fallback_threshold:
            total_time = round((time.time() - search_start_time) * 1000, 2)
            logger.info(f"✅ ML Search successful. Method: ML. Time: {total_time}ms")
            return {
                "products": ml_results,
                "message": f"Found {len(ml_results)} matches using ML!",
                "search_method": "ML Semantic Search",
                "search_time_ms": total_time
            }

    total_time = round((time.time() - search_start_time) * 1000, 2)
    logger.warning("⚠️ ML confidence too low or model not loaded. Falling back to Gemini.")
    gemini_results = _run_gemini_search(request, MODELS.get('products_list', []))
    
    if gemini_results:
        logger.info(f"✅ Gemini Fallback successful. Method: Gemini. Time: {total_time}ms")
        return {
            "products": gemini_results,
            "message": f"Found {len(gemini_results)} matches using Gemini Fallback!",
            "search_method": "Gemini Fallback",
            "search_time_ms": total_time
        }
    
    return {
        "products": [],
        "message": "No products found. Try a different query!",
        "search_method": "Failed",
        "search_time_ms": round((time.time() - search_start_time) * 1000, 2)
    }

# --- Other Endpoints (as before) ---
@app.get("/")
def root():
    return {
        "message": "Vibe Search API v4.0 - A Hybrid Search System",
        "status": "running",
        "search_method": "ML + Gemini"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": 'sentence_model' in MODELS,
        "gemini_available": GEMINI_AVAILABLE,
        "timestamp": time.time()
    }

@app.get("/trending")
def get_trending():
    trending_vibes = [
        "dark academia", "cottagecore", "Y2K", "minimalist", 
        "grunge", "soft girl", "indie sleaze", "coquette"
    ]
    return {"trending_vibes": trending_vibes, "status": "success"}

@app.get("/categories")
def get_categories():
    if 'df_products' in MODELS:
        categories = MODELS['df_products']['category'].unique().tolist()
        return {"categories": sorted(categories)}
    return {"categories": []}

@app.get("/quiz/generate", response_model=QuizQuestionsResponse)
def generate_quiz_questions():
    """
    Generates a set of 5 quiz questions using the Gemini API.
    """
    if not GEMINI_AVAILABLE or not model_gemini:
        raise HTTPException(status_code=503, detail="Gemini API is not available.")

    prompt = """
    Create a 5-question multiple-choice quiz about clothing styles and aesthetics.
    Each question should have exactly two options.
    The options should be distinct and represent different fashion vibes.
    The quiz should help determine a user's primary and secondary fashion vibe.
    Do NOT include a correct answer. The options are preference-based.
    
    Format the response as a JSON array of objects, with each object having the following keys:
    - 'question' (string)
    - 'options' (array of two strings)
    - 'correct_answer' (string, can be empty)
    
    Example JSON response:
    [
      {"question": "What's your ideal weekend setting?", "options": ["A cozy cabin in the woods", "A high-energy city festival"], "correct_answer": ""},
      {"question": "Choose a color palette.", "options": ["Earthy tones and pastels", "Bold and neon colors"], "correct_answer": ""}
    ]
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        quiz_data = json.loads(response_text)
        return {"questions": quiz_data}
    except Exception as e:
        logger.error(f"Error generating quiz questions with Gemini: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions.")

@app.post("/quiz/recommendation", response_model=QuizRecommendationResponse)
def quiz_recommendation(request: QuizAnswersRequest):
    """
    Generates a fashion vibe recommendation based on quiz answers using the Gemini API.
    """
    if not GEMINI_AVAILABLE or not model_gemini:
        raise HTTPException(status_code=503, detail="Gemini API is not available.")

    answers_str = json.dumps(request.answers)
    prompt = f"""
    Analyze the following quiz answers to determine a user's primary and secondary fashion vibe.
    Also, provide a short, concise reasoning for the primary vibe.

    Quiz Answers: {answers_str}

    Primary vibes to choose from: dark academia, cottagecore, Y2K, minimalist, grunge, soft girl, indie sleaze, coquette.
    Secondary vibes can be a blend of any of these.

    Format the response as a JSON object with the following keys:
    - 'primary_vibe' (string)
    - 'secondary_vibe' (string)
    - 'reasoning' (string, a single sentence explaining the primary vibe)
    
    Example JSON response:
    {{"primary_vibe": "dark academia", "secondary_vibe": "cottagecore", "reasoning": "Your preference for classic literature and muted colors points to a studious, vintage aesthetic."}}
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        recommendation = json.loads(response_text)
        return recommendation
    except Exception as e:
        logger.error(f"Error getting quiz recommendation from Gemini: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quiz recommendation.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)