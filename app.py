import os
import logging
import time
import pandas as pd
import numpy as np
import json
import re
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import random

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

class EnhancedVibeSearcher:
    """Enhanced vibe searcher with aesthetic understanding."""
    
    def __init__(self):
        self.aesthetic_vocabulary = {}
        self.cultural_references = {}
        self.load_aesthetic_knowledge()
    
    def load_aesthetic_knowledge(self):
        """Load aesthetic vocabulary and cultural references."""
        try:
            if os.path.exists('aesthetic_vocabulary.json'):
                with open('aesthetic_vocabulary.json', 'r') as f:
                    self.aesthetic_vocabulary = json.load(f)
                logger.info("✅ Loaded aesthetic vocabulary")
            else:
                self.aesthetic_vocabulary = {
                    'dark academia': ['vintage', 'classic', 'scholarly', 'tweed', 'library', 'books', 'gothic', 'romantic academia'],
                    'cottagecore': ['rural', 'pastoral', 'vintage floral', 'cozy', 'handmade', 'rustic', 'cottage', 'farmhouse'],
                    'y2k': ['futuristic', 'metallic', 'cyber', 'holographic', 'tech', 'millennium', 'digital', 'space age'],
                    'minimalist': ['clean', 'simple', 'neutral', 'basic', 'essential', 'uncluttered', 'modern', 'sleek'],
                    'grunge': ['edgy', 'distressed', 'alternative', 'punk', 'rebellious', 'vintage denim', 'band tees'],
                    'soft girl': ['pastel', 'cute', 'kawaii', 'blush', 'feminine', 'dreamy', 'innocent', 'sweet'],
                    'indie sleaze': ['vintage', 'thrifted', 'eclectic', 'artsy', 'bohemian', 'creative', 'unconventional'],
                    'coquette': ['feminine', 'romantic', 'delicate', 'bows', 'lace', 'girly', 'pretty', 'dainty']
                }
                logger.info("⚠️ Using fallback aesthetic vocabulary")
                
            self.cultural_references = {
                'priyanka chopra barfi': ['vintage', 'retro bollywood', 'classic indian', 'colorful traditional'],
                'taylor swift folklore': ['cottagecore', 'indie folk', 'nature', 'cozy cabin', 'cardigans'],
                'audrey hepburn': ['classic elegant', 'vintage chic', 'timeless', 'little black dress'],
                'french girl': ['effortless chic', 'minimalist', 'neutral tones', 'classic pieces'],
                'rachel green friends': ['90s preppy', 'layered looks', 'casual chic', 'vintage 90s']
            }
            
        except Exception as e:
            logger.error(f"Error loading aesthetic knowledge: {e}")
    
    def expand_query(self, query: str) -> str:
        """Expand query with aesthetic and cultural understanding."""
        query_lower = query.lower()
        expanded_terms = [query]
        
        for aesthetic, keywords in self.aesthetic_vocabulary.items():
            if aesthetic in query_lower:
                expanded_terms.extend(keywords[:3])
        
        for reference, keywords in self.cultural_references.items():
            if any(word in query_lower for word in reference.split()):
                expanded_terms.extend(keywords)
        
        if any(word in query_lower for word in ['movie', 'character', 'era', 'style of']):
            expanded_terms.extend(['vintage', 'classic', 'iconic', 'aesthetic'])
        
        return ' '.join(expanded_terms)

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

def _get_top_score(ml_results: List[Dict[str, Any]]) -> float:
    if not ml_results:
        return 0.0
    return max([r['similarity_score'] for r in ml_results])

def _run_enhanced_ml_search(request: VibeSearchRequest):
    """
    Enhanced ML-based semantic search with aesthetic understanding.
    """
    try:
        model = MODELS['sentence_model']
        product_embeddings = MODELS['product_embeddings']
        df_products = MODELS['df_products']
        vibe_searcher = MODELS['vibe_searcher']

        expanded_query = vibe_searcher.expand_query(request.vibe)
        logger.info(f"🔍 Original query: '{request.vibe}' → Expanded: '{expanded_query}'")

        query_embedding = model.encode([expanded_query], convert_to_tensor=True).detach().cpu().numpy()
        similarities = cosine_similarity(query_embedding, product_embeddings)
        
        filtered_indices = []
        for idx in range(len(df_products)):
            product = df_products.iloc[idx]
            
            if request.price_min is not None and product['price'] < request.price_min:
                continue
            if request.price_max is not None and product['price'] > request.price_max:
                continue
            
            if request.category and request.category.lower() != 'all' and product['category'].lower() != request.category.lower():
                continue
                
            filtered_indices.append(idx)
        
        if not filtered_indices:
            logger.warning("⚠️ ML search returned no products after filtering.")
            return []
        
        filtered_similarities = similarities[0, filtered_indices]
        top_n = min(request.max_results, len(filtered_indices))
        top_indices_in_filtered = np.argsort(filtered_similarities)[-top_n:][::-1]
        top_indices = [filtered_indices[i] for i in top_indices_in_filtered]
        
        results = []
        for index in top_indices:
            product_info = df_products.iloc[index].to_dict()
            product_info['similarity_score'] = float(similarities[0][index])
            results.append(product_info)
        
        return results
    except Exception as e:
        logger.error(f"❌ Enhanced ML Search Error: {str(e)}")
        return None

def _run_enhanced_gemini_search(request: VibeSearchRequest, products: list):
    """
    Enhanced Gemini API search with better cultural understanding.
    """
    try:
        if not GEMINI_AVAILABLE or not model_gemini:
            return None
        
        filtered_products = [
            p for p in products
            if (request.price_min is None or p['price'] >= request.price_min) and
               (request.price_max is None or p['price'] <= request.price_max) and
               (not request.category or request.category.lower() == 'all' or p['category'].lower() == request.category.lower())
        ]
        
        if not filtered_products:
            logger.info("ℹ️ Filters returned no products. Gemini search cannot proceed.")
            return []
        
        sample_size = min(50, len(filtered_products))
        if len(filtered_products) >= sample_size:
            sampled_products = random.sample(filtered_products, sample_size)
        else:
            sampled_products = filtered_products
        
        product_descriptions = "\n".join([
            f"ID: {p['id']}, Title: {p['title']}, Description: {p.get('description', '')[:100]}, "
            f"Category: {p['category']}, Vibe Tags: {p.get('vibe_tags', '')}"
            for p in sampled_products
        ])
        
        prompt = f"""
        User is searching for: "{request.vibe}"
        
        This could be:
        - A fashion aesthetic (like "dark academia", "cottagecore")
        - A cultural reference (like "Priyanka Chopra in Barfi", "Taylor Swift folklore era")
        - A movie/character style reference
        - A lifestyle or personality-based fashion query
        
        From the following products, identify the top {min(request.max_results, len(sampled_products))} that best match this vibe/aesthetic:
        
        {product_descriptions}
        
        Consider:
        - Visual aesthetics and style elements
        - Color palettes and patterns
        - Cultural and historical context
        - Lifestyle associations
        - Emotional/mood connections
        
        Respond with ONLY the Product IDs (numbers) separated by commas, ordered by relevance.
        Example: 123, 456, 789
        """
        
        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip().replace(' ', '')
        
        product_ids = [int(id_str) for id_str in re.findall(r'\b\d+\b', response_text)]
        
        products_dict = {p['id']: p for p in sampled_products}
        matched_products = []
        
        for pid in product_ids:
            if pid in products_dict:
                product = products_dict[pid].copy()
                product['similarity_score'] = 0.8
                matched_products.append(product)
        
        return matched_products[:request.max_results]
        
    except Exception as e:
        logger.error(f"❌ Enhanced Gemini Search Error: {str(e)}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the enhanced ML model and product data on application startup.
    """
    logger.info("🚀 Starting Enhanced Vibe Search API...")
    
    enhanced_data_file = 'products_enhanced_corpus.csv'
    enhanced_embeddings_file = 'product_embeddings_enhanced.npy'
    fallback_data_file = 'products_with_corpus.csv'
    fallback_embeddings_file = 'product_embeddings.npy'
    
    try:
        if os.path.exists(enhanced_data_file) and os.path.exists(enhanced_embeddings_file):
            logger.info("📊 Loading enhanced model data...")
            MODELS['df_products'] = pd.read_csv(enhanced_data_file)
            MODELS['product_embeddings'] = np.load(enhanced_embeddings_file)
            corpus_column = 'enhanced_corpus'
            logger.info("✅ Enhanced model data loaded successfully")
        
        elif os.path.exists(fallback_data_file) and os.path.exists(fallback_embeddings_file):
            logger.info("📊 Loading fallback model data...")
            MODELS['df_products'] = pd.read_csv(fallback_data_file)
            MODELS['product_embeddings'] = np.load(fallback_embeddings_file)
            corpus_column = 'corpus'
            logger.info("⚠️ Using fallback model data")
        
        else:
            raise FileNotFoundError("No model data files found")
        
        MODELS['sentence_model'] = SentenceTransformer('vibe_search_model_tuned')
        MODELS['vibe_searcher'] = EnhancedVibeSearcher()
        MODELS['products_list'] = MODELS['df_products'].to_dict('records')
        
        logger.info(f"✅ Loaded {len(MODELS['df_products'])} products with enhanced search capabilities")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model data: {str(e)}")
        MODELS.clear()
    
    yield
    
    logger.info("🛑 Shutting down Enhanced Vibe Search API...")
    MODELS.clear()

app = FastAPI(
    title="Enhanced Vibe Search API", 
    version="5.0.0",
    description="An enhanced hybrid search engine with ML and Gemini fallback, featuring aesthetic and cultural understanding.",
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
    Performs an enhanced hybrid search with aesthetic and cultural understanding.
    """
    search_start_time = time.time()
    fallback_threshold = 0.25

    ml_results = None
    if 'sentence_model' in MODELS and 'product_embeddings' in MODELS:
        logger.info(f"🔍 Searching for vibe: '{request.vibe}' with ML model...")
        ml_results = _run_enhanced_ml_search(request)
    
    if ml_results is not None:
        top_score = _get_top_score(ml_results)
        logger.info(f"📊 ML search top score: {top_score:.3f} (threshold: {fallback_threshold})")
        
        if top_score > fallback_threshold:
            total_time = round((time.time() - search_start_time) * 1000, 2)
            logger.info(f"✅ Enhanced ML Search successful. Time: {total_time}ms")
            return {
                "products": ml_results,
                "message": f"Found {len(ml_results)} matches using Enhanced ML Search!",
                "search_method": "Enhanced ML Semantic Search",
                "search_time_ms": total_time,
                "top_similarity_score": top_score
            }

    if ml_results is None:
        fallback_reason = "ML model or data not loaded"
    else:
        fallback_reason = f"ML score too low: {_get_top_score(ml_results):.3f}"
        
    logger.warning(f"⚠️ ML search failed or low confidence. Falling back to Enhanced Gemini. Reason: {fallback_reason}")
    
    gemini_results = _run_enhanced_gemini_search(request, MODELS.get('products_list', []))
    
    total_time = round((time.time() - search_start_time) * 1000, 2)
    
    if gemini_results:
        logger.info(f"✅ Enhanced Gemini Search successful. Time: {total_time}ms")
        return {
            "products": gemini_results,
            "message": f"Found {len(gemini_results)} matches using Enhanced Gemini Search!",
            "search_method": "Enhanced Gemini Cultural Search",
            "search_time_ms": total_time,
            "fallback_reason": fallback_reason
        }
    
    return {
        "products": [],
        "message": "No products found. Try a different query or check your filters!",
        "search_method": "Failed",
        "search_time_ms": total_time,
        "debug_info": f"ML search failed. Gemini fallback also failed. Reason: {fallback_reason}"
    }

@app.get("/debug/search/{query}")
def debug_search(query: str):
    """Debug endpoint to test search with detailed information."""
    if 'vibe_searcher' not in MODELS:
        return {"error": "Vibe searcher not loaded"}
    
    vibe_searcher = MODELS['vibe_searcher']
    expanded_query = vibe_searcher.expand_query(query)
    
    request = VibeSearchRequest(vibe=query, max_results=5)
    
    debug_info = {
        "original_query": query,
        "expanded_query": expanded_query,
        "ml_available": 'sentence_model' in MODELS,
        "gemini_available": GEMINI_AVAILABLE
    }
    
    if 'sentence_model' in MODELS:
        ml_results = _run_enhanced_ml_search(request)
        if ml_results:
            debug_info["ml_top_score"] = max([r['similarity_score'] for r in ml_results])
            debug_info["ml_results_count"] = len(ml_results)
            debug_info["ml_top_3_titles"] = [r['title'] for r in ml_results[:3]]
    
    if GEMINI_AVAILABLE:
        gemini_results = _run_enhanced_gemini_search(request, MODELS.get('products_list', []))
        if gemini_results:
            debug_info["gemini_results_count"] = len(gemini_results)
            debug_info["gemini_top_3_titles"] = [r['title'] for r in gemini_results[:3]]
    
    return debug_info

@app.get("/evaluate/test-queries")
def evaluate_test_queries():
    """Evaluate the system on a set of test queries."""
    if 'sentence_model' not in MODELS:
        return {"error": "Model not loaded"}
    
    test_queries = [
        "priyanka chopra in barfi",
        "cottagecore aesthetic", 
        "dark academia style",
        "taylor swift folklore era",
        "french girl minimalist",
        "90s grunge outfit",
        "audrey hepburn breakfast at tiffanys",
        "soft girl kawaii",
        "y2k futuristic",
        "indie sleaze vintage"
    ]
    
    results = {}
    
    for query in test_queries:
        request = VibeSearchRequest(vibe=query, max_results=3)
        
        ml_results = _run_enhanced_ml_search(request)
        ml_score = max([r['similarity_score'] for r in ml_results]) if ml_results else 0
        
        results[query] = {
            "ml_top_score": ml_score,
            "ml_results_count": len(ml_results) if ml_results else 0,
            "ml_top_result": ml_results[0]['title'] if ml_results else "No results"
        }
    
    avg_score = np.mean(list(results.values()))
    
    return {
        "results": results,
        "average_ml_score": avg_score,
        "fallback_threshold": 0.25,
        "queries_above_threshold": sum(1 for r in results.values() if r['ml_top_score'] > 0.25)
    }

@app.get("/")
def root():
    return {
        "message": "Enhanced Vibe Search API v5.0 - Cultural & Aesthetic Understanding",
        "status": "running",
        "search_method": "Enhanced ML + Enhanced Gemini",
        "features": ["Cultural References", "Aesthetic Vocabulary", "Query Expansion"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": 'sentence_model' in MODELS,
        "enhanced_data_loaded": 'vibe_searcher' in MODELS,
        "gemini_available": GEMINI_AVAILABLE,
        "products_count": len(MODELS.get('df_products', [])),
        "timestamp": time.time()
    }

@app.get("/trending")
def get_trending():
    trending_vibes = [
        "dark academia", "cottagecore", "Y2K", "minimalist", 
        "grunge", "soft girl", "indie sleaze", "coquette",
        "priyanka chopra barfi", "taylor swift folklore", "french girl aesthetic"
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
    
    Output the questions in a JSON array format, with each object having the following keys:
    - 'question' (string): The quiz question.
    - 'options' (array of strings): The two multiple-choice options.
    - 'correct_answer' (string): An empty string.
    
    Make sure the output is valid, parsable JSON.
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

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

    Output the recommendation in a JSON object with the following keys:
    - 'primary_vibe' (string): The main fashion vibe.
    - 'secondary_vibe' (string): The secondary fashion vibe.
    - 'reasoning' (string): A single sentence explaining the primary vibe.
    
    Make sure the output is valid, parsable JSON.
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        recommendation = json.loads(response_text)
        return recommendation
    except Exception as e:
        logger.error(f"Error getting quiz recommendation from Gemini: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quiz recommendation.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)