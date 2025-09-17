import os
import logging
import time
import pandas as pd
import json
import re
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import random
import math

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables
from dotenv import load_dotenv
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

def clean_float_value(value):
    """Clean float values to ensure JSON compliance"""
    if pd.isna(value) or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 0.0
    return float(value)

def clean_product_data(products_list):
    """Clean all product data to ensure JSON compliance"""
    cleaned_products = []
    for product in products_list:
        cleaned_product = {}
        for key, value in product.items():
            if isinstance(value, float):
                cleaned_product[key] = clean_float_value(value)
            elif pd.isna(value) if hasattr(pd, 'isna') else value is None:
                cleaned_product[key] = "" if key in ['title', 'description', 'category', 'vibe_tags'] else 0
            else:
                cleaned_product[key] = value
        cleaned_products.append(cleaned_product)
    return cleaned_products

def load_products():
    """Load product data from CSV"""
    try:
        data_file = 'products_enhanced_corpus.csv'
        if os.path.exists(data_file):
            products_df = pd.read_csv(data_file)
            products_list = products_df.to_dict('records')
            products_list = clean_product_data(products_list)
            
            for i, product in enumerate(products_list):
                if 'id' not in product or pd.isna(product['id']):
                    product['id'] = i + 1
            
            logger.info(f"✅ Loaded {len(products_list)} products from CSV")
            return products_list
        else:
            raise FileNotFoundError(f"{data_file} not found")
    except Exception as e:
        logger.error(f"❌ Failed to load product data: {str(e)}")
        return []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads product data on application startup."""
    logger.info("🚀 Starting Enhanced Vibe Search API...")
    MODELS['products_list'] = load_products()
    yield
    logger.info("🛑 Shutting down Enhanced Vibe Search API...")
    MODELS.clear()

app = FastAPI(
    title="Enhanced Vibe Search API", 
    version="5.0.0",
    description="An enhanced hybrid search engine with Gemini, featuring aesthetic and cultural understanding.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def advanced_text_match(vibe: str, products: list, max_results: int = 12):
    """Advanced text matching with improved scoring"""
    vibe_lower = vibe.lower()
    vibe_words = vibe_lower.split()
    
    style_keywords = {
        'cottagecore': ['floral', 'vintage', 'romantic', 'feminine', 'natural', 'peasant', 'embroidered'],
        'dark academia': ['blazer', 'formal', 'sophisticated', 'vintage', 'intellectual', 'tweed', 'plaid'],
        'y2k': ['crop', 'platform', 'chunky', 'metallic', 'edgy', 'holographic', 'futuristic'],
        'minimalist': ['clean', 'simple', 'basic', 'neutral', 'white', 'minimal'],
        'grunge': ['oversized', 'denim', 'combat', 'flannel', 'alternative', 'distressed'],
        'boho': ['flowing', 'ethnic', 'patterns', 'kimono', 'festival', 'bohemian'],
        'coquette': ['pink', 'pearl', 'feminine', 'romantic', 'delicate', 'bow'],
        'streetwear': ['urban', 'casual', 'sporty', 'hoodie', 'sneakers', 'logo']
    }
    
    scored_products = []
    
    for product in products:
        score = 0
        searchable_text = f"{product.get('title', '')} {product.get('description', '')} {product.get('vibe_tags', '')}".lower()
        
        for word in vibe_words:
            if len(word) > 2 and word in searchable_text:
                score += 3
        
        for style, keywords in style_keywords.items():
            if style in vibe_lower:
                for keyword in keywords:
                    if keyword in searchable_text:
                        score += 4
        
        color_words = ['pastel', 'bright', 'dark', 'neutral', 'black', 'white', 'pink', 'blue', 'red', 'green', 'brown', 'navy']
        for color in color_words:
            if color in vibe_lower and color in searchable_text:
                score += 2
        
        if vibe_lower in searchable_text:
            score += 10
        
        if score > 0:
            product['similarity_score'] = score
            scored_products.append((score, product))
    
    scored_products.sort(key=lambda x: x[0], reverse=True)
    return [product for _, product in scored_products[:max_results]]

def _run_enhanced_gemini_search(request: VibeSearchRequest, products: list):
    """Enhanced Gemini API search with better cultural understanding."""
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
        sampled_products = random.sample(filtered_products, sample_size)
        
        product_descriptions = "\n".join([
            f"ID: {p['id']}, Title: {p['title']}, Description: {p.get('description', '')[:100]}, "
            f"Category: {p['category']}, Vibe Tags: {p.get('vibe_tags', '')}"
            for p in sampled_products
        ])
        
        # New, more powerful prompt
        prompt = f"""
You are an expert fashion curator specializing in aesthetic-based product discovery, like Pinterest's visual search but for fashion vibes.

User is searching for: "{vibe}"

Here are the available products:
{product_context}

Instructions:
1. Analyze the vibe "{vibe}" and understand its key aesthetic elements
2. Select the {min(max_results, 15)} product IDs that BEST match this specific vibe
3. Consider: style, colors, textures, mood, cultural references, and overall aesthetic feeling
4. Prioritize products that authentically represent the vibe over generic matches
5. For aesthetic vibes like "dark academia", "cottagecore", "grunge", etc., be very specific about matching authentic pieces

Respond with ONLY a comma-separated list of product IDs (numbers only).
Example format: 1,5,8,12,15,23,29,34,41,45,52,58

Product IDs only:"""
        
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

@app.post("/search/vibe", response_model=Dict[str, Any])
def vibe_search_hybrid(request: VibeSearchRequest):
    """
    Performs a hybrid search using the Gemini API and text matching as a fallback.
    """
    search_start_time = time.time()
    
    products_list = MODELS.get('products_list', [])
    if not products_list:
        raise HTTPException(status_code=500, detail="Product data not loaded.")

    # Try Gemini AI search first
    gemini_results = None
    if GEMINI_AVAILABLE:
        gemini_results = _run_enhanced_gemini_search(request, products_list)

    total_time = round((time.time() - search_start_time) * 1000, 2)
    
    if gemini_results and len(gemini_results) > 0:
        logger.info(f"✅ Enhanced Gemini Search successful. Time: {total_time}ms")
        return {
            "products": gemini_results,
            "message": f"AI found {len(gemini_results)} items perfectly matching your vibe! ✨",
            "search_method": "Gemini AI",
            "search_time_ms": total_time,
        }

    # Fallback to advanced text matching
    logger.info("➡️ Falling back to Advanced Text Matching.")
    text_results = advanced_text_match(request.vibe, products_list, request.max_results)
    
    total_time = round((time.time() - search_start_time) * 1000, 2)

    if text_results:
        return {
            "products": text_results,
            "message": f"Found {len(text_results)} items matching your vibe using text matching.",
            "search_method": "Text Matching Fallback",
            "search_time_ms": total_time,
        }
    
    return {
        "products": [],
        "message": "No products found. Try a different query or check your filters!",
        "search_method": "Failed",
        "search_time_ms": total_time,
    }

@app.get("/")
def root():
    return {
        "message": "Enhanced Vibe Search API v5.0 - Cultural & Aesthetic Understanding",
        "status": "running",
        "search_method": "Enhanced Gemini",
        "features": ["Cultural References", "Aesthetic Vocabulary"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gemini_available": GEMINI_AVAILABLE,
        "products_count": len(MODELS.get('products_list', [])),
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
    if 'products_list' in MODELS:
        categories = set(p.get('category') for p in MODELS['products_list'] if p.get('category'))
        return {"categories": sorted(list(categories))}
    return {"categories": []}

@app.get("/quiz/generate", response_model=QuizQuestionsResponse)
def generate_quiz_questions():
    """Generates a set of 5 quiz questions using the Gemini API."""
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
    """Generates a fashion vibe recommendation based on quiz answers using the Gemini API."""
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

@app.post("/search/image")
async def image_vibe_search(
    file: UploadFile = File(...), 
    additional_text: str = Form("")
):
    """Search products based on uploaded image + optional text using Gemini."""
    try:
        logger.info(f"📸 Processing image upload: {file.filename}")
        
        image_data = await file.read()
        image_pil = Image.open(io.BytesIO(image_data))
        
        if not GEMINI_AVAILABLE or not model_gemini:
            raise HTTPException(status_code=503, detail="Gemini AI is not available for image search.")

        # Multimodal prompt with image and text
        prompt_parts = [
            image_pil,
            f"You are a fashion stylist. Based on this image, what are the key clothing elements, colors, and overall fashion aesthetic? List the top 10 most relevant products from the following catalog that match this vibe. {additional_text}",
            "\n\nProduct Catalog (IDs, Titles, Tags):\n"
        ]
        
        products_list = MODELS.get('products_list', [])
        
        # Add products to the prompt to ground the response
        for product in random.sample(products_list, min(50, len(products_list))):
            prompt_parts.append(f"ID:{product['id']}, Title:{product['title']}, Vibe Tags:{product.get('vibe_tags', '')}\n")
        
        prompt_parts.append("\n\nRespond with only the comma-separated product IDs (no explanations): 1,3,5,2")

        response = await model_gemini.generate_content(prompt_parts, stream=False)
        
        product_ids = [int(id_str) for id_str in re.findall(r'\b\d+\b', response.text.strip())]
        
        matched_products = []
        for pid in product_ids:
            product = next((p for p in products_list if p['id'] == pid), None)
            if product:
                product['similarity_score'] = 0.9 # High score for AI-matched items
                matched_products.append(product)
        
        return {
            "products": matched_products,
            "message": f"Found {len(matched_products)} items with a similar style based on your image! 📸",
            "search_method": "Gemini AI Image Search"
        }
    
    except Exception as e:
        logger.error(f"❌ Error in image_vibe_search: {e}")
        raise HTTPException(status_code=500, detail="Error processing image search")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Myntra Vibe Shopping API...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")