import os
import logging
import time
import pandas as pd
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models for request and response data validation
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
    vibe_tags: List[str] = Field(default_factory=list)
    rating: float = Field(default=4.0, ge=0, le=5)
    reviews_count: int = Field(default=0, ge=0)

# Configure Gemini AI with enhanced error handling
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
model = None

if GEMINI_API_KEY:
    try:
        logger.info("Attempting to configure Gemini AI...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        test_response = model.generate_content("Test connection. Respond with 'OK'.")
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

# Global products list
products_list: List[Dict[str, Any]] = []

# Vibe matching keywords for enhanced fallback search (static, but still useful)
VIBE_KEYWORDS = {
    'dark academia': ['tweed', 'plaid', 'oxford'],
    'cottagecore': ['floral', 'lace', 'gingham'],
    'grunge': ['black', 'leather', 'ripped'],
    'Y2K': ['metallic', 'holographic', 'butterfly'],
    'minimalist': ['clean', 'simple', 'white'],
    'coquette': ['bow', 'pink', 'lace'],
    'soft girl': ['pastel', 'cute', 'kawaii'],
    'indie sleaze': ['silver', 'metallic', 'mesh'],
    'desi': ['kurta', 'kurti', 'saree'],
    'office siren': ['blazer', 'trousers', 'formal'],
    'boho chic': ['bohemian', 'flowy', 'tassels'],
    'streetwear': ['hoodie', 'sneakers', 'cargo pants']
}

def load_products():
    """Load products from a CSV file, with fallbacks for missing data."""
    try:
        logger.info("Loading products from CSV...")
        df = pd.read_csv('products_enhanced_corpus.csv')
        logger.info(f"Successfully read CSV with {len(df)} rows")
        
        # Clean and validate data types
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(4.0)
        df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0).astype(int)
        
        # Fill missing string/url values
        df = df.fillna({
            'title': 'Unknown Product',
            'description': '',
            'image_url': 'https://via.placeholder.com/400x500/FF3F6C/FFFFFF?text=Product+Image',
            'category': 'uncategorized',
            'vibe_tags': ''
        })

        # Process vibe_tags from comma-separated string to list
        df['vibe_tags'] = df['vibe_tags'].apply(
            lambda x: [tag.strip() for tag in str(x).split(',')] if pd.notna(x) else []
        )
        
        products = df.to_dict('records')
        logger.info(f"✅ Successfully processed {len(products)} products")
        return products
    
    except FileNotFoundError:
        logger.error("❌ products_enhanced_corpus.csv not found. Creating sample data...")
        return create_sample_products()
    except Exception as e:
        logger.error(f"❌ Error loading products: {str(e)}. Creating sample data...")
        return create_sample_products()

def create_sample_products():
    """Create sample products if CSV loading fails."""
    logger.info("Creating sample products...")
    return [
        {
            "id": 1, "title": "Classic Tweed Blazer", "description": "Academic-inspired wool tweed blazer in rich brown",
            "price": 5499.0, "image_url": "https://images.unsplash.com/photo-1594633313593-bab3825d0caf?w=400",
            "category": "blazers", "vibe_tags": ["dark academia", "scholarly", "tweed"], "rating": 4.6, "reviews_count": 342
        },
        {
            "id": 2, "title": "Floral Smocked Midi Dress", "description": "Flowing midi dress with hand-smocked bodice and delicate rose print",
            "price": 4599.0, "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400",
            "category": "dresses", "vibe_tags": ["cottagecore", "smocked", "floral"], "rating": 4.7, "reviews_count": 324
        }
    ]

def enhanced_fallback_search(vibe: str, products: list, max_results: int = 12, ai_keywords: Optional[List[str]] = None):
    """Enhanced text matching with weighted scoring and dynamic keywords from AI."""
    start_time = time.time()
    logger.info(f"🔎 Starting enhanced fallback search for vibe: '{vibe}'")
    
    vibe_lower = vibe.lower()
    scored_products = []
    
    keywords_to_search = set()
    if ai_keywords:
        keywords_to_search.update(ai_keywords)
        logger.info(f"🤖 Using AI-generated keywords: {keywords_to_search}")
    
    for known_vibe, keywords in VIBE_KEYWORDS.items():
        if known_vibe in vibe_lower:
            keywords_to_search.update(keywords)
            logger.info(f"✨ Matched known vibe '{known_vibe}', adding preset keywords")
    
    for product in products:
        score = 0
        # Combine all searchable text fields for comprehensive matching
        searchable_text = f"{product.get('title', '')} {product.get('description', '')} {' '.join(product.get('vibe_tags', []))}".lower()
        
        # High priority for AI-generated keywords
        if ai_keywords:
            for keyword in ai_keywords:
                if keyword in searchable_text:
                    score += 15
        
        # High priority for exact phrase match
        if vibe_lower in searchable_text:
            score += 20
        
        # Match against static keywords if AI-generated ones don't exist
        for keyword in keywords_to_search:
            if keyword in searchable_text:
                score += 8
        
        # Medium priority for individual word matches
        vibe_words = vibe_lower.split()
        for word in vibe_words:
            if len(word) > 2 and word in searchable_text:
                score += 5
        
        if score > 0:
            scored_products.append((score, product))
    
    scored_products.sort(key=lambda x: x[0], reverse=True)
    results = [product for _, product in scored_products[:max_results]]
    search_time = round((time.time() - start_time) * 1000, 2)
    logger.info(f"✅ Enhanced fallback search completed in {search_time}ms, found {len(results)} matches")
    return results

def get_ai_keywords(vibe_query: str):
    """Uses Gemini to interpret a vibe query and extract keywords."""
    logger.info(f"🧠 Asking Gemini to interpret '{vibe_query}'")
    
    # This prompt is clean, direct, and focused on a single task: keyword extraction.
    prompt = f"""You are a fashion AI. Analyze the user's vibe query and break it down into a list of key fashion elements, aesthetics, and specific keywords.
User query: "{vibe_query}"
Instructions:
1. Identify the core aesthetic, celebrity reference, or theme.
2. List specific clothing items, colors, materials, and accessories that define this look.
3. Be as specific as possible.
4. Respond ONLY with a comma-separated list of keywords. DO NOT include any other text, explanations, or conversational filler.

Example:
'pariyanaka chopra barfi' -> 'vintage, retro, 1970s, cardigan, mary jane, innocent, classic, blouse, skirt, nostalgic, sweet'
'aishwarya rai dhoom' -> 'biker jacket, denim hot pants, crop top, sleek, corset, leather, bold, y2k'
'sabrina carpenter' -> 'sequin, sparkly, bodysuit, glam, stage wear, mini dress, platform, boots, concert'

Keywords:"""
    try:
        response = model.generate_content(prompt)
        if response and response.text:
            keywords = [k.strip() for k in response.text.split(',')]
            logger.info(f"✅ Gemini interpreted vibe as keywords: {keywords}")
            return keywords
    except Exception as e:
        logger.error(f"❌ Gemini keyword extraction failed: {str(e)}")
    return []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load products on startup and manage shutdown."""
    global products_list
    logger.info("🚀 Starting Vibe Search API...")
    logger.info(f"Gemini AI Status: {'✅ Available' if GEMINI_AVAILABLE else '❌ Not Available'}")
    products_list = load_products()
    if products_list:
        categories = list(set(p.get('category', 'uncategorized') for p in products_list))
        logger.info(f"📦 Loaded {len(products_list)} products across {len(categories)} categories")
        logger.info(f"Categories: {', '.join(sorted(categories))}")
    else:
        logger.error("❌ No products loaded!")
    yield
    logger.info("🛑 Shutting down Vibe Search API...")

# Initialize FastAPI app
app = FastAPI(title="Vibe Search API", version="2.0.0", description="Pinterest-style aesthetic search for fashion products", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Main API Endpoint
@app.post("/search/vibe")
def vibe_search(request: VibeSearchRequest):
    """Enhanced Pinterest-style vibe search with a dynamic, AI-driven approach."""
    search_start_time = time.time()
    logger.info(f"🔍 NEW VIBE SEARCH: '{request.vibe}'")
    
    # 1. First, apply filters to the entire product list to reduce the search space
    filtered_products = products_list.copy()
    if request.price_min is not None:
        filtered_products = [p for p in filtered_products if p['price'] >= request.price_min]
    if request.price_max is not None:
        filtered_products = [p for p in filtered_products if p['price'] <= request.price_max]
    if request.category and request.category.lower() != "all":
        filtered_products = [
            p for p in filtered_products 
            if p.get('category', '').lower() == request.category.lower()
        ]
    
    if not filtered_products:
        logger.warning("❌ No products match the applied filters.")
        return {"products": [], "message": "No products found matching your filters."}
    
    # 2. Use Gemini to get keywords from a vague query
    ai_keywords = []
    if GEMINI_AVAILABLE:
        ai_keywords = get_ai_keywords(request.vibe)
    
    # 3. Use the enhanced fallback search with the AI-generated keywords
    matched_products = enhanced_fallback_search(request.vibe, filtered_products, request.max_results, ai_keywords)
    
    total_search_time = round((time.time() - search_start_time) * 1000, 2)
    search_method = "gemini_dynamic_fallback" if GEMINI_AVAILABLE else "enhanced_fallback"
    
    result = {
        "products": matched_products, "vibe_query": request.vibe, "total_matches": len(matched_products),
        "search_method": search_method, "search_time_ms": total_search_time, "gemini_available": GEMINI_AVAILABLE
    }
    
    if matched_products:
        result["message"] = f"Found {len(matched_products)} perfect matches for '{request.vibe}'!"
        logger.info(f"✅ SEARCH COMPLETED: {len(matched_products)} matches in {total_search_time}ms using {search_method}")
    else:
        result["message"] = f"No matches found for '{request.vibe}'. Try different keywords!"
        logger.warning(f"⚠️ SEARCH COMPLETED: No matches found in {total_search_time}ms")
    
    return result

# Other Endpoints (Unchanged)
@app.get("/trending")
def get_trending():
    logger.info("📈 Fetching trending content...")
    trending_vibes = ["dark academia", "cottagecore", "Y2K", "minimalist", "grunge", "soft girl", "indie sleaze", "coquette"]
    try:
        top_products = sorted(products_list, key=lambda x: (x.get('rating', 0), x.get('reviews_count', 0)), reverse=True)[:6]
        logger.info(f"Retrieved {len(top_products)} featured products")
        return {"trending_vibes": trending_vibes, "featured_products": top_products, "status": "success"}
    except Exception as e:
        logger.error(f"❌ Error getting trending content: {str(e)}")
        return {"trending_vibes": trending_vibes, "featured_products": [], "status": "error"}

@app.get("/categories")
def get_categories():
    try:
        categories = list(set(p.get('category', 'uncategorized').lower() for p in products_list if p.get('category')))
        logger.info(f"📂 Retrieved {len(categories)} categories")
        return {"categories": sorted(categories)}
    except Exception as e:
        logger.error(f"❌ Error fetching categories: {str(e)}")
        return {"categories": []}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", "products_loaded": len(products_list) > 0,
        "gemini_available": GEMINI_AVAILABLE, "total_products": len(products_list), "timestamp": time.time()
    }
    
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Vibe Search API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)