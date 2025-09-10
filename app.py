# app.py - Enhanced Vibe Search API for Myntra Hackathon
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

# Configure Gemini AI with enhanced error handling
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
model = None

if GEMINI_API_KEY:
    try:
        logger.info("Attempting to configure Gemini AI...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Test the model with a simple prompt
        test_response = model.generate_content("Test connection. Respond with 'OK'.")
        if test_response and test_response.text:
            GEMINI_AVAILABLE = True
            logger.info("✅ Gemini AI configured and tested successfully")
        else:
            logger.warning("⚠️ Gemini AI test failed - no response received")

    except Exception as e:
        logger.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        GEMINI_AVAILABLE = False
else:
    logger.warning("❌ GEMINI_API_KEY not found in environment variables")

# Global products list
products_list: List[Dict[str, Any]] = []

# Enhanced vibe matching keywords for better fallback search
# In your app.py file, locate the VIBE_KEYWORDS dictionary and replace it with this:
VIBE_KEYWORDS = {
    'dark academia': ['tweed', 'plaid', 'oxford', 'burgundy', 'navy', 'brown', 'leather', 'blazer', 'scholarly', 'vintage', 'academic', 'preppy', 'structured'],
    'cottagecore': ['floral', 'lace', 'gingham', 'prairie', 'ruffles', 'cream', 'sage', 'dusty pink', 'flowing', 'romantic', 'embroidered', 'smocked'],
    'grunge': ['black', 'leather', 'ripped', 'distressed', 'flannel', 'combat boots', 'mesh', 'oversized', 'alternative', 'edgy', '90s', 'band'],
    'Y2K': ['metallic', 'holographic', 'butterfly', 'cargo', 'platform', 'chunky', 'iridescent', 'futuristic', '2000s', 'rhinestone', 'cyber'],
    'minimalist': ['clean', 'simple', 'white', 'beige', 'grey', 'structured', 'minimal', 'tailored', 'neutral', 'basic'],
    'coquette': ['bow', 'pink', 'lace', 'pearl', 'ribbon', 'ruffle', 'satin', 'romantic', 'feminine', 'sweet', 'heart', 'delicate'],
    'soft girl': ['pastel', 'cute', 'kawaii', 'pink', 'lavender', 'soft', 'sweet', 'feminine', 'oversized', 'cozy'],
    'indie sleaze': ['silver', 'metallic', 'mesh', 'sequin', 'party', 'night', 'glam', 'edgy', 'disco', '2000s'],
    'desi': ['kurta', 'kurti', 'saree', 'lehenga', 'suit', 'bangles', 'jhumka', 'earrings', 'embroidered', 'zari', 'traditional', 'ethnic'],
    'office siren': ['blazer', 'trousers', 'formal', 'tailored', 'workwear', 'professional', 'sleek', 'tote', 'watches', 'pumps', 'heels', 'structured'],
    'boho chic': ['bohemian', 'flowy', 'tassels', 'fringe', 'embroidery', 'tribal', 'earth tones', 'natural fabric', 'sandals', 'kimono', 'maxi dress'],
    'streetwear': ['hoodie', 'sneakers', 'cargo pants', 'denim jacket', 'oversized', 'urban', 'graphic tee', 'bomber jacket', 'joggers', 'street style']
}

def load_products():
    """Load products from CSV file with enhanced error handling"""
    try:
        logger.info("Loading products from CSV...")
        df = pd.read_csv('products.csv')
        logger.info(f"Successfully read CSV with {len(df)} rows")
        
        # Clean and validate data
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(4.0)
        df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0).astype(int)
        
        # Fill missing values
        df = df.fillna({
            'title': 'Unknown Product',
            'description': '',
            'image_url': 'https://via.placeholder.com/400x500/FF3F6C/FFFFFF?text=Product+Image',
            'category': 'uncategorized',
            'vibe_tags': ''
        })
        
        products = df.to_dict('records')
        logger.info(f"✅ Successfully processed {len(products)} products")
        return products
        
    except FileNotFoundError:
        logger.error("❌ products.csv not found. Creating sample data...")
        return create_sample_products()
    except Exception as e:
        logger.error(f"❌ Error loading products: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return create_sample_products()

def create_sample_products():
    """Create sample products if CSV loading fails"""
    logger.info("Creating sample products...")
    return [
        {
            "id": 1,
            "title": "Classic Tweed Blazer",
            "description": "Academic-inspired wool tweed blazer in rich brown",
            "price": 5499.0,
            "image_url": "https://images.unsplash.com/photo-1594633313593-bab3825d0caf?w=400",
            "category": "blazers",
            "vibe_tags": "dark academia, scholarly, tweed, vintage, structured, brown",
            "rating": 4.6,
            "reviews_count": 342
        },
        {
            "id": 2,
            "title": "Floral Smocked Midi Dress",
            "description": "Flowing midi dress with hand-smocked bodice and delicate rose print",
            "price": 4599.0,
            "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400",
            "category": "dresses",
            "vibe_tags": "cottagecore, smocked, floral, cream, romantic, flowing, roses",
            "rating": 4.7,
            "reviews_count": 324
        }
    ]

def enhanced_fallback_search(vibe: str, products: list, max_results: int = 12):
    """Enhanced text matching with weighted scoring for different vibes"""
    start_time = time.time()
    logger.info(f"🔍 Starting enhanced fallback search for vibe: '{vibe}'")
    
    vibe_lower = vibe.lower()
    scored_products = []
    
    # Get enhanced keywords for known vibes
    enhanced_keywords = []
    for known_vibe, keywords in VIBE_KEYWORDS.items():
        if known_vibe in vibe_lower or any(word in vibe_lower for word in keywords[:3]):
            enhanced_keywords.extend(keywords)
            logger.info(f"📝 Matched known vibe '{known_vibe}', using enhanced keywords")
    
    for product in products:
        score = 0
        searchable_text = f"{product.get('title', '')} {product.get('description', '')} {product.get('vibe_tags', '')}".lower()
        
        # Exact vibe phrase match (highest score)
        if vibe_lower in searchable_text:
            score += 20
            logger.debug(f"Exact match found for product {product.get('id')}: '{vibe_lower}'")
        
        # Enhanced keyword matching
        if enhanced_keywords:
            for keyword in enhanced_keywords:
                if keyword in searchable_text:
                    score += 8
                    logger.debug(f"Enhanced keyword '{keyword}' matched for product {product.get('id')}")
        
        # Individual word matching from search query
        vibe_words = vibe_lower.split()
        for word in vibe_words:
            if len(word) > 2:  # Skip very short words
                if word in searchable_text:
                    score += 5
        
        # Vibe tag priority boost (products with matching vibe tags get higher scores)
        vibe_tags = product.get('vibe_tags', '').lower()
        if any(word in vibe_tags for word in vibe_words):
            score += 10
            logger.debug(f"Vibe tag match bonus for product {product.get('id')}")
        
        # Category relevance boost
        category_boosts = {
            'dark academia': ['blazers', 'skirts', 'shirts', 'knitwear', 'shoes', 'coats'],
            'cottagecore': ['dresses', 'blouses', 'skirts', 'tops', 'accessories'],
            'grunge': ['jackets', 'jeans', 't-shirts', 'shoes', 'accessories'],
            'Y2K': ['tops', 'pants', 'skirts', 'shoes', 'accessories'],
            'minimalist': ['all'],
            'coquette': ['blouses', 'skirts', 'tops', 'accessories', 'shoes']
        }
        
        product_category = product.get('category', '').lower()
        for known_vibe, relevant_categories in category_boosts.items():
            if known_vibe in vibe_lower and (product_category in relevant_categories or 'all' in relevant_categories):
                score += 5
                break
        
        if score > 0:
            scored_products.append((score, product))
    
    # Sort by score and return top results
    scored_products.sort(key=lambda x: x[0], reverse=True)
    results = [product for _, product in scored_products[:max_results]]
    
    search_time = round((time.time() - start_time) * 1000, 2)
    logger.info(f"✅ Enhanced fallback search completed in {search_time}ms, found {len(results)} matches")
    
    return results

def gemini_vibe_search(vibe: str, products: list, max_results: int = 12):
    """Enhanced Gemini AI search with better error handling and logging"""
    start_time = time.time()
    logger.info(f"🤖 Starting Gemini AI search for vibe: '{vibe}'")
    
    try:
        if not GEMINI_AVAILABLE or not model:
            logger.warning("⚠️ Gemini AI not available, falling back to text search")
            return None
        
        # Limit products to avoid token limits but ensure good variety
        sample_products = products[:100]  # Increased sample size
        
        # Create enhanced product context for AI
        product_context = "\n".join([
            f"ID: {p['id']} | Title: {p['title'][:50]} | Category: {p['category']} | Vibe Tags: {p['vibe_tags']} | Description: {p['description'][:100]}"
            for p in sample_products
        ])
        
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
        
        logger.info("📤 Sending request to Gemini AI...")
        response = model.generate_content(prompt)
        ai_time = round((time.time() - start_time) * 1000, 2)
        
        if not response or not response.text:
            logger.error(f"❌ Gemini AI returned empty response after {ai_time}ms")
            return None
        
        logger.info(f"📥 Received Gemini AI response in {ai_time}ms: {response.text[:100]}...")
        
        # Parse AI response with enhanced error handling
        try:
            # Clean the response and extract numbers
            clean_response = response.text.strip().replace('\n', ',').replace(' ', '')
            product_ids = []
            
            for id_str in clean_response.split(','):
                id_str = id_str.strip()
                if id_str.isdigit():
                    product_ids.append(int(id_str))
                else:
                    logger.debug(f"Skipping non-numeric ID: '{id_str}'")
            
            if not product_ids:
                logger.error(f"❌ No valid product IDs found in AI response: '{response.text}'")
                return None
            
            # Get matched products in the order suggested by AI
            matched_products = []
            products_dict = {p['id']: p for p in products}
            
            for pid in product_ids:
                if pid in products_dict:
                    matched_products.append(products_dict[pid])
                else:
                    logger.debug(f"Product ID {pid} not found in products list")
            
            total_time = round((time.time() - start_time) * 1000, 2)
            logger.info(f"✅ Gemini AI search completed successfully in {total_time}ms")
            logger.info(f"🎯 AI selected {len(matched_products)} products: {[p['id'] for p in matched_products]}")
            
            return matched_products
            
        except Exception as parse_error:
            logger.error(f"❌ Error parsing Gemini AI response: {str(parse_error)}")
            logger.error(f"Raw AI response: '{response.text}'")
            return None
            
    except Exception as ai_error:
        ai_time = round((time.time() - start_time) * 1000, 2)
        logger.error(f"❌ Gemini AI error after {ai_time}ms: {str(ai_error)}")
        logger.error(f"Error type: {type(ai_error).__name__}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load products on startup with enhanced logging"""
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
app = FastAPI(
    title="Vibe Search API", 
    version="2.0.0", 
    description="Pinterest-style aesthetic search for fashion products",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
def root():
    return {
        "message": "Vibe Search API v2.0 - Pinterest meets Myntra",
        "total_products": len(products_list),
        "gemini_available": GEMINI_AVAILABLE,
        "status": "running",
        "supported_vibes": list(VIBE_KEYWORDS.keys())
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "products_loaded": len(products_list) > 0,
        "gemini_available": GEMINI_AVAILABLE,
        "total_products": len(products_list),
        "timestamp": time.time()
    }

@app.get("/products", response_model=List[Product])
def get_products(limit: int = 20, category: str = None):
    """Get all products with optional category filter"""
    try:
        logger.info(f"📋 Fetching products: limit={limit}, category={category}")
        filtered_products = products_list.copy()
        
        if category and category.lower() != "all":
            filtered_products = [
                p for p in filtered_products 
                if p.get('category', '').lower() == category.lower()
            ]
            logger.info(f"Filtered to {len(filtered_products)} products for category '{category}'")
        
        return filtered_products[:limit]
    except Exception as e:
        logger.error(f"❌ Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching products")

@app.get("/categories")
def get_categories():
    """Get all available categories"""
    try:
        categories = list(set(
            p.get('category', 'uncategorized').lower() 
            for p in products_list 
            if p.get('category')
        ))
        logger.info(f"📂 Retrieved {len(categories)} categories")
        return {"categories": sorted(categories)}
    except Exception as e:
        logger.error(f"❌ Error fetching categories: {str(e)}")
        return {"categories": []}

@app.post("/search/vibe")
def vibe_search(request: VibeSearchRequest):
    """Enhanced Pinterest-style vibe search with comprehensive logging"""
    search_start_time = time.time()
    logger.info(f"🔍 NEW VIBE SEARCH: '{request.vibe}' (max_results: {request.max_results})")
    
    if request.price_min is not None or request.price_max is not None:
        logger.info(f"💰 Price filter: {request.price_min} - {request.price_max}")
    if request.category:
        logger.info(f"📂 Category filter: {request.category}")
    
    try:
        # Apply filters
        filtered_products = products_list.copy()
        original_count = len(filtered_products)
        
        if request.price_min is not None:
            filtered_products = [p for p in filtered_products if p['price'] >= request.price_min]
            logger.info(f"Price min filter: {len(filtered_products)}/{original_count} products remaining")
        
        if request.price_max is not None:
            filtered_products = [p for p in filtered_products if p['price'] <= request.price_max]
            logger.info(f"Price max filter: {len(filtered_products)}/{original_count} products remaining")
        
        if request.category and request.category.lower() != "all":
            filtered_products = [
                p for p in filtered_products 
                if p.get('category', '').lower() == request.category.lower()
            ]
            logger.info(f"Category filter: {len(filtered_products)}/{original_count} products remaining")
        
        if not filtered_products:
            logger.warning("❌ No products match the applied filters")
            return {
                "products": [],
                "message": "No products found matching your filters",
                "vibe_query": request.vibe,
                "total_matches": 0,
                "search_method": "filtered_out",
                "search_time_ms": round((time.time() - search_start_time) * 1000, 2)
            }
        
        # Try Gemini AI first
        matched_products = None
        search_method = "fallback"
        
        if GEMINI_AVAILABLE:
            logger.info("🤖 Attempting Gemini AI search...")
            matched_products = gemini_vibe_search(request.vibe, filtered_products, request.max_results)
            
            if matched_products:
                search_method = "gemini_ai"
                logger.info(f"✅ Gemini AI search successful: {len(matched_products)} products")
            else:
                logger.warning("⚠️ Gemini AI search failed, falling back to text matching")
        else:
            logger.info("ℹ️ Gemini AI not available, using enhanced fallback search")
        
        # Fallback to enhanced text matching if AI failed
        if not matched_products:
            logger.info("🔤 Starting enhanced fallback search...")
            matched_products = enhanced_fallback_search(request.vibe, filtered_products, request.max_results)
            search_method = "enhanced_fallback"
        
        total_search_time = round((time.time() - search_start_time) * 1000, 2)
        
        result = {
            "products": matched_products,
            "vibe_query": request.vibe,
            "total_matches": len(matched_products),
            "search_method": search_method,
            "search_time_ms": total_search_time,
            "gemini_available": GEMINI_AVAILABLE
        }
        
        if matched_products:
            result["message"] = f"Found {len(matched_products)} perfect matches for '{request.vibe}'!"
            logger.info(f"✅ SEARCH COMPLETED: {len(matched_products)} matches in {total_search_time}ms using {search_method}")
        else:
            result["message"] = f"No matches found for '{request.vibe}'. Try different keywords!"
            logger.warning(f"⚠️ SEARCH COMPLETED: No matches found in {total_search_time}ms")
        
        return result
        
    except Exception as e:
        search_time = round((time.time() - search_start_time) * 1000, 2)
        logger.error(f"❌ SEARCH ERROR after {search_time}ms: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Error processing vibe search")

@app.get("/trending")
def get_trending():
    """Get trending vibes and featured products"""
    logger.info("📈 Fetching trending content...")
    
    trending_vibes = [
        "dark academia", "cottagecore", "Y2K", "minimalist", 
        "grunge", "soft girl", "indie sleaze", "coquette"
    ]
    
    # Get top-rated products
    try:
        top_products = sorted(
            products_list, 
            key=lambda x: (x.get('rating', 0), x.get('reviews_count', 0)), 
            reverse=True
        )[:6]
        
        logger.info(f"Retrieved {len(top_products)} featured products")
        
        return {
            "trending_vibes": trending_vibes,
            "featured_products": top_products,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting trending content: {str(e)}")
        return {
            "trending_vibes": trending_vibes,
            "featured_products": [],
            "status": "error"
        }

@app.get("/debug/vibe/{vibe}")
def debug_vibe_search(vibe: str):
    """Debug endpoint to test vibe matching logic"""
    logger.info(f"🐛 DEBUG: Testing vibe matching for '{vibe}'")
    
    # Test both search methods
    ai_results = None
    if GEMINI_AVAILABLE:
        ai_results = gemini_vibe_search(vibe, products_list[:20], 5)
    
    fallback_results = enhanced_fallback_search(vibe, products_list, 5)
    
    return {
        "vibe": vibe,
        "gemini_available": GEMINI_AVAILABLE,
        "ai_results": [p['id'] for p in ai_results] if ai_results else None,
        "fallback_results": [p['id'] for p in fallback_results],
        "ai_products": ai_results,
        "fallback_products": fallback_results
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Vibe Search API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)