from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import logging
import google.generativeai as genai
from typing import List, Optional
import random
import re
import base64
import json
from dotenv import load_dotenv
from PIL import Image
import io

# Load env vars first
load_dotenv()

# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Myntra Vibe Shopping API", version="2.0.0")

# FIXED: Updated CORS origins to include your frontend
origins = [
    "http://localhost:8080",
    "http://0.0.0.0:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "*"  # Allow all origins for development (remove in production)
]

# Enable CORS (for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# FIXED: Better Gemini setup with error handling
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        model_gemini = gemini_model
        GEMINI_AVAILABLE = True
        logger.info("✅ Gemini API configured successfully")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")
        GEMINI_AVAILABLE = False
        model_gemini = None
except Exception as e:
    logger.error(f"❌ Error configuring Gemini API: {str(e)}")
    GEMINI_AVAILABLE = False
    model_gemini = None

# ============================================================
# DATA LOADING
# ============================================================

try:
    products_df = pd.read_csv("products.csv")
    products_list = products_df.to_dict("records")
    logger.info(f"✅ Loaded {len(products_list)} products from CSV")
except FileNotFoundError:
    logger.warning("⚠️ products.csv not found. Using sample fallback products.")
    products_list = [
        {
            "id": 1,
            "title": "Floral Midi Dress",
            "description": "Beautiful floral print midi dress perfect for spring outings",
            "price": 2499.0,
            "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400",
            "category": "dresses",
            "vibe_tags": "cottagecore, feminine, spring, romantic, floral",
        },
        {
            "id": 2,
            "title": "Oversized Denim Jacket",
            "description": "Classic oversized denim jacket with vintage wash",
            "price": 3299.0,
            "image_url": "https://images.unsplash.com/photo-1551537482-f2075a1d41f2?w=400",
            "category": "jackets",
            "vibe_tags": "grunge, casual, Y2K, streetwear, denim",
        },
        {
            "id": 3,
            "title": "Black Leather Ankle Boots",
            "description": "Edgy black leather boots with chunky sole",
            "price": 4299.0,
            "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400",
            "category": "shoes",
            "vibe_tags": "dark academia, grunge, edgy, minimalist, black",
        },
        {
            "id": 4,
            "title": "Bohemian Maxi Skirt",
            "description": "Flowing boho maxi skirt with intricate patterns",
            "price": 1899.0,
            "image_url": "https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400",
            "category": "skirts",
            "vibe_tags": "bohemian, boho, free-spirited, patterns, flowing",
        },
        {
            "id": 5,
            "title": "Minimalist White T-Shirt",
            "description": "Clean, simple white cotton tee",
            "price": 899.0,
            "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400",
            "category": "tops",
            "vibe_tags": "minimalist, clean, basic, casual, white",
        },
        {
            "id": 6,
            "title": "Vintage High-Waist Jeans",
            "description": "Classic high-waisted denim with vintage wash",
            "price": 2899.0,
            "image_url": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=400",
            "category": "bottoms",
            "vibe_tags": "vintage, retro, high-waist, denim, classic",
        },
        {
            "id": 7,
            "title": "Elegant Evening Gown",
            "description": "Sophisticated black evening gown with sequin details",
            "price": 8999.0,
            "image_url": "https://images.unsplash.com/photo-1566479179817-c8f6e5d9b5aa?w=400",
            "category": "dresses",
            "vibe_tags": "elegant, formal, evening, sophisticated, sequin, black",
        },
        {
            "id": 8,
            "title": "Festive Silk Saree",
            "description": "Traditional silk saree with gold border for festivals",
            "price": 12999.0,
            "image_url": "https://images.unsplash.com/photo-1583391733956-6c78276477e1?w=400",
            "category": "ethnic",
            "vibe_tags": "traditional, festive, silk, elegant, indian, gold",
        },
        {
            "id": 9,
            "title": "Casual Cotton Kurta",
            "description": "Comfortable cotton kurta for everyday wear",
            "price": 1299.0,
            "image_url": "https://images.unsplash.com/photo-1603252109303-2751441dd157?w=400",
            "category": "ethnic",
            "vibe_tags": "casual, comfortable, cotton, traditional, everyday",
        },
        {
            "id": 10,
            "title": "Designer Lehenga",
            "description": "Embroidered designer lehenga for weddings",
            "price": 25999.0,
            "image_url": "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=400",
            "category": "ethnic",
            "vibe_tags": "wedding, designer, embroidered, traditional, bridal",
        },
        {
            "id": 11,
            "title": "Casual Sneakers",
            "description": "Comfortable white sneakers for daily wear",
            "price": 2999.0,
            "image_url": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400",
            "category": "shoes",
            "vibe_tags": "casual, comfortable, white, sneakers, daily",
        },
        {
            "id": 12,
            "title": "Party Crop Top",
            "description": "Sparkly crop top perfect for parties",
            "price": 1599.0,
            "image_url": "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400",
            "category": "tops",
            "vibe_tags": "party, sparkly, crop, trendy, nightlife",
        }
    ]

# ============================================================
# MODELS
# ============================================================

class VibeSearchRequest(BaseModel):
    vibe: str
    max_results: Optional[int] = 12
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category: Optional[str] = None

class LocationEventRequest(BaseModel):
    location: str
    max_results: Optional[int] = 12
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category: Optional[str] = None

# ============================================================
# HELPERS
# ============================================================

def _run_enhanced_gemini_search(request: VibeSearchRequest, products: list):
    """
    Enhanced Gemini API search with better cultural understanding.
    """
    try:
        if not GEMINI_AVAILABLE or not model_gemini:
            logger.warning("⚠️ Gemini not available, falling back to keyword search")
            return _fallback_keyword_search(request, products)
        
        # Apply filters first
        filtered_products = [
            p for p in products
            if (request.price_min is None or p['price'] >= request.price_min) and
               (request.price_max is None or p['price'] <= request.price_max) and
               (not request.category or request.category.lower() == 'all' or p.get('category', '').lower() == request.category.lower())
        ]
        
        if not filtered_products:
            logger.info("ℹ️ Filters returned no products. Gemini search cannot proceed.")
            return []
        
        # Sample products if too many
        sample_size = min(50, len(filtered_products))
        if len(filtered_products) > sample_size:
            sampled_products = random.sample(filtered_products, sample_size)
        else:
            sampled_products = filtered_products
        
        # Create product descriptions for Gemini
        product_descriptions = "\n".join([
            f"ID: {p['id']}, Title: {p['title']}, Description: {p.get('description', '')[:100]}, "
            f"Category: {p.get('category', '')}, Vibe Tags: {p.get('vibe_tags', '')}"
            for p in sampled_products
        ])
        
        prompt = f"""
You are a fashion stylist AI.

User vibe query: "{request.vibe}"

You are given a product catalog below. Each product has:
- ID
- Title
- Short description
- Category
- Vibe tags

Your task:
1. Analyze the user's vibe query to understand the implied aesthetics, mood, style, occasion or cultural references and translate them to recognizable vibe keywords.
2. Identify the vibe keywords implied by the user's query 
   (e.g., mood, aesthetic, style, occasion).
3. Match them against the catalog.
4. Pick the top {min(request.max_results, len(sampled_products))} most relevant product IDs.

Rules:
- Match using aesthetics, tags, mood, colors, patterns, lifestyle cues.
- Prefer products that share multiple overlapping vibe tags with the query.
- If nothing matches well, return the closest vibe tags (do not leave empty).
- Respond ONLY in valid JSON.

Output format (strict):
{{
  "ids": [1, 2, 3]
}}

Catalog:
{product_descriptions}
"""

        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to parse JSON response
        try:
            parsed_response = json.loads(response_text)
            product_ids = parsed_response.get("ids", [])
        except json.JSONDecodeError:
            # Fallback to regex if JSON parsing fails
            product_ids = [int(id_str) for id_str in re.findall(r'\b\d+\b', response_text)]
        
        # Build result products
        products_dict = {p['id']: p for p in sampled_products}
        matched_products = []
        
        for pid in product_ids:
            if pid in products_dict:
                product = products_dict[pid].copy()
                product['similarity_score'] = 0.9  # Gemini only, mock score
                matched_products.append(product)
        
        return matched_products[:request.max_results]
        
    except Exception as e:
        logger.error(f"❌ Enhanced Gemini Search Error: {str(e)}")
        return _fallback_keyword_search(request, products)

def _fallback_keyword_search(request: VibeSearchRequest, products: list):
    """
    Fallback keyword-based search when Gemini is not available
    """
    logger.info("🔍 Using fallback keyword search")
    
    # Apply filters first
    filtered_products = [
        p for p in products
        if (request.price_min is None or p['price'] >= request.price_min) and
           (request.price_max is None or p['price'] <= request.price_max) and
           (not request.category or request.category.lower() == 'all' or p.get('category', '').lower() == request.category.lower())
    ]
    
    if not filtered_products:
        return []
    
    # Simple keyword matching
    vibe_keywords = request.vibe.lower().split()
    scored_products = []
    
    for product in filtered_products:
        score = 0
        text_to_search = f"{product.get('title', '')} {product.get('description', '')} {product.get('vibe_tags', '')}".lower()
        
        for keyword in vibe_keywords:
            if keyword in text_to_search:
                score += 1
        
        if score > 0:
            product_copy = product.copy()
            product_copy['similarity_score'] = score / len(vibe_keywords)
            scored_products.append(product_copy)
    
    # Sort by score and return top results
    scored_products.sort(key=lambda x: x['similarity_score'], reverse=True)
    return scored_products[:request.max_results]

def _analyze_location_events(location: str):
    """
    Use Gemini AI to analyze location and determine local events/festivals
    """
    try:
        if not GEMINI_AVAILABLE or not model_gemini:
            logger.warning("⚠️ Gemini not available for location analysis")
            return {"events": [], "dress_codes": ["casual"], "error": "AI analysis not available"}
        
        prompt = f"""
You are a cultural events and fashion expert specializing in Indian locations and festivities.

Location: "{location}"

Your task:
1. Identify the location (city, state, region in India)
2. Determine current/upcoming festivals, events, or cultural celebrations specific to this location
3. Consider seasonal events, local traditions, religious festivals, cultural programs
4. Suggest appropriate dress codes and fashion styles for these events

Focus on:
- Regional festivals (like Durga Puja in Bengal, Ganesh Chaturthi in Maharashtra, Onam in Kerala)
- Local cultural events and fairs
- Seasonal celebrations
- Wedding seasons
- Religious observances
- Modern events (concerts, parties, exhibitions)

Respond ONLY in valid JSON format:
{{
  "location_info": {{
    "city": "city_name",
    "state": "state_name",
    "region": "region_description"
  }},
  "events": [
    {{
      "name": "event_name",
      "type": "festival/cultural/religious/modern",
      "description": "brief_description",
      "dress_code": "traditional/semi-formal/casual/ethnic",
      "style_keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ],
  "recommended_styles": ["style1", "style2", "style3"],
  "seasonal_note": "seasonal_context"
}}
"""

        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                parsed_response = json.loads(json_str)
                return parsed_response
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ Location analysis JSON parsing failed: {e}")
            return {
                "events": [{"name": "Local Events", "style_keywords": ["casual", "comfortable"]}],
                "recommended_styles": ["casual", "comfortable"],
                "error": "Partial analysis available"
            }
        
    except Exception as e:
        logger.error(f"❌ Location analysis error: {str(e)}")
        return {
            "events": [],
            "recommended_styles": ["casual"],
            "error": f"Analysis failed: {str(e)}"
        }

def _search_products_by_event_style(style_keywords: list, request: LocationEventRequest, products: list):
    """
    Search products based on event style keywords
    """
    # Apply filters first
    filtered_products = [
        p for p in products
        if (request.price_min is None or p['price'] >= request.price_min) and
           (request.price_max is None or p['price'] <= request.price_max) and
           (not request.category or request.category.lower() == 'all' or p.get('category', '').lower() == request.category.lower())
    ]
    
    if not filtered_products:
        return []
    
    # Score products based on style keywords
    scored_products = []
    
    for product in filtered_products:
        score = 0
        text_to_search = f"{product.get('title', '')} {product.get('description', '')} {product.get('vibe_tags', '')}".lower()
        
        for keyword in style_keywords:
            if keyword.lower() in text_to_search:
                score += 1
        
        if score > 0:
            product_copy = product.copy()
            product_copy['similarity_score'] = score / len(style_keywords)
            product_copy['matched_keywords'] = [kw for kw in style_keywords if kw.lower() in text_to_search]
            scored_products.append(product_copy)
    
    # Sort by score and return top results
    scored_products.sort(key=lambda x: x['similarity_score'], reverse=True)
    return scored_products[:request.max_results]

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def read_root():
    return {
        "message": "Myntra Vibe Shopping API is running!", 
        "total_products": len(products_list),
        "gemini_available": GEMINI_AVAILABLE
    }

@app.get("/products")
def get_all_products(limit: int = 20, category: Optional[str] = None):
    filtered = products_list
    if category and category.lower() != 'all':
        filtered = [p for p in filtered if p.get("category", "").lower() == category.lower()]
    return {"products": filtered[:limit], "total": len(filtered)}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    product = next((p for p in products_list if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.get("/categories")
def get_categories():
    categories = list(set(p.get("category", "uncategorized") for p in products_list))
    return {"categories": categories}

@app.post("/search/vibe")
def vibe_search(request: VibeSearchRequest):
    vibe = request.vibe
    logger.info(f"🔍 NEW VIBE SEARCH: '{vibe}'")

    # Use enhanced search (Gemini or fallback)
    search_results = _run_enhanced_gemini_search(request, products_list)
    
    if search_results:
        logger.info(f"✅ Search returned {len(search_results)} results")
        return {
            "products": search_results,
            "query": vibe,
            "total_matches": len(search_results),
            "search_method": "gemini" if GEMINI_AVAILABLE else "fallback",
            "message": f"Found {len(search_results)} items matching your vibe",
        }
    
    # If no results found
    logger.warning("⚠️ Search returned no results")
    return {
        "products": [],
        "query": vibe,
        "total_matches": 0,
        "search_method": "gemini" if GEMINI_AVAILABLE else "fallback",
        "message": "No matches found. Try a different vibe query.",
    }

@app.post("/search/location-events")
def location_event_search(request: LocationEventRequest):
    """
    NEW FEATURE: Search products based on location and local events/festivals
    """
    location = request.location
    logger.info(f"📍 NEW LOCATION EVENT SEARCH: '{location}'")
    
    try:
        # Step 1: Analyze location to determine events
        location_analysis = _analyze_location_events(location)
        
        if "error" in location_analysis and not location_analysis.get("events"):
            logger.warning(f"⚠️ Location analysis failed for: {location}")
            return {
                "products": [],
                "location": location,
                "events": [],
                "total_matches": 0,
                "message": "Could not analyze location for events. Please try a more specific location.",
                "error": location_analysis.get("error")
            }
        
        # Step 2: Extract style keywords from events
        all_style_keywords = []
        events_info = location_analysis.get("events", [])
        
        for event in events_info:
            style_keywords = event.get("style_keywords", [])
            all_style_keywords.extend(style_keywords)
        
        # Add recommended styles
        recommended_styles = location_analysis.get("recommended_styles", [])
        all_style_keywords.extend(recommended_styles)
        
        # Remove duplicates and filter empty
        unique_keywords = list(set([kw for kw in all_style_keywords if kw]))
        
        if not unique_keywords:
            unique_keywords = ["casual", "comfortable"]  # Fallback
        
        logger.info(f"🎯 Style keywords for {location}: {unique_keywords}")
        
        # Step 3: Search products based on style keywords
        search_results = _search_products_by_event_style(unique_keywords, request, products_list)
        
        # Step 4: Return results
        if search_results:
            logger.info(f"✅ Location event search returned {len(search_results)} results")
            return {
                "products": search_results,
                "location": location,
                "location_info": location_analysis.get("location_info", {}),
                "events": events_info,
                "style_keywords": unique_keywords,
                "seasonal_note": location_analysis.get("seasonal_note", ""),
                "total_matches": len(search_results),
                "search_method": "location_event_analysis",
                "message": f"Found {len(search_results)} items perfect for events in {location}"
            }
        else:
            logger.warning("⚠️ No products found for location events")
            return {
                "products": [],
                "location": location,
                "location_info": location_analysis.get("location_info", {}),
                "events": events_info,
                "style_keywords": unique_keywords,
                "total_matches": 0,
                "search_method": "location_event_analysis",
                "message": f"No suitable products found for events in {location}. Try adjusting your filters."
            }
            
    except Exception as e:
        logger.error(f"❌ Location event search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Location event search failed: {str(e)}")

@app.post("/search/image")
async def image_search(file: UploadFile = File(...), max_results: int = 12):
    """
    FIXED: Search products based on an uploaded image using Gemini Vision.
    """
    try:
        if not GEMINI_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Image search requires Gemini API which is not available. Please set GEMINI_API_KEY."
            )
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_bytes = await file.read()
        logger.info(f"📷 Processing image: {file.filename}, size: {len(image_bytes)} bytes")
        
        # Optimize image size for Gemini (optional but recommended)
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=85)
                image_bytes = img_buffer.getvalue()
                logger.info("📷 Image resized for optimization")
        except Exception as e:
            logger.warning(f"⚠️ Image optimization failed: {e}, using original")
        
        # Prepare product context for Gemini (limit for token efficiency)
        sample_products = random.sample(products_list, min(40, len(products_list)))
        product_context = "\n".join([
            f"ID: {p['id']}, Title: {p['title']}, Description: {p.get('description','')[:100]}, "
            f"Category: {p.get('category','')}, Vibe Tags: {p.get('vibe_tags','')}"
            for p in sample_products
        ])

        # Enhanced prompt for Gemini Vision
        prompt = f"""
You are an expert fashion stylist and visual analyst.

Analyze the uploaded image and extract fashion/style elements:

1. VISUAL ANALYSIS:
   - Colors (dominant and accent colors)
   - Patterns (floral, geometric, solid, etc.)
   - Silhouettes (oversized, fitted, flowy, structured)
   - Textures (denim, leather, cotton, silk, etc.)
   - Style aesthetics (minimalist, bohemian, grunge, preppy, etc.)
   - Occasion/mood (casual, formal, party, work, vacation)

2. VIBE KEYWORDS:
   Extract 5-10 relevant style keywords that describe this image's aesthetic

3. PRODUCT MATCHING:
   Match these vibes against the product catalog below and select the {max_results} most relevant product IDs.

Rules:
- Focus on style similarity, not exact product matching
- Consider color harmony, aesthetic compatibility, and vibe alignment
- Prefer products with overlapping vibe tags
- Return diverse product types when possible

Product Catalog:
{product_context}

Respond ONLY in valid JSON format:
{{
  "analysis": {{
    "colors": ["color1", "color2"],
    "patterns": ["pattern1", "pattern2"],
    "aesthetic": "aesthetic_description",
    "vibe_keywords": ["keyword1", "keyword2", "keyword3"]
  }},
  "ids": [1, 2, 3, 4, 5]
}}
"""

        # Create the image part for Gemini
        image_part = {
            "mime_type": file.content_type,
            "data": image_bytes
        }

        # Call Gemini Vision API
        logger.info("🤖 Calling Gemini Vision API for image analysis")
        response = model_gemini.generate_content([prompt, image_part])
        raw_response = response.text.strip()
        
        logger.info(f"📋 Raw Gemini response: {raw_response[:200]}...")

        # Parse Gemini response
        product_ids = []
        analysis = {}
        try:
            # ✅ CORRECTED LOGIC: Use a more reliable regex to find the JSON object.
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                json_str = json_match.group(0)
                # Parse the extracted JSON string
                parsed_response = json.loads(json_str)
                product_ids = parsed_response.get("ids", [])
                analysis = parsed_response.get("analysis", {})
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}, using regex fallback")
            # Fallback: extract numbers from response
            product_ids = [int(x) for x in re.findall(r'\b\d+\b', raw_response)]
            analysis = {"vibe_keywords": ["image-based-search"]}
        

        # Build products dictionary for quick lookup
        products_dict = {p['id']: p for p in products_list}
        
        # Collect matched products
        matched_products = []
        for pid in product_ids[:max_results]:
            if pid in products_dict:
                product = products_dict[pid].copy()
                product['similarity_score'] = 0.9  # Mock score for image search
                matched_products.append(product)

        logger.info(f"✅ Image search completed: {len(matched_products)} products matched")
        
        return {
            "products": matched_products,
            "total_matches": len(matched_products),
            "analysis": analysis,
            "search_method": "gemini_vision",
            "message": f"Found {len(matched_products)} products matching your image style"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Image search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")

@app.get("/trending")
def get_trending():
    """
    Get trending vibes and featured products
    """
    try:
        # Sample some random products for featured section
        featured_products = random.sample(products_list, min(6, len(products_list)))
        
        return {
            "trending_vibes": [
                "cozy cottagecore aesthetic",
                "dark academia vibes", 
                "Y2K nostalgia",
                "minimalist clean girl",
                "bohemian free spirit",
                "vintage retro style"
            ],
            "featured_products": featured_products,
            "message": "Trending data loaded successfully"
        }
    except Exception as e:
        logger.error(f"❌ Error in /trending endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load trending data: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "total_products": len(products_list),
        "gemini_available": GEMINI_AVAILABLE,
        "message": "API is healthy and running"
    }

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")