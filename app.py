from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import logging
import google.generativeai as genai
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
import torch
import random
import re
import base64

# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Myntra Vibe Shopping API", version="2.0.0")

# Enable CORS (for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
model_gemini = gemini_model  # Alias for consistency
GEMINI_AVAILABLE = True

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# DATA LOADING
# ============================================================

try:
    products_df = pd.read_csv("products.csv")
    products_list = products_df.to_dict("records")
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
        }
    ]

# Precompute embeddings
logger.info("🔄 Encoding product embeddings...")
product_texts = [
    f"{p['title']} {p['description']} {p.get('vibe_tags', '')}" for p in products_list
]
product_embeddings = embedding_model.encode(product_texts, convert_to_tensor=True)
logger.info("✅ Product embeddings ready!")

# ============================================================
# MODELS
# ============================================================

class VibeSearchRequest(BaseModel):
    vibe: str
    max_results: Optional[int] = 12
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def semantic_search(query: str, top_k: int = 20) -> List[int]:
    """
    Perform semantic search using sentence transformers.
    Returns indices of top matching products.
    """
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, product_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(products_list)))
    return top_results.indices.tolist()


def get_ai_keywords(vibe: str) -> List[str]:
    """
    Use Gemini to extract relevant keywords from a vibe description.
    """
    try:
        if not GEMINI_AVAILABLE or not model_gemini:
            return []
        
        prompt = f"""
You are an AI fashion stylist and shopping assistant.  
Your task: map vague or cultural user queries to specific catalog items.  

Follow these exact steps:

Step 1: Interpret the user query
- Identify if the query is about an aesthetic (e.g., cottagecore, Y2K), a celebrity look (e.g., Taylor Swift Eras Tour), an event (e.g., rooftop karaoke night), or a mood/emotion (e.g., cozy, confident).  
- Expand the query into a list of STYLE ELEMENTS: { "colors, fabrics, silhouettes, aesthetics, moods, cultural references" }.  

Step 2: Match with catalog
The catalog is provided below. Each product has: ID, title, description, category, vibe_tags.  

Catalog:
{product_context}

Compare STYLE ELEMENTS against each product’s title, description, and vibe_tags.  
Decide which products BEST align with the query.  
Ranking criteria (in order of importance):
1. Aesthetic/mood alignment (most critical).  
2. Relevance to cultural reference (if any).  
3. Visual elements (colors, fabrics, silhouettes).  
4. Secondary match: versatility or adjacent style.  

Step 3: Output
Return ONLY the product IDs of the top {min(request.max_results, len(filtered_products))} matches.  
Format: a single line with comma-separated IDs. Example: 2,5,7,9  

Rules:
- DO NOT output any explanation.  
- DO NOT repeat the query.  
- DO NOT include extra words or symbols, only numbers separated by commas.  
---
User query: "{request.vibe}"
"""

        
        response = model_gemini.generate_content(prompt)
        keywords_text = response.text.strip()
        keywords = [k.strip().lower() for k in keywords_text.split(',')]
        return keywords[:8]  # Limit to 8 keywords
        
    except Exception as e:
        logger.error(f"❌ Keyword extraction error: {str(e)}")
        return []


def _run_enhanced_gemini_search(request: VibeSearchRequest, products: list):
    """
    Enhanced Gemini API search with better cultural understanding.
    """
    try:
        if not GEMINI_AVAILABLE or not model_gemini:
            return None
        
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
        
        # Extract product IDs from response
        product_ids = [int(id_str) for id_str in re.findall(r'\b\d+\b', response_text)]
        
        # Build result products
        products_dict = {p['id']: p for p in sampled_products}
        matched_products = []
        
        for pid in product_ids:
            if pid in products_dict:
                product = products_dict[pid].copy()
                product['similarity_score'] = 0.8  # Mock similarity score
                matched_products.append(product)
        
        return matched_products[:request.max_results]
        
    except Exception as e:
        logger.error(f"❌ Enhanced Gemini Search Error: {str(e)}")
        return None


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def read_root():
    return {"message": "Myntra Vibe Shopping API is running!", "total_products": len(products_list)}


@app.get("/products")
def get_all_products(limit: int = 20, category: Optional[str] = None):
    filtered = products_list
    if category:
        filtered = [p for p in filtered if p.get("category", "").lower() == category.lower()]
    return filtered[:limit]


@app.get("/products/{product_id}")
def get_product(product_id: int):
    product = next((p for p in products_list if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.get("/categories")
def get_categories():
    return {"categories": list(set(p.get("category", "uncategorized") for p in products_list))}


@app.post("/search/vibe")
def vibe_search(request: VibeSearchRequest):
    vibe = request.vibe
    logger.info(f"🔍 NEW VIBE SEARCH: '{vibe}'")

    # Try Gemini search first (most sophisticated)
    gemini_results = _run_enhanced_gemini_search(request, products_list)
    
    if gemini_results:
        logger.info(f"✅ Gemini search returned {len(gemini_results)} results")
        return {
            "products": gemini_results,
            "query": vibe,
            "total_matches": len(gemini_results),
            "search_method": "gemini_enhanced",
            "message": f"Found {len(gemini_results)} items matching your vibe using AI",
        }
    
    # Fallback to semantic search
    logger.info("🔄 Falling back to semantic search")
    indices = semantic_search(vibe, top_k=request.max_results)
    matched = [products_list[i] for i in indices]

    # Refine with AI keywords
    keywords = get_ai_keywords(vibe)
    if keywords:
        refined = []
        for product in matched:
            text = f"{product['title']} {product['description']} {product.get('vibe_tags', '')}".lower()
            if any(kw in text for kw in keywords):
                refined.append(product)
        if refined:
            matched = refined

    # Apply filters
    if request.price_min is not None:
        matched = [p for p in matched if p["price"] >= request.price_min]
    if request.price_max is not None:
        matched = [p for p in matched if p["price"] <= request.price_max]
    if request.category:
        matched = [p for p in matched if p.get("category", "").lower() == request.category.lower()]

    return {
        "products": matched,
        "query": vibe,
        "total_matches": len(matched),
        "keywords": keywords,
        "search_method": "semantic_fallback",
        "message": f"Found {len(matched)} items matching your vibe",
    }

@app.post("/search/image")
async def image_search(file: UploadFile = File(...), additional_text: str = ""):
    """
    Search products based on an uploaded image + optional user text.
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare product context (limit for efficiency)
        products = MODELS.get('products_list', [])[:40]
        product_context = "\n".join([
            f"ID: {p['id']}, Title: {p['title']}, Description: {p.get('description','')}, Tags: {p.get('vibe_tags','')}"
            for p in products
        ])

        # Prompt for Gemini Vision
        prompt = f"""
You are a fashion stylist AI.
Analyze the uploaded image + optional text "{additional_text}".
1. Extract STYLE ELEMENTS (aesthetics, mood, colors, patterns, silhouettes, cultural references).
2. Match them against this catalog:

{product_context}

Return ONLY JSON in this format:
{{
  "ids": [ID1, ID2, ID3]
}}
"""

        # Call Gemini Vision
        response = model_gemini.generate_content(
            [prompt, {"mime_type": file.content_type, "data": b64_image}]
        )

        raw = response.text.strip()

        # Parse Gemini response
        try:
            parsed = json.loads(raw)
            ids = parsed.get("ids", [])
        except:
            ids = [int(x) for x in re.findall(r"\b\d+\b", raw)]

        # Collect matched products
        products_dict = {p['id']: p for p in products}
        matched = [products_dict[i] for i in ids if i in products_dict]

        return {"products": matched, "message": f"Found {len(matched)} image-based matches"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@app.get("/trending")
def get_trending():
    return {
        "trending_vibes": [
            "cozy cottagecore aesthetic",
            "dark academia vibes",
            "Y2K nostalgia",
            "minimalist clean girl",
        ],
        "featured_products": products_list[:6],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "total_products": len(products_list)}


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)