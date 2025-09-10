# streamlit_app.py - Final Vibe Search UI for Myntra Hackathon
import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time

# --- Page Config ---
st.set_page_config(
    page_title="Vibe Search by Myntra",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main app theme colors - light mode */
    :root {
        --primary-color: #ff3f6c;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #222222;
        --font: sans-serif;
    }

    /* Apply theme colors */
    .stApp {
        background-color: var(--background-color);
        color: #ff005b;
        font-family: var(--font);
    }
    .st-emotion-cache-1r651z9 { /* Sidebar background */
        background-color: var(--secondary-background-color);
    }
    .st-emotion-cache-1n667ek { /* Widget labels */
        color: var(--text-color);
    }

    /* Header and sub-header styling */
    .main-header {
        color: #ff3f6c;
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        color: #555;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Vibe Search Input */
    .st-emotion-cache-1cpx97b {
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 10px 20px;
    }

    /* Product Cards Styling */
    .product-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 15px;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        text-align: center;
        text-decoration: none !important;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    .product-image {
        border-radius: 8px;
        width: 100%;
        height: 250px;
        object-fit: cover;
    }
    .product-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-color);
        margin-top: 10px;
        height: 2.5em; /* Fixed height for titles */
        overflow: hidden;
    }
    .product-price {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--primary-color);
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .product-details {
        font-size: 0.9rem;
        color: #666;
    }
    .search-button {
        background-color: var(--primary-color);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 20px;
        padding: 10px 25px;
        width: 100%;
        cursor: pointer;
    }

    /* Fix for some default Streamlit text elements */
    p, h1, h2, h3, h4, h5, h6, label {
        color: var(--text-color);
    }

</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
API_URL = "http://localhost:8000/search/vibe"
TRENDING_URL = "http://localhost:8000/trending"
CATEGORIES_URL = "http://localhost:8000/categories"

# --- Functions to interact with the API ---
def get_categories():
    try:
        response = requests.get(CATEGORIES_URL)
        response.raise_for_status()
        return response.json().get('categories', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching categories: {e}")
        return []

def get_trending():
    try:
        response = requests.get(TRENDING_URL)
        response.raise_for_status()
        return response.json().get('trending_vibes', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API. Please ensure the backend is running. Error: {e}")
        return []

def search_vibe(vibe: str, max_results: int, price_min: float, price_max: float, category: str):
    payload = {
        "vibe": vibe,
        "max_results": max_results,
        "price_min": price_min,
        "price_max": price_max,
        "category": category
    }
    try:
        with st.spinner("Searching for your vibe..."):
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# --- Main UI Logic ---

st.markdown("<h1 class='main-header'>Myntra Vibe Search</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Discover products that match your aesthetic. Powered by Gemini AI.</p>", unsafe_allow_html=True)

# Main search bar and trending vibes display
trending_vibes = get_trending()
if trending_vibes:
    st.markdown("##### Trending Vibes:")
    cols = st.columns(len(trending_vibes))
    for i, vibe in enumerate(trending_vibes):
        with cols[i]:
            if st.button(vibe, use_container_width=True):
                st.session_state.vibe_input = vibe

vibe_query = st.text_input(
    label="Enter your vibe here:",
    placeholder="e.g., dark academia, cottagecore, minimalist, grunge...",
    label_visibility="collapsed",
    value=st.session_state.get('vibe_input', ''),
    key='vibe_search_input'
)

# Sidebar for filters
st.sidebar.header("Filters")
max_results = st.sidebar.slider("Number of Results", 1, 50, 20)
min_price, max_price = st.sidebar.slider(
    "Price Range", 0.0, 10000.0, (0.0, 10000.0)
)
categories = get_categories()
categories.insert(0, "all")
category = st.sidebar.selectbox("Category", categories)

if st.button("Search Vibe", use_container_width=True):
    if vibe_query:
        st.session_state.last_search = search_vibe(
            vibe_query,
            max_results,
            min_price,
            max_price,
            category
        )
    else:
        st.warning("Please enter a vibe to search.")

# Display search results
if 'last_search' in st.session_state and st.session_state.last_search:
    results = st.session_state.last_search.get("products", [])
    message = st.session_state.last_search.get("message", "No results found.")
    search_method = st.session_state.last_search.get("search_method", "N/A")
    search_time_ms = st.session_state.last_search.get("search_time_ms", 0)

    st.info(f"**Method:** {search_method} | **Time:** {search_time_ms:.2f}ms")
    st.subheader(message)

    if results:
        # Create a grid of product cards
        num_cols = 4
        num_rows = (len(results) + num_cols - 1) // num_cols
        
        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                product_index = i * num_cols + j
                if product_index < len(results):
                    product = results[product_index]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="product-card">
                            <img src="{product['image_url']}" class="product-image" />
                            <h5 class="product-title">{product['title']}</h5>
                            <p class="product-price">₹{product['price']:,}</p>
                            <p class="product-details">
                                🌟 {product['rating']} ({product['reviews_count']} reviews)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No products found matching your vibe and filters. Try a different query!")