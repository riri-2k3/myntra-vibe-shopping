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

# --- Enhanced Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main app theme colors */
    :root {
        --primary-color: #ff3f6c;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #222222;
        --font: montserrat;
    }

    /* Apply theme colors */
    .stApp {
        background-color: var(--background-color);
        color: #ff3f6c;
        font-family: montserrat;
    }
    .st-emotion-cache-1r651z9 { 
        background-color: var(--secondary-background-color);
    }
    .st-emotion-cache-1n667ek { 
        color: var(--text-color);
    }
    
    .main-header {
        color: var(--primary-color);
        text-align: center;
        font-family: montserrat;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        color: #555;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-family: montserrat;
    }

    .quiz-launch-container {
        text-align: center;
        margin: 2rem 0;
    }
    .st-emotion-cache-1cpx97b {
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 10px 20px;
    }

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
        height: 2.5em;
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

    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}

    p, h1, h2, h3, h4, h5, h6, label {
        color: var(--text-color;
        font-family: 'Caveat', cursive;
    }
</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
API_URL = "http://localhost:8000/search/vibe-hybrid"
TRENDING_URL = "http://localhost:8000/trending"
CATEGORIES_URL = "http://localhost:8000/categories"
QUIZ_GENERATE_URL = "http://localhost:8000/quiz/generate"
QUIZ_RECOMMENDATION_URL = "http://localhost:8000/quiz/recommendation"

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

def generate_quiz_questions_from_api():
    try:
        response = requests.get(QUIZ_GENERATE_URL)
        response.raise_for_status()
        return response.json().get('questions', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating quiz questions: {e}")
        return []

def get_quiz_recommendation(answers: Dict[str, str]):
    payload = {"answers": answers}
    try:
        response = requests.post(QUIZ_RECOMMENDATION_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting quiz recommendation: {e}")
        return {"primary_vibe": "unknown", "secondary_vibe": "unknown", "reasoning": "Failed to get a recommendation."}


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'last_search' not in st.session_state:
    st.session_state.last_search = None
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = []
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'quiz_complete' not in st.session_state:
    st.session_state.quiz_complete = False
if 'quiz_result' not in st.session_state:
    st.session_state.quiz_result = None

def reset_quiz_state():
    st.session_state.quiz_data = []
    st.session_state.quiz_answers = {}
    st.session_state.current_question_index = 0
    st.session_state.quiz_complete = False
    st.session_state.quiz_result = None
    st.session_state.card_transition_class = 'slide-in-card'

# --- Main Page ---
if st.session_state.page == 'main':
    st.markdown("<h1 class='main-header'>Myntra Vibe Search</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Discover products that match your aesthetic. Powered by Gemini AI.</p>", unsafe_allow_html=True)

    # Quiz launch button (centered)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🧭 Find My Style", key="quiz_launch", help="Take our interactive style quiz!", use_container_width=True):
            reset_quiz_state()
            st.session_state.quiz_data = generate_quiz_questions_from_api()
            if st.session_state.quiz_data:
                st.session_state.page = 'quiz'
                st.rerun()
            else:
                st.error("Failed to generate quiz questions. Please try again.")

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

    if 'last_search' in st.session_state and st.session_state.last_search:
        results = st.session_state.last_search.get("products", [])
        message = st.session_state.last_search.get("message", "No results found.")
        search_method = st.session_state.last_search.get("search_method", "N/A")
        search_time_ms = st.session_state.last_search.get("search_time_ms", 0)

        st.info(f"**Method:** {search_method} | **Time:** {search_time_ms:.2f}ms")
        st.subheader(message)

        if results:
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

# --- Quiz Page ---
elif st.session_state.page == 'quiz':
    if st.button("← Back to Search", key="back_to_main"):
        st.session_state.page = 'main'
        st.rerun()

    st.markdown("<h1 class='main-header'>🧭 Style Compass Quiz</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Answer a few quick questions to find your vibe!</p>", unsafe_allow_html=True)

    if not st.session_state.quiz_data:
        st.warning("Quiz questions are not loaded. Please go back and try again.")
    elif not st.session_state.quiz_complete:
        current_q_data = st.session_state.quiz_data[st.session_state.current_question_index]
        
        progress = (st.session_state.current_question_index / len(st.session_state.quiz_data)) * 100
        st.markdown(f"""
        <div class="quiz-progress">
            <div class="quiz-progress-fill" style="width: {progress}%;"></div>
        </div>
        <p style="text-align: center; margin-bottom: 2rem;">Question {st.session_state.current_question_index + 1} of {len(st.session_state.quiz_data)}</p>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="quiz-card">
            <h2 class="quiz-question">{current_q_data['question']}</h2>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        for i, option in enumerate(current_q_data['options']):
            with cols[i % 2]:
                if st.button(option, key=f"option_{st.session_state.current_question_index}_{i}", use_container_width=True):
                    st.session_state.quiz_answers[current_q_data['question']] = option
                    st.session_state.current_question_index += 1
                    
                    if st.session_state.current_question_index >= len(st.session_state.quiz_data):
                        st.session_state.quiz_result = get_quiz_recommendation(st.session_state.quiz_answers)
                        st.session_state.quiz_complete = True
                        st.rerun()
                    else:
                        st.rerun()
    else:
        result = st.session_state.quiz_result
        if result:
            primary_vibe = result.get('primary_vibe', 'Your Vibe').title()
            secondary_vibe = result.get('secondary_vibe', 'A Secondary Vibe').title()
            reasoning = result.get('reasoning', "Based on your answers, we've found your perfect match!")

            st.markdown(f"""
            <div class="quiz-card" style="text-align: center;">
                <h2 class="quiz-question">Quiz Complete!</h2>
                <h1 style="font-size: 2.5rem; margin: 1rem 0; font-family: 'Kalam', cursive;">Your vibe is<br><strong>{primary_vibe}</strong>!</h1>
                <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                    {reasoning}
                </p>
                <p>Your secondary vibe is **{secondary_vibe}**.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🛍️ Shop {primary_vibe}", key="shop_primary_vibe", use_container_width=True):
                    st.session_state.vibe_input = primary_vibe
                    st.session_state.page = 'main'
                    st.session_state.last_search = search_vibe(
                        primary_vibe, 20, 0.0, 10000.0, "all"
                    )
                    st.rerun()
            
            with col2:
                if st.button(f"✨ Explore {secondary_vibe}", key="shop_secondary_vibe", use_container_width=True):
                    st.session_state.vibe_input = secondary_vibe
                    st.session_state.page = 'main'
                    st.session_state.last_search = search_vibe(
                        secondary_vibe, 20, 0.0, 10000.0, "all"
                    )
                    st.rerun()

            st.button("🔄 Retake Quiz", key="retake_quiz_final", use_container_width=True, on_click=reset_quiz_state)
        else:
            st.error("Failed to get quiz recommendation. Please try again.")
            st.button("Try Again", on_click=reset_quiz_state)