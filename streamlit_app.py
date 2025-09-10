# streamlit_app.py - Enhanced Vibe Search UI for Myntra Hackathon
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

# --- Enhanced Custom CSS for Styling (keeping original colors) ---
st.markdown("""
<style>
    /* Main app theme colors - ORIGINAL MYNTRA COLORS */
    :root {
        --primary-color: #ff3f6c;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #222222;
        --font: montserrat;
    }

    /* Apply theme colors - KEEPING ORIGINAL */
    .stApp {
        background-color: var(--background-color);
        color: #ff3f6c;
        font-family: montserrat;;
    }
    .st-emotion-cache-1r651z9 { /* Sidebar background */
        background-color: var(--secondary-background-color);
    }
    .st-emotion-cache-1n667ek { /* Widget labels */
        color: var(--text-color);
    }

    /* Enhanced header styling with better fonts - CORRECTED ALIGNMENT */
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

    /* Quiz launch button styling */
    .quiz-launch-container {
        text-align: center;
        margin: 2rem 0;
    }

    /* Vibe Search Input - keeping original style */
    .st-emotion-cache-1cpx97b {
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 10px 20px;
    }

    /* Quiz card styling - more elegant */
    .quiz-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border: 2px solid #ff3f6c;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        color: var(--text-color);
        box-shadow: 0 8px 32px rgba(255, 63, 108, 0.1);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .quiz-card.slide-out {
        transform: translateX(-100%);
        opacity: 0;
    }
    
    .quiz-card.slide-in {
        transform: translateX(100%);
        opacity: 0;
        animation: slideIn 0.5s ease-out forwards;
    }
    
    @keyframes slideIn {
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    .quiz-question {
        font-family: 'Caveat', cursive;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 2rem;
        text-align: center;
        color: var(--primary-color);
    }
    
    /* NEW: Styled quiz option buttons */
    .stButton > button {
        background-color: white !important;
        color: var(--primary-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 20px !important;
        padding: 10px 25px !important;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: bold;
        font-family: var(--font);
    }
    
    .stButton > button:hover {
        background-color: var(--primary-color) !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 63, 108, 0.3);
    }
    
    /* Product Cards Styling - ORIGINAL */
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

    /* Progress bar for quiz */
    .quiz-progress {
        width: 100%;
        height: 6px;
        background: #f0f2f6;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    
    .quiz-progress-fill {
        height: 100%;
        background: var(--primary-color);
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* Hide default streamlit elements */
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

# --- VIBE_KEYWORDS ---
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

# Quiz questions
quiz_questions = {
    "q1": {
        "question": "Which word best describes your ideal day?",
        "options": {
            "Cozy": "soft girl",
            "Adventurous": "boho chic",
            "Mysterious": "dark academia",
            "Edgy": "grunge",
        }
    },
    "q2": {
        "question": "What's your go-to outfit color palette?",
        "options": {
            "Earthy tones (browns, greens, creams)": "cottagecore",
            "Black and bold neons": "grunge",
            "Classic neutrals (white, black, beige)": "minimalist",
            "Soft pastels (pink, lavender, baby blue)": "soft girl",
        }
    },
    "q3": {
        "question": "What kind of accessories do you love?",
        "options": {
            "Pearl necklaces and lace ribbons": "coquette",
            "Layered chains and leather belts": "grunge",
            "Simple silver jewelry": "minimalist",
            "Chunky platforms and butterfly clips": "Y2K",
        }
    },
    "q4": {
        "question": "Choose a setting that inspires you:",
        "options": {
            "An old library": "dark academia",
            "A lush garden": "cottagecore",
            "A busy city street": "streetwear",
            "A vintage disco club": "indie sleaze",
        }
    },
    "q5": {
        "question": "What's your favorite texture?",
        "options": {
            "Soft, flowing fabrics": "soft girl",
            "Rough denim and worn leather": "grunge",
            "Structured wool and tweed": "dark academia",
            "Delicate lace and silk": "coquette",
        }
    }
}

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = {
        "current_q": 0,
        "scores": {vibe: 0 for vibe in get_trending() + list(VIBE_KEYWORDS.keys())},
        "completed": False,
    }
if 'last_search' not in st.session_state:
    st.session_state.last_search = None

def reset_quiz_state():
    st.session_state.quiz_state = {
        "current_q": 0,
        "scores": {vibe: 0 for vibe in get_trending() + list(VIBE_KEYWORDS.keys())},
        "completed": False,
    }

# --- Main Page ---
if st.session_state.page == 'main':
    st.markdown("<h1 class='main-header'>Myntra Vibe Search</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Discover products that match your aesthetic. Powered by Gemini AI.</p>", unsafe_allow_html=True)

    # Quiz launch button (centered)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🧭 Find My Style", key="quiz_launch", help="Take our interactive style quiz!", use_container_width=True):
            reset_quiz_state()
            st.session_state.page = 'quiz'
            st.rerun()

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
    st.markdown("<p class='sub-header'>Discover your unique fashion aesthetic through our interactive quiz</p>", unsafe_allow_html=True)

    # Quiz logic
    if not st.session_state.quiz_state["completed"]:
        if st.session_state.quiz_state["current_q"] < len(quiz_questions):
            current_q_key = list(quiz_questions.keys())[st.session_state.quiz_state["current_q"]]
            current_q_data = quiz_questions[current_q_key]
            
            # Progress bar
            progress = (st.session_state.quiz_state["current_q"] / len(quiz_questions)) * 100
            st.markdown(f"""
            <div class="quiz-progress">
                <div class="quiz-progress-fill" style="width: {progress}%;"></div>
            </div>
            <p style="text-align: center; margin-bottom: 2rem;">Question {st.session_state.quiz_state["current_q"] + 1} of {len(quiz_questions)}</p>
            """, unsafe_allow_html=True)

            # Quiz card with question
            st.markdown(f"""
            <div class="quiz-card">
                <h2 class="quiz-question">{current_q_data["question"]}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Create option buttons
            options = list(current_q_data["options"].keys())
            
            # Use columns for a more compact layout
            cols = st.columns(2)
            
            for i, option in enumerate(options):
                col = cols[i % 2]
                with col:
                    if st.button(option, key=f"option_{i}", use_container_width=True):
                        selected_vibe = current_q_data["options"][option]
                        st.session_state.quiz_state["scores"][selected_vibe] += 1
                        st.session_state.quiz_state["current_q"] += 1
                        
                        if st.session_state.quiz_state["current_q"] >= len(quiz_questions):
                            st.session_state.quiz_state["completed"] = True
                        
                        st.rerun()
        
    else:
        # Quiz completed
        top_vibe = max(st.session_state.quiz_state["scores"], key=st.session_state.quiz_state["scores"].get)
        
        st.markdown(f"""
        <div class="quiz-card" style="text-align: center;">
            <h2 class="quiz-question">Quiz Complete!</h2>
            <h1 style="font-size: 2.5rem; margin: 1rem 0; font-family: 'Kalam', cursive;">Your vibe is<br><strong>{top_vibe.title()}</strong>!</h1>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">Ready to explore your style?</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🛍️ Shop {top_vibe.title()}", key="shop_result", use_container_width=True):
                st.session_state.vibe_input = top_vibe
                st.session_state.page = 'main'
                st.session_state.last_search = search_vibe(
                    top_vibe, 20, 0.0, 10000.0, "all"
                )
                st.rerun()
        
        with col2:
            if st.button("🔄 Retake Quiz", key="retake_quiz", use_container_width=True):
                reset_quiz_state()
                st.session_state.page = 'main'
                st.rerun()