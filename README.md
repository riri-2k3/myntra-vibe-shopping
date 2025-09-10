# 🛍️ Myntra Vibe Search: A Vibe-Based Fashion Discovery Engine

### Project Overview
Welcome to Myntra Vibe Search, an innovative e-commerce platform built for the Myntra Hackathon. This project re-imagines online shopping by allowing users to discover fashion products based on a unique "vibe" or aesthetic, much like a Pinterest-style search. Powered by Google's Gemini AI, the app intelligently matches user queries like "dark academia" or "boho chic" to a curated product catalog, providing a truly personalized shopping experience.

The application is split into two components:
- A **FastAPI backend** that handles the search logic and API requests.
- A **Streamlit frontend** that provides a beautiful, responsive, and user-friendly interface.

### ✨ Key Features
- **Vibe-Based Search**: Search for products using descriptive aesthetic vibes like `cottagecore`, `grunge`, `desi`, or `office siren`.
- **Gemini AI Integration**: Leverages Google's Gemini 1.5 Flash model for highly accurate and sophisticated product matching.
- **Robust Fallback Search**: An enhanced text-based search is available for when the AI model is unavailable, ensuring the application remains functional.
- **Comprehensive Filtering**: Filter products by price range and category to narrow down your search.
- **Myntra-Inspired UI**: A clean, light, and professional user interface built with Streamlit and custom CSS for a mobile-friendly experience.
- **Trending Vibes**: A dynamic display of trending aesthetics to inspire new searches.

### 🚀 Tech Stack
- **Backend**: Python, FastAPI, Pandas, Google Gemini API
- **Frontend**: Streamlit
- **Data Storage**: CSV file (`products.csv`)

### ⚙️ Installation and Setup

Follow these steps to get the application running on your local machine.

#### 1. Clone the repository
```bash
git clone [https://github.com/riri-2k3/myntra-vibe-shopping.git](https://github.com/riri-2k3/myntra-vibe-shopping.git)
cd myntra-vibe-shopping