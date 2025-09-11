import pandas as pd
import numpy as np
import json
import os
import logging
import re
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, losses, InputExample
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import random
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibeSearchTrainingPipeline:
    def __init__(self, gemini_api_key: str):
        """Initialize the training pipeline with Gemini for data augmentation."""
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.gemini_api_key = gemini_api_key
        self.setup_gemini()
        
        # Load base sentence transformer
        self.base_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Aesthetic vocabulary for better understanding
        self.aesthetic_vocabulary = {
            'dark academia': ['vintage', 'classic', 'scholarly', 'tweed', 'library', 'books', 'gothic', 'romantic academia'],
            'cottagecore': ['rural', 'pastoral', 'vintage floral', 'cozy', 'handmade', 'rustic', 'cottage', 'farmhouse'],
            'y2k': ['futuristic', 'metallic', 'cyber', 'holographic', 'tech', 'millennium', 'digital', 'space age'],
            'minimalist': ['clean', 'simple', 'neutral', 'basic', 'essential', 'uncluttered', 'modern', 'sleek'],
            'grunge': ['edgy', 'distressed', 'alternative', 'punk', 'rebellious', 'vintage denim', 'band tees'],
            'soft girl': ['pastel', 'cute', 'kawaii', 'blush', 'feminine', 'dreamy', 'innocent', 'sweet'],
            'indie sleaze': ['vintage', 'thrifted', 'eclectic', 'artsy', 'bohemian', 'creative', 'unconventional'],
            'coquette': ['feminine', 'romantic', 'delicate', 'bows', 'lace', 'girly', 'pretty', 'dainty'],
            'jackie kennedy': ['classic elegance', 'vintage chic', 'timeless', 'pillbox hats', 'tuxedo dress', 'clean lines']
        }
        
        self.cultural_references = {
            'priyanka chopra barfi': ['vintage', 'retro bollywood', 'classic indian', 'colorful traditional'],
            'taylor swift folklore': ['cottagecore', 'indie folk', 'nature', 'cozy cabin', 'cardigans'],
            'audrey hepburn': ['classic elegant', 'vintage chic', 'timeless', 'little black dress'],
            'french girl': ['effortless chic', 'minimalist', 'neutral tones', 'classic pieces'],
            'rachel green friends': ['90s preppy', 'layered looks', 'casual chic', 'vintage 90s'],
            'jackie kennedy': ['classic elegance', 'vintage chic', 'timeless', 'pillbox hats', 'tuxedo dress', 'clean lines']
        }
        
    def setup_gemini(self):
        """Setup Gemini AI for data augmentation."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("✅ Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini AI: {str(e)}")
            raise
    
    def load_and_prepare_data(self, data_file: str) -> pd.DataFrame:
        """Load and prepare the product data with enhanced corpus."""
        logger.info("📊 Loading and preparing product data...")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found!")
        
        df = pd.read_csv(data_file)
        
        df['enhanced_corpus'] = df.apply(self._create_enhanced_corpus, axis=1)
        
        logger.info(f"✅ Loaded {len(df)} products with enhanced corpus")
        return df
    
    def _create_enhanced_corpus(self, row) -> str:
        """Create an enhanced corpus for each product with aesthetic understanding."""
        base_text = f"{row.get('title', '')} {row.get('description', '')} {row.get('vibe_tags', '')}"
        
        category_context = f"category: {row.get('category', '')}"
        
        price = row.get('price', 0)
        if price < 1000:
            price_tier = "affordable budget-friendly"
        elif price < 3000:
            price_tier = "mid-range"
        else:
            price_tier = "premium luxury"
        
        enhanced_corpus = f"{base_text} {category_context} {price_tier}"
        
        return enhanced_corpus.lower().strip()
    
    def generate_aesthetic_training_pairs(self, df: pd.DataFrame, num_pairs: int = 1000) -> List[Tuple[str, str, float]]:
        """Generate training pairs using Gemini for aesthetic understanding."""
        logger.info(f"🎨 Generating {num_pairs} aesthetic training pairs using Gemini...")
        
        training_pairs = []
        aesthetic_queries = []
        
        for aesthetic, keywords in self.aesthetic_vocabulary.items():
            aesthetic_queries.extend([
                f"{aesthetic} style",
                f"{aesthetic} aesthetic",
                f"{aesthetic} vibe",
                f"{aesthetic} outfit",
                f"{aesthetic} look"
            ])
            
            for keyword in keywords[:3]:
                aesthetic_queries.extend([
                    f"{keyword} style",
                    f"{keyword} aesthetic",
                    f"clothes with {keyword} vibe"
                ])
        
        cultural_queries = self._generate_cultural_reference_queries()
        aesthetic_queries.extend(cultural_queries)
        
        for query in tqdm(aesthetic_queries[:num_pairs//2], desc="Creating positive pairs"):
            relevant_products = self._find_relevant_products_with_gemini(query, df, top_k=3)
            
            for product_idx in relevant_products:
                if product_idx in df.index:
                    product_text = df.loc[product_idx]['enhanced_corpus']
                    training_pairs.append((query, product_text, 1.0))
        
        for _ in tqdm(range(num_pairs//2), desc="Creating negative pairs"):
            query = random.choice(aesthetic_queries)
            random_product_idx = random.randint(0, len(df)-1)
            product_text = df.iloc[random_product_idx]['enhanced_corpus']
            
            is_negative = self._verify_negative_pair_with_gemini(query, product_text)
            if is_negative:
                training_pairs.append((query, product_text, 0.0))
        
        logger.info(f"✅ Generated {len(training_pairs)} training pairs")
        return training_pairs
    
    def _generate_cultural_reference_queries(self, num_queries: int = 50) -> List[str]:
        """Generate cultural reference queries using Gemini."""
        prompt = f"""
        Generate {num_queries} diverse cultural reference queries for fashion/aesthetic search.
        Include references to:
        - Bollywood movies and characters
        - Western movies and TV shows
        - Music artists and their iconic styles
        - Historical periods and their fashion
        - Lifestyle and personality types
        
        Format as a simple list, one query per line.
        Examples:
        - Audrey Hepburn breakfast at tiffanys
        - 90s rachel green friends
        - Taylor Swift folklore era
        - French girl aesthetic
        - Old money aesthetic
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            queries = [q.strip().strip('-').strip() for q in response.text.split('\n') if q.strip()]
            return queries[:num_queries]
        except Exception as e:
            logger.error(f"Error generating cultural queries: {e}")
            return []
    
    def _find_relevant_products_with_gemini(self, query: str, df: pd.DataFrame, top_k: int = 3) -> List[int]:
        """Use Gemini to find relevant products for a query."""
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        product_descriptions = "\n".join([
            f"ID: {row['id']}, Title: {row['title']}, Description: {row['description'][:100]}" 
            for _, row in sample_df.iterrows()
        ])
        
        prompt = f"""
        Query: "{query}"
        
        From the following products, identify the {top_k} most relevant ones:
        {product_descriptions}
        
        Respond with only the Product IDs separated by commas.
        Example: 123, 456, 789
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            ids = [int(x.strip()) for x in re.findall(r'\b\d+\b', response.text)]
            return [idx for idx in ids if idx in df['id'].values][:top_k]
        except Exception as e:
            logger.error(f"Error finding relevant products: {e}")
            return []
    
    def _verify_negative_pair_with_gemini(self, query: str, product_text: str) -> bool:
        """Verify if a query-product pair should be negative."""
        prompt = f"""
        Query: "{query}"
        Product: "{product_text[:200]}"
        
        Does this product match the query/vibe? Respond with only "YES" or "NO".
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return "NO" in response.text.upper()
        except Exception as e:
            logger.error(f"Error verifying negative pair: {e}")
            return random.choice([True, False])
    
    def fine_tune_model(self, training_pairs: List[Tuple[str, str, float]]):
        """Fine-tunes the SentenceTransformer model on the generated aesthetic data."""
        logger.info("📚 Starting model fine-tuning...")

        train_examples = []
        for query, product, score in training_pairs:
            train_examples.append(InputExample(texts=[query, product], label=score))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        # CosineSimilarityLoss is a great loss function for semantic search, as it optimizes for cosine similarity.
        train_loss = losses.CosineSimilarityLoss(model=self.base_model)

        # Fine-tune the model for one epoch.
        logger.info("⏳ Fine-tuning in progress. This may take a few minutes...")
        self.base_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
            output_path='vibe_search_model_tuned'
        )

        logger.info("✅ Model fine-tuning completed and saved to 'vibe_search_model_tuned'")

    def create_enhanced_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Create enhanced embeddings using the fine-tuned model with aesthetic context."""
        logger.info("🔄 Creating enhanced product embeddings using the fine-tuned model...")
        
        embeddings = self.base_model.encode(
            df['enhanced_corpus'].tolist(),
            convert_to_tensor=True,
            show_progress_bar=True
        ).detach().numpy()
        
        logger.info(f"✅ Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_enhanced_model_data(self, df: pd.DataFrame, embeddings: np.ndarray):
        """Save the enhanced model data."""
        logger.info("💾 Saving enhanced model data...")
        
        df.to_csv('products_enhanced_corpus.csv', index=False)
        np.save('product_embeddings_enhanced.npy', embeddings)
        
        with open('aesthetic_vocabulary.json', 'w') as f:
            json.dump(self.aesthetic_vocabulary, f, indent=2)
        
        logger.info("✅ Enhanced model data saved successfully")
    
    def evaluate_model(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test queries."""
        logger.info("📊 Evaluating enhanced model...")
        
        test_queries = [
            "priyanka chopra in barfi",
            "cottagecore aesthetic", 
            "dark academia style",
            "taylor swift folklore era",
            "french girl minimalist",
            "90s grunge outfit"
        ]
        
        results = {}
        for query in test_queries:
            query_embedding = self.base_model.encode([query], convert_to_tensor=True).detach().numpy()
            
            similarities = cosine_similarity(query_embedding, embeddings)
            top_score = np.max(similarities[0])
            
            results[query] = top_score
            
            top_indices = np.argsort(similarities[0])[-3:][::-1]
            logger.info(f"\nQuery: '{query}' (Max Score: {top_score:.3f})")
            for i, idx in enumerate(top_indices):
                product = df.iloc[idx]
                logger.info(f"  {i+1}. {product['title']} (Score: {similarities[0][idx]:.3f})")
        
        avg_score = np.mean(list(results.values()))
        logger.info(f"\n📊 Average similarity score: {avg_score:.3f}")
        
        return results
    
    def run_full_pipeline(self, data_file: str, num_training_pairs: int = 1000):
        """Run the complete training and fine-tuning pipeline."""
        logger.info("🚀 Starting enhanced vibe search training pipeline...")
        
        df = self.load_and_prepare_data(data_file)
        
        training_pairs = self.generate_aesthetic_training_pairs(df, num_training_pairs)
        
        self.fine_tune_model(training_pairs)
        
        embeddings = self.create_enhanced_embeddings(df)
        
        self.save_enhanced_model_data(df, embeddings)
        
        evaluation_results = self.evaluate_model(df, embeddings)
        
        logger.info("🎉 Enhanced vibe search training pipeline completed!")
        return df, embeddings, evaluation_results


if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    pipeline = VibeSearchTrainingPipeline(GEMINI_API_KEY)
    
    try:
        df, embeddings, results = pipeline.run_full_pipeline('products_with_corpus.csv', num_training_pairs=100)
        print("\n🎉 Training completed successfully!")
        print(f"📊 Evaluation results: {results}")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")