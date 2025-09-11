import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def debug_search_issues():
    """Debug why vibe search isn't working for basic queries."""
    
    print("=" * 50)
    print("DEBUG: Analyzing Search Issues")
    print("=" * 50)
    
    # Load your data
    try:
        df = pd.read_csv('products_with_corpus.csv')
        print(f"✓ Loaded {len(df)} products")
    except:
        print("✗ Failed to load products_with_corpus.csv")
        return
    
    # Check what your data looks like
    print("\n1. SAMPLE DATA INSPECTION:")
    print("-" * 30)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"Product {i+1}:")
        print(f"  Title: {row.get('title', 'N/A')}")
        print(f"  Category: {row.get('category', 'N/A')}")
        print(f"  Vibe Tags: {row.get('vibe_tags', 'N/A')}")
        print(f"  Corpus: {row.get('corpus', 'N/A')[:100]}...")
        print()
    
    # Check if corpus contains aesthetic terms
    print("2. AESTHETIC TERMS IN DATA:")
    print("-" * 30)
    aesthetic_terms = ['cottagecore', 'dark academia', 'minimalist', 'grunge', 
                      'vintage', 'floral', 'rustic', 'gothic', 'romantic', 'cozy']
    
    for term in aesthetic_terms:
        count = df['corpus'].str.contains(term, case=False, na=False).sum()
        print(f"  '{term}': {count} products")
    
    # Test basic search
    print("\n3. SEARCH TEST:")
    print("-" * 30)
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = np.load('product_embeddings.npy')
        
        test_queries = ['cottagecore', 'dark academia', 'minimalist style']
        
        for query in test_queries:
            query_embedding = model.encode([query]).reshape(1, -1)
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            top_5_indices = np.argsort(similarities)[-5:][::-1]
            top_5_scores = similarities[top_5_indices]
            
            print(f"\nQuery: '{query}'")
            print(f"Max similarity: {max(similarities):.3f}")
            print("Top 5 results:")
            
            for i, (idx, score) in enumerate(zip(top_5_indices, top_5_scores)):
                product = df.iloc[idx]
                print(f"  {i+1}. {product['title']} (Score: {score:.3f})")
                
    except Exception as e:
        print(f"✗ Search test failed: {e}")
    
    # Analyze the problem
    print("\n4. PROBLEM ANALYSIS:")
    print("-" * 30)
    
    # Check if vibe_tags column has useful data
    vibe_tags_filled = df['vibe_tags'].notna().sum()
    print(f"Products with vibe_tags: {vibe_tags_filled}/{len(df)}")
    
    # Check corpus quality
    avg_corpus_length = df['corpus'].str.len().mean()
    print(f"Average corpus length: {avg_corpus_length:.0f} characters")
    
    # Check category distribution
    print(f"\nCategory distribution:")
    category_counts = df['category'].value_counts().head(10)
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")

def suggest_fixes():
    """Suggest fixes based on the analysis."""
    print("\n5. SUGGESTED FIXES:")
    print("-" * 30)
    print("a) If vibe_tags are mostly empty:")
    print("   - Need to add aesthetic descriptions to products")
    print("   - Use Gemini to generate vibe tags from titles/descriptions")
    print()
    print("b) If corpus is too generic:")
    print("   - Enhance product descriptions with style keywords")
    print("   - Add color, pattern, and aesthetic descriptors")
    print()
    print("c) If similarity scores are universally low:")
    print("   - Your product data might not match aesthetic queries")
    print("   - Consider using Gemini as primary search method")
    print()
    print("Want me to generate a script to fix your data?")

if __name__ == "__main__":
    debug_search_issues()
    suggest_fixes()