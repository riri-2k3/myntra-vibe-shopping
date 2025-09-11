import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Step 1: Data and Model Setup ---

# Define the path to your prepared product data
DATA_FILE = 'products_with_corpus.csv'
EMBEDDINGS_FILE = 'product_embeddings.npy'
MODEL_NAME = 'all-MiniLM-L6-v2'

# Check if the prepared data file exists
if not os.path.exists(DATA_FILE):
    print(f"Error: The file '{DATA_FILE}' was not found.")
    print("Please make sure you have run the data preparation step to create this file.")
    exit()

# Load the data into a pandas DataFrame
df_products = pd.read_csv(DATA_FILE)

# Initialize the pre-trained ML model for generating embeddings
# This will automatically download the model if it's not already on your machine.
print("Loading the Sentence-Transformer model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")


# --- Step 2: Generate and Save Embeddings ---

# This is the most resource-intensive part. We only need to do it once.
if not os.path.exists(EMBEDDINGS_FILE):
    print("Generating product embeddings for the first time...")
    # Generate embeddings for the combined text corpus
    product_embeddings = model.encode(df_products['corpus'].tolist(), convert_to_tensor=True)
    # Save the embeddings to a file so we don't have to re-generate them every time
    np.save(EMBEDDINGS_FILE, product_embeddings.detach().numpy())
    print(f"Embeddings saved to '{EMBEDDINGS_FILE}'.")
else:
    # If embeddings already exist, load them from the file
    print("Loading pre-generated embeddings...")
    product_embeddings = np.load(EMBEDDINGS_FILE)
    print("Embeddings loaded successfully.")


# --- Step 3: Define the Vibe Search Function ---

def vibe_search(query: str, top_n: int = 5):
    """
    Performs a semantic search to find products that match a given vibe query.
    
    Args:
        query (str): The user's vibe search query (e.g., "dark academia").
        top_n (int): The number of top results to return.
        
    Returns:
        A list of dictionaries containing the best matching products.
    """
    print(f"\nSearching for vibe: '{query}'...")
    
    # Generate an embedding for the user's query
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Calculate the cosine similarity between the query and all products
    # Cosine similarity measures how similar the vectors are (from -1 to 1).
    similarities = cosine_similarity(query_embedding.detach().numpy(), product_embeddings)
    
    # Get the indices of the most similar products
    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]
    
    # Retrieve the top N products and their similarity scores
    results = []
    for index in top_n_indices:
        product_info = df_products.loc[index].to_dict()
        product_info['similarity_score'] = similarities[0][index]
        results.append(product_info)
        
    return results


# --- Step 4: Run a few example searches ---

if __name__ == "__main__":
    
    # Example 1: A vague, conceptual search
    query_1 = "bohemian festival outfit"
    results_1 = vibe_search(query_1, top_n=3)
    
    print("\n--- Example 1 Results: ---")
    for i, product in enumerate(results_1):
        print(f"{i+1}. Product: {product['title']} (Score: {product['similarity_score']:.4f})")
        print(f"   Vibe: {product['vibe_tags']}")
        
    # Example 2: A specific vibe search
    query_2 = "clean and structured look for office"
    results_2 = vibe_search(query_2, top_n=3)
    
    print("\n--- Example 2 Results: ---")
    for i, product in enumerate(results_2):
        print(f"{i+1}. Product: {product['title']} (Score: {product['similarity_score']:.4f})")
        print(f"   Vibe: {product['vibe_tags']}")

    # Example 3: A stylistic search
    query_3 = "girly clothes with bows and lace"
    results_3 = vibe_search(query_3, top_n=3)

    print("\n--- Example 3 Results: ---")
    for i, product in enumerate(results_3):
        print(f"{i+1}. Product: {product['title']} (Score: {product['similarity_score']:.4f})")
        print(f"   Vibe: {product['vibe_tags']}")