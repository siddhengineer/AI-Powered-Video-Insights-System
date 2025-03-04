import logging
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load SBERT model
logging.info("Loading SBERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("SBERT model loaded.")

def load_faiss_index(index_path):
    """Loads the FAISS index."""
    logging.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    logging.info("FAISS index loaded successfully.")
    return index

def load_text_chunks(metadata_path):
    """Loads the text chunks metadata."""
    logging.info(f"Loading text chunks metadata from {metadata_path}...")
    with open(metadata_path, "rb") as f:
        text_chunks = pickle.load(f)

    # Remove duplicate chunks for diversity
    unique_chunks = list(set(text_chunks))
    logging.info(f"Text chunks metadata loaded successfully. Unique chunks: {len(unique_chunks)}")
    return unique_chunks

def search_faiss(query, index, text_chunks, top_k=3):
    """Searches FAISS index and retrieves top-k relevant text chunks."""
    logging.info(f"Processing query: {query}")

    # Generate query embedding and normalize it
    query_embedding = sbert_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)  # Normalization improves search accuracy

    # Perform search
    distances, indices = index.search(query_embedding, top_k)

    # Ensure indices are within valid range
    results = []
    for i in indices[0]:
        if 0 <= i < len(text_chunks):
            results.append(text_chunks[i])

    logging.info(f"Top-{len(results)} results retrieved successfully.")
    return results

if __name__ == "__main__":
    # Define paths
    index_path = "data/vector_store.index"
    metadata_path = "data/processed_chunks.pkl"
    
    # Load FAISS index and metadata
    index = load_faiss_index(index_path)
    text_chunks = load_text_chunks(metadata_path)
    
    # Example Query
    query = "What was discussed in the video?"
    results = search_faiss(query, index, text_chunks)
    
    # Display results
    for i, result in enumerate(results, 1):
        logging.info(f"Result {i}: {result}")
