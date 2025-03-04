import os
import pickle
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load SBERT model once at the start
logging.info("Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("SBERT model loaded.")

def generate_embeddings(text_chunks):
    """Generates embeddings for a list of text chunks using SBERT."""
    logging.info("Generating embeddings for text chunks...")
    embeddings = sbert_model.encode(text_chunks, convert_to_numpy=True)  # Fix applied
    logging.info("Embeddings generated.")
    return embeddings

def store_embeddings(embeddings, index_path):
    """Stores embeddings in a FAISS index."""
    logging.info(f"Storing embeddings in FAISS index at {index_path}...")
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    logging.info(f"Embeddings stored in FAISS index at {index_path}.")

def load_embeddings(index_path):
    """Loads embeddings from a FAISS index."""
    logging.info(f"Loading embeddings from FAISS index at {index_path}...")
    index = faiss.read_index(index_path)
    logging.info(f"Embeddings loaded from FAISS index at {index_path}.")
    return index

if __name__ == "__main__":
    # Example usage
    text_chunks_path = "data/processed_chunks.pkl"
    index_path = "data/vector_store.index"

    # Ensure directories exist
    Path(os.path.dirname(index_path)).mkdir(parents=True, exist_ok=True)

    # Load text chunks safely
    if not os.path.exists(text_chunks_path) or os.stat(text_chunks_path).st_size == 0:
        logging.error(f"No text chunks found at {text_chunks_path}. Ensure transcripts are processed first.")
        exit(1)

    logging.info(f"Loading text chunks from {text_chunks_path}...")
    with open(text_chunks_path, 'rb') as f:
        text_chunks = pickle.load(f)
    
    # Ensure unique text chunks
    unique_chunks = list(set(text_chunks))
    logging.info(f"Total unique text chunks: {len(unique_chunks)}")
    
    # Generate embeddings
    embeddings = generate_embeddings(unique_chunks)

    # Store embeddings
    store_embeddings(embeddings, index_path)
