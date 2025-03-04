import logging
from fastapi import FastAPI
from pydantic import BaseModel
from search_engine import load_faiss_index, load_text_chunks, search_faiss
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI
app = FastAPI()

# Load models and FAISS index
logging.info("Loading SBERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("SBERT model loaded.")

logging.info("Loading FAISS index and metadata...")
index = load_faiss_index("data/vector_store.index")
text_chunks = load_text_chunks("data/processed_chunks.pkl")
logging.info("FAISS index and metadata loaded.")

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/ask")
def ask_question(request: QueryRequest):
    """Handles user queries and retrieves relevant video insights."""
    results = search_faiss(request.query, index, text_chunks, top_k=request.top_k)
    return {"query": request.query, "responses": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)