import logging, asyncio, time, os, json
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    filename="agent_intent.log",
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

vectors = None
texts = None
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_sample_data():
    global vectors, texts

    logger.info("[load_sample_data] Loading sample data...")
    # Check if the examples.json file exists
    RAG_SAMPLE_SET_PATH = BASE_DIR / "data_files" / "examples.json"
    if not RAG_SAMPLE_SET_PATH.exists():
        raise FileNotFoundError(f"The examples.json file does not exist at {RAG_SAMPLE_SET_PATH}")
    
    #  Load the data if the file exists
    data = []
    with RAG_SAMPLE_SET_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert the vectors to a numpy array
    vectors = np.array([item["vector"] for item in data], dtype=np.float32)
    texts = [item["text"] for item in data]
    
    logger.info(f"[load_sample_data] Sample data loaded successfully with {len(vectors)} items!")

async def classify_query(query: str, threshold: float = 0.6):
    """Async method to classify a query into a intent bucket.

    Args:
        query (str): The user query to classify
        threshold (float, optional): The threshold for the classification. Defaults to 0.6.

    Returns:
        A dictionary with the score 
    """
    try:
        logger.info("[classify_query] Query classification started!")
        
        if vectors is None:
            raise RuntimeError("Sample data not loaded!")
        
        try:
            # Embed query
            query_vector = model.encode(query)
            
            # Compute max cosine similarity
            max_score, best_idx = max_cosine_similarity(query_vector, vectors)
            
            needs_rag = max_score >= threshold
            
            return {
                "query": query,
                "needs_rag": needs_rag,
                "rag_score": max_score,
                "best_match": texts[best_idx],
            }
        finally:
            logger.info(f"[classify_query] Query classification ended!")
                    
    except FileNotFoundError as e:
        logger.error("Error: %s", e)
    
def max_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> float:
    """
    Computes max cosine similarity between query vector and matrix of vectors.
    matrix shape: (N, D)
    query_vec shape: (D,)
    """
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    similarities = np.dot(matrix, query_vec) / (matrix_norms * query_norm)
    return float(similarities.max()), int(similarities.argmax())
    
async def main():
    query = "How do I find county housing assistance programs?"
    
    start_time = time.monotonic()
    result = await classify_query(query, threshold=0.65)
    end_time = time.monotonic()
    execution_time = (end_time - start_time) * 1000
    
    print(f"Execution time: {execution_time:.2f} ms")
    
# Run the async main
load_sample_data()
asyncio.run(main())