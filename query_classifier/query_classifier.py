import logging, asyncio
from transformers import pipeline
import time

logger = logging.getLogger(__name__)

# Initialize zero-shot classifier once (blockingly on import or startup)
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

# Candidate labels — intent buckets
CANDIDATE_LABELS = ["needs_rag", "does_not_need_rag"]

async def classify_query(query: str, threshold: float = 0.6):
    """Async method to classify a query into a intent bucket.

    Args:
        query (str): The user query to classify
        threshold (float, optional): The threshold for the classification. Defaults to 0.6.

    Returns:
        A dictionary with the score 
    """
    # Run blocking Hugging Face pipeline in a separate thread
    result = await asyncio.to_thread(
        classifier,
        query,
        CANDIDATE_LABELS,
    )

    labels = result.get("labels", [])
    scores = result.get("scores", [])

    # Find score for the RAG intent
    rag_score = 0.0
    for label, score in zip(labels, scores):
        if label == "needs_rag":
            rag_score = score
            break

    needs_rag = rag_score >= threshold
    
    print("Query:", query)
    print("RAG score:", rag_score)
    print("Needs RAG:", needs_rag)
    print("All labels:", labels)
    print("All scores:", scores)
    print("--------------------------------")

    return {
        "query": query,
        "rag_score": rag_score,
        "needs_rag": needs_rag,
        "raw_labels": labels,
        "raw_scores": scores
    }
    
async def main():
    query = "How do I find county housing assistance programs?"
    
    start_time = time.monotonic()
    result = await classify_query(query, threshold=0.65)
    end_time = time.monotonic()
    execution_time = (end_time - start_time) * 1000
    
    print(f"Execution time: {execution_time:.2f} ms")
    
# Run the async main
asyncio.run(main())