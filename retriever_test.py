import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def recall_precision_at_k(
    query: str,
    gold_contexts: List[str],
    retrieved_contexts: List[str],
    k: int
) -> Tuple[float, float]:
    """Compute Recall@k and Precision@k for a single query."""
    retrieved_top_k = retrieved_contexts[:k]
    hits = sum(any(gold in retrieved for retrieved in retrieved_top_k) for gold in gold_contexts)
    
    recall = hits / len(gold_contexts) if gold_contexts else 0
    precision = hits / k
    return recall, precision

def cosine_sim(a, b):
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

def embedding_drift_analysis(
    docs_v1: List[str],
    docs_v2: List[str],
    model_v1: SentenceTransformer,
    model_v2: SentenceTransformer
) -> Dict:
    """Compare document embeddings from two versions of embedding models."""
    drift_scores = []
    for doc1, doc2 in zip(docs_v1, docs_v2):
        emb1 = model_v1.encode(doc1, convert_to_numpy=True)
        emb2 = model_v2.encode(doc2, convert_to_numpy=True)
        drift_scores.append(cosine_sim(emb1, emb2))

    return {
        "avg_cosine_similarity": np.mean(drift_scores),
        "min": np.min(drift_scores),
        "max": np.max(drift_scores),
    }

def evaluate_retrieval(
    dataset: List[Dict],  # Each entry: {'question': str, 'gold_contexts': List[str], 'retrieved_contexts': List[str]}
    k: int = 5
) -> Dict:
    recalls, precisions = [], []
    for ex in dataset:
        recall, precision = recall_precision_at_k(
            query=ex["question"],
            gold_contexts=ex["gold_contexts"],
            retrieved_contexts=ex["retrieved_contexts"],
            k=k
        )
        recalls.append(recall)
        precisions.append(precision)

    return {
        f"Recall@{k}": round(np.mean(recalls), 4),
        f"Precision@{k}": round(np.mean(precisions), 4)
    }

# Example mock dataset
dataset = [
    {
        "question": "What are side effects of aspirin?",
        "gold_contexts": ["...side effects of aspirin include..."],
        "retrieved_contexts": [
            "...side effects of aspirin include stomach bleeding...",
            "...aspirin is a pain reliever...",
            "...unrelated context..."
        ]
    },
    # More examples...
]

results = evaluate_retrieval(dataset, k=3)
print(results)

# Optional: Drift between two models
model_v1 = SentenceTransformer("all-MiniLM-L6-v2")
model_v2 = SentenceTransformer("all-mpnet-base-v2")
docs = ["Aspirin may cause nausea.", "It can also lead to bleeding."]

drift_report = embedding_drift_analysis(docs, docs, model_v1, model_v2)
print(drift_report)

