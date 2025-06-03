from collections import defaultdict
from typing import List, Dict
import json

def compute_ixn_score(
    baseline_docs: List[str],
    secured_docs: List[str],
    k: int = 10
) -> float:
    baseline_top_k = set(baseline_docs[:k])
    secured_top_k = set(secured_docs[:k])
    intersection = baseline_top_k.intersection(secured_top_k)

    return len(intersection) / k

def evaluate_ixn_for_users(
    results: List[Dict], 
    k: int = 10
) -> Dict[str, float]:
    user_scores = defaultdict(list)

    for entry in results:
        ixn = compute_ixn_score(entry["baseline_docs"], entry["secured_docs"], k)
        user_scores[entry["user"]].append(ixn)

    avg_scores = {user: round(sum(scores) / len(scores), 4) for user, scores in user_scores.items()}
    return avg_scores

with open("retrieved_docs_9.json") as f: 
    revelant_documents = json.load(f)

print(type(revelant_documents))
print(len(revelant_documents[0]))

evaluate_ixn_for_users(revelant_documents)