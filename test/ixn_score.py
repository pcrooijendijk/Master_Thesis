from typing import List, Dict
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path1", help="Put in the path of the first retrieved documents dataset.", required=True)
parser.add_argument("--path2", help="Put in the path of the second retrieved documents dataset.", required=True)
args = parser.parse_args()

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
    baseline: List[List[str]],
    comparison: List[List[str]],
    k: int = 10
) -> Dict[str, float]:
    ixn_scores = []
    for base_docs, comp_docs in zip(baseline, comparison):
        score = compute_ixn_score(base_docs, comp_docs, k)
        ixn_scores.append(score)

    avg_score = round(sum(ixn_scores) / len(ixn_scores), 4)
    print("IXN scores:", ixn_scores)
    return {"avg_ixn": avg_score}

with open(args.path1) as f: 
    revelant_documents_9 = json.load(f)

with open(args.path2) as f: 
    revelant_documents_2 = json.load(f)

evaluate_ixn_for_users(revelant_documents_9, revelant_documents_2)