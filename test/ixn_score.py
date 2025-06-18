from typing import List, Dict
import json
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Put in the path of the retrieved documents dataset.", required=True)
parser.add_argument("--output", help="Output CSV filename", default="ixn_scores.pickle")
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
    output_file: str,
    k: int = 10
) -> Dict[str, float]:
    ixn_scores = []
    for base_docs, comp_docs in zip(baseline, comparison):
        score = compute_ixn_score(base_docs, comp_docs, k)
        ixn_scores.append(score)

    avg_score = round(sum(ixn_scores) / len(ixn_scores), 4)
    print("IXN scores:", ixn_scores)

    os.makedirs("ixn_output", exist_ok=True)
    with open("ixn_output/" + output_file, 'wb') as f:
        pickle.dump(ixn_scores, f)

with open("retrieved_docs/retrieved_docs_baseline.json") as f: 
    baseline_docs = json.load(f)

with open(args.path) as f: 
    revelant_documents = json.load(f)

evaluate_ixn_for_users(baseline_docs, revelant_documents, args.output)