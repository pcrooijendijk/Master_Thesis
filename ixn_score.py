from collections import defaultdict


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
