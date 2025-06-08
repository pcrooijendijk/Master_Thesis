from ragas.metrics import RougeScore, BleuScore
from ragas import SingleTurnSample

example = {
    "question": "How many categories of aggression were participants asked to classify texts into?",
    "ground_truth": "3 categories: overt aggression, covert aggression, and non-aggression.",
    "answer": "Participants were asked to classify texts into three categories: overt aggression, covert aggression, and non-aggression. The task aimed to assess their ability to perform a top-level classification of aggression in a language-agnostic manner.",
}

sample = SingleTurnSample(
    user_input=example["question"],
    response=example["answer"],
    reference=example["ground_truth"]
)

metrics = {
    "BLEU": BleuScore(),
    **{
        f"ROUGE-{t.upper()}-{m}": RougeScore(rouge_type=t, mode=m)
        for t in ["rouge1", "rougeL"]
        for m in ["precision", "recall", "fmeasure"]
    }
}

for name, metric in metrics.items():
    score = metric.single_turn_score(sample)
    print(f"{name:<30} {score:.4f}")