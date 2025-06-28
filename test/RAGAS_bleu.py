from ragas.metrics import RougeScore, BleuScore
from ragas import SingleTurnSample
from datasets import load_dataset
import csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Put in the path to the evaluation dataset.")
parser.add_argument("--output", help="Output CSV filename", default="scores.csv")
parser.add_argument("--bleu", help="Output CSV filename for evaluation")
args = parser.parse_args()

if args.path:
    # Loading the dataset
    all_questions = load_dataset("json", data_files=args.path)["train"]

    metrics = {
        "BLEU": BleuScore(),
        **{
            f"ROUGE-{t.upper()}-{m}": RougeScore(rouge_type=t, mode=m)
            for t in ["rouge1", "rougeL"]
            for m in ["precision", "recall", "fmeasure"]
        }
    }

    results = []
    for i, data in enumerate(all_questions):
        sample = SingleTurnSample(
            user_input=data["question"],
            response=data["answer"],
            reference=data["ground_truth"]
        )

        row = {"index": i}
        for name, metric in metrics.items():
            row[name] = round(metric.single_turn_score(sample), 4)
        results.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index"] + list(metrics.keys()))
        writer.writeheader()
        writer.writerows(results)

if args.bleu:
    df = pd.read_csv(args.bleu)

    # Drop index column if it exists
    if 'index' in df.columns:
        df = df.drop(columns=["index"])

    # Compute means
    means = df.mean()

    mean_row = " & ".join(f"{val:.4f}" for val in means)

    print(mean_row)