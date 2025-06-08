from ragas.metrics import RougeScore, BleuScore
from ragas import SingleTurnSample
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Put in the path to the evaluation dataset.")
args = parser.parse_args()
print(args.path)

# Loading the dataset
all_documents = load_dataset("json", data_files=args.path)["train"]
print(all_documents)