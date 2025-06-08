import os
import json
import sys
from typing import Optional
import argparse

from datasets import load_dataset, Dataset, DatasetDict
from ragas import evaluate, RunConfig
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Choose evaluation mode. Use --mode full or --mode base", required=True, default="full")
parser.add_argument("--client_id", help="Choose which client to evaluate.", required=True, default=2)
parser.add_argument("eval", help="Whether to perform evaluation with RAGAS.")
args = parser.parse_args()

client_id: int = args.client_id
print(client_id)

# ------------------------------------------------------------------------------------------------------
if args.eval:
    print("heeeyy")