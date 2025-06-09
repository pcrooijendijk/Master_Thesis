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
parser.add_argument("--client_id", help="Choose which client to evaluate.", default=2)
parser.add_argument("--eval", help="Whether to perform evaluation with RAGAS. Use --eval eval to enable evaluation.")
args = parser.parse_args()

client_id: int = args.client_id

if args.mode == "full":
    from DeepSeek.deepseek_application import DeepSeekApplication
    output_path_retrieved = f"retrieved_docs/retrieved_docs_{client_id}.json"
    output_path_evaluation = f"eval_dataset/eval_dataset_{client_id}.json"
elif args.mode == "base":
    from DeepSeek.baseline_deepseek import BaselineDeepSeekApplication as DeepSeekApplication
    output_path_retrieved = f"retrieved_docs/retrieved_docs_baseline.json"
    output_path_evaluation = f"eval_dataset/eval_dataset_baseline.json"

os.environ["RAGAS_DEBUG"] = "true"

# Variables for the client
ori_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # The original model
lora_weights_path: str = "FL_output/pytorch_model.bin"        # Path to the weights after LoRA
lora_config_path: str = "FL_output"                           # Path to the config.json file after LoRA
prompt_template: str = 'utils/prompt_template.json'           # Prompt template for LLM

uploaded_documents: Optional[str] = None                      # The corresponding document(s)
custom_text: Optional[str] = None                             # Custom text input instead of documents

temp: float = 0.1                                             # Temperature for token sampling
top_p: float = 0.75                                           # Top-p sampling
top_k: int = 40                                               # Top-k filtering
num_beams: int = 4                                            # Beam search size
max_new_tokens: int = 256                                     # Max tokens to generate

# Loading the dataset
all_documents = load_dataset("json", data_files="test_documents_1.json")["train"]
questions = all_documents["question"]
contexts = all_documents["context"]
answers = all_documents["answer"]

# Initialize DeepSeek
deepseek = DeepSeekApplication(
    client_id,
    ori_model,
    lora_weights_path,
    lora_config_path,
    prompt_template
)

# Process and load documents 
# documents, metadata = [], {}

# print("Processing documents...")
# for idx, context in enumerate(contexts):
#     file_path = f"tmp_{idx}.txt"
#     with open(file_path, "w") as f:
#         f.write(context)

#     content, meta, file_name = deepseek.doc_processor.process_file(file_path)
#     documents.append(content)
#     metadata[file_name] = meta
#     os.remove(file_path)

deepseek.load_documents([], [])

# Generate responses using the local LLM's of the clients
eval_dataset = []
retrieved_documents_log = []

print("Generating responses...\n")
for query, reference in zip(questions, answers):
    relevant_docs = deepseek.retrieve_relevant_docs(query, top_k=10, sim_threshold=0.5)
    chunks, _ = deepseek.return_relevant_chunks()[0]

    response = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)

    eval_dataset.append({
        "question": query,
        "ground_truth": reference,
        "answer": response[0]["content"],
        "contexts": [chunks.page_content],
    })

    retrieved_documents_log.append([doc.page_content for doc in relevant_docs])

# Save intermediate files 
os.makedirs("retrieved_docs", exist_ok=True)
os.makedirs("eval_dataset", exist_ok=True)

with open(output_path_retrieved, "w", encoding="utf-8") as f:
    json.dump(retrieved_documents_log, f, indent=4)

with open(output_path_evaluation, "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, ensure_ascii=False, indent=4)

# ------------------------------------------------------------------------------------------------------
if args.eval == "eval":
    with open(f"eval_dataset/eval_dataset_{client_id}_b.json") as f: 
        dataset = json.load(f)

    eval_subset = Dataset.from_list(dataset)
    ds_dict = DatasetDict({"eval": eval_subset})

    print("Evaluating...\n")
    evaluation = evaluate(
        ds_dict["eval"],
        metrics=[context_precision, answer_relevancy, faithfulness, context_recall],
        run_config=RunConfig(timeout=300, log_tenacity=True, max_workers=5),
        llm=Ollama(model="llama3"),
        embeddings=OllamaEmbeddings(model="llama3"),
        batch_size=4
    )

    evaluation.to_pandas().to_csv("results.csv")
    print("Evaluation complete. Results saved to results.csv")