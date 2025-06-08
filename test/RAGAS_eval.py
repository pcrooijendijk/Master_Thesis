import os
import json
import sys
from typing import Optional

from datasets import load_dataset, Dataset, DatasetDict
from ragas import evaluate, RunConfig
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Mode selector
mode = next((arg.split("=")[1] for arg in sys.argv if arg.startswith("--mode=")), "full").lower()

if mode == "full":
    from DeepSeek.deepseek_application import DeepSeekApplication
elif mode == "base":
    from DeepSeek.baseline_deepseek import BaselineDeepSeekApplication as DeepSeekApplication
else:
    print("Invalid evaluation mode. Use --mode=full or --mode=base")
    sys.exit(1)

os.environ["RAGAS_DEBUG"] = "true"

# Variables for the client
client_id: int = 2
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
all_documents = load_dataset("json", data_files="test_documents.json")["train"]
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
documents, metadata = [], {}

print("Processing documents...")
for idx, context in enumerate(contexts):
    file_path = f"tmp_{idx}.txt"
    with open(file_path, "w") as f:
        f.write(context)

    content, meta, file_name = deepseek.doc_processor.process_file(file_path)
    documents.append(content)
    metadata[file_name] = meta
    os.remove(file_path)

deepseek.load_documents(documents, metadata)

# Generate responses using the local LLM's of the clients
eval_dataset = []
retrieved_documents_log = []

print("Generating responses...\n")
for question, reference in zip(questions, answers):
    relevant_docs = deepseek.retrieve_relevant_docs(question, top_k=10, sim_threshold=0.5)
    chunks, _ = deepseek.return_relevant_chunks()[0]

    response = deepseek.generate_response(
        query=question,
        deepseek=deepseek,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        repetition_penalty=0.28,
        temperature=temp,
        stream_output=False
    )

    eval_dataset.append({
        "question": question,
        "ground_truth": reference,
        "answer": response[0]["content"],
        "contexts": [chunks.page_content],
    })

    retrieved_documents_log.append([doc.page_content for doc in relevant_docs])

# Save intermediate files 
os.makedirs("retrieved_docs", exist_ok=True)
os.makedirs("eval_dataset", exist_ok=True)

with open(f"retrieved_docs/retrieved_docs_{client_id}_b.json", "w", encoding="utf-8") as f:
    json.dump(retrieved_documents_log, f, indent=4)

with open(f"eval_dataset/eval_dataset_{client_id}_b.json", "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, ensure_ascii=False, indent=4)

# ------------------------------------------------------------------------------------------------------
if "--eval" in sys.argv:
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