from DeepSeek import DeepSeekApplication
from typing import Optional
from ragas import evaluate, RunConfig
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import os
import json

os.environ["RAGAS_DEBUG"] = "true"

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

from utils import Custom_Dataset

all_documents = load_dataset("json", data_files="test_documents.json")
print(f"all documents {all_documents.__str__}")
questions = all_documents['train']['question']
contexts = all_documents['train']['context']
answers = all_documents['train']['answer']

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
max_new_tokens: int = 256

# Initalize a DeepSeek application for processing documents
deepseek = DeepSeekApplication(
    client_id, 
    ori_model,
    lora_weights_path,
    lora_config_path,
    prompt_template,
)

documents = []
metadata = {}

print("Starting loop for contexts")
for index, doc in enumerate(contexts):
    with open(f"index_{index}.txt", "w") as file:
        file.write(doc)
        file.flush()
        print("file written")
        content, metadata_doc, file_name = deepseek.doc_processor.process_file(f"index_{index}.txt")
        documents.append(content)
        metadata[file_name] = metadata_doc
    os.remove(f"index_{index}.txt")

print("Ended loop and now loading documents")

# Load documents
deepseek.load_documents(documents, metadata)

dataset = []
revelant_documents = []

print("Retrieving answers:\n")

for query, reference in zip(questions, answers):
    relevant_docs = deepseek.retrieve_relevant_docs(query, 10, 0.5)
    chunks, _ = deepseek.return_relevant_chunks()[0]
    revelant_documents.append([doc.page_content for doc in relevant_docs])
    response = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)
    print("Response", response, "\n")

    dataset.append({
        "question": query,
        "ground_truth": reference,
        "answer": response[0]['content'],
        "contexts": [chunks.page_content],  
    })

    with open(f"retrieved_docs/retrieved_docs_{client_id}_b.json", 'w', encoding='utf-8') as f: 
        json.dump(revelant_documents, f, indent=4)
    
    with open(f"eval_dataset/eval_dataset_{client_id}_b.json", 'w', encoding='utf-8') as f: 
        json.dump(dataset, f, ensure_ascii=False, indent=4)

# with open(f"eval_dataset_{client_id}.json") as f: 
#     dataset = json.load(f)

# eval_set = Dataset.from_list(dataset[:5])

# ds_dict = DatasetDict({
#     "eval": eval_set
# })

# run_config = RunConfig(timeout=300, log_tenacity=True, max_workers=5)  
# # RunConfig(timeout: int = 180, max_retries: int = 10, max_wait: int = 60, max_workers: int = 16, exception_types: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = (Exception,), log_tenacity: bool = False, seed: int = 42)

# langchain_llm = Ollama(model="llama3")
# langchain_embeddings = OllamaEmbeddings(model="llama3")

# result = evaluate(ds_dict['eval'],
#         metrics=[
#         context_precision,
#         answer_relevancy,
#         faithfulness,
#         context_recall,
#         ], run_config=run_config, llm=langchain_llm,embeddings=langchain_embeddings, batch_size=4)

# print(result)
# result.to_pandas().to_csv('results.csv')
