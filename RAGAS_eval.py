from DeepSeek import DeepSeekApplication
from typing import Optional
from ragas import evaluate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import os

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

all_documents = load_dataset("json", data_files="test_documents.json")
questions = all_documents['train']['question']
contexts = all_documents['train']['context']
answers = all_documents['train']['answer']

client_id: int = 9
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

for index, doc in enumerate(contexts):
    with open(f"index_{index}.txt", "w") as file:
        file.write(doc)
        file.flush()
        content, metadata_doc, file_name = deepseek.doc_processor.process_file(f"index_{index}.txt")
        documents.append(content)
        metadata[file_name] = metadata_doc
    os.remove(f"index_{index}.txt")

# Load documents
deepseek.load_documents(documents, metadata)

dataset = []

for query, reference in zip(questions, answers):
    relevant_docs = deepseek.retrieve_relevant_docs(query, 10, 0.5)
    response = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)

    dataset.append({
        "question": query,
        "ground_truth": reference,
        "answer": response[0]['content'],
        "contexts": [relevant_docs],  
    })
    import json
    with open("eval_dataset.json", 'w', encoding='utf-8') as f: 
      json.dump(dataset, f, ensure_ascii=False, indent=4)

eval_set = Dataset.from_list(dataset)

ds_dict = DatasetDict({
    "eval": eval_set
})

langchain_llm = ChatOllama(model="llama3")
langchain_embeddings = OllamaEmbeddings(model="llama3")

result = evaluate(ds_dict['eval'],
        metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall], llm=langchain_llm,embeddings=langchain_embeddings)

print(result)