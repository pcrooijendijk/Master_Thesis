from DeepSeek.baseline_deepseek import DeepSeekApplication
from typing import Optional
from ragas import evaluate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import os
import json

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

from utils import Custom_Dataset

dataset = Custom_Dataset("data/")
dataset.convert_to_json(1, "test_documents.json", 10)

all_documents = load_dataset("json", data_files="test_documents.json")
print(f"all documents {all_documents.__str__}")
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
    revelant_documents.append([doc.page_content for doc in relevant_docs])
    response = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)
    print("Response", response, "\n")

    dataset.append({
        "question": query,
        "ground_truth": reference,
        "answer": response[0]['content'],
        "contexts": [relevant_docs[0].page_content],  
    })

    with open("retrieved_docs.json", 'w', encoding='utf-8') as f: 
        json.dump(revelant_documents, f, indent=4)
    
    with open("eval_dataset.json", 'w', encoding='utf-8') as f: 
        json.dump(dataset, f, ensure_ascii=False, indent=4)

eval_set = Dataset.from_list(dataset)

ds_dict = DatasetDict({
    "eval": eval_set
})

ds_dict.save_to_disk("eval_dataset")

# langchain_llm = ChatOllama(model="llama3")
# langchain_embeddings = OllamaEmbeddings(model="llama3")

# result = evaluate(ds_dict['eval'],
#         metrics=[
#         context_precision,
#         faithfulness,
#         answer_relevancy,
#         context_recall], llm=langchain_llm,embeddings=langchain_embeddings)

# print(result)