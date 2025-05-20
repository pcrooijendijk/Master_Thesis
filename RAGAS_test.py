from DeepSeek import DeepSeekApplication
from typing import Optional
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import os
from ragas.run_config import RunConfig
from datasets import Dataset
import pickle

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]

client_id: int = 9
ori_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # The original model
lora_weights_path: str = "FL_output/pytorch_model.bin"        # Path to the weights after LoRA
lora_config_path: str = "FL_output"                           # Path to the config.json file after LoRA
prompt_template: str = 'utils/prompt_template.json'           # Prompt template for LLM
question: str = ""                                            # The question to be asked
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

if sample_docs: 
    for index, doc in enumerate(sample_docs):
        with open(f"index_{index}.txt", "w") as file:
            file.write(doc)
            file.flush()
            content, metadata_doc, file_name = deepseek.doc_processor.process_file(f"index_{index}.txt")
            documents.append(content)
            metadata[file_name] = metadata_doc

# Load documents
deepseek.load_documents(documents, metadata)

sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]

from datasets import Dataset, DatasetDict

# 👇 Assuming this is what you're building during evaluation:
dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = deepseek.retrieve_relevant_docs(query, 10, 0.5)
    response = deepseek.generate_response(query, deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)
    
    dataset.append({
        "question": query,
        "ground_truth": reference,
        "answer": response[0]['content'],
        "contexts": [relevant_docs],  
    })

eval_set = Dataset.from_list(dataset)

# Create a DatasetDict if you want splits
ds_dict = DatasetDict({
    "eval": eval_set
})

with open("qa_dataset.pkl", "wb") as f:
    pickle.dump(ds_dict, f)

langchain_llm = ChatOllama(model="llama3")
langchain_embeddings = OllamaEmbeddings(model="llama3")

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

result = evaluate(ds_dict['eval'],
        metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall], llm=langchain_llm,embeddings=langchain_embeddings)

print(result)