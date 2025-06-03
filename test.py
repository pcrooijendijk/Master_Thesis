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

deepseek.load_documents([], [])
response, content_doc = deepseek.generate_response("What is one limitation of using Google Ngrams for studying semantic shifts?", deepseek, top_k, top_p, num_beams, max_new_tokens, 0.28, temp, False)

documents = []