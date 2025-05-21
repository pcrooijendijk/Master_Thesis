from DeepSeek import DeepSeekApplication
from typing import Optional
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from datasets import Dataset, DatasetDict
from datasets import load_dataset

all_documents = load_dataset("json", data_files="test_documents.json")
print(all_documents['train']['question'])