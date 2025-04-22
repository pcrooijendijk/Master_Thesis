from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Step 1: Load and parse the JSON
with open("utils/documents.json", "r", encoding="utf-8") as f:
    raw_json = json.load(f)

# Step 2: Extract only the specific entries (e.g., the "content" field)
documents = [Document(page_content=entry["context"], metadata={"space_key_index": entry["space_key_index"]}) for entry in raw_json]
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

from langchain_community.vectorstores import FAISS

model_kwargs = {
            'device': device
        }
# Ensure that there are no out of memory issues
encode_kwargs = { 
    'normalize_embeddings': True, 
    'batch_size': 8
}

embeddings = HuggingFaceEmbeddings(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

db = FAISS.from_documents(docs, embeddings)

query = "Are LIME and Alvarez-Melis and Jaakkola (2017) methods dependent on model properties?"
result_docs = db.similarity_search(query)
print(result_docs[0].page_content)