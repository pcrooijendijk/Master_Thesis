from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import json
from more_itertools import chunked

# Set up device and clear CUDA cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Step 1: Load and parse the JSON
with open("utils/documents.json", "r", encoding="utf-8") as f:
    raw_json = json.load(f)

# Step 2: Extract only the "context" field and metadata
documents = [
    Document(
        page_content=entry["context"],
        metadata={"space_key_index": entry["space_key_index"]}
    )
    for entry in raw_json
]

# Step 3: Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Step 4: Set up HuggingFace embeddings
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True, 'batch_size': 8}

embeddings = HuggingFaceEmbeddings(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Step 5: Build FAISS index in batches
faiss_index = None
batch_size = 50  # You can tweak this depending on your RAM

for doc_batch in chunked(docs, batch_size):
    if faiss_index is None:
        faiss_index = FAISS.from_documents(doc_batch, embeddings)
    else:
        faiss_index.add_documents(doc_batch)

# Step 6: Run a similarity search
query = "Are LIME and Alvarez-Melis and Jaakkola (2017) methods dependent on model properties?"
result_docs = faiss_index.similarity_search(query)

print(result_docs[0].page_content)
