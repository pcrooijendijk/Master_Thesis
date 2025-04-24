from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
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
    model_name="BAAI/bge-small-en-v1.5", 
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

splitted_docs = text_splitter.split_documents(documents)
document_store = FAISS.from_documents(splitted_docs, embeddings)

question = "Are LIME and Alvarez-Melis and Jaakkola (2017) methods dependent on model properties?"

scores = document_store.similarity_search(
        query=question, 
        k=40
    )

recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
    )

text_splits = recursive_text_splitter.split_documents(scores)