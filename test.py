from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import json
from langchain_core.documents import Document

# Step 1: Load and parse the JSON
with open("utils/documents.json", "r", encoding="utf-8") as f:
    raw_json = json.load(f)

# Step 2: Extract only the specific entries (e.g., the "content" field)
documents = [Document(page_content=entry["context"], metadata={"space_key_index": entry["space_key_index"]}) for entry in raw_json]
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
