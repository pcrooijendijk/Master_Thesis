from pathlib import Path
import pymupdf
import random
from typing import List

class Dataset:
    def __init__(self, path: str):
        self.path = path 

    def get_documents(self) -> List:        
        # Get the documents
        file_path = Path(self.path)
        documents = []

        # Append the documents to texts lists to get the content of the documents
        for pdf in file_path.rglob("*.pdf"): # Use rglob to find all PDFs
            with pymupdf.open(pdf) as file: 
                space_key_index = random.randint(0, 3)  # For the space key
                documents.append(
                    Document(
                        chr(12).join([page.get_text() for page in file]),
                        file.metadata,
                        space_key_index,
                    )
                )
            
        return documents

class Document: 
    def __init__(self, content: str, metadata: str, space_key_index: str):
        self.content = content
        self.metadata = metadata
        self.space_key_index = space_key_index
    
    def get_content(self) -> str:
        return self.content

    def get_metadata(self) -> str:
        return self.metadata

    def get_space_key_index(self) -> str:
        return self.space_key_index
    
    def set_content(self, content) -> None:
        self.content = content

    def set_metadata(self, metadata) -> None:
        self.metadata = metadata

    def set_space_key_index(self, space_key_index) -> None:
        self.space_key_index = space_key_index