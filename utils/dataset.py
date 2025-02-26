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
                    [
                        chr(12).join([page.get_text() for page in file]),
                        file.metadata,
                        space_key_index,
                    ]
                )
            
        return documents