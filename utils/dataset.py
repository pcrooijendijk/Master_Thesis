from pathlib import Path
import pymupdf
import random
from typing import List
import glob
import json
import os

random.seed(42)

class Custom_Dataset:
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

    def convert_to_json(self, qa_index: int, output_file: str, last_file: int = None, random_files: List = None, space_indices: List = None) -> None:
        # Function to convert the questions, documents and answers to JSON format instead of
        # PDF and seperate JSON file
        documents = []
        
        if random_files:
            directory_length = random_files
        else: 
            directory_length = len(next(os.walk(self.path))[1])
            directory_length = range(directory_length) if last_file == 0 or None else range(last_file)

        for index, cur_folder_num in enumerate(directory_length): 
            print("Appending document {} to the JSON file.".format(cur_folder_num))
            q_list = [json.loads(line)['question'] for line in open(f'{self.path}/{cur_folder_num}/{cur_folder_num}_qa.jsonl').readlines()]
            a_list = [json.loads(line)['answer'] for line in open(f'{self.path}/{cur_folder_num}/{cur_folder_num}_qa.jsonl').readlines()]
            pdf_path = glob.glob(f'{self.path}/{cur_folder_num}/*.pdf')[0]

            # Append the documents to texts lists to get the content of the documents
            with pymupdf.open(pdf_path) as file:
                try:
                    question = q_list[qa_index]
                except IndexError:
                    qa_index -= 1
                if space_indices:
                    space_key_index = space_indices[index]
                else:
                    space_key_index = random.randint(0, 3) # For the space key
                documents.append(
                    {
                        "question": q_list[qa_index],
                        "context": chr(12).join([page.get_text().replace('\n', '') for page in file]),
                        "answer": a_list[qa_index],
                        "space_key_index": space_key_index,
                        "metadata": file.metadata,
                    }
                )
        with open(output_file, 'w', encoding='utf-8') as f: 
            json.dump(documents, f, ensure_ascii=False, indent=4, separators=(",", ": "))

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