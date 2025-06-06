import json
import glob
import random
import pymupdf
import os
from datasets import load_dataset

random.seed(42)

def convert_to_json(path, qa_index: int, output_file: str, last_file: int) -> None:
    # Function to convert the questions, documents and answers to JSON format instead of
    # PDF and seperate JSON file
    documents = []
    directory_length = len(next(os.walk(path))[1])
    # directory_length = directory_length if last_file == 0 or None else last_file

    random_files = random.sample(range(0, directory_length + 1), 20)

    for cur_folder_num in random_files: 
        print("Appending document {} to the JSON file.".format(cur_folder_num))
        q_list = [json.loads(line)['question'] for line in open(f'{path}/{cur_folder_num}/{cur_folder_num}_qa.jsonl').readlines()]
        a_list = [json.loads(line)['answer'] for line in open(f'{path}/{cur_folder_num}/{cur_folder_num}_qa.jsonl').readlines()]
        pdf_path = glob.glob(f'{path}/{cur_folder_num}/*.pdf')[0]

        # Append the documents to texts lists to get the content of the documents
        with pymupdf.open(pdf_path) as file:
            space_key_index = random.randint(0, 3) # For the space key
            try:
                question = q_list[qa_index]
            except IndexError:
                print(f"Index {qa_index} is out of range. Using last valid index instead.")
                qa_index -= qa_index
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
        json.dump(documents, f, ensure_ascii=False, indent=4)

# convert_to_json("data/", qa_index=1, output_file="test_documents.json", last_file=20)

all_documents = load_dataset("json", data_files="test_documents.json")
data = []
for i in range(20):
    data.append(all_documents['train'][i]['space_key_index'])
    print(all_documents['train'][i]['space_key_index'])

from collections import Counter

counts = Counter(data)
print(counts)
