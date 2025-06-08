import json
import glob
import random
import pymupdf
import os
from datasets import load_dataset

random.seed(42)

all_documents = load_dataset("json", data_files="utils/documents.json")

data = []
counter = 0

for index in range(len(all_documents['train'])):
    counter += 1
    if all_documents["train"][index]["question"].lower() == "Does SenseBERT employ a whole-word-masking strategy for out-of-vocabulary words?".lower():
        print("counter", counter)
        break

from collections import Counter

counts = Counter(data)
print(counts)