import random
from collections import Counter, defaultdict
from datasets import load_dataset
from utils.dataset import Custom_Dataset
import json

# Setting seed for reproducability
random.seed(42)

# Target indices
target_indices = [
    167, 159, 204, 168, 170,  # space_key_index: 0
    133, 55, 187, 88, 39,     # space_key_index: 1
    129, 174, 181, 54, 153,   # space_key_index: 2
    163, 6, 123, 46, 122      # space_key_index: 3
]

# Load source dataset
dataset = load_dataset("json", data_files="utils/documents.json")["train"]

# Group entries by space_key_index
grouped = defaultdict(list)
for idx, item in enumerate(dataset):
    grouped[item["space_key_index"]].append((idx, item))

# Pick 20 random samples from the first 4 eligible categories (at least 5 items)
eligible_keys = [k for k, items in grouped.items() if len(items) >= 5]
random.shuffle(eligible_keys)
selected = [sample for k in eligible_keys[:4] for sample in random.sample(grouped[k], 20)]

# Build question mapping
questions = {
    str(idx): {
        "space_key_index": item["space_key_index"],
        "question": item.get("question"),
    }
    for idx, item in selected
}

# Print summary
print(f"Selected {len(selected)} items:")
for idx in questions:
    q = questions[idx]
    print(f"Index: {idx}, space_key_index: {q['space_key_index']}, question: {q['question']}")

# Save selected questions
with open("question_indices.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)

# Extract filtered test questions
filtered_questions = {
    str(idx): questions[str(idx)]
    for idx in target_indices if str(idx) in questions
}

# Build aligned space_key_index list
space_indices = [filtered_questions[str(idx)]["space_key_index"] for idx in target_indices]

# Create dataset subset
Custom_Dataset("data/").convert_to_json(
    qa_index=1,
    output_file="test_documents.json",
    random_files=target_indices,
    space_indices=space_indices
)

# Verify and print distribution
test_data = load_dataset("json", data_files="test_documents.json")["train"]
counts = Counter([item["space_key_index"] for item in test_data])
print("Test set distribution by space_key_index:", counts)