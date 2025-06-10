import json

with open("question_indices.json") as f: 
    indices = json.load(f)

target_indices = [
    107, 132, 199, 186, 151, # space_key_index: 0
    25, 17, 136, 193, 130, # space_key_index: 1
    139, 152, 82, 46, 24, # space_key_index: 2
    214, 180, 172, 146, 176 # space_key_index: 3
]

questions = []
for id in target_indices:
    questions.append(indices[str(id)]["question"])

print(questions)