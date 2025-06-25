import json

with open("eval_dataset_f/eval_dataset_1.json") as f: 
    eval = json.load(f)

answers = []
for i in eval:
    # print(i["answer"])
    answers.append(i["answer"])
    # print("---------------------------------------------------------------------\n")

def clean_outputs(outputs):
    return [output.replace("<｜end▁of▁sentence｜>", "").strip() for output in outputs]

cleaned = clean_outputs(answers)

for c in cleaned:
    print(c)
    print("---------------------------------------------------------------------\n")