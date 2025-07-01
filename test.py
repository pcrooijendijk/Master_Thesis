import json 

for i in range(1,12):
    with open(f"eval_dataset_p/eval_dataset_{i}.json") as f: 
        eval = json.load(f)

    print(i)
    print(eval[12]["answer"]) # 2
    print("\n")