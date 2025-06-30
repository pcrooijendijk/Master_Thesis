# import json

# with open("eval_dataset_f/eval_dataset_1.json") as f: 
#     eval = json.load(f)

# # answers = []
# # for i in eval:
# #     print(i["answer"])
# #     answers.append(i["answer"])
# #     print("---------------------------------------------------------------------\n")
# import pickle
# import numpy as np

# vies = []
# mean = []
# for i in range(1,12):
#     with open(f"ixn_output/ixn_score_{i}.pkl", "rb") as f: 
#         ar = pickle.load(f)
#     print("ar", ar)
#     mean.append(np.mean(ar))
#     vies.append(ar)
# print(" & ".join([str(i) for i in mean]))
# transposed = list(map(list, zip(*vies)))

# # Print the first few transposed columns
# for col in transposed:  # print only first 5 for brevity
#     print(" & ".join([str(x) for x in col]))

# print(transposed)

# import csv
# with open('BLEU_scores/scores_1.csv', newline='') as csvfile:
#     scores = csv.reader(csvfile, delimiter=',')
#     for row in scores:
#         print(" & ".join(row))

# import pandas as pd

# df = pd.read_csv("BLEU_scores/scores_2.csv")

# # Drop index column if it exists
# if 'index' in df.columns:
#     df = df.drop(columns=["index"])

# # Compute means
# means = df.mean()

# mean_row = " & ".join(f"{val:.4f}" for val in means)

# print(mean_row)

# import pandas as pd

# df = pd.read_csv(f"RAGAS_scores/results_{1}.csv")
# metrics = ["context_precision", "answer_relevancy", "faithfulness", "context_recall"]
# metric_arr = []
# for metric in metrics: 
#     metric_arr.append(df[metric])

# mean_row = " & ".join(f"{val:.4f}" for val in metric_arr)
# print(mean_row)

