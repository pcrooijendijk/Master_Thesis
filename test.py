import pickle

with open("eval_dataset.pkl", 'rb') as f:
    dataset = pickle.load(f)

print(dataset[0])