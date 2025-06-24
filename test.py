import pickle

with open("ixn_output/" + "ixn_score_1.pkl", "rb") as f:
    ixn_score = pickle.load(f)

import numpy as np

# Convert to numpy array (optional but convenient)
arr = np.array(ixn_score)

# Reshape into rows of 5 elements (only works if length is divisible by 5)
arr_reshaped = arr.reshape(-1, 5)

# Compute mean along rows
averages = arr_reshaped.mean(axis=1)

print(averages)